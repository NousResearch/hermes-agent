"""Warm-backend client for ``hermes -z`` — reuse the persistent ``hermes serve``
daemon (JSON-RPC over WebSocket, port 9119, ``hermes-serve.service``) instead
of cold-building a brand new ``AIAgent`` per invocation.

Port of Crucible's proven client (``hermesWarm.ts`` in the agent-os repo):
connect to ``ws://127.0.0.1:9119/api/ws?token=...``, wait for
``gateway.ready``, ``session.create`` (with ``close_on_disconnect=True`` so
the throwaway session is reaped the instant we close the socket — no manual
``session.close`` needed), ``config.set`` yolo=1, ``prompt.submit``, then wait
for the ``message.complete`` event.

The win: MCP servers (e.g. ``hostinger``, spawned via ``npx`` and taking
~4s to connect) are already discovered process-wide on the warm daemon by the
time this client connects — a fresh session on that daemon inherits them
immediately. The cold path has to rediscover them from scratch every
invocation, racing a ~1.5s discovery bound it can lose (see #<mcp_discovery
timing bug>).

Scope, by design (mirrors hermesWarm.ts's own documented limitation):
``session.create`` has no per-session toolset-restriction equivalent to
``-z``'s ``--toolsets``/``-t`` flag — a session always gets the profile's
full default toolset. So this client must ONLY be used for oneshot calls
that did NOT request an explicit toolset restriction; ``hermes_cli/oneshot.py``
enforces that gate before importing this module. A toolset-scoped call (e.g.
``-t hostinger``) must keep using the cold, per-invocation ``AIAgent`` path —
routing it through a shared warm session would silently drop the restriction
and grant the daemon's full default toolset instead.

Safety contract (same as hermesWarm.ts): every failure resolves to a
``WarmOneshotResult`` instead of raising.  ``submitted`` distinguishes two
situations for the caller:
  - ``submitted=False`` — failed before ``prompt.submit`` was acked
    (connect/token/session-create/config-set). Nothing ran server-side —
    safe to fall back to the cold path.
  - ``submitted=True`` — the turn was accepted by the server and may have
    already run real tools before we lost track of it (timeout/dropped
    connection). The caller must NOT retry via the cold path in this case —
    that risks double-executing side effects — surface the error instead.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

TOKEN_FILE = Path.home() / ".hermes" / "dashboard_session_token.env"
WARM_HOST = "127.0.0.1"
WARM_PORT = 9119

# Mirrors hermesWarm.ts's SETUP_TIMEOUT_MS / SUBMIT_TIMEOUT_MS exactly, so the
# two clients degrade the same way against the same daemon.
SETUP_TIMEOUT = 8.0
SUBMIT_TIMEOUT = 6 * 60.0


class WarmOneshotResult:
    __slots__ = ("ok", "text", "submitted", "error", "usage")

    def __init__(
        self,
        ok: bool,
        text: str,
        submitted: bool,
        error: Optional[str] = None,
        usage: Optional[dict] = None,
    ) -> None:
        self.ok = ok
        self.text = text
        self.submitted = submitted
        self.error = error
        self.usage = usage or {}


def _read_token() -> Optional[str]:
    try:
        for line in TOKEN_FILE.read_text(encoding="utf-8").splitlines():
            if line.startswith("HERMES_DASHBOARD_SESSION_TOKEN="):
                tok = line.split("=", 1)[1].strip()
                return tok or None
    except Exception:
        return None
    return None


async def _run(
    prompt: str,
    cwd: str,
    model: Optional[str],
    provider: Optional[str],
) -> WarmOneshotResult:
    try:
        import websockets
    except Exception as exc:
        return WarmOneshotResult(False, "", False, error=f"websockets not available: {exc}")

    token = _read_token()
    if not token:
        return WarmOneshotResult(False, "", False, error=f"no warm token at {TOKEN_FILE}")

    uri = f"ws://{WARM_HOST}:{WARM_PORT}/api/ws?token={token}"

    try:
        ws = await asyncio.wait_for(websockets.connect(uri), timeout=SETUP_TIMEOUT)
    except Exception as exc:
        return WarmOneshotResult(False, "", False, error=f"warm connect failed: {exc}")

    submitted = False
    session_id: Optional[str] = None
    next_id = 1

    async def call(method: str, params: dict, timeout: float) -> Any:
        nonlocal next_id
        rid = next_id
        next_id += 1
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"warm rpc timeout: {method}")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("id") == rid:
                if msg.get("error"):
                    raise RuntimeError((msg["error"] or {}).get("message") or f"{method} rpc error")
                return msg.get("result")
            # Interleaved event frame (status.update, etc.) or a stale reply —
            # not our turn's RPC reply. Keep waiting for the matching id.

    try:
        # Wait for gateway.ready before issuing any RPC — mirrors hermesWarm.ts's
        # connect(): the socket is open but the gateway isn't accepting RPCs yet.
        deadline = time.monotonic() + SETUP_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return WarmOneshotResult(False, "", False, error="warm connect timeout (waiting for gateway.ready)")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("method") == "event" and (msg.get("params") or {}).get("type") == "gateway.ready":
                break

        title = f"cli-oneshot-{uuid.uuid4().hex[:8]}"
        create_params: dict = {"cwd": cwd, "title": title, "close_on_disconnect": True}
        if model:
            create_params["model"] = model
        if provider:
            create_params["provider"] = provider

        created = await call("session.create", create_params, SETUP_TIMEOUT)
        session_id = (created or {}).get("session_id")
        if not session_id:
            return WarmOneshotResult(False, "", False, error="session.create returned no session_id")

        # Session-scoped only (never touches global approvals.mode) — a headless
        # oneshot call has no user to approve a dangerous-command prompt.
        await call("config.set", {"session_id": session_id, "key": "yolo", "value": "1"}, SETUP_TIMEOUT)

        await call("prompt.submit", {"session_id": session_id, "text": prompt}, SETUP_TIMEOUT)
        submitted = True

        # Past this point the turn is live server-side — any failure below
        # means submitted=True, and the caller must not retry via the cold path.
        deadline = time.monotonic() + SUBMIT_TIMEOUT
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return WarmOneshotResult(False, "", True, error="warm turn timeout")
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
            msg = json.loads(raw)
            if msg.get("method") != "event":
                continue
            params = msg.get("params") or {}
            if params.get("session_id") != session_id:
                continue
            evt_type = params.get("type")
            if evt_type == "message.complete":
                payload = params.get("payload") or {}
                text = str(payload.get("text") or "")
                status = str(payload.get("status") or "complete")
                usage = payload.get("usage") or {}
                ok = status == "complete" and bool(text)
                return WarmOneshotResult(
                    ok, text, True, error=None if ok else f"warm status={status}", usage=usage
                )
            if evt_type == "error":
                err = (params.get("payload") or {}).get("error") or "warm session error event"
                return WarmOneshotResult(False, "", True, error=err)
            # message.delta, tool.start, status.update, etc. — not needed here.
    except Exception as exc:
        return WarmOneshotResult(False, "", submitted, error=str(exc))
    finally:
        try:
            await ws.close()
        except Exception:
            pass


def try_warm_oneshot(
    prompt: str,
    cwd: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> WarmOneshotResult:
    """Synchronous entry point for ``hermes_cli/oneshot.py``. Never raises."""
    try:
        return asyncio.run(_run(prompt, cwd, model, provider))
    except Exception as exc:
        return WarmOneshotResult(False, "", False, error=str(exc))
