"""Boardstate dashboard plugin — backend API routes.

Mounted at ``/api/plugins/boardstate/`` by the Hermes dashboard plugin system.

This layer does two things:

1. **Sidecar lifecycle.** On first use it spawns exactly one Node sidecar
   (``sidecar/server.js`` — a self-contained bundle of ``@boardstate/server``'s
   control plane over an fs-backed store) bound to an ephemeral loopback port, and
   reaps it on dashboard exit. The sidecar owns the Boardstate control plane; this
   process never re-implements it.

2. **WebSocket bridge.** ``/api/plugins/boardstate/ws`` is the browser-facing,
   authenticated endpoint the plugin's ``<boardstate-view>`` connects to via the SDK
   ``buildWsUrl`` + ``createWsTransport``. Each accepted upgrade opens a client
   WebSocket to the loopback sidecar and relays JSON text frames verbatim in both
   directions. The wire format is symmetric ({id,method,params} / {id,result|error}
   / {event,payload}), so the bridge is a transparent byte-for-byte relay.

WHY A PROXY (not a direct browser→sidecar connection)
-----------------------------------------------------
The browser connects to the *dashboard origin*, so:

* Auth is the dashboard's canonical WS gate (``web_server._ws_auth_ok``) — the same
  gate kanban's live-events WS uses. It transparently accepts the right credential in
  every mode: loopback ``?token=``, gated single-use ``?ticket=``, server-internal
  ``?internal=``. No bespoke sidecar token scheme, and it works under ``--host`` /
  gated OAuth / HTTPS where a direct ``ws://127.0.0.1:<port>`` from the page would be
  blocked (mixed content) or unreachable. The sidecar binds loopback-only and is
  never exposed to the browser.

Security note
-------------
``/api/plugins/*`` requests are gated by the dashboard's plugin-enable allow-list
(``plugins.enabled``) and, for the WS upgrade, by ``_ws_upgrade_authorized`` below.
The sidecar listens on ``127.0.0.1`` only and receives traffic solely from this
in-process bridge.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import secrets
import shutil
from pathlib import Path
from typing import Optional

import httpx
import websockets
from fastapi import APIRouter, Request, Response, WebSocket, WebSocketDisconnect, status as http_status
from fastapi.responses import JSONResponse, StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter()

_DASHBOARD_DIR = Path(__file__).resolve().parent
_SIDECAR_JS = _DASHBOARD_DIR / "sidecar" / "server.js"

try:
    import fcntl  # POSIX file locking (macOS/Linux). Absent on Windows → best-effort.

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - Windows
    _HAVE_FCNTL = False

# ONE sidecar per STATE DIR, shared across backend processes. `hermes dashboard` (web)
# and the desktop app each run their own Python backend; without cross-process sharing
# they would each spawn a sidecar and both `FsStorageAdapter` would read-modify-write the
# same `workspace.json` → silent lost updates (the exact hazard fs-watch was rejected
# over). A port-file in the state dir lets a second backend ADOPT the first's sidecar
# (single writer). `_sidecar_lock` serializes within a process; a POSIX file lock closes
# the cross-process first-connect race. `owned` marks whether WE spawned it (reap only
# our own on exit).
_sidecar_lock = asyncio.Lock()
# ``operator_secret`` (SEC-1) is the DEDICATED credential for the sidecar's /operator endpoint.
# Unlike ``nonce`` (the /ws + /mcp adoption credential, published to the port file so a second
# backend can render the board), it is NEVER written to disk — only THIS spawning process holds
# it — so knowing the port-file contents is not sufficient to drive operator actions. An adopted
# (not-spawned) sidecar has ``operator_secret = None`` here → operator actions are unavailable on
# that backend (documented, acceptable).
_sidecar: dict = {"proc": None, "port": None, "nonce": None, "operator_secret": None, "owned": False}
_atexit_registered = False


def _portfile_path(state_dir: Path) -> Path:
    return state_dir / ".boardstate-sidecar.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user — treat as alive
        return True
    except OSError:
        return False
    return True


async def _port_listening(port: int) -> bool:
    """A live sidecar accepts a loopback TCP connection on its port."""
    try:
        _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout=1.0)
    except Exception:
        return False
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    return True


async def _try_adopt(state_dir: Path) -> Optional[tuple[int, str]]:
    """Adopt a sidecar recorded in the port-file iff its pid is alive AND its port is
    listening — so a stale/dead record (crash, port reuse) is ignored, not adopted."""
    try:
        rec = json.loads(_portfile_path(state_dir).read_text())
    except Exception:
        return None
    port, nonce, pid = rec.get("port"), rec.get("nonce"), rec.get("pid")
    if not (isinstance(port, int) and isinstance(nonce, str) and nonce):
        return None
    if isinstance(pid, int) and not _pid_alive(pid):
        return None
    if not await _port_listening(port):
        return None
    return port, nonce


def _state_dir() -> Path:
    """Where the sidecar keeps board state. Honors an explicit override, else a
    per-HERMES_HOME directory so isolated dashboards don't share a board."""
    override = os.environ.get("BOARDSTATE_HERMES_STATE_DIR")
    if override:
        return Path(override)
    hermes_home = os.environ.get("HERMES_HOME")
    base = Path(hermes_home) if hermes_home else (Path.home() / ".hermes")
    return base / "boardstate-state"


def _node_bin() -> str:
    return os.environ.get("HERMES_NODE_BIN") or shutil.which("node") or "node"


def _hermes_data_credentials() -> tuple[Optional[str], Optional[str]]:
    """Best-effort dashboard base URL + session token for the sidecar's Hermes REST
    data resolver. Reads the dashboard's own loopback session token + bound port from
    ``hermes_cli.web_server``. Returns (None, None) if unavailable (older dashboard,
    gated/OAuth mode, or import failure) — the sidecar then serves no live Hermes data.
    """
    try:
        from hermes_cli import web_server as _ws  # local import: avoid load-order coupling

        token = getattr(_ws, "_SESSION_TOKEN", None)
        app = getattr(_ws, "app", None)
        port = getattr(getattr(app, "state", None), "bound_port", None)
        # Only the loopback token path is wired here; gated/OAuth data-fetch is a
        # follow-up (would use the process-internal credential instead).
        if token and port:
            return f"http://127.0.0.1:{int(port)}", str(token)
    except Exception as exc:  # pragma: no cover - dashboard internals unavailable
        log.info("boardstate: Hermes data credentials unavailable (%s); live data off", exc)
    return None, None


# ---------------------------------------------------------------------------
# WebSocket auth — delegate to the dashboard's canonical gate (see kanban).
# ---------------------------------------------------------------------------

def _ws_upgrade_authorized(ws: "WebSocket") -> bool:
    """Authorize a WS upgrade via ``hermes_cli.web_server._ws_auth_ok`` so the right
    credential is accepted in every mode. Older dashboards without the gate fall back
    to accept (loopback-only anyway)."""
    try:
        from hermes_cli import web_server as _ws  # local import: avoid load-order coupling
    except Exception:  # pragma: no cover - dashboard internals unavailable
        return True
    checker = getattr(_ws, "_ws_auth_ok", None)
    if checker is None:
        return True
    try:
        return bool(checker(ws))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("boardstate: WS auth check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Sidecar lifecycle.
# ---------------------------------------------------------------------------

async def _drain(stream: Optional[asyncio.StreamReader], label: str) -> None:
    if stream is None:
        return
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            log.info("boardstate-sidecar[%s]: %s", label, line.decode(errors="replace").rstrip())
    except Exception:  # pragma: no cover - best-effort logging
        return


async def _read_port(proc: "asyncio.subprocess.Process") -> int:
    """Read the sidecar's announced port from its stdout JSON handshake line."""
    assert proc.stdout is not None
    while True:
        line = await asyncio.wait_for(proc.stdout.readline(), timeout=20.0)
        if not line:
            raise RuntimeError("boardstate sidecar exited before announcing its port")
        try:
            data = json.loads(line.decode().strip())
        except Exception:
            continue  # non-JSON log noise before the handshake
        info = data.get("boardstateSidecar") if isinstance(data, dict) else None
        if isinstance(info, dict) and "port" in info:
            return int(info["port"])


def _kill_sidecar() -> None:
    # Reap ONLY a sidecar this process spawned (never one we adopted from another
    # backend), and clear the port-file so the next start doesn't adopt a corpse.
    if not _sidecar.get("owned"):
        return
    proc = _sidecar.get("proc")
    if proc is not None and proc.returncode is None:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        _portfile_path(_state_dir()).unlink(missing_ok=True)
    except Exception:
        pass


async def _spawn_sidecar(state_dir: Path) -> tuple[int, str]:
    """Spawn a fresh sidecar and record it in the port-file. Caller holds the locks."""
    global _atexit_registered

    # Per-spawn shared secret: the sidecar's WS/MCP gate (verifyClient) requires it as a
    # `?nonce=` query param, so a random other local process can't drive the control plane just
    # by finding the ephemeral loopback port. Published to the port file for adoption.
    nonce = secrets.token_urlsafe(32)
    # SEC-1: a SEPARATE operator secret for the sidecar's /operator endpoint — NEVER written to
    # the port file, so port-file knowledge alone cannot approve/confirm/deny. Only this process
    # holds it.
    operator_secret = secrets.token_urlsafe(32)

    env = os.environ.copy()
    env["BOARDSTATE_STATE_DIR"] = str(state_dir)
    env["BOARDSTATE_SIDECAR_NONCE"] = nonce
    env["BOARDSTATE_OPERATOR_SECRET"] = operator_secret
    env.setdefault("PORT", "0")  # ephemeral loopback port

    # Inject the dashboard base URL + session token so the sidecar can resolve
    # `source:"rpc"` data bindings (sessions/usage/status/cron) against Hermes REST.
    # Server-side only — the credential never enters the board document or a browser.
    # Best-effort: without it the sidecar simply serves no live Hermes data (graceful).
    hermes_url, hermes_token = _hermes_data_credentials()
    if hermes_url and hermes_token:
        env["HERMES_DASHBOARD_URL"] = hermes_url
        env["HERMES_SESSION_TOKEN"] = hermes_token

    proc = await asyncio.create_subprocess_exec(
        _node_bin(),
        str(_SIDECAR_JS),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    try:
        port = await _read_port(proc)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
        raise

    _sidecar["proc"] = proc
    _sidecar["port"] = port
    _sidecar["nonce"] = nonce
    _sidecar["operator_secret"] = operator_secret  # in-memory only; NEVER persisted (SEC-1)
    _sidecar["owned"] = True
    # Publish port + adoption nonce for other backend processes (single writer per state dir).
    # The operator secret is DELIBERATELY excluded — adoption grants board rendering, not the
    # operator plane. chmod 600 so only this user can even read the adoption record.
    try:
        pf = _portfile_path(state_dir)
        pf.write_text(json.dumps({"port": port, "nonce": nonce, "pid": os.getpid()}))
        try:
            os.chmod(pf, 0o600)
        except OSError:  # pragma: no cover - best-effort (e.g. Windows)
            pass
    except Exception as exc:  # pragma: no cover - non-fatal
        log.warning("boardstate: could not write sidecar port-file: %s", exc)
    # Keep draining both streams so the pipe buffers never fill and stall node.
    asyncio.create_task(_drain(proc.stdout, "out"))
    asyncio.create_task(_drain(proc.stderr, "err"))
    if not _atexit_registered:
        atexit.register(_kill_sidecar)
        _atexit_registered = True
    log.info("boardstate: sidecar up on 127.0.0.1:%d (state %s)", port, state_dir)
    return port, nonce


async def _ensure_sidecar() -> tuple[int, str]:
    async with _sidecar_lock:
        # (1) This process already owns/adopted a live sidecar.
        proc = _sidecar.get("proc")
        if _sidecar.get("port") and (proc is None or proc.returncode is None):
            if proc is None:  # adopted (not ours) — re-confirm it's still listening
                if await _port_listening(int(_sidecar["port"])):
                    return int(_sidecar["port"]), str(_sidecar["nonce"])
            else:
                return int(_sidecar["port"]), str(_sidecar["nonce"])

        if not _SIDECAR_JS.exists():
            raise RuntimeError(f"boardstate sidecar bundle missing: {_SIDECAR_JS} (run the build)")

        state_dir = _state_dir()
        state_dir.mkdir(parents=True, exist_ok=True)

        # (2) Cross-process critical section: under a POSIX file lock, adopt a live
        # sidecar for this state dir, else spawn one. The lock closes the race where two
        # backends both find no port-file and both spawn.
        lock_fd = None
        if _HAVE_FCNTL:
            try:
                lock_fd = os.open(str(state_dir / ".boardstate-sidecar.lock"), os.O_CREAT | os.O_RDWR, 0o600)
                await asyncio.get_running_loop().run_in_executor(None, fcntl.flock, lock_fd, fcntl.LOCK_EX)
            except Exception:  # pragma: no cover - lock best-effort
                if lock_fd is not None:
                    os.close(lock_fd)
                    lock_fd = None
        try:
            adopted = await _try_adopt(state_dir)
            if adopted:
                _sidecar["proc"] = None  # not ours: never reap it
                _sidecar["owned"] = False
                _sidecar["port"], _sidecar["nonce"] = adopted
                log.info("boardstate: adopted existing sidecar on 127.0.0.1:%d (state %s)", adopted[0], state_dir)
                return adopted
            return await _spawn_sidecar(state_dir)
        finally:
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(lock_fd)


# ---------------------------------------------------------------------------
# HTTP: a small health/status probe for verification + debugging.
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    proc = _sidecar.get("proc")
    running = proc is not None and proc.returncode is None
    return {
        "ok": True,
        "sidecar_running": running,
        "sidecar_port": _sidecar.get("port"),
        "state_dir": str(_state_dir()),
        "bundle_present": _SIDECAR_JS.exists(),
    }


# ---------------------------------------------------------------------------
# MCP: a stable, dashboard-authed route the Hermes agent connects to (as a
# `url:` MCP server, StreamableHTTP) so its boardstate_* tool calls build the
# board live. Requests are proxied to the sidecar's ephemeral-port /mcp with the
# per-spawn nonce appended — the agent reaches a STABLE URL, the sidecar stays
# loopback-only, and auth is the dashboard's own session-token gate on
# /api/plugins/*. Streamed so both JSON and SSE responses pass through.
# ---------------------------------------------------------------------------

_MCP_FWD_REQ_HEADERS = ("content-type", "accept", "mcp-session-id", "mcp-protocol-version", "last-event-id")
_MCP_FWD_RESP_HEADERS = ("content-type", "mcp-session-id")


@router.api_route("/mcp", methods=["POST", "GET", "DELETE"])
async def mcp_proxy(request: "Request") -> "Response":
    try:
        port, nonce = await _ensure_sidecar()
    except Exception as exc:  # defensive: never crash the dashboard worker
        log.warning("boardstate: sidecar unavailable for MCP: %s", exc)
        return Response(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, content=b"sidecar unavailable")

    body = await request.body()
    fwd_headers = {h: request.headers[h] for h in _MCP_FWD_REQ_HEADERS if h in request.headers}
    url = f"http://127.0.0.1:{port}/mcp?nonce={nonce}"

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_req = client.build_request(request.method, url, content=body, headers=fwd_headers)
        upstream = await client.send(upstream_req, stream=True)
    except Exception as exc:
        await client.aclose()
        log.warning("boardstate: MCP upstream error: %s", exc)
        return Response(status_code=http_status.HTTP_502_BAD_GATEWAY, content=b"mcp upstream error")

    resp_headers = {h: upstream.headers[h] for h in _MCP_FWD_RESP_HEADERS if h in upstream.headers}

    async def _iter():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(_iter(), status_code=upstream.status_code, headers=resp_headers)


# ---------------------------------------------------------------------------
# Operator gate: the ONLY privileged path to the sidecar's operator verbs.
#
# Hermes has no role primitive, so operator approve/confirm can NEVER travel the browser
# WS or the MCP proxy (both stay blocked by the sidecar's OPERATOR_ONLY_METHODS). This
# route is the one privileged surface. It (a) requires the dashboard session auth (same
# family as the WS gate), (b) consults an allowlist policy (``boardstate.operators.json``
# in the state dir) — ABSENT ⇒ allowed only in loopback-token mode, DENIED (403) in
# gated/multi-user mode; PRESENT ⇒ the caller principal must be listed — then forwards
# ``{method, params}`` to the sidecar's nonce-gated ``/operator``. The nonce never leaves
# these two processes.
# ---------------------------------------------------------------------------

# EXACTLY the four operator verbs the sidecar executes (the sidecar re-enforces this from
# @boardstate/server's OPERATOR_ONLY_METHODS; we gate here too so a non-operator verb never
# even reaches the loopback endpoint).
_OPERATOR_METHODS: frozenset[str] = frozenset({
    "dashboard.widget.approve",
    "dashboard.capability.approve",
    "dashboard.action.confirm",
    "dashboard.action.deny",
})


def _operators_allowlist_path(state_dir: Path) -> Path:
    return state_dir / "boardstate.operators.json"


def _load_operators_allowlist(state_dir: Path) -> Optional[list[str]]:
    """The operator allowlist policy. Returns None when the file is ABSENT (⇒ loopback-only
    policy), else the list of allowed principals (malformed ⇒ empty list = nobody, fail
    closed). The file lists WHO may approve; it is authored by the operator, like the
    connectors config, and lives in the state dir (never the board doc)."""
    path = _operators_allowlist_path(state_dir)
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except Exception:  # pragma: no cover - unreadable file: fail closed
        return []
    try:
        rec = json.loads(raw)
    except Exception:
        return []  # malformed ⇒ nobody is blessed (fail closed)
    ops = rec.get("operators") if isinstance(rec, dict) else None
    return [str(o) for o in ops] if isinstance(ops, list) else []


def _authorize_operator(request: "Request") -> tuple[Optional[str], Optional[Response]]:
    """Authorize an operator request. Returns ``(principal, None)`` when allowed, or
    ``(None, error_response)`` (401/403) when not. Enforces BOTH the dashboard session
    auth and the allowlist policy above."""
    try:
        from hermes_cli import web_server as _ws  # local import: avoid load-order coupling
    except Exception:  # pragma: no cover - dashboard internals unavailable
        _ws = None

    # AUTH-1 (fail CLOSED): only a POSITIVELY-confirmed loopback single-user bind — the
    # dashboard set `app.state.auth_required = False` at startup (see web_server:should_require_auth)
    # — is trusted as a self-service operator without an allowlist. If gating is INDETERMINATE
    # (the attribute is absent — e.g. an atypical embedding) or the bind is gated/multi-user
    # (True), the operator allowlist is MANDATORY. `auth_required` defaulting to False here would
    # be fail-OPEN, so we treat "not exactly False" as untrusted.
    state = getattr(getattr(request, "app", None), "state", None)
    auth_required = getattr(state, "auth_required", None) if state is not None else None
    loopback_single_user = auth_required is False

    if loopback_single_user:
        # Loopback: validate the dashboard session token exactly as the WS gate does.
        # FAIL CLOSED when the checker is unavailable (hermes_cli unimportable or an
        # older dashboard without _has_valid_session_token): unlike the render WS, the
        # operator plane is consequential, so "can't verify" must not mean "allow".
        # Tests/harnesses without hermes_cli set BOARDSTATE_OPERATOR_TEST_BYPASS=1.
        checker = getattr(_ws, "_has_valid_session_token", None) if _ws else None
        if checker is None:
            if os.environ.get("BOARDSTATE_OPERATOR_TEST_BYPASS") != "1":
                return None, JSONResponse(
                    status_code=401,
                    content={"error": "operator auth unavailable (dashboard session gate not found)"},
                )
        elif not checker(request):
            return None, JSONResponse(status_code=401, content={"error": "unauthorized"})
        principal = "loopback"
    else:
        # Gated OR indeterminate → the allowlist is mandatory (below). Identify the caller from
        # the gate-attached session when present; a gated bind with no session is a gate failure.
        session = getattr(request.state, "session", None)
        if auth_required is True and session is None:
            return None, JSONResponse(status_code=401, content={"error": "unauthorized"})
        principal = (getattr(session, "email", None) or getattr(session, "user_id", None)) if session else None

    allowlist = _load_operators_allowlist(_state_dir())
    if allowlist is None:
        # No allowlist file: permitted ONLY for a confirmed loopback single-user bind. Otherwise
        # (gated or indeterminate) refuse — an undifferentiated session token must not be a
        # self-service operator on a shared/unknown host.
        if not loopback_single_user:
            return None, JSONResponse(
                status_code=403,
                content={"error": "operator allowlist (boardstate.operators.json) required unless the dashboard is a confirmed loopback single-user bind"},
            )
    else:
        if not principal or principal not in allowlist:
            return None, JSONResponse(status_code=403, content={"error": "caller is not an authorized operator"})

    return principal, None


@router.post("/operator")
async def operator(request: "Request") -> "Response":
    principal, denied = _authorize_operator(request)
    if denied is not None:
        return denied

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "body must be JSON { method, params }"})

    method = payload.get("method") if isinstance(payload, dict) else None
    params = (payload.get("params") if isinstance(payload, dict) else None) or {}
    if not isinstance(method, str) or method not in _OPERATOR_METHODS:
        return JSONResponse(status_code=400, content={"error": f"method not allowed on the operator endpoint: {method}"})

    try:
        port, _nonce = await _ensure_sidecar()
    except Exception as exc:  # defensive: never crash the dashboard worker
        log.warning("boardstate: sidecar unavailable for operator: %s", exc)
        return JSONResponse(status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, content={"error": "sidecar unavailable"})

    # SEC-1: forward with the DEDICATED operator secret (in-memory, never in the port file),
    # NOT the adoption nonce. If this backend ADOPTED the sidecar (didn't spawn it) it has no
    # operator secret — operator actions are unavailable here (drive them from the spawning
    # backend, or restart so this process owns the sidecar).
    operator_secret = _sidecar.get("operator_secret")
    if not operator_secret:
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"error": "operator actions unavailable on this backend (adopted sidecar); use the backend that spawned it"},
        )

    # Forward the EXACT { method, params } shape to the sidecar's secret-gated /operator.
    url = f"http://127.0.0.1:{port}/operator?nonce={operator_secret}"
    log.info("boardstate: operator %s by %s", method, principal)
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.post(url, json={"method": method, "params": params})
    except Exception as exc:
        log.warning("boardstate: operator upstream error: %s", exc)
        return JSONResponse(status_code=http_status.HTTP_502_BAD_GATEWAY, content={"error": "operator upstream error"})

    # Relay the sidecar's status + JSON body verbatim (200 result / 400 refusal / 401).
    try:
        body = upstream.json()
    except Exception:
        body = {"error": upstream.text}
    return JSONResponse(status_code=upstream.status_code, content=body)


# ---------------------------------------------------------------------------
# WebSocket: authenticated browser endpoint bridged to the loopback sidecar.
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def board_ws(ws: WebSocket) -> None:
    # Authorize the upgrade via the dashboard's canonical WS gate (browsers can't set
    # Authorization on an upgrade, so the credential rides in the query string that
    # the SDK buildWsUrl() assembled).
    if not _ws_upgrade_authorized(ws):
        await ws.close(code=http_status.WS_1008_POLICY_VIOLATION)
        return

    await ws.accept()

    try:
        port, nonce = await _ensure_sidecar()
    except Exception as exc:
        log.warning("boardstate: sidecar unavailable: %s", exc)
        await ws.close(code=http_status.WS_1011_INTERNAL_ERROR)
        return

    uri = f"ws://127.0.0.1:{port}/ws?nonce={nonce}"
    try:
        async with websockets.connect(uri, max_size=2 ** 20) as upstream:
            await _bridge(ws, upstream)
    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        return
    except Exception as exc:  # defensive: never crash the dashboard worker
        log.warning("boardstate: bridge error: %s", exc)
        try:
            await ws.close(code=http_status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass


async def _bridge(ws: "WebSocket", upstream) -> None:
    """Relay JSON text frames verbatim in both directions until either side closes."""

    async def client_to_sidecar() -> None:
        while True:
            msg = await ws.receive_text()
            await upstream.send(msg)

    async def sidecar_to_client() -> None:
        async for msg in upstream:
            if isinstance(msg, (bytes, bytearray)):
                msg = msg.decode("utf-8", errors="replace")
            await ws.send_text(msg)

    tasks = {asyncio.create_task(client_to_sidecar()), asyncio.create_task(sidecar_to_client())}
    try:
        _done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
