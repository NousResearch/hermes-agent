"""Worker daemon — the agent server that runs *inside* the isolation boundary.

One daemon per employee. It stays resident (this is the "warm" state the
supervisor manages) and answers chat requests over a unix socket by invoking
Hermes headless in the employee's own HERMES_HOME.

The router NEVER execs Hermes across the tenant boundary — it only speaks this
socket protocol. In production this same daemon runs inside the employee's
container / microVM; the socket becomes a vsock/TCP endpoint.

Protocol (newline-delimited JSON, one request+response per connection):
    -> {"type": "ping"}
    <- {"ok": true, "pong": true}
    -> {"type": "chat", "session": "<name>", "message": "<text>"}
    <- {"ok": true, "reply": "<text>"}   |   {"ok": false, "error": "<msg>"}

Config comes from env: ORCHARD_SOCKET, HERMES_HOME, ORCHARD_HERMES_BIN,
ORCHARD_WORKSPACE, ORCHARD_IDLE_TTL (optional self-shutdown).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path


class WorkerDaemon:
    def __init__(self) -> None:
        self.socket_path = Path(os.environ["ORCHARD_SOCKET"])
        self.hermes_home = os.environ["HERMES_HOME"]
        self.hermes_bin = os.environ.get("ORCHARD_HERMES_BIN", "hermes")
        self.workspace = os.environ.get("ORCHARD_WORKSPACE", self.hermes_home)
        self.idle_ttl = float(os.environ.get("ORCHARD_IDLE_TTL", "0") or 0)
        # Shared secret with the router. Read from STDIN (first line) so it never
        # lands in the process environment — otherwise a same-UID sibling could
        # scrape it with `ps eww`. Falls back to env for the container backend,
        # where env is already namespace-isolated.
        self.token = _read_token_from_stdin() or os.environ.get("ORCHARD_WORKER_TOKEN", "")
        # One agent turn at a time per employee (sessions are not reentrant).
        self._lock = asyncio.Lock()
        self._last_activity = time.monotonic()

    async def run(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        idle_task = asyncio.create_task(self._idle_watchdog(server)) if self.idle_ttl else None
        async with server:
            await server.serve_forever()
        if idle_task:
            idle_task.cancel()

    async def _idle_watchdog(self, server: asyncio.AbstractServer) -> None:
        while True:
            await asyncio.sleep(5)
            if time.monotonic() - self._last_activity > self.idle_ttl:
                server.close()
                return

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            resp = await self._dispatch(req)
        except Exception as e:  # never crash the daemon on a bad request
            resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        try:
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            writer.close()

    async def _dispatch(self, req: dict) -> dict:
        self._last_activity = time.monotonic()
        kind = req.get("type")
        if kind == "ping":
            return {"ok": True, "pong": True}
        # Everything privileged requires the shared token.
        if self.token and req.get("token") != self.token:
            return {"ok": False, "error": "unauthorized"}
        if kind == "chat":
            async with self._lock:
                return await self._chat(req["session"], req["message"])
        return {"ok": False, "error": f"unknown request type {kind!r}"}

    def _load_secrets(self) -> dict:
        """Read this tenant's skill secrets fresh each turn (so updates apply
        without a re-wake). Injected only into the Hermes subprocess env, not the
        daemon's own env. Same-UID note: env is still ps-visible on the Hermes
        process locally — the container backend removes that; here the confined
        home already blocks other tenants from reading secrets.json."""
        import json
        f = os.path.join(self.hermes_home, "secrets.json")
        try:
            with open(f) as fh:
                data = json.load(fh)
            return {k: str(v) for k, v in data.items() if isinstance(k, str)}
        except Exception:
            return {}

    async def _chat(self, session: str, message: str) -> dict:
        env = dict(os.environ)
        env["HERMES_HOME"] = self.hermes_home
        # Inject skill tokens. Hermes strips well-known credential names (e.g.
        # GITHUB_TOKEN) from tool subprocess env; the `_HERMES_FORCE_` prefix is
        # its opt-in to pass them through, so the agent's tools actually see them.
        for k, v in self._load_secrets().items():
            env[k] = v
            env[f"_HERMES_FORCE_{k}"] = v
        # `-z` prints ONLY the reply; `-c <session>` resumes the named conversation.
        proc = await asyncio.create_subprocess_exec(
            self.hermes_bin, "-z", message, "-c", session, "--yolo",
            cwd=self.workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            return {"ok": False, "error": f"hermes exit {proc.returncode}: {err.decode(errors='replace')[:2000]}"}
        return {"ok": True, "reply": out.decode(errors="replace").strip()}


def _read_token_from_stdin() -> str:
    """Read the one-line token the router pipes at spawn. Skip if stdin is a
    TTY (manual run) so we don't block."""
    try:
        if sys.stdin and not sys.stdin.isatty():
            return (sys.stdin.readline() or "").strip()
    except Exception:
        pass
    return ""


def main() -> None:
    asyncio.run(WorkerDaemon().run())


if __name__ == "__main__":
    main()
