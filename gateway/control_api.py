"""
Local-only HTTP control API for the Hermes gateway.

Runs alongside the gateway as a background aiohttp server on 127.0.0.1.
Provides programmatic control over live agent sessions — session listing,
message injection, etc.

External tools (e.g., desloppify) can call these endpoints to influence
a running Hermes agent without needing direct process access.

All mutations go through POST /sessions/{key}/message — inject any text
(including /commands) as if the user typed it.

Default port: 47823 (override with HERMES_CONTROL_PORT env var).
"""

import json
import logging
import os

from aiohttp import web

logger = logging.getLogger(__name__)

DEFAULT_PORT = 47823

def _get_available_commands():
    """Pull command list from hermes_cli.commands (single source of truth)."""
    try:
        from hermes_cli.commands import COMMANDS
        return [{"command": cmd, "description": desc} for cmd, desc in COMMANDS.items()]
    except Exception:
        return []


def _get_port() -> int:
    return int(os.getenv("HERMES_CONTROL_PORT", str(DEFAULT_PORT)))


def _json_response(data: dict, status: int = 200) -> web.Response:
    return web.Response(
        text=json.dumps(data, ensure_ascii=False),
        content_type="application/json",
        status=status,
    )


REQUIRED_HEADER = "X-Hermes-Control"


class ControlAPI:
    """Lightweight control API that holds a reference to the GatewayRunner."""

    def __init__(self, gateway_runner):
        self.runner = gateway_runner
        self.app = web.Application(middlewares=[self._require_header])
        self._setup_routes()
        self._site = None
        self._app_runner = None

    @web.middleware
    async def _require_header(self, request: web.Request, handler):
        """Reject requests missing the X-Hermes-Control header.

        This blocks browser-based CSRF attacks: any non-standard header
        forces a CORS preflight, which fails because we don't serve
        Access-Control-Allow-* headers.  Non-browser callers just add
        the header.
        """
        if request.headers.get(REQUIRED_HEADER) != "1":
            return _json_response(
                {"error": f"Missing required header: {REQUIRED_HEADER}: 1"},
                status=403,
            )
        return await handler(request)

    def _setup_routes(self):
        self.app.router.add_get("/health", self.health)
        self.app.router.add_get("/sessions", self.list_sessions)
        self.app.router.add_get("/sessions/{key}", self.get_session)
        self.app.router.add_get("/commands", self.list_commands)
        self.app.router.add_post("/sessions/{key}/message", self.send_message)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _resolve_agent(self, key: str):
        """Look up a running agent by session key.  '_any' returns the first.
        Returns (resolved_key, agent) or (key, None) if not found."""
        if key == "_any":
            if not self.runner._running_agents:
                return key, None
            key = next(iter(self.runner._running_agents))
        return key, self.runner._running_agents.get(key)

    # ── Endpoints ────────────────────────────────────────────────────────

    async def health(self, request: web.Request) -> web.Response:
        return _json_response({"status": "ok"})

    async def list_sessions(self, request: web.Request) -> web.Response:
        """List all sessions, marking which have a live (running) agent."""
        sessions = []
        for key, agent in self.runner._running_agents.items():
            entry = {
                "session_key": key,
                "status": "running",
                "model": getattr(agent, "model", None),
                "provider": getattr(agent, "provider", None),
                **self.runner.get_session_info(key),
            }
            sessions.append(entry)
        return _json_response({"sessions": sessions})

    async def get_session(self, request: web.Request) -> web.Response:
        key = request.match_info["key"]
        key, agent = self._resolve_agent(key)
        if not agent:
            return _json_response(
                {"error": f"No running agent for '{key}'"},
                status=404,
            )
        return _json_response({
            "session_key": key,
            "status": "running",
            "model": getattr(agent, "model", None),
            "provider": getattr(agent, "provider", None),
            **self.runner.get_session_info(key),
        })

    async def list_commands(self, request: web.Request) -> web.Response:
        """List available slash commands that can be sent via /message."""
        return _json_response({"commands": _get_available_commands()})

    async def send_message(self, request: web.Request) -> web.Response:
        """Inject a message (or /command) into a session.

        POST body: {"text": "...", "mode": "interrupt"|"queue"}

        Modes:
        - interrupt (default): cancel current agent run, process this message
        - queue: let current run finish, then process this message next
        """
        key = request.match_info["key"]

        try:
            body = await request.json()
        except Exception:
            return _json_response({"error": "Invalid JSON body"}, status=400)

        text = body.get("text", "").strip()
        if not text:
            return _json_response({"error": "'text' is required"}, status=400)

        mode = body.get("mode", "interrupt")
        if mode not in ("interrupt", "queue"):
            return _json_response(
                {"error": f"Invalid mode '{mode}', must be 'interrupt' or 'queue'"},
                status=400,
            )

        logger.info(
            "Control API: send_message session=%s mode=%s text=%s",
            key, mode, text[:80],
        )

        try:
            result = self.runner.inject_message(key, text, interrupt=(mode != "queue"))
            # await if coroutine (GatewayRunner is async, CLI shim is sync)
            if hasattr(result, "__await__"):
                await result
        except LookupError as e:
            return _json_response({"error": str(e)}, status=404)

        return _json_response({"success": True, "message": "Message submitted"})

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def start(self):
        port = _get_port()
        self._app_runner = web.AppRunner(self.app, access_log=None)
        await self._app_runner.setup()
        self._site = web.TCPSite(self._app_runner, "127.0.0.1", port)
        try:
            await self._site.start()
            logger.info("Control API listening on http://127.0.0.1:%d", port)
            self._write_port_file(port)
        except OSError as e:
            logger.warning("Control API failed to bind port %d: %s", port, e)

    async def stop(self):
        if self._site:
            await self._site.stop()
        if self._app_runner:
            await self._app_runner.cleanup()
        self._remove_port_file()
        logger.info("Control API stopped")

    @staticmethod
    def _write_port_file(port: int):
        path = os.path.expanduser("~/.hermes/control_api.port")
        try:
            with open(path, "w") as f:
                f.write(str(port))
        except OSError:
            pass

    @staticmethod
    def _remove_port_file():
        path = os.path.expanduser("~/.hermes/control_api.port")
        try:
            os.remove(path)
        except OSError:
            pass
