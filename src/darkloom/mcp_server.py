"""MCP server exposing Tor management tools to Hermes.

Register with:
    hermes mcp add darkloom --command "python -m darkloom.mcp_server"

Provides 6 tools:
    tor_download    — Download Tor Expert Bundle
    tor_start       — Start Tor daemon with bridges
    tor_stop        — Stop Tor daemon
    tor_status      — Get current Tor status
    tor_verify      — Verify traffic routes through Tor
    tor_add_bridge  — Add a bridge line
"""
import json
import sys
from pathlib import Path

from darkloom.manager import TorManager, TorState
from darkloom.constants import BRIDGES_PATH
from darkloom.policy import NetworkChannel, authorize
from darkloom.privacy import classify_error, private_diagnostic, require_local_admin
from darkloom.privacy import get_logger

logger = get_logger(__name__)

# Module-level singleton — one TorManager per process
_manager: TorManager | None = None


def _error(error: object, component: str, code: str | None = None) -> str:
    """Serialize only the stable public classification to an MCP caller."""
    private_diagnostic(component, error)
    public = classify_error(error)
    return json.dumps({
        "ok": False,
        "error": {
            "code": code or public.code,
            "message": str(error) if code else public.message,
        },
    })


def get_manager(auto_download: bool = True) -> TorManager:
    global _manager
    if _manager is None:
        _manager = TorManager(auto_download=auto_download)
        # Load existing bridges
        _manager.load_bridges()
    return _manager


# ── Tool handlers ─────────────────────────────────────────────

def tor_download() -> str:
    """Download the Tor Expert Bundle for the current platform.

    One-time setup. Downloads ~22-32MB. Subsequent calls return
    immediately if already installed.
    """
    authorize(NetworkChannel.TOR_BOOTSTRAP)
    try:
        mgr = get_manager()
        mgr.ensure_installed()
        return json.dumps({
            "ok": True,
            "installed": True,
        })
    except Exception as e:
        return _error(e, "tor_download")


def tor_start(socks_port: int = 9050, timeout: float = 60.0) -> str:
    """Start the Tor daemon with configured bridges.

    Bridges are loaded from ~/.hermes/tor/bridges.txt.
    If no bridges are configured, Tor uses public relays.

    Get bridges from @GetBridgesBot on Telegram, then use
    tor_add_bridge to configure them before starting.
    """
    authorize(NetworkChannel.TOR_BOOTSTRAP)
    try:
        mgr = get_manager()
        mgr.socks_port = socks_port
        mgr.load_bridges()
        status = mgr.start(timeout=timeout)
    except Exception as exc:
        return _error(exc, "tor_start")
    if status.error:
        return _error(status.error, "tor_start", status.error_code)
    return json.dumps({
        "ok": True,
        "state": status.state.name,
        "socks_proxy_url": status.socks_proxy_url,
        "process_healthy": status.process_healthy,
        "socks_healthy": status.socks_healthy,
        "bootstrap_percent": status.bootstrap_percent,
        "bootstrap_complete": status.bootstrap_complete,
        "external_route_verified": status.external_route_verified,
        "bridge_count": status.bridge_count,
        "uptime_seconds": status.uptime_seconds,
        "error": status.error,
    })


def tor_stop() -> str:
    """Stop the Tor daemon."""
    try:
        mgr = get_manager()
        status = mgr.stop()
        return json.dumps({"ok": True, "state": status.state.name})
    except Exception as exc:
        return _error(exc, "tor_stop")


def tor_status() -> str:
    """Get current Tor daemon status including bridge count and uptime."""
    try:
        mgr = get_manager()
        status = mgr.status()
        return json.dumps({
            "ok": True,
            "state": status.state.name,
            "socks_proxy_url": status.socks_proxy_url,
            "process_healthy": status.process_healthy,
            "socks_healthy": status.socks_healthy,
            "bootstrap_percent": status.bootstrap_percent,
            "bootstrap_complete": status.bootstrap_complete,
            "external_route_verified": status.external_route_verified,
            "bridge_count": status.bridge_count,
            "uptime_seconds": status.uptime_seconds,
            "error": status.error,
        })
    except Exception as exc:
        return _error(exc, "tor_status")


def tor_verify() -> str:
    """Verify traffic routes through Tor.

    Requires Tor's structured HTTPS API and an independent HTTPS service to
    report the same exit address through the SOCKS5 proxy.
    """
    try:
        mgr = get_manager()
        authorize(NetworkChannel.MCP, proxy_url=mgr.socks_proxy_url)
        result = mgr.verify()
    except Exception as exc:
        return _error(exc, "tor_verify")
    if result.error:
        return _error(result.error, "tor_verify")
    return json.dumps({
        "ok": True,
        "using_tor": result.using_tor,
        "is_anonymous": result.is_anonymous,
    })


def tor_add_bridge(bridge_line: str) -> str:
    """Add a Tor bridge line to the configuration.

    Bridges are persisted to ~/.hermes/tor/bridges.txt.
    After adding bridges, restart Tor with tor_stop + tor_start
    to use the new bridges.

    Get bridges from:
      1. Telegram: @GetBridgesBot (send /bridges)
      2. Web: https://bridges.torproject.org/bridges?transport=obfs4

    Example bridge lines:
      obfs4 1.2.3.4:443 FINGERPRINT cert=... iat-mode=0
    """
    try:
        mgr = get_manager()
        result = mgr.add_bridge(bridge_line)
    except Exception as exc:
        return _error(exc, "tor_add_bridge")
    return json.dumps({
        "ok": True,
        "added": result.added,
        "total_bridges": result.total_bridges,
        "error": result.error,
        "hint": "Restart Tor with tor_stop + tor_start to use new bridges",
    })


# ── MCP Tool definitions ──────────────────────────────────────

TOOLS = [
    {
        "name": "tor_download",
        "description": "Download the Tor Expert Bundle (~22-32MB). One-time setup for anonymous routing.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "tor_start",
        "description": "Start the Tor daemon with configured bridges. Loads bridges from ~/.hermes/tor/bridges.txt. Blocks until bootstrapped or timeout.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "socks_port": {
                    "type": "integer",
                    "description": "SOCKS5 port (default: 9050)",
                    "default": 9050,
                },
                "timeout": {
                    "type": "number",
                    "description": "Bootstrap timeout in seconds (default: 60)",
                    "default": 60.0,
                },
            },
        },
    },
    {
        "name": "tor_stop",
        "description": "Stop the Tor daemon gracefully.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "tor_status",
        "description": "Get layered Tor health: managed process, SOCKS handshake, authenticated bootstrap state, and externally verified route.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "tor_verify",
        "description": "Verify traffic is routing through Tor by checking check.torproject.org. Reports exit IP and anonymity status.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "tor_add_bridge",
        "description": "Add a Tor bridge line. Bridges are persisted to ~/.hermes/tor/bridges.txt. Get bridges from @GetBridgesBot on Telegram.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "bridge_line": {
                    "type": "string",
                    "description": "Full bridge line, e.g. 'obfs4 1.2.3.4:443 FINGERPRINT cert=... iat-mode=0'",
                },
            },
            "required": ["bridge_line"],
        },
    },
]

HANDLERS = {
    "tor_download": tor_download,
    "tor_start": tor_start,
    "tor_stop": tor_stop,
    "tor_status": tor_status,
    "tor_verify": tor_verify,
    "tor_add_bridge": tor_add_bridge,
}


# ── MCP Server entry point ────────────────────────────────────


def serve():
    """Run the MCP server via stdio.

    Hermes connects via:
      hermes mcp add darkloom --command "python -m darkloom.mcp_server"
    """
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
    except ImportError:
        print(
            "MCP SDK not installed. Run: pip install mcp\n"
            "Or install with: uv pip install -e '.[mcp]'",
            file=sys.stderr,
        )
        sys.exit(1)

    server = Server("darkloom")

    for tool_def in TOOLS:
        handler = HANDLERS[tool_def["name"]]
        name = tool_def["name"]
        desc = tool_def["description"]

        # Register with closure capturing the handler
        def _register(n, d, h):
            @server.tool(n, d)
            def tool_fn(**kwargs):
                return h(**kwargs)

        _register(name, desc, handler)

    import asyncio

    async def run():
        async with stdio_server() as (read, write):
            await server.run(read, write)

    asyncio.run(run())


if __name__ == "__main__":
    serve()
