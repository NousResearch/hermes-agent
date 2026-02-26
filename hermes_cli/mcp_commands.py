"""
CLI command: hermes mcp-server

Add to hermes_cli/commands.py:

    from hermes_cli.mcp_commands import click_mcp_server_command
    cli.add_command(click_mcp_server_command)
"""

from __future__ import annotations

import sys


def mcp_server_command(
    http: bool = False,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    try:
        from gateway.mcp_server import main
    except ImportError:
        import importlib.util
        import os
        spec_path = os.path.join(
            os.path.dirname(__file__), "..", "gateway", "mcp_server.py"
        )
        spec = importlib.util.spec_from_file_location("mcp_server", spec_path)
        if spec is None or spec.loader is None:
            print("Error: could not locate gateway/mcp_server.py", file=sys.stderr)
            sys.exit(1)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        main = mod.main

    main(http=http, host=host, port=port)


try:
    import click  # type: ignore[import]

    @click.command("mcp-server")
    @click.option("--http", is_flag=True, default=False, help="Use HTTP transport (default: stdio)")
    @click.option("--host", default="127.0.0.1", show_default=True, help="HTTP host")
    @click.option("--port", default=8765, show_default=True, type=int, help="HTTP port")
    def click_mcp_server_command(http: bool, host: str, port: int) -> None:
        """Start the Hermes MCP Server (stdio or HTTP transport)."""
        mcp_server_command(http=http, host=host, port=port)

except ImportError:
    click_mcp_server_command = None  # type: ignore[assignment]
