"""stdio MCP server for MoneyPrinterTurbo Video Studio.

Run:
  python -m capabilities.moneyprinter.mcp.server

Register with Hermes:
  hermes mcp add moneyprinter --command \"python -m capabilities.moneyprinter.mcp.server\"
  hermes mcp test moneyprinter
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Optional

logger = logging.getLogger("moneyprinter.mcp")


def create_server() -> Any:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "moneyprinter MCP server requires the 'mcp' package. "
            f"Install with: {sys.executable} -m pip install 'mcp'"
        ) from exc

    from capabilities.moneyprinter.mcp import tools as mp_tools

    mcp = FastMCP(
        "moneyprinter",
        instructions=(
            "MoneyPrinterTurbo video generation capability for Hermes. "
            "Use moneyprinter_generate_video to create a task, then poll "
            "moneyprinter_get_task until complete. Do not block waiting for "
            "full render in a single turn when possible. Prefer 9:16 short "
            "videos with subtitles on by default."
        ),
    )

    # Register each tool with FastMCP. Prefer add_tool; fall back to decorator.
    for spec in mp_tools.TOOL_SPECS:
        name = spec["name"]
        description = spec["description"]
        handler = spec["fn"]
        try:
            mcp.add_tool(handler, name=name, description=description)
        except TypeError:
            mcp.tool(name=name, description=description)(handler)
        except Exception:
            # Some FastMCP versions only support the decorator form.
            mcp.tool(name=name, description=description)(handler)

    logger.info("moneyprinter MCP server registered %d tools", len(mp_tools.TOOL_SPECS))
    return mcp


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or sys.argv[1:]
    verbose = "--verbose" in argv or "-v" in argv
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        stream=sys.stderr,  # MCP protocol uses stdout
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        server = create_server()
    except ImportError as exc:
        sys.stderr.write(f"moneyprinter MCP cannot start: {exc}\n")
        return 2

    # FastMCP defaults to stdio transport when run as a subprocess.
    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
