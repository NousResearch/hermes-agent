"""cube_sandbox plugin — routes terminal and execute_code to CubeSandbox microVMs."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    from .checks import check_cube_high_risk_requirements
    from .config import is_plugin_active
    from .handlers import handle_execute_code, handle_terminal
    from tools.code_execution_tool import EXECUTE_CODE_SCHEMA
    from tools.terminal_tool import TERMINAL_SCHEMA

    if not is_plugin_active():
        logger.debug("cube_sandbox plugin skipped (SANDBOX_TYPE is not 'cube')")
        return

    ctx.register_tool(
        name="terminal",
        toolset="terminal",
        schema=TERMINAL_SCHEMA,
        handler=handle_terminal,
        check_fn=check_cube_high_risk_requirements,
        override=True,
        emoji="💻",
    )
    ctx.register_tool(
        name="execute_code",
        toolset="code_execution",
        schema=EXECUTE_CODE_SCHEMA,
        handler=handle_execute_code,
        check_fn=check_cube_high_risk_requirements,
        override=True,
        emoji="🐍",
    )

    logger.info(
        "cube_sandbox plugin registered: terminal + execute_code → Cube; "
        "file tools remain on host workspace (split mode)"
    )
