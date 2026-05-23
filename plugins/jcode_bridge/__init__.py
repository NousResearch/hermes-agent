"""Hermes <-> jcode bridge plugin.

The first bridge is deliberately narrow: call jcode's documented wrapper CLI
from Hermes, parse its JSON-oriented output, and keep all integration code
outside both projects' core agent loops.
"""

from __future__ import annotations

from plugins.jcode_bridge.tools import (
    JCODE_CONTRACT_CHECK_SCHEMA,
    JCODE_RUN_SCHEMA,
    JCODE_STATUS_SCHEMA,
    handle_jcode_contract_check,
    handle_jcode_run,
    handle_jcode_status,
)

try:
    from plugins.jcode_bridge.webhook_dispatch import on_pre_gateway_dispatch
except ModuleNotFoundError as exc:
    if exc.name != "gateway":
        raise
    on_pre_gateway_dispatch = None


def register(ctx) -> None:
    """Register jcode bridge tools."""
    ctx.register_tool(
        name="jcode_run",
        toolset="jcode",
        schema=JCODE_RUN_SCHEMA,
        handler=handle_jcode_run,
    )
    ctx.register_tool(
        name="jcode_status",
        toolset="jcode",
        schema=JCODE_STATUS_SCHEMA,
        handler=handle_jcode_status,
    )
    ctx.register_tool(
        name="jcode_contract_check",
        toolset="jcode",
        schema=JCODE_CONTRACT_CHECK_SCHEMA,
        handler=handle_jcode_contract_check,
    )
    if on_pre_gateway_dispatch is not None:
        ctx.register_hook("pre_gateway_dispatch", on_pre_gateway_dispatch)
