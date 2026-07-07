"""Synchronous MCP write-tool gate at the Hermes transport layer."""

from __future__ import annotations

from typing import Any, Optional

from hermes_trader.audit.rate_limit import check_write_rate_limit
from hermes_trader.config import load_trader_config
from hermes_trader.risk.gate import is_kill_switch_active
from hermes_trader.risk.mandate import default_mandate_path, load_mandate, validate_mandate
from hermes_trader.tools import LIVE_WRITE_TOOLS


def intercept_mcp_tool_call(
    server_name: str,
    tool_name: str,
    args: dict[str, Any] | None = None,
) -> Optional[str]:
    """Block defi-trading write tools before they reach the MCP server.

    Returns an error message when the call must be rejected, else ``None``.
    """
    _ = args  # reserved for P2 argument-level checks
    if tool_name not in LIVE_WRITE_TOOLS:
        return None

    cfg = load_trader_config()
    if server_name != cfg.mcp_server_name:
        return None

    if is_kill_switch_active():
        return (
            "HERMES_TRADER_KILL_SWITCH active — "
            f"{tool_name} blocked by deterministic risk gate"
        )

    if cfg.mode == "paper":
        return (
            f"Paper mode blocks {tool_name}. "
            "Set mode: live in hermes_trader.yaml and sign mandate.json (P1)."
        )

    mandate = load_mandate(default_mandate_path())
    if mandate is None:
        return (
            f"{tool_name} blocked: mandate.json missing at {default_mandate_path()}. "
            "Sign a mandate before live execution."
        )

    ok, err = validate_mandate(mandate)
    if not ok:
        return f"{tool_name} blocked: mandate invalid ({err})"

    allowed, rate_msg = check_write_rate_limit(max_per_hour=cfg.max_write_tools_per_hour)
    if not allowed:
        return f"{tool_name} blocked: {rate_msg}"

    return None