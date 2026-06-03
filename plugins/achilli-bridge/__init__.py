"""
Achilli Bridge -- MCP tool chain resilience, circuit breaker, format translation.

IMPORTANT: The core hook this plugin needs (post_tool_call) is DECLARED in
Hermes VALID_HOOKS but NEVER INVOKED in the Python runtime. This plugin
loads, registers its tools, and does not crash -- but the interception logic
cannot function until Hermes fixes the missing invoke_hook("post_tool_call")
call site.

Hermes needs to add the invocation in the tool dispatch path after a tool
returns but before the result is appended to the conversation messages.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-server circuit breaker state
# ---------------------------------------------------------------------------

_SERVER_STATE: Dict[str, Dict[str, Any]] = {}


def _get_failure_threshold() -> int:
    import os
    try:
        return int(os.environ.get("ACHILLI_BRIDGE_FAILURE_THRESHOLD", "3"))
    except (ValueError, TypeError):
        return 3


def _get_cooldown() -> float:
    import os
    try:
        return float(os.environ.get("ACHILLI_BRIDGE_COOLDOWN", "60"))
    except (ValueError, TypeError):
        return 60.0


# ---------------------------------------------------------------------------
# Hook -- subagent_stop is the only one that fires reliably
# ---------------------------------------------------------------------------


def _on_subagent_stop(
    child_role: Optional[str] = None,
    child_status: Optional[str] = None,
    duration_ms: int = 0,
    **_: Any,
) -> None:
    """Track delegation completion for bridge status reporting."""
    # Without post_tool_call, we can't intercept MCP tool calls directly.
    # We at least track delegation activity as a proxy for MCP usage.
    logger.debug(
        "bridge: observed subagent_stop role=%s status=%s",
        child_role, child_status,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _bridge_status_tool(*_, **__) -> dict:
    """Return bridge/circuit-breaker status. Limited without post_tool_call."""
    return {
        "status": "degraded",
        "reason": "post_tool_call hook not invoked in Hermes core",
        "loaded": True,
        "circuit_breakers": {
            name: {
                "state": info.get("state", "closed"),
                "consecutive_failures": info.get("failures", 0),
                "last_failure": info.get("last_failure"),
            }
            for name, info in _SERVER_STATE.items()
        },
        "config": {
            "failure_threshold": _get_failure_threshold(),
            "cooldown_s": _get_cooldown(),
        },
        "note": (
            "Full circuit breaker and format translation features require "
            "post_tool_call hook, which Hermes does not currently invoke. "
            "This tool reports static configuration only."
        ),
    }


def _achilli_bridge_status_tool(*_, **__) -> str:
    """Human-readable bridge status."""
    status = _bridge_status_tool()
    lines = [
        "# Achilli Bridge Status",
        "",
        "Status: **{}**".format(status["status"]),
        "Reason: {}".format(status["reason"]),
        "",
        "## Configuration",
        "- Failure threshold: {} consecutive failures".format(
            status["config"]["failure_threshold"]
        ),
        "- Cooldown: {}s".format(status["config"]["cooldown_s"]),
        "",
        "## Circuit Breakers",
    ]
    if status["circuit_breakers"]:
        for name, cb in status["circuit_breakers"].items():
            lines.append("- **{}**: {} ({} failures)".format(
                name, cb["state"], cb["consecutive_failures"]
            ))
    else:
        lines.append("- No MCP servers tracked yet")

    lines.extend(["", "Note: {}".format(status["note"])])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_tool(
        name="bridge_status",
        description=(
            "Return MCP server circuit breaker status and health. "
            "(achilli-bridge, limited: post_tool_call not invoked in core)"
        ),
        parameters={"type": "object", "properties": {}},
        handler=_bridge_status_tool,
    )
    ctx.register_tool(
        name="achilli_bridge_status",
        description="Human-readable bridge status report (achilli-bridge)",
        parameters={"type": "object", "properties": {}},
        handler=_achilli_bridge_status_tool,
    )
