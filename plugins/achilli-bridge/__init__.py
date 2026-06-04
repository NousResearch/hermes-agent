"""
Achilli Bridge -- MCP tool chain resilience, circuit breaker, format translation.

Uses the post_tool_call hook (available in Hermes >= 0.15.2 / fork) to observe
all tool call results. Tracks per-MCP-server failure rates and manages circuit
breaker state (closed -> open -> half_open -> closed).

Compatible with Hermes Agent >= 0.15.2 (fork with post_tool_call hook).
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


def _get_server(tool_name: str) -> Dict[str, Any]:
    """Return (and lazily initialise) the state dict for a given MCP server."""
    if tool_name not in _SERVER_STATE:
        _SERVER_STATE[tool_name] = {
            "state": "closed",
            "failures": 0,
            "last_failure": None,
            "opened_at": None,
            "total_calls": 0,
            "total_failures": 0,
        }
    return _SERVER_STATE[tool_name]


def _is_mcp_tool(tool_name: str) -> bool:
    """Heuristic: MCP tools are prefixed with their server name (e.g. 'mcp_...')."""
    return tool_name.startswith("mcp_")


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def _on_post_tool_call(
    tool_name: str = "",
    result: Any = None,
    status: str = "",
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    duration_ms: int = 0,
    **_: Any,
) -> None:
    """Observe every tool call result and update circuit breaker state."""
    if not tool_name or not _is_mcp_tool(tool_name):
        return

    server = _get_server(tool_name)
    server["total_calls"] += 1

    is_error = status == "error" or error_type is not None

    if is_error:
        server["total_failures"] += 1
        server["failures"] += 1
        server["last_failure"] = time.time()
        logger.debug(
            "bridge: tool=%s failure #%d type=%s",
            tool_name, server["failures"], error_type,
        )

        # Check if circuit should open
        if server["failures"] >= _get_failure_threshold():
            if server["state"] != "open":
                server["state"] = "open"
                server["opened_at"] = time.time()
                logger.warning(
                    "bridge: circuit OPEN for %s after %d consecutive failures",
                    tool_name, server["failures"],
                )
    else:
        # Success: reset consecutive failures
        if server["state"] == "open":
            # Check if cooldown has elapsed
            cooldown = _get_cooldown()
            opened_at = server.get("opened_at") or 0
            if time.time() - opened_at >= cooldown:
                server["state"] = "half_open"
                logger.info(
                    "bridge: circuit HALF_OPEN for %s (cooldown elapsed)", tool_name
                )
        elif server["state"] == "half_open":
            # Successful call in half_open -> close the circuit
            server["state"] = "closed"
            server["failures"] = 0
            server["opened_at"] = None
            logger.info("bridge: circuit CLOSED for %s (recovery confirmed)", tool_name)
        else:
            # Normal closed state: reset consecutive failure count
            server["failures"] = 0


def _on_subagent_stop(
    child_role: Optional[str] = None,
    child_status: Optional[str] = None,
    duration_ms: int = 0,
    **_: Any,
) -> None:
    """Track delegation completion for bridge status reporting."""
    logger.debug(
        "bridge: observed subagent_stop role=%s status=%s",
        child_role, child_status,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _bridge_status_tool(*_, **__) -> dict:
    """Return bridge/circuit-breaker status."""
    active = any(
        s.get("state") in ("open", "half_open")
        for s in _SERVER_STATE.values()
    )
    return {
        "status": "active" if _SERVER_STATE else "active (no MCP calls yet)",
        "loaded": True,
        "circuit_breakers": {
            name: {
                "state": info.get("state", "closed"),
                "consecutive_failures": info.get("failures", 0),
                "total_calls": info.get("total_calls", 0),
                "total_failures": info.get("total_failures", 0),
                "last_failure": info.get("last_failure"),
            }
            for name, info in _SERVER_STATE.items()
        },
        "config": {
            "failure_threshold": _get_failure_threshold(),
            "cooldown_s": _get_cooldown(),
        },
        "any_open_circuits": active,
    }


def _achilli_bridge_status_tool(*_, **__) -> str:
    """Human-readable bridge status."""
    status = _bridge_status_tool()
    lines = [
        "# Achilli Bridge Status",
        "",
        "Status: **{}**".format(status["status"]),
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
            lines.append("- **{}**: {} ({} consecutive, {} total / {} failures)".format(
                name, cb["state"], cb["consecutive_failures"],
                cb["total_calls"], cb["total_failures"],
            ))
    else:
        lines.append("- No MCP tool calls observed yet")

    if status.get("any_open_circuits"):
        lines.extend(["", "⚠ At least one circuit is open or half-open."])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("subagent_stop", _on_subagent_stop)
    ctx.register_tool(
        name="bridge_status",
        description=(
            "Return MCP server circuit breaker status and health. "
            "(achilli-bridge)"
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
