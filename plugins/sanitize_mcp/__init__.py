"""
sanitize-mcp plugin — MCP tool output validation.

Hooks into ``transform_tool_result`` to scan MCP tool outputs for prompt
injection before they enter the LLM context.

Behaviour:
- Only intercepts tools whose name starts with ``mcp__``
- Runs the result string through ``core/sanitize.sanitize_input(channel='mcp')``
- If ``trust_score < 0.3`` → replaces result with ``[MCP OUTPUT BLOCKED]``
- Everything else passes through unchanged (including non-string results)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Blocked message — opaque, no clues to attacker
_BLOCKED_MSG = "[MCP OUTPUT BLOCKED]"

# Threshold: trust score below this = output replaced
_TRUST_THRESHOLD = 0.3


def _is_mcp_tool(tool_name: str) -> bool:
    """Check if tool_name belongs to an MCP server."""
    return tool_name.startswith("mcp__")


def _on_transform_tool_result(
    tool_name: str = "",
    args: Optional[dict] = None,
    result: Any = None,
    **_: Any,
) -> Optional[str]:
    """Rewrite MCP tool outputs that fail sanitization.

    Returns:
        A replacement string if the output was blocked, or None to pass through.
    """
    if not _is_mcp_tool(tool_name):
        return None

    if not isinstance(result, str):
        return None

    # MCP output is DATA (tool result, not user instruction)
    try:
        from core.sanitize import sanitize_input

        san = sanitize_input(
            result,
            channel="mcp",
            is_data=True,
            enable_semantic=True,
        )
    except ImportError:
        logger.warning("core.sanitize not available — MCP output validation skipped")
        return None
    except Exception as exc:
        logger.debug("sanitize_input error on MCP output: %s", exc)
        return None

    if san.blocked or san.trust_score < _TRUST_THRESHOLD:
        logger.info(
            "MCP output blocked (trust=%.2f, patterns=%s): tool=%s",
            san.trust_score,
            san.redacted_patterns,
            tool_name,
        )
        return _BLOCKED_MSG

    # If sanitize modified the text (redacted something) but didn't block,
    # return the sanitized version
    if san.text != result:
        logger.info(
            "MCP output sanitized (trust=%.2f, patterns=%s): tool=%s",
            san.trust_score,
            san.redacted_patterns,
            tool_name,
        )
        return san.text

    return None


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the transform_tool_result hook."""
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)
