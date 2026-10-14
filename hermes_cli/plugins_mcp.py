"""Result normalization for ``PluginContext.call_mcp`` (moved out of the ``plugins.py`` facade)."""

from __future__ import annotations

import json
from typing import Any

MCP_RESULT_CHAR_CAP = 65536


def mcp_envelope(raw: Any) -> dict[str, Any]:
    """Normalize an MCP handler result string into a stable envelope."""
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    truncated = len(raw) > MCP_RESULT_CHAR_CAP
    if truncated:
        raw = raw[:MCP_RESULT_CHAR_CAP] + "… [truncated]"
    try:
        parsed: Any = json.loads(raw)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and "error" in parsed:
        envelope: dict[str, Any] = {"ok": False, "error": parsed["error"]}
    elif isinstance(parsed, dict) and "result" in parsed:
        envelope = {"ok": True, "result": parsed["result"]}
        if "structuredContent" in parsed:
            envelope["structuredContent"] = parsed["structuredContent"]
    else:
        envelope = {"ok": True, "result": parsed if parsed is not None else raw}
    return {**envelope, "truncated": True} if truncated else envelope
