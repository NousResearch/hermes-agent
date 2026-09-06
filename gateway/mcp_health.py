"""Sanitized MCP discovery health persisted by the messaging gateway."""

from __future__ import annotations

from typing import Any, Iterable


def summarize_mcp_health(statuses: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-server discovery status into a local runtime-health receipt."""
    enabled = [entry for entry in statuses if not entry.get("disabled")]
    connected = [entry for entry in enabled if entry.get("connected")]
    failed = [entry for entry in enabled if entry.get("status") == "failed"]
    return {
        "status": "degraded" if failed else "ok",
        "configured_servers": len(enabled),
        "connected_servers": len(connected),
        "failed_servers": len(failed),
        "failures": [
            {"name": str(entry.get("name") or "unknown"), "error": str(entry.get("error") or "unknown error")}
            for entry in failed
        ],
    }


__all__ = ["summarize_mcp_health"]