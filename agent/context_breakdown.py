"""Live session context-window breakdown for UI surfaces.

The UI breakdown is rendered from ``agent.token_ledger`` so the same source
attribution used for report-only P1 telemetry feeds operator-facing context
usage. Numbers remain rough estimates, matching Hermes' existing char/4
accounting rather than provider-specific tokenizers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.token_ledger import compute_token_ledger

_CATEGORY_COLORS = {
    "system_prompt": "var(--context-usage-system)",
    "tool_definitions": "var(--context-usage-tools)",
    "rules": "var(--context-usage-rules)",
    "skills": "var(--context-usage-skills)",
    "mcp": "var(--context-usage-mcp)",
    "subagent_definitions": "var(--context-usage-subagents)",
    "memory": "var(--context-usage-memory)",
    "conversation": "var(--context-usage-conversation)",
}


def compute_session_context_breakdown(
    agent: Any,
    messages: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Return a Cursor-style context usage breakdown for one live agent."""
    ledger = compute_token_ledger(agent, messages)
    return {
        "categories": [
            {
                "color": _CATEGORY_COLORS.get(segment.source, "var(--ui-text-tertiary)"),
                "id": segment.source,
                "label": segment.label,
                "tokens": segment.token_count,
            }
            for segment in ledger.segments
            if segment.token_count > 0
        ],
        "context_max": ledger.context_max,
        "context_percent": ledger.context_percent,
        "context_used": ledger.context_used,
        "estimated_total": ledger.estimated_total,
        "model": ledger.model,
    }
