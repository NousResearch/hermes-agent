"""Fold an agent-as-provider's own activity back into Hermes' turn state.

Agent providers (ACP CLI shims, the codex app-server) run their own tools, so that
work must never come back as pending ``tool_calls`` (Hermes would re-run it) — but
the self-improvement loop (replays ``messages``) and the skill-review nudge
(``_iters_since_skill`` counter) go blind if it is merely summarised into
``reasoning``. The client hands back ``hermes_projected_messages`` (completed
assistant/tool rows) and ``hermes_provider_tool_iterations`` on the completion
object; this helper applies them append-only via ``append_message`` (timestamped,
persisted). Ordinary OpenAI-compatible clients set neither and are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.message_metadata import append_message

logger = logging.getLogger(__name__)

__all__ = ["splice_provider_projection"]


def splice_provider_projection(agent: Any, response: Any, messages: list[dict[str, Any]]) -> int:
    """Append the provider's projected history rows and tick the nudge counter.

    Returns the number of rows spliced. Tolerates absent/garbage attributes so a
    third-party OpenAI-compatible client can't break the turn.
    """
    projected = getattr(response, "hermes_projected_messages", None)
    rows = [m for m in projected if isinstance(m, dict)] if isinstance(projected, list) else []
    for row in rows:
        append_message(messages, row)
    if rows:
        logger.debug(
            "spliced %d provider-projected transcript row(s) from %s", len(rows), getattr(agent, "provider", "?"),
        )

    try:
        iterations = int(getattr(response, "hermes_provider_tool_iterations", 0) or 0)
    except (TypeError, ValueError):
        iterations = 0
    if iterations > 0:
        agent._iters_since_skill = getattr(agent, "_iters_since_skill", 0) + iterations

    return len(rows)
