"""The proactive tool-result prune's two call sites: after a tool round (every iteration where full
compression stands down) and at turn start (cold provider cache only, where most expiries happen:
a person comes back after a pause). Both share the safety gates and the commit contract.
Nothing here imports ``agent.conversation_loop`` at module level (cycle)."""

from __future__ import annotations

import logging
from typing import Any

from agent.conversation_compression import _reset_read_dedup_caches
from agent.prompt_caching import provider_cache_expired

logger = logging.getLogger("agent.conversation_loop")


def run_proactive_prune(
    agent: Any, messages: list[dict[str, Any]], current_tokens: int, effective_task_id: Any,
) -> list[dict[str, Any]]:
    """Run the post-tool prune helper once (same gates, suppression accounting and commit contract);
    reset the read dedup on a commit. Returns the list the loop continues with."""
    from agent.turn_preflight import _proactive_tool_result_prune

    pruned = _proactive_tool_result_prune(agent, agent.context_compressor, messages, current_tokens)
    if pruned is not messages:
        # A committed prune is a content-loss boundary like compaction: demoted skill_view / read_file bodies
        # survive only as one-line markers, so the repeat-read dedup must stop answering "unchanged" for them
        # or the reload the marker asks for is refused.
        _reset_read_dedup_caches(effective_task_id, session_id=agent.session_id or "")
    return pruned


def turn_start_cold_prune(agent: Any, out: Any, effective_task_id: Any) -> None:
    """Turn-start prune, only on an already-expired provider cache (the rewrite then costs no cache break).

    Runs on the history BEFORE this turn's user row, which is not persisted yet: the commit never writes it,
    so the turn-start persist and ``api_content`` stamp handle it exactly as on an unpruned turn. The prune
    replaces bodies in place and never drops rows, so ``current_turn_user_idx`` stays valid.
    """
    idx = out.current_turn_user_idx
    if (out.compressed or out.blocked or not agent.compression_enabled
            or not isinstance(idx, int) or not 0 < idx < len(out.messages)
            or not provider_cache_expired(agent, out.messages[:idx])):
        return
    from agent import turn_context as _tc

    prefix = out.messages[:idx]
    tokens = _tc._preflight_request_tokens(agent, out.messages, out.active_system_prompt or "")
    pruned = run_proactive_prune(agent, prefix, tokens, effective_task_id)
    if pruned is not prefix and len(pruned) == len(prefix):
        out.messages = pruned + out.messages[idx:]
