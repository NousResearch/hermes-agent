"""Adopt an operator's prompt refresh request on a continuing session's next turn.

A continuing session replays its stored system prompt verbatim so the provider prefix cache stays
warm, which also means SOUL.md, skills and project-context edits never reach it before ``/new``.
``hermes sessions refresh-prompts`` records a refresh epoch in the profile's state.db
(``hermes_state_prompt_refresh``). At the start of the session's next turn this module compares
that epoch with the one stamped when the session's prompt was built (``model_config``) and, when
the prompt is older, rebuilds it once through the normal builder: history, session id and the
tools[] pin are kept, ``on_session_start`` does not fire, and the new bytes plus the epoch are
persisted so every later turn restores them verbatim. One prefix-cache break per request.

No request recorded means the stored prompt is reused exactly as before: the only extra work is
one ``state_meta`` point read. Nothing watches files and nothing changes mid-turn.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# model_config key: the refresh epoch the session's stored prompt already satisfies.
SESSION_EPOCH_KEY = "prompt_refresh_epoch"


def requested_epoch(agent: Any) -> int:
    """Newest refresh epoch recorded for the agent's session; 0 without a request (or a store that
    keeps none, such as a managed worker's facade)."""
    reader = getattr(getattr(agent, "_session_db", None), "prompt_refresh_epoch", None)
    session_id = getattr(agent, "session_id", None)
    if reader is None or not session_id:
        return 0
    try:
        epoch = reader(session_id)
    except Exception:
        logger.warning("Prompt refresh epoch read failed (session=%s); reusing the stored prompt.",
                       session_id, exc_info=True)
        return 0
    return epoch if isinstance(epoch, int) else 0


def _applied_epoch(session_row: Any) -> int:
    from hermes_state_sessions import _parse_model_config

    value = _parse_model_config((session_row or {}).get("model_config")).get(SESSION_EPOCH_KEY)
    return value if isinstance(value, int) else 0


def stored_prompt_refresh_epoch(agent: Any, session_row: Any) -> int:
    """The epoch to rebuild the stored prompt at, or 0 when the stored prompt is current."""
    epoch = requested_epoch(agent)
    if epoch and _applied_epoch(session_row) < epoch:
        return epoch
    agent._prompt_refresh_epoch = epoch
    return 0


def stamp_prompt_epoch(agent: Any, epoch: int) -> None:
    """Record that the agent's freshly built prompt satisfies ``epoch`` (read BEFORE the build, so a
    request that lands mid-build still fires on the next turn)."""
    agent._prompt_refresh_epoch = epoch
    if epoch:
        _persist_epoch(agent, epoch)


def _persist_epoch(agent: Any, epoch: int) -> bool:
    db = getattr(agent, "_session_db", None)
    if not db:
        return True
    try:
        db.patch_session_model_config(agent.session_id, {SESSION_EPOCH_KEY: epoch})
    except Exception:
        logger.warning("Could not record the prompt refresh epoch (session=%s); the refresh re-fires "
                       "next turn.", agent.session_id, exc_info=True)
        return False
    return True


def rebuild_for_refresh(agent: Any, system_message: Any, conversation_history: Any, epoch: int) -> None:
    """Rebuild the session's prompt from current sources and persist it with ``epoch``."""
    from agent.prompt_builder import clear_skills_system_prompt_cache
    from agent.surface_switch import stage_surface_switch_note
    from hermes_cli.observability.shared_metrics_efficiency import record_cache_break

    logger.info("Operator requested a system prompt refresh (epoch %d) for session %s; rebuilding it "
                "from current SOUL.md, skills and context files (one-time prefix-cache break).",
                epoch, agent.session_id)
    # Same boundary compaction uses: memory reloads from disk, plugin sections re-render.
    agent._invalidate_system_prompt()
    agent._auto_load_skills_resolved = False  # pinned skills (skills.auto_load) re-read too
    # As in the Bot Chat capability refresh: build the index from disk, not a cached render.
    clear_skills_system_prompt_cache(clear_snapshot=True)
    agent._cached_system_prompt = agent._build_system_prompt(system_message)
    record_cache_break(agent, "system_prompt_rebuild")
    stage_surface_switch_note(agent, agent._cached_system_prompt, conversation_history)
    db = getattr(agent, "_session_db", None)
    if db:
        try:
            db.update_system_prompt(agent.session_id, agent._cached_system_prompt)
        except Exception:
            logger.warning("Could not persist the refreshed system prompt (session=%s); the refresh "
                           "re-fires next turn.", agent.session_id, exc_info=True)
            return
    if _persist_epoch(agent, epoch):
        agent._prompt_refresh_epoch = epoch


def refresh_warm_prompt(agent: Any, system_message: Any, conversation_history: Any) -> None:
    """An agent kept warm across turns (gateway agent cache, TUI/Desktop) adopts a refresh request
    recorded after its prompt was set. Agents whose prompt did not come from the restore/build path
    (seeded maintenance agents, forks) carry no epoch and are left alone."""
    seen = getattr(agent, "_prompt_refresh_epoch", None)
    if agent._cached_system_prompt is None or not isinstance(seen, int) or not conversation_history:
        return
    epoch = requested_epoch(agent)
    if epoch > seen:
        rebuild_for_refresh(agent, system_message, conversation_history, epoch)
