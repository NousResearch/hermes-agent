"""Carry SQLite session-scoped state across compression rotation."""

from __future__ import annotations

from typing import Any


def carry_session_state_to_child(agent: Any, old_session_id: str, old_title: Any, *, _swallow: Any, logger: Any) -> None:
    """Migrate /goal, /heartbeat, /loop state, rejected-thinking fingerprints and the title from the parent to the child.
    Each lookup is a flat per-session read with no parent walk, so state would silently die at the boundary. The title
    is carried unchanged (renumbering per rotation made one session look like many); its provenance is read BEFORE the
    transfer clears the ancestor's row, then restored so an inherited auto-title stays upgradeable.
    """
    # PostgreSQL moves controls inside publish_compression_child's transaction.
    # The post-publication carrier below is therefore strictly the established
    # SQLite compatibility path; a PG retry here would create duplicate active
    # controls after an acknowledged publication.
    try:
        from hermes_cli.config import load_config
        from state_store import resolve_state_store_config
        if resolve_state_store_config(load_config() or {}).backend == "postgresql":
            return
    except ImportError:
        pass
    with _swallow('Could not migrate goal on compression: %s'):
        from hermes_cli.goals import migrate_goal_to_session
        migrate_goal_to_session(old_session_id, agent.session_id, reason="compression")
    with _swallow('Could not migrate heartbeat on compression: %s'):
        from hermes_cli.heartbeat import migrate_heartbeat_to_session
        migrate_heartbeat_to_session(old_session_id, agent.session_id)
    with _swallow('Could not migrate loop on compression: %s'):
        from hermes_cli.loops import migrate_loop_to_session
        migrate_loop_to_session(old_session_id, agent.session_id, reason="compression")
    with _swallow('Could not carry rejected thinking on compression: %s'):
        from agent.anthropic_thinking_replay import carry_rejected_thinking_to_session
        carry_rejected_thinking_to_session(agent, old_session_id)
    if not old_title:
        return
    _src = None
    with _swallow('Could not read title provenance: %s'):
        _src = agent._session_db.get_session_title_source(old_session_id)
    try:
        agent._session_db.set_session_title(agent.session_id, old_title)
    except Exception as e:
        logger.debug("Could not propagate title on compression: %s", e, exc_info=True)
        return
    # set_session_title() records "user"; restore the original authority.
    if _src is not None:
        with _swallow('Could not propagate title provenance: %s'):
            agent._session_db.set_session_title_source(agent.session_id, _src)
