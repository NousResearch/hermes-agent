"""Carry per-session state from a compression parent onto its child session.

Sibling of ``agent/conversation_compression.py`` (the facade), which late-imports this module at its
single call site; this module late-imports the facade's ``_swallow`` inside the function (no
module-level cycle).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _carry_session_state_to_child(agent: Any, old_session_id: str, old_title: Any) -> None:
    """Migrate /goal, /heartbeat, /loop state, rejected-thinking fingerprints and the title from the parent to the child.
    Each lookup is a flat per-session read with no parent walk, so state would silently die at the boundary. The title
    is carried unchanged (renumbering per rotation made one session look like many); its provenance is read BEFORE the
    transfer clears the ancestor's row, then restored so an inherited auto-title stays upgradeable.
    """
    from agent.conversation_compression import _swallow

    # /goal, /heartbeat, /loop: carried locally, or atomically by the owner on a RuntimeSessionStore.
    from agent.compression_automation import carry_compression_automation
    carry_compression_automation(agent, old_session_id)
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
        logger.debug("Could not propagate title on compression: %s", e)
        return
    # set_session_title() records "user"; restore the original authority.
    if _src is not None:
        with _swallow('Could not propagate title provenance: %s'):
            agent._session_db.set_session_title_source(agent.session_id, _src)
