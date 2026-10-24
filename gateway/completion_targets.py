"""Read-only admission rules for gateway completion targets."""

import json
import logging

logger = logging.getLogger("gateway.run")


def is_delegate_target(row: dict) -> bool:
    """Identify delegate lineage independently of its current transport.

    Transport metadata can overwrite source with discord/telegram. The creation
    source and inherited delegation marker still identify the owner.
    """
    if row.get("source") == "subagent" or row.get("created_source") == "subagent":
        return True
    config = row.get("model_config")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except ValueError:
            return False
    return isinstance(config, dict) and bool(config.get("_delegate_from"))


async def delegate_route_verdict(session_db, target_id: str, current_id: str) -> str:
    """A delegate may wake only a route already bound to its conversation.

    Explicit /resume is allowed, including subsequent compression. A descendant
    relation alone is not ownership: a delegate is NOT its parent's continuation.
    """
    if not current_id:
        return "terminal"
    if current_id == target_id:
        return "deliver"
    try:
        current_tip = await session_db.get_compression_tip(current_id)
        target_tip = await session_db.get_compression_tip(target_id)
    except Exception:
        logger.debug("Delegate notification lineage lookup failed", exc_info=True)
        return "retry"
    return "deliver" if current_tip and current_tip == target_tip else "terminal"


async def _delegate_preflight(session_db, target_id, session_store, session_key):
    if session_store is None or not session_key:
        return "terminal"
    try:
        entry = await session_store.lookup_by_session_key(session_key)
    except Exception:
        logger.debug("Delegate notification route lookup failed", exc_info=True)
        return "retry"
    return await delegate_route_verdict(session_db, target_id, entry.session_id if entry else "")


async def classify_process_target_ownership(session_db, parent_session_id: str, *, session_store=None, session_key="") -> str:
    """Check the delegate boundary before the host classifies closure/compression.

    Children poll their processes or hand them to the parent. Their notifications
    must not switch the chat into a child transcript. Lifecycle disposition stays
    with the host owner classifier, including notes on closed local conversations.
    """
    if session_db is None:
        return "retry"
    try:
        parent = await session_db.get_session(parent_session_id)
    except Exception:
        logger.debug("Async-completion pre-flight parent lookup failed for %s", parent_session_id, exc_info=True)
        return "retry"
    if parent is None:
        return "terminal"
    if is_delegate_target(parent):
        return await _delegate_preflight(session_db, parent_session_id, session_store, session_key)
    return "deliver"
