"""Admission of internal events that are pinned to one conversation."""


async def pinned_session_continues(runner, entry, pinned_session_id: str) -> bool:
    """Whether the route's current session is ``pinned_session_id`` or its compression continuation.

    Compression moves a conversation to a child session without ending it, so an event pinned to
    the parent still belongs to the route. ``/new`` and ``/reset`` start an unrelated session.
    """
    if entry.session_id == pinned_session_id:
        return True

    def compression_tip():
        db = runner.session_store._db_for_key(entry.session_key)
        return db.get_compression_tip(pinned_session_id) if db is not None else None

    return await runner._run_in_executor_with_context(compression_tip) == entry.session_id
