"""Session title settlement before the cron session database closes."""


def set_cron_session_title(session_db, session_id, base_title):
    """Persist a non-blank, unique title for a finished cron session; returns it (None if unset).
    Runs BEFORE end_session()/close() so no write races the close. Duplicate title (unique-index
    ValueError) -> get_next_title_in_lineage(); if unavailable, raise rather than end up untitled.

    Centralizes the title write so the cron finally block can guarantee a non-blank, unique title is
    persisted before end_session()/close() tear the connection down (issues #50535, #50536, #50537):
    - #50535: never leaves the session blank. base_title already carries a cron-id fallback for nameless
    jobs; this also guards a failed write. Recover by appending a #N suffix via get_next_title_in_lineage()
    when supported, instead of swallowing the error and ending up untitled. - #50536: this runs
    synchronously in the cron finally block ahead of the session close, so no in-flight title write can race
    the close.
    """
    if not session_db or not session_id:
        return None
    title = (base_title or "").strip()
    if not title:
        return None
    try:
        session_db.set_session_title(session_id, title)
        return title
    except ValueError:
        # Unique-title collision: fall back to the next lineage title (base #2, #3, ...).
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        session_db.set_session_title(session_id, deduped)
        return deduped
