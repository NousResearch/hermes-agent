"""Desktop-native cron delivery: per-job persistent chat sessions.

Each recurring cron job can have one stable Desktop delivery session that
accumulates output over time.  Runs still execute in fresh isolated sessions;
the delivery session is a separate, user-facing conversation.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Session ID prefix for desktop delivery sessions
DESKTOP_DELIVERY_SESSION_PREFIX = "cron_delivery_"


def _delivery_session_id(job: dict) -> str:
    """Return the stable session ID for a job's desktop delivery session.

    Deterministic from the job ID so the same job always maps to the same
    delivery session across runs.
    """
    job_id = job.get("id", "?")
    return f"{DESKTOP_DELIVERY_SESSION_PREFIX}{job_id}"


def _deliver_to_desktop_session(
    job: dict,
    content: str,
    session_db=None,
    session_name_hint: Optional[str] = None,
) -> Optional[str]:
    """Deliver cron output to the job's persistent Desktop delivery session.

    Creates the session on first delivery, appends to it on subsequent runs.
    When *session_db* is None (the common case — delivery happens after the
    agent's SessionDB has been closed), opens its own short-lived SessionDB
    instance for the write.

    *session_name_hint* overrides the session title when set (from a
    ``desktop-session:<name>`` delivery target).  The session id is always
    derived from the job id, so multiple runs of the same job always append
    to the same delivery session regardless of the name hint.

    Returns None on success, error string on failure.
    """
    job_id = job.get("id", "?")
    job_name = job.get("name", job_id)

    # Open our own SessionDB when none is provided.  Delivery runs after the
    # agent's session DB has been closed, so we need a fresh handle.
    _owned_db = None
    try:
        if session_db is None:
            from hermes_state import SessionDB

            _owned_db = SessionDB()
            session_db = _owned_db

        session_id = _delivery_session_id(job)

        # Try to create the session row (no-op on conflict — the upsert in
        # _insert_session_row handles it).  Source "cron_desktop" marks this
        # as a cron desktop-delivery session, distinct from regular cron runs.
        session_db.create_session(
            session_id,
            source="cron_desktop",
        )

        # Attempt to give the session a readable title.  Best-effort: a title
        # collision with a different job (extremely unlikely given the
        # deterministic ID prefix) is non-fatal.
        try:
            display_name = session_name_hint or job_name
            title = f"Cron Delivery: {display_name}"
            session_db.set_session_title(session_id, title)
        except Exception:
            pass

        # Append the cron output as an assistant-role message so the Desktop
        # chat shows it as a delivered piece of content (not as a user message
        # that would feel like the user wrote it).
        session_db.append_message(
            session_id,
            role="assistant",
            content=content,
        )

        logger.info(
            "Job '%s': delivered to Desktop session %s",
            job_id, session_id,
        )
        return None
    except Exception as e:
        msg = f"desktop delivery to session failed: {e}"
        logger.warning("Job '%s': %s", job_id, msg, exc_info=True)
        return msg
    finally:
        if _owned_db is not None:
            try:
                _owned_db.close()
            except Exception:
                logger.debug(
                    "Job '%s': failed to close owned SessionDB for desktop delivery",
                    job_id,
                    exc_info=True,
                )
