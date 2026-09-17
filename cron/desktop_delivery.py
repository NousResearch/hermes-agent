"""Desktop-native cron delivery: one Desktop chat session per cron invocation.

A job that delivers to ``desktop-session`` lands each invocation in its OWN
session in the Desktop sidebar — the same shape as the run session the runner
creates (``cron_<job id>_<stamp>``), just user-facing.  Nothing accumulates
across invocations, so replying to today's brief carries today's context and
not every previous day's.
"""

import logging
from datetime import datetime
from typing import Optional

from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

# Session ID prefix for desktop delivery sessions
DESKTOP_DELIVERY_SESSION_PREFIX = "cron_delivery_"


def _delivery_session_id(job: dict, now: Optional[datetime] = None) -> str:
    """Fresh session ID per invocation: ``cron_delivery_<job id>_<YYYYmmdd_HHMMSS>``.

    Unique by construction via the run stamp, and job-scoped so every delivery
    session of a job is identifiable (and greppable) as such.
    """
    stamp = (now or _hermes_now()).strftime("%Y%m%d_%H%M%S")
    job_id = job.get("id", "?")
    return f"{DESKTOP_DELIVERY_SESSION_PREFIX}{job_id}_{stamp}"


def _delivery_title(job: dict, session_name_hint: Optional[str], now: datetime) -> str:
    """``"<job name> · Sep 17 09:05"`` — the shape a cron run session gets.

    The run stamp is not decoration: ``sessions.title`` carries a UNIQUE index,
    so without it the second invocation of a job could not hold its own name.
    """
    job_id = job.get("id", "?")
    display_name = (session_name_hint or job.get("name") or job_id).strip() or job_id
    return f"{display_name} · {now.strftime('%b %d %H:%M')}"


def _set_delivery_title(session_db, session_id: str, title: str) -> Optional[str]:
    """Persist the title, falling back to the next free lineage title (``#2``, ...).

    ``_set_cron_session_title`` already owns that invariant for run sessions
    (never leave a session blank; a duplicate title becomes ``base #N``), so
    reuse it rather than growing a second copy of the rule.  Late import: this
    module is called from ``cron.scheduler``'s delivery path.
    """
    from cron.scheduler import _set_cron_session_title

    return _set_cron_session_title(session_db, session_id, title)


def _deliver_to_desktop_session(
    job: dict,
    content: str,
    session_db=None,
    session_name_hint: Optional[str] = None,
) -> Optional[str]:
    """Deliver this invocation's cron output to its own Desktop session.

    Creates the session (each invocation gets a fresh row — see
    ``_delivery_session_id``) and appends the output as a single assistant
    message.  When *session_db* is None (the common case — delivery happens
    after the agent's SessionDB has been closed), opens its own short-lived
    SessionDB instance for the write.

    *session_name_hint* replaces the job name in the session title when set
    (from a ``desktop-session:<name>`` delivery target).

    Returns None on success, error string on failure.
    """
    job_id = job.get("id", "?")
    now = _hermes_now()

    # Open our own SessionDB when none is provided.  Delivery runs after the
    # agent's session DB has been closed, so we need a fresh handle.
    _owned_db = None
    try:
        if session_db is None:
            from hermes_state import SessionDB

            _owned_db = SessionDB()
            session_db = _owned_db

        session_id = _delivery_session_id(job, now)

        # Source "cron_desktop" marks this as a cron Desktop-delivery session,
        # distinct from regular cron run sessions (source "cron").
        session_db.create_session(
            session_id,
            source="cron_desktop",
        )

        # Title best-effort: untitled is survivable, an error is not a delivery
        # failure.  A same-minute second delivery of the job collides on the
        # unique title index and lands as "<name> · Sep 17 09:05 #2".
        try:
            title = _delivery_title(job, session_name_hint, now)
            _set_delivery_title(session_db, session_id, title)
        except Exception:
            logger.debug(
                "Job '%s': could not title Desktop delivery session %s",
                job_id, session_id, exc_info=True,
            )

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
