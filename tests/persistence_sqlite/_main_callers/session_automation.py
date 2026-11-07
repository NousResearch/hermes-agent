"""Main's ``gateway/session_automation.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

from persistence.core.errors import AuthorityRefusal


def _replace_queued(admission_id, session_id, replaced):
    """Write guard: cancel the replaced queued tick in the replacement's own transaction; refuse the
    replacement when that tick was claimed or settled first (its output already ran)."""
    def guard(conn):
        changed = conn.execute("UPDATE session_admissions SET status='terminal',outcome='cancelled' "
                               "WHERE admission_id=? AND target_session_id=? AND status='queued'",
                               (admission_id, session_id)).rowcount
        if changed != 1:
            raise AuthorityRefusal('stale_generation')
        replaced[:] = [admission_id]
    return guard
