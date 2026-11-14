"""Error text bound for cron's durable stores (jobs.json, executions.db): scrubbed and bounded (#102700)."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Same bound as cron/incidents.py MAX_ERROR_CHARS-style caps: enough to diagnose, never unbounded.
MAX_PERSISTED_ERROR_CHARS = 2000


def sanitize_persisted_error(error: Optional[str]) -> Optional[str]:
    """Scrub secrets from, and bound, error text bound for jobs.json / executions.db.

    Those stores outlive restarts and get backed up or synced, so raw provider / stderr text must
    not land there verbatim — the same rule cron/incidents.py and cron/delivery_queue.py already
    apply. A redactor failure stores a placeholder rather than the unscanned text, and never
    raises: the caller is recording that a job failed.
    """
    if not error:
        return error
    try:
        from agent.redact import redact_sensitive_text

        text = redact_sensitive_text(str(error), force=True, redact_url_credentials=True)
    except Exception:
        logger.warning("Failed to redact cron error before persisting it", exc_info=True)
        return "[REDACTED - redaction failed]"
    return text[:MAX_PERSISTED_ERROR_CHARS]
