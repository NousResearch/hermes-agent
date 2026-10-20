"""Fail-closed redaction for outward cron payloads (split from scheduler_delivery)."""

import logging

logger = logging.getLogger(__name__)


def _redact_cron_payload(text: str, what: str) -> str:
    """Fail-closed secret redaction for anything a cron job emits outward.

    Every outward lane — chat message, session mirror, bot-chat turn — must apply the same policy,
    so the policy lives in one place. ``force=True`` because this is a safety boundary, not
    logging: the ``security.redact_secrets`` preference governs how much is scrubbed from the
    user's own logs and must not be able to turn scrubbing off on the way out to a chat (same
    reasoning as ``tools/delegation_live_log.py``). Empty input is returned as-is; any failure
    inside the redactor replaces the payload entirely rather than letting an unscanned value out.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception as e:
        logger.warning("Failed to redact secrets from cron %s: %s", what, e)
        return "[REDACTED - redaction failed]"
