"""Text redacted exactly once, so a later storage pass leaves it byte-identical.

A document that carries length stamps over its own text (the cron run archive's
``**Prompt Characters:**`` / ``**Response Characters:**`` frames) must be redacted
BEFORE those stamps are computed, and never again afterwards: the redactor is not
idempotent (a ``KEY='exampl...-32c'`` head/tail mask collapses to ``KEY='***'`` on a
second pass), so a second pass shifts the text under the stamps and the frame no
longer parses (#153079).

``redact_once`` returns a :class:`RedactedText` only when redaction actually ran for
the active profile. Storage boundaries (``gateway.session_results._redacted``) keep a
``RedactedText`` as is and still redact every plain ``str``, so nothing reaches
state.db unredacted that would have been redacted before.
"""

from __future__ import annotations


class RedactedText(str):
    """A ``str`` already passed through ``redact_sensitive_text`` by ``redact_once``."""

    __slots__ = ()


def redact_once(text: str) -> str:
    """``text`` redacted for the active profile, marked so storage does not redact it again.

    Redaction off for this profile (``security.redact_secrets: false``): the text comes back
    unmarked, so a storage pass that resolves the policy differently still applies its own.
    """
    if isinstance(text, RedactedText):
        return text
    from agent.redact import _redact_enabled, redact_sensitive_text
    if not _redact_enabled():
        return text
    return RedactedText(redact_sensitive_text(text))
