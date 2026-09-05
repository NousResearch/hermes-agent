"""Fail-closed checks for user-controlled text sent to the Hindsight backend.

Rejects the entire operation when Hermes redaction would alter the payload.
Never redacts-and-stores, and never echoes the original value.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__.rpartition(".")[0])


class HindsightSensitiveContentError(ValueError):
    """Raised when user-controlled text must not be transmitted to Hindsight."""

    def __init__(self, field: str = "content"):
        self.field = field
        super().__init__(
            f"Refused to transmit {field}: sensitive content detected. "
            "The operation was rejected so no altered memory is stored."
        )


def assert_safe_for_hindsight(text: Any, *, field: str = "content") -> None:
    """Force-check *text* with strict URL-credential redaction.

    No-op on empty values. Fail closed if redaction changes the text or if
    the checker itself errors. The original value is never logged or raised.
    """
    if text is None:
        return
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return
    try:
        redacted = redact_sensitive_text(
            text, force=True, redact_url_credentials=True,
        )
    except Exception:
        logger.warning(
            "Hindsight safety checker failed; refusing transmission of %s", field,
        )
        raise HindsightSensitiveContentError(field) from None
    if redacted != text:
        raise HindsightSensitiveContentError(field)


def assert_safe_texts(values: Iterable[Any], *, field: str) -> None:
    for value in values:
        if value is None:
            continue
        assert_safe_for_hindsight(value, field=field)


def assert_retain_item_safe(item: dict) -> None:
    """Check retain-item fields that carry user-controlled text."""
    assert_safe_for_hindsight(item.get("content"), field="content")
    assert_safe_for_hindsight(item.get("context"), field="context")
    assert_safe_for_hindsight(item.get("timestamp"), field="occurred_at")
    tags = item.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    assert_safe_texts(tags, field="tags")
