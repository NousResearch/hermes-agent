"""Regression tests for #71643: Telegram streaming — successful finalize edit
carries the stale preview text, ``content_delivered=True`` suppresses the
complete send.

When the stream consumer's last sent text is substantially shorter than the
completed response (stale finalize), the gateway must NOT suppress the normal
final send path.
"""

import pytest


class _StubStreamConsumer:
    """Minimal stub matching the attrs the suppression logic reads."""

    def __init__(self, *, last_sent_text="", final_content_delivered=False):
        self._last_sent_text = last_sent_text
        self._final_content_delivered = final_content_delivered

    @property
    def final_content_delivered(self):
        return self._final_content_delivered

    @property
    def final_response_sent(self):
        return False

    @staticmethod
    def _clean_for_display(text: str) -> str:
        """Minimal clean_for_display matching stream_consumer behavior."""
        return text.replace("\u2589", "").strip()


def _apply_stale_finalize_guard(sc, final_response, *, streamed=False):
    """Reproduce the stale-finalize guard logic from gateway/run.py."""
    _content_delivered = bool(sc.final_content_delivered)
    _final = final_response
    _streamed = streamed

    if _content_delivered and not _streamed and _final:
        _delivered_text = getattr(sc, "_last_sent_text", "") or ""
        _clean_delivered = (
            sc._clean_for_display(_delivered_text).strip()
            if hasattr(sc, "_clean_for_display")
            else _delivered_text.strip()
        )
        _clean_final = _final.strip()
        if (
            _clean_delivered
            and _clean_final
            and len(_clean_delivered) < len(_clean_final) * 0.8
        ):
            _content_delivered = False

    return _content_delivered


def test_stale_finalize_does_not_suppress_normal_send():
    """When the stream consumer delivered stale text (e.g. preview buffer
    instead of completed response), the gateway must fall through to the
    normal final send path (#71643)."""
    final_response = "A" * 272
    stale_text = "A" * 155

    sc = _StubStreamConsumer(
        last_sent_text=stale_text,
        final_content_delivered=True,
    )

    result = _apply_stale_finalize_guard(sc, final_response)
    assert result is False


def test_complete_finalize_still_suppresses():
    """When the stream consumer delivered the complete response (>= 80% match),
    the gateway should still suppress the normal final send."""
    final_response = "A" * 272
    complete_text = "A" * 260

    sc = _StubStreamConsumer(
        last_sent_text=complete_text,
        final_content_delivered=True,
    )

    result = _apply_stale_finalize_guard(sc, final_response)
    assert result is True


def test_streamed_delivery_not_affected_by_stale_guard():
    """When _streamed is True (stream_confirmed_final_delivery passed),
    the stale guard should not interfere."""
    sc = _StubStreamConsumer(
        last_sent_text="short",
        final_content_delivered=True,
    )

    result = _apply_stale_finalize_guard(sc, "A" * 272, streamed=True)
    assert result is True


def test_empty_delivered_text_skips_guard():
    """When _last_sent_text is empty (e.g. never sent), the guard should
    not trigger — let the existing logic handle it."""
    sc = _StubStreamConsumer(
        last_sent_text="",
        final_content_delivered=True,
    )

    result = _apply_stale_finalize_guard(sc, "A" * 272)
    assert result is True


def test_empty_final_skips_guard():
    """When the final response is empty, the guard should not trigger."""
    sc = _StubStreamConsumer(
        last_sent_text="A" * 100,
        final_content_delivered=True,
    )

    result = _apply_stale_finalize_guard(sc, "")
    assert result is True
