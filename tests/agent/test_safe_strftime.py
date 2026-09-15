"""safe_strftime must never let locale decoding crash the caller (#102910).

On Windows with some system locales the timezone / day / month names come
back through the ANSI code page carrying lone surrogates, and ``strftime``
itself RAISES ``UnicodeEncodeError`` before producing output — so merely
scrubbing the returned string never runs. That crash landed in the
system-prompt builder (rebuilt every conversation + compaction).
"""

from datetime import datetime

from agent.message_sanitization import safe_strftime


class _FrWindowsDt:
    """Mimics a fr-FR Windows datetime: locale-name codes raise, the rest work."""

    _LOCALE_CODES = ("%Z", "%A", "%a", "%B", "%b")

    def __init__(self, inner: datetime):
        self._inner = inner

    def strftime(self, fmt: str) -> str:
        if any(code in fmt for code in self._LOCALE_CODES):
            raise UnicodeEncodeError("utf-8", "x\udc80y", 1, 2, "surrogates not allowed")
        return self._inner.strftime(fmt)


class _SurrogateDt:
    """strftime succeeds but the text carries a lone surrogate (tz name)."""

    def strftime(self, fmt: str) -> str:
        assert fmt == "%Z"
        return "Paris, Madrid (heure d'\udc82t\u00e9)"


def test_passthrough_matches_strftime():
    dt = datetime(2026, 9, 4, 12, 30)
    assert safe_strftime(dt, "%Y-%m-%d %H:%M %Z") == dt.strftime("%Y-%m-%d %H:%M %Z")
    assert safe_strftime(dt, "%A, %B %d, %Y") == dt.strftime("%A, %B %d, %Y")


def test_unicode_error_falls_back_without_locale_codes():
    dt = _FrWindowsDt(datetime(2026, 9, 4, 12, 30))
    # %Z dropped, numeric codes kept — the UTC offset survives.
    assert safe_strftime(dt, "%Y-%m-%d %H:%M %Z") == "2026-09-04 12:30"
    assert safe_strftime(dt, "%A, %B %d, %Y") == ", 04, 2026"


def test_format_of_only_locale_codes_degrades_to_empty():
    dt = _FrWindowsDt(datetime(2026, 9, 4, 12, 30))
    assert safe_strftime(dt, "%Z") == ""


def test_surrogate_output_is_scrubbed():
    out = safe_strftime(_SurrogateDt(), "%Z")
    assert "\udc82" not in out
    assert "\ufffd" in out
    out.encode("utf-8")  # must not raise
