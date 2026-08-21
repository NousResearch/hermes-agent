"""Retry-After must be honoured in both RFC 7231 forms.

`conversation_loop` parsed the header with a bare `float()`, so the HTTP-date
form — which edge proxies (Cloudflare, nginx) in front of a provider emit on
429/503 — raised ValueError and was discarded. The loop then fell back to a
~2s jittered backoff that cannot outlast the rate-limit window, exhausted its
retries and surfaced a rate-limit failure to the user.

`agent.retry_utils.parse_retry_after_seconds` already handled both forms; it
simply was not wired into this call site.
"""

import datetime
import email.utils

from agent.retry_utils import parse_retry_after_seconds


def _http_date(seconds_from_now):
    when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
        seconds=seconds_from_now
    )
    return email.utils.format_datetime(when)


class TestRetryAfterParsing:
    def test_delta_seconds_form(self):
        assert parse_retry_after_seconds({"retry-after": "42"}) == 42.0

    def test_header_lookup_is_case_insensitive(self):
        assert parse_retry_after_seconds({"Retry-After": "42"}) == 42.0

    def test_http_date_form_is_honoured(self):
        """The form a bare float() rejected."""
        parsed = parse_retry_after_seconds({"retry-after": _http_date(300)})
        assert parsed is not None
        assert 290 <= parsed <= 300

    def test_past_http_date_clamps_to_zero(self):
        assert parse_retry_after_seconds({"retry-after": _http_date(-300)}) == 0.0

    def test_unparseable_value_is_none(self):
        assert parse_retry_after_seconds({"retry-after": "garbage"}) is None

    def test_absent_header_is_none(self):
        assert parse_retry_after_seconds({}) is None

    def test_bare_float_rejects_the_http_date_form(self):
        """Documents the defect this call site had."""
        try:
            float(_http_date(300))
        except ValueError:
            return
        raise AssertionError("expected float() to reject an HTTP-date")


class TestConversationLoopWiring:
    def test_conversation_loop_imports_the_parser(self):
        """Guard against the call site drifting back to a bare float()."""
        import agent.conversation_loop as cl

        assert hasattr(cl, "parse_retry_after_seconds")
