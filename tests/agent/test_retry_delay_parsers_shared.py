"""Invariants for the shared retry-delay parsers in ``agent/retry_utils.py``.

Cluster: every consumer of ``Retry-After`` / free-text reset grammars goes through one parser,
so an HTTP-date header or a "resets in 2 hours 5 minutes" body yields the same wait everywhere.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from agent.retry_utils import parse_retry_after_seconds, reset_delay_from_message


def _http_date(seconds_ahead: int) -> str:
    return format_datetime(datetime.now(timezone.utc) + timedelta(seconds=seconds_ahead), usegmt=True)


class TestRetryAfterHeaderOneParser:
    def test_http_date_header_parsed_identically_at_formerly_divergent_sites(self):
        """anon_auth, the error-context extractor and nous_rate_guard used to float() the header
        and silently drop the RFC 7231 date form; all three must now agree with the canonical."""
        from agent.agent_runtime_helpers import extract_api_error_context
        from agent.nous_rate_guard import _parse_reset_seconds
        from hermes_cli.anon_auth import _retry_after_seconds as anon_retry_after
        import time

        header = _http_date(90)
        canonical = parse_retry_after_seconds(header)
        assert 85 <= canonical <= 90

        anon = anon_retry_after(SimpleNamespace(headers={"Retry-After": header}), default=1.0)
        assert abs(anon - canonical) < 2

        guard = _parse_reset_seconds({"Retry-After": header})
        assert guard is not None and abs(guard - canonical) < 2

        err = Exception("rate limited")
        err.response = SimpleNamespace(headers={"Retry-After": header})
        ctx = extract_api_error_context(err)
        assert 85 <= ctx["reset_at"] - time.time() <= 91

    def test_metrics_sender_clamps_on_top_of_the_shared_parser(self):
        from hermes_cli.observability.shared_metrics_sender import _retry_after_seconds

        assert _retry_after_seconds(_http_date(120), 7) in (119, 120)
        assert _retry_after_seconds("0", 7) == 1          # floor survives
        assert _retry_after_seconds("99999999", 7) == 86_400  # cap survives
        assert _retry_after_seconds("garbage", 7) == 7


class TestResetDelayOneTable:
    @pytest.mark.parametrize("message, seconds", [
        ("Weekly usage limit reached. Resets in 6hr 29min.", 6 * 3600 + 29 * 60),
        ("resets in 2 hours 5 minutes", 2 * 3600 + 5 * 60),
        ("Limit hit; resets in 45s", 45.0),
        ('"quotaResetDelay": "1500ms"', 1.5),
        ("please retry after 12 seconds", 12.0),
        # Both grammars in one body: the explicit retry-after wins (pool precedence), not the
        # multi-hour quota window.
        ("Rate limited. Retry after 30s; resets in 4hr", 30.0),
    ])
    def test_credential_pool_and_error_context_agree(self, message, seconds):
        """The pooled-credential cooldown and the UI's error context read the same table, so the
        long-form "hours/minutes" grammar (which the pool used to miss) resolves at both sites."""
        import time
        from agent.credential_pool import _normalize_error_context

        assert reset_delay_from_message(message) == pytest.approx(seconds)
        normalized = _normalize_error_context({"message": message})
        assert normalized["reset_at"] - time.time() == pytest.approx(seconds, abs=2)

    def test_no_grammar_means_no_reset(self):
        from agent.credential_pool import _normalize_error_context

        assert reset_delay_from_message("resets in the future, maybe") is None
        assert "reset_at" not in _normalize_error_context({"message": "resets in the future, maybe"})
