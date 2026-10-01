"""Invariants for the shared retry-delay parsers in ``agent/retry_utils.py``.

Cluster: every consumer of ``Retry-After`` / free-text reset grammars goes through one parser,
so an HTTP-date header or a "resets in 2 hours 5 minutes" body yields the same wait everywhere.
"""

from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

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


class TestClaudeWallClockReset:
    """The Claude-subscription session-limit grammar names a wall-clock reopen in an explicit zone.

    DirectSDK's relay raises the 429 as a ``RuntimeError`` whose text ends
    ``You've hit your session limit · resets 12:30pm (America/Chicago)``. None of the delta
    grammars match, so the cooldown armed a generic 60s, ``restore_primary_runtime`` re-primed the
    spent primary every turn, and each attempt re-uploaded the whole conversation for another 429.
    """

    @staticmethod
    def _seconds_until_clock(hour12: int, minute: int, meridiem: str, tz_name: str) -> float:
        tz = ZoneInfo(tz_name)
        now = datetime.now(tz)
        hour = hour12 % 12 + (12 if meridiem == "pm" else 0)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    def test_named_zone_resolves_to_that_zones_next_occurrence(self):
        """The delay is measured in the NAMED zone, not the host's — a host on UTC must still read
        ``resets 5:30pm (UTC)`` as ~its own evening and ``12:30pm (America/Chicago)`` in Chicago."""
        for hour12, minute, meridiem, tz_name in (
            (12, 30, "pm", "America/Chicago"),
            (5, 30, "pm", "UTC"),
            (5, 0, "pm", "America/Chicago"),
        ):
            message = f"You've hit your session limit · resets {hour12}:{minute:02d}{meridiem} ({tz_name})"
            expected = self._seconds_until_clock(hour12, minute, meridiem, tz_name)
            actual = reset_delay_from_message(message)
            assert actual == pytest.approx(expected, abs=5), message

    def test_zone_is_required_and_invalid_zones_fail_open(self):
        """A bare ``resets 5pm`` is somebody's local clock and guessing it can be hours wrong;
        an unknown zone name must fall through to the pre-existing behavior, not crash."""
        assert reset_delay_from_message("You've hit your session limit · resets 5pm") is None
        assert reset_delay_from_message("resets 5pm (Mars/Olympus_Mons)") is None
        assert reset_delay_from_message("resets (America/Chicago)") is None

    def test_just_past_instant_is_not_rolled_a_whole_day(self):
        """An error surfacing seconds after the stated minute means the window just reopened.
        Rolling to "next occurrence" would wait ~24h for a limit that is already over."""
        now_ct = datetime.now(ZoneInfo("America/Chicago"))
        message = f"resets {now_ct.strftime('%I:%M%p').lower()} (America/Chicago)"
        assert reset_delay_from_message(message) == 0.0

    def test_clearly_past_instant_rolls_to_next_day_within_the_cap(self):
        """A stale error re-read the next day still describes a forward-looking window; the
        roll-forward must stay bounded (<= 24h) rather than inflate beyond a day."""
        an_hour_ago = datetime.now(ZoneInfo("America/Chicago")) - timedelta(hours=1)
        message = f"resets {an_hour_ago.strftime('%I:%M%p').lower()} (America/Chicago)"
        seconds = reset_delay_from_message(message)
        assert seconds is not None
        assert 22 * 3600 <= seconds <= 24 * 3600

    def test_resets_at_variant_and_existing_grammars_are_unchanged(self):
        assert reset_delay_from_message("resets at 12:30pm (America/Chicago)") is not None
        assert reset_delay_from_message("Weekly usage limit reached. Resets in 6hr 29min.") == 6 * 3600 + 29 * 60

    def test_directsdk_runtimeerror_text_survives_into_the_reset_context(self):
        """End-to-end through the exact text DirectSDK's relay raises: the extractor the retry
        loop consults (``extract_api_error_context``) must carry ``reset_at`` so
        ``_arm_rate_limit_cooldown`` gates primary restoration until the named reopen."""
        from agent.agent_runtime_helpers import extract_api_error_context
        from agent.fallback_cooldown import _provider_reset_delay

        message = (
            "Incomplete upstream response (first upstream attempt: status 429, capture incomplete, "
            "native retries denied: 0, upstream said: This request would exceed your account's rate limit. "
            "Please try again later.): You've hit your session limit \u00b7 resets 12:30pm (America/Chicago)"
        )
        ctx = extract_api_error_context(RuntimeError(message))
        assert "reset_at" in ctx
        expected = self._seconds_until_clock(12, 30, "pm", "America/Chicago")
        delay = _provider_reset_delay(ctx["reset_at"])
        assert delay is not None
        assert delay == pytest.approx(expected, abs=5)
        # The cooldown it arms is the real window (tens of minutes to hours), nowhere near the 60s
        # default that made the primary look re-usable a minute after the limit fired.
        assert delay > 60
