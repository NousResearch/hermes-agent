"""Phase2 P2: clamped 429-park + interactive-capped Console Go backoff.

Covers the ``compute_error_backoff`` Console Go 503/``service_overloaded`` +
429 branch (300s -> 1200s -> 3600s ladder, jittered), the attempt-reducing
retry ceiling, structured ``resets_at`` / ``resets_in_seconds`` park-until
(clamped at 3600s, ms-vs-s/ISO handled, unparseable falls through), the
interactive <=300s sleep cap, and Retry-After precedence (600s cap stands).
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.real_retry_backoff

from agent import retry_utils
from agent.retry_utils import (
    console_go_overload_backoff,
    console_go_overload_retry_ceiling,
    is_console_go_overload_error,
    park_seconds_from_error,
    _CONSOLE_GO_OVERLOAD_LONG_BACKOFF,
)
from agent.turn_recovery import compute_error_backoff, route_classified_error

GO_BASE = "https://opencode.ai/zen/go/v1"
GO_MODEL = "muse-spark-1.3-contributor"


def _go_503(**kw):
    return SimpleNamespace(status_code=503, body={"error": {"message": "Service overloaded"}}, **kw)


def _go_429_overload(**kw):
    return SimpleNamespace(
        status_code=429,
        body={"error": {"code": "service_overloaded", "message": "service_overloaded: relay saturated"}},
        **kw,
    )


def _agent():
    return MagicMock()


def _backoff(err, **kw):
    params: dict[str, Any] = dict(
        retry_count=1, max_retries=8, is_rate_limited=False,
        is_zai_coding_overload=False, base_url=GO_BASE, model=GO_MODEL,
    )
    params.update(kw)
    return compute_error_backoff(_agent(), err, **params)


# ---------------------------------------------------------------------------
# Detection: narrow Console Go overload shape
# ---------------------------------------------------------------------------

class TestIsConsoleGoOverloadError:
    def test_go_503_qualifies(self):
        assert is_console_go_overload_error(base_url=GO_BASE, model=GO_MODEL, error=_go_503())

    def test_go_429_with_overload_text_qualifies(self):
        assert is_console_go_overload_error(base_url=GO_BASE, model=GO_MODEL, error=_go_429_overload())

    def test_go_429_without_overload_text_fails_fast(self):
        err = SimpleNamespace(status_code=429, body={"error": {"message": "quota exceeded"}})
        assert not is_console_go_overload_error(base_url=GO_BASE, model=GO_MODEL, error=err)

    def test_non_go_503_does_not_qualify(self):
        assert not is_console_go_overload_error(
            base_url="https://api.anthropic.com/v1", model="claude", error=_go_503())

    def test_other_status_does_not_qualify(self):
        err = SimpleNamespace(status_code=500, body={"error": {"message": "service_overloaded"}})
        assert not is_console_go_overload_error(base_url=GO_BASE, model=GO_MODEL, error=err)


# ---------------------------------------------------------------------------
# Ladder: 300 -> 1200 -> 3600, jittered
# ---------------------------------------------------------------------------

class TestConsoleGoLadder:
    def test_ladder_bases(self, monkeypatch):
        monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
        assert _CONSOLE_GO_OVERLOAD_LONG_BACKOFF == (300.0, 1200.0, 3600.0)
        waits = [
            console_go_overload_backoff(a, error=_go_503(), default_wait=1.0)[0]
            for a in (2, 3, 4)
        ]
        assert waits == [300.0, 1200.0, 3600.0]

    def test_short_attempt_keeps_default_wait(self):
        wait, policy = console_go_overload_backoff(1, error=_go_503(), default_wait=2.0)
        assert (wait, policy) == (2.0, "console_go_overload_short")

    def test_ladder_sticks_at_top(self, monkeypatch):
        monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
        wait, policy = console_go_overload_backoff(99, error=_go_503(), default_wait=1.0)
        assert (wait, policy) == (3600.0, "console_go_overload_long")

    def test_compute_uses_ladder_with_jitter(self):
        wait = _backoff(_go_503(), retry_count=2, is_console_go_overload=True, interactive=False)
        assert 300.0 <= wait <= 360.0

    def test_ladder_has_light_jitter(self):
        waits = {
            _backoff(_go_503(), retry_count=3, is_console_go_overload=True, interactive=False)
            for _ in range(10)
        }
        assert all(1200.0 <= w <= 1440.0 for w in waits)


# ---------------------------------------------------------------------------
# Ceiling: reduces total attempts, never extends
# ---------------------------------------------------------------------------

class TestConsoleGoCeiling:
    def test_ceiling_value_and_headroom(self):
        ceiling = console_go_overload_retry_ceiling()
        assert ceiling == 5  # 1 short + 3 long + 1 headroom
        last_attempt_with_backoff = ceiling - 1
        assert last_attempt_with_backoff - 1 >= len(_CONSOLE_GO_OVERLOAD_LONG_BACKOFF)

    def test_route_shrinks_max_retries(self):
        from agent.error_classifier import classify_api_error
        err = _go_503()
        classified = classify_api_error(err, provider="opencode-go", model=GO_MODEL, base_url=GO_BASE)
        verdict = route_classified_error(
            SimpleNamespace(), err, classified, SimpleNamespace(), error_msg=str(err),
            error_context={}, recovered_with_pool=False, base_url=GO_BASE, model=GO_MODEL,
            messages=[], api_messages=[], system_message=None, active_system_prompt="",
            conversation_history=None, retry_count=0, max_retries=8,
            compression_attempts=0, max_compression_attempts=3, api_call_count=1,
            effective_task_id="t",
        )
        assert verdict.is_console_go_overload is True
        assert verdict.max_retries == console_go_overload_retry_ceiling() < 8

    def test_route_never_extends_small_ceiling(self):
        from agent.error_classifier import classify_api_error
        err = _go_503()
        classified = classify_api_error(err, provider="opencode-go", model=GO_MODEL, base_url=GO_BASE)
        verdict = route_classified_error(
            SimpleNamespace(), err, classified, SimpleNamespace(), error_msg=str(err),
            error_context={}, recovered_with_pool=False, base_url=GO_BASE, model=GO_MODEL,
            messages=[], api_messages=[], system_message=None, active_system_prompt="",
            conversation_history=None, retry_count=0, max_retries=3,
            compression_attempts=0, max_compression_attempts=3, api_call_count=1,
            effective_task_id="t",
        )
        assert verdict.max_retries == 3


# ---------------------------------------------------------------------------
# Retry-After still wins; 600s cap stands
# ---------------------------------------------------------------------------

class TestRetryAfterPrecedence:
    def test_retry_after_wins_over_ladder(self):
        err = SimpleNamespace(
            status_code=503, body={"error": {"message": "Service overloaded"}},
            response=SimpleNamespace(headers={"Retry-After": "120"}),
        )
        assert _backoff(err, retry_count=4, is_console_go_overload=True, interactive=False) == 120.0

    def test_retry_after_cap_stands(self):
        err = SimpleNamespace(
            status_code=429, body={"error": {"code": "service_overloaded"}},
            response=SimpleNamespace(headers={"Retry-After": "3600"}),
        )
        assert _backoff(err, retry_count=4, is_console_go_overload=True, interactive=False) == 600.0


# ---------------------------------------------------------------------------
# Park: structured resets_at / resets_in_seconds, clamped at 3600s
# ---------------------------------------------------------------------------

class TestParkUntil:
    def test_resets_in_seconds_parks(self):
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 900}})
        assert _backoff(err, is_rate_limited=True, interactive=False) == 900.0

    def test_park_clamped_at_3600(self):
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 7200}})
        assert _backoff(err, is_rate_limited=True, interactive=False) == 3600.0

    def test_resets_at_epoch_seconds(self):
        err = SimpleNamespace(
            status_code=429, body={"error": {"resets_at": time.time() + 600}})
        wait = _backoff(err, is_rate_limited=True, interactive=False)
        assert 590.0 < wait <= 600.0

    def test_resets_at_epoch_milliseconds(self):
        err = SimpleNamespace(
            status_code=429, body={"error": {"resets_at": (time.time() + 600) * 1000}})
        wait = _backoff(err, is_rate_limited=True, interactive=False)
        assert 590.0 < wait <= 600.0

    def test_resets_at_iso_string(self):
        iso = (datetime.now(timezone.utc) + timedelta(seconds=600)).isoformat()
        err = SimpleNamespace(status_code=429, body={"error": {"resets_at": iso}})
        wait = _backoff(err, is_rate_limited=True, interactive=False)
        assert 590.0 < wait <= 600.0

    def test_top_level_body_fields(self):
        err = SimpleNamespace(status_code=429, body={"resets_in_seconds": 450})
        assert park_seconds_from_error(err) == 450.0

    def test_shortest_candidate_wins(self):
        err = SimpleNamespace(
            status_code=429,
            body={"error": {"resets_in_seconds": 900, "resets_at": time.time() + 300}},
        )
        wait = park_seconds_from_error(err)
        assert wait is not None
        assert 290.0 < wait <= 300.0


# ---------------------------------------------------------------------------
# Fallthrough: unparseable / missing / expired never freezes
# ---------------------------------------------------------------------------

class TestParkFallthrough:
    def test_unparseable_resets_at_falls_through_to_ladder(self):
        err = SimpleNamespace(status_code=429, body={"error": {"resets_at": "soon-ish"}})
        wait = _backoff(err, retry_count=2, is_console_go_overload=True, interactive=False)
        assert 300.0 <= wait <= 360.0

    def test_missing_fields_fall_through_to_ladder(self):
        err = SimpleNamespace(status_code=503, body={"error": {"message": "Service overloaded"}})
        wait = _backoff(err, retry_count=2, is_console_go_overload=True, interactive=False)
        assert 300.0 <= wait <= 360.0

    def test_expired_reset_falls_through(self):
        err = SimpleNamespace(
            status_code=429, body={"error": {"resets_at": time.time() - 60}})
        assert park_seconds_from_error(err) is None

    def test_non_adaptive_error_ignores_park(self, monkeypatch):
        monkeypatch.setattr(retry_utils, "jittered_backoff", lambda *a, **kw: kw["base_delay"])
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 900}})
        # Neither rate-limited nor overload: existing default backoff, unpacked.
        assert _backoff(err, retry_count=1) == 2.0

    def test_bool_and_empty_values_ignored(self):
        assert park_seconds_from_error(SimpleNamespace(body={"resets_in_seconds": True})) is None
        assert park_seconds_from_error(SimpleNamespace(body={"resets_at": ""})) is None
        assert park_seconds_from_error(SimpleNamespace(body=None)) is None


# ---------------------------------------------------------------------------
# Interactive guard: single sleep <= 300s, then fallback/surface
# ---------------------------------------------------------------------------

class TestInteractiveCap:
    def test_long_ladder_capped_interactive(self):
        err = _go_503()
        assert _backoff(err, retry_count=4, is_console_go_overload=True, interactive=True) == 300.0

    def test_long_ladder_uncapped_non_interactive(self):
        err = _go_503()
        wait = _backoff(err, retry_count=4, is_console_go_overload=True, interactive=False)
        assert 3600.0 <= wait <= 4320.0

    def test_park_capped_interactive(self):
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 3600}})
        assert _backoff(err, is_rate_limited=True, interactive=True) == 300.0

    def test_retry_after_capped_by_interactive_guard(self):
        # H1: Retry-After takes the "retry_after" policy tag, which is in the
        # interactive-capped set — a 600s header on a user-facing turn sleeps
        # 300s, then the loop re-enters toward fallback/surface. The 600s
        # provider-window cap itself is untouched (see non-interactive parity
        # below); only the single-sleep cap applies.
        err = SimpleNamespace(
            status_code=429, body={"error": {"code": "service_overloaded"}},
            response=SimpleNamespace(headers={"Retry-After": "600"}),
        )
        assert _backoff(err, retry_count=4, is_console_go_overload=True, interactive=True) == 300.0


# ---------------------------------------------------------------------------
# Shared plumbing: extract_api_error_context carries resets_in_seconds
# ---------------------------------------------------------------------------

class TestErrorContextPlumbing:
    def test_resets_in_seconds_becomes_reset_at(self):
        from agent.agent_runtime_helpers import extract_api_error_context
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 300}})
        before = time.time()
        ctx = extract_api_error_context(err)
        assert before + 290.0 < ctx["reset_at"] <= time.time() + 300.0

    def test_resets_at_still_wins_over_duration(self):
        from agent.agent_runtime_helpers import extract_api_error_context
        err = SimpleNamespace(
            status_code=429,
            body={"error": {"resets_at": time.time() + 999, "resets_in_seconds": 300}},
        )
        ctx = extract_api_error_context(err)
        assert abs(ctx["reset_at"] - (time.time() + 999)) < 5.0


# ---------------------------------------------------------------------------
# H1/H2 fix: Retry-After takes the capped "retry_after" policy; the
# interactive flag threads handle_api_error -> settle -> compute_error_backoff
# ---------------------------------------------------------------------------

def _retry_after_600_err():
    return SimpleNamespace(
        status_code=429, body={"error": {"code": "service_overloaded"}},
        response=SimpleNamespace(headers={"Retry-After": "600"}),
    )


class TestH1H2RetryAfterCapAndThreading:
    def test_retry_after_600_capped_interactive_rate_limited(self):
        # (1) Retry-After 600 + interactive -> 300 (rate-limited shape, no
        # Console Go ladder involved: the cap rides the policy tag alone).
        assert _backoff(_retry_after_600_err(), is_rate_limited=True, interactive=True) == 300.0

    def test_park_3600_uncapped_non_interactive(self):
        # (2) interactive=False + 3600 park -> 3600 uncapped: headless
        # callers honour the full provider window.
        err = SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 3600}})
        assert _backoff(err, is_rate_limited=True, interactive=False) == 3600.0

    def test_handle_api_error_forwards_interactive_false(self, monkeypatch):
        # (3) handle_api_error(interactive=False) forwards False all the way
        # to compute_error_backoff (mock assert on the deepest hop).
        import agent.turn_api_error as tae
        from agent.error_classifier import FailoverReason

        seen: dict[str, Any] = {}

        def fake_backoff(agent, api_error, **kw):
            seen.update(kw)
            return 0.0

        classified = SimpleNamespace(
            reason=FailoverReason.rate_limit, status_code=429, retryable=True,
            should_compress=False, should_rotate_credential=False, should_fallback=False,
        )
        route_verdict = SimpleNamespace(
            action="fallthrough", status_code=429, messages=[], active_system_prompt="sp",
            conversation_history=None, retry_count=1, max_retries=8, compression_attempts=0,
            is_rate_limited=True, wrapped_output_cap_budget=None,
            is_zai_coding_overload=False, is_console_go_overload=False,
            provider_overflow_recovery_pending=False, result=None,
        )
        overflow_verdict = SimpleNamespace(
            action="fallthrough", messages=[], active_system_prompt="sp",
            conversation_history=None, approx_tokens=10, compression_attempts=0,
            is_context_length_error=False, provider_overflow_recovery_pending=False,
            result=None,
        )
        monkeypatch.setattr(tae, "compute_error_backoff", fake_backoff)
        monkeypatch.setattr(tae, "interruptible_backoff_sleep", lambda *a, **k: None)
        monkeypatch.setattr(tae, "recover_before_classification", lambda *a, **k: (False, "sp"))
        monkeypatch.setattr(tae, "recover_after_classification", lambda *a, **k: (False, False))
        monkeypatch.setattr(
            tae, "log_api_error_attempt", lambda *a, **k: ("t", "m", "p", "b", "m"))
        monkeypatch.setattr(tae, "classify_api_error", lambda *a, **k: classified)
        monkeypatch.setattr(tae, "route_classified_error", lambda *a, **k: route_verdict)
        monkeypatch.setattr(tae, "recover_from_overflow", lambda *a, **k: overflow_verdict)
        import tools.interpreter_shutdown as _shut
        monkeypatch.setattr(_shut, "interpreter_shutting_down", lambda *a, **k: False)

        agent = SimpleNamespace(
            thinking_callback=None,
            _extract_api_error_context=lambda e: {},
            _invoke_api_request_error_hook=lambda **k: None,
            _touch_activity=lambda *a, **k: None,
            _interrupt_requested=False,
            log_prefix="", provider="", model="", base_url="", api_key=None,
        )
        retry = SimpleNamespace(
            primary_recovery_attempted=True, restart_with_redirected_messages=False,
        )
        verdict = tae.handle_api_error(
            agent, api_error=_retry_after_600_err(), _retry=retry, thinking_spinner=None,
            messages=[], api_messages=[{"role": "user", "content": "x"}], api_kwargs={},
            system_message=None, active_system_prompt="sp", conversation_history=None,
            approx_tokens=10, retry_count=0, max_retries=8, compression_attempts=0,
            max_compression_attempts=3, api_call_count=1, api_request_id="r",
            api_start_time=time.time(), effective_task_id="t", turn_id="t1",
            interactive=False,
        )
        assert seen.get("interactive") is False
        assert verdict.action == "fallthrough"

    def test_interactive_parity_same_input(self):
        # (4) Same Retry-After 600 input: interactive True -> 300,
        # interactive False -> 600 (provider-window cap untouched).
        assert _backoff(_retry_after_600_err(), is_rate_limited=True, interactive=True) == 300.0
        assert _backoff(_retry_after_600_err(), is_rate_limited=True, interactive=False) == 600.0
