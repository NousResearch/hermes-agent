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

    def test_retry_after_not_capped_by_interactive_guard(self):
        err = SimpleNamespace(
            status_code=429, body={"error": {"code": "service_overloaded"}},
            response=SimpleNamespace(headers={"Retry-After": "600"}),
        )
        assert _backoff(err, retry_count=4, is_console_go_overload=True, interactive=True) == 600.0


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
