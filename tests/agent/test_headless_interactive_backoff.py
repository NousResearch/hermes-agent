"""Phase2 follow-up (t_21ec84e1): headless signal threads to the backoff cap.

``resolve_turn_interactive``: an explicit per-turn value wins, otherwise the
agent's construction-time platform decides (cron/batch → False, everything
else → True). ``_LoopState`` carries it into ``handle_api_error`` via
``_run_phase``, so a cron-shaped turn honours the full provider park while
the interactive default still caps one sleep at 300s.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.real_retry_backoff

from agent.conversation_loop import (
    _LoopState,
    _run_phase,
    resolve_turn_interactive,
)
from agent.turn_recovery import compute_error_backoff


def _agent(platform: Any) -> Any:
    return MagicMock(platform=platform)


def _backoff(err: Any, **kw: Any) -> float:
    params: dict[str, Any] = dict(
        retry_count=1, max_retries=8, is_rate_limited=False,
        is_zai_coding_overload=False, base_url="https://api.example.test/v1",
        model="test-model",
    )
    params.update(kw)
    return compute_error_backoff(MagicMock(), err, **params)


def _park_3600_err() -> Any:
    return SimpleNamespace(status_code=429, body={"error": {"resets_in_seconds": 3600}})


def _retry_after_600_err() -> Any:
    return SimpleNamespace(
        status_code=429, body={"error": {"code": "service_overloaded"}},
        response=SimpleNamespace(headers={"Retry-After": "600"}),
    )


def _loop_state(interactive: bool) -> _LoopState:
    return _LoopState(
        user_message=None, system_message=None, moa_config=None,
        original_user_message=None, conversation_history=None,
        effective_task_id="task", turn_id="turn", _should_review_memory=None,
        _plugin_user_context=None, _ext_prefetch_cache=None, messages=None,
        active_system_prompt=None, current_turn_user_idx=None,
        _preflight_compression_blocked=None, max_compression_attempts=3,
        interactive=interactive,
    )


# ---------------------------------------------------------------------------
# Signal resolution: explicit wins, else construction-time platform
# ---------------------------------------------------------------------------

class TestResolveTurnInteractive:
    def test_interactive_default_unchanged(self):
        # Current callers (CLI/gateway/unknown surfaces) keep the 300s cap:
        # platform None, empty, cli-like, and agents without the attribute.
        assert resolve_turn_interactive(_agent("cli")) is True
        assert resolve_turn_interactive(_agent("telegram")) is True
        assert resolve_turn_interactive(_agent(None)) is True
        assert resolve_turn_interactive(_agent("")) is True
        assert resolve_turn_interactive(MagicMock(spec=[])) is True

    def test_headless_platforms_resolve_false(self):
        assert resolve_turn_interactive(_agent("cron")) is False
        assert resolve_turn_interactive(_agent("batch")) is False
        # Construction sites pass lowercase, but resolution is case-tolerant.
        assert resolve_turn_interactive(_agent("Cron")) is False
        assert resolve_turn_interactive(_agent("BATCH")) is False

    def test_subagent_stays_interactive(self):
        # Out of scope for this card (cron/batch only): delegation timing is
        # unchanged, so subagent turns keep the interactive cap by default.
        assert resolve_turn_interactive(_agent("subagent")) is True

    def test_explicit_value_always_wins(self):
        assert resolve_turn_interactive(_agent("cron"), True) is True
        assert resolve_turn_interactive(_agent("batch"), True) is True
        assert resolve_turn_interactive(_agent("cli"), False) is False
        assert resolve_turn_interactive(_agent(None), False) is False

    def test_loop_state_default_true(self):
        assert _loop_state(True).interactive is True
        assert _LoopState(
            user_message=None, system_message=None, moa_config=None,
            original_user_message=None, conversation_history=None,
            effective_task_id="t", turn_id="t", _should_review_memory=None,
            _plugin_user_context=None, _ext_prefetch_cache=None, messages=None,
            active_system_prompt=None, current_turn_user_idx=None,
            _preflight_compression_blocked=None, max_compression_attempts=3,
        ).interactive is True


# ---------------------------------------------------------------------------
# Phase threading: _LoopState.interactive reaches the phase helper by name
# (the same _run_phase hop handle_api_error travels)
# ---------------------------------------------------------------------------

@dataclass
class _StubVerdict:
    action: str
    interactive: bool
    result: Any = None


class TestLoopStatePhaseThreading:
    def test_false_threads_through_run_phase(self):
        seen: dict[str, Any] = {}

        def _stub_phase(agent: Any, interactive: bool) -> _StubVerdict:
            seen["interactive"] = interactive
            return _StubVerdict(action="return", interactive=interactive)

        state = _loop_state(False)
        verdict = _run_phase(_stub_phase, MagicMock(), state)
        assert seen["interactive"] is False
        assert verdict.interactive is False
        assert state.interactive is False

    def test_true_round_trips_unchanged(self):
        seen: dict[str, Any] = {}

        def _stub_phase(agent: Any, interactive: bool) -> _StubVerdict:
            seen["interactive"] = interactive
            return _StubVerdict(action="return", interactive=interactive)

        state = _loop_state(True)
        _run_phase(_stub_phase, MagicMock(), state)
        assert seen["interactive"] is True
        assert state.interactive is True


# ---------------------------------------------------------------------------
# Backoff parity: cron-shaped turns honour full parks, interactive caps
# ---------------------------------------------------------------------------

class TestHeadlessParkParity:
    def test_cron_shaped_turn_honours_full_park(self):
        interactive = resolve_turn_interactive(_agent("cron"))
        assert interactive is False
        assert _backoff(_park_3600_err(), is_rate_limited=True, interactive=interactive) == 3600.0

    def test_batch_shaped_turn_honours_full_park(self):
        interactive = resolve_turn_interactive(_agent("batch"))
        assert interactive is False
        assert _backoff(_park_3600_err(), is_rate_limited=True, interactive=interactive) == 3600.0

    def test_interactive_default_still_caps_park(self):
        interactive = resolve_turn_interactive(_agent("cli"))
        assert interactive is True
        assert _backoff(_park_3600_err(), is_rate_limited=True, interactive=interactive) == 300.0

    def test_park_parity_both_modes(self):
        err = _park_3600_err()
        assert _backoff(err, is_rate_limited=True, interactive=True) == 300.0
        assert _backoff(err, is_rate_limited=True, interactive=False) == 3600.0

    def test_retry_after_parity_both_modes(self):
        err = _retry_after_600_err()
        assert _backoff(err, is_rate_limited=True, interactive=True) == 300.0
        assert _backoff(err, is_rate_limited=True, interactive=False) == 600.0
