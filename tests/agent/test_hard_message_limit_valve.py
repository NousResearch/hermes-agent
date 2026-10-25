"""Hard message-count safety valve (#56034) — the bounded recovery contract.

A count breach (``compression.hygiene_hard_message_limit`` reached) forces compaction
through the REAL ``_compress_context(force=True)`` pipeline at both guard sites — the
turn-start preflight (``agent.turn_context_compaction``) and the in-turn pre-API guard
(``agent.turn_preflight.run_preflight_compression``) — past the summary-failure cooldown
and the anti-thrash breaker (two ineffective compactions). Only the summary LLM boundary
(``call_llm``) is stubbed; the engine gate, the cooldown/strike state, the would-grow
guard and the commit path are real. Without a breach the same guards keep deferring.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.turn_context_compaction import hard_message_limit_breached, run_turn_start_compaction
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression
from hermes_state import SessionDB

# Below/above the transcript produced by _transcript() (41 rows with the user turn).
BREACH_LIMIT = 30
NO_BREACH_LIMIT = 500


def _make_agent(tmp_path: Path, tag: str):
    """Real AIAgent + real ContextCompressor + real compress_context (stall-test shape)."""
    db = SessionDB(db_path=Path(tmp_path) / f"state-{tag}.db")
    session_id = f"HARD_VALVE_{tag}"
    db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, session_db=db, session_id=session_id,
            skip_context_files=True, skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    compressor = agent.context_compressor
    compressor.threshold_tokens = 1_000
    # The provider is known to omit usage, so rough estimates decide (no deferral).
    compressor.note_usage_less_response()
    compressor.hygiene_hard_message_limit = NO_BREACH_LIMIT
    agent.iteration_budget = MagicMock()
    agent._api_call_count = 1
    return agent


def _transcript():
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + ("lorem ipsum " * 300)}
        for i in range(40)
    ]


@contextlib.contextmanager
def _summary_stub():
    """LLM boundary only: a well-formed summary response, instantly."""
    resp = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content="COMPACTED SUMMARY of the earlier turns."),
            finish_reason="stop",
        )]
    )
    with (
        patch("agent.context_compressor.call_llm", return_value=resp),
        patch(
            "agent.auxiliary_client._get_auxiliary_task_config",
            return_value={"fallback_chain": []},
        ),
    ):
        yield


def _arm_cooldown(agent) -> None:
    """Arm the summary-failure cooldown in memory AND durably (the gate refreshes)."""
    agent.context_compressor._summary_failure_cooldown_until = time.monotonic() + 600
    agent._session_db.record_compression_failure_cooldown(
        agent.session_id, time.time() + 600, "timeout"
    )


def _trip_breaker(agent) -> None:
    """Two ineffective compactions: the anti-thrash breaker, in memory AND durably."""
    agent.context_compressor._ineffective_compression_count = 2
    agent._session_db.set_compression_ineffective_count(agent.session_id, 2)


def _turn_start(agent, *, user_message="hello"):
    messages = _transcript() + [{"role": "user", "content": user_message}]
    out = run_turn_start_compaction(
        agent, messages=messages, system_message="sys", active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=len(messages) - 1,
        user_message=user_message, effective_task_id="t",
    )
    return messages, out


def _in_turn(agent, messages, *, blocked=False, attempts=0, max_attempts=3):
    v = PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None,
        messages=list(messages), active_system_prompt="sys", conversation_history=None,
        api_call_count=1, compression_attempts=attempts, final_response=None, failed=False,
        _turn_exit_reason="unknown", _compression_timeout_exhausted=False,
        _preflight_compression_blocked=blocked,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=None,
    )
    return run_preflight_compression(
        agent, v, compressor=agent.context_compressor,
        request_pressure_tokens=50_000, provider_overflow_preflight=False,
        defer_preflight=lambda _t: False, moa_prepared_request=None,
        system_message="sys", user_message="hello",
        max_compression_attempts=max_attempts, effective_task_id="t",
    )


class TestHardMessageLimitBreached:
    """The trigger is count-only; doubles and junk never trip it."""

    def test_zero_limit_disables(self):
        assert hard_message_limit_breached(
            SimpleNamespace(hygiene_hard_message_limit=0), [{}] * 100
        ) is False

    def test_reaching_the_limit_is_a_breach(self):
        c = SimpleNamespace(hygiene_hard_message_limit=10)
        assert hard_message_limit_breached(c, [{}] * 9) is False
        assert hard_message_limit_breached(c, [{}] * 10) is True

    def test_missing_or_junk_limit_never_breaches(self):
        assert hard_message_limit_breached(SimpleNamespace(), [{}] * 100) is False
        assert hard_message_limit_breached(
            SimpleNamespace(hygiene_hard_message_limit=MagicMock()), [{}] * 100
        ) is False
        assert hard_message_limit_breached(
            SimpleNamespace(hygiene_hard_message_limit="10"), [{}] * 100
        ) is False


class TestTurnStartValve:
    """``run_turn_start_compaction`` — the turn-start preflight chain."""

    def test_count_breach_forces_compaction_during_summary_failure_cooldown(
        self, tmp_path
    ):
        agent = _make_agent(tmp_path, "TS_COOLDOWN")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        _arm_cooldown(agent)
        with _summary_stub():
            messages, out = _turn_start(agent)
        assert out.compressed is True
        assert len(out.messages) < len(messages), (
            "a count breach must commit a real compaction even while the "
            "summary-failure cooldown is armed"
        )

    def test_count_breach_forces_compaction_after_two_ineffective_compactions(
        self, tmp_path
    ):
        agent = _make_agent(tmp_path, "TS_THRASH")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        _trip_breaker(agent)
        with _summary_stub():
            messages, out = _turn_start(agent)
        assert out.compressed is True
        assert len(out.messages) < len(messages), (
            "a count breach must commit a real compaction even while the "
            "anti-thrash breaker is tripped (two ineffective compactions)"
        )

    def test_cooldown_still_defers_without_a_breach(self, tmp_path):
        agent = _make_agent(tmp_path, "TS_COOLDOWN_CTRL")
        _arm_cooldown(agent)
        with _summary_stub():
            messages, out = _turn_start(agent)
        assert out.compressed is False
        assert out.messages is messages, (
            "without a count breach the summary-failure cooldown must still defer"
        )

    def test_anti_thrash_still_defers_without_a_breach(self, tmp_path):
        agent = _make_agent(tmp_path, "TS_THRASH_CTRL")
        _trip_breaker(agent)
        with _summary_stub():
            messages, out = _turn_start(agent)
        assert out.compressed is False
        assert out.messages is messages, (
            "without a count breach the anti-thrash breaker must still defer"
        )

    def test_valve_excluded_for_codex_native_threads(self, tmp_path):
        """Documented exclusion: the provider owns compaction on native Codex threads."""
        agent = _make_agent(tmp_path, "TS_CODEx")
        agent.api_mode = "codex_app_server"
        agent.codex_app_server_auto_compaction = "native"
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        with _summary_stub():
            messages, out = _turn_start(agent)
        assert out.compressed is False
        assert out.messages is messages


class TestInTurnPreApiValve:
    """``agent.turn_preflight.run_preflight_compression`` — the in-turn pre-API guard."""

    def test_count_breach_forces_compaction_during_summary_failure_cooldown(
        self, tmp_path
    ):
        agent = _make_agent(tmp_path, "IT_COOLDOWN")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        _arm_cooldown(agent)
        messages = _transcript()
        with _summary_stub():
            v = _in_turn(agent, messages)
        assert v.action == "continue"
        assert len(v.messages) < len(messages), (
            "the in-turn guard must force a real compaction on a count breach "
            "even while the summary-failure cooldown is armed"
        )

    def test_count_breach_forces_compaction_after_two_ineffective_compactions(
        self, tmp_path
    ):
        agent = _make_agent(tmp_path, "IT_THRASH")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        _trip_breaker(agent)
        messages = _transcript()
        with _summary_stub():
            v = _in_turn(agent, messages)
        assert v.action == "continue"
        assert len(v.messages) < len(messages), (
            "the in-turn guard must force a real compaction on a count breach "
            "even while the anti-thrash breaker is tripped"
        )

    def test_cooldown_still_defers_without_a_breach(self, tmp_path):
        agent = _make_agent(tmp_path, "IT_COOLDOWN_CTRL")
        _arm_cooldown(agent)
        messages = _transcript()
        with _summary_stub():
            v = _in_turn(agent, messages)
        assert v.action == "fallthrough"
        assert len(v.messages) == len(messages), (
            "without a count breach the in-turn guard must still defer to the cooldown"
        )

    def test_valve_respects_the_per_turn_attempt_cap(self, tmp_path):
        """Documented bound: the shared compression attempt budget caps valve passes."""
        agent = _make_agent(tmp_path, "IT_CAP")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        messages = _transcript()
        with _summary_stub():
            v = _in_turn(agent, messages, attempts=3, max_attempts=3)
        assert v.action == "fallthrough"
        assert len(v.messages) == len(messages)

    def test_valve_respects_the_insufficient_progress_blocker(self, tmp_path):
        """Documented bound: a no-progress pass stops further valve passes this turn."""
        agent = _make_agent(tmp_path, "IT_BLOCKED")
        agent.context_compressor.hygiene_hard_message_limit = BREACH_LIMIT
        messages = _transcript()
        with _summary_stub():
            v = _in_turn(agent, messages, blocked=True)
        assert v.action == "fallthrough"
        assert len(v.messages) == len(messages)
