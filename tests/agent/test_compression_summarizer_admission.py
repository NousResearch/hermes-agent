"""Candidate admission must ignore no-op entry claims (#112482).

``_begin_compression_attempt`` claims compressor-attribute ownership before
the lease and breaker gates. Sit-outs and other early returns therefore bump
``_compression_attempt_generation`` without invoking the summarizer. The
commit gate must not treat those claims as a newer working attempt, or a
completed summary is discarded and the session livelocks over threshold.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent.conversation_compression import (
    _candidate_rejected,
    _claim_compressor_attempt,
    _record_summarizer_invocation,
    compress_context,
)
from hermes_state import SessionDB


def _messages():
    return [{"role": "user", "content": f"m{i}"} for i in range(20)]


def _agent_for_gate(compressor):
    return SimpleNamespace(
        context_compressor=compressor,
        session_id="s",
        _last_compaction_in_place=True,
        _emit_warning=lambda _m: None,
    )


def _build_agent(tmp_path: Path, session_id: str):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._cached_system_prompt = "sys"
    agent.context_compressor.threshold_tokens = 1_000
    return db, agent


class TestNoopEntryClaimsDoNotSupersede:
    def test_candidate_gate_accepts_after_later_noop_claims(self):
        compressor = SimpleNamespace()
        in_flight = _claim_compressor_attempt(compressor)
        for _ in range(5):
            _claim_compressor_attempt(compressor)
        before = [{"role": "user", "content": "keep"}]
        candidate = [{"role": "assistant", "content": "completed summary"}]
        rejected = _candidate_rejected(
            _agent_for_gate(compressor),
            candidate,
            list(before),
            before,
            attempt_generation=in_flight,
            attempt_started_at=0.0,
        )
        assert rejected is False, (
            "no-op entry claims must not discard a completed candidate (#112482)"
        )

    def test_lock_sitout_during_summary_still_commits(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "SUMMARIZER_ADMISSION")
        live = _messages()
        nested_results = []
        entered = False

        def compress_with_lock_sitout(messages, **_kwargs):
            nonlocal entered
            if not entered:
                entered = True
                nested, _ = compress_context(
                    agent, list(messages), "sys", approx_tokens=500_000
                )
                nested_results.append(nested)
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                messages[-1],
            ]

        agent.context_compressor.compress = compress_with_lock_sitout
        out, _prompt = compress_context(agent, live, "sys", approx_tokens=500_000)

        assert nested_results == [live]
        assert len(out) == 2
        assert out[0]["content"] == "[CONTEXT COMPACTION] summary"
        assert db.get_compression_lock_holder("SUMMARIZER_ADMISSION") is None

    def test_claim_loop_during_summary_still_commits(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "CLAIM_LOOP")
        live = _messages()
        original = copy.deepcopy(live)
        summary = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "summary"},
        ]

        def compress_while_noop_claims(messages, **_kwargs):
            for _ in range(5):
                _claim_compressor_attempt(agent.context_compressor)
            return summary

        agent.context_compressor.compress = compress_while_noop_claims
        out, _prompt = compress_context(agent, live, "sys", approx_tokens=500_000)
        assert out == summary
        assert live != original or out is not live
        assert db.get_compression_lock_holder("CLAIM_LOOP") is None

    def test_later_summarizer_still_discards_late_candidate(self):
        compressor = SimpleNamespace()
        in_flight = _claim_compressor_attempt(compressor)
        _record_summarizer_invocation(compressor, in_flight)
        newer = _claim_compressor_attempt(compressor)
        _record_summarizer_invocation(compressor, newer)
        before = [{"role": "user", "content": "keep"}]
        candidate = [{"role": "assistant", "content": "stale summary"}]
        rejected = _candidate_rejected(
            _agent_for_gate(compressor),
            candidate,
            list(before),
            before,
            attempt_generation=in_flight,
            attempt_started_at=0.0,
        )
        assert rejected is True

    def test_cancelled_dispatch_without_summarizer_does_not_supersede(self):
        compressor = SimpleNamespace()
        in_flight = _claim_compressor_attempt(compressor)
        _record_summarizer_invocation(compressor, in_flight)
        _claim_compressor_attempt(compressor)
        before = [{"role": "user", "content": "keep"}]
        candidate = [{"role": "assistant", "content": "completed summary"}]
        rejected = _candidate_rejected(
            _agent_for_gate(compressor),
            candidate,
            list(before),
            before,
            attempt_generation=in_flight,
            attempt_started_at=0.0,
        )
        assert rejected is False
