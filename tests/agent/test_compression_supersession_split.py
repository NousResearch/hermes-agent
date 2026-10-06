"""Candidate supersession must key on working attempts, not entry claims.

Regression for the compression livelock: ``_begin_compression_attempt()``
claims the monotonic generation on entry — before the breaker gates and the
per-session lock — so every early no-op return (lock sit-outs, transient
gates) has already bumped the shared counter without doing any work. The
commit-side gate then discarded a *completed* summary as "superseded" by
those no-op claims: nothing committed, the session stayed over threshold and
re-triggered compression every turn.
"""

import copy
from pathlib import Path
from unittest.mock import patch

import os

from agent.conversation_compression import (
    _claim_compressor_attempt,
    _mark_compressor_working_attempt,
    compress_context,
)
from hermes_state import SessionDB


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


def _messages():
    return [{"role": "user", "content": f"m{i}"} for i in range(20)]


class TestSupersessionKeysOnWorkingAttempts:
    def test_noop_claims_do_not_discard_a_completed_summary(self, tmp_path: Path):
        """Lock sit-outs claim the entry generation without doing work; the
        attempt that actually ran the summary must still commit. Sabotage:
        removing the working-attempt split regresses this to the livelock
        (candidate discarded, input returned unchanged)."""
        db, agent = _build_agent(tmp_path, "LIVELOCK")
        live = _messages()
        original = copy.deepcopy(live)
        summary = [
            {"role": "user", "content": "m0"},
            {"role": "assistant", "content": "summary"},
        ]

        def compress_while_sitouts_claim(messages, **_kwargs):
            # Sit-outs while this summary is in flight: each claims the entry
            # generation and returns without touching a provider (what
            # _sit_out_lock_contention / transient gates do on other paths).
            for _ in range(5):
                _claim_compressor_attempt(agent.context_compressor)
            return summary

        agent.context_compressor.compress = compress_while_sitouts_claim
        out, _prompt = compress_context(agent, live, "sys", approx_tokens=500_000)
        assert out == summary, (
            "a completed summary was discarded as superseded by no-op entry "
            "claims — the compression livelock (#112482)"
        )
        assert live != original or out is not live
        assert db.get_compression_lock_holder("LIVELOCK") is None

    def test_genuinely_newer_working_attempt_still_discards_late_candidate(
        self, tmp_path: Path
    ):
        """A newer attempt that began its own summary work (stall-fallback
        overlap) still supersedes: the late candidate is discarded, never
        committed over newer state."""
        db, agent = _build_agent(tmp_path, "WORKING_SUPERSEDE")
        live = _messages()
        original = copy.deepcopy(live)

        def compress_and_get_superseded(messages, **_kwargs):
            # While this attempt's summary was in flight, a NEWER attempt
            # claimed the compressor AND began its own summary work (what a
            # retry/fallback does when it reaches dispatch).
            newer = _claim_compressor_attempt(agent.context_compressor)
            _mark_compressor_working_attempt(agent.context_compressor, newer)
            return [{"role": "assistant", "content": "stale summary"}]

        agent.context_compressor.compress = compress_and_get_superseded
        out, _prompt = compress_context(agent, live, "sys", approx_tokens=500_000)
        assert out == original, (
            "late candidate from an attempt superseded by a newer WORKING "
            "attempt must be discarded, never committed over newer state"
        )
        assert live == original
        assert db.get_compression_lock_holder("WORKING_SUPERSEDE") is None
        db.append_message("WORKING_SUPERSEDE", "assistant", "still writable")
