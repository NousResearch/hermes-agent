"""Attempt-lifecycle regression tests for #97488 / #96775.

Pins the three attempt-level lifecycle guarantees added after PR #98628
collapsed lean compaction to one auxiliary request per attempt:

1. A ceiling/idle-timeout host TEARS DOWN its worker: a cooperative worker is
   joined within the bounded grace (releasing the lease normally), while an
   uninterruptible worker is orphaned behind the poison fence — its late
   result is discarded and, on the total-ceiling path, the durable lease is
   retained until it exits so no new attempt overlaps the unchanged session.
2. A failed/stalled/cancelled attempt records a durable per-session backoff
   (strategy + failure kind stamped into the state.db cooldown row) that
   SURVIVES a gateway restart; the next automatic turn skips the same
   strategy inside the window, and a successful compression clears it.
3. Late results from a superseded attempt are discarded, never committed
   over newer state (generation counter), and a transiently-blocked no-op is
   reported as a soft defer — never compression_exhausted (false auto-reset).
"""

from __future__ import annotations

import copy
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.conversation_compression import (
    CompressionCommitFence,
    _claim_compressor_attempt,
    compress_context,
    compression_blocked_transiently,
    run_compress_context_with_progress_timeout,
)
from hermes_state import SessionDB


def _build_agent(tmp_path: Path, session_id: str, db: SessionDB | None = None):
    if db is None:
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


class TestWorkerTeardownOnCeiling:
    def test_cooperative_worker_joined_within_grace(self):
        """A worker that exits promptly after cancel is joined; the lease is
        released normally (no retention) — the sabotage check for this test
        is removing the `_join_cancelled_worker` call, which makes
        `worker_done_when_host_returned` False."""
        original = [{"role": "user", "content": "keep"}]
        worker_done = threading.Event()

        def cooperative_worker(fence: CompressionCommitFence):
            # Poll the poison fence like the production worker does between
            # provider phases; exit as soon as cancellation is visible.
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if fence.is_cancelled:
                    break
                fence_idle = fence.seconds_since_progress()
                assert fence_idle >= 0  # precondition: fence clock live
                time.sleep(0.01)
            worker_done.set()
            return (original, "late")

        fence = CompressionCommitFence()
        msgs, prompt = run_compress_context_with_progress_timeout(
            worker=cooperative_worker,
            messages=original,
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.05,
            total_ceiling_seconds=0.15,
            fence=fence,
            stall_fallback=False,
        )
        # The bounded-grace join must have reaped the cooperative worker
        # BEFORE the host returned.
        assert worker_done.is_set(), (
            "host returned before tearing down a cooperative cancelled "
            "worker — bounded-grace join missing (#97488)"
        )
        assert prompt == "fallback"
        # Teardown proved quiescence, so the lease must NOT stay retained.
        assert fence._retain_cancelled_lock_until_worker_done is False

    def test_uninterruptible_worker_is_orphaned_with_lease_retained(self):
        """A worker stuck in an uninterruptible provider call is orphaned:
        the host returns after the grace, the poison fence discards its late
        result, and on the total-ceiling path the durable lease release hook
        does NOT fire while the worker is alive (no overlap window)."""
        original = [{"role": "user", "content": "keep"}]
        release = threading.Event()
        worker_finished = threading.Event()
        lock_released: list[float] = []

        def stuck_worker(fence: CompressionCommitFence):
            # Continuous progress so only the TOTAL ceiling can expire
            # (the #97488 'last progress 0.0s ago' shape).
            while not release.wait(timeout=0.02):
                fence.touch_progress()
            worker_finished.set()
            if not fence.begin_commit():
                return (original, "")
            try:
                return ([{"role": "assistant", "content": "late"}], "late")
            finally:
                fence.finish_commit()

        fence = CompressionCommitFence()
        fence.register_cancelled_lock_release(
            lambda: lock_released.append(time.monotonic())
        )
        msgs, prompt = run_compress_context_with_progress_timeout(
            worker=stuck_worker,
            messages=original,
            system_prompt_fallback="fallback",
            idle_timeout_seconds=0.1,
            total_ceiling_seconds=0.3,
            fence=fence,
            stall_fallback=False,
        )
        # Precondition: the worker is genuinely still running.
        assert not worker_finished.is_set()
        assert msgs is original and prompt == "fallback"
        # Total-ceiling path: lease retained until the worker exits, so no
        # new attempt can overlap the unchanged session.
        assert not lock_released, (
            "durable lease released while the timed-out worker was still "
            "alive — overlap window reopened (#97488)"
        )
        release.set()
        assert worker_finished.wait(timeout=2)
        # Late result was fence-poisoned, never adopted.
        assert msgs == [{"role": "user", "content": "keep"}]


class TestDurableAttemptBackoff:
    def test_backoff_row_records_strategy_and_kind(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_KIND")
        agent.context_compressor.record_timeout_failure(
            "host ceiling exhausted", failure_kind="ceiling_exhausted"
        )
        row = db.get_compression_failure_cooldown("BACKOFF_KIND")
        assert row is not None, "backoff must persist to state.db"
        assert row["remaining_seconds"] > 0
        assert "backoff:ceiling_exhausted:strategy=lean" in (row["error"] or "")

    def test_backoff_blocks_same_strategy_reentry_next_turn(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_REENTRY")
        agent.context_compressor.record_timeout_failure(
            "stall", failure_kind="stalled"
        )
        # Precondition: over threshold, so only the backoff can block.
        assert 500_000 >= agent.context_compressor.threshold_tokens
        should, reason = agent.context_compressor.should_compress_info(500_000)
        assert should is False
        assert reason and reason.startswith("cooldown"), (
            "next-turn re-entry of the same strategy must be skipped inside "
            "the backoff window (#96775)"
        )

    def test_backoff_survives_simulated_gateway_restart(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_RESTART")
        agent.context_compressor.record_timeout_failure(
            "stall before restart", failure_kind="stall_interrupted"
        )
        # Precondition: row is durable in this DB file.
        assert db.get_compression_failure_cooldown("BACKOFF_RESTART")
        # Simulated restart: brand-new SessionDB handle + brand-new agent
        # objects rebuilt from the same state.db file.
        db2 = SessionDB(db_path=tmp_path / "state.db")
        _db2, agent2 = _build_agent(tmp_path, "BACKOFF_RESTART", db=db2)
        cooldown = agent2.context_compressor.get_active_compression_failure_cooldown(
            refresh=True
        )
        assert cooldown is not None, (
            "backoff must survive a gateway restart via state.db (#96775)"
        )
        assert "stall_interrupted" in (cooldown["error"] or "")
        should, reason = agent2.context_compressor.should_compress_info(500_000)
        assert should is False and reason.startswith("cooldown")

    def test_success_clears_backoff(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "BACKOFF_CLEAR")
        compressor = agent.context_compressor
        compressor.record_timeout_failure("stall", failure_kind="stalled")
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR")
        # What a successful compression does on commit:
        compressor._clear_compression_failure_cooldown()
        assert db.get_compression_failure_cooldown("BACKOFF_CLEAR") is None
        assert compressor.should_compress_info(500_000)[0] is True


class TestSupersessionDiscardsLateResults:
    def test_superseded_attempt_candidate_never_commits(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "SUPERSEDE")
        live = _messages()
        original = copy.deepcopy(live)

        def compress_and_get_superseded(messages, **_kwargs):
            # While this attempt's summary was in flight, a NEWER attempt
            # claimed the compressor (what a retry/fallback does).
            _claim_compressor_attempt(agent.context_compressor)
            return [{"role": "assistant", "content": "stale summary"}]

        agent.context_compressor.compress = compress_and_get_superseded
        out, _prompt = compress_context(
            agent, live, "sys", approx_tokens=500_000
        )
        assert out == original, (
            "late candidate from a superseded attempt must be discarded, "
            "never committed over newer state (#97488)"
        )
        assert live == original
        # Session stayed writable and unrotated.
        assert db.get_compression_lock_holder("SUPERSEDE") is None
        db.append_message("SUPERSEDE", "assistant", "still writable")


class TestTransientBlockIsNotExhaustion:
    def test_cooldown_blocked_noop_sets_transient_signal(self, tmp_path: Path):
        db, agent = _build_agent(tmp_path, "TRANSIENT_SIGNAL")
        agent.context_compressor.record_timeout_failure(
            "host ceiling", failure_kind="ceiling_exhausted"
        )
        live = _messages()
        before = copy.deepcopy(live)
        out, _ = compress_context(agent, live, "sys", approx_tokens=500_000)
        # Preconditions: the pass no-oped and it was NOT a lock skip.
        assert out == before
        assert getattr(agent, "_compression_skipped_due_to_lock", None) is None
        assert compression_blocked_transiently(agent) is True, (
            "a cooldown-blocked no-op must be distinguishable from "
            "exhaustion or the gateway falsely auto-resets (#97488)"
        )

    def test_signal_cleared_per_attempt_and_not_set_when_unblocked(
        self, tmp_path: Path
    ):
        db, agent = _build_agent(tmp_path, "TRANSIENT_CLEAR")
        # Stale signal from a previous pass must not leak.
        agent._compression_blocked_transient = "cooldown:999"
        agent.context_compressor.compress = lambda messages, **kw: list(messages)
        live = _messages()
        compress_context(agent, live, "sys", approx_tokens=500_000)
        assert compression_blocked_transiently(agent) is False

    def test_type_pinned_against_magicmock_agents(self):
        from unittest.mock import MagicMock

        mock_agent = MagicMock()
        # MagicMock auto-attributes are truthy but not str.
        assert compression_blocked_transiently(mock_agent) is False
