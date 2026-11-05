"""Stall-interrupted preflight compression must persist a durable backoff.

#96775: an explicit /stop after the summary stream has already crossed the
no-progress stall window restores the original transcript but must record a
stall-specific cooldown so the next automatic turn does not re-enter the
same strategy. An ordinary early /stop stays cooldown-neutral.

Native Codex app-server compaction is a sibling path (#75364) and is not
covered here. Stall classification reads the commit fence's progress clock,
not Chat Completions vs Responses frame shapes.
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.conversation_compression import (
    STALL_INTERRUPTED_FAILURE_CLASS,
    CompressionCommitFence,
    compress_context,
    compression_attempt_stalled,
)
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def _build_agent(tmp_path: Path, session_id: str = "STALL_INTERRUPT_96775"):
    db = storage_handle(tmp_path)
    seed_session(db, session_id, "cli")
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            persistence=db,
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


def _age_fence(fence: CompressionCommitFence, idle_seconds: float) -> None:
    fence._last_progress = time.monotonic() - float(idle_seconds)


class TestStallClassificationIsFenceIdle:
    def test_fresh_fence_is_not_stalled(self):
        fence = CompressionCommitFence()
        assert compression_attempt_stalled(
            commit_fence=fence,
            started_at=time.monotonic(),
            idle_timeout_seconds=1.0,
        ) is False

    def test_idle_fence_is_stalled(self):
        fence = CompressionCommitFence()
        _age_fence(fence, 2.0)
        assert compression_attempt_stalled(
            commit_fence=fence,
            started_at=time.monotonic(),
            idle_timeout_seconds=1.0,
        ) is True

    def test_recent_progress_is_not_stalled(self):
        fence = CompressionCommitFence()
        _age_fence(fence, 2.0)
        fence.touch_progress()
        assert compression_attempt_stalled(
            commit_fence=fence,
            started_at=time.monotonic() - 5.0,
            idle_timeout_seconds=1.0,
        ) is False

    def test_chat_and_responses_share_the_fence_clock(self):
        """Transport-agnostic: both APIs tick the same fence or they don't."""
        chat_fence = CompressionCommitFence()
        responses_fence = CompressionCommitFence()
        _age_fence(chat_fence, 2.0)
        responses_fence.touch_progress()
        assert compression_attempt_stalled(
            commit_fence=chat_fence,
            started_at=time.monotonic(),
            idle_timeout_seconds=1.0,
        ) is True
        assert compression_attempt_stalled(
            commit_fence=responses_fence,
            started_at=time.monotonic(),
            idle_timeout_seconds=1.0,
        ) is False


class TestEarlyStopStaysNeutral:
    @pytest.mark.parametrize(("user_stop", "outcome"), [(True, "skipped"), (False, "failed")])
    def test_explicit_interrupt_before_stall_does_not_arm_cooldown(
        self, tmp_path: Path, monkeypatch, user_stop, outcome
    ):
        """Only the user's stop counts ``skipped``; a host timeout that cancels the fence (gateway hygiene
        budget, total ceiling) without the stop event is a real failure even before the stall window."""
        from hermes_cli.observability import shared_metrics_events

        recorded = []
        monkeypatch.setattr(shared_metrics_events, "record_compression", lambda **kw: recorded.append(kw["outcome"]))
        db, agent = _build_agent(tmp_path, "EARLY_STOP_96775")
        original = _messages()
        live = copy.deepcopy(original)
        fence = CompressionCommitFence()

        def _early_stop(messages, **_kwargs):
            messages[0]["content"] = "must be rolled back"
            if user_stop:
                agent._hard_interrupt_requested.set()
            fence.cancel_before_commit()
            raise AuxiliaryExplicitCancellation()

        agent.context_compressor.compress = _early_stop

        compressed, _prompt = compress_context(
            agent,
            live,
            "sys",
            approx_tokens=50_000,
            commit_fence=fence,
        )

        assert compressed == original
        assert live == original
        assert db.coordination.get_compression_lock_holder(session_id="EARLY_STOP_96775") is None
        assert db.messages.read_compression_cooldown("EARLY_STOP_96775", time.time()) is None
        assert agent.context_compressor.should_compress(50_000) is True
        assert recorded == [outcome]
        seed_messages(db, "EARLY_STOP_96775", {"role": "assistant", "content": "still writable"})


class TestStallInterruptedBackoff:
    def test_aux_explicit_cancel_after_stall_persists_backoff(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (1.0, 10.0),
        )
        db, agent = _build_agent(tmp_path, "STALL_AUX_CANCEL")
        original = _messages()
        live = copy.deepcopy(original)
        fence = CompressionCommitFence()
        compress_calls = {"n": 0}

        def _stalled_then_stop(messages, **_kwargs):
            compress_calls["n"] += 1
            _age_fence(fence, 2.0)
            messages[0]["content"] = "must be rolled back"
            raise AuxiliaryExplicitCancellation()

        agent.context_compressor.compress = _stalled_then_stop

        compressed, _prompt = compress_context(
            agent,
            live,
            "sys",
            approx_tokens=50_000,
            commit_fence=fence,
        )

        assert compressed == original
        assert live == original
        assert db.coordination.get_compression_lock_holder(session_id="STALL_AUX_CANCEL") is None
        state = db.messages.read_compression_cooldown("STALL_AUX_CANCEL", time.time())
        assert state is not None
        assert STALL_INTERRUPTED_FAILURE_CLASS in str(state.error)
        assert "msgs=20" in str(state.error)
        assert agent.context_compressor.should_compress(50_000) is False

        compress_calls["n"] = 0
        again, _ = compress_context(
            agent,
            copy.deepcopy(original),
            "sys",
            approx_tokens=50_000,
            commit_fence=CompressionCommitFence(),
        )
        assert again == original
        assert compress_calls["n"] == 0

        seed_messages(db, "STALL_AUX_CANCEL", {"role": "assistant", "content": "still writable"})

    def test_commit_fence_cancel_after_stall_persists_backoff(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (1.0, 10.0),
        )
        db, agent = _build_agent(tmp_path, "STALL_FENCE_CANCEL")
        original = _messages()
        live = copy.deepcopy(original)
        fence = CompressionCommitFence()

        def _summary_then_cancel(messages, **_kwargs):
            _age_fence(fence, 2.0)
            fence.cancel_before_commit()
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "assistant", "content": "tail"},
            ]

        agent.context_compressor.compress = _summary_then_cancel

        compressed, _prompt = compress_context(
            agent,
            live,
            "sys",
            approx_tokens=50_000,
            commit_fence=fence,
        )

        assert compressed == original
        assert live == original
        state = db.messages.read_compression_cooldown("STALL_FENCE_CANCEL", time.time())
        assert state is not None
        assert STALL_INTERRUPTED_FAILURE_CLASS in str(state.error)
        assert agent.context_compressor.should_compress(50_000) is False
        seed_messages(db, "STALL_FENCE_CANCEL", {"role": "user", "content": "next turn"})

    def test_commit_fence_cancel_with_fresh_progress_stays_neutral(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (1.0, 10.0),
        )
        db, agent = _build_agent(tmp_path, "FRESH_FENCE_CANCEL")
        original = _messages()
        live = copy.deepcopy(original)
        fence = CompressionCommitFence()

        def _healthy_then_cancel(messages, **_kwargs):
            fence.touch_progress()
            fence.cancel_before_commit()
            return [
                {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                {"role": "assistant", "content": "tail"},
            ]

        agent.context_compressor.compress = _healthy_then_cancel

        compressed, _prompt = compress_context(
            agent,
            live,
            "sys",
            approx_tokens=50_000,
            commit_fence=fence,
        )

        assert compressed == original
        assert db.messages.read_compression_cooldown("FRESH_FENCE_CANCEL", time.time()) is None
        assert agent.context_compressor.should_compress(50_000) is True

    def test_manual_force_bypasses_stall_backoff(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (1.0, 10.0),
        )
        _db, agent = _build_agent(tmp_path, "FORCE_BYPASS_STALL")
        original = _messages()
        fence = CompressionCommitFence()

        def _stalled_then_stop(messages, **_kwargs):
            _age_fence(fence, 2.0)
            raise AuxiliaryExplicitCancellation()

        agent.context_compressor.compress = _stalled_then_stop
        compress_context(
            agent,
            copy.deepcopy(original),
            "sys",
            approx_tokens=50_000,
            commit_fence=fence,
        )
        assert agent.context_compressor.should_compress(50_000) is False

        forced_calls = {"n": 0}

        def _forced(messages, **kwargs):
            forced_calls["n"] += 1
            assert kwargs.get("force") is True
            return messages

        agent.context_compressor.compress = _forced
        compress_context(
            agent,
            copy.deepcopy(original),
            "sys",
            approx_tokens=50_000,
            force=True,
            commit_fence=CompressionCommitFence(),
        )
        assert forced_calls["n"] == 1

    def test_stall_backoff_merges_with_longer_active_deadline(
        self, tmp_path: Path, monkeypatch
    ):
        monkeypatch.setattr(
            "agent.conversation_compression.resolve_context_compression_timeouts",
            lambda compression_cfg=None: (1.0, 10.0),
        )
        db, agent = _build_agent(tmp_path, "MERGE_MAX_STALL")
        agent.context_compressor._record_compression_failure_cooldown(
            900.0, "prior timeout"
        )
        before = db.messages.read_compression_cooldown("MERGE_MAX_STALL", time.time())
        assert before is not None
        prior_until = float(before.until)

        fence = CompressionCommitFence()

        def _stalled_then_stop(messages, **_kwargs):
            _age_fence(fence, 2.0)
            raise AuxiliaryExplicitCancellation()

        agent.context_compressor.compress = _stalled_then_stop
        # force=True so the pre-existing cooldown does not skip the attempt
        # before the stall-interrupt restore/merge path can run.
        compress_context(
            agent,
            _messages(),
            "sys",
            approx_tokens=50_000,
            force=True,
            commit_fence=fence,
        )

        after = db.messages.read_compression_cooldown("MERGE_MAX_STALL", time.time())
        assert after is not None
        assert float(after.until) >= prior_until - 1.0
        assert STALL_INTERRUPTED_FAILURE_CLASS in str(after.error)


def test_session_db_cooldown_write_does_not_shorten_longer_deadline(
    tmp_path: Path,
):
    db = storage_handle(tmp_path)
    seed_session(db, "merge-max", "cli")
    later = time.time() + 900.0
    sooner = time.time() + 60.0
    db.messages.record_compression_cooldown("merge-max", later, "timeout")
    db.messages.record_compression_cooldown(
        "merge-max", sooner, STALL_INTERRUPTED_FAILURE_CLASS
    )
    row = db.messages.read_compression_cooldown("merge-max", time.time())
    assert row is not None
    assert float(row.until) >= later - 1.0
    assert row.error == STALL_INTERRUPTED_FAILURE_CLASS
