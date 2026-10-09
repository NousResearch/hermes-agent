"""Regression: out-of-order completion of overlapping ``_compress_context`` calls
must not corrupt ``_active_compression_commit_fence``.

The fence slot used to be a save/restore cell: attempt B saved previous=A and
published B. When A completed first, its finally popped the slot, deleting B's
live fence mid-attempt (hard_interrupt lost the handle serializing cancel
admission against B's begin_commit). B's finally then republished A's dead
fence, which lingered until the next compression. The slot must instead track
the newest *live* attempt: a registration stack where each attempt removes only
its own entry.
"""

import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent.compression_facade as compression_facade
import agent.conversation_compression as conversation_compression
from agent.conversation_compression import CompressionCommitFence
from hermes_state import SessionDB


def _build_agent(db: SessionDB, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            platform="telegram",
            quiet_mode=True,
            session_db=db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    compressor = MagicMock()
    compressor.compression_count = 0
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_summary_auth_failure = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    agent.context_compressor = compressor
    return agent


def _overlapping_attempts(agent, messages):
    """Drive two _compress_context attempts whose compress_context calls block on
    release events; returns the controls needed to order their completion."""
    fence_a, fence_b = CompressionCommitFence(), CompressionCommitFence()
    entered_a, entered_b = threading.Event(), threading.Event()
    release_a, release_b = threading.Event(), threading.Event()
    errors = []

    def _fake_compress(_agent, _msgs, _sysmsg, *, commit_fence=None, **_kw):
        if commit_fence is fence_a:
            entered_a.set()
            assert release_a.wait(10), "attempt A never released"
        elif commit_fence is fence_b:
            entered_b.set()
            assert release_b.wait(10), "attempt B never released"
        return [{"role": "user", "content": "x"}], "prompt"

    def _attempt(fence):
        try:
            agent._compress_context(messages, "sys", commit_fence=fence)
        except Exception as exc:  # noqa: BLE001 - surfaced via assertion below
            errors.append(exc)

    return fence_a, fence_b, entered_a, entered_b, release_a, release_b, errors, _fake_compress, _attempt


class TestCommitFenceCompletionOrder:
    def test_earlier_attempt_finishing_first_keeps_newer_live_fence(
        self, tmp_path: Path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("S", source="cli")
        agent = _build_agent(db, "S")
        messages = [{"role": "user", "content": "live question"}]

        (fence_a, fence_b, entered_a, entered_b, release_a, release_b,
         errors, fake, attempt) = _overlapping_attempts(agent, messages)
        monkeypatch.setattr(conversation_compression, "compress_context", fake)

        thread_a = threading.Thread(target=attempt, args=(fence_a,), daemon=True)
        thread_b = threading.Thread(target=attempt, args=(fence_b,), daemon=True)
        thread_a.start()
        assert entered_a.wait(5)
        thread_b.start()
        assert entered_b.wait(5)
        assert vars(agent)["_active_compression_commit_fence"] is fence_b

        # A completes while B is still in its try block: B's fence must survive.
        release_a.set()
        thread_a.join(10)
        assert vars(agent).get("_active_compression_commit_fence") is fence_b

        release_b.set()
        thread_b.join(10)
        assert "_active_compression_commit_fence" not in vars(agent)
        assert not errors

    def test_lifo_completion_restores_outer_fence_while_it_runs(
        self, tmp_path: Path, monkeypatch
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("S", source="cli")
        agent = _build_agent(db, "S")
        messages = [{"role": "user", "content": "live question"}]

        (fence_a, fence_b, entered_a, entered_b, release_a, release_b,
         errors, fake, attempt) = _overlapping_attempts(agent, messages)
        monkeypatch.setattr(conversation_compression, "compress_context", fake)

        thread_a = threading.Thread(target=attempt, args=(fence_a,), daemon=True)
        thread_b = threading.Thread(target=attempt, args=(fence_b,), daemon=True)
        thread_a.start()
        assert entered_a.wait(5)
        thread_b.start()
        assert entered_b.wait(5)

        # B completes first: A's still-running attempt gets its fence back.
        release_b.set()
        thread_b.join(10)
        assert vars(agent).get("_active_compression_commit_fence") is fence_a

        release_a.set()
        thread_a.join(10)
        assert "_active_compression_commit_fence" not in vars(agent)
        assert not errors


def _capture_new_fence(monkeypatch):
    """Stub the progress-timeout runner; return the dict that receives its kwargs."""
    captured = {}

    def _stub_runner(**kw):
        captured.update(kw)
        return (["m"], "p")

    monkeypatch.setattr(
        conversation_compression, "run_compress_context_with_progress_timeout", _stub_runner)
    monkeypatch.setattr(
        conversation_compression, "request_exceeds_model_window", lambda *a, **k: False)
    return captured


class TestStallFallbackFencePublication:
    """``_publish_new_fence`` swaps the owning registration's fence and publishes to the
    slot only while that attempt still holds the top registration."""

    def _drive(self, monkeypatch, agent, registration, fence, lock):
        captured = _capture_new_fence(monkeypatch)
        compression_facade._run_under_progress_timeout(
            agent, lambda *a, **k: (["m"], "p"), [{"role": "user", "content": "x"}], "sys",
            active_fence=fence, registration=registration, fence_registration_lock=lock,
            idle_timeout=1, total_ceiling=2)
        return captured["new_fence"]

    def test_retry_fence_publishes_while_attempt_owns_slot(self, monkeypatch):
        agent = SimpleNamespace()
        lock = threading.RLock()
        fence = CompressionCommitFence()
        registration = compression_facade._CommitFenceRegistration(fence)
        with lock:
            vars(agent)["_compression_commit_fence_stack"] = [registration]
            agent._active_compression_commit_fence = fence

        retry = self._drive(monkeypatch, agent, registration, fence, lock)()

        assert isinstance(retry, CompressionCommitFence)
        assert registration.fence is retry
        assert vars(agent)["_active_compression_commit_fence"] is retry

    def test_retry_fence_does_not_clobber_newer_attempt(self, monkeypatch):
        agent = SimpleNamespace()
        lock = threading.RLock()
        fence = CompressionCommitFence()
        registration = compression_facade._CommitFenceRegistration(fence)
        other_fence = CompressionCommitFence()
        other = compression_facade._CommitFenceRegistration(other_fence)
        with lock:
            vars(agent)["_compression_commit_fence_stack"] = [registration, other]
            agent._active_compression_commit_fence = other_fence

        retry = self._drive(monkeypatch, agent, registration, fence, lock)()

        # The retry still gets its own fresh fence for begin_commit/cancel serialization,
        # but the published slot stays with the newer live attempt.
        assert isinstance(retry, CompressionCommitFence)
        assert registration.fence is retry
        assert vars(agent)["_active_compression_commit_fence"] is other_fence
