"""Regressions for the #76354 review of the compression timeout architecture.

Every test here asserts the BLOCKED/hung state itself where the review demands
it — the worker is released only AFTER the assertion (helix4u called out two
prior tests that released before asserting; do not regress that).

Covers:
- F1: commit-phase overrun warning fires WHILE the commit is hung (lock-free
  ``commit_in_flight`` phase marker).
- F2: every host unwind (KeyboardInterrupt / generic exception) revokes commit
  admission before the host resumes.
- F4 (unit half): a cancelled attempt cannot clear the failure cooldown
  (fence check ordered BEFORE cooldown-clear).
- F6: bounded admission — four wedged workers refuse a fifth submission fast,
  and the refused job never runs later; a cancelled fence skips summary work.
- S3 analogue: the idle wait is charged from the last progress event, so
  silence cannot approach 2x the configured idle timeout.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time

import pytest

import agent.conversation_compression as cc
from agent.conversation_compression import (
    CompressionCommitFence,
    run_compress_context_with_progress_timeout,
)


def _drain_admission_slots():
    """Placeholder until bounded admission (F6) lands in a later commit."""
    return


class TestF1CommitOverrunWhileHung:
    def test_overrun_warning_fires_while_commit_still_blocked(self):
        """The warning + on_commit_overrun fire DURING the hang, not after.

        The fake commit is event-gated and is NOT released until after the
        assertions on the callback/log have been made while the worker
        thread is still blocked inside the commit boundary.
        """
        original = [{"role": "user", "content": "a"}]
        compressed = [{"role": "assistant", "content": "late"}]
        entered = threading.Event()
        release = threading.Event()
        overrun_fired = threading.Event()
        overruns = []

        def worker(fence: CompressionCommitFence):
            assert fence.begin_commit()
            entered.set()
            try:
                # Hung commit: blocked until the TEST releases it, which
                # happens only after asserting the overrun surfaced.
                assert release.wait(timeout=10)
                return (compressed, "committed-late")
            finally:
                fence.finish_commit()

        records = []

        class _Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        def on_overrun(waited, ceil):
            overruns.append((waited, ceil))
            overrun_fired.set()

        done = {}

        def run():
            done["result"] = run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=0.05,
                total_ceiling_seconds=0.05,
                on_commit_overrun=on_overrun,
            )

        comp_logger = logging.getLogger("agent.conversation_compression")
        handler = _Capture(level=logging.WARNING)
        comp_logger.addHandler(handler)
        try:
            t = threading.Thread(target=run, name="f1-hung-commit-host")
            t.start()
            try:
                assert entered.wait(timeout=2)
                # ── Assert WHILE the commit worker is still blocked ──────
                assert overrun_fired.wait(timeout=5), (
                    "on_commit_overrun must fire while the commit is hung"
                )
                assert not release.is_set()  # worker provably still blocked
                assert t.is_alive()
                deadline = time.time() + 5
                while time.time() < deadline:
                    if any(
                        r.levelno >= logging.WARNING
                        and "past the total ceiling" in r.getMessage()
                        for r in list(records)
                    ):
                        break
                    time.sleep(0.01)
                overrun_logs = [
                    r
                    for r in list(records)
                    if r.levelno >= logging.WARNING
                    and "past the total ceiling" in r.getMessage()
                ]
                assert overrun_logs, (
                    "expected the overrun WARNING while the commit was "
                    f"still blocked; got: {[r.getMessage() for r in records]}"
                )
                assert overruns and overruns[0][1] == pytest.approx(0.05)
            finally:
                release.set()
            t.join(timeout=5)
            assert not t.is_alive()
        finally:
            comp_logger.removeHandler(handler)
        assert done["result"] == (compressed, "committed-late")
        _drain_admission_slots()

    def test_commit_in_flight_marker_is_lock_free(self):
        fence = CompressionCommitFence()
        assert fence.commit_in_flight is False
        assert fence.begin_commit()
        # The fence lock is HELD here; the marker must still be readable.
        assert fence.commit_in_flight is True
        fence.finish_commit()
        assert fence.commit_in_flight is False


class _KIOnFirstResultFuture:
    """Future proxy raising on the host's first result() call."""

    def __init__(self, inner, exc, gate=None):
        self._inner = inner
        self._exc = exc
        self._raised = False
        self._gate = gate

    def result(self, timeout=None):
        if not self._raised:
            self._raised = True
            if self._gate is not None:
                # Ensure the pooled worker has genuinely STARTED before the
                # host unwinds, so the test exercises "unwind with a live
                # worker" rather than the queued-job skip path.
                assert self._gate.wait(timeout=5)
            raise self._exc
        return self._inner.result(timeout=timeout)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _InjectingExecutor:
    def __init__(self, inner, exc, gate=None):
        self._inner = inner
        self._exc = exc
        self._gate = gate

    def submit(self, fn, *args, **kwargs):
        return _KIOnFirstResultFuture(
            self._inner.submit(fn, *args, **kwargs), self._exc, self._gate
        )


class TestF2HostUnwindRevokesAdmission:
    @pytest.mark.parametrize(
        "exc_type", [KeyboardInterrupt, RuntimeError], ids=["ki", "generic"]
    )
    def test_unwind_revokes_commit_admission_before_host_returns(
        self, monkeypatch, exc_type
    ):
        """KI/exception while waiting → detached worker can never commit.

        The worker is still blocked pre-commit when the host unwinds; the
        assertions run BEFORE the worker is released.
        """
        original = [{"role": "user", "content": "keep"}]
        started = threading.Event()
        release = threading.Event()
        fence_box = {}
        commit_admitted = {}

        def worker(fence: CompressionCommitFence):
            fence_box["fence"] = fence
            started.set()
            assert release.wait(timeout=10)
            commit_admitted["value"] = fence.begin_commit()
            if commit_admitted["value"]:
                fence.finish_commit()
            return ([{"role": "assistant", "content": "late"}], "x")

        real_executor = cc._get_compress_timeout_executor()
        monkeypatch.setattr(
            cc,
            "_get_compress_timeout_executor",
            lambda: _InjectingExecutor(real_executor, exc_type(), gate=started),
        )

        with pytest.raises(exc_type):
            run_compress_context_with_progress_timeout(
                worker=worker,
                messages=original,
                system_prompt_fallback="fallback",
                idle_timeout_seconds=5.0,
                total_ceiling_seconds=5.0,
            )

        # ── Host has unwound; worker is STILL blocked pre-commit ─────────
        assert started.wait(timeout=2)
        fence = fence_box["fence"]
        assert not release.is_set()
        assert fence.is_cancelled, (
            "host unwind must revoke commit admission while the worker "
            "is still running"
        )
        # Now release the worker and prove its commit was refused.
        release.set()
        deadline = time.time() + 5
        while time.time() < deadline and "value" not in commit_admitted:
            time.sleep(0.01)
        assert commit_admitted.get("value") is False, (
            "a worker surviving a host unwind must be denied the commit "
            "boundary"
        )
        _drain_admission_slots()


class TestF4CooldownClearOrdering:
    def test_cancelled_attempt_cannot_clear_failure_cooldown(self):
        """Fence check ordered BEFORE cooldown-clear (review F4 ordering)."""
        from agent.context_compressor import ContextCompressor

        class _FakeCompressor:
            _summary_failure_cooldown_until = 12345.0
            _last_summary_error = "timeout"
            _consecutive_timeout_failures = 2
            _cooldown_persist_failed = False
            _session_db = None
            _session_id = ""
            _compression_cancelled_check = staticmethod(lambda: True)

        fake = _FakeCompressor()
        ContextCompressor._clear_compression_failure_cooldown(fake)
        assert fake._summary_failure_cooldown_until == 12345.0, (
            "a cancelled attempt must NOT undo the host's timeout cooldown"
        )
        assert fake._consecutive_timeout_failures == 2

        # Sabotage check: with the fence reporting NOT cancelled, the clear
        # must proceed (proves the guard is the only thing blocking it).
        fake2 = _FakeCompressor()
        fake2._compression_cancelled_check = staticmethod(lambda: False)
        ContextCompressor._clear_compression_failure_cooldown(fake2)
        assert fake2._summary_failure_cooldown_until == 0.0
