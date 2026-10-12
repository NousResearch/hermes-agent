"""A summary adoption and its cancellation check are one step for the attempt's own commit fence.

``_generate_summary`` used to read the host's cancellation, then write ``_previous_summary`` as a separate
step: a host cancellation could win between the two (the host then moves on, a stall fallback may reuse the
compressor) while the late worker still adopted its summary. The adoption now runs inside a window that
holds the attempt's fence lock, the lock every host-side cancellation takes, and re-checks before leaving
it, because revocation and the deadline flip without that lock.
"""
from __future__ import annotations

import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.auxiliary_client import AuxiliaryExplicitCancellation
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import (
    CompressionCommitFence,
    _claim_compressor_attempt,
    _run_summary_dispatch,
)
from agent.interrupt_control import _fence_cancel_before_commit

PRIOR = "Prior synthetic summary."
SNAPSHOT = "## Historical Task Snapshot\nNone. This session contains no user-authored turns.\n\n## Goal\n"
TURNS = [{"role": "assistant", "content": "Synthetic assistant note about a fictional public record."}]


def _answer(text):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.choices[0].finish_reason = "stop"
    return response


class _RecordingCompressor(ContextCompressor):
    """Records every ``_previous_summary`` write once armed; ``on_adopt`` runs right after the write."""

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        probe = self.__dict__.get("_probe")
        if name == "_previous_summary" and probe is not None and probe.armed.is_set():
            probe.events.append(("write", value))
            if value != PRIOR and probe.on_adopt is not None:
                on_adopt, probe.on_adopt = probe.on_adopt, None
                on_adopt()


class _InterposingFence(CompressionCommitFence):
    """Runs ``on_check`` once, on the worker thread's first cancellation read after the last answer."""

    def __init__(self, probe):
        super().__init__()
        self._probe = probe

    @property
    def is_cancelled(self):
        value = CompressionCommitFence.is_cancelled.fget(self)
        probe = self._probe
        if probe.armed.is_set() and probe.on_check is not None and threading.get_ident() == probe.worker:
            on_check, probe.on_check = probe.on_check, None
            on_check()
        return value


def _probe():
    return SimpleNamespace(armed=threading.Event(), events=[], on_check=None, on_adopt=None,
                           worker=threading.get_ident())


def _dispatch(compressor, fence):
    """The real fenced dispatch around the real multi-chunk summary, armed after the last answer."""
    agent = SimpleNamespace(context_compressor=compressor, session_id="adoption-window")
    real_once = compressor._generate_summary_once

    def answer_then_arm(*args, **kwargs):
        result = real_once(*args, **kwargs)
        compressor._probe.armed.set()
        return result

    with patch.object(compressor, "_generate_summary_once", side_effect=answer_then_arm), \
         patch("agent.context_compressor.call_llm", return_value=_answer(SNAPSHOT + "complete")):
        return _run_summary_dispatch(
            agent, [], lambda _messages: compressor._generate_summary(TURNS), {}, commit_fence=fence,
            attempt_generation=_claim_compressor_attempt(compressor), hard_cancel_event=None)


def _compressor(probe):
    compressor = _RecordingCompressor(model="test/model", provider="test", config_context_length=100_000,
                                      quiet_mode=True)
    compressor._previous_summary = PRIOR
    compressor._probe = probe
    return compressor


def test_a_host_cancellation_cannot_win_between_the_check_and_the_adoption():
    probe = _probe()
    compressor = _compressor(probe)
    fence = _InterposingFence(probe)
    host_done = threading.Event()

    def host_cancels():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            won = fence.try_cancel_before_commit()
            if won is not None:
                probe.events.append(("host_cancel_won", won))
                host_done.set()
                return
            time.sleep(0.005)

    host = threading.Thread(target=host_cancels, daemon=True)

    def interpose():
        # The worker has just read "not cancelled": the host cancels now, as fast as the fence lets it.
        host.start()
        host_done.wait(0.3)

    probe.on_check = interpose
    try:
        _dispatch(compressor, fence)
    except AuxiliaryExplicitCancellation:
        pass
    host.join(5)
    assert not host.is_alive()

    kinds = [kind for kind, _ in probe.events]
    assert ("host_cancel_won", True) in probe.events
    adopted = [i for i, (kind, value) in enumerate(probe.events) if kind == "write" and value != PRIOR]
    if adopted:
        # Adopted only if the host's cancellation came after the adoption (the commit then refuses).
        assert adopted[0] < kinds.index("host_cancel_won"), probe.events
        assert fence.begin_commit() is False
    else:
        assert compressor._previous_summary == PRIOR


def test_a_revocation_at_the_instant_of_the_adoption_is_undone():
    probe = _probe()
    compressor = _compressor(probe)
    fence = _InterposingFence(probe)
    # Revocation flips without the fence lock (it must never block on a hung commit).
    probe.on_adopt = fence.revoke_commit_admission

    with pytest.raises(AuxiliaryExplicitCancellation):
        _dispatch(compressor, fence)

    assert compressor._previous_summary == PRIOR


def test_an_uncancelled_attempt_adopts_its_summary():
    probe = _probe()
    compressor = _compressor(probe)
    fence = _InterposingFence(probe)

    _dispatch(compressor, fence)

    assert compressor._previous_summary == compressor._strip_summary_prefix(SNAPSHOT + "complete")
    assert fence.begin_commit() is True
    fence.finish_commit()


def _cancel_and_wait(fence):
    fence.cancel_before_commit()
    return True


def _try_cancel(fence):
    return fence.try_cancel_before_commit()


def _host_fence_cancel(fence):
    _fence_cancel_before_commit(fence, when_in_flight=False, failure_log="adoption-window test")
    return True


@pytest.mark.parametrize("durable_write", ["_clear_compression_failure_cooldown", "_reset_consecutive_overload_aborts"])
@pytest.mark.parametrize("host_cancel", [_cancel_and_wait, _try_cancel, _host_fence_cancel],
                         ids=["cancel_before_commit", "try_cancel_before_commit", "interrupt_control_site"])
def test_a_host_cancellation_never_waits_for_the_success_bookkeeping(durable_write, host_cancel):
    """The window covers the adoption only: the durable success writes after it never hold the fence lock."""
    probe = _probe()
    compressor = _compressor(probe)
    fence = CompressionCommitFence()
    reached, release = threading.Event(), threading.Event()
    real_write = getattr(compressor, durable_write)

    def slow_durable_write():
        reached.set()
        release.wait(10)
        real_write()

    worker_error = []

    def work():
        try:
            _dispatch(compressor, fence)
        except BaseException as exc:  # surfaced below
            worker_error.append(exc)

    host_result = []
    with patch.object(compressor, durable_write, side_effect=slow_durable_write):
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        host = threading.Thread(target=lambda: host_result.append(host_cancel(fence)), daemon=True)
        try:
            assert reached.wait(10), "the worker never reached its success bookkeeping"
            assert not fence.commit_in_flight
            host.start()
            host.join(2)
            host_blocked = host.is_alive()
        finally:
            release.set()
            worker.join(10)
            host.join(10)
    assert not worker.is_alive() and not host.is_alive()
    assert not host_blocked, f"the host waited on {durable_write} before any commit began"
    assert host_result == [True]  # a pending commit is cancelled; try_cancel never reports a busy fence
    assert worker_error == []
    assert fence.begin_commit() is False  # the late adoption cannot commit; compress_context rolls it back


def _cooldown_row(db, session_id):
    return db.get_compression_failure_cooldown_row(session_id)


def test_a_cancellation_during_the_success_bookkeeping_is_rolled_back_end_to_end(tmp_path):
    """Through compress_context: the host cancels once the success bookkeeping has written, after the adoption.
    The commit is refused and the attempt's rollback restores the adopted summary, the in-memory bookkeeping and
    the durable cooldown row that the success had cleared."""
    from agent.conversation_compression import compress_context
    from tests.agent.test_compression_attempt_lifecycle import _build_agent, _summary_response

    session_id = "ADOPTION_WINDOW_ROLLBACK"
    db, agent = _build_agent(tmp_path, session_id)
    compressor = agent.context_compressor
    compressor.record_timeout_failure("stall", failure_kind="stalled")  # durable row the success clears
    compressor._previous_summary = PRIOR
    compressor._consecutive_overload_aborts = 2
    bookkeeping = ("_previous_summary", "_summary_failure_cooldown_until", "_last_summary_error",
                   "_consecutive_timeout_failures", "_consecutive_overload_aborts")
    memory_before = {name: getattr(compressor, name) for name in bookkeeping}
    durable_before = _cooldown_row(db, session_id)
    assert durable_before and durable_before != _cooldown_row(db, "NO_SUCH_SESSION"), "the row must exist first"
    fence = CompressionCommitFence()
    real_reset = ContextCompressor._reset_consecutive_overload_aborts
    host = {}

    def reset_then_the_host_cancels(self):
        real_reset(self)  # the last success write has landed
        if self is not compressor or host or self._previous_summary in (PRIOR, None):
            return
        host["adopted"] = {"summary": self._previous_summary, "cooldown": self._summary_failure_cooldown_until,
                           "overload": self._consecutive_overload_aborts, "durable": _cooldown_row(db, session_id)}
        canceller = threading.Thread(target=lambda: host.setdefault("cancelled", fence.cancel_before_commit()),
                                     daemon=True)
        canceller.start()
        canceller.join(2)
        host["blocked"] = canceller.is_alive()
        canceller.join(10)

    live = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " * 400} for i in range(20)]
    before = copy.deepcopy(live)
    with patch.object(ContextCompressor, "_reset_consecutive_overload_aborts", reset_then_the_host_cancels), \
         patch("agent.context_compressor.call_llm", return_value=_summary_response("## Goal\nSynthetic rollback.")):
        out, _prompt = compress_context(agent, live, "sys", approx_tokens=500_000, bypass_cooldown=True,
                                        commit_fence=fence)

    adopted = host.pop("adopted", None)
    assert host == {"cancelled": True, "blocked": False}, "the host must cancel after the adoption, without waiting"
    # The success bookkeeping had really been written when the host cancelled: the rollback compensates writes.
    assert adopted is not None and adopted["summary"] not in (PRIOR, None)
    assert (adopted["cooldown"], adopted["overload"]) == (0.0, 0)
    assert adopted["durable"] != durable_before, "the success had cleared the durable cooldown row"
    assert out == before and live == before, "a refused commit leaves the transcript untouched"
    after = {name: getattr(compressor, name) for name in bookkeeping}
    expected = dict(memory_before)
    # Under the lease the attempt re-derives its in-memory deadline from the authoritative durable row (wall clock to
    # monotonic), so the restored deadline may differ from the pre-call float by that conversion jitter only.
    assert after.pop("_summary_failure_cooldown_until") == pytest.approx(
        expected.pop("_summary_failure_cooldown_until"), abs=1.0)
    assert after == expected
    assert _cooldown_row(db, session_id) == durable_before
    assert db.get_compression_lock_holder(session_id) is None


def test_a_revocation_deferred_by_the_window_releases_the_lease_on_exit():
    fence = CompressionCommitFence()
    released = []
    fence.register_cancelled_lock_release(lambda: released.append("lease"))

    with fence.adoption_window():
        # The host's revoke never blocks: the window holds the fence lock, so it defers the lease release.
        revoker = threading.Thread(target=fence.revoke_commit_admission)
        revoker.start()
        revoker.join(5)
        assert not revoker.is_alive()
        assert fence.try_cancel_before_commit() is None  # hosts see a busy fence and retry
        assert released == []

    assert released == ["lease"]
    assert fence.begin_commit() is False
