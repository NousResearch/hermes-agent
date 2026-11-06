"""The live turn's follow-up probe for background-review admission (``run_turn_runner`` sibling).

``_wire_turn_agent_callbacks`` installs one :class:`FollowupPendingProbe` per turn as
``agent.followup_pending_callback``: a non-consuming, body-free answer to "has the gateway
accepted a follow-up that becomes this session's next live turn?". It reads three things: the
adapter's process-local queue state (pending slot, busy-text debounce buffer, held busy burst,
the runner's overflow FIFO), the durable session FIFO (``SessionAuthority.queued_followup_on_route``,
a SQLite read), and the turn's ``FollowupAdmissionState`` epoch, which every accepted follow-up
bumps under the state lock (``_record_and_fence_followup_locked``).

The gateway event loop takes that same per-session lock for every follow-up it accepts
(``apply_followup_queue_mutation``), so the SQLite read must never run under it: a cold
per-thread connection, a WAL checkpoint or a busy store would stall the loop for the session.
The probe therefore samples the durable FIFO with no lock held, tagged with the epoch read just
before it, and takes the lock only for the epoch compare and the in-memory reads. The fence
guarantee survives: a follow-up accepted after the epoch tag moved the epoch, so the compare
reports it pending whatever the sample said; a follow-up accepted before the tag is in the
sample or the slot. ``_BackgroundReviewRun.begin_request`` holds the lock across its gate and
publication, so it calls :meth:`FollowupPendingProbe.sample` first (``admission_sample``), and
the gate under the lock consumes that sample instead of reading SQLite.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

# Re-samples the probe allows itself when the queue moved under a sample before giving the
# fail-safe answer (pending): the epoch only grows, so one re-sample is the practical maximum.
_RESAMPLE_BOUND = 3


class FollowupPendingProbe:
    """One turn's follow-up probe: sample the durable FIFO off the lock, compare under it."""

    def __init__(
        self,
        adapter: Any,
        session_key: str,
        state: Any,
        turn_epoch: int,
        overflow_queue: Callable[[str], Any],
    ) -> None:
        self._adapter = adapter
        self._session_key = session_key
        self._state = state
        self._turn_epoch = turn_epoch
        self._overflow_queue = overflow_queue
        # Per thread: the sample ``begin_request`` took before it entered the admission lock.
        self._samples = threading.local()

    # ── the lock-free half ──────────────────────────────────────────────────────────────

    def sample(self) -> None:
        """Read the durable FIFO with no lock held and keep the answer for this thread's next
        call. ``begin_request`` runs this outside its admission scope; a read that fails is kept
        too, so the call under the lock re-raises it (``admission_probe_failed``) instead of
        reading the store again there."""
        try:
            self._samples.value = self._sample_durable()
        except Exception as exc:  # health: allow BLE001 -- re-raised by the gated call
            self._samples.value = exc

    def _sample_durable(self) -> tuple[int, bool]:
        """(epoch tag, durable follow-up). The tag is read before the SQLite probe, so a follow-up
        accepted during the read moves the epoch past the tag."""
        with self._state.lock:
            epoch = self._state.epoch
        durable = getattr(self._adapter, "has_durable_followup", None)
        return epoch, bool(callable(durable) and durable(self._session_key))

    def _take_sample(self) -> Any:
        """This thread's kept sample, consumed. A sample left behind by a gate that never ran
        (the run was fenced first) is still safe to use later: its epoch tag decides."""
        sample = getattr(self._samples, "value", None)
        self._samples.value = None
        return sample

    # ── the gated half ──────────────────────────────────────────────────────────────────

    def __call__(self) -> bool:
        """Body-free: has the gateway accepted a follow-up that becomes the next live turn?
        Raises when a read fails (review admission reads that as ``admission_probe_failed``)."""
        sample = self._take_sample()
        if isinstance(sample, BaseException):
            raise sample
        for _attempt in range(_RESAMPLE_BOUND):
            epoch_seen, durable = sample if sample is not None else self._sample_durable()
            sample = None
            with self._state.lock:
                if self._state.epoch != self._turn_epoch:
                    # Accepted since this turn began: pending, whatever the sample says. Checked
                    # first so a caller already inside the lock (``begin_request`` after a stale
                    # sample) never reads the store under it.
                    return True
                if self._pending_in_memory():
                    return True
                if self._state.epoch == epoch_seen:
                    return durable
            # The epoch moved under the sample yet reads as the turn's own: unreachable while the
            # epoch only grows, but a stale sample is never the answer.
        return True

    def _pending_in_memory(self) -> bool:
        """Process-local queue state, read under the state lock: the adapter's slot/debounce/burst
        and the runner's overflow FIFO."""
        in_memory = getattr(self._adapter, "has_pending_message_in_memory", None)
        if not callable(in_memory):
            # An adapter without the split contract: its whole probe is the in-memory read.
            in_memory = self._adapter.has_pending_message
        return bool(in_memory(self._session_key) or self._overflow_queue(self._session_key))


def install_followup_probe(
    agent: Any, adapter: Any, session_key: Optional[str], overflow_queue: Callable[[str], Any]
) -> None:
    """Bind this turn's probe, its lock-free sample and its admission lock on ``agent``; every
    attribute is rebound per turn (a reused agent keeps nothing from the previous turn)."""
    probe = getattr(adapter, "has_pending_message", None)
    state_factory = getattr(adapter, "followup_admission_state", None)
    if not (session_key and callable(probe)):
        agent.followup_pending_callback = lambda: False
        agent.followup_pending_sample = None
        agent.followup_pending_lock = None
        return
    if not callable(state_factory):
        # A minimal adapter without the admission fence: the plain probe, no lock to keep I/O off.
        agent.followup_pending_callback = lambda: bool(
            probe(session_key) or overflow_queue(session_key)
        )
        agent.followup_pending_sample = None
        agent.followup_pending_lock = None
        return
    state = state_factory(session_key)
    with state.lock:
        turn_epoch = state.epoch
    pending = FollowupPendingProbe(adapter, session_key, state, turn_epoch, overflow_queue)
    agent.followup_pending_callback = pending
    agent.followup_pending_sample = pending.sample
    agent.followup_pending_lock = state.lock
