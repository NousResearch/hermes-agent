"""The classic CLI's ``/queue`` over the gateway's durable FIFO (refs #134889, N5).

On main a prompt queued while a turn ran sat in the CLI's ``_pending_input``: no
``session_admissions`` row, so an owner restart or a disconnect lost it. Over the session authority
``/queue <prompt>`` typed while a turn runs is an ordinary queued admission (``prompt.submit
{queued: true}``, the same path Ink and Desktop queue busy input on): the owner commits it at accept,
keeps it across a restart and runs it once, in FIFO order, after the current turn.

The CLI keeps no copy to resend. It remembers only which admissions it queued
(``SessionState.queued_inputs``) so the turn the owner runs for one is painted like a typed turn
(``gateway_tools_notify.queue_admitted_turn`` adopts it), and ``/queue list|edit|rm|clear`` read and
change the owner's queue (``prompt.receipt`` / ``prompt.replace_queued`` / ``prompt.cancel``).
"""
from __future__ import annotations

import logging
import uuid

from hermes_cli.gateway_client import GatewayClientError

logger = logging.getLogger(__name__)


def _state(cli, store):
    return store.connection.state(cli.session_id)


class QueueOutcomeUnknown(GatewayClientError):
    """The submit was sent but no answer came back (timeout, disconnect): the owner may hold it.

    No admission id came back, so nothing enters ``queued_inputs``. If the owner did queue it, the row
    shows in ``/queue list`` (from the owner's published list) and runs once, but this CLI never adopts
    its turn: the reply lands in the session's history, not painted live here. The user is told to
    check ``/queue list`` instead of re-sending. This is rare (two lost answers in a row).
    """


def submit_queued(cli, store, text: str) -> str:
    """Admit ``text`` behind the running turn in the owner's FIFO; returns its admission id.

    A refusal (``GatewayRPCError``) or a request never sent raises as is: nothing was queued. A lost
    answer is asked once more under the SAME ``submission_id``, which the owner answers with the
    admission it already holds (``admit_session_input`` keys it by principal, session and request id),
    so the prompt is queued exactly once. Still no answer: :class:`QueueOutcomeUnknown`.
    """
    from hermes_cli.gateway_client import GatewayRPCError
    from hermes_cli.gateway_connection import NOT_SENT
    submission = uuid.uuid4().hex
    for attempt in (1, 2):
        try:
            receipt = store.call("prompt.submit", session_id=cli.session_id, submission_id=submission,
                                 text=text, queued=True)
            break
        except GatewayRPCError:
            raise
        except GatewayClientError as exc:
            if str(exc) == NOT_SENT and attempt == 1:
                raise  # never reached a socket: nothing is in doubt
            if attempt == 2:
                raise QueueOutcomeUnknown(str(exc)) from exc
            logger.debug("prompt.submit answer lost (%s); asking again under the same identity", exc)
    admission = (receipt or {}).get("admission_id")
    if not admission:
        raise GatewayClientError("the gateway did not admit the queued prompt")
    state = _state(cli, store)
    state.queued_inputs[admission] = text
    state.queued_at[admission] = state.pending_seq
    return admission


def queued_rows(cli, store) -> list[tuple[str, str]]:
    """``[(admission_id, text)]`` still queued in the owner's FIFO, in run order.

    This viewer's own admissions are read back with ``prompt.receipt`` (one it finds started or settled
    is left for the idle tick to paint); other viewers' queued rows come from the last pending list the owner published.
    """
    state = _state(cli, store)
    rows: dict[str, tuple[int, str]] = {}
    for row in state.pending or ():
        if isinstance(row, dict) and row.get("status") == "queued" and row.get("admission_id"):
            rows[row["admission_id"]] = (int(row.get("sequence") or 0), str(row.get("text") or ""))
    for admission, text in list(state.queued_inputs.items()):
        try:
            receipt = store.call("prompt.receipt", session_id=cli.session_id, admission_id=admission)
        except GatewayClientError as exc:
            logger.debug("prompt.receipt %s failed: %s", admission, exc)
            rows.setdefault(admission, (1 << 62, text))  # unread: still shown, never dropped
            continue
        if receipt.get("status") == "queued":
            rows[admission] = (int(receipt.get("sequence") or 0), text)
        else:
            # Started or settled: no longer queued, but its turn may not be painted yet. The idle tick
            # paints it (``gateway_tools_notify.recover_queued``); listing never erases that.
            rows.pop(admission, None)
            state.missed.add(admission)
    return [(admission, text) for admission, (_, text) in sorted(rows.items(), key=lambda kv: kv[1][0])]


def published_queue(cli, store) -> list[str]:
    """The texts the owner last published as queued, plus this viewer's admissions submitted after that
    list (no RPC: the dock repaints on every tick).

    The owner's published list is authoritative: an admission it no longer lists (cancelled, replaced or
    steered by another client, or started) is not shown as queued, even while ``queued_inputs`` still
    owes its turn a paint."""
    state = _state(cli, store)
    for gone in [admission for admission in state.queued_at if admission not in state.queued_inputs]:
        state.queued_at.pop(gone, None)
    rows = [row for row in state.pending or () if isinstance(row, dict)]
    known = {row.get("admission_id") for row in rows}
    fresh = [text for admission, text in state.queued_inputs.items()
             if admission not in known and state.queued_at.get(admission, -1) >= state.pending_seq]
    return [str(row.get("text") or "") for row in rows if row.get("status") == "queued"] + fresh


# How long the input loop holds a local input behind this viewer's own ``/queue`` admission at most.
OWN_QUEUE_HOLD_S = 2.0


def _own_head(state):
    """This viewer's ``/queue`` admission at the head of the owner's queue, not yet adopted; None when
    the head is someone else's, nothing is queued, or the FIFO is paused on an ``unknown`` row."""
    rows = [row for row in state.pending or () if isinstance(row, dict)]
    if any(row.get("status") == "unknown" for row in rows):
        return None  # paused until the owner resolves that row: never hold typing behind it
    queued = sorted((row for row in rows if row.get("status") == "queued" and row.get("admission_id")),
                    key=lambda row: int(row.get("sequence") or 0))
    if queued:
        head = queued[0]["admission_id"]
    else:  # submitted after the last published list (``published_queue``'s fresh rows)
        listed = {row.get("admission_id") for row in rows}
        head = next((a for a in state.queued_inputs
                     if a not in listed and state.queued_at.get(a, -1) >= state.pending_seq), None)
    return head if head in state.queued_inputs and head not in state.adopted else None


class OwnQueueHold:
    """A local input (a slash command, a typed turn) taken while this viewer's own ``/queue`` admission
    is the owner's next turn. On main both sat in ``_pending_input`` in entry order; now the admission
    is in the owner's FIFO and its turn is adopted only when the owner's ``message.start`` arrives, so
    the input loop holds the local item until that admission was taken (or left the queue).

    Bounded: an ``unknown`` row (a paused FIFO) never starts a hold; an owner that publishes an idle
    list still queuing it (beyond the one a just-settled turn publishes before its claim) releases it;
    and ``OWN_QUEUE_HOLD_S`` caps the wait. The admission itself is never touched.
    """

    def __init__(self, item, state, admission):
        import time
        self.item, self.state, self.admission = item, state, admission
        self.seq = state.pending_seq
        # A turn the owner reports running settles with one idle list (still queuing this admission)
        # before its drain claims it: that list is expected, a further one means the FIFO is paused.
        self.allowed_idle_lists = 1 if state.info.get("running") else 0
        self.idle_lists = 0
        self.deadline = time.monotonic() + OWN_QUEUE_HOLD_S
        self.taken = False

    @classmethod
    def begin(cls, cli, item):
        from cli import _SeededQueryMessage
        from hermes_cli.gateway_tools_notify import gateway_store
        from tools.process_registry_notifications import TimelineNotification
        if getattr(cli, "_agent_running", False) or isinstance(item, (_SeededQueryMessage, TimelineNotification)):
            return None  # a turn the owner already runs (or a seeded prompt) is never held
        store = gateway_store(cli) if getattr(cli, "session_id", None) else None
        if store is None:
            return None
        state = _state(cli, store)
        admission = _own_head(state)
        return cls(item, state, admission) if admission else None

    def take_gateway_turn(self, pending):
        """Pop a turn the owner is running now (an adopted admission, a notification header) from the
        head of ``_pending_input``: those run before the held item, typed items stay behind it."""
        from tools.process_registry_notifications import TimelineNotification
        with pending.mutex:
            if not pending.queue:
                return None
            head = pending.queue[0]
            if not (getattr(head, "admission_id", None) or isinstance(head, TimelineNotification)):
                return None
            pending.queue.popleft()
            pending.unfinished_tasks = max(0, pending.unfinished_tasks - 1)
        if getattr(head, "admission_id", None) == self.admission:
            self.taken = True
        return head

    def done(self) -> bool:
        import time
        if self.taken or time.monotonic() >= self.deadline:
            return True
        state = self.state
        if _own_head(state) != self.admission:
            # Adopted: its item is at the head of ``_pending_input``, taken on the next step. Otherwise
            # it started unseen (the idle tick recovers it), was cancelled, or is no longer next.
            return self.admission not in state.adopted
        if state.pending_seq != self.seq:
            self.seq = state.pending_seq
            if not state.info.get("running"):
                self.idle_lists += 1
        return self.idle_lists > self.allowed_idle_lists  # idle and still not claiming it: stop waiting


STARTED_REFUSALS = frozenset({"stale_generation", "admission_started"})


def cancel(cli, store, admission: str) -> str:
    """Retire one queued admission: ``"cancelled"`` when it never ran, ``"stopped"`` when the drain had
    claimed it and the owner interrupted that run instead (``session_controls.cancel``), ``"started"``
    when it already ran. A turn that ran is still owed a paint, so only a cancelled one leaves
    ``queued_inputs``."""
    receipt = store.call("prompt.cancel", session_id=cli.session_id, admission_id=admission) or {}
    if receipt.get("status") == "terminal" and receipt.get("outcome") == "cancelled":
        state = _state(cli, store)
        state.queued_inputs.pop(admission, None)
        state.missed.discard(admission)
        return "cancelled"
    _state(cli, store).missed.add(admission)  # the idle tick paints whatever ran
    if receipt.get("status") == "started" or receipt.get("outcome") == "interrupted":
        return "stopped"
    return "started"  # it already ran to its end: nothing was stopped


def replace(cli, store, admission: str, text: str) -> str | None:
    """``prompt.replace_queued``: the edited prompt is a new admission at the back of the FIFO."""
    receipt = store.call("prompt.replace_queued", session_id=cli.session_id, admission_id=admission,
                         submission_id=uuid.uuid4().hex, text=text)
    state = _state(cli, store)
    state.queued_inputs.pop(admission, None)
    new = (receipt or {}).get("admission_id")
    if new:
        state.queued_inputs[new] = text
        state.queued_at[new] = state.pending_seq
    return new
