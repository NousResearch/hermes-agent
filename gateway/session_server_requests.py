"""The session owner's server→client requests to a local viewer, and who may settle them.

The owner sends a request frame (``tui_gateway/contracts/server_requests.py``: ``sudo``, ``secret``,
``clarify``, the Desktop UI requests) to the viewers attached to a session's event stream and blocks the
agent thread until one answers, the turn is interrupted, or it times out: the canonical side of main's
sidecar ``tui_gateway/server_requests.py``. A viewer answers with a JSON-RPC response frame
(:func:`resolve`) or, for a batch clarify, per question with ``clarify.lock`` (:func:`lock_answer`); both go
through :func:`may_settle`, and a batch's terminal result comes from ``tools.clarify_batch.settle_batch``,
the rule the sidecar uses too. Values are never published, journaled or logged: they travel only in the
viewer's response frame.
"""
from __future__ import annotations

import contextlib
from collections import OrderedDict
import json
import threading
import uuid

from tools.clarify_batch import settle_batch

REQUEST_TIMEOUT_S = 120  # main: ``_ask("sudo", ..., timeout=120)``
_POLL_S = 0.25


class RequestIdentityMismatch(Exception):
    """A fenced caller named a different authority epoch or execution generation."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _check_fences(pending, authority_epoch, execution_generation):
    if authority_epoch is not None and pending.authority_epoch != authority_epoch:
        raise RequestIdentityMismatch('stale_epoch')
    if execution_generation is not None and pending.execution_generation != execution_generation:
        raise RequestIdentityMismatch('stale_generation')


class _Pending:
    __slots__ = ("authority_epoch", "event", "events", "execution_generation", "locked", "method", "params",
                 "qids", "replay", "result", "session_id", "value")

    def __init__(self, session_id, qids=None, events=None, method=None, params=None, replay=True,
                 authority_epoch=None, execution_generation=None):
        self.event = threading.Event()
        self.session_id = session_id
        self.method = method
        # The frame's params as sent, so a viewer that reconnects gets the request again; ``replay`` False
        # for an automatic window request (a preview click), which a second delivery would run twice.
        self.params = {"session_id": session_id, **dict(params or {})}
        self.replay = replay
        # The live session's own event stream the request went out on: one per authority (profile) and
        # session mount, so a viewer of another profile's same-named session can never settle it.
        self.events = events
        self.authority_epoch = authority_epoch
        self.execution_generation = execution_generation
        self.value = ""
        self.result = None
        # A batch clarify: its question ids and the answers ``clarify.lock`` has locked so far.
        self.qids = tuple(qids) if qids else None
        self.locked = {}

    def snapshot(self, request_id):
        """Reconnect frame with only locks already accepted by this authority."""
        params = dict(self.params)
        if self.qids is not None:
            params["answers"] = _answers_snapshot(self)
        return {"id": request_id, "method": self.method, "params": params}


_lock = threading.Lock()
_open: dict[str, _Pending] = {}
_settled: OrderedDict[str, _Pending] = OrderedDict()
_SETTLED_LIMIT = 128


def _answers_snapshot(pending):
    """JSON values for replay/reconcile; native response frames encode multi-select as JSON text."""
    answers = dict(pending.locked)
    questions = pending.params.get("questions")
    if not isinstance(questions, list):
        return answers
    for question in questions:
        if not isinstance(question, dict) or not question.get("multi_select"):
            continue
        qid = question.get("qid")
        value = answers.get(qid)
        if not isinstance(value, str):
            continue
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            continue
        if isinstance(decoded, list):
            answers[qid] = decoded
    return answers


def _settle(request_id, pending, result):
    """Commit one outcome and retain only batch receipts, bounded and under the registry lock."""
    _open.pop(request_id, None)
    pending.result = result
    if pending.qids is not None:
        if isinstance(result, dict) and isinstance(result.get("answers"), dict):
            # A competing response frame may settle the form instead of the final clarify.lock.
            # Retained terminal replay must describe that accepted result, not stale partial locks.
            pending.locked = dict(result["answers"])
        _settled[request_id] = pending
        _settled.move_to_end(request_id)
        while len(_settled) > _SETTLED_LIMIT:
            _settled.popitem(last=False)


def owned_by(authority, session_id, events) -> bool:
    """True when *authority*'s live *session_id* is the very stream a request was sent on."""
    live = authority.sessions.get(session_id)
    return live is not None and events is not None and live.event_stream is events


def may_settle(authority, actor, session_id, events) -> bool:
    """The one rule for who may settle an owner request (a response frame or ``clarify.lock``): the
    request went out on *authority*'s own live stream for that session (session ids repeat across
    profiles and ``_open`` is process-wide), *actor* holds ``session:respond`` there, and *actor* is
    attached to that session, the same set of connections the request frame was written to."""
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import RuntimeStoreError
    if not owned_by(authority, session_id, events):
        return False
    try:
        authority.authorize(actor, SessionRef(authority.profile_id, session_id), "session:respond")
    except RuntimeStoreError:
        return False
    live = authority.sessions.get(session_id)
    return live is not None and actor in live.subscribers.values()


def is_response_frame(frame) -> bool:
    """A viewer's answer to one of the owner's requests: a string ``srq-`` id, a result/error, no method."""
    return (isinstance(frame, dict) and "method" not in frame and isinstance(frame.get("id"), str)
            and frame["id"].startswith("srq-") and ("result" in frame or "error" in frame))


def resolve(frame, *, session_ok=lambda session_id, events: True) -> bool:
    """Settle the open request *frame* answers; False when nothing (here) waits for that id.
    ``session_ok(session_id, events)`` lets the caller refuse an answer from a connection not entitled
    to that session: the caller's own authority must own ``events``, the stream the request was sent on
    (session ids are only unique per profile, and ``_open`` is process-wide)."""
    with _lock:
        pending = _open.get(frame["id"])
        if pending is None or not session_ok(pending.session_id, pending.events):
            return False
        result = frame.get("result") if "error" not in frame else None
        if pending.qids is not None:
            # One rule with main's sidecar registry: the frame's answers over the locks, or a cancel-all.
            result = settle_batch(dict(pending.locked), result)
        value = result.get("value") if isinstance(result, dict) else ""
        pending.value = value if isinstance(value, str) else ""
        _settle(frame["id"], pending, result if isinstance(result, dict) else None)
    pending.event.set()
    return True


def ask_viewer(events, session_id, method, params, *, cancelled=lambda: False, timeout=REQUEST_TIMEOUT_S,
               replay=True) -> str:
    """Send one one-string server→client request (``sudo``, ``secret``: ``ValueResult``) to the
    session's attached viewers and wait for its ``value``; "" on skip, interrupt, timeout or no viewer.
    ``replay=False``: a reconnecting viewer is not handed it again (``open_requests``)."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout, replay=replay)
    return pending.value if pending is not None else ""


def ask_viewer_result(events, session_id, method, params, *, cancelled=lambda: False, timeout=REQUEST_TIMEOUT_S,
                      qids=None, authority_epoch=None, execution_generation=None):
    """The same round trip for a structured answer (``clarify``: ``ClarifyResult``): the viewer's
    ``result`` dict, or None on interrupt, timeout, an error response or no attached viewer.
    ``qids`` makes it a batch, which always ends with ``{answers: <locked so far>, outcome}`` once
    asked: ``lock_answer`` (the ``clarify.lock`` verb) answers it per question."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout, qids=qids,
                   authority_epoch=authority_epoch, execution_generation=execution_generation)
    return pending.result if pending is not None else None


def waiting(events, session_id, method) -> bool:
    """Whether a *method* request is open on this live session's stream (the card still waits)."""
    with _lock:
        return any(p.session_id == session_id and p.events is events and p.method == method for p in _open.values())


def open_requests(events, session_id, *, batch_only=False) -> list[dict]:
    """Live requests for this exact stream; admission-only snapshots expose only batch forms."""
    with _lock:
        return [pending.snapshot(request_id) for request_id, pending in _open.items()
                if pending.session_id == session_id and pending.events is events and pending.replay
                and (pending.qids is not None or not batch_only)]


def request_receipts(events, session_id) -> list[dict]:
    """Latest settled batch forms for this stream, so a fresh viewer can reconcile their receipts."""
    with _lock:
        retained = [(request_id, pending) for request_id, pending in _settled.items()
                    if pending.session_id == session_id and pending.events is events]
        return [pending.snapshot(request_id) for request_id, pending in retained[-16:]]


def withdraw(events, session_id, method) -> int:
    """Settle every open *method* request on this stream: no answer for scalar requests, a
    ``cancelled`` result retaining locks for batches, and ``request.cancel reason: resolved``."""
    with _lock:
        settled = []
        for request_id, pending in list(_open.items()):
            if pending.session_id == session_id and pending.events is events and pending.method == method:
                result = settle_batch(dict(pending.locked), ended="cancelled") if pending.qids is not None else None
                _settle(request_id, pending, result)
                settled.append(pending)
    for pending in settled:
        pending.event.set()
    return len(settled)


def lock_answer(request_id, question_id, answer, *, session_ok=lambda session_id, events: False,
                authority_epoch=None, execution_generation=None):
    """Lock one answer of an open batch request (``clarify.lock``; editable until the last lock;
    ``None`` = skipped). Returns the question ids still unanswered; the lock that empties it settles
    the request with ``{answers, outcome: submitted}``. ``None`` when no open batch has that id or the
    caller may not answer its session (expired, or another profile's / session's request: see
    :func:`resolve`); ``ValueError`` for an unknown question id."""
    with _lock:
        pending = _open.get(request_id)
        if pending is None or pending.qids is None or not session_ok(pending.session_id, pending.events):
            return None
        _check_fences(pending, authority_epoch, execution_generation)
        if question_id not in pending.qids:
            raise ValueError(question_id)
        pending.locked[question_id] = answer
        remaining = [qid for qid in pending.qids if qid not in pending.locked]
        if not remaining:
            _settle(request_id, pending, settle_batch(dict(pending.locked), ended="submitted"))
    if not remaining:
        pending.event.set()
    return remaining


def reconcile(request_id, *, session_ok=lambda session_id, events: False,
              authority_epoch=None, execution_generation=None):
    """Read one open batch or its retained decision; never lock, settle or replay anything."""
    with _lock:
        pending = _open.get(request_id)
        if pending is not None:
            if pending.qids is None or not session_ok(pending.session_id, pending.events):
                return {"status": "expired", "request_id": request_id}
            _check_fences(pending, authority_epoch, execution_generation)
            return {"status": "open", "request_id": request_id,
                    "authority_epoch": pending.authority_epoch,
                    "execution_generation": pending.execution_generation,
                    "answers": _answers_snapshot(pending),
                    "remaining": [qid for qid in pending.qids if qid not in pending.locked]}
        pending = _settled.get(request_id)
        if pending is None or not session_ok(pending.session_id, pending.events):
            return {"status": "expired", "request_id": request_id}
        _check_fences(pending, authority_epoch, execution_generation)
        result = pending.result or {}
        outcome = result.get("outcome")
        return {"status": outcome if outcome in {"submitted", "cancelled", "timed_out"} else "cancelled",
                "request_id": request_id,
                "authority_epoch": pending.authority_epoch,
                "execution_generation": pending.execution_generation,
                "answers": _answers_snapshot(pending), "remaining": None}


def _ask(events, session_id, method, params, *, cancelled, timeout, qids=None, replay=True,
         authority_epoch=None, execution_generation=None):
    if not events.fanout.transports():
        return None
    request_id = f"srq-{uuid.uuid4().hex[:12]}"
    pending = _Pending(session_id, qids, events, method, params, replay=replay,
                       authority_epoch=authority_epoch, execution_generation=execution_generation)
    with _lock:
        _open[request_id] = pending
    events.fanout.write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": pending.params})
    waited = 0.0
    reason = "timeout"
    while not pending.event.wait(_POLL_S):
        waited += _POLL_S
        if cancelled():
            reason = "interrupted"
            break
        if timeout is not None and waited >= timeout:
            break
    with _lock:
        still_open = _open.get(request_id) is pending
        if still_open:
            # A batch keeps every answer locked before the wait ended (main's sidecar: "locked
            # answers survive a timeout"); sudo/secret have no question ids and stay unanswered.
            result = (settle_batch(dict(pending.locked),
                                   ended="cancelled" if reason == "interrupted" else "timed_out")
                      if pending.qids is not None else None)
            _settle(request_id, pending, result)
    if still_open:
        # The same teardown main's renderers key on (``request.cancel``): the card closes and a
        # timed-out password prompt says so.
        with contextlib.suppress(Exception):
            events.publish(session_id, {"id": request_id, "method": method, "reason": reason},
                           event_type="request.cancel")
        return pending if pending.qids is not None else None
    # Settled by a viewer (a response frame or the last ``clarify.lock``): every OTHER viewer still
    # shows the card, and a classic CLI is parked inside its modal callback. Withdraw it everywhere
    # with ``reason: resolved`` (``RequestCancelReason``) so each renderer closes the matching card and
    # the CLI's connection thread releases its modal (``hermes_cli.gateway_agent``). No answer rides it.
    with contextlib.suppress(Exception):
        events.publish(session_id, {"id": request_id, "method": method, "reason": "resolved"},
                       event_type="request.cancel")
    return pending
