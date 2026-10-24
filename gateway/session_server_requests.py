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
import threading
import uuid

from tools.clarify_batch import settle_batch

REQUEST_TIMEOUT_S = 120  # main: ``_ask("sudo", ..., timeout=120)``
_POLL_S = 0.25


class _Pending:
    __slots__ = ("event", "events", "locked", "method", "qids", "result", "session_id", "value")

    def __init__(self, session_id, qids=None, events=None, method=None):
        self.event = threading.Event()
        self.session_id = session_id
        self.method = method
        # The live session's own event stream the request went out on: one per authority (profile) and
        # session mount, so a viewer of another profile's same-named session can never settle it.
        self.events = events
        self.value = ""
        self.result = None
        # A batch clarify: its question ids and the answers ``clarify.lock`` has locked so far.
        self.qids = tuple(qids) if qids else None
        self.locked = {}


_lock = threading.Lock()
_open: dict[str, _Pending] = {}


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
        _open.pop(frame["id"], None)
        locked = dict(pending.locked) if pending.qids is not None else None
    result = frame.get("result") if "error" not in frame else None
    if locked is not None:
        # One rule with main's sidecar registry: the frame's answers over the locks, or a cancel-all.
        result = settle_batch(locked, result)
    value = result.get("value") if isinstance(result, dict) else ""
    pending.value = value if isinstance(value, str) else ""
    pending.result = result if isinstance(result, dict) else None
    pending.event.set()
    return True


def ask_viewer(events, session_id, method, params, *, cancelled=lambda: False, timeout=REQUEST_TIMEOUT_S) -> str:
    """Send one one-string server→client request (``sudo``, ``secret``: ``ValueResult``) to the
    session's attached viewers and wait for its ``value``; "" on skip, interrupt, timeout or no viewer."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout)
    return pending.value if pending is not None else ""


def ask_viewer_result(events, session_id, method, params, *, cancelled=lambda: False, timeout=REQUEST_TIMEOUT_S,
                      qids=None):
    """The same round trip for a structured answer (``clarify``: ``ClarifyResult``): the viewer's
    ``result`` dict, or None on interrupt, timeout, an error response or no attached viewer.
    ``qids`` makes it a batch, which always ends with ``{answers: <locked so far>, outcome}`` once
    asked: ``lock_answer`` (the ``clarify.lock`` verb) answers it per question."""
    pending = _ask(events, session_id, method, params, cancelled=cancelled, timeout=timeout, qids=qids)
    return pending.result if pending is not None else None


def waiting(events, session_id, method) -> bool:
    """Whether a *method* request is open on this live session's stream (the card still waits)."""
    with _lock:
        return any(p.session_id == session_id and p.events is events and p.method == method for p in _open.values())


def withdraw(events, session_id, method) -> int:
    """Settle every open *method* request on this live session's stream with no answer, the way a
    viewer that answered would: the asker reads None and every renderer gets ``request.cancel
    reason: resolved``."""
    with _lock:
        settled = [_open.pop(rid) for rid, p in list(_open.items())
                   if p.session_id == session_id and p.events is events and p.method == method]
    for pending in settled:
        pending.event.set()
    return len(settled)


def lock_answer(request_id, question_id, answer, *, session_ok=lambda session_id, events: False):
    """Lock one answer of an open batch request (``clarify.lock``; editable until the last lock;
    ``None`` = skipped). Returns the question ids still unanswered; the lock that empties it settles
    the request with ``{answers, outcome: submitted}``. ``None`` when no open batch has that id or the
    caller may not answer its session (expired, or another profile's / session's request: see
    :func:`resolve`); ``ValueError`` for an unknown question id."""
    with _lock:
        pending = _open.get(request_id)
        if pending is None or pending.qids is None or not session_ok(pending.session_id, pending.events):
            return None
        if question_id not in pending.qids:
            raise ValueError(question_id)
        pending.locked[question_id] = answer
        remaining = [qid for qid in pending.qids if qid not in pending.locked]
        if not remaining:
            _open.pop(request_id, None)
            pending.result = settle_batch(pending.locked, ended="submitted")
    if not remaining:
        pending.event.set()
    return remaining


def _ask(events, session_id, method, params, *, cancelled, timeout, qids=None):
    if not events.fanout.transports():
        return None
    request_id = f"srq-{uuid.uuid4().hex[:12]}"
    pending = _Pending(session_id, qids, events, method)
    with _lock:
        _open[request_id] = pending
    events.fanout.write({"jsonrpc": "2.0", "id": request_id, "method": method,
                         "params": {"session_id": session_id, **params}})
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
        still_open = _open.pop(request_id, None) is pending
        if still_open and pending.qids is not None:
            # A batch keeps every answer locked before the wait ended (main's sidecar: "locked
            # answers survive a timeout"); sudo/secret have no question ids and stay unanswered.
            pending.result = settle_batch(pending.locked,
                                          ended="cancelled" if reason == "interrupted" else "timed_out")
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
