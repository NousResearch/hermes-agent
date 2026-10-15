"""Every order in which the drain and ``admit_message`` can complete one drained turn's review
ownership (``gateway/session_ingress_review.py``) releases its token exactly once and holds
nothing afterwards.

The drain's own sequence is fixed: ``hand_over_review_completion`` once the turn ran; after the
terminal commit, ``deliver_settled`` (``hand_review_completion_to_waiter`` when a native waiter
is registered on a row the drain owes the reply to, then ``complete_settled_review``); then
``_resolve_settled_waiter``. When the commit failed, ``release_unsettled_review`` and a pause
instead. ``admit_message`` interleaves at any of the drain's awaits: it registers before the
commit (``note_review_waiter``), its cancel (``forget_review_waiter``) can land anywhere, and a
redelivered message deduped onto the same row registers again after a cancel. Replayed here
against the module with a counting completion and the authority's own waiter bookkeeping (no
SQLite, no loop), so each 'completed exactly once' claim is checked against the module itself.
"""

import itertools
import types

import pytest

from agent import review_admission
from gateway import session_ingress_review as sir

_ID = "a"
# The drain's awaits a waiter event can land between: before the claim; before the handover;
# before the settlement commit; inside ``deliver_settled`` before its own completion step;
# before ``_resolve_settled_waiter``; after it, before the resolved task resumes.
_SLOTS = range(6)
_LAST_REGISTRATION_SLOT = (
    2  # a waiter registering after the commit reads the row terminal
)


class _Completion:
    """``_GatewayReviewAdmission.finish``'s shape: a bound method whose owner has a tag."""

    def __init__(self):
        self.calls = []
        self.__self__ = types.SimpleNamespace(owner_tag="owner")

    def __call__(self, *, delivery_succeeded, cause=None, outer=None):
        self.calls.append((delivery_succeeded, cause))


class _Authority:
    """The authority's process-local waiter bookkeeping on one admission, kept the way
    ``admit_message``, ``execute_admission``, ``_pause``/``_settle_cancelled`` and
    ``_resolve_settled_waiter`` keep it."""

    def __init__(self):
        self.waiters = {}
        self.native_waiters = set()
        self.live = False

    def register(self):
        """``admit_message`` before it parks on the row."""
        self.native_waiters.add(_ID)
        self.waiters.setdefault(_ID, object())
        sir.note_review_waiter(self, _ID)
        self.live = True

    def cancel(self):
        """The parked task's ``CancelledError``: the maps stay as they are."""
        sir.forget_review_waiter(self, _ID)
        self.live = False

    def claim(self):
        """``execute_admission``'s claim-time read of a registered native waiter."""
        native = _ID in self.native_waiters
        self.native_waiters.discard(_ID)
        return native

    def resolve(self):
        """``_resolve_settled_waiter``."""
        self.waiters.pop(_ID, None)

    def pause(self):
        """``_pause`` / ``_settle_cancelled``: the waiter is released with a reason, not a reply."""
        self.native_waiters.discard(_ID)
        self.waiters.pop(_ID, None)
        sir.drop_review_waiter_mark(self, _ID)


def _event(completion):
    event = types.SimpleNamespace()
    event._gateway_review_delivery_complete = completion
    return event


def _scripts(last_slot):
    """Waiter event scripts: one live waiter at a time (a registration after a cancel is the
    redelivered duplicate), registrations before the commit, later events at the same or a
    later slot."""
    shapes = (
        (),
        ("register",),
        ("register", "cancel"),
        ("register", "cancel", "register"),
        ("register", "cancel", "register", "cancel"),
    )
    for shape in shapes:
        for slots in itertools.product(range(last_slot + 1), repeat=len(shape)):
            script = tuple(zip(shape, slots, strict=True))
            if list(slots) != sorted(slots):
                continue
            if any(
                kind == "register" and slot > _LAST_REGISTRATION_SLOT
                for kind, slot in script
            ):
                continue
            yield script


def _name(script):
    return "-".join(f"{kind}{slot}" for kind, slot in script) or "no-waiter"


def _waiter_events(authority, script):
    def at(slot):
        for kind, when in script:
            if when == slot:
                getattr(authority, kind)()

    return at


def _walk_settled(script):
    done, authority = _Completion(), _Authority()
    waiter = _waiter_events(authority, script)
    waiter(0)
    native = authority.claim()
    waiter(1)
    sir.hand_over_review_completion(authority, _ID, _event(done), native=native)
    waiter(2)
    # ``deliver_settled``: the drain owes the reply of a turn it ran with no waiter, and hands
    # it to a waiter registered since (a cancel leaves the registration in place).
    handed = not native and _ID in authority.native_waiters
    if handed:
        authority.native_waiters.discard(_ID)
        sir.hand_review_completion_to_waiter(authority, _ID)
    waiter(3)
    sir.complete_settled_review(authority, _ID, delivery_succeeded=True)
    waiter(4)
    authority.resolve()
    waiter(5)
    adopted = False
    if authority.live:  # resumes with the reply, sends it, completes what it adopted
        event = types.SimpleNamespace()
        sir.adopt_review_completion(authority, _ID, event)
        completion = sir.take_review_completion(event)
        adopted = completion is not None
        if adopted:
            sir.complete_review_delivery(completion, delivery_succeeded=True)
    return done, authority, native, handed, adopted


@pytest.mark.parametrize("script", list(_scripts(5)), ids=_name)
def test_every_settled_order_completes_the_token_exactly_once(script):
    """One completion per token in every order: the drain's own delivery and a live waiter's
    adopted send complete it delivered, everything else releases it unconfirmed. A waiter
    alive at the end sent the reply (its own, or the one the drain resolved it with), so it
    is the one completing the turn, whatever an earlier waiter's cancel left behind."""
    done, authority, native, handed, adopted = _walk_settled(script)

    assert len(done.calls) == 1, done.calls
    assert authority.review_completions == {}
    drain_sent = not native and not handed
    if adopted or drain_sent:
        assert done.calls == [(True, None)]
    else:
        assert done.calls == [(False, review_admission.REASON_DELIVERY_UNCONFIRMED)]
    assert adopted == authority.live


@pytest.mark.parametrize("script", list(_scripts(3)), ids=_name)
def test_every_unsettled_order_releases_the_token_once_and_marks_nothing(script):
    """The commit failed: the drain releases what it held (``release_unsettled_review``) and
    pauses, releasing the waiter with the reason. A waiter that resumes with it, or is cancelled
    before resuming, completes nothing and leaves no mark behind."""
    done, authority = _Completion(), _Authority()
    waiter = _waiter_events(authority, script)
    waiter(0)
    native = authority.claim()
    waiter(1)
    sir.hand_over_review_completion(authority, _ID, _event(done), native=native)
    waiter(2)
    sir.release_unsettled_review(authority, _ID)
    authority.pause()
    waiter(3)
    if authority.live:
        sir.adopt_review_completion(authority, _ID, None)

    assert done.calls == [(False, review_admission.REASON_TURN_UNSETTLED)]
    assert authority.review_completions == {}


def test_a_drain_owned_completion_is_released_when_its_dead_late_waiter_is_handed_the_reply():
    """The settlement-window sequence: handover with ``native=False``, a waiter registering on
    the started row and cancelled, ``deliver_settled`` handing the reply to that registration."""
    done, authority = _Completion(), _Authority()
    authority.claim()
    sir.hand_over_review_completion(authority, _ID, _event(done), native=False)
    authority.register()
    authority.cancel()
    assert done.calls == []  # the drain may still deliver this turn itself
    sir.hand_review_completion_to_waiter(authority, _ID)
    sir.complete_settled_review(authority, _ID, delivery_succeeded=True)

    assert done.calls == [(False, review_admission.REASON_DELIVERY_UNCONFIRMED)]
    assert authority.review_completions == {}


def test_a_queued_row_whose_cancelled_waiter_is_released_keeps_no_mark():
    """The mark a cancel leaves on a pending row is for the handover that would otherwise
    leave the completion for a dead waiter. A row cancelled while queued never hands over,
    and one the FIFO paused on runs later with no waiter at all."""
    authority = _Authority()
    authority.register()
    authority.cancel()
    assert authority.review_completions

    authority.pause()

    assert authority.review_completions == {}


def test_a_cancel_after_the_drain_released_the_waiter_marks_nothing():
    """A row discarded while its settlement was recovering: the drain releases its waiter with
    ``unknown_execution`` and moves on, so no handover follows the cancel that can still reach
    the task before it resumes."""
    authority = _Authority()
    authority.register()
    authority.waiters.pop(_ID)

    authority.cancel()

    assert authority.review_completions == {}
