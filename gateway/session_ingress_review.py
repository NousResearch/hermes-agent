"""Who completes a drained turn's gateway review ownership (``gateway/session_ingress.py``).

``gateway.run_turn._handle_message_with_agent`` registers every gateway turn as live in
``agent.review_admission`` (``_GatewayReviewAdmission``) and parks the completion that releases
the token — and spawns the turn's automatic review once its reply is confirmed delivered — where
the turn's delivery owner reads it. A turn the session authority's drain executes has no adapter
task of its own, so ``execute_admission`` parks the completion on the event it executes and hands
it over here, the way it hands over the turn's crash marker (``_hand_over_turn_marker``):

* a native waiter (the adapter task that admitted the first message, ``dispatch_shared_busy``'s
  observer, ``/retry``'s inline dispatch) takes it back from ``admit_message`` with the reply and
  completes it with its own send;
* otherwise the drain completes it once the terminal outcome is committed: with the send outcome
  when it owes the reply (``deliver_settled``), else the commit itself is the delivery (an API
  caller, a local viewer or a finite CLI turn reads the committed result);
* a turn the drain could not settle, or whose waiter went away, is released unconfirmed.

Every release that is not a delivery logs an owner-tagged, body-free slug, so a token that
outlives its turn is greppable.
"""

import contextlib
import logging

logger = logging.getLogger(__name__)

_FIELD = "_gateway_review_delivery_complete"
# A waiter that will never read the completion the drain is still to leave (its wait was cancelled).
_WAITER_GONE = object()


def _held(authority):
    found = getattr(authority, "review_completions", None)
    if found is None:
        found = authority.review_completions = {}
    return found


def take_review_completion(event):
    """Detach the completion ``_handle_message_with_agent`` parked on ``event``, if any."""
    completion = getattr(event, _FIELD, None)
    if completion is None:
        return None
    with contextlib.suppress(Exception):
        delattr(event, _FIELD)
    return completion if callable(completion) else None


def hand_over_review_completion(authority, admission_id, event, *, native):
    """``execute_admission``: hold the executed turn's completion for its delivery owner —
    ``native`` names a live ``admit_message`` waiter whose lifecycle sends the reply. A waiter
    that went away while the turn ran is gone whatever ``native`` says: one that registered
    after the claim (``native`` False) is still registered when ``deliver_settled`` hands it
    the reply, so the completion must not be left for it."""
    completion = take_review_completion(event)
    if completion is None:
        return
    held = _held(authority)
    if held.get(admission_id) is _WAITER_GONE:
        held.pop(admission_id)
        _release(completion, _slugs().REASON_DELIVERY_UNCONFIRMED)
        return
    held[admission_id] = (completion, native)


def note_review_waiter(authority, admission_id):
    """``admit_message`` registered a live delivery waiter: a 'waiter gone' mark an earlier
    waiter of this admission left (a redelivered message deduped onto the executing row) no
    longer describes it, so the drain leaves the completion for this one."""
    held = _held(authority)
    if held.get(admission_id) is _WAITER_GONE:
        held.pop(admission_id)


def adopt_review_completion(authority, admission_id, event):
    """``admit_message``: carry the completion the drain left for this waiter on ``event``, the
    object its delivery lifecycle reads once the handler returns; ``event=None`` when no
    lifecycle will send (the row settled before this waiter saw it)."""
    held = _held(authority)
    entry = held.get(admission_id)
    if entry is None or entry is _WAITER_GONE or not entry[1]:
        return  # nothing left, or the drain's own to complete
    held.pop(admission_id)
    if event is None:
        _release(entry[0], _slugs().REASON_DELIVERY_UNCONFIRMED)
    else:
        setattr(event, _FIELD, entry[0])


def forget_review_waiter(authority, admission_id):
    """``admit_message``'s wait was cancelled: no lifecycle will send this reply. Release the
    completion if the drain already left it; while the turn is still pending or executing,
    remember that the drain must release what it is about to leave."""
    held = _held(authority)
    entry = held.get(admission_id)
    if entry is None:
        if (
            admission_id in authority.waiters
            or admission_id in authority.native_waiters
        ):
            held[admission_id] = _WAITER_GONE
        return
    if entry is not _WAITER_GONE and entry[1]:
        held.pop(admission_id)
        _release(entry[0], _slugs().REASON_DELIVERY_UNCONFIRMED)


def hand_review_completion_to_waiter(authority, admission_id):
    """``deliver_settled``: a native waiter registered after the turn started sends the reply."""
    entry = _held(authority).get(admission_id)
    if entry is not None and entry is not _WAITER_GONE:
        _held(authority)[admission_id] = (entry[0], True)


def complete_settled_review(authority, admission_id, *, delivery_succeeded):
    """The drain's own completion, after the terminal commit: ``delivery_succeeded`` is the send
    outcome when a reply was owed, else the commit itself delivered."""
    held = _held(authority)
    entry = held.get(admission_id)
    if entry is _WAITER_GONE:
        held.pop(admission_id)
        return
    if entry is None or entry[1]:
        return
    held.pop(admission_id)
    complete_review_delivery(entry[0], delivery_succeeded=delivery_succeeded)


def release_unsettled_review(authority, admission_id):
    """The drain could not commit the turn's outcome (fenced unknown or discarded): nobody is
    told an answer, so nothing is reviewed; the token is released either way."""
    entry = _held(authority).pop(admission_id, None)
    if entry is None or entry is _WAITER_GONE:
        return
    _release(entry[0], _slugs().REASON_TURN_UNSETTLED)


def release_failed_review(event):
    """``execute_admission`` unwound before handing over: the handler raised."""
    completion = take_review_completion(event)
    if completion is not None:
        _release(completion, _slugs().REASON_DELIVERY_UNCONFIRMED)


def complete_review_delivery(completion, *, delivery_succeeded, cause=None):
    """``_GatewayReviewAdmission.finish`` with a delivery outcome. It releases ownership before
    the spawn hop, so a failure here only loses the review: one owner-tagged line, never fatal."""
    try:
        completion(delivery_succeeded=delivery_succeeded, cause=cause)
    except Exception:
        logger.warning(
            "Background review delivery completion failed (owner=%s, reason=%s)",
            _owner(completion),
            _slugs().REASON_COMPLETION_ERROR,
            exc_info=True,
        )


def _release(completion, slug):
    logger.info(
        "Background review ownership released without delivery (owner=%s): %s",
        _owner(completion),
        slug,
    )
    complete_review_delivery(completion, delivery_succeeded=False, cause=slug)


def _owner(completion):
    return getattr(getattr(completion, "__self__", None), "owner_tag", "unbound")


def _slugs():
    from agent import review_admission

    return review_admission
