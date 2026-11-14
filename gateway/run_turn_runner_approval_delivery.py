"""Exec-approval prompt delivery helpers for ``TurnRunner._approval_notify_sync``.

The combined approval description carries an optional, unverified model annotation
(purpose/effect/risk) composed for uncapped surfaces. A chat prompt re-fits that annotation to the
adapter's own budget at send time, or leaves it out where the budget is not established; the
scanner text is never shortened here. Two ``BasePlatformAdapter`` class attributes declare those budgets:

* ``_EA_CARD_CAP`` (default 0): the finished card text's cap as the adapter delivers it (rejected or
  cut past it), in ``message_len_fn`` units; the card template already sizes the command preview
  from it (``base_exec_approval.render_exec_approval``). 0 = not established here, e.g. the relay,
  whose connector renders the card natively under a per-platform cap the contract does not
  negotiate, or Matrix, Feishu, Teams and QQ, which budget only the preview. Optional approval
  context is then left off the card, which renders as without it.
* ``_EA_CHAT_LIMIT_ESTABLISHED`` (default True): whether ``max_message_length_for_chat()`` /
  ``message_len_fn_for_chat()`` are each chat's own limit. False where they can stand in another
  value, e.g. the relay: ``_descriptor_for_chat`` stands the primary's descriptor in for a chat whose
  platform descriptor is unknown, absent or failed, and the handshake reads a malformed cap as 4096
  and an unknown unit as chars. The connector splits against that same negotiated cap
  (``splits_long_messages``), so splitting does not establish it either. The annotation is then left
  off a text prompt, split by ``send()`` or not, which renders as without it.

The annotation never changes what else a prompt shows or offers: a card keeps the plain card's
command preview, fence and cut notice (so its Deny-only decision is the plain card's), and a text
prompt keeps the plain prompt's layout and choices.

``gateway.run`` internals are imported lazily (import cycle), so ``patch("gateway.run.X")`` keeps
intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import Optional

from gateway.platforms.base import BasePlatformAdapter

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class ApprovalDeliveryError(RuntimeError):
    """No actionable approval prompt reached the user; fail the pending request."""


def _fit_approval_description(adapter, desc: str, approval_data: dict, session_key: str, fits_for) -> Optional[str]:
    """Fit the unverified model annotation in ``desc`` to one approval prompt.

    ``desc`` was composed for uncapped surfaces. ``fits_for()`` returns ``fits(candidate, plain)``,
    which tells whether the prompt shows ``candidate`` whole within the prompt's budget and shows
    everything it shows for the scanner-only ``plain``, or None when that budget is not established
    for this adapter and chat; the annotation is then left out and the prompt renders exactly as it
    would without it. Otherwise the annotation is shortened between both delimiters, or left out,
    until it fits. The scanner text comes from the queued request, which never carries the
    annotation, and is never shortened here. None when the request is no longer queued (answered or
    withdrawn since): there is nothing left to approve.
    """
    from gateway.run import _redact_approval_command
    from tools.approval import _build_enhanced_description_with_context, list_gateway_approvals

    explanation = approval_data.get("explanation")
    if not explanation or not isinstance(adapter, BasePlatformAdapter):
        return desc
    pending = next((a for a in list_gateway_approvals(session_key)
                    if a.get("request_id") == approval_data.get("request_id")), None)
    if pending is None:
        return None
    scanner = pending.get("description")
    if not scanner or not str(scanner).strip():
        return desc
    scanner = _redact_approval_command(scanner)
    fits = fits_for()
    if fits is None:
        return scanner
    if fits(desc, scanner):
        return desc
    best, lo, hi = scanner, 0, len(desc)
    while lo <= hi:  # longest annotation budget that still fits
        mid = (lo + hi) // 2
        candidate = _redact_approval_command(_build_enhanced_description_with_context(scanner, explanation, mid))
        if fits(candidate, scanner):
            best, lo = candidate, mid + 1
        else:
            hi = mid - 1
    return best


def _card_fits(adapter, command: str, smart_denied: bool):
    """``fits`` for ``adapter``'s approval card, or None when it declares no ``_EA_CARD_CAP``. The card
    may cut the reason (``_EA_REASON_BUDGET``, the cap's share) and sizes the command preview from the
    reason's length (``card_layout``). A candidate fits when, on both cards ``send_exec_approval`` may
    send (full command above, or Deny only), its reason is uncut, its preview, fence and cut notice
    are the plain card's, and the whole text is within the cap."""
    from gateway.platforms.base_exec_approval import card_layout, render_exec_approval
    cap = adapter._EA_CARD_CAP
    if not cap or cap <= 0:
        return None

    def fits(candidate: str, plain: str) -> bool:
        for shown_above in (True, False):
            reason, *shown = card_layout(adapter, command, candidate, smart_denied, shown_above)
            if reason != candidate or shown != list(card_layout(adapter, command, plain, smart_denied, shown_above)[1:]):
                return False
            if adapter.message_len_fn(render_exec_approval(adapter, command, candidate, smart_denied, shown_above)[0]) > cap:
                return False
        return True
    return fits


def _text_fits(adapter, render):
    """``fits`` for the text prompt ``render(description) -> (text, layout)`` (``text_prompt``): None
    where the adapter does not establish its per-chat cap as the chat's own
    (``_EA_CHAT_LIMIT_ESTABLISHED``; e.g. the relay, whose connector also splits against that cap, so
    splitting does not establish it either). Else a candidate fits when it keeps the plain prompt's
    layout and that layout offers every choice: sent whole where ``send()`` splits long messages
    and can prove it, or within the chat's own cap (``max_message_length_for_chat``, counted with
    ``message_len_fn_for_chat``) with the instructions first."""
    if not getattr(adapter, "_EA_CHAT_LIMIT_ESTABLISHED", True):
        return None
    return lambda candidate, plain: render(candidate)[1] == render(plain)[1] != "deny"


def _timeout_notice_data(approval_data: dict, session_key: str) -> dict:
    """``approval_data`` as the expiry notice shows it. The timed-out card keeps why the command was
    flagged (``register_timeout_notice`` reads ``description``), which is the queued request's scanner
    text. The optional annotation is context for the live prompt only: cut to the notice's reason
    budget it could lose its closing delimiter, and a prompt that left it out would show it only
    after expiry. Not queued any more: no notice can be armed; the annotation is dropped all the same."""
    if not approval_data.get("explanation"):
        return approval_data
    from gateway.run import _redact_approval_command
    from tools.approval import list_gateway_approvals

    pending = next((a for a in list_gateway_approvals(session_key)
                    if a.get("request_id") == approval_data.get("request_id")), None)
    scanner = pending.get("description") if pending else None
    return {**approval_data,
            "description": _redact_approval_command(scanner) if scanner and str(scanner).strip() else None}


def arm_timeout_notice(runner, approval_data: dict, *, command: str, card_message_id) -> None:
    """Register the expiry notice for a prompt that was (or may have been) delivered. Notice
    bookkeeping failing must neither re-send the prompt as text nor raise, which would withdraw
    the user's pending decision."""
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    try:
        register_timeout_notice(runner, _timeout_notice_data(approval_data, runner._ctx.session_key or ""),
                                command=command, card_message_id=card_message_id)
    except Exception:
        logger.warning("Approval expiry-notice registration failed", exc_info=True)


def send_text_approval_prompt(runner, approval_data: dict, *, desc: str, command: str, flags: dict,
                              background: bool = False) -> None:
    """Send the plain-text ``/approve`` prompt for ``approval_data`` from the agent thread.

    Uses the adapter's typed prefix (e.g. ``!approve``): typed "/" is blocked in Slack threads and
    reserved by Matrix clients. A prompt that offers Deny alone marks the request Deny-only
    (``approval_text_prompt``). Raises :class:`ApprovalDeliveryError` when no actionable prompt
    reached the user (the pending request then fails); an ambiguous send (a receipt that timed out,
    a lost ack) stays armed. A turn whose session authority publishes the request to its own viewers
    after this send (``runner._approval_owner``: hosted rooms, local sessions; ``_approval_notify_sync``)
    has a second surface, so only a connector decline fails it; a failed chat send is logged, as on
    main. A ``background`` child's prompt (``run_turn_runner_approval_transport``) has no other surface.
    """
    from gateway.run import _approval_send_outcome, _interim_metadata
    from gateway.run_turn_runner_approval_text import approval_text_prompt, text_prompt
    ctx = runner._ctx
    adapter, chat_id, session_key = ctx._status_adapter, ctx._status_chat_id, ctx.session_key or ""
    prefix = getattr(adapter, "typed_command_prefix", "/")
    text_desc = _fit_approval_description(
        adapter, desc, approval_data, session_key,
        lambda: _text_fits(adapter, lambda description: text_prompt(adapter, chat_id, command, description, prefix, **flags)))
    if text_desc is None:
        logger.info("Approval request settled before its prompt was sent; not sending it")
        return
    msg = approval_text_prompt(adapter, chat_id, session_key, approval_data.get("request_id"), command, text_desc,
                               prefix, **flags)
    fut, cause = None, "Failed to send approval request"
    try:
        # Mark as approval prompt: WeCom routes it through the control lane and Telegram pushes it
        # in "important" mode (#132516). Never ``notify`` — A2A reads that as the turn-final reply.
        metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
        fut = runner._schedule(
            adapter.send(chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
        )
        if fut is None:
            cause = "Approval text-send: loop unavailable"
        outcome = _approval_send_outcome(fut, timeout=15)  # no future: "failed", logged there
    except Exception as e:
        logger.error("Failed to send approval request: %s", e, exc_info=True)
        outcome = "failed"
    if fut is not None:
        _log_text_receipt(fut, outcome, approval_data.get("request_id"), ctx.session_key)
    if outcome == "declined" or (outcome == "failed" and (background or getattr(runner, "_approval_owner", None) is None)):
        raise ApprovalDeliveryError(cause)
    # Ambiguous delivery keeps the pending request armed for a late reply;
    # the decision wait still blocks on silence. Never send a duplicate.
    # No card to edit on the text path: the prompt has no buttons to drop and carries
    # the /approve instructions, so the timeout notice is posted as a new message.
    arm_timeout_notice(runner, approval_data, command=command, card_message_id=None)


def _settled_result(fut, outcome: str):
    """The send's result for the receipt log; None when it never resolved or raised."""
    if outcome == "ambiguous":
        return None
    try:
        return fut.result(timeout=0)
    except Exception:  # health: allow BLE001 -- already classified and logged by _approval_send_outcome
        return None


def _log_text_receipt(fut, outcome: str, request_id, session_key) -> None:
    """Text-lane receipt (#138133): a card failure lands here, and a failed fallback was silent too.
    Correlation fields only, never the command text."""
    res = _settled_result(fut, outcome)
    logger.log(logging.INFO if outcome == "sent" else logging.WARNING,
               "Exec approval text prompt send (success=%s, outcome=%s, message_id=%s, error=%s, request_id=%s, session=%s)",
               getattr(res, "success", None), outcome, getattr(res, "message_id", None), getattr(res, "error", None),
               request_id, session_key)
