"""Exec-approval prompt delivery helpers for ``TurnRunner._approval_notify_sync``.

The combined approval description carries an optional, unverified model annotation
(purpose/effect/risk) composed for uncapped surfaces. A chat prompt re-fits that annotation to the
adapter's own budget at send time, or leaves it out where the budget is not established; the
scanner text is never shortened. Two ``BasePlatformAdapter`` class attributes declare those budgets:

* ``_EA_TEXT_BUDGET`` (default None): the finished card text's cap as the adapter delivers it
  (rejected or cut past it after the ``_format_exec_approval`` template), in ``message_len_fn``
  units. None = not established here, e.g. the relay, whose connector renders the card natively
  under a per-platform cap the contract does not negotiate (``max_message_length`` is the chat's
  text cap). Optional approval context is then left off the card, which renders as without it.
* ``_EA_CHAT_LIMIT_ESTABLISHED`` (default True): whether ``max_message_length_for_chat()`` /
  ``message_len_fn_for_chat()`` are each chat's own limit. False where they can stand in another
  value, e.g. the relay: ``_descriptor_for_chat`` stands the primary's descriptor in for a chat whose
  platform descriptor is unknown, absent or failed, and the handshake reads a malformed cap as 4096
  and an unknown unit as chars. The connector splits against that same negotiated cap
  (``splits_long_messages``), so splitting does not establish it either. The annotation is then left
  off a text prompt, split by ``send()`` or not, which renders as without it.

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


class _ExecApprovalDeclined(RuntimeError):
    """The connector refused the approval card's destination.

    Raised (not returned) so it propagates out of `_approval_notify_sync` to
    `_await_gateway_decision`, whose notify-failure path drops the central
    approval queue entry and unblocks the waiting tool. A plain return
    suppressed the text fallback but left that entry pending.
    """


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
    """``fits`` for ``adapter``'s approval card, or None when it declares no ``_EA_TEXT_BUDGET``
    (e.g. the relay connector renders the card natively under a per-platform cap the contract does
    not negotiate). The card cuts the reason at ``_EA_REASON_BUDGET`` and may size the command
    preview from the reason's length (``_format_exec_approval``): a candidate fits when its reason
    is uncut, its command preview is the plain card's, and the whole text is within the budget."""
    budget = adapter._EA_TEXT_BUDGET
    if not budget or budget <= 0:
        return None

    def rendered(reason: str) -> tuple:
        if adapter._EA_REASON_BUDGET:
            reason = adapter._ea_fit(reason, adapter._EA_REASON_BUDGET)
        return reason, adapter._ea_fit(command, adapter._exec_approval_cmd_budget(reason, smart_denied))

    def fits(candidate: str, plain: str) -> bool:
        return rendered(candidate) == (candidate, rendered(plain)[1]) and adapter.message_len_fn(
            adapter._format_exec_approval(command, candidate, smart_denied)) <= budget
    return fits


def _text_fits(adapter, chat_id, render):
    """``fits`` for the text prompt ``render(description)``: None where the adapter does not
    establish its per-chat cap as the chat's own (``_EA_CHAT_LIMIT_ESTABLISHED``; e.g. the relay,
    whose connector also splits against that cap, so splitting does not establish it either), else
    sent whole where ``send()`` splits long messages, else within the chat's own cap
    (``max_message_length_for_chat``, counted with ``message_len_fn_for_chat``)."""
    if not adapter._EA_CHAT_LIMIT_ESTABLISHED:
        return None
    if adapter.splits_long_messages:
        return lambda candidate, plain: True
    cap, len_fn = adapter.max_message_length_for_chat(chat_id), adapter.message_len_fn_for_chat(chat_id)
    return lambda candidate, plain: len_fn(render(candidate)) <= cap


def arm_timeout_notice(runner, approval_data: dict, *, command: str, card_message_id) -> None:
    """Register the expiry notice for a prompt that was (or may have been) delivered. Notice
    bookkeeping failing must neither re-send the prompt as text nor raise, which would withdraw
    the user's pending decision."""
    from gateway.run_turn_runner_approval_settle import register_timeout_notice
    try:
        register_timeout_notice(runner, approval_data, command=command, card_message_id=card_message_id)
    except Exception:
        logger.warning("Approval expiry-notice registration failed", exc_info=True)


def send_text_approval_prompt(runner, approval_data: dict, *, desc: str, command: str, flags: dict) -> None:
    """Send the plain-text ``/approve`` prompt for ``approval_data`` from the agent thread.

    Uses the adapter's typed prefix (e.g. ``!approve``): typed "/" is blocked in Slack threads and
    reserved by Matrix clients. Raises :class:`ApprovalDeliveryError` when no actionable prompt
    reached the user (the pending request then fails); an ambiguous send stays armed.
    """
    from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata
    ctx = runner._ctx
    adapter = ctx._status_adapter
    prefix = getattr(adapter, "typed_command_prefix", "/")

    def text_prompt(description: str) -> str:
        return _format_exec_approval_fallback(command, description, prefix, **flags)

    text_desc = _fit_approval_description(
        adapter, desc, approval_data, ctx.session_key or "",
        lambda: _text_fits(adapter, ctx._status_chat_id, text_prompt))
    if text_desc is None:
        logger.info("Approval request settled before its prompt was sent; not sending it")
        return
    msg = text_prompt(text_desc)
    try:
        # Mark as approval prompt: WeCom routes it through the control lane and Telegram pushes it
        # in "important" mode (#132516). Never ``notify`` — A2A reads that as the turn-final reply.
        metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
        fut = runner._schedule(
            adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
        )
        if fut is None:
            raise ApprovalDeliveryError("Approval text-send: loop unavailable")
        outcome = _approval_send_outcome(fut, timeout=15)
    except ApprovalDeliveryError:
        raise
    except Exception as e:
        raise ApprovalDeliveryError("Failed to send approval request") from e
    if outcome in {"failed", "declined"}:
        raise ApprovalDeliveryError("Failed to send approval request")
    # Ambiguous delivery keeps the pending request armed for a late reply;
    # the decision wait still blocks on silence. Never send a duplicate.
    # Preserve upstream expiry notices for delivered or possibly-delivered prompts.
    arm_timeout_notice(runner, approval_data, command=command, card_message_id=None)
