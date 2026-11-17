"""Keep a messaging exec-approval card truthful after its wait ends (messaging platforms).

``tools.approval_gateway_wait._await_gateway_decision`` calls ``entry.settle(reason)`` once the
wait ends. The turn runner registers the hook here right after the prompt was delivered. A card
whose timer ran out is edited to the timed-out notice (or the notice is posted). A BUTTON card is
also edited, in place, to its decided state (``ApprovalCardState``) whenever its own buttons did
not draw it: answered by typed ``/approve`` / ``/deny`` or another surface (denied, with the deny
reason), withdrawn when the turn ended or the session was reset under it (nobody refused it), and, once approved,
running, then finished or failed when the tool call that asked reports back, or unknown when the
turn ended first. Without this the card kept live buttons whose tap said "expired".

Best-effort by design: a failed notice or edit is logged at debug — the approval already
resolved, and nothing here may block the agent thread.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from agent.display import _detect_tool_failure
from agent.i18n import t
from gateway.platforms.base import fit_prefix
from gateway.platforms.base_exec_approval import (
    ApprovalCardOutcome, ApprovalCardState, approval_card_notice, approval_card_record, approval_timeout_seconds,
    ea_default_reason_text, ea_reason_label_text, format_approval_timed_out_notice)
from tools.approval_gateway_wait import ApprovalSettlement, current_settlement

logger = logging.getLogger(__name__)


def register_timeout_notice(
    runner, approval_data: dict, *, command: str, card_message_id: Optional[str]) -> None:
    """Arm a settle hook that posts the timed-out notice for ``approval_data['request_id']``.

    ``runner`` is the ``TurnRunner`` (for ``_ctx`` and ``_schedule``); ``card_message_id`` is the
    delivered BUTTON card's id when the adapter returned one, so the card itself is edited in place
    (which also drops its buttons). The plain-text prompt passes ``None``: it has no buttons to
    drop and rewriting it would erase the record of what was asked. ``command`` is the
    already-redacted command shown to the user. The notice is skipped when the run is no longer
    current (``ctx._run_still_current``).
    """
    from tools.approval import register_gateway_settle

    request_id = approval_data.get("request_id")
    session_key = runner._ctx.session_key or ""
    if not request_id or not session_key:
        return
    timeout_s = approval_data.get("timeout_s")
    if timeout_s is None:
        timeout_s = approval_timeout_seconds()
    flagged = approval_data.get("description") or ea_default_reason_text()

    card = _Card(request_id, card_message_id, command, flagged) if card_message_id else None

    def settle(reason: str) -> None:
        if reason != "timeout":
            # A text prompt has no buttons to retire; its /approve or /deny reply is the record.
            if card is not None:
                _settle_card(runner, card, reason, current_settlement())
            return
        # Same guard as every other late notice in TurnRunner: after /stop, /new or a restart the
        # turn is over and this chat belongs to a newer run — do not edit or post into it.
        still_current = getattr(runner._ctx, "_run_still_current", None)
        if callable(still_current) and not still_current():
            return
        runner._schedule(
            _post_timeout_notice(runner._ctx, command, card_message_id, timeout_s, flagged),
            "Approval timeout notice scheduling error")

    register_gateway_settle(session_key, request_id, settle)


async def _post_timeout_notice(
        ctx, command: str, card_message_id: Optional[str], timeout_s: int, reason: str = "") -> None:
    from gateway.run import _interim_metadata

    from gateway.bot_peer_notices import automatic_notices_allowed

    adapter = ctx._status_adapter
    if not automatic_notices_allowed(getattr(ctx, "source", None)):
        return
    notice = format_approval_timed_out_notice(timeout_s)
    metadata = _interim_metadata(ctx._status_thread_metadata)
    try:
        # Plain markdown, not the card's platform markup: ``edit_message`` re-formats it itself.
        if card_message_id and await _edit_card(adapter, ctx._status_chat_id, card_message_id, notice, command, reason):
            return
        await adapter.send(ctx._status_chat_id, notice, metadata=metadata)
    except Exception:
        logger.debug("Approval timeout notice failed", exc_info=True)


def _timed_out_card_body(adapter, chat_id: str, notice: str, command: str, reason: str = "") -> str:
    """The notice plus the reason and command, which are now the card's only record of the request. It must fit one
    message in the platform's own units (Telegram counts UTF-16), or the edit is truncated in
    place behind a dangling "(1/N)" and keeps streaming-preview state. A run of backticks in the
    command would close the card's fence and let the platform's formatter rewrite, and grow,
    the rest, so it is split with zero-width spaces (which also ride along in a copy)."""
    command, reason = (re.sub(r"`{3,}", lambda run: "\u200b".join(run.group()), s) for s in (command, reason))
    measure, suffix = adapter.message_len_fn_for_chat(chat_id), "..."
    if reason:  # capped like the card's own reason, so the command keeps most of the budget
        notice = f"{notice}\n{ea_reason_label_text()}: {fit_prefix(reason, 500, measure, suffix)}"
    frame = measure(t("gateway.exec_approval.timed_out_card_body", notice=notice, command=""))
    budget = adapter.max_message_length_for_chat(chat_id) - frame - measure(suffix)
    return t("gateway.exec_approval.timed_out_card_body", notice=notice,
             command=fit_prefix(command, max(0, budget), measure, suffix))


async def _edit_card(adapter, chat_id: str, message_id: str, notice: str, command: str, reason: str = "") -> bool:
    """Edit the card in place to the timed-out body; False sends the notice as a new message.
    Platforms whose edit replaces the markup drop the buttons here; Discord's edit keeps them,
    and its card's settle observer removes them."""
    edit: Optional[Callable[..., Any]] = getattr(adapter, "edit_message", None)
    if edit is None:
        return False
    try:
        result = await edit(chat_id, message_id, _timed_out_card_body(adapter, chat_id, notice, command, reason))
    except Exception:
        logger.debug("Approval card edit failed; sending the notice as a new message", exc_info=True)
        return False
    return bool(getattr(result, "success", False))


@dataclass(frozen=True)
class _Card:
    """A delivered button card: its request, message id, and the redacted command and flag reason."""
    request_id: str
    message_id: str
    command: str
    flagged: str


# A wait that ended under the prompt with nobody answering (turn over, session reset). An
# ``interrupted`` wait (/stop) is left to the stop reply, as before.
_WITHDRAWN_REASONS = frozenset({"session_closed"})
# Edits Telegram defers while the chat's send+edit slot is busy are retried this many times, 1s apart.
_SKIPPED_EDIT_RETRIES = 3


def _settled_outcome(reason: str, settled: Optional[ApprovalSettlement]) -> Optional[ApprovalCardOutcome]:
    """What a settled card should say when its own buttons did not say it; None leaves it as drawn."""
    if reason in _WITHDRAWN_REASONS:
        return ApprovalCardOutcome(ApprovalCardState.WITHDRAWN)
    if reason != "resolved" or settled is None or settled.choice is None or settled.via_card:
        return None
    if settled.choice == "deny":
        return ApprovalCardOutcome(ApprovalCardState.DENIED, reason=settled.deny_reason or "")
    return ApprovalCardOutcome(ApprovalCardState.APPROVED)


def _settle_card(runner, card: _Card, reason: str, settled: Optional[ApprovalSettlement]) -> None:
    """Edit ``card`` to how its wait ended. Runs on the waiting agent thread."""
    approved = reason == "resolved" and settled is not None and settled.choice not in (None, "deny")
    tracked = approved and _track_completion(runner, card, settled)
    if tracked and settled.via_card:
        return  # the button already shows who approved; the completion edit follows
    outcome = ApprovalCardOutcome(ApprovalCardState.APPLYING) if tracked else _settled_outcome(reason, settled)
    if outcome is not None:
        runner._schedule(_edit_decided_card(runner._ctx, card, outcome), "Approval card edit scheduling error")


def _track_completion(runner, card: _Card, settled: ApprovalSettlement) -> bool:
    """Watch this turn's own tool call for the approved command's result. False when it cannot be
    watched: a delegated child asked (another agent runs the call), a muted turn (no tool callbacks),
    or a prompt outside a tool call."""
    ctx = runner._ctx
    agent = (getattr(ctx, "agent_holder", None) or [None])[0]
    if (agent is None or ctx.mute_notification_reply or not settled.tool_call_id
            or settled.session_id != (getattr(agent, "session_id", "") or "")):
        return False
    inner = agent.tool_complete_callback
    if not getattr(inner, "watches_approval_cards", False):
        agent.tool_complete_callback = _completion_observer(runner, inner)
    ctx.open_approval_cards[settled.tool_call_id] = card
    return True


def _completion_observer(runner, inner: Optional[Callable[..., Any]]) -> Callable[..., Any]:
    """``inner`` (the turn's own tool-complete callback, if any) plus the approved cards' final edit."""
    def observe(call_id, tool_name, args, result):
        if inner is not None:
            inner(call_id, tool_name, args, result)
        card = runner._ctx.open_approval_cards.pop(call_id, None)
        if card is None:
            return
        failed, detail = _detect_tool_failure(tool_name, result)
        outcome = (ApprovalCardOutcome(ApprovalCardState.FAILED, reason=detail.strip().strip("[]").strip())
                   if failed else ApprovalCardOutcome(ApprovalCardState.APPROVED))
        runner._schedule(_edit_decided_card(runner._ctx, card, outcome), "Approval card edit scheduling error")

    observe.watches_approval_cards = True
    return observe


def settle_unreported_cards(runner) -> None:
    """At turn end: an approved command whose tool call never reported back may or may not have run."""
    cards = getattr(runner._ctx, "open_approval_cards", None) or {}
    while cards:
        _call_id, card = cards.popitem()
        runner._schedule(_edit_decided_card(runner._ctx, card, ApprovalCardOutcome(ApprovalCardState.UNKNOWN)),
                         "Approval card edit scheduling error")


async def _edit_decided_card(ctx, card: _Card, outcome: ApprovalCardOutcome) -> None:
    """Edit the card in place (never a new message) behind the card's lock, so a button handler's
    own decision edit cannot land after, and hide, this one. Keeps who answered on the card."""
    from gateway.bot_peer_notices import automatic_notices_allowed

    if not automatic_notices_allowed(getattr(ctx, "source", None)):
        return
    adapter, chat_id = ctx._status_adapter, ctx._status_chat_id
    edit: Optional[Callable[..., Any]] = getattr(adapter, "edit_message", None)
    if edit is None:
        return
    record = approval_card_record(card.request_id)
    try:
        async with record.lock:
            notice = approval_card_notice(replace(outcome, by=outcome.by or record.by))
            body = _timed_out_card_body(adapter, chat_id, notice, card.command, card.flagged)
            for attempt in range(_SKIPPED_EDIT_RETRIES):
                result = await edit(chat_id, card.message_id, body)
                raw = getattr(result, "raw_response", None)
                if not (isinstance(raw, dict) and raw.get("skipped")) or attempt == _SKIPPED_EDIT_RETRIES - 1:
                    return
                await asyncio.sleep(1.0)
    except Exception:
        logger.debug("Approval card edit (%s) failed", outcome.state.value, exc_info=True)
