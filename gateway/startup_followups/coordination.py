"""Transfer an admitted messaging follow-up into the same startup generation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
import logging
from typing import Any

from agent.startup_followups import StartupFollowups
from gateway.session_persistence import authority_persistence
from persistence.core.errors import AuthorityRefusal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candidate:
    key: str
    run_generation: int
    mode: str
    stop_epoch: int = 0


def candidate(runner: Any, event: Any, source: Any, key: str) -> Candidate | None:
    """Only normal, same-sender text may bypass its next-turn queue position."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    if (runner._draining or getattr(event, "_fifo_only", False) or getattr(event, "internal", False)
            or getattr(source, "is_bot", False) or event.get_command() or not runner._hm_text_only(event)
            or not (event.text or "").strip() or runner._clarify_takeover(event, key)):
        return None
    mode = runner._effective_busy_input_mode(source)
    if mode not in ("interrupt", "steer"):
        return None
    if mode == "interrupt" and (runner._effective_busy_text_mode(source) == "queue"
                                or runner._hm_in_telegram_grace(event, source, key)):
        return None
    state = runner._peek_session_state(key)
    sender = event.sender_id()
    if state is None or state.turn.event is None or not sender or state.turn.event.sender_id() != sender:
        return None
    agent = state.turn.agent
    if agent is not _AGENT_PENDING_SENTINEL:
        startup = getattr(agent, "_gateway_startup_followups", None)
        if not isinstance(startup, StartupFollowups) or not startup.is_open():
            return None
    return Candidate(key, runner._current_session_run_generation(key), mode, runner._route_stop_epoch(key))


def _current(authority: Any, ref: Any, generation: int, selected: Candidate) -> bool:
    return (authority.executing.get(ref.session_id) == generation
            and authority.route_epoch(selected.key) == selected.stop_epoch
            and authority.runner._is_session_run_current(selected.key, selected.run_generation))


def cancel_unread(authority: Any, session_id: str, live: Any) -> None:
    """Fence unread transferred messages as cancelled, before stopping their owner."""
    with live.event_stream.lock:
        generation = authority.executing.get(session_id)
        entry = authority.steer_transfers.get(session_id)
        if generation is None or entry is None or entry[0] != generation:
            return
        pending = authority.startup_pending.pop((session_id, generation), {})
        tickets = list(pending)
        adopted_generation, agent = authority.adopted.get(session_id, (None, None))
        startup = getattr(agent, "_gateway_startup_followups", None)
        if adopted_generation == generation and isinstance(startup, StartupFollowups):
            tickets.extend(startup.cancel_unread())
        for ticket in tickets:
            if ticket in entry[1]:
                entry[1][ticket] = 'cancelled'  # settlement cancels it rather than requeueing it


def _deliver(authority: Any, ref: Any, generation: int, ticket: str, text: str,
             selected: Candidate, event: Any) -> bool | None:
    """Caller holds the stream lock. None means the current turn has no agent yet."""
    if not _current(authority, ref, generation, selected):
        return False
    adopted_generation, agent = authority.adopted.get(ref.session_id, (None, None))
    if adopted_generation != generation:
        return None
    startup = getattr(agent, "_gateway_startup_followups", None)
    if not isinstance(startup, StartupFollowups):
        return False
    accepted = startup.offer(ticket, text, (selected, event), display_text=event.text)
    if accepted:
        authority.steer_transfers[ref.session_id][1][ticket] = True
    return accepted


async def transfer(authority: Any, receipt: Any, event: Any, selected: Candidate) -> None:
    """Move a durably queued head to its running claim, or leave it in FIFO order."""
    live = authority.sessions.get(receipt.ref.session_id)
    if live is None:
        return
    async with live.startup_transfer_lock:
        await _transfer_locked(authority, receipt, event, selected, live)


async def _transfer_locked(authority: Any, receipt: Any, event: Any, selected: Candidate, live: Any) -> None:
    from gateway.session_queued_controls import _register, _unregister
    from gateway.session_runtime_workers import tracked_write

    ref, ticket = receipt.ref, receipt.admission_id
    store = authority_persistence(authority)
    rows = await asyncio.to_thread(store.admissions.list, session_id=ref.session_id)
    head = next((row for row in rows if row.status == "queued"), None)
    if head is None or head.admission_id != ticket:
        return
    runner = authority.runner
    try:
        text = runner._prepend_inbound_reply_context(event, event.source, event.text)
        text = runner._steer_text_with_origin(text, event)
    except Exception:
        logger.warning("Could not prepare a startup follow-up; keeping it queued", exc_info=True)
        return
    with live.event_stream.lock:
        generation = authority.executing.get(ref.session_id)
        if generation is None or not _current(authority, ref, generation, selected):
            return
        _register(authority, ref.session_id, generation, ticket)
    try:
        _row, fresh = await tracked_write(authority, partial(
            store.admissions.steer, epoch=authority.epoch, admission_id=ticket, generation=generation))
    except AuthorityRefusal:
        with live.event_stream.lock:
            _unregister(authority, ref.session_id, generation, ticket)
        return
    if not fresh:
        return
    with live.event_stream.lock:
        accepted = _deliver(authority, ref, generation, ticket, text, selected, event)
        if accepted is None:
            authority.startup_pending.setdefault((ref.session_id, generation), {})[ticket] = (ref, text, selected, event)
    if accepted is False:
        await tracked_write(authority, partial(store.admissions.revert_steer, epoch=authority.epoch,
                                               admission_id=ticket, generation=generation))
        with live.event_stream.lock:
            _unregister(authority, ref.session_id, generation, ticket)
        authority._publish_pending(ref)
        authority._schedule(ref)


def deliver_adopted(authority: Any, session_id: str, generation: int) -> None:
    """Deliver latches after this generation adopts its agent; caller holds the stream lock."""
    pending = authority.startup_pending.pop((session_id, generation), {})
    for ticket, (ref, text, selected, event) in pending.items():
        _deliver(authority, ref, generation, ticket, text, selected, event)


def note_sent(runner: Any, ctx: Any, agent: Any, authority: Any, session_id: str,
              generation: int, entries: list[Any]) -> None:
    """Apply the existing per-mode reply ownership once the request takes these inputs."""
    from gateway.run import _AGENT_PENDING_SENTINEL

    live = authority.sessions.get(session_id)
    if live is None:
        return
    with live.event_stream.lock:
        if authority.executing.get(session_id) != generation:
            return
        state = runner._peek_session_state(ctx.session_key)
        if state is None or (state.turn.agent is not agent and state.turn.agent is not _AGENT_PENDING_SENTINEL):
            return
        turn = state.turn
        for entry in entries:
            selected, event = entry.origin
            if not runner._is_session_run_current(selected.key, selected.run_generation):
                continue
            running = turn.running_event()
            if running is not None and running is not event:
                running.absorb_turn_input(event)
            if turn.event is None or turn.event is event:
                continue
            turn.event.absorb_reply_expected(event)
            ctx.reply_expected = turn.event.reply_expected
            if selected.mode == "interrupt":
                anchor = runner._reply_anchor_for_event(event)
                inbound_id = str(event.message_id) if event.message_id else None
                turn.event.reply_anchor_override, turn.event.ledger_message_id = anchor, inbound_id
                ctx.event_message_id, ctx.inbound_message_id = anchor, inbound_id
                holder = getattr(ctx, 'stream_consumer_holder', None)
                if holder and holder[0] is not None:
                    holder[0]._initial_reply_to_id = anchor
                admission_id = live.event_stream.execution.get('admission_id')
                original = getattr(authority, 'marker_adopters', {}).get(admission_id)
                if original is not None:
                    original.reply_anchor_override, original.ledger_message_id = anchor, inbound_id
        from gateway.response_filters import silence_allowed
        agent._silence_marker_needs_reply = not silence_allowed(ctx.persist_user_display_kind, ctx.reply_expected)
