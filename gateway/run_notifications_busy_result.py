"""Which finished background results may ride a running turn, and the process-completion carry.

``delegation.busy_result_mode`` governs subagent results and
``display.background_process_busy_result_mode`` governs ``notify_on_complete`` process completions.
With ``steer`` a result that finishes while its session's turn runs is appended to that turn's next
tool result (``agent.background_results``) instead of queueing a new turn.

A process completion has no durable claim to hold, so its delivery call waits on the carry instead:
the future resolves True once a tool result carried it and False when the turn ended without one,
and the caller then queues it as the next turn exactly as ``queue`` mode does.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

BUSY_RESULT_MODE_KEYS = {
    "async_delegation": ("delegation", "busy_result_mode"),
    "completion": ("display", "background_process_busy_result_mode"),
}


@dataclasses.dataclass(frozen=True)
class CarriedCompletion:
    """A claimed completion riding a running turn (``delegation.busy_result_mode: steer``): what
    settling it later needs once a tool result carried it, or the turn ended without one."""

    evt: dict
    delegation_id: str
    claim_id: str
    identity: Any
    siblings: tuple
    waiter: Optional[asyncio.Future] = None  # a process completion's carry outcome


def busy_result_mode(evt_type: str) -> str:
    """The profile in scope's busy-result mode for ``evt_type``: ``queue`` (default) or ``steer``."""
    from gateway.run import _load_gateway_config
    from hermes_cli.config import cfg_get

    section, key = BUSY_RESULT_MODE_KEYS[evt_type]
    mode = str(cfg_get(_load_gateway_config(), section, key, default="") or "queue").strip().lower()
    if mode not in ("queue", "steer"):
        logger.warning("%s.%s=%r is not 'queue' or 'steer'; using 'queue'", section, key, mode)
        return "queue"
    return mode


def offer_process_completion(agent: Any, carried: Any, synth_text: str) -> asyncio.Future:
    """Offer a process completion to ``agent``'s next tool result; return the carry outcome future."""
    from agent.background_results import PROCESS_RESULT_MARKER, offer, withdraw

    waiter = asyncio.get_running_loop().create_future()
    carried = dataclasses.replace(carried, waiter=waiter)
    session_ids = [str(sid or "") for sid in carried.evt.get("_batch_session_ids") or [carried.evt.get("session_id")]]

    def already_read() -> bool:
        # The agent read every output with process(wait/log) this turn: showing it again is noise.
        import tools.process_registry as process_registry_module
        return all(process_registry_module.process_registry.is_completion_consumed(sid) for sid in session_ids)

    offer(agent, carried, synth_text, marker=PROCESS_RESULT_MARKER, already_read=already_read)
    # A cancelled delivery (shutdown) is retried by its watcher, so the turn must not carry it too.
    waiter.add_done_callback(lambda done: done.cancelled() and withdraw(agent, carried))
    logger.info("Carrying process completion %s into the running turn of %s",
                ", ".join(session_ids), carried.evt.get("session_key"))
    return waiter


def carry_completion_into_running_turn(mixin: Any, synth_text: str, evt: dict, claim: Any, identity: Any,
                                       sibling_claims) -> Any:
    """``steer`` mode: hand a claimed completion to the session's running agent, which appends it to
    its next tool result (``agent.background_results``), instead of queueing a new turn. A delegation
    returns True and its claims stay held until :func:`settle_carried_completions` sees whether a tool
    batch carried it; a process completion returns that outcome as a future."""
    if evt.get("type") not in BUSY_RESULT_MODE_KEYS or evt.get("task_failure_notice") or evt.get("origin_session_id"):
        return False
    from gateway.run import _AGENT_PENDING_SENTINEL
    state = mixin._peek_session_state(str(evt.get("session_key") or ""))
    agent = state.turn.agent if state is not None else None
    if agent is None or agent is _AGENT_PENDING_SENTINEL or not callable(getattr(agent, "steer", None)):
        return False
    if busy_result_mode(evt["type"]) != "steer":
        return False
    from agent.background_results import offer
    carried = CarriedCompletion(evt, claim.delegation_id, claim.claim_id, identity, tuple(sibling_claims))
    mixin.__dict__.setdefault("_carrying_agents", {})[id(agent)] = agent
    if evt["type"] == "completion":
        return offer_process_completion(agent, carried, synth_text)
    offer(agent, carried, synth_text)
    logger.info("Carrying async delegation %s into the running turn of %s",
                claim.delegation_id or "<legacy>", evt.get("session_key"))
    return True


async def settle_carried_completions(mixin: Any, agent: Any = None, *, final: bool = False) -> None:
    """Acknowledge completions a tool result carried; at turn end (``final``) hand the rest back to
    the completion queue (claims deferred, no attempt spent) so they arrive as the next turn."""
    try:
        await _settle_carried_completions_unguarded(mixin, agent, final=final)
    except Exception:  # health: allow BLE001 -- runs in a turn's cleanup; the claim lease re-offers anything unsettled
        logger.warning("Could not settle carried subagent completions", exc_info=True)


async def _settle_carried_completions_unguarded(mixin: Any, agent: Any, *, final: bool) -> None:
    from agent.background_results import take_delivered, take_outcome
    agents = mixin.__dict__.get("_carrying_agents") or {}
    if agent is not None and id(agent) not in agents:
        return
    for current in [agent] if agent is not None else list(agents.values()):
        if final:
            agents.pop(id(current), None)
        # Settling an outcome can register a new carry: run_turn settles while the agent slot is
        # still held, so a delivery landing now sees that agent and re-registers. Drain until
        # quiet, bounded so a hot queue still ends.
        for _ in range(64):
            if final:
                delivered, undelivered = take_outcome(current)
            else:
                delivered, undelivered = take_delivered(current), []
            if not delivered and not undelivered:
                break
            for carried in delivered:
                await _settle_carried_completion(mixin, carried, "complete")
            for carried, _text in undelivered:
                await _settle_carried_completion(mixin, carried, "defer")


async def _settle_carried_completion(mixin: Any, carried: CarriedCompletion, operation: str) -> None:
    from tools.process_registry import process_registry as _pr
    if carried.waiter is not None:  # a process completion: its waiting delivery call settles it
        if not carried.waiter.done():
            carried.waiter.set_result(operation == "complete")
        return
    async with mixin._completion_event_scope(carried.evt):
        if carried.claim_id:
            mixin._settle_durable_claim(operation, carried.delegation_id, carried.claim_id)
        for sibling, claim_id in carried.siblings:
            if claim_id:
                mixin._settle_durable_claim(operation, sibling["delegation_id"], claim_id)
    if operation == "complete":
        if carried.identity is not None:
            with mixin._completion_delivery_lock:
                mixin._mark_completions_delivered_locked((carried.identity,))
        if carried.siblings:
            record_coalesced_completion_siblings(mixin, [event for event, _claim_id in carried.siblings])
        return
    if carried.identity is not None:
        with mixin._completion_delivery_lock:
            mixin._completion_deliveries_inflight.discard(carried.identity)
    evt = {k: v for k, v in carried.evt.items() if k != "completion_ids"}
    for event in (evt, *(sibling for sibling, _claim_id in carried.siblings)):
        _pr.completion_queue.put(event)


def record_coalesced_completion_siblings(mixin: Any, events: list[dict]) -> None:
    """Extend a successful primary delivery claim to its batched siblings."""
    identities = [i for i in map(mixin._completion_delivery_identity, events) if i is not None]
    with mixin._completion_delivery_lock:
        mixin._mark_completions_delivered_locked(identities)


def settle_completion_claims(mixin: Any, claim: Any, sibling_claims, identity: Any, identity_claimed: bool,
                             accepted: bool, refused: bool) -> None:
    """Settle a delivery's durable claims on the way out when no running turn is carrying it."""
    if identity_claimed and not accepted:
        with mixin._completion_delivery_lock:
            mixin._completion_deliveries_inflight.discard(identity)
    operation = "complete" if accepted else "defer" if refused or claim.defer else "release"
    if claim.claim_id:
        mixin._settle_durable_claim(operation, claim.delegation_id, claim.claim_id)
    for sibling, claim_id in sibling_claims:
        if claim_id:
            mixin._settle_durable_claim(operation, sibling["delegation_id"], claim_id)
    if accepted and sibling_claims:
        record_coalesced_completion_siblings(mixin, [event for event, _claim_id in sibling_claims])
