"""Receiver-side drainer for peer deliveries (spec §3.1 triggers 1-4).

``_bot_send_turn`` in ``api_server.py`` runs the delivery its own request was
admitted for. This module owns the other half of the acceptance promise: a
delivery that was accepted with ``receipt`` because the target's turn slot was
busy must still RUN once that slot frees. Without it the queue is write-only --
the live build held 53 ``queued`` records with a full body each and no code path
that advanced or expired a single one (VERIFICATION D1).

Four triggers converge here, all under the target's durable turn lock so a
drained delivery can never race the lock holder any more than a live request can:

1. admission -- :func:`drain_under_lock` keeps draining the queue under the lock
   the caller just acquired, until the caller's own delivery is at the head (or
   the caller's promised window elapses -- the caller then keeps its receipt).
2. turn end -- the same loop is what runs after each turn, so an idle target with
   a backlog drains it back-to-back instead of idling between chores.
3. the 30s sweep -- :func:`sweep_loop`, started beside ``_sweep_orphaned_runs``.
4. the hourly chore -- ``bot_delivery_queue.cleanup_bot_delivery_queue``.

The turn lock may be held by a UI/desktop session, not just a card worker (the
live build reproduced a 1800.3s lease wait against a desktop holder), so nothing
here expires or abandons a record merely because its target is mid-turn: the
holder is the one that will drain it.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tools import bot_delivery_queue as delivery_queue

#: One log line per delivery decision, in the queue module's own format.
log_delivery_event = delivery_queue._log

logger = logging.getLogger(__name__)

#: Triggers 1-2 run inside a caller's HTTP window: stop *starting* new turns once
#: the caller's promised window has elapsed. A turn already running is never
#: abandoned (it is the delivery that was promised).
IN_CALL_DRAIN_MIN_SECONDS = 1.0

#: Trigger 3's budget per tick, so one deep backlog cannot monopolise the loop.
SWEEP_DRAIN_BUDGET_SECONDS = 60.0


def delivery_run_kwargs(record: dict[str, Any]) -> Dict[str, Any]:
    """Run kwargs for a delivery with no live request behind it (§2.5).

    A drained record carries its own message and target session, so the turn is
    replayed into that session with the session's stored runtime selection --
    there is no sender body to re-read and no request-scoped model lock to honour.
    """
    return {
        "user_message": record.get("message") or "",
        "ephemeral_system_prompt": None,
        "session_id": record.get("target_session_id"),
        "gateway_session_key": None,
        "route": None,
        "session_model": None,
        "requested_runtime": {},
        "route_source": "global",
        "confirmed_runtime_lock": False,
    }


def _failure_reason(error: str) -> str:
    """Classify a turn failure with the fork's shared reason table."""
    try:
        from tools.bot_failure_reasons import UNKNOWN, classify_agent_error
    except Exception:
        return "unknown"
    try:
        return classify_agent_error(error)
    except Exception:
        return UNKNOWN


def _finalize_reply(payload: Any) -> str:
    """Resolve a turn's final response into the reply text stored on the record."""
    if not isinstance(payload, dict):
        return ""
    try:
        from gateway.platforms.api_server import _resolve_media_to_data_urls

        return _resolve_media_to_data_urls(payload.get("final_response", "") or "")
    except Exception:
        return str(payload.get("final_response", "") or "")


async def run_record(
    adapter: Any,
    home: Path,
    record: dict[str, Any],
    *,
    ctx: Optional[Dict[str, Any]] = None,
    waited: float = 0.0,
) -> Tuple[dict[str, Any], Optional[str]]:
    """Run one CLAIMED record's turn and settle it. Returns ``(record, reply)``.

    ``ctx`` is the live request's context when the record belongs to the caller
    (its body-selected runtime wins); a drained record passes ``ctx=None``.

    A contended session lease is never surfaced to a sender and never charged an
    attempt (§2.7 step 2): the record goes back to ``queued`` for the next free
    slot (``status: queued``, so the caller keeps its receipt).
    """
    delivery_id = str(record.get("delivery_id"))
    session_id = str(record.get("target_session_id") or "")
    if ctx is None:
        kwargs = delivery_run_kwargs(record)
    else:
        kwargs = dict(ctx.get("run_kwargs") or {})
        kwargs["session_id"] = session_id
    history = await adapter._conversation_history_for_session(session_id)
    started = time.monotonic()
    log_delivery_event("turn_start", record, drained=ctx is None)
    try:
        result, _usage = await adapter._run_agent(
            conversation_history=history,
            lease_wait_seconds=delivery_queue.lease_probe_seconds(),
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - any turn failure settles the record
        logger.exception("[api_server] peer delivery turn failed: %s", delivery_id)
        error = str(exc) or exc.__class__.__name__
        settled = delivery_queue.settle(
            home,
            delivery_id,
            status=delivery_queue.STATUS_FAILED,
            reply=None,
            error=error,
            reason=_failure_reason(error),
        )
        adapter._bot_send_record_status(
            delivery_id, settled, result=delivery_queue.RESULT_FAILED, error=error
        )
        log_delivery_event("turn_failed", settled, seconds=round(time.monotonic() - started, 3))
        return settled, None

    lease_timed_out = (
        isinstance(result, dict)
        and bool(result.get("failed"))
        and "lease" in str(result.get("error") or "").lower()
    )
    if lease_timed_out:
        requeued = delivery_queue.requeue_unstarted(home, delivery_id)
        adapter._bot_send_record_status(
            delivery_id, requeued, result=delivery_queue.RESULT_RECEIPT
        )
        log_delivery_event("turn_lease_contended", requeued, waited_seconds=waited)
        return requeued, None

    reply = _finalize_reply(result)
    settled = delivery_queue.settle(
        home, delivery_id, status=delivery_queue.STATUS_DELIVERED, reply=reply
    )
    adapter._bot_send_record_status(
        delivery_id, settled, result=delivery_queue.RESULT_DELIVERED, reply=reply
    )
    log_delivery_event("turn_end", settled, seconds=round(time.monotonic() - started, 3))
    return settled, reply


async def drain_under_lock(
    adapter: Any,
    *,
    home: Path,
    target_profile: str,
    delivery_id: str,
    ctx: Dict[str, Any],
    waited: float,
) -> Any:
    """Triggers 1-2: drain the queue under the turn lock this caller just took.

    HEAD-FIRST, always. Handing a foreign head back with another receipt is what
    wedged the live backlog (VERIFICATION D1): the record's own request is long
    gone, so nothing would ever pick it up again. The caller's delivery keeps its
    place in FIFO order and receives the receipt the caller was promised once its
    window elapses.
    """
    started = time.monotonic()
    window = delivery_queue.receipt_after_seconds() - waited
    deadline = started + max(IN_CALL_DRAIN_MIN_SECONDS, window)
    drained = 0
    while True:
        record = delivery_queue.claim_next(home, target_profile=target_profile, lease_ok=True)
        if record is None:
            break
        record_id = str(record.get("delivery_id"))
        if record_id == delivery_id:
            settled, _reply = await run_record(adapter, home, record, ctx=ctx, waited=waited)
            elapsed = time.monotonic() - started
            if str(settled.get("status")) == delivery_queue.STATUS_QUEUED:
                return adapter._bot_send_receipt(
                    delivery_id, settled, waited=elapsed, home=home
                )
            return adapter._bot_send_json(
                delivery_queue.build_envelope(settled, waited_seconds=elapsed)
            )
        log_delivery_event("draining_ahead", record, drained=drained + 1)
        await run_record(adapter, home, record)
        drained += 1
        if time.monotonic() >= deadline:
            break

    current = delivery_queue.read_record(home, delivery_id)
    if current is None:
        logger.error("[api_server] peer delivery vanished mid-flight: %s", delivery_id)
        from gateway.platforms.api_server import _error_response

        return _error_response(
            "Delivery is already being handled.", 409, code="delivery_in_progress"
        )
    return adapter._bot_send_receipt(
        delivery_id, current, waited=time.monotonic() - started, home=home
    )


async def drain_once(
    adapter: Any, home: Path, *, budget_seconds: Optional[float] = None
) -> int:
    """Run queued deliveries for every profile in ``home`` whose slot is free.

    Returns the number of turns actually run. A slot held by any other turn (a
    desktop session, a card worker) is skipped, never fought for -- the holder
    drains the queue when it releases.
    """
    from tools.bot_mode_probe import _hermes_root
    from tools.bot_relay import TurnBusyError, acquire_turn_lock

    root = _hermes_root(home)
    budget = SWEEP_DRAIN_BUDGET_SECONDS if budget_seconds is None else budget_seconds
    deadline = time.monotonic() + budget
    drained = 0
    for profile in delivery_queue.queued_target_profiles(home):
        while time.monotonic() < deadline:
            try:
                with acquire_turn_lock(root, profile, timeout_seconds=0):
                    record = delivery_queue.claim_next(
                        home, target_profile=profile, lease_ok=True
                    )
                    if record is None:
                        break
                    settled, _reply = await run_record(adapter, home, record)
                    drained += 1
                    log_delivery_event("drained", settled, drained=drained)
            except TurnBusyError:
                log_delivery_event("drain_deferred", None, target=profile, reason="slot_held")
                break
    return drained


async def sweep_loop(adapter: Any) -> None:
    """Trigger 3: every ``sweep_seconds``, recover expiries then drain free slots."""
    from tools.bot_mode_probe import _default_home

    home = Path(_default_home())
    logger.info("[api_server] bot delivery drainer started (home=%s)", home)
    while True:
        await asyncio.sleep(delivery_queue.sweep_seconds())
        try:
            # Recover orphaned claims / expire over-age records first, so the
            # drain below sees the real queue (and never expires a busy target's).
            delivery_queue.sweep_delivery_queue(home)
            drained = await drain_once(adapter, home)
            if drained:
                log_delivery_event("sweep_drained", None, drained=drained)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a sweep failure must never kill the loop
            logger.exception("[api_server] bot delivery sweep failed")
