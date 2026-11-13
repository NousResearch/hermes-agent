"""Cron's standalone delivery lane: the out-of-process send for a target the live adapter did not
deliver, and the delivery-ledger hand-off for a payload both lanes failed (#125363, #127187).

Split out of ``hermes_agent.cron.scheduler_delivery``, which imports these names and remains the seam: its
``_deliver_standalone`` reads them as module globals, and this module reads the facade's
``_get_standalone_send_timeout`` late-bound, so patching ``hermes_agent.cron.scheduler_delivery`` still works.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_agent.cron.scheduler_delivery import _TargetDelivery

logger = logging.getLogger("hermes_agent.cron.scheduler")

# Live-lane rejections an adapter returns BEFORE sending anything (no client while it rebuilds): the
# only ones whose ``retryable`` proves the chat has nothing yet. ``SendResult.retryable`` alone means
# "can try again" and is also set after a failed POST that may have landed (QQBot ``_send_chunk``,
# Teams, Slack), so it never licenses a full replay by itself.
_PRE_SEND_LIVE_REJECTIONS = frozenset({"Not connected"})


def _standalone_send(
    t: _TargetDelivery, content: str, media_files: list) -> tuple[Any, Optional[str]]:
    """Run the standalone sender for one target: ``(result, None)`` or ``(None, error)`` (already
    logged — WARNING for a shutdown race, ERROR with traceback otherwise). A timeout also sets
    ``t.standalone_in_flight``: the dispatch shield keeps a timed-out send un-cancelled, so the
    payload may STILL be delivered and the caller must not queue it for redelivery (a replay
    would duplicate it)."""
    from tools.send_message_tool import _send_to_platform
    job = t.job
    shutdown_msg = f"delivery to {t.where} skipped — interpreter is shutting down"

    send_timeout = _script._media_scaled_timeout(base_timeout := _delivery._get_standalone_send_timeout(), media_files)

    async def _send():
        # The bound lives inside the coroutine: the running-loop fallback below closes ``coro``
        # unstarted, and a wait_for wrapper created out here would be left never awaited.
        # notify: the live lane's metadata["notify"] (cron.delivery.notify), so a brief rings either way.
        return await asyncio.wait_for(_send_to_platform(
            t.platform, t.pconfig, t.chat_id, content, thread_id=t.thread_id,
            media_files=media_files, notify=t.notify_delivery, force_document=t.force_document),
            timeout=send_timeout)

    def _warned(msg: str) -> tuple[None, str]:
        logger.warning("Job '%s': %s", job["id"], msg)
        return None, msg

    def _failed(e) -> tuple[None, str]:
        msg = f"delivery to {t.where} failed: {e}"
        logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
        # An exception escaping the sender says nothing about what already went out, unless it
        # proves the request never left.
        from hermes_agent.gateway.platforms._shared import request_never_left
        t.standalone_in_flight = t.standalone_in_flight or not request_never_left(e)
        return None, msg

    def _in_flight(seconds: float) -> tuple[None, str]:
        # The send may still complete on the gateway loop (the dispatch shield keeps an in-flight
        # send un-cancelled); the run is released instead of waiting on it unbounded (#115469).
        msg = (f"standalone send to {t.where} timed out after {seconds:g}s "
               "(the send may still be in flight)")
        logger.error("Job '%s': %s", job["id"], msg)
        t.standalone_in_flight = True
        return None, msg

    # Interpreter finalizing (SIGTERM/restart/OOM): asyncio.run and a fresh ThreadPoolExecutor both
    # raise "cannot schedule new futures after interpreter shutdown" — warn, not ERROR traceback.
    if _sched._interpreter_shutting_down():
        return _warned(shutdown_msg)
    # The live lane failed closed on an empty payload; standalone senders don't (Telegram returns
    # success=True for empty content WITHOUT an API call) — a phantom delivery would result.
    if not content.strip() and not media_files:
        return _warned(f"standalone send skipped (empty text and no media) for {t.where}")
    coro = _send()
    try:
        return asyncio.run(coro), None
    except TimeoutError:
        return _in_flight(send_timeout)
    except RuntimeError as run_err:
        # asyncio.run() refuses inside a running loop; close the unstarted coro, retry in a thread.
        coro.close()
        if _sched._interpreter_shutting_down(run_err):
            return _warned(shutdown_msg)
        # The fallback can itself raise (SMTP, result timeout); catch it or remaining targets skip.
        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                # A fresh thread does NOT inherit the profile ContextVars (home override + secret
                # scope); run in the active context or the sender reads the default bot token.
                return pool.submit(contextvars.copy_context().run, asyncio.run, _send()).result(
                    timeout=30 + send_timeout - base_timeout), None
            finally:
                pool.shutdown(wait=False)
        except Exception as e:
            if _sched._interpreter_shutting_down(e):
                return _warned(shutdown_msg)
            # A result timeout leaves the pooled thread's send running: same in-flight ambiguity.
            if isinstance(e, TimeoutError):
                return _in_flight(30 + send_timeout - base_timeout)
            return _failed(e)
    except Exception as e:
        return _failed(e)


def _queue_for_live_reconnect(t: _TargetDelivery, content: str, media_files: list, delivery_errors: list) -> None:
    """Hand a payload the live lane rejected retryably and the standalone lane then definitively
    failed (or provably could not attempt) to the delivery ledger, as a failed row owned by the
    adapter that rejected it: the post-reconnect sweep / retry timer redelivers it (#125363).
    Retryable = the reconnect-only ``send_path_degraded`` string, or a pre-send rejection
    (``_PRE_SEND_LIVE_REJECTIONS``) the adapter itself marked retryable (``SendResult.retryable``
    carried on ``DeliverySendError``) — e.g. Telegram's "Not connected" while the bot client
    rebuilds; the same wording with ``retryable=False`` is a permanent fatal. Only reached when the standalone
    lane proved nothing was sent (a sender dict's explicit ``retryable``, or a failure before the
    sender ran); a timeout, a lost answer or a partial send is NOT queued, so a replay cannot
    duplicate. The ledger carries text only; dropped attachments are reported."""
    try:
        from hermes_agent.gateway.delivery_ledger import (
            compute_obligation_id, is_reconnect_only, ledger_enabled, mark_failed, record_obligation)
        retryable_live_rejection = is_reconnect_only(t.live_error) or (
            t.live_retryable is True and t.live_error in _PRE_SEND_LIVE_REJECTIONS)
        if not retryable_live_rejection or not ledger_enabled():
            return
        session_key = f"cron:{t.platform_name}:{t.chat_id}" + (f":{t.thread_id}" if t.thread_id else "")
        obligation_id = compute_obligation_id(session_key, f"job:{t.job.get('id', '?')}", content)
        record_obligation(
            obligation_id=obligation_id, session_key=session_key, platform=t.platform_name,
            chat_id=str(t.chat_id), thread_id=t.thread_id, content=content,
            adapter_profile=getattr(getattr(t.transport, "adapter", None), "_owner_profile", None))
        mark_failed(obligation_id, str(t.live_error))
        # Arm the in-process drain: this row lands in ``pending_retries`` (backoff tier 0, due in
        # ~30s), but the only arming sites for the flood worker are boot adoption and the
        # agent-reply finalizer — cron enters neither, so without this a retryable-marked row on
        # an adapter that never reconnects waits for the next boot's sweep (#127188 review).
        # Idempotent per (platform, profile): an already-parked worker is just woken to re-scan
        # deadlines in case this row is due sooner. Marshal onto the gateway loop — the cron
        # worker runs on its own thread without one.
        adapter = getattr(t.transport, "adapter", None)
        schedule = getattr(getattr(adapter, "gateway_runner", None), "_schedule_flood_redelivery", None)
        loop = getattr(t, "loop", None)
        if callable(schedule) and loop is not None:
            loop.call_soon_threadsafe(
                functools.partial(
                    schedule, t.platform_name,
                    profile=getattr(adapter, "_owner_profile", None)))
    except Exception:
        logger.warning("Job '%s': could not queue %s for post-reconnect redelivery",
                       t.job.get("id"), t.where, exc_info=True)
        return
    note = f"queued text for {t.where} for redelivery once the live adapter reconnects"
    if media_files:
        note += f" ({len(media_files)} attachment(s) not queued)"
    logger.warning("Job '%s': %s", t.job.get("id"), note)
    delivery_errors.append(note)


# Late-bound (see module docstring): the facade imports this module, so these load after it.
from hermes_agent.cron import scheduler as _sched
from hermes_agent.cron import scheduler_delivery as _delivery
from hermes_agent.cron import scheduler_script as _script
