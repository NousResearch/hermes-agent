"""``gateway_message_delivered`` observer hook (#64176, delivery half).

One normalized notice per successful outbound delivery the gateway itself decided is complete.
Fired at gateway-level delivery sites (never inside a platform adapter), so every adapter is
covered by the same contract; the ids are whatever the adapter reported in its ``SendResult``.

v1 fire sites (see hooks.md for the per-kind coverage table):

* ``kind="final"``  — the agent's final reply to a user turn: the non-streamed ledgered send
  (``BasePlatformAdapter.send_final_ledgered``) and the streamed final once the runner confirms
  the stream already delivered it (``GatewayRunner._run_agent_mark_streamed_delivery``).
* ``kind="cron"``   — cron job output: the live-adapter lane and the standalone lane of
  ``cron/scheduler_delivery.py``.

Observer-only: return values are ignored and callbacks are isolated. Dispatch happens AFTER the send
succeeded and never on the delivery path: loop-side sites start a background task, the cron thread
schedules onto the gateway loop, and the hook is in the plugin dispatcher's timeout-bounded set. A
failing or slow callback therefore can neither fail nor delay a delivery. Payload fields are plain strings/lists (no SDK objects, no
adapter handles). To act on a delivered message use the capability-gated
``ctx.platform_actions`` facade.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

HOOK_NAME = "gateway_message_delivered"
KINDS = ("final", "cron")
_ID_MAX = 128
_TEXT_MAX = 8192


def _id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "__no_edit__":  # stream-consumer sentinel, not a platform id
        return None
    return text[:_ID_MAX]


def _field(result: Any, name: str) -> Any:
    if isinstance(result, dict):
        return result.get(name)
    return getattr(result, name, None)


def delivered_message_ids(result: Any) -> List[str]:
    """Every message id a send produced, in send order, from any adapter's result shape.

    Adapters disagree on ``SendResult.message_id`` for split payloads (the base contract says it is
    the LAST id; Telegram's ``send`` reports the first and lists all ids in
    ``raw_response["message_ids"]``). Prefer the explicit ordered list, else
    ``continuation_message_ids`` + ``message_id``; duplicates and sentinels are dropped.
    """
    ordered: List[Any] = []
    raw = _field(result, "raw_response")
    raw_ids = raw.get("message_ids") if isinstance(raw, dict) else None
    if isinstance(raw_ids, (list, tuple)) and raw_ids:
        ordered.extend(raw_ids)
    else:
        continuation = _field(result, "continuation_message_ids") or ()
        if isinstance(continuation, (list, tuple)):
            ordered.extend(continuation)
        ordered.append(_field(result, "message_id"))
    ids: List[str] = []
    for value in ordered:
        normalized = _id(value)
        if normalized is not None and normalized not in ids:
            ids.append(normalized)
    return ids


def _active_profile() -> Optional[str]:
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or None
    except Exception:
        return None


def build_payload(
    *,
    kind: str,
    platform: Any,
    chat_id: Any,
    thread_id: Any = None,
    message_ids: Optional[Iterable[Any]] = None,
    text: Optional[str] = None,
    session_key: Optional[str] = None,
    job_id: Optional[str] = None,
    streamed: bool = False,
) -> Dict[str, Any]:
    """The v1 keyword payload. All ids are bounded strings; unknown values are ``None``."""
    ids: List[str] = []
    for value in message_ids or ():
        normalized = _id(value)
        if normalized is not None and normalized not in ids:
            ids.append(normalized)
    platform_id = getattr(platform, "value", platform)
    return {
        "kind": kind,
        "platform": str(platform_id).strip().lower() if platform_id is not None else None,
        "chat_id": _id(chat_id),
        "thread_id": _id(thread_id),
        "message_ids": ids,
        "last_message_id": ids[-1] if ids else None,
        "text": text[:_TEXT_MAX] if isinstance(text, str) else None,
        "session_key": session_key or None,
        "job_id": _id(job_id),
        "streamed": bool(streamed),
        "profile": _active_profile(),
    }


def _has_subscribers() -> bool:
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(HOOK_NAME)
    except Exception:
        return False


# Strong references to in-flight loop-side dispatches (a bare create_task result may be GC'd mid-run).
_PENDING: "set[asyncio.Task]" = set()


async def _dispatch(payload: Dict[str, Any]) -> None:
    try:
        from hermes_cli.lifecycle import ainvoke_hook

        await ainvoke_hook(HOOK_NAME, **payload)
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.debug("%s dispatch failed", HOOK_NAME, exc_info=True)


def notify_message_delivered(**fields: Any) -> None:
    """For callers ON the gateway loop: build the payload now (inside the caller's profile scope) and
    dispatch as a background task, so delivery never waits on an observer. The task inherits the
    caller's context (profile scope) as ``create_task`` copies it. Never raises."""
    try:
        if not _has_subscribers():
            return
        task = asyncio.get_running_loop().create_task(_dispatch(build_payload(**fields)))
        _PENDING.add(task)
        task.add_done_callback(_PENDING.discard)
    except Exception:
        logger.debug("%s scheduling failed", HOOK_NAME, exc_info=True)


def notify_message_delivered_from_thread(loop: Any, **fields: Any) -> None:
    """For callers OFF the loop (the cron ticker thread): schedule onto the running gateway ``loop``
    — async callbacks then share the loop that owns the adapters' clients — and do not wait. The
    calling thread's context (profile scope) travels with the scheduled callback. Without a running
    loop (``hermes cron run`` in a CLI process) dispatch synchronously. Never raises."""
    try:
        if not _has_subscribers():
            return
        payload = build_payload(**fields)
        if loop is not None and getattr(loop, "is_running", lambda: False)():
            from agent.async_utils import safe_schedule_threadsafe

            # A scheduling failure is logged by the helper; never fall back to a private loop here.
            safe_schedule_threadsafe(_dispatch(payload), loop, logger=logger,
                                     log_message=f"Failed to schedule {HOOK_NAME}")
            return
        from hermes_cli.lifecycle import invoke_hook

        invoke_hook(HOOK_NAME, **payload)
    except Exception:
        logger.debug("%s dispatch failed", HOOK_NAME, exc_info=True)


async def wait_for_pending_notifications(timeout: Optional[float] = None) -> None:
    """Await in-flight loop-side dispatches (tests, orderly shutdown)."""
    if _PENDING:
        await asyncio.wait(list(_PENDING), timeout=timeout)
