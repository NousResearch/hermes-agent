"""Durable Telegram task bubbles, scoped to the existing authorized subscription."""
from __future__ import annotations

import re
import time
from functools import partial

from agent.redact import redact_sensitive_text
from gateway.kanban_watchers_common import _to_thread_process_service, logger


class StatusDeferred(Exception):
    """Keep the subscription and retry cursor without a tight transport retry loop."""


def _text(value):
    text = redact_sensitive_text(str(value or ""), force=True, redact_url_credentials=True)
    return "\n".join(line for line in text.splitlines()
                     if not re.match(r"\s*(pid|session(?:_id)?|last[_ ]output)\s*[:=]", line, re.IGNORECASE)).strip()


def _section(value, budget):
    """Fit whole sentences/lines, never an arbitrary prefix of the outcome."""
    text = _text(value)
    if len(text) <= budget:
        return text
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = []
    for part in parts:
        if len(" ".join(kept + [part])) > budget - 60:
            break
        kept.append(part)
    return (" ".join(kept) + "\nFull details retained on the task.").strip()


def render_status(n, ev):
    payload = {**(ev.payload or {}), **n.d.get("handoffs", {}).get(ev.id, {})}
    kind = ev.kind
    labels = {"crashed": "Worker stopped; retry pending", "timed_out": "Attempt timed out; retry pending",
              "gave_up": "Automatic retries stopped; management attention needed",
              "unblocked": "Recovery dispatched; success not yet verified",
              "review_requested": "Review requested", "changes_requested": "Changes requested",
              "blocked": "Blocked", "block_loop_detected": "Management attention needed",
              "completed": "Done", "status": str(payload.get("status") or "Status changed")}
    status = labels.get(kind, "Working")
    if kind == "completed" and getattr(n.task, "status", "done") != "done":
        status = "Completion recorded; task is now " + str(n.task.status)
    lines = [f"{status} · {n.task_id}", _section(n.title, 160)]
    if kind in {"completed", "review_requested"}:
        summary = payload.get("summary") or getattr(n.task, "result", "") or "No outcome summary recorded."
        lines = [_section(summary, 1800), *lines]
        meta = payload.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        work = meta.get("completed_work") or meta.get("changed_files")
        if work:
            lines.append("Completed work: " + _section(work, 450))
        limits = meta.get("verification_limits") or meta.get("unverified")
        tests = meta.get("tests_run")
        if tests:
            lines.append("Verification: " + _section(tests, 450))
        lines.append("Verification limits: " + (_section(limits, 650) if limits else
                     "No additional verification evidence recorded here; completion is the worker's report."))
    elif payload.get("reason"):
        lines.append(_section(payload["reason"], 1200))
    if kind in {"blocked", "block_loop_detected"} and payload.get("kind") == "needs_input":
        lines.append("Decision needed: " + _section(payload.get("reason") or "See task for the requested decision.", 500))
    return "\n\n".join(filter(None, lines))


async def send_status(n, ev, metadata):
    """Only known missing/uneditable messages permit a replacement send.

    Transient edit failures retain the ID and back off durably. A replacement
    send is reserved before I/O: an ambiguous result cannot cause a send storm.
    Telegram has no idempotency key, so an interrupted first send cannot offer
    exactly-once delivery; ambiguous replacements require operator reconciliation.
    """
    state = dict(n.sub.get("delivery_metadata") or {})
    now = time.time()
    if int(state.get("task_status_event", 0)) >= ev.id:
        return
    if float(state.get("task_status_retry_at", 0)) > now:
        raise StatusDeferred("status transport is in backoff")
    if state.get("task_status_send_pending"):
        logger.error("kanban status %s: ambiguous send requires reconciliation; retaining subscription", n.task_id)
        raise StatusDeferred("ambiguous status send requires reconciliation")

    async def save(**changes):
        state.update(changes)
        await _to_thread_process_service(partial(
            n.runner._kanban_sub_op, n.board_slug, "record_notify_status", n.sub, state=state))
        n.sub["delivery_metadata"] = dict(state)

    content = render_status(n, ev)
    message_id = state.get("task_status_message_id")
    if message_id:
        try:
            result = await n.adapter.edit_message(n.sub["chat_id"], str(message_id), content,
                                                  finalize=True, metadata=metadata)
        except Exception as exc:
            result = None
            error = str(exc)
        else:
            error = str(getattr(result, "error", "") or "")
        if getattr(result, "success", False) or "message is not modified" in error.lower():
            await save(task_status_event=ev.id, task_status_retry_at=0, task_status_failures=0)
            return
        missing = any(s in error.lower() for s in ("message to edit not found", "message can't be edited", "message can not be edited"))
        if not missing:
            failures = int(state.get("task_status_failures", 0)) + 1
            await save(task_status_failures=failures, task_status_retry_at=now + min(3600, 30 * 2 ** min(failures, 7)))
            logger.warning("kanban status %s: edit failed; durable backoff, message retained", n.task_id)
            raise StatusDeferred("status edit deferred")
    await save(task_status_send_pending=True)
    try:
        result = await n.adapter.send(n.sub["chat_id"], content, metadata=metadata)
    except Exception:
        logger.exception("kanban status %s: send outcome unknown; refusing automatic duplicate", n.task_id)
        raise StatusDeferred("status send outcome unknown") from None
    if getattr(result, "success", True) is False:
        # The adapter reports definite failure; retain pending on ambiguous failures.
        if getattr(result, "retryable", False):
            await save(task_status_send_pending=False, task_status_retry_at=now + 60)
        raise StatusDeferred("status send failed")
    new_id = getattr(result, "message_id", None)
    if not new_id:
        logger.error("kanban status %s: successful send has no editable ID; reconciliation required", n.task_id)
        raise StatusDeferred("status message ID unavailable")
    await save(task_status_message_id=str(new_id), task_status_event=ev.id,
               task_status_send_pending=False, task_status_retry_at=0, task_status_failures=0)
