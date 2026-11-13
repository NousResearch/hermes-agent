"""Durable Telegram task bubbles, scoped to the existing authorized subscription."""
from __future__ import annotations

import hashlib
import re
import tempfile
import time
from functools import partial
from pathlib import Path

from agent.redact import redact_sensitive_text
from gateway.kanban_watchers_common import _to_thread_process_service, logger


class StatusDeferred(Exception):
    """Keep the subscription and retry cursor without a tight transport retry loop."""


class StatusUncertain(StatusDeferred):
    """Delivery needs management reconciliation, never a blind duplicate send."""


def _text(value):
    text = redact_sensitive_text(str(value or ""), force=True, redact_url_credentials=True)
    return "\n".join(line for line in text.splitlines()
                     if not re.match(r"\s*(pid|session(?:_id)?|last[_ ]output)\s*[:=]", line, re.IGNORECASE)).strip()


def _wire_size(text):
    # A MarkdownV2 escape adds at most one BMP code unit per input code unit.
    # This conservative bound covers both raw UTF-16 and formatted payloads,
    # without importing a platform plugin across the core/plugin boundary.
    return 2 * len(text.encode("utf-16-le")) // 2


def _section(value, budget):
    """Keep complete sentences; overflow is delivered as a native document."""
    text = _text(value)
    if _wire_size(text) <= budget:
        return text
    kept = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        if _wire_size(" ".join(kept + [part])) > budget:
            break
        kept.append(part)
    return " ".join(kept) or "Full recorded text is in the status details attachment."


def render_status(n, ev, *, full=False):
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
    section = (lambda value, budget: _text(value)) if full else _section
    lines = [f"{status} · {n.task_id}", section(n.title, 200)]
    if kind in {"completed", "review_requested"}:
        summary = payload.get("summary") or getattr(n.task, "result", "") or "No outcome summary recorded."
        outcome = section(summary, 1200)
        if not full and outcome == "Full recorded text is in the status details attachment.":
            outcome = f"{status}. Full recorded outcome is in the status details attachment."
        lines = [outcome, *lines]
        meta = payload.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        work = meta.get("completed_work") or meta.get("changed_files")
        if work:
            lines.append("Completed work: " + section(work, 350))
        if meta.get("tests_run"):
            lines.append("Verification: " + section(meta["tests_run"], 350))
        limits = meta.get("verification_limits") or meta.get("unverified")
        lines.append("Verification limits: " + (section(limits, 650) if limits else
                     "No additional verification evidence recorded here; completion is the worker's report."))
    elif payload.get("reason"):
        lines.append(section(payload["reason"], 1000))
    if kind in {"blocked", "block_loop_detected"} and payload.get("kind") == "needs_input":
        lines.append("Decision needed: " + section(payload.get("reason") or "See task for the requested decision.", 500))
    return "\n\n".join(filter(None, lines))


async def _details(n, ev, metadata, state, save, full, content):
    """One native text document per distinct overflow, reserved before I/O."""
    if full == content:
        return content
    digest = hashlib.sha256(full.encode()).hexdigest()
    if state.get("task_status_details_" + digest) or state.get("task_status_details_hash") == digest:
        return content + "\n\nFull status details attached."
    if state.get("task_status_details_pending"):
        raise StatusUncertain("status details delivery requires reconciliation")
    await save(task_status_details_pending=digest)
    with tempfile.TemporaryDirectory(prefix="kanban-status-") as directory:
        path = Path(directory) / f"{n.task_id}-status.txt"
        path.write_text(full, encoding="utf-8")
        try:
            result = await n.adapter.send_document(n.sub["chat_id"], str(path),
                                                   caption=f"Full status details · {n.task_id}", metadata=metadata)
        except Exception:
            raise StatusUncertain("status details delivery outcome unknown") from None
    if not getattr(result, "success", False):
        if getattr(result, "retryable", False):
            await save(task_status_details_pending="", task_status_retry_at=time.time() + 60)
            raise StatusDeferred("details delivery deferred")
        raise StatusUncertain("status details delivery outcome unknown")
    await save(task_status_details_hash=digest, task_status_details_pending="", **{"task_status_details_" + digest: True})
    return content + "\n\nFull status details attached."


async def send_status(n, ev, metadata):
    """Edit durably; ambiguous sends escalate instead of looping or hiding decisions."""
    state = dict(n.sub.get("delivery_metadata") or {})
    now = time.time()
    if int(state.get("task_status_event", 0)) >= ev.id:
        return
    if state.get("task_status_send_pending"):
        raise StatusUncertain("ambiguous status send requires reconciliation")
    if float(state.get("task_status_retry_at", 0)) > now:
        raise StatusDeferred("status transport is in backoff")

    async def save(**changes):
        state.update(changes)
        await _to_thread_process_service(partial(
            n.runner._kanban_sub_op, n.board_slug, "record_notify_status", n.sub, state=state))
        n.sub["delivery_metadata"] = dict(state)

    content = await _details(n, ev, metadata, state, save, render_status(n, ev, full=True), render_status(n, ev))
    # Whole-message guard, including fixed labels, identifiers and attachment notice.
    # It uses the actual adapter's formatting, not the estimate used for allocation.
    formatted = n.adapter.format_message(content) if callable(getattr(n.adapter, "format_message", None)) else content
    if max(len(s.encode("utf-16-le")) // 2 for s in (content, formatted)) > 3900:
        raise RuntimeError("status exceeds transport budget; refusing continuation sends")
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
        raise StatusUncertain("status send outcome unknown") from None
    if getattr(result, "success", True) is False:
        if getattr(result, "retryable", False):
            await save(task_status_send_pending=False, task_status_retry_at=now + 60)
            raise StatusDeferred("status send deferred")
        raise StatusUncertain("status send outcome unknown")
    new_id = getattr(result, "message_id", None)
    if not new_id:
        raise StatusUncertain("status message ID unavailable")
    await save(task_status_message_id=str(new_id), task_status_event=ev.id,
               task_status_send_pending=False, task_status_retry_at=0, task_status_failures=0)
