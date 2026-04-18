"""Shared durable background-job delivery and recovery helpers for the gateway."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from gateway.session import SessionSource


logger = logging.getLogger(__name__)


def _truncate_status_preview(value: Any, *, limit: int = 120) -> str:
    """Return a single-line preview for delivery and approval messages."""
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."


def sanitize_background_visible_text(text: str) -> str:
    cleaned = str(text or "").replace("[[NO_REPLY]]", "").replace("(empty)", "")
    return cleaned.strip()


def background_completion_should_stay_silent(*, job_kind: str, worker_name: str = "") -> bool:
    return job_kind == "auto" or bool(worker_name)


def build_background_delivery_header(
    *,
    task_id: str,
    preview: str = "",
    worker_name: str = "",
    state: str = "completed",
) -> str:
    normalized_state = str(state or "").strip().lower()
    title = {
        "completed": "后台任务完成",
        "failed": "后台任务失败",
        "approval": "后台任务待授权",
    }.get(normalized_state, "后台任务更新")
    icon = {
        "completed": "✅",
        "failed": "❌",
        "approval": "⚠️",
    }.get(normalized_state, "ℹ️")
    lines = [f"{icon} {title} · `{str(task_id or '').strip()}`"]
    if worker_name:
        lines.append(f"负责人：{worker_name}")
    if preview:
        lines.append(f"任务：{preview}")
    return "\n".join(lines)


async def deliver_background_job_updates_once(runner: Any) -> None:
    """Deliver one polling pass of durable job completions and approvals."""
    runner._ensure_background_job_state()
    store = runner._get_background_job_store()
    claimer = f"gateway:{os.getpid()}"

    from tools.approval import build_gateway_approval_message

    approval_requests = store.claim_approval_notifications(
        claimer=claimer,
        limit=20,
        lease_seconds=60,
    )
    for request in approval_requests:
        source_data = request.get("source") or {}
        try:
            source = SessionSource.from_dict(source_data)
        except Exception:
            store.release_approval_notification_claim(int(request["request_id"]))
            continue
        adapter = runner.adapters.get(source.platform)
        if not adapter:
            store.release_approval_notification_claim(int(request["request_id"]))
            continue
        task_id = str(request.get("task_id") or "").strip()
        preview = str(
            request.get("prompt")
            or request.get("preview")
            or (store.get_job(task_id) or {}).get("preview")
            or (store.get_job(task_id) or {}).get("prompt")
            or ""
        ).strip()
        header = build_background_delivery_header(
            task_id=task_id,
            preview=_truncate_status_preview(preview, limit=200),
            state="approval",
        )
        message = build_gateway_approval_message(
            command=str(request.get("command") or ""),
            description=str(request.get("description") or ""),
            prompt_title=str(request.get("prompt_title") or "Dangerous command requires approval"),
            approver_name=str(request.get("approver_name") or "管理员"),
            allow_persistence=bool(request.get("allow_persistence", True)),
        )
        if header:
            message = f"{header}\n\n{message}"
        admin_note = runner._admin_only_message(source, "approve dangerous commands")
        if admin_note:
            message = f"{message}\n\n{admin_note}"
        metadata = {"thread_id": source.thread_id} if source.thread_id else None
        try:
            await adapter.send(
                chat_id=source.chat_id,
                content=message,
                metadata=metadata,
            )
            store.mark_approval_notified(int(request["request_id"]))
        except Exception:
            logger.exception("Failed to deliver background approval request %s", request.get("request_id"))
            store.release_approval_notification_claim(int(request["request_id"]))

    jobs = store.claim_delivery_jobs(
        claimer=claimer,
        limit=20,
        lease_seconds=60,
    )
    for job in jobs:
        task_id = str(job.get("task_id") or "")
        source_data = job.get("source") or {}
        try:
            source = SessionSource.from_dict(source_data)
        except Exception:
            store.release_delivery_claim(task_id, error="invalid source metadata")
            continue
        adapter = runner.adapters.get(source.platform)
        if not adapter:
            store.release_delivery_claim(task_id, error="adapter unavailable")
            continue

        metadata = {"thread_id": source.thread_id} if source.thread_id else None
        preview = str(job.get("preview") or job.get("prompt") or "").strip()
        worker_name = str(job.get("worker_name") or "").strip()
        job_kind = str(job.get("kind") or "manual")
        raw_response = str(job.get("raw_response") or "")
        try:
            if str(job.get("status") or "").strip().lower() == "failed":
                error = str(job.get("error") or "background task failed")
                if job_kind == "btw":
                    message = f"❌ /btw failed: {error}"
                else:
                    header = build_background_delivery_header(
                        task_id=task_id,
                        preview=_truncate_status_preview(preview, limit=200),
                        worker_name=worker_name,
                        state="failed",
                    )
                    message = f"{header}\n错误：{error}"
                await adapter.send(
                    chat_id=source.chat_id,
                    content=message,
                    metadata=metadata,
                )
                store.mark_job_delivered(task_id)
                continue

            media_files, response = adapter.extract_media(raw_response)
            images, text_content = adapter.extract_images(response)
            text_content = sanitize_background_visible_text(text_content)

            if job_kind == "btw":
                header = f'💬 /btw: "{preview}"\n\n'
            else:
                header = build_background_delivery_header(
                    task_id=task_id,
                    preview=_truncate_status_preview(preview, limit=200),
                    worker_name=worker_name,
                    state="completed",
                )

            if text_content:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"{header}\n\n{text_content}",
                    metadata=metadata,
                )
            elif not images and not media_files and not background_completion_should_stay_silent(
                job_kind=job_kind,
                worker_name=worker_name,
            ):
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"{header}\n\n未生成可见结果。",
                    metadata=metadata,
                )

            for image_url, alt_text in (images or []):
                try:
                    await adapter.send_image(
                        chat_id=source.chat_id,
                        image_url=image_url,
                        caption=alt_text,
                    )
                except Exception:
                    logger.debug("Background job image delivery failed for %s", task_id, exc_info=True)

            for media_path in (media_files or []):
                try:
                    await adapter.send_document(
                        chat_id=source.chat_id,
                        file_path=media_path,
                    )
                except Exception:
                    logger.debug("Background job document delivery failed for %s", task_id, exc_info=True)

            store.mark_job_delivered(task_id)
        except Exception as exc:
            logger.exception("Failed to deliver background job %s", task_id)
            store.release_delivery_claim(task_id, error=str(exc))


async def background_job_delivery_poller(runner: Any, interval: float = 2.0) -> None:
    """Background poller for durable job completions and approval prompts."""
    while runner._running:
        try:
            await recover_stale_background_jobs_once(runner)
            await deliver_background_job_updates_once(runner)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Background job delivery poller error")
        await asyncio.sleep(max(float(interval or 0.0), 0.2))


async def recover_stale_background_jobs_once(
    runner: Any,
    *,
    queued_grace_seconds: float = 120.0,
    heartbeat_stale_seconds: float = 120.0,
    now_ts: float | None = None,
) -> list[dict[str, Any]]:
    runner._ensure_background_job_state()
    recovered = runner._get_background_job_store().recover_stale_jobs(
        now_ts=now_ts,
        queued_grace_seconds=queued_grace_seconds,
        heartbeat_stale_seconds=heartbeat_stale_seconds,
    )
    return recovered
