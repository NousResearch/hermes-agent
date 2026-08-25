"""Attachment readiness outcomes for BlueBubbles webhook revisions."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Sequence


class AttachmentReadiness(str, Enum):
    READY = "ready"
    PENDING = "pending"
    TERMINAL_FAILURE = "terminal_failure"


@dataclass(frozen=True)
class MaterializedAttachments:
    readiness: AttachmentReadiness
    paths: tuple[str, ...] = ()
    mime_types: tuple[str, ...] = ()
    failed_guid: str | None = None


DownloadAttachment = Callable[[str, dict[str, Any]], Awaitable[str | None]]
logger = logging.getLogger(__name__)


def metadata_readiness(attachment: Mapping[str, Any]) -> AttachmentReadiness:
    """Classify provider metadata before attempting to read attachment bytes.

    BlueBubbles documents transferState=5 as complete. Notification webhooks may
    omit transferState, so an absent state remains eligible for an immediate
    materialization attempt. Explicit provider errors are terminal; incomplete
    transfer states are pending rather than failures.
    """
    if attachment.get("isCorrupt") is True:
        return AttachmentReadiness.TERMINAL_FAILURE
    error = attachment.get("error")
    if error not in (None, 0, "", False):
        return AttachmentReadiness.TERMINAL_FAILURE
    transfer_state = attachment.get("transferState")
    if transfer_state is not None and transfer_state != 5:
        return AttachmentReadiness.PENDING
    return AttachmentReadiness.READY


async def materialize_attachments(
    attachments: Sequence[Any],
    download: DownloadAttachment,
) -> MaterializedAttachments:
    """Materialize a whole revision atomically; never return partial media."""
    paths: list[str] = []
    mime_types: list[str] = []
    pending_guid: str | None = None

    for raw in attachments:
        if not isinstance(raw, Mapping):
            return MaterializedAttachments(AttachmentReadiness.TERMINAL_FAILURE)
        attachment = dict(raw)
        guid = attachment.get("guid")
        if not isinstance(guid, str) or not guid.strip():
            return MaterializedAttachments(
                AttachmentReadiness.TERMINAL_FAILURE,
                failed_guid=None,
            )
        guid = guid.strip()
        readiness = metadata_readiness(attachment)
        if readiness is AttachmentReadiness.TERMINAL_FAILURE:
            return MaterializedAttachments(readiness, failed_guid=guid)
        if readiness is AttachmentReadiness.PENDING:
            pending_guid = pending_guid or guid
            continue

        cached = await download(guid, attachment)
        if not cached:
            pending_guid = pending_guid or guid
            continue
        paths.append(cached)
        mime_types.append(str(attachment.get("mimeType") or "").lower())

    if pending_guid is not None:
        return MaterializedAttachments(
            AttachmentReadiness.PENDING,
            failed_guid=pending_guid,
        )
    return MaterializedAttachments(
        AttachmentReadiness.READY,
        paths=tuple(paths),
        mime_types=tuple(mime_types),
    )


async def _retry_pending_attachments(
    adapter: Any,
    message_key: tuple[str, str, str],
    revision_serial: int,
    payload: dict[str, Any],
    record: dict[str, Any],
    chat_guid: str,
    attempts: int,
) -> None:
    """Poll the authoritative message and replay only its current attachment set."""
    current_task = asyncio.current_task()
    try:
        for attempt in range(attempts):
            delay = adapter._attachment_retry_delay_seconds * min(2**attempt, 8)
            await asyncio.sleep(delay)
            async with adapter._inbound_dedup_lock:
                if (
                    adapter._pending_attachment_tasks.get(message_key) is not current_task
                    or adapter._message_revision_serials.get(message_key) != revision_serial
                ):
                    return
            try:
                refreshed = await adapter._query_exact_message(
                    chat_guid,
                    message_key[2],
                    include_attachments=True,
                )
            except Exception:
                continue
            if refreshed is None:
                continue
            current_attachments = refreshed.get("attachments") or []
            readiness = [
                metadata_readiness(attachment)
                for attachment in current_attachments
                if isinstance(attachment, Mapping)
            ]
            if any(
                state is AttachmentReadiness.TERMINAL_FAILURE
                for state in readiness
            ):
                logger.warning("BlueBubbles attachment materialization failed terminally")
                return
            if any(state is AttachmentReadiness.PENDING for state in readiness):
                continue

            refreshed_record = {**record, **refreshed}
            # The exact lookup is authoritative for removals as well as
            # additions; never merge stale webhook attachments back in.
            refreshed_record["attachments"] = list(current_attachments)
            refreshed_payload = dict(payload)
            if isinstance(refreshed_payload.get("data"), dict):
                refreshed_payload["data"] = refreshed_record
            elif isinstance(refreshed_payload.get("message"), dict):
                refreshed_payload["message"] = refreshed_record
            else:
                refreshed_payload.update(refreshed_record)
            await adapter._handle_webhook(
                None,
                _trusted_payload=refreshed_payload,
                _authoritative_chat_guid=chat_guid,
            )
            return
        logger.warning("BlueBubbles attachment materialization retries exhausted")
    finally:
        async with adapter._inbound_dedup_lock:
            if adapter._pending_attachment_tasks.get(message_key) is current_task:
                adapter._pending_attachment_tasks.pop(message_key, None)


async def schedule_pending_attachment_retry(
    adapter: Any,
    message_key: tuple[str, str, str],
    revision_serial: int,
    payload: dict[str, Any],
    record: dict[str, Any],
    chat_guid: str,
    *,
    attempts: int,
) -> None:
    """Keep exactly one current attachment-readiness worker per logical message."""
    async with adapter._inbound_dedup_lock:
        prior = adapter._pending_attachment_tasks.get(message_key)
        if prior and prior is not asyncio.current_task() and not prior.done():
            prior.cancel()
        task = asyncio.create_task(
            _retry_pending_attachments(
                adapter,
                message_key,
                revision_serial,
                dict(payload),
                dict(record),
                chat_guid,
                attempts,
            )
        )
        adapter._pending_attachment_tasks[message_key] = task
        adapter._background_tasks.add(task)
        task.add_done_callback(adapter._background_tasks.discard)
