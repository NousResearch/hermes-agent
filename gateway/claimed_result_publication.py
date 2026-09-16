"""Durable per-part publication for claim-owned final responses.

Payloads stay in the existing private raw-result record and in process memory.
Only stable hashes, ordinals, kinds, state, and optional remote receipts cross
into the part ledger.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional


class ClaimedResultPartDeliveryError(RuntimeError):
    """One claim-owned response part did not receive a successful ACK."""


@dataclass(frozen=True)
class ClaimedResultPart:
    part_id: str
    ordinal: int
    kind: str


@dataclass(frozen=True)
class ClaimedResponsePartsSnapshot:
    """One bounded parse of every claimed-result publication part."""

    visible_text: str
    images: tuple[tuple[str, str], ...]
    media_files: tuple[tuple[str, bool], ...]
    local_files: tuple[str, ...]
    force_document_attachments: bool


def _accepts_keyword(function: Any, keyword: str) -> bool:
    """Return whether a callable explicitly accepts a snapshot keyword."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def snapshot_claimed_response_parts(
    response: str,
    adapter: Any,
) -> ClaimedResponsePartsSnapshot:
    """Derive visible text and attachment intents from one parse snapshot."""
    from gateway.platforms.base import BasePlatformAdapter, _strip_media_directives

    media_extractor = getattr(adapter, "extract_media", None)
    if not callable(media_extractor) or not _accepts_keyword(
        media_extractor,
        "include_unavailable",
    ):
        media_extractor = BasePlatformAdapter.extract_media
    media_files, cleaned = media_extractor(
        response,
        include_unavailable=True,
    )
    image_extractor = getattr(adapter, "extract_images", None)
    if not callable(image_extractor):
        image_extractor = BasePlatformAdapter.extract_images
    images, text_content = image_extractor(cleaned)
    text_content = _strip_media_directives(text_content).strip()
    local_file_extractor = getattr(adapter, "extract_local_files", None)
    if not callable(local_file_extractor) or not _accepts_keyword(
        local_file_extractor,
        "include_unavailable",
    ):
        local_file_extractor = BasePlatformAdapter.extract_local_files
    local_files, text_content = local_file_extractor(
        text_content,
        include_unavailable=True,
    )
    return ClaimedResponsePartsSnapshot(
        visible_text=text_content.strip(),
        images=tuple(images),
        media_files=tuple(media_files),
        local_files=tuple(local_files),
        force_document_attachments="[[as_document]]" in response,
    )


def plan_claimed_result_parts(
    obligation_id: str,
    entries: Iterable[tuple[str, str]],
) -> list[ClaimedResultPart]:
    """Build stable part identities without retaining private payload values."""
    from gateway.delivery_ledger import compute_claimed_result_part_id

    parts: list[ClaimedResultPart] = []
    for ordinal, (kind, payload_identity) in enumerate(entries):
        parts.append(
            ClaimedResultPart(
                part_id=compute_claimed_result_part_id(
                    obligation_id,
                    ordinal,
                    kind,
                    payload_identity,
                ),
                ordinal=ordinal,
                kind=kind,
            )
        )
    return parts


async def register_claimed_result_parts(
    obligation_id: str,
    parts: list[ClaimedResultPart],
) -> None:
    from gateway.delivery_ledger import register_claimed_result_parts as _register

    await asyncio.to_thread(
        _register,
        obligation_id,
        [(part.part_id, part.kind) for part in parts],
    )


async def deliver_claimed_result_part(
    obligation_id: str,
    part: ClaimedResultPart,
    send: Callable[[], Awaitable[Any]],
    *,
    failure_code: str = "platform_part_delivery_failed",
) -> Optional[Any]:
    """Checkpoint, send, and acknowledge one response part.

    ``None`` means this exact part was already acknowledged and must not be
    replayed. Ordinary exceptions become a payload-free durable failure;
    cancellation and process-level BaseExceptions leave the part ``attempting``
    so crash recovery preserves ambiguity.
    """
    from gateway.delivery_ledger import (
        mark_claimed_result_part_delivered,
        mark_claimed_result_part_failed,
        prepare_claimed_result_part,
    )

    should_send = await asyncio.to_thread(
        prepare_claimed_result_part,
        obligation_id,
        part.part_id,
    )
    if not should_send:
        return None
    try:
        result = await send()
    except BaseException as exc:
        if isinstance(exc, Exception):
            try:
                await asyncio.to_thread(
                    mark_claimed_result_part_failed,
                    obligation_id,
                    part.part_id,
                    failure_code,
                )
            except Exception:
                pass
            raise ClaimedResultPartDeliveryError(
                "claimed continuation attachment delivery failed"
            ) from exc
        raise

    if result is None or not getattr(result, "success", False):
        try:
            await asyncio.to_thread(
                mark_claimed_result_part_failed,
                obligation_id,
                part.part_id,
                failure_code,
            )
        except Exception:
            pass
        raise ClaimedResultPartDeliveryError(
            "claimed continuation attachment delivery failed"
        )

    acknowledged = await asyncio.to_thread(
        mark_claimed_result_part_delivered,
        obligation_id,
        part.part_id,
        remote_receipt=getattr(result, "message_id", None),
    )
    if not acknowledged:
        raise ClaimedResultPartDeliveryError(
            "claimed continuation publication ownership changed"
        )
    return result
