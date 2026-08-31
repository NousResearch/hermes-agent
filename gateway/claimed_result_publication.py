"""Durable per-part publication for claim-owned final responses.

Payloads stay in the existing private raw-result record and in process memory.
Only stable hashes, ordinals, kinds, state, and optional remote receipts cross
into the part ledger.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable, Optional


class ClaimedResultPartDeliveryError(RuntimeError):
    """One claim-owned response part did not receive a successful ACK."""


@dataclass(frozen=True)
class ClaimedResultPart:
    part_id: str
    ordinal: int
    kind: str


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
