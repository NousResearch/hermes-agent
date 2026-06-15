"""Startup reconciliation for Discord protocol-v2 outbox deliveries.

The reconciler is intentionally transport-agnostic: tests/native adapters inject a
recent-history fetcher, and the reconciler only decides whether an old
leased/sending/uncertain outbox row was already committed or should be made
pending for the normal outbox sender.
"""

from __future__ import annotations

import inspect
import json
import uuid
from hashlib import sha256
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from gateway.discord_protocol_v2_outbox import (
    _decode_payload,
    _extract_content,
    record_outbox_message_map,
    split_discord_message,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store

RecentHistoryFetcher = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class DiscordProtocolV2ReconciliationResult:
    """Summary of one startup reconciliation pass."""

    run_id: str
    scanned: int = 0
    acked: int = 0
    enqueued: int = 0
    exhausted: int = 0
    ignored: int = 0


async def reconcile_discord_protocol_v2_outbox(
    *,
    store: DiscordProtocolV2Store,
    recent_history_fetcher: RecentHistoryFetcher | None = None,
    max_attempts: int = 3,
    run_id: str | None = None,
) -> DiscordProtocolV2ReconciliationResult:
    """Reconcile stale outbox rows after a gateway restart.

    Rows in ``leased`` with an expired lease, plus crash-sticky ``sending`` and
    ``uncertain`` rows, are inspected.  Evidence from existing ``message_map`` /
    ``outbox_parts`` or injected recent Discord history is enough to mark the
    delivery ``acked`` when every part is accounted for.  Partial evidence is
    persisted before rows with remaining retry budget are reset to ``pending``
    so the regular outbox sender sends only missing parts.
    """

    run_id = run_id or f"reconcile:{uuid.uuid4().hex}"
    candidates = _list_reconciliation_candidates(store)
    acked = enqueued = exhausted = ignored = 0

    for outbox in candidates:
        status = "ignored"
        payload: dict[str, Any] = {}
        details: dict[str, Any] = {}
        try:
            payload = _decode_payload(outbox.get("payload_json"))
            chunks = split_discord_message(_extract_content(payload))
            if not chunks:
                store.mark_outbox_acked(str(outbox["outbox_delivery_id"]))
                status = "acked_empty"
                acked += 1
                continue

            mapped = _mapped_parts(store, outbox, chunks)
            if _has_all_parts(mapped, chunks):
                _persist_reconciled_parts(store, outbox, payload, chunks, mapped)
                store.mark_outbox_acked(str(outbox["outbox_delivery_id"]))
                status = "acked_message_map"
                acked += 1
                continue

            history = []
            if recent_history_fetcher is not None:
                history = list(await _maybe_await(recent_history_fetcher(dict(outbox))))
            found = dict(mapped)
            for part_index, evidence in _match_history_parts(store, outbox, chunks, history).items():
                found.setdefault(part_index, evidence)
            if _has_all_parts(found, chunks):
                _persist_reconciled_parts(store, outbox, payload, chunks, found)
                store.mark_outbox_acked(str(outbox["outbox_delivery_id"]))
                status = "acked_discord_history"
                acked += 1
                continue

            if found:
                _persist_reconciled_parts(store, outbox, payload, chunks, found)
                details["reconciled_part_indexes"] = sorted(found)
                details["reconciled_part_count"] = len(found)
                details["total_part_count"] = len(chunks)

            if int(outbox.get("attempts") or 0) < int(max_attempts):
                _reset_outbox_to_pending(store, str(outbox["outbox_delivery_id"]))
                status = "retry_enqueued"
                enqueued += 1
            else:
                _mark_outbox_uncertain_if_needed(store, outbox)
                status = "retry_exhausted"
                exhausted += 1
        except Exception as exc:  # pragma: no cover - defensive audit path
            _mark_outbox_uncertain_if_needed(store, outbox)
            status = "error"
            exhausted += 1
            details["error"] = type(exc).__name__
        finally:
            if status == "ignored":
                ignored += 1
            details.update(
                {
                    "run_id": run_id,
                    "previous_status": outbox.get("status"),
                    "attempts": outbox.get("attempts"),
                    "payload_content_sha256": _content_sha256(payload),
                }
            )
            store.create_reconciliation_run(
                reconciliation_run_id=f"{run_id}:{outbox['outbox_delivery_id']}",
                source_agent_event_id=outbox.get("source_agent_event_id"),
                outbox_delivery_id=str(outbox["outbox_delivery_id"]),
                status=status,
                payload=details,
            )

    return DiscordProtocolV2ReconciliationResult(
        run_id=run_id,
        scanned=len(candidates),
        acked=acked,
        enqueued=enqueued,
        exhausted=exhausted,
        ignored=ignored,
    )


def _list_reconciliation_candidates(store: DiscordProtocolV2Store) -> list[dict[str, Any]]:
    now = _now()
    rows = store.conn.execute(
        """
        SELECT * FROM outbox_deliveries
        WHERE (status = 'leased' AND (lease_until IS NULL OR lease_until <= ?))
           OR status IN ('sending', 'uncertain')
        ORDER BY created_at, outbox_delivery_id
        """,
        (now,),
    ).fetchall()
    return [dict(row) for row in rows]


def _mapped_parts(
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
    chunks: Sequence[str],
) -> dict[int, dict[str, Any]]:
    outbox_id = str(outbox["outbox_delivery_id"])
    found: dict[int, dict[str, Any]] = {}
    part_rows = store.conn.execute(
        """
        SELECT * FROM outbox_parts
        WHERE outbox_delivery_id = ? AND discord_message_id IS NOT NULL
        ORDER BY part_index
        """,
        (outbox_id,),
    ).fetchall()
    for row in part_rows:
        part = dict(row)
        found[int(part["part_index"])] = {
            "discord_message_id": str(part["discord_message_id"]),
            "content": chunks[int(part["part_index"])] if int(part["part_index"]) < len(chunks) else "",
            "source_client_agent_id": str(outbox["target_agent_id"]),
            "author_bot_user_id": _identity_bot_user_id(store, str(outbox["target_agent_id"])),
        }

    map_rows = store.conn.execute(
        """
        SELECT * FROM message_map
        WHERE outbox_delivery_id = ?
        ORDER BY created_at, discord_message_id
        """,
        (outbox_id,),
    ).fetchall()
    for row in map_rows:
        mapped = dict(row)
        part_index = _part_index_from_message_map(mapped, default=0 if len(chunks) == 1 else None)
        if part_index is None or part_index in found or part_index >= len(chunks):
            continue
        found[part_index] = {
            "discord_message_id": str(mapped["discord_message_id"]),
            "content": chunks[part_index],
            "source_client_agent_id": mapped.get("source_client_agent_id")
            or str(outbox["target_agent_id"]),
            "author_bot_user_id": mapped.get("author_bot_user_id")
            or _identity_bot_user_id(store, str(outbox["target_agent_id"])),
        }
    return found


def _match_history_parts(
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
    chunks: Sequence[str],
    history: Iterable[Any],
) -> dict[int, dict[str, Any]]:
    remaining = list(history)
    found: dict[int, dict[str, Any]] = {}
    for part_index, chunk in enumerate(chunks):
        for message in list(remaining):
            if not _history_message_matches(store, outbox, chunk, message):
                continue
            found[part_index] = {
                "discord_message_id": str(_history_value(message, "discord_message_id") or _history_value(message, "id")),
                "content": str(_history_value(message, "content") or ""),
                "source_client_agent_id": str(
                    _history_value(message, "source_client_agent_id") or outbox["target_agent_id"]
                ),
                "author_bot_user_id": _history_value(message, "author_bot_user_id")
                or _identity_bot_user_id(store, str(outbox["target_agent_id"])),
            }
            remaining.remove(message)
            break
    return found


def _history_message_matches(
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
    chunk: str,
    message: Any,
) -> bool:
    message_id = _history_value(message, "discord_message_id") or _history_value(message, "id")
    if not message_id:
        return False
    if str(_history_value(message, "content") or "") != chunk:
        return False
    channel_id = _history_value(message, "channel_id")
    if channel_id is not None and str(channel_id) != str(outbox["channel_id"]):
        return False
    thread_id = _history_value(message, "thread_id")
    if thread_id is not None and str(thread_id) != str(outbox.get("thread_id")):
        return False

    target_agent_id = str(outbox["target_agent_id"])
    bot_user_id = _identity_bot_user_id(store, target_agent_id)
    author_values = {
        str(value)
        for value in (
            _history_value(message, "author_id"),
            _history_value(message, "author_bot_user_id"),
            _history_value(message, "source_client_agent_id"),
        )
        if value is not None
    }
    return not author_values or target_agent_id in author_values or bool(
        bot_user_id and bot_user_id in author_values
    )


def _persist_reconciled_parts(
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
    payload: dict[str, Any],
    chunks: Sequence[str],
    found: Mapping[int, Mapping[str, Any]],
) -> None:
    outbox_id = str(outbox["outbox_delivery_id"])
    for part_index in sorted(found):
        if part_index < 0 or part_index >= len(chunks):
            continue
        chunk = chunks[part_index]
        evidence = found[part_index]
        discord_message_id = str(evidence["discord_message_id"])
        store.add_outbox_part(
            outbox_delivery_id=outbox_id,
            part_index=part_index,
            status="sent",
            discord_message_id=discord_message_id,
        )
        record_outbox_message_map(
            store=store,
            outbox=outbox,
            payload=payload,
            part_index=part_index,
            content=str(evidence.get("content") or chunk),
            discord_message_id=discord_message_id,
            source_client_agent_id=str(
                evidence.get("source_client_agent_id") or outbox["target_agent_id"]
            ),
            author_bot_user_id=evidence.get("author_bot_user_id"),
        )


def _has_all_parts(found: Mapping[int, Mapping[str, Any]], chunks: Sequence[str]) -> bool:
    return bool(chunks) and all(index in found for index in range(len(chunks)))


def _reset_outbox_to_pending(store: DiscordProtocolV2Store, outbox_delivery_id: str) -> None:
    now = _now()
    with store.conn:
        store.conn.execute(
            """
            UPDATE outbox_deliveries
            SET status = 'pending', lease_owner = NULL, lease_until = NULL,
                updated_at = ?, state_version = state_version + 1
            WHERE outbox_delivery_id = ? AND status IN ('leased', 'sending', 'uncertain')
            """,
            (now, outbox_delivery_id),
        )


def _mark_outbox_uncertain_if_needed(
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
) -> None:
    if outbox.get("status") in {"leased", "sending"}:
        store.mark_outbox_uncertain(str(outbox["outbox_delivery_id"]))


def _part_index_from_message_map(row: Mapping[str, Any], default: int | None = None) -> int | None:
    try:
        payload = json.loads(str(row.get("payload_json") or "{}"))
    except json.JSONDecodeError:
        payload = {}
    value = payload.get("part_index") if isinstance(payload, dict) else None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default


def _history_value(message: Any, key: str) -> Any:
    if isinstance(message, Mapping):
        return message.get(key)
    return getattr(message, key, None)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _identity_bot_user_id(store: DiscordProtocolV2Store, target_agent_id: str) -> str | None:
    identity = store.get_identity(target_agent_id)
    if identity is None:
        return None
    return identity.get("discord_bot_user_id")


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _content_sha256(payload: Mapping[str, Any]) -> str | None:
    content = _extract_content(dict(payload)) if payload else ""
    if not content:
        return None
    return sha256(content.encode("utf-8")).hexdigest()
