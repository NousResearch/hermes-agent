from __future__ import annotations

import json
import os
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from gateway.whatsapp_identity import canonical_whatsapp_identifier

SCHEMA_VERSION = 1

_QUERYABLE_RECORD_KINDS = {"message_record", "conversation_record"}

_APPEND_LOCK = threading.Lock()
_SEQUENCE_LOCK = threading.Lock()
_SEQUENCE_COUNTER = 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_whatsapp_event_datetime(value: Any) -> datetime | None:
    if value in {None, ""} or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        raw = float(value)
        if raw <= 0:
            return None
        if raw > 10_000_000_000:
            raw /= 1000.0
        return datetime.fromtimestamp(raw, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return parse_whatsapp_event_datetime(float(text))
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(
                timezone.utc
            )
        except ValueError:
            return None
    return None


def build_whatsapp_destination_fields(chat_id: str) -> dict[str, Any]:
    destination_chat_id = str(chat_id or "")
    if destination_chat_id.endswith("@g.us"):
        group_chat_id = destination_chat_id
        return {
            "destination_key": f"whatsapp:group:{group_chat_id}",
            "destination_context_type": "group_chat",
            "destination_chat_id": destination_chat_id,
            "destination_target_id": group_chat_id,
            "group_chat_id": group_chat_id,
            "dm_counterparty_id": None,
        }

    dm_counterparty_id = canonical_whatsapp_identifier(destination_chat_id)
    return {
        "destination_key": f"whatsapp:dm:{dm_counterparty_id}",
        "destination_context_type": "direct_message",
        "destination_chat_id": destination_chat_id,
        "destination_target_id": dm_counterparty_id,
        "group_chat_id": None,
        "dm_counterparty_id": dm_counterparty_id,
    }


def next_whatsapp_record_sequence(effective_event_at: datetime) -> int:
    global _SEQUENCE_COUNTER
    base = int(effective_event_at.astimezone(timezone.utc).timestamp() * 1_000_000)
    with _SEQUENCE_LOCK:
        _SEQUENCE_COUNTER = (_SEQUENCE_COUNTER + 1) % 1000
        return (base * 1000) + _SEQUENCE_COUNTER


def get_whatsapp_record_store_dir(base_dir: Path | None = None) -> Path:
    return base_dir or (get_hermes_home() / "gateway" / "whatsapp-records")


def whatsapp_daily_partition_path(
    effective_event_at: datetime,
    *,
    base_dir: Path | None = None,
) -> Path:
    store_dir = get_whatsapp_record_store_dir(base_dir)
    day = effective_event_at.astimezone(timezone.utc).date().isoformat()
    return store_dir / f"{day}.jsonl"


def append_whatsapp_record(
    record: dict[str, Any],
    *,
    effective_event_at: datetime,
    base_dir: Path | None = None,
) -> Path:
    path = whatsapp_daily_partition_path(effective_event_at, base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with _APPEND_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
    return path


def _has_destination_scope(
    *,
    conversation_key: str | None,
    destination_key: str | None,
    group_chat_id: str | None,
    dm_counterparty_id: str | None,
) -> bool:
    return any(
        isinstance(value, str) and bool(value.strip())
        for value in (
            conversation_key,
            destination_key,
            group_chat_id,
            dm_counterparty_id,
        )
    )


def _iter_overlapping_partition_dates(
    start_at_utc: datetime, end_at_utc: datetime
) -> list[date]:
    if start_at_utc >= end_at_utc:
        return []

    current_day = start_at_utc.date()
    last_overlap_day = (end_at_utc - timedelta(microseconds=1)).date()
    partition_dates: list[date] = []

    while current_day <= last_overlap_day:
        partition_dates.append(current_day)
        current_day += timedelta(days=1)

    return partition_dates


def _is_queryable_record(record: dict[str, Any]) -> bool:
    record_kind = record.get("record_kind")
    if record_kind is None:
        return True
    return record_kind in _QUERYABLE_RECORD_KINDS


def _record_sequence_value(record: dict[str, Any]) -> int:
    try:
        return int(record.get("record_sequence") or 0)
    except (TypeError, ValueError):
        return 0


def _matches_exact_filter(
    record: dict[str, Any], field_name: str, expected: str | None
) -> bool:
    if expected is None:
        return True
    return record.get(field_name) == expected


def _iter_existing_partition_paths(*, base_dir: Path | None = None) -> list[Path]:
    store_dir = get_whatsapp_record_store_dir(base_dir)
    if not store_dir.exists():
        return []
    return sorted(path for path in store_dir.glob("*.jsonl") if path.is_file())


def _load_filtered_whatsapp_records(
    *,
    partition_paths: list[Path],
    start: datetime | None,
    end: datetime | None,
    conversation_key: str | None,
    destination_key: str | None,
    destination_context_type: str | None,
    group_chat_id: str | None,
    dm_counterparty_id: str | None,
    direction: str | None,
) -> list[dict[str, Any]]:
    results: list[tuple[datetime, int, dict[str, Any]]] = []

    for path in partition_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if not _is_queryable_record(record):
                    continue

                if not _matches_exact_filter(
                    record, "conversation_key", conversation_key
                ):
                    continue
                if not _matches_exact_filter(
                    record, "destination_key", destination_key
                ):
                    continue
                if not _matches_exact_filter(
                    record,
                    "destination_context_type",
                    destination_context_type,
                ):
                    continue
                if not _matches_exact_filter(record, "group_chat_id", group_chat_id):
                    continue
                if not _matches_exact_filter(
                    record, "dm_counterparty_id", dm_counterparty_id
                ):
                    continue
                if not _matches_exact_filter(record, "direction", direction):
                    continue

                effective = parse_whatsapp_event_datetime(
                    record.get("effective_event_at_utc")
                )
                if effective is None:
                    continue
                if start is not None and effective < start:
                    continue
                if end is not None and effective >= end:
                    continue

                results.append((effective, _record_sequence_value(record), record))

    results.sort(key=lambda item: (item[0], item[1]))
    return [record for _, _, record in results]


def query_whatsapp_records(
    start_at_utc: datetime,
    end_at_utc: datetime,
    *,
    base_dir: Path | None = None,
    conversation_key: str | None = None,
    destination_key: str | None = None,
    destination_context_type: str | None = None,
    group_chat_id: str | None = None,
    dm_counterparty_id: str | None = None,
    direction: str | None = None,
) -> list[dict[str, Any]]:
    start = start_at_utc.astimezone(timezone.utc)
    end = end_at_utc.astimezone(timezone.utc)

    if not _has_destination_scope(
        conversation_key=conversation_key,
        destination_key=destination_key,
        group_chat_id=group_chat_id,
        dm_counterparty_id=dm_counterparty_id,
    ):
        raise ValueError(
            "query_whatsapp_records requires at least one destination scope: "
            "conversation_key, destination_key, group_chat_id, or dm_counterparty_id"
        )

    store_dir = get_whatsapp_record_store_dir(base_dir)
    partition_paths = [
        store_dir / f"{partition_day.isoformat()}.jsonl"
        for partition_day in _iter_overlapping_partition_dates(start, end)
        if (store_dir / f"{partition_day.isoformat()}.jsonl").exists()
    ]
    return _load_filtered_whatsapp_records(
        partition_paths=partition_paths,
        start=start,
        end=end,
        conversation_key=conversation_key,
        destination_key=destination_key,
        destination_context_type=destination_context_type,
        group_chat_id=group_chat_id,
        dm_counterparty_id=dm_counterparty_id,
        direction=direction,
    )


def query_whatsapp_records_any_time(
    *,
    base_dir: Path | None = None,
    conversation_key: str | None = None,
    destination_key: str | None = None,
    destination_context_type: str | None = None,
    group_chat_id: str | None = None,
    dm_counterparty_id: str | None = None,
    direction: str | None = None,
) -> list[dict[str, Any]]:
    if not _has_destination_scope(
        conversation_key=conversation_key,
        destination_key=destination_key,
        group_chat_id=group_chat_id,
        dm_counterparty_id=dm_counterparty_id,
    ):
        raise ValueError(
            "query_whatsapp_records_any_time requires at least one destination scope: "
            "conversation_key, destination_key, group_chat_id, or dm_counterparty_id"
        )

    return _load_filtered_whatsapp_records(
        partition_paths=_iter_existing_partition_paths(base_dir=base_dir),
        start=None,
        end=None,
        conversation_key=conversation_key,
        destination_key=destination_key,
        destination_context_type=destination_context_type,
        group_chat_id=group_chat_id,
        dm_counterparty_id=dm_counterparty_id,
        direction=direction,
    )


__all__ = [
    "SCHEMA_VERSION",
    "append_whatsapp_record",
    "build_whatsapp_destination_fields",
    "get_whatsapp_record_store_dir",
    "next_whatsapp_record_sequence",
    "parse_whatsapp_event_datetime",
    "query_whatsapp_records",
    "query_whatsapp_records_any_time",
    "utc_isoformat",
    "utc_now",
    "whatsapp_daily_partition_path",
]


def query_latest_whatsapp_record(
    *,
    base_dir: Path | None = None,
    conversation_key: str | None = None,
    destination_key: str | None = None,
    destination_context_type: str | None = None,
    group_chat_id: str | None = None,
    dm_counterparty_id: str | None = None,
    direction: str | None = None,
) -> dict[str, Any] | None:
    records = query_whatsapp_records_any_time(
        base_dir=base_dir,
        conversation_key=conversation_key,
        destination_key=destination_key,
        destination_context_type=destination_context_type,
        group_chat_id=group_chat_id,
        dm_counterparty_id=dm_counterparty_id,
        direction=direction,
    )
    if not records:
        return None
    return records[-1]


__all__.append("query_latest_whatsapp_record")
