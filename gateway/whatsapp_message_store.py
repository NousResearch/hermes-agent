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
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
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


def query_whatsapp_records(
    start_at_utc: datetime,
    end_at_utc: datetime,
    *,
    base_dir: Path | None = None,
    destination_key: str | None = None,
) -> list[dict[str, Any]]:
    start = start_at_utc.astimezone(timezone.utc)
    end = end_at_utc.astimezone(timezone.utc)
    current_day: date = start.date()
    end_day: date = end.date()
    results: list[dict[str, Any]] = []
    store_dir = get_whatsapp_record_store_dir(base_dir)

    while current_day <= end_day:
        path = store_dir / f"{current_day.isoformat()}.jsonl"
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if destination_key and record.get("destination_key") != destination_key:
                        continue
                    effective = parse_whatsapp_event_datetime(record.get("effective_event_at_utc"))
                    if effective is None:
                        continue
                    if start <= effective <= end:
                        results.append(record)
        current_day += timedelta(days=1)

    results.sort(
        key=lambda record: (
            record.get("effective_event_at_utc") or "",
            int(record.get("record_sequence") or 0),
        )
    )
    return results


__all__ = [
    "SCHEMA_VERSION",
    "append_whatsapp_record",
    "build_whatsapp_destination_fields",
    "get_whatsapp_record_store_dir",
    "next_whatsapp_record_sequence",
    "parse_whatsapp_event_datetime",
    "query_whatsapp_records",
    "utc_isoformat",
    "utc_now",
    "whatsapp_daily_partition_path",
]
