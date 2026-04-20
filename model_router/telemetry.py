from __future__ import annotations

import json
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


def build_event(
    *,
    router_version: str,
    config_path: str,
    request_input: Any,
    decision: Any,
    request_id: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": utc_now_iso(),
        "event_type": "decision",
        "request_id": request_id or str(uuid.uuid4()),
        "router_version": router_version,
        "config_path": config_path,
        "input": to_jsonable(request_input),
        "decision": to_jsonable(decision),
    }


def append_jsonl(path: str | Path, event: dict[str, Any]) -> None:
    log_path = Path(path)
    ensure_parent_dir(log_path)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
