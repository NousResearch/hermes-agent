"""Worker protocol validation and deterministic hashing."""
from __future__ import annotations
import hashlib, json, re
from datetime import datetime, timezone
from uuid import UUID

WORKER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")

def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

def validate_system_echo_payload(payload: object, max_bytes: int = 4096) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {"message"}:
        raise ValueError("invalid_task_payload")
    message = payload["message"]
    if not isinstance(message, str) or not message or len(message.encode("utf-8")) > max_bytes:
        raise ValueError("invalid_task_payload")
    return {"message": message}

def require_uuid(value: object, name: str) -> str:
    if not isinstance(value, str): raise ValueError(f"invalid_{name}")
    try: UUID(value)
    except (ValueError, TypeError): raise ValueError(f"invalid_{name}") from None
    return value

def require_worker_id(value: object) -> str:
    if not isinstance(value, str) or not WORKER_ID_RE.fullmatch(value): raise ValueError("invalid_worker_id")
    return value

def require_timestamp(value: object, name: str) -> str:
    if not isinstance(value, str): raise ValueError(f"invalid_{name}")
    try: datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError: raise ValueError(f"invalid_{name}") from None
    return value

def closed(data: object, fields: set[str]) -> dict:
    if not isinstance(data, dict) or set(data) - fields: raise ValueError("malformed_request")
    return data
