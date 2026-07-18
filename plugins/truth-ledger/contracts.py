from __future__ import annotations

import json
from typing import Any, Mapping

SCHEMA_VERSION = 1


def assert_schema_version(schema_version: int) -> None:
    """Fail closed on unknown schema versions."""
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={schema_version}; expected {SCHEMA_VERSION}"
        )


def serialize_jsonl_record(payload: Mapping[str, Any]) -> bytes:
    """Serialize one JSONL record as compact UTF-8 with trailing newline."""
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return text.encode("utf-8") + b"\n"
