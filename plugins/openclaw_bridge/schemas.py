from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _load_schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _validate_required(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    missing = [field for field in schema.get("required", []) if field not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")


def _validate(payload: dict[str, Any], schema_name: str) -> dict[str, Any]:
    schema = _load_schema(schema_name)
    _validate_required(payload, schema)
    try:
        import jsonschema

        jsonschema.validate(payload, schema)
    except ImportError:
        return payload
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return payload


def validate_delegated_task(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate(payload, "delegated-task.schema.json")


def validate_delegated_result(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate(payload, "delegated-result.schema.json")

