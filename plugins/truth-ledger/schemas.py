from __future__ import annotations

import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMAS_DIR = Path(__file__).with_name("schemas")

_FORMAT_CHECKER = FormatChecker()
_RFC3339_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})$"
)


@_FORMAT_CHECKER.checks("date-time", raises=(TypeError, ValueError))
def _is_rfc3339_datetime(value: object) -> bool:
    if not isinstance(value, str) or _RFC3339_RE.fullmatch(value) is None:
        return False
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    return parsed.tzinfo is not None

_SCHEMA_FILES = {
    "source-envelope.v1": "source-envelope-v1.schema.json",
    "fact-candidates.v1": "fact-candidates-v1.schema.json",
    "ledger-event.v1": "ledger-event-v1.schema.json",
    "spool-record.v1": "spool-record-v1.schema.json",
    "dead-letter.v1": "dead-letter-v1.schema.json",
    "current-projection.v1": "current-projection-v1.schema.json",
}


def available_schema_names() -> list[str]:
    return list(_SCHEMA_FILES.keys())


@lru_cache(maxsize=len(_SCHEMA_FILES))
def load_schema(schema_name: str) -> dict[str, Any]:
    file_name = _SCHEMA_FILES.get(schema_name)
    if file_name is None:
        raise ValueError(f"unknown schema name: {schema_name}")
    path = SCHEMAS_DIR / file_name
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=len(_SCHEMA_FILES))
def _get_validator(schema_name: str) -> Draft202012Validator:
    schema = load_schema(schema_name)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)


def validate_all_metaschemas() -> dict[str, str]:
    errors: dict[str, str] = {}
    for schema_name in _SCHEMA_FILES:
        try:
            _get_validator(schema_name)
        except Exception as exc:  # pragma: no cover - exercised by tests
            errors[schema_name] = str(exc)
    return errors


def validate_document(schema_name: str, document: Any) -> None:
    validator = _get_validator(schema_name)
    errors = sorted(validator.iter_errors(document), key=lambda e: list(e.path))
    if errors:
        first = errors[0]
        path = "/".join(str(p) for p in first.absolute_path)
        where = path or "<root>"
        raise ValueError(f"{schema_name} validation failed at {where}: {first.message}")
