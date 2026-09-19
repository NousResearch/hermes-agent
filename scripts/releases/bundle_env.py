"""Non-secret environment defaults and explicit clears carried by a desktop bundle."""
from __future__ import annotations

from collections.abc import Sequence
import json
import re


def validate(values: object) -> dict[str, str | None]:
    if not isinstance(values, dict):
        raise ValueError("Bundle environment must be a JSON object")
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Bundle environment names must be valid environment identifiers")
        if value is not None and (not isinstance(value, str) or "\0" in value):
            raise ValueError(f"Bundle environment value for {key} must be a string without NUL or null")
    return values


def parse_assignments(assignments: list[str], unset: Sequence[str] = ()) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise ValueError("--bundle-env requires NAME=VALUE")
        if key in values:
            raise ValueError(f"Duplicate bundle environment name: {key}")
        values[key] = value
    for key in unset:
        if key in values:
            raise ValueError(f"Duplicate bundle environment name: {key}")
        values[key] = None
    return validate(values)


def decode(raw: str) -> dict[str, str | None]:
    return validate(json.loads(raw or "{}"))
