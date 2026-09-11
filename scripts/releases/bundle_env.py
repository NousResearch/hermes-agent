"""Non-secret environment defaults carried by a desktop bundle."""
from __future__ import annotations

import json
import re


def validate(values: object) -> dict[str, str]:
    if not isinstance(values, dict):
        raise ValueError("Bundle environment must be a JSON object")
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError("Bundle environment names must be valid environment identifiers")
        if not isinstance(value, str) or "\0" in value:
            raise ValueError(f"Bundle environment value for {key} must be a string without NUL")
    return values


def parse_assignments(assignments: list[str]) -> dict[str, str]:
    values = {}
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        if not separator:
            raise ValueError("--bundle-env requires NAME=VALUE")
        if key in values:
            raise ValueError(f"Duplicate bundle environment name: {key}")
        values[key] = value
    return validate(values)


def decode(raw: str) -> dict[str, str]:
    return validate(json.loads(raw or "{}"))
