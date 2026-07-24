"""Structured HEGI run logging without credential material."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


_SECRET = re.compile(
    r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*\S+|"
    r"\b(?:sk-|gh[pousr]_)[A-Za-z0-9_-]{12,}"
)


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET.sub("[REDACTED]", value)
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if _SECRET.search(str(key)) else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(redact(event), ensure_ascii=False) + "\n")
