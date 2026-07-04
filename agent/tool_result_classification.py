"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

import json
import logging
from typing import Any


logger = logging.getLogger(__name__)

FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    if tool_name not in FILE_MUTATING_TOOL_NAMES or not isinstance(result, str):
        return False
    try:
        data = json.loads(result.strip())
    except Exception as exc:
        logger.debug(
            "Ignoring non-JSON file mutation tool result for %s: %s",
            tool_name,
            exc,
            exc_info=True,
        )
        return False
    if not isinstance(data, dict) or data.get("error"):
        return False
    if tool_name == "write_file":
        return "bytes_written" in data
    if tool_name == "patch":
        return data.get("success") is True
    return False
