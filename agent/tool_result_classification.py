"""Shared helpers for classifying tool result payloads."""

from __future__ import annotations

from enum import Enum
import json
from typing import Any


FILE_MUTATING_TOOL_NAMES = frozenset({"write_file", "patch"})


class FileMutationOutcome(str, Enum):
    """High-level outcome for a tool result from a file-mutating tool."""

    LANDED = "landed"
    PROTECTED_REFUSAL = "protected_refusal"
    FAILED_EDIT = "failed_edit"


def _decode_tool_result(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, str):
        return None
    stripped = result.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except Exception:
        if not stripped.startswith("{"):
            return None
        try:
            data, _end = json.JSONDecoder().raw_decode(stripped)
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def classify_file_mutation_result(tool_name: str, result: Any) -> FileMutationOutcome:
    """Classify whether a file mutation landed, was guardrail-blocked, or failed.

    ``protected_refusal`` is reserved for deliberate Hermes file-tool guardrails
    that block mutation before any write occurs (sensitive paths, cross-profile
    guardrails, traversal guards, etc.). It is still unresolved work if the
    target file was required, but it should not be conflated with a normal edit
    failure such as a missing ``old_string``.
    """
    if tool_name not in FILE_MUTATING_TOOL_NAMES:
        return FileMutationOutcome.FAILED_EDIT

    data = _decode_tool_result(result)
    if data is None:
        return FileMutationOutcome.FAILED_EDIT

    if data.get("file_mutation_status") == FileMutationOutcome.PROTECTED_REFUSAL.value:
        return FileMutationOutcome.PROTECTED_REFUSAL

    if data.get("error"):
        return FileMutationOutcome.FAILED_EDIT

    if tool_name == "write_file" and "bytes_written" in data:
        return FileMutationOutcome.LANDED
    if tool_name == "patch" and data.get("success") is True:
        return FileMutationOutcome.LANDED
    return FileMutationOutcome.FAILED_EDIT


def file_mutation_result_landed(tool_name: str, result: Any) -> bool:
    """Return True when a file mutation result proves the write landed."""
    return classify_file_mutation_result(tool_name, result) is FileMutationOutcome.LANDED
