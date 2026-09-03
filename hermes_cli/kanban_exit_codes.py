"""Stable process exit codes for dispatcher-spawned Kanban workers."""

from __future__ import annotations

import os
from typing import Any


# BSD sysexits values, with portable fallbacks for platforms that do not expose
# the names through ``os`` (notably Windows).
KANBAN_RATE_LIMIT_EXIT_CODE = getattr(os, "EX_TEMPFAIL", 75)
KANBAN_PROTOCOL_EXIT_CODE = getattr(os, "EX_PROTOCOL", 76)


def single_query_exit_code(result: Any, *, kanban_worker: bool) -> int:
    """Map a quiet single-query result to a process exit code."""
    if not isinstance(result, dict) or not result.get("failed"):
        return 0
    if not kanban_worker:
        return 1
    reason = str(result.get("failure_reason") or "")
    if reason in {"rate_limit", "billing"}:
        return KANBAN_RATE_LIMIT_EXIT_CODE
    if reason == "kanban_protocol":
        return KANBAN_PROTOCOL_EXIT_CODE
    return 1


__all__ = [
    "KANBAN_PROTOCOL_EXIT_CODE",
    "KANBAN_RATE_LIMIT_EXIT_CODE",
    "single_query_exit_code",
]
