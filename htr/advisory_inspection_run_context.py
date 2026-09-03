"""Task 29 — read-only run-context detection (R5-13)."""

from __future__ import annotations

from typing import Any, Literal

from htr.advisory_inspection_secure import (
    classify_regular_file_presence,
    read_regular_control_file,
)

RunContextStatus = Literal[
    "run_not_finalized",
    "run_finalized_source_read_only",
    "run_successor_read_only",
    "run_context_indeterminate",
]

_CLOSURE_FILENAME = "run_final_closure_record.json"
_ORIGIN_FILENAME = "recovery_origin.json"


def _bind_state(parent_fd: int, filename: str, *, decode_kind: str, context: str) -> str:
    """Return ``absent``, ``success``, or ``untrusted`` for a control JSON file."""
    presence, _, _ = classify_regular_file_presence(parent_fd, filename)
    if presence == "absent":
        return "absent"

    read_result = read_regular_control_file(
        parent_fd,
        filename,
        decode_kind=decode_kind,  # type: ignore[arg-type]
        context=context,
    )
    if read_result.budget_exceeded:
        return "untrusted"
    if not read_result.ok or read_result.decode is None or not read_result.decode.ok:
        return "untrusted"
    if read_result.decode.obj is None or not isinstance(read_result.decode.obj, dict):
        return "untrusted"
    return "success"


def _origin_successor_matches(origin: dict[str, Any], run_id: str) -> bool:
    successor = origin.get("successor_run_id")
    return isinstance(successor, str) and successor == run_id


def _closure_run_matches(closure: dict[str, Any], run_id: str) -> bool:
    bound_run = closure.get("run_id")
    return isinstance(bound_run, str) and bound_run == run_id


def detect_run_context(run_fd: int, *, run_id: str) -> RunContextStatus:
    """Detect run context from closure and recovery-origin control files only."""
    origin_state = _bind_state(
        run_fd,
        _ORIGIN_FILENAME,
        decode_kind="link",
        context="run_context/recovery_origin",
    )
    closure_state = _bind_state(
        run_fd,
        _CLOSURE_FILENAME,
        decode_kind="link",
        context="run_context/run_final_closure",
    )

    origin_obj: dict[str, Any] | None = None
    if origin_state == "success":
        read_result = read_regular_control_file(
            run_fd,
            _ORIGIN_FILENAME,
            decode_kind="link",
            context="run_context/recovery_origin_rebind",
        )
        if read_result.ok and read_result.decode and read_result.decode.ok:
            obj = read_result.decode.obj
            if isinstance(obj, dict):
                origin_obj = obj

    if origin_state == "success" and origin_obj is not None:
        if not _origin_successor_matches(origin_obj, run_id):
            return "run_context_indeterminate"
        return "run_successor_read_only"

    if origin_state == "untrusted":
        return "run_context_indeterminate"

    if closure_state == "success":
        read_result = read_regular_control_file(
            run_fd,
            _CLOSURE_FILENAME,
            decode_kind="link",
            context="run_context/closure_rebind",
        )
        if read_result.ok and read_result.decode and read_result.decode.ok:
            obj = read_result.decode.obj
            if isinstance(obj, dict) and _closure_run_matches(obj, run_id):
                return "run_finalized_source_read_only"
        return "run_context_indeterminate"

    if closure_state == "untrusted":
        return "run_context_indeterminate"

    return "run_not_finalized"
