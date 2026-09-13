"""Typed, task-level evidence for opt-in Kanban completion contracts."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


EVIDENCE_CONTRACT = "evidence"
_EVIDENCE_TYPES = "path, url, task, attachment"


class CompletionEvidenceError(ValueError):
    """Completion evidence is missing, malformed, or cannot be resolved."""


def validate_completion_evidence(conn, task, proof: Iterable[str] | None, *,
                                 accept_unproven: bool = False) -> tuple[list[dict], bool]:
    """Validate typed proof and return normalized records plus override state.

    Relative ``path:`` values resolve against the task's persisted workspace.
    The ``evidence`` contract requires at least one record unless the caller uses
    the explicit, auditable override. Other contracts remain backwards compatible
    but may still attach validated evidence.
    """
    values = list(proof or ())
    if accept_unproven and values:
        raise CompletionEvidenceError("accept_unproven cannot be combined with proof")
    if accept_unproven and task.completion_contract != EVIDENCE_CONTRACT:
        raise CompletionEvidenceError("accept_unproven is only valid for the evidence completion contract")
    if not values:
        if task.completion_contract == EVIDENCE_CONTRACT and not accept_unproven:
            raise CompletionEvidenceError(
                "completion requires proof; pass one or more typed values "
                f"({_EVIDENCE_TYPES}) or explicitly accept an unproven completion"
            )
        return [], accept_unproven

    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw in values:
        if not isinstance(raw, str) or ":" not in raw:
            raise CompletionEvidenceError(
                f"invalid proof {raw!r}; expected TYPE:VALUE where TYPE is one of {_EVIDENCE_TYPES}"
            )
        kind, value = raw.split(":", 1)
        kind, value = kind.strip().lower(), value.strip()
        if kind not in {"path", "url", "task", "attachment"} or not value:
            raise CompletionEvidenceError(
                f"invalid proof {raw!r}; expected TYPE:VALUE where TYPE is one of {_EVIDENCE_TYPES}"
            )
        record = _validate_one(conn, task, kind, value)
        key = (record["type"], str(record["resolved"]))
        if key not in seen:
            normalized.append(record)
            seen.add(key)
    return normalized, False


def _validate_one(conn, task, kind: str, value: str) -> dict:
    if kind == "path":
        candidate = Path(value)
        if not candidate.is_absolute():
            if not task.workspace_path:
                raise CompletionEvidenceError(
                    f"relative proof path {value!r} cannot resolve because the task has no workspace_path"
                )
            candidate = Path(task.workspace_path) / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise CompletionEvidenceError(f"proof path does not exist: {value}") from exc
        return {"type": kind, "value": value, "resolved": str(resolved)}

    if kind == "url":
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CompletionEvidenceError(f"proof URL must be an absolute http(s) URL: {value}")
        return {"type": kind, "value": value, "resolved": value}

    if kind == "task":
        row = conn.execute("SELECT id FROM tasks WHERE id = ?", (value,)).fetchone()
        if row is None:
            raise CompletionEvidenceError(f"proof task does not exist: {value}")
        return {"type": kind, "value": value, "resolved": value}

    try:
        attachment_id = int(value)
    except ValueError as exc:
        raise CompletionEvidenceError(f"proof attachment id must be an integer: {value}") from exc
    row = conn.execute(
        "SELECT id FROM task_attachments WHERE id = ? AND task_id = ?", (attachment_id, task.id)
    ).fetchone()
    if row is None:
        raise CompletionEvidenceError(
            f"proof attachment {attachment_id} does not belong to task {task.id}"
        )
    return {"type": kind, "value": attachment_id, "resolved": attachment_id}
