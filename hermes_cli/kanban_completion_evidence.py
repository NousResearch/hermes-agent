"""Typed, task-level evidence for opt-in Kanban completion contracts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


EVIDENCE_CONTRACT = "evidence"
_EVIDENCE_TYPES = "path, url, task, attachment"


class CompletionEvidenceError(ValueError):
    """Completion evidence is missing, malformed, or cannot be resolved."""


def prepare_completion_evidence(
    conn, task, proof: Iterable[str] | None, *, accept_unproven: bool = False,
    max_path_bytes: int,
) -> tuple[list[str], list[dict], bool, int]:
    """Capture typed proof for mandatory revalidation at settlement.

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
        return values, [], accept_unproven, max_path_bytes

    return values, _validate_values(conn, task, values, max_path_bytes), False, max_path_bytes


def settle_completion_evidence(
    conn, task, prepared: tuple[list[str], list[dict], bool, int],
) -> tuple[list[dict], bool]:
    """Revalidate a prepared receipt under the terminal write transaction."""
    values, prior_records, accept_unproven, max_path_bytes = prepared
    _, records, settled_override, _ = prepare_completion_evidence(
        conn, task, values, accept_unproven=accept_unproven, max_path_bytes=max_path_bytes,
    )
    if records != prior_records or settled_override != accept_unproven:
        raise CompletionEvidenceError("completion evidence changed before settlement")
    return records, settled_override


def _validate_values(conn, task, values: list[str], max_path_bytes: int) -> list[dict]:
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
        record = _validate_one(conn, task, kind, value, max_path_bytes)
        key = (record["type"], str(record["resolved"]))
        if key not in seen:
            normalized.append(record)
            seen.add(key)
    return normalized


def _validate_one(conn, task, kind: str, value: str, max_path_bytes: int) -> dict:
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
            if not resolved.is_file():
                raise CompletionEvidenceError(f"proof path must identify a file: {value}")
            digest, size = _digest_stable_file(resolved, max_path_bytes)
        except CompletionEvidenceError:
            raise
        except (OSError, RuntimeError) as exc:
            raise CompletionEvidenceError(f"proof path does not exist: {value}") from exc
        return {
            "type": kind,
            "value": value,
            "resolved": str(resolved),
            "size": size,
            "sha256": digest,
        }

    if kind == "url":
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise CompletionEvidenceError(f"proof URL must be an absolute http(s) URL: {value}")
        from agent.redact import redact_sensitive_text
        if redact_sensitive_text(value, force=True, redact_url_credentials=True) != value:
            raise CompletionEvidenceError("proof URL must not contain reusable credentials")
        return {"type": kind, "value": value, "resolved": value}

    if kind == "task":
        row = conn.execute(
            """SELECT t.id
                 FROM tasks t
                WHERE t.id = ? AND t.id != ? AND t.status = 'done'
                  AND EXISTS (
                      SELECT 1 FROM task_links l
                       WHERE (l.parent_id = t.id AND l.child_id = ?)
                          OR (l.child_id = t.id AND l.parent_id = ?)
                  )""",
            (value, task.id, task.id, task.id),
        ).fetchone()
        if row is None:
            raise CompletionEvidenceError(
                f"proof task must be a different, completed task directly linked to {task.id}: {value}"
            )
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


def _digest_stable_file(path: Path, max_bytes: int) -> tuple[str, int]:
    """Hash one stable file identity so the durable receipt binds observed bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if before.st_size > max_bytes:
            raise CompletionEvidenceError(
                f"proof path exceeds the {max_bytes}-byte evidence limit: {path}"
            )
        read_bytes = 0
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            read_bytes += len(chunk)
            if read_bytes > max_bytes:
                raise CompletionEvidenceError(
                    f"proof path exceeds the {max_bytes}-byte evidence limit: {path}"
                )
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if identity(before) != identity(after):
        raise CompletionEvidenceError(f"proof path changed while it was being recorded: {path}")
    return digest.hexdigest(), after.st_size
