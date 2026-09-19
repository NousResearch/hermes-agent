"""Kanban attachment bridge for diagnostic engineering receipts."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from .receipts import ReceiptValidationError, validate_receipt


_FILENAME_PREFIX = "engineering-evidence-"


def _kanban_modules():
    """Load Hermes's existing Kanban layer lazily so plugin discovery stays bounded."""

    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    return kb, kbc


def _open_board(db_path: Path | None, board: str | None):
    kb, kbc = _kanban_modules()
    if db_path is not None:
        kbc.init_db(db_path, board=board)
        return kb, kbc, kbc.connect(db_path, board=board)
    kbc.init_db(board=board)
    return kb, kbc, kbc.connect(board=board)


def attach_receipt_to_task(
    receipt: dict[str, Any],
    *,
    task_id: str,
    db_path: Path | None = None,
    board: str | None = None,
    uploaded_by: str = "engineering-evidence",
    attachments_root: Path | None = None,
) -> dict[str, Any]:
    """Attach a validated receipt through ``kanban_db``'s attachment API."""

    validated = validate_receipt(receipt)
    receipt_task_id = validated.get("task_id")
    if receipt_task_id and receipt_task_id != task_id:
        raise ValueError("receipt task_id does not match attachment task_id")
    kb, _kbc, conn = _open_board(db_path, board)
    try:
        if kb.get_task(conn, task_id) is None:
            raise ValueError(f"unknown task {task_id}")
        filename = f"{_FILENAME_PREFIX}{validated['receipt_type']}-{validated['receipt_id']}.json"
        data = json.dumps(validated, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if attachments_root is None:
            destination_dir = kb.task_attachments_dir(task_id, board=board)
        else:
            destination_dir = attachments_root.expanduser().resolve() / task_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        safe_name = kb._safe_attachment_name(filename)
        destination = kb._collision_free_path(destination_dir, safe_name)
        if len(data) > kb.KANBAN_ATTACHMENT_MAX_BYTES:
            raise ValueError("receipt exceeds Kanban attachment size limit")
        destination.write_bytes(data)
        try:
            attachment_id = kb.add_attachment(
                conn,
                task_id,
                filename=destination.name,
                stored_path=str(destination.resolve()),
                content_type="application/json",
                size=len(data),
                uploaded_by=uploaded_by,
            )
        except Exception:
            with contextlib.suppress(OSError):
                destination.unlink(missing_ok=True)
            raise
        attachment = kb.get_attachment(conn, attachment_id)
        if attachment is None:  # pragma: no cover - defensive against a broken DB adapter
            raise RuntimeError("Kanban attachment row was not readable after insertion")
        return {
            "id": attachment.id,
            "task_id": attachment.task_id,
            "filename": attachment.filename,
            "stored_path": attachment.stored_path,
            "content_type": attachment.content_type,
            "size": attachment.size,
        }
    finally:
        conn.close()


def list_task_receipts(
    *,
    task_id: str,
    db_path: Path | None = None,
    board: str | None = None,
) -> list[dict[str, Any]]:
    """Read only valid engineering receipts attached to a Kanban task."""

    kb, _kbc, conn = _open_board(db_path, board)
    try:
        if kb.get_task(conn, task_id) is None:
            raise ValueError(f"unknown task {task_id}")
        records: list[dict[str, Any]] = []
        for attachment in kb.list_attachments(conn, task_id):
            if not attachment.filename.startswith(_FILENAME_PREFIX):
                continue
            try:
                receipt = validate_receipt(json.loads(Path(attachment.stored_path).read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, ReceiptValidationError):
                continue
            if receipt.get("task_id") and receipt["task_id"] != task_id:
                continue
            records.append(receipt)
        return records
    finally:
        conn.close()
