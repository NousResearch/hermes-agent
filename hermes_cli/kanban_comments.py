from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
from pathlib import Path

from hermes_cli import kanban_db as kb


def _profile_author() -> str:
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "user"
    except Exception:
        return "user"


def _cmd_comment(args: argparse.Namespace) -> int:
    body = " ".join(args.text).strip()
    if args.max_len is not None:
        if args.max_len < 1:
            print("kanban: --max-len must be positive", file=sys.stderr)
            return 2
        if len(body) > args.max_len:
            suffix = f"\n\n[trimmed to {args.max_len} chars by --max-len]"
            body = body[: max(0, args.max_len - len(suffix))].rstrip() + suffix
    profile = _profile_author()
    author = f"cli:{args.author}" if args.author else f"cli:{profile}"
    with kb.connect_closing() as conn:
        kb.add_comment(conn, args.task_id, author, body)
    print(f"Comment added to {args.task_id}")
    return 0


def _cmd_attach(args: argparse.Namespace) -> int:
    """Attach a local file to a task.

    Reads the file off disk, writes it under the task's attachments dir,
    and records the metadata row via the shared ``store_attachment_bytes``
    path (same code the dashboard upload and the agent tool use), so the
    25 MB cap and name-sanitisation behave identically everywhere.
    """
    src = Path(args.path).expanduser()
    if not src.is_file():
        print(f"kanban: no such file: {src}", file=sys.stderr)
        return 1
    data = src.read_bytes()
    name = args.name or src.name
    content_type = args.content_type or mimetypes.guess_type(name)[0]
    profile = _profile_author()
    uploaded_by = f"cli:{args.author}" if args.author else f"cli:{profile}"
    try:
        with kb.connect_closing() as conn:
            att_id = kb.store_attachment_bytes(
                conn,
                args.task_id,
                name,
                data,
                content_type=content_type,
                uploaded_by=uploaded_by,
            )
    except kb.AttachmentTooLarge as exc:
        print(f"kanban: {exc}", file=sys.stderr)
        return 1
    print(f"Attached {name} to {args.task_id} (attachment {att_id}, {len(data)} bytes)")
    return 0


def _cmd_attachments(args: argparse.Namespace) -> int:
    """List a task's attachments."""
    with kb.connect_closing() as conn:
        if kb.get_task(conn, args.task_id) is None:
            print(f"no such task: {args.task_id}", file=sys.stderr)
            return 1
        atts = kb.list_attachments(conn, args.task_id)
    if getattr(args, "json", False):
        print(json.dumps([
            {
                "id": a.id,
                "filename": a.filename,
                "content_type": a.content_type,
                "size": a.size,
                "uploaded_by": a.uploaded_by,
                "stored_path": a.stored_path,
                "created_at": a.created_at,
            }
            for a in atts
        ], indent=2))
        return 0
    if not atts:
        print(f"No attachments on {args.task_id}")
        return 0
    print(f"Attachments on {args.task_id}:")
    for a in atts:
        ct = a.content_type or "-"
        print(f"  [{a.id}] {a.filename}  ({a.size} bytes, {ct}, by {a.uploaded_by or '-'})")
        print(f"        {a.stored_path}")
    return 0


def _cmd_attach_rm(args: argparse.Namespace) -> int:
    """Delete an attachment by id (removes the row and the on-disk blob)."""
    with kb.connect_closing() as conn:
        removed = kb.delete_attachment(conn, args.attachment_id)
    if removed is None:
        print(f"no such attachment: {args.attachment_id}", file=sys.stderr)
        return 1
    print(f"Deleted attachment {args.attachment_id} ({removed.filename}) from {removed.task_id}")
    return 0
