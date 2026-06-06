#!/usr/bin/env python3
"""
CLI subcommands for inbox management.

Usage (via executors.cli):
    python -m executors.cli inbox add --source cli --title "Fix bug" --body "Fix login error"
    python -m executors.cli inbox add --source feishu --title "Review PR" --body "..."
    python -m executors.cli inbox list [--status pending|confirmed|rejected|archived] [--source cli|feishu|...]
    python -m executors.cli inbox show <item_id>
    python -m executors.cli inbox convert <item_id> --task-id <task_id>
    python -m executors.cli inbox reject <item_id> [--reason "not relevant"]
    python -m executors.cli inbox archive <item_id>
    python -m executors.cli inbox edit <item_id> --title "New title" --prompt "New prompt"
    python -m executors.cli inbox summary
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from executors.inbox import InboxManager
from executors.types import InboxSource, InboxStatus


SOURCE_ICONS = {
    "desktop": "✎",
    "cli": "⌨",
    "feishu": "飞",
    "discord": "◌",
    "scheduler": "⏱",
}

STATUS_ICONS = {
    "pending": "●",
    "confirmed": "✓",
    "rejected": "✗",
    "archived": "○",
    "expired": "⊗",
}


async def cmd_add(
    mgr: InboxManager,
    source: str,
    title: str,
    body: str,
    executor: str = "",
    project: str = "",
    priority: str = "normal",
) -> None:
    """Add a new inbox item."""
    try:
        src = InboxSource(source)
    except ValueError:
        valid = ", ".join(s.value for s in InboxSource)
        print(f"Invalid source: {source}. Valid: {valid}", file=sys.stderr)
        sys.exit(1)

    raw = {"title": title, "body": body}
    if project:
        raw["project_hint"] = project

    item = mgr.add(
        source=src,
        title=title,
        body=body,
        raw_payload=raw,
        suggested_executor=executor or None,
        project_hint=project or None,
        priority=priority,
    )
    icon = SOURCE_ICONS.get(item.source.value, "?")
    print(f"{icon} Added inbox item: {item.id}")
    print(f"   Title:  {title}")
    print(f"   Source: {item.source.value}")
    print(f"   Status: {item.status.value}")


async def cmd_list(
    mgr: InboxManager,
    status: str = "",
    source: str = "",
    json_output: bool = False,
) -> None:
    """List inbox items with optional filters."""
    status_filter = InboxStatus(status) if status else None
    source_filter = InboxSource(source) if source else None
    items = mgr.list_items(status=status_filter, source=source_filter)

    if json_output:
        data = []
        for it in items:
            data.append({
                "id": it.id,
                "source": it.source.value,
                "title": it.draft.title,
                "status": it.status.value,
                "created_at": it.created_at.isoformat(),
                "linked_task_id": it.linked_task_id,
            })
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    if not items:
        print("No inbox items found.")
        return

    print(f"{'ID':<18} {'S':<3} {'Title':<40} {'Status':<12} {'Created'}")
    print("-" * 100)
    for it in items:
        sid = it.id[-14:]
        src_icon = SOURCE_ICONS.get(it.source.value, "?")
        st_icon = STATUS_ICONS.get(it.status.value, "?")
        title = it.draft.title[:38]
        created = it.created_at.strftime("%m-%d %H:%M")
        linked = f" → {it.linked_task_id[:12]}" if it.linked_task_id else ""
        print(f"{sid:<18} {src_icon:<3} {title:<40} {st_icon} {it.status.value:<9} {created}{linked}")


async def cmd_show(mgr: InboxManager, item_id: str, json_output: bool = False) -> None:
    """Show full details of an inbox item."""
    item = mgr.get(item_id)
    if item is None:
        print(f"Item not found: {item_id}", file=sys.stderr)
        sys.exit(1)

    if json_output:
        data = {
            "id": item.id,
            "source": item.source.value,
            "status": item.status.value,
            "title": item.draft.title,
            "prompt": item.draft.suggested_prompt,
            "executor": item.draft.suggested_executor,
            "project": item.draft.project_hint,
            "priority": item.draft.priority,
            "user_edited": item.draft.user_edited,
            "created_at": item.created_at.isoformat(),
            "expires_at": item.expires_at.isoformat() if item.expires_at else None,
            "linked_task_id": item.linked_task_id,
            "rejected_reason": item.rejected_reason,
            "raw_payload": item.raw_payload,
            "writeback_dest": InboxManager.writeback_destination(item),
        }
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return

    icon = SOURCE_ICONS.get(item.source.value, "?")
    print(f"{icon} Inbox Item: {item.id}")
    print(f"  Source:    {item.source.value}")
    print(f"  Status:    {STATUS_ICONS.get(item.status.value, '?')} {item.status.value}")
    print(f"  Title:     {item.draft.title}")
    print(f"  Prompt:    {item.draft.suggested_prompt[:200]}")
    if item.draft.suggested_executor:
        print(f"  Executor:  {item.draft.suggested_executor}")
    if item.draft.project_hint:
        print(f"  Project:   {item.draft.project_hint}")
    print(f"  Priority:  {item.draft.priority}")
    print(f"  Edited:    {'yes' if item.draft.user_edited else 'no'}")
    print(f"  Created:   {item.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if item.expires_at:
        print(f"  Expires:   {item.expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if item.linked_task_id:
        print(f"  Task:      {item.linked_task_id}")
    if item.rejected_reason:
        print(f"  Rejected:  {item.rejected_reason}")
    print(f"  Writeback: {InboxManager.writeback_destination(item)}")


async def cmd_convert(mgr: InboxManager, item_id: str, task_id: str) -> None:
    """Convert an inbox item to a task thread."""
    item = mgr.convert_to_task(item_id, task_id)
    if item is None:
        print(f"Item not found: {item_id}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Converted {item_id} → task {task_id}")
    print(f"  Title: {item.draft.title}")
    print(f"  Status: {item.status.value}")


async def cmd_reject(mgr: InboxManager, item_id: str, reason: str = "") -> None:
    """Reject an inbox item."""
    item = mgr.reject(item_id, reason)
    if item is None:
        print(f"Item not found: {item_id}", file=sys.stderr)
        sys.exit(1)
    print(f"✗ Rejected: {item_id}")
    if reason:
        print(f"  Reason: {reason}")


async def cmd_archive(mgr: InboxManager, item_id: str) -> None:
    """Archive an inbox item."""
    item = mgr.archive(item_id)
    if item is None:
        print(f"Item not found: {item_id}", file=sys.stderr)
        sys.exit(1)
    print(f"○ Archived: {item_id}")


async def cmd_edit(
    mgr: InboxManager,
    item_id: str,
    title: str = "",
    prompt: str = "",
    executor: str = "",
    project: str = "",
    priority: str = "",
) -> None:
    """Edit the task draft of a pending inbox item."""
    item = mgr.update_draft(
        item_id,
        title=title or None,
        prompt=prompt or None,
        executor=executor or None,
        project=project or None,
        priority=priority or None,
    )
    if item is None:
        print(f"Item not found: {item_id}", file=sys.stderr)
        sys.exit(1)
    print(f"✓ Updated draft for {item_id}")
    if title:
        print(f"  Title: {title}")


async def cmd_summary(mgr: InboxManager) -> None:
    """Show inbox summary (counts by status and source)."""
    counts = mgr.count_by_status()
    src_counts = mgr.count_pending_by_source()

    print("=== Inbox Summary ===")
    print(f"Pending:   {counts.get('pending', 0)}")
    print(f"Confirmed: {counts.get('confirmed', 0)}")
    print(f"Rejected:  {counts.get('rejected', 0)}")
    print(f"Archived:  {counts.get('archived', 0)}")
    print(f"Expired:   {counts.get('expired', 0)}")
    if src_counts:
        print(f"\nPending by source:")
        for src, n in sorted(src_counts.items()):
            icon = SOURCE_ICONS.get(src, "?")
            print(f"  {icon} {src:<12} {n}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def handle_inbox_command(
    project_root: Path,
    args,
) -> None:
    """Dispatch inbox subcommands."""
    mgr = InboxManager(project_root)

    sub = args.inbox_subcommand
    if sub is None:
        print("Usage: inbox {add|list|show|convert|reject|archive|edit|summary} [...]", file=sys.stderr)
        sys.exit(1)

    try:
        if sub == "add":
            await cmd_add(
                mgr,
                source=args.source,
                title=args.title,
                body=args.body,
                executor=getattr(args, "executor", ""),
                project=getattr(args, "project", ""),
                priority=getattr(args, "priority", "normal"),
            )
        elif sub == "list":
            await cmd_list(
                mgr,
                status=getattr(args, "status", ""),
                source=getattr(args, "source", ""),
                json_output=getattr(args, "json", False),
            )
        elif sub == "show":
            await cmd_show(mgr, args.item_id, getattr(args, "json", False))
        elif sub == "convert":
            await cmd_convert(mgr, args.item_id, args.task_id)
        elif sub == "reject":
            await cmd_reject(mgr, args.item_id, getattr(args, "reason", ""))
        elif sub == "archive":
            await cmd_archive(mgr, args.item_id)
        elif sub == "edit":
            await cmd_edit(
                mgr, args.item_id,
                title=getattr(args, "title", ""),
                prompt=getattr(args, "prompt", ""),
                executor=getattr(args, "executor", ""),
                project=getattr(args, "project", ""),
                priority=getattr(args, "priority", ""),
            )
        elif sub == "summary":
            await cmd_summary(mgr)
        else:
            print(f"Unknown inbox command: {sub}", file=sys.stderr)
            sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
