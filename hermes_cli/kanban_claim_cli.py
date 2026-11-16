"""Manual Kanban claim command backed by the atomic claim-result contract."""

from __future__ import annotations

import argparse

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli.kanban_output import _err


def _cmd_claim(args: argparse.Namespace) -> int:
    with kbc.connect_closing() as conn:
        attempt = kb.claim_task_result(conn, args.task_id, ttl_seconds=args.ttl)
        task = attempt.task
        if task is None:
            if attempt.reason is not None:
                return _err(f"cannot claim {args.task_id}: {attempt.reason}")
            existing = kb.get_task(conn, args.task_id)
            if existing is None:
                return _err(f"no such task: {args.task_id}")
            return _err(
                f"cannot claim {args.task_id}: status={existing.status} "
                f"lock={existing.claim_lock or '(none)'}",
            )
        workspace = kbw.resolve_workspace(task)
        kbw.set_workspace_path(conn, task.id, str(workspace))
    print(f"Claimed {task.id}\nWorkspace: {workspace}")
    return 0
