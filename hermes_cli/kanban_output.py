"""Text / ``--json`` output helpers shared by the ``hermes kanban`` CLI modules."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb

_STATUS_ICONS = {
    "todo": "◻", "ready": "▶", "running": "●", "scheduled": "⏱",
    "blocked": "⊘", "done": "✓", "archived": "—",
}

_TASK_DICT_FIELDS = (
    "id", "title", "body", "assignee", "status", "priority", "tenant",
    "workspace_kind", "workspace_path", "branch_name", "project_id",
    "created_by", "created_at", "started_at", "completed_at", "result",
    "skills", "max_retries", "model_override", "provider_override",
    "session_id", "workflow_template_id", "current_step_key",
)
_SHOW_RUN_FIELDS = (
    "id", "profile", "step_key", "status", "outcome", "summary", "error",
    "metadata", "worker_pid", "started_at", "ended_at",
)
_RUNS_RUN_FIELDS = (
    "id", "profile", "status", "outcome", "started_at", "ended_at",
    "summary", "error", "metadata", "worker_pid", "step_key",
)
_ATTACHMENT_FIELDS = ("id", "filename", "content_type", "size", "uploaded_by", "stored_path", "created_at")


def _fmt_ts(ts: Optional[int]) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else ""


def _print_json(obj: Any, *, ascii: bool = False) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=ascii))


def _json_out(args: argparse.Namespace, obj: Any, *, ascii: bool = False) -> bool:
    """Print ``obj`` as JSON and return True when ``--json`` was passed."""
    if not getattr(args, "json", False):
        return False
    _print_json(obj, ascii=ascii)
    return True


def _fmt_counts(counts: dict, empty: str = "") -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or empty


def _err(msg: str, rc: int = 1) -> int:
    print(msg, file=sys.stderr)
    return rc


def _bulk_apply(ids: Iterable[str], op: Callable[[str], Any],
                ok_msg: Callable[[str], str], fail_msg: Callable[[str], str]) -> int:
    """Run ``op(tid) -> bool`` per id, print ok/fail lines, exit 1 if any failed."""
    failed = False
    for tid in ids:
        if op(tid):
            print(ok_msg(tid))
        else:
            failed = True
            print(fail_msg(tid), file=sys.stderr)
    return 1 if failed else 0


def _fmt_task_line(t: kb.Task) -> str:
    icon = _STATUS_ICONS.get(t.status, "?")
    assignee = t.assignee or "(unassigned)"
    tenant = f" [{t.tenant}]" if t.tenant else ""
    return f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"


def _obj_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in fields}


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    d = _obj_dict(t, _TASK_DICT_FIELDS)
    d["skills"] = list(t.skills) if t.skills else []
    return d
