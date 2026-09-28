"""Text / ``--json`` output helpers shared by the ``hermes kanban`` CLI modules."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Callable, Iterable, Optional

from hermes_cli import kanban_db as kb

_STATUS_ICONS = {
    "todo": "◻", "ready": "▶", "running": "●", "scheduled": "⏱",
    "blocked": "⊘", "done": "✓", "archived": "—",
}

# Compact failure-text classifier: tells ``_fmt_task_line`` why a ready card is
# sitting in ``ready`` instead of being dispatched. Mirrors the split in
# ``kanban_db_dispatch.check_respawn_guard`` so the CLI surfaces the same
# information without invoking the dispatcher. Catches the same patterns:
#   "quota_cooldown" — quota / billing / rate-limit text (transient)
#   "blocker_auth"   — 401 / forbidden / token-missing (terminal)
#   None             — no failure text or unknown reason
#
# Operators reading ``hermes kanban list`` should be able to spot a parked
# card WITHOUT running ``hermes kanban dispatch --json`` (#t_cfbbb112, AC#3).
#
# The auth sniff is the SAME tight set as ``kanban_db_dispatch._RESPAWN_AUTH_RE``
# so a card whose TITLE / BODY happens to mention "authentication" or
# "authorise" is NOT misclassified here (#t_258dc2c3, t_961892f2 live proof).
_QUOTA_SNIFF_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit(?:ed)?|429|billing|subscription|"
    r"out[\s_]of[\s_]credits|entitlement[\s_]exhausted|exhausted|"
    r"token[\s_]plan)\b",
    re.IGNORECASE,
)
_AUTH_SNIFF_RE = re.compile(
    r"(?:"
    r"\b(?:401|403)(?:[\s_]forbidden)?\b|"
    r"\b(?:"
    r"unauthorized|"
    r"invalid[\s_]api[\s_]key|"
    r"access[\s_]denied|"
    r"access[\s_]forbidden|"
    r"request[\s_]forbidden|"
    r"permission[\s_]denied|"
    r"authentication[\s_]required|"
    r"auth[\s_](?:token|error|failed|header)|"
    r"missing[\s_](?:api[\s_]key|credentials|token)|"
    r"credential[s]?[\s_]missing|"
    r"credential[s]?[\s_](?:not[\s_]set|invalid|expired)|"
    r"credential[s]?[\s_]not[\s_]configured"
    r")\b"
    r")",
    re.IGNORECASE,
)

_TASK_DICT_FIELDS = (
    "id", "title", "body", "assignee", "status", "priority", "tenant",
    "workspace_kind", "workspace_path", "branch_name", "project_id",
    "created_by", "created_at", "started_at", "completed_at", "result",
    "skills", "max_runtime_seconds", "max_retries", "model_override", "provider_override",
    "session_id", "workflow_template_id", "current_step_key", "completion_contract", "last_failure_error",
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
    line = f"{icon} {t.id}  {t.status:8s}  {assignee:20s}{tenant}  {t.title}"
    # Surface why a ready card isn't dispatching. Operators don't have to run
    # ``hermes kanban dispatch --json`` to spot a quota wall or a 401 (#t_cfbbb112).
    parked = _parked_reason(t)
    if parked:
        line += f"  [parked: {parked}]"
    return line


def _parked_reason(t: kb.Task) -> Optional[str]:
    """Classify a task's ``last_failure_error`` text using the same patterns as
    ``kanban_db_dispatch.check_respawn_guard``. Returns a short human tag like
    ``"quota_cooldown"`` or ``"blocker_auth"`` for tasks the dispatcher would
    defer (status=ready, with stamped failure text). Returns ``None`` for tasks
    that aren't parked or whose reason the rules don't classify."""
    if t.status != "ready":
        return None
    err = getattr(t, "last_failure_error", None)
    if not err:
        return None
    # Auth is checked first — auth text in a quota-flavoured message (rare)
    # should surface as terminal, not transient.
    if _AUTH_SNIFF_RE.search(err):
        return "blocker_auth"
    if _QUOTA_SNIFF_RE.search(err):
        return "quota_cooldown"
    return None


def _obj_dict(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: getattr(obj, k) for k in fields}


def _task_to_dict(t: kb.Task) -> dict[str, Any]:
    d = _obj_dict(t, _TASK_DICT_FIELDS)
    d["skills"] = list(t.skills) if t.skills else []
    return d
