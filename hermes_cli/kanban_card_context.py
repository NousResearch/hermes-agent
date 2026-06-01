"""Build deterministic Kanban-card context packets for external coding agents."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from hermes_cli import kanban_db


class CardContextError(RuntimeError):
    """Raised when a card cannot be converted into launchable agent context."""


def _trim(text: Optional[str], limit: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n…[truncated]"


def _workspace_path(task: kanban_db.Task) -> Path:
    if not task.workspace_path:
        if task.workspace_kind == "scratch":
            return Path.home().resolve()
        raise CardContextError(
            f"task {task.id} has no workspace_path; set a concrete repo/workspace before launching Claude Code"
        )
    path = Path(task.workspace_path).expanduser()
    if not path.exists():
        raise CardContextError(f"task {task.id} workspace_path does not exist: {path}")
    if not path.is_dir():
        raise CardContextError(f"task {task.id} workspace_path is not a directory: {path}")
    return path.resolve()


def build_card_context(task_id: str, *, board: Optional[str] = None) -> dict[str, Any]:
    """Return canonical card data plus a launchable workspace path.

    This intentionally fails when the card lacks a real workspace. Claude Code
    sessions should not silently fall back to an unrelated repo.
    """

    kanban_db.init_db(board=board)
    conn = kanban_db.connect(board=board)
    try:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise CardContextError(f"task {task_id} not found")
        workspace = _workspace_path(task)
        list_criteria = getattr(kanban_db, "list_acceptance_criteria", None)
        criteria = list_criteria(conn, task_id) if callable(list_criteria) else []
        task_dict = asdict(task)
        return {
            "task": task_dict,
            "workspace_path": str(workspace),
            "acceptance_criteria": criteria,
            "latest_summary": kanban_db.latest_summary(conn, task_id),
            "links": {
                "parents": [r["parent_id"] for r in conn.execute("SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,)).fetchall()],
                "children": [r["child_id"] for r in conn.execute("SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id", (task_id,)).fetchall()],
            },
        }
    finally:
        conn.close()


def build_claude_prompt(context: dict[str, Any]) -> str:
    """Build the initial Claude Code prompt for a card-scoped session."""

    task = context["task"]
    links = context.get("links") or {}
    criteria = context.get("acceptance_criteria") or []
    lines: list[str] = [
        "You are Claude Code working on a Rolly Kanban card.",
        "Do exactly the card asks: no scope creep, no mock implementations, fail fast.",
        "If acceptance/verification is unclear, stop and explain the blocker instead of guessing.",
        "Respect global Rolly/MIX constraints: preserve user work, do not reset/stash, run narrow verification, and report exact files/tests.",
        "",
        f"Card: {task.get('id')} — {task.get('title')}",
        f"Status: {task.get('status')} | Priority: {task.get('priority')} | Tenant: {task.get('tenant') or 'none'} | Assignee: {task.get('assignee') or 'none'}",
        f"Workspace: {context.get('workspace_path')}",
        f"Execution mode: {task.get('execution_mode') or 'manual Claude Code'}",
        "",
        "## Body",
        _trim(task.get("body"), 6000) or "(empty)",
    ]
    if context.get("latest_summary"):
        lines += ["", "## Latest worker summary", _trim(context.get("latest_summary"), 2000)]
    lines += [
        "",
        "## Acceptance criteria",
    ]
    if criteria:
        for item in criteria:
            lines.append(
                f"- [{ 'x' if item.get('passed') else ' ' }] ({item.get('verifier')}) "
                f"{_trim(item.get('text'), 1000)}"
                + (f" — evidence: {_trim(item.get('evidence'), 700)}" if item.get('evidence') else "")
            )
    else:
        lines.append("(none available in this board schema — rely on the card body and stop if acceptance is unclear)")
    lines += [
        "",
        "## Links",
        f"Parents: {', '.join(links.get('parents') or []) or 'none'}",
        f"Children: {', '.join(links.get('children') or []) or 'none'}",
    ]
    lines += [
        "",
        "## Expected behavior",
        "- First inspect the repo and card context.",
        "- Make only changes needed for this card.",
        "- Run the card's verification commands or the narrowest relevant tests.",
        "- End with a concise handoff: files changed, tests run, remaining blockers.",
    ]
    return "\n".join(lines).strip() + "\n"


def build_claude_card_launch(task_id: str, *, board: Optional[str] = None) -> dict[str, Any]:
    context = build_card_context(task_id, board=board)
    return {
        "task_id": context["task"]["id"],
        "workspace_path": context["workspace_path"],
        "prompt": build_claude_prompt(context),
    }
