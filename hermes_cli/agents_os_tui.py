"""Interactive Mission Control TUI for local Agents OS.

The curses wrapper in this module is intentionally thin.  State transitions,
queries, rendering, and local actions are plain Python functions/classes so they
can be tested without a terminal.  All writes stay inside the existing local
Agents OS SQLite/artifact paths; there are no network, deploy, server, gateway,
or credential side effects.
"""

from __future__ import annotations

import argparse
import curses
import json
import textwrap
import uuid
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

from hermes_cli import agents_os

VIEWS = ["tasks", "next", "approvals", "runs", "events", "doctor"]
VIEW_TITLES = {
    "tasks": "Tasks",
    "next": "Next",
    "approvals": "Approvals",
    "runs": "Runs",
    "events": "Events",
    "doctor": "Doctor / Mirror",
}
VIEW_KEYS = {"1": "tasks", "2": "next", "3": "approvals", "4": "runs", "5": "events", "6": "doctor"}


@dataclass(frozen=True)
class MissionControlState:
    view: str = "tasks"
    selected: int = 0
    message: str = ""
    quit: bool = False


def apply_key(state: MissionControlState, key: str, item_count: int) -> MissionControlState:
    """Pure navigation reducer for keyboard-testable TUI behaviour."""
    if key in {"q", "Q", "esc"}:
        return replace(state, quit=True)
    if key in VIEW_KEYS:
        return MissionControlState(view=VIEW_KEYS[key], selected=0, message=state.message)
    if key in {"j", "down", "KEY_DOWN"}:
        return replace(state, selected=min(max(item_count - 1, 0), state.selected + 1))
    if key in {"k", "up", "KEY_UP"}:
        return replace(state, selected=max(0, state.selected - 1))
    if key in {"g", "home", "KEY_HOME"}:
        return replace(state, selected=0)
    if key in {"G", "end", "KEY_END"}:
        return replace(state, selected=max(item_count - 1, 0))
    return state


def _row(row: Any) -> dict[str, Any]:
    return agents_os.row_to_dict(row)


class MissionControlCore:
    """Local-only data/action facade used by both tests and curses UI."""

    def __init__(self, paths: agents_os.AgentsOSPaths | None = None):
        self.paths = paths or agents_os.resolve_paths(None)

    def items_for_view(self, view: str) -> list[dict[str, Any]]:
        with agents_os.connect(self.paths) as conn:
            if view == "tasks":
                return [_row(r) for r in conn.execute(
                    """
                    SELECT * FROM tasks
                    ORDER BY CASE status
                        WHEN 'blocked' THEN 0 WHEN 'needs_approval' THEN 1
                        WHEN 'review' THEN 2 WHEN 'in_progress' THEN 3
                        WHEN 'ready' THEN 4 WHEN 'pending' THEN 5
                        WHEN 'routed' THEN 6 WHEN 'new' THEN 7
                        WHEN 'completed' THEN 8 ELSE 9 END,
                        priority ASC, created_at ASC
                    LIMIT 100
                    """
                ).fetchall()]
            if view == "next":
                task = conn.execute(
                    "SELECT * FROM tasks WHERE status IN ('ready','pending','routed') AND approval_required=0 "
                    "ORDER BY CASE status WHEN 'ready' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, priority ASC, created_at ASC LIMIT 1"
                ).fetchone()
                return [_row(task)] if task else []
            if view == "approvals":
                return [_row(r) for r in conn.execute(
                    "SELECT * FROM approvals ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, created_at ASC LIMIT 100"
                ).fetchall()]
            if view == "runs":
                return [_row(r) for r in conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 100").fetchall()]
            if view == "events":
                return [_row(r) for r in conn.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT 100").fetchall()]
            if view == "doctor":
                schema = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
                pending = conn.execute("SELECT COUNT(*) FROM approvals WHERE status='pending'").fetchone()[0]
                orphan_artifacts = conn.execute("SELECT COUNT(*) FROM artifacts WHERE task_id IS NOT NULL AND task_id != '' AND task_id NOT IN (SELECT id FROM tasks)").fetchone()[0]
            dashboard_path = self.paths.vault_root / "00-command-center" / "RUNTIME-DASHBOARD.md"
            mirror_ok = dashboard_path.exists()
            root_text = str(self.paths.root).lower()
            policy_ok = str(self.paths.root).startswith(str(self.paths.home)) and not any(marker in root_text for marker in ("separate-profile", "external-runtime", "shared-runtime"))
            return [
                {"id": "doctor", "status": "ok" if policy_ok and orphan_artifacts == 0 else "attention", "schema_version": schema, "pending_approvals": pending, "orphan_artifacts": orphan_artifacts, "policy_home_isolated": policy_ok},
                {"id": "mirror", "status": "ok" if mirror_ok else "attention", "dashboard_path": str(dashboard_path), "exists": mirror_ok},
                {"id": "safety", "status": "ok", "network_side_effects": False, "deploy": False, "gateway_restart": False},
            ]
        return []

    def counts(self) -> dict[str, Any]:
        with agents_os.connect(self.paths) as conn:
            return {
                "tasks": dict(conn.execute("SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall()),
                "approvals": dict(conn.execute("SELECT status, COUNT(*) c FROM approvals GROUP BY status").fetchall()),
                "runs": conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            }

    def detail_for_state(self, state: MissionControlState) -> dict[str, Any] | None:
        items = self.items_for_view(state.view)
        if not items:
            return None
        return items[min(state.selected, len(items) - 1)]

    def screen_payload(self, state: MissionControlState) -> dict[str, Any]:
        items = self.items_for_view(state.view)
        return {state.view: items, "counts": self.counts(), "detail": items[min(state.selected, len(items) - 1)] if items else None}

    def create_task(self, title: str, workflow: str = "code-task", priority: int = 3, notes: str = "") -> dict[str, Any]:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        now = agents_os.utc_now()
        spec = agents_os.SAFE_WORKFLOWS.get(workflow, {})
        approval_required = 1 if spec.get("requires_approval") else 0
        status = "needs_approval" if approval_required else "pending"
        with agents_os.connect(self.paths) as conn:
            conn.execute(
                "INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,approval_required) VALUES(?,?,?,?,?,?,?,?,?)",
                (task_id, title, status, workflow, priority, now, now, notes, approval_required),
            )
            agents_os.log_event(conn, "task_created", task_id=task_id, payload={"source": "mission_control_tui", "workflow": workflow, "status": status})
            conn.commit()
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            return _row(row)

    def close_task(self, task_id: str, evidence: str) -> dict[str, Any]:
        evidence = evidence.strip()
        if not evidence:
            return {"task_id": task_id, "status": "error", "reason": "evidence_required"}
        with agents_os.connect(self.paths) as conn:
            task = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                return {"task_id": task_id, "status": "error", "reason": "task_not_found"}
            if task["approval_required"] or agents_os._pending_approval_count(conn, task_id) > 0 or task["status"] == "needs_approval":
                agents_os.log_event(conn, "close_blocked", task_id=task_id, payload={"reason": "approval_required", "source": "mission_control_tui"})
                conn.commit()
                return {"task_id": task_id, "status": "blocked", "reason": "approval_required"}
            now = agents_os.utc_now()
            conn.execute("UPDATE tasks SET status='completed', updated_at=? WHERE id=?", (now, task_id))
            agents_os.log_event(conn, "task_closed", task_id=task_id, payload={"evidence": evidence, "source": "mission_control_tui"})
            conn.commit()
        return {"task_id": task_id, "status": "completed", "evidence": evidence}

    def resolve_approval(self, approval_id: str, status: str, notes: str = "") -> dict[str, Any]:
        if status not in {"approved", "rejected", "cancelled"}:
            return {"approval_id": approval_id, "status": "error", "reason": "invalid_approval_status"}
        with agents_os.connect(self.paths) as conn:
            approval = conn.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
            if approval is None:
                return {"approval_id": approval_id, "status": "error", "reason": "approval_not_found"}
            now = agents_os.utc_now()
            conn.execute("UPDATE approvals SET status=?, resolved_at=? WHERE id=?", (status, now, approval_id))
            task_id = approval["task_id"]
            if task_id:
                if status == "approved":
                    pending = conn.execute("SELECT COUNT(*) FROM approvals WHERE task_id=? AND status='pending' AND id != ?", (task_id, approval_id)).fetchone()[0]
                    if pending == 0:
                        conn.execute("UPDATE tasks SET approval_required=0, status=CASE WHEN status='needs_approval' THEN 'ready' ELSE status END, updated_at=? WHERE id=?", (now, task_id))
                elif status == "rejected":
                    conn.execute("UPDATE tasks SET status='blocked', updated_at=? WHERE id=?", (now, task_id))
                agents_os.log_event(conn, "approval_resolved", task_id=task_id, payload={"approval_id": approval_id, "status": status, "notes": notes, "source": "mission_control_tui"})
            conn.commit()
        return {"approval_id": approval_id, "status": status, "task_id": task_id, "notes": notes}


def _format_item(view: str, item: dict[str, Any]) -> str:
    if view in {"tasks", "next"}:
        gate = " gate" if item.get("approval_required") else ""
        return f"{item.get('id')} [{item.get('status')}] p{item.get('priority')} {item.get('title')}{gate}"
    if view == "approvals":
        return f"{item.get('id')} [{item.get('status')}] {item.get('risk')} task={item.get('task_id') or '-'} {item.get('title')}"
    if view == "runs":
        return f"{item.get('id')} [{item.get('status')}] workflow={item.get('workflow')} task={item.get('task_id') or '-'}"
    if view == "events":
        return f"{item.get('id')} {item.get('event_type')} task={item.get('task_id') or '-'}"
    return f"{item.get('id')} [{item.get('status')}]"


def _detail_lines(detail: dict[str, Any] | None) -> list[str]:
    if not detail:
        return ["No item selected."]
    lines: list[str] = []
    for key, value in detail.items():
        if isinstance(value, str) and len(value) > 120:
            value = value[:117] + "..."
        lines.append(f"{key}: {value}")
    return lines


def _wrap(lines: Iterable[str], width: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if len(line) <= width:
            out.append(line)
        else:
            out.extend(textwrap.wrap(line, width=width, replace_whitespace=False) or [""])
    return out


def render_screen(state: MissionControlState, payload: dict[str, Any], width: int = 100, height: int = 30) -> str:
    """Render a deterministic text frame for curses or scripted CLI tests."""
    width = max(width, 60)
    height = max(height, 12)
    items = payload.get(state.view, [])
    counts = payload.get("counts", {})
    header = f"Agents OS Mission Control — {VIEW_TITLES[state.view]}"
    tabs = "  ".join(f"[{idx}] {VIEW_TITLES[name]}" for idx, name in enumerate(VIEWS, start=1))
    summary = f"tasks={counts.get('tasks', {})} approvals={counts.get('approvals', {})} runs={counts.get('runs', 0)} events={counts.get('events', 0)}"
    commands = "Commands: 1-6 views | j/k/up/down select | n new | c close+evidence | a approve | d deny | r refresh | q quit"
    split = max(30, width // 2)
    list_lines = [VIEW_TITLES[state.view]]
    if not items:
        list_lines.append("  (empty)")
    for idx, item in enumerate(items):
        marker = ">" if idx == state.selected else " "
        list_lines.append(f"{marker} {_format_item(state.view, item)}")
    detail = ["Detail"] + _detail_lines(payload.get("detail"))
    body: list[str] = [header[:width], tabs[:width], summary[:width], "-" * min(width, 120)]
    rows = max(1, height - 7)
    left = _wrap(list_lines, max(10, split - 2))[:rows]
    right = _wrap(detail, max(10, width - split - 3))[:rows]
    for i in range(max(len(left), len(right), 1)):
        l = left[i] if i < len(left) else ""
        r = right[i] if i < len(right) else ""
        body.append(f"{l:<{split}} | {r}"[:width])
    if state.message:
        body.append(("Message: " + state.message)[:width])
    body.append(commands[:width])
    return "\n".join(body[:height]) + "\n"


def _prompt(stdscr: Any, prompt: str) -> str:
    curses.echo()
    stdscr.addstr(curses.LINES - 2, 0, " " * (curses.COLS - 1))
    stdscr.addstr(curses.LINES - 2, 0, prompt[: curses.COLS - 2])
    stdscr.refresh()
    try:
        raw = stdscr.getstr(curses.LINES - 2, min(len(prompt), curses.COLS - 2), 240)
        return raw.decode("utf-8", errors="replace").strip()
    finally:
        curses.noecho()


def _key_name(code: int) -> str:
    if code == curses.KEY_DOWN:
        return "down"
    if code == curses.KEY_UP:
        return "up"
    if code == curses.KEY_HOME:
        return "home"
    if code == curses.KEY_END:
        return "end"
    if code == 27:
        return "esc"
    try:
        return chr(code)
    except ValueError:
        return str(code)


def run_curses(core: MissionControlCore) -> int:
    def _main(stdscr: Any) -> int:
        curses.curs_set(0)
        stdscr.keypad(True)
        state = MissionControlState()
        while not state.quit:
            payload = core.screen_payload(state)
            frame = render_screen(state, payload, curses.COLS - 1, curses.LINES - 1)
            stdscr.erase()
            for y, line in enumerate(frame.splitlines()[: curses.LINES - 1]):
                stdscr.addstr(y, 0, line[: curses.COLS - 1])
            stdscr.refresh()
            key = _key_name(stdscr.getch())
            if key == "n":
                title = _prompt(stdscr, "New task title: ")
                if title:
                    created = core.create_task(title=title)
                    state = MissionControlState(view="tasks", message=f"created {created['id']}")
                continue
            if key == "c":
                detail = core.detail_for_state(state)
                if detail and detail.get("id") and str(detail.get("id")).startswith("task-"):
                    evidence = _prompt(stdscr, "Close evidence: ")
                    result = core.close_task(str(detail["id"]), evidence)
                    state = replace(state, message=f"close {result.get('status')}: {result.get('reason') or result.get('task_id')}")
                continue
            if key in {"a", "d"}:
                detail = core.detail_for_state(state)
                if detail and detail.get("id") and str(detail.get("id")).startswith("approval-"):
                    result = core.resolve_approval(str(detail["id"]), "approved" if key == "a" else "rejected")
                    state = replace(state, message=f"approval {result.get('status')}: {detail['id']}")
                continue
            state = apply_key(state, key, len(payload.get(state.view, [])))
        return 0
    return curses.wrapper(_main)


def run_script(core: MissionControlCore, script: str, width: int = 100, height: int = 30) -> str:
    state = MissionControlState()
    frame = ""
    for ch in script:
        payload = core.screen_payload(state)
        if ch == "r":
            pass
        else:
            state = apply_key(state, ch, len(payload.get(state.view, [])))
        frame = render_screen(state, core.screen_payload(state), width=width, height=height)
        if state.quit:
            break
    if not frame:
        frame = render_screen(state, core.screen_payload(state), width=width, height=height)
    return frame


def tui_status_payload(core: MissionControlCore) -> dict[str, Any]:
    return {
        "status": "ok",
        "views": VIEWS,
        "counts": core.counts(),
        "paths": {"agents_os_home": str(core.paths.root), "state_db": str(core.paths.db), "vault_root": str(core.paths.vault_root)},
        "launcher": "export HERMES_HOME=/path/to/hermes-home && hermes agents-os tui",
        "safety": {"local_only": True, "network_side_effects": False, "deploy": False, "gateway_restart": False, "credentials_touched": False},
    }


def tui_cmd(args: argparse.Namespace) -> int:
    core = MissionControlCore(agents_os.resolve_paths(args))
    if getattr(args, "json", False):
        print(json.dumps(tui_status_payload(core), ensure_ascii=False, indent=2))
        return 0
    if getattr(args, "script", None) is not None:
        print(run_script(core, args.script, width=getattr(args, "width", 100), height=getattr(args, "height", 30)), end="")
        return 0
    return run_curses(core)
