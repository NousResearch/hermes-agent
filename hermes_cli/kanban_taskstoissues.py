"""Deterministic Spec-Kit tasks.md -> kanban tickets support."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb


_TASK_RE = re.compile(r"^\s*[-*]\s+\[[ xX]\]\s+(T\d{3,})\b[:.)-]?\s*(.*)$")
_DEPEND_RE = re.compile(
    r"\bDepend(?:s|encies)?(?:\s+on)?\s*:\s*([Tt]\d{3,}(?:\s*,\s*[Tt]\d{3,})*)"
)
_TASK_ID_RE = re.compile(r"\b[Tt]\d{3,}\b")
_TITLE_STOP_RE = re.compile(r"^\s*(Depend(?:s|encies)?(?:\s+on)?|Satisfies)\s*:", re.I)


@dataclass(frozen=True)
class ParsedTask:
    task_id: str
    title: str
    body: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketPlanItem:
    task: ParsedTask
    idempotency_key: str
    existing_ticket_id: Optional[str] = None
    planned_ticket_id: Optional[str] = None
    dependency_ticket_ids: tuple[str, ...] = ()

    @property
    def ticket_id(self) -> Optional[str]:
        return self.existing_ticket_id or self.planned_ticket_id

    @property
    def action(self) -> str:
        return "existing" if self.existing_ticket_id else "create"


@dataclass(frozen=True)
class TicketPlan:
    spec_id: str
    root_ticket: str
    tasks_file: Path
    items: tuple[TicketPlanItem, ...]

    @property
    def creates(self) -> int:
        return sum(1 for item in self.items if item.action == "create")

    @property
    def existing(self) -> int:
        return sum(1 for item in self.items if item.action == "existing")


@dataclass(frozen=True)
class ApplyResult:
    spec_id: str
    root_ticket: str
    tasks_file: Path
    created_ticket_ids: tuple[str, ...]
    existing_ticket_ids: tuple[str, ...]
    task_ticket_ids: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.task_ticket_ids)


def resolve_tasks_file(
    spec_id: str,
    *,
    spec_root: Optional[Path | str] = None,
    tasks_file: Optional[Path | str] = None,
) -> Path:
    if tasks_file:
        return Path(tasks_file).expanduser()
    if not spec_id:
        raise ValueError("spec_id is required")
    root = Path(spec_root).expanduser() if spec_root else Path(".specify/specs")
    return root / spec_id / "tasks.md"


def parse_tasks_md(path: Path | str) -> list[ParsedTask]:
    tasks_path = Path(path).expanduser()
    text = tasks_path.read_text(encoding="utf-8")
    parsed: list[ParsedTask] = []
    current_id: Optional[str] = None
    current_title: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_id, current_title, current_lines
        if current_id is None or current_title is None:
            return
        body = "\n".join(line.rstrip() for line in current_lines).strip()
        title_parts = [current_title.strip()]
        for line in current_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _TITLE_STOP_RE.match(stripped):
                break
            title_parts.append(stripped)
        title = " ".join(part for part in title_parts if part).strip()
        depends: list[str] = []
        for match in _DEPEND_RE.finditer("\n".join([current_title, body])):
            for dep in _TASK_ID_RE.findall(match.group(1)):
                dep_id = dep.upper()
                if dep_id != current_id and dep_id not in depends:
                    depends.append(dep_id)
        parsed.append(
            ParsedTask(
                task_id=current_id,
                title=title,
                body=body,
                depends_on=tuple(depends),
            )
        )
        current_id = None
        current_title = None
        current_lines = []

    for raw in text.splitlines():
        match = _TASK_RE.match(raw)
        if match:
            flush()
            current_id = match.group(1).upper()
            current_title = match.group(2).strip()
            current_lines = []
            continue
        if current_id is not None:
            current_lines.append(raw)
    flush()

    seen: set[str] = set()
    for task in parsed:
        if not task.title:
            raise ValueError(f"{task.task_id} is missing a title")
        if task.task_id in seen:
            raise ValueError(f"duplicate task id {task.task_id}")
        seen.add(task.task_id)
    known = {task.task_id for task in parsed}
    missing = sorted({dep for task in parsed for dep in task.depends_on if dep not in known})
    if missing:
        raise ValueError(f"unknown dependency task id(s): {', '.join(missing)}")
    return parsed


def idempotency_key(spec_id: str, task_id: str) -> str:
    return f"tasks-to-issues:{spec_id}:{task_id.upper()}"


def _existing_by_key(conn: sqlite3.Connection, keys: Iterable[str]) -> dict[str, str]:
    keys = tuple(keys)
    if not keys:
        return {}
    placeholders = ",".join("?" for _ in keys)
    rows = conn.execute(
        "SELECT idempotency_key, id FROM tasks "
        f"WHERE idempotency_key IN ({placeholders}) AND status != 'archived'",
        keys,
    ).fetchall()
    return {row["idempotency_key"]: row["id"] for row in rows}


def build_plan(
    conn: sqlite3.Connection,
    *,
    spec_id: str,
    root_ticket: str,
    tasks_file: Path | str,
) -> TicketPlan:
    tasks_path = Path(tasks_file).expanduser()
    tasks = parse_tasks_md(tasks_path)
    existing = _existing_by_key(conn, [idempotency_key(spec_id, t.task_id) for t in tasks])
    task_to_ticket: dict[str, str] = {}
    items: list[TicketPlanItem] = []
    for task in tasks:
        key = idempotency_key(spec_id, task.task_id)
        existing_id = existing.get(key)
        if existing_id:
            task_to_ticket[task.task_id] = existing_id
        dep_ids = tuple(task_to_ticket[dep] for dep in task.depends_on if dep in task_to_ticket)
        items.append(
            TicketPlanItem(
                task=task,
                idempotency_key=key,
                existing_ticket_id=existing_id,
                dependency_ticket_ids=dep_ids,
            )
        )
    return TicketPlan(spec_id=spec_id, root_ticket=root_ticket, tasks_file=tasks_path, items=tuple(items))


def _ticket_body(*, spec_id: str, root_ticket: str, task: ParsedTask) -> str:
    parts = [
        f"Spec: {spec_id}",
        f"Root ticket: {root_ticket}",
        f"Source task: {task.task_id}",
        "",
        task.body.strip() if task.body else task.title,
        "",
        f"tasks-to-issues:{spec_id}:{task.task_id}",
    ]
    return "\n".join(parts).strip()


def _link_if_missing(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    row = conn.execute(
        "SELECT 1 FROM task_links WHERE parent_id = ? AND child_id = ?",
        (parent_id, child_id),
    ).fetchone()
    if row:
        return
    kb.link_tasks(conn, parent_id, child_id)


def apply_tasks_to_issues(
    conn: sqlite3.Connection,
    *,
    spec_id: str,
    root_ticket: str,
    tasks_file: Path | str,
    author: str = "taskstoissues",
) -> ApplyResult:
    if not kb.get_task(conn, root_ticket):
        raise ValueError(f"root ticket {root_ticket} not found")

    tasks_path = Path(tasks_file).expanduser()
    tasks = parse_tasks_md(tasks_path)
    task_ticket_ids: dict[str, str] = {}
    created: list[str] = []
    existing: list[str] = []

    for task in tasks:
        key = idempotency_key(spec_id, task.task_id)
        found = _existing_by_key(conn, [key]).get(key)
        if found:
            ticket_id = found
            existing.append(ticket_id)
        else:
            ticket_id = kb.create_task(
                conn,
                title=f"{task.task_id}: {task.title}",
                body=_ticket_body(spec_id=spec_id, root_ticket=root_ticket, task=task),
                created_by=author,
                idempotency_key=key,
                workspace_kind="scratch",
            )
            created.append(ticket_id)
        task_ticket_ids[task.task_id] = ticket_id

    for task in tasks:
        child_id = task_ticket_ids[task.task_id]
        for dep in task.depends_on:
            _link_if_missing(conn, task_ticket_ids[dep], child_id)
        _link_if_missing(conn, child_id, root_ticket)

    if created:
        kb.add_comment(
            conn,
            root_ticket,
            author,
            "Created Spec-Kit task tickets: " + ", ".join(created),
        )

    return ApplyResult(
        spec_id=spec_id,
        root_ticket=root_ticket,
        tasks_file=tasks_path,
        created_ticket_ids=tuple(created),
        existing_ticket_ids=tuple(existing),
        task_ticket_ids=task_ticket_ids,
    )


def plan_to_dict(plan: TicketPlan) -> dict[str, Any]:
    return {
        "spec_id": plan.spec_id,
        "root_ticket": plan.root_ticket,
        "tasks_file": str(plan.tasks_file),
        "creates": plan.creates,
        "existing": plan.existing,
        "items": [
            {
                "task_id": item.task.task_id,
                "title": item.task.title,
                "depends_on": list(item.task.depends_on),
                "idempotency_key": item.idempotency_key,
                "action": item.action,
                "ticket_id": item.ticket_id,
            }
            for item in plan.items
        ],
    }


def result_to_dict(result: ApplyResult) -> dict[str, Any]:
    return {
        "spec_id": result.spec_id,
        "root_ticket": result.root_ticket,
        "tasks_file": str(result.tasks_file),
        "created": list(result.created_ticket_ids),
        "existing": list(result.existing_ticket_ids),
        "task_ticket_ids": dict(result.task_ticket_ids),
        "total": result.total,
    }
