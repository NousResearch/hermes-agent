"""Slack-to-Kanban intake helpers.

This module is intentionally small and dependency-light so the Slack gateway
adapter can create Kanban cards directly without starting an agent turn or
exposing terminal/code execution from Slack.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Optional


_ALLOWED_COLUMNS = {"triage", "todo"}
_COLUMN_ALIASES = {
    "triage": "triage",
    "inbox": "triage",
    "spec": "triage",
    "specify": "triage",
    "todo": "todo",
    "to-do": "todo",
    "to_do": "todo",
}


@dataclass(frozen=True)
class SlackKanbanCreateRequest:
    title: str
    body: str = ""
    column: str = "triage"
    assignee: Optional[str] = None
    board: Optional[str] = None
    tenant: Optional[str] = None
    priority: int = 0


@dataclass(frozen=True)
class SlackKanbanCreateResult:
    task_id: str
    title: str
    column: str
    status: str
    board: str


def _strip_surrounding_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _normalize_column(raw: str) -> str:
    value = (raw or "").strip().lower()
    value = value.removeprefix("#")
    column = _COLUMN_ALIASES.get(value)
    if column not in _ALLOWED_COLUMNS:
        allowed = ", ".join(sorted(_ALLOWED_COLUMNS))
        raise ValueError(f"column must be one of: {allowed}")
    return column


def _parse_tokenized(text: str) -> tuple[dict[str, str], list[str]]:
    try:
        tokens = shlex.split(text, posix=True)
    except ValueError as exc:
        raise ValueError(f"could not parse quoted input: {exc}") from exc

    fields: dict[str, str] = {}
    title_parts: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        lowered = token.lower()
        if lowered in {"--todo", "todo"}:
            fields["column"] = "todo"
        elif lowered in {"--triage", "triage"}:
            fields["column"] = "triage"
        elif lowered.startswith("--") and "=" in lowered:
            key, value = token[2:].split("=", 1)
            fields[key.lower().replace("-", "_")] = value
        elif "=" in token and not token.startswith("="):
            key, value = token.split("=", 1)
            fields[key.lower().replace("-", "_")] = value
        elif lowered.startswith("--"):
            key = lowered[2:].replace("-", "_")
            if key in {"title", "body", "column", "assignee", "board", "tenant", "priority"}:
                if i + 1 >= len(tokens):
                    raise ValueError(f"{token} requires a value")
                fields[key] = tokens[i + 1]
                i += 1
            else:
                title_parts.append(token)
        else:
            title_parts.append(token)
        i += 1
    return fields, title_parts


def parse_slack_kanban_intake(text: str) -> SlackKanbanCreateRequest:
    """Parse `/kanban-add` text into a bounded Kanban create request.

    Supported examples:
    - `/kanban-add Fix login bug`
    - `/kanban-add column=todo title="Fix login" body="Steps..." assignee=default`
    - `/kanban-add --todo "Fix login"`
    - multi-line: first line title, remaining lines body
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("title is required. Example: /kanban-add Fix login bug")

    fields, title_parts = _parse_tokenized(raw)

    # If no explicit body was provided and the raw input is multi-line, keep
    # the first non-empty line as title and remaining lines as body. This is
    # useful when users paste a brief spec into Slack.
    body_from_lines = ""
    if "title" not in fields and "body" not in fields and "\n" in raw:
        lines = [line.rstrip() for line in raw.splitlines()]
        nonempty = [line for line in lines if line.strip()]
        if nonempty:
            fields["title"] = nonempty[0].strip()
            body_from_lines = "\n".join(nonempty[1:]).strip()
            title_parts = []

    title = _strip_surrounding_quotes(fields.get("title", "") or " ".join(title_parts))
    body = _strip_surrounding_quotes(fields.get("body", "") or body_from_lines)
    column = _normalize_column(fields.get("column", "triage"))

    if not title.strip():
        raise ValueError("title is required. Example: /kanban-add Fix login bug")
    if len(title) > 200:
        raise ValueError("title must be 200 characters or fewer")
    if len(body) > 8000:
        raise ValueError("body must be 8000 characters or fewer")

    priority = 0
    if fields.get("priority") not in (None, ""):
        try:
            priority = int(fields["priority"])
        except ValueError as exc:
            raise ValueError("priority must be an integer") from exc

    def optional(name: str) -> Optional[str]:
        value = _strip_surrounding_quotes(fields.get(name, ""))
        return value.strip() or None

    return SlackKanbanCreateRequest(
        title=title.strip(),
        body=body,
        column=column,
        assignee=optional("assignee"),
        board=optional("board"),
        tenant=optional("tenant"),
        priority=priority,
    )


def create_slack_kanban_task(
    request: SlackKanbanCreateRequest,
    *,
    created_by: str = "slack",
) -> SlackKanbanCreateResult:
    """Create the requested Kanban task and return the created id/status."""
    from hermes_cli import kanban_db

    board = request.board or kanban_db.get_current_board()
    conn = kanban_db.connect(board=board)
    try:
        task_id = kanban_db.create_task(
            conn,
            title=request.title,
            body=request.body or None,
            assignee=request.assignee,
            created_by=created_by,
            tenant=request.tenant,
            priority=request.priority,
            triage=request.column == "triage",
            board=board,
        )

        if request.column == "todo":
            with kanban_db.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    ("todo", task_id),
                )
                kanban_db._append_event(  # internal helper; keeps event log truthful
                    conn,
                    task_id,
                    "status_changed",
                    {"status": "todo", "source": "slack_kanban_intake"},
                )

        task = kanban_db.get_task(conn, task_id)
        status = task.status if task else request.column
        return SlackKanbanCreateResult(
            task_id=task_id,
            title=request.title,
            column=request.column,
            status=status,
            board=board,
        )
    finally:
        conn.close()


def is_kanban_add_text(text: str) -> bool:
    return bool(re.match(r"^\s*(?:kanban-add|kanban_add|add-kanban)\b", text or "", re.I))
