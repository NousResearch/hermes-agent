"""Slack Block Kit view for Hermes Kanban boards.

This module is intentionally Slack-specific. The source of truth remains the
existing kanban SQLite DB and CLI/tooling; this file only renders and applies
small control-plane actions from Slack.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from dataclasses import asdict, dataclass
from typing import Any

from hermes_cli import kanban_db as kb


STATUS_ORDER = ("triage", "todo", "ready", "running", "blocked", "done")
MANUAL_MOVE_STATUSES = ("triage", "todo", "ready", "blocked", "done")
CREATE_TASK_STATUSES = ("triage", "todo", "ready")
EDITABLE_DETAIL_STATUSES = ("todo", "ready", "running", "blocked")
STATUS_LABELS = {
    "triage": "Triage",
    "todo": "Todo",
    "ready": "Ready",
    "running": "In Progress",
    "blocked": "Blocked",
    "done": "Done",
}
SYSTEM_DEFAULT_RESULTS = {
    "Completed from Slack /board.",
    "Completed from Slack /board",
}
APPROVAL_KEYWORDS = (
    "approval",
    "approve",
    "승인",
    "confirm",
    "confirmation",
)
PRIORITY_CHOICES = (
    (-1, "Low"),
    (0, "Normal"),
    (1, "Medium"),
    (2, "High"),
    (3, "Critical"),
)
PRIORITY_LABELS = dict(PRIORITY_CHOICES)


@dataclass
class BoardFilters:
    board: str | None = None
    status: str | None = None
    assignee: str | None = None
    tenant: str | None = None
    query: str | None = None
    approval_only: bool = False
    include_archived: bool = False
    limit: int = 5
    page: int = 0


@dataclass
class BoardCommand:
    filters: BoardFilters
    action: str = "view"
    render: str = "ui"
    text_detail: str = "list"
    task_id: str | None = None
    title: str | None = None
    public: bool = False
    natural_request: str | None = None


TASK_ID_RE = re.compile(r"\bt_[A-Za-z0-9_-]+\b")
STATUS_ALIASES = {
    "all": None,
    "triage": "triage",
    "inbox": "triage",
    "분류": "triage",
    "todo": "todo",
    "to-do": "todo",
    "할일": "todo",
    "할 일": "todo",
    "ready": "ready",
    "준비": "ready",
    "대기": "ready",
    "running": "running",
    "run": "running",
    "in-progress": "running",
    "in_progress": "running",
    "progress": "running",
    "진행": "running",
    "진행중": "running",
    "진행 중": "running",
    "blocked": "blocked",
    "block": "blocked",
    "막힘": "blocked",
    "차단": "blocked",
    "done": "done",
    "complete": "done",
    "completed": "done",
    "완료": "done",
}


def _normalize_status(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    return STATUS_ALIASES.get(key, key if key in kb.VALID_STATUSES else None)


def _consume_value(parts: list[str], idx: int) -> tuple[str | None, int]:
    nxt = parts[idx + 1] if idx + 1 < len(parts) else None
    if nxt and not nxt.startswith("-"):
        return nxt, idx + 2
    return None, idx + 1


def parse_board_args(rest: str) -> BoardFilters:
    """Parse `/board` args.

    Supported forms are intentionally small and deterministic:
    `/board --board product-launch --tenant acme --assignee writer --status ready`.
    """
    return parse_board_command(rest).filters


def parse_board_command(rest: str) -> BoardCommand:
    """Parse `/board` args into a board command.

    Explicit flags are deterministic. Remaining non-option text is kept as a
    natural request and mapped with lightweight Korean/English aliases.
    """
    filters = BoardFilters()
    command = BoardCommand(filters=filters)
    try:
        parts = shlex.split(rest or "")
    except ValueError:
        command.natural_request = (rest or "").strip() or None
        return _apply_natural_board_request(command)

    idx = 0
    natural_parts: list[str] = []
    while idx < len(parts):
        part = parts[idx]
        if part in {"-h", "--help"}:
            command.action = "help"
            command.render = "text"
            idx += 1
            continue
        if part in {"--archived"}:
            filters.include_archived = True
            idx += 1
            continue
        if part in {"-a", "--approval", "--approvals", "--approval-required"}:
            filters.approval_only = True
            idx += 1
            continue
        if part in {"-b", "--board"}:
            value, idx = _consume_value(parts, idx)
            if value:
                filters.board = value
            continue
        if part in {"-s", "--status"}:
            value, idx = _consume_value(parts, idx)
            filters.status = _normalize_status(value)
            continue
        if part in {"-u", "--assignee"}:
            value, idx = _consume_value(parts, idx)
            if value:
                filters.assignee = value
            continue
        if part in {"-p", "--project", "--tenant"}:
            value, idx = _consume_value(parts, idx)
            if value:
                filters.tenant = value
            continue
        if part in {"-q", "--query", "--search"}:
            value, idx = _consume_value(parts, idx)
            if value:
                filters.query = value
            continue
        if part in {"-l", "--limit"}:
            value, idx = _consume_value(parts, idx)
            try:
                filters.limit = max(1, min(10, int(value or "")))
            except (TypeError, ValueError):
                pass
            continue
        if part == "--page":
            value, idx = _consume_value(parts, idx)
            try:
                filters.page = max(0, int(value or "") - 1)
            except (TypeError, ValueError):
                pass
            continue
        if part in {"-t", "--text", "--plain"}:
            command.render = "text"
            idx += 1
            continue
        if part == "--summary":
            command.render = "text"
            command.text_detail = "summary"
            idx += 1
            continue
        if part == "--full":
            command.render = "text"
            command.text_detail = "full"
            idx += 1
            continue
        if part == "--public":
            command.public = True
            idx += 1
            continue
        if part == "--ephemeral":
            command.public = False
            idx += 1
            continue
        if part in {"-n", "--new"}:
            command.action = "new"
            value, idx = _consume_value(parts, idx)
            if value:
                command.title = value
            continue
        if part in {"-e", "--edit"}:
            command.action = "edit"
            value, idx = _consume_value(parts, idx)
            if value:
                command.task_id = value
            continue
        if part in {"-d", "--delete", "--archive"}:
            command.action = "delete"
            value, idx = _consume_value(parts, idx)
            if value:
                command.task_id = value
            continue
        if part in {"--detail", "--show", "-i"}:
            command.action = "detail"
            value, idx = _consume_value(parts, idx)
            if value:
                command.task_id = value
            continue
        natural_parts.append(part)
        idx += 1

    if filters.status and filters.status not in kb.VALID_STATUSES:
        filters.status = None
    if natural_parts:
        command.natural_request = " ".join(natural_parts).strip() or None
        command = _apply_natural_board_request(command)
    return command


def _extract_task_id(text: str) -> str | None:
    match = TASK_ID_RE.search(text or "")
    return match.group(0) if match else None


def _strip_task_id(text: str) -> str:
    return TASK_ID_RE.sub("", text or "").strip(" ,:·")


def _apply_natural_board_request(command: BoardCommand) -> BoardCommand:
    text = (command.natural_request or "").strip()
    if not text:
        return command
    lowered = text.lower()

    task_id = _extract_task_id(text)
    if task_id and not command.task_id:
        command.task_id = task_id

    if any(word in lowered for word in ("텍스트", "text", "plain", "목록", "보고")):
        command.render = "text"
    if any(word in lowered for word in ("요약", "summary", "brief")):
        command.render = "text"
        command.text_detail = "summary"
    if any(word in lowered for word in ("상세 보고", "full report", "full")):
        command.render = "text"
        command.text_detail = "full"
    if any(word in lowered for word in ("승인", "approval", "approve")):
        command.filters.approval_only = True

    for alias, status in STATUS_ALIASES.items():
        if status and alias and alias in lowered:
            command.filters.status = status
            break

    project_match = re.search(r"([A-Za-z0-9_.가-힣-]+)\s*프로젝트", text, re.I)
    if not project_match:
        project_match = re.search(r"(?:project|프로젝트)\s*[:=]?\s*([A-Za-z0-9_.가-힣-]+)", text, re.I)
    if project_match and not command.filters.tenant:
        command.filters.tenant = project_match.group(1)

    query_match = re.search(r"(?:query|search|검색)\s*[:=]?\s*(.+)$", text, re.I)
    if query_match and not command.filters.query:
        command.filters.query = query_match.group(1).strip()

    if any(word in lowered for word in ("추가", "생성", "등록", "new", "create", "add")):
        command.action = "new"
        if not command.title:
            title = _strip_task_id(text)
            title = re.sub(r"\b(new|create|add)\b", "", title, flags=re.I)
            title = re.sub(r"(추가|생성|등록)(해줘|하기|해|)$", "", title).strip(" ,:·")
            command.title = title or None
    elif any(word in lowered for word in ("삭제", "아카이브", "archive", "delete")):
        command.action = "delete"
    elif any(word in lowered for word in ("수정", "edit", "change")):
        command.action = "edit"
    elif task_id and any(word in lowered for word in ("상세", "detail", "show", "보기")):
        command.action = "detail"

    return command


def filters_from_value(value: str | None) -> BoardFilters:
    if not value:
        return BoardFilters()
    try:
        data = json.loads(value)
    except Exception:
        return BoardFilters()
    if isinstance(data, dict) and "filters" in data:
        data = data.get("filters") or {}
    if not isinstance(data, dict):
        return BoardFilters()
    if any(key in data for key in ("b", "f", "a", "n", "x", "q", "r", "l", "p")):
        data = {
            "board": data.get("b") or data.get("board") or None,
            "status": data.get("f") or data.get("status") or None,
            "assignee": data.get("a") or data.get("assignee") or None,
            "tenant": data.get("n") or data.get("tenant") or None,
            "query": data.get("x") or data.get("query") or None,
            "approval_only": bool(data.get("q", data.get("approval_only", False))),
            "include_archived": bool(data.get("r", data.get("include_archived", False))),
            "limit": data.get("l") or data.get("limit") or 5,
            "page": data.get("p") or data.get("page") or 0,
        }
    return BoardFilters(
        board=data.get("board") or None,
        status=data.get("status") or None,
        assignee=data.get("assignee") or None,
        tenant=data.get("tenant") or None,
        query=data.get("query") or None,
        approval_only=bool(data.get("approval_only", False)),
        include_archived=bool(data.get("include_archived", False)),
        limit=max(1, min(10, int(data.get("limit") or 5))),
        page=max(0, int(data.get("page") or 0)),
    )


def filters_value(filters: BoardFilters) -> str:
    return json.dumps(asdict(filters), ensure_ascii=False, separators=(",", ":"))


def compact_filters_dict(filters: BoardFilters, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    board = overrides.get("board", filters.board)
    status = overrides.get("status", filters.status)
    assignee = overrides.get("assignee", filters.assignee)
    tenant = overrides.get("tenant", filters.tenant)
    query = overrides.get("query", filters.query)
    approval_only = overrides.get("approval_only", filters.approval_only)
    include_archived = overrides.get("include_archived", filters.include_archived)
    limit = int(overrides.get("limit", filters.limit) or 5)
    page = int(overrides.get("page", filters.page) or 0)
    if board:
        data["b"] = board
    if status:
        data["f"] = status
    if assignee:
        data["a"] = assignee
    if tenant:
        data["n"] = tenant
    if query:
        data["x"] = query
    if approval_only:
        data["q"] = 1
    if include_archived:
        data["r"] = 1
    if limit != 5:
        data["l"] = max(1, min(10, limit))
    if page:
        data["p"] = max(0, page)
    return data


def compact_filters_value(filters: BoardFilters, **overrides: Any) -> str:
    return json.dumps(
        compact_filters_dict(filters, **overrides),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def action_value(action: str, task_id: str, filters: BoardFilters) -> str:
    return json.dumps(
        {"action": action, "task_id": task_id, "filters": asdict(filters)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def move_action_value(task_id: str, target_status: str, filters: BoardFilters) -> str:
    """Compact value for Slack static_select options.

    Slack caps option values at 150 characters. Keep the common board filters
    with one-letter keys, and fall back to task/status only for unusually long
    filter values.
    """
    payload: dict[str, Any] = {"t": task_id, "s": target_status}
    if filters.board:
        payload["b"] = filters.board
    if filters.status:
        payload["f"] = filters.status
    if filters.assignee:
        payload["a"] = filters.assignee
    if filters.tenant:
        payload["n"] = filters.tenant
    if filters.query:
        payload["x"] = filters.query
    if filters.approval_only:
        payload["q"] = 1
    if filters.include_archived:
        payload["r"] = 1
    if filters.limit != 5:
        payload["l"] = filters.limit
    if filters.page:
        payload["p"] = filters.page

    value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    if len(value) < 151:
        return value
    return json.dumps(
        {"t": task_id, "s": target_status},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def add_action_value(status: str, filters: BoardFilters) -> str:
    return json.dumps(
        {"action": "add", "status": status, "filters": asdict(filters)},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def page_action_value(filters: BoardFilters, page: int) -> str:
    return compact_filters_value(filters, page=max(0, int(page)))


def approval_filter_value(filters: BoardFilters, enabled: bool) -> str:
    return compact_filters_value(filters, approval_only=bool(enabled), page=0)


def parse_action_value(value: str | None) -> tuple[str, str, BoardFilters]:
    if not value:
        return ("", "", BoardFilters())
    try:
        data = json.loads(value)
    except Exception:
        return ("", "", BoardFilters())
    if not isinstance(data, dict):
        return ("", "", BoardFilters())
    return (
        str(data.get("action") or ""),
        str(data.get("task_id") or ""),
        filters_from_value(json.dumps(data.get("filters") or {})),
    )


def parse_move_action_value(value: str | None) -> tuple[str, str, BoardFilters]:
    if not value:
        return ("", "", BoardFilters())
    try:
        data = json.loads(value)
    except Exception:
        return ("", "", BoardFilters())
    if not isinstance(data, dict):
        return ("", "", BoardFilters())
    # New compact shape: {"t":"task","s":"ready","b":"board",...}
    # Legacy shape remains supported for existing rendered boards.
    task_id = str(data.get("t") or data.get("task_id") or "")
    status = str(data.get("s") or data.get("status") or "")
    if status not in STATUS_ORDER:
        status = ""
    filters_data = data.get("filters")
    if not isinstance(filters_data, dict):
        filters_data = {
            "board": data.get("b") or None,
            "status": data.get("f") or None,
            "assignee": data.get("a") or None,
            "tenant": data.get("n") or None,
            "query": data.get("x") or None,
            "approval_only": bool(data.get("q", False)),
            "include_archived": bool(data.get("r", False)),
            "limit": data.get("l") or 5,
            "page": data.get("p") or 0,
        }
    return (
        task_id,
        status,
        filters_from_value(json.dumps(filters_data)),
    )


def parse_add_action_value(value: str | None) -> tuple[str, BoardFilters]:
    if not value:
        return ("todo", BoardFilters())
    try:
        data = json.loads(value)
    except Exception:
        return ("todo", BoardFilters())
    if not isinstance(data, dict):
        return ("todo", BoardFilters())
    status = str(data.get("status") or "todo")
    if status not in STATUS_ORDER:
        status = "todo"
    return (status, filters_from_value(json.dumps(data.get("filters") or {})))


def _mrkdwn(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _task_age(created_at: int | None) -> str:
    if not created_at:
        return ""
    seconds = max(0, int(time.time()) - int(created_at))
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _current_board(filters: BoardFilters) -> str:
    if filters.board:
        return kb._normalize_board_slug(filters.board)
    return kb.get_current_board()


def _status_option(status: str, task_id: str, filters: BoardFilters) -> dict[str, Any]:
    return {
        "text": {
            "type": "plain_text",
            "text": STATUS_LABELS.get(status, status),
        },
        "value": move_action_value(task_id, status, filters),
    }


def _move_select(task: kb.Task, filters: BoardFilters) -> dict[str, Any]:
    options = [_status_option(status, task.id, filters) for status in STATUS_ORDER]
    initial = next(
        (option for status, option in zip(STATUS_ORDER, options) if status == task.status),
        None,
    )
    element: dict[str, Any] = {
        "type": "static_select",
        "placeholder": {"type": "plain_text", "text": "Move"},
        "action_id": "hermes_board_task_move",
        "options": options,
    }
    if initial:
        element["initial_option"] = initial
    return element


def _plain(text: str) -> dict[str, Any]:
    return {"type": "plain_text", "text": text[:75], "emoji": False}


def _card_text(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) > limit:
        return value[: limit - 3].rstrip() + "..."
    return value


def _first_line(text: str | None) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    lines = value.splitlines()
    return lines[0].strip() if lines else ""


def _truncate(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) > limit:
        return value[: limit - 15].rstrip() + "\n...(truncated)"
    return value


def _badge(label: str, value: str) -> str:
    clean_label = _mrkdwn(label).strip()
    clean_value = _mrkdwn(value).strip()
    return f"{clean_label} `{clean_value}`"


def _mrkdwn_obj(text: str) -> dict[str, Any]:
    return {"type": "mrkdwn", "text": text, "verbatim": False}


def _alert_block(text: str, level: str = "default") -> dict[str, Any]:
    return {
        "type": "alert",
        "level": level,
        "text": _mrkdwn_obj(text),
    }


def _card_block(
    *,
    title: str,
    subtitle: str | None = None,
    body: str | None = None,
    actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "card",
        "title": _mrkdwn_obj(_card_text(title, 150)),
    }
    if subtitle:
        block["subtitle"] = _mrkdwn_obj(_card_text(subtitle, 150))
    if body:
        block["body"] = _mrkdwn_obj(_card_text(body, 200))
    if actions:
        block["actions"] = actions[:2]
    return block


def _status_alert_level(status: str) -> str:
    if status == "done":
        return "success"
    if status == "blocked":
        return "error"
    if status == "running":
        return "warning"
    if status in {"ready", "triage"}:
        return "info"
    return "default"


def _status_header_block(status: str, count: int) -> dict[str, Any]:
    label = STATUS_LABELS.get(status, status)
    # Slack message surfaces reject the newer `alert` block type even when
    # modals support it. Use a supported section block with quote styling for
    # a compact callout-like status header.
    return {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f">*{_mrkdwn(label)}* `{count}`",
        },
    }


def _assignee_label(assignee: str | None) -> str:
    value = (assignee or "").strip()
    return value or "Default profile"


def _priority_label(priority: int | None) -> str:
    value = int(priority or 0)
    if value <= -1:
        return "Low"
    if value == 0:
        return "Normal"
    if value == 1:
        return "Medium"
    if value == 2:
        return "High"
    return "Critical"


def _created_by_label(created_by: str | None) -> str:
    value = (created_by or "").strip()
    if not value:
        return "-"
    if (
        len(value) >= 8
        and value[0] in {"U", "W"}
        and all(ch.isdigit() or ("A" <= ch <= "Z") for ch in value[1:])
    ):
        return f"<@{value}>"
    return f"`{_mrkdwn(value)}`"


def _is_system_default_result(text: str | None) -> bool:
    return (text or "").strip() in SYSTEM_DEFAULT_RESULTS


def _duration(started_at: int | None, ended_at: int | None) -> str:
    if not started_at:
        return "-"
    end = int(ended_at or time.time())
    seconds = max(0, end - int(started_at))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _payload_preview(payload: Any, limit: int = 220) -> str:
    if payload is None:
        return ""
    if isinstance(payload, dict):
        payload = {
            key: value
            for key, value in payload.items()
            if not (key in {"result", "summary"} and _is_system_default_result(str(value)))
        }
        if not payload:
            return ""
    try:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        text = str(payload)
    return _truncate(_mrkdwn(text), limit)


def _code_block(text: str, limit: int = 1800) -> str:
    value = _truncate(text or "", limit).replace("```", "'''")
    return f"```{_mrkdwn(value)}```"


def _latest_block_reason(events: list[kb.Event]) -> str:
    for event in reversed(events):
        if event.kind != "blocked":
            continue
        payload = event.payload or {}
        reason = str(payload.get("reason") or "").strip()
        if reason:
            return reason
    return ""


def _approval_context(
    task: kb.Task,
    comments: list[kb.Comment],
    events: list[kb.Event],
    latest_summary: str | None,
) -> dict[str, str] | None:
    if task.status != "blocked":
        return None
    reason = _latest_block_reason(events) or (latest_summary or "").strip()
    draft = comments[-1].body.strip() if comments else ""
    haystack = " ".join(
        part for part in [reason, latest_summary or "", draft, task.body or ""] if part
    ).lower()
    if not any(keyword in haystack for keyword in APPROVAL_KEYWORDS):
        return None
    return {
        "reason": reason or "Approval is required before this task can continue.",
        "draft": draft,
    }


def task_approval_context(task_id: str, filters: BoardFilters) -> dict[str, str] | None:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return None
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)
        latest_summary = kb.latest_summary(conn, task_id)
    return _approval_context(task, comments, events, latest_summary)


def _approval_task_ids(tasks: list[kb.Task], filters: BoardFilters) -> set[str]:
    blocked_ids = [task.id for task in tasks if task.status == "blocked"]
    if not blocked_ids:
        return set()
    board = _current_board(filters)
    kb.init_db(board=board)
    out: set[str] = set()
    with kb.connect(board=board) as conn:
        for task_id in blocked_ids:
            task = kb.get_task(conn, task_id)
            if not task:
                continue
            comments = kb.list_comments(conn, task_id)
            events = kb.list_events(conn, task_id)
            latest_summary = kb.latest_summary(conn, task_id)
            if _approval_context(task, comments, events, latest_summary):
                out.add(task_id)
    return out


def _slack_user_label(user_id: str | None) -> str:
    value = (user_id or "").strip()
    if not value:
        return "Slack user"
    if (
        len(value) >= 8
        and value[0] in {"U", "W"}
        and all(ch.isdigit() or ("A" <= ch <= "Z") for ch in value[1:])
    ):
        return f"<@{value}>"
    return value


def approve_task_and_continue(
    task_id: str,
    filters: BoardFilters,
    *,
    approved_by: str | None = None,
) -> tuple[str, BoardFilters]:
    board = _current_board(filters)
    kb.init_db(board=board)
    approver = _slack_user_label(approved_by)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return f"No such task: `{task_id}`.", filters
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)
        latest_summary = kb.latest_summary(conn, task_id)
        if not _approval_context(task, comments, events, latest_summary):
            return f"`{task_id}` is not waiting for approval.", filters
        kb.add_comment(conn, task_id, "slack", f"Approved by {approver}. Continuing the task.")
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "approved",
                {"source": "slack_board", "approved_by": approved_by or ""},
            )
        ok = kb.unblock_task(conn, task_id)
        if not ok:
            return f"Could not continue `{task_id}`.", filters

    next_filters = BoardFilters(
        board=filters.board,
        status=filters.status,
        assignee=filters.assignee,
        tenant=filters.tenant,
        query=filters.query,
        approval_only=filters.approval_only,
        include_archived=filters.include_archived,
        limit=filters.limit,
        page=filters.page,
    )
    if next_filters.status == "blocked":
        next_filters.status = "ready"
    return f"Approved `{task_id}` and moved it to `ready`.", next_filters


def request_task_changes(
    task_id: str,
    filters: BoardFilters,
    *,
    requested_by: str | None = None,
    feedback: str,
) -> tuple[str, BoardFilters]:
    feedback = (feedback or "").strip()
    if not feedback:
        raise ValueError("feedback is required")
    board = _current_board(filters)
    kb.init_db(board=board)
    requester = _slack_user_label(requested_by)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return f"No such task: `{task_id}`.", filters
        comments = kb.list_comments(conn, task_id)
        events = kb.list_events(conn, task_id)
        latest_summary = kb.latest_summary(conn, task_id)
        if not _approval_context(task, comments, events, latest_summary):
            return f"`{task_id}` is not waiting for approval.", filters
        kb.add_comment(conn, task_id, "slack", f"Changes requested by {requester}:\n\n{feedback}")
        with kb.write_txn(conn):
            kb._append_event(
                conn,
                task_id,
                "approval_changes_requested",
                {
                    "source": "slack_board",
                    "requested_by": requested_by or "",
                    "feedback_len": len(feedback),
                },
            )
    return f"Requested changes for `{task_id}`. It remains blocked.", filters


def _task_card(task: kb.Task, filters: BoardFilters, *, approval_required: bool = False) -> dict[str, Any]:
    assignee = _assignee_label(task.assignee)
    priority = _priority_label(task.priority)
    project = task.tenant or "-"

    body = task.body.strip().splitlines()[0] if task.body else ""
    if not body:
        body = "No description."
    body_lines = [
        f"Assignee `{_mrkdwn(assignee)}`",
        f"Priority `{priority}`",
        f"Project `{_mrkdwn(project)}`",
    ]
    if approval_required:
        body_lines.append("Approval `Required`")
    body_lines.append(_mrkdwn(body))

    actions: list[dict[str, Any]] = [
        {
            "type": "button",
            "text": _plain("Move"),
            "action_id": "hermes_board_task_move_open",
            "value": action_value("move_open", task.id, filters),
        },
        {
            "type": "button",
            "text": _plain("Detail"),
            "action_id": "hermes_board_task_show",
            "value": action_value("show", task.id, filters),
        },
    ]

    return {
        "type": "card",
        "block_id": f"task-{task.id}",
        "title": {
            "type": "mrkdwn",
            "text": _card_text(f"`{task.id}` { _mrkdwn(task.title) }", 150),
            "verbatim": False,
        },
        "body": {
            "type": "mrkdwn",
            "text": _card_text("\n".join(body_lines), 200),
            "verbatim": False,
        },
        # Slack card blocks allow at most two actions.
        "actions": actions,
    }


def dependency_options(filters: BoardFilters, *, limit: int = 99) -> list[dict[str, Any]]:
    """Return Slack multi-select options for parent task dependencies.

    Values are compact task IDs so they stay under Slack's option value limit.
    Options are ordered by project match, active status, and recency, which keeps
    likely dependencies near the top without requiring Slack external-select
    plumbing.
    """
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        tasks = kb.list_tasks(conn, include_archived=False)

    status_rank = {status: idx for idx, status in enumerate(STATUS_ORDER)}

    def _rank(task: kb.Task) -> tuple[int, int, int, str]:
        project_match = 0 if filters.tenant and task.tenant == filters.tenant else 1
        active_rank = 1 if task.status in {"done", "blocked"} else 0
        return (
            project_match,
            active_rank,
            status_rank.get(task.status, 99),
            str(task.created_at or 0).rjust(20, "0"),
        )

    options: list[dict[str, Any]] = []
    for task in sorted(tasks, key=_rank)[:limit]:
        title = _card_text(task.title or task.id, 54)
        project = task.tenant or "-"
        assignee = _assignee_label(task.assignee)
        description = _card_text(
            f"{STATUS_LABELS.get(task.status, task.status)} · {project} · {assignee}",
            75,
        )
        options.append(
            {
                "text": {"type": "plain_text", "text": _card_text(f"{task.id} {title}", 75)},
                "value": task.id,
                "description": {"type": "plain_text", "text": description},
            }
        )
    return options


def project_options(filters: BoardFilters, *, limit: int = 99) -> list[dict[str, Any]]:
    """Return Slack static-select options for existing project names."""
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        tasks = kb.list_tasks(conn, include_archived=False)

    projects: list[str] = []
    seen: set[str] = set()
    if filters.tenant:
        selected = str(filters.tenant).strip()
        if selected and len(selected) <= 150:
            projects.append(selected)
            seen.add(selected)
    for task in tasks:
        project = str(task.tenant or "").strip()
        if not project or project in seen or len(project) > 150:
            continue
        seen.add(project)
        projects.append(project)

    options: list[dict[str, Any]] = [
        {
            "text": {"type": "plain_text", "text": "No project"},
            "value": "__none__",
        }
    ]
    for project in sorted(projects, key=lambda item: (item != (filters.tenant or ""), item.lower()))[:limit]:
        options.append(
            {
                "text": {"type": "plain_text", "text": _card_text(project, 75)},
                "value": project,
            }
        )
    return options


def _load_tasks(filters: BoardFilters) -> list[kb.Task]:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        tasks = kb.list_tasks(
            conn,
            assignee=filters.assignee,
            status=filters.status,
            tenant=filters.tenant,
            include_archived=filters.include_archived,
        )
    query = (filters.query or "").strip().lower()
    if query:
        tasks = [
            task for task in tasks
            if query in (task.title or "").lower()
            or query in (task.body or "").lower()
            or query in (task.id or "").lower()
        ]
    return tasks


def build_board_blocks(filters: BoardFilters) -> tuple[str, list[dict[str, Any]]]:
    """Return `(fallback_text, blocks)` for a Slack Kanban board view."""
    board = _current_board(filters)
    tasks = _load_tasks(filters)
    approval_ids = _approval_task_ids(tasks, filters)
    if filters.approval_only:
        tasks = [task for task in tasks if task.id in approval_ids]
    grouped: dict[str, list[kb.Task]] = {status: [] for status in STATUS_ORDER}
    archived_count = 0
    for task in tasks:
        if task.status == "archived":
            archived_count += 1
            continue
        grouped.setdefault(task.status, []).append(task)

    total = sum(len(v) for v in grouped.values())
    filter_bits = [f"board `{_mrkdwn(board)}`"]
    if filters.status:
        filter_bits.append(f"status `{_mrkdwn(filters.status)}`")
    if filters.assignee:
        filter_bits.append(f"assignee `{_mrkdwn(filters.assignee)}`")
    if filters.tenant:
        filter_bits.append(f"project `{_mrkdwn(filters.tenant)}`")
    if filters.query:
        filter_bits.append(f"query `{_mrkdwn(filters.query)}`")
    if filters.approval_only:
        filter_bits.append("approval required")
    if filters.include_archived:
        filter_bits.append(f"archived {archived_count}")

    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Hermes Kanban Board"},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"{' · '.join(filter_bits)} · {total} visible tasks",
                }
            ],
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Refresh"},
                    "action_id": "hermes_board_refresh",
                    "value": filters_value(filters),
                },
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Add"},
                    "style": "primary",
                    "action_id": "hermes_board_task_add",
                    "value": add_action_value("todo", filters),
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "All Tasks" if filters.approval_only else "Approvals",
                    },
                    "action_id": "hermes_board_filter_approval",
                    "value": approval_filter_value(filters, not filters.approval_only),
                },
                {
                    "type": "static_select",
                    "placeholder": {"type": "plain_text", "text": "Status"},
                    "action_id": "hermes_board_filter_status",
                    "options": [
                        {
                            "text": {"type": "plain_text", "text": "All statuses"},
                            "value": json.dumps(
                                {"s": "", **compact_filters_dict(filters, status=None, page=0)},
                                separators=(",", ":"),
                            ),
                        },
                        *[
                            {
                                "text": {"type": "plain_text", "text": STATUS_LABELS[s]},
                                "value": json.dumps(
                                    {"s": s, **compact_filters_dict(filters, status=s, page=0)},
                                    separators=(",", ":"),
                                ),
                            }
                            for s in STATUS_ORDER
                        ],
                    ],
                },
            ],
        },
        {"type": "divider"},
    ]

    visible_statuses = [filters.status] if filters.status else list(STATUS_ORDER)
    rendered_status_count = 0
    for status in visible_statuses:
        if status not in grouped:
            continue
        if rendered_status_count:
            blocks.append({"type": "divider"})
        rendered_status_count += 1
        items = grouped.get(status, [])
        blocks.append(_status_header_block(status, len(items)))
        if not items:
            blocks.append(
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": "_No cards_"}],
                }
            )
            continue

        visible_items = items[: min(filters.limit, 10)]
        page_count = 1
        page = 0
        if filters.status:
            limit = min(filters.limit, 10)
            page_count = max(1, (len(items) + limit - 1) // limit)
            page = min(filters.page, page_count - 1)
            start = page * limit
            visible_items = items[start:start + limit]
        blocks.append(
            {
                "type": "carousel",
                "elements": [
                    _task_card(task, filters, approval_required=task.id in approval_ids)
                    for task in visible_items
                ],
            }
        )

        if filters.status and page_count > 1:
            page_elements: list[dict[str, Any]] = []
            if page > 0:
                page_elements.append(
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Previous"},
                        "action_id": "hermes_board_page",
                        "value": page_action_value(filters, page - 1),
                    }
                )
            if page + 1 < page_count:
                page_elements.append(
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Next"},
                        "action_id": "hermes_board_page",
                        "value": page_action_value(filters, page + 1),
                    }
                )
            page_elements.append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": f"Page {page + 1}/{page_count}"},
                    "action_id": "hermes_board_page_current",
                    "value": page_action_value(filters, page),
                }
            )
            blocks.append({"type": "actions", "elements": page_elements})

        remaining = len(items) - (filters.limit if not filters.status else (page + 1) * min(filters.limit, 10))
        if remaining > 0:
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": (
                                f"_+{remaining} more. "
                                f"Use `/board --status {status} --limit 10` to focus this column._"
                                if not filters.status
                                else "_Use Previous/Next to page through this status._"
                            ),
                        }
                    ],
                }
            )

    # Slack allows 50 blocks per message. Keep the top and append a clear note
    # instead of failing the API call if a board is crowded.
    if len(blocks) > 49:
        blocks = blocks[:48]
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                            "text": "_Board truncated for Slack. Add `--status`, `--tenant`, or `--assignee` filters._",
                    }
                ],
            }
        )

    return (f"Hermes Kanban Board: {board} ({total} visible tasks)", blocks)


def build_board_text(filters: BoardFilters, *, detail: str = "list") -> str:
    """Return a plain-text Slack report for `/board --text`."""
    board = _current_board(filters)
    tasks = _load_tasks(filters)
    approval_ids = _approval_task_ids(tasks, filters)
    if filters.approval_only:
        tasks = [task for task in tasks if task.id in approval_ids]

    grouped: dict[str, list[kb.Task]] = {status: [] for status in STATUS_ORDER}
    for task in tasks:
        if task.status == "archived" and not filters.include_archived:
            continue
        grouped.setdefault(task.status, []).append(task)

    total = sum(len(items) for items in grouped.values())
    filters_line = [f"board: {board}"]
    if filters.tenant:
        filters_line.append(f"project: {filters.tenant}")
    if filters.status:
        filters_line.append(f"status: {filters.status}")
    if filters.assignee:
        filters_line.append(f"assignee: {filters.assignee}")
    if filters.query:
        filters_line.append(f"query: {filters.query}")
    if filters.approval_only:
        filters_line.append("approval: required")

    lines = [
        "*Hermes Kanban Board*",
        " · ".join(filters_line),
        f"visible tasks: {total}",
    ]
    if detail == "summary":
        lines.append("")
        for status in STATUS_ORDER:
            count = len(grouped.get(status, []))
            if count or not filters.status:
                lines.append(f"- {STATUS_LABELS.get(status, status)}: {count}")
        if approval_ids:
            lines.append(f"- Approval required: {len(approval_ids)}")
        return "\n".join(lines)

    max_items = max(1, min(20 if detail == "full" else 10, int(filters.limit or 5)))
    for status in STATUS_ORDER:
        items = grouped.get(status, [])
        if filters.status and status != filters.status:
            continue
        lines.append("")
        lines.append(f"*{STATUS_LABELS.get(status, status)}* `{len(items)}`")
        if not items:
            lines.append("- No tasks")
            continue
        page_start = max(0, int(filters.page or 0)) * max_items
        visible = items[page_start: page_start + max_items]
        for idx, task in enumerate(visible, start=page_start + 1):
            approval = " · Approval required" if task.id in approval_ids else ""
            lines.append(f"{idx}. `{task.id}` {task.title}{approval}")
            lines.append(f"   assignee: {task.assignee or 'Default profile'}")
            if task.tenant:
                lines.append(f"   project: {task.tenant}")
            lines.append(f"   priority: {PRIORITY_LABELS.get(task.priority, str(task.priority))}")
            if detail == "full" and task.body:
                lines.append(f"   description: {_first_line(task.body)[:240]}")
        if len(items) > page_start + max_items:
            lines.append(f"   ... {len(items) - page_start - max_items} more")
    return "\n".join(lines)


def build_board_help_text() -> str:
    """Return `/board` slash command help text."""
    return """*Hermes Kanban /board*

*Open board*
`/board`
`/board --help`
`/board -h`
`/board -p youtube -s ready`
`/board -a`

*Text report*
`/board -t`
`/board -t --summary`
`/board -t --full`

*Task actions*
`/board -n`
`/board -n "AI 뉴스기사 수집" -p youtube -s todo`
`/board -e t_425b5e75`
`/board --detail t_425b5e75`
`/board -d t_425b5e75`

*Search and filters*
`-p, --project` project filter
`-s, --status` status filter
`-a, --approval` approval-required tasks
`-q, --query` search title, description, or task id
`-u, --assignee` assignee/profile filter
`-l, --limit` max cards/items
`-h, --help` show this help
`--page` page number
`--archived` include archived tasks

*Natural requests*
`/board youtube 프로젝트 ready 텍스트로 보여줘`
`/board bright data 조사 추가`
`/board t_425b5e75 상세 보기`
`/board 승인 필요한 일만 요약`
"""


def apply_task_action(action: str, task_id: str, filters: BoardFilters) -> str:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        if action == "done":
            ok = kb.complete_task(conn, task_id, result="Completed from Slack /board.")
            return f"Marked `{task_id}` done." if ok else f"Could not complete `{task_id}`."
        if action == "unblock":
            ok = kb.unblock_task(conn, task_id)
            return f"Unblocked `{task_id}`." if ok else f"Could not unblock `{task_id}`."
        if action == "archive":
            ok = kb.archive_task(conn, task_id)
            return f"Archived `{task_id}`." if ok else f"Could not archive `{task_id}`."
        if action == "delete":
            task = kb.get_task(conn, task_id)
            if not task:
                return f"No such task: `{task_id}`."
            if task.status == "archived":
                return f"Already deleted `{task_id}`."
            ok = kb.archive_task(conn, task_id)
            return f"Deleted `{task_id}`." if ok else f"Could not delete `{task_id}`."
    return f"Unknown board action `{action}`."


def move_task_status(task_id: str, target_status: str, filters: BoardFilters) -> str:
    if target_status not in STATUS_ORDER:
        return f"Unknown target status `{target_status}`."
    requested_status = target_status
    if target_status == "running":
        target_status = "ready"
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return f"No such task: `{task_id}`."
        if task.status == "archived":
            return f"Cannot move deleted task `{task_id}`."
        if task.status == target_status:
            return f"`{task_id}` is already in `{target_status}`."

        with kb.write_txn(conn):
            if task.current_run_id and target_status != "running":
                kb._end_run(
                    conn,
                    task_id,
                    outcome="reclaimed",
                    status=target_status,
                    summary=f"moved from {task.status} to {target_status} from Slack /board",
                )
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status = ?,
                       claim_lock = NULL,
                       claim_expires = NULL,
                       worker_pid = NULL,
                       current_run_id = CASE
                           WHEN ? = 'running' THEN current_run_id
                           ELSE NULL
                       END,
                       completed_at = CASE
                           WHEN ? = 'done' THEN COALESCE(completed_at, ?)
                           ELSE completed_at
                       END
                 WHERE id = ?
                   AND status != 'archived'
                """,
                (target_status, target_status, target_status, int(time.time()), task_id),
            )
            if cur.rowcount != 1:
                return f"Could not move `{task_id}`."
            event_payload = {"from": task.status, "to": target_status, "source": "slack_board"}
            if requested_status != target_status:
                event_payload["requested_to"] = requested_status
            kb._append_event(
                conn,
                task_id,
                "moved",
                event_payload,
            )
    if requested_status == "running":
        return (
            f"Queued `{task_id}` by moving it from `{task.status}` to `ready`. "
            "The Kanban dispatcher will move it to `running` when a worker claims it."
        )
    return f"Moved `{task_id}` from `{task.status}` to `{target_status}`."


def create_task_for_status(
    *,
    status: str,
    title: str,
    body: str | None = None,
    assignee: str | None = None,
    tenant: str | None = None,
    priority: int = 0,
    parents: list[str] | tuple[str, ...] | None = None,
    filters: BoardFilters | None = None,
    created_by: str | None = "slack",
) -> tuple[str, BoardFilters]:
    filters = filters or BoardFilters()
    board = _current_board(filters)
    if status not in STATUS_ORDER:
        status = "triage"
    if not tenant and filters.tenant:
        tenant = filters.tenant
    if not assignee and filters.assignee:
        assignee = filters.assignee
    parents = tuple(str(parent).strip() for parent in (parents or ()) if str(parent).strip())

    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task_id = kb.create_task(
            conn,
            title=title,
            body=body or None,
            assignee=assignee or None,
            tenant=tenant or None,
            priority=priority,
            parents=parents,
            triage=status == "triage",
            created_by=created_by,
        )
        if status not in {"triage", "ready"}:
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = ? WHERE id = ?",
                    (status, task_id),
                )
        created_task = kb.get_task(conn, task_id)
        actual_status = created_task.status if created_task else status
    next_filters = BoardFilters(
        board=filters.board,
        status=filters.status,
        assignee=filters.assignee,
        tenant=filters.tenant,
        query=filters.query,
        approval_only=filters.approval_only,
        include_archived=filters.include_archived,
        limit=filters.limit,
        page=filters.page,
    )
    if next_filters.status and next_filters.status != actual_status:
        next_filters.status = actual_status
    return task_id, next_filters


def task_edit_values(task_id: str, filters: BoardFilters) -> dict[str, Any] | None:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return None
        return {
            "id": task.id,
            "title": task.title or "",
            "body": task.body or "",
            "assignee": task.assignee or "",
            "tenant": task.tenant or "",
            "status": task.status,
        }


def update_task_fields(
    task_id: str,
    *,
    title: str,
    body: str | None = None,
    assignee: str | None = None,
    tenant: str | None = None,
    filters: BoardFilters | None = None,
) -> tuple[str, BoardFilters]:
    filters = filters or BoardFilters()
    board = _current_board(filters)
    title = (title or "").strip()
    if not title:
        raise ValueError("title is required")
    body = (body or "").strip() or None
    assignee = kb._canonical_assignee(assignee) if assignee else None
    tenant = (tenant or "").strip() or None

    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return f"No such task: `{task_id}`.", filters
        if task.status == "archived":
            return f"Cannot edit archived task `{task_id}`.", filters
        if task.status not in EDITABLE_DETAIL_STATUSES:
            return (
                f"Cannot edit `{task_id}` while it is `{task.status}`. "
                "Move it to Todo, Ready, In Progress, or Blocked first."
            ), filters

        changed: list[str] = []
        if task.title != title:
            changed.append("title")
        if (task.body or None) != body:
            changed.append("description")
        if (task.assignee or None) != assignee:
            changed.append("assignee")
        if (task.tenant or None) != tenant:
            changed.append("project")

        if changed:
            with kb.write_txn(conn):
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET title = ?, body = ?, assignee = ?, tenant = ?
                     WHERE id = ?
                       AND status IN ('todo', 'ready', 'running', 'blocked')
                    """,
                    (title, body, assignee, tenant, task_id),
                )
                if cur.rowcount != 1:
                    return f"Could not update `{task_id}`.", filters
                kb._append_event(
                    conn,
                    task_id,
                    "updated",
                    {"source": "slack_board", "fields": changed},
                )

    next_filters = BoardFilters(
        board=filters.board,
        status=filters.status,
        assignee=filters.assignee,
        tenant=filters.tenant,
        query=filters.query,
        approval_only=filters.approval_only,
        include_archived=filters.include_archived,
        limit=filters.limit,
        page=filters.page,
    )
    if filters.assignee is not None and filters.assignee != assignee:
        next_filters.assignee = assignee
    if filters.tenant is not None and filters.tenant != tenant:
        next_filters.tenant = tenant
    if not changed:
        return f"No changes for `{task_id}`.", next_filters
    return f"Updated `{task_id}`: {', '.join(changed)}.", next_filters


def task_detail_text(task_id: str, filters: BoardFilters) -> str:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return f"No such task: `{task_id}`"
        parents = kb.parent_ids(conn, task_id)
        children = kb.child_ids(conn, task_id)
        comments = kb.list_comments(conn, task_id)[-3:]
        runs = kb.list_runs(conn, task_id)[-3:]

    lines = [
        f"*`{task.id}`* {_mrkdwn(task.title)}",
        f"status: `{task.status}` · assignee: `{_mrkdwn(task.assignee or '-')}`",
    ]
    if task.tenant:
        lines.append(f"project: `{_mrkdwn(task.tenant)}`")
    if parents:
        lines.append(f"parents: `{', '.join(parents)}`")
    if children:
        lines.append(f"children: `{', '.join(children)}`")
    if task.body:
        lines.append(f"\n{_mrkdwn(task.body[:1200])}")
    if runs:
        lines.append("\n*Recent runs*")
        for run in runs:
            summary = _first_line(run.summary or run.error)
            if not summary:
                continue
            if len(summary) > 140:
                summary = summary[:137].rstrip() + "..."
            lines.append(f"- `{run.outcome or run.status}` @{_mrkdwn(run.profile or '-')} { _mrkdwn(summary) }")
    if comments:
        lines.append("\n*Recent comments*")
        for comment in comments:
            body = _first_line(comment.body)
            if not body:
                continue
            if len(body) > 140:
                body = body[:137].rstrip() + "..."
            lines.append(f"- @{_mrkdwn(comment.author or '-')}: {_mrkdwn(body)}")
    return "\n".join(lines)


def task_detail_blocks(task_id: str, filters: BoardFilters) -> list[dict[str, Any]]:
    board = _current_board(filters)
    kb.init_db(board=board)
    with kb.connect(board=board) as conn:
        task = kb.get_task(conn, task_id)
        if not task:
            return [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"No such task: `{_mrkdwn(task_id)}`"},
                }
            ]
        parents = kb.parent_ids(conn, task_id)
        children = kb.child_ids(conn, task_id)
        comments = kb.list_comments(conn, task_id)[-3:]
        all_runs = kb.list_runs(conn, task_id)
        runs = all_runs[-3:]
        all_events = kb.list_events(conn, task_id)
        events = all_events[-5:]
        latest_summary = kb.latest_summary(conn, task_id)
        diagnostics: list[dict[str, Any]] = []
        try:
            from hermes_cli import kanban_diagnostics as kd

            diagnostics = [
                diag.to_dict()
                for diag in kd.compute_task_diagnostics(task, all_events, all_runs)
            ]
        except Exception:
            diagnostics = []
    worker_log = kb.read_worker_log(task_id, tail_bytes=4096, board=board)
    approval = _approval_context(task, comments, all_events, latest_summary)

    status_label = STATUS_LABELS.get(task.status, task.status)
    assignee = _assignee_label(task.assignee)
    project = task.tenant or "-"
    priority = _priority_label(task.priority)
    created_by = _created_by_label(task.created_by)
    fields = [
        {"type": "mrkdwn", "text": f"*Status*\n`{_mrkdwn(status_label)}`"},
        {"type": "mrkdwn", "text": f"*Assignee*\n`{_mrkdwn(assignee)}`"},
        {"type": "mrkdwn", "text": f"*Project*\n`{_mrkdwn(project)}`"},
        {"type": "mrkdwn", "text": f"*Priority*\n`{priority}`"},
        {"type": "mrkdwn", "text": f"*Created*\n`{_task_age(task.created_at) or '-'}`"},
        {"type": "mrkdwn", "text": f"*Created by*\n{created_by}"},
    ]

    blocks: list[dict[str, Any]] = [
        _alert_block(f"*{_mrkdwn(status_label)}* task", _status_alert_level(task.status)),
        _card_block(
            title=_mrkdwn(task.title),
            subtitle=f"Project `{_mrkdwn(project)}` · Priority `{priority}`",
            body=_truncate(_mrkdwn(task.body or "No description."), 200),
        ),
        {"type": "section", "fields": fields},
    ]

    if approval:
        approval_blocks: list[dict[str, Any]] = [
            {"type": "divider"},
            _alert_block("*Approval Required*", "warning"),
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": _truncate(_mrkdwn(approval.get("reason") or ""), 1000),
                },
            },
        ]
        draft = approval.get("draft") or ""
        if draft:
            approval_blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Draft / Pending Output*\n{_truncate(_mrkdwn(draft), 2400)}",
                    },
                }
            )
        blocks.extend(approval_blocks)

    if task.body:
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Description*", "info"),
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": _truncate(_mrkdwn(task.body), 1800),
                    },
                },
            ]
        )

    relation_fields = []
    if parents:
        relation_fields.append({"type": "mrkdwn", "text": f"*Parents*\n`{_mrkdwn(', '.join(parents))}`"})
    if children:
        relation_fields.append({"type": "mrkdwn", "text": f"*Children*\n`{_mrkdwn(', '.join(children))}`"})
    if relation_fields:
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Relations*", "default"),
                {"type": "section", "fields": relation_fields},
            ]
        )

    if task.result and not _is_system_default_result(task.result):
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Result*", "success"),
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": _truncate(_mrkdwn(task.result), 1200),
                    },
                },
            ]
        )

    if (
        (not task.result or _is_system_default_result(task.result))
        and latest_summary
        and not _is_system_default_result(latest_summary)
    ):
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Latest Summary*", "success"),
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": _truncate(_mrkdwn(latest_summary), 1600),
                    },
                },
            ]
        )

    if task.last_failure_error:
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Last Failure*", "error"),
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": _truncate(_mrkdwn(task.last_failure_error), 1400),
                    },
                },
            ]
        )

    if diagnostics:
        diagnostic_lines = []
        for diag in diagnostics[:4]:
            title = _mrkdwn(str(diag.get("title") or diag.get("kind") or "Diagnostic"))
            severity = _mrkdwn(str(diag.get("severity") or "warning")).upper()
            detail = _truncate(_mrkdwn(str(diag.get("detail") or "")), 320)
            actions = [
                str(action.get("label") or action.get("kind") or "").strip()
                for action in (diag.get("actions") or [])
                if str(action.get("label") or action.get("kind") or "").strip()
            ]
            line = f"- `{severity}` *{title}*"
            if detail:
                line += f"\n  {detail}"
            if actions:
                line += f"\n  Suggested: `{_mrkdwn(', '.join(actions[:3]))}`"
            diagnostic_lines.append(line)
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Diagnostics*", "warning"),
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _truncate("\n".join(diagnostic_lines), 2400)},
                },
            ]
        )

    if runs:
        run_lines = []
        for run in reversed(runs):
            outcome = run.outcome or run.status or "running"
            elapsed = _duration(run.started_at, run.ended_at)
            age = _task_age(run.ended_at or run.started_at) or "-"
            line = (
                f"- Run `{run.id}` `{_mrkdwn(outcome)}` "
                f"Assignee `{_mrkdwn(_assignee_label(run.profile))}` "
                f"elapsed `{elapsed}` updated `{age}`"
            )
            if run.summary and not _is_system_default_result(run.summary):
                line += f"\n  Summary: {_truncate(_mrkdwn(run.summary), 420)}"
            if run.error:
                line += f"\n  Error: {_truncate(_mrkdwn(run.error), 520)}"
            if run.metadata:
                line += f"\n  Metadata: `{_payload_preview(run.metadata, 260)}`"
            run_lines.append(line)
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Run History*", "default"),
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _truncate("\n".join(run_lines), 2600)},
                },
            ]
        )

    if events:
        event_lines = []
        for event in reversed(events):
            payload = _payload_preview(event.payload, 260)
            age = _task_age(event.created_at) or "-"
            line = f"- `{_mrkdwn(event.kind)}` `{age}`"
            if event.run_id:
                line += f" run `{event.run_id}`"
            if payload:
                line += f"\n  `{payload}`"
            event_lines.append(line)
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Recent Events*", "default"),
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _truncate("\n".join(event_lines), 2400)},
                },
            ]
        )

    if worker_log:
        blocks.extend(
            [
                {"type": "divider"},
                _alert_block("*Worker Log Tail*", "default"),
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": _code_block(worker_log, 2200)},
                },
            ]
        )

    if comments:
        comment_lines = []
        for comment in comments:
            body = _first_line(comment.body)
            if not body:
                continue
            comment_lines.append(
                f"- `{_mrkdwn(comment.author or '-')}`: {_mrkdwn(_card_text(body, 140))}"
            )
        if comment_lines:
            blocks.extend(
                [
                    {"type": "divider"},
                    _alert_block("*Recent Comments*", "default"),
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": _truncate("\n".join(comment_lines), 1800)},
                    },
                ]
            )

    return blocks
