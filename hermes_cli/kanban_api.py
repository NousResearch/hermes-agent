"""Sanitized REST adapter for the shared Hermes Kanban store.

The dashboard plugin mounts this router at ``/api/plugins/kanban/v1``.  The
adapter intentionally exposes workflow/task concepts rather than SQLite rows:
private task bodies, results, comments, workspace paths, claims, process data,
profile internals, and raw event payloads are never serialized. The one
exception, the run transcript, is off unless ``kanban.api_expose_transcripts``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Annotated, Any, Iterator, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from agent.redact import redact_sensitive_text
from hermes_cli import kanban_db
from hermes_cli import kanban_db_connect as kbc

log = logging.getLogger(__name__)

router = APIRouter()

_API_VERSION = "1"
_DEFAULT_LOG_TAIL = 8_192
# Transcript field caps: tool results can be whole 1C/DB dumps, so they get
# a tighter limit than the agent's own prose/reasoning.
_TRANSCRIPT_TEXT_CAP = 20_000
_TRANSCRIPT_TOOL_CAP = 4_000
_MAX_LOG_TAIL = 32_768
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:])(?:[A-Za-z]:[\\/](?:[^\s\\/]+[\\/])*[^\s\\/]*|/[^/\s]+(?:/[^/\s]*)*)"
)
# Match an ``Authorization: Bearer/Basic <token>`` header anywhere in a line,
# not just at its start — the token can appear mid-line inside a dumped curl
# command (``curl -H 'Authorization: Bearer ...'``) or a shell trace. A bare
# ``Bearer <token>`` (logged without its header name) is redacted too.
_AUTH_HEADER_RE = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+|\bbearer\s+)[^\s]+"
)

# Known-safe validation messages from ``kanban_db`` that may be echoed to an
# external caller verbatim: they carry no filesystem paths, SQL, or other
# internal detail — only user-facing input guidance. Anything else collapses
# to a stable generic fallback while the raw error is logged server-side.
_SAFE_ERROR_MARKERS = (
    "title is required",
    "unknown parent task",
    "unknown task",
    "a task cannot depend on itself",
    "would create a cycle",
    "is a toolset name",
    "are toolset names",
    "skill name cannot contain comma",
    "must be one of",
    "has no result or summary evidence",
)


def _client_error(
    exc: Exception,
    *,
    fallback: str,
    status_code: int = 400,
) -> HTTPException:
    """Build an HTTPException with a sanitized, stable ``detail``.

    The original exception is always logged server-side. The client sees a
    known-safe validation message when one is recognised, otherwise the
    generic ``fallback`` — raw internal text never reaches the wire.
    """
    message = str(exc)
    detail = message if any(m in message for m in _SAFE_ERROR_MARKERS) else fallback
    log.warning("kanban_api request rejected (detail=%r): %s", detail, message)
    return HTTPException(status_code=status_code, detail=detail)


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateTaskRequest(_RequestModel):
    title: str = Field(min_length=1, max_length=300)
    body: Optional[str] = Field(default=None, max_length=100_000)
    assignee: Optional[str] = Field(default=None, max_length=128)
    tenant: Optional[str] = Field(default=None, max_length=128)
    priority: int = Field(default=0, ge=-1_000_000, le=1_000_000)
    parents: list[str] = Field(default_factory=list, max_length=100)
    triage: bool = False
    idempotency_key: Optional[str] = Field(default=None, min_length=1, max_length=200)
    max_runtime_seconds: Optional[int] = Field(default=None, ge=1, le=31_536_000)
    skills: Optional[list[str]] = Field(default=None, max_length=50)


class UpdateTaskRequest(_RequestModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=300)
    body: Optional[str] = Field(default=None, max_length=100_000)
    assignee: Optional[str] = Field(default=None, max_length=128)
    priority: Optional[int] = Field(default=None, ge=-1_000_000, le=1_000_000)


class CommentRequest(_RequestModel):
    body: str = Field(min_length=1, max_length=20_000)


class CompleteRequest(_RequestModel):
    summary: Optional[str] = Field(default=None, max_length=50_000)


class BlockRequest(_RequestModel):
    reason: Optional[str] = Field(default=None, max_length=20_000)
    kind: Optional[str] = None


def _actor(request: Request) -> str:
    """Durable provenance (comment author, task creator) for this request: ``api:`` + the
    principal the token seam verified, else ``api:external``. Never caller-supplied, and never
    a valid profile name (the colon): the live comment bridge skips comments whose author
    equals the worker's profile as its own (``inject_new_comments_from_env``), so an author
    that could match a profile would silence an operator steer."""
    principal = getattr(request.state, "token_principal", None)
    return f"api:{getattr(principal, 'principal', None) or 'external'}"


@contextmanager
def _connection(board: Optional[str]) -> Iterator[sqlite3.Connection]:
    # ``connect`` creates the parent dir and auto-runs the schema/migration
    # pass on a path's first open, caching it afterwards. Calling ``init_db``
    # here instead would evict that cache entry and force the full integrity
    # probe + migration executescript under the cross-process init lock on
    # *every* request — the same redundancy already dropped in
    # ``_board_counts``. This is a network-facing polling surface, so it stays
    # on the cached path.
    slug = _resolve_board_slug(board)
    with kbc.connect_closing(board=slug) as conn:
        yield conn


def _resolve_board_slug(board: Optional[str]) -> str:
    if board is None or not str(board).strip():
        return kanban_db.get_current_board()
    try:
        slug = kanban_db.normalize_board_slug(board)
    except ValueError as exc:
        raise _client_error(exc, fallback="invalid board id") from exc
    if not slug or not kanban_db.board_exists(slug):
        raise HTTPException(status_code=404, detail="board not found")
    return slug


def _resolve_board_id_or_name(value: str) -> str:
    candidate = value.strip()
    try:
        slug = kanban_db.normalize_board_slug(candidate)
    except ValueError:
        slug = None
    if slug and kanban_db.board_exists(slug):
        return slug

    matches = [
        item["slug"]
        for item in kanban_db.list_boards(include_archived=False)
        if str(item.get("name") or "").casefold() == candidate.casefold()
    ]
    if not matches:
        raise HTTPException(status_code=404, detail="board not found")
    if len(matches) > 1:
        raise HTTPException(status_code=409, detail="board name is ambiguous; use its id")
    return matches[0]


def _board_counts(slug: str) -> dict[str, int]:
    # One connection per board, one GROUP BY query. ``connect`` auto-runs the
    # schema/migration pass on first open, so the separate ``init_db`` call
    # (which opened and threw away its own connection) is redundant — dropping
    # it halves the connections ``list_boards`` opens while enumerating boards.
    # Boards are separate SQLite files, so per-board counts genuinely need one
    # connection each; the query itself stays single.
    with kbc.connect_closing(board=slug) as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
        ).fetchall()
        return {row["status"]: int(row["count"]) for row in rows}


def _board_dto(meta: dict[str, Any], *, current: str) -> dict[str, Any]:
    slug = str(meta["slug"])
    counts = _board_counts(slug)
    return {
        "id": slug,
        "name": str(meta.get("name") or slug),
        "description": str(meta.get("description") or ""),
        "icon": str(meta.get("icon") or ""),
        "color": str(meta.get("color") or ""),
        "created_at": meta.get("created_at"),
        "is_current": slug == current,
        "counts": counts,
        "total": sum(counts.values()),
    }


def _task_dto(conn: sqlite3.Connection, task: kanban_db.Task) -> dict[str, Any]:
    return {
        "id": task.id,
        "title": task.title,
        "assignee": task.assignee,
        # Attribution: which profile (or surface, e.g. "api:external" /
        # "dashboard") created the card — this is what lets an external
        # control plane visualise orchestrator fan-out, not just who the
        # work was routed to.
        "created_by": task.created_by,
        "status": task.status,
        "priority": task.priority,
        "tenant": task.tenant,
        "created_at": task.created_at,
        "started_at": task.started_at,
        "completed_at": task.completed_at,
        "workflow_template_id": task.workflow_template_id,
        "current_step_key": task.current_step_key,
        "block_kind": task.block_kind,
        "links": {
            "parents": kanban_db.parent_ids(conn, task.id),
            "children": kanban_db.child_ids(conn, task.id),
        },
    }


def _require_task(conn: sqlite3.Connection, task_id: str) -> kanban_db.Task:
    task = kanban_db.get_task(conn, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


def _task_response(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    return {"task": _task_dto(conn, _require_task(conn, task_id))}


def _transition_response(
    conn: sqlite3.Connection,
    task_id: str,
    ok: bool,
    action: str,
) -> dict[str, Any]:
    if not ok:
        if kanban_db.get_task(conn, task_id) is None:
            raise HTTPException(status_code=404, detail="task not found")
        raise HTTPException(status_code=409, detail=f"task cannot be {action} from its current state")
    return _task_response(conn, task_id)


def _idempotency_key(
    payload_key: Optional[str],
    header_key: Optional[str],
) -> Optional[str]:
    body = (payload_key or "").strip()
    header = (header_key or "").strip()
    if body and header and body != header:
        raise HTTPException(status_code=409, detail="conflicting idempotency keys")
    key = header or body or None
    if key and len(key) > 200:
        raise HTTPException(status_code=422, detail="idempotency key is too long")
    return key


def _sanitize_log(content: str) -> str:
    redacted = redact_sensitive_text(content, force=True, redact_url_credentials=True)
    redacted = _AUTH_HEADER_RE.sub(r"\1[REDACTED]", redacted)
    return _ABSOLUTE_PATH_RE.sub("[PATH]", redacted)


@router.get("/health")
def health() -> dict[str, Any]:
    current = kanban_db.get_current_board()
    try:
        with _connection(current) as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="kanban store unavailable") from exc
    return {
        "status": "ok",
        "service": "hermes-kanban",
        "api_version": _API_VERSION,
        "current_board": current,
    }


@router.get("/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "api_version": _API_VERSION,
        "boards": {"read": True, "write": False},
        "tasks": {"read": True, "create": True, "update": True},
        "actions": ["comment", "complete", "block", "unblock", "archive"],
        "links": {"create": True, "delete": True},
        "observability": ["events", "runs", "sanitized_log_excerpt"]
        + (["transcript"] if _transcripts_enabled() else []),
        "task_statuses": sorted(kanban_db.VALID_STATUSES),
        "block_kinds": sorted(kanban_db.VALID_BLOCK_KINDS),
        "idempotent_task_creation": True,
        "profile_execution": False,
        # Read-only roster (GET /profiles): name + description only, for
        # assignee pickers in external control planes. No profile
        # management or execution surface exists here.
        "profiles_api": True,
    }


@router.get("/profiles")
def list_profiles() -> dict[str, Any]:
    """Sanitized assignee roster for external control planes.

    Returns only what an external dashboard needs to route work — the
    profile name and its operator-facing description (the same pair the
    built-in decomposer feeds its routing LLM). Models, providers,
    filesystem paths, env/config state, and skill inventories are
    deliberately not exposed.
    """
    from hermes_cli import profiles as profiles_mod  # lazy: heavy CLI import

    try:
        infos = profiles_mod.list_profiles()
    except Exception as exc:
        log.warning("kanban_api profiles listing failed: %s", exc)
        raise HTTPException(status_code=503, detail="profiles unavailable") from exc
    roster = [
        {
            "name": info.name,
            "description": (info.description or "").strip(),
            "has_description": bool((info.description or "").strip()),
        }
        for info in infos
    ]
    roster.sort(key=lambda item: item["name"])
    return {"profiles": roster, "count": len(roster)}


@router.get("/boards")
def list_boards() -> dict[str, Any]:
    current = kanban_db.get_current_board()
    boards = [
        _board_dto(meta, current=current)
        for meta in kanban_db.list_boards(include_archived=False)
    ]
    return {"boards": boards, "current": current, "count": len(boards)}


@router.get("/boards/{board_id_or_name}")
def get_board(board_id_or_name: str) -> dict[str, Any]:
    slug = _resolve_board_id_or_name(board_id_or_name)
    current = kanban_db.get_current_board()
    return {"board": _board_dto(kanban_db.read_board_metadata(slug), current=current)}


@router.get("/tasks")
def list_tasks(
    board: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    assignee: Optional[str] = Query(default=None),
    tenant: Optional[str] = Query(default=None),
    include_archived: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    try:
        with _connection(board) as conn:
            tasks = kanban_db.list_tasks(
                conn,
                status=status,
                assignee=assignee,
                tenant=tenant,
                include_archived=include_archived,
                limit=limit,
            )
            items = [_task_dto(conn, task) for task in tasks]
    except ValueError as exc:
        raise _client_error(exc, fallback="invalid task filter") from exc
    return {"tasks": items, "count": len(items), "limit": limit}


@router.post("/tasks", status_code=201)
def create_task(
    payload: CreateTaskRequest,
    request: Request,
    response: Response,
    board: Optional[str] = Query(default=None),
    idempotency_header: Annotated[Optional[str], Header(alias="Idempotency-Key")] = None,
) -> dict[str, Any]:
    key = _idempotency_key(payload.idempotency_key, idempotency_header)
    with _connection(board) as conn:
        try:
            # The storage layer owns idempotency end to end: a repeated key
            # (fast-path hit or a lost UNIQUE-index race with a concurrent
            # POST) resolves to the existing task with created=False.
            task_id, created = kanban_db.create_task_idempotent(
                conn,
                title=payload.title,
                body=payload.body,
                assignee=payload.assignee,
                created_by=_actor(request),
                workspace_kind="scratch",
                tenant=payload.tenant,
                priority=payload.priority,
                parents=payload.parents,
                triage=payload.triage,
                idempotency_key=key,
                max_runtime_seconds=payload.max_runtime_seconds,
                skills=payload.skills,
            )
        except ValueError as exc:
            raise _client_error(exc, fallback="task could not be created") from exc
        if not created:
            response.status_code = 200
        return {**_task_response(conn, task_id), "created": created}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        return _task_response(conn, task_id)


@router.patch("/tasks/{task_id}")
def update_task(
    task_id: str,
    payload: UpdateTaskRequest,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=400, detail="at least one field is required")
    slug = _resolve_board_slug(board)
    with _connection(slug) as conn:
        # One storage-layer transaction for the whole patch: a transition landing
        # mid-request rolls back every field, so a 409 never leaves the assignee
        # applied (and announced) while the title/body edit was refused.
        try:
            applied = kanban_db.update_task_fields(
                conn,
                task_id,
                assign="assignee" in fields,
                assignee=payload.assignee or None,
                title=payload.title,
                # An explicit ``"body": null`` clears the body; omitted leaves it.
                body="" if "body" in fields and payload.body is None else payload.body,
                priority=payload.priority,
                board=slug,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not applied:
            raise HTTPException(status_code=404, detail="task not found")
        return _task_response(conn, task_id)


@router.post("/tasks/{task_id}/comment", status_code=201)
def comment_task(
    task_id: str,
    payload: CommentRequest,
    request: Request,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        try:
            comment_id = kanban_db.add_comment(conn, task_id, _actor(request), payload.body)
        except ValueError as exc:
            raise _client_error(exc, fallback="comment rejected") from exc
        row = conn.execute(
            "SELECT created_at FROM task_comments WHERE id = ?", (comment_id,)
        ).fetchone()
        return {
            "comment": {
                "id": comment_id,
                "task_id": task_id,
                "created_at": row["created_at"] if row else int(time.time()),
            }
        }


@router.post("/tasks/{task_id}/complete")
def complete_task(
    task_id: str,
    payload: Optional[CompleteRequest] = None,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        try:
            ok = kanban_db.complete_task(
                conn, task_id, summary=payload.summary if payload else None
            )
        except ValueError as exc:
            raise _client_error(exc, fallback="task could not be completed") from exc
        return _transition_response(conn, task_id, ok, "completed")


@router.post("/tasks/{task_id}/block")
def block_task(
    task_id: str,
    payload: Optional[BlockRequest] = None,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        try:
            ok = kanban_db.block_task(
                conn,
                task_id,
                reason=payload.reason if payload else None,
                kind=payload.kind if payload else None,
            )
        except ValueError as exc:
            raise _client_error(exc, fallback="task could not be blocked") from exc
        return _transition_response(conn, task_id, ok, "blocked")


@router.post("/tasks/{task_id}/unblock")
def unblock_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        return _transition_response(
            conn, task_id, kanban_db.unblock_task(conn, task_id), "unblocked"
        )


@router.post("/tasks/{task_id}/archive")
def archive_task(task_id: str, board: Optional[str] = Query(default=None)) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        return _transition_response(
            conn, task_id, kanban_db.archive_task(conn, task_id), "archived"
        )


@router.post("/tasks/{parent_id}/links/{child_id}")
def link_tasks(
    parent_id: str,
    child_id: str,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        # A nonexistent parent/child is a 404 (consistent with unlink), not a
        # 400 from kanban_db's "unknown task(s)" ValueError.
        _require_task(conn, parent_id)
        _require_task(conn, child_id)
        try:
            kanban_db.link_tasks(conn, parent_id, child_id)
        except ValueError as exc:
            raise _client_error(exc, fallback="link rejected") from exc
        return {"parent_id": parent_id, "child_id": child_id, "linked": True}


@router.delete("/tasks/{parent_id}/links/{child_id}")
def unlink_tasks(
    parent_id: str,
    child_id: str,
    board: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, parent_id)
        _require_task(conn, child_id)
        removed = kanban_db.unlink_tasks(conn, parent_id, child_id)
        return {"parent_id": parent_id, "child_id": child_id, "removed": removed}


@router.get("/tasks/{task_id}/events")
def task_events(
    task_id: str,
    board: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        events = kanban_db.list_events(conn, task_id, limit=limit)
        items = [
            {
                "id": event.id,
                "task_id": event.task_id,
                "kind": event.kind,
                "created_at": event.created_at,
                "run_id": event.run_id,
            }
            for event in events
        ]
        return {"events": items, "count": len(items)}


@router.get("/tasks/{task_id}/runs")
def task_runs(
    task_id: str,
    board: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    with _connection(board) as conn:
        _require_task(conn, task_id)
        runs = kanban_db.list_runs(conn, task_id, limit=limit)
        items = [
            {
                "id": run.id,
                # The profile that executed this attempt. Usually equals the
                # task's assignee, but reassignment between retries makes the
                # per-run value the only accurate execution record.
                "profile": run.profile,
                "status": run.status,
                "outcome": run.outcome,
                "started_at": run.started_at,
                "ended_at": run.ended_at,
            }
            for run in runs
        ]
        return {"runs": items, "count": len(items)}


@router.get("/tasks/{task_id}/log")
def task_log(
    task_id: str,
    board: Optional[str] = Query(default=None),
    tail_bytes: int = Query(default=_DEFAULT_LOG_TAIL, ge=1, le=_MAX_LOG_TAIL),
) -> dict[str, Any]:
    slug = _resolve_board_slug(board)
    with _connection(slug) as conn:
        _require_task(conn, task_id)
    content = kanban_db.read_worker_log(task_id, tail_bytes=tail_bytes, board=slug, whole_lines=True)
    size = 0
    log_path = kanban_db.worker_log_path(task_id, board=slug)
    try:
        size = log_path.stat().st_size
    except OSError:
        pass
    return {
        "task_id": task_id,
        "exists": content is not None,
        "size_bytes": size,
        "tail_bytes": tail_bytes,
        "truncated": size > tail_bytes,
        "excerpt": _sanitize_log(content or ""),
    }


def _transcripts_enabled() -> bool:
    """Opt-in: transcripts carry the task body and worker output that the rest of ``/v1`` withholds."""
    from hermes_cli.config import cfg_get, load_config

    return cfg_get(load_config(), "kanban", "api_expose_transcripts", default=False) is True


def _transcript_text(value: Any, cap: int) -> tuple[Optional[str], bool]:
    """Sanitize + cap one transcript field; multimodal parts keep text only."""
    if value is None:
        return None, False
    if isinstance(value, list):
        value = "\n".join(
            str(part.get("text") or "") for part in value if isinstance(part, dict)
        )
    text = _sanitize_log(str(value))
    if not text:
        return None, False
    if len(text) > cap:
        return text[:cap], True
    return text, False


def _transcript_tool_call(raw: Any) -> tuple[dict[str, Any], bool]:
    raw = raw if isinstance(raw, dict) else {}
    fn = raw.get("function") if isinstance(raw.get("function"), dict) else raw
    arguments, truncated = _transcript_text(fn.get("arguments"), _TRANSCRIPT_TOOL_CAP)
    return {
        "id": str(raw.get("id") or ""),
        "name": str(fn.get("name") or ""),
        "arguments": arguments or "",
    }, truncated


def _transcript_message(msg: dict[str, Any]) -> dict[str, Any]:
    role = msg.get("role")
    content, truncated = _transcript_text(
        msg.get("content"),
        _TRANSCRIPT_TOOL_CAP if role == "tool" else _TRANSCRIPT_TEXT_CAP,
    )
    reasoning, cut = _transcript_text(
        msg.get("reasoning") or msg.get("reasoning_content"), _TRANSCRIPT_TEXT_CAP
    )
    truncated = truncated or cut
    tool_calls = []
    for raw in msg.get("tool_calls") or []:
        call, cut = _transcript_tool_call(raw)
        tool_calls.append(call)
        truncated = truncated or cut
    return {
        "id": msg["id"],
        "role": role,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "tool_name": msg.get("tool_name"),
        "tool_call_id": msg.get("tool_call_id"),
        "timestamp": msg.get("timestamp"),
        "truncated": truncated,
    }


def _read_session_messages(
    profile: str, session_id: str, after_id: int, limit: int, latest: bool = False
) -> list[dict[str, Any]]:
    """Read a worker session (plus its compression continuations) from the
    worker profile's own state.db. Message ids are one AUTOINCREMENT per DB,
    so ``after_id`` is a valid cursor across the whole chain. ``latest``
    returns the newest ``limit`` rows (still oldest-first) instead."""
    from pathlib import Path

    from hermes_cli.profiles import resolve_profile_env
    from hermes_state import SessionDB

    try:
        db_path = Path(resolve_profile_env(profile)) / "state.db"
    except (FileNotFoundError, ValueError):
        return []
    if not db_path.is_file():
        return []
    db = SessionDB(db_path, read_only=True)
    try:
        rows: list[dict[str, Any]] = []
        # ponytail: a compression continuation re-inserts the compacted
        # context, so post-compression transcripts repeat a summary block.
        # ``include_inactive``: in-place compaction soft-archives the steps it
        # summarized (``compacted=1``) and those are still the run's history; it is
        # the only id-cursor read that reaches them. The caller drops rewound rows.
        chain = db.get_compression_chain(session_id)
        if latest:
            for sid in reversed(chain):
                rows[:0] = db.get_messages(sid, include_inactive=True, latest=True, limit=limit - len(rows))
                if len(rows) >= limit:
                    break
            return rows
        for sid in chain:
            rows.extend(db.get_messages(sid, include_inactive=True, after_id=after_id, limit=limit - len(rows)))
            if len(rows) >= limit:
                break
        return rows
    finally:
        db.close()


@router.get("/tasks/{task_id}/transcript")
def task_transcript(
    task_id: str,
    board: Optional[str] = Query(default=None),
    run_id: Optional[int] = Query(default=None, ge=1),
    after_id: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=500),
    latest: bool = Query(default=False),
) -> dict[str, Any]:
    """Step-by-step transcript of one run: reasoning, tool calls/results and
    replies, sanitized like the log excerpt. Poll with ``after_id`` =
    ``next_after_id`` for near-live progress while the run is going, or with
    ``latest=true`` for the newest ``limit`` steps (``has_more`` then means
    older steps exist; ``after_id`` is ignored). 404 unless
    ``kanban.api_expose_transcripts`` is on."""
    if not _transcripts_enabled():
        raise HTTPException(status_code=404, detail="transcripts are disabled")
    with _connection(board) as conn:
        _require_task(conn, task_id)
        if run_id is None:
            run = kanban_db.latest_run(conn, task_id)
        else:
            run = kanban_db.get_run(conn, run_id)
            if run is None or run.task_id != task_id:
                raise HTTPException(status_code=404, detail="run not found")
    # Only the column the worker stamps at start: run metadata is caller-writable
    # (``kanban complete --metadata``), so it can't pick which session is read.
    session_id = run.worker_session_id if run is not None else None
    messages: list[dict[str, Any]] = []
    has_more = False
    if run is not None and session_id:
        # The run's own profile, never the task's assignee: PATCH can repoint the assignee.
        profile = run.profile or "default"
        try:
            rows = _read_session_messages(
                profile, str(session_id), 0 if latest else after_id, limit + 1, latest
            )
        except Exception as exc:
            log.warning("kanban transcript read failed for %s: %s", task_id, exc)
            raise HTTPException(status_code=503, detail="transcript unavailable") from exc
        has_more = len(rows) > limit
        page = rows[-limit:] if latest else rows[:limit]
        # Rewound rows (neither live nor compaction-archived) include the tail that a
        # compaction re-inserts as fresh rows; showing both would repeat it.
        messages = [
            _transcript_message(row) for row in page
            if row.get("role") != "system" and (row.get("active") or row.get("compacted"))
        ]
        next_after_id = page[-1]["id"] if page else after_id
    else:
        next_after_id = after_id
    return {
        "task_id": task_id,
        "run_id": run.id if run else None,
        "run_status": run.status if run else None,
        "messages": messages,
        "next_after_id": next_after_id,
        "has_more": has_more,
    }
