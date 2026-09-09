"""Read-only Phase-C task/project resolution and status snapshots.

This module is deliberately an adapter over the Kanban and Projects readers. It
opens existing SQLite files in ``mode=ro`` so a status request cannot run schema
initialization, migrations, WAL configuration, or any lifecycle write.
"""

from __future__ import annotations

import contextlib
import shlex
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_notify as kbn
from hermes_cli import projects_db as pdb
from hermes_cli.sqlite_safe_read import connect_tracked
from utils import is_truthy_value


SKILL_CONTRACT_VERSION = "phase-c-v1"
_MAX_CANDIDATES = 10
_MAX_TEXT = 1_000
_MAX_COLLECTION = 50


def _bounded_text(value: Any, limit: int = _MAX_TEXT) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Bound persisted JSON-like metadata without changing scalar semantics."""
    if isinstance(value, str):
        return _bounded_text(value)
    if depth >= 3:
        return "…"
    if isinstance(value, dict):
        return {
            str(key)[:200]: _bounded_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_COLLECTION]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_value(item, depth=depth + 1) for item in value[:_MAX_COLLECTION]]
    return value


@dataclass(frozen=True)
class StatusCandidate:
    kind: str
    id: str
    name: str
    board: Optional[str] = None


@dataclass(frozen=True)
class StatusResolution:
    skill_contract_version: str = SKILL_CONTRACT_VERSION
    ok: bool = False
    scope: Optional[str] = None
    reference: str = ""
    board: Optional[str] = None
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    candidates: tuple[StatusCandidate, ...] = field(default_factory=tuple)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectStatusResult:
    """Stable internal result for the first read-only Phase-C command."""

    skill_contract_version: str = SKILL_CONTRACT_VERSION
    ok: bool = False
    read_only: bool = True
    scope: Optional[str] = None
    task_id: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    board: Optional[str] = None
    state: Optional[str] = None
    assignee: Optional[str] = None
    profile: Optional[str] = None
    profile_source: Optional[str] = None
    active_run: Optional[dict[str, Any]] = None
    latest_run: Optional[dict[str, Any]] = None
    latest_lifecycle_event: Optional[dict[str, Any]] = None
    provenance: dict[str, Optional[str]] = field(
        default_factory=lambda: {"implementer": None, "reviewer": None}
    )
    review_loop_count: int = 0
    failure_loop_count: int = 0
    dependency_state: dict[str, Any] = field(
        default_factory=lambda: {"satisfied": True, "parents": []}
    )
    workspace: dict[str, Any] = field(default_factory=dict)
    branch: Optional[str] = None
    github: dict[str, Any] = field(
        default_factory=lambda: {"pr": None, "merge_state": None}
    )
    deployment: dict[str, Any] = field(
        default_factory=lambda: {
            "deployment_state": None,
            "production_state": None,
            "inferred": False,
        }
    )
    policy_block_reason: Optional[str] = None
    notification_route: dict[str, Any] = field(
        default_factory=lambda: {"advisory_unowned": False, "owned_profiles": []}
    )
    project_task_counts: dict[str, int] = field(default_factory=dict)
    project_tasks: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    project_tasks_truncated: bool = False
    next_action: Optional[str] = None
    candidates: tuple[StatusCandidate, ...] = field(default_factory=tuple)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@contextlib.contextmanager
def _read_only_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Open one existing Hermes DB without creating or configuring it."""
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = connect_tracked(uri, tracking_path=path, uri=True, timeout=5.0)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
    finally:
        conn.close()


def _board_slugs(explicit: Optional[str]) -> tuple[str, ...]:
    if explicit is not None:
        # Kanban exposes board readers but no public slug normalizer; reuse its canonical
        # normalizer rather than creating a subtly different board identity algorithm.
        slug = kb._normalize_board_slug(explicit)
        if not slug or not kb.board_exists(slug):
            raise ValueError(f"board {explicit!r} does not exist")
        return (slug,)
    return tuple(sorted({str(item.get("slug") or kb.DEFAULT_BOARD) for item in kb.list_boards(include_archived=False)}))


def _tasks_on_board(
    board: str, *, include_archived: bool = False, **filters: Any,
) -> list[kb.Task]:
    path = kb.kanban_db_path(board=board)
    if not path.is_file():
        return []
    with _read_only_connection(path) as conn:
        return kb.list_tasks(conn, include_archived=include_archived, **filters)


def _task_on_board(
    board: str, task_id: str, *, include_archived: bool = False,
) -> Optional[kb.Task]:
    path = kb.kanban_db_path(board=board)
    if not path.is_file():
        return None
    with _read_only_connection(path) as conn:
        task = kb.get_task(conn, task_id)
        if task is None or (task.status == "archived" and not include_archived):
            return None
        return task


def _projects(**filters: Any) -> list[pdb.Project]:
    path = pdb.projects_db_path()
    if not path.is_file():
        return []
    with _read_only_connection(path) as conn:
        return pdb.list_projects(conn, include_archived=False, **filters)


def _project_identity(reference: str) -> Optional[pdb.Project]:
    path = pdb.projects_db_path()
    if not path.is_file():
        return None
    with _read_only_connection(path) as conn:
        project = pdb.get_project(conn, reference)
        return project if project and not project.archived else None


def _project_board(project: pdb.Project, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return kb._normalize_board_slug(explicit)
    if project.board_slug and kb.board_exists(project.board_slug):
        return project.board_slug
    associated = sorted(
        str(meta.get("slug") or kb.DEFAULT_BOARD)
        for meta in kb.list_boards(include_archived=False)
        if meta.get("project_id") == project.id
    )
    return associated[0] if len(associated) == 1 else None


def _candidate_for_task(task: kb.Task, board: str) -> StatusCandidate:
    return StatusCandidate("task", task.id, _bounded_text(task.title, 200) or "", board)


def _candidate_for_project(project: pdb.Project, explicit: Optional[str]) -> StatusCandidate:
    return StatusCandidate(
        "project", project.id, _bounded_text(project.name, 200) or "",
        _project_board(project, explicit),
    )


def _ambiguous(reference: str, candidates: list[StatusCandidate]) -> StatusResolution:
    ordered = sorted(candidates, key=lambda item: (item.kind, item.board or "", item.id))[:_MAX_CANDIDATES]
    return StatusResolution(
        ok=False,
        reference=reference,
        candidates=tuple(ordered),
        error=f"ambiguous reference {_bounded_text(reference, 200)!r}",
    )


def _task_reference_candidates(
    boards: tuple[str, ...], reference: str, *, partial: bool,
    include_archived: bool = False,
) -> list[StatusCandidate]:
    """Read no more task rows than are needed to prove ambiguity."""
    candidates: list[StatusCandidate] = []
    for slug in boards:
        remaining = _MAX_CANDIDATES + 1 - len(candidates)
        if remaining <= 0:
            break
        filters = {"title_contains" if partial else "title": reference}
        candidates.extend(
            _candidate_for_task(task, slug)
            for task in _tasks_on_board(
                slug, include_archived=include_archived, limit=remaining,
                order_by="title", **filters,
            )
        )
    return candidates


def _project_reference_candidates(
    reference: str, *, board: Optional[str], partial: bool,
) -> list[StatusCandidate]:
    filters = {"reference_contains" if partial else "name": reference}
    if board is None:
        projects = _projects(limit=_MAX_CANDIDATES + 1, **filters)
    else:
        slug = kb._normalize_board_slug(board)
        metadata_project_id = kb.read_board_metadata(slug).get("project_id")
        linked = _projects(board_slug=slug, limit=_MAX_CANDIDATES + 1, **filters)
        metadata_project = _project_identity(str(metadata_project_id)) if metadata_project_id else None
        if metadata_project is not None:
            folded = reference.casefold()
            matches = (
                folded in metadata_project.name.casefold()
                or folded in metadata_project.slug.casefold()
                if partial
                else folded == metadata_project.name.casefold()
            )
            if matches:
                linked.append(metadata_project)
        associated = bool(metadata_project_id or _projects(board_slug=slug, limit=1))
        projects = linked if associated else _projects(limit=_MAX_CANDIDATES + 1, **filters)
    deduplicated = {project.id: project for project in projects}
    return [
        _candidate_for_project(project, board)
        for project in list(deduplicated.values())[:_MAX_CANDIDATES + 1]
    ]


def resolve_status_reference(
    reference: str, *, board: Optional[str] = None, include_archived: bool = False,
) -> StatusResolution:
    """Resolve a task or project deterministically; ambiguity never selects a winner."""
    ref = str(reference or "").strip()
    if not ref:
        return StatusResolution(reference=ref, error="task or project reference is required")
    all_boards = _board_slugs(None)
    try:
        exact_ids = [
            _candidate_for_task(task, slug)
            for slug in all_boards
            if (task := _task_on_board(slug, ref, include_archived=include_archived)) is not None
        ]
    except (OSError, sqlite3.Error) as exc:
        return StatusResolution(reference=ref, error=f"status read failed: {exc}")

    # Exact task identity is the first and strongest authority.
    if len(exact_ids) == 1:
        candidate = exact_ids[0]
        return StatusResolution(
            ok=True, scope="task", reference=ref, board=candidate.board, task_id=candidate.id,
        )
    if len(exact_ids) > 1:
        return _ambiguous(ref, exact_ids)

    try:
        boards = _board_slugs(board)
    except ValueError as exc:
        return StatusResolution(reference=ref, error=str(exc))

    try:
        project = _project_identity(ref)
        if project is not None and board is not None:
            slug = boards[0]
            board_project_id = kb.read_board_metadata(slug).get("project_id")
            associated = bool(board_project_id or _projects(board_slug=slug, limit=1))
            if associated and project.id != board_project_id and project.board_slug != slug:
                project = None
    except (OSError, sqlite3.Error) as exc:
        return StatusResolution(reference=ref, error=f"status read failed: {exc}")
    if project is not None:
        candidate = _candidate_for_project(project, board)
        return StatusResolution(
            ok=True,
            scope="project",
            reference=ref,
            board=candidate.board,
            project_id=candidate.id,
        )

    try:
        exact_matches = _task_reference_candidates(
            boards, ref, partial=False, include_archived=include_archived,
        )
        exact_matches += _project_reference_candidates(ref, board=board, partial=False)
    except (OSError, sqlite3.Error, ValueError) as exc:
        return StatusResolution(reference=ref, error=f"status read failed: {exc}")
    if len(exact_matches) == 1:
        candidate = exact_matches[0]
        return StatusResolution(
            ok=True,
            scope=candidate.kind,
            reference=ref,
            board=candidate.board,
            task_id=candidate.id if candidate.kind == "task" else None,
            project_id=candidate.id if candidate.kind == "project" else None,
        )
    if len(exact_matches) > 1:
        return _ambiguous(ref, exact_matches)

    try:
        partial_matches = _task_reference_candidates(
            boards, ref, partial=True, include_archived=include_archived,
        )
        partial_matches += _project_reference_candidates(ref, board=board, partial=True)
    except (OSError, sqlite3.Error, ValueError) as exc:
        return StatusResolution(reference=ref, error=f"status read failed: {exc}")
    if len(partial_matches) == 1:
        candidate = partial_matches[0]
        return StatusResolution(
            ok=True,
            scope=candidate.kind,
            reference=ref,
            board=candidate.board,
            task_id=candidate.id if candidate.kind == "task" else None,
            project_id=candidate.id if candidate.kind == "project" else None,
        )
    if len(partial_matches) > 1:
        return _ambiguous(ref, partial_matches)
    return StatusResolution(
        reference=ref,
        error=f"no task or project matches {_bounded_text(ref, 200)!r}",
    )


_NON_LIFECYCLE_EVENTS = frozenset({
    "commented", "attached", "attachment_removed", "heartbeat",
})
_PROJECT_TASK_LIMIT = 50


def _run_dict(run: Optional[kb.Run]) -> Optional[dict[str, Any]]:
    if run is None:
        return None
    return {
        "id": run.id,
        "profile": run.profile,
        "status": run.status,
        "outcome": run.outcome,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "summary": _bounded_text(run.summary),
    }


def _event_dict(event: Optional[kb.Event]) -> Optional[dict[str, Any]]:
    if event is None:
        return None
    return {
        "id": event.id,
        "kind": event.kind,
        "run_id": event.run_id,
        "created_at": event.created_at,
        "payload": _bounded_value(event.payload),
    }


def _metadata_status(metadata: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    meta = metadata if isinstance(metadata, dict) else {}
    raw_pr = meta.get("pull_request") or meta.get("pr")
    pr: Optional[dict[str, Any]] = None
    if isinstance(raw_pr, dict):
        pr = {
            key: _bounded_value(raw_pr.get(key))
            for key in ("id", "number", "url", "state")
            if raw_pr.get(key) is not None
        } or None
    elif any(meta.get(key) is not None for key in ("pr_id", "pr_number", "pr_url", "pr_state")):
        pr = {
            target: _bounded_value(meta.get(source))
            for target, source in (
                ("id", "pr_id"), ("number", "pr_number"), ("url", "pr_url"), ("state", "pr_state")
            )
            if meta.get(source) is not None
        }
    github = {"pr": pr, "merge_state": _bounded_value(meta.get("merge_state"))}
    deployment = {
        "deployment_state": _bounded_value(meta.get("deployment_state")),
        "production_state": _bounded_value(meta.get("production_state")),
        # This marker is always false: workflow/PR facts never infer runtime state.
        "inferred": False,
    }
    return github, deployment


def _provenance(event: Optional[kb.Event]) -> dict[str, Optional[str]]:
    payload = event.payload if event and isinstance(event.payload, dict) else {}
    implementer = payload.get("implementer")
    reviewer = payload.get("reviewer")
    return {
        "implementer": _bounded_text(implementer, 200)
        if isinstance(implementer, str) and implementer else None,
        "reviewer": _bounded_text(reviewer, 200)
        if isinstance(reviewer, str) and reviewer else None,
    }


def _block_reason(
    task: kb.Task, event: Optional[kb.Event], latest: Optional[kb.Run],
) -> Optional[str]:
    if task.status != "blocked":
        return None
    payload = event.payload if event and isinstance(event.payload, dict) else {}
    for key in ("policy_reason", "reason", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:400]
    if task.last_failure_error:
        return task.last_failure_error[:400]
    if latest and latest.error:
        return latest.error[:400]
    return task.block_kind


def _next_task_action(state: str, dependencies_satisfied: bool) -> str:
    if state == "triage":
        return "specify and assign the task"
    if state == "todo":
        return "wait for dependencies" if not dependencies_satisfied else "promote with /kanban promote"
    if state == "ready":
        return "wait for the existing dispatcher"
    if state == "running":
        return "inspect the active run"
    if state == "review":
        return "perform independent review"
    if state == "blocked":
        return "resolve the block, then use /kanban unblock"
    if state == "scheduled":
        return "wait for the schedule or use /kanban unblock"
    return "verify merge, deployment, and production separately" if state == "done" else "inspect task"


def _task_status(resolution: StatusResolution) -> ProjectStatusResult:
    assert resolution.board and resolution.task_id
    path = kb.kanban_db_path(board=resolution.board)
    with _read_only_connection(path) as conn:
        task = kb.get_task(conn, resolution.task_id)
        if task is None:
            return ProjectStatusResult(error=f"task {resolution.task_id!r} no longer exists")
        active = kb.get_run(conn, task.current_run_id) if task.current_run_id else None
        latest = kb.latest_run(conn, task.id)
        graph = kb.task_graph_context(conn, task.id)
        lifecycle = kb.find_latest_event(
            conn, task.id, exclude_kinds=_NON_LIFECYCLE_EVENTS,
        )
        review_event = kb.find_latest_event(
            conn, task.id, kinds=("review_requested", "changes_requested"),
        )
        block_event = kb.find_latest_event(
            conn, task.id, kinds=("blocked", "spawn_auto_blocked", "scheduled"),
        )
        review_loop_count = kb.count_events(conn, task.id, kind="changes_requested")
        route_summary = kbn.summarize_notify_subs(
            conn, task.id, profile_limit=_MAX_COLLECTION,
        )

    all_parents = graph.get("parents", [])
    dependencies_satisfied = all(parent.get("status") in {"done", "archived"} for parent in all_parents)
    parents = _bounded_value(all_parents)
    selected_run = active or latest
    profile = selected_run.profile if selected_run and selected_run.profile else task.assignee
    profile_source = "active_run" if active and active.profile else "latest_run" if latest and latest.profile else (
        "assignee" if task.assignee else None
    )
    owned_profiles = [
        _bounded_text(profile, 200) or ""
        for profile in route_summary["owned_profiles"]
    ]
    github, deployment = _metadata_status(latest.metadata if latest else None)
    project = _project_identity(task.project_id) if task.project_id else None
    repository_path = project.primary_path if project else None
    return ProjectStatusResult(
        ok=True,
        scope="task",
        task_id=task.id,
        project_id=task.project_id,
        project_name=_bounded_text(project.name, 200) if project else None,
        board=resolution.board,
        state=task.status,
        assignee=_bounded_text(task.assignee, 200),
        profile=_bounded_text(profile, 200),
        profile_source=profile_source,
        active_run=_run_dict(active),
        latest_run=_run_dict(latest),
        latest_lifecycle_event=_event_dict(lifecycle),
        provenance=_provenance(review_event),
        review_loop_count=review_loop_count,
        failure_loop_count=task.consecutive_failures,
        dependency_state={"satisfied": dependencies_satisfied, "parents": parents},
        workspace={
            "kind": task.workspace_kind,
            "path": _bounded_text(task.workspace_path),
            "repository_path": _bounded_text(repository_path),
            "project_id": task.project_id,
        },
        branch=_bounded_text(task.branch_name),
        github=github,
        deployment=deployment,
        policy_block_reason=_block_reason(task, block_event, latest),
        notification_route={
            "advisory_unowned": route_summary["advisory_unowned"],
            "owned_profiles": owned_profiles,
        },
        next_action=_next_task_action(task.status, dependencies_satisfied),
    )


def _project_status(resolution: StatusResolution) -> ProjectStatusResult:
    assert resolution.project_id
    project = _project_identity(resolution.project_id)
    if project is None:
        return ProjectStatusResult(error=f"project {resolution.project_id!r} no longer exists")
    boards = (resolution.board,) if resolution.board else _board_slugs(None)
    tasks: list[tuple[str, kb.Task]] = []
    counts: dict[str, int] = {}
    for board in boards:
        path = kb.kanban_db_path(board=board)
        if not path.is_file():
            continue
        with _read_only_connection(path) as conn:
            for state, count in kb.count_tasks_by_status(conn, project_id=project.id).items():
                counts[state] = counts.get(state, 0) + count
            remaining = _PROJECT_TASK_LIMIT + 1 - len(tasks)
            if remaining > 0:
                tasks.extend(
                    (board, task)
                    for task in kb.list_tasks(
                        conn, project_id=project.id, include_archived=False,
                        limit=remaining, order_by="created",
                    )
                )
    ordered = sorted(tasks, key=lambda item: (item[0], item[1].created_at, item[1].id))
    bounded = tuple(
        {
            "id": task.id,
            "name": _bounded_text(task.title, 200),
            "board": board,
            "state": task.status,
        }
        for board, task in ordered[:_PROJECT_TASK_LIMIT]
    )
    return ProjectStatusResult(
        ok=True,
        scope="project",
        project_id=project.id,
        project_name=_bounded_text(project.name, 200),
        board=resolution.board,
        workspace={
            "repository_path": _bounded_text(project.primary_path),
            "project_id": project.id,
        },
        project_task_counts=dict(sorted(counts.items())),
        project_tasks=bounded,
        project_tasks_truncated=sum(counts.values()) > _PROJECT_TASK_LIMIT,
        next_action="inspect active or blocked project tasks" if ordered else "create or link a Kanban task",
    )


def get_project_status(reference: str, *, board: Optional[str] = None) -> ProjectStatusResult:
    resolution = resolve_status_reference(reference, board=board)
    if not resolution.ok:
        return ProjectStatusResult(candidates=resolution.candidates, error=resolution.error)
    try:
        return _task_status(resolution) if resolution.scope == "task" else _project_status(resolution)
    except (OSError, sqlite3.Error) as exc:
        return ProjectStatusResult(error=f"status read failed: {exc}")


def project_status_command_enabled() -> bool:
    """Read the rollout gate dynamically; malformed or missing config fails closed."""
    try:
        from hermes_cli.config import cfg_get, read_raw_config

        config = read_raw_config()
        return is_truthy_value(
            cfg_get(config, "kanban", "project_status_command"), default=False
        )
    except Exception:
        return False


def _parse_command_args(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return ``(reference, board, error)`` for the canonical slash syntax."""
    raw = str(text or "").strip().lstrip("/")
    for command_name in ("project-status", "project_status"):
        if raw == command_name or raw.startswith(command_name + " "):
            raw = raw[len(command_name):].lstrip()
            break
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return None, None, f"Invalid arguments: {exc}"
    reference_parts: list[str] = []
    board: Optional[str] = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--board":
            index += 1
            if index >= len(tokens) or tokens[index].startswith("--"):
                return None, None, "Usage: /project-status <task|project> [--board <board>]"
            if board is not None:
                return None, None, "--board may be specified only once"
            board = tokens[index]
        elif token.startswith("--board="):
            if board is not None or not token.partition("=")[2]:
                return None, None, "--board requires one board name"
            board = token.partition("=")[2]
        elif token.startswith("--"):
            return None, None, f"Unknown option: {token}"
        else:
            reference_parts.append(token)
        index += 1
    reference = " ".join(reference_parts).strip()
    if not reference:
        return None, None, "Usage: /project-status <task|project> [--board <board>]"
    return reference, board, None


def run_project_status_slash(text: str) -> str:
    """Execute the canonical read-only command and return its operator rendering."""
    reference, board, error = _parse_command_args(text)
    if error:
        return error
    assert reference is not None
    return render_project_status(get_project_status(reference, board=board))


def render_project_status(result: ProjectStatusResult) -> str:
    """Render the bounded internal result as a concise Telegram-safe response."""
    if not result.ok:
        lines = [result.error or "Status unavailable. No action taken."]
        if result.candidates:
            lines.append("No action taken. Use an exact ID:")
            lines.extend(
                f"- {item.id} — {item.name}" + (f" [{item.board}]" if item.board else "")
                for item in result.candidates
            )
        return "\n".join(lines)
    if result.scope == "project":
        counts = ", ".join(f"{state}={count}" for state, count in result.project_task_counts.items()) or "no tasks"
        return "\n".join((
            f"{result.project_id} — project",
            f"Board: {result.board or '-'}",
            f"Tasks: {counts}",
            "Production: not inferred",
            f"Next: {result.next_action or '-'}",
        ))

    run = result.active_run or result.latest_run
    run_text = "-" if not run else f"{run['id']}/{run.get('outcome') or run.get('status') or '-'}"
    reviewer = result.provenance.get("reviewer")
    review_text = reviewer or ("pending" if result.state == "review" else "-")
    pr = result.github.get("pr")
    pr_text = "not recorded"
    if isinstance(pr, dict):
        identity = pr.get("number") or pr.get("id") or pr.get("url") or "recorded"
        pr_text = f"{identity}/{pr.get('state') or 'state not recorded'}"
    production = result.deployment.get("production_state") or "not inferred"
    return "\n".join((
        f"{result.task_id} — {result.state}",
        f"Profile: {result.profile or '-'}",
        f"Run: {run_text}",
        f"Review: {review_text}",
        f"Failures: {result.failure_loop_count}",
        f"PR: {pr_text}",
        f"Production: {production}",
        f"Next: {result.next_action or '-'}",
    ))
