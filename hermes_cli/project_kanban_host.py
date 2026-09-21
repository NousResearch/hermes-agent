"""Versioned in-process contract for canonical Hermes Projects and Kanban.

Product adapters consume this service instead of opening Hermes database files.
The implementation composes only public domain APIs and returns plain,
versioned data or one safe error family.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator

import hermes_constants
from hermes_cli import kanban_db
from hermes_cli import projects_db

CONTRACT_VERSION = 2
MAX_PAGE_SIZE = 500

_READ_CAPABILITIES = frozenset(
    {
        "get_board",
        "get_project",
        "get_task",
        "list_comments",
        "list_epics",
        "list_events",
        "list_links",
        "list_projects",
        "list_task_children",
        "list_tasks",
        "list_profiles",
        "validate_project",
    }
)
_WRITE_CAPABILITIES = frozenset(
    {
        "add_comment",
        "assign_task",
        "attach_task_to_epic",
        "block_task",
        "create_epic",
        "create_task",
        "detach_task_from_epic",
        "get_epic",
        "link_tasks",
        "transition_task",
        "unlink_tasks",
        "unblock_task",
        "update_epic",
        "update_task",
        "provision_project",
    }
)
_CAPABILITIES = _READ_CAPABILITIES | _WRITE_CAPABILITIES
_UNSET = object()
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_PROJECT_NAME = 120
MAX_PROJECT_DESCRIPTION = 4000
MAX_IDEMPOTENCY_KEY = 200


class HostError(RuntimeError):
    """Safe, stable error returned by the host contract."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        fields: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fields = fields or {}

    def to_envelope(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "error": {
                "code": self.code,
                "message": self.message,
                "fields": dict(self.fields),
            },
        }


def _plain(value: Any) -> Any:
    """Convert public domain values into JSON-compatible primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _record(value: Any) -> dict[str, Any]:
    return _plain(asdict(value))


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_PAGE_SIZE:
        raise HostError(
            "invalid_limit",
            "limit must be between 1 and 500",
            fields={"limit": value},
        )
    return value


def _page(items: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    size = _limit(limit)
    return {
        "items": items[:size],
        "limit": size,
        "has_more": len(items) > size,
    }


@contextmanager
def _safe_write_errors() -> Iterator[None]:
    """Map canonical domain write failures to the stable host error family."""
    try:
        yield
    except HostError:
        raise
    except PermissionError:
        raise HostError(
            "write_forbidden",
            "canonical write is not permitted",
        ) from None
    except ValueError:
        raise HostError(
            "validation_error",
            "canonical write was rejected",
        ) from None
    except RuntimeError:
        raise HostError(
            "write_conflict",
            "canonical write could not be applied",
        ) from None


class ProjectKanbanHost:
    """Versioned host service scoped to one explicit Hermes home and board."""

    def __init__(
        self,
        *,
        hermes_home: str | Path,
        board: str = "default",
        allowed_repo_root: str | Path | None = None,
    ) -> None:
        self.hermes_home = Path(hermes_home)
        self.board = (board or "").strip() or "default"
        self.allowed_repo_root = (
            Path(allowed_repo_root).expanduser().resolve()
            if allowed_repo_root is not None
            else None
        )

    @contextmanager
    def _scope(self, board: str | None = None) -> Iterator[str]:
        selected = (board or self.board or "").strip() or "default"
        token = hermes_constants.set_hermes_home_override(self.hermes_home)
        try:
            with kanban_db.scoped_kanban_home(self.hermes_home), kanban_db.scoped_current_board(selected):
                try:
                    yield selected
                except HostError:
                    raise
                except Exception:
                    raise HostError(
                        "host_unavailable",
                        "canonical host operation failed",
                    ) from None
        finally:
            hermes_constants.reset_hermes_home_override(token)

    def capabilities(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "methods": sorted(_CAPABILITIES),
        }

    def require_capability(self, name: str) -> None:
        if name not in _CAPABILITIES:
            raise HostError(
                "unsupported_capability",
                "capability is not available",
                fields={"capability": name},
            )

    def list_profiles(self) -> list[dict[str, Any]]:
        """List profile identities only from this host's explicit root."""
        profiles = [{"name": "default", "is_default": True}]
        profiles_root = self.hermes_home / "profiles"
        if not profiles_root.is_dir():
            return profiles
        for entry in sorted(profiles_root.iterdir(), key=lambda item: item.name.lower()):
            if (
                not entry.is_dir()
                or not _PROFILE_NAME_RE.fullmatch(entry.name)
                or not ((entry / "SOUL.md").is_file() or (entry / "config.yaml").is_file())
            ):
                continue
            profiles.append({"name": entry.name, "is_default": False})
        return profiles

    def _ensure_board(self, board: str) -> dict[str, Any]:
        with projects_db.connect_closing() as conn:
            provisioning = projects_db.get_project_provisioning_by_board(conn, board)
        if provisioning is not None and provisioning.get("status") != "complete":
            raise HostError(
                "board_not_found",
                "board was not found",
                fields={"board": board},
            )
        match = next(
            (
                item
                for item in kanban_db.list_boards(include_archived=True)
                if item.get("slug") == board
            ),
            None,
        )
        if match is None:
            raise HostError(
                "board_not_found",
                "board was not found",
                fields={"board": board},
            )
        metadata = kanban_db.read_board_metadata(board)
        return {
            "slug": board,
            "name": metadata.get("name") or match.get("name") or board,
            "description": metadata.get("description") or "",
            "icon": metadata.get("icon"),
            "color": metadata.get("color"),
            "project_id": metadata.get("project_id") or match.get("project_id"),
            "archived": bool(metadata.get("archived", match.get("archived", False))),
        }

    def list_projects(self) -> list[dict[str, Any]]:
        with self._scope():
            with projects_db.connect_closing() as conn:
                return [project.to_dict() for project in projects_db.list_projects(conn)]

    def get_project(self, id_or_slug: str) -> dict[str, Any]:
        with self._scope():
            with projects_db.connect_closing() as conn:
                project = projects_db.get_project(conn, id_or_slug)
            if project is None:
                raise HostError(
                    "project_not_found",
                    "project was not found",
                    fields={"project": id_or_slug},
                )
            return project.to_dict()

    def get_board(self, board: str | None = None) -> dict[str, Any]:
        with self._scope(board) as selected:
            return self._ensure_board(selected)

    @staticmethod
    def _project_request_digest(
        *,
        name: str,
        slug: str,
        description: str,
        repo_path: str,
        lead_profile: str,
        board_slug: str | None,
    ) -> str:
        path = str(Path(str(repo_path or "")).expanduser().resolve(strict=False))
        payload = {
            "name": str(name or "").strip(),
            "slug": str(slug or "").strip().lower(),
            "description": str(description or "").strip(),
            "repo_path": path,
            "lead_profile": str(lead_profile or "").strip(),
            "board_slug": str(board_slug or slug or "").strip().lower(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _validate_project(
        self,
        *,
        name: str,
        slug: str,
        description: str,
        repo_path: str,
        lead_profile: str,
        board_slug: str | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        errors: dict[str, list[str]] = {}
        clean_name = str(name or "").strip()
        if not clean_name:
            errors["name"] = ["Project name is required"]
        elif len(clean_name) > MAX_PROJECT_NAME:
            errors["name"] = ["Project name is too long"]

        raw_slug = str(slug or "").strip()
        clean_slug: str | None = None
        try:
            clean_slug = projects_db.normalize_slug(raw_slug)
            if clean_slug is None or raw_slug != raw_slug.lower():
                raise ValueError
        except ValueError:
            errors["slug"] = ["Use a lowercase project slug with letters, numbers, hyphens or underscores"]

        clean_description = str(description or "").strip()
        if not clean_description:
            errors["description"] = ["Project mission is required"]
        elif len(clean_description) > MAX_PROJECT_DESCRIPTION:
            errors["description"] = ["Project mission is too long"]

        raw_board = str(board_slug or clean_slug or "").strip()
        clean_board: str | None = None
        try:
            clean_board = kanban_db.normalize_board_slug(raw_board)
            if clean_board is None or raw_board != raw_board.lower():
                raise ValueError
        except ValueError:
            errors["board_slug"] = ["Use a lowercase board slug with letters, numbers, hyphens or underscores"]

        clean_path: Path | None = None
        candidate = Path(str(repo_path or "")).expanduser()
        if not candidate.is_absolute():
            errors["repo_path"] = ["Repository path must be absolute"]
        else:
            try:
                clean_path = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                errors["repo_path"] = ["Repository path does not exist"]
            if clean_path is not None and not clean_path.is_dir():
                errors["repo_path"] = ["Repository path must be a directory"]
            if (
                clean_path is not None
                and self.allowed_repo_root is not None
                and not clean_path.is_relative_to(self.allowed_repo_root)
            ):
                errors["repo_path"] = ["Repository path is outside the allowed root"]

        clean_lead = str(lead_profile or "").strip()
        profile_names = {item["name"] for item in self.list_profiles()}
        if clean_lead not in profile_names:
            errors["lead_profile"] = ["Lead profile is not available"]

        if not errors and clean_slug and clean_board and clean_path:
            with projects_db.connect_closing() as conn:
                existing = projects_db.get_project(conn, clean_slug)
                if existing is not None:
                    errors["slug"] = ["Project slug is already in use"]
                if any(
                    project.name.casefold() == clean_name.casefold()
                    for project in projects_db.list_projects(conn)
                ):
                    errors["name"] = ["Project name is already in use"]
                if projects_db.find_by_primary_path(conn, str(clean_path)) is not None:
                    errors["repo_path"] = ["Repository path already belongs to a project"]
                board_journal = projects_db.get_project_provisioning_by_board(
                    conn,
                    clean_board,
                )
            board_owned_by_request = (
                idempotency_key is not None
                and board_journal is not None
                and board_journal.get("idempotency_key") == idempotency_key
            )
            if kanban_db.board_exists(clean_board) and not board_owned_by_request:
                errors["board_slug"] = ["Board slug is already in use"]

        if errors:
            raise HostError(
                "validation_error",
                "Project details are invalid",
                fields=errors,
            )
        return {
            "name": clean_name,
            "slug": clean_slug,
            "description": clean_description,
            "repo_path": str(clean_path),
            "lead_profile": clean_lead,
            "board_slug": clean_board,
        }

    def validate_project(
        self,
        *,
        name: str,
        slug: str,
        description: str,
        repo_path: str,
        lead_profile: str,
        board_slug: str | None = None,
    ) -> dict[str, Any]:
        with self._scope():
            return self._validate_project(
                name=name,
                slug=slug,
                description=description,
                repo_path=repo_path,
                lead_profile=lead_profile,
                board_slug=board_slug,
            )

    def _completed_provisioning_result(
        self,
        journal: dict[str, Any],
        *,
        replayed: bool,
    ) -> dict[str, Any]:
        project_id = str(journal.get("project_id") or "")
        if not project_id:
            raise HostError(
                "host_unavailable",
                "canonical project provisioning state is unavailable",
            )
        project = self.get_project(project_id)
        board = self.get_board(str(journal["board_slug"]))
        return {
            "status": "complete",
            "idempotency_key": journal["idempotency_key"],
            "replayed": replayed,
            "project": project,
            "board": board,
        }

    def provision_project(
        self,
        *,
        name: str,
        slug: str,
        description: str,
        repo_path: str,
        lead_profile: str,
        idempotency_key: str,
        board_slug: str | None = None,
    ) -> dict[str, Any]:
        key = str(idempotency_key or "").strip()
        if not key or len(key) > MAX_IDEMPOTENCY_KEY:
            raise HostError(
                "validation_error",
                "Project details are invalid",
                fields={"idempotency_key": ["A valid idempotency key is required"]},
            )
        digest = self._project_request_digest(
            name=name,
            slug=slug,
            description=description,
            repo_path=repo_path,
            lead_profile=lead_profile,
            board_slug=board_slug,
        )

        with self._scope():
            kanban_db.assert_mutation_allowed()
            with projects_db.connect_closing() as conn:
                journal = projects_db.get_project_provisioning(conn, key)
            if journal is not None:
                if journal.get("request_digest") != digest:
                    raise HostError(
                        "idempotency_conflict",
                        "Idempotency key was already used for different project details",
                        fields={"idempotency_key": ["Use a new idempotency key"]},
                    )
                if journal.get("status") == "complete":
                    return self._completed_provisioning_result(journal, replayed=True)

            validated = self._validate_project(
                name=name,
                slug=slug,
                description=description,
                repo_path=repo_path,
                lead_profile=lead_profile,
                board_slug=board_slug,
                idempotency_key=key,
            )
            if journal is None:
                try:
                    with projects_db.connect_closing() as conn:
                        journal = projects_db.begin_project_provisioning(
                            conn,
                            idempotency_key=key,
                            request_digest=digest,
                            project_slug=validated["slug"],
                            board_slug=validated["board_slug"],
                        )
                except Exception:
                    raise HostError(
                        "write_conflict",
                        "canonical project could not be provisioned",
                    ) from None

            board = validated["board_slug"]
            board_created = False
            project_created = False
            project = None
            try:
                with projects_db.connect_closing() as conn:
                    project = projects_db.find_project_by_provisioning_key(conn, key)
                if not kanban_db.board_exists(board):
                    kanban_db.create_board(
                        board,
                        name=f"{validated['name']} Board",
                        description=validated["description"],
                        default_workdir=validated["repo_path"],
                    )
                    board_created = True
                if project is None:
                    with projects_db.connect_closing() as conn:
                        project_id = projects_db.create_project(
                            conn,
                            name=validated["name"],
                            slug=validated["slug"],
                            description=validated["description"],
                            primary_path=validated["repo_path"],
                            folders=[validated["repo_path"]],
                            board_slug=board,
                            provisioning_key=key,
                        )
                        project_created = True
                        if not projects_db.bind_project_provisioning(conn, key, project_id):
                            raise RuntimeError("provisioning journal could not be bound")
                        project = projects_db.find_project_by_provisioning_key(conn, key)
                if project is None:
                    raise RuntimeError("provisioned project could not be read")
                kanban_db.create_board(
                    board,
                    name=f"{validated['name']} Board",
                    description=validated["description"],
                    default_workdir=validated["repo_path"],
                    project_id=project.id,
                )
                with projects_db.connect_closing() as conn:
                    if not projects_db.complete_project_provisioning(conn, key, project.id):
                        raise RuntimeError("provisioning journal could not be completed")
                    journal = projects_db.get_project_provisioning(conn, key)
            except Exception:
                compensation_ok = True
                if project_created and project is not None:
                    try:
                        with projects_db.connect_closing() as conn:
                            compensation_ok = (
                                projects_db.delete_project(conn, project.id)
                                and compensation_ok
                            )
                    except Exception:
                        compensation_ok = False
                if board_created:
                    try:
                        kanban_db.remove_board(board, archive=False)
                    except Exception:
                        compensation_ok = False
                if not compensation_ok:
                    raise HostError(
                        "write_conflict",
                        "canonical project provisioning needs recovery",
                    ) from None
                raise HostError(
                    "write_conflict",
                    "canonical project could not be provisioned",
                ) from None

            if journal is None:
                raise HostError(
                    "host_unavailable",
                    "canonical project provisioning state is unavailable",
                )
            return self._completed_provisioning_result(journal, replayed=False)

    def list_tasks(
        self,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                tasks = kanban_db.list_tasks(conn, limit=size + 1)
            return _page([_record(task) for task in tasks], size)

    def list_epics(
        self,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                epics = kanban_db.list_epics(conn, board_slug=selected)
            items = sorted((_record(epic) for epic in epics), key=lambda item: (item["created_at"], item["id"]))
            return _page(items, size)

    @staticmethod
    def _require_task(conn: Any, task_id: str) -> Any:
        task = kanban_db.get_task(conn, task_id)
        if task is None:
            raise HostError(
                "task_not_found",
                "task was not found",
                fields={"task": task_id},
            )
        return task

    @staticmethod
    def _require_epic(conn: Any, epic_id: str) -> Any:
        epic = kanban_db.get_epic(conn, epic_id)
        if epic is None:
            raise HostError(
                "epic_not_found",
                "epic was not found",
                fields={"epic": epic_id},
            )
        return epic

    @staticmethod
    def _comments(conn: Any, task_id: str, limit: int) -> dict[str, Any]:
        items = sorted(
            (_record(comment) for comment in kanban_db.list_comments(conn, task_id)),
            key=lambda item: (item["created_at"], item["id"]),
        )
        return _page(items, limit)

    @staticmethod
    def _events(conn: Any, task_id: str, limit: int) -> dict[str, Any]:
        items = sorted(
            (_record(event) for event in kanban_db.list_events(conn, task_id)),
            key=lambda item: (item["created_at"], item["id"]),
        )
        return _page(items, limit)

    @staticmethod
    def _links(conn: Any, task_id: str, limit: int) -> dict[str, Any]:
        items = [
            {"direction": "parent", "task_id": item}
            for item in sorted(kanban_db.parent_ids(conn, task_id))
        ]
        items.extend(
            {"direction": "child", "task_id": item}
            for item in sorted(kanban_db.child_ids(conn, task_id))
        )
        return _page(items, limit)

    @staticmethod
    def _children(conn: Any, task_id: str, limit: int) -> dict[str, Any]:
        items = []
        for child_id in sorted(kanban_db.get_subtask_children(conn, task_id)):
            child = kanban_db.get_task(conn, child_id)
            if child is not None:
                items.append(_record(child))
        return _page(items, limit)

    def get_task(
        self,
        task_id: str,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                task = self._require_task(conn, task_id)
                epic = kanban_db.get_epic(conn, task.epic_id) if task.epic_id else None
                parent = kanban_db.get_task_parent(conn, task_id)
                return {
                    "task": _record(task),
                    "epic": _record(epic) if epic is not None else None,
                    "parent_task": _record(parent) if parent is not None else None,
                    "children": self._children(conn, task_id, size),
                    "links": self._links(conn, task_id, size),
                    "comments": self._comments(conn, task_id, size),
                    "events": self._events(conn, task_id, size),
                    "workflow_run": None,
                }

    def list_comments(
        self,
        task_id: str,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                return self._comments(conn, task_id, size)

    def list_events(
        self,
        task_id: str,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                return self._events(conn, task_id, size)

    def list_links(
        self,
        task_id: str,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                return self._links(conn, task_id, size)

    def list_task_children(
        self,
        task_id: str,
        *,
        board: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        size = _limit(limit)
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                return self._children(conn, task_id, size)

    def create_task(
        self,
        *,
        title: str,
        body: str | None = None,
        assignee: str | None = None,
        created_by: str = "project-kanban-host",
        tenant: str | None = None,
        priority: int = 0,
        workspace_kind: str = "scratch",
        workspace_path: str | None = None,
        parents: list[str] | tuple[str, ...] | None = None,
        triage: bool = False,
        idempotency_key: str | None = None,
        initial_status: str = "running",
        max_runtime_seconds: int | None = None,
        skills: list[str] | None = None,
        goal_mode: bool = False,
        goal_max_turns: int | None = None,
        model_override: str | None = None,
        provider_override: str | None = None,
        reasoning_effort: str | None = None,
        project_id: str | None = None,
        task_kind: str | None = None,
        parent_task_id: str | None = None,
        epic_id: str | None = None,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                with _safe_write_errors():
                    task_id = kanban_db.create_task(
                        conn,
                        title=title,
                        body=body,
                        assignee=assignee,
                        created_by=created_by,
                        workspace_kind=workspace_kind,
                        workspace_path=workspace_path,
                        tenant=tenant,
                        priority=priority,
                        parents=parents or (),
                        triage=triage,
                        idempotency_key=idempotency_key,
                        initial_status=initial_status,
                        max_runtime_seconds=max_runtime_seconds,
                        skills=skills,
                        goal_mode=goal_mode,
                        goal_max_turns=goal_max_turns,
                        model_override=model_override,
                        provider_override=provider_override,
                        reasoning_effort=reasoning_effort,
                        board=selected,
                        project_id=project_id,
                        task_kind=task_kind,
                        parent_task_id=parent_task_id,
                        epic_id=epic_id,
                    )
                return _record(self._require_task(conn, task_id))

    def update_task(
        self,
        task_id: str,
        *,
        title: Any = _UNSET,
        body: Any = _UNSET,
        priority: Any = _UNSET,
        task_kind: Any = _UNSET,
        parent_task_id: Any = _UNSET,
        epic_id: Any = _UNSET,
        model_override: Any = _UNSET,
        provider_override: Any = _UNSET,
        reasoning_effort: Any = _UNSET,
        board: str | None = None,
    ) -> dict[str, Any]:
        changes = {
            name: value
            for name, value in (
                ("title", title),
                ("body", body),
                ("priority", priority),
                ("task_kind", task_kind),
                ("parent_task_id", parent_task_id),
                ("epic_id", epic_id),
            )
            if value is not _UNSET
        }
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                current = self._require_task(conn, task_id)
                with _safe_write_errors():
                    task = kanban_db.update_task(
                        conn,
                        task_id,
                        board=selected,
                        **changes,
                    )
                    if task is None:
                        raise HostError("task_not_found", "task was not found")
                    if (
                        model_override is not _UNSET
                        or provider_override is not _UNSET
                    ):
                        selected_model = (
                            current.model_override
                            if model_override is _UNSET
                            else model_override
                        )
                        selected_provider = (
                            current.provider_override
                            if provider_override is _UNSET
                            else provider_override
                        )
                        changed = kanban_db.set_model_override(
                            conn,
                            task_id,
                            selected_model,
                            provider=selected_provider,
                        )
                        if not changed:
                            raise HostError("task_not_found", "task was not found")
                    if reasoning_effort is not _UNSET:
                        changed = kanban_db.set_reasoning_effort(
                            conn,
                            task_id,
                            reasoning_effort,
                        )
                        if not changed:
                            raise HostError("task_not_found", "task was not found")
                return _record(self._require_task(conn, task_id))

    def transition_task(
        self,
        task_id: str,
        status: str,
        *,
        reason: str | None = None,
        block_kind: str | None = None,
        result: str | None = None,
        summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        reviewer: str | None = None,
        force_review: bool = False,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    changed = kanban_db.transition_task(
                        conn,
                        task_id,
                        status,
                        reason=reason,
                        block_kind=block_kind,
                        result=result,
                        summary=summary,
                        metadata=metadata,
                        reviewer=reviewer,
                        force_review=force_review,
                        actor="project-kanban-host",
                    )
                if not changed:
                    # A dependency refusal must say WHICH parent is open, not the
                    # generic text (parity with the dashboard's _open_parent_refusal;
                    # the host-backed PATCH path bypasses that helper).
                    detail = None
                    if status in ("done", "review"):
                        blockers = kanban_db.unsatisfied_parents(conn, task_id)
                        if blockers:
                            detail = "; ".join(f"{pid} ({st})" for pid, st in blockers)
                    if detail:
                        raise HostError(
                            "transition_conflict",
                            f"cannot move {task_id} to {status!r}: unsatisfied parent "
                            f"dependencies: {detail}; complete the parents first "
                            f"(done or archived)",
                        )
                    raise HostError(
                        "transition_conflict",
                        "task cannot enter the requested state",
                    )
                return _record(self._require_task(conn, task_id))

    def assign_task(
        self,
        task_id: str,
        assignee: str | None,
        *,
        reclaim_first: bool = False,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    if reclaim_first:
                        changed = kanban_db.reassign_task(
                            conn,
                            task_id,
                            assignee,
                            reclaim_first=True,
                        )
                    else:
                        changed = kanban_db.assign_task(conn, task_id, assignee)
                if not changed:
                    raise HostError(
                        "write_conflict",
                        "task assignment could not be applied",
                    )
                return _record(self._require_task(conn, task_id))

    def block_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        kind: str | None = None,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    changed = kanban_db.block_task(
                        conn,
                        task_id,
                        reason=reason,
                        kind=kind,
                    )
                if not changed:
                    raise HostError(
                        "transition_conflict",
                        "task cannot be blocked from its current state",
                    )
                return _record(self._require_task(conn, task_id))

    def unblock_task(
        self,
        task_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    changed = kanban_db.unblock_task(conn, task_id)
                if not changed:
                    raise HostError(
                        "transition_conflict",
                        "task cannot be unblocked from its current state",
                    )
                return _record(self._require_task(conn, task_id))

    def create_epic(
        self,
        *,
        title: str,
        description: str | None = None,
        parent_epic_id: str | None = None,
        status: str = "active",
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                with _safe_write_errors():
                    epic_id = kanban_db.create_epic(
                        conn,
                        title=title,
                        description=description,
                        board_slug=selected,
                        parent_epic_id=parent_epic_id,
                        status=status,
                    )
                return _record(self._require_epic(conn, epic_id))

    def get_epic(
        self,
        epic_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                return _record(self._require_epic(conn, epic_id))

    def update_epic(
        self,
        epic_id: str,
        *,
        title: Any = _UNSET,
        description: Any = _UNSET,
        status: Any = _UNSET,
        parent_epic_id: Any = _UNSET,
        board: str | None = None,
    ) -> dict[str, Any]:
        changes = {
            name: value
            for name, value in (
                ("title", title),
                ("description", description),
                ("status", status),
                ("parent_epic_id", parent_epic_id),
            )
            if value is not _UNSET
        }
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_epic(conn, epic_id)
                with _safe_write_errors():
                    epic = kanban_db.update_epic(conn, epic_id, **changes)
                if epic is None:
                    raise HostError("epic_not_found", "epic was not found")
                return _record(epic)

    def attach_task_to_epic(
        self,
        task_id: str,
        epic_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                self._require_epic(conn, epic_id)
                with _safe_write_errors():
                    changed = kanban_db.set_task_epic(
                        conn,
                        task_id,
                        epic_id,
                        board=selected,
                    )
                if not changed:
                    raise HostError("task_not_found", "task was not found")
                return _record(self._require_task(conn, task_id))

    def detach_task_from_epic(
        self,
        task_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    changed = kanban_db.set_task_epic(
                        conn,
                        task_id,
                        None,
                        board=selected,
                    )
                if not changed:
                    raise HostError("task_not_found", "task was not found")
                return _record(self._require_task(conn, task_id))

    def add_comment(
        self,
        task_id: str,
        *,
        author: str,
        body: str,
        board: str | None = None,
    ) -> dict[str, Any]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, task_id)
                with _safe_write_errors():
                    comment_id = kanban_db.add_comment(conn, task_id, author, body)
                comment = next(
                    (
                        item
                        for item in kanban_db.list_comments(conn, task_id)
                        if item.id == comment_id
                    ),
                    None,
                )
                if comment is None:
                    raise HostError(
                        "host_unavailable",
                        "canonical host operation failed",
                    )
                return _record(comment)

    def link_tasks(
        self,
        parent_task_id: str,
        child_task_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, str]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, parent_task_id)
                self._require_task(conn, child_task_id)
                with _safe_write_errors():
                    kanban_db.link_tasks(conn, parent_task_id, child_task_id)
                return {
                    "parent_task_id": parent_task_id,
                    "child_task_id": child_task_id,
                }

    def unlink_tasks(
        self,
        parent_task_id: str,
        child_task_id: str,
        *,
        board: str | None = None,
    ) -> dict[str, str]:
        with self._scope(board) as selected:
            self._ensure_board(selected)
            with kanban_db.connect_closing(board=selected) as conn:
                self._require_task(conn, parent_task_id)
                self._require_task(conn, child_task_id)
                with _safe_write_errors():
                    changed = kanban_db.unlink_tasks(
                        conn,
                        parent_task_id,
                        child_task_id,
                    )
                if not changed:
                    raise HostError(
                        "link_not_found",
                        "task link was not found",
                    )
                return {
                    "parent_task_id": parent_task_id,
                    "child_task_id": child_task_id,
                }
