"""Import external task stores into the native Kanban lifecycle.

The adapter boundary deliberately owns source reads and writeback.  Importers
never edit an adapter's storage directly, which lets future stores provide
transactional APIs without changing Kanban.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import yaml
from yaml.nodes import MappingNode, ScalarNode
from yaml.tokens import AliasToken, AnchorToken

from hermes_cli import kanban_db as kb


_SOURCE_RUNNABLE = {"pending", "todo", "ready"}
_SOURCE_OWNED = {"importing", "imported", "running", "review", "blocked", "done", "archived"}
_MIRROR_STATUS = {
    "triage": "blocked",
    "todo": "imported",
    "scheduled": "imported",
    "ready": "imported",
    "running": "running",
    "review": "review",
    "blocked": "blocked",
    "done": "done",
    "archived": "archived",
}
_SUPPORTED_FIELDS = {
    "id", "title", "status", "assignee", "priority", "depends_on",
    "workspace", "skills", "tenant", "max_runtime", "hermes_kanban",
}


class ImportAdapter(Protocol):
    def scan(self) -> Iterable["ExternalTask"]: ...
    def write_state(self, task: "ExternalTask", state: str, marker: dict[str, Any]) -> None: ...
    def claim(self, task: "ExternalTask", marker: dict[str, Any]): ...


@dataclass(frozen=True)
class ExternalTask:
    source_id: str
    title: str
    body: str
    status: str
    assignee: str | None
    priority: int
    depends_on: tuple[str, ...]
    workspace_kind: str
    workspace_path: str | None
    skills: tuple[str, ...]
    tenant: str | None
    max_runtime_seconds: int | None
    revision: str
    definition_revision: str
    path: Path
    metadata: dict[str, Any]


@dataclass
class ImportResult:
    source_id: str
    action: str
    task_id: str | None = None
    error: str | None = None


class MarkdownAdapter:
    """Frontmatter-backed adapter with atomic same-directory writeback."""

    def __init__(self, source: Path):
        self.source = source.expanduser().resolve()
        if not self.source.exists():
            raise ValueError(f"source does not exist: {self.source}")
        if not self.source.is_dir():
            raise ValueError("markdown source must be a directory")

    def scan(self) -> Iterable[ExternalTask]:
        for path in sorted(self.source.glob("*.md")):
            resolved = path.resolve()
            if resolved.parent != self.source:
                raise ValueError(f"task path escapes source directory: {path}")
            yield self._read(resolved)

    @staticmethod
    def _split(text: str, path: Path) -> tuple[dict[str, Any], str]:
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError(f"{path.name}: missing YAML frontmatter")
        try:
            end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration as exc:
            raise ValueError(f"{path.name}: unterminated YAML frontmatter") from exc
        frontmatter = "\n".join(lines[1:end])
        try:
            if any(isinstance(token, (AliasToken, AnchorToken)) for token in yaml.scan(frontmatter)):
                raise ValueError(f"{path.name}: YAML anchors and aliases are not supported")
            metadata = yaml.safe_load(frontmatter) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name}: invalid YAML frontmatter: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{path.name}: YAML frontmatter must be an object")
        return metadata, "\n".join(lines[end + 1 :]).strip()

    @staticmethod
    def _workspace(value: Any, path: Path) -> tuple[str, str | None]:
        raw = str(value or "scratch").strip()
        if raw in {"scratch", "worktree"}:
            return raw, None
        for prefix, kind in (("dir:", "dir"), ("worktree:", "worktree")):
            if raw.startswith(prefix):
                target = Path(raw[len(prefix):].strip()).expanduser()
                if not target.is_absolute():
                    raise ValueError(f"{path.name}: workspace path must be absolute")
                return kind, str(target.resolve())
        raise ValueError(f"{path.name}: unsupported workspace {raw!r}")

    @staticmethod
    def _duration(value: Any, path: Path) -> int | None:
        if value in (None, ""):
            return None
        raw = str(value).strip().lower()
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        try:
            return int(raw) if raw[-1:].isdigit() else int(raw[:-1]) * units[raw[-1]]
        except (ValueError, KeyError) as exc:
            raise ValueError(f"{path.name}: invalid max_runtime {value!r}") from exc

    def _read(self, path: Path, text: str | None = None) -> ExternalTask:
        if text is None:
            text = path.read_text(encoding="utf-8")
        metadata, body = self._split(text, path)
        unknown = sorted(set(metadata) - _SUPPORTED_FIELDS)
        if unknown:
            raise ValueError(f"{path.name}: unsupported field(s): {', '.join(unknown)}")
        for field in ("id", "title", "status", "assignee", "tenant", "workspace"):
            value = metadata.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{path.name}: {field} must be a string")
        marker = metadata.get("hermes_kanban")
        if marker is not None and not isinstance(marker, dict):
            raise ValueError(f"{path.name}: hermes_kanban must be an object")
        priority_value = metadata.get("priority", 0)
        if isinstance(priority_value, bool) or not isinstance(priority_value, int):
            raise ValueError(f"{path.name}: priority must be an integer")
        source_id = (metadata.get("id") or "").strip()
        title = (metadata.get("title") or "").strip()
        status = (metadata.get("status") or "pending").strip().lower()
        if not source_id or not title:
            raise ValueError(f"{path.name}: id and title are required")
        if status not in _SOURCE_RUNNABLE | _SOURCE_OWNED:
            raise ValueError(f"{path.name}: unsupported status {status!r}")
        depends = metadata.get("depends_on") or []
        skills = metadata.get("skills") or []
        if (
            not isinstance(depends, list)
            or not all(isinstance(value, str) for value in depends)
            or not isinstance(skills, list)
            or not all(isinstance(value, str) for value in skills)
        ):
            raise ValueError(f"{path.name}: depends_on and skills must be lists of strings")
        kind, workspace_path = self._workspace(metadata.get("workspace"), path)
        semantic = {k: v for k, v in metadata.items() if k != "hermes_kanban"}
        revision = hashlib.sha256(
            (json.dumps(semantic, sort_keys=True, ensure_ascii=False) + "\n" + body).encode("utf-8")
        ).hexdigest()
        definition = {
            k: v for k, v in metadata.items()
            if k not in {"status", "hermes_kanban"}
        }
        definition_revision = hashlib.sha256(
            (json.dumps(definition, sort_keys=True, ensure_ascii=False) + "\n" + body).encode("utf-8")
        ).hexdigest()
        return ExternalTask(
            source_id=source_id,
            title=title,
            body=body,
            status=status,
            assignee=(str(metadata["assignee"]).strip() if metadata.get("assignee") else None),
            priority=priority_value,
            depends_on=tuple(str(v).strip() for v in depends if str(v).strip()),
            workspace_kind=kind,
            workspace_path=workspace_path,
            skills=tuple(str(v).strip() for v in skills if str(v).strip()),
            tenant=(str(metadata["tenant"]).strip() if metadata.get("tenant") else None),
            max_runtime_seconds=self._duration(metadata.get("max_runtime"), path),
            revision=revision,
            definition_revision=definition_revision,
            path=path,
            metadata=metadata,
        )

    @staticmethod
    def _dump_value(value: Any) -> str:
        rendered = yaml.safe_dump(
            value,
            allow_unicode=True,
            default_flow_style=True,
            sort_keys=False,
        ).strip()
        if rendered.endswith("\n..."):
            rendered = rendered[:-4]
        return rendered

    @classmethod
    def _patch_frontmatter(
        cls,
        text: str,
        path: Path,
        updates: dict[str, Any],
    ) -> str:
        lines = text.splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            raise ValueError(f"{path.name}: missing YAML frontmatter")
        try:
            end_line = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        except StopIteration as exc:
            raise ValueError(f"{path.name}: unterminated YAML frontmatter") from exc

        content_start = len(lines[0])
        content_end = sum(len(line) for line in lines[:end_line])
        frontmatter = text[content_start:content_end]
        try:
            document = yaml.compose(frontmatter)
        except yaml.YAMLError as exc:
            raise ValueError(f"{path.name}: invalid YAML frontmatter: {exc}") from exc
        if not isinstance(document, MappingNode):
            raise ValueError(f"{path.name}: YAML frontmatter must be an object")

        replacements: list[tuple[int, int, str]] = []
        found: set[str] = set()
        for key_node, value_node in document.value:
            if not isinstance(key_node, ScalarNode) or key_node.value not in updates:
                continue
            found.add(key_node.value)
            replacements.append((
                value_node.start_mark.index,
                value_node.end_mark.index,
                cls._dump_value(updates[key_node.value]),
            ))
        for start, end, replacement in reversed(replacements):
            frontmatter = frontmatter[:start] + replacement + frontmatter[end:]

        missing = [key for key in updates if key not in found]
        if missing:
            newline = "\r\n" if lines[0].endswith("\r\n") else "\n"
            if frontmatter and not frontmatter.endswith(("\n", "\r")):
                frontmatter += newline
            frontmatter += "".join(
                f"{key}: {cls._dump_value(updates[key])}{newline}"
                for key in missing
            )
        return text[:content_start] + frontmatter + text[content_end:]

    def _write_state_locked(
        self,
        task: ExternalTask,
        state: str,
        marker: dict[str, Any],
    ) -> None:
        with task.path.open("r", encoding="utf-8", newline="") as source:
            original = source.read()
        current = self._read(task.path, original)
        current_marker = current.metadata.get("hermes_kanban") or {}
        if current.status == state and current_marker == marker:
            return
        unchanged = (
            current.revision == task.revision
            and current_marker == (task.metadata.get("hermes_kanban") or {})
        )
        resumable_claim = (
            task.status in _SOURCE_RUNNABLE | {"imported"}
            and current.status == "importing"
            and current_marker.get("import_id") == marker.get("import_id")
            and current.definition_revision == task.definition_revision
        )
        if not unchanged and not resumable_claim:
            raise ValueError(f"{task.path.name}: source changed before writeback")
        text = self._patch_frontmatter(
            original,
            task.path,
            {"status": state, "hermes_kanban": marker},
        )
        fd, temp_name = tempfile.mkstemp(prefix=f".{task.path.name}.", dir=task.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            if os.name != "nt":
                os.chmod(temp_name, stat.S_IMODE(task.path.stat().st_mode))
            os.replace(temp_name, task.path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    @contextmanager
    def _source_lock(self, path: Path):
        lock_path = path.with_name(f".{path.name}.hermes.lock")
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+b") as handle:
            deadline = time.monotonic() + 5.0
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        handle.seek(0)
                        # Windows byte-range locks require one byte to exist.
                        if handle.read(1) == b"":
                            handle.seek(0)
                            handle.write(b"0")
                            handle.flush()
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (OSError, BlockingIOError):
                    if time.monotonic() >= deadline:
                        raise ValueError(f"{path.name}: timed out acquiring source ownership")
                    time.sleep(0.05)
            try:
                yield
            finally:
                if os.name == "nt":
                    import msvcrt
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def write_state(self, task: ExternalTask, state: str, marker: dict[str, Any]) -> None:
        with self._source_lock(task.path):
            self._write_state_locked(task, state, marker)

    @contextmanager
    def claim(self, task: ExternalTask, marker: dict[str, Any]):
        """Serialize ownership transfer and compare against the scanned record."""
        with self._source_lock(task.path):
            current = self._read(task.path)
            current_marker = current.metadata.get("hermes_kanban") or {}
            same_resume = (
                current.status in {"importing", "imported"}
                and current_marker.get("import_id") == marker["import_id"]
                and current.definition_revision == task.definition_revision
            )
            if not same_resume and (
                current.revision != task.revision or current.status not in _SOURCE_RUNNABLE
            ):
                raise ValueError(
                    f"{task.path.name}: source changed or is already owned by another importer"
                )
            self._write_state_locked(current, "importing", marker)
            yield current


def _ensure_schema(conn) -> None:
    with kb.write_txn(conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS task_imports (
                import_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_revision TEXT NOT NULL,
                task_id TEXT NOT NULL,
                mirrored_status TEXT NOT NULL,
                updated_at INTEGER NOT NULL,
                PRIMARY KEY (import_id, source_id),
                UNIQUE (source_path),
                UNIQUE (task_id)
            )
        """)


def _event(conn, task_id: str, kind: str, payload: dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
        (task_id, kind, json.dumps(payload, ensure_ascii=False), int(time.time())),
    )


def _mapped_assignee(task: ExternalTask, mapping: dict[str, str]) -> str | None:
    return mapping.get(task.assignee, task.assignee) if task.assignee else None


def sync_import(
    conn,
    *,
    adapter: ImportAdapter,
    import_id: str,
    assignee_map: dict[str, str] | None = None,
    dry_run: bool = False,
) -> list[ImportResult]:
    """Import runnable records and mirror native state back to their source."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_imports'"
    ).fetchone() is not None
    if not dry_run:
        _ensure_schema(conn)
        table_exists = True
    mapping = assignee_map or {}
    results: list[ImportResult] = []
    try:
        tasks = list(adapter.scan())
    except (OSError, TypeError, ValueError) as exc:
        return [ImportResult(source_id="", action="error", error=str(exc))]
    by_id: dict[str, ExternalTask] = {}
    duplicate_ids: set[str] = set()
    for task in tasks:
        if task.source_id in by_id:
            results.append(ImportResult(task.source_id, "error", error="duplicate source id"))
            duplicate_ids.add(task.source_id)
        else:
            by_id[task.source_id] = task

    ledger = (
        {
            row["source_id"]: row
            for row in conn.execute(
                "SELECT * FROM task_imports WHERE import_id = ?", (import_id,)
            )
        }
        if table_exists else {}
    )
    path_owners = (
        {
            row["source_path"]: row
            for row in conn.execute("SELECT * FROM task_imports")
        }
        if table_exists else {}
    )
    foreign_owned: set[str] = set()
    for source_id, task in by_id.items():
        if source_id in duplicate_ids:
            continue
        marker = task.metadata.get("hermes_kanban") or {}
        owner = marker.get("import_id")
        if owner and owner != import_id:
            results.append(ImportResult(
                source_id,
                "conflict",
                marker.get("task_id"),
                f"source is owned by importer {owner!r}",
            ))
            foreign_owned.add(source_id)
            continue
        path_owner = path_owners.get(str(task.path))
        if path_owner and (
            path_owner["import_id"] != import_id
            or path_owner["source_id"] != source_id
        ):
            results.append(ImportResult(
                source_id,
                "conflict",
                path_owner["task_id"],
                f"source path is owned by importer {path_owner['import_id']!r}",
            ))
            foreign_owned.add(source_id)
    known_profiles = set(kb.list_profiles_on_disk())

    # Existing imports are mirror-only. A source-side transition back to a
    # runnable state is a conflict; Kanban remains the single lifecycle owner.
    for source_id, row in ledger.items():
        if source_id in foreign_owned or source_id in duplicate_ids:
            continue
        task = by_id.get(source_id)
        native = kb.get_task(conn, row["task_id"])
        if task is None:
            results.append(ImportResult(source_id, "error", row["task_id"], "source task was deleted"))
            continue
        if native is None:
            results.append(ImportResult(source_id, "error", row["task_id"], "native task is missing"))
            continue
        desired = _MIRROR_STATUS[native.status]
        if task.status in _SOURCE_RUNNABLE:
            results.append(ImportResult(source_id, "conflict", native.id, "source became runnable after import"))
            continue
        marker = {"import_id": import_id, "task_id": native.id, "state": desired}
        if task.status != desired or task.metadata.get("hermes_kanban") != marker:
            if not dry_run:
                try:
                    adapter.write_state(task, desired, marker)
                except ValueError as exc:
                    results.append(ImportResult(source_id, "conflict", native.id, str(exc)))
                    continue
                except OSError as exc:
                    results.append(ImportResult(source_id, "error", native.id, str(exc)))
                    continue
                with kb.write_txn(conn):
                    conn.execute(
                        "UPDATE task_imports SET mirrored_status=?, updated_at=? WHERE import_id=? AND source_id=?",
                        (desired, int(time.time()), import_id, source_id),
                    )
                    _event(conn, native.id, "import_mirrored", {"import_id": import_id, "source_id": source_id, "status": desired})
            results.append(ImportResult(source_id, "would_mirror" if dry_run else "mirrored", native.id))
        else:
            results.append(ImportResult(source_id, "unchanged", native.id))

    pending = [
        task for task in by_id.values()
        if task.source_id not in ledger
        and task.source_id not in foreign_owned
        and task.source_id not in duplicate_ids
        and task.status in _SOURCE_RUNNABLE | {"importing"}
    ]
    unresolved = {task.source_id: task for task in pending}
    planned_task_ids: dict[str, str] = {
        source_id: row["task_id"] for source_id, row in ledger.items()
    }
    while unresolved:
        progressed = False
        for source_id, task in list(unresolved.items()):
            ambiguous = [dep for dep in task.depends_on if dep in duplicate_ids]
            if ambiguous:
                results.append(ImportResult(
                    source_id,
                    "error",
                    error=f"ambiguous duplicate dependencies: {', '.join(ambiguous)}",
                ))
                del unresolved[source_id]
                continue
            missing = [dep for dep in task.depends_on if dep not in by_id]
            if missing:
                results.append(ImportResult(source_id, "error", error=f"unknown dependencies: {', '.join(missing)}"))
                del unresolved[source_id]
                continue
            if any(dep in unresolved for dep in task.depends_on):
                continue
            assignee = _mapped_assignee(task, mapping)
            if not assignee or assignee not in known_profiles:
                results.append(ImportResult(source_id, "error", error=f"unknown or unassigned profile: {assignee or '(none)'}"))
                del unresolved[source_id]
                continue
            parent_ids = []
            dep_error = None
            for dep in task.depends_on:
                if dry_run:
                    dep_task_id = planned_task_ids.get(dep)
                else:
                    dep_row = conn.execute(
                        "SELECT task_id FROM task_imports WHERE import_id=? AND source_id=?",
                        (import_id, dep),
                    ).fetchone()
                    dep_task_id = dep_row["task_id"] if dep_row else None
                if not dep_task_id:
                    dep_error = f"dependency {dep!r} was not imported"
                    break
                parent_ids.append(dep_task_id)
            if dep_error:
                results.append(ImportResult(source_id, "error", error=dep_error))
                del unresolved[source_id]
                continue
            if dry_run:
                results.append(ImportResult(source_id, "would_import"))
                planned_task_ids[source_id] = f"planned:{source_id}"
                del unresolved[source_id]
                progressed = True
                continue
            # Source lock first: after this durable transition no external
            # consumer should claim the record. A crash here is recoverable
            # because subsequent syncs accept `importing` as resumable.
            claim_marker = {"import_id": import_id, "state": "importing"}
            try:
                with adapter.claim(task, claim_marker) as claimed:
                    source_key = hashlib.sha256(str(task.path).encode("utf-8")).hexdigest()
                    with kb.write_txn(conn):
                        existing = conn.execute(
                            "SELECT task_id FROM task_imports WHERE import_id=? AND source_id=?",
                            (import_id, source_id),
                        ).fetchone()
                        resumed_existing = existing is not None
                        if resumed_existing:
                            native_id = existing["task_id"]
                        else:
                            native_id = kb.create_task(
                                conn,
                                title=task.title,
                                body=task.body or None,
                                assignee=assignee,
                                created_by=f"import:{import_id}",
                                workspace_kind=task.workspace_kind,
                                workspace_path=task.workspace_path,
                                priority=task.priority,
                                parents=parent_ids,
                                tenant=task.tenant,
                                idempotency_key=f"external-import:{source_key}",
                                max_runtime_seconds=task.max_runtime_seconds,
                                skills=task.skills,
                            )
                            conn.execute(
                                "INSERT INTO task_imports VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (import_id, source_id, str(task.path), task.revision, native_id, "imported", int(time.time())),
                            )
                            _event(conn, native_id, "imported", {"import_id": import_id, "source_id": source_id, "source_revision": task.revision})
                adapter.write_state(claimed, "imported", {
                    "import_id": import_id, "task_id": native_id, "state": "imported",
                })
            except (OSError, ValueError) as exc:
                results.append(ImportResult(source_id, "conflict", error=str(exc)))
                del unresolved[source_id]
                continue
            results.append(ImportResult(
                source_id, "unchanged" if resumed_existing else "imported", native_id,
            ))
            del unresolved[source_id]
            progressed = True
        if not progressed:
            for source_id in unresolved:
                results.append(ImportResult(source_id, "error", error="dependency cycle"))
            break
    return results
