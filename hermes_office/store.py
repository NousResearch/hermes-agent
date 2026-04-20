"""Persistence layer for Hermes Digital Office.

Profile-aware: every file lives under ``$HERMES_HOME/office/`` so each Hermes
profile gets its own isolated office.  All writes go through
:func:`atomic_write_text` which uses temp+rename to be crash-safe.

Layout (see ``design.md`` §3.1)::

    $HERMES_HOME/office/
    ├── departments/<dept_id>.json
    ├── employees/<emp_id>.json
    ├── tasks/<YYYYMMDD>.jsonl       (append-only)
    ├── activity/<emp_id>.jsonl      (ring-rotated)
    ├── telemetry.jsonl
    ├── weights.json                 (optional, set by `hermes office optimize`)
    └── .quarantine/<ts>/            (corrupt files moved here at boot)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import ActivityEvent, Department, Employee, Task

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# Path helpers
# ────────────────────────────────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Return the active Hermes profile root.

    Imports ``hermes_constants`` lazily so the office package can be imported in
    isolation (e.g. for tests that monkey-patch the env)."""
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def office_root() -> Path:
    return _hermes_home() / "office"


def _now_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ────────────────────────────────────────────────────────────────────────────
# Atomic file IO
# ────────────────────────────────────────────────────────────────────────────


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    Uses a per-pid temp file plus :func:`os.replace` (atomic on POSIX,
    near-atomic on Windows). Best-effort ``fsync`` on platforms that expose it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except (OSError, AttributeError):
        pass
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def append_jsonl(path: Path, payload: Any, *, max_lines: int | None = None) -> None:
    """Append a single JSON line to ``path``. If ``max_lines`` is set and the
    file exceeds it, rotate (truncate to the last ``max_lines // 2`` lines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    if max_lines is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > max_lines:
                kept = lines[-(max_lines // 2):]
                atomic_write_text(path, "".join(kept))
        except OSError:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Secret redaction
# ────────────────────────────────────────────────────────────────────────────

import re

# Lifted from the spirit of hermes_cli.config._CREDENTIAL_NAMES; kept local so
# we never break the office if the CLI module shape changes.
_SECRET_KEY_PAT = re.compile(
    r"\b(?:api[_-]?key|secret|token|bearer|password|passwd|client_secret|"
    r"openai[_-]?api[_-]?key|anthropic[_-]?api[_-]?key|"
    r"hf[_-]?token|access[_-]?token|refresh[_-]?token)\b"
    r"\s*[:=]\s*([A-Za-z0-9_\-\.+/=]{16,})",
    re.IGNORECASE,
)
_BEARER_PAT = re.compile(r"\b(Bearer\s+)([A-Za-z0-9_\-\.]{16,})", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    if not text:
        return text
    text = _SECRET_KEY_PAT.sub(lambda m: m.group(0).replace(m.group(1), "***REDACTED***"), text)
    text = _BEARER_PAT.sub(lambda m: m.group(1) + "***REDACTED***", text)
    return text


# ────────────────────────────────────────────────────────────────────────────
# Store
# ────────────────────────────────────────────────────────────────────────────


class Store:
    """In-memory cache backed by JSON files on disk.

    Thread-safe via a single :class:`threading.RLock`.  All write methods
    persist to disk before returning.  Reads return copies (Pydantic models are
    immutable-ish; callers should treat returned values as read-only and use
    update methods to mutate)."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else office_root()
        self._lock = threading.RLock()
        self._employees: dict[str, Employee] = {}
        self._departments: dict[str, Department] = {}

    # ── path helpers ───────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        return self._root

    def _emp_path(self, emp_id: str) -> Path:
        return self._root / "employees" / f"{emp_id}.json"

    def _dept_path(self, dept_id: str) -> Path:
        return self._root / "departments" / f"{dept_id}.json"

    def _activity_path(self, emp_id: str) -> Path:
        return self._root / "activity" / f"{emp_id}.jsonl"

    def _task_log_path(self, when: datetime | None = None) -> Path:
        when = when or datetime.now(tz=timezone.utc)
        return self._root / "tasks" / f"{when.strftime('%Y%m%d')}.jsonl"

    @property
    def telemetry_path(self) -> Path:
        return self._root / "telemetry.jsonl"

    @property
    def weights_path(self) -> Path:
        return self._root / "weights.json"

    # ── boot ───────────────────────────────────────────────────────────────

    def boot_from_disk(self) -> dict[str, int]:
        """Load all employees + departments from disk; quarantine bad files.

        Returns a counts dict for logging: {departments, employees, quarantined}.
        """
        with self._lock:
            self._employees.clear()
            self._departments.clear()
            counts = {"departments": 0, "employees": 0, "quarantined": 0}
            (self._root / "departments").mkdir(parents=True, exist_ok=True)
            (self._root / "employees").mkdir(parents=True, exist_ok=True)

            for path in (self._root / "departments").glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    dept = Department.model_validate(payload)
                    self._departments[dept.id] = dept
                    counts["departments"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Quarantining bad department file %s: %s", path, exc)
                    self._quarantine(path)
                    counts["quarantined"] += 1

            for path in (self._root / "employees").glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    emp = Employee.model_validate(payload)
                    if emp.department_id not in self._departments:
                        # Orphan; quarantine rather than crash.
                        logger.warning(
                            "Employee %s references missing dept %s — quarantining",
                            emp.id,
                            emp.department_id,
                        )
                        self._quarantine(path)
                        counts["quarantined"] += 1
                        continue
                    self._employees[emp.id] = emp
                    counts["employees"] += 1
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Quarantining bad employee file %s: %s", path, exc)
                    self._quarantine(path)
                    counts["quarantined"] += 1

            # Reconcile dept.employee_ids with reality (drop unknown ids).
            for dept in self._departments.values():
                live = [eid for eid in dept.employee_ids if eid in self._employees]
                if live != dept.employee_ids:
                    self._departments[dept.id] = dept.model_copy(update={"employee_ids": live})
                    self._persist_department(self._departments[dept.id])

            # Pick up employees whose dept doesn't list them (defensive).
            for emp in self._employees.values():
                dept = self._departments.get(emp.department_id)
                if dept and emp.id not in dept.employee_ids:
                    self._departments[dept.id] = dept.model_copy(
                        update={"employee_ids": dept.employee_ids + [emp.id]}
                    )
                    self._persist_department(self._departments[dept.id])

            return counts

    def _quarantine(self, path: Path) -> None:
        target = self._root / ".quarantine" / _now_str()
        target.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(path), str(target / path.name))
        except OSError:
            # Last resort: at least delete it so we don't loop on next boot.
            try:
                path.unlink()
            except OSError:
                pass

    # ── department CRUD ────────────────────────────────────────────────────

    def list_departments(self) -> list[Department]:
        with self._lock:
            return [d.model_copy() for d in self._departments.values()]

    def get_department(self, dept_id: str) -> Department | None:
        with self._lock:
            d = self._departments.get(dept_id)
            return d.model_copy() if d else None

    def create_department(self, dept: Department) -> Department:
        with self._lock:
            if dept.id in self._departments:
                raise ValueError(f"department {dept.id} already exists")
            self._departments[dept.id] = dept
            self._persist_department(dept)
            return dept.model_copy()

    def update_department(self, dept_id: str, **fields: Any) -> Department:
        with self._lock:
            existing = self._departments.get(dept_id)
            if existing is None:
                raise KeyError(dept_id)
            updated = existing.model_copy(update={**fields, "updated_at": datetime.now(tz=timezone.utc)})
            self._departments[dept_id] = updated
            self._persist_department(updated)
            return updated.model_copy()

    def delete_department(self, dept_id: str) -> list[str]:
        """Delete dept + cascade-delete its employees.  Returns deleted employee ids."""
        with self._lock:
            existing = self._departments.pop(dept_id, None)
            if existing is None:
                raise KeyError(dept_id)
            self._dept_path(dept_id).unlink(missing_ok=True)
            removed_emps: list[str] = []
            for emp_id in list(existing.employee_ids):
                if emp_id in self._employees:
                    self._employees.pop(emp_id, None)
                    self._emp_path(emp_id).unlink(missing_ok=True)
                    removed_emps.append(emp_id)
            return removed_emps

    def _persist_department(self, dept: Department) -> None:
        atomic_write_json(self._dept_path(dept.id), dept.model_dump(mode="json"))

    # ── employee CRUD ──────────────────────────────────────────────────────

    def list_employees(self, dept_id: str | None = None) -> list[Employee]:
        with self._lock:
            return [
                e.model_copy()
                for e in self._employees.values()
                if dept_id is None or e.department_id == dept_id
            ]

    def get_employee(self, emp_id: str) -> Employee | None:
        with self._lock:
            e = self._employees.get(emp_id)
            return e.model_copy() if e else None

    def create_employee(self, emp: Employee) -> Employee:
        with self._lock:
            if emp.department_id not in self._departments:
                raise ValueError(f"unknown department {emp.department_id!r}")
            if emp.id in self._employees:
                raise ValueError(f"employee {emp.id} already exists")
            self._employees[emp.id] = emp
            self._persist_employee(emp)
            dept = self._departments[emp.department_id]
            if emp.id not in dept.employee_ids:
                self._departments[dept.id] = dept.model_copy(
                    update={"employee_ids": dept.employee_ids + [emp.id]}
                )
                self._persist_department(self._departments[dept.id])
            return emp.model_copy()

    def update_employee(self, emp_id: str, **fields: Any) -> Employee:
        with self._lock:
            existing = self._employees.get(emp_id)
            if existing is None:
                raise KeyError(emp_id)
            merged = {**fields, "updated_at": datetime.now(tz=timezone.utc), "revision": existing.revision + 1}
            updated = existing.model_copy(update=merged)
            self._employees[emp_id] = updated
            self._persist_employee(updated)
            return updated.model_copy()

    def delete_employee(self, emp_id: str) -> None:
        with self._lock:
            existing = self._employees.pop(emp_id, None)
            if existing is None:
                raise KeyError(emp_id)
            self._emp_path(emp_id).unlink(missing_ok=True)
            dept = self._departments.get(existing.department_id)
            if dept and emp_id in dept.employee_ids:
                self._departments[dept.id] = dept.model_copy(
                    update={"employee_ids": [x for x in dept.employee_ids if x != emp_id]}
                )
                self._persist_department(self._departments[dept.id])
            # Activity log left in place; user can clear via "Reset memory".

    def _persist_employee(self, emp: Employee) -> None:
        atomic_write_json(self._emp_path(emp.id), emp.model_dump(mode="json"))

    # ── activity ───────────────────────────────────────────────────────────

    def append_activity(self, evt: ActivityEvent) -> None:
        # NB: the eventbus already redacts; we redact again here to defend in
        # depth (callers may bypass the bus during recovery).
        payload = evt.model_dump(mode="json")
        payload["text"] = redact_secrets(payload.get("text", ""))
        append_jsonl(self._activity_path(evt.employee_id), payload, max_lines=10_000)

    def read_activity(self, emp_id: str, *, limit: int = 50, cursor: int | None = None) -> tuple[list[dict[str, Any]], int | None]:
        path = self._activity_path(emp_id)
        if not path.exists():
            return [], None
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return [], None
        end = cursor if cursor is not None else len(lines)
        end = max(0, min(end, len(lines)))
        start = max(0, end - limit)
        out: list[dict[str, Any]] = []
        for ln in lines[start:end]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        next_cursor = start if start > 0 else None
        return out, next_cursor

    # ── tasks ──────────────────────────────────────────────────────────────

    def append_task(self, task: Task) -> None:
        append_jsonl(self._task_log_path(), task.model_dump(mode="json"))

    def read_recent_tasks(self, days: int = 1) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen_paths: list[Path] = []
        from datetime import timedelta

        today = datetime.now(tz=timezone.utc)
        for delta in range(days):
            seen_paths.append(self._task_log_path(today - timedelta(days=delta)))
        for p in seen_paths:
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            out.append(json.loads(ln))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
        return out

    # ── export / import ────────────────────────────────────────────────────

    def export(self) -> dict[str, Any]:
        with self._lock:
            return {
                "version": 1,
                "exported_at": datetime.now(tz=timezone.utc).isoformat(),
                "departments": [d.model_dump(mode="json") for d in self._departments.values()],
                "employees": [e.model_dump(mode="json") for e in self._employees.values()],
            }

    def import_(self, payload: dict[str, Any]) -> dict[str, int]:
        """Replace state with payload's; back up the previous state first."""
        if not isinstance(payload, dict):
            raise ValueError("import payload must be a dict")
        if int(payload.get("version", 0)) != 1:
            raise ValueError("unsupported export version")
        with self._lock:
            backup_dir = self._root / ".backups" / _now_str()
            backup_dir.mkdir(parents=True, exist_ok=True)
            for sub in ("departments", "employees"):
                src = self._root / sub
                if src.exists():
                    shutil.copytree(src, backup_dir / sub)
                    shutil.rmtree(src)
                src.mkdir(parents=True, exist_ok=True)

            self._employees.clear()
            self._departments.clear()
            for d in payload.get("departments", []):
                dept = Department.model_validate(d)
                self._departments[dept.id] = dept
                self._persist_department(dept)
            for e in payload.get("employees", []):
                emp = Employee.model_validate(e)
                self._employees[emp.id] = emp
                self._persist_employee(emp)
            return {"departments": len(self._departments), "employees": len(self._employees)}

    # ── iteration helper ───────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Employee]:
        return iter(self.list_employees())
