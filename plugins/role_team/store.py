"""Concurrency-safe, single-document role-team plan persistence."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional

fcntl: Any
try:  # POSIX
    import fcntl as _fcntl
    fcntl = _fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

msvcrt: Any
try:  # Windows
    import msvcrt as _msvcrt
    msvcrt = _msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ACTIVE = frozenset({"preparing", "queued", "running", "delegated"})
_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_LOCK_DEPTH = threading.local()


class PlanLockError(RuntimeError):
    pass


class ActiveRoleInvocation(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class PlanStore:
    """One atomic JSON document plus short per-plan publication locks.

    Manifest, findings, execution plan, utilization, summary, and invocation
    records share one document, so readers cannot observe a half-published
    multi-file transition. Packet/output files are immutable artifacts and are
    written atomically while holding the same short plan lock; no role work is
    executed while that lock is held.
    """

    def __init__(self, workspace_root: Path | str, plan_id: str):
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        if not self.workspace_root.is_dir():
            raise ValueError(f"workspace root does not exist: {self.workspace_root}")
        if not _ID_RE.fullmatch(str(plan_id or "")):
            raise ValueError("plan_id must be a safe 1-128 character identifier")
        self.plan_id = str(plan_id)
        self.plan_dir = self.workspace_root / "_plans" / self.plan_id
        self.state_path = self.plan_dir / "role-team-state.json"
        self.lock_path = self.plan_dir / ".role-team.lock"

    def _open_lock_file(self):
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        return self.lock_path.open("a+b")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        key = str(self.lock_path)
        depths = getattr(_LOCK_DEPTH, "values", None)
        if depths is None:
            depths = {}
            _LOCK_DEPTH.values = depths
        if depths.get(key, 0):
            depths[key] += 1
            try:
                yield
            finally:
                depths[key] -= 1
            return

        with _LOCKS_GUARD:
            local_lock = _LOCKS.setdefault(key, threading.RLock())
        with local_lock:
            handle = None
            acquired = False
            depths[key] = 1
            try:
                handle = self._open_lock_file()
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    handle.seek(0)
                    getattr(msvcrt, "locking")(
                        handle.fileno(), getattr(msvcrt, "LK_LOCK"), 1
                    )
                else:  # pragma: no cover - unsupported runtime
                    raise OSError("no cross-process file lock implementation")
                acquired = True
                yield
            except PlanLockError:
                raise
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                if not acquired:
                    raise PlanLockError(f"could not acquire plan lock: {exc}") from exc
                raise
            finally:
                try:
                    if acquired and handle is not None:
                        if fcntl is not None:
                            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                        elif msvcrt is not None:  # pragma: no cover - Windows
                            handle.seek(0)
                            getattr(msvcrt, "locking")(
                                handle.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                            )
                finally:
                    if handle is not None:
                        handle.close()
                    depths.pop(key, None)

    def _default_state(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "plan_id": self.plan_id,
            "manifest": {"schema_version": 1, "role_sessions": []},
            "execution_plan": {
                "schema_version": 1,
                "workflow_sequence": [],
                "roles": [],
            },
            "findings": {"schema_version": 1, "items": []},
            "utilization": {"schema_version": 1, "roles": []},
            "summary": "",
            "invocations": {},
            "updated_at": _now(),
        }

    def _normalize(self, raw: Any) -> Dict[str, Any]:
        state = copy.deepcopy(raw) if isinstance(raw, dict) else {}
        default = self._default_state()
        for key, value in default.items():
            if key not in state:
                state[key] = copy.deepcopy(value)
        if state.get("plan_id") != self.plan_id:
            raise ValueError("plan state belongs to a different plan_id")
        for section in ("manifest", "execution_plan", "findings", "utilization"):
            if not isinstance(state.get(section), dict):
                raise ValueError(f"invalid {section} section")
            for key, value in default[section].items():
                if key not in state[section]:
                    state[section][key] = copy.deepcopy(value)
        if not isinstance(state.get("invocations"), dict):
            raise ValueError("invalid invocations section")
        return state

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return self._default_state()
        return self._normalize(json.loads(self.state_path.read_text(encoding="utf-8")))

    def _write_unlocked(self, state: Dict[str, Any]) -> None:
        state["updated_at"] = _now()
        payload = json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _atomic_write_text(self.state_path, payload)

    def snapshot(self) -> Dict[str, Any]:
        with self._locked():
            return copy.deepcopy(self._read_unlocked())

    def mutate(self, callback: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        with self._locked():
            state = self._read_unlocked()
            replacement = callback(state)
            if replacement is not None:
                if not isinstance(replacement, dict):
                    raise TypeError("plan mutation must return a mapping or None")
                state = replacement
            state = self._normalize(state)
            self._write_unlocked(state)
            return copy.deepcopy(state)

    @staticmethod
    def _section_record(record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in {"packet", "output", "error"}
        }

    def reserve_invocation(self, record: Dict[str, Any]) -> Dict[str, Any]:
        invocation_id = str(record.get("invocation_id") or "")
        role_slug = str(record.get("role_slug") or "")
        if not _ID_RE.fullmatch(invocation_id) or not _ID_RE.fullmatch(role_slug):
            raise ValueError("invocation_id and role_slug must be safe identifiers")

        def reserve(state: Dict[str, Any]) -> None:
            for existing in state["invocations"].values():
                if existing.get("role_slug") == role_slug and existing.get("status") in _ACTIVE:
                    raise ActiveRoleInvocation(
                        f"role {record.get('role') or role_slug} already has an active invocation"
                    )
            item = copy.deepcopy(record)
            item.setdefault("created_at", _now())
            state["invocations"][invocation_id] = item
            projection = self._section_record(item)
            state["manifest"]["role_sessions"].append(copy.deepcopy(projection))
            state["execution_plan"]["roles"].append(copy.deepcopy(projection))
            state["utilization"]["roles"].append(copy.deepcopy(projection))

        return self.mutate(reserve)

    def transition(self, invocation_id: str, **patch: Any) -> Dict[str, Any]:
        def update(state: Dict[str, Any]) -> None:
            if invocation_id not in state["invocations"]:
                raise KeyError(f"unknown invocation {invocation_id}")
            state["invocations"][invocation_id].update(copy.deepcopy(patch))
            state["invocations"][invocation_id]["updated_at"] = _now()
            for section, key in (
                ("manifest", "role_sessions"),
                ("execution_plan", "roles"),
                ("utilization", "roles"),
            ):
                for item in state[section][key]:
                    if item.get("invocation_id") == invocation_id:
                        item.update(self._section_record(patch))
                        item["updated_at"] = state["invocations"][invocation_id]["updated_at"]
            if patch.get("status") == "completed" and "summary" in patch:
                state["summary"] = str(patch["summary"])
            elif patch.get("status") in {"blocked", "cancelled", "error"}:
                detail = patch.get("error") or patch.get("end_reason") or "terminal failure"
                state["summary"] = f"{patch['status']}: {detail}"

        return self.mutate(update)

    def write_artifact(
        self,
        role_slug: str,
        invocation_id: str,
        kind: str,
        content: str,
    ) -> str:
        if not _ID_RE.fullmatch(role_slug) or not _ID_RE.fullmatch(invocation_id):
            raise ValueError("unsafe artifact identifier")
        if kind not in {"packet", "output", "evidence"}:
            raise ValueError("unsupported artifact kind")
        path = self.plan_dir / "roles" / role_slug / f"{invocation_id}-{kind}.md"
        with self._locked():
            _atomic_write_text(path, str(content))
        return str(path.relative_to(self.workspace_root))
