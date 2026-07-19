"""Persistent identity and safe backfill planning for Discord workspaces.

A Discord thread is the user-visible workspace. Header identity and the small
workspace context map are profile-local and partitioned first by canonical
Discord guild ``SessionSource.scope_id`` and then by thread id. Registry reads
fail closed, and every read-modify-write transaction holds both an in-process
lock and an OS file lock so gateway and backfill processes cannot lose each
other's bindings.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from utils import atomic_json_write

if os.name == "nt":
    import msvcrt
else:
    import fcntl


_STATE_VERSION = 1
_STATE_FILENAME = "discord_workspace_headers.json"
_KNOWN_HERMES_PLACEHOLDER_TITLES = frozenset(
    {"hermes", "hermes chat", "new post"}
)
_LOCAL_LOCKS_GUARD = threading.Lock()
_LOCAL_LOCKS: dict[str, threading.RLock] = {}


class WorkspaceHeaderStoreError(RuntimeError):
    """Raised when registry identity cannot be read or safely persisted."""


def _clean_id(value: Any) -> str:
    return str(value or "").strip()


def _clean_workspace_text(value: Any, field_name: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if len(text) > 1024:
        raise ValueError(f"{field_name} exceeds the 1024-character limit")
    return text


def _thread_scope_id(thread: Any) -> str:
    guild = getattr(thread, "guild", None)
    return _clean_id(getattr(guild, "id", None))


def is_known_hermes_placeholder_title(title: Any) -> bool:
    """Return whether a title is one of Hermes' exact generic placeholders."""
    normalized = " ".join(str(title or "").split()).casefold()
    return normalized in _KNOWN_HERMES_PLACEHOLDER_TITLES


def _local_registry_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCAL_LOCKS_GUARD:
        return _LOCAL_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_registry_lock(path: Path) -> Iterator[None]:
    """Hold a blocking cross-process lock for one registry transaction."""
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _local_registry_lock(lock_path):
        with lock_path.open("a+b") as handle:
            if os.name == "nt":
                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True, slots=True)
class WorkspaceHeaderState:
    owner: str = "Hermes"
    status: str = "Active"
    linked_issue_or_artifact: str = "Not linked"
    last_decision: str = "No decision recorded"
    next_action: str = "Awaiting next assistant turn"

    @classmethod
    def from_mapping(cls, payload: Any) -> "WorkspaceHeaderState":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise WorkspaceHeaderStoreError("workspace state must be an object")
        unknown = set(payload) - {
            "owner",
            "status",
            "linked_issue_or_artifact",
            "last_decision",
            "next_action",
        }
        if unknown:
            raise WorkspaceHeaderStoreError(
                "workspace state has unknown fields: " + ", ".join(sorted(unknown))
            )
        defaults = cls()
        try:
            return cls(
                owner=_clean_workspace_text(payload.get("owner", defaults.owner), "owner"),
                status=_clean_workspace_text(payload.get("status", defaults.status), "status"),
                linked_issue_or_artifact=_clean_workspace_text(
                    payload.get(
                        "linked_issue_or_artifact",
                        defaults.linked_issue_or_artifact,
                    ),
                    "linked_issue_or_artifact",
                ),
                last_decision=_clean_workspace_text(
                    payload.get("last_decision", defaults.last_decision),
                    "last_decision",
                ),
                next_action=_clean_workspace_text(
                    payload.get("next_action", defaults.next_action),
                    "next_action",
                ),
            )
        except ValueError as error:
            raise WorkspaceHeaderStoreError(str(error)) from error

    def to_mapping(self) -> dict[str, str]:
        return {
            "owner": self.owner,
            "status": self.status,
            "linked_issue_or_artifact": self.linked_issue_or_artifact,
            "last_decision": self.last_decision,
            "next_action": self.next_action,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceHeaderBinding:
    scope_id: str
    thread_id: str
    message_id: Optional[str]
    pending: bool = False


@dataclass(frozen=True, slots=True)
class WorkspaceHeaderResult:
    success: bool
    action: str
    message_id: Optional[str] = None
    error: Optional[str] = None


_WorkspaceRecord = dict[str, Any]
_WorkspaceRegistry = dict[str, dict[str, _WorkspaceRecord]]


class WorkspaceHeaderStore:
    """Locked atomic registry for header identity and tiny workspace state."""

    def __init__(self, path: Optional[Path] = None):
        if path is None:
            from hermes_constants import get_hermes_home

            path = get_hermes_home() / "gateway" / _STATE_FILENAME
        self.path = Path(path)

    @staticmethod
    def _default_record() -> _WorkspaceRecord:
        return {
            "message_id": None,
            "pending_token": None,
            "state": WorkspaceHeaderState().to_mapping(),
        }

    def _read_unlocked(self) -> _WorkspaceRegistry:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as error:
            raise WorkspaceHeaderStoreError(
                f"workspace header registry is unreadable: {type(error).__name__}"
            ) from error
        if not isinstance(payload, dict):
            raise WorkspaceHeaderStoreError("workspace header registry must be an object")
        if payload.get("version") != _STATE_VERSION:
            raise WorkspaceHeaderStoreError("workspace header registry version is unsupported")
        raw_workspaces = payload.get("workspaces")
        if not isinstance(raw_workspaces, dict):
            raise WorkspaceHeaderStoreError("workspace header registry has invalid workspaces")

        workspaces: _WorkspaceRegistry = {}
        for raw_scope_id, raw_threads in raw_workspaces.items():
            scope_id = _clean_id(raw_scope_id)
            if not scope_id or not isinstance(raw_threads, dict):
                raise WorkspaceHeaderStoreError("workspace header registry has invalid scope")
            threads: dict[str, _WorkspaceRecord] = {}
            for raw_thread_id, raw_record in raw_threads.items():
                thread_id = _clean_id(raw_thread_id)
                if not thread_id:
                    raise WorkspaceHeaderStoreError("workspace header registry has invalid thread id")
                # Accept the initial message-id-only shape written by early
                # OE-178 builds, but always rewrite it in the richer shape.
                if isinstance(raw_record, str):
                    message_id = _clean_id(raw_record)
                    if not message_id:
                        raise WorkspaceHeaderStoreError(
                            "workspace header registry has an empty message id"
                        )
                    record = self._default_record()
                    record["message_id"] = message_id
                elif isinstance(raw_record, dict):
                    unknown = set(raw_record) - {
                        "message_id",
                        "pending_token",
                        "state",
                    }
                    if unknown:
                        raise WorkspaceHeaderStoreError(
                            "workspace header registry record has unknown fields"
                        )
                    message_id = _clean_id(raw_record.get("message_id")) or None
                    pending_token = _clean_id(raw_record.get("pending_token")) or None
                    if message_id and pending_token:
                        raise WorkspaceHeaderStoreError(
                            "workspace header registry record has conflicting identity"
                        )
                    state = WorkspaceHeaderState.from_mapping(
                        raw_record.get("state")
                    )
                    record = {
                        "message_id": message_id,
                        "pending_token": pending_token,
                        "state": state.to_mapping(),
                    }
                else:
                    raise WorkspaceHeaderStoreError(
                        "workspace header registry record must be an object"
                    )
                threads[thread_id] = record
            if threads:
                workspaces[scope_id] = threads
        return workspaces

    def _write_unlocked(self, workspaces: _WorkspaceRegistry) -> None:
        try:
            atomic_json_write(
                self.path,
                {"version": _STATE_VERSION, "workspaces": workspaces},
                indent=2,
            )
        except Exception as error:
            raise WorkspaceHeaderStoreError(
                f"workspace header registry write failed: {type(error).__name__}"
            ) from error

    @staticmethod
    def _keys(scope_id: Any, thread_id: Any) -> tuple[str, str]:
        scope_key = _clean_id(scope_id)
        thread_key = _clean_id(thread_id)
        if not scope_key or not thread_key:
            raise ValueError("scope_id and thread_id are required")
        return scope_key, thread_key

    @staticmethod
    def _record(
        workspaces: _WorkspaceRegistry,
        scope_key: str,
        thread_key: str,
    ) -> Optional[_WorkspaceRecord]:
        return workspaces.get(scope_key, {}).get(thread_key)

    def get(self, scope_id: Any, thread_id: Any) -> Optional[WorkspaceHeaderBinding]:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        with _exclusive_registry_lock(self.path):
            record = self._record(self._read_unlocked(), scope_key, thread_key)
        if record is None:
            return None
        message_id = record.get("message_id")
        pending = bool(record.get("pending_token"))
        if not message_id and not pending:
            return None
        return WorkspaceHeaderBinding(scope_key, thread_key, message_id, pending)

    def get_state(
        self, scope_id: Any, thread_id: Any
    ) -> Optional[WorkspaceHeaderState]:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        with _exclusive_registry_lock(self.path):
            record = self._record(self._read_unlocked(), scope_key, thread_key)
        if record is None:
            return None
        return WorkspaceHeaderState.from_mapping(record.get("state"))

    def put(
        self, scope_id: Any, thread_id: Any, message_id: Any
    ) -> WorkspaceHeaderBinding:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        message_key = _clean_id(message_id)
        if not message_key:
            raise ValueError("message_id is required")
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            record = self._record(workspaces, scope_key, thread_key)
            if record is None:
                record = self._default_record()
                workspaces.setdefault(scope_key, {})[thread_key] = record
            record["message_id"] = message_key
            record["pending_token"] = None
            self._write_unlocked(workspaces)
        return WorkspaceHeaderBinding(scope_key, thread_key, message_key)

    def update_state(
        self,
        scope_id: Any,
        thread_id: Any,
        *,
        owner: Any = None,
        status: Any = None,
        linked_issue_or_artifact: Any = None,
        last_decision: Any = None,
        next_action: Any = None,
    ) -> WorkspaceHeaderState:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            record = self._record(workspaces, scope_key, thread_key)
            if record is None:
                record = self._default_record()
                workspaces.setdefault(scope_key, {})[thread_key] = record
            current = WorkspaceHeaderState.from_mapping(record.get("state"))
            changes = {
                "owner": current.owner if owner is None else owner,
                "status": current.status if status is None else status,
                "linked_issue_or_artifact": (
                    current.linked_issue_or_artifact
                    if linked_issue_or_artifact is None
                    else linked_issue_or_artifact
                ),
                "last_decision": (
                    current.last_decision if last_decision is None else last_decision
                ),
                "next_action": current.next_action if next_action is None else next_action,
            }
            state = WorkspaceHeaderState.from_mapping(changes)
            record["state"] = state.to_mapping()
            self._write_unlocked(workspaces)
        return state

    def reserve_creation(
        self,
        scope_id: Any,
        thread_id: Any,
        *,
        token: Any,
        expected_message_id: Any = None,
    ) -> bool:
        """Reserve one create/recreate before Discord receives a send."""
        scope_key, thread_key = self._keys(scope_id, thread_id)
        token_key = _clean_id(token)
        expected_key = _clean_id(expected_message_id) or None
        if not token_key:
            raise ValueError("reservation token is required")
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            record = self._record(workspaces, scope_key, thread_key)
            if record is None:
                if expected_key is not None:
                    return False
                record = self._default_record()
                workspaces.setdefault(scope_key, {})[thread_key] = record
            current_message_id = record.get("message_id")
            if record.get("pending_token") or current_message_id != expected_key:
                return False
            record["message_id"] = None
            record["pending_token"] = token_key
            self._write_unlocked(workspaces)
        return True

    def complete_creation(
        self,
        scope_id: Any,
        thread_id: Any,
        *,
        token: Any,
        message_id: Any,
    ) -> WorkspaceHeaderBinding:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        token_key = _clean_id(token)
        message_key = _clean_id(message_id)
        if not token_key or not message_key:
            raise ValueError("reservation token and message_id are required")
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            record = self._record(workspaces, scope_key, thread_key)
            if record is None or record.get("pending_token") != token_key:
                raise WorkspaceHeaderStoreError(
                    "workspace header creation reservation was lost"
                )
            record["message_id"] = message_key
            record["pending_token"] = None
            self._write_unlocked(workspaces)
        return WorkspaceHeaderBinding(scope_key, thread_key, message_key)

    def cancel_creation(
        self, scope_id: Any, thread_id: Any, *, token: Any
    ) -> bool:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        token_key = _clean_id(token)
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            record = self._record(workspaces, scope_key, thread_key)
            if record is None or record.get("pending_token") != token_key:
                return False
            record["pending_token"] = None
            self._write_unlocked(workspaces)
        return True

    def remove(self, scope_id: Any, thread_id: Any) -> None:
        scope_key, thread_key = self._keys(scope_id, thread_id)
        with _exclusive_registry_lock(self.path):
            workspaces = self._read_unlocked()
            threads = workspaces.get(scope_key)
            if not threads or thread_key not in threads:
                return
            del threads[thread_key]
            if not threads:
                workspaces.pop(scope_key, None)
            self._write_unlocked(workspaces)


@dataclass(frozen=True, slots=True)
class WorkspaceHeaderCandidate:
    scope_id: str
    thread_id: str
    observed_title: str
    reasons: tuple[str, ...]
    proposed_title: None = None


def collect_workspace_header_candidates(
    threads: Iterable[Any],
    *,
    participated_thread_ids: Iterable[Any],
    store: WorkspaceHeaderStore,
) -> list[WorkspaceHeaderCandidate]:
    """Plan only known Hermes workspaces; never propose a title mutation."""
    participated = {_clean_id(value) for value in participated_thread_ids}
    candidates: list[WorkspaceHeaderCandidate] = []
    for thread in threads:
        thread_id = _clean_id(getattr(thread, "id", None))
        scope_id = _thread_scope_id(thread)
        if not thread_id or not scope_id or thread_id not in participated:
            continue
        title = " ".join(str(getattr(thread, "name", "") or "").split())
        reasons: list[str] = []
        if is_known_hermes_placeholder_title(title):
            reasons.append("placeholder_title")
        if store.get(scope_id, thread_id) is None:
            reasons.append("header_missing")
        if reasons:
            candidates.append(
                WorkspaceHeaderCandidate(
                    scope_id=scope_id,
                    thread_id=thread_id,
                    observed_title=title,
                    reasons=tuple(reasons),
                )
            )
    candidates.sort(key=lambda item: (item.scope_id, item.thread_id))
    return candidates


def revalidate_workspace_header_candidate(
    candidate: WorkspaceHeaderCandidate,
    *,
    live_thread: Any,
    participated_thread_ids: Iterable[Any],
    store: WorkspaceHeaderStore,
) -> bool:
    """Fail closed if any live fact changed since the dry-run observation."""
    thread_id = _clean_id(getattr(live_thread, "id", None))
    scope_id = _thread_scope_id(live_thread)
    live_title = " ".join(str(getattr(live_thread, "name", "") or "").split())
    participated = {_clean_id(value) for value in participated_thread_ids}
    if (
        thread_id != candidate.thread_id
        or scope_id != candidate.scope_id
        or thread_id not in participated
        or live_title != candidate.observed_title
    ):
        return False
    if "placeholder_title" in candidate.reasons and not is_known_hermes_placeholder_title(
        live_title
    ):
        return False
    if (
        "header_missing" in candidate.reasons
        and store.get(scope_id, thread_id) is not None
    ):
        return False
    return True
