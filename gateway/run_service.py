"""Thread-safe retained status and control state for API runs.

This module is deliberately transport-agnostic.  HTTP parsing and response
translation remain in the API server adapter.
"""

from collections.abc import MutableMapping, MutableSet
from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional


PUBLIC_RUN_STATUS_FIELDS = frozenset(
    {
        "object",
        "run_id",
        "status",
        "session_id",
        "model",
        "created_at",
        "updated_at",
        "last_event",
        "output",
        "usage",
        "error",
    }
)
TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
# "stopping" (set by _handle_stop_run) is not terminal: the run stays in this
# state, doing real executor-thread work, until the agent actually notices the
# interrupt and the task settles to "cancelled" — an unbounded window, not the
# old ~5s hard-timeout wait. Excluding it here undercounts active runs for the
# whole duration of a cooperative stop.
ACTIVE_RUN_STATUSES = frozenset({"queued", "running", "waiting_for_approval", "stopping"})
RUN_STATUSES = ACTIVE_RUN_STATUSES | TERMINAL_RUN_STATUSES


def _finite_timestamp(value: Any) -> float:
    try:
        timestamp = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return timestamp if math.isfinite(timestamp) else 0.0


def _validate_run_status(status: Any) -> None:
    if type(status) is not str or status not in RUN_STATUSES:
        raise ValueError(f"unsupported run status: {status!r}")


def _snapshot_status_value(value: Any) -> Any:
    """Copy JSON-shaped status data without invoking user-defined hooks."""
    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("run status float values must be finite")
        return value
    if value is None or value_type in (bool, int, str):
        return value
    if value_type is list:
        return [_snapshot_status_value(item) for item in value]
    if value_type is tuple:
        return tuple(_snapshot_status_value(item) for item in value)
    if value_type is dict:
        snapshot = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("run status mappings must use string keys")
            snapshot[key] = _snapshot_status_value(item)
        return snapshot
    raise TypeError(
        "run status values must be JSON-compatible builtin data, "
        f"not {value_type.__name__}"
    )


def _snapshot_status_record(value: Any) -> Dict[str, Any]:
    if type(value) is not dict:
        raise TypeError("run status records must be builtin dictionaries")
    snapshot = {}
    for key, item in value.items():
        if type(key) is not str:
            raise TypeError("run status mappings must use string keys")
        if key in {"created_at", "updated_at"}:
            snapshot[key] = _finite_timestamp(item)
        else:
            snapshot[key] = _snapshot_status_value(item)
    _validate_run_status(snapshot.get("status"))
    return snapshot


@dataclass(frozen=True)
class StopTarget:
    """Control references claimed for one stop decision."""

    agent: Any
    task: Any
    owns_lease: bool = True


class _LockedMapping(MutableMapping):
    """Minimal lock-aware compatibility view over an authoritative dict."""

    def __init__(
        self,
        data: Dict[Any, Any],
        lock: threading.RLock,
        *,
        snapshotter: Optional[Callable[[Any], Any]] = None,
        mutation_allowed: Optional[Callable[[Any], bool]] = None,
    ):
        self._data = data
        self._lock = lock
        self._snapshotter = snapshotter
        self._mutation_allowed = mutation_allowed

    def _snapshot(self, value: Any) -> Any:
        if self._snapshotter is not None:
            return self._snapshotter(value)
        return value

    def __getitem__(self, key: Any) -> Any:
        with self._lock:
            return self._snapshot(self._data[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        snapshot = self._snapshot(value)
        with self._lock:
            self._ensure_mutation_allowed(key)
            self._data[key] = snapshot

    def __delitem__(self, key: Any) -> None:
        with self._lock:
            self._ensure_mutation_allowed(key)
            del self._data[key]

    def _ensure_mutation_allowed(self, key: Any) -> None:
        if self._mutation_allowed is not None and not self._mutation_allowed(key):
            raise RuntimeError("run control mutation rejected while stop is pending")

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            return iter(tuple(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class _LockedSet(MutableSet):
    """Minimal lock-aware compatibility view over an authoritative set."""

    def __init__(
        self,
        data: set[Any],
        lock: threading.RLock,
        *,
        mutation_allowed: Optional[Callable[[Any], bool]] = None,
    ):
        self._data = data
        self._lock = lock
        self._mutation_allowed = mutation_allowed

    def __contains__(self, value: object) -> bool:
        with self._lock:
            return value in self._data

    def __iter__(self) -> Iterator[Any]:
        with self._lock:
            return iter(tuple(self._data))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def add(self, value: Any) -> None:
        with self._lock:
            self._ensure_mutation_allowed(value)
            self._data.add(value)

    def discard(self, value: Any) -> None:
        with self._lock:
            self._ensure_mutation_allowed(value)
            self._data.discard(value)

    def _ensure_mutation_allowed(self, value: Any) -> None:
        if self._mutation_allowed is not None and not self._mutation_allowed(value):
            raise RuntimeError("run control mutation rejected while stop is pending")


class RunRegistry:
    """Own retained run statuses and active stop-control references."""

    def __init__(self, *, clock: Callable[[], Any] = time.time):
        self._clock = clock
        self._lock = threading.RLock()
        self._statuses: Dict[str, Dict[str, Any]] = {}
        self._agents: Dict[str, Any] = {}
        self._tasks: Dict[str, Any] = {}
        self._approval_sessions: Dict[str, str] = {}
        self._stopping_ids: set[str] = set()
        self._claimed_stop_ids: set[str] = set()
        self._deferred_control_removals: set[str] = set()
        self._status_view = _LockedMapping(
            self._statuses,
            self._lock,
            snapshotter=_snapshot_status_record,
            mutation_allowed=self._compat_mutation_allowed,
        )
        self._agent_view = _LockedMapping(
            self._agents,
            self._lock,
            mutation_allowed=self._compat_mutation_allowed,
        )
        self._task_view = _LockedMapping(
            self._tasks,
            self._lock,
            mutation_allowed=self._compat_mutation_allowed,
        )
        self._approval_session_view = _LockedMapping(
            self._approval_sessions,
            self._lock,
            mutation_allowed=self._compat_mutation_allowed,
        )
        self._stopping_view = _LockedSet(
            self._stopping_ids,
            self._lock,
            mutation_allowed=self._compat_mutation_allowed,
        )

    def _compat_mutation_allowed(self, run_id: Any) -> bool:
        return (
            run_id not in self._claimed_stop_ids
            and run_id not in self._stopping_ids
        )

    @property
    def statuses(self) -> MutableMapping:
        """Authoritative status map, exposed only for legacy compatibility."""
        return self._status_view

    @property
    def agents(self) -> MutableMapping:
        """Authoritative agent map, exposed only for legacy compatibility."""
        return self._agent_view

    @property
    def tasks(self) -> MutableMapping:
        """Authoritative task map, exposed only for legacy compatibility."""
        return self._task_view

    @property
    def approval_sessions(self) -> MutableMapping:
        """Authoritative approval-session map for legacy adapter compatibility."""
        return self._approval_session_view

    @property
    def stopping_ids(self) -> MutableSet:
        """Authoritative stopping set, exposed only for legacy compatibility."""
        return self._stopping_view

    def _replace_mapping(
        self,
        target: Dict[Any, Any],
        values: Dict[Any, Any],
        *,
        snapshotter: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        if type(values) is not dict:
            raise TypeError("run compatibility maps must be builtin dictionaries")
        replacement = {
            key: snapshotter(value) if snapshotter is not None else value
            for key, value in values.items()
        }
        with self._lock:
            for run_id in set(target) | set(replacement):
                if not self._compat_mutation_allowed(run_id):
                    raise RuntimeError(
                        "run control mutation rejected while stop is pending"
                    )
            target.clear()
            target.update(replacement)

    def replace_statuses(self, values: Dict[str, Dict[str, Any]]) -> None:
        self._replace_mapping(
            self._statuses,
            values,
            snapshotter=_snapshot_status_record,
        )

    def replace_agents(self, values: Dict[str, Any]) -> None:
        self._replace_mapping(self._agents, values)

    def replace_tasks(self, values: Dict[str, Any]) -> None:
        self._replace_mapping(self._tasks, values)

    def replace_approval_sessions(self, values: Dict[str, str]) -> None:
        self._replace_mapping(self._approval_sessions, values)

    def replace_stopping_ids(self, values: set[str]) -> None:
        if type(values) is not set:
            raise TypeError("run compatibility stopping IDs must be a builtin set")
        replacement = set(values)
        with self._lock:
            for run_id in self._stopping_ids | replacement:
                if not self._compat_mutation_allowed(run_id):
                    raise RuntimeError(
                        "run control mutation rejected while stop is pending"
                    )
            self._stopping_ids.clear()
            self._stopping_ids.update(replacement)

    @staticmethod
    def finite_timestamp(value: Any) -> float:
        return _finite_timestamp(value)

    @classmethod
    def public_status(cls, status: Dict[str, Any]) -> Dict[str, Any]:
        public = {}
        for key, value in status.items():
            if key not in PUBLIC_RUN_STATUS_FIELDS:
                continue
            if key in {"created_at", "updated_at"}:
                public[key] = cls.finite_timestamp(value)
            else:
                public[key] = _snapshot_status_value(value)
        return public

    def _set_status_locked(
        self,
        run_id: str,
        status: str,
        fields: Dict[str, Any],
        *,
        now: float,
    ) -> Dict[str, Any]:
        incoming = dict(fields)
        created_at = incoming.pop("created_at", now)
        current = self._statuses.get(run_id, {})
        if (
            (
                run_id in self._claimed_stop_ids
                or run_id in self._stopping_ids
            )
            and "session_id" in incoming
            and incoming["session_id"] != current.get("session_id")
        ):
            raise RuntimeError(
                "run ownership mutation rejected while stop is pending"
            )
        candidate = dict(current)
        candidate.update(
            {
                "object": "hermes.run",
                "run_id": run_id,
                "status": status,
                "updated_at": now,
            }
        )
        candidate.setdefault(
            "created_at",
            created_at,
        )
        candidate.update(incoming)
        public = self.public_status(candidate)
        self._statuses[run_id] = candidate
        return public

    def set_status(self, run_id: str, status: str, **fields: Any) -> Dict[str, Any]:
        _validate_run_status(status)
        now = self.finite_timestamp(self._clock())
        for field in ("created_at", "updated_at"):
            if field in fields:
                fields[field] = self.finite_timestamp(fields[field])
        incoming = _snapshot_status_value(fields)
        with self._lock:
            return self._set_status_locked(
                run_id,
                status,
                incoming,
                now=now,
            )

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            status = self._statuses.get(run_id)
            if status is None:
                return None
            return self.public_status(status)

    def contains(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._statuses

    def status_value(self, run_id: str, field: str, default: Any = None) -> Any:
        with self._lock:
            return _snapshot_status_value(
                self._statuses.get(run_id, {}).get(field, default)
            )


    def active_status_count(self) -> int:
        with self._lock:
            return sum(
                status.get("status") in ACTIVE_RUN_STATUSES
                for status in self._statuses.values()
            )

    def expire_terminal_statuses(self, now: float, ttl: float) -> None:
        with self._lock:
            stale = [
                run_id
                for run_id, status in self._statuses.items()
                if status.get("status") in TERMINAL_RUN_STATUSES
                and now - self.finite_timestamp(status.get("updated_at", 0)) > ttl
            ]
            for run_id in stale:
                self._statuses.pop(run_id, None)

    def register_agent(self, run_id: str, agent: Any) -> bool:
        with self._lock:
            if run_id in self._stopping_ids or run_id in self._claimed_stop_ids:
                return False
            self._agents[run_id] = agent
            return True

    def register_task(self, run_id: str, task: Any) -> bool:
        with self._lock:
            if run_id in self._stopping_ids or run_id in self._claimed_stop_ids:
                return False
            self._tasks[run_id] = task
            return True

    def agent_for(self, run_id: str) -> Any:
        with self._lock:
            return self._agents.get(run_id)

    def task_for(self, run_id: str) -> Any:
        with self._lock:
            return self._tasks.get(run_id)

    def register_approval_session(
        self, run_id: str, approval_session_key: str
    ) -> bool:
        """Bind one run to its isolated approval namespace."""
        with self._lock:
            if run_id in self._stopping_ids or run_id in self._claimed_stop_ids:
                return False
            self._approval_sessions[run_id] = approval_session_key
            return True

    def approval_session_for(self, run_id: str) -> Optional[str]:
        with self._lock:
            return self._approval_sessions.get(run_id)

    def active_task_count(self) -> int:
        with self._lock:
            tasks = tuple(self._tasks.values())
        return sum(not task.done() for task in tasks)

    def is_stopping(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._stopping_ids

    def remove_control(self, run_id: str) -> None:
        with self._lock:
            if run_id in self._claimed_stop_ids:
                self._deferred_control_removals.add(run_id)
                return
            self._remove_control_locked(run_id)

    def claim_stop_target(self, run_id: str) -> Optional[StopTarget]:
        """Atomically lease a stop target without running external code locked."""
        now = self.finite_timestamp(self._clock())
        with self._lock:
            agent = self._agents.get(run_id)
            task = self._tasks.get(run_id)
            if agent is None and task is None:
                return None

            if run_id in self._claimed_stop_ids or run_id in self._stopping_ids:
                return StopTarget(agent=None, task=task, owns_lease=False)

            self._set_status_locked(
                run_id,
                "stopping",
                {"last_event": "run.stopping"},
                now=now,
            )
            self._stopping_ids.add(run_id)
            self._claimed_stop_ids.add(run_id)
            return StopTarget(agent=agent, task=task)

    def release_stop_target(self, run_id: str) -> None:
        """Release a stop lease and apply cleanup deferred during interruption."""
        with self._lock:
            self._claimed_stop_ids.discard(run_id)
            if run_id in self._deferred_control_removals:
                self._deferred_control_removals.discard(run_id)
                self._remove_control_locked(run_id)

    def _remove_control_locked(self, run_id: str) -> None:
        self._agents.pop(run_id, None)
        self._tasks.pop(run_id, None)
        self._approval_sessions.pop(run_id, None)
        self._stopping_ids.discard(run_id)
        self._deferred_control_removals.discard(run_id)
