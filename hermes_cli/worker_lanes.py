"""Kanban worker-lane registry.

A *worker lane* tells the kanban dispatcher how to spawn a worker for a given
task ``assignee``. The default lane shape is a Hermes profile — the dispatcher
runs ``hermes -p <assignee> chat`` (see :func:`kanban_db._default_spawn`). A
*plugin lane* lets an out-of-tree integration register its own ``spawn_fn`` for
a non-Hermes assignee (e.g. a Junie / Codex CLI runner) so the dispatcher
spawns that runtime directly instead of a Hermes process.

Plugins register lanes at load time via the plugin context
(``ctx.register_worker_lane(...)``); the dispatcher consults the registry when
it resolves each ready task's assignee. The registry is **process-local**, so
registration must happen in the same process that runs the dispatcher
(typically the gateway that owns kanban dispatch).

See ``website/docs/user-guide/features/kanban-worker-lanes.md`` for the lane
contract a spawned worker must satisfy (exactly one terminal kanban action,
heartbeat, review-required convention).
"""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# spawn_fn(task, workspace, *, board=None) -> Optional[int]   (worker pid)
SpawnFn = Callable[..., Optional[int]]

_LANE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}$")

_REGISTRY_LOCK = threading.RLock()
_WORKER_LANES: "dict[str, WorkerLane]" = {}


def normalize_lane_name(name: str) -> str:
    """Lower-case and validate a lane name (the assignee it matches)."""
    if not isinstance(name, str):
        name = str(name)
    lane = name.strip().lower()
    if not lane:
        raise ValueError("worker lane name cannot be empty")
    if not _LANE_ID_RE.match(lane):
        raise ValueError(
            f"invalid worker lane name {name!r}: must match "
            "[a-z0-9][a-z0-9_-]{0,63}"
        )
    return lane


@dataclass
class WorkerLane:
    """How to spawn a worker for an assignee.

    Behaviour comes from two fields:

    * ``name`` — the assignee the dispatcher matches against (the routing key);
    * ``spawn_fn`` — called as ``spawn_fn(task, workspace, board=...)`` to launch
      the worker (typically detached) and return its pid for crash detection.

    ``kind`` is an optional label for logs/telemetry; ``max_concurrency`` is an
    optional per-lane in-flight cap (``None`` falls back to the global
    per-profile cap).
    """

    name: str
    spawn_fn: SpawnFn
    kind: str = ""
    max_concurrency: Optional[int] = None

    def __post_init__(self) -> None:
        self.name = normalize_lane_name(self.name)
        if not callable(self.spawn_fn):
            raise ValueError("worker lane spawn_fn must be callable")
        self.kind = str(self.kind or "").strip()
        if self.max_concurrency is not None:
            mc = int(self.max_concurrency)
            if mc < 1:
                raise ValueError("worker lane max_concurrency must be >= 1")
            self.max_concurrency = mc


def register_worker_lane(lane: WorkerLane, *, replace: bool = False) -> WorkerLane:
    """Register ``lane`` by name. Duplicate names are rejected unless ``replace``."""
    if not isinstance(lane, WorkerLane):
        raise TypeError("register_worker_lane expects a WorkerLane instance")
    name = normalize_lane_name(lane.name)
    with _REGISTRY_LOCK:
        existing = _WORKER_LANES.get(name)
        if existing is not None and not replace:
            raise ValueError(
                f"worker lane {name!r} already registered (kind={existing.kind or '?'})"
            )
        _WORKER_LANES[name] = lane
    logger.debug("registered worker lane %s (kind=%s)", name, lane.kind)
    return lane


def get_worker_lane(name: Optional[str]) -> Optional[WorkerLane]:
    """Return the registered lane for ``name``, or None."""
    if not name:
        return None
    try:
        key = normalize_lane_name(name)
    except ValueError:
        return None
    with _REGISTRY_LOCK:
        return _WORKER_LANES.get(key)


def is_worker_lane_assignee(name: Optional[str]) -> bool:
    """True if ``name`` resolves to a registered worker lane."""
    return get_worker_lane(name) is not None


def list_worker_lanes() -> "list[WorkerLane]":
    with _REGISTRY_LOCK:
        return [_WORKER_LANES[n] for n in sorted(_WORKER_LANES)]


def clear_worker_lanes() -> None:
    """Remove all registered lanes. Mainly for tests."""
    with _REGISTRY_LOCK:
        _WORKER_LANES.clear()


def kanban_worker_env(task, workspace, *, board=None, base=None) -> "dict[str, Any]":
    """Build the standard kanban worker env contract for a spawned worker.

    Returns a dict (copied from ``base``, default ``os.environ``) with the
    ``HERMES_KANBAN_*`` pins a worker process needs to converge on the exact
    board / DB / workspace / run / claim the dispatcher resolved — the same
    contract :func:`kanban_db._default_spawn` sets for Hermes profile workers,
    factored out so external lane ``spawn_fn`` implementations don't each
    re-derive it.

    Lane-specific vars (``HERMES_PROFILE``, ``HERMES_WORKER_LANE``, runtime
    auth, ``PYTHONPATH``, …) are intentionally left to the caller.
    """
    from hermes_cli import kanban_db as kb  # local import: avoids import cycle

    env = dict(os.environ if base is None else base)
    env["HERMES_KANBAN_TASK"] = task.id
    if workspace:
        env["HERMES_KANBAN_WORKSPACE"] = workspace
    env["HERMES_KANBAN_HOME"] = str(kb.kanban_home())
    env["HERMES_KANBAN_DB"] = str(kb.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(kb.workspaces_root(board=board))
    env["HERMES_KANBAN_BOARD"] = kb._normalize_board_slug(board) or kb.get_current_board()
    if getattr(task, "current_run_id", None) is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if getattr(task, "claim_lock", None):
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    if getattr(task, "tenant", None):
        env["HERMES_TENANT"] = task.tenant
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    return env
