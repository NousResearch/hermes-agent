"""SQLite-backed Kanban board shared across profiles (the cross-profile coordination primitive).

Lives under the shared Hermes root: ``default`` board DB at ``<root>/kanban.db`` (pre-boards
back-compat), other boards at ``<root>/kanban/boards/<slug>/``; a worker on one board never sees
another. Board resolution: ``board=`` arg > ``HERMES_KANBAN_BOARD`` > ``HERMES_KANBAN_DB`` (pins the
file path) > ``<root>/kanban/current`` > ``default``; the dispatcher injects these into workers.
Concurrency: WAL + ``BEGIN IMMEDIATE`` + compare-and-swap on ``tasks.status``/``claim_lock`` —
SQLite serializes writers so one claimer wins, losers see zero rows (no retries, no distributed
locks). Schema: tasks, task_links, task_comments, task_events, task_runs, attachments, notify subs.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from toolsets import get_toolset_names

_log = logging.getLogger(__name__)


# --- Shared micro-helpers (row access, JSON, env, git) ---

def _row_get(row: Any, col: str, default: Any = None) -> Any:
    """``row[col]`` tolerant of the column being absent from the SELECT / schema."""
    if row is None or col not in row.keys():
        return default
    return row[col]


def _json_or(value: Any, default: Any = None) -> Any:
    """Decode a JSON text column; any decode failure or empty value yields ``default``."""
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _json_dict(value: Any) -> dict:
    """Decode a JSON text column that must be an object; anything else yields ``{}``."""
    parsed = _json_or(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    """Integer env override: absent/empty/non-integer/below ``minimum`` falls back to ``default``."""
    raw = os.environ.get(name, "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            return default
        if parsed >= minimum:
            return parsed
    return default


def _git_out(cwd: Path, *args: str, timeout: int = 30) -> Optional[str]:
    """Run ``git -C cwd args`` and return stripped stdout, or ``None`` on any failure / empty output."""
    try:
        result = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or "").strip() or None


# --- Constants ---

VALID_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"}
VALID_INITIAL_STATUSES = {"running", "blocked"}

# Typed block reasons (routing in ``_route_block``); ``None`` = legacy un-typed.
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# Same-reason block -> unblock -> re-block cycles before routing to ``triage``.
# Counts unblock recurrences, NOT dispatcher failures (``DEFAULT_FAILURE_LIMIT``).
BLOCK_RECURRENCE_LIMIT = 2
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}


def normalize_reasoning_effort(effort: Optional[str]) -> Optional[str]:
    """``VALID_REASONING_EFFORTS`` or ``"none"`` (thinking off), case-insensitive;
    empty/None = inherit the profile's own effort (NULL). Anything else raises —
    a typo'd level must not quietly hand the task back to the profile default."""
    from hermes_constants import VALID_REASONING_EFFORTS

    value = str(effort or "").strip().lower()
    if not value:
        return None
    if value == "none" or value in VALID_REASONING_EFFORTS:
        return value
    allowed = ", ".join(("none", *VALID_REASONING_EFFORTS))
    raise ValueError(f"reasoning_effort must be one of {allowed}, got {effort!r}")


KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())
_IS_WINDOWS = sys.platform == "win32"
KANBAN_ATTACHMENT_MAX_BYTES = 25 * 1024 * 1024  # one cap for dashboard, tools and CLI


def _assert_not_delegated_child_mutation() -> None:
    """Reject Kanban mutations from ``delegate_task`` child contexts.

    The tool/CLI fast-fail guards are UX, not a trust boundary (a child can shell
    out or import this module); the invariant lives here so every ``write_txn``
    user and board-metadata mutator fails closed before touching durable state.
    """
    try:
        from agent.delegation_context import is_delegated_child_process_context

        delegated = is_delegated_child_process_context()
    except Exception:
        delegated = bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))
    if delegated:
        raise PermissionError("delegate_task child contexts cannot mutate Kanban tasks or boards")


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **fields: Any) -> None:
    """Best-effort lifecycle hook. Call AFTER the write txn commits (plugins never
    run under the SQLite write lock, always see durable state); failures are
    swallowed so an observer can never break a transition."""
    try:
        from hermes_cli.lifecycle import invoke_hook

        invoke_hook(event, task_id=task_id, profile_name=_hook_profile_name(), **fields)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban lifecycle hook %s failed: %s", event, exc)


def _fire_task_hook(event: str, task: Optional["Task"], task_id: str, run_id: Optional[int], **fields: Any) -> None:
    """Lifecycle hook for a task transition; ``assignee`` from the (possibly missing) row."""
    _fire_kanban_lifecycle_hook(
        event, task_id, board=get_current_board(),
        assignee=task.assignee if task else None, run_id=run_id, **fields,
    )


def _hook_profile_name() -> str:
    """Active profile for hook payloads; ``"default"`` when it cannot be resolved."""
    from hermes_cli.profiles import get_active_profile_name

    try:
        return get_active_profile_name()
    except Exception:
        return "default"


def _kanban_observer_consumed(event: str) -> bool:
    """Hot-path short-circuit: skip payload assembly when nothing subscribes.
    Inspection failure counts as unconsumed (dropping an observer is always safe)."""
    try:
        from hermes_cli.lifecycle import has_hook

        return has_hook(event)
    except Exception:  # pragma: no cover - defensive
        return False


def _fire_worker_spawned_hook(
    conn: sqlite3.Connection, task: "Task", workspace_path: str, pid: Optional[int], *,
    board: Optional[str] = None,
) -> None:
    """``on_kanban_worker_spawned`` AFTER the PID is durably persisted; best-effort."""
    if not _kanban_observer_consumed("on_kanban_worker_spawned"):
        return
    try:
        _fire_kanban_lifecycle_hook(
            "on_kanban_worker_spawned", task.id, board=board or get_current_board(),
            assignee=task.assignee, run_id=_current_run_id(conn, task.id),
            worker_pid=int(pid) if pid else None, workspace_path=str(workspace_path),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban worker spawned hook failed: %s", exc)


def notify_task_updated(
    conn: sqlite3.Connection, task_id: str, changed_fields: Iterable[str], *,
    board: Optional[str] = None,
) -> None:
    """``on_kanban_task_updated`` AFTER a non-lifecycle task mutation commits
    (also for direct-SQL surfaces like dashboard field editors).
    ``changed_fields`` carries field NAMES only, never values."""
    if not _kanban_observer_consumed("on_kanban_task_updated"):
        return
    try:
        row = conn.execute(
            "SELECT assignee, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        _fire_kanban_lifecycle_hook(
            "on_kanban_task_updated", task_id, board=board or get_current_board(),
            assignee=row["assignee"] if row else None,
            run_id=row["current_run_id"] if row else None, changed_fields=list(changed_fields),
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban task updated hook failed: %s", exc)


# DispatchResult counters whose non-zero value means the tick did something.
_TICK_ACTIVITY_FIELDS = (
    "spawned", "reclaimed", "promoted", "reconciled_orphans", "crashed", "stale",
    "timed_out", "auto_blocked", "rate_limited", "auto_assigned_default",
    "respawn_guarded", "skipped_per_profile_capped", "skipped_unassigned",
    "skipped_nonspawnable",
)


def _fire_dispatch_tick_hook(
    result: "DispatchResult", *, board: Optional[str] = None, dry_run: bool = False,
) -> None:
    """``on_kanban_dispatch_tick`` — strictly AFTER ``_dispatch_tick_lock`` is
    released so a slow subscriber cannot stall a sibling dispatcher.

    Re-port of PR #56066 per the #64231 batch disposition: renamed to the taxonomy form and called by
    ``dispatch_once`` strictly AFTER ``_dispatch_tick_lock`` has been released — the original fired inside
    the lock, so a slow subscriber could extend the single-writer critical section and stall a sibling
    dispatcher's tick. Observer-only and fully best-effort: any subscriber failure is swallowed.
    """
    if not _kanban_observer_consumed("on_kanban_dispatch_tick"):
        return
    try:
        from hermes_cli.lifecycle import invoke_hook

        profile_name = _hook_profile_name()
        if board is None:
            try:
                board = get_current_board()
            except Exception:
                board = None
        outcome = "ok"
        if result.skipped_locked:
            outcome = "skipped_locked"
        elif not any(getattr(result, f) for f in _TICK_ACTIVITY_FIELDS):
            outcome = "idle"
        invoke_hook(
            "on_kanban_dispatch_tick", board=board, profile_name=profile_name,
            dry_run=bool(dry_run), outcome=outcome, result=result,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban dispatch tick hook failed: %s", exc)


# Claim window before the next tick reclaims a running task; long workers
# ``heartbeat_claim`` or raise it via HERMES_KANBAN_CLAIM_TTL_SECONDS.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60

# A live PID with a heartbeat older than this is wedged and reclaimed anyway
# (``_touch_activity`` keeps genuinely active workers fresh).
# If a worker's PID is still alive but its ``last_heartbeat_at`` is older than this when
# ``release_stale_claims`` runs, treat the worker as wedged and reclaim regardless of PID liveness (#29747
# gap 3). This catches the logic-loop case where the process is technically running but not making
# observable progress. ``_touch_activity`` bridges chunk-level liveness into ``last_heartbeat_at`` via
# #31752, so any genuinely active worker keeps its heartbeat fresh as a side effect of normal API traffic.
DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS = 60 * 60

# Grace when a host-local worker survived termination (e.g. parked in D state
# under memory.high, SIGKILL pending): releasing now would spawn a duplicate.
RECLAIM_DEFER_GRACE_SECONDS = 120


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Explicit ``ttl_seconds`` > ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` > default."""
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    return _env_int("HERMES_KANBAN_CLAIM_TTL_SECONDS", DEFAULT_CLAIM_TTL_SECONDS, minimum=1)


# ``detect_crashed_workers`` skips ``_pid_alive`` this long after start: the
# fork -> /proc window can report a fresh worker dead.
DEFAULT_CRASH_GRACE_SECONDS = 30

# Worker exit "provider rate-limited": released WITHOUT counting a failure (the
# breaker must never trip on a throttle). 75 == BSD EX_TEMPFAIL.
KANBAN_RATE_LIMIT_EXIT_CODE = 75

# _signal_handler_q (cli.py) intentionally calls os._exit() rather than
# letting SIGINT/SIGTERM/SIGHUP kill the process via the default disposition
# (issue #28181 — a controlled unwind can leave a worker thread parked in
# _wait_for_process, orphaning its subprocess). os._exit(N) always reports
# WIFEXITED, never WIFSIGNALED, so _classify_worker_exit cannot tell "worker
# caught a termination signal and exited fast on purpose" apart from
# "worker's own turn quietly finished" unless the two use different exit
# codes. Historically both exited 0, so a worker that was killed via signal
# recorded as the misleading `clean_exit` -> protocol_violation. Standard
# 128+SIGTERM, and well clear of 0/1/2/KANBAN_RATE_LIMIT_EXIT_CODE.
KANBAN_SIGNAL_EXIT_CODE = 143


def _resolve_crash_grace_seconds() -> int:
    """``HERMES_KANBAN_CRASH_GRACE_SECONDS`` (0 = immediate, for tests) else default."""
    return _env_int("HERMES_KANBAN_CRASH_GRACE_SECONDS", DEFAULT_CRASH_GRACE_SECONDS)


def _resolve_rate_limit_cooldown_seconds() -> int:
    """``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` (0 = next tick, for tests) else default."""
    return _env_int("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS)


# build_worker_context() caps, sized for a ~100k-char prompt with headroom.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # per comment


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """``just now`` / ``18h ago`` / ``3d ago``; "" for a missing/invalid ts. An LLM
    reads a bare absolute timestamp as current fact — the relative age is what
    prompts a worker to re-verify stale sibling work."""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 60:  # includes negative = clock skew across machines; never claim "in the future"
        return "just now"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


# --- Paths ---

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override", default=None,
)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)


# Slug = directory name: strict enough to stop traversal / separators, loose
# enough for kebab-case. Display names (spaces, emoji) live in board.json.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    s = str(slug).strip().lower() if slug is not None else ""
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def _slug_or_default(board: Optional[str]) -> str:
    return _normalize_board_slug(board) or DEFAULT_BOARD


def _require_slug(slug: str) -> str:
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    return normed


def kanban_home() -> Path:
    """``HERMES_KANBAN_HOME`` else ``get_default_hermes_root()``. Shared across
    profiles BY DESIGN: resolving through the active profile's HERMES_HOME would
    fork the board per profile and break the dispatcher/worker handoff."""
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """``<root>/kanban/boards`` — parent of the *additional* named boards.
    ``default`` is deliberately not here (its DB stays at ``<root>/kanban.db``)."""
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """``<root>/kanban/current`` — one-line slug written by ``boards switch``; absent = ``default``."""
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Active slug: context override -> ``HERMES_KANBAN_BOARD`` -> ``<root>/kanban/current``
    (only while that board exists) -> ``DEFAULT_BOARD``. A malformed/stale slug
    falls through — the dispatcher must never crash on a hand-edited file."""
    def _existing(candidate: str) -> Optional[str]:
        if not candidate:
            return None
        try:
            normed = _normalize_board_slug(candidate)
        except ValueError:
            return None
        return normed if normed and board_exists(normed) else None

    for candidate in (
        (_CURRENT_BOARD_OVERRIDE.get() or "").strip(),
        os.environ.get("HERMES_KANBAN_BOARD", "").strip(),
    ):
        found = _existing(candidate)
        if found:
            return found
    try:
        f = current_board_path()
        if f.exists():
            found = _existing(f.read_text(encoding="utf-8").strip())
            if found:
                return found
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board; returns the file written. Does NOT
    check the board exists — callers do (so ``boards switch <typo>`` errors)."""
    _assert_not_delegated_child_mutation()
    normed = _require_slug(slug)
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    _assert_not_delegated_child_mutation()
    with contextlib.suppress(FileNotFoundError):
        current_board_path().unlink()


def board_dir(board: Optional[str] = None) -> Path:
    """``<root>/kanban/boards/<slug>/``. For ``default`` this holds metadata
    only (board.json, workspaces/, logs/) — its DB stays at ``<root>/kanban.db``
    for back-compat (:func:`kanban_db_path`).
    """
    return boards_root() / _slug_or_default(board)


def board_exists(board: Optional[str] = None) -> bool:
    """Board has ``board.json`` or ``kanban.db`` on disk; ``default`` always exists."""
    slug = _slug_or_default(board)
    if slug == DEFAULT_BOARD:
        return True
    return _dir_holds_board(board_dir(slug))


def _dir_holds_board(d: Path) -> bool:
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def _board_path(
    env_var: Optional[str], board: Optional[str], default_parts: tuple[str, ...], leaf: str,
) -> Path:
    """Shared resolver: ``env_var`` override, else legacy ``<root>/<default_parts>``
    for the ``default`` board, else ``board_dir(slug)/leaf``."""
    if env_var:
        override = os.environ.get(env_var, "").strip()
        if override:
            return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home().joinpath(*default_parts)
    return board_dir(slug) / leaf


def kanban_db_path(board: Optional[str] = None) -> Path:
    """``kanban.db`` path: ``HERMES_KANBAN_DB`` pins it (injected into workers);
    ``default`` -> ``<root>/kanban.db`` (back-compat), else the board dir."""
    return _board_path("HERMES_KANBAN_DB", board, ("kanban.db",), "kanban.db")


def workspaces_root(board: Optional[str] = None) -> Path:
    """Per-board scratch workspace root (``HERMES_KANBAN_WORKSPACES_ROOT`` wins);
    ``default`` keeps the legacy ``<root>/kanban/workspaces/``."""
    return _board_path("HERMES_KANBAN_WORKSPACES_ROOT", board, ("kanban", "workspaces"), "workspaces")


def attachments_root(board: Optional[str] = None) -> Path:
    """Per-board attachments root (``HERMES_KANBAN_ATTACHMENTS_ROOT`` wins). Workers
    read attachments by absolute path, so remote terminal backends must mount it."""
    return _board_path("HERMES_KANBAN_ATTACHMENTS_ROOT", board, ("kanban", "attachments"), "attachments")


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Per-board worker log dir (logs follow the board so ``hermes kanban log``
    is unambiguous when two boards share a task id)."""
    return _board_path(None, board, ("kanban", "logs"), "logs")


def board_metadata_path(board: Optional[str] = None) -> Path:
    """``board.json`` path — display metadata only; the directory slug is the identity."""
    return board_dir(_slug_or_default(board)) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """``atm10-server`` -> ``Atm10 Server``."""
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


def read_board_metadata(board: Optional[str] = None) -> dict:
    """``board.json`` merged over defaults, plus ``slug`` and ``db_path``. Never
    raises — a missing/malformed file yields the synthesized entry."""
    slug = _slug_or_default(board)
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        # Project scope: new tasks inherit it (deterministic worktree + branch).
        "project_id": None,
        "created_at": None,
        "archived": False,
    }
    try:
        p = board_metadata_path(slug)
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                # Never let the metadata file claim a different slug than
                # its directory — trust the filesystem.
                raw["slug"] = slug
                meta.update(raw)
    except (OSError, json.JSONDecodeError):
        pass
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def write_board_metadata(
    board: Optional[str], *, name: Optional[str] = None, description: Optional[str] = None,
    icon: Optional[str] = None, color: Optional[str] = None, archived: Optional[bool] = None,
    default_workdir: Optional[str] = None, project_id: Optional[str] = None,
) -> dict:
    """Create/update ``board.json``; unmentioned fields are preserved, ``created_at``
    set on first write. ``project_id``/``default_workdir``: ``None`` = unchanged,
    "" = clear (``project_id`` is not validated here)."""
    _assert_not_delegated_child_mutation()
    slug = _slug_or_default(board)
    meta = read_board_metadata(slug)
    # db_path is derived on every read; never persist it into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    for key, value in (("description", description), ("icon", icon), ("color", color)):
        if value is not None:
            meta[key] = str(value)
    if archived is not None:
        meta["archived"] = bool(archived)
    for key, value in (("default_workdir", default_workdir), ("project_id", project_id)):
        if value is not None:
            meta[key] = str(value) if value else None
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def create_board(
    slug: str, *, name: Optional[str] = None, description: Optional[str] = None,
    icon: Optional[str] = None, color: Optional[str] = None, default_workdir: Optional[str] = None,
    project_id: Optional[str] = None,
) -> dict:
    """Create board dir + DB + metadata (``mkdir -p`` semantics: existing board returns its metadata)."""
    normed = _require_slug(slug)
    meta = write_board_metadata(
        normed, name=name, description=description, icon=icon, color=color,
        default_workdir=default_workdir, project_id=project_id,
    )
    # Touch the DB so list_boards() sees it immediately.
    init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Metadata for every board: ``default`` first (always present), then
    ``boards/<slug>/`` dirs holding a ``kanban.db`` or ``board.json``, sorted."""
    entries = [read_board_metadata(DEFAULT_BOARD)]
    seen = {DEFAULT_BOARD}
    root = boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            try:
                normed = _normalize_board_slug(child.name)  # skip junk dirs, don't raise
            except ValueError:
                continue
            if not normed or normed in seen or not _dir_holds_board(child):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Archive (to ``boards/_archived/<slug>-<ts>/``) or delete a board;
    ``default`` cannot be removed. Returns ``{"slug", "action", "new_path"}``."""
    _assert_not_delegated_child_mutation()
    normed = _require_slug(slug)
    if normed == DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if get_current_board() == normed:
        clear_current_board()

    # A concurrent connect() after the rename recreates an empty DB file; drop
    # the init cache first so the schema pass re-runs on it.
    _INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        archive_root = boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        suffix = 1
        while target.exists():  # rapid double-archive
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    import shutil
    shutil.rmtree(d)
    return {"slug": normed, "action": "deleted", "new_path": ""}


# --- Data classes ---

@dataclass
class Task:
    """In-memory view of a row from the ``tasks`` table."""

    id: str
    title: str
    body: Optional[str]
    assignee: Optional[str]
    status: str
    priority: int
    created_by: Optional[str]
    created_at: int
    started_at: Optional[int]
    completed_at: Optional[int]
    workspace_kind: str
    workspace_path: Optional[str]
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    tenant: Optional[str]
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    result: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Column semantics: see SCHEMA_SQL.
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    skills: Optional[list] = None            # None = defaults only; [] = explicitly none
    model_override: Optional[str] = None
    provider_override: Optional[str] = None  # provider ``model_override`` belongs to
    reasoning_effort: Optional[str] = None   # VALID_REASONING_EFFORTS | "none"; NULL = profile's
    # Breaker trip count; None -> ``kanban.failure_limit`` -> DEFAULT_FAILURE_LIMIT.
    max_retries: Optional[int] = None
    # ``/goal``-style loop: a judge re-checks each turn IN THE SAME SESSION until
    # done / budget exhausted (-> kanban_block); ``goal_max_turns`` None -> goals default.
    goal_mode: bool = False
    goal_max_turns: Optional[int] = None
    session_id: Optional[str] = None         # originating HERMES_SESSION_ID; NULL from CLI/dashboard
    # VALID_BLOCK_KINDS or None (legacy); kept across unblock so a same-kind re-block reads as a loop.
    block_kind: Optional[str] = None
    block_recurrences: int = 0               # unblock-loop counter, see BLOCK_RECURRENCE_LIMIT
    completion_contract: Optional[str] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        g = lambda col, default=None: _row_get(row, col, default)  # noqa: E731
        parsed = _json_or(g("skills"))
        skills_value = [str(s) for s in parsed if s] if isinstance(parsed, list) else None
        return cls(
            **{col: row[col] for col in _TASK_REQUIRED_COLUMNS},
            **{col: g(col) for col in _TASK_OPTIONAL_COLUMNS},
            **{col: g(col) or None for col in _TASK_EMPTY_IS_NULL_COLUMNS},
            # Pre-migration fallbacks (spawn_failures / last_spawn_error) are only
            # reachable on a DB never opened since the rename migration landed.
            consecutive_failures=g("consecutive_failures", g("spawn_failures", 0)),
            last_failure_error=g("last_failure_error", g("last_spawn_error")),
            skills=skills_value,
            goal_mode=bool(g("goal_mode")),
            block_recurrences=int(g("block_recurrences") or 0),
        )


# Columns every schema version has (KeyError if the SELECT omitted them).
_TASK_REQUIRED_COLUMNS = (
    "id", "title", "body", "assignee", "status", "priority", "created_by", "created_at",
    "started_at", "completed_at", "workspace_kind", "workspace_path", "claim_lock", "claim_expires",
)
# Later-added columns read as NULL when absent from the row.
_TASK_OPTIONAL_COLUMNS = (
    "branch_name", "project_id", "tenant", "result", "idempotency_key", "worker_pid",
    "max_runtime_seconds", "last_heartbeat_at", "current_run_id", "workflow_template_id",
    "current_step_key", "max_retries", "session_id", "completion_contract",
)
# Text columns where "" is stored/read as "not set".
_TASK_EMPTY_IS_NULL_COLUMNS = (
    "model_override", "provider_override", "reasoning_effort", "goal_max_turns", "block_kind",
)


@dataclass
class Run:
    """One attempt at a task (``task_runs`` row): opened on claim, closed on
    complete/block/crash/timeout/reclaim; carries the handoff summary."""

    id: int
    task_id: str
    profile: Optional[str]
    step_key: Optional[str]
    status: str
    claim_lock: Optional[str]
    claim_expires: Optional[int]
    worker_pid: Optional[int]
    max_runtime_seconds: Optional[int]
    last_heartbeat_at: Optional[int]
    started_at: int
    ended_at: Optional[int]
    outcome: Optional[str]
    summary: Optional[str]
    metadata: Optional[dict]
    error: Optional[str]

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        return cls(
            **{
                col: row[col] for col in (
                    "task_id", "profile", "step_key", "status", "claim_lock", "claim_expires",
                    "worker_pid", "max_runtime_seconds", "last_heartbeat_at", "outcome", "summary", "error",
                )
            },
            id=int(row["id"]),
            started_at=int(row["started_at"]),
            ended_at=_opt_int(row["ended_at"]),
            metadata=_json_or(row["metadata"]),
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Comment":
        return cls(
            id=r["id"], task_id=r["task_id"], author=r["author"],
            body=r["body"], created_at=r["created_at"],
        )


@dataclass
class Attachment:
    """In-memory view of a row from the ``task_attachments`` table."""

    id: int
    task_id: str
    filename: str
    stored_path: str
    content_type: Optional[str]
    size: int
    uploaded_by: Optional[str]
    created_at: int

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> "Attachment":
        return cls(
            id=r["id"], task_id=r["task_id"], filename=r["filename"],
            stored_path=r["stored_path"], content_type=r["content_type"],
            size=r["size"] or 0, uploaded_by=r["uploaded_by"], created_at=r["created_at"],
        )


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Event":
        run_id = _row_get(row, "run_id")
        return cls(
            id=row["id"], task_id=row["task_id"], kind=row["kind"],
            payload=_json_or(row["payload"]), created_at=row["created_at"], run_id=_opt_int(run_id),
        )


# --- Schema ---

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    title                TEXT NOT NULL,
    body                 TEXT,
    assignee             TEXT,
    status               TEXT NOT NULL,
    priority             INTEGER DEFAULT 0,
    created_by           TEXT,
    created_at           INTEGER NOT NULL,
    started_at           INTEGER,
    completed_at         INTEGER,
    workspace_kind       TEXT NOT NULL DEFAULT 'scratch',
    workspace_path       TEXT,
    branch_name          TEXT,
    -- Optional link to a first-class Project (hermes_cli/projects_db). When set,
    -- the task's worktree is anchored under the project's primary repo with a
    -- deterministic branch name instead of a random wt/<task-id> fallback.
    project_id           TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    result               TEXT,
    idempotency_key      TEXT,
    -- Unified consecutive-failure counter. Incremented on spawn
    -- failure, timeout, or crash; reset only on successful completion.
    -- The circuit breaker in _record_task_failure trips when this
    -- exceeds DEFAULT_FAILURE_LIMIT consecutive non-successes.
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    worker_pid           INTEGER,
    -- Short excerpt of the most recent failure's error text.
    last_failure_error   TEXT,
    max_runtime_seconds  INTEGER,
    last_heartbeat_at    INTEGER,
    -- Pointer into task_runs for the currently-active run (NULL if no
    -- run is in-flight). Denormalised for cheap reads.
    current_run_id       INTEGER,
    -- Forward-compat for v2 workflow routing. In v1 the kernel writes
    -- these when the task is opted into a template but otherwise ignores
    -- them; the dispatcher doesn't consult them for routing yet.
    workflow_template_id TEXT,
    current_step_key     TEXT,
    -- Force-loaded skills for the worker on this task, stored as JSON.
    -- Passed to the worker via `--skills`. NULL or empty array = no extras.
    skills               TEXT,
    -- Per-task model override. When set, the dispatcher passes -m <model>
    -- to the worker, overriding the profile's default model. NULL = use
    -- the profile default.
    model_override       TEXT,
    -- Provider the model override belongs to. When set (alongside
    -- model_override), the dispatcher passes --provider <name> so the
    -- worker resolves the model against the right backend instead of the
    -- profile's configured provider. NULL = profile provider.
    provider_override    TEXT,
    -- Per-task reasoning effort for the worker (minimal|low|medium|high|
    -- xhigh|max|ultra, or 'none' for thinking off). When set, the dispatcher
    -- passes --reasoning <level> so the worker runs at that depth regardless
    -- of the profile's agent.reasoning_effort. NULL = profile setting.
    reasoning_effort     TEXT,
    -- Per-task override for the consecutive-failure circuit breaker.
    -- The value is the failure count at which the breaker trips — e.g.
    -- ``max_retries=1`` blocks on the first failure. NULL (the common
    -- case) falls through to the dispatcher-level ``kanban.failure_limit``
    -- config and then ``DEFAULT_FAILURE_LIMIT``.
    max_retries          INTEGER,
    -- When 1, the dispatched worker runs in a Ralph-style goal loop: an
    -- auxiliary judge re-evaluates the worker's response against the
    -- card title/body after each turn and feeds a continuation prompt
    -- back into the SAME session until the judge agrees the work is done
    -- or ``goal_max_turns`` is exhausted. NULL/0 = classic single-shot
    -- worker (the default).
    goal_mode            INTEGER NOT NULL DEFAULT 0,
    -- Goal-loop turn budget for ``goal_mode`` workers. NULL = use the
    -- goals-engine default.
    goal_max_turns       INTEGER,
    -- Originating chat/agent session id when the task was created from
    -- inside an agent loop that propagated ``HERMES_SESSION_ID``. NULL
    -- for tasks created from the CLI, dashboard, or any path that doesn't
    -- set the env var. Indexed so per-session list queries stay cheap on
    -- larger boards.
    session_id           TEXT,
    -- Typed block reason set by ``block_task`` (one of VALID_BLOCK_KINDS, or
    -- NULL for legacy/un-typed blocks). Drives routing: ``dependency`` never
    -- sits in ``blocked`` (goes to ``todo`` for parent-gating); the others go
    -- to ``blocked`` for a human. Preserved across unblock so a re-block for
    -- the SAME kind can be recognised as a loop.
    block_kind           TEXT,
    -- Unblock-loop counter. Incremented each time a task is re-blocked for the
    -- same truly-blocked reason after having been unblocked. When it reaches
    -- BLOCK_RECURRENCE_LIMIT the task is routed to ``triage`` instead of
    -- ``blocked`` so a cron can't spin it forever. Reset to 0 only on a
    -- successful completion — NOT on unblock (resetting on unblock is exactly
    -- the amnesia that let the loop run unbounded).
    block_recurrences    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS task_links (
    parent_id  TEXT NOT NULL,
    child_id   TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS task_comments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS task_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
);

-- Historical attempt record. Each time the dispatcher claims a task, a
-- new row is created here; claim state, PID, heartbeat, runtime cap,
-- and structured summary all live on the run, not the task. Multiple
-- rows per task id when the task was retried after crash/timeout/block.
-- v2 of the kanban schema will use ``step_key`` to drive per-stage
-- workflow routing; in v1 the column is nullable and unused (kernel
-- ignores it).
CREATE TABLE IF NOT EXISTS task_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id             TEXT NOT NULL,
    profile             TEXT,
    step_key            TEXT,
    status              TEXT NOT NULL,
    -- status: running | done | blocked | crashed | timed_out | failed | released
    claim_lock          TEXT,
    claim_expires       INTEGER,
    worker_pid          INTEGER,
    max_runtime_seconds INTEGER,
    last_heartbeat_at   INTEGER,
    started_at          INTEGER NOT NULL,
    ended_at            INTEGER,
    outcome             TEXT,
    -- outcome: completed | blocked | crashed | timed_out | spawn_failed |
    --          gave_up | reclaimed | (null while still running)
    summary             TEXT,
    metadata            TEXT,
    error               TEXT
);

-- Files attached to a task (PDFs, images, source documents). The blob
-- lives on disk under ``attachments_root(board)/<task_id>/<stored_name>``;
-- this row carries metadata + the absolute ``stored_path`` so the
-- dashboard can list/download and ``build_worker_context`` can surface
-- the absolute path to the worker (which has full file-tool access). See
-- #35338.
CREATE TABLE IF NOT EXISTS task_attachments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    filename     TEXT NOT NULL,
    stored_path  TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL DEFAULT 0,
    uploaded_by  TEXT,
    created_at   INTEGER NOT NULL
);

-- Subscription from a gateway source (platform + chat + thread) to a
-- task. The gateway's kanban-notifier watcher tails task_events and
-- pushes ``completed`` / ``blocked`` / ``spawn_auto_blocked`` events to
-- the original requester so human-in-the-loop workflows close the loop.
CREATE TABLE IF NOT EXISTS kanban_notify_subs (
    task_id       TEXT NOT NULL,
    platform      TEXT NOT NULL,
    chat_id       TEXT NOT NULL,
    thread_id     TEXT NOT NULL DEFAULT '',
    user_id       TEXT,
    user_id_alt   TEXT,
    chat_type     TEXT,
    notifier_profile TEXT,
    delivery_mode TEXT NOT NULL DEFAULT 'notify',
    delivery_metadata TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
    last_ping_event_id INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task_id, platform, chat_id, thread_id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_assignee_status ON tasks(assignee, status);
CREATE INDEX IF NOT EXISTS idx_tasks_status          ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_links_child           ON task_links(child_id);
CREATE INDEX IF NOT EXISTS idx_links_parent          ON task_links(parent_id);
CREATE INDEX IF NOT EXISTS idx_comments_task         ON task_comments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_task           ON task_events(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_runs_task             ON task_runs(task_id, started_at);
CREATE INDEX IF NOT EXISTS idx_runs_status           ON task_runs(status);
CREATE INDEX IF NOT EXISTS idx_attachments_task      ON task_attachments(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notify_task           ON kanban_notify_subs(task_id);
"""


# --- ID generation ---

def _new_task_id() -> str:
    """``t_`` + 4 hex bytes (collision ~1e-3 at 100k tasks; 2 bytes would hit 50%
    by 10k). Idempotency belongs to ``idempotency_key``, not id uniqueness."""
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def _host_prefix() -> str:
    """``"<host>:"`` prefix shared by every claim lock issued from this host."""
    return f"{_claimer_id().split(':', 1)[0]}:"


# --- Task creation / mutation ---

def _validate_model_override(model: Optional[str], provider: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Strip both; a provider without a model is rejected (a bare ``--provider``
    would re-resolve the profile's model against another backend — exactly
    the mismatch the override exists to kill)."""
    model = (model or "").strip() or None
    provider = (provider or "").strip() or None
    if provider and not model:
        raise ValueError("provider_override requires a model_override")
    return model, provider


def _canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def _resolve_project_link(
    conn: sqlite3.Connection, project_id: Optional[str], project_source_task_id: Optional[str],
    workspace_kind: str, workspace_path: Optional[str],
) -> tuple[Optional[str], Any, Optional[str], str]:
    """``(project_id, project_obj, project_repo, workspace_kind)`` for ``create_task``.

    A project-linked task is anchored to the project's primary repo as a
    worktree with a deterministic branch (slug + task id). Projects live in the
    creator's per-profile projects.db, but the stored repo path is absolute so
    the cross-profile dispatcher needs no projects.db access. ``project_repo``
    is set when the worktree path must still be derived from the new task id.
    """
    project_id = (str(project_id).strip() or None) if project_id is not None else None
    if not project_id:
        return None, None, None, workspace_kind
    from hermes_cli import projects_db as _pdb

    project_repo: Optional[str] = None
    try:
        with _pdb.connect_closing() as _pconn:
            project_obj = _pdb.get_project(_pconn, project_id)
    except Exception:
        project_obj = None
    if project_obj is None and project_source_task_id:
        project_obj, project_repo = _project_from_source_task(
            conn, _pdb, project_id, str(project_source_task_id),
        )
        if project_obj is not None and workspace_kind == "scratch":
            workspace_kind = "worktree"
    if project_obj is None:
        # Unresolvable id/slug: drop the link (never a dangling reference,
        # never a crash) and create an ordinary scratch task.
        return None, None, None, workspace_kind
    # Canonicalise (a slug may have been passed) and anchor the worktree
    # under the project's primary repo.
    if workspace_kind == "scratch" and project_obj.primary_path:
        workspace_kind = "worktree"
    if workspace_kind == "worktree" and workspace_path is None and project_obj.primary_path:
        # Concrete path is deferred to the insert loop: a fresh
        # ``<repo>/.worktrees/<task-id>`` keyed on the new task id.
        project_repo = str(project_obj.primary_path)
    return project_obj.id, project_obj, project_repo, workspace_kind


def _project_from_source_task(
    conn: sqlite3.Connection, _pdb: Any, project_id: str, source_task_id: str,
) -> tuple[Any, Optional[str]]:
    """Recover a Project (and its repo) from a canonical project-linked
    worktree task on this board. Worker profiles have their own projects.db
    while the Kanban DB is shared, so this carries the repo + branch
    convention forward without opening the creator's store and without
    reusing the source task's literal worktree path. ``(None, None)`` when
    the source task is not a ``<repo>/.worktrees/<id>`` project worktree."""
    source_task = get_task(conn, source_task_id)
    if not (
        source_task is not None
        and source_task.project_id == project_id
        and source_task.workspace_kind == "worktree"
        and source_task.workspace_path
    ):
        return None, None
    source_path = Path(source_task.workspace_path)
    if not (
        source_path.is_absolute()
        and source_path.name == source_task.id
        and source_path.parent.name == ".worktrees"
    ):
        return None, None
    project_slug = None
    if source_task.branch_name:
        prefix, separator, leaf = source_task.branch_name.partition("/")
        if separator and (leaf == source_task.id or leaf.startswith(f"{source_task.id}-")):
            with contextlib.suppress(ValueError):
                project_slug = _pdb.normalize_slug(prefix)
    if project_slug is None:
        with contextlib.suppress(ValueError):
            project_slug = _pdb.normalize_slug(project_id)
    if not project_slug:
        return None, None
    project_repo = str(source_path.parent.parent)
    project_obj = _pdb.Project(
        id=project_id, slug=project_slug, name=project_slug, created_at=0, primary_path=project_repo,
    )
    return project_obj, project_repo


def _normalize_task_skills(skills: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Strip/dedupe a skills list. Commas are refused (a comma-joined string must
    not land in one argv slot); toolset names are rejected all at once because
    agents that confuse the two usually pass several."""
    if skills is None:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    toolset_typos: list[str] = []
    for s in skills:
        if not s:
            continue
        name = str(s).strip()
        if not name:
            continue
        if "," in name:
            raise ValueError(
                f"skill name cannot contain comma: {name!r} "
                f"(pass a list of separate names instead of a comma-joined string)"
            )
        if name.casefold() in KNOWN_TOOLSET_NAMES:
            toolset_typos.append(name)
            continue
        if name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if toolset_typos:
        quoted = ", ".join(repr(n) for n in toolset_typos)
        noun = "is a toolset name" if len(toolset_typos) == 1 else "are toolset names"
        raise ValueError(
            f"{quoted} {noun}, not skill name(s). "
            "Put toolsets in the assignee profile's `toolsets:` config "
            "instead of per-task skills. Skills are named skill bundles "
            "(e.g. `blogwatcher`, `github-code-review`); toolsets are runtime "
            "capabilities (e.g. `web`, `browser`, `terminal`)."
        )
    return cleaned


def create_task(
    conn: sqlite3.Connection, *, title: str, body: Optional[str] = None,
    assignee: Optional[str] = None, created_by: Optional[str] = None,
    workspace_kind: Optional[str] = None, workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None, tenant: Optional[str] = None, priority: int = 0,
    parents: Iterable[str] = (), triage: bool = False, idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None, skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None, model_override: Optional[str] = None,
    provider_override: Optional[str] = None, reasoning_effort: Optional[str] = None,
    goal_mode: bool = False, goal_max_turns: Optional[int] = None, initial_status: str = "running",
    session_id: Optional[str] = None, board: Optional[str] = None, project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    creator_task_id: Optional[str] = None,
    completion_contract: Optional[str] = None,
) -> str:
    """Create a task (optionally under ``parents``); returns its id.

    Status: ``ready`` unless a parent is not ``done`` (``todo``); ``triage=True``
    forces ``triage``; ``initial_status="blocked"`` parks it for human ops.
    ``idempotency_key``: an existing non-archived task with the key is returned
    instead of a duplicate. ``max_runtime_seconds``: cap before the dispatcher
    SIGTERMs and re-queues. ``model_override``/``provider_override`` pin the
    worker model (provider requires model); ``reasoning_effort`` is independent.
    ``creator_task_id``: inherit durable session/subscriptions independently of
    dependency edges; an explicit ``session_id`` still wins.
    ``project_source_task_id``: cross-profile fallback when ``project_id`` is not
    in the active profile's projects.db — see ``_resolve_project_link``.
    ``workspace_kind=None`` (omitted) inherits a project-scoped board's project;
    an explicit ``"scratch"`` or ``project_id=""`` is a request for no project.
    """
    from hermes_cli.kanban_db_graph import initial_task_state, inherit_creator_origin
    from hermes_cli.kanban_pr_acceptance import validate_contract

    completion_contract = validate_contract(completion_contract)
    model_override, provider_override = _validate_model_override(model_override, provider_override)
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    assignee = _canonical_assignee(assignee)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}")
    # A project-scoped board anchors every new task to its project's repo
    # (deterministic worktree + branch) without each surface repeating it.
    # An explicit ``scratch`` (or ``project_id=""``) is a request for no project:
    # it must not be upgraded to a worktree in the board's repo (#106342).
    if project_id is None and workspace_kind != "scratch":
        try:
            project_id = (_board_meta_for(board).get("project_id") or "").strip() or None
        except Exception:
            pass
    if workspace_kind is None:
        workspace_kind = "scratch"
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    project_id, project_obj, project_repo, workspace_kind = _resolve_project_link(
        conn, project_id, project_source_task_id, workspace_kind, workspace_path
    )
    parents = tuple(p for p in parents if p)
    skills_list = _normalize_task_skills(skills)

    # Idempotency check BEFORE the write txn (no lock held); a concurrent-create
    # race may insert twice, the next lookup stabilises on the newest.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1", (idempotency_key,),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Only persistent kinds inherit the board ``default_workdir``: a scratch
    # task inheriting it would point cleanup at the user's source tree.
    if workspace_path is None and project_repo is None and workspace_kind in {"dir", "worktree"}:
        board_default = _board_meta_for(board).get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            # allow_nested: graph builders compose create_task under one outer
            # commit so the dispatcher never sees a half-built graph.
            with write_txn(conn, allow_nested=True):
                task_status, tenant = initial_task_state(conn, parents, initial_status, triage, tenant)
                # Project worktree: fresh dir under the repo + deterministic
                # branch, instead of the random ``wt/<id>`` worker fallback.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(project_repo, ".worktrees", task_id)
                    if not branch_name:
                        branch_name = _project_branch_name(project_obj, task_id, title)

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort,
                        goal_mode, goal_max_turns, session_id, completion_contract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, title.strip(), body, assignee, task_status, priority,
                        created_by, now, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        _opt_int(max_runtime_seconds),
                        json.dumps(skills_list) if skills_list is not None else None,
                        _opt_int(max_retries), model_override, provider_override, reasoning_effort,
                        1 if goal_mode else 0, _opt_int(goal_max_turns), session_id, completion_contract,
                    ),
                )
                for pid in parents:
                    _link(conn, pid, task_id)
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "creator_task_id": creator_task_id,
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "project_id": project_id,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                    },
                )
                # ACK-edge: the originating channel hears a child BLOCK, not just the fan-in.
                inherit_creator_origin(conn, task_id, creator_task_id, created_at=now)
                _inherit_notify_subs(conn, task_id, parents, created_at=now)
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
    raise RuntimeError("unreachable")


def _board_meta_for(board: Optional[str]) -> dict:
    return read_board_metadata(board if board else get_current_board())


def _project_branch_name(project_obj: Any, task_id: str, title: Optional[str]) -> Optional[str]:
    from hermes_cli import projects_db as _pdb

    try:
        return _pdb.branch_name_for(project_obj, task_id, title=title or "")
    except Exception:
        return None


def _link(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
        (parent_id, child_id),
    )


def _missing_task_ids(conn: sqlite3.Connection, ids: Iterable[str]) -> list[str]:
    """Subset of ``ids`` (order kept) with no ``tasks`` row."""
    ids = list(ids)
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT id FROM tasks WHERE id IN ({placeholders})", ids).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in ids if p not in present]


def _inherit_notify_subs(
    conn: sqlite3.Connection, child_id: str, parents: Iterable[str], *,
    created_at: Optional[int] = None,
) -> None:
    """Copy parents' notify subscriptions to a child, cursor caught up to the
    child's current event so a late ``link_tasks`` never replays history.

    Single owner of inheritance (create_task, link_tasks, decompose). It must
    copy EVERY routing/delivery column: dropping ``chat_type`` made DM-originated
    completions wake a fresh group session instead of the originating DM.

    Omitting columns here silently degrades routing: a DM-originated child completion falls back to
    chat_type='group' and wakes a fresh group-scoped session instead of the originating DM (issue #73030).
    """
    parent_ids = tuple(dict.fromkeys(p for p in parents if p))
    if not parent_ids:
        return
    row = conn.execute(
        "SELECT COALESCE(MAX(id), 0) AS cursor FROM task_events WHERE task_id = ?", (child_id,),
    ).fetchone()
    cursor = int(row["cursor"] if row is not None else 0)
    placeholders = ",".join("?" * len(parent_ids))
    conn.execute(
        f"""
        INSERT OR IGNORE INTO kanban_notify_subs
            (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
             chat_type, notifier_profile, delivery_mode, delivery_metadata,
             created_at, last_event_id)
        SELECT ?, platform, chat_id, thread_id, user_id, user_id_alt,
               COALESCE(chat_type, 'dm'), notifier_profile,
               COALESCE(delivery_mode, 'notify'), delivery_metadata, ?, ?
          FROM kanban_notify_subs
         WHERE task_id IN ({placeholders})
        """,
        (child_id, int(created_at if created_at is not None else time.time()), cursor, *parent_ids),
    )


def get_task(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return Task.from_row(row) if row else None


# Canonical sort-order mappings for ``hermes kanban list --sort``.
# Each value is a raw SQL fragment appended after ``ORDER BY``.
VALID_SORT_ORDERS: dict[str, str] = {
    "created": "created_at ASC, id ASC",
    "created-desc": "created_at DESC, id DESC",
    "priority": "priority DESC, created_at ASC",
    "priority-desc": "priority ASC, created_at ASC",
    "status": "status ASC, created_at ASC",
    "assignee": "assignee ASC, created_at ASC",
    "title": "title ASC, id ASC",
    "updated": "started_at DESC NULLS LAST, created_at DESC",
}


def list_tasks(
    conn: sqlite3.Connection, *, assignee: Optional[str] = None, status: Optional[str] = None,
    tenant: Optional[str] = None, session_id: Optional[str] = None, include_archived: bool = False,
    limit: Optional[int] = None, order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None, current_step_key: Optional[str] = None,
) -> list[Task]:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    for col, val in (
        ("assignee", _canonical_assignee(assignee)), ("status", status), ("tenant", tenant),
        ("session_id", session_id), ("workflow_template_id", workflow_template_id),
        ("current_step_key", current_step_key),
    ):
        if val is not None:
            query += f" AND {col} = ?"
            params.append(val)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}")
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign/reassign; raises RuntimeError while the task is running under a claim."""
    profile = _canonical_assignee(profile)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row:
            return False
        if row["claim_lock"] is not None and row["status"] == "running":
            raise RuntimeError(
                f"cannot reassign {task_id}: currently running (claimed). "
                "Wait for completion or reclaim the stale lock first."
            )
        if row["assignee"] != profile:
            # The failure streak is per task/profile; a new profile starts fresh.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL WHERE id = ?", (profile, task_id),
            )
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _append_event(conn, task_id, "assigned", {"assignee": profile})
    # Observer fires AFTER commit so subscribers see durable state.
    notify_task_updated(conn, task_id, ("assignee",))
    return True


def set_model_override(
    conn: sqlite3.Connection, task_id: str, model: Optional[str], provider: Optional[str] = None,
) -> bool:
    """Set (empty ``model`` clears BOTH) the per-task model/provider override.
    Allowed while ``running``: it applies on the NEXT dispatch, which is the
    rate-limit-recovery flow (set, then reclaim/retry)."""
    model, provider = _validate_model_override(model, provider)
    return _set_task_override(
        conn, task_id,
        "UPDATE tasks SET model_override = ?, provider_override = ? WHERE id = ?", (model, provider),
        "model_override_set", {"model": model, "provider": provider},
        ("model_override", "provider_override"), archived_msg="cannot set model override",
    )


def _set_task_override(
    conn: sqlite3.Connection, task_id: str, sql: str, params: tuple, event_kind: str, payload: dict,
    changed_fields: tuple[str, ...], *, archived_msg: str,
) -> bool:
    """Per-task override write: refuse archived tasks, record ``event_kind``,
    then fire the task-updated observer AFTER commit (RFC #58548)."""
    with write_txn(conn):
        status = _task_status(conn, task_id)
        if status is None:
            return False
        if status == "archived":
            raise RuntimeError(f"{archived_msg} on archived task {task_id}")
        conn.execute(sql, (*params, task_id))
        _append_event(conn, task_id, event_kind, payload)
    notify_task_updated(conn, task_id, changed_fields)
    return True


def set_reasoning_effort(conn: sqlite3.Connection, task_id: str, effort: Optional[str]) -> bool:
    """Set (empty clears; ``"none"`` pins thinking OFF) the per-task reasoning
    effort. Independent of the model override so clearing one never resets the
    other; applies on the NEXT dispatch, so settable while running."""
    effort = normalize_reasoning_effort(effort)
    return _set_task_override(
        conn, task_id, "UPDATE tasks SET reasoning_effort = ? WHERE id = ?", (effort,),
        "reasoning_effort_set", {"reasoning_effort": effort},
        ("reasoning_effort",), archived_msg="cannot set reasoning effort",
    )


# --- Links ---

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _missing_task_ids(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(f"linking {parent_id} -> {child_id} would create a cycle")
        _link(conn, parent_id, child_id)
        # If child was ready but parent is not yet done, demote child to todo.
        if _task_status(conn, parent_id) != "done":
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'", (child_id,),
            )
        _append_event(
            conn, child_id, "linked", {"parent": parent_id, "child": child_id},
        )
        _inherit_notify_subs(conn, child_id, (parent_id,))


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """True iff ``parent_id`` is already a descendant of ``child_id``."""
    seen = set()
    stack = [child_id]
    while stack:
        node = stack.pop()
        if node == parent_id:
            return True
        if node in seen:
            continue
        seen.add(node)
        rows = conn.execute(
            "SELECT child_id FROM task_links WHERE parent_id = ?", (node,)
        ).fetchall()
        stack.extend(r["child_id"] for r in rows)
    return False


def unlink_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?", (parent_id, child_id),
        )
        removed = cur.rowcount > 0
        if removed:
            _append_event(conn, child_id, "unlinked", {"parent": parent_id, "child": child_id})
    if removed:
        # Re-gate the child now (as complete_task/unblock_task do) instead of
        # leaving it in todo until the next tick.
        recompute_ready(conn)
    return removed


def _linked_ids(conn: sqlite3.Connection, want: str, where: str, task_id: str) -> list[str]:
    rows = conn.execute(
        f"SELECT {want} FROM task_links WHERE {where} = ? ORDER BY {want}", (task_id,)
    ).fetchall()
    return [r[want] for r in rows]


# Dependency edge removed — re-evaluate promotion eligibility for the child immediately. Matches the
# contract of complete_task and unblock_task; without this the child stays stuck in todo until the next
# dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return _linked_ids(conn, "parent_id", "child_id", task_id)


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    return _linked_ids(conn, "child_id", "parent_id", task_id)


def task_graph_contexts(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, dict]:
    """Bulk-load compact direct graph state for graph-aware diagnostics."""
    ordered_ids = list(dict.fromkeys(str(task_id) for task_id in task_ids if task_id))
    contexts = {task_id: {"parents": [], "children": []} for task_id in ordered_ids}
    if not ordered_ids:
        return contexts

    placeholders = ",".join("?" for _ in ordered_ids)
    for bucket, own, other in (("parents", "child_id", "parent_id"), ("children", "parent_id", "child_id")):
        for row in conn.execute(
            f"SELECT l.{own} AS owner_id, t.id, t.title, t.status "
            f"FROM task_links l JOIN tasks t ON t.id = l.{other} "
            f"WHERE l.{own} IN ({placeholders}) ORDER BY l.{own}, t.id", tuple(ordered_ids),
        ).fetchall():
            contexts[row["owner_id"]][bucket].append(
                {"id": row["id"], "title": row["title"], "status": row["status"]}
            )
    return contexts


def task_graph_context(conn: sqlite3.Connection, task_id: str) -> dict:
    """Return compact direct parent/child state for one task."""
    return task_graph_contexts(conn, [task_id])[task_id]


# --- Comments & events ---

def add_comment(conn: sqlite3.Connection, task_id: str, author: str, body: str) -> int:
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    now = int(time.time())
    # ``allow_nested=True``: graph builders (kanban_swarm blackboard seeding)
    # compose comment writes under one outer commit.
    with write_txn(conn, allow_nested=True):
        _require_task(conn, task_id)
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)", (task_id, author.strip(), body.strip(), now),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


def _require_task(conn: sqlite3.Connection, task_id: str) -> None:
    if not conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone():
        raise ValueError(f"unknown task {task_id}")


def _task_rows(conn: sqlite3.Connection, table: str, task_id: str, order: str) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM {table} WHERE task_id = ? ORDER BY {order}", (task_id,)
    ).fetchall()


def list_comments(conn: sqlite3.Connection, task_id: str) -> list[Comment]:
    return [Comment.from_row(r) for r in _task_rows(conn, "task_comments", task_id, "created_at ASC")]


def list_comments_after(
    conn: sqlite3.Connection, task_id: str, *, after_id: int = 0
) -> list[Comment]:
    """Comments with ``id > after_id`` — keyed on rowid, not ``created_at``, so a
    same-second burst is never skipped (live worker comment bridge)."""
    rows = conn.execute(
        "SELECT id, task_id, author, body, created_at FROM task_comments "
        "WHERE task_id = ? AND id > ? ORDER BY id ASC", (task_id, int(after_id)),
    ).fetchall()
    return [Comment.from_row(r) for r in rows]


# --- Attachments ---

class AttachmentTooLarge(ValueError):
    """Attachment over the size cap. A ``ValueError`` so generic 400 handlers
    still catch it while the tool/CLI can give a 413-style message."""


def _safe_attachment_name(raw: str) -> str:
    """Client filename -> safe basename: strip directories (both separators),
    control chars and leading dots (no dotfiles, no traversal); ValueError when
    nothing usable remains. Only ever joined under the per-task attachments dir."""
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = "".join(ch for ch in name if ch.isprintable() and ch not in "\x00").strip()
    name = name.lstrip(".").strip()
    if not name:
        raise ValueError("invalid attachment filename")
    return name[:200]


def _collision_free_path(dest_dir: Path, safe_name: str) -> Path:
    """``foo.pdf`` -> ``foo.pdf``, ``foo (1).pdf``, ... first one that doesn't exist."""
    stem, dot, ext = safe_name.partition(".")
    candidate = safe_name
    n = 1
    while (dest_dir / candidate).exists():
        candidate = f"{stem} ({n}){dot}{ext}"
        n += 1
    return dest_dir / candidate


def store_attachment_bytes(
    conn: sqlite3.Connection, task_id: str, filename: str, data: bytes, *,
    content_type: Optional[str] = None, uploaded_by: Optional[str] = None,
    board: Optional[str] = None, max_bytes: Optional[int] = None,
) -> int:
    """Single attachment write path (dashboard, tools, CLI): size cap, safe
    basename, collision-free blob under :func:`task_attachments_dir`, then the
    metadata row. Raises :class:`AttachmentTooLarge` / ``ValueError``; a blob
    whose row insert fails is removed before re-raising. Returns the new id."""
    if max_bytes is None:
        max_bytes = KANBAN_ATTACHMENT_MAX_BYTES
    if len(data) > max_bytes:
        raise AttachmentTooLarge(f"attachment exceeds {max_bytes // (1024 * 1024)} MB limit")
    safe_name = _safe_attachment_name(filename)
    dest_dir = task_attachments_dir(task_id, board=board)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = _collision_free_path(dest_dir, safe_name)
    dest_path.write_bytes(data)
    try:
        return add_attachment(
            conn, task_id, filename=dest_path.name, stored_path=str(dest_path.resolve()),
            content_type=content_type, size=len(data), uploaded_by=uploaded_by,
        )
    except Exception:
        # Don't leave an orphan blob if the metadata insert fails (most
        # commonly: the task id doesn't exist).
        with contextlib.suppress(OSError):
            dest_path.unlink(missing_ok=True)
        raise


def add_attachment(
    conn: sqlite3.Connection, task_id: str, *, filename: str, stored_path: str,
    content_type: Optional[str] = None, size: int = 0, uploaded_by: Optional[str] = None,
) -> int:
    """Record the metadata row (+ ``attached`` event) for a blob the caller already wrote."""
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        _require_task(conn, task_id)
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (task_id, filename.strip(), stored_path, content_type, int(size), uploaded_by, now),
        )
        _append_event(
            conn, task_id, "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    return [Attachment.from_row(r) for r in _task_rows(conn, "task_attachments", task_id, "created_at ASC, id ASC")]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute("SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)).fetchone()
    return None if r is None else Attachment.from_row(r)


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete the row (source of truth) and best-effort its blob; None when no row matched."""
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(conn, att.task_id, "attachment_removed", {"filename": att.filename})
    with contextlib.suppress(OSError):
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    return att


def list_events(conn: sqlite3.Connection, task_id: str) -> list[Event]:
    return [Event.from_row(r) for r in _task_rows(conn, "task_events", task_id, "created_at ASC, id ASC")]


def _insert_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str, created_at: int,
) -> None:
    """Raw comment INSERT for callers already inside a write txn (``add_comment``
    opens its own txn and emits ``commented``)."""
    conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?, ?, ?, ?)", (task_id, author, body, created_at),
    )


def _append_event(
    conn: sqlite3.Connection, task_id: str, kind: str, payload: Optional[dict] = None, *,
    run_id: Optional[int] = None,
) -> None:
    """Insert an event row inside the caller's txn; ``run_id`` groups it by attempt (NULL = task-scoped)."""
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)", (task_id, run_id, kind, _json_or_null(payload), int(time.time())),
    )


def _end_run(
    conn: sqlite3.Connection, task_id: str, *, outcome: str, summary: Optional[str] = None,
    error: Optional[str] = None, metadata: Optional[dict] = None, status: Optional[str] = None,
) -> Optional[int]:
    """Close the active run (``status`` defaults to ``outcome``) and clear
    ``current_run_id``; None when no run was active (never-claimed task)."""
    now = int(time.time())
    run_id = _current_run_id(conn, task_id)
    if run_id is None:
        return None
    conn.execute(
        """
        UPDATE task_runs
           SET status        = ?,
               outcome       = ?,
               summary       = ?,
               error         = ?,
               metadata      = ?,
               ended_at      = ?,
               claim_lock    = NULL,
               claim_expires = NULL,
               worker_pid    = NULL
         WHERE id = ?
           AND ended_at IS NULL
        """,
        (status or outcome, outcome, summary, error, _json_or_null(metadata), now, run_id),
    )
    conn.execute("UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,))
    return run_id


def _first_line(text: Optional[str], limit: int) -> str:
    """First non-blank-stripped line of ``text`` capped at ``limit`` chars; "" when empty."""
    lines = (text or "").strip().splitlines()
    return lines[0][:limit] if lines else ""


def _opt_int(value: Any) -> Optional[int]:
    """``int(value)`` or ``None`` when ``value`` is ``None`` (NULL column passthrough)."""
    return int(value) if value is not None else None


def _json_or_null(obj: Any) -> Optional[str]:
    """JSON text for a payload/metadata column; falsy -> NULL."""
    return json.dumps(obj, ensure_ascii=False) if obj else None


def _task_status(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Current ``tasks.status`` for ``task_id``, or ``None`` when no such row."""
    row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row["status"] if row else None


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return int(row["current_run_id"]) if row and row["current_run_id"] else None


def _end_or_synthesize_run(
    conn: sqlite3.Connection, task_id: str, *, outcome: str, status: str,
    summary: Optional[str] = None, metadata: Optional[dict] = None, synthesize: bool,
) -> Optional[int]:
    """:func:`_end_run`; when no run was active and ``synthesize`` holds, record a
    zero-duration run instead so the handoff fields survive in attempt history."""
    run_id = _end_run(conn, task_id, outcome=outcome, status=status, summary=summary, metadata=metadata)
    if run_id is None and synthesize:
        run_id = _synthesize_ended_run(conn, task_id, outcome=outcome, summary=summary, metadata=metadata)
    return run_id


def _synthesize_ended_run(
    conn: sqlite3.Connection, task_id: str, *, outcome: str, summary: Optional[str] = None,
    error: Optional[str] = None, metadata: Optional[dict] = None,
) -> int:
    """Zero-duration closed run for a terminal transition on a never-claimed
    task, so the handoff fields aren't silently dropped (``_end_run`` is a
    no-op then). ``started_at == ended_at`` keeps elapsed stats honest. Does
    NOT touch the tasks row."""
    now = int(time.time())
    trow = conn.execute(
        "SELECT assignee, current_step_key FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    profile = trow["assignee"] if trow else None
    step_key = trow["current_step_key"] if trow else None
    cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key,
            status, outcome,
            summary, error, metadata,
            started_at, ended_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_id, profile, step_key, outcome, outcome, summary, error, _json_or_null(metadata),
            now, now,
        ),
    )
    return int(cur.lastrowid or 0)


# --- Dependency resolution (todo -> ready) ---

def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the newest ``blocked``/``unblocked`` event is ``blocked`` — an
    explicit ``kanban_block`` that must wait for an operator. A breaker trip
    emits ``gave_up`` (not ``blocked``) and so auto-recovers, as does a task
    with no such event at all (direct DB edit).

    See #28712.
    Returns ``False`` when there is no such event at all (e.g. the task was set to ``status='blocked'`` by
    the circuit breaker or by direct DB manipulation) — preserves the pre-#28712 auto-recover semantics for
    that path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def _latest_event(
    conn: sqlite3.Connection, task_id: str, kind: str, run_id: Optional[int] = None,
) -> Optional[sqlite3.Row]:
    """Newest ``task_events`` row of ``kind`` (optionally scoped to one run)."""
    sql = "SELECT payload FROM task_events WHERE task_id = ? AND kind = ?"
    params: tuple[Any, ...] = (task_id, kind)
    if run_id is not None:
        sql += " AND run_id = ?"
        params = (*params, int(run_id))
    return conn.execute(sql + " ORDER BY id DESC LIMIT 1", params).fetchone()


def _resume_status_from_events(conn: sqlite3.Connection, task_id: str) -> str:
    """``review`` when the newest lifecycle event carries a review
    ``resume_status``/``retry_status``/``source_status``, else ``ready`` (legacy)."""
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind IN ("
        "'blocked', 'block_loop_detected', 'dependency_wait', 'gave_up', "
        "'unblocked', 'changes_requested', 'review_reopened', 'status', 'reclaimed', "
        "'stale', 'timed_out', 'crashed', 'spawn_failed', 'rate_limited'"
        ") ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    payload = _json_dict(_row_get(row, "payload"))
    for key in ("resume_status", "retry_status", "source_status"):
        if payload.get(key) == "review":
            return "review"
    return "ready"


def recompute_ready(conn: sqlite3.Connection, failure_limit: int = None) -> int:
    """Promote ``todo``/``blocked`` tasks whose parents are all done/archived;
    returns the count. Opens its own IMMEDIATE txn — call OUTSIDE any write txn.

    ``blocked`` is skipped when sticky (explicit ``kanban_block``) or when
    ``consecutive_failures`` reached the limit (else the breaker could never
    trip). Limit order matches ``_record_task_failure``: ``max_retries`` >
    ``failure_limit`` > ``DEFAULT_FAILURE_LIMIT``.

    1. The most recent block event was a worker-initiated ``kanban_block`` — those stay blocked until an
    explicit ``kanban_unblock`` (#28712).
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Explicit human-intervention block; only ``unblock_task`` may exit it.
                continue
            parents = conn.execute(
                "SELECT t.status FROM tasks t "
                "JOIN task_links l ON l.parent_id = t.id "
                "WHERE l.child_id = ?", (task_id,),
            ).fetchall()
            if all(p["status"] in ("done", "archived") for p in parents):
                resume_status = _resume_status_from_events(conn, task_id)
                if cur_status == "blocked":
                    # At the breaker limit, no auto-recovery (else block ->
                    # recover -> respawn -> exhaust -> block forever). The
                    # counter is preserved so it accumulates across cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = ? "
                        "WHERE id = ? AND status = 'blocked'", (resume_status, task_id),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ? AND status = 'todo'",
                        (resume_status, task_id),
                    )
                _append_event(
                    conn, task_id, "promoted",
                    {"status": resume_status} if resume_status != "ready" else None,
                )
                promoted += 1
    return promoted


# --- Claim / complete / block ---

def _parents_satisfied(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return whether every direct parent is terminal for dependency gating."""
    return conn.execute(
        # Check if this task has children that still need the workspace. If any child is not yet
        # done/archived, defer cleanup so the child can read handoff artifacts from the workspace (#33774).
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? "
        "AND p.status NOT IN ('done', 'archived') LIMIT 1", (task_id,),
    ).fetchone() is None


def _claim_and_open_run(
    conn: sqlite3.Connection, task_id: str, source_status: str, lock: str, expires: int, now: int,
    *, event_extra: Optional[dict] = None,
) -> Optional[int]:
    """CAS ``source_status -> running``, open a run row, emit ``claimed``; None
    when the CAS lost. Caller holds the txn."""
    cur = conn.execute(
        f"""
        UPDATE tasks
           SET status        = 'running',
               claim_lock    = ?,
               claim_expires = ?,
               started_at    = COALESCE(started_at, ?)
         WHERE id = ?
           AND status = '{source_status}'
           AND claim_lock IS NULL
        """,
        (lock, expires, now, task_id),
    )
    if cur.rowcount != 1:
        return None
    trow = conn.execute(
        "SELECT assignee, max_runtime_seconds, current_step_key "
        "FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    run_cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, step_key, status,
            claim_lock, claim_expires, max_runtime_seconds,
            started_at
        ) VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
        """,
        (
            task_id, trow["assignee"] if trow else None, trow["current_step_key"] if trow else None,
            lock, expires, trow["max_runtime_seconds"] if trow else None, now,
        ),
    )
    run_id = run_cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id = ? WHERE id = ?", (run_id, task_id))
    _append_event(
        conn, task_id, "claimed",
        {"lock": lock, "expires": expires, "run_id": run_id, **(event_extra or {})}, run_id=run_id,
    )
    return run_id


def claim_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``ready -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``ready`` status).
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        # Single enforcement point: never ready -> running with an undone
        # parent, whichever writer set 'ready'. Demote to 'todo';
        # recompute_ready re-promotes when the parents finish.
        if not _parents_satisfied(conn, task_id):
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready'", (task_id,),
            )
            _append_event(conn, task_id, "claim_rejected", {"reason": "parents_not_done"})
            return None
        # Close a leaked prior run so the CAS below doesn't strand it.
        _reclaim_dangling_run(
            conn, task_id, statuses=("ready",), now=now, note="invariant recovery on re-claim",
        )
        run_id = _claim_and_open_run(conn, task_id, "ready", lock, expires, now)
        if run_id is None:
            return None
        claimed = get_task(conn, task_id)
    _fire_task_hook("kanban_task_claimed", claimed, task_id, run_id)
    return claimed


def claim_review_task(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomic ``review -> running`` (None when lost). Parents are re-checked
    (one may have reopened meanwhile) and a NEW run tracks the reviewer
    separately from the implementer."""
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            demoted = conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL", (task_id,),
            )
            if demoted.rowcount == 1:
                _append_event(
                    conn, task_id, "dependency_wait",
                    {"reason": "parent_reopened", "source_status": "review"},
                )
            return None
        run_id = _claim_and_open_run(
            conn, task_id, "review", lock, expires, now, event_extra={"source_status": "review"},
        )
        if run_id is None:
            return None
        return get_task(conn, task_id)


def _retry_status_for_run(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int] = None,
) -> str:
    """``review`` when the run's ``claimed`` event says ``source_status=review``,
    else ``ready`` — one place, so crash/timeout/reclaim can't silently turn a
    reviewer run into an implementation run."""
    if run_id is None:
        run_id = _current_run_id(conn, task_id)
    if run_id is None:
        return "ready"
    event = _latest_event(conn, task_id, "claimed", run_id)
    payload = _json_dict(_row_get(event, "payload"))
    return "review" if payload.get("source_status") == "review" else "ready"


# Run outcome -> lifecycle status a goal loop should report for a handed-off run.
_RUN_OUTCOME_TERMINAL_STATUS = {
    "completed": "done",
    "review_requested": "review",
    "changes_requested": "changes_requested",
    "blocked": "blocked",
    "dependency_wait": "blocked",
}


def goal_run_status(
    conn: sqlite3.Connection, task_id: str, expected_run_id: Optional[int] = None,
) -> Optional[str]:
    """Lifecycle status as seen by ONE run: terminal handoffs bind to that run,
    any other ownership loss is ``superseded`` — otherwise an old goal loop
    would read the successor's live ``running`` and mutate it."""
    task = get_task(conn, task_id)
    if task is None:
        return None
    if expected_run_id is not None:
        row = conn.execute(
            "SELECT outcome FROM task_runs WHERE id = ? AND task_id = ?",
            (int(expected_run_id), task_id),
        ).fetchone()
        outcome = str(row["outcome"]) if row and row["outcome"] is not None else None
        terminal_status = _RUN_OUTCOME_TERMINAL_STATUS.get(outcome)
        if terminal_status is not None:
            return terminal_status
        if outcome is not None or task.current_run_id != int(expected_run_id):
            return "superseded"
    if task.status in {"ready", "todo"}:
        event = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1", (task_id,),
        ).fetchone()
        if event and event["kind"] == "changes_requested":
            return "changes_requested"
    return task.status


def heartbeat_claim(
    conn: sqlite3.Connection, task_id: str, *, ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim; True if we still own it."""
    expires = int(time.time()) + _resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?", (expires, task_id, lock),
        )
        if cur.rowcount != 1:
            return False
        _extend_run_claim(conn, task_id, expires)
        return True


def _extend_run_claim(conn: sqlite3.Connection, task_id: str, expires: int) -> Optional[int]:
    """Mirror a task claim extension onto its active run row; returns that run id."""
    run_id = _current_run_id(conn, task_id)
    if run_id is not None:
        conn.execute("UPDATE task_runs SET claim_expires = ? WHERE id = ?", (expires, run_id))
    return run_id


def release_stale_claims(conn: sqlite3.Connection, *, signal_fn=None) -> int:
    """Reclaim ``running`` tasks whose claim expired; returns the count reclaimed.

    A host-local worker that is still alive gets its claim *extended* instead
    (a slow model can sit longer than the TTL inside one tool-free call, so no
    heartbeat) — unless ``last_heartbeat_at`` is older than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (wedged; ``_touch_activity``
    keeps any genuinely active worker fresh). Safe to call often.

    Reclaiming a live worker mid-flight produces the spawn- then-immediately-reclaim loop seen on slow
    models that spend longer than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM call (#23025):
    no tool calls means no ``kanban_heartbeat``, even though the subprocess is healthy.
    Backstop (#29747 gap 3): if the worker's PID is still alive but its ``last_heartbeat_at`` is stale by
    more than ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has been making no observable
    progress and we reclaim anyway — even if ``_pid_alive`` is still true. This catches the
    wedged-in-a-logic-loop case where the process is technically running but accomplishing nothing.
    ``_touch_activity`` (run_agent.py) bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
    so any genuinely active worker keeps its heartbeat fresh as a side effect of normal API traffic.
    ``enforce_max_runtime`` and ``detect_crashed_workers`` remain the upper bounds for genuinely wedged or
    dead workers.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = _host_prefix()
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at, "
        "       assignee "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?", (now,),
    ).fetchall()
    for row in stale:
        host_local = (row["claim_lock"] or "").startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Backstop: a heartbeat older than the max-stale threshold means no
        # observable progress — reclaim even if the PID is alive (logic loop).
        heartbeat_stale = hb is not None and (now - int(hb)) > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        if host_local and row["worker_pid"] and _pid_alive(row["worker_pid"]) and not heartbeat_stale:
            _extend_live_stale_claim(conn, row, now)
            continue

        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # A live worker of ours must keep its claim (else a duplicate spawns beside it).
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, row["id"])
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (retry_status, row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _record_reclaim(
                conn, row["id"], termination,
                error=f"stale_lock={row['claim_lock']}",
                payload={
                    "stale_lock": row["claim_lock"],
                    "worker_pid": _opt_int(row["worker_pid"]),
                    "claim_expires": int(row["claim_expires"]),
                    "last_heartbeat_at": _opt_int(row["last_heartbeat_at"]),
                    "now": now,
                    "host_local": host_local,
                    "heartbeat_stale": bool(heartbeat_stale),
                    "retry_status": retry_status,
                },
            )
            reclaimed += 1
        # Post-commit observer; every non-reclaim branch ``continue``d above.
        if _kanban_observer_consumed("on_kanban_worker_stale_claim"):
            _fire_kanban_lifecycle_hook(
                "on_kanban_worker_stale_claim", row["id"], board=get_current_board(),
                assignee=row["assignee"], run_id=run_id, worker_pid=_opt_int(row["worker_pid"]),
                heartbeat_stale=bool(heartbeat_stale), retry_status=retry_status,
            )
    return reclaimed


def _record_reclaim(
    conn: sqlite3.Connection, task_id: str, termination: dict, *, error: str, payload: dict,
) -> Optional[int]:
    """Close the active run as ``reclaimed`` and emit the ``reclaimed`` event
    (payload merged with the termination report). Caller holds the txn."""
    run_id = _end_run(
        conn, task_id, outcome="reclaimed", status="reclaimed", error=error, metadata=termination,
    )
    payload.update(termination)
    _append_event(conn, task_id, "reclaimed", payload, run_id=run_id)
    return run_id


def _extend_live_stale_claim(conn: sqlite3.Connection, row: sqlite3.Row, now: int) -> None:
    """TTL-expired claim whose host-local worker is alive: extend instead of
    reclaiming (``claim_extended`` event). CAS on the same expired lock so a
    concurrent reclaimer wins cleanly."""
    new_expires = now + _resolve_claim_ttl_seconds()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' "
            "  AND claim_lock IS ? "
            "  AND claim_expires IS NOT NULL "
            "  AND claim_expires < ?", (new_expires, row["id"], row["claim_lock"], now),
        )
        if cur.rowcount != 1:
            return
        run_id = _extend_run_claim(conn, row["id"], new_expires)
        _append_event(
            conn, row["id"], "claim_extended",
            {
                "reason": "pid_alive",
                "worker_pid": int(row["worker_pid"]),
                "claim_lock": row["claim_lock"],
                "claim_expires_was": int(row["claim_expires"]),
                "claim_expires_now": new_expires,
                "last_heartbeat_at": _opt_int(row["last_heartbeat_at"]),
            },
            run_id=run_id,
        )


def reclaim_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None, signal_fn=None,
) -> bool:
    """Operator reclaim regardless of TTL: release the claim, restore the source
    phase, reset the failure counter. False when not running."""
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] != "running" and row["claim_lock"] is None:
        # Nothing to reclaim — already ready / blocked / done.
        return False
    prev_lock = row["claim_lock"]
    termination = _terminate_reclaimed_worker(row["worker_pid"], prev_lock, signal_fn=signal_fn)
    with write_txn(conn):
        retry_status = _retry_status_for_run(conn, task_id)
        cur = conn.execute(
            "UPDATE tasks SET status = ?, claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status IN ('running', 'ready', 'blocked') "
            "AND claim_lock IS ?", (retry_status, task_id, prev_lock),
        )
        if cur.rowcount != 1:
            return False
        _record_reclaim(
            conn, task_id, termination,
            error=f"manual_reclaim: {reason}" if reason else f"manual_reclaim lock={prev_lock}",
            payload={"manual": True, "reason": reason, "prev_lock": prev_lock, "retry_status": retry_status},
        )
    # Operator intervention = fresh retry budget (own txn, runs after commit).
    _clear_failure_counter(conn, task_id)
    return True


def reassign_task(
    conn: sqlite3.Connection, task_id: str, profile: Optional[str], *, reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign (None unassigns); a running task is refused unless
    ``reclaim_first`` releases its claim — the "this profile's model is broken" path."""
    if reclaim_first:
        # Safe to call even if nothing to reclaim.
        reclaim_task(conn, task_id, reason=reason or "reassign")
    # assign_task handles its own txn + the still-running guard.
    try:
        return assign_task(conn, task_id, profile)
    except RuntimeError:
        # Task is still running and reclaim_first was False; caller
        # needs to decide whether to retry with reclaim.
        return False


def _verify_created_cards(
    conn: sqlite3.Connection, completing_task_id: str, claimed_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Partition ``claimed_ids`` into (verified, phantom). Verified = the row
    exists AND ``created_by`` is the completing task's assignee or id, OR the
    card is linked as its child (created elsewhere, attached by the worker).
    Never mutates."""
    ordered = list(dict.fromkeys(str(x).strip() for x in (claimed_ids or []) if str(x).strip()))
    if not ordered:
        return [], []

    row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (completing_task_id,)).fetchone()
    if row is None:
        # Completing task not found — nothing resolves.
        return [], ordered
    completing_assignee = row["assignee"]

    # Batch-fetch existence + created_by in one query.
    placeholders = ",".join(["?"] * len(ordered))
    rows = conn.execute(
        f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders})", tuple(ordered),
    ).fetchall()
    found = {r["id"]: r["created_by"] for r in rows}

    # Pull the set of cards linked as children of the completing task.
    # Cheap: one query, indexed on parent_id.
    linked_children: set[str] = set(child_ids(conn, completing_task_id))

    verified: list[str] = []
    phantom: list[str] = []
    for cid in ordered:
        created_by = found.get(cid)
        trusted = created_by is not None and (
            (completing_assignee and created_by == completing_assignee)
            or created_by == completing_task_id
            or cid in linked_children
        )
        (verified if trusted else phantom).append(cid)
    return verified, phantom


# Matches ``kanban_create`` (12 hex) and ``_new_task_id`` (8 hex) ids; 8+ for forward compat.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def _scan_prose_for_phantom_ids(conn: sqlite3.Connection, text: str) -> list[str]:
    """``t_<hex>`` references in ``text`` that don't resolve to a task (deduped; advisory)."""
    if not text:
        return []
    return _missing_task_ids(conn, dict.fromkeys(_TASK_ID_PROSE_RE.findall(text)))


class HallucinatedCardsError(ValueError):
    """``complete_task`` refused: ``created_cards`` has ids that don't exist or
    weren't created by this worker (``.phantom``). A ``ValueError`` so tool
    error handlers treat it as recoverable."""

    def __init__(self, phantom: list[str], completing_task_id: str):
        self.phantom = list(phantom)
        self.completing_task_id = completing_task_id
        super().__init__(
            f"completion blocked: claimed created_cards that do not exist "
            f"or were not created by this worker: {', '.join(phantom)}"
        )


class ArtifactPreservationError(RuntimeError):
    """Raised when a declared scratch deliverable cannot be preserved."""


def complete_task(
    conn: sqlite3.Connection, task_id: str, *, result: Optional[str] = None,
    summary: Optional[str] = None, metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None, expected_run_id: Optional[int] = None,
    fire_lifecycle_hook: bool = True,
) -> bool:
    """``running|ready|blocked|review -> done``; records ``result``.

    ``ready`` is accepted for manual CLI completion, ``review`` for human
    approval; with no active run the handoff fields survive via
    :func:`_synthesize_ended_run`. ``summary`` (defaults to ``result``) and
    ``metadata`` land on the closing run for :func:`build_worker_context`.
    ``created_cards`` are verified first — a phantom id raises
    :class:`HallucinatedCardsError` after an auditable event; afterwards the
    prose is scanned for unresolvable ``t_<hex>`` refs (advisory event only).
    """
    now = int(time.time())
    # Cheap pre-check; re-checked inside the txn to close the parent-reopen race.
    if not _parents_satisfied(conn, task_id):
        return False
    from hermes_cli.kanban_pr_acceptance_store import prepare_acceptance, record_acceptance
    verified_cards = _gate_created_cards(conn, task_id, created_cards, summary or result)
    metadata = _merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    handoff_summary = summary if summary is not None else result
    acceptance = prepare_acceptance(conn, task_id, expected_run_id, metadata)
    if acceptance is False:
        return False
    with write_txn(conn):
        # Hard invariant even for human review approval: a parent may have
        # reopened while this task waited.
        if not _parents_satisfied(conn, task_id):
            return False
        if acceptance is not None and not record_acceptance(conn, task_id, acceptance):
            return False
        prior_status = _task_status(conn, task_id)
        sql = """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked', 'review')
                """
        params: tuple = (result, now, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        if isinstance(metadata, dict):
            _stage_completion_artifacts(conn, task_id, metadata, now)
        run_id = _end_run(
            conn, task_id, outcome="completed", status="done", summary=handoff_summary,
            metadata=metadata,
        )
        # Never-claimed task: synthesize a run so the handoff fields survive.
        if run_id is None and (summary or metadata or result or prior_status == "review"):
            synth_summary, synth_metadata = handoff_summary, metadata
            if prior_status == "review" and not synth_summary and not synth_metadata:
                synth_summary = _REVIEW_APPROVED_NOTE
                synth_metadata = {"source_status": "review", "approval": "manual"}
            run_id = _synthesize_ended_run(
                conn, task_id, outcome="completed", summary=synth_summary, metadata=synth_metadata,
            )
        event_summary = handoff_summary
        if prior_status == "review" and not event_summary:
            event_summary = _REVIEW_APPROVED_NOTE
        _append_event(
            conn, task_id, "completed",
            _completed_event_payload(result, event_summary, verified_cards, metadata),
            run_id=run_id,
        )
    _flag_phantom_prose_refs(conn, task_id, run_id, summary, result, verified_cards)
    # Success wipes the breaker counter (history stays on the event log).
    _clear_failure_counter(conn, task_id)
    recompute_ready(conn)  # separate txn so children see ``done``
    _cleanup_workspace(conn, task_id)
    _done_task = get_task(conn, task_id)
    if fire_lifecycle_hook:
        _fire_task_hook("kanban_task_completed", _done_task, task_id, run_id, summary=handoff_summary)
    return True


_REVIEW_APPROVED_NOTE = "Review approved without additional evidence."


def _gate_created_cards(
    conn: sqlite3.Connection, task_id: str, created_cards: Optional[Iterable[str]], preview_text: Optional[str],
) -> list[str]:
    """Verify ``created_cards`` BEFORE the main write txn; returns the verified
    ids. A phantom id is recorded in its own tiny txn (auditable) then raised
    as :class:`HallucinatedCardsError` without touching task state."""
    if not created_cards:
        return []
    verified_cards, phantom_cards = _verify_created_cards(conn, task_id, created_cards)
    if phantom_cards:
        with write_txn(conn):
            _append_event(
                conn, task_id, "completion_blocked_hallucination",
                {
                    "phantom_cards": phantom_cards,
                    "verified_cards": verified_cards,
                    "summary_preview": _first_line(preview_text, 200) or None,
                },
            )
        raise HallucinatedCardsError(phantom_cards, task_id)
    return verified_cards


def _stage_completion_artifacts(conn: sqlite3.Connection, task_id: str, metadata: dict, now: int) -> None:
    """Copy scratch artifacts to the attachments dir and record each as an attachment row."""
    _persist_scratch_completion_artifacts(conn, task_id, metadata)
    for stored_path in metadata.pop("_staged_artifacts", []):
        path = Path(stored_path)
        _insert_completion_attachment(
            conn, task_id, filename=path.name, stored_path=str(path),
            size=path.stat().st_size, created_at=now,
        )


def _completed_event_payload(
    result: Optional[str], event_summary: Optional[str], verified_cards: list[str], metadata: Any,
) -> dict:
    """``completed`` event payload: first summary line (400 chars) so gateway
    notifiers / dashboard WS render without a second round-trip; verified
    cards; and ``metadata["artifacts"]`` promoted so the notifier can upload
    them as native attachments without fetching the run row."""
    # Mirror CLI's _show_voice_status: include STT/TTS provider availability so the user can tell at a
    # glance *why* voice mode isn't working ("STT provider: MISSING ..." is the common case). ``record_key``
    # mirrors the configured ``voice.record_key`` so the TUI can both bind it (frontend
    # ``isVoiceToggleKey``) and display it in /voice status — previously the TUI hardcoded Ctrl+B and
    # ignored the config (#18994).
    payload: dict = {
        "result_len": len(result) if result else 0,
        "summary": _first_line(event_summary, 400) or None,
    }
    if verified_cards:
        payload["verified_cards"] = verified_cards
    if isinstance(metadata, dict):
        md_artifacts = metadata.get("artifacts")
        if isinstance(md_artifacts, (list, tuple)):
            cleaned = [str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()]
            if cleaned:
                payload["artifacts"] = cleaned
    return payload


def _flag_phantom_prose_refs(
    conn: sqlite3.Connection, task_id: str, run_id: Optional[int],
    summary: Optional[str], result: Optional[str], verified_cards: list[str],
) -> None:
    """Advisory post-commit scan of summary+result for unresolvable ``t_<hex>``
    references; emits ``suspected_hallucinated_references`` in its own txn so
    the completion is already durable. Never blocks."""
    scan_text = " ".join(filter(None, [summary, result]))
    if not scan_text:
        return
    phantom_refs = [p for p in _scan_prose_for_phantom_ids(conn, scan_text) if p not in set(verified_cards)]
    if phantom_refs:
        with write_txn(conn):
            _append_event(
                conn, task_id, "suspected_hallucinated_references",
                {"phantom_refs": phantom_refs, "source": "completion_summary"}, run_id=run_id,
            )


def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: Optional[dict], *, summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Legacy workers named deliverables only by absolute path in prose; add
    those that exist under the scratch workspace to ``metadata["artifacts"]``
    before cleanup can erase them."""
    workspace = _scratch_workspace(conn, task_id)
    if workspace is None:
        return metadata
    if not _is_managed_scratch_path(workspace):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        if candidate.is_file():
            discovered.append(str(candidate))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated


def _persist_scratch_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict,
) -> None:
    """Copy scratch-workspace completion artifacts before cleanup removes them."""
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return

    workspace = _scratch_workspace(conn, task_id)
    if workspace is None:
        return
    is_managed, board = _managed_scratch_path_info(workspace)
    if not is_managed:
        return

    try:
        workspace_root = workspace.resolve()
    except OSError:
        return

    attachment_dir = task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    def _discard_copies() -> None:
        for copied in used_destinations:
            with contextlib.suppress(OSError):
                copied.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            attachment_dir.rmdir()

    for item in raw_artifacts:
        artifact = str(item).strip() if isinstance(item, str) else ""
        if not artifact:
            continue
        src = Path(artifact).expanduser()
        try:
            resolved_src = src.resolve()
        except OSError:
            persisted.append(artifact)
            continue

        if not resolved_src.is_relative_to(workspace_root):
            persisted.append(artifact)
            continue

        problem = None
        if not src.is_file():
            problem = f"declared scratch artifact is unavailable or not a regular file: {artifact}"
        elif resolved_src.stat().st_size > KANBAN_ATTACHMENT_MAX_BYTES:
            problem = (
                f"declared scratch artifact exceeds the "
                f"{KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
            )
        if problem:
            _discard_copies()
            raise ArtifactPreservationError(problem)

        dest: Optional[Path] = None
        try:
            attachment_dir.mkdir(parents=True, exist_ok=True)
            dest = _unique_attachment_path(attachment_dir, resolved_src.name, used_destinations)
            _copy_capped(resolved_src, dest, artifact)
        except Exception as exc:
            if dest is not None:
                with contextlib.suppress(OSError):
                    dest.unlink(missing_ok=True)
            _discard_copies()
            if isinstance(exc, ArtifactPreservationError):
                raise
            raise ArtifactPreservationError(
                f"could not preserve declared scratch artifact {artifact}: {exc}"
            ) from exc
        used_destinations.add(dest)
        persisted.append(str(dest.resolve()))
        changed = True

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = [
            path for path in persisted if path.startswith(str(attachment_dir.resolve()))
        ]


def _copy_capped(src: Path, dest: Path, artifact: str) -> None:
    """Chunked copy that aborts if the file grows past the attachment cap mid-copy."""
    with src.open("rb") as source_file, dest.open("xb") as destination_file:
        copied = 0
        while chunk := source_file.read(1024 * 1024):
            copied += len(chunk)
            if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                raise ArtifactPreservationError(
                    f"declared scratch artifact grew beyond the size limit: {artifact}"
                )
            destination_file.write(chunk)


def _insert_completion_attachment(
    conn: sqlite3.Connection, task_id: str, *, filename: str, stored_path: str, size: int,
    created_at: int,
) -> None:
    """Record a worker-produced artifact in the existing attachment table."""
    conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
        "VALUES (?, ?, ?, NULL, ?, 'kanban_complete', ?)",
        (task_id, filename, stored_path, size, created_at),
    )
    _append_event(conn, task_id, "attached", {"filename": filename, "size": size, "by": "kanban_complete"})


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return a non-conflicting path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    stem, suffix = Path(safe_name).stem or "artifact", Path(safe_name).suffix
    candidate = directory / safe_name
    idx = 1
    while candidate in used or candidate.exists():
        candidate = directory / f"{stem}_{idx}{suffix}"
        idx += 1
    return candidate


def edit_completed_task_result(
    conn: sqlite3.Connection, task_id: str, *, result: str, summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        if _task_status(conn, task_id) != "done":
            return False
        conn.execute("UPDATE tasks SET result = ? WHERE id = ?", (result, task_id))
        run = conn.execute(
            """
            SELECT id FROM task_runs
             WHERE task_id = ?
               AND outcome = 'completed'
             ORDER BY COALESCE(ended_at, started_at, 0) DESC, id DESC
             LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        if run is None:
            run_id = _synthesize_ended_run(
                conn, task_id, outcome="completed", summary=handoff_summary, metadata=metadata,
            )
        else:
            run_id = int(run["id"])
            conn.execute("UPDATE task_runs SET summary = ? WHERE id = ?", (handoff_summary, run_id))
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        _append_event(
            conn, task_id, "edited",
            {
                "fields": ["result", "summary"] + (["metadata"] if metadata is not None else []),
                "result_len": len(result) if result else 0,
                "summary": _first_line(handoff_summary, 400) or None,
            },
            run_id=run_id,
        )
    return True


def block_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    kind: Optional[str] = None, expected_run_id: Optional[int] = None,
) -> bool:
    """``running``/``ready`` -> ``blocked`` (or ``todo`` / ``triage``, see
    :func:`_route_block`). ``transient`` still counts toward the loop breaker
    so a forever-flaky task escalates. True on any transition."""
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None")
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        source_status = _retry_status_for_run(conn, task_id) if cur_row["status"] == "running" else "ready"
        new_status, event_kind, set_sql, params, payload = _route_block(
            kind, reason, source_status, prev_kind=_row_get(cur_row, "block_kind"),
            prev_recurrences=int(_row_get(cur_row, "block_recurrences") or 0),
        )
        sql = f"""
                UPDATE tasks
                   SET status        = '{new_status}',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       {set_sql}
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """
        params = (*params, task_id)
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params = (*params, int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="blocked", status="blocked", summary=reason, synthesize=bool(reason),
        )
        _append_event(conn, task_id, event_kind, payload, run_id=run_id)
        blocked_task = get_task(conn, task_id)
        if kind == "dependency":
            # Historical ordering: the dependency lane fires inside the txn.
            _fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
            return True
    _fire_task_hook("kanban_task_blocked", blocked_task, task_id, run_id, reason=reason)
    return True


def _route_block(
    kind: Optional[str], reason: Optional[str], source_status: str, *,
    prev_kind: Optional[str], prev_recurrences: int,
) -> tuple[str, str, str, tuple, dict]:
    """``(new_status, event_kind, set_sql, params, payload)`` for :func:`block_task`.

    ``dependency`` never enters the human ``blocked`` bucket: it waits in
    ``todo`` for ``recompute_ready``, so a cron never sees a dependency-wait
    as something to "unblock". Every other kind counts unblock-loop
    recurrences: block_task only fires from running/ready (AFTER an unblock
    returned the task to the pool), so a stored ``block_kind`` equal to the
    incoming one means blocked -> unblocked -> re-block for the same cause
    (un-typed None compares equal to a prior un-typed block). At
    ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
    """
    payload = {"reason": reason, "kind": kind, "source_status": source_status}
    if kind == "dependency":
        return "todo", "dependency_wait", "block_kind    = ?", (kind,), payload
    recurrences = prev_recurrences + 1 if prev_kind == kind else 1
    set_sql = "block_kind    = ?,\n                       block_recurrences = ?"
    payload = {"reason": reason, "kind": kind, "recurrences": recurrences, "source_status": source_status}
    if recurrences >= BLOCK_RECURRENCE_LIMIT:
        payload["limit"] = BLOCK_RECURRENCE_LIMIT
        return "triage", "block_loop_detected", set_sql, (kind, recurrences), payload
    return "blocked", "blocked", set_sql, (kind, recurrences), payload


def redact_review_value(value: Any) -> Any:
    """Redact secrets at the domain boundary for durable review handoffs."""
    if isinstance(value, str):
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    if isinstance(value, dict):
        return {key: redact_review_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_review_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_review_value(item) for item in value)
    return value


def request_review(
    conn: sqlite3.Connection, task_id: str, *, summary: Optional[str] = None,
    metadata: Optional[dict] = None, reviewer: Optional[str] = None,
    expected_run_id: Optional[int] = None, force: bool = False, with_reason: bool = False,
):
    """``running``/``ready`` -> ``review``; never touches block recurrence accounting.

    Implementer and reviewer are recorded on the event so requested changes
    route back to the right profile; ``reviewer`` reassigns the task, and on
    re-review defaults to the latest ``changes_requested`` provenance. A live
    claim is only cleared with proof of ownership (``expected_run_id``) or
    ``force=True``. Returns ``bool``, or ``(ok, reason)`` with ``with_reason``.
    """

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = redact_review_value(metadata)
    with write_txn(conn):
        if not _parents_satisfied(conn, task_id):
            return _ret(False, "parent dependencies are not satisfied")
        trow = conn.execute(
            "SELECT assignee, status, claim_lock, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if trow is None:
            return _ret(False, "task not found")
        # Refuse to clear a live worker's claim without proof of ownership
        # (expected_run_id) or an explicit human override (force=True).
        if (
            expected_run_id is None
            and not force
            and trow["status"] == "running"
            and trow["claim_lock"] is not None
        ):
            return _ret(
                False, "task is running under a live claim; pass expected_run_id "
                "(worker ownership) or force=True (explicit operator "
                "override) instead of clearing the live run's claim",
            )
        implementer = trow["assignee"]
        if reviewer is None:
            reviewer = _prior_reviewer(conn, task_id)
            if reviewer is False:
                return _ret(
                    False, "re-review has no durable reviewer provenance (the "
                    "latest changes_requested event is missing or "
                    "malformed); pass reviewer= explicitly",
                )
        reviewer = _canonical_assignee(reviewer)
        assignee_sql = ", assignee = ?" if reviewer is not None else ""
        run_guard = "" if expected_run_id is None else " AND current_run_id = ?"
        params: tuple[Any, ...] = (
            *(() if reviewer is None else (reviewer,)), task_id,
            *(() if expected_run_id is None else (int(expected_run_id),)),
        )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'review',
                   claim_lock    = NULL,
                   claim_expires = NULL,
                   worker_pid    = NULL
            """ + assignee_sql + """
             WHERE id = ?
               AND status IN ('running', 'ready')
            """ + run_guard,
            params,
        )
        if cur.rowcount != 1:
            return _ret(
                False, "task is not in running/ready (or expected_run_id did not match the current run)",
            )
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="review_requested", status="review",
            summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
        )
        _append_event(
            conn,
            task_id,
            "review_requested",
            {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
            },
            run_id=run_id,
        )
    return _ret(True)


def _prior_reviewer(conn: sqlite3.Connection, task_id: str):
    """Reviewer recorded by the latest ``changes_requested`` run's event.
    ``None`` = first review (no such run); ``False`` = a run exists but its
    provenance is missing/malformed."""
    changes_run = conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'changes_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if changes_run is None:
        return None
    changes_event = _latest_event(conn, task_id, "changes_requested", changes_run["id"])
    reviewer = _json_dict(_row_get(changes_event, "payload")).get("reviewer")
    return reviewer if isinstance(reviewer, str) and reviewer.strip() else False


def _nonblank_str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def request_changes(
    conn: sqlite3.Connection, task_id: str, *, reason: str, expected_run_id: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """Close an active reviewer run (claimed from ``review``) and hand the task
    back to the implementer from the latest ``review_requested`` event, parent
    gating reapplied. Returns ``(ok, implementer | reason)``."""
    reason = str(redact_review_value(reason or "")).strip()
    if not reason:
        return False, "reason is required"

    with write_txn(conn):
        task_row = conn.execute(
            "SELECT status, assignee, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task_row is None:
            return False, "task not found"
        current_run_id = task_row["current_run_id"]
        if task_row["status"] != "running" or current_run_id is None:
            return False, "task is not in an active review run"
        if expected_run_id is not None and int(current_run_id) != int(expected_run_id):
            return False, "run_id mismatch"

        claimed_event = _latest_event(conn, task_id, "claimed", current_run_id)
        claimed_payload = _json_dict(_row_get(claimed_event, "payload"))
        if claimed_payload.get("source_status") != "review":
            return False, "active run was not claimed from review"

        requested_event = _latest_event(conn, task_id, "review_requested")
        if requested_event is None:
            return False, "no prior review_requested event"
        implementer = _nonblank_str(_json_dict(requested_event["payload"]).get("implementer"))
        if implementer is None:
            return False, "review handoff has no valid implementer provenance"
        reviewer = _canonical_assignee(_nonblank_str(task_row["assignee"]))

        new_status = _landing_status_after_parents(conn, task_id)
        # consecutive_failures deliberately PRESERVED: a review transition is
        # not evidence the pathology cleared; only complete_task resets it.
        cur = conn.execute(
            """
            UPDATE tasks
               SET status = ?,
                   assignee = COALESCE(?, assignee),
                   claim_lock = NULL,
                   claim_expires = NULL,
                   worker_pid = NULL
             WHERE id = ? AND status = 'running' AND current_run_id = ?
            """,
            (new_status, implementer, task_id, int(current_run_id)),
        )
        if cur.rowcount != 1:
            return False, "task changed during review handoff"
        run_id = _end_run(
            conn, task_id, outcome="changes_requested", status=new_status, summary=reason,
        )
        _append_event(
            conn,
            task_id,
            "changes_requested",
            {
                "reason": reason,
                "implementer": implementer,
                "reviewer": reviewer,
                "status": new_status,
            },
            run_id=run_id,
        )
    return True, implementer


def promote_task(
    conn: sqlite3.Connection, task_id: str, *, actor: str, reason: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Operator promotion ``todo``/``blocked`` -> ``ready`` with an audit event.
    Refused while a parent is unfinished; ``dry_run`` only validates.
    Returns ``(ok, reason)``."""
    cur_status = _task_status(conn, task_id)
    if cur_status is None:
        return False, f"task {task_id} not found"

    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    # No override: claim_task demotes ready -> todo on an undone parent whichever
    # writer set 'ready', so a forced promotion would only report a success the
    # first claim silently reverts (#106195). The dependency itself is the knob.
    parents = conn.execute(
        "SELECT t.id, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?", (task_id,),
    ).fetchall()
    unsatisfied = [p["id"] for p in parents if p["status"] not in ("done", "archived")]
    if unsatisfied:
        return False, (
            f"unsatisfied parent dependencies: {', '.join(unsatisfied)} "
            f"(the ready -> running claim re-checks parents, so promotion cannot "
            f"bypass them; complete the parents or drop the link with "
            f"`hermes kanban unlink <parent_id> {task_id}`)"
        )

    if dry_run:
        return True, None

    with write_txn(conn):
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked')", (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(conn, task_id, "promoted_manual", {"actor": actor, "reason": reason})

    return True, None


def _reclaim_dangling_run(
    conn: sqlite3.Connection, task_id: str, *, statuses, now: int, note: str,
) -> None:
    """Close a leaked open run before a status flip so the invariant
    ``current_run_id IS NULL <=> run row terminal`` holds; no-op normally."""
    placeholders = ", ".join("?" for _ in statuses)
    stale = conn.execute(
        f"SELECT current_run_id FROM tasks WHERE id = ? AND status IN ({placeholders})",
        (task_id, *statuses),
    ).fetchone()
    if stale and stale["current_run_id"]:
        conn.execute(
            """
            UPDATE task_runs
               SET status = 'reclaimed', outcome = 'reclaimed',
                   summary = COALESCE(summary, ?),
                   ended_at = ?,
                   claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
             WHERE id = ? AND ended_at IS NULL
            """,
            (note, now, int(stale["current_run_id"])),
        )


def _landing_status_after_parents(conn: sqlite3.Connection, task_id: str) -> str:
    """``ready`` if every parent is terminal else ``todo`` — the re-gate shared by
    unblock/reopen so neither can spawn a child whose upstream is unfinished."""
    return "ready" if _parents_satisfied(conn, task_id) else "todo"


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``blocked``/``scheduled`` -> its resumable phase (parent re-gated; ``review``
    when that is where it left off), closing any leaked run first."""
    now = int(time.time())
    with write_txn(conn):
        resume_status = (
            _resume_status_from_events(conn, task_id)
            if _task_status(conn, task_id) == "blocked"
            else "ready"
        )
        _reclaim_dangling_run(
            conn, task_id, statuses=("blocked", "scheduled"), now=now,
            note="invariant recovery on unblock",
        )
        # Re-gate on parent completion before restoring the source phase.
        landing_status = _landing_status_after_parents(conn, task_id)
        new_status = (
            "review"
            if landing_status == "ready" and resume_status == "review"
            else landing_status
        )
        # ``block_kind``/``block_recurrences`` deliberately survive the unblock:
        # resetting them is the amnesia that let cron-unblock <-> re-block loop
        # unbounded; only complete_task clears them. ``consecutive_failures``
        # (the dispatcher's spawn/crash counter) IS reset — a deliberate unblock
        # is a fresh start for the retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')", (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            (
                {"status": new_status, "resume_status": resume_status}
                if new_status != "ready" or resume_status != "ready"
                else None
            ),
        )
        return True


def reopen_review_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """``review`` -> ``ready``/``todo`` so the implementer re-runs on the new
    comments; restores the implementer from the ``review_requested`` event.
    Preserves ``consecutive_failures`` and the block loop counter (review is
    not a block; only :func:`complete_task` clears them)."""
    now = int(time.time())
    with write_txn(conn):
        _reclaim_dangling_run(
            conn, task_id, statuses=("review",), now=now,
            note="invariant recovery on review reopen",
        )
        new_status = _landing_status_after_parents(conn, task_id)
        review_event = _latest_event(conn, task_id, "review_requested")
        handoff = _json_dict(_row_get(review_event, "payload"))
        implementer = _nonblank_str(handoff.get("implementer"))
        params: tuple[Any, ...] = (new_status, *((implementer,) if implementer else ()), task_id)
        cur = conn.execute(
            # consecutive_failures deliberately PRESERVED: review reopen is not
            # a success signal; only complete_task resets the breaker (#35072).
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            + (", assignee = ?" if implementer else "")
            + " WHERE id = ? AND status = 'review'",
            params,
        )
        if cur.rowcount != 1:
            return False
        payload: dict[str, Any] = {"status": new_status}
        if implementer:
            payload["implementer"] = implementer
        _append_event(
            conn, task_id, "review_reopened", payload if payload != {"status": "ready"} else None,
        )
        return True


def invalidate_descendants_for_parent_reopen(
    conn: sqlite3.Connection, task_id: str, *, author: str,
) -> dict[str, Any]:
    """THE done-reopen invalidation: every ``ready``/``review``/``running``/``done``
    descendant of a reopened ancestor is demoted to ``todo`` and re-gated.
    Every surface that reopens a done task (dashboard PATCH/drag) routes here.

    Composes under the caller's txn (``allow_nested=True``) so the flip and the
    retractions commit atomically. Each descendant gets a
    ``descendant_invalidated`` event, the legacy ``status`` event the live feed
    renders, and a comment naming the ancestor. Running descendants are closed
    ``reclaimed`` and their workers killed strictly post-commit (audit trail
    before death) — when composed, the CALLER must drain ``terminations``
    after its own commit. ``consecutive_failures`` resets (deliberate operator
    action), the opposite of :func:`reopen_review_task`.

    Returns ``{"invalidated": [{id, prior_status, new_status, resume_status}],
    "terminations": [(worker_pid, claim_lock)]}``.
    """
    caller_owns_txn = bool(conn.in_transaction)
    now = int(time.time())
    invalidated: list[dict[str, Any]] = []
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    with write_txn(conn, allow_nested=True):
        rows = conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
                SELECT child_id FROM task_links WHERE parent_id = ?
                UNION
                SELECT l.child_id
                FROM task_links l
                JOIN descendants d ON d.id = l.parent_id
            )
            SELECT t.id, t.status, t.current_run_id, t.worker_pid, t.claim_lock
            FROM descendants d
            JOIN tasks t ON t.id = d.id
            ORDER BY t.id
            """,
            (task_id,),
        ).fetchall()
        for row in rows:
            previous_status = row["status"]
            if previous_status not in {"ready", "review", "running", "done"}:
                continue
            resume_status = "ready"
            run_id = None
            if previous_status == "review":
                resume_status = "review"
            elif previous_status == "running":
                resume_status = _retry_status_for_run(conn, row["id"], row["current_run_id"])
                terminations.append((row["worker_pid"], row["claim_lock"]))
                run_id = _end_run(
                    conn, row["id"], outcome="reclaimed", status="todo",
                    summary=f"ancestor {task_id} reopened",
                )
            # consecutive_failures = 0: deliberate operator reset — see
            # docstring for why this diverges from reopen_review_task.
            conn.execute(
                "UPDATE tasks SET status = 'todo', completed_at = NULL, "
                "claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, "
                "current_run_id = NULL, consecutive_failures = 0 WHERE id = ?", (row["id"],),
            )
            entry = {
                "id": row["id"], "prior_status": previous_status,
                "new_status": "todo", "resume_status": resume_status,
            }
            _append_event(
                conn, row["id"], "descendant_invalidated",
                {"ancestor": task_id, **{k: v for k, v in entry.items() if k != "id"}},
                run_id=run_id,
            )
            # Legacy 'status' event so existing live-feed consumers still see
            # the move without learning the new event kind.
            _append_event(
                conn, row["id"], "status",
                {
                    "status": "todo", "reason": "ancestor_reopened", "parent": task_id,
                    "previous_status": previous_status, "resume_status": resume_status,
                },
                run_id=run_id,
            )
            _insert_comment(
                conn, row["id"], author, f"Invalidated: ancestor {task_id} was reopened; "
                f"retracted from '{previous_status}' to 'todo' "
                f"(will resume via '{resume_status}').", now,
            )
            invalidated.append(entry)
    if not caller_owns_txn:
        # Standalone: committed above, audit trail durable, safe to kill now.
        # Composed calls leave this to the caller post-commit.
        for pid, claim_lock in terminations:
            _terminate_reclaimed_worker(pid, claim_lock)
    return {"invalidated": invalidated, "terminations": terminations}


def specify_triage_task(
    conn: sqlite3.Connection, task_id: str, *, title: Optional[str] = None,
    body: Optional[str] = None, assignee: Optional[str] = None, author: Optional[str] = None,
) -> bool:
    """Update title/body/assignee (when given) and move ``triage -> todo`` in one
    txn; False when not in triage. Lands in ``todo`` (not ``ready``) so parent
    gating still applies; the audit comment is written only when a field changed.
    """
    if title is not None and not title.strip():
        raise ValueError("title cannot be blank")
    assignee = _canonical_assignee(assignee)
    with write_txn(conn):
        existing = conn.execute(
            "SELECT title, body, assignee FROM tasks WHERE id = ? AND status = 'triage'",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        sets: list[str] = ["status = 'todo'"]
        params: list[Any] = []
        changed_fields: list[str] = []
        if title is not None and title.strip() != (existing["title"] or ""):
            sets.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")
        if body is not None and (body or "") != (existing["body"] or ""):
            sets.append("body = ?")
            params.append(body)
            changed_fields.append("body")
        if assignee is not None and assignee != (existing["assignee"] or None):
            sets.append("assignee = ?")
            params.append(assignee)
            changed_fields.append("assignee")
        params.append(task_id)
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} "
            f"WHERE id = ? AND status = 'triage'", tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Not add_comment (own txn + 'commented' event); 'specified' below records it.
            _insert_comment(
                conn, task_id, author.strip(),
                "Specified — updated " + ", ".join(changed_fields) + " and promoted to todo.",
                int(time.time()),
            )
        _append_event(
            conn, task_id, "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Own IMMEDIATE txn (outside the one above): a parent-free specified task
    # flips to 'ready' now instead of idling until the next tick.
    recompute_ready(conn)
    return True


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status != 'archived'", (task_id,),
        )
        if cur.rowcount != 1:
            return False
        # Archived mid-run (dashboard): close the run so history isn't orphaned.
        run_id = _end_run(
            conn, task_id, outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # ``archived`` parents no longer block children; promote them now.
    recompute_ready(conn)
    # Reap the workspace on archive too (never-completed tasks kept it forever).
    _cleanup_workspace(conn, task_id)
    return True


def _delete_task_relations(conn: sqlite3.Connection, task_id: str) -> None:
    """Delete every row referencing ``task_id`` (schema has no ON DELETE CASCADE)."""
    conn.execute("DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id))
    for table in ("task_comments", "task_events", "task_runs", "kanban_notify_subs"):
        conn.execute(f"DELETE FROM {table} WHERE task_id = ?", (task_id,))


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete an ARCHIVED task (+ related rows); anything else must be
    archived first so data loss takes two deliberate actions."""
    with write_txn(conn):
        if _task_status(conn, task_id) != "archived":
            return False
        _delete_task_relations(conn, task_id)
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        return cur.rowcount == 1


def delete_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Hard-delete a task and its related rows in one txn; False when not found."""
    with write_txn(conn):
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount != 1:
            return False
        _delete_task_relations(conn, task_id)
    recompute_ready(conn)
    return True


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    try:
        return Path(out).expanduser().resolve()
    except Exception:
        return Path(out).expanduser()


def _git_branch_exists(repo_root: Path, branch_name: str) -> bool:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show-ref", "--verify", f"refs/heads/{branch_name}"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _git_common_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_dir(path: Path) -> Optional[Path]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-dir"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    return Path(out).expanduser().resolve(strict=False)


def _git_current_branch(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "branch", "--show-current"],
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def _is_linked_worktree_checkout(path: Path) -> bool:
    git_dir = _git_dir(path)
    common_dir = _git_common_dir(path)
    if git_dir is None or common_dir is None:
        return False
    return git_dir != common_dir


def _nearest_existing_path(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    current = _nearest_existing_path(path).resolve(strict=False)
    while True:
        repo_root = _git_toplevel(current)
        if repo_root is not None:
            return repo_root
        if current == current.parent:
            return None
        current = current.parent


def _bootstrap_worktree_environments(
    repo_root: Path,
    target: Path,
    *,
    require_python: bool = False,
    allow_venv_fallback: bool = False,
) -> None:
    """Link usable project-local environments into a child worktree."""
    from hermes_cli.worktree_environment import bootstrap_worktree_environments

    # Preserve the existing Kanban compatibility contract for non-Python
    # repositories and older environment directories. New Hermes agent
    # creation paths request the stricter executable check explicitly.
    bootstrap_worktree_environments(
        repo_root,
        target,
        require_python=require_python,
        allow_venv_fallback=allow_venv_fallback,
    )


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str) -> None:
    """Materialize ``target`` as a linked git worktree under ``repo_root``."""
    target = target.expanduser()
    repo_common = _git_common_dir(repo_root)
    if target.exists() and repo_common is not None:
        target_common = _git_common_dir(target)
        if target_common == repo_common:
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_branch_exists(repo_root, branch_name):
        cmd = ["git", "-C", str(repo_root), "worktree", "add", str(target), branch_name]
    else:
        cmd = [
            "git", "-C", str(repo_root), "worktree", "add", "-b", branch_name,
            str(target), "HEAD",
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True, encoding='utf-8', errors='replace',
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
        )
    _bootstrap_worktree_environments(
        repo_root,
        target,
        require_python=True,
        allow_venv_fallback=True,
    )


def _resolve_worktree_workspace(
    task: Task, *, board: Optional[str] = None
) -> tuple[Path, str]:
    """Resolve + materialize a linked git worktree for ``task``.

    When ``task.workspace_path`` is unset, the anchor is the board's
    ``default_workdir`` (a persistent project checkout). This keeps every
    worktree task under a meaningful, board-owned repo — ``<repo>/.worktrees/
    <task-id>`` — instead of silently landing under the dispatcher's current
    working directory (which is whatever directory the gateway happened to be
    launched from, e.g. the Hermes checkout). If no anchor is configured
    anywhere, we fail loudly rather than guess.
    """
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    if not task.workspace_path:
        # Anchor on the board's configured default_workdir, not Path.cwd().
        # The dispatcher's CWD is incidental (gateway launch dir) and using it
        # scatters worktrees under whatever repo the gateway started in.
        board_slug = board if board else get_current_board()
        board_default = (read_board_metadata(board_slug).get("default_workdir") or "").strip()
        if not board_default:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                f"and board {board_slug!r} has no default_workdir set. Set a board "
                "default workdir (a git repo) or create the task with "
                "--workspace worktree:<absolute-repo-path>."
            )
        anchor = Path(board_default).expanduser()
        if not anchor.is_absolute():
            raise ValueError(
                f"board {board_slug!r} default_workdir {board_default!r} is not "
                "absolute; use an absolute path to a git repo"
            )
        repo_root = _git_toplevel(anchor)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but board "
                f"{board_slug!r} default_workdir {board_default!r} is not inside a git repo"
            )
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path "
            f"{task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)

    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch == branch_name:
            repo_root = _repo_root_for_worktree_target(requested.parent)
            if repo_root is not None:
                _bootstrap_worktree_environments(
                    repo_root,
                    requested_resolved,
                    require_python=True,
                    allow_venv_fallback=True,
                )
            return requested_resolved, actual_branch
        # The requested path is an existing checkout of a DIFFERENT
        # task's branch. Decompose children inherit the root's
        # workspace_path verbatim, so siblings all point here; reusing
        # the checkout as-is would run this task on the other task's
        # branch — silent cross-task provenance corruption, and unsafe
        # when siblings run concurrently. Fall back to a fresh worktree
        # of our own under the same repo.
        fallback_root = _repo_root_for_worktree_target(requested.parent)
        if fallback_root is not None:
            fallback = fallback_root / ".worktrees" / task.id
            if fallback.resolve(strict=False) != requested_resolved:
                _ensure_git_worktree(fallback_root, fallback, branch_name)
                return fallback.resolve(strict=False), branch_name
        # No repo to anchor a fallback on (or the occupied path IS this
        # task's own canonical worktree): keep the legacy reuse rather
        # than failing dispatch.
        return requested_resolved, actual_branch or branch_name

    repo_root = _git_toplevel(requested)
    if repo_root is not None and requested_resolved == repo_root:
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name)
        return target, branch_name

    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo "
            "and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name)
    return requested, branch_name


def resolve_workspace(task: Task, *, board: Optional[str] = None) -> Path:
    """Resolve (and create if needed) the workspace for a task.

    - ``scratch``: a fresh dir under ``<board-root>/workspaces/<id>/``,
      where ``<board-root>`` is the active board's root. The path is the
      same for the dispatcher and every profile worker, so handoff is
      path-stable.
    - ``dir:<path>``: the path stored in ``workspace_path``.  Created
      if missing.  MUST be absolute — relative paths are rejected to
      prevent confused-deputy traversal where ``../../../tmp/attacker``
      resolves against the dispatcher's CWD instead of a meaningful
      root.  Users who want a kanban-root-relative workspace should
      compute the absolute path themselves.
    - ``worktree``: a real linked git worktree. If ``workspace_path`` names
      a repo root, Hermes treats it as an anchor and materializes a linked
      worktree at ``<repo>/.worktrees/<task-id>``. If ``workspace_path`` names
      a concrete target path, Hermes creates/reuses that linked worktree. With
      no ``workspace_path``, Hermes anchors on the board's ``default_workdir``
      and materializes ``<repo>/.worktrees/<task-id>`` per task; if no
      ``default_workdir`` is configured it raises rather than guessing from the
      dispatcher's CWD. When ``branch_name`` is empty, Hermes uses
      ``wt/<task-id>``.

    Persist the resolved path back to the task row via ``set_workspace_path``
    so subsequent runs reuse the same directory.
    """
    kind = task.workspace_kind or "scratch"
    if kind == "scratch":
        if task.workspace_path:
            # Legacy scratch tasks that were set to an explicit path get the
            # same absolute-path guard as dir: — consistent with the
            # threat model.
            p = Path(task.workspace_path).expanduser()
            if not p.is_absolute():
                raise ValueError(
                    f"task {task.id} has non-absolute workspace_path "
                    f"{task.workspace_path!r}; workspace paths must be absolute"
                )
        else:
            p = workspaces_root(board=board) / task.id
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "dir":
        if not task.workspace_path:
            raise ValueError(
                f"task {task.id} has workspace_kind=dir but no workspace_path"
            )
        p = Path(task.workspace_path).expanduser()
        if not p.is_absolute():
            raise ValueError(
                f"task {task.id} has non-absolute workspace_path "
                f"{task.workspace_path!r}; use an absolute path "
                f"(relative paths are ambiguous against the dispatcher's CWD)"
            )
        p.mkdir(parents=True, exist_ok=True)
        return p
    if kind == "worktree":
        p, _branch_name = _resolve_worktree_workspace(task, board=board)
        return p
    raise ValueError(f"unknown workspace_kind: {kind}")


def set_workspace_path(
    conn: sqlite3.Connection, task_id: str, path: Path | str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_path = ? WHERE id = ?",
            (str(path), task_id),
        )


def set_branch_name(
    conn: sqlite3.Connection, task_id: str, branch_name: str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET branch_name = ? WHERE id = ?",
            (str(branch_name), task_id),
        )


# ---------------------------------------------------------------------------
def schedule_task(
    conn: sqlite3.Connection, task_id: str, *, reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park in ``scheduled`` (waiting on time, not a human; not dispatchable)
    until ``unblock_task`` re-gates it."""
    with write_txn(conn):
        params: list[Any] = [task_id]
        sql = """
            UPDATE tasks
               SET status       = 'scheduled',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('todo', 'ready', 'running', 'blocked')
        """
        if expected_run_id is not None:
            sql += " AND current_run_id = ?"
            params.append(int(expected_run_id))
        if conn.execute(sql, params).rowcount != 1:
            return False
        run_id = _end_or_synthesize_run(
            conn, task_id, outcome="scheduled", status="scheduled", summary=reason, synthesize=bool(reason),
        )
        _append_event(conn, task_id, "scheduled", {"reason": reason}, run_id=run_id)
        return True


# Dispatcher (one-shot pass)
# ---------------------------------------------------------------------------

# After this many consecutive non-success attempts on a task/profile, the
# dispatcher stops retrying and parks the task in ``blocked`` with a reason so
# a human can investigate. Prevents retry storms when a worker repeatedly times
# out, crashes, or cannot spawn.
DEFAULT_FAILURE_LIMIT = 2
# Legacy alias — callers / tests still reference the old name.
DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT

# Max bytes to keep in a single worker log file. The dispatcher truncates
# and rotates on spawn if the file is larger than this at spawn time.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Keep a little wall-clock budget for the worker to observe a terminal timeout
# and call kanban_block/kanban_complete before max_runtime_seconds kills it.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|auth\w*|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Cooldown after a rate-limited (quota-wall) requeue before the dispatcher
# re-spawns the worker. Without this, a task released by the rate-limit path
# would be re-spawned on the very next tick and immediately bounce off the
# same quota wall, burning a worker slot every tick for hours. The cooldown
# spaces retries out so the board keeps cheaply probing whether quota is back
# without thrashing. Overridable via ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS``
# for operators who want a tighter/looser probe cadence.
DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 300  # 5 minutes

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

# Pattern matching a GitHub PR URL in task comments.
_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


@dataclass
class DispatchResult:
    """Outcome of a single ``dispatch`` pass."""

    reclaimed: int = 0
    promoted: int = 0
    reconciled_orphans: list[str] = field(default_factory=list)
    """Task ids requeued by :func:`reconcile_orphaned_running` this tick —
    ``running`` cards whose claim bookkeeping was broken (no valid claim,
    dead/gone worker). See the reconciliation pass for details."""
    spawned: list[tuple[str, str, str]] = field(default_factory=list)
    """List of ``(task_id, assignee, workspace_path)`` triples."""
    skipped_unassigned: list[str] = field(default_factory=list)
    """Ready task ids skipped because they have no assignee at all.
    Operator-actionable — usually a misfiled task waiting for routing."""
    auto_assigned_default: list[str] = field(default_factory=list)
    """Task ids that were unassigned in the DB and had
    ``kanban.default_assignee`` applied this tick before spawning (#27145).
    Surfaces the auto-assignment to telemetry / CLI / dashboard so the
    operator can see when the dispatcher is acting on the fallback rule
    rather than on explicit per-task assignments."""
    skipped_nonspawnable: list[str] = field(default_factory=list)
    """Ready task ids skipped because their assignee names a control-plane
    lane (a Claude Code terminal like ``orion-cc``) rather than a Hermes
    profile. Expected steady-state on multi-lane setups; NOT an
    operator-actionable failure. Tracked separately so health telemetry
    can distinguish "real stuck" (nothing spawned but spawnable work
    available) from "correctly idle" (nothing spawnable in the queue)."""
    skipped_per_profile_capped: list[tuple[str, str, int]] = field(default_factory=list)
    """Tasks deferred this tick because their assignee is already at
    ``kanban.max_in_progress_per_profile`` (#21582). Each entry is
    ``(task_id, assignee, current_running_count)``. NOT an
    operator-actionable failure — the task will be picked up on a
    subsequent tick when the assignee has capacity. Separate bucket so
    telemetry / dashboards can show "this profile is busy" vs
    "task is genuinely stuck"."""
    crashed: list[str] = field(default_factory=list)
    """Task ids reclaimed because their worker PID disappeared."""
    auto_blocked: list[str] = field(default_factory=list)
    """Task ids auto-blocked by the spawn-failure circuit breaker."""
    timed_out: list[str] = field(default_factory=list)
    """Task ids whose workers exceeded ``max_runtime_seconds``."""
    stale: list[str] = field(default_factory=list)
    """Task ids reclaimed because no progress (heartbeat) was seen
    within ``dispatch_stale_timeout_seconds``."""
    respawn_guarded: list[tuple[str, str]] = field(default_factory=list)
    """Tasks skipped by the respawn guard, as ``(task_id, reason)`` pairs.

    Reasons: ``"blocker_auth"`` (quota/auth error — also auto-blocked),
    ``"recent_success"`` (completed run within guard window),
    ``"active_pr"`` (GitHub PR URL in a recent comment)."""
    rate_limited: list[str] = field(default_factory=list)
    """Task ids whose workers bailed on a provider rate-limit / quota wall
    (EX_TEMPFAIL sentinel exit) and were released back to ``ready`` WITHOUT
    counting a failure. These never trip the circuit breaker — a long quota
    window just makes the task bounce cheaply until the window clears."""
    skipped_locked: bool = False
    """True when this tick was skipped because another process already held
    the board's dispatch lock (issue #35240). A losing dispatcher does no
    DB writes this tick — the lock holder is making progress on the same
    board. This is the steady-state signal that a single-writer guard is
    actively preventing two dispatchers from racing on ``kanban.db``."""
    memory_pressure: Optional[str] = None
    """System memory pressure observed at spawn time when the memory guard
    restricted this tick (OOF-30/OOF-77): ``"critical"`` — no new workers
    were spawned this tick; ``"elevated"`` — at most one new worker was
    spawned. ``None`` when memory was fine/unknown and the guard imposed
    no restriction. Reclaim/promotion bookkeeping still ran either way;
    deferred tasks stay queued for the next tick."""


# Bounded registry of recently-reaped worker child exits, populated by the
# reap loop at the top of ``dispatch_once`` and consulted by
# ``detect_crashed_workers`` to classify a dead-pid task.
#
# Entry: ``pid -> (raw_wait_status, reaped_at_epoch)``. We keep raw status
# so both ``os.WIFEXITED`` / ``os.WEXITSTATUS`` and ``os.WIFSIGNALED`` can
# be consulted. Entries are trimmed by age (and total size cap as a
# belt-and-braces against unbounded growth on exotic platforms).
_RECENT_WORKER_EXIT_TTL_SECONDS = 600
_RECENT_WORKER_EXITS_MAX = 4096
_recent_worker_exits: "dict[int, tuple[int, float]]" = {}


def _record_worker_exit(pid: int, raw_status: int) -> None:
    """Record a reaped child's exit status for later classification.

    Called from the reap loop in ``dispatch_once``. Safe to call many
    times; duplicate pids overwrite (pids can cycle, latest wins).
    """
    if not pid or pid <= 0:
        return
    now = time.time()
    _recent_worker_exits[int(pid)] = (int(raw_status), now)
    # Age-based trim: drop entries older than the TTL.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX // 2:
        cutoff = now - _RECENT_WORKER_EXIT_TTL_SECONDS
        for _pid in [p for p, (_s, t) in _recent_worker_exits.items() if t < cutoff]:
            _recent_worker_exits.pop(_pid, None)
    # Size cap as a final guard.
    if len(_recent_worker_exits) > _RECENT_WORKER_EXITS_MAX:
        # Drop oldest half.
        ordered = sorted(_recent_worker_exits.items(), key=lambda kv: kv[1][1])
        for _pid, _ in ordered[: len(ordered) // 2]:
            _recent_worker_exits.pop(_pid, None)


def _classify_worker_exit(pid: int) -> "tuple[str, Optional[int]]":
    """Classify a recently-reaped worker by pid.

    Returns ``(kind, code)`` where ``kind`` is one of:

    * ``"clean_exit"`` — ``WIFEXITED`` with ``WEXITSTATUS == 0``. When the
      task is still ``running`` in the DB, this is a protocol violation
      (worker exited without calling ``kanban_complete`` / ``kanban_block``)
      and should be auto-blocked immediately — retrying will just loop.
    * ``"rate_limited"`` — ``WIFEXITED`` with status
      ``KANBAN_RATE_LIMIT_EXIT_CODE``. The worker bailed because the
      provider rate-limited / exhausted quota, NOT because the task failed.
      ``detect_crashed_workers`` releases the task back to ``ready`` without
      counting a failure, so a long quota window can't trip the breaker.
    * ``"terminated_by_signal"`` — ``WIFEXITED`` with status
      ``KANBAN_SIGNAL_EXIT_CODE``. ``_signal_handler_q`` caught SIGINT /
      SIGTERM / SIGHUP and called ``os._exit()`` on purpose (issue #28181),
      which always reports ``WIFEXITED`` rather than ``WIFSIGNALED`` — so
      without this dedicated code a genuinely signal-killed worker was
      indistinguishable from ``clean_exit`` and got the misleading
      "exited cleanly without calling kanban_complete" protocol-violation
      diagnostic. Accounted as a real failure like ``nonzero_exit``, just
      with an honest error message about *why*.
    * ``"nonzero_exit"`` — ``WIFEXITED`` with non-zero status. Real error.
    * ``"signaled"`` — ``WIFSIGNALED`` (OOM killer, SIGKILL, etc). Real crash.
    * ``"unknown"`` — pid was not in the reap registry (either reaped by
      something else, or died between reap tick and liveness check). Fall
      back to existing crashed-counter behavior.

    ``code`` is the exit status (for ``clean_exit`` / ``rate_limited`` /
    ``terminated_by_signal`` / ``nonzero_exit``) or the signal number (for
    ``signaled``), or ``None`` for ``unknown``.
    """
    entry = _recent_worker_exits.get(int(pid))
    if entry is None:
        return ("unknown", None)
    raw, _ = entry
    try:
        if os.WIFEXITED(raw):
            code = os.WEXITSTATUS(raw)
            if code == 0:
                return ("clean_exit", 0)
            if code == KANBAN_RATE_LIMIT_EXIT_CODE:
                return ("rate_limited", code)
            if code == KANBAN_SIGNAL_EXIT_CODE:
                return ("terminated_by_signal", code)
            return ("nonzero_exit", code)
        if os.WIFSIGNALED(raw):
            return ("signaled", os.WTERMSIG(raw))
    except Exception:
        pass
    return ("unknown", None)


def reap_worker_zombies() -> "list[int]":
    """Reap all zombie children of this process without blocking.

    Returns the list of reaped PIDs. Safe to call when there are no
    children (returns []). No-op on Windows.
    """
    reaped: "list[int]" = []
    if os.name != "nt":
        try:
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid == 0:
                    break
                _record_worker_exit(pid, status)
                reaped.append(pid)
        except Exception:
            pass
    return reaped


def _pid_alive(pid: Optional[int]) -> bool:
    """Return True if ``pid`` is still running on this host.

    Cross-platform: uses ``OpenProcess`` + ``WaitForSingleObject`` on
    Windows (via ``gateway.status._pid_exists``) and ``os.kill(pid, 0)``
    on POSIX. Returns False for falsy PIDs or on any OS error.

    **DO NOT** use ``os.kill(pid, 0)`` directly on Windows — Python's
    Windows ``os.kill`` treats ``sig=0`` as ``CTRL_C_EVENT`` (bpo-14484)
    and will broadcast it to the target's console group, potentially
    killing unrelated processes.

    **Zombie handling:** the existence check succeeds against zombie
    processes (post-exit, pre-reap) because the process table entry
    still exists. A worker that exits without being reaped by its
    parent would stay "alive" to the dispatcher forever. Dispatcher
    workers are started via ``start_new_session=True`` + intentional
    Popen handle abandonment, so init reaps them quickly — but during
    the window between exit and reap, we'd otherwise see stale "alive"
    signals. On Linux we peek at ``/proc/<pid>/status`` and treat
    ``State: Z`` as dead. On macOS we ask ``ps`` for the BSD ``stat``
    field and treat values containing ``Z`` as dead.
    """
    if not pid or pid <= 0:
        return False
    from gateway.status import _pid_exists
    if not _pid_exists(int(pid)):
        return False
    # Still here → process exists. Check for zombie on platforms
    # where we have a cheap, deterministic process-state probe.
    if sys.platform == "linux":
        try:
            with open(f"/proc/{int(pid)}/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("State:"):
                        # "State:\tZ (zombie)" → dead
                        if "Z" in line.split(":", 1)[1]:
                            return False
                        break
        except (FileNotFoundError, PermissionError, OSError):
            # proc entry gone → already reaped; treat as dead.
            # PermissionError shouldn't happen for our own children but
            # be defensive.
            pass
    elif sys.platform == "darwin":
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(int(pid))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True, encoding='utf-8', errors='replace',
                timeout=1,
                check=False,
            )
            if proc.returncode != 0:
                return False
            if "Z" in (proc.stdout or "").strip():
                return False
        except (OSError, subprocess.SubprocessError, TimeoutError):
            # If the secondary probe fails, keep the kill(0) answer.
            pass
    return True


def _terminate_reclaimed_worker(
    pid: Optional[int],
    claim_lock: Optional[str],
    *,
    signal_fn=None,
) -> dict[str, Any]:
    """Best-effort host-local worker termination for reclaim paths."""
    import signal

    info: dict[str, Any] = {
        "prev_pid": int(pid) if pid else None,
        "host_local": False,
        "termination_attempted": False,
        "terminated": False,
        "sigkill": False,
    }
    if not pid or pid <= 0 or not claim_lock:
        return info

    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    if not str(claim_lock).startswith(host_prefix):
        return info
    info["host_local"] = True

    kill = signal_fn if signal_fn is not None else (
        os.kill if hasattr(os, "kill") else None
    )
    if kill is None:
        return info

    info["termination_attempted"] = True
    try:
        kill(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        # Process is already gone — that's a successful termination, not a
        # survival. Leaving terminated=False here would make the reclaim guard
        # misread a dead worker as still-alive and defer forever.
        info["terminated"] = True
        return info
    except OSError:
        return info

    for _ in range(10):
        if not _pid_alive(pid):
            info["terminated"] = True
            return info
        time.sleep(0.5)

    if _pid_alive(pid):
        try:
            # signal.SIGKILL doesn't exist on Windows; fall back to SIGTERM
            # (which maps to TerminateProcess via the stdlib shim).
            _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
            kill(int(pid), _sigkill)
            info["sigkill"] = True
        except (ProcessLookupError, OSError):
            return info

    info["terminated"] = not _pid_alive(pid)
    return info


def _worker_survived_termination(termination: dict) -> bool:
    """True when we tried to kill our own host-local worker and it is still alive.

    Reclaiming in this state would release the claim and let the dispatcher
    spawn a second worker while the first is still running — the duplication
    loop. Only host-local workers we actually signalled count: a non-local
    claim lock or a no-op attempt (no ``os.kill`` available) must fall through
    to the normal release path, since we cannot manage that worker anyway.
    """
    return bool(
        termination.get("termination_attempted")
        and termination.get("host_local")
        and not termination.get("terminated")
    )


def _defer_reclaim_for_live_worker(
    conn: sqlite3.Connection,
    task_id: str,
    claim_lock: Optional[str],
    now: int,
    termination: dict,
    *,
    reason: str,
) -> None:
    """Hold a claim whose worker survived termination instead of releasing it.

    Extends ``claim_expires`` by ``RECLAIM_DEFER_GRACE_SECONDS`` so the task
    stays ``running`` (no duplicate spawn) and records a ``reclaim_deferred``
    event so the hold is visible in ``hermes kanban tail``. The next dispatch
    tick retries the kill; this is self-correcting because not spawning a
    duplicate is what lets the throttled worker finally die.
    """
    grace = now + RECLAIM_DEFER_GRACE_SECONDS
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock IS ?",
            (grace, task_id, claim_lock),
        )
        if cur.rowcount != 1:
            return
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                (grace, run_id),
            )
        payload = {
            "reason": reason,
            "claim_lock": claim_lock,
            "claim_expires_now": grace,
        }
        payload.update(termination)
        _append_event(conn, task_id, "reclaim_deferred", payload, run_id=run_id)


def heartbeat_worker(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    note: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Record a ``heartbeat`` event + touch ``last_heartbeat_at``.

    Called by long-running workers as a liveness signal orthogonal to
    the PID check. A worker that forks a long-lived child (train loop,
    video encode, web crawl) can have its Python still alive while the
    actual work process is stuck; periodic heartbeats catch that.

    Returns True on success, False if the task is not in a state that
    should be heartbeating (not running, or claim expired).
    """
    now = int(time.time())
    with write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running'",
                (now, task_id),
            )
        else:
            cur = conn.execute(
                "UPDATE tasks SET last_heartbeat_at = ? "
                "WHERE id = ? AND status = 'running' AND current_run_id = ?",
                (now, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _current_run_id(conn, task_id)
        )
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?",
                (now, run_id),
            )
        _append_event(
            conn, task_id, "heartbeat",
            {"note": note} if note else None,
            run_id=run_id,
        )
    return True


def enforce_max_runtime(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> list[str]:
    """Terminate workers whose per-task ``max_runtime_seconds`` has elapsed.

    Sends SIGTERM, waits a short grace window, then SIGKILL. Emits a
    ``timed_out`` event and restores the task's source phase so the next
    dispatcher tick re-spawns the same kind of worker — unless the circuit
    breaker has already given up, in which case the task stays blocked
    where ``_record_spawn_failure`` parked it.

    Runs host-local: only tasks claimed by this host are candidates
    (same reasoning as ``detect_crashed_workers``). ``signal_fn`` is a
    test hook; defaults to ``os.kill`` on POSIX.
    """
    import signal
    timed_out: list[str] = []
    now = int(time.time())
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at, "
        "       t.max_runtime_seconds, t.claim_lock "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running' AND t.max_runtime_seconds IS NOT NULL "
        "  AND COALESCE(r.started_at, t.started_at) IS NOT NULL "
        "  AND t.worker_pid IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        # Runtime is per attempt, not lifetime-of-task. ``tasks.started_at``
        # intentionally records the first time a task ever started, so retries
        # must be measured from the active task_runs row when present.
        elapsed = now - int(row["active_started_at"])
        if elapsed < int(row["max_runtime_seconds"]):
            continue

        pid = int(row["worker_pid"])
        tid = row["id"]
        # SIGTERM then SIGKILL. Keep it simple: 5 s grace. Workers that
        # want a cleaner shutdown can install their own SIGTERM handler
        # before the grace expires.
        killed = False
        kill = signal_fn if signal_fn is not None else (
            os.kill if hasattr(os, "kill") else None
        )
        if kill is not None:
            try:
                kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
            # Short polling wait — no time.sleep on the write txn.
            for _ in range(10):
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            if _pid_alive(pid):
                try:
                    # signal.SIGKILL doesn't exist on Windows.
                    _sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
                    kill(pid, _sigkill)
                    killed = True
                except (ProcessLookupError, OSError):
                    pass

        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(row["max_runtime_seconds"]),
                    "sigkill": killed,
                    "retry_status": retry_status,
                }
                run_id = _end_run(
                    conn, tid,
                    outcome="timed_out", status="timed_out",
                    error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                    metadata=payload,
                )
                _append_event(
                    conn, tid, "timed_out", payload, run_id=run_id,
                )
                timed_out.append(tid)
        # Increment the unified failure counter. Outside the write_txn
        # above because ``_record_task_failure`` opens its own. If the
        # breaker trips, this flips the retried task to ``blocked`` and
        # emits a ``gave_up`` event on top of the ``timed_out`` we
        # already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={
                    "pid": pid,
                    "sigkill": killed,
                    "retry_status": retry_status,
                },
            )
    return timed_out


# Heartbeat staleness heartbeat gap — if a running task hasn't sent a
# heartbeat in this many seconds it's considered inactive regardless of
# the ``dispatch_stale_timeout_seconds`` threshold.  Hardcoded at 1 hour
# to match the original spec (">4h started + no commits in 1h").
_STALE_HEARTBEAT_GAP_SECONDS = 3600


def detect_stale_running(
    conn: sqlite3.Connection,
    *,
    stale_timeout_seconds: int = 0,
    signal_fn=None,
) -> list[str]:
    """Reclaim ``running`` tasks that show no progress (heartbeat) within the
    staleness window.

    A task is considered stale when BOTH of these hold:

    1. It has been running for longer than ``stale_timeout_seconds``
       (measured from the active run's ``started_at``, falling back to
       ``tasks.started_at`` on older runs).
    2. Its ``last_heartbeat_at`` is older than
       ``_STALE_HEARTBEAT_GAP_SECONDS`` (or NULL — never sent a heartbeat).

    On reclaim the task is restored to its source phase, the run is closed with
    ``outcome='stale'``, and the host-local worker (if still running) is
    terminated.

    Only considers ``status='running'`` tasks. Blocked tasks are never
    candidates.  Returns the list of reclaimed task IDs.

    ``stale_timeout_seconds=0`` disables the check entirely (returns ``[]``
    immediately).  ``signal_fn`` is a test hook; defaults to ``os.kill``
    on POSIX.
    """
    if stale_timeout_seconds <= 0:
        return []


    now = int(time.time())
    reclaimed: list[str] = []

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.last_heartbeat_at, t.claim_lock, "
        "       COALESCE(r.started_at, t.started_at) AS active_started_at "
        "FROM tasks t "
        "LEFT JOIN task_runs r ON r.id = t.current_run_id "
        "WHERE t.status = 'running'"
    ).fetchall()

    for row in rows:
        # Skip if no started_at (shouldn't happen for running, but be safe).
        if row["active_started_at"] is None:
            continue

        elapsed = now - int(row["active_started_at"])
        if elapsed < stale_timeout_seconds:
            continue  # not old enough to check

        last_hb = row["last_heartbeat_at"]
        hb_age = (now - int(last_hb)) if last_hb is not None else None
        if hb_age is not None and hb_age < _STALE_HEARTBEAT_GAP_SECONDS:
            continue  # recent heartbeat → still alive

        pid = row["worker_pid"]
        tid = row["id"]
        lock = row["claim_lock"] or ""

        # Terminate the worker if it's still host-local.
        termination = _terminate_reclaimed_worker(
            pid, lock, signal_fn=signal_fn,
        )

        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, tid, lock, now, termination,
                reason="heartbeat_stale_worker_alive",
            )
            continue

        with write_txn(conn):
            retry_status = _retry_status_for_run(conn, tid)
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (retry_status, tid, row["claim_lock"]),
            )
            if cur.rowcount != 1:
                continue

            payload = {
                "elapsed_seconds": int(elapsed),
                "last_heartbeat_at": (
                    int(last_hb) if last_hb is not None else None
                ),
                "heartbeat_age_seconds": (
                    int(hb_age) if hb_age is not None else None
                ),
                "timeout_seconds": stale_timeout_seconds,
                "pid": int(pid) if pid else None,
                "retry_status": retry_status,
            }
            payload.update(termination)

            run_id = _end_run(
                conn, tid,
                outcome="stale", status="stale",
                error=(
                    f"no heartbeat for {int(hb_age)}s "
                    if hb_age is not None
                    else "no heartbeat ever"
                ) + f" after {int(elapsed)}s running",
                metadata=payload,
            )
            _append_event(
                conn, tid, "stale", payload, run_id=run_id,
            )
            reclaimed.append(tid)

        # Intentionally NOT calling _record_task_failure here. Stale reclaim
        # is dispatcher-side detection of an absent heartbeat; the task is
        # going straight back to its source phase for re-dispatch. Counting it as
        # a worker failure would let two legitimately-long-running tasks
        # (>4h without explicit heartbeat) trip the circuit breaker and
        # auto-block, even though no worker actually failed. The 'stale'
        # event already lives in task_events for auditability; that's the
        # right surface for "this happened" without conflating with the
        # spawn_failed / timed_out / crashed counters.

    return reclaimed


def reconcile_orphaned_running(
    conn: sqlite3.Connection,
) -> list[str]:
    """Reconcile ``running`` cards whose claim bookkeeping is broken.

    Tracked-state vs. reality divergence: a task can sit in
    ``status='running'`` with ``claim_lock IS NULL`` or ``claim_expires IS
    NULL`` (crash mid-claim, manual SQL, DB restore). None of the other
    recovery paths ever touch such a card — ``release_stale_claims``
    requires a non-NULL ``claim_expires``, ``detect_crashed_workers``
    requires a host-local claim_lock + worker_pid, and
    ``detect_stale_running`` is disabled by default — so the card shows
    Running forever (a zombie).

    This pass finds those orphans, requeues them to ``ready`` with an
    explanatory comment, closes any leaked run, and appends a
    ``reconciled`` event. If the orphan row still records a live PID on
    this host, requeueing is deferred to a later tick so we never spawn a
    duplicate beside a possibly-alive worker.

    Returns the list of reconciled task ids. Safe to call every tick.

    Idea from openai/symphony's tracker reconciliation (Apache-2.0).
    """
    now = int(time.time())
    reconciled: list[str] = []
    rows = conn.execute(
        "SELECT id, claim_lock, claim_expires, worker_pid FROM tasks "
        "WHERE status = 'running' "
        "  AND (claim_lock IS NULL OR claim_expires IS NULL)"
    ).fetchall()
    for row in rows:
        tid = row["id"]
        pid = row["worker_pid"]
        if pid and _pid_alive(pid):
            # The recorded worker may still be doing real work — never
            # requeue beside a live process. Retry next tick.
            _log.debug(
                "kanban reconcile: task %s has broken claim bookkeeping but "
                "pid %s is alive on this host — deferring", tid, pid,
            )
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ? AND claim_expires IS ?",
                (tid, row["claim_lock"], row["claim_expires"]),
            )
            if cur.rowcount != 1:
                continue
            payload = {
                "reason": "orphaned_running",
                "claim_lock": row["claim_lock"],
                "claim_expires": (
                    int(row["claim_expires"])
                    if row["claim_expires"] is not None else None
                ),
                "worker_pid": int(pid) if pid else None,
                "now": now,
            }
            run_id = _end_run(
                conn, tid,
                outcome="reclaimed", status="reclaimed",
                error="orphaned running card (broken claim bookkeeping)",
                metadata=payload,
            )
            # Inline comment INSERT — add_comment opens its own write_txn
            # and would raise on nesting (see write_txn pitfalls).
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    tid, "dispatcher",
                    "reconciliation: card was 'running' with no valid claim "
                    "(dead/gone worker) — requeued to ready",
                    now,
                ),
            )
            _append_event(conn, tid, "reconciled", payload, run_id=run_id)
            reconciled.append(tid)
        _log.info(
            "kanban reconcile: requeued orphaned running task %s "
            "(claim_lock=%r, worker_pid=%r)", tid, row["claim_lock"], pid,
        )
    return reconciled


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message for grouping identical failures.

    Strips host-specific details (PIDs, timestamps) so that errors
    with the same root cause produce the same fingerprint.
    """
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


# Empirically ~96% of "clean exit without a terminal tool call" tasks complete
# on a later run (a goal-mode finalize nudge, or the model simply emitting the
# tool call next time), so a protocol violation is NOT deterministic — give it a
# bounded retry before the breaker trips instead of blocking on the first hit.
#
# The budget is a violation-only STREAK, not a share of the unified
# ``consecutive_failures`` counter: it counts consecutive clean-exit protocol
# violations (derived from run history by ``_protocol_violation_streak``), so
# earlier timeouts / nonzero exits neither consume nor extend it, and a
# below-budget violation does not tick the unified counter either. A per-task
# ``max_retries`` overrides this bound — the same "task override wins"
# precedence ``_record_task_failure`` documents for every other failure kind.
_PROTOCOL_VIOLATION_FAILURE_LIMIT = 3

# How far back to walk a task's closed runs when counting the violation
# streak. The streak trips at a handful of violations, so anything beyond a
# few dozen rows (violations interleaved with neutral rate-limited requeues)
# can only mean "way past the bound" anyway.
_PROTOCOL_VIOLATION_SCAN_LIMIT = 50


def _protocol_violation_streak(conn: sqlite3.Connection, task_id: str) -> int:
    """Count the task's trailing run of clean-exit protocol violations.

    Walks the task's closed runs newest-first — including the violation run
    ``detect_crashed_workers`` just closed — and counts how many in a row were
    clean-exit protocol violations:

    * ``rate_limited`` runs are neutral and skipped: a quota wall says nothing
      about the task, exactly as it is neutral for the unified
      ``consecutive_failures`` counter.
    * Any other closed run (completed, plain crash, timeout, spawn failure,
      reclaim, …) breaks the streak, so the bounded retry budget counts ONLY
      protocol violations — mixed failure kinds can neither consume nor
      extend it.

    Violation runs are recognized by the ``protocol_violation`` marker that
    ``detect_crashed_workers`` stamps into the run metadata; the violation
    error text is matched as a fallback for runs recorded before the marker
    existed.
    """
    streak = 0
    rows = conn.execute(
        "SELECT outcome, error, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (task_id, _PROTOCOL_VIOLATION_SCAN_LIMIT),
    ).fetchall()
    for row in rows:
        outcome = row["outcome"] or ""
        if outcome == "rate_limited":
            continue
        if outcome == "crashed":
            is_violation = False
            raw_meta = row["metadata"]
            if raw_meta:
                try:
                    is_violation = bool(
                        json.loads(raw_meta).get("protocol_violation")
                    )
                except (ValueError, TypeError):
                    is_violation = False
            if not is_violation:
                is_violation = "protocol violation" in (row["error"] or "")
            if is_violation:
                streak += 1
                continue
        break
    return streak


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Appends a ``crashed`` event and restores the task's source phase.
    Different from ``release_stale_claims``: this checks liveness
    immediately rather than waiting for the claim TTL.

    Only considers tasks claimed by *this host* — PIDs from other hosts
    are meaningless here. The host-local check is enough because
    ``_default_spawn`` always runs the worker on the same host as the
    dispatcher (the whole design is single-host).

    When the reap registry shows the worker exited cleanly (rc=0) but
    the task was still ``running`` in the DB, treat it as a protocol
    violation (worker answered conversationally without calling
    ``kanban_complete`` / ``kanban_block``) and trip the circuit breaker
    on the first occurrence — retrying a worker whose CLI keeps
    returning 0 without a terminal transition just loops forever.

    When the reap registry shows the worker exited with the rate-limit
    sentinel (``KANBAN_RATE_LIMIT_EXIT_CODE``), the worker bailed on a
    provider quota wall, NOT a task failure. Such tasks are released back
    to its source phase WITHOUT counting a failure (so a long quota window can't
    trip the breaker) and stamped with a quota-blocker error so
    ``check_respawn_guard`` defers their respawn until the window clears.
    The ids are returned via the ``_last_rate_limited`` function attribute
    (the public return stays the crashed-only ``list[str]``).
    """
    crashed: list[str] = []
    rate_limited: list[str] = []
    # Per-crash details collected inside the main txn, used after it
    # closes to run ``_record_task_failure`` (which needs its own
    # write_txn so can't nest). ``protocol_violation`` flags the
    # clean-exit-but-still-running case, which is accounted against its
    # own bounded violation streak instead of the unified failure
    # counter (see the post-txn loop below).
    crash_details: list[tuple[str, int, str, bool, str]] = []
    # (task_id, pid, claimer, protocol_violation, error_text)
    # Worker-exit observer payloads (RFC #58548), collected inside the main
    # txn and fired only after every reclaim/accounting txn has committed.
    exited_hook_payloads: list[dict] = []
    with write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, claim_lock, started_at, assignee "
            "FROM tasks "
            "WHERE status = 'running' AND worker_pid IS NOT NULL"
        ).fetchall()
        host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
        for row in rows:
            # Only check liveness for claims owned by this host.
            lock = row["claim_lock"] or ""
            if not lock.startswith(host_prefix):
                continue
            # Skip liveness check inside the launch-window grace period
            # so a freshly-spawned worker isn't reclaimed before its PID
            # is visible on /proc.
            started_at = row["started_at"] if "started_at" in row.keys() else None
            if started_at is not None:
                grace = _resolve_crash_grace_seconds()
                if time.time() - started_at < grace:
                    continue
            if _pid_alive(row["worker_pid"]):
                continue

            pid = int(row["worker_pid"])
            kind, code = _classify_worker_exit(pid)
            rate_limited_exit = False
            if kind == "clean_exit":
                # Worker subprocess returned 0 but its task is still
                # ``running`` in the DB — it exited without calling
                # ``kanban_complete`` / ``kanban_block``. Overwhelmingly the
                # work itself succeeded and only the paperwork was skipped, so
                # a retry usually completes; the corrective sentence below is
                # surfaced to the retry worker via the prior-attempt error in
                # ``build_worker_context`` (guidance approach from #61817).
                protocol_violation = True
                error_text = (
                    "worker exited cleanly (rc=0) without calling "
                    "kanban_complete or kanban_block — protocol violation. "
                    "If the prior run already did the work, verify it and "
                    "report the result via kanban_complete; a run that ends "
                    "without a terminal kanban call counts as failed no "
                    "matter what it did."
                )
                event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                    # Durable marker for _protocol_violation_streak: _end_run
                    # copies this payload into the run metadata, which is how
                    # the violation-only retry budget is derived later.
                    "protocol_violation": True,
                }
            elif kind == "rate_limited":
                # Worker bailed because the provider rate-limited / exhausted
                # quota (EX_TEMPFAIL sentinel). This is NOT a task failure —
                # the task is fine, the account just hit a wall. Release it
                # back to its source phase so the respawn guard defers it until the
                # quota window clears, and crucially do NOT count a failure
                # (skip ``_record_task_failure``) so a long quota window can't
                # trip the circuit breaker and permanently block the card.
                protocol_violation = False
                rate_limited_exit = True
                error_text = (
                    f"pid {pid} exited rate-limited (quota wall) — "
                    f"requeued without counting a failure"
                )
                event_kind = "rate_limited"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            else:
                protocol_violation = False
                if kind == "nonzero_exit":
                    error_text = f"pid {pid} exited with code {code}"
                elif kind == "signaled":
                    error_text = f"pid {pid} killed by signal {code}"
                elif kind == "terminated_by_signal":
                    error_text = (
                        f"pid {pid} caught a termination signal "
                        f"(SIGINT/SIGTERM/SIGHUP) and exited fast on purpose "
                        f"before it could call kanban_complete or "
                        f"kanban_block (see _signal_handler_q, #28181). Not a "
                        f"protocol violation — something outside the worker "
                        f"(dispatcher requeue, gateway restart, OS/OOM) sent "
                        f"it a termination signal."
                    )
                else:
                    error_text = f"pid {pid} not alive"
                event_kind = "crashed"
                event_payload = {"pid": pid, "claimer": row["claim_lock"]}
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            retry_status = _retry_status_for_run(conn, row["id"])
            event_payload["retry_status"] = retry_status
            cur = conn.execute(
                "UPDATE tasks SET status = ?, claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (retry_status, row["id"], pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                # Rate-limited requeues are a clean release, not a crash —
                # record the run outcome as ``rate_limited`` so the board
                # history doesn't show a phantom crash for a quota wall.
                _run_outcome = "rate_limited" if rate_limited_exit else "crashed"
                run_id = _end_run(
                    conn, row["id"],
                    outcome=_run_outcome, status=_run_outcome,
                    error=error_text,
                    metadata=dict(event_payload),
                )
                _append_event(
                    conn, row["id"], event_kind,
                    event_payload,
                    run_id=run_id,
                )
                exited_hook_payloads.append({
                    "task_id": row["id"],
                    "assignee": row["assignee"],
                    "run_id": run_id,
                    "worker_pid": pid,
                    "exit_kind": kind,
                    "exit_code": code,
                    "outcome": _run_outcome,
                    "retry_status": retry_status,
                })
                if rate_limited_exit:
                    # Stamp the failure-error column so ``check_respawn_guard``
                    # recognizes this as a quota blocker and defers the
                    # respawn until the window clears — WITHOUT touching
                    # ``consecutive_failures`` (that's the whole point: no
                    # breaker trip on a throttle).
                    conn.execute(
                        "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
                        (error_text[:500], row["id"]),
                    )
                    rate_limited.append(row["id"])
                else:
                    if protocol_violation:
                        # Stamp the failure error now: a below-budget
                        # violation never reaches ``_record_task_failure``
                        # (which stamps this column for every other failure
                        # kind), yet the board UI and the retry worker's
                        # context still need the violation message + the
                        # corrective guidance it carries.
                        conn.execute(
                            "UPDATE tasks SET last_failure_error = ? "
                            "WHERE id = ?",
                            (error_text[:500], row["id"]),
                        )
                    crashed.append(row["id"])
                    crash_details.append(
                        (row["id"], pid, row["claim_lock"],
                         protocol_violation, error_text)
                    )
    # Outside the main txn: account each crashed task and maybe trip the
    # breaker (the retried task transitions to blocked with a ``gave_up`` event
    # on top of the event we already emitted).
    #
    # Protocol-violation crashes (clean exit, no terminal tool call) get a
    # BOUNDED retry, not an immediate trip: empirically ~96% of these tasks
    # complete on a later run (a goal-mode finalize nudge, or the model simply
    # emitting kanban_complete/kanban_block next time), so blocking on the first
    # occurrence just churned them through the respawn cycle. The retry budget
    # is a violation-only streak (``_protocol_violation_streak``): earlier
    # timeouts / nonzero exits neither consume nor extend it, and a
    # below-budget violation does not tick the unified
    # ``consecutive_failures`` counter, so the two budgets stay independent.
    # A per-task ``max_retries`` overrides the violation bound with the same
    # top precedence it has for every other failure kind. Systemic same-error
    # crashes still trip immediately.
    auto_blocked: list[str] = []
    if crash_details:
        # Fingerprint errors to detect systemic failures.
        _fp_counts: dict[str, int] = {}
        for _, _, _, _, err_text in crash_details:
            fp = _error_fingerprint(err_text)
            _fp_counts[fp] = _fp_counts.get(fp, 0) + 1
        for tid, pid, claimer, protocol_violation, error_text in crash_details:
            if protocol_violation:
                streak = _protocol_violation_streak(conn, tid)
                trow = conn.execute(
                    "SELECT max_retries FROM tasks WHERE id = ?", (tid,),
                ).fetchone()
                if trow is None:
                    continue  # task deleted mid-loop
                task_override = (
                    trow["max_retries"] if "max_retries" in trow.keys() else None
                )
                violation_limit = (
                    int(task_override)
                    if task_override is not None
                    else _PROTOCOL_VIOLATION_FAILURE_LIMIT
                )
                if streak < violation_limit:
                    # Below budget: the task is already back at ``ready``
                    # (respawn allowed) with ``last_failure_error`` stamped.
                    # Deliberately no ``_record_task_failure`` call — a
                    # below-budget violation must not consume the unified
                    # failure budget, just as other failure kinds don't
                    # consume this one.
                    continue
                # Streak reached the bound: trip the breaker. ``force_trip``
                # skips the threshold resolution inside
                # ``_record_task_failure`` because the decision — including
                # the per-task ``max_retries`` override — was already made
                # against the violation streak above.
                tripped = _record_task_failure(
                    conn, tid,
                    error=error_text,
                    outcome="crashed",
                    failure_limit=violation_limit,
                    force_trip=True,
                    release_claim=False,
                    end_run=False,
                    event_payload_extra={
                        "pid": pid,
                        "claimer": claimer,
                        "protocol_violations": streak,
                        "protocol_violation_limit": violation_limit,
                    },
                )
                if tripped:
                    auto_blocked.append(tid)
                continue
            fp = _error_fingerprint(error_text)
            is_systemic = _fp_counts.get(fp, 0) >= 3
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if is_systemic else None,
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "claimer": claimer},
            )
            if tripped:
                auto_blocked.append(tid)
    # Stash auto-blocked ids on the function for the dispatch loop to pick up.
    # Keeps the public return type (``list[str]``) stable for direct callers
    # and tests that destructure the result; ``dispatch_once`` reads this
    # side-channel attribute to populate ``DispatchResult.auto_blocked``.
    detect_crashed_workers._last_auto_blocked = auto_blocked  # type: ignore[attr-defined]
    # Same side-channel for rate-limited requeues — these did NOT count a
    # failure and are NOT crashes, so they stay out of the ``crashed`` return.
    detect_crashed_workers._last_rate_limited = rate_limited  # type: ignore[attr-defined]
    # Worker-lifecycle observer (RFC #58548): exit events are tick-derived
    # from this reclaim pass — fired only now, after the main reclaim txn
    # AND the breaker accounting above have committed, so subscribers always
    # observe fully durable board state.
    if exited_hook_payloads and _kanban_observer_consumed("on_kanban_worker_exited"):
        _board = get_current_board()
        for hook_fields in exited_hook_payloads:
            hook_fields = dict(hook_fields)
            _fire_kanban_lifecycle_hook(
                "on_kanban_worker_exited",
                hook_fields.pop("task_id"),
                board=_board,
                **hook_fields,
            )
    return crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
    force_trip: bool = False,
    release_claim: bool = False,
    end_run: bool = False,
    event_payload_extra: Optional[dict] = None,
) -> bool:
    """Record a non-success outcome (spawn_failed / crashed / timed_out)
    and maybe trip the circuit breaker.

    Unified replacement for the old spawn-only ``_record_spawn_failure``.
    Every path that ends a task with a non-success outcome funnels
    through here so the ``consecutive_failures`` counter and the
    auto-block threshold stay consistent.

    Returns True when the task was auto-blocked (counter reached
    ``failure_limit``), False when it was just updated in place.

    Modes:

    * ``release_claim=True, end_run=True`` — spawn-failure path.
      Caller has a running task with an open run; this transitions
      it back to its source phase (or ``blocked`` when the breaker trips),
      releases the claim, and closes the run with ``outcome=<outcome>``.

    * ``release_claim=False, end_run=False`` — timeout/crash path.
      Caller has ALREADY restored the task's source phase and closed the
      run with the appropriate outcome. This just increments the
      counter; if the breaker trips, the task is re-transitioned
      into ``blocked`` and a ``gave_up`` event is emitted.

    ``event_payload_extra`` merges into the ``gave_up`` event payload
    when the breaker trips, so callers can include outcome-specific
    context (e.g. pid on crash, elapsed on timeout).

    Resolution order for the effective threshold:
      1. per-task ``max_retries`` if set (nothing else overrides)
      2. caller-supplied ``failure_limit`` (gateway passes the config
         value from ``kanban.failure_limit``; tests pass fixed values)
      3. ``DEFAULT_FAILURE_LIMIT``

    ``force_trip=True`` trips the breaker unconditionally, skipping the
    counter-vs-threshold comparison (the resolution order above is then
    only reported in the ``gave_up`` payload, not re-evaluated). Callers
    use it when they have already applied their own bounded-retry policy
    — e.g. the clean-exit protocol-violation streak in
    ``detect_crashed_workers``, which resolves the per-task
    ``max_retries`` override against the violation streak itself. The
    failure is still counted into ``consecutive_failures``.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    blocked = False
    with write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries, current_run_id "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        retry_status = (
            _retry_status_for_run(conn, task_id, row["current_run_id"])
            if release_claim
            else ("review" if row["status"] == "review" else "ready")
        )
        failures = int(row["consecutive_failures"]) + 1

        # Per-task override wins over both caller-supplied and default
        # thresholds. None (the common case) falls through.
        task_override = (
            row["max_retries"] if "max_retries" in row.keys() else None
        )
        if task_override is not None:
            effective_limit = int(task_override)
            limit_source = "task"
        else:
            effective_limit = int(failure_limit)
            limit_source = "dispatcher"

        if force_trip or failures >= effective_limit:
            # Trip the breaker.
            if release_claim:
                # Spawn path: still running, also clear claim state.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('running', 'ready', 'review')",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: source phase already restored with claim
                # cleared; just flip to blocked + update
                # counter fields.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('ready', 'review', 'running')",
                    (failures, error[:500], task_id),
                )
            run_id = None
            if end_run:
                # Only the spawn path has an open run to close.
                run_id = _end_run(
                    conn, task_id,
                    outcome="gave_up", status="gave_up",
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "trigger_outcome": outcome,
                        "effective_limit": effective_limit,
                        "limit_source": limit_source,
                        "retry_status": retry_status,
                    },
                )
            payload = {
                "failures": failures,
                "effective_limit": effective_limit,
                "limit_source": limit_source,
                "error": error[:500],
                "trigger_outcome": outcome,
                "retry_status": retry_status,
            }
            if event_payload_extra:
                payload.update(event_payload_extra)
            _append_event(
                conn, task_id, "gave_up", payload, run_id=run_id,
            )
            blocked = True
        else:
            # Below threshold.
            if release_claim:
                # Spawn path: restore the claimed source phase + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = ?, claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (retry_status, failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: caller already restored the source phase.
                conn.execute(
                    "UPDATE tasks SET consecutive_failures = ?, "
                    "last_failure_error = ? WHERE id = ?",
                    (failures, error[:500], task_id),
                )
            if end_run:
                # Spawn path: close the open run with outcome.
                run_id = _end_run(
                    conn, task_id,
                    outcome=outcome, status=outcome,
                    error=error[:500],
                    metadata={
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                )
                _append_event(
                    conn, task_id, outcome,
                    {
                        "error": error[:500],
                        "failures": failures,
                        "retry_status": retry_status,
                    },
                    run_id=run_id,
                )
            # Timeout/crash path's caller already emitted its own event.
    return blocked


# Backward-compat alias. Old name is referenced from tests and possibly
# third-party callers. New code should call ``_record_task_failure``.
def _record_spawn_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    failure_limit: int = None,
) -> bool:
    return _record_task_failure(
        conn, task_id, error,
        outcome="spawn_failed",
        failure_limit=failure_limit,
        release_claim=True,
        end_run=True,
    )


def _set_worker_pid(conn: sqlite3.Connection, task_id: str, pid: int) -> None:
    """Record the spawned child's pid + emit a ``spawned`` event.

    The event's payload carries the pid so a human reading ``hermes kanban
    tail`` can correlate log lines with OS-level traces without opening
    the drawer.
    """
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ? WHERE id = ?",
            (int(pid), task_id),
        )
        run_id = _current_run_id(conn, task_id)
        if run_id is not None:
            conn.execute(
                "UPDATE task_runs SET worker_pid = ? WHERE id = ?",
                (int(pid), run_id),
            )
        _append_event(conn, task_id, "spawned", {"pid": int(pid)}, run_id=run_id)


def _clear_failure_counter(conn: sqlite3.Connection, task_id: str) -> None:
    """Reset the unified consecutive-failures counter.

    Called from ``complete_task`` on successful completion — a fresh
    success means the task + profile combination is working and any
    past failures are history. NOT called on spawn success anymore:
    a successful spawn proves the worker could start but says nothing
    about whether the run will succeed, so we need to let timeouts and
    crashes accumulate across spawn boundaries.
    """
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 0, "
            "last_failure_error = NULL WHERE id = ?",
            (task_id,),
        )


# Legacy alias for test-code and anything else that still imports it.
_clear_spawn_failures = _clear_failure_counter


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review task in ``dispatch_once`` before any claim attempt.
    Returning a reason defers the spawn this tick; the task stays in its
    source phase and gets another chance on the next dispatcher tick.

    ``lane`` names the dispatch column the task is being spawned from
    (``"ready"`` or ``"review"``). In the review lane the
    ``recent_success`` and ``active_pr`` rules are skipped: a recent PR
    URL comment (and often a recent completed run) is the *precondition*
    of the canonical review handoff — a worker opened a PR and requested
    review — not a duplicate-work signal. Rate-limit cooldown and the
    auth-blocker check still apply in every lane.

    Checks in priority order:

    ``"rate_limit_cooldown"``
        The task's most recent run ended with the ``rate_limited`` outcome
        (a worker bailed on a provider quota wall via the EX_TEMPFAIL
        sentinel) within ``_resolve_rate_limit_cooldown_seconds()``. The
        quota almost certainly hasn't reset yet, so defer the respawn until
        the cooldown elapses — then allow a cheap probe. This is checked
        BEFORE ``blocker_auth`` because the rate-limit requeue stamps a
        quota-flavored ``last_failure_error`` that would otherwise match the
        auth-blocker regex and park the task forever (the rate-limit path
        never increments ``consecutive_failures``, so the breaker can't free
        it). Once the cooldown elapses the task falls through and respawns.

    ``"blocker_auth"``
        The task's last failure error matches a quota / authentication
        pattern. Retrying immediately is unlikely to help (rate limits
        reset on a timer; auth needs human action), so we defer to the
        next tick. The existing ``consecutive_failures`` counter still
        trips the auto-block circuit breaker after ``failure_limit``
        consecutive failures, so a persistent auth error eventually
        blocks via the normal path — but a transient 429 gets a few
        ticks of recovery first.

    ``"recent_success"``
        A completed run exists within ``_RESPAWN_GUARD_SUCCESS_WINDOW``
        seconds. Useful work already succeeded for this task; wait for an
        explicit re-queue rather than immediately re-spawning. Bypassed when an
        explicit re-queue event (status change, promote, unblock, reclaim)
        arrives AFTER that completion — that's a deliberate re-run request.

    ``"active_pr"``
        A GitHub PR URL appears in a recent task comment (within
        ``_RESPAWN_GUARD_PR_WINDOW`` seconds).  A prior worker already
        opened a PR; re-spawning risks a duplicate PR on the same task.

    Stale / dead claim locks are NOT a guard reason — they are handled
    by ``release_stale_claims`` and ``detect_crashed_workers`` which
    reset the task to ``ready`` only after verifying the lock is
    genuinely dead (no live PID on this host).
    """
    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown. The most recent run ended ``rate_limited``
    #    (quota wall) — defer while inside the cooldown window, then allow a
    #    cheap probe. Must run BEFORE the blocker_auth regex check, because a
    #    rate-limit requeue stamps a quota-flavored last_failure_error that
    #    the regex would otherwise match → defer forever (no failure counter
    #    increment on this path means the breaker can never free it).
    #
    #    We look at the LATEST run only (ORDER BY ended_at DESC LIMIT 1): if a
    #    newer crash/completion superseded the rate-limit run, this guard
    #    no longer applies and the normal paths take over.
    rl_cooldown = _resolve_rate_limit_cooldown_seconds()
    latest_run = conn.execute(
        "SELECT outcome, ended_at FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if (
        latest_run is not None
        and latest_run["outcome"] == "rate_limited"
    ):
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, and skip the
            # blocker_auth regex so the stamped rate-limit text doesn't
            # re-trap the task.
            return None
        ended_at = latest_run["ended_at"]
        if ended_at is not None and (now - int(ended_at)) < rl_cooldown:
            return "rate_limit_cooldown"
        # Cooldown elapsed — allow the respawn. Return early so the
        # blocker_auth check below doesn't catch the rate-limit text we
        # stamped on the task; this path intentionally retries forever
        # (cheaply, spaced by the cooldown) until quota returns or a real
        # crash/completion supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.
    err = row["last_failure_error"]
    if err and _RESPAWN_BLOCKER_RE.search(err):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR
    # URL comment are the canonical *inputs* to a review handoff (worker
    # opened a PR, then requested review), not signals of duplicate work.
    if lane == "review":
        return None

    # 3. Completed run within guard window — proof of recent success.
    #    Exception: an explicit re-queue AFTER that success (an operator
    #    dragging done→ready, a dependency re-promotion, an unblock, a
    #    reclaim) is a deliberate "run it again" — honor it instead of
    #    deferring. Without this, a manual done→ready just sits there,
    #    silently held by the guard, until the window elapses.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    recent_completed = conn.execute(
        "SELECT ended_at FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ? "
        "ORDER BY ended_at DESC LIMIT 1",
        (task_id, cutoff),
    ).fetchone()
    if recent_completed:
        completed_at = int(recent_completed["ended_at"] or 0)
        requeued_after = conn.execute(
            "SELECT 1 FROM task_events "
            "WHERE task_id = ? AND created_at >= ? "
            "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed') "
            "LIMIT 1",
            (task_id, completed_at),
        ).fetchone()
        if not requeued_after:
            return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    pr_cutoff = now - _RESPAWN_GUARD_PR_WINDOW
    for c in conn.execute(
        "SELECT body FROM task_comments WHERE task_id = ? AND created_at >= ?",
        (task_id, pr_cutoff),
    ).fetchall():
        if c["body"] and _RESPAWN_GUARD_PR_URL_RE.search(c["body"]):
            return "active_pr"

    return None


def has_spawnable_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one ready+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Used by the gateway- and CLI-embedded dispatchers' health telemetry to
    decide whether ``0 spawned`` is a "stuck" condition (real spawnable
    work waiting) or a "correctly idle" condition (only control-plane
    lanes like ``orion-cc`` / ``orion-research`` waiting on terminals
    that pull tasks via ``claim_task`` directly).

    Falls back to "any ready+assigned" if ``profile_exists`` is not
    importable (e.g. partial install) — preserves the old behavior so
    the warning still fires in degraded environments.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'ready' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        # Can't introspect — assume spawnable, preserve legacy behavior.
        return True
    for row in rows:
        if profile_exists(row["assignee"]):
            return True
    return False


def has_spawnable_review(conn: sqlite3.Connection) -> bool:
    """Return True iff there is at least one review+assigned+unclaimed task
    whose assignee maps to a real Hermes profile.

    Mirror of :func:`has_spawnable_ready` for the review column —
    used by the health telemetry to decide whether the dispatcher
    should have spawned a review agent.
    """
    rows = conn.execute(
        "SELECT DISTINCT assignee FROM tasks "
        "WHERE status = 'review' AND assignee IS NOT NULL "
        "    AND claim_lock IS NULL"
    ).fetchall()
    if not rows:
        return False
    try:
        from hermes_cli.profiles import profile_exists  # local import: avoids cycle
    except Exception:
        return True
    for row in rows:
        if profile_exists(row["assignee"]):
            return True
    return False


def review_dispatch_enabled() -> bool:
    """Return whether first-class review tasks should dispatch automatically.

    The default is true because Hermes ships the ``sdlc-review`` skill and the
    review lifecycle includes a supported reviewer-owned changes-requested
    transition. Operators can disable it for human-only review boards.
    """
    try:
        from hermes_cli.config import load_config
        return bool(
            (load_config() or {}).get("kanban", {}).get("review_dispatch", True)
        )
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Memory-aware dispatch guard (OOF-30 / OOF-77)
#
# Two production incidents ("larrikin-lollies", "synclare-task-manager")
# followed the same shape: no ``kanban.max_in_progress`` configured, a busy
# board, and a 1 GiB VM — the dispatcher fanned out 26-31 concurrent workers,
# the host went into swap-thrash/OOM, and the dashboard (and everything else
# on the machine) became unreachable. Two complementary safeguards:
#
#   1. A memory-DERIVED default concurrency cap when the operator never set
#      ``kanban.max_in_progress`` (``resolve_max_in_progress``) — sized from
#      MemTotal so a 1 GiB VM defaults to 2 workers, not unlimited.
#   2. A live memory-PRESSURE guard inside the dispatch tick itself
#      (``_memory_pressure_level``) — even a correctly-sized static cap can't
#      see other tenants of the box, so under real observed pressure the
#      dispatcher stops adding workers regardless of configured caps.
#
# Both fail open: on non-Linux hosts or any read error the sample is empty,
# the derived default is None (no cap — unchanged behaviour), and the
# pressure level is "unknown" (no spawn restriction).
# ---------------------------------------------------------------------------

# Assumed per-worker memory footprint for the derived default cap. Hermes
# workers are full agent processes (Python + model client + tool subprocesses);
# ~512 MiB is a deliberately conservative planning number so the derived cap
# errs toward fewer workers on small VMs.
MEMORY_GUARD_MB_PER_WORKER = 512
# Bounds for the derived default: never below 2 (a board must still make
# progress on the smallest hosted VM) and never above 8 (operators who want
# more fan-out on big iron should say so explicitly in config).
DERIVED_MAX_IN_PROGRESS_FLOOR = 2
DERIVED_MAX_IN_PROGRESS_CEILING = 8


def _system_memory_sample() -> dict:
    """Best-effort system memory snapshot (KiB values), ``{}`` when unknown.

    Delegates to :func:`gateway.lifecycle_ledger.sample_memory` (pure /proc
    reads, Linux-only, never raises). Local import keeps ``kanban_db``
    importable in stripped-down environments without the gateway package.
    Module-level indirection is also the test seam — the shared conftest
    patches this to ``{}`` so suite results don't depend on the CI runner's
    live memory state.
    """
    try:
        from gateway.lifecycle_ledger import sample_memory
        return sample_memory() or {}
    except Exception:
        return {}


def derive_default_max_in_progress(sample: Optional[Mapping[str, Any]] = None) -> Optional[int]:
    """Memory-derived default for ``kanban.max_in_progress`` when unset.

    ``clamp(MemTotal / MEMORY_GUARD_MB_PER_WORKER, FLOOR, CEILING)`` — e.g.
    a 1 GiB VM derives 2, a 4 GiB VM derives 8. Returns ``None`` (no cap,
    pre-fix behaviour) when total memory can't be determined, so dev
    machines on macOS/Windows are unaffected.
    """
    if sample is None:
        sample = _system_memory_sample()
    total_kib = sample.get("mem_total_kib")
    if isinstance(total_kib, bool) or not isinstance(total_kib, int) or total_kib <= 0:
        return None
    workers = (total_kib // 1024) // MEMORY_GUARD_MB_PER_WORKER
    return max(
        DERIVED_MAX_IN_PROGRESS_FLOOR,
        min(workers, DERIVED_MAX_IN_PROGRESS_CEILING),
    )


def resolve_max_in_progress(configured: Optional[int]) -> Optional[int]:
    """Return the effective global concurrency cap for a dispatch tick.

    An explicit operator-configured value always wins. When unset, fall back
    to the memory-derived default (see :func:`derive_default_max_in_progress`).
    Callers that parse config (gateway dispatcher, ``hermes kanban dispatch``)
    should route through this so both paths agree.
    """
    if configured is not None:
        return configured
    return derive_default_max_in_progress()


def configured_max_in_progress() -> Optional[int]:
    """Read ``kanban.max_in_progress`` from config, or None when unset/invalid.

    Small shared parser so every dispatch entry point (gateway watcher, CLI
    dispatch, standalone daemon) agrees on what "explicitly configured"
    means: a positive integer wins, anything else falls through to the
    memory-derived default via :func:`resolve_max_in_progress`.
    """
    try:
        from hermes_cli.config import load_config_readonly
        raw = (load_config_readonly() or {}).get("kanban", {}).get(
            "max_in_progress"
        )
    except Exception:
        return None
    if raw is None:
        return None
    try:
        ival = int(raw)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def count_running_tasks(conn: sqlite3.Connection) -> int:
    """Return the number of tasks currently in ``status='running'``.

    Used by the gateway's multi-board sweep to account for workers on
    OTHER boards against the host-level concurrency budget (OOF-30): the
    memory-derived cap bounds the machine, so each board's tick must see
    the machine's total, not just its own. Fails open to 0 — a broken
    board must not brick dispatch on healthy ones (corruption is handled
    separately by the watcher's quarantine logic).
    """
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )
    except Exception:
        return 0


def count_running_tasks_other_boards(board: Optional[str] = None) -> int:
    """Total ``running`` tasks across every board EXCEPT ``board``.

    The concurrency caps bound the HOST (workers are OS processes sharing
    one machine's memory), but each board's dispatch tick only sees its own
    DB. Without this, a memory-derived cap of N gets multiplied by the
    number of active boards — reproduced in review of OOF-30: two boards
    each spawned N workers on a derived N-worker host budget.

    Boards are matched by resolved DB path, so the ``HERMES_KANBAN_DB``
    override (which pins every board to one file) naturally yields 0.
    Fails open per board: one broken/corrupt board must not brick dispatch
    on the healthy ones.
    """
    try:
        current_path = str(kanban_db_path(board=board).expanduser().resolve())
    except Exception:
        current_path = None
    try:
        boards = list_boards(include_archived=False)
    except Exception:
        return 0
    total = 0
    for meta in boards:
        slug = meta.get("slug") or DEFAULT_BOARD
        try:
            path = kanban_db_path(board=slug).expanduser()
            resolved = str(path.resolve())
            if current_path is not None and resolved == current_path:
                continue
            if not path.exists():
                continue
            other = connect(board=slug)
            try:
                total += count_running_tasks(other)
            finally:
                try:
                    other.close()
                except Exception:
                    pass
        except Exception:
            continue
    return total


def _memory_pressure_level(sample: Optional[Mapping[str, Any]] = None) -> str:
    """Classify current system memory pressure: ok/elevated/critical/unknown.

    Reuses :func:`gateway.memory_status.classify_pressure` so the dispatcher's
    idea of "critical" matches the memory banner users see on the dashboard
    and the lifecycle ledger's OOM-suspicion heuristics (NS-608/NS-656).
    ``unknown`` (non-Linux, read failure) imposes no restriction — the guard
    must never brick dispatch on hosts where /proc isn't available.
    """
    if sample is None:
        sample = _system_memory_sample()
    if not sample:
        return "unknown"
    try:
        from gateway.memory_status import classify_pressure
        return classify_pressure(
            sample.get("mem_available_kib"), sample.get("mem_total_kib")
        )
    except Exception:
        return "unknown"


def dispatch_once(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick under the board's single-writer lock.

    Thin wrapper around :func:`_dispatch_once_locked`. It acquires a
    non-blocking, board-scoped dispatch lock (issue #35240) so that two
    dispatchers pointed at the same ``kanban.db`` — e.g. the service-
    managed gateway and a shell-spawned orphan that escaped the service
    cgroup — can never run a reclaim/spawn/write tick concurrently and
    race on WAL frames. The losing dispatcher returns an empty
    ``DispatchResult`` with ``skipped_locked=True`` and does no DB writes;
    the holder is already making progress on the same board.

    The lock is keyed off the board's resolved DB path, so unrelated
    boards tick in parallel. See :func:`_dispatch_tick_lock` for the
    cross-process / cross-platform mechanics.
    """
    try:
        db_path = kanban_db_path(board=board)
    except Exception:
        # Path resolution should never fail, but if it somehow does we
        # must not lose the tick — fall through to an unguarded dispatch
        # rather than dropping work.
        result = _dispatch_once_locked(
            conn,
            spawn_fn=spawn_fn,
            ttl_seconds=ttl_seconds,
            dry_run=dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=failure_limit,
            stale_timeout_seconds=stale_timeout_seconds,
            board=board,
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
            reconcile_orphans=reconcile_orphans,
        )
        _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
        return result
    with _dispatch_tick_lock(db_path) as held:
        if not held:
            result = DispatchResult(skipped_locked=True)
        else:
            result = _dispatch_once_locked(
                conn,
                spawn_fn=spawn_fn,
                ttl_seconds=ttl_seconds,
                dry_run=dry_run,
                max_spawn=max_spawn,
                max_in_progress=max_in_progress,
                failure_limit=failure_limit,
                stale_timeout_seconds=stale_timeout_seconds,
                board=board,
                default_assignee=default_assignee,
                max_in_progress_per_profile=max_in_progress_per_profile,
                reconcile_orphans=reconcile_orphans,
            )
            # Still under the dispatch lock: run the periodic PASSIVE WAL
            # checkpoint (see _maybe_checkpoint_wal; the -wal file size is
            # bounded by journal_size_limit on the writer's natural reset).
            _maybe_checkpoint_wal(conn, db_path)
    # The dispatch lock has been released here. Fire the tick observer
    # strictly OUTSIDE the single-writer critical section (#56066 sweeper
    # finding / #64231 disposition): a slow subscriber must never extend
    # the lock hold and stall a sibling dispatcher's tick.
    _fire_dispatch_tick_hook(result, board=board, dry_run=dry_run)
    return result


def _dispatch_once_locked(
    conn: sqlite3.Connection,
    *,
    spawn_fn=None,
    ttl_seconds: Optional[int] = None,
    dry_run: bool = False,
    max_spawn: Optional[int] = None,
    max_in_progress: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stale_timeout_seconds: int = 0,
    board: Optional[str] = None,
    default_assignee: Optional[str] = None,
    max_in_progress_per_profile: Optional[int] = None,
    reconcile_orphans: bool = True,
) -> DispatchResult:
    """Run one dispatcher tick.

    Steps:
      1. Reclaim stale running tasks (TTL expired).
      2. Reclaim stale running tasks (no recent heartbeat).
      3. Reclaim crashed running tasks (host-local PID no longer alive).
      3. Promote todo -> ready where all parents are done.
      4. For each ready task with an assignee, atomically claim and call
         ``spawn_fn(task, workspace_path, board) -> Optional[int]``. The
         return value (if any) is recorded as ``worker_pid`` so subsequent
         ticks can detect crashes before the TTL expires.

    Spawn failures are counted per-task. After ``failure_limit`` consecutive
    failures the task is auto-blocked with the last error as its reason —
    prevents the dispatcher from thrashing forever on an unfixable task.

    ``max_spawn`` is a **live concurrency cap**, not a per-tick spawn budget:
    it counts tasks already in ``status='running'`` plus this tick's spawns
    against the limit. So ``max_spawn=4`` means "at most 4 workers running
    at any time across the whole board" — matching the gateway's stated
    intent ("limit concurrent kanban tasks"). With a per-tick interpretation
    a 60-second tick interval could grow concurrency by N every minute on a
    busy board and accumulate without bound.

    ``max_in_progress`` is a **host-level** concurrency cap (OOF-30): it
    counts running tasks on every active board — not just this one — plus
    this tick's spawns. Workers are OS processes sharing one machine's
    memory, so a per-board interpretation would multiply the cap by the
    number of active boards. ``max_spawn`` retains its historical per-board
    semantics.

    ``spawn_fn`` defaults to ``_default_spawn``. Tests pass a stub.
    ``board`` pins workspace/log/db resolution for this tick to a specific
    board. When omitted, the current-board resolution chain is used.
    """
    # Reap zombie children from previously spawned workers. See
    # reap_worker_zombies() for the full rationale.
    reap_worker_zombies()

    result = DispatchResult()
    result.reclaimed = release_stale_claims(conn)
    if reconcile_orphans:
        # Orphaned-card reconciliation: requeue 'running' cards whose claim
        # bookkeeping is broken (no valid claim, dead/gone worker) that the
        # TTL/crash/stale paths can never see. See reconcile_orphaned_running.
        result.reconciled_orphans = reconcile_orphaned_running(conn)
    result.stale = detect_stale_running(
        conn, stale_timeout_seconds=stale_timeout_seconds,
    )
    result.crashed = detect_crashed_workers(conn)
    # detect_crashed_workers stashes protocol-violation auto-blocks on
    # itself so the public list-return stays stable. Pull them into the
    # DispatchResult here so telemetry / tests see the trip.
    _crash_auto_blocked = getattr(
        detect_crashed_workers, "_last_auto_blocked", []
    )
    if _crash_auto_blocked:
        result.auto_blocked.extend(_crash_auto_blocked)
    # Rate-limited requeues (quota wall, no failure counted) — surface for
    # telemetry / tests. These tasks went back to ``ready`` and the respawn
    # guard will defer them until the quota window clears.
    _crash_rate_limited = getattr(
        detect_crashed_workers, "_last_rate_limited", []
    )
    if _crash_rate_limited:
        result.rate_limited.extend(_crash_rate_limited)
    result.timed_out = enforce_max_runtime(conn)
    result.promoted = recompute_ready(conn, failure_limit=failure_limit)

    # Count tasks already running so max_spawn enforces concurrency rather
    # than a per-tick spawn budget. See the docstring above for the full
    # rationale; the short version is that a 60-second tick interval with a
    # per-tick budget of N would grow concurrency by N every tick on a busy
    # board, since "running" tasks aren't reclaimed by completion alone —
    # they sit in status='running' until the worker calls
    # kanban_complete/kanban_block (or the dispatcher TTL-reclaims them).
    running_count = 0
    spawn_budget: Optional[int] = None
    if max_spawn is not None or max_in_progress is not None:
        running_count = count_running_tasks(conn)

    # Convert any concurrency caps into a shared additional-spawns budget
    # for this tick. Both ready and review loops consume from the same
    # budget so the total number of new workers stays bounded.
    if max_spawn is not None:
        if running_count >= max_spawn:
            return result
        spawn_budget = max_spawn - running_count

    # Honour kanban.max_in_progress across both ready and review queues: if
    # the board already has enough running tasks, skip this tick entirely.
    # When there is room left, intersect the remaining in-progress budget
    # with any explicit max_spawn cap above.
    #
    # max_in_progress is a HOST-level cap, not a per-board one (OOF-30):
    # workers are OS processes sharing one machine's memory, so running
    # workers on every other board count against the same budget. Without
    # this, N active boards multiply the cap by N — exactly the fan-out
    # the memory-derived default exists to prevent.
    if max_in_progress is not None:
        total_running = running_count + count_running_tasks_other_boards(board)
        if total_running >= max_in_progress:
            return result
        remaining = max_in_progress - total_running
        if spawn_budget is None or spawn_budget > remaining:
            spawn_budget = remaining

    # Memory-pressure guard (OOF-30/OOF-77): even a well-chosen static cap
    # can't see the host's actual memory state (other tenants, bloated
    # long-lived workers, dashboard growth). Under observed pressure the
    # dispatcher stops adding load: critical -> spawn nothing this tick;
    # elevated -> at most one new worker. Reclaim/promotion above already
    # ran, so board bookkeeping stays live either way, and deferred tasks
    # simply wait for a later tick. "unknown" imposes no restriction.
    pressure = _memory_pressure_level()
    if pressure == "critical":
        result.memory_pressure = pressure
        _log.warning(
            "kanban dispatch: system memory pressure is critical; "
            "spawning no new workers this tick (deferred, not dropped)"
        )
        return result
    if pressure == "elevated":
        result.memory_pressure = pressure
        if spawn_budget is None or spawn_budget > 1:
            _log.warning(
                "kanban dispatch: system memory pressure is elevated; "
                "limiting to at most 1 new worker this tick"
            )
            spawn_budget = 1

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    # Review rows are enumerated up front (not after the ready loop) so the
    # budget split below can see whether review work exists at all.
    review_rows = []
    if review_dispatch_enabled():
        review_rows = conn.execute(
            "SELECT id, assignee FROM tasks "
            "WHERE status = 'review' AND claim_lock IS NULL "
            "ORDER BY priority DESC, created_at ASC"
        ).fetchall()
    # Review-lane reservation (OOF-30 review finding): the ready loop runs
    # first and used to consume the ENTIRE shared budget, so a sustained
    # ready backlog permanently starved autonomous reviews — completed work
    # sat in 'review' forever while new work kept spawning. When spawnable
    # review work exists and the tick has any budget, hold one slot back
    # from the ready loop so the review lane always gets a spawn
    # opportunity. The reservation is per-tick and self-releasing: with no
    # spawnable review work (or no cap at all) the ready loop keeps the
    # full budget. "Spawnable" mirrors the review loop's own gate
    # (assigned + real profile) so a review column full of human-pulled
    # control-plane lanes doesn't permanently tax ready throughput.
    def _any_spawnable_review() -> bool:
        if not review_rows:
            return False
        try:
            from hermes_cli.profiles import profile_exists as _rpe
        except Exception:
            # Profiles module unavailable (test stubs, exotic envs) —
            # assume spawnable, matching the review loop's own fallback.
            return any(row["assignee"] for row in review_rows)
        return any(
            row["assignee"] and _rpe(row["assignee"]) for row in review_rows
        )

    ready_budget = spawn_budget
    if spawn_budget is not None and spawn_budget > 0 and _any_spawnable_review():
        ready_budget = max(spawn_budget - 1, 0)
    spawned = 0
    # Per-profile concurrency cap (#21582): when set, track how many
    # workers each assignee already has in flight, and refuse to spawn
    # when this would push that assignee past the cap. Prevents
    # fan-out workloads from melting a single profile's local model /
    # API quota / browser pool while leaving other profiles idle.
    # Tasks blocked this way go to skipped_per_profile_capped (not
    # skipped_unassigned — the operator-actionable signal is different:
    # "this profile is busy, try again later" not "this needs routing").
    _per_profile_cap = max_in_progress_per_profile if (
        isinstance(max_in_progress_per_profile, int)
        and max_in_progress_per_profile > 0
    ) else None
    _per_profile_running: dict[str, int] = {}
    if _per_profile_cap is not None:
        for prow in conn.execute(
            "SELECT assignee, COUNT(*) AS n FROM tasks "
            "WHERE status = 'running' AND assignee IS NOT NULL "
            "GROUP BY assignee"
        ):
            _per_profile_running[prow["assignee"]] = int(prow["n"])
    # Normalize default_assignee once: empty/whitespace string → None so the
    # rest of the loop can use ``if default_assignee:`` as a single check.
    # We also resolve profile_exists once here for the same reason.
    _default_assignee = (default_assignee or "").strip() or None
    _default_assignee_resolved = False
    if _default_assignee:
        try:
            from hermes_cli.profiles import profile_exists as _pe
            _default_assignee_resolved = bool(_pe(_default_assignee))
        except Exception:
            # Profiles module not importable (test stubs, exotic envs).
            # Trust the operator's config and try the assignment; the
            # downstream profile_exists check on the assigned row will
            # bucket it as nonspawnable if the profile genuinely isn't
            # there, with the existing diagnostic.
            _default_assignee_resolved = True
    for row in ready_rows:
        if ready_budget is not None and spawned >= ready_budget:
            break
        row_assignee = row["assignee"]
        if not row_assignee:
            # Honour kanban.default_assignee: when the dispatcher hits an
            # unassigned ready task and an operator-configured fallback
            # exists, persist the assignment and proceed. This removes the
            # dashboard footgun where a task created without an assignee
            # parks in 'ready' forever even though the operator's intent
            # ("default") was perfectly clear (#27145). Mutating the row
            # (not just the in-memory view) keeps diagnostics and the
            # board state consistent: the task is now legitimately owned
            # by ``kanban.default_assignee``, not "unassigned but secretly
            # routed".
            if _default_assignee and _default_assignee_resolved:
                # Dry-run: show what WOULD happen (auto-assign + spawn) without
                # mutating the DB. Real run: mutate the row + emit the
                # 'assigned' event so the board state matches what just happened.
                if not dry_run:
                    try:
                        with write_txn(conn):
                            conn.execute(
                                "UPDATE tasks SET assignee = ? WHERE id = ? "
                                "AND (assignee IS NULL OR assignee = '')",
                                (_default_assignee, row["id"]),
                            )
                            _append_event(
                                conn, row["id"], "assigned",
                                {
                                    "assignee": _default_assignee,
                                    "source": "kanban.default_assignee",
                                },
                            )
                    except Exception:
                        _log.debug(
                            "kanban dispatch: failed to apply default_assignee=%r "
                            "to task %s",
                            _default_assignee, row["id"], exc_info=True,
                        )
                        result.skipped_unassigned.append(row["id"])
                        continue
                row_assignee = _default_assignee
                result.auto_assigned_default.append(row["id"])
            else:
                result.skipped_unassigned.append(row["id"])
                continue
        # Skip ready tasks whose assignee is not a real Hermes profile.
        # `_default_spawn` invokes ``hermes -p <assignee>`` which fails
        # with "Profile 'X' does not exist" when the assignee names a
        # control-plane lane (e.g. an interactive Claude Code terminal
        # like ``orion-cc`` / ``orion-research``) rather than a Hermes
        # profile. Those task lanes are pulled by terminals via
        # ``claim_task`` directly and should NEVER auto-spawn — the
        # subprocess would crash on startup, get reaped as a zombie,
        # the task would loop back to ``ready`` on next tick, and we'd
        # burn CPU forever (#kanban-dispatcher-crash-loop 2026-05-05).
        try:
            from hermes_cli.profiles import profile_exists  # local import: avoids cycle
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row_assignee):
            # Bucket separately from skipped_unassigned: the operator
            # cannot fix this by assigning a profile (the assignee IS the
            # intended owner — a terminal lane). Health telemetry uses
            # this distinction to suppress spurious "stuck" warnings on
            # multi-lane setups where the ready queue is steadily full
            # of human-pulled work.
            result.skipped_nonspawnable.append(row["id"])
            continue
        # Per-profile concurrency cap (#21582): even if there's global
        # headroom, refuse to spawn for an assignee that's already at
        # its in-flight cap. Prevents one profile's local model / API
        # quota / browser pool from being overwhelmed by a fan-out
        # while the global max_in_progress / max_spawn caps still allow
        # work on OTHER profiles.
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row_assignee, 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row_assignee, current)
                )
                continue
        # Respawn guard: refuse to re-spawn when useful work is already
        # in-flight/recent, or when the last failure is a deterministic
        # blocker (quota / auth). The guard defers the spawn this tick so
        # the task gets a chance to clear (rate limits often reset in
        # seconds-to-minutes); the existing consecutive_failures counter
        # still trips the auto-block circuit breaker after failure_limit
        # consecutive failures, so a persistent auth error eventually
        # blocks via the normal path rather than on first occurrence.
        guard_reason = check_respawn_guard(conn, row["id"])
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            # Emit an event so operators can see why the task was
            # skipped when reading `hermes kanban tail` — without
            # this the task appears stuck in ready with no diagnosis.
            if not dry_run:
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
            continue
        if dry_run:
            result.spawned.append((row["id"], row_assignee, ""))
            spawned += 1
            # Increment per-profile counter even in dry_run so the cap
            # check sees the would-be spawn on subsequent iterations.
            # Without this, dry_run reports every task as spawnable and
            # under-reports the capped subset (#21582).
            if _per_profile_cap is not None and row_assignee:
                _per_profile_running[row_assignee] = (
                    _per_profile_running.get(row_assignee, 0) + 1
                )
            continue
        claimed = claim_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            # Back-compat: older spawn_fn signatures accept only
            # (task, workspace). Test stubs in the suite rely on that.
            # Introspect the callable and pass `board` only when supported.
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): fires AFTER spawn_fn
            # returned and the PID (when reported) is durably persisted,
            # per the RFC timing contract. Best-effort — can never break
            # the dispatch loop.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            # NOTE: we intentionally do NOT reset consecutive_failures
            # here. A successful spawn proves the worker can start but
            # doesn't prove the run will succeed. Under unified
            # failure counting, resetting on spawn would let a task
            # that keeps timing out after spawn loop forever. The
            # counter is cleared only on successful completion (see
            # complete_task).
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            # Track the new in-flight count for this profile so later
            # iterations in this same tick respect the per-profile cap
            # (#21582). Subsequent ticks re-query from the DB.
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)

    # ---- review column dispatch ----
    # Review tasks are tasks that a worker moved to 'review' after
    # creating a PR.  The dispatcher spawns a review agent (loading
    # sdlc-review skill) that verifies the candidate and either approves
    # (→ done) or requests changes (→ ready/todo for the implementer).
    #
    # Same concurrency model as ready dispatch: review spawns count
    # against max_spawn alongside ready tasks, so the total number of
    # running workers stays bounded.
    # Auto-dispatch is enabled by default because Hermes bundles the
    # ``sdlc-review`` skill and reviewer workers can now approve, request
    # changes without block-loop accounting, or escalate a genuine blocker.
    # Human-only boards can disable it with ``kanban.review_dispatch``.
    #
    # ``review_rows`` was enumerated before the ready loop; when it is
    # non-empty the ready loop ran against ``ready_budget`` (one slot held
    # back) so this lane cannot be permanently starved by a sustained
    # ready backlog. The review loop itself still checks the FULL shared
    # ``spawn_budget`` — the reservation caps the ready lane, it does not
    # grant the review lane extra capacity.
    for row in review_rows:
        if spawn_budget is not None and spawned >= spawn_budget:
            break
        if not row["assignee"]:
            result.skipped_unassigned.append(row["id"])
            continue
        try:
            from hermes_cli.profiles import profile_exists
        except Exception:
            profile_exists = None  # type: ignore[assignment]
        if profile_exists is not None and not profile_exists(row["assignee"]):
            result.skipped_nonspawnable.append(row["id"])
            continue
        if _per_profile_cap is not None:
            current = _per_profile_running.get(row["assignee"], 0)
            if current >= _per_profile_cap:
                result.skipped_per_profile_capped.append(
                    (row["id"], row["assignee"], current)
                )
                continue
        guard_reason = check_respawn_guard(conn, row["id"], lane="review")
        if guard_reason is not None:
            result.respawn_guarded.append((row["id"], guard_reason))
            if not dry_run:
                with write_txn(conn):
                    _append_event(
                        conn, row["id"], "respawn_guarded",
                        {"reason": guard_reason},
                    )
            continue
        if dry_run:
            result.spawned.append((row["id"], row["assignee"], ""))
            spawned += 1
            if _per_profile_cap is not None:
                _per_profile_running[row["assignee"]] = (
                    _per_profile_running.get(row["assignee"], 0) + 1
                )
            continue
        claimed = claim_review_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        try:
            resolved_branch_name = None
            if claimed.workspace_kind == "worktree":
                workspace, resolved_branch_name = _resolve_worktree_workspace(claimed, board=board)
            else:
                workspace = resolve_workspace(claimed, board=board)
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, f"workspace: {exc}",
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
            continue
        # Persist the resolved workspace path so the worker can cd there.
        set_workspace_path(conn, claimed.id, str(workspace))
        if claimed.workspace_kind == "worktree":
            set_branch_name(conn, claimed.id, resolved_branch_name or (claimed.branch_name or "").strip() or f"wt/{claimed.id}")
        _maybe_emit_scratch_tip(conn, claimed.id, claimed.workspace_kind)
        # Force-load the sdlc-review skill for review agents — it carries
        # the review logic (AC verification, merge, etc.). The mandatory
        # kanban lifecycle is already injected into every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill the
        # review agent needs.
        claimed.skills = list(
            dict.fromkeys([*(claimed.skills or []), "sdlc-review"])
        )
        _spawn = spawn_fn if spawn_fn is not None else _default_spawn
        try:
            import inspect
            try:
                sig = inspect.signature(_spawn)
                if "board" in sig.parameters:
                    pid = _spawn(claimed, str(workspace), board=board)
                else:
                    pid = _spawn(claimed, str(workspace))
            except (TypeError, ValueError):
                pid = _spawn(claimed, str(workspace))
            if pid:
                _set_worker_pid(conn, claimed.id, int(pid))
            # Worker-lifecycle observer (RFC #58548): same contract as the
            # ready-lane fire above — after spawn + PID persistence.
            _fire_worker_spawned_hook(
                conn, claimed, str(workspace), pid, board=board,
            )
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
            if _per_profile_cap is not None and claimed.assignee:
                _per_profile_running[claimed.assignee] = (
                    _per_profile_running.get(claimed.assignee, 0) + 1
                )
        except Exception as exc:
            auto = _record_spawn_failure(
                conn, claimed.id, str(exc),
                failure_limit=failure_limit,
            )
            if auto:
                result.auto_blocked.append(claimed.id)
    return result


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.

    Defaults preserve the historical behavior: rotate at 2 MiB and keep one
    backup generation (``.log.1``). Operators with long-running workers can
    raise either value from ``config.yaml`` without changing dispatcher code.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = (load_config().get("kanban") or {})
        except Exception:
            kanban_cfg = {}
    max_bytes = _positive_int(
        (kanban_cfg or {}).get("worker_log_rotate_bytes"),
        DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        (kanban_cfg or {}).get("worker_log_backup_count"),
        DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``.

    ``backup_count=1`` preserves the legacy single-generation behavior:
    ``<log>`` moves to ``<log>.1`` and any previous ``.1`` is replaced.
    Higher values shift older generations up to ``backup_count``.
    """
    try:
        if not log_path.exists():
            return
        if log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count,
            DEFAULT_LOG_BACKUP_COUNT,
            minimum=0,
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        try:
            if oldest.exists():
                oldest.unlink()
        except OSError:
            pass
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            try:
                src.rename(_rotated_log_path(log_path, generation + 1))
            except OSError:
                pass
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Return the interpreter-bound Hermes CLI invocation."""
    # ``hermes_cli.main`` is the console-script target declared in
    # pyproject.toml, NOT a top-level ``hermes`` package — there is no
    # ``hermes`` package to import.
    return [sys.executable, "-m", "hermes_cli.main"]


def _absolute_hermes_path(path: str) -> str:
    """Return an absolute filesystem path for a resolved Hermes shim."""
    expanded = os.path.expanduser(path)
    return expanded if os.path.isabs(expanded) else os.path.abspath(expanded)


def _looks_like_path(value: str) -> bool:
    """Return true when a command override is an explicit path, not a name."""
    expanded = os.path.expanduser(value)
    return (
        expanded.startswith("~")
        or os.path.isabs(expanded)
        or bool(os.path.dirname(expanded))
        or "\\" in expanded
        or bool(re.match(r"^[A-Za-z]:", expanded))
    )


def _is_windows_batch_shim(path: str) -> bool:
    """Return true for Windows shell/batch shims that should not be argv[0]."""
    return path.lower().endswith((".cmd", ".bat"))


def _path_search_names(command: str) -> list[str]:
    """Return executable names to try for an unqualified command."""
    if not _IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    exts = [ext for ext in raw.split(";") if ext]
    return [command + ext for ext in exts]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    ``shutil.which`` follows platform search behavior. On Windows that can
    include the current directory before PATH for bare names, which is not a
    safe dispatcher primitive. This resolver only considers explicit PATH
    entries and skips empty / ``.`` entries.
    """
    path_env = os.environ.get("PATH", "")
    for raw_dir in path_env.split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if not os.path.isfile(candidate):
                continue
            if _IS_WINDOWS or os.access(candidate, os.X_OK):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """Return argv for a resolved Hermes executable path.

    Windows batch shims (`.cmd` / `.bat`) are not safe as argv[0] for
    worker launches because the argument vector includes task-derived
    values. Prefer the interpreter-bound module form whenever the resolved
    executable is only a shell shim.
    """
    if _IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv parts for ``Popen``.

    Tries in order:

    1. ``$HERMES_BIN`` — explicit operator override. Path-like values are
       normalized to absolute paths; bare command names keep normal PATH
       semantics and never prefer a same-directory file before ``PATH``.
    2. ``shutil.which("hermes")`` — the console-script shim, normalized to
       an absolute path. On Windows, ``which`` can return a relative
       ``.\\hermes.CMD`` when the current directory is on ``PATH``; directly
       launching batch shims is also unsafe with task-derived argv. The
       dispatcher therefore falls back to the interpreter-bound module form
       for implicit ``.cmd`` / ``.bat`` shims.
    3. ``sys.executable -m hermes_cli.main`` — fallback for setups where
       Hermes is launched from a venv and the ``hermes`` shim is not on
       the dispatcher's ``$PATH`` (cron, systemd ``User=`` services,
       launchd jobs, detached processes, etc.). Goes through the running
       interpreter so the result is independent of ``$PATH``.

    Mirrors ``gateway.run._resolve_hermes_bin`` for the same reason. Kept
    local (not imported from gateway) because ``hermes_cli`` sits below
    ``gateway`` in the dependency order.
    """
    import shutil

    env_bin = os.environ.get("HERMES_BIN", "").strip()
    if env_bin:
        if _looks_like_path(env_bin):
            return _hermes_path_argv(env_bin)
        resolved_env_bin = _safe_which_no_cwd(env_bin)
        if resolved_env_bin:
            return _hermes_path_argv(resolved_env_bin)
        return _module_hermes_argv()

    hermes_bin = _safe_which_no_cwd("hermes") if _IS_WINDOWS else shutil.which("hermes")
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    Kanban's ``max_runtime_seconds`` bounds the whole worker attempt. The
    terminal tool has its own default timeout via ``TERMINAL_TIMEOUT``; when
    the worker runtime is longer, raise only the child process default so a
    long command is not killed by the generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(1, runtime - KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS)
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Dispatcher-spawned workers are launched from a long-lived gateway process,
    then the child re-enters the CLI with ``-p <assignee>``. Resolve the
    assignee profile's CLI tool surface at dispatch time and pass it as an
    explicit ``--toolsets`` pin so worker startup cannot fall back to a stale
    root/active-profile config or a profile whose top-level ``toolsets`` entry
    is only the kanban orchestrator surface. ``model_tools`` still appends the
    task-scoped kanban lifecycle tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort and gated — the durable ``state_meta`` gate lives in
    ``retag_kanban_worker_sessions``; the in-process set keeps a busy
    dispatcher from reopening state.db on every spawn just to read it. A
    dispatcher tick must never fail because a session DB was busy or missing.
    """
    if workspaces_root_path in _retagged_workspace_roots:
        return
    try:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            db.retag_kanban_worker_sessions(workspaces_root_path)
        finally:
            db.close()
        _retagged_workspace_roots.add(workspaces_root_path)
    except Exception as exc:
        _log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _default_spawn(
    task: Task,
    workspace: str,
    *,
    board: Optional[str] = None,
) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    Returns the spawned child's PID so the dispatcher can detect crashes
    before the claim TTL expires. The child's completion is still observed
    via the ``complete`` / ``block`` transitions the worker writes itself;
    the PID check is a safety net for crashes, OOM kills, and Ctrl+C.

    ``board`` pins the child's kanban context to that board: the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root env
    vars all resolve to the same board the dispatcher claimed the task
    from. Workers cannot accidentally see other boards.
    """
    import subprocess
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name

    profile_arg = normalize_profile_name(task.assignee)

    prompt = f"work kanban task {task.id}"
    env = dict(os.environ)
    # The dispatcher is detached from every conversation. Its worker must never
    # inherit routing mirrored by a previous gateway turn, even before the first
    # session binds ContextVars in this process.
    from gateway.session_context import _VAR_MAP
    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml
    # (fallback_providers, toolsets, agent settings, etc.) instead of the root
    # config.  Without this, `env = dict(os.environ)` copies only the parent's
    # env, and when the child process starts `hermes -p <name>` the
    # _apply_profile_override() runs *before* hermes_constants is imported.
    # If HERMES_HOME is absent from the child's env, get_hermes_home() falls
    # back to Path.home() / ".hermes" (the DEFAULT profile root), ignoring the
    # profile-specific config entirely.  Fixes profile-scoped fallback_providers
    # being invisible to kanban workers.
    from hermes_cli.profiles import resolve_profile_env
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
    except FileNotFoundError:
        # Profile dir doesn't exist — defer resolution to the CLI's
        # _apply_profile_override() via HERMES_PROFILE (set below).
        # This only happens in test fixtures where the isolated
        # HERMES_HOME never had profiles created.
        pass
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Tag the worker's session so it lands in state.db as `kanban`, not as an
    # untitled `cli` row. A worker is a dispatcher-owned run whose transcript is
    # read on the board and in `hermes kanban log` — it is not a conversation
    # the user started, so every session-browsing surface (desktop sidebar, TUI
    # resume picker, session_search) filters it out by source. Without this the
    # sidebar renders one row per attempt, labeled with the worker's own prompt
    # ("work kanban task t_…").
    env["HERMES_SESSION_SOURCE"] = "kanban"
    # Pin TERMINAL_CWD to the task's workspace so the worker's file tools and
    # context-file loader anchor on the workspace, not whatever cwd the
    # dispatching gateway happened to export. The worker subprocess is already
    # launched with cwd=workspace, but TERMINAL_CWD takes precedence over the
    # process cwd in both file_tools._resolve_base_dir (#41312 — relative
    # write_file paths were landing in the gateway user's home) and
    # build_context_files_prompt (#34619 — workers loaded the dispatching
    # gateway's AGENTS.md instead of the task's). Setting it to the workspace
    # fixes both: the workspace is where the task's work actually happens.
    # Only pin a real, absolute directory — file_tools rejects relative /
    # sentinel TERMINAL_CWD values, so a non-dir workspace must NOT be set
    # here (leave the inherited value rather than write a meaningless one).
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode: the worker reads these and wraps its run in the
    # Ralph-style /goal judge loop (see cli.py quiet-mode path). Only set
    # when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    terminal_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_TIMEOUT"),
    )
    if terminal_timeout is not None:
        env["TERMINAL_TIMEOUT"] = terminal_timeout
    foreground_timeout = _worker_terminal_timeout_env(
        task.max_runtime_seconds,
        env.get("TERMINAL_MAX_FOREGROUND_TIMEOUT"),
    )
    if foreground_timeout is not None:
        env["TERMINAL_MAX_FOREGROUND_TIMEOUT"] = foreground_timeout
    # Pin the shared board + workspaces root the dispatcher resolved, so
    # that even when the worker activates a profile (`hermes -p <name>`
    # rewrites HERMES_HOME), its kanban paths still match the
    # dispatcher's. Belt-and-braces with the `get_default_hermes_root()`
    # resolution in `kanban_home()` — symmetric resolution is the norm,
    # but unusual symlink / Docker layouts are caught here too.
    env["HERMES_KANBAN_DB"] = str(kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(workspaces_root(board=board))
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    # Board slug — the final defense-in-depth pin. If the worker ever
    # resolves kanban paths without the DB / workspaces env vars, the
    # board slug still forces it to the right directory.
    resolved_board = _normalize_board_slug(board) or get_current_board()
    env["HERMES_KANBAN_BOARD"] = resolved_board
    # HERMES_PROFILE is the author the kanban_comment tool defaults to.
    # `hermes -p <assignee>` activates the profile, but the env var is
    # what the tool reads — set it explicitly here so comments are
    # attributed correctly regardless of how the child loads config.
    env["HERMES_PROFILE"] = profile_arg

    # A worker must NEVER boot the interactive TUI: an inherited HERMES_TUI=1
    # or a `display.interface: tui` in the profile's config would send the
    # quiet chat run into the Ink TUI, whose no-TTY bail-out exits 0 without
    # doing the task → "protocol violation" on every attempt. `--cli` is the
    # highest-precedence interface override; dropping the env var covers
    # older hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
        "--cli",
        # Worker subprocesses switch to a profile-scoped HERMES_HOME above,
        # so they see that profile's shell-hook allowlist instead of the
        # dispatcher's root allowlist. Pass --accept-hooks explicitly so
        # profile-local worker sessions still register configured hooks.
        "--accept-hooks",
    ]
    # Per-task force-loaded skills. Each name goes in its own
    # `--skills X` pair rather than a single comma-joined arg: the CLI
    # accepts both forms (action='append' + comma-split), but
    # per-name pairs are easier to read in `ps` output and avoid any
    # quoting ambiguity if a skill name ever contains unusual chars.
    if task.skills:
        for sk in task.skills:
            if sk:
                cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too when the override names one, so the worker
        # resolves the model against the intended backend instead of the
        # profile's configured provider (mixing model X with provider Y is
        # the classic mis-set that stalls a board).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Per-task thinking depth. Independent of the model override — a task can
    # run the profile's own model at a different depth — so this is its own
    # branch, not a nested one.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _resolve_worker_cli_toolsets(env.get("HERMES_HOME"))
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend([
        "chat",
        "-q", prompt,
    ])
    if task.goal_mode:
        # Goal-mode workers must take the fully-quiet single-query path:
        # the kanban goal-loop hook (_run_kanban_goal_loop_q) only runs in
        # cli.py's quiet branch. Without -Q the worker gets exactly one
        # turn, prints text, exits rc=0, and the dispatcher records a
        # protocol violation (incident 2026-06-09 t_d9cbe312).
        cmd.append("-Q")
    # Redirect output to a per-task log under <board-root>/logs/.
    # Anchored at the board root (not the shared kanban root), so
    # `hermes kanban log` on a specific board reads its own file and
    # logs don't collide across boards that happen to share task ids.
    log_dir = worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")
    try:
        proc = subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _IS_WINDOWS else 0,
        )
    except FileNotFoundError:
        log_f.close()
        raise RuntimeError(
            "`hermes` executable not found on PATH. "
            "Install Hermes Agent or activate its venv before running the kanban dispatcher."
        )
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.
    return proc.pid


# ---------------------------------------------------------------------------
# Long-lived dispatcher daemon
# ---------------------------------------------------------------------------

def run_daemon(
    *,
    interval: float = 60.0,
    max_spawn: Optional[int] = None,
    failure_limit: int = DEFAULT_SPAWN_FAILURE_LIMIT,
    stop_event=None,
    on_tick=None,
) -> None:
    """Run the dispatcher in a loop until interrupted.

    Calls :func:`dispatch_once` every ``interval`` seconds. Exits cleanly
    on SIGINT / SIGTERM so ``hermes kanban daemon`` is systemd-friendly.
    ``stop_event`` (a :class:`threading.Event`) and ``on_tick`` (a
    callable receiving the :class:`DispatchResult`) are test hooks.

    Each tick resolves ``kanban.max_in_progress`` (explicit config, else
    the memory-derived default) exactly like the gateway-embedded
    dispatcher and ``hermes kanban dispatch`` — the standalone daemon must
    not be the one uncapped entry point (OOF-30).
    """
    import signal
    import threading

    if stop_event is None:
        stop_event = threading.Event()

    def _handle(_signum, _frame):
        stop_event.set()

    # Install handlers only when running on the main thread — tests call
    # this inline from worker threads and signal() would raise there.
    if threading.current_thread() is threading.main_thread():
        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    signal.signal(sig, _handle)
                except (ValueError, OSError):
                    pass

    while not stop_event.is_set():
        try:
            # Resolve the global concurrency cap the same way the gateway
            # dispatcher and `hermes kanban dispatch` do (OOF-30): explicit
            # kanban.max_in_progress wins, otherwise the memory-derived
            # default applies. The standalone daemon previously passed no
            # cap at all — the shipped systemd path could still fan out an
            # entire backlog in one tick even with the derived default in
            # place everywhere else. Re-resolved every tick (config load is
            # mtime-cached) so operator edits apply without a restart.
            max_in_progress = resolve_max_in_progress(
                configured_max_in_progress()
            )
            with contextlib.closing(connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
                    max_in_progress=max_in_progress,
                    failure_limit=failure_limit,
                )
            if on_tick is not None:
                try:
                    on_tick(res)
                except Exception:
                    pass
        except Exception:
            # Don't let any single tick kill the daemon.
            import traceback
            traceback.print_exc()
        stop_event.wait(timeout=interval)


# ---------------------------------------------------------------------------
# Worker context builder (what a spawned worker sees)
# ---------------------------------------------------------------------------

def build_worker_context(conn: sqlite3.Connection, task_id: str) -> str:
    """Everything a worker should read about its task: header, body,
    attachments, prior attempts, done-parent handoffs, the assignee's recent
    work, comments. Lists are tail-capped and fields char-capped
    (``_CTX_MAX_*``) so the prompt stays bounded on pathological boards."""
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")
    # One clock reading so every relative age in this rendering agrees.
    now = int(time.time())
    lines: list[str] = []
    _ctx_header(lines, task)
    _ctx_attachments(lines, list_attachments(conn, task_id))
    _ctx_prior_attempts(lines, conn, task_id, now)
    _ctx_parent_results(lines, conn, task_id, now)
    _ctx_role_history(lines, conn, task, now)
    _ctx_comments(lines, list_comments(conn, task_id), now)
    return "\n".join(lines).rstrip() + "\n"


def _ctx_cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
    """Truncate to ``limit`` chars with a visible ellipsis."""
    if not s:
        return ""
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"


def _ctx_stamp(ts: int, now: int) -> str:
    """``YYYY-MM-DD HH:MM`` plus a relative age when one is available."""
    disp = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
    age = _relative_age(ts, now)
    return f"{disp}, {age}" if age else disp


def _ctx_metadata_line(metadata: Any) -> Optional[str]:
    if not metadata:
        return None
    try:
        return f"_metadata_: `{_ctx_cap(json.dumps(metadata, ensure_ascii=False, sort_keys=True))}`"
    except Exception:
        return None


def _ctx_tail(items: list, cap: int, noun: str) -> tuple[list, Optional[str]]:
    """Keep the newest ``cap`` items; describe the omitted head, if any."""
    omitted = max(0, len(items) - cap)
    if not omitted:
        return items, None
    return items[-cap:], (
        f"_({omitted} earlier {noun}{'s' if omitted != 1 else ''} "
        f"omitted; showing most recent {cap})_"
    )


def _ctx_header(lines: list[str], task: Task) -> None:
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {task.workspace_kind} @ {task.workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _worker_terminal_timeout_env(
            task.max_runtime_seconds, os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if task.branch_name:
        lines.append(f"Branch:   {task.branch_name}")
    lines.append("")
    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_ctx_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")


def _ctx_attachments(lines: list[str], attachments: list[Attachment]) -> None:
    """Absolute on-disk paths so the worker's file tools read them directly
    (remote terminal backends need the attachments dir mounted)."""
    if not attachments:
        return
    lines.append("## Attachments")
    lines.append(
        "Files attached to this task. Read them with the file/terminal "
        "tools at the absolute paths below:"
    )
    for att in attachments:
        size_kb = max(1, (att.size + 1023) // 1024) if att.size else 0
        size_str = f", {size_kb} KB" if size_kb else ""
        ctype = f", {att.content_type}" if att.content_type else ""
        lines.append(f"- `{att.filename}`{ctype}{size_str} → `{att.stored_path}`")
    lines.append("")


def _ctx_prior_attempts(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Closed runs on this task (the active run is this worker), newest
    ``_CTX_MAX_PRIOR_ATTEMPTS`` in full, older ones as a one-line marker."""
    all_prior = [r for r in list_runs(conn, task_id) if r.ended_at is not None]
    shown, omitted_note = _ctx_tail(all_prior, _CTX_MAX_PRIOR_ATTEMPTS, "attempt")
    if not shown:
        return
    first_shown_idx = len(all_prior) - len(shown) + 1
    lines.append("## Prior attempts on this task")
    if omitted_note:
        lines.append(omitted_note)
    for offset, run in enumerate(shown):
        profile = run.profile or "(unknown)"
        outcome = run.outcome or run.status
        lines.append(
            f"### Attempt {first_shown_idx + offset} — {outcome} ({profile}, {_ctx_stamp(run.started_at, now)})"
        )
        if run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        if run.error and run.error.strip():
            lines.append(f"_error_: {_ctx_cap(run.error)}")
        meta_line = _ctx_metadata_line(run.metadata)
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_parent_results(lines: list[str], conn: sqlite3.Connection, task_id: str, now: int) -> None:
    """Done-parent handoffs: newest ``completed`` run's summary+metadata,
    falling back to ``task.result`` for pre-runs-table data. Stamped with a
    relative age so the worker re-verifies stale upstream results."""
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id", (task_id,),
    ).fetchall()
    wrote_header = False
    for pid in (r["parent_id"] for r in parent_rows):
        pt = get_task(conn, pid)
        if not pt or pt.status != "done":
            continue
        runs = [r for r in list_runs(conn, pid) if r.outcome == "completed"]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        run = runs[0] if runs else None
        if not wrote_header:
            lines.append("## Parent task results")
            lines.append(
                "_Handoffs from upstream tasks, captured when each parent "
                "completed (see age below). These are point-in-time "
                "snapshots, not live state — if a result drives your "
                "current work and it's not recent, re-verify against the "
                "source before acting on it as current._"
            )
            wrote_header = True
        done_ts = run.ended_at if run is not None and run.ended_at else (pt.completed_at or None)
        age = _relative_age(done_ts, now)
        lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))
        if run is not None and run.summary and run.summary.strip():
            lines.append(_ctx_cap(run.summary))
        elif pt.result:
            lines.append(_ctx_cap(pt.result))
        else:
            lines.append("(no result recorded)")
        meta_line = _ctx_metadata_line(run.metadata) if run is not None else None
        if meta_line:
            lines.append(meta_line)
        lines.append("")


def _ctx_role_history(lines: list[str], conn: sqlite3.Connection, task: Task, now: int) -> None:
    """The assignee's 5 most recent completed runs on OTHER tasks — implicit
    role continuity without wiring anything into SOUL.md / MEMORY.md."""
    if not task.assignee:
        return
    role_rows = conn.execute(
        "SELECT t.id, t.title, r.summary, r.ended_at "
        "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
        "WHERE r.profile = ? AND r.task_id != ? "
        "  AND r.outcome = 'completed' "
        "ORDER BY r.ended_at DESC LIMIT 5", (task.assignee, task.id),
    ).fetchall()
    if not role_rows:
        return
    lines.append(f"## Recent work by @{task.assignee}")
    for row in role_rows:
        first = _first_line(row["summary"], 200) or "(no summary)"
        lines.append(
            f"- {row['id']} — {row['title']} ({_ctx_stamp(int(row['ended_at']), now)}): {first}"
        )
    lines.append("")


def _ctx_comments(lines: list[str], comments: list[Comment], now: int) -> None:
    """Newest ``_CTX_MAX_COMMENTS`` comments. The explicit "comment from
    worker" framing stops an operator-controlled HERMES_PROFILE like
    "hermes-system" being read as a system directive above an
    attacker-influenceable body (defense-in-depth)."""
    shown, omitted_note = _ctx_tail(comments, _CTX_MAX_COMMENTS, "comment")
    if not shown:
        return
    lines.append("## Comment thread")
    if omitted_note:
        lines.append(omitted_note)
    for c in shown:
        # Render author with explicit "comment from worker" framing so operator-controlled HERMES_PROFILE
        # values like "hermes-system" or "operator" can't be misread by the next worker as a system
        # directive above the (attacker-influenceable) comment body. Defense-in-depth — the LLM-controlled
        # author-forgery surface was already closed in #22435. See #22452.
        safe_author = (c.author or "").replace("`", "")
        lines.append(f"comment from worker `{safe_author}` at {_ctx_stamp(c.created_at, now)}:")
        lines.append(_ctx_cap(c.body, _CTX_MAX_COMMENT_BYTES))
        lines.append("")


# --- Stats + SLA helpers ---

def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts and the oldest ``ready`` age (staleness signal)."""
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee = _counts_by_assignee(conn)

    oldest_row = conn.execute(
        "SELECT MIN(created_at) AS ts FROM tasks WHERE status = 'ready'"
    ).fetchone()
    now = int(time.time())
    oldest_ready_age = (
        (now - int(oldest_row["ts"]))
        if oldest_row and oldest_row["ts"] is not None else None
    )

    return {
        "by_status": by_status,
        "by_assignee": by_assignee,
        "oldest_ready_age_seconds": oldest_ready_age,
        "now": now,
    }


def _counts_by_assignee(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """``{assignee: {status: n}}`` over non-archived tasks."""
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])
    return counts


def _to_epoch(val) -> Optional[int]:
    """Epoch seconds from int/float/numeric string/ISO-8601; None for empty/invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    # ISO-8601 fallback (e.g. '2026-05-10T15:00:00Z')
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, OSError):
        return None


def task_age(task: Task) -> dict:
    """Return age metrics for a single task. All values are seconds or None."""
    now = int(time.time())
    _c = _to_epoch(task.created_at)
    _s = _to_epoch(task.started_at)
    _co = _to_epoch(task.completed_at)
    return {
        "created_age_seconds": now - _c if _c is not None else None,
        "started_age_seconds": now - _s if _s is not None else None,
        "time_to_complete_seconds": _co - (_s or _c) if _co is not None else None,
    }


# --- Retention + garbage collection ---

def gc_events(conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600) -> int:
    """Prune old done/archived events, retaining decomposition identity until task deletion."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND kind != 'decomposed' AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))", (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(*, older_than_seconds: int = 30 * 24 * 3600, board: Optional[str] = None) -> int:
    """Delete worker log files older than the cutoff on one board; returns the count."""
    log_dir = worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        with contextlib.suppress(OSError):
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
    return removed


# --- Worker log accessor ---

def worker_log_path(task_id: str, *, board: Optional[str] = None) -> Path:
    """Worker log path (may not exist). The dispatcher always passes ``board``
    explicitly to avoid resolution ambiguity."""
    return worker_logs_dir(board=board) / f"{task_id}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None, board: Optional[str] = None,
) -> Optional[str]:
    """Worker log text (last ``tail_bytes`` when set); None when the file is missing."""
    path = worker_log_path(task_id, board=board)
    if not path.exists():
        return None
    try:
        if tail_bytes is None:
            return path.read_text(encoding="utf-8", errors="replace")
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                # Skip the partial first line unless the window has no newline
                # at all (readline() would eat everything).
                probe = f.tell()
                if not f.readline().endswith(b"\n") and f.tell() >= size:
                    f.seek(probe)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return None


# --- Assignee enumeration (known profiles + per-profile board stats) ---

def list_profiles_on_disk() -> list[str]:
    """Profiles with a ``config.yaml`` plus the implicit ``default``; reads paths
    directly to avoid importing ``hermes_cli.profiles`` at startup."""
    try:
        from hermes_constants import get_default_hermes_root
        default_root = get_default_hermes_root()
        profiles_dir = default_root / "profiles"
    except Exception:
        return []

    names: set[str] = set()
    if default_root.exists():
        names.add("default")
    if profiles_dir.is_dir():
        try:
            names.update(e.name for e in profiles_dir.iterdir() if e.is_dir() and (e / "config.yaml").is_file())
        except OSError:
            pass
    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict]:
    """``{"name", "on_disk", "counts"}`` for every on-disk profile or task
    assignee, so a fresh profile appears in pickers before it has a task."""
    on_disk = set(list_profiles_on_disk())
    counts = _counts_by_assignee(conn)
    return [
        {"name": name, "on_disk": name in on_disk, "counts": counts.get(name, {})}
        for name in sorted(on_disk | set(counts))
    ]


# --- Runs (attempt history on a task) ---

def list_runs(
    conn: sqlite3.Connection, task_id: str, *, include_active: bool = True,
    state_type: Optional[str] = None, state_name: Optional[str] = None,
) -> list[Run]:
    """Runs in start order; ``include_active=False`` = closed only; ``state_type``
    (``status``/``outcome``) + ``state_name`` filter together."""
    if (state_type is None) ^ (state_name is None):
        raise ValueError("state_type and state_name must both be set or both omitted")
    if state_type is not None and state_type not in ("status", "outcome"):
        raise ValueError("state_type must be 'status' or 'outcome'")
    q = "SELECT * FROM task_runs WHERE task_id = ?"
    params: list[Any] = [task_id]
    if not include_active:
        q += " AND ended_at IS NOT NULL"
    if state_type is not None:
        q += f" AND {state_type} = ?"
        params.append(state_name)
    q += " ORDER BY started_at ASC, id ASC"
    rows = conn.execute(q, params).fetchall()
    return [Run.from_row(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> Optional[Run]:
    row = conn.execute("SELECT * FROM task_runs WHERE id = ?", (int(run_id),)).fetchone()
    return Run.from_row(row) if row else None


def latest_run(conn: sqlite3.Connection, task_id: str) -> Optional[Run]:
    """Return the most recent run regardless of outcome (active or closed)."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Newest non-empty run summary, or None. Workers hand off via ``summary`` and
    leave ``tasks.result`` NULL, so views need this or a done task looks empty."""
    row = conn.execute(
        "SELECT summary FROM task_runs "
        "WHERE task_id = ? AND summary IS NOT NULL AND summary != '' "
        "ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return row["summary"] if row else None


def latest_summaries(conn: sqlite3.Connection, task_ids: Iterable[str]) -> dict[str, str]:
    """``{task_id: newest non-empty run summary}`` in one query (window function,
    SQLite >= 3.25); tasks without a summary are omitted."""
    ids = list(task_ids)
    if not ids:
        return {}
    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"""
        SELECT task_id, summary FROM (
            SELECT task_id, summary,
                   ROW_NUMBER() OVER (
                       PARTITION BY task_id
                       ORDER BY COALESCE(ended_at, started_at) DESC, id DESC
                   ) AS rn
              FROM task_runs
             WHERE task_id IN ({placeholders})
               AND summary IS NOT NULL AND summary != ''
        ) WHERE rn = 1
        """,
        ids,
    ).fetchall()
    return {r["task_id"]: r["summary"] for r in rows}


# --- Split modules (imported at the tail: they import this module as ``_kb``) ---
from hermes_cli.kanban_db_connect import (  # noqa: E402
    _INITIALIZED_PATHS,
    init_db,
    write_txn,
)
from hermes_cli.kanban_db_workspace import (  # noqa: E402
    _cleanup_workspace,
    _is_managed_scratch_path,
    _managed_scratch_path_info,
    _scratch_workspace,
)
from hermes_cli.kanban_db_dispatch import (  # noqa: E402
    DEFAULT_FAILURE_LIMIT,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    DispatchResult,
    _clear_failure_counter,
    _defer_reclaim_for_live_worker,
    _pid_alive,
    _terminate_reclaimed_worker,
    _worker_survived_termination,
    _worker_terminal_timeout_env,
)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Mapping  # noqa: F401,E402
from dataclasses import field  # noqa: F401,E402
import hashlib  # noqa: F401,E402
import random  # noqa: F401,E402
import shutil  # noqa: F401,E402
import threading  # noqa: F401,E402

DEFAULT_SPAWN_FAILURE_LIMIT = DEFAULT_FAILURE_LIMIT

def parent_results(conn: sqlite3.Connection, task_id: str) -> list[tuple[str, Optional[str]]]:
    """Return ``(parent_id, result)`` for every done parent of ``task_id``."""
    rows = conn.execute(
        """
        SELECT t.id AS id, t.result AS result
        FROM tasks t
        JOIN task_links l ON l.parent_id = t.id
        WHERE l.child_id = ? AND t.status = 'done'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [(r["id"], r["result"]) for r in rows]


_PLUGIN_COMPAT_LAZY = {
    'DEFAULT_BUSY_TIMEOUT_MS': ('hermes_cli.kanban_db_connect', 'DEFAULT_BUSY_TIMEOUT_MS'),
    'DEFAULT_LOG_BACKUP_COUNT': ('hermes_cli.kanban_db_dispatch', 'DEFAULT_LOG_BACKUP_COUNT'),
    'DEFAULT_LOG_ROTATE_BYTES': ('hermes_cli.kanban_db_dispatch', 'DEFAULT_LOG_ROTATE_BYTES'),
    'DERIVED_MAX_IN_PROGRESS_CEILING': ('hermes_cli.kanban_db_dispatch', 'DERIVED_MAX_IN_PROGRESS_CEILING'),
    'DERIVED_MAX_IN_PROGRESS_FLOOR': ('hermes_cli.kanban_db_dispatch', 'DERIVED_MAX_IN_PROGRESS_FLOOR'),
    'KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS': ('hermes_cli.kanban_db_dispatch', 'KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS'),
    'KanbanDbCorruptError': ('hermes_cli.kanban_db_connect', 'KanbanDbCorruptError'),
    'MEMORY_GUARD_MB_PER_WORKER': ('hermes_cli.kanban_db_dispatch', 'MEMORY_GUARD_MB_PER_WORKER'),
    'RepairResult': ('hermes_cli.kanban_db_connect', 'RepairResult'),
    'add_notify_sub': ('hermes_cli.kanban_db_notify', 'add_notify_sub'),
    'advance_notify_cursor': ('hermes_cli.kanban_db_notify', 'advance_notify_cursor'),
    'check_respawn_guard': ('hermes_cli.kanban_db_dispatch', 'check_respawn_guard'),
    'claim_unseen_events_for_sub': ('hermes_cli.kanban_db_notify', 'claim_unseen_events_for_sub'),
    'configured_max_in_progress': ('hermes_cli.kanban_db_dispatch', 'configured_max_in_progress'),
    'connect': ('hermes_cli.kanban_db_connect', 'connect'),
    'connect_closing': ('hermes_cli.kanban_db_connect', 'connect_closing'),
    'count_notify_subs': ('hermes_cli.kanban_db_notify', 'count_notify_subs'),
    'count_running_tasks': ('hermes_cli.kanban_db_dispatch', 'count_running_tasks'),
    'count_running_tasks_other_boards': ('hermes_cli.kanban_db_dispatch', 'count_running_tasks_other_boards'),
    'derive_default_max_in_progress': ('hermes_cli.kanban_db_dispatch', 'derive_default_max_in_progress'),
    'detect_crashed_workers': ('hermes_cli.kanban_db_dispatch', 'detect_crashed_workers'),
    'detect_stale_running': ('hermes_cli.kanban_db_dispatch', 'detect_stale_running'),
    'dispatch_once': ('hermes_cli.kanban_db_dispatch', 'dispatch_once'),
    'enforce_max_runtime': ('hermes_cli.kanban_db_dispatch', 'enforce_max_runtime'),
    'has_spawnable_ready': ('hermes_cli.kanban_db_dispatch', 'has_spawnable_ready'),
    'has_spawnable_review': ('hermes_cli.kanban_db_dispatch', 'has_spawnable_review'),
    'heartbeat_worker': ('hermes_cli.kanban_db_dispatch', 'heartbeat_worker'),
    'list_notify_subs': ('hermes_cli.kanban_db_notify', 'list_notify_subs'),
    'purge_stale_done_notify_subs': ('hermes_cli.kanban_db_notify', 'purge_stale_done_notify_subs'),
    'reap_worker_zombies': ('hermes_cli.kanban_db_dispatch', 'reap_worker_zombies'),
    'reconcile_orphaned_running': ('hermes_cli.kanban_db_dispatch', 'reconcile_orphaned_running'),
    'remove_notify_sub': ('hermes_cli.kanban_db_notify', 'remove_notify_sub'),
    'repair_db': ('hermes_cli.kanban_db_connect', 'repair_db'),
    'resolve_max_in_progress': ('hermes_cli.kanban_db_dispatch', 'resolve_max_in_progress'),
    'resolve_workspace': ('hermes_cli.kanban_db_workspace', 'resolve_workspace'),
    'review_dispatch_enabled': ('hermes_cli.kanban_db_dispatch', 'review_dispatch_enabled'),
    'rewind_notify_cursor': ('hermes_cli.kanban_db_notify', 'rewind_notify_cursor'),
    'run_daemon': ('hermes_cli.kanban_db_dispatch', 'run_daemon'),
    'set_branch_name': ('hermes_cli.kanban_db_workspace', 'set_branch_name'),
    'set_workspace_path': ('hermes_cli.kanban_db_workspace', 'set_workspace_path'),
    'unseen_events_for_sub': ('hermes_cli.kanban_db_notify', 'unseen_events_for_sub'),
    'worker_log_rotation_config': ('hermes_cli.kanban_db_dispatch', 'worker_log_rotation_config'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
