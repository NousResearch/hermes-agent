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
from dataclasses import dataclass
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

VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient", "cost_cap", "operator_hold"}

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
    # FLEET buckets — a tick that only cost-capped or parent-gated still did
    # something, and the observer must see it.
    "parent_gated", "cost_capped",
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
    # Per-card dollar cap on cumulative worker spend (state.db session costs).
    # ``None`` = uncapped (backward compat); when set and exceeded the
    # dispatcher SIGTERMs the worker and BLOCKs the card with kind
    # ``cost_cap`` (never retried — routes to the jobsy triage lane).
    max_cost: Optional[float] = None
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
    # fleet: per-card dollar cap (applied via effective_max_cost at create)
    "max_cost",
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
    -- Per-card dollar cap on cumulative worker spend. REAL, nullable; when
    -- set, the dispatcher compares the card's cumulative session cost
    -- (state.db session_model_usage) against this each heartbeat tick and,
    -- if exceeded, SIGTERMs the worker and BLOCKs the card with kind
    -- ``cost_cap`` (never retried — it routes to the jobsy triage lane).
    -- NULL = uncapped (backward compat).
    max_cost             REAL,
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

-- Global dispatch circuit breaker (fleet-wide provider-outage gate). A single
-- row (id=1) records whether the dispatcher is paused because a provider-side
-- outage (e.g. OpenRouter 403/429/5xx at worker spawn) tripped it. When
-- state='open' the dispatcher spawns NO workers fleet-wide; a cheap probe
-- every CIRCUIT_PROBE_INTERVAL_SECONDS closes it again. The reason field is
-- the detection signature (a serialized signature string); the outbox state
-- lives in a sibling JSON file so the no_agent alert cron can deliver it.
CREATE TABLE IF NOT EXISTS dispatch_circuit (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    state          TEXT NOT NULL DEFAULT 'closed',   -- 'open' | 'closed'
    reason         TEXT NOT NULL DEFAULT '',
    opened_at      INTEGER,
    last_probe_at   INTEGER,
    probe_count     INTEGER NOT NULL DEFAULT 0
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
    max_runtime_seconds: Optional[int] = None, max_cost: Optional[float] = None,
    skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None, model_override: Optional[str] = None,
    provider_override: Optional[str] = None, reasoning_effort: Optional[str] = None,
    goal_mode: bool = False, goal_max_turns: Optional[int] = None, initial_status: str = "running",
    block_kind: Optional[str] = None,
    points: Optional[int] = None,
    session_id: Optional[str] = None, board: Optional[str] = None, project_id: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    creator_task_id: Optional[str] = None,
    completion_contract: Optional[str] = None,
    _assignee_parked: Optional[dict] = None,
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

    Fleet additions:
    ``max_cost`` caps cumulative worker spend in USD: when the card's session
    costs in state.db exceed it the dispatcher SIGTERMs the worker and BLOCKs
    the card with kind ``cost_cap`` (never retried — routes to the jobsy triage
    lane). ``None`` means uncapped; the configured default and the new-card
    ceiling are applied here via ``effective_max_cost`` so every caller is
    covered. ``block_kind`` types a hold created directly in ``blocked``.
    ``points`` is the charter §4 estimate: None -> the conservative
    ``auto-points`` placeholder comment is written so no card leaves the mint
    path unestimated; an explicit >=1 value writes that estimate instead.
    ``_assignee_parked`` is an out-parameter the CLI reads to report a card
    parked in triage because its assignee is not a real profile.
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
    if block_kind is not None:
        if initial_status != "blocked":
            raise ValueError("block_kind requires initial_status='blocked'")
        if block_kind not in VALID_BLOCK_KINDS:
            raise ValueError(f"block kind must be one of {sorted(VALID_BLOCK_KINDS)}")
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
    if max_cost is not None:
        try:
            max_cost = float(max_cost)
        except (TypeError, ValueError):
            raise ValueError(f"max_cost must be a number, got {max_cost!r}")
        if max_cost < 0:
            raise ValueError("max_cost must be >= 0")
    if points is not None:
        try:
            points = int(points)
        except (TypeError, ValueError):
            raise ValueError(f"points must be an integer, got {points!r}")
        if points < 1:
            # 0 is falsy, and the ledger's coverage metric is
            # ``[r for r in rows if r.get("points")]`` — a 0 would read as
            # "no estimate" and silently defeat the point of writing it.
            raise ValueError("points must be >= 1 (0 reads as 'no estimate')")

    project_id, project_obj, project_repo, workspace_kind = _resolve_project_link(
        conn, project_id, project_source_task_id, workspace_kind, workspace_path
    )

    # 2026-09-06 (W1): a tenant's code lands in that tenant's repo whoever
    # mints the card. When no Project resolved (creator's projects.db is
    # per-profile and usually empty) and the card would otherwise be an empty
    # ``scratch`` folder, anchor it to the fleet tenant map instead. An
    # explicit ``dir`` or ``worktree`` request is left alone.
    if project_obj is None and tenant and workspace_kind == "scratch":
        _tp = _tenant_project(tenant)
        if _tp is not None:
            from hermes_cli import projects_db as _pdb

            project_obj = _tp
            project_id = _tp.id
            workspace_kind = "worktree"
            if workspace_path is None and _tp.primary_path:
                project_repo = str(_tp.primary_path)

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

                # Assignee guard: an unknown assignee (not a real profile) is
                # parked in triage for the PM (Jobsy) to accept/reject — it is
                # NEVER dispatched, which would strand the card. `default` is a
                # valid assignee (Agent Smith). The bogus name is preserved on
                # the row and recorded in a comment so the triage lane sees the
                # original intent, per the 2026-08-30 phantom-28-card incident.
                assignee_unknown = assignee is not None and not _assignee_is_known(
                    assignee
                )
                # An explicit initial_status="blocked" (human-ops review) takes
                # precedence — a blocked card is never dispatched anyway, and we
                # must not clobber VALID_INITIAL_STATUSES semantics. A blocked
                # + unknown-assignee card stays blocked, so it must NOT get the
                # "parked in triage" comment below (which would be a lie).
                assignee_parked_in_triage = (
                    assignee_unknown and task_status != "blocked"
                )
                if assignee_parked_in_triage:
                    task_status = "triage"
                    if _assignee_parked is not None:
                        _assignee_parked["assignee"] = assignee

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
                        max_runtime_seconds, max_cost,
                        skills, max_retries, model_override, provider_override,
                        reasoning_effort,
                        goal_mode, goal_max_turns, session_id, completion_contract
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, title.strip(), body, assignee, task_status, priority,
                        created_by, now, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        _opt_int(max_runtime_seconds),
                        # WeRoll cost policy 2026-09-02: apply the configured
                        # default and the $1.00 new-card ceiling HERE so every
                        # caller of create_task is covered — CLI, kanban_create
                        # tool, dashboard API, Deploy: follow-ups and swarm alike.
                        effective_max_cost(max_cost),
                        json.dumps(skills_list) if skills_list is not None else None,
                        _opt_int(max_retries), model_override, provider_override, reasoning_effort,
                        1 if goal_mode else 0, _opt_int(goal_max_turns), session_id, completion_contract,
                    ),
                )
                for pid in parents:
                    _link(conn, pid, task_id)
                # Charter §4 estimation at the mint path (est-mintpath-20260911):
                # EVERY card leaves create_task carrying a points estimate, so
                # the CLI, the kanban_create tool, the dashboard API, the
                # Deploy: follow-ups and the swarm are all covered by
                # construction — and the machine path is covered without asking
                # the machine to remember. Written first so any later estimate
                # comment wins (the cost ledger takes the last match).
                _insert_comment(
                    conn, task_id, "(system)", auto_points_comment(points), now,
                )
                if assignee_parked_in_triage:
                    conn.execute(
                        "INSERT INTO task_comments "
                        "(task_id, author, body, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            task_id,
                            "(system)",
                            f"Created with unknown assignee {assignee!r} — "
                            "not a real profile, so this card is parked in triage "
                            "for the PM to accept or reject. Transfer to a valid "
                            "assignee to dispatch.",
                            now,
                        ),
                    )
                # Notify-sub inheritance (ACK-edge: the originating channel
                # still hears about a child that BLOCKs, not just the final
                # fan-in) is handled by the single-owner helper below —
                # _inherit_notify_subs copies every routing/delivery column.
                if task_status == "blocked" and block_kind:
                    # Typed hold at creation: the kind is what escalation-watch,
                    # fleet-preflight and stalled-card-watch key on, and the
                    # `blocked` event is what dates the hold.
                    conn.execute(
                        "UPDATE tasks SET block_kind = ? WHERE id = ?", (block_kind, task_id)
                    )
                    _append_event(
                        conn, task_id, "blocked",
                        {"reason": f"created with block_kind={block_kind}", "kind": block_kind,
                         "recurrences": 0, "source_status": "created"},
                    )
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
    # operator_hold (2026-09-03) is sticky by definition — even when it was set
    # by hand in SQL with no event row (as the 09-02 park scripts did), it must
    # never auto-promote. Only a human unblock, which clears the status, ends it.
    kind_row = conn.execute(
        "SELECT block_kind FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if kind_row and kind_row["block_kind"] == "operator_hold":
        return True
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
    # An approved platform card (worktree under this hermes-agent repo) is
    # approved-but-undeployed until the branch actually lands on main and the
    # gateway restarts. Auto-create ONE deploy follow-up so that gap closes.
    # Idempotency-keyed on the approved task id, so a repeat completion of the
    # same card cannot spawn a second deploy card.
    try:
        _maybe_create_deploy_followup(conn, task_id, metadata)
    except Exception:
        # Creation must never block the completion itself.
        _log.exception(
            "deploy follow-up creation failed for completed task %s", task_id
        )
    # Clean up the scratch workspace and any stale tmux session for the worker.
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
    """Transition ``blocked``/``scheduled``/``triage`` to a resumable phase.

    ``blocked``/``scheduled`` return to their prior phase (or ``ready``).
    ``triage`` is *accepted*: parked triage tasks (e.g. decision-shaped
    children the auto-decomposer routed to triage) promote ``triage -> todo``,
    and ``recompute_ready`` lifts them to ``ready`` on the next tick (or
    immediately if parent-free). This is the acceptance primitive for a parked
    decision — deliberately body-less, so a PM ratifies the decision without an
    LLM fabricating one (the phantom-PM-decision class).

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.
    """
    now = int(time.time())
    with write_txn(conn):
        current = conn.execute(
            "SELECT status FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if current is not None and current["status"] == "triage":
            # Accept a parked triage task: promote triage -> todo (parent
            # gating + ready promotion handled by recompute_ready). No resume
            # phase to reconstruct — a triage card has no prior lane.
            _reclaim_dangling_run(
                conn, task_id, statuses=("triage",), now=now,
                note="invariant recovery on triage accept",
            )
            cur = conn.execute(
                "UPDATE tasks SET status = 'todo', current_run_id = NULL, "
                "consecutive_failures = 0, last_failure_error = NULL "
                "WHERE id = ? AND status = 'triage'",
                (task_id,),
            )
            if cur.rowcount != 1:
                return False
            _append_event(
                conn, task_id, "accepted",
                {"status": "todo", "resume_status": "triage"},
            )
            return True
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
        task_row = conn.execute(
            "SELECT status, block_kind FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if task_row is not None and task_row["status"] == "blocked" and task_row["block_kind"] == "operator_hold":
            return False
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


# --- Worker context builder (what a spawned worker sees) ---

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




# ===========================================================================
# FLEET ADDITIONS (WeRoll) — not present upstream.
# Kept in this module so their internal references resolve unchanged; the
# split modules reach them late-bound via ``_kb.<name>``.
# ===========================================================================

# ---------------------------------------------------------------------------
# Global dispatch circuit breaker (fleet-wide provider-outage gate)
# ---------------------------------------------------------------------------
# On 2026-08-30 the OpenRouter monthly-key 403 bounced every worker at spawn
# inside ~2s (rc=0, protocol violation), and the dispatcher retried through 22
# crashed runs before a human parked the cards. A per-card failure budget can't
# see this — each retry is "legitimately" from one card's viewpoint. So the
# fix is a FLEET-wide breaker: when a worker exits fast+clean with a provider
# 4xx/5xx in its log, pause ALL dispatch, alert ONCE, and probe-resume.
#
# A worker that exits within this window with rc=0 and a provider 4xx/5xx in
# its log is the provider-outage signature — not a genuine worker fault.
CIRCUIT_FAST_EXIT_SECONDS = 10


# Probe cadence between paused probes ("every 5-10 min"). Overridable so tests
# and operators can tighten/loosen it.
DEFAULT_CIRCUIT_PROBE_INTERVAL_SECONDS = 300


# Hard timeout for a single probe request (bounded network call).
DEFAULT_CIRCUIT_PROBE_TIMEOUT_SECONDS = 30


# State row id for the singleton dispatch_circuit row.
CIRCUIT_ROW_ID = 1


# Outbox file for alerts delivered by the no_agent cron (same channel as
# budget-watch). Relative to the board root; "default" board keeps the legacy
# kanban root.
CIRCUIT_OUTBOX_FILENAME = "circuit-outbox.json"


def _resolve_circuit_probe_interval_seconds() -> int:
    """Return the probe interval (seconds) between paused probes.

    Reads ``HERMES_KANBAN_CIRCUIT_PROBE_INTERVAL_SECONDS``; falls back to
    ``DEFAULT_CIRCUIT_PROBE_INTERVAL_SECONDS``. A value of 0 disables probes
    (the breaker stays paused until manually closed).
    """
    raw = os.environ.get(
        "HERMES_KANBAN_CIRCUIT_PROBE_INTERVAL_SECONDS", ""
    ).strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_CIRCUIT_PROBE_INTERVAL_SECONDS


def _worktree_holds_unmerged(worktree_path: str) -> bool:
    """Return True when a task worktree has unmerged work: a dirty tree OR
    commits not reachable from any remote-tracking ref.

    Guard 2 of the auto-decomposer fix (t_c52b9bc3): when a decomposition
    happens, the implementation child MUST inherit the parent's worktree if it
    holds dirty/unpushed work — never mint a fresh worktree while the parent's
    holds the only copy of a half-done diff. Reuses the same fail-safe cli
    predicates the teardown path uses (``_cleanup_worktree_workspace``), so the
    verdict here matches the verdict there. Any error resolves to ``True``
    (preserve the worktree / inherit it) — fail-safe, never silently discard.
    """
    try:
        # 2026-09-11: was ``from cli import ...`` — cli stopped exporting _worktree_is_dirty when
        # upstream moved both predicates to hermes_cli.worktree_ops, so this import always failed
        # and the guard always answered True (every decomposed child inherited the parent tree).
        from hermes_cli.worktree_ops import _worktree_has_unpushed_commits, _worktree_is_dirty
    except Exception:
        _log.warning(
            "decompose: cannot import cli worktree predicates for %s — "
            "assuming unmerged work (fail-safe inherit)",
            worktree_path,
        )
        return True
    try:
        if _worktree_is_dirty(worktree_path):
            return True
        if _worktree_has_unpushed_commits(worktree_path):
            return True
    except Exception:
        _log.warning(
            "decompose: worktree predicate failed for %s — assuming unmerged "
            "work (fail-safe inherit)",
            worktree_path,
        )
        return True
    return False


def _assignee_is_known(assignee: Optional[str]) -> bool:
    """Whether ``assignee`` names a real (spawnable) Hermes profile.

    ``default`` is valid — it is the built-in root profile (Agent Smith).
    Uses :func:`hermes_cli.profiles.profile_exists`, the same guard the
    dispatcher consults at spawn time so create-time and dispatch-time agree
    on what is a phantom. Local import here stays local: it lives in a module
    that peers import eagerly and would otherwise risk an import cycle.

    2026-09-10 (catch-up merge): the ONLY thing added to the fleet's original
    is the two ``except`` arms. A guard must not become a crash surface: if the
    profile machinery cannot be imported or raises, the answer is "known", so a
    broken profiles layer degrades to the pre-guard behaviour instead of
    parking every card. An *empty* roster is deliberately NOT fail-open —
    ``tests/hermes_cli/test_kanban_decompose_db.py`` pins parking as the
    designed behaviour there (the 2026-08-30 junk-card incident), and a
    fresh-install roster that answers No to a phantom name is answering
    correctly. Whether a bare root should park is a product question, not
    something to change inside a merge.
    """
    if not assignee:
        return True
    try:
        from hermes_cli.profiles import profile_exists
    except Exception:  # noqa: BLE001 — a guard must not become a crash surface
        return True
    try:
        return bool(profile_exists(assignee))
    except Exception:  # noqa: BLE001
        return True

def resolve_default_max_cost() -> Optional[float]:
    """Resolve the ``kanban.default_max_cost`` new-card fallback.

    Single shared spot for both the CLI ``kanban create`` path and the
    worker-facing ``kanban_create`` tool: when a caller passes no explicit
    ``max_cost``, the card inherits the configured default. Absent/unreadable
    config or an absent key returns ``None`` (uncapped, backward compat). An
    unparseable or negative config value raises ``ValueError`` so callers can
    surface it as an error rather than silently uncap a card.
    """
    try:
        from hermes_cli.config import load_config

        raw = (load_config() or {}).get("kanban", {}).get(
            "default_max_cost", None
        )
    except Exception:
        return None
    if raw is None:
        return None
    val = float(raw)  # raises ValueError on garbage -> caller decides
    if val < 0:
        raise ValueError("kanban.default_max_cost must be >= 0")
    return val


def _resolve_cost_key(key: str, fallback: float) -> float:
    """Read a float ``kanban.<key>`` from config, falling back on any error."""
    try:
        from hermes_cli.config import load_config

        raw = (load_config() or {}).get("kanban", {}).get(key, None)
        if raw is None:
            return fallback
        val = float(raw)
        return val if val > 0 else fallback
    except Exception:
        return fallback


def resolve_max_cost_ceiling() -> float:
    """Hardest cap allowed on a NEW card (``kanban.max_cost_ceiling``).

    WeRoll cost policy 2026-09-02 (Richie): every card carries an estimate and a
    cap of estimate + 20% contingency, and **no new card may be created above
    $1.00**. Steve-o owns estimates and caps; a card needing more than the
    ceiling must be split, not raised at creation.
    """
    return _resolve_cost_key("max_cost_ceiling", 1.00)


def resolve_max_cost_hard_ceiling() -> float:
    """Absolute lifetime ceiling for a card including Steve-o's extensions.

    Steve-o may grant at most two extensions to a single card, in increments of
    his choosing, up to a total card cost of ``kanban.max_cost_hard_ceiling``
    ($1.50). A third cap break is never extended: the card is blocked and Richie
    is asked for a decision on Slack and iMessage.
    """
    return _resolve_cost_key("max_cost_hard_ceiling", 1.50)


def _profile_skills_dir(profile: Optional[str]) -> Path:
    """Where a profile's skills actually live.

    A profile sees ONLY its own ``skills/`` directory — root's are invisible to
    it (``tools/skills_tool.py:148-159`` and ``tools/skill_ledger.py:192-193``
    both return ``get_hermes_home()/"skills"`` with no parent walk, and the
    dispatcher sets ``HERMES_HOME=<profiles/x>`` per worker).
    """
    home = Path(os.path.expanduser("~/.hermes"))
    name = (profile or "").strip()
    if not name or name in ("default", "root"):
        return home / "skills"
    return home / "profiles" / name / "skills"


def missing_skills_for(profile: Optional[str], skills) -> list[str]:
    """Names in ``skills`` that the profile cannot load. Fail-open.

    **2026-09-05.** Skills were validated NOWHERE — not at creation, not at
    dispatch. ``Unknown skill(s): X`` is raised inside the spawned worker at
    agent init (``cli.py:8976`` / ``hermes_cli/oneshot.py:80``), after the card
    is claimed, the workspace built and the PID forked; the dispatcher sees only
    ``exit_code 1``. Six cards burned exactly two runs each that way
    (t_7f4ea155, t_0ec6abcf, t_867e31c9, t_03e5426e, t_8c40251d, t_34e858c9),
    and `switch` still has no ``sdlc-review`` — which the review lane injects
    unconditionally — so the failure is armed today.

    Cheap: a filesystem walk, no Hermes import, no subprocess, ~ms. Returns []
    (allow) whenever the answer is uncertain: no skills requested, no skills
    directory, unreadable tree. A dispatch gate must never invent a blocker.
    """
    wanted = [s for s in (skills or []) if s]
    if not wanted:
        return []
    root = _profile_skills_dir(profile)
    try:
        if not root.is_dir():
            return []
        have: set[str] = set()
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            if "SKILL.md" in filenames:
                have.add(os.path.basename(dirpath))
            for d in dirnames:  # a symlinked skill dir is a real candidate
                p = Path(dirpath) / d
                if p.is_symlink() and (p / "SKILL.md").exists():
                    have.add(d)
    except Exception:
        return []
    if not have:
        return []
    return [s for s in wanted if s not in have]


def effective_max_cost(max_cost: Optional[float]) -> Optional[float]:
    """Apply the WeRoll cost policy to a cap supplied at card creation.

    Used by EVERY task-insert path so no card can be minted uncapped: the CLI,
    the ``kanban_create`` tool, the dashboard API, the auto-decomposer, the
    auto-generated ``Deploy:`` follow-ups and the swarm. Before 2026-09-02 the
    default was applied on the CLI/tool paths only, which left 68% of cards
    uncapped and let one card reach $2.17 against a $0.60 cap.

    ``None`` -> the configured default. Any value -> clamped to the new-card
    ceiling. A card that genuinely needs more gets split by Steve-o.
    """
    try:
        val = resolve_default_max_cost() if max_cost is None else float(max_cost)
    except Exception:
        val = None
    if val is None:
        return None
    ceiling = resolve_max_cost_ceiling()
    return min(val, ceiling) if val > 0 else None


# --- Charter §4 estimation at the mint path (est-mintpath-20260911) ----------
#
# §4 mandates every card carry points + a cost estimate. Measured 2026-09-11:
# 21/241 cards (8.7%) carried one. The 2026-09-04 attempt
# (self-improvement-si-20260904-estimation) edited SOUL prose and moved
# nothing — workers and the auto-decomposer never read a SOUL, and ~70% of
# cards are minted by the machine path. This is the same shape as the cost-cap
# fix that did work: apply the default at the ONE choke point every caller
# goes through, so the machine path is covered by construction.
#
# Points travel as a ``points-estimate: N`` COMMENT, not a column. The cost
# ledger (``scripts/cost-ledger.py``, ``PTS_RE``) and the charter §2 estimation
# metric both read them from comments, so a column would be a second source of
# truth that moves no number. The comment IS the mechanism — the same
# convention as ``scheduled-until:`` for date waits.
AUTO_POINTS_PLACEHOLDER = 1
AUTO_POINTS_NOTE = (
    "auto-points: §4 placeholder — written at creation because this card "
    "carried no estimate. The specifier replaces it on first touch by posting "
    "the real estimate; a later estimate comment wins over this one."
)


def auto_points_comment(points: Optional[int] = None) -> str:
    """The §4 estimate marker for a card minted without one.

    ``points`` None -> the conservative placeholder plus the note explaining
    it is not anybody's estimate. An explicit value -> just the marker, so a
    creator who knows the estimate writes a real one and no placeholder text
    is ever written. ``points-estimate:`` is the line the cost ledger parses.
    """
    value = AUTO_POINTS_PLACEHOLDER if points is None else int(points)
    body = f"points-estimate: {value}"
    if points is None:
        body += f"\n{AUTO_POINTS_NOTE}"
    return body


def _fleet_home() -> Path:
    """The FLEET root (~/.hermes), never a profile dir.

    ``HERMES_HOME`` is set per worker to ``profiles/<x>`` by the dispatcher, so
    anything that must agree across every profile is anchored here instead.
    """
    h = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes")).expanduser()
    if h.parent.name == "profiles":
        h = h.parent.parent
    return h


def _tenant_project(tenant: Optional[str]):
    """Resolve a tenant to a Project from the fleet-level tenant map.

    2026-09-06 (plan card W1). Projects live in the CREATOR's per-profile
    ``projects.db``, so the same brief produced worktree cards from Jobsy (who
    has the ``backupbrain`` project) and empty ``scratch`` cards from Karl (who
    does not): 12 scratch children in one job, four "workspace is empty" blocks
    within nine minutes, eight review/verify cards closed as "SUPERSEDED (no
    work done)". The map at ``<fleet home>/kanban-tenants.json`` is read the
    same way from every profile and every mint path::

        {"backupbrain": {"id": "p_5fe7127d", "slug": "backupbrain",
                         "name": "BackupBrain",
                         "primary_path": "/Users/.../Meeting notes transcription tool"}}

    ``HERMES_KANBAN_TENANTS`` overrides the path (tests). Fail-open: any
    problem returns ``None`` and the caller behaves exactly as before.
    """
    if not tenant:
        return None
    try:
        path = os.environ.get("HERMES_KANBAN_TENANTS")
        p = Path(path) if path else _fleet_home() / "kanban-tenants.json"
        if not p.is_file():
            return None
        data = json.loads(p.read_text() or "{}")
        ent = data.get(str(tenant).strip())
        if not isinstance(ent, dict) or not ent.get("primary_path"):
            return None
        from hermes_cli import projects_db as _pdb

        slug = ent.get("slug") or str(tenant)
        try:
            slug = _pdb.normalize_slug(slug) or slug
        except Exception:
            pass
        return _pdb.Project(
            id=str(ent.get("id") or f"tenant:{tenant}"),
            slug=slug,
            name=str(ent.get("name") or slug),
            created_at=0,
            primary_path=str(ent["primary_path"]),
        )
    except Exception:
        return None


def _maybe_create_deploy_followup(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
) -> Optional[str]:
    """When a reviewed platform card is approved, auto-create a deploy follow-up.

    An approval (``metadata["review_outcome"] == "approved"``) of a task that was
    implemented in a git worktree under *this* hermes-agent repo means the
    approved branch is still sitting on ``wt/`` while the live gateway runs old
    code — exactly the approved-but-undeployed gap behind the duplicate-send
    incident (2026-08-30/31, 9 more live duplicates while approved-unmerged).

    Close the gap: create exactly one follow-up card that merges the branch to
    the parked branch and restarts the gateway. The card is idempotency-keyed on
    the approved task id, so re-completion of the same card cannot spawn a second
    deploy card. Since 2026-09-03 (charter §8) it is created HELD
    (``blocked`` / ``operator_hold``): approval precedes deploy, and only Richie
    releases it.

    Returns the new card id, or ``None`` when no follow-up is warranted.
    """
    if not isinstance(metadata, dict) or metadata.get("review_outcome") != "approved":
        return None
    task = get_task(conn, task_id)
    if task is None or task.workspace_kind != "worktree":
        return None
    ws = (task.workspace_path or "").strip()
    if not ws:
        return None
    # Platform cards live in a worktree under THIS hermes-agent repo.
    try:
        ws_resolved = Path(ws).expanduser().resolve()
    except OSError:
        return None
    worktrees_root = (Path(__file__).resolve().parents[1] / ".worktrees").resolve()
    if worktrees_root not in ws_resolved.parents:
        return None

    branch = (task.branch_name or "").strip() or f"wt/{task_id}"
    repo = worktrees_root.parent

    # 2026-09-02: the deploy target is whatever branch the repo is PARKED on,
    # not a hardcoded "main". On 09-02 06:53:28 the auto-updater reset main to
    # origin/main and erased 59 fleet commits; the fleet now lives on `fleet`
    # with updates.parked_branch_strategy: update_in_place. A deploy card that
    # still said "main" would merge approved work onto upstream's branch, where
    # the next update would discard it — i.e. silently re-create that incident.
    # Resolve dynamically so this cannot drift again.
    deploy_branch = "main"
    try:
        _hb = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
        _name = (_hb.stdout or "").strip()
        if _hb.returncode == 0 and _name and _name != "HEAD":
            deploy_branch = _name
    except Exception:  # never block a deploy card on a git hiccup
        pass

    title = f"Deploy: merge {branch} + restart gateway"
    idempotency_key = f"deploy-followup:{task_id}"
    tag = f"deploy/{task_id}"
    body = (
        f"Deploy the approved platform branch `{branch}` to the live gateway.\n\n"
        f"**This card is HELD (`operator_hold`). Charter §8: approval precedes deploy on "
        f"every repo; Rodge's approval is a review verdict, not a deploy trigger. Richie "
        f"releases it with `hermes kanban unblock {{this card}}`. Do not unblock it yourself.**\n\n"
        f"The parent card `{task_id}` passed independent review, but approved is not "
        f"deployed: the branch sits on `{branch}` while the live gateway is still "
        f"running old code.\n\n"
        f"Repo: `{repo}`  ·  target branch: `{deploy_branch}`  ·  tag on success: `{tag}`\n\n"
        "Steps:\n"
        f"1. `cd {repo}` and confirm the tree is clean (`git status --porcelain` empty) and "
        f"HEAD is on `{deploy_branch}`. A dirty tree or wrong branch = stop and comment; "
        "never merge onto a dirty tree (charter §11).\n"
        f"2. Ensure the branch is present: `git fetch origin`, `git branch --list {branch}`. If the "
        "post-completion cleanup pruned the local worktree branch, recreate it from the parent "
        "card's commit rather than abandoning the deploy.\n"
        f"3. Capture which fleet-watchdog files this deploy touches: "
        f"`git diff --name-only {deploy_branch}...{branch} -- scripts/fleet-watchdogs/ > /tmp/wd-changed-{task_id}`.\n"
        f"4. Merge as an ORDINARY MERGE COMMIT — `git merge --no-ff --no-edit {branch}` — never "
        "squash on this repo: the fleet-integrity and pre-flight ancestry checks depend on the "
        "sentinel commits staying ancestors of HEAD. On conflict: `git merge --abort`, comment, "
        "stop. Never force-merge or force-push.\n"
        f"5. Tag the merge: `git tag -a {tag} -m \"deploy {task_id}\"`. Rollback later is "
        f"`~/.hermes/scripts/fleet-rollback.sh <change-id>` or a revert PR — never a force-push.\n"
        "6. Install changed fleet watchdog scripts into the live cron dir (copy-on-merge only — "
        f"no symlinks): `python3 scripts/fleet-watchdogs/install_fleet_watchdogs.py --dest ~/.hermes/scripts $(cat /tmp/wd-changed-{task_id})`. "
        "Empty capture file = nothing to do.\n"
        "7. Run the repo's focused tests for the changed area — e.g. "
        "`venv/bin/python -m pytest tests/hermes_cli/test_kanban_review_lifecycle_complete.py -q` "
        "plus any test the parent card named. If watchdog files changed, also run "
        "`python3 scripts/fleet-watchdogs/test_install_fleet_watchdogs.py`.\n"
        "8. Restart the live (root) gateway: `bash ~/.hermes/profiles/axel/scripts/restart-root-gateway.sh`, "
        "then verify a fresh start_time.\n"
        f"9. Prove it: `git merge-base --is-ancestor {branch} HEAD` exits 0, `git tag --list {tag}` "
        "prints the tag, `python3 ~/.hermes/scripts/fleet-preflight.py` is GREEN, and the gateway "
        "start_time changed. Put all four in the completion summary, plus which watchdog files "
        "(if any) were installed.\n\n"
        "Comment or reopen this card if the merge, focused tests, or restart fails."
    )
    child_id = create_task(
        conn,
        title=title,
        body=body,
        assignee="default",
        created_by=task.assignee,
        parents=[task_id],
        idempotency_key=idempotency_key,
        # Charter §8 (2026-09-03): minted HELD; only Richie releases it.
        initial_status="blocked",
        block_kind="operator_hold",
    )
    _log.debug(
        "Created deploy follow-up %s for approved platform card %s (branch %s)",
        child_id, task_id, branch,
    )
    return child_id


def _wip_commit_worktree(worktree_path: str, task_id: str) -> bool:
    """Commit all staged/unstaged changes in a task worktree onto its branch.

    ``git add -A && git commit -m 'WIP: force-finalized <task_id>'`` — run
    from inside the worktree (``-C`` its path) so the commit lands on the
    task branch as the current HEAD is the worktree's checked-out branch.
    Only ever commits real, user/agent-authored changes; a commit with
    nothing to add is a no-op (``git commit`` refuses on an empty tree).

    Runs BEFORE any teardown step so the branch is the durable owner of a
    half-done diff. Returns True when a WIP commit was created (or nothing
    was dirty); False when the commit failed or the tree could not be
    committed — callers preserve the worktree in that case.
    """
    try:
        add = subprocess.run(
            ["git", "-C", worktree_path, "add", "-A"],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=60, check=False,
        )
        if add.returncode != 0:
            _log.warning("git add -A failed for worktree %s: %s",
                         worktree_path, (add.stderr or add.stdout or "").strip())
            return False
        commit = subprocess.run(
            ["git", "-C", worktree_path, "commit", "-m",
             f"WIP: force-finalized {task_id}"],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=60, check=False,
        )
        # rc 0 = committed; rc 1 = nothing staged (clean/no-op) — both fine.
        if commit.returncode not in (0, 1):
            _log.warning("git commit failed for worktree %s: %s",
                         worktree_path, (commit.stderr or commit.stdout or "").strip())
            return False
        return True
    except Exception as exc:
        _log.warning("_wip_commit_worktree failed for %s: %s", worktree_path, exc)
        return False


# Toolchain directories that are git-ignored in every project this fleet runs
# and therefore ABSENT from a fresh worktree. Charter §7 (2026-09-03): every
# worktree is provisioned with the project's test runner before the worker
# starts, so a missing `pytest` is an environment fault charged to Smith, never
# to the card (a stdlib-only worktree venv false-bounced a good card twice on
# 09-01). Symlinks only: zero network, zero tokens, and the worker sees exactly
# the interpreter and node_modules the repo root already has.
WORKTREE_TOOLCHAIN_DIRS = ("venv", ".venv", "node_modules")


# How deep to look for a nested toolchain dir. 2026-09-07: the top-level scan
# above missed BackupBrain entirely — its install is at ``frontend/node_modules``,
# so every fresh worktree failed `npm run build` before it tested anything, and
# the 09-06 gate patch worked around it by SKIPPING the UI rung when `dist` was
# absent, which masks the fault instead of fixing it. One level is enough for the
# monorepo layouts this fleet actually has, and bounded so a big repo cannot turn
# worktree creation into a filesystem walk.
_WORKTREE_NESTED_SCAN_DEPTH = 1


_WORKTREE_NESTED_MAX_DIRS = 40


def _tenant_provision_paths(repo_root: Path) -> list[str]:
    """Repo-relative paths a tenant declares must exist in every worktree.

    2026-09-07. Symlinking `node_modules` is not enough: a project's *data* can
    be gitignored too, and then a fresh worktree is silently unable to verify
    anything. BackupBrain's 69MB corpus (`data/backupbrain.db`) is the live
    case — two cards had to be hand-provisioned on 09-07 before they could check
    their own acceptance criteria, which is the same failure the toolchain links
    exist to prevent.

    Declared per tenant in ``kanban-tenants.json``::

        "backupbrain": {..., "provision": ["data/backupbrain.db",
                                           "frontend/node_modules"]}

    Looked up by ``primary_path`` so any caller with a repo root can resolve it
    without threading the tenant through. Fail-open: any problem returns ``[]``
    and worktree creation behaves exactly as before.
    """
    try:
        path = os.environ.get("HERMES_KANBAN_TENANTS")
        p = Path(path) if path else _fleet_home() / "kanban-tenants.json"
        if not p.is_file():
            return []
        data = json.loads(p.read_text() or "{}")
        if not isinstance(data, dict):
            return []
        want = repo_root.resolve(strict=False)
        for ent in data.values():
            if not isinstance(ent, dict):
                continue
            primary = ent.get("primary_path")
            if not primary:
                continue
            if Path(str(primary)).expanduser().resolve(strict=False) != want:
                continue
            raw = ent.get("provision") or []
            if not isinstance(raw, list):
                return []
            return [str(x) for x in raw if isinstance(x, str) and x.strip()]
    except Exception as exc:  # noqa: BLE001 — provisioning must never fail a dispatch
        _log.debug("tenant provision lookup for %s failed: %s", repo_root, exc)
    return []


def _safe_relative_target(repo_root: Path, target: Path, rel: str):
    """Resolve a repo-relative provision entry, or ``None`` if it escapes.

    An entry is data from a config file, so it is treated as untrusted: absolute
    paths and anything that escapes the repo (``../../etc/passwd``) are refused
    rather than linked. Returns ``(src, dst)``.

    The bounds check is LEXICAL (``normpath``), not ``resolve()``. Resolving
    followed an existing symlink at the destination out of the worktree and
    refused the entry — so an entry the nested scan had *already* linked was
    reported as a traversal attempt on every worktree. Observed on the first
    live run, 2026-09-07: `frontend/node_modules` was linked correctly and
    warned about in the same breath. A security check that cries wolf on its own
    successful output is how a fleet learns to scroll past its warnings.
    Lexical normalisation is also the right tool here: it collapses ``..``
    without consulting the filesystem, which is exactly the traversal question
    being asked.
    """
    rel = (rel or "").strip()
    if not rel or Path(rel).is_absolute():
        return None
    root = Path(os.path.normpath(str(repo_root)))
    tgt = Path(os.path.normpath(str(target)))
    src = Path(os.path.normpath(str(root / rel)))
    dst = Path(os.path.normpath(str(tgt / rel)))
    try:
        src.relative_to(root)
        dst.relative_to(tgt)
    except ValueError:
        return None
    if src == root or dst == tgt:
        return None
    return src, dst


def _provision_worktree_toolchain(repo_root: Path, target: Path) -> list[str]:
    """Symlink the repo root's toolchain and tenant data into a fresh worktree.

    Best effort throughout: a worktree that is missing a convenience link is
    recoverable, a dispatch that raises here is not.
    """
    linked: list[str] = []

    def _link(rel: str, src: Path, dst: Path, is_dir: bool) -> None:
        try:
            if not src.exists():
                return
            if dst.exists() or dst.is_symlink():
                return
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src, target_is_directory=is_dir)
            linked.append(rel)
        except Exception as exc:  # noqa: BLE001
            _log.debug("worktree link %s -> %s failed: %s", dst, src, exc)

    # 1. Top-level toolchain dirs (the 2026-09-03 behaviour, unchanged).
    for name in WORKTREE_TOOLCHAIN_DIRS:
        src = repo_root / name
        if src.is_dir():
            _link(name, src, target / name, True)

    # 2. The same dirs one level down — `frontend/node_modules`, `api/.venv`.
    try:
        children = sorted(
            d for d in repo_root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name != ".worktrees"
        )[:_WORKTREE_NESTED_MAX_DIRS]
        for child in children:
            for name in WORKTREE_TOOLCHAIN_DIRS:
                src = child / name
                if src.is_dir():
                    rel = f"{child.name}/{name}"
                    _link(rel, src, target / child.name / name, True)
    except Exception as exc:  # noqa: BLE001
        _log.debug("worktree nested toolchain scan of %s failed: %s", repo_root, exc)

    # 3. Whatever the tenant declares it cannot work without — typically
    #    gitignored data. Untrusted config, so each entry is bounds-checked.
    for rel in _tenant_provision_paths(repo_root):
        resolved = _safe_relative_target(repo_root, target, rel)
        if resolved is None:
            _log.warning(
                "worktree %s: refusing provision entry %r — it escapes the repo",
                target, rel,
            )
            continue
        src, dst = resolved
        if src.exists():
            _link(rel, src, dst, src.is_dir())

    if linked:
        _exclude_provisioned_links(target, linked)
        _log.info("worktree %s: linked %s from %s", target, ",".join(linked), repo_root)
    return linked


def _exclude_provisioned_links(target: Path, linked: list[str]) -> None:
    """Make git ignore the links we just created, in this repo's local excludes.

    2026-09-07, found within an hour of shipping the links above. A `.gitignore`
    entry written as ``.venv/`` or ``node_modules/`` is a DIRECTORY pattern, and
    git does not follow symlinks — so a *symlink* named ``.venv`` is NOT matched
    by ``.venv/`` and is perfectly addable. The force-finalize teardown runs
    ``git add -A``, so it committed a symlink pointing outside the repo; the
    BackupBrain working line carried one until a deploy worker had to de-track it
    by hand (`37dd6d3`), and card `t_2c2086ac`'s worktree still has one staged.

    Fixed here rather than in each tenant's `.gitignore` because provisioning is
    what creates these paths, so provisioning is what should hide them — and
    `info/exclude` is local-only: never committed, never pushed, invisible to the
    tenant's repo history.

    Best effort: an un-excluded link is untidy, a raised exception is a failed
    dispatch.
    """
    try:
        common = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if common.returncode != 0:
            return
        gitdir = Path((common.stdout or "").strip())
        if not gitdir.is_absolute():
            gitdir = (target / gitdir).resolve(strict=False)
        exclude = gitdir / "info" / "exclude"
        exclude.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude.read_text().splitlines() if exclude.exists() else []
        have = {ln.strip() for ln in existing}
        # A bare name (no trailing slash) matches a symlink, a file OR a
        # directory — which is the whole point of this function.
        want = [rel for rel in linked if rel not in have and f"/{rel}" not in have]
        if not want:
            return
        with exclude.open("a") as fh:
            if existing and existing[-1].strip():
                fh.write("\n")
            fh.write("# hermes worktree provisioning (2026-09-07): these are symlinks\n")
            fh.write("# into the repo root. A directory pattern like `.venv/` does NOT\n")
            fh.write("# match a symlink, so without these a `git add -A` commits them.\n")
            for rel in want:
                fh.write(f"{rel}\n")
        _log.info("worktree %s: excluded %s via %s", target, ",".join(want), exclude)
    except Exception as exc:  # noqa: BLE001
        _log.debug("worktree %s: could not write local excludes: %s", target, exc)


# Workspace / worktree contention signatures in a worker log that indicate a
# clean exit (rc=0) was NOT a genuine protocol violation but a broken
# workspace: the worker bailed because of git-worktree / workspace-path
# contention. Retrying into the same broken workspace would just loop, so the
# card is blocked (``capability``) instead (2026-09-01 split, t_8fc16a73).
#
# Only SPECIFIC contention phrases are matched — never the bare nouns
# ``workspace`` / ``worktree`` / the ``.worktrees`` path — because those appear
# in benign model reasoning text (measured 126/166 real worker logs, 2026-09-01)
# and would misclassify a genuine no_checkpoint (finished but forgot to
# checkpoint) whose log tail merely mentions "workspace" as workspace_error →
# blocked instead of the bounded retry it needs. A true git worktree /
# workspace-path contention failure always carries one of these exact
# signatures.
_WORKSPACE_ERROR_RE = re.compile(
    r"\b(not inside a git repo|does not point at a git "
    r"repo root|non-absolute workspace|is already checked out at|"
    r"failed to create worktree|already exists and is not an empty "
    r"directory)\b",
    re.IGNORECASE,
)


def _circuit_outbox_path(board: Optional[str] = None) -> Path:
    """Path to the circuit alert outbox (JSON) for a board.

    Mirrors ``worker_logs_dir`` semantics: the ``default`` board uses the legacy
    kanban root, other boards nest under ``kanban/boards/<slug>/``. The outbox
    is a file the no_agent alert cron watches; a signature inside drives dedup
    so a paused breaker does not re-alert every tick.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / CIRCUIT_OUTBOX_FILENAME
    return board_dir(slug) / CIRCUIT_OUTBOX_FILENAME


def _ensure_dispatch_circuit_row(conn: sqlite3.Connection) -> None:
    """Ensure the singleton ``dispatch_circuit`` row (id=1) exists (closed)."""
    conn.execute(
        "INSERT OR IGNORE INTO dispatch_circuit (id, state, reason, "
        "opened_at, last_probe_at, probe_count) "
        "VALUES (?, 'closed', '', NULL, NULL, 0)",
        (CIRCUIT_ROW_ID,),
    )


def _circuit_status(conn: sqlite3.Connection) -> dict:
    """Return the current circuit state as a dict with defaults for a fresh DB."""
    _ensure_dispatch_circuit_row(conn)
    row = conn.execute(
        "SELECT state, reason, opened_at, last_probe_at, probe_count "
        "FROM dispatch_circuit WHERE id = ?",
        (CIRCUIT_ROW_ID,),
    ).fetchone()
    if row is None:
        return {"state": "closed", "reason": "", "opened_at": None,
                "last_probe_at": None, "probe_count": 0}
    keys = row.keys()
    return {
        "state": row["state"],
        "reason": row["reason"],
        "opened_at": row["opened_at"] if "opened_at" in keys else None,
        "last_probe_at": row["last_probe_at"],
        "probe_count": row["probe_count"],
    }


def _circuit_open(conn: sqlite3.Connection) -> bool:
    """True when the global dispatch circuit breaker is open (paused)."""
    try:
        return _circuit_status(conn).get("state") == "open"
    except Exception:
        # Uninitialized/corrupt — treat as closed (not paused). Safe default:
        # never block dispatch on a bookkeeping read failure.
        return False


def _circuit_trip(
    conn: sqlite3.Connection,
    reason: str,
    *,
    board: Optional[str] = None,
) -> None:
    """Open (pause) the global dispatch circuit breaker.

    ``reason`` is a human-readable detection signature stored for the alert /
    audit trail. The breaker stays open until a probe succeeds and closes it.
    """
    _ensure_dispatch_circuit_row(conn)
    conn.execute(
        "UPDATE dispatch_circuit SET state = 'open', reason = ?, "
        "opened_at = COALESCE(opened_at, ?) WHERE id = ?",
        (reason[:500], int(time.time()), CIRCUIT_ROW_ID),
    )


def _circuit_close(conn: sqlite3.Connection) -> None:
    """Close (resume) the global breaker. ``last_probe_at`` is bumped so a
    resume is not immediately re-probed on the next tick."""
    _ensure_dispatch_circuit_row(conn)
    conn.execute(
        "UPDATE dispatch_circuit SET state = 'closed', last_probe_at = ? "
        "WHERE id = ?",
        (int(time.time()), CIRCUIT_ROW_ID),
    )


def _circuit_probe(probe_fn=None) -> bool:
    """Run one cheap real probe that the provider is back.

    ``probe_fn`` (test hook) returns True when the provider is healthy. The
    default probe issues a tiny completion on the default model via OpenRouter;
    any error (network, 4xx/5xx, missing key) returns False — the breaker stays
    paused. Bounded by ``DEFAULT_CIRCUIT_PROBE_TIMEOUT_SECONDS`` so a hang
    never blocks a dispatch tick indefinitely.
    """
    if probe_fn is not None:
        try:
            return bool(probe_fn())
        except Exception:
            return False
    key = _circuit_openrouter_api_key()
    if not key:
        return False
    import json as _json
    import urllib.error as _uerror
    import urllib.request as _urequest
    model = os.environ.get(
        "HERMES_KANBAN_CIRCUIT_PROBE_MODEL",
        "deepseek/deepseek-v4-flash-0731",
    ).strip() or "deepseek/deepseek-v4-flash-0731"
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
    }).encode()
    req = _urequest.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _urequest.urlopen(
            req, timeout=DEFAULT_CIRCUIT_PROBE_TIMEOUT_SECONDS
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


def _circuit_openrouter_api_key() -> Optional[str]:
    """Return the OpenRouter API key used for the probe (env or ``~/.hermes``)."""
    val = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if val:
        return val
    try:
        env_path = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
        env_file = env_path / ".env"
        for line in env_file.read_text(errors="ignore").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    except Exception:
        pass
    return None


def _circuit_dispatch_paused(
    conn: sqlite3.Connection,
    *,
    circuit_probe_fn=None,
    board: Optional[str] = None,
) -> DispatchResult:
    """Run the paused tick: possibly probe to resume, never spawn.

    Called in place of the ready/review spawn loops when the circuit is open.
    If due for a probe and the probe succeeds, the breaker closes (resumes)
    and a resume alert is queued; the remainder of THIS tick still returns
    paused so a health-checking caller sees a clean ``frozen_by_circuit``
    result — the next tick spawns normally. Returns the result to return from
    ``_dispatch_once_locked``.
    """
    result = DispatchResult(frozen_by_circuit=True)
    st = _circuit_status(conn)
    interval = _resolve_circuit_probe_interval_seconds()
    due = interval > 0 and (
        st.get("last_probe_at") is None
        or (time.time() - int(st.get("last_probe_at") or 0)) >= interval
    )
    if due:
        with write_txn(conn):
            now = int(time.time())
            conn.execute(
                "UPDATE dispatch_circuit SET last_probe_at = ?, "
                "probe_count = probe_count + 1 WHERE id = ?",
                (now, CIRCUIT_ROW_ID),
            )
        ok = _circuit_probe(probe_fn=circuit_probe_fn)
        if ok:
            with write_txn(conn):
                _circuit_close(conn)
            _queue_circuit_alert(
                conn, "resumed",
                "kanban dispatch resumed — provider probe succeeded; "
                "circuit breaker closed.",
                board=board,
            )
            _log.info(
                "kanban dispatch: circuit breaker probe succeeded; "
                "dispatch resumed."
            )
    return result


def _detect_provider_outage(
    task_id: str,
    started_at: Optional[int],
    reaped_at: Optional[int],
    *,
    board: Optional[str] = None,
) -> bool:
    """True when a dead worker shows the provider-outage-at-spawn signature.

    A worker that (a) exited cleanly within ``CIRCUIT_FAST_EXIT_SECONDS`` of
    being spawned (rc=0, which also implies no real work / no tool calls) AND
    (b) has a provider-side HTTP 4xx/5xx (billing / rate-limit / auth) in its
    log is the fleet-wide outage signature — NOT a genuine worker fault.
    ``_RESPAWN_BLOCKER_RE`` is the same auth/quota pattern the respawn guard
    already trusts, so detection reuses the established auth-pool
    classification. No log, or a missing/inconclusive runtime, is conservatively
    False: we only trip the fleet-wide breaker on a positive, grounded signal.

    Timing measure: ``reaped_at - started_at`` is the worker's ACTUAL runtime,
    taken from the reap registry (``_RECENT_WORKER_EXITS``). We deliberately do
    NOT use ``time.time() - started_at`` here — by the time
    ``detect_crashed_workers`` inspects a dead worker it has already survived
    the ``DEFAULT_CRASH_GRACE_SECONDS`` (30s) launch window, so elapsed
    wall-clock at inspection time is always past the fast-exit window and the
    fingerprint would never fire. The spawn-bounce it fingerprints is a short
    *runtime*, not a short *wait-until-inspected*.
    """
    if started_at is not None and reaped_at is not None:
        try:
            runtime = int(reaped_at) - int(started_at)
        except (TypeError, ValueError):
            # Inconclusive timestamps — don't trip on a guess.
            return False
        if runtime >= CIRCUIT_FAST_EXIT_SECONDS or runtime < 0:
            # Ran too long to be the spawn bounce, or the clock is inconsistent
            # (reaped before started) — not confidently the outage fingerprint.
            return False
    try:
        log = _read_worker_log_text(task_id, board=board)
    except Exception:
        log = ""
    if not log:
        return False
    return bool(_RESPAWN_BLOCKER_RE.search(log)) and bool(
        re.search(r"\b(403|429|4\d\d|5\d\d|forbidden|too_many_requests|"
                 r"monthly\s*limit|credits|quota|forbidden)\b",
                 log, re.IGNORECASE))


def _classify_clean_exit(
    task_id: str, *, board: Optional[str] = None,
) -> str:
    """Classify a clean-exit (rc=0) worker whose task is still ``running``.

    Decisions are grounded in the worker log (never a guess). One of:

    * ``workspace_error`` — a workspace / worktree / git-contention signature
      in the log. The card must be blocked (``capability``), not retried into
      the same broken workspace (2026-09-01 split, t_8fc16a73).
    * ``no_checkpoint`` — the genuine case: a model that finished but never
      checkpointed (no workspace error in the log). Keeps today's bounded
      retry.

    The provider class (``provider_error``) is NOT derived here: the fast
    provider spawn-bounce is a distinct, timing-gated path
    (``_detect_provider_outage`` → ``violation_class=provider_error``), and a
    long-running clean exit with provider text in the log is deliberately kept
    a plain crash (see test_kanban_circuit_breaker.py ``[False]`` — the outage
    fingerprint is specifically the *spawn bounce*, not any clean exit).
    """
    try:
        log = _read_worker_log_text(task_id, board=board)
    except Exception:
        log = ""
    if not log:
        return "no_checkpoint"
    if _WORKSPACE_ERROR_RE.search(log):
        return "workspace_error"
    return "no_checkpoint"


def _read_worker_log_text(
    task_id: str, *, board: Optional[str] = None,
    limit: int = 20000,
) -> str:
    """Return the tail of a worker log as text for outage-signature scanning."""
    import io
    try:
        path = worker_log_path(task_id, board=board)
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit))
            data = f.read()
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _queue_circuit_alert(
    conn: sqlite3.Connection,
    kind: str,
    message: str,
    *,
    board: Optional[str] = None,
) -> None:
    """Append one alert to the circuit outbox (dedup by kind+board).

    The no_agent cron (``kanban-circuit-watch.py``) delivers the newest
    unsent alert exactly once via the budget-watch delivery channel and clears
    the outbox, so a paused breaker does not nag on every tick.
    """
    path = _circuit_outbox_path(board=board)
    saw = False
    data = {}
    try:
        if path.exists():
            data = json.loads(path.read_text(errors="ignore") or "{}")
            if isinstance(data, dict):
                last = data.get("messages", [])
                if isinstance(last, list) and last:
                    if last[-1].get("kind") == kind:
                        saw = True
        if not saw:
            messages = []
            if isinstance(data, dict) and isinstance(data.get("messages", []), list):
                messages = data.get("messages", [])
            messages.append({
                "kind": kind, "message": message, "at": int(time.time()),
            })
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"messages": messages}))
    except Exception:
        _log.warning("kanban circuit: failed to queue %s alert", kind,
                     exc_info=True)


def stamp_worker_run_metadata(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    extra: dict,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Best-effort merge ``extra`` into the active run's ``metadata``.

    Same identity contract as ``heartbeat_worker``: the worker's own current
    run, pinned by ``expected_run_id`` when provided. Used by the kanban
    finalize-in-process instrumentation to record, on the run row itself,
    whether a forced finalize turn fired (so the rate is measurable whether or
    not the worker later completes). Never raises. Returns True only when the
    merge actually landed on the task's pinned (or current) run row — False
    when ``extra`` is empty, when that run does not exist, when a pinned id
    is stale or belongs to a different task, or when the write failed.

    The merge is JSON-level (read current metadata, update, write back) rather
    than the outer ``metadata = ?`` overwrite used by complete/block — a worker
    may stamp the fired flag mid-run, before the terminal transition writes its
    own handoff metadata. The lookup is scoped to both ``id`` AND ``task_id``,
    and the write is verified by rowcount, so a stale/foreign pinned run is a
    real no-op (never silently lands on a sibling run).
    """
    if not extra:
        return False
    try:
        run_id = (
            int(expected_run_id)
            if expected_run_id is not None
            else _current_run_id(conn, task_id)
        )
        if run_id is None:
            return False
        # Scoped to this task so a pinned id that is stale or belongs to a
        # different task is rejected, not merged onto the wrong row.
        row = conn.execute(
            "SELECT metadata FROM task_runs WHERE id = ? AND task_id = ?",
            (run_id, task_id),
        ).fetchone()
        if row is None:
            return False
        meta: dict = {}
        if row["metadata"]:
            try:
                meta = json.loads(row["metadata"])
            except (TypeError, ValueError):
                meta = {}
        if not isinstance(meta, dict):
            meta = {}
        meta.update(extra)
        cur = conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(meta, ensure_ascii=False), run_id),
        )
        # The row existed a moment ago; rowcount confirms the write actually
        # landed (guards against a racing delete between SELECT and UPDATE).
        return cur.rowcount > 0
    except Exception:
        _log.debug("stamp_worker_run_metadata failed: %s", task_id, exc_info=True)
        return False


def _escape_like(text: str) -> str:
    """Escape a literal for SQLite ``LIKE ... ESCAPE '\\'`` matching."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _state_db_path_for_assignee(assignee: Optional[str]) -> Optional[Path]:
    """Resolve the worker's state.db for ``assignee``.

    A kanban worker runs ``hermes -p <assignee>`` with HERMES_HOME pinned to
    ``resolve_profile_env(assignee)`` (see ``_default_spawn``), so the worker's
    session cost ledger lives in **that** profile home, not the dispatching
    profile's. Newer workers also self-tag their sessions with
    ``HERMES_SESSION_SOURCE=kanban``. Falls back to the dispatching profile's
    home when the assignee profile cannot be resolved (safe — a worker still
    reads a home either way). Returns None only when no home can be found at
    all, which the cost check treats as "no ledger available → $0".
    """
    try:
        from hermes_cli.profiles import resolve_profile_env

        home = resolve_profile_env(assignee) if assignee else None
    except Exception:
        home = None
    if not home:
        try:
            from hermes_constants import get_hermes_home

            home = str(get_hermes_home())
        except Exception:
            return None
    return Path(home) / "state.db"


def _cumulative_session_cost(state_db_path, workspace, task_id=None, all_ledgers=True) -> float:
    """Sum ``estimated_cost_usd`` across a card's kanban worker sessions.

    **2026-09-02 — this function was structurally blind and the cap never
    fired once in the fleet's history.** It matched sessions by ``s.cwd``
    under the task workspace, on the assumption that a worker is spawned with
    ``cwd=workspace``. In practice ``sessions.cwd`` is NULL for every kanban
    worker session on this fleet (170/170 on rodge alone), so the query
    matched nothing, returned 0.0, and ``enforce_max_cost`` compared $0.00
    against every cap. Measured on real cards: t_0af25226 had spent $0.2146
    and t_296c6855 $0.1406, and this returned 0.0 for both. Cards therefore
    ran to completion regardless of their cap — five of them overran by up to
    64% — and zero ``cost_cap`` blocks exist in the entire board history.

    The fix matches the way the card's sessions are actually identifiable: the
    worker titles each session ``Work kanban task <task_id> #<n>``, which IS
    populated. The cwd match is kept as an OR so nothing regresses if a future
    spawn path does set it, and so a workspace-only match still works when no
    task id is passed.

    Pure read — opens state.db read-only. Any failure (missing file, busy
    lock, malformed ledger) yields 0.0 so a ledger problem can never block
    the tick or fake a false positive.
    """
    if not workspace and not task_id:
        return 0.0
    # A card's LIFETIME cost is not one profile's spend. t_0ad7eded cost $2.1742
    # of which $1.9435 was Bob's and $0.2307 Rodge's review; reading only the
    # assignee's ledger sees 89% of it, and on a review-heavy card far less.
    # Richie's policy caps the card, so sum every profile that worked on it.
    if task_id:
        total = 0.0
        seen = set()
        try:
            roots = []
            # 2026-09-06 (C1): the CAP counts the assignee's own spend only.
            # Reviewer spend is measured on the review card and overwatch spend
            # on the job; summing every ledger let a $0.60 build + $0.50 review
            # breach after the work was done (Rodge's own p90 is $1.00), and
            # 59% of t_eaaa3434's "spend" was adjudication and block-loop churn.
            # ``all_ledgers=True`` keeps the lifetime figure for reporting.
            if all_ledgers:
                hermes_home = Path(os.path.expanduser("~/.hermes"))
                roots.append(hermes_home / "state.db")
                roots.extend(sorted((hermes_home / "profiles").glob("*/state.db")))
            if state_db_path:
                roots.append(Path(str(state_db_path)))
            for db in roots:
                key = str(db)
                if key in seen or not os.path.isfile(key):
                    continue
                seen.add(key)
                total += _session_cost_in_db(key, prefix_for=workspace, task_id=task_id)
        except Exception:
            return 0.0
        return total
    if not state_db_path or not os.path.isfile(str(state_db_path)):
        return 0.0
    return _session_cost_in_db(str(state_db_path), prefix_for=workspace, task_id=None)


def _session_cost_in_db(state_db_path, prefix_for=None, task_id=None) -> float:
    """Sum matching session cost inside ONE state.db. Fail-open on any error."""
    prefix = str(prefix_for).rstrip("/\\") if prefix_for else ""
    try:
        sconn = sqlite3.connect(
            f"file:{state_db_path}?mode=ro", uri=True, timeout=10
        )
        sconn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return 0.0
    try:
        clauses, params = [], []
        if prefix:
            clauses.append("(s.cwd = ? OR s.cwd LIKE ? ESCAPE '\\')")
            params += [prefix, _escape_like(prefix) + "/%"]
        if task_id:
            clauses.append("s.title LIKE ? ESCAPE '\\'")
            params.append("%" + _escape_like(str(task_id)) + "%")
        if not clauses:
            return 0.0
        where = " OR ".join(clauses)
        # Prefer the per-session total: session_model_usage is a per-model
        # breakdown that can be empty even when the session recorded a cost.
        row = sconn.execute(
            "SELECT COALESCE(SUM(s.estimated_cost_usd), 0) AS total "
            f"FROM sessions s WHERE {where}",
            params,
        ).fetchone()
        total = float(row["total"]) if row and row["total"] is not None else 0.0
        if total > 0.0:
            return total
        # Fall back to the per-model ledger if the session column is unset.
        row = sconn.execute(
            "SELECT COALESCE(SUM(u.estimated_cost_usd), 0) AS total "
            "FROM session_model_usage u JOIN sessions s ON s.id = u.session_id "
            f"WHERE {where}",
            params,
        ).fetchone()
        return float(row["total"]) if row and row["total"] is not None else 0.0
    except sqlite3.Error:
        return 0.0
    finally:
        sconn.close()


def set_task_max_cost(
    conn: sqlite3.Connection,
    task_id: str,
    new_cap: float,
    *,
    by: str,
    reason: str = "",
) -> float:
    """Raise a card's cap ONCE, to at most ``kanban.max_cost_hard_ceiling``.

    2026-09-06 (C1/O1): the overwatch extension. Smith may extend a breached
    card once by up to +$0.50 (config: ceiling 1.00 -> hard ceiling 1.50). A
    second extension is refused — a second breach on one card is the
    deeper-error signal that goes to Richie. Records a ``cost-extension:``
    comment on the card so the ledger review can see it. Returns the cap set.
    """
    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    hard = resolve_max_cost_hard_ceiling()
    try:
        new_cap = float(new_cap)
    except (TypeError, ValueError):
        raise ValueError(f"cap must be a number, got {new_cap!r}")
    if new_cap <= 0:
        raise ValueError("cap must be > 0")
    if new_cap > hard:
        raise ValueError(
            f"cap ${new_cap:.2f} exceeds the hard ceiling ${hard:.2f}; "
            "only Richie can move a card past it (split it instead)"
        )
    old = task.max_cost
    prior = conn.execute(
        "SELECT COUNT(*) FROM task_comments WHERE task_id = ? AND body LIKE 'cost-extension:%'",
        (task_id,),
    ).fetchone()[0]
    if prior:
        raise ValueError(
            "this card has already been extended once; a second breach stays "
            "blocked for Richie (overwatch rule)"
        )
    if old is not None and new_cap <= float(old):
        raise ValueError(f"cap ${new_cap:.2f} is not above the current ${float(old):.2f}")
    with write_txn(conn):
        conn.execute("UPDATE tasks SET max_cost = ? WHERE id = ?", (new_cap, task_id))
        add_comment(
            conn, task_id, author=by,
            body=(f"cost-extension: ${float(old):.2f} -> ${new_cap:.2f} by {by}"
                  if old is not None else f"cost-extension: (none) -> ${new_cap:.2f} by {by}")
                 + (f"\nreason: {reason}" if reason else ""),
        )
        _append_event(conn, task_id, "cap_extended",
                      {"by": by, "old": old, "new": new_cap, "reason": reason})
    return new_cap


def enforce_max_cost(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
    state_db_path=None,
) -> list[str]:
    """Block workers whose cumulative session spend exceeded ``max_cost``.

    Each heartbeat tick: for every host-local ``running`` task with a
    ``max_cost`` set, compare the card's cumulative estimated cost (sum of
    ``estimated_cost_usd`` across all its kanban worker sessions — see
    :func:`_cumulative_session_cost`) against the cap. When spend exceeds the
    cap:

      1. the host-local worker is SIGTERMed (then SIGKILLed after the grace
         window) via the SAME shared teardown the runtime-cap path and stale
         reclaim use — no second teardown implementation;
      2. the card is BLOCKED with ``kind='cost_cap'``.

    A ``cost_cap`` block is never re-queued: ``block_task`` parks the card in
    ``blocked`` (the dispatcher never re-promotes a blocked card) and the
    dependency-gate watchdog carries it to the jobsy triage lane like every
    other dead letter. Because enforcement only considers ``status='running'``
    cards, a blocked card cannot re-fire, so there is no double-SIGTERM and no
    retry loop.

    ``signal_fn`` and ``state_db_path`` are test hooks (defaults: ``os.kill``
    and the assignee's profile state.db). Returns the list of blocked task
    ids. Never raises for a ledger problem; a card with no workspace, no
    ledger, or an unreadable state.db is treated as $0 spend and left running.
    """
    cost_capped: list[str] = []
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"

    rows = conn.execute(
        "SELECT t.id, t.worker_pid, t.assignee, t.workspace_path, "
        "       t.max_cost, t.claim_lock "
        "FROM tasks t "
        "WHERE t.status = 'running' AND t.max_cost IS NOT NULL "
        "  AND t.claim_lock IS NOT NULL"
    ).fetchall()
    for row in rows:
        lock = row["claim_lock"] or ""
        if not lock.startswith(host_prefix):
            continue
        workspace = row["workspace_path"]
        if not workspace:
            continue
        cap = float(row["max_cost"])
        spend = _cumulative_session_cost(
            state_db_path or _state_db_path_for_assignee(row["assignee"]),
            workspace,
            # 2026-09-02: pass the card id. sessions.cwd is NULL on this fleet,
            # so workspace matching alone saw $0 and the cap never fired.
            task_id=row["id"],
            # 2026-09-06 (C1): assignee's ledger only — see _cumulative_session_cost.
            all_ledgers=False,
        )
        if spend <= cap:
            continue

        tid = row["id"]
        pid = row["worker_pid"]
        # SIGTERM -> grace -> SIGKILL, the single shared teardown.
        termination = _terminate_reclaimed_worker(
            pid, row["claim_lock"], signal_fn=signal_fn,
        )
        sigkill_used = bool(termination.get("sigkill"))
        worker_survived = bool(
            termination.get("termination_attempted")
            and not termination.get("terminated")
        )
        reason = f"cumulative spend ${spend:.2f} exceeded max_cost ${cap:.2f}"

        # BLOCK, never requeue. block_task parks the card in ``blocked`` with
        # kind ``cost_cap``; the dispatcher does not re-promote blocked cards,
        # so this run is terminal and the card flows to the jobsy triage lane.
        blocked = block_task(conn, tid, reason=reason, kind="cost_cap")
        if not blocked:
            # Task left 'running' concurrently (worker completed or was
            # reclaimed) — nothing to block. Log and move on.
            _log.debug("kanban cost-cap: %s left running before block", tid)
        try:
            add_comment(
                conn, tid, author="dispatcher",
                body=(
                    f"cost cap tripped ({reason}). "
                    f"Worker PID {pid or '(none)'} "
                    f"{'SIGKILLed' if sigkill_used else 'SIGTERMed'};"
                    f"{' worker survived termination' if worker_survived else ''}"
                ),
            )
        except Exception as exc:  # pragma: no cover - never fail the tick
            _log.warning("kanban cost-cap: comment write failed for %s: %s", tid, exc)
        cost_capped.append(tid)
    return cost_capped


def _parents_terminal_successful(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return True iff every parent of ``task_id`` has a terminal-successful
    status (``done`` or ``archived``). A task with no parents is a root and
    returns True.

    Mirrors the parent-completion status set used by ``recompute_ready``,
    ``claim_task``'s structural gate, and ``_landing_status_after_parents`` so
    the dispatcher's pre-spawn readiness check can never disagree with the
    promotion/claim gates about what "ready to run" means.
    """
    row = conn.execute(
        "SELECT 1 FROM task_links l "
        "JOIN tasks p ON p.id = l.parent_id "
        "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived') LIMIT 1",
        (task_id,),
    ).fetchone()
    return row is None


def _resolve_model_rule(
    title: str,
    rules: Optional[list] = None,
    kanban_cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Return the first ``model_rules`` entry whose ``match`` regex fires on
    ``title``, or ``None``.

    ``rules`` takes precedence (tests pass it explicitly); otherwise the
    rules are loaded from ``kanban.model_rules`` in config. Matching is
    case-insensitive and against the TITLE only. A malformed regex in a
    rule is skipped with a logged warning — it must never crash dispatch.
    """
    title = title or ""
    if rules is None:
        if kanban_cfg is None:
            try:
                from hermes_cli.config import load_config
                kanban_cfg = (load_config().get("kanban") or {})
            except Exception:
                kanban_cfg = {}
        rules = (kanban_cfg or {}).get("model_rules") or []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        match = rule.get("match")
        if not match:
            continue
        try:
            if re.search(match, title, re.IGNORECASE):
                return rule
        except re.error as exc:
            _log.warning(
                "kanban model_rule has malformed match regex %r (skipped): %s",
                match, exc,
            )
            continue
    return None


def apply_model_rule(
    task: Task,
    *,
    rules: Optional[list] = None,
    kanban_cfg: Optional[dict] = None,
) -> Optional[dict]:
    """Apply dispatch-time model routing to an in-memory ``Task``.

    If the card already has an explicit ``model_override``, it always wins
    and no rule fires. Otherwise the first matching rule sets the worker's
    model (and provider, when the rule names one) FOR THIS SPAWN ONLY — the
    card is never mutated, so a later dispatch or a re-run applies the rule
    freshly. Returns the fired rule (for the caller's audit event), else
    ``None``.
    """
    if task.model_override:
        return None
    rule = _resolve_model_rule(task.title, rules=rules, kanban_cfg=kanban_cfg)
    if not rule:
        return None
    model = rule.get("model")
    if not model:
        return None
    task.model_override = model
    if rule.get("provider"):
        task.provider_override = rule["provider"]
    return rule


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
    _RESPAWN_BLOCKER_RE,
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
