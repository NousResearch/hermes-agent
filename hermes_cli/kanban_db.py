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

try:  # KENSEI FIX: move helpers reference _fcntl/_os/_hashlib aliases never imported
    import fcntl as _fcntl
except ImportError:  # Windows
    _fcntl = None
import os as _os  # noqa: E402  (KENSEI FIX: move helper aliases)
import hashlib as _hashlib  # noqa: E402
import tempfile as _tempfile  # noqa: E402

from toolsets import get_toolset_names
# KENSEI CUSTOM (fork re-anchor): activity-ledger best-effort import — must never break board writes.
try:
    from hermes_cli.profile_activity_ledger import record_event_if_enabled
except Exception:  # pragma: no cover — ledger unavailable in stripped environments
    def record_event_if_enabled(**_kw):
        return None

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

VALID_STATUSES = {
    "triage", "todo", "scheduled", "ready", "running", "blocked", "review",
    "done", "archived", "backlog",
    # Feature pipeline stages (Phase A→D)
    "research", "prd", "spec", "council", "sign_off", "tech_review",
    "decompose", "execute", "pr+qa", "audit", "final_sign_off", "document",
    # Terminal blocked state set by ``block_task`` when a worker hits a
    # stakeholder-decision gate (``escalation_target IS NOT NULL`` and a
    # decision-kind block). Differs from ``blocked`` because it is meant
    # to be human-owned: ``unblock_task`` refuses to flip it, the
    # kensei-blocked-unblocker cron pre-check skips it, and dispatch
    # never claims it. The only legitimate exits are a direct operator
    # write (e.g. ``hermes kanban set-status ... done``) or reschedule.
    "decision-needed",
}
VALID_INITIAL_STATUSES = {"running", "blocked", "backlog"}  # KENSEI CUSTOM (restored): fork allowed backlog

# Typed block reasons (routing in ``_route_block``); ``None`` = legacy un-typed.
VALID_BLOCK_KINDS = {
    "dependency", "needs_input", "capability", "transient",
    # Decision kinds: blocks that need a human to decide before the task can
    # continue. Routed by ``block_task`` to the ``decision-needed`` terminal
    # status when ``escalation_target`` is set on the task, instead of the
    # generic ``blocked`` bucket. See ``DECISION_BLOCK_KINDS`` below.
    "stakeholder-decision", "escalation",
}

DECISION_BLOCK_KINDS = {"stakeholder-decision", "escalation"}

# Same-reason block -> unblock -> re-block cycles before routing to ``triage``.
# Counts unblock recurrences, NOT dispatcher failures (``DEFAULT_FAILURE_LIMIT``).
BLOCK_RECURRENCE_LIMIT = 2

# KENSEI CUSTOM (fork re-anchor):
VALID_TASK_KINDS = {"task", "bug", "spike", "subtask", "gate"}
DEFAULT_TASK_KIND = "task"
VALID_EPIC_STATUSES = {"active", "done", "archived"}

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


def assert_mutation_allowed() -> None:
    """Public domain guard for cross-store mutations such as provisioning."""
    _assert_not_delegated_child_mutation()


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
def scoped_kanban_home(root: str | Path):
    """Temporarily pin Kanban storage to ``root`` for this context only."""
    token: Token[Path | None] = _KANBAN_HOME_OVERRIDE.set(Path(root))
    try:
        yield
    finally:
        _KANBAN_HOME_OVERRIDE.reset(token)


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


_KANBAN_HOME_OVERRIDE: ContextVar[Path | None] = ContextVar(
    "hermes_kanban_home_override",
    default=None,
)

# KENSEI CUSTOM (fork re-anchor):
def normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Public board-slug normalizer for host validation contracts."""
    return _normalize_board_slug(slug)


def kanban_home() -> Path:
    """``HERMES_KANBAN_HOME`` else ``get_default_hermes_root()``. Shared across
    profiles BY DESIGN: resolving through the active profile's HERMES_HOME would
    fork the board per profile and break the dispatcher/worker handoff."""
    # KENSEI CUSTOM (fork re-anchor): context-local override wins first.
    scoped = _KANBAN_HOME_OVERRIDE.get()
    if scoped is not None:
        return scoped
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
    # KENSEI CUSTOM (fork re-anchor): theme tag + tier + pipeline fields.
    theme: Optional[str] = None
    tier: Optional[str] = None
    pipeline_stage: Optional[str] = None
    pipeline_mode: Optional[str] = None
    status_reason: Optional[str] = None
    reviewer: Optional[str] = None
    task_kind: str = DEFAULT_TASK_KIND
    parent_task_id: Optional[str] = None
    epic_id: Optional[str] = None
    done_at: Optional[int] = None
    archived_at: Optional[int] = None
    # VALID_BLOCK_KINDS or None (legacy); kept across unblock so a same-kind re-block reads as a loop.
    block_kind: Optional[str] = None
    block_recurrences: int = 0               # unblock-loop counter, see BLOCK_RECURRENCE_LIMIT

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
            # KENSEI CUSTOM (fork re-anchor): task_kind defaults for legacy rows
            # (other fork columns ride the _TASK_OPTIONAL_COLUMNS loop above).
            task_kind=g("task_kind") or DEFAULT_TASK_KIND,
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
    "current_step_key", "max_retries", "session_id",
    # KENSEI CUSTOM (fork re-anchor): fork-only task columns.
    "theme", "tier", "pipeline_stage", "pipeline_mode", "status_reason", "reviewer",
    "parent_task_id", "epic_id", "done_at", "archived_at",
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

@dataclass
class Epic:
    """In-memory view of a row from the ``epics`` table (Kanban v2)."""

    id: str
    title: str
    description: Optional[str]
    board_slug: Optional[str]
    status: str
    parent_epic_id: Optional[str]
    created_at: int
    updated_at: int

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Epic":
        keys = set(row.keys())
        return cls(
            id=row["id"],
            title=row["title"],
            description=row["description"] if "description" in keys else None,
            board_slug=row["board_slug"] if "board_slug" in keys else None,
            status=row["status"] if "status" in keys else "active",
            parent_epic_id=row["parent_epic_id"] if "parent_epic_id" in keys else None,
            created_at=int(row["created_at"]),
            updated_at=int(row["updated_at"]),
        )

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
    -- KENSEI CUSTOM (fork re-anchor): human-escalation target for the daily
    -- briefing "NEEDS YOU" section; NULL = no escalation pending.
    escalation_target    TEXT,
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
    block_recurrences    INTEGER NOT NULL DEFAULT 0,
    -- KENSEI CUSTOM (fork re-anchor): flat theme tag for grouping related work.
    theme                TEXT,
    -- KENSEI CUSTOM (fork re-anchor): canonical task kind (VALID_TASK_KINDS).
    task_kind            TEXT NOT NULL DEFAULT 'task',
    -- KENSEI CUSTOM (fork re-anchor): non-blocking containment parent (NOT a dependency).
    parent_task_id       TEXT,
    -- KENSEI CUSTOM (fork re-anchor): tier classification set at triage ('fast' | 'full').
    tier                 TEXT,
    -- KENSEI CUSTOM (fork re-anchor): current feature-pipeline stage (NULL = not in pipeline).
    pipeline_stage       TEXT,
    -- KENSEI CUSTOM (fork re-anchor): pipeline mode ('full' | 'express'; NULL = full).
    pipeline_mode        TEXT,
    -- KENSEI CUSTOM (fork re-anchor): Kanban v2 epic grouping (FK to epics.id).
    epic_id              TEXT,
    -- KENSEI CUSTOM (fork re-anchor): done→archive automation timestamps.
    done_at              INTEGER,
    archived_at          INTEGER
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
CREATE INDEX IF NOT EXISTS idx_events_kind           ON task_events(kind, created_at);
-- KENSEI CUSTOM (fork re-anchor): Kanban v2 epics — JIRA-style epic/feature grouping.
CREATE TABLE IF NOT EXISTS epics (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT,
    board_slug      TEXT,
    status          TEXT NOT NULL DEFAULT 'active',
    parent_epic_id  TEXT,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    FOREIGN KEY (parent_epic_id) REFERENCES epics(id)
);

-- KENSEI CUSTOM (fork re-anchor): pending human-approval records for profile
-- lifecycle ops (create / delete); resolved from Discord approve/reject.
CREATE TABLE IF NOT EXISTS profile_lifecycle_approvals (
    id            TEXT PRIMARY KEY,
    task_id       TEXT NOT NULL,
    op            TEXT NOT NULL,
    profile       TEXT NOT NULL,
    args_json     TEXT NOT NULL DEFAULT '{}',
    requested_by  TEXT,
    blast_summary TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    token         TEXT NOT NULL,
    error         TEXT,
    resolved_by   TEXT,
    created_at    INTEGER NOT NULL,
    resolved_at   INTEGER,
    notified_at   INTEGER
);

-- KENSEI CUSTOM (fork re-anchor): cost governance — daily agent-spawn counter
-- (P2-1), one row per UTC date, recycled by the dispatcher tick.
CREATE TABLE IF NOT EXISTS daily_spawn_counter (
    date_utc   TEXT NOT NULL PRIMARY KEY,
    count      INTEGER NOT NULL DEFAULT 0,
    last_tick  INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_profile_approvals_status ON profile_lifecycle_approvals(status, created_at);
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
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: Optional[str] = None,
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    tenant: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
    triage: bool = False,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    model_override: Optional[str] = None,
    provider_override: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    goal_mode: bool = False,
    goal_max_turns: Optional[int] = None,
    initial_status: str = "running",
    session_id: Optional[str] = None,
    board: Optional[str] = None,
    project_id: Optional[str] = None,
    # KENSEI CUSTOM port: upstream's creator_task_id (ACK-edge origin inheritance;
    # the graph helper is imported inside create_task below).
    creator_task_id: Optional[str] = None,
    theme: Optional[str] = None,
    # WS-3 tier inheritance: when children are created from a decomposed
    # full-tier parent, they inherit the parent's tier so WS-1 (contract
    # gate) and WS-4 (review gate) both apply to them. NULL = untyped.
    tier: Optional[str] = None,
    # WS-2 universal subscription: auto-subscribe a gateway source
    # to the new task's terminal events on creation. Accepts the
    # same args as add_notify_sub(). When set, the subscription
    # is created atomically with the task in the same transaction.
    notify_platform: Optional[str] = None,
    notify_chat_id: Optional[str] = None,
    notify_thread_id: Optional[str] = None,
    notify_user_id: Optional[str] = None,
    notify_profile: Optional[str] = None,
    # Phase D: per-task pipeline mode selector. 'full' (default) walks all
    # 12 stages; 'express' skips PRD/Council/Tech Review/Audit (design
    # doc §3 express path). Stored on the task so dispatch_once branches
    # on it without re-reading config.
    pipeline_mode: Optional[str] = None,
    project_source_task_id: Optional[str] = None,
    # P2a: canonical task kind. One of VALID_TASK_KINDS. Defaults to 'task'.
    task_kind: Optional[str] = None,
    # P2a: non-blocking containment parent. Must exist on the SAME board;
    # never treated as a dependency (no readiness gating, no task_links row).
    parent_task_id: Optional[str] = None,
    # P2a: attach the new task to an epic (same-board). Validated on create.
    epic_id: Optional[str] = None,
) -> str:
    """Create a new task and optionally link it under parent tasks.

    Returns the new task id.  Status is ``ready`` when there are no
    parents (or all parents already ``done``), otherwise ``todo``.
    If ``triage=True``, status is forced to ``triage`` regardless of
    parents — a specifier/triager is expected to promote the task to
    ``todo`` once the spec is fleshed out.

    If ``idempotency_key`` is provided and a non-archived task with the
    same key already exists, returns the existing task's id instead of
    creating a duplicate. Useful for retried webhooks / automation that
    should not double-write.

    ``max_runtime_seconds`` caps how long a worker may run before the
    dispatcher SIGTERMs (then SIGKILLs after a grace window) and
    re-queues the task. ``None`` means no cap (default).

    ``skills`` is an optional list of skill names to force-load into
    the worker when dispatched. Stored as JSON; the dispatcher passes
    each name to ``hermes --skills ...``. Use this to pin a task to a
    specialist skill (e.g. ``skills=["translation"]`` so the worker loads the
    translation skill regardless of the profile's default config).

    ``model_override`` / ``provider_override`` pin the worker to a specific
    model (and optionally its provider) without touching the profile's
    config — passed to the worker as ``-m <model> [--provider <name>]``.
    ``provider_override`` requires ``model_override``.

    ``reasoning_effort`` pins the worker's thinking depth for this task
    (``minimal``…``ultra``, or ``none`` to disable thinking), passed as
    ``--reasoning <level>``. It is independent of ``model_override``: a task
    can run the profile's own model at a different depth.

    ``project_source_task_id`` is an internal cross-profile fallback for a
    worker-created child. When the active profile cannot resolve ``project_id``
    in its own projects.db, a matching canonical project-linked task in this
    board can supply the repo and branch convention. Its literal worktree is
    never reused; the new task still gets its own task-id-keyed path.
    ``creator_task_id``: inherit durable session/subscriptions independently of
    dependency edges; an explicit ``session_id`` still wins.
    ``workspace_kind=None`` (omitted) inherits a project-scoped board's project;
    an explicit ``"scratch"`` or ``project_id=""`` is a request for no project.
    """
    from hermes_cli.kanban_db_graph import inherit_creator_origin
    model_override = (model_override or "").strip() or None
    provider_override = (provider_override or "").strip() or None
    reasoning_effort = normalize_reasoning_effort(reasoning_effort)
    if provider_override and not model_override:
        raise ValueError("provider_override requires a model_override")
    # ── P2a: validate task_kind + containment + epic attachment ──────
    task_kind = (task_kind or "").strip().lower() or DEFAULT_TASK_KIND
    if task_kind not in VALID_TASK_KINDS:
        raise ValueError(
            f"task_kind must be one of {sorted(VALID_TASK_KINDS)}, got {task_kind!r}"
        )
    parent_task_id = (parent_task_id or "").strip() or None
    if task_kind == "subtask" and parent_task_id is None:
        raise ValueError("subtask requires parent_task_id")
    epic_id = (epic_id or "").strip() or None
    assignee = _canonical_assignee(assignee)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(
            f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}"
        )

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

    # Idempotency check — return the existing task instead of creating a
    # duplicate. Done BEFORE entering write_txn to keep the fast path fast
    # and to avoid holding a write lock during the lookup. Race is
    # acceptable: two concurrent creators with the same key might both
    # insert, at which point both rows exist but the next lookup stabilises.
    if idempotency_key:
        row = conn.execute(
            "SELECT id FROM tasks WHERE idempotency_key = ? "
            "AND status != 'archived' "
            "ORDER BY created_at DESC LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if row:
            return row["id"]

    now = int(time.time())

    # Resolve workspace_path from board-level default_workdir when the
    # caller did not specify one explicitly. Board defaults represent
    # persistent project checkouts, so only persistent workspace kinds may
    # inherit them. Scratch workspaces are auto-deleted on completion and
    # must stay under the per-board scratch root created by
    # ``resolve_workspace``; inheriting ``default_workdir`` for a scratch
    # task would point cleanup at the user's source tree (#28818). The
    # containment guard in ``_cleanup_workspace`` is the safety rail, but
    # we also stop the bad state from being created in the first place.
    if (
        workspace_path is None
        and project_repo is None
        and workspace_kind in {"dir", "worktree"}
    ):
        board_slug = board if board else get_current_board()
        board_meta = read_board_metadata(board_slug)
        board_default = board_meta.get("default_workdir")
        if board_default:
            workspace_path = str(board_default)

    # Retry once on the extremely unlikely id collision.
    for attempt in range(2):
        task_id = _new_task_id()
        try:
            # ``allow_nested=True``: graph builders (kanban_swarm.create_swarm)
            # compose create_task calls under one outer commit so the
            # dispatcher can never observe a partially constructed graph.
            with write_txn(conn, allow_nested=True):
                # Determine task status from parent status, unless the caller
                # parks it directly in blocked for human-ops review or in
                # triage for a specifier.
                if initial_status == "blocked":
                    task_status = "blocked"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                elif initial_status == "backlog":
                    # Backlog is a non-auto-promoting landing pad for
                    # approved follow-up specs. Parents are still validated
                    # so future ``hermes kanban promote`` doesn't dangle.
                    task_status = "backlog"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                elif triage:
                    task_status = "triage"
                else:
                    task_status = "ready"
                    if parents:
                        missing = _find_missing_parents(conn, parents)
                        if missing:
                            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
                        # If any parent is not yet done, we're todo.
                        rows = conn.execute(
                            "SELECT status FROM tasks WHERE id IN "
                            "(" + ",".join("?" * len(parents)) + ")",
                            parents,
                        ).fetchall()
                        if any(r["status"] != "done" for r in rows):
                            task_status = "todo"
                # Even in triage mode we still need to validate parent ids
                # so the eventual link rows don't dangle.
                if triage and parents:
                    missing = _find_missing_parents(conn, parents)
                    if missing:
                        raise ValueError(f"unknown parent task(s): {', '.join(missing)}")

                # ── P2a: validate non-blocking containment parent ──────
                # parent_task_id is NOT a dependency: no task_links row, no
                # readiness gating. It must exist on THIS board and must not
                # be the task itself.
                if parent_task_id is not None:
                    if parent_task_id == task_id:
                        raise ValueError("a task cannot contain itself")
                    parent_row = conn.execute(
                        "SELECT id FROM tasks WHERE id = ?", (parent_task_id,)
                    ).fetchone()
                    if parent_row is None:
                        raise ValueError(
                            f"unknown parent_task_id: {parent_task_id!r} "
                            "(containment parent must exist on the same board)"
                        )

                # ── P2a: validate epic attachment (same-board) ─────────
                if epic_id is not None:
                    _validate_epic_for_attachment(conn, epic_id, board_slug_for_epic=board)

                # Project-linked worktree: a fresh worktree dir under the repo
                # plus a deterministic branch (project slug + task id). Together
                # these kill the random ``wt/<task-id>`` worker fallback and the
                # unanchored ``.worktrees/<id>`` under the dispatcher's cwd.
                if project_obj is not None and workspace_kind == "worktree":
                    if project_repo and not workspace_path:
                        workspace_path = os.path.join(
                            project_repo, ".worktrees", task_id
                        )
                    if not branch_name:
                        # KENSEI re-anchor: _pdb used to be imported by the inline
                        # project-resolution block; _resolve_project_link replaced
                        # that block, so import it here (best-effort, never fatal).
                        try:
                            from hermes_cli import projects_db as _pdb
                            branch_name = _pdb.branch_name_for(
                                project_obj, task_id, title=title or ""
                            )
                        except Exception:
                            branch_name = None

                conn.execute(
                    """
                    INSERT INTO tasks (
                        id, title, body, assignee, status, priority,
                        created_by, created_at, workspace_kind, workspace_path,
                        branch_name, project_id, tenant, idempotency_key,
                        max_runtime_seconds, skills, max_retries,
                        model_override, provider_override, reasoning_effort,
                        goal_mode, goal_max_turns, session_id, theme, tier, pipeline_mode,
                        task_kind, parent_task_id, epic_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        title.strip(),
                        body,
                        assignee,
                        task_status,
                        priority,
                        created_by,
                        now,
                        workspace_kind,
                        workspace_path,
                        branch_name,
                        project_id,
                        tenant,
                        idempotency_key,
                        int(max_runtime_seconds) if max_runtime_seconds is not None else None,
                        json.dumps(skills_list) if skills_list is not None else None,
                        int(max_retries) if max_retries is not None else None,
                        model_override,
                        provider_override,
                        reasoning_effort,
                        1 if goal_mode else 0,
                        int(goal_max_turns) if goal_max_turns is not None else None,
                        session_id,
                        theme.strip() if isinstance(theme, str) and theme.strip() else None,
                        tier.strip() if isinstance(tier, str) and tier.strip() else None,
                        pipeline_mode.strip() if isinstance(pipeline_mode, str) and pipeline_mode.strip() else None,
                        task_kind,
                        parent_task_id,
                        epic_id,
                    ),
                )
                for pid in parents:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                        (pid, task_id),
                    )
                # Notify-sub inheritance (ACK-edge: the originating channel
                # still hears about a child that BLOCKs, not just the final
                # fan-in) is handled by the single-owner helper below —
                # _inherit_notify_subs copies every routing/delivery column.
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "workspace_kind": workspace_kind,
                        "workspace_path": workspace_path,
                        "branch_name": branch_name,
                        "project_id": project_id,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                        "model_override": model_override,
                        "provider_override": provider_override,
                        "task_kind": task_kind,
                        "parent_task_id": parent_task_id,
                        "epic_id": epic_id,
                    },
                )
                if task_status == "blocked":
                    _append_event(
                        conn,
                        task_id,
                        "blocked",
                        {"reason": "initial_status", "status": "blocked", "actor": created_by or "user"},
                    )
                # ACK-edge: the originating channel hears a child BLOCK, not just the fan-in.
                inherit_creator_origin(conn, task_id, creator_task_id, created_at=now)
                _inherit_notify_subs(conn, task_id, parents, created_at=now)
            # WS-2 universal subscription: if the caller passed notify_
            # params, create the subscription inside the write_txn. This
            # covers ALL task creation paths — cron scripts, agent tool
            # calls, voice intake — not just /kanban create slash commands.
            if notify_platform and notify_chat_id:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO kanban_notify_subs
                        (task_id, platform, chat_id, thread_id, user_id,
                         notifier_profile, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task_id, notify_platform, notify_chat_id,
                        notify_thread_id or "", notify_user_id,
                        notify_profile, now,
                    ),
                )
            record_event_if_enabled(
                source="kanban.db",
                actor_profile=created_by,
                target_profile=assignee,
                event_type="kanban.task.created",
                object_type="kanban_task",
                object_id=task_id,
                board=board,
                status_to=task_status,
                summary=f"Created kanban task {task_id}",
                payload={
                    "title": title.strip(),
                    "parents": list(parents),
                    "tenant": tenant,
                    "skills": list(skills_list) if skills_list else None,
                },
            )
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _board_meta_for(board: Optional[str]) -> dict:
    return read_board_metadata(board if board else get_current_board())


def _initial_task_status(
    conn: sqlite3.Connection, parents: tuple[str, ...], initial_status: str, triage: bool,
) -> str:
    """Status for a new task: ``blocked``/``triage`` when parked by the caller,
    else ``ready`` unless a parent is not yet ``done`` (-> ``todo``). Parent ids
    are validated in every mode (even triage) so link rows never dangle."""
    if parents:
        missing = _missing_task_ids(conn, parents)
        if missing:
            raise ValueError(f"unknown parent task(s): {', '.join(missing)}")
    if initial_status == "blocked":
        return "blocked"
    if triage:
        return "triage"
    if parents:
        rows = conn.execute(
            "SELECT status FROM tasks WHERE id IN "
            "(" + ",".join("?" * len(parents)) + ")", parents,
        ).fetchall()
        if any(r["status"] != "done" for r in rows):
            return "todo"
    return "ready"


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


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    """KENSEI CUSTOM (fork re-anchor): parent-id validation used by create_task /
    decompose paths (same semantics as :func:`_missing_task_ids`)."""
    return _missing_task_ids(conn, parents)


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
    theme: Optional[str] = None, epic_id: Optional[str] = None,
) -> list[Task]:
    if status is not None and status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    # KENSEI CUSTOM (fork re-anchor): theme / epic_id filters.
    for col, val in (
        ("assignee", _canonical_assignee(assignee)), ("status", status), ("tenant", tenant),
        ("session_id", session_id), ("workflow_template_id", workflow_template_id),
        ("current_step_key", current_step_key), ("theme", theme), ("epic_id", epic_id),
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

_TASK_UPDATE_UNSET = object()


def update_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Any = _TASK_UPDATE_UNSET,
    body: Any = _TASK_UPDATE_UNSET,
    priority: Any = _TASK_UPDATE_UNSET,
    task_kind: Any = _TASK_UPDATE_UNSET,
    parent_task_id: Any = _TASK_UPDATE_UNSET,
    epic_id: Any = _TASK_UPDATE_UNSET,
    board: Optional[str] = None,
) -> Optional[Task]:
    """Atomically edit canonical task fields and return the updated task.

    Omitted fields remain unchanged. ``body=None``, ``parent_task_id=None`` and
    ``epic_id=None`` explicitly clear those fields. Containment stays separate
    from dependency links and all hierarchy checks happen before the write.
    """
    task_id = (task_id or "").strip()
    if not task_id:
        raise ValueError("task_id must be non-empty")

    updates: list[str] = []
    params: list[Any] = []
    changed_fields: list[str] = []
    with write_txn(conn):
        current = conn.execute(
            "SELECT task_kind, parent_task_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if current is None:
            return None

        resulting_kind = current["task_kind"] or DEFAULT_TASK_KIND
        resulting_parent = current["parent_task_id"]

        if title is not _TASK_UPDATE_UNSET:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("title must be non-empty text")
            updates.append("title = ?")
            params.append(title.strip())
            changed_fields.append("title")

        if body is not _TASK_UPDATE_UNSET:
            if body is not None and not isinstance(body, str):
                raise ValueError("body must be text or null")
            updates.append("body = ?")
            params.append(body)
            changed_fields.append("body")

        if priority is not _TASK_UPDATE_UNSET:
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ValueError("priority must be an integer")
            updates.append("priority = ?")
            params.append(priority)
            changed_fields.append("priority")

        if task_kind is not _TASK_UPDATE_UNSET:
            if not isinstance(task_kind, str):
                raise ValueError("task_kind must be text")
            resulting_kind = task_kind.strip().lower()
            if resulting_kind not in VALID_TASK_KINDS:
                raise ValueError(
                    f"task_kind must be one of {sorted(VALID_TASK_KINDS)}"
                )
            updates.append("task_kind = ?")
            params.append(resulting_kind)
            changed_fields.append("task_kind")

        if parent_task_id is not _TASK_UPDATE_UNSET:
            if parent_task_id is not None and not isinstance(parent_task_id, str):
                raise ValueError("parent_task_id must be text or null")
            resulting_parent = (
                (parent_task_id or "").strip()
                if parent_task_id is not None
                else None
            ) or None
            _validate_containment_parent(conn, task_id, resulting_parent)
            updates.append("parent_task_id = ?")
            params.append(resulting_parent)
            changed_fields.append("parent_task_id")

        if resulting_kind == "subtask" and resulting_parent is None:
            raise ValueError("subtask requires parent_task_id")

        if epic_id is not _TASK_UPDATE_UNSET:
            if epic_id is not None and not isinstance(epic_id, str):
                raise ValueError("epic_id must be text or null")
            normalized_epic = (
                (epic_id or "").strip() if epic_id is not None else None
            ) or None
            if normalized_epic is not None:
                _validate_epic_for_attachment(
                    conn,
                    normalized_epic,
                    board_slug_for_epic=board,
                )
            updates.append("epic_id = ?")
            params.append(normalized_epic)
            changed_fields.append("epic_id")

        if updates:
            params.append(task_id)
            conn.execute(
                f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            _append_event(
                conn,
                task_id,
                "edited",
                {"fields": sorted(changed_fields)},
            )

    if changed_fields:
        notify_task_updated(conn, task_id, changed_fields)
    return get_task(conn, task_id)


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

def _new_epic_id() -> str:
    return "epic_" + secrets.token_hex(4)


def _epic_board_slug(board: Optional[str]) -> Optional[str]:
    """Resolve the board an epic will live on (normalised, default if None)."""
    return _normalize_board_slug(board) or DEFAULT_BOARD


def create_epic(
    conn: sqlite3.Connection,
    *,
    title: str,
    description: Optional[str] = None,
    board_slug: Optional[str] = None,
    parent_epic_id: Optional[str] = None,
    status: str = "active",
    epic_id: Optional[str] = None,
) -> str:
    """Create an epic and return its id.

    Validates the title, status, and — when set — that ``parent_epic_id``
    exists on the same board and does not introduce a hierarchy cycle.
    ``epic_id`` allows an explicit id (used by tests and idempotent callers).
    """
    title = (title or "").strip()
    if not title:
        raise ValueError("epic title is required")
    status = (status or "").strip().lower() or "active"
    if status not in VALID_EPIC_STATUSES:
        raise ValueError(
            f"epic status must be one of {sorted(VALID_EPIC_STATUSES)}, got {status!r}"
        )
    board_slug = _epic_board_slug(board_slug)
    parent_epic_id = (parent_epic_id or "").strip() or None
    now = int(time.time())
    with write_txn(conn):
        if parent_epic_id is not None:
            _validate_epic_parent(conn, parent_epic_id, board_slug)
        if epic_id is not None:
            epic_id = str(epic_id).strip()
            if not epic_id:
                raise ValueError("epic_id must be non-empty")
        else:
            epic_id = _new_epic_id()
        conn.execute(
            "INSERT INTO epics (id, title, description, board_slug, status, "
            "parent_epic_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (epic_id, title, description, board_slug, status, parent_epic_id, now, now),
        )
    return epic_id


def get_epic(conn: sqlite3.Connection, epic_id: str) -> Optional[Epic]:
    row = conn.execute("SELECT * FROM epics WHERE id = ?", (epic_id,)).fetchone()
    return Epic.from_row(row) if row else None


def list_epics(
    conn: sqlite3.Connection,
    *,
    board_slug: Optional[str] = None,
    parent_epic_id: Optional[str] = None,
) -> list[Epic]:
    query = "SELECT * FROM epics WHERE 1=1"
    params: list[Any] = []
    if board_slug is not None:
        query += " AND board_slug = ?"
        params.append(_epic_board_slug(board_slug))
    if parent_epic_id is not None:
        query += " AND parent_epic_id = ?"
        params.append(parent_epic_id)
    query += " ORDER BY created_at ASC, id ASC"
    return [Epic.from_row(r) for r in conn.execute(query, params).fetchall()]


_EPIC_UNSET = object()


def update_epic(
    conn: sqlite3.Connection,
    epic_id: str,
    *,
    title: Any = _EPIC_UNSET,
    description: Any = _EPIC_UNSET,
    status: Any = _EPIC_UNSET,
    parent_epic_id: Any = _EPIC_UNSET,
) -> Optional[Epic]:
    """Update an epic atomically and return its new canonical representation.

    Omitted values remain unchanged. ``description=None`` clears the description
    and ``parent_epic_id=None`` detaches the epic from its parent. Hierarchy
    updates validate same-board identity and reject self-parenting and cycles.
    """
    epic_id = (epic_id or "").strip()
    if not epic_id:
        raise ValueError("epic_id must be non-empty")

    updates: list[str] = []
    params: list[Any] = []
    with write_txn(conn):
        row = conn.execute(
            "SELECT board_slug FROM epics WHERE id = ?", (epic_id,)
        ).fetchone()
        if row is None:
            return None
        board_slug = row["board_slug"] or DEFAULT_BOARD

        if title is not _EPIC_UNSET:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("epic title is required")
            updates.append("title = ?")
            params.append(title.strip())

        if description is not _EPIC_UNSET:
            if description is not None and not isinstance(description, str):
                raise ValueError("epic description must be text or null")
            updates.append("description = ?")
            params.append(description)

        if status is not _EPIC_UNSET:
            if not isinstance(status, str) or not status.strip():
                raise ValueError("epic status is required")
            normalized_status = status.strip().lower()
            if normalized_status not in VALID_EPIC_STATUSES:
                raise ValueError(
                    f"epic status must be one of {sorted(VALID_EPIC_STATUSES)}, "
                    f"got {normalized_status!r}"
                )
            updates.append("status = ?")
            params.append(normalized_status)

        if parent_epic_id is not _EPIC_UNSET:
            if parent_epic_id is not None and not isinstance(parent_epic_id, str):
                raise ValueError("parent_epic_id must be text or null")
            normalized_parent = (
                (parent_epic_id or "").strip() if parent_epic_id is not None else None
            ) or None
            if normalized_parent is not None:
                _validate_epic_parent(conn, normalized_parent, board_slug)
                if normalized_parent == epic_id:
                    raise ValueError("an epic cannot be its own parent")
                _assert_no_epic_cycle(conn, epic_id, normalized_parent)
            updates.append("parent_epic_id = ?")
            params.append(normalized_parent)

        if updates:
            updates.append("updated_at = ?")
            params.extend((int(time.time()), epic_id))
            conn.execute(
                f"UPDATE epics SET {', '.join(updates)} WHERE id = ?", params
            )

    return get_epic(conn, epic_id)


def set_epic_parent(
    conn: sqlite3.Connection, epic_id: str, parent_epic_id: Optional[str]
) -> bool:
    """Reparent an epic (hierarchy). Validates same-board existence + cycles."""
    parent_epic_id = (parent_epic_id or "").strip() or None
    with write_txn(conn):
        row = conn.execute(
            "SELECT board_slug FROM epics WHERE id = ?", (epic_id,)
        ).fetchone()
        if row is None:
            return False
        board_slug = row["board_slug"] or DEFAULT_BOARD
        if parent_epic_id is not None:
            _validate_epic_parent(conn, parent_epic_id, board_slug)
            if parent_epic_id == epic_id:
                raise ValueError("an epic cannot be its own parent")
            _assert_no_epic_cycle(conn, epic_id, parent_epic_id)
        conn.execute(
            "UPDATE epics SET parent_epic_id = ?, updated_at = ? WHERE id = ?",
            (parent_epic_id, int(time.time()), epic_id),
        )
    return True


def _validate_epic_parent(
    conn: sqlite3.Connection, parent_epic_id: str, board_slug: Optional[str]
) -> None:
    """Fail closed: the parent epic must exist on the SAME board."""
    row = conn.execute(
        "SELECT board_slug FROM epics WHERE id = ?", (parent_epic_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown parent_epic_id: {parent_epic_id!r}")
    parent_board = row["board_slug"] or DEFAULT_BOARD
    expected = board_slug or DEFAULT_BOARD
    if parent_board != expected:
        raise ValueError(
            f"epic {parent_epic_id!r} is on board {parent_board!r}, "
            f"not {expected!r}"
        )


def _assert_no_epic_cycle(
    conn: sqlite3.Connection, epic_id: str, parent_epic_id: str
) -> None:
    """Raise if parenting ``epic_id`` under ``parent_epic_id`` closes a cycle."""
    seen: set[str] = set()
    cursor: Optional[str] = parent_epic_id
    while cursor is not None and cursor not in seen:
        if cursor == epic_id:
            raise ValueError(
                f"setting parent_epic_id={parent_epic_id!r} on {epic_id!r} "
                "would create an epic hierarchy cycle"
            )
        seen.add(cursor)
        row = conn.execute(
            "SELECT parent_epic_id FROM epics WHERE id = ?", (cursor,)
        ).fetchone()
        cursor = row["parent_epic_id"] if row else None


def _validate_epic_for_attachment(
    conn: sqlite3.Connection, epic_id: str, board_slug_for_epic: Optional[str] = None
) -> None:
    """Fail closed: an epic referenced by a task must exist on this board."""
    row = conn.execute(
        "SELECT board_slug FROM epics WHERE id = ?", (epic_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"unknown epic_id: {epic_id!r}")
    epic_board = row["board_slug"] or DEFAULT_BOARD
    expected = _epic_board_slug(board_slug_for_epic)
    if epic_board != expected:
        raise ValueError(
            f"epic {epic_id!r} is on board {epic_board!r}, not {expected!r}"
        )


def set_task_epic(
    conn: sqlite3.Connection,
    task_id: str,
    epic_id: Optional[str],
    *,
    board: Optional[str] = None,
) -> bool:
    """Attach (or detach) a task to an epic on the selected board.

    ``board`` is explicit for non-default board connections. Containment is
    validated against that canonical board and never creates a dependency.
    """
    epic_id = (epic_id or "").strip() or None
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
            return False
        if epic_id is not None:
            _validate_epic_for_attachment(conn, epic_id, board_slug_for_epic=board)
        conn.execute(
            "UPDATE tasks SET epic_id = ? WHERE id = ?", (epic_id, task_id)
        )
    return True


def clear_task_epic(conn: sqlite3.Connection, task_id: str) -> bool:
    """Detach a task from its epic. Returns False when task absent."""
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET epic_id = NULL WHERE id = ?", (task_id,)
        )
    return cur.rowcount > 0


def epic_task_counts(conn: sqlite3.Connection) -> dict[str, dict[str, int]]:
    """Return ``{epic_id: {'total': N, <status>: N, ...}}`` across all epics."""
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT epic_id, status, COUNT(*) AS n FROM tasks "
        "WHERE epic_id IS NOT NULL GROUP BY epic_id, status"
    ).fetchall():
        bucket = counts.setdefault(row["epic_id"], {"total": 0})
        bucket[row["status"]] = int(row["n"])
        bucket["total"] += int(row["n"])
    return counts


def _add_comment_inline(
    conn: sqlite3.Connection, task_id: str, *, author: str, body: str,
) -> int:
    """Insert a comment row inside an already-open write_txn.

    Helper for callers that need to bundle the comment with other state
    changes in a single atomic write. Same input contract as
    ``add_comment`` but no transaction of its own.
    """
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    if not conn.execute(
        "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
    ).fetchone():
        raise ValueError(f"unknown task {task_id}")
    now = int(time.time())
    # ``allow_nested=True``: graph builders (kanban_swarm blackboard seeding)
    # compose comment writes under one outer commit.
    with write_txn(conn, allow_nested=True):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, author.strip(), body.strip(), now),
        )
        _append_event(conn, task_id, "commented", {"author": author, "len": len(body)})
        return int(cur.lastrowid or 0)


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
    conn: sqlite3.Connection,
    task_id: str,
    kind: str,
    payload: Optional[dict] = None,
    *,
    run_id: Optional[int] = None,
) -> None:
    """Record an event row.  Called from within an already-open txn.

    ``run_id`` is optional: pass the current run id so UIs can group
    events by attempt. For events that aren't scoped to a single run
    (task created/edited/archived, dependency promotion) leave it None
    and the row carries NULL.
    """
    now = int(time.time())
    pl = json.dumps(payload, ensure_ascii=False) if payload else None
    cur = conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, pl, now),
    )
    try:
        task_row = conn.execute(
            "SELECT assignee, status FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        record_event_if_enabled(
            source="kanban.db",
            actor_profile=os.environ.get("HERMES_PROFILE"),
            target_profile=(task_row["assignee"] if task_row else None),
            event_type=f"kanban.{kind}",
            object_type="kanban_task",
            object_id=task_id,
            board=get_current_board(),
            status_to=(task_row["status"] if task_row else None),
            summary=f"Kanban event {kind} for {task_id}",
            payload={"run_id": run_id, "kind": kind, "payload": payload},
            event_id=f"kanban:{get_current_board()}:{task_id}:event:{cur.lastrowid}",
        )
    except Exception:
        # Ledger hooks are best-effort and must never break board writes.
        pass


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


def _recompute_ready_locked(
    conn: sqlite3.Connection,
    failure_limit: int = None,
    *,
    task_filter: Optional[str] = None,
) -> int:
    """Transaction-aware core of :func:`recompute_ready`.

    Does **not** open ``write_txn`` — the caller must already hold a
    write transaction (or an attached multi-database transaction).  When
    ``task_filter`` is set, only that task id is recomputed; otherwise
    all ``todo``/``blocked`` rows on the main database are scanned.

    ``task_filter`` is used by the cross-board move helper to recompute
    only the moved task (on the target attached DB) and affected children
    (on the source main DB) without opening a nested ``write_txn``.
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    if task_filter:
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE id = ? AND status IN ('todo', 'blocked')",
            (task_filter,),
        ).fetchall()
    else:
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
    for row in todo_rows:
        task_id = row["id"]
        cur_status = row["status"]
        if cur_status == "blocked" and _has_sticky_block(conn, task_id):
            continue
        parents = conn.execute(
            "SELECT t.status FROM tasks t "
            "JOIN task_links l ON l.parent_id = t.id "
            "WHERE l.child_id = ?",
            (task_id,),
        ).fetchall()
        if all(p["status"] in ("done", "archived") for p in parents):
            resume_status = _resume_status_from_events(conn, task_id)
            if cur_status == "blocked":
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
                    "WHERE id = ? AND status = 'blocked'",
                    (resume_status, task_id),
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


def validate_task_contract(
    conn: sqlite3.Connection, task_id: str
) -> Optional[str]:
    """Input-presence check: full-tier tasks must carry Acceptance Criteria and
    a Test Plan in their body before dispatch.

    This is NOT a quality gate; it only checks the required sections are
    present. Quality judgement is the job of the council and audit stages.

    Returns ``None`` if the contract is satisfied, or a human-readable
    reason string when the required sections are missing.
    """
    row = conn.execute(
        "SELECT tier, body FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None  # task doesn't exist — let claim_task handle it

    tier = (row["tier"] or "").lower()
    if tier != "full":
        return None  # fast-tier or unclassified — no contract required

    body = (row["body"] or "").lower()

    # Check for Acceptance Criteria section
    has_ac = any(
        marker in body
        for marker in ("## acceptance criteria", "# acceptance criteria",
                       "acceptance criteria:", "**acceptance criteria**")
    )

    # Check for Test Plan section
    has_test_plan = any(
        marker in body
        for marker in ("## test plan", "# test plan",
                       "test plan:", "**test plan**")
    )

    missing = []
    if not has_ac:
        missing.append("Acceptance Criteria")
    if not has_test_plan:
        missing.append("Test Plan")

    if missing:
        return (
            f"Full-tier task missing required contract sections: "
            f"{', '.join(missing)}. "
            f"Full-tier tasks must carry ## Acceptance Criteria and ## Test Plan "
            f"in the body before they can be dispatched."
        )
    return None


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
        # KENSEI CUSTOM (fork re-anchor): WS-1 contract gate — full-tier tasks must
        # carry AC + Test Plan sections in the body before they can be dispatched.
        contract_violation = validate_task_contract(conn, task_id)
        if contract_violation:
            conn.execute(
                "UPDATE tasks SET status = 'blocked', status_reason = ? "
                "WHERE id = ? AND status = 'ready'",
                (contract_violation, task_id),
            )
            _append_event(
                conn, task_id, "claim_rejected",
                {"reason": "contract_violation", "detail": contract_violation},
            )
            _append_event(conn, task_id, "blocked", {"reason": contract_violation})
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


def _set_task_status_direct(
    conn: sqlite3.Connection,
    task_id: str,
    new_status: str,
    *,
    actor: str,
) -> bool:
    """Apply a guarded direct lane move and preserve run/graph invariants."""
    terminations: list[tuple[Optional[int], Optional[str]]] = []
    effective_status = new_status
    with write_txn(conn):
        previous = conn.execute(
            "SELECT status, current_run_id, worker_pid, claim_lock "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if previous is None:
            return False

        if previous["status"] == "running" and new_status == "ready":
            resume_status = _retry_status_for_run(
                conn,
                task_id,
                previous["current_run_id"],
            )
            if resume_status == "review":
                effective_status = (
                    "review" if _parents_satisfied(conn, task_id) else "todo"
                )

        if effective_status == "ready" and not _parents_satisfied(conn, task_id):
            return False

        was_running = previous["status"] == "running"
        reopening_satisfied_parent = (
            previous["status"] in {"done", "archived"}
            and effective_status not in {"done", "archived"}
        )
        changed = conn.execute(
            "UPDATE tasks SET status = ?, "
            "claim_lock = CASE WHEN ? = 'running' THEN claim_lock ELSE NULL END, "
            "claim_expires = CASE WHEN ? = 'running' THEN claim_expires ELSE NULL END, "
            "worker_pid = CASE WHEN ? = 'running' THEN worker_pid ELSE NULL END "
            "WHERE id = ?",
            (
                effective_status,
                effective_status,
                effective_status,
                effective_status,
                task_id,
            ),
        )
        if changed.rowcount != 1:
            return False

        run_id = None
        if (
            was_running
            and effective_status != "running"
            and previous["current_run_id"]
        ):
            run_id = _end_run(
                conn,
                task_id,
                outcome="reclaimed",
                status="reclaimed",
                summary=f"status changed to {effective_status} ({actor})",
            )
            terminations.append(
                (previous["worker_pid"], previous["claim_lock"])
            )
        _append_event(
            conn,
            task_id,
            "status",
            {
                "status": effective_status,
                "requested_status": new_status,
                "actor": actor,
            },
            run_id=run_id,
        )
        if reopening_satisfied_parent:
            invalidated = invalidate_descendants_for_parent_reopen(
                conn,
                task_id,
                author=actor,
            )
            terminations.extend(invalidated["terminations"])

    for worker_pid, claim_lock in terminations:
        _terminate_reclaimed_worker(worker_pid, claim_lock)
    if effective_status in {"done", "ready", "review"}:
        recompute_ready(conn)
    return True


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


def reassign_task_with_note(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    handoff_note: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Reassign and append a handoff comment in one logical operation.

    ``reassign_task`` runs reclaim + assign as its own write_txns, then
    this helper appends the comment in a tightly-coupled follow-up txn
    on the same connection. The two-step sequence is on a local SQLite
    file with WAL, so the inter-txn window is microseconds. Callers
    that need true single-txn atomicity should inline the SQL directly.
    """
    ok = reassign_task(conn, task_id, profile, reclaim_first=True,
                       reason="kanban_reassign")
    if not ok:
        return False
    if handoff_note and handoff_note.strip():
        with write_txn(conn):
            _add_comment_inline(
                conn, task_id,
                author=author or "orchestrator",
                body=f"reassign handoff: {handoff_note.strip()}",
            )
    return True


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
    verified_cards = _gate_created_cards(conn, task_id, created_cards, summary or result)
    metadata = _merge_completion_prose_artifacts(
        conn, task_id, metadata, summary=summary, result=result,
    )
    # KENSEI CUSTOM (fork re-anchor): WS-4 review gate + P2-1 tiered review —
    # classified tasks auto-route running→review on completion; untyped
    # (NULL-tier) automation tasks go straight to done.
    tier_row = conn.execute(
        "SELECT tier FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    task_tier = (tier_row["tier"] or "").lower() if tier_row else ""
    if _should_review(conn, task_tier, task_id):
        # WS-4 loop guard: only skip redirect when the latest review is
        # UNRESOLVED (no subsequent approve/reject). Once a review is
        # resolved (passed or rejected), the next completion must re-enter
        # review so reject→fix→re-review works.
        passes = review_pass_count(conn, task_id)
        latest_review_id = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'review_requested'",
            (task_id,),
        ).fetchone()[0] or 0
        latest_resolution_id = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ? "
            "AND kind IN ('review_passed', 'review_rejected')",
            (task_id,),
        ).fetchone()[0] or 0
        review_is_open = latest_review_id > latest_resolution_id
        if passes > 0 or review_is_open:
            _log.debug(
                "complete_task: %s — passes=%d, review_is_open=%s — skipping redirect",
                task_id, passes, review_is_open,
            )
        else:
            return request_review(
                conn, task_id,
                summary=(summary or result or "complete"),
                artefacts=None,
                next_steps="Review per sdlc-review skill and multi-gate-qa.md criteria",
                expected_run_id=expected_run_id,
            )
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        # Hard invariant even for human review approval: a parent may have
        # reopened while this task waited.
        if not _parents_satisfied(conn, task_id):
            return False
        prior_status = _task_status(conn, task_id)
        sql = """
                UPDATE tasks
                   SET status       = 'done',
                       result       = ?,
                       completed_at = ?,
                       done_at      = ?,
                       claim_lock   = NULL,
                       claim_expires= NULL,
                       worker_pid   = NULL,
                       block_kind   = NULL,
                       block_recurrences = 0
                 WHERE id = ?
                   AND status IN ('running', 'ready', 'blocked', 'review')
                """
        params: tuple = (result, now, now, task_id)
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
        completed_payload = _completed_event_payload(result, event_summary, verified_cards, metadata)
        # KENSEI CUSTOM (fork re-anchor): WS-7 Denji loop — flag the completion
        # event for governance review scanning.
        completed_payload["denji_review_signal"] = True
        _append_event(
            conn, task_id, "completed", completed_payload, run_id=run_id,
        )
    _flag_phantom_prose_refs(conn, task_id, run_id, summary, result, verified_cards)
    # Success wipes the breaker counter (history stays on the event log).
    _clear_failure_counter(conn, task_id)
    recompute_ready(conn)  # separate txn so children see ``done``
    # KENSEI CUSTOM (fork re-anchor): auto-revoke task-scoped skill/tool grants
    # (Phase 3). Best-effort — never fail completion.
    try:
        from tools.skill_grants import revoke_grants_for_task
        revoke_grants_for_task(task_id, "completed")
    except Exception:  # noqa: BLE001
        _log.debug("skill grant auto-revoke failed for %s", task_id, exc_info=True)
    try:
        from tools.tool_grants import revoke_grants_for_task as _revoke_tool_grants
        _revoke_tool_grants(task_id, "completed")
    except Exception:  # noqa: BLE001
        _log.debug("tool grant auto-revoke failed for %s", task_id, exc_info=True)
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


def _stage_completion_artifacts(
    conn: sqlite3.Connection, task_id: str, metadata: dict, now: int, *,
    uploaded_by: str = "kanban_complete",
) -> list[Path]:
    """Copy scratch artifacts to the attachments dir and record each as an
    attachment row; returns the copies so the caller can discard them if its
    transaction rolls back."""
    _persist_scratch_completion_artifacts(conn, task_id, metadata)
    staged = [Path(stored_path) for stored_path in metadata.pop("_staged_artifacts", [])]
    for path in staged:
        _insert_completion_attachment(
            conn, task_id, filename=path.name, stored_path=str(path),
            size=path.stat().st_size, created_at=now, uploaded_by=uploaded_by,
        )
    return staged


def _cleaned_artifact_paths(metadata: Any) -> list[str]:
    """Non-blank string paths declared in ``metadata["artifacts"]``."""
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("artifacts")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(p).strip() for p in raw if isinstance(p, str) and str(p).strip()]


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
        cleaned = _cleaned_artifact_paths(metadata)
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
        _discard_staged_copies(used_destinations, attachment_dir)

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


def _discard_staged_copies(copies: Iterable[Path], attachment_dir: Path) -> None:
    """Remove staged attachment copies whose DB rows never committed; a leaked
    copy would make the retry stage ``name_1.ext`` next to an orphan."""
    for copied in copies:
        with contextlib.suppress(OSError):
            Path(copied).unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        attachment_dir.rmdir()


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
    created_at: int, uploaded_by: str = "kanban_complete",
) -> None:
    """Record a worker-produced artifact in the existing attachment table."""
    conn.execute(
        "INSERT INTO task_attachments "
        "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
        "VALUES (?, ?, ?, NULL, ?, ?, ?)",
        (task_id, filename, stored_path, size, uploaded_by, created_at),
    )
    _append_event(conn, task_id, "attached", {"filename": filename, "size": size, "by": uploaded_by})


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
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    kind: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Transition ``running``/``ready`` → ``blocked`` (or route elsewhere).

    ``kind`` (one of :data:`VALID_BLOCK_KINDS`, or ``None`` for a legacy
    un-typed block) drives routing instead of every block landing in one
    undifferentiated ``blocked`` bucket:

    * ``dependency`` — the task is only waiting on another task. It does NOT
      sit in ``blocked`` (where a cron would keep "unblocking" it); it goes to
      ``todo`` so the existing parent-gating / ``recompute_ready`` machinery
      promotes it automatically once its parents finish. No human, no cron, no
      retry storm. This is Dale's "Type 2 — dependency blocked".

    * ``needs_input`` / ``capability`` / ``None`` — "truly blocked" (Dale's
      "Type 1"). Lands in ``blocked`` for a human. BUT: each time such a task
      is re-blocked for the SAME kind after having been unblocked, the
      unblock-loop counter (``block_recurrences``) increments. When it reaches
      :data:`BLOCK_RECURRENCE_LIMIT`, the task is routed to ``triage`` instead
      of ``blocked`` — breaking the cron-unblock ↔ worker-re-block loop and
      forcing a human-in-the-loop triage decision.

    * ``transient`` — treated like a generic block for routing, but a worker
      can use it to signal "this might clear on its own"; it still participates
      in the loop breaker so a forever-flaky task eventually escalates.

    * ``stakeholder-decision`` / ``escalation`` — decision kinds
      (subset :data:`DECISION_BLOCK_KINDS`). When the task already has an
      ``escalation_target`` set, the block routes the task to the
      ``decision-needed`` terminal status instead of ``blocked`` — surfacing
      it as human-owned so the auto-unblocker can't churn on it. Without an
      ``escalation_target`` the block falls through to the regular
      ``blocked`` routing (the decision-kind alone is not enough; a human
      must have explicitly escalated the task for it to enter the terminal
      decision gate).

    Returns True on any successful transition (to ``blocked``, ``todo``,
    ``triage``, or ``decision-needed``), False when the task wasn't in a
    blockable state.
    """
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(
            f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None"
        )
    # S7 (Sahil 30/08/26): an un-typed block must never land as a NULL
    # block_kind — that is the attribution gap that left 11 Aug tasks
    # unascribable. Callers that don't know the kind get 'unspecified'.
    if kind is None:
        kind = "unspecified"
    recurrences = 0
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences, escalation_target "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        source_status = (
            _retry_status_for_run(conn, task_id)
            if cur_row["status"] == "running"
            else "ready"
        )
        prev_kind = cur_row["block_kind"] if "block_kind" in cur_row.keys() else None
        prev_recurrences = (
            int(cur_row["block_recurrences"])
            if "block_recurrences" in cur_row.keys()
            and cur_row["block_recurrences"] is not None
            else 0
        )
        # Pre-compute the decision-gate routing. A decision-kind block with an
        # ``escalation_target`` is the explicit human signal that this task
        # needs a decision, not another auto-recovery cycle. We evaluate this
        # once up-front so both the loop-detected (``triage``) and the normal
        # (``blocked``) branches can fall through to it. The recurrence
        # counter still increments — a forever-flaky decision-kind task will
        # still hit BLOCK_RECURRENCE_LIMIT and be routed to ``triage`` so a
        # human can see "this decision has been bouncing" rather than the
        # task silently sitting in ``decision-needed`` while the underlying
        # problem keeps re-asserting itself.
        is_decision_block = kind in DECISION_BLOCK_KINDS
        is_escalated = (
            "escalation_target" in cur_row.keys()
            and cur_row["escalation_target"] is not None
            and str(cur_row["escalation_target"]).strip() != ""
        )

        # Dependency blocks never enter the human ``blocked`` bucket — they
        # wait in ``todo`` and let ``recompute_ready`` gate on parents. Routing
        # here (rather than ``blocked``) is what keeps a cron from ever seeing
        # a dependency-wait as something to "unblock".
        if kind == "dependency":
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'todo',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       block_kind    = ?
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
                (kind, task_id) if expected_run_id is None
                else (kind, task_id, int(expected_run_id)),
            )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=reason,
                )
            _append_event(
                conn, task_id, "dependency_wait",
                {
                    "reason": reason,
                    "kind": kind,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
            _blocked_task = get_task(conn, task_id)
            _fire_kanban_lifecycle_hook(
                "kanban_task_blocked",
                task_id,
                board=get_current_board(),
                assignee=_blocked_task.assignee if _blocked_task else None,
                run_id=run_id,
                reason=reason,
            )
            return True

        # Truly-blocked kinds. Increment the unblock-loop counter when this is a
        # re-block for the SAME reason after a prior unblock. block_task only
        # fires from running/ready (i.e. AFTER an unblock returned the task to
        # the work pool), so a stored block_kind that matches the incoming kind
        # means: blocked → unblocked → about-to-re-block for the same cause.
        # An un-typed (None) block compares as "same" to a prior un-typed block.
        same_cause = prev_kind == kind
        recurrences = prev_recurrences + 1 if same_cause else 1

        if recurrences >= BLOCK_RECURRENCE_LIMIT:
            # Loop detected — stop letting the unblocker spin this task. Route
            # to triage for a human-in-the-loop decision instead of blocked —
            # UNLESS this is a decision-kind block on an already-escalated
            # task, in which case the terminal decision gate
            # (``decision-needed``) wins so the human sees the decision
            # surfaced for action rather than re-buried in a triage loop.
            dest_status = (
                "decision-needed" if is_decision_block and is_escalated
                else "triage"
            )
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = ?,
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL,
                       block_kind    = ?,
                       block_recurrences = ?
                 WHERE id = ?
                   AND status IN ('running', 'ready')
                """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
                (dest_status, kind, recurrences, task_id)
                if expected_run_id is None
                else (dest_status, kind, recurrences, task_id, int(expected_run_id)),
            )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id, outcome="blocked", summary=reason,
                )
            _append_event(
                conn, task_id, "block_loop_detected",
                {
                    "reason": reason,
                    "kind": kind,
                    "recurrences": recurrences,
                    "limit": BLOCK_RECURRENCE_LIMIT,
                    "decision_gate": is_decision_block and is_escalated,
                    "routed_to": dest_status,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
            routed_to = dest_status
        else:
            # Decision-gate routing: a decision-kind block with an
            # ``escalation_target`` set on the task lands the task in the
            # ``decision-needed`` terminal status (human-owned, unblocker
            # refuses to touch it). Without ``escalation_target`` the regular
            # ``blocked`` bucket is used so the human escalation flow remains
            # the explicit signal.
            dest_status = (
                "decision-needed" if is_decision_block and is_escalated
                else "blocked"
            )
            if expected_run_id is None:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = ?,
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """,
                    (dest_status, kind, recurrences, task_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = ?,
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                       AND current_run_id = ?
                    """,
                    (dest_status, kind, recurrences, task_id, int(expected_run_id)),
                )
            if cur.rowcount != 1:
                return False
            run_id = _end_run(
                conn, task_id,
                outcome="blocked", status="blocked",
                summary=reason,
            )
            # Synthesize a run when blocking a never-claimed task so the
            # reason is preserved in attempt history.
            if run_id is None and reason:
                run_id = _synthesize_ended_run(
                    conn, task_id,
                    outcome="blocked",
                    summary=reason,
                )
            _append_event(
                conn, task_id, "blocked",
                {
                    "reason": reason,
                    "kind": kind,
                    "recurrences": recurrences,
                    "decision_gate": is_decision_block and is_escalated,
                    "routed_to": dest_status,
                    "source_status": source_status,
                },
                run_id=run_id,
            )
            routed_to = dest_status
        # Loop-diagnostics: attach the failure report for this terminal
        # attempt failure (a worker block = a terminal non-success attempt).
        # Dependency-wait blocks return early above, so this covers the
        # normal / loop-detected / decision-gate routes. Only when a real
        # run was closed — synthesized runs (never claimed) have no trace.
        if run_id is not None and reason:
            _attach_loop_diagnosis(
                conn, task_id,
                run_id=run_id,
                outcome="blocked",
                error=reason,
            )
        _blocked_task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=get_current_board(),
        assignee=_blocked_task.assignee if _blocked_task else None,
        run_id=run_id,
        reason=reason,
    )
    # ── KENSEI governance: record block event ──
    task = get_task(conn, task_id)
    record_event_if_enabled(
        source="kanban.db",
        actor_profile=os.environ.get("HERMES_PROFILE"),
        target_profile=task.assignee if task else None,
        event_type="kanban.task.blocked",
        object_type="kanban_task",
        object_id=task_id,
        # Follow the actual destination so dashboards and event consumers see
        # the right landing state (``decision-needed`` for decision gates,
        # ``triage`` for loop-detected, ``blocked`` for normal).
        status_to=routed_to,
        summary=reason or f"Blocked kanban task {task_id}",
        payload={"reason": reason, "run_id": run_id, "routed_to": routed_to},
    )
    return True


_HUMAN_APPROVAL_PREFIX = "needs_human_approval"

def request_human_approval(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    detail: str,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Block a task awaiting explicit human approval (WS-8).

    Used for full-tier go/no-go decisions. The task is blocked with
    reason ``needs_human_approval: <detail>`` and must be explicitly
    unblocked by a human before the dispatcher will touch it.
    """
    reason = f"{_HUMAN_APPROVAL_PREFIX}: {detail}" if detail else _HUMAN_APPROVAL_PREFIX
    return block_task(
        conn, task_id,
        reason=reason,
        expected_run_id=expected_run_id,
    )


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


def _block_task_locked(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> "tuple[bool, Optional[int]]":
    """Transactional body of ``block_task``. MUST run inside a ``write_txn``.

    Returns ``(blocked, run_id)``. Factored out so callers that must block a
    task atomically with other writes (e.g. recording a profile-lifecycle
    approval) can do both in a single transaction rather than two.
    """
    if expected_run_id is None:
        cur = conn.execute(
            """
            UPDATE tasks
               SET status       = 'blocked',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('running', 'ready', 'triage')
            """,
            (task_id,),
        )
    else:
        cur = conn.execute(
            """
            UPDATE tasks
               SET status       = 'blocked',
                   claim_lock   = NULL,
                   claim_expires= NULL,
                   worker_pid   = NULL
             WHERE id = ?
               AND status IN ('running', 'ready', 'triage')
               AND current_run_id = ?
            """,
            (task_id, int(expected_run_id)),
        )
    if cur.rowcount != 1:
        return False, None
    run_id = _end_run(
        conn, task_id,
        outcome="blocked", status="blocked",
        summary=reason,
    )
    # Synthesize a run when blocking a never-claimed task so the
    # reason is preserved in attempt history.
    if run_id is None and reason:
        run_id = _synthesize_ended_run(
            conn, task_id,
            outcome="blocked",
            summary=reason,
        )
    _append_event(conn, task_id, "blocked", {"reason": reason}, run_id=run_id)
    # Loop-diagnostics: attach the failure report for this terminal
    # attempt failure (a worker blocked = a terminal non-success attempt).
    # Only when a real run was closed — synthesized runs (never claimed)
    # have no trace to diagnose.
    if run_id is not None and reason:
        _attach_loop_diagnosis(
            conn, task_id,
            run_id=run_id,
            outcome="blocked",
            error=reason,
        )
    return True, run_id

_VALID_LIFECYCLE_OPS = ("create", "delete")

def request_profile_lifecycle_approval(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    op: str,
    profile: str,
    args: Optional[dict] = None,
    requested_by: Optional[str] = None,
    blast_summary: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> str:
    """Record a pending profile-lifecycle approval and block the task.

    Returns the approval id. The task is blocked with a
    ``needs_human_approval`` reason; the Discord bridge delivers the
    prompt and the resolver executes the op once Sahil approves.
    """
    if op not in _VALID_LIFECYCLE_OPS:
        raise ValueError(f"op must be one of {_VALID_LIFECYCLE_OPS}, got {op!r}")
    approval_id = "pla-" + secrets.token_hex(4)
    token = secrets.token_urlsafe(24)
    now = int(time.time())
    detail = f"profile {op} {profile} (approval {approval_id})"
    reason = f"{_HUMAN_APPROVAL_PREFIX}: {detail}"
    # Atomic: record the approval AND block the task in one transaction, so
    # there is never an approval row without a blocked task (or vice versa).
    with write_txn(conn):
        conn.execute(
            """
            INSERT INTO profile_lifecycle_approvals
                (id, task_id, op, profile, args_json, requested_by,
                 blast_summary, status, token, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
            """,
            (
                approval_id, task_id, op, profile,
                json.dumps(args or {}), requested_by, blast_summary,
                token, now,
            ),
        )
        _append_event(
            conn, task_id, "profile_lifecycle_approval_requested",
            {"approval_id": approval_id, "op": op, "profile": profile},
        )
        blocked, run_id = _block_task_locked(
            conn, task_id, reason=reason, expected_run_id=expected_run_id,
        )
        if not blocked:
            # Roll the whole transaction back (no orphan approval row) by
            # raising; the task was not in a blockable state.
            raise RuntimeError(
                f"cannot request approval {approval_id}: task {task_id} is not "
                f"in a blockable (running/ready/triage) state"
            )
    # Emit the same external blocked-telemetry block_task does, so dashboards
    # and event consumers see profile-gate blocks (the inlined _block_task_locked
    # only writes the task_events row, not the external event bus).
    task = get_task(conn, task_id)
    record_event_if_enabled(
        source="kanban.db",
        actor_profile=os.environ.get("HERMES_PROFILE"),
        target_profile=task.assignee if task else None,
        event_type="kanban.task.blocked",
        object_type="kanban_task",
        object_id=task_id,
        status_to="blocked",
        summary=reason,
        payload={"reason": reason, "run_id": run_id},
    )
    return approval_id


def get_profile_lifecycle_approval(
    conn: sqlite3.Connection, approval_id: str
) -> Optional[dict]:
    """Return the approval row as a dict, or None if unknown."""
    row = conn.execute(
        "SELECT * FROM profile_lifecycle_approvals WHERE id = ?", (approval_id,)
    ).fetchone()
    return dict(row) if row else None


def list_pending_profile_lifecycle_approvals(
    conn: sqlite3.Connection, *, undelivered_only: bool = False
) -> list[dict]:
    """List pending approvals, oldest first.

    ``undelivered_only`` returns just the rows the Discord bridge has not
    yet posted (``notified_at IS NULL``).
    """
    sql = "SELECT * FROM profile_lifecycle_approvals WHERE status = 'pending'"
    if undelivered_only:
        sql += " AND notified_at IS NULL"
    sql += " ORDER BY created_at ASC"
    return [dict(r) for r in conn.execute(sql).fetchall()]


def mark_profile_lifecycle_notified(
    conn: sqlite3.Connection, approval_id: str
) -> None:
    """Stamp when the Discord prompt was delivered (idempotency guard)."""
    with write_txn(conn):
        conn.execute(
            "UPDATE profile_lifecycle_approvals SET notified_at = ? "
            "WHERE id = ? AND notified_at IS NULL",
            (int(time.time()), approval_id),
        )


def resolve_profile_lifecycle_approval(
    conn: sqlite3.Connection,
    approval_id: str,
    *,
    decision: str,
    resolved_by: str,
) -> dict:
    """Mark an approval approved or rejected. Fail-closed on bad state.

    Returns the updated row dict (including ``token`` so the caller can
    execute the op on approve). This only flips the approval status; closing
    the requester task is the caller's job (profile_lifecycle_gate._close_task).
    Raises ``ValueError`` if the approval is unknown or not pending (so a
    double-click cannot execute the op twice).
    """
    if decision not in ("approve", "reject"):
        raise ValueError(f"decision must be approve|reject, got {decision!r}")
    row = get_profile_lifecycle_approval(conn, approval_id)
    if row is None:
        raise ValueError(f"unknown approval {approval_id}")
    if row["status"] != "pending":
        raise ValueError(
            f"approval {approval_id} already {row['status']}; refusing to re-resolve"
        )
    new_status = "approved" if decision == "approve" else "rejected"
    now = int(time.time())
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE profile_lifecycle_approvals "
            "SET status = ?, resolved_by = ?, resolved_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (new_status, resolved_by, now, approval_id),
        )
        if cur.rowcount != 1:
            # Lost a race with a concurrent resolve.
            raise ValueError(
                f"approval {approval_id} was resolved concurrently; ignoring"
            )
    _append_event(
        conn, row["task_id"], "profile_lifecycle_approval_resolved",
        {"approval_id": approval_id, "decision": new_status, "by": resolved_by},
    )
    return get_profile_lifecycle_approval(conn, approval_id)


def mark_profile_lifecycle_executed(
    conn: sqlite3.Connection,
    approval_id: str,
    *,
    ok: bool,
    error: Optional[str] = None,
) -> None:
    """Record the outcome of running an approved op (executed | failed)."""
    with write_txn(conn):
        conn.execute(
            "UPDATE profile_lifecycle_approvals SET status = ?, error = ? "
            "WHERE id = ? AND status = 'approved'",
            ("executed" if ok else "failed", error, approval_id),
        )


class ApproverProfileError(ValueError):
    """Raised when the approver's profile matches the most recent worker
    (first-pass) or the previous reviewer (chained). Signals that the
    caller must not self-approve."""


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
    artefacts: Optional[Iterable[str]] = None, next_steps: Optional[str] = None,
):
    """``running``/``ready`` -> ``review``; never touches block recurrence accounting.

    Implementer and reviewer are recorded on the event so requested changes
    route back to the right profile; ``reviewer`` reassigns the task, and on
    re-review defaults to the latest ``changes_requested`` provenance. A live
    claim is only cleared with proof of ownership (``expected_run_id``) or
    ``force=True``. Returns ``bool``, or ``(ok, reason)`` with ``with_reason``.

    ``metadata["artifacts"]`` names the handoff's deliverable
    files; a review handoff is the last implementer transition, and the
    *reviewer's* completion is what cleans the managed scratch workspace up, so
    the files are staged into the task's durable attachments dir here and the
    staged paths ride the ``review_requested`` payload for the notifier to
    upload. A declared artifact that cannot be preserved raises
    :class:`ArtifactPreservationError`, rolling the whole transition back: the
    task stays ``running`` and retryable, with no attachments and no event.
    """

    def _ret(ok: bool, reason: Optional[str] = None):
        return (ok, reason) if with_reason else ok

    summary = redact_review_value(summary)
    metadata = dict(redact_review_value(metadata) or {})
    # KENSEI CUSTOM (fork re-anchor): structured review handoff fields from the
    # worker tool surface (artefacts + next_steps) fold into run metadata.
    if artefacts is not None:
        artefacts_list = [str(a).strip() for a in artefacts if str(a).strip()]
        if artefacts_list:
            metadata["artefacts"] = artefacts_list
    if isinstance(next_steps, str) and next_steps.strip():
        metadata["next_steps"] = next_steps.strip()
    # Declared (metadata["artifacts"]) and prose-referenced files
    # must be durable BEFORE anything can clean the scratch workspace up: for a
    # review-bound card the reviewer's completion is the cleanup trigger.
    metadata = _merge_completion_prose_artifacts(conn, task_id, metadata, summary=summary, result=None)
    now = int(time.time())
    # Staged copies live outside the txn: a rollback after staging must not
    # leave orphans that make the retry stage ``name_1.ext`` beside them.
    staged_copies: list[Path] = []
    try:
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
            if isinstance(metadata, dict):
                staged_copies = _stage_completion_artifacts(
                    conn, task_id, metadata, now, uploaded_by="kanban_request_review",
                )
            run_id = _end_or_synthesize_run(
                conn, task_id, outcome="review_requested", status="review",
                summary=summary, metadata=metadata, synthesize=bool(summary or metadata),
            )
            payload: dict = {
                "summary": _first_line(summary, 400) or None,
                "implementer": implementer,
                "reviewer": reviewer,
            }
            staged = _cleaned_artifact_paths(metadata)
            if staged:
                payload["artifacts"] = staged
            _append_event(conn, task_id, "review_requested", payload, run_id=run_id)
    except Exception:
        if staged_copies:
            _discard_staged_copies(staged_copies, staged_copies[0].parent)
        raise
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
        # KENSEI CUSTOM (fork re-anchor): decision-gate refusal — a task that is
        # human-owned (status 'decision-needed', or a decision-kind block with an
        # escalation_target) must not be auto-unblocked.
        gated = conn.execute(
            "SELECT status, block_kind, escalation_target FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if gated is not None:
            status = gated["status"]
            block_kind = gated["block_kind"]
            escalation = gated["escalation_target"]
            is_decision_needed = status == "decision-needed"
            is_decision_gated = (
                status == "blocked"
                and block_kind in DECISION_BLOCK_KINDS
                and bool(escalation is not None and str(escalation).strip() != "")
            )
            if is_decision_needed or is_decision_gated:
                _log.warning(
                    "unblock_task refused: task %s is decision-gated "
                    "(status=%r, block_kind=%r, escalation_target=%r); human decision required",
                    task_id, status, block_kind, escalation,
                )
                _append_event(
                    conn, task_id, "unblock_refused",
                    {
                        "reason": "decision-gated",
                        "status": status,
                        "block_kind": block_kind,
                        "escalation_target": escalation,
                    },
                )
                return False
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
            "consecutive_failures = 0, last_failure_error = NULL, "
            "escalation_target = NULL "
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


def _count_events(conn: sqlite3.Connection, task_id: str, kind: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM task_events WHERE task_id = ? AND kind = ?",
        (task_id, kind),
    ).fetchone()
    return int(row["n"]) if row else 0


def review_pass_count(conn: sqlite3.Connection, task_id: str) -> int:
    """How many chained `review_passed` events this task has accumulated.

    Used by `approve_review_task` to enforce `kanban.max_review_passes`.
    """
    return _count_events(conn, task_id, "review_passed")


def reassign_count(conn: sqlite3.Connection, task_id: str) -> int:
    """How many `assigned` events this task has accumulated.

    Used by `kanban_reassign` to enforce `kanban.max_reassigns`. Mirrors
    the `assigned` event emitted by `assign_task`.
    """
    return _count_events(conn, task_id, "assigned")


def latest_worker_profile(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Profile of the most recent worker run that requested review.

    Conditional approval pins the task back to this profile so the same
    worker resumes the task with full context on the requested fix.
    """
    row = conn.execute(
        """
        SELECT profile FROM task_runs
         WHERE task_id = ? AND outcome = 'review_requested'
         ORDER BY id DESC LIMIT 1
        """,
        (task_id,),
    ).fetchone()
    return row["profile"] if row and row["profile"] else None


def _latest_event_payload(
    conn: sqlite3.Connection, task_id: str, kind: str,
) -> Optional[dict]:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()
    if not row or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_reviewer_profile(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Profile from the most recent `review_passed` event payload.

    Used by the chained approver-profile guard so the same reviewer cannot
    rubber-stamp their own pass-along.
    """
    payload = _latest_event_payload(conn, task_id, "review_passed")
    if not payload:
        return None
    value = payload.get("approver_profile") or payload.get("by_profile")
    return str(value) if value else None


def preferred_reviewer_profile(
    conn: sqlite3.Connection, task_id: str,
) -> Optional[str]:
    """Profile recorded as the rejecting reviewer on the latest `review_rejected`.

    Sticky-reviewer rule: when a task re-enters `review` after fixes,
    spawn the same reviewer who logged the original findings.
    """
    payload = _latest_event_payload(conn, task_id, "review_rejected")
    if not payload:
        return None
    value = payload.get("rejected_by_profile")
    return str(value) if value else None


def _should_review(
    conn: sqlite3.Connection,
    task_tier: str,
    task_id: str,
    *,
    kanban_cfg: Optional[dict] = None,
) -> bool:
    """Decide whether a completed task should enter review based on its tier.

    When ``kanban.tiered_review`` is enabled:
      * ``full`` tier → always review (mandatory).
      * ``fast`` tier → sampled review (1 in N + all failures).
      * Unclassified / NULL tier → no review (system automation).

    When ``tiered_review`` is disabled (or unset), the legacy WS-4
    "review everything" behaviour applies: all ``full`` and ``fast``
    tier tasks go to review.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import get_kanban_config
            kanban_cfg = get_kanban_config()
        except Exception:
            kanban_cfg = {}

    tiered_enabled = bool(kanban_cfg.get("tiered_review", False))
    tier = (task_tier or "").lower().strip()

    if tier not in ("full", "fast"):
        return False

    if tier == "full":
        return True  # mandatory review

    # Fast tier — sampled review.
    # Always review tasks whose last run failed (non-completed).
    last_outcome_row = conn.execute(
        "SELECT outcome FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    last_failed = (
        last_outcome_row is not None
        and last_outcome_row["outcome"] != "completed"
    )
    if last_failed:
        return True

    if not tiered_enabled:
        # Legacy WS-4: review everything (fast tier just gets sampled by default)
        return True

    # Sample 1 in N fast-tier tasks.
    sample_rate = max(1, int(kanban_cfg.get("review_sample_rate", 5) or 5))
    # Deterministic sampling by task_id so the same task always gets the same
    # decision across dispatcher ticks AND across gateway restarts. Built-in
    # hash() is per-process salted (PYTHONHASHSEED), so it would flip the
    # decision after a restart; sha256 is stable.
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    sample_bucket = int(digest, 16) % sample_rate
    return sample_bucket == 0


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


class ReviewCircuitBreakError(ValueError):
    """Raised when the chain or reassign cap is hit."""


def _kanban_setting(name: str, default: int) -> int:
    try:
        from hermes_cli.config import load_config
        cfg = load_config().get("kanban", {})
    except Exception:
        return default
    value = cfg.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def approve_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    approver_profile: str,
    summary: Optional[str] = None,
    follow_up_spec: Optional[dict] = None,
    next_reviewer: Optional[str] = None,
    comment: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> dict:
    """Three-outcome approval handler called by `kanban_approve`.

    - ``outcome="terminal"`` flips to ``done`` via ``complete_task`` so
      ``created_cards`` verification and run-summary persistence behave
      consistently. Optional ``follow_up_spec`` creates a child task in
      ``backlog`` linked under the approved task.
    - ``outcome="chained"`` keeps the task in ``review`` and reassigns to
      ``next_reviewer``. Records a ``review_passed`` event carrying the
      approver. Caller must not pass ``follow_up_spec``.
    - ``outcome="conditional"`` returns the task to ``todo`` with assignee
      pinned to the most recent worker (the implementer who requested
      review). The optional ``comment`` is appended for the worker to
      read on next claim.

    Returns a dict describing the resulting state. Raises
    ``ApproverProfileError`` if the approver is the original implementer
    (first pass) or the previous reviewer (chained), and
    ``ReviewCircuitBreakError`` when ``kanban.max_review_passes`` is hit
    on a chained call.
    """
    if not approver_profile:
        raise ValueError("approver_profile is required")
    if outcome not in {"terminal", "chained", "conditional"}:
        raise ValueError(
            "outcome must be one of terminal | chained | conditional"
        )
    if outcome != "terminal" and follow_up_spec is not None:
        raise ValueError(
            "follow_up_spec is only valid for outcome='terminal'"
        )
    if outcome == "chained" and not next_reviewer:
        raise ValueError("next_reviewer is required for chained approval")

    # First-pass vs chained approver guard. Reads must be inside the txn
    # so a concurrent reject doesn't slip in between the check and the
    # status transition.
    prior_passes = review_pass_count(conn, task_id)
    if prior_passes == 0:
        original = latest_worker_profile(conn, task_id)
        if original and approver_profile == original:
            raise ApproverProfileError(
                f"approver_profile={approver_profile!r} matches the most "
                "recent worker; the original implementer cannot approve "
                "their own work. Reassign to a different reviewer first."
            )
    else:
        prev = latest_reviewer_profile(conn, task_id)
        if prev and approver_profile == prev:
            raise ApproverProfileError(
                f"approver_profile={approver_profile!r} matches the previous "
                "reviewer; chained approvals must hand off to a different "
                "profile."
            )

    if outcome == "chained":
        cap = _kanban_setting("max_review_passes", 4)
        if prior_passes >= cap:
            raise ReviewCircuitBreakError(
                f"chained approvals exhausted (max_review_passes={cap}); "
                "terminal-approve, reject, or reassign instead."
            )

    if outcome == "terminal":
        ok = complete_task(
            conn, task_id,
            summary=summary,
            metadata={"approver_profile": approver_profile},
            expected_run_id=expected_run_id,
        )
        if not ok:
            return {"ok": False, "reason": "complete_task refused"}
        # Lay down a `review_approved` event so review history is
        # explicit alongside the standard `completed` event.
        with write_txn(conn):
            _append_event(
                conn, task_id, "review_approved",
                {"approver_profile": approver_profile, "outcome": "terminal"},
            )
        result = {"ok": True, "status": "done", "outcome": "terminal"}
        if follow_up_spec:
            child_id = _create_followup_from_approval(
                conn, parent_id=task_id,
                spec=follow_up_spec, approver_profile=approver_profile,
            )
            result["follow_up_task_id"] = child_id
        return result

    if outcome == "chained":
        with write_txn(conn):
            # End the reviewer's run, reset assignee + status to review
            # so claim_review_task picks the next reviewer up on the next
            # dispatcher tick. Clear the claim under the same txn.
            run_id = _end_run(
                conn, task_id,
                outcome="approved_chained",
                status="approved_chained",
                summary=summary,
                metadata={
                    "approver_profile": approver_profile,
                    "next_reviewer": next_reviewer,
                },
            )
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'review',
                       assignee      = ?,
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL
                 WHERE id = ?
                """,
                (_canonical_assignee(next_reviewer), task_id),
            )
            if cur.rowcount != 1:
                return {"ok": False, "reason": "task not found"}
            _append_event(
                conn, task_id, "review_passed",
                {
                    "approver_profile": approver_profile,
                    "next_reviewer": _canonical_assignee(next_reviewer),
                    "summary": summary,
                },
                run_id=run_id,
            )
        return {
            "ok": True, "status": "review",
            "outcome": "chained",
            "next_reviewer": _canonical_assignee(next_reviewer),
        }

    # conditional
    original_worker = latest_worker_profile(conn, task_id)
    if not original_worker:
        raise ValueError(
            "no prior worker run found; cannot pin assignee for "
            "conditional approval"
        )
    with write_txn(conn):
        run_id = _end_run(
            conn, task_id,
            outcome="approved_conditional",
            status="approved_conditional",
            summary=summary,
            metadata={
                "approver_profile": approver_profile,
                "comment": comment,
            },
        )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'todo',
                   assignee      = ?,
                   claim_lock    = NULL,
                   claim_expires = NULL,
                   worker_pid    = NULL
             WHERE id = ?
            """,
            (original_worker, task_id),
        )
        if cur.rowcount != 1:
            return {"ok": False, "reason": "task not found"}
        if comment and comment.strip():
            _add_comment_inline(
                conn, task_id,
                author=approver_profile,
                body=f"conditional approval: {comment.strip()}",
            )
        _append_event(
            conn, task_id, "review_approved",
            {
                "approver_profile": approver_profile,
                "outcome": "conditional",
                "pinned_assignee": original_worker,
            },
            run_id=run_id,
        )
    # Recompute so the task either lands in ``ready`` (no parents) or
    # waits in ``todo`` until parents resolve. Matches the
    # ``unblock_task`` policy.
    recompute_ready(conn)
    return {
        "ok": True, "status": "todo",
        "outcome": "conditional",
        "pinned_assignee": original_worker,
    }


def _create_followup_from_approval(
    conn: sqlite3.Connection,
    *,
    parent_id: str,
    spec: dict,
    approver_profile: str,
) -> str:
    """Land a follow_up_spec from a terminal approval as a new backlog task.

    ``spec`` must include ``title`` and ``assignee``. ``body``, ``theme``,
    and ``sub_goals`` are optional. The new task is linked as a child of
    ``parent_id`` so the audit trail joins them.
    """
    title = (spec.get("title") or "").strip()
    assignee = (spec.get("assignee") or "").strip()
    if not title:
        raise ValueError("follow_up_spec.title is required")
    if not assignee:
        raise ValueError("follow_up_spec.assignee is required")
    body_parts: list[str] = []
    if spec.get("body"):
        body_parts.append(str(spec["body"]).strip())
    sub_goals = spec.get("sub_goals")
    if isinstance(sub_goals, (list, tuple)) and sub_goals:
        body_parts.append("Sub-goals:")
        body_parts.extend(f"- {str(g).strip()}" for g in sub_goals if str(g).strip())
    body = "\n\n".join(p for p in body_parts if p) or None
    child_id = create_task(
        conn,
        title=title,
        body=body,
        assignee=assignee,
        created_by=approver_profile,
        parents=(parent_id,),
        theme=spec.get("theme"),
        initial_status="backlog",
    )
    return child_id


def reject_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    findings: dict,
    rejected_by_profile: str,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Transition ``running -> blocked`` from a review claim.

    The task was in ``review`` and got claimed by the rejecting reviewer
    (so its live status is ``running`` with the reviewer's run active).
    This helper closes that run, flips the task to ``blocked``, records
    the rejecting reviewer's profile under ``rejected_by_profile`` on
    the ``review_rejected`` event so the sticky-reviewer logic can route
    the next pass back to the same reviewer.
    """
    if not isinstance(findings, dict):
        raise ValueError("findings must be an object")
    if not rejected_by_profile:
        raise ValueError("rejected_by_profile is required")
    with write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'blocked',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL
                 WHERE id = ? AND status = 'running'
                """,
                (task_id,),
            )
        else:
            cur = conn.execute(
                """
                UPDATE tasks
                   SET status        = 'blocked',
                       claim_lock    = NULL,
                       claim_expires = NULL,
                       worker_pid    = NULL
                 WHERE id = ? AND status = 'running'
                   AND current_run_id = ?
                """,
                (task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        summary_preview = None
        reasons = findings.get("reasons")
        if isinstance(reasons, (list, tuple)) and reasons:
            summary_preview = str(reasons[0])[:400]
        run_id = _end_run(
            conn, task_id,
            outcome="review_rejected",
            status="review_rejected",
            summary=summary_preview,
            metadata={"findings": findings, "rejected_by_profile": rejected_by_profile},
        )
        payload = {
            "findings": findings,
            "rejected_by_profile": rejected_by_profile,
        }
        _append_event(
            conn, task_id, "review_rejected", payload, run_id=run_id,
        )
        # Emit `blocked` second so _has_sticky_block sees a worker-style
        # block and recompute_ready will not auto-promote on the next tick.
        _append_event(
            conn, task_id, "blocked",
            {"reason": "review-rejected", "rejected_by_profile": rejected_by_profile},
            run_id=run_id,
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


def edit_task_skills(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    skills: Iterable[str],
    author: str,
) -> dict:
    """Replace a task's ``skills`` list. Scope-locked to ``skills`` only.

    Allowed on any non-terminal status. For a ``running`` task the row
    updates but the live worker keeps its already-loaded skill set; the
    return value carries ``applies_on_next_spawn=True`` so the caller
    knows whether to follow up with a reassign or wait for a natural
    respawn.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in skills or ():
        s = str(raw).strip()
        if not s or s in seen:
            continue
        if "," in s:
            raise ValueError(f"skill name {s!r} contains a comma; pass a list instead")
        seen.add(s)
        cleaned.append(s)
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"unknown task {task_id!r}")
        status = row["status"]
        if status in {"done", "archived"}:
            raise ValueError(
                f"cannot edit skills on terminal task (status={status!r})"
            )
        payload = json.dumps(cleaned) if cleaned else None
        conn.execute(
            "UPDATE tasks SET skills = ? WHERE id = ?", (payload, task_id),
        )
        if author:
            _add_comment_inline(
                conn, task_id,
                author=author,
                body=f"edit: skills <- {cleaned!r}",
            )
        _append_event(
            conn, task_id, "edited",
            {"fields": ["skills"], "skills": cleaned, "author": author},
        )
    return {
        "ok": True,
        "skills": cleaned,
        "applies_on_next_spawn": status == "running",
        "status": status,
    }


def set_escalation_target(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    target: Optional[str],
    author: str,
) -> dict:
    """Set or clear a task's ``escalation_target``. Scope-locked to that column.

    ``target`` is a recipient slug (e.g. ``'sahil'``) when a job decides the
    task genuinely needs a human; pass ``None``/empty to clear it once resolved.
    The daily briefing's "NEEDS YOU" section reads every still-open task whose
    ``escalation_target`` is set, so this is the single structured signal that
    drives human escalation. Allowed on any non-terminal status.

    Refuses to set the target on a task already in the ``decision-needed``
    terminal state — that row has already been promoted to a human gate by
    ``block_task`` and any further ``escalation_target`` write would mutate a
    human-owned task through automation. The escalation column is intended
    to be set *before* the block transitions the status; re-setting it after
    is meaningless and is treated as automation touching a human-owned row.
    The same refusal applies to *clearing* the target while the task is in
    the terminal decision gate — only a direct status flip (e.g. ``hermes
    kanban set-status ... done``) should move it out, not column-level
    writes from the escalate CLI.
    """
    cleaned = (str(target).strip() or None) if target is not None else None
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, escalation_target FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if not row:
            raise ValueError(f"unknown task {task_id!r}")
        status = row["status"]
        if status in {"done", "archived"}:
            raise ValueError(
                f"cannot set escalation_target on terminal task (status={status!r})"
            )
        if status == "decision-needed":
            raise ValueError(
                f"cannot set escalation_target on decision-gated task "
                f"(status='decision-needed'); human decision required"
            )
        if row["escalation_target"] == cleaned:
            return {"ok": True, "escalation_target": cleaned, "changed": False,
                    "status": status}
        conn.execute(
            "UPDATE tasks SET escalation_target = ? WHERE id = ?",
            (cleaned, task_id),
        )
        if author:
            _add_comment_inline(
                conn, task_id,
                author=author,
                body=f"edit: escalation_target <- {cleaned!r}",
            )
        _append_event(
            conn, task_id, "edited",
            {"fields": ["escalation_target"], "escalation_target": cleaned,
             "author": author},
        )
    return {"ok": True, "escalation_target": cleaned, "changed": True,
            "status": status}


def list_open_escalations(conn: sqlite3.Connection) -> list[dict]:
    """Return non-terminal tasks awaiting a human, for the briefing NEEDS-YOU list.

    A task is "open and escalated" when ``escalation_target`` is set and the
    status is not ``done``/``archived``. Ordered oldest-first so the briefing
    surfaces the longest-waiting decisions at the top.
    """
    rows = conn.execute(
        """
        SELECT id, title, status, assignee, escalation_target,
               status_reason, priority, created_at
        FROM tasks
        WHERE escalation_target IS NOT NULL
          AND status NOT IN ('done', 'archived')
        ORDER BY created_at ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def _pin_sticky_reviewers(conn: sqlite3.Connection) -> int:
    """Reassign unclaimed review-column tasks to their preferred reviewer.

    Called from ``dispatch_once`` right before the review-row sweep.
    Bounded scan: at most one UPDATE per review task with a recorded
    rejection. Tasks without a prior rejection are no-ops.

    R-2: only pin when the preferred reviewer is spawnable; otherwise the
    task would sit in review with a non-spawnable assignee and the
    dispatcher would skip it forever.
    """
    rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'review' AND claim_lock IS NULL"
    ).fetchall()
    pinned = 0
    for row in rows:
        preferred = preferred_reviewer_profile(conn, row["id"])
        if not preferred:
            continue
        if not _is_profile_spawnable(preferred):
            # Sticky reviewer is not spawnable; don't pin to avoid strand
            continue
        if row["assignee"] == preferred:
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET assignee = ? "
                "WHERE id = ? AND status = 'review' AND claim_lock IS NULL",
                (preferred, row["id"]),
            )
            if cur.rowcount == 1:
                _append_event(
                    conn, row["id"], "assigned",
                    {"profile": preferred, "via": "sticky_reviewer"},
                )
                pinned += 1
    return pinned


def promote_from_backlog(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``backlog -> triage`` so the task enters the specifier path.

    Returns ``True`` on a successful transition, ``False`` when the task
    is unknown or not in ``backlog``. Idempotent: second call returns ``False``.

    Emits a ``promoted_from_backlog`` event so the move is auditable.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id = ? AND status = 'backlog'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        _append_event(conn, task_id, "promoted_from_backlog", None)
        return True


def specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
) -> bool:
    """Flesh out a triage task and promote it to ``todo``.

    Atomically updates ``title`` / ``body`` / ``assignee`` (when provided)
    and transitions ``status: triage -> todo`` in a single write txn. Returns
    False when the task is missing or not in the ``triage`` column — callers
    should surface that as "nothing to specify" rather than an error.

    ``todo`` (not ``ready``) is the correct landing column: ``recompute_ready``
    promotes parent-free / parent-done todos to ``ready`` on the next
    dispatcher tick, which keeps the normal parent-gating behaviour intact
    for specified tasks that happen to have open parents.

    ``author`` is recorded on an audit comment only when at least one of
    ``title`` / ``body`` / ``assignee`` actually changed — avoids noisy
    comment spam for status-only promotions.
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
            f"WHERE id = ? AND status = 'triage'",
            tuple(params),
        )
        if cur.rowcount != 1:
            return False
        if changed_fields and author and author.strip():
            # Inline INSERT (rather than ``add_comment``) because we're
            # already inside this function's write_txn — nested BEGIN
            # IMMEDIATE would raise OperationalError. We also skip the
            # 'commented' event that ``add_comment`` emits, since the
            # 'specified' event below already records the change.
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Specified — updated "
                    + ", ".join(changed_fields)
                    + " and promoted to todo.",
                    int(time.time()),
                ),
            )
        _append_event(
            conn,
            task_id,
            "specified",
            {"changed_fields": changed_fields} if changed_fields else None,
        )
    # Outside the write_txn above, so we don't nest BEGIN IMMEDIATE — the
    # ready-promotion pass opens its own IMMEDIATE txn. This runs the same
    # logic the dispatcher would on its next tick, so a specified task
    # with no open parents flips straight to 'ready' here instead of
    # idling in 'todo' until the next sweep.
    recompute_ready(conn)
    return True


def _validate_children_graph(children: list) -> None:
    """DB-free shape check + Kahn's cycle check on the sibling graph (a cycle
    would deadlock every involved child in ``todo`` forever)."""
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(f"child[{idx}].parents[{p}] is not a valid index into children")
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    in_deg = [0] * len(children)
    adj: list[list[int]] = [[] for _ in children]
    for i, c in enumerate(children):
        for p in (c.get("parents") or []):
            adj[p].append(i)
            in_deg[i] += 1
    queue = [i for i in range(len(children)) if in_deg[i] == 0]
    seen = 0
    while queue:
        seen += 1
        for nb in adj[queue.pop()]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    if seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
) -> Optional[list[str]]:
    """Fan a triage task out into child tasks and promote the root to ``todo``.

    The root task stays alive and becomes the parent of every child —
    when all children reach ``done``, the root promotes to ``ready`` and
    its assignee (typically the orchestrator profile) wakes back up to
    judge completion or spawn more work.

    ``children`` is a list of dicts, each shaped like::

        {
            "title": "...",
            "body": "...",                     # optional
            "assignee": "profile-name",        # optional, None -> default fallback
            "parents": [0, 2],                 # indices into this same children list
        }

    Returns the list of created child task ids (in input order) on
    success. Returns ``None`` when:
      - The root task does not exist
      - The root task is not in ``triage``
      - A cycle would result (caller built a bad graph)

    Validation of titles/assignees happens inside the same write_txn as
    the inserts so a malformed entry aborts the whole decomposition
    cleanly (no orphan children).
    """
    if not children:
        return None
    if root_assignee is not None:
        root_assignee = _canonical_assignee(root_assignee)

    # Pre-validate the children list shape outside the txn. Cheap checks
    # that don't need DB access. Bad input aborts before we touch the DB.
    for idx, child in enumerate(children):
        if not isinstance(child, dict):
            raise ValueError(f"child[{idx}] is not a dict")
        title = child.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"child[{idx}].title is required")
        parents_idx = child.get("parents") or []
        if not isinstance(parents_idx, list):
            raise ValueError(f"child[{idx}].parents must be a list")
        for p in parents_idx:
            if not isinstance(p, int) or p < 0 or p >= len(children):
                raise ValueError(
                    f"child[{idx}].parents[{p}] is not a valid index into children"
                )
            if p == idx:
                raise ValueError(f"child[{idx}] cannot list itself as a parent")

    # Detect cycles in the sibling parent graph (Kahn's topological sort).
    # link_tasks() calls _would_cycle() for every new edge; here we check
    # the entire sibling graph before touching the DB.  A cycle silently
    # deadlocks every involved child in 'todo' because recompute_ready()
    # can never promote them.
    _in_deg = [0] * len(children)
    _adj: list[list[int]] = [[] for _ in range(len(children))]
    for _i, _c in enumerate(children):
        for _p in (_c.get("parents") or []):
            _adj[_p].append(_i)
            _in_deg[_i] += 1
    _queue = [_i for _i in range(len(children)) if _in_deg[_i] == 0]
    _seen = 0
    while _queue:
        _node = _queue.pop()
        _seen += 1
        for _nb in _adj[_node]:
            _in_deg[_nb] -= 1
            if _in_deg[_nb] == 0:
                _queue.append(_nb)
    if _seen != len(children):
        raise ValueError("cyclic dependency detected in decomposed children list")

    # We do the full decomposition in a SINGLE write_txn so it's
    # atomic: either every child is created AND the root flips to
    # ``todo``, or nothing changes. We deliberately do NOT call any
    # kb helper that opens its own write_txn (create_task, link_tasks,
    # add_comment) from inside this block — see architecture.md
    # write_txn pitfalls. Instead we inline the INSERTs and
    # _append_event calls.
    now = int(time.time())
    child_ids: list[str] = []
    with write_txn(conn):
        root_row = conn.execute(
            "SELECT id, status, tenant, workspace_kind, workspace_path, tier "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] != "triage":
            return None
        tenant = root_row["tenant"]
        # WS-3 tier inheritance: children of a full-tier parent inherit
        # the parent's tier so WS-1 (contract gate) and WS-4 (review gate)
        # apply to them. Fast-tier or untyped parents leave children untyped.
        parent_tier = root_row["tier"]
        # Children inherit the root's workspace by default so a fan-out
        # of a code-gen task lands in the parent's project dir/worktree
        # rather than throwaway scratch tmp dirs. A child dict can still
        # override with its own 'workspace_kind' / 'workspace_path'.
        root_ws_kind = root_row["workspace_kind"] or "scratch"
        root_ws_path = root_row["workspace_path"]

        # Create children. Status is 'todo' regardless of parents — we
        # link them under the root AFTER creation so the dispatcher
        # sees a coherent state, and recompute_ready() at the end
        # promotes parent-free children to 'ready'.
        for idx, child in enumerate(children):
            new_id = _new_task_id()
            title = child["title"].strip()
            body = child.get("body")
            assignee = _canonical_assignee(child.get("assignee"))
            # Per-child override wins; otherwise inherit the root's
            # workspace. A child that sets workspace_kind without a path
            # falls back to the root path only when kinds match (so a
            # child can't accidentally point a 'dir' at the root's
            # worktree path or vice versa).
            child_ws_kind = child.get("workspace_kind") or root_ws_kind
            if child.get("workspace_path"):
                child_ws_path = child.get("workspace_path")
            elif child_ws_kind == "worktree":
                # Never share one worktree checkout between siblings: the
                # root's literal path would put every child in the same
                # directory on the first-dispatched sibling's branch, with
                # no lock — siblings can be promoted and dispatched
                # concurrently. Leave the path unset so dispatch
                # materializes a fresh <repo>/.worktrees/<child-id> per
                # child from the board anchor.
                child_ws_path = None
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            # Decomposed children default to goal-mode: an implementation
            # slice split out of a proposal is exactly the "keep working
            # until a judge agrees it's done" shape the goal loop exists
            # for, and without it a complex child dies on the single-shot
            # per-turn iteration cap (V9 incident 2026-08-11: "Iteration
            # budget exhausted (30/30)" on sanitizer/regression/skill
            # scoring tasks). A child dict can still opt out with
            # goal_mode=False or tune turns via goal_max_turns.
            child_goal_mode = child.get("goal_mode", True)
            child_goal_max_turns = child.get("goal_max_turns")
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, created_at, created_by, tier, "
                " goal_mode, goal_max_turns) "
                "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    now,
                    (author or "decomposer"),
                    parent_tier,
                    1 if child_goal_mode else 0,
                    child_goal_max_turns,
                ),
            )
            _append_event(
                conn, new_id, "created",
                {"by": author or "decomposer", "from_decompose_of": task_id},
            )
            _inherit_notify_subs(conn, new_id, (task_id,), created_at=now)
            child_ids.append(new_id)

        # Link children to their sibling parents (within the decomposed graph).
        for idx, child in enumerate(children):
            for p_idx in child.get("parents") or []:
                parent_id = child_ids[p_idx]
                child_id = child_ids[idx]
                conn.execute(
                    "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                    "VALUES (?, ?)",
                    (parent_id, child_id),
                )
                _append_event(
                    conn, child_id, "linked",
                    {"parent": parent_id, "child": child_id},
                )

        # Link the ROOT task as a child of every leaf child — i.e. the
        # root waits for the whole graph. Simpler than computing leaves:
        # link root under every child. Cycle-free because the root is
        # only ever a child here, never a parent of children.
        for cid in child_ids:
            conn.execute(
                "INSERT OR IGNORE INTO task_links (parent_id, child_id) "
                "VALUES (?, ?)",
                (cid, task_id),
            )

        # Flip the root: triage -> todo, set assignee to the orchestrator.
        sets = ["status = 'todo'"]
        params: list[Any] = []
        if root_assignee is not None:
            sets.append("assignee = ?")
            params.append(root_assignee)
        params.append(task_id)
        conn.execute(
            f"UPDATE tasks SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )

        # Audit comment + event on the root so the timeline shows the fan-out.
        if author and author.strip():
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    task_id,
                    author.strip(),
                    "Decomposed into "
                    + ", ".join(child_ids)
                    + ". Root will wake when all children complete.",
                    now,
                ),
            )
        _append_event(
            conn, task_id, "decomposed",
            {
                "child_ids": child_ids,
                "root_assignee": root_assignee,
            },
        )

    # Outside the write_txn: promote parent-free children to 'ready'
    # so the dispatcher picks them up on its next tick. Same pattern
    # specify_triage_task uses.  When auto_promote is False children
    # stay in 'todo' until the user manually promotes them — useful
    # for manual-review-first workflows.
    if auto_promote:
        recompute_ready(conn)
    return child_ids


def _insert_decomposed_child(
    conn: sqlite3.Connection, root_id: str, root_row: sqlite3.Row, child: dict,
    author: Optional[str], now: int,
) -> str:
    """Insert one decomposed child as ``todo`` (linked under the root later so
    the dispatcher only ever sees a coherent graph); returns its id.

    Workspace: per-child override wins, else inherit the root's kind. Path
    inherits only when kinds match (a 'dir' child must not point at the
    root's worktree) and NEVER for worktrees — siblings dispatch concurrently
    and one shared checkout would put them all on the first sibling's branch
    with no lock; leaving it unset makes dispatch materialize a fresh
    ``<repo>/.worktrees/<child-id>`` per child from the board anchor.
    """
    root_ws_kind = root_row["workspace_kind"] or "scratch"
    child_ws_kind = child.get("workspace_kind") or root_ws_kind
    if child.get("workspace_path"):
        child_ws_path = child.get("workspace_path")
    elif child_ws_kind == "worktree":
        child_ws_path = None
    elif child_ws_kind == root_ws_kind:
        child_ws_path = root_row["workspace_path"]
    else:
        child_ws_path = None
    new_id = _new_task_id()
    body = child.get("body")
    conn.execute(
        "INSERT INTO tasks "
        "(id, title, body, assignee, status, workspace_kind, "
        " workspace_path, tenant, created_at, created_by) "
        "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?)",
        (
            new_id, child["title"].strip(), body if isinstance(body, str) else None,
            _canonical_assignee(child.get("assignee")), child_ws_kind, child_ws_path,
            root_row["tenant"], now, (author or "decomposer"),
        ),
    )
    _append_event(
        conn, new_id, "created", {"by": author or "decomposer", "from_decompose_of": root_id},
    )
    _inherit_notify_subs(conn, new_id, (root_id,), created_at=now)
    return new_id


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
    """Hard-delete a task and cascade to all related rows.

    Because the schema does not use ``ON DELETE CASCADE`` foreign keys,
    we explicitly delete from child tables first, then the task row.
    This keeps the operation atomic (single ``write_txn``).

    Returns ``True`` if the task existed and was deleted, ``False``
    if the task was not found.
    """
    with write_txn(conn):
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if cur.rowcount != 1:
            return False
        conn.execute("DELETE FROM task_links WHERE parent_id = ? OR child_id = ?", (task_id, task_id))
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
    recompute_ready(conn)
    return True


class MoveError(Exception):
    """Raised when a cross-board move cannot proceed safely."""


_MOVE_INT_PK_TABLES = ["task_runs", "task_events", "task_comments", "task_attachments"]


_MOVE_TEXT_PK_TABLES = ["kanban_notify_subs", "profile_lifecycle_approvals"]


_MOVE_ALL_TASK_TABLES = ["tasks"] + _MOVE_INT_PK_TABLES + _MOVE_TEXT_PK_TABLES


_TERMINAL_RUN_STATUSES = frozenset({
    "done", "blocked", "crashed", "timed_out", "failed", "released",
})


def _fsync_file(fh):
    """fsync a file handle, ignoring OS-level errors on platforms without it."""
    try:
        _os.fsync(fh.fileno())
    except (OSError, AttributeError):
        pass


def _fsync_dir(path):
    """fsync a directory to flush rename operations."""
    try:
        fd = _os.open(str(path), _os.O_RDONLY)
        try:
            _os.fsync(fd)
        finally:
            _os.close(fd)
    except OSError:
        pass


def _sha256_file(path):
    """Return the SHA-256 hex digest of a file."""
    h = _hashlib.sha256()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _table_columns(conn, schema, table):
    """Return list of (name, notnull, dflt_value) for a table."""
    rows = conn.execute(
        "PRAGMA " + schema + ".table_info(" + table + ")"
    ).fetchall()
    return [(r["name"], r["notnull"], r["dflt_value"]) for r in rows]


def _check_schema_compat(conn, src_schema, tgt_schema, table):
    """Check both directions for schema compatibility.

    Fail if either side has a required (NOT NULL, no default) column
    that the other lacks.  Fixed schema identifiers only.
    """
    src_cols = {name: (notnull, dflt) for name, notnull, dflt in
                _table_columns(conn, src_schema, table)}
    tgt_cols = {name: (notnull, dflt) for name, notnull, dflt in
                _table_columns(conn, tgt_schema, table)}

    # Source has a column target lacks — reject if source value could be non-null
    for col, (notnull, dflt) in src_cols.items():
        if col not in tgt_cols:
            if notnull and dflt is None:
                raise MoveError(
                    "schema incompatible: source table " + table +
                    " has required column " + col +
                    " not present in target"
                )

    # Target has a column source lacks — reject if target requires it
    for col, (notnull, dflt) in tgt_cols.items():
        if col not in src_cols:
            if notnull and dflt is None:
                raise MoveError(
                    "schema incompatible: target table " + table +
                    " has required column " + col +
                    " not present in source"
                )


def _resolve_board_slug(path):
    """Best-effort board slug from a DB path for the moved-event payload.

    For the default board the DB sits at ``<root>/kanban.db`` (parent is
    the root, NOT 'boards').  For a named board the DB sits at
    ``<root>/kanban/boards/<slug>/kanban.db`` (parent is <slug>,
    grandparent is 'boards').
    """
    try:
        p = Path(path)
        if p.parent.parent.name == "boards":
            return p.parent.name
        return "default"
    except Exception:
        return "unknown"


def _is_safe_filename(name):
    """Reject filenames with path separators, traversal, or empty."""
    if not name or not name.strip():
        return False
    if "/" in name or "\\" in name:
        return False
    if name == "." or name == "..":
        return False
    if name.startswith("."):
        return False
    return True


def _copy_attachment_blob(src_path, tgt_dir, expected_size, expected_hash=None):
    """Copy a file to tgt_dir with atomic rename, fsync, and verification.

    Returns the final path.  Raises MoveError on any safety violation.
    """
    src_path = Path(src_path)
    if not src_path.exists():
        raise MoveError("attachment source missing: " + str(src_path))
    if src_path.is_symlink():
        raise MoveError("attachment source is symlink: " + str(src_path))
    if not src_path.is_file():
        raise MoveError("attachment source not a regular file: " + str(src_path))

    actual_size = src_path.stat().st_size
    if actual_size != expected_size:
        raise MoveError(
            "attachment size mismatch: expected " + str(expected_size) +
            " got " + str(actual_size)
        )

    tgt_dir.mkdir(parents=True, exist_ok=True)
    final_name = src_path.name
    final_path = tgt_dir / final_name

    # If final already exists (retry), verify hash matches
    if final_path.exists():
        if final_path.is_symlink() or not final_path.is_file():
            raise MoveError("attachment final exists but is not a regular file: " + str(final_path))
        if expected_hash is not None:
            existing_hash = _sha256_file(final_path)
            if existing_hash != expected_hash:
                raise MoveError("attachment final exists with different hash")
        # Already correct — return it
        return final_path

    # Copy to unique temp file, fsync, atomic rename
    fd, tmp_name = _tempfile.mkstemp(dir=str(tgt_dir), prefix=".move_tmp_")
    try:
        with _os.fdopen(fd, "wb") as tmp_f:
            with open(str(src_path), "rb") as src_f:
                while True:
                    chunk = src_f.read(65536)
                    if not chunk:
                        break
                    tmp_f.write(chunk)
            _fsync_file(tmp_f)
        _os.rename(tmp_name, str(final_path))
        _fsync_dir(tgt_dir)
    except Exception:
        # Clean up temp on failure
        try:
            _os.unlink(tmp_name)
        except OSError:
            pass
        raise

    return final_path


def _cleanup_attachment_artifacts(tgt_dir):
    """Clean up .move_tmp_ temp files and orphaned finals from a failed attempt.

    Only removes temp files (``.move_tmp_*``); orphaned final files are
    left in place for retry/hash verification — a retry must recognise,
    verify and reuse matching finals, and fail closed on mismatched ones.
    """
    if not tgt_dir.exists():
        return
    for child in tgt_dir.iterdir():
        if child.name.startswith(".move_tmp_"):
            try:
                child.unlink()
            except OSError:
                pass


def _acquire_dual_locks(src_path, tgt_path):
    """Acquire both .write_lock sidecars in sorted resolved-path order.

    Returns (lock_handles, lock_paths) or raises MoveError on same-path.
    """
    src_resolved = str(Path(src_path).resolve())
    tgt_resolved = str(Path(tgt_path).resolve())

    if src_resolved == tgt_resolved:
        raise MoveError("source and target are the same database path")

    # Sorted order for deadlock avoidance
    paths = sorted([src_resolved, tgt_resolved])
    handles = []
    lock_paths = []
    try:
        for p in paths:
            lock_path = Path(p).with_name(Path(p).name + ".write_lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fh = lock_path.open("a+b")
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_EX)
            handles.append(fh)
            lock_paths.append(lock_path)
    except Exception:
        for fh in handles:
            try:
                _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
                fh.close()
            except Exception:
                pass
        raise

    return handles, lock_paths


def _release_locks(handles):
    """Release all acquired sidecar locks."""
    for fh in handles:
        try:
            _fcntl.flock(fh.fileno(), _fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass


def _raw_connect(path):
    """Open a raw sqlite3 connection (isolation_level=None, autocommit).

    Does NOT set WAL mode or any PRAGMAs beyond busy_timeout.  Used for
    the checkpoint/switch and WAL-restore connections that must not
    interfere with the main move transaction.
    """
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=120000")
    return conn


def _checkpoint_and_switch_to_delete(path, label):
    """Checkpoint a WAL database and switch it to DELETE journal mode.

    Opens its own raw connection, checkpoints, switches to DELETE, and
    closes the connection.  Returns the original journal mode.  Raises
    MoveError on any failure — never swallow.
    """
    conn = _raw_connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        orig_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if orig_mode and orig_mode.lower() == "wal":
            conn.execute("PRAGMA journal_mode=DELETE")
            mode_after = conn.execute("PRAGMA journal_mode").fetchone()[0]
            if mode_after.lower() != "delete":
                raise MoveError(
                    "failed to switch " + label + " from WAL to DELETE mode"
                )
        return orig_mode
    finally:
        conn.close()


def _restore_wal_mode(path, label):
    """Restore WAL journal mode on a database using a fresh connection.

    Raises on failure — the caller must surface the inability to prove
    restoration.  Never swallow.
    """
    conn = _raw_connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        mode_after = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if mode_after.lower() != "wal":
            raise MoveError(
                "failed to restore WAL mode on " + label +
                " (database is safe in DELETE mode but WAL is not restored)"
            )
    finally:
        conn.close()


def move_task_atomic(
    source_path,
    target_path,
    task_id,
    *,
    _test_hooks=None,
):
    """Atomically move a task and all its owned rows from source to target board.

    Architecture (binding contract):

    1. Reject same resolved source/target path.
    2. Acquire both .write_lock sidecars in sorted resolved-path order.
    3. Under those locks, use **two separate raw connections** to
       checkpoint each WAL database and switch each to journal_mode=DELETE.
    4. **Close both switching connections** before opening the move connection.
    5. Open one source connection, ATTACH target, begin one BEGIN IMMEDIATE
       transaction spanning both rollback-journal databases.
    6. Do NOT switch the attached target journal mode.
    7. Perform all guards and mutations inside that attached transaction.
    8. Commit once, close/detach.
    9. Restore WAL on both databases using **separate connections**.
       Fail closed / surface inability to prove restoration.

    Returns a dict: {"status": "moved"|"already_moved"|"rejected"|"error", ...}
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    hooks = _test_hooks or {}

    # --- Same-path rejection (before any lock) ---
    try:
        if source_path.resolve() == target_path.resolve():
            return {"status": "rejected", "reason": "source and target are the same database path"}
    except Exception as exc:
        return {"status": "rejected", "reason": "path resolution error: " + str(exc)}

    src_board = _resolve_board_slug(source_path)
    tgt_board = _resolve_board_slug(target_path)

    # --- Acquire dual locks in sorted resolved-path order ---
    try:
        lock_handles, lock_paths = _acquire_dual_locks(source_path, target_path)
    except MoveError as exc:
        return {"status": "rejected", "reason": str(exc)}
    except Exception as exc:
        return {"status": "error", "reason": "lock acquisition failed: " + str(exc)}

    conn = None
    src_orig_mode = None
    tgt_orig_mode = None
    cleanup_dirs = []  # attachment dirs that may have temp files on failure
    wal_restored = {"src": False, "tgt": False}

    try:
        # --- Journal switch: checkpoint + switch to DELETE (separate conns) ---
        # Use two separate raw connections, one per DB. Close both before
        # opening the move connection. This avoids the "database is locked"
        # that occurs when switching journal_mode through an ATTACHed WAL
        # connection.
        try:
            src_orig_mode = _checkpoint_and_switch_to_delete(source_path, "source")
        except MoveError:
            raise
        except Exception as exc:
            raise MoveError("source journal switch failed: " + str(exc)) from exc

        try:
            tgt_orig_mode = _checkpoint_and_switch_to_delete(target_path, "target")
        except MoveError:
            raise
        except Exception as exc:
            raise MoveError("target journal switch failed: " + str(exc)) from exc

        # --- Open source connection and ATTACH target ---
        # Both DBs are now in DELETE mode. Do NOT switch the attached
        # target's journal mode.
        conn = _raw_connect(source_path)
        conn.execute(
            "ATTACH DATABASE '" + str(target_path.resolve()) + "' AS tgt"
        )

        # --- Schema compatibility check (both directions) ---
        for table in _MOVE_ALL_TASK_TABLES:
            _check_schema_compat(conn, "main", "tgt", table)

        # --- Idempotency check (under locks, pre-transaction) ---
        src_exists = conn.execute(
            "SELECT 1 FROM main.tasks WHERE id = ?", (task_id,)
        ).fetchone()
        tgt_exists = conn.execute(
            "SELECT 1 FROM tgt.tasks WHERE id = ?", (task_id,)
        ).fetchone()

        if src_exists is None and tgt_exists is not None:
            moved_count = conn.execute(
                "SELECT COUNT(*) FROM tgt.task_events WHERE task_id = ? AND kind = 'moved'",
                (task_id,),
            ).fetchone()[0]
            if moved_count == 1:
                # A SIGKILL after COMMIT but before the normal post-commit
                # restoration leaves both DBs safely committed in DELETE mode.
                # Idempotent retry is the recovery path: close the ATTACHed
                # connection first, then heal both journals through separate
                # raw connections while the canonical sidecar locks are held.
                try:
                    conn.execute("DETACH DATABASE tgt")
                    conn.close()
                    conn = None
                    _restore_wal_mode(source_path, "source idempotent recovery")
                    wal_restored["src"] = True
                    _restore_wal_mode(target_path, "target idempotent recovery")
                    wal_restored["tgt"] = True
                except Exception as exc:
                    return {
                        "status": "error",
                        "reason": "idempotent recovery could not restore WAL: " + str(exc),
                    }
                return {"status": "already_moved", "reason": "task already moved to target"}
            return {"status": "rejected", "reason": "ambiguous state: task on target with " + str(moved_count) + " moved events"}

        if src_exists is not None and tgt_exists is not None:
            return {"status": "rejected", "reason": "ambiguous state: task exists on both source and target"}

        if src_exists is None:
            return {"status": "rejected", "reason": "task " + task_id + " not found on source"}

        # --- Begin attached transaction ---
        conn.execute("BEGIN IMMEDIATE")

        try:
            # --- Rejection guards (under locked transaction) ---
            task_row = conn.execute(
                "SELECT * FROM main.tasks WHERE id = ?", (task_id,)
            ).fetchone()
            if task_row is None:
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task vanished from source under lock"}

            if task_row["status"] == "running":
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task status is 'running'"}

            if task_row["current_run_id"] is not None:
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task has non-null current_run_id"}

            # Check for active/open runs
            active_run = conn.execute(
                "SELECT 1 FROM main.task_runs WHERE task_id = ? AND status = 'running'",
                (task_id,),
            ).fetchone()
            if active_run is not None:
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task has a running task_run"}

            # Check for runs without terminal end state — a nominally
            # terminal run (e.g. status='done') with ended_at IS NULL is
            # still "open" and must be rejected.
            unended_run = conn.execute(
                "SELECT 1 FROM main.task_runs WHERE task_id = ? AND ended_at IS NULL",
                (task_id,),
            ).fetchone()
            if unended_run is not None:
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task has a task_run without terminal end state (ended_at is null)"}

            # Check for pending/approved lifecycle approvals
            pending_approval = conn.execute(
                "SELECT 1 FROM main.profile_lifecycle_approvals "
                "WHERE task_id = ? AND status IN ('pending', 'approved')",
                (task_id,),
            ).fetchone()
            if pending_approval is not None:
                conn.execute("ROLLBACK")
                return {"status": "rejected", "reason": "task has pending or approved lifecycle approval"}

            # --- Pre-copy attachment files (before DB commit) ---
            attachments = conn.execute(
                "SELECT * FROM main.task_attachments WHERE task_id = ?", (task_id,)
            ).fetchall()
            src_att_root = attachments_root(board=src_board)
            tgt_att_root = attachments_root(board=tgt_board)
            tgt_att_dir = tgt_att_root / task_id
            attachment_new_paths = {}  # old_id -> new_stored_path

            for att in attachments:
                att_id = att["id"]
                stored_path = Path(att["stored_path"])
                filename = att["filename"]

                # Containment: source stored_path must be under source att root
                try:
                    expected_src_dir = src_att_root / task_id
                    stored_resolved = stored_path.resolve()
                    expected_resolved = expected_src_dir.resolve()
                    stored_resolved.relative_to(expected_resolved)
                except (ValueError, RuntimeError):
                    conn.execute("ROLLBACK")
                    return {"status": "rejected", "reason": "attachment stored_path not contained under source attachment root: " + str(stored_path)}

                # Reject symlinks
                if stored_path.is_symlink():
                    conn.execute("ROLLBACK")
                    return {"status": "rejected", "reason": "attachment source is a symlink: " + str(stored_path)}

                # Reject unsafe filename (traversal)
                if not _is_safe_filename(filename):
                    conn.execute("ROLLBACK")
                    return {"status": "rejected", "reason": "unsafe attachment filename: " + repr(filename)}

                # Copy the blob (hash-verified, atomic rename, fsync)
                try:
                    src_hash = _sha256_file(stored_path)
                    new_path = _copy_attachment_blob(
                        stored_path, tgt_att_dir, att["size"], src_hash
                    )
                    attachment_new_paths[att_id] = str(new_path)
                    if tgt_att_dir not in cleanup_dirs:
                        cleanup_dirs.append(tgt_att_dir)
                except MoveError as exc:
                    conn.execute("ROLLBACK")
                    for d in cleanup_dirs:
                        _cleanup_attachment_artifacts(d)
                    return {"status": "rejected", "reason": str(exc)}

            # --- 7-table copy with ID remap ---

            # 1. task_runs (integer PK, needs remap)
            run_id_map = {}  # old_id -> new_id
            runs = conn.execute(
                "SELECT * FROM main.task_runs WHERE task_id = ? ORDER BY id", (task_id,)
            ).fetchall()
            run_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(task_runs)"
            ).fetchall() if r["name"] != "id"]

            for run in runs:
                old_id = run["id"]
                values = [run[c] for c in run_cols]
                placeholders = ", ".join("?" for _ in run_cols)
                col_list = ", ".join(run_cols)
                cur = conn.execute(
                    "INSERT INTO tgt.task_runs (" + col_list + ") VALUES (" + placeholders + ")",
                    values,
                )
                run_id_map[old_id] = cur.lastrowid

            # 2. task_events (integer PK, run_id remap)
            events = conn.execute(
                "SELECT * FROM main.task_events WHERE task_id = ? ORDER BY id", (task_id,)
            ).fetchall()
            ev_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(task_events)"
            ).fetchall() if r["name"] != "id"]

            for ev in events:
                ev_values = []
                for c in ev_cols:
                    if c == "run_id" and ev["run_id"] is not None:
                        ev_values.append(run_id_map.get(ev["run_id"]))
                    else:
                        ev_values.append(ev[c])
                placeholders = ", ".join("?" for _ in ev_cols)
                col_list = ", ".join(ev_cols)
                conn.execute(
                    "INSERT INTO tgt.task_events (" + col_list + ") VALUES (" + placeholders + ")",
                    ev_values,
                )

            # 3. task_comments (integer PK, no remap needed)
            comments = conn.execute(
                "SELECT * FROM main.task_comments WHERE task_id = ? ORDER BY id", (task_id,)
            ).fetchall()
            cm_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(task_comments)"
            ).fetchall() if r["name"] != "id"]

            for cm in comments:
                values = [cm[c] for c in cm_cols]
                placeholders = ", ".join("?" for _ in cm_cols)
                col_list = ", ".join(cm_cols)
                conn.execute(
                    "INSERT INTO tgt.task_comments (" + col_list + ") VALUES (" + placeholders + ")",
                    values,
                )

            # 4. task_attachments (integer PK, stored_path update)
            att_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(task_attachments)"
            ).fetchall() if r["name"] != "id"]

            for att in attachments:
                att_values = []
                for c in att_cols:
                    if c == "stored_path":
                        att_values.append(attachment_new_paths.get(att["id"], att["stored_path"]))
                    else:
                        att_values.append(att[c])
                placeholders = ", ".join("?" for _ in att_cols)
                col_list = ", ".join(att_cols)
                conn.execute(
                    "INSERT INTO tgt.task_attachments (" + col_list + ") VALUES (" + placeholders + ")",
                    att_values,
                )

            # 5. kanban_notify_subs (composite text PK, no remap)
            subs = conn.execute(
                "SELECT * FROM main.kanban_notify_subs WHERE task_id = ?", (task_id,)
            ).fetchall()
            sub_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(kanban_notify_subs)"
            ).fetchall()]

            for sub in subs:
                values = [sub[c] for c in sub_cols]
                placeholders = ", ".join("?" for _ in sub_cols)
                col_list = ", ".join(sub_cols)
                conn.execute(
                    "INSERT INTO tgt.kanban_notify_subs (" + col_list + ") VALUES (" + placeholders + ")",
                    values,
                )

            # 6. profile_lifecycle_approvals (text PK, no remap)
            # Only terminal approvals (rejected/executed/failed) reach here —
            # pending/approved were rejected above.
            approvals = conn.execute(
                "SELECT * FROM main.profile_lifecycle_approvals WHERE task_id = ?", (task_id,)
            ).fetchall()
            appr_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(profile_lifecycle_approvals)"
            ).fetchall()]

            for appr in approvals:
                values = [appr[c] for c in appr_cols]
                placeholders = ", ".join("?" for _ in appr_cols)
                col_list = ", ".join(appr_cols)
                conn.execute(
                    "INSERT INTO tgt.profile_lifecycle_approvals (" + col_list + ") VALUES (" + placeholders + ")",
                    values,
                )

            # 7. tasks (text PK, clear runtime state + null cross-board epic_id)
            task_cols = [r["name"] for r in conn.execute(
                "PRAGMA main.table_info(tasks)"
            ).fetchall()]
            task_values = []
            for c in task_cols:
                if c == "epic_id":
                    task_values.append(None)  # sever cross-board epic FK
                elif c == "current_run_id":
                    task_values.append(None)  # movable task has no active run
                elif c == "claim_lock":
                    task_values.append(None)
                elif c == "claim_expires":
                    task_values.append(None)
                elif c == "worker_pid":
                    task_values.append(None)
                elif c == "last_heartbeat_at":
                    task_values.append(None)
                else:
                    task_values.append(task_row[c])
            placeholders = ", ".join("?" for _ in task_cols)
            col_list = ", ".join(task_cols)
            conn.execute(
                "INSERT INTO tgt.tasks (" + col_list + ") VALUES (" + placeholders + ")",
                task_values,
            )

            # --- Sever all source-board links involving the moved task ---
            conn.execute(
                "DELETE FROM main.task_links WHERE parent_id = ? OR child_id = ?",
                (task_id, task_id),
            )

            # --- Emit exactly one moved event in target ---
            now = int(time.time())
            moved_payload = json.dumps({"from_board": src_board, "to_board": tgt_board})
            conn.execute(
                "INSERT INTO tgt.task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'moved', ?, ?)",
                (task_id, moved_payload, now),
            )

            # --- In-transaction source dependency recompute ---
            # Links are already severed; recompute all todo/blocked on
            # source (the moved task's children may now be promotable).
            _recompute_ready_locked(conn, task_filter=None)

            # --- In-transaction target dependency recompute ---
            # Recompute the moved task on target. We can't use
            # _recompute_ready_locked directly (it queries main.*); use
            # transaction-aware manual logic against tgt.* that preserves
            # manual/sticky and circuit-breaker blocks exactly.
            _recompute_target_task_locked(conn, task_id, now)

            # --- Failpoint: after_target_copy ---
            if "after_target_copy" in hooks:
                hooks["after_target_copy"]()

            # --- Delete source rows in dependency-safe order ---
            conn.execute("DELETE FROM main.task_events WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.task_comments WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.task_runs WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.task_attachments WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.kanban_notify_subs WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.profile_lifecycle_approvals WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM main.tasks WHERE id = ?", (task_id,))

            # --- Failpoint: before_commit ---
            if "before_commit" in hooks:
                hooks["before_commit"]()

            # --- COMMIT ---
            conn.execute("COMMIT")

            # --- Failpoint: after_commit ---
            if "after_commit" in hooks:
                hooks["after_commit"]()

        except Exception as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            # Clean up attachment temp files from failed attempt
            for d in cleanup_dirs:
                _cleanup_attachment_artifacts(d)
            raise

        # --- Post-commit: detach and close move connection ---
        # Must close before WAL restoration (separate connections).
        try:
            conn.execute("DETACH DATABASE tgt")
        except Exception:
            pass
        conn.close()
        conn = None

        # --- Post-commit: WAL restoration (separate connections) ---
        # Never swallow restoration errors — surface inability to prove
        # restoration. A crash after commit but before restoration leaves
        # safe DELETE-mode databases; normal connect() restores WAL.
        wal_errors = []
        if src_orig_mode and src_orig_mode.lower() == "wal":
            try:
                _restore_wal_mode(source_path, "source")
                wal_restored["src"] = True
            except Exception as exc:
                wal_errors.append("source WAL restore failed: " + str(exc))
        else:
            wal_restored["src"] = True

        if tgt_orig_mode and tgt_orig_mode.lower() == "wal":
            try:
                _restore_wal_mode(target_path, "target")
                wal_restored["tgt"] = True
            except Exception as exc:
                wal_errors.append("target WAL restore failed: " + str(exc))
        else:
            wal_restored["tgt"] = True

        if wal_errors:
            # Databases are durable in DELETE mode but WAL is not restored.
            # Surface the error — do not report ordinary success.
            return {
                "status": "error",
                "reason": "; ".join(wal_errors),
                "task_id": task_id,
                "wal_restored": wal_restored,
            }

        # --- Post-commit: source blob deletion (best-effort, retryable) ---
        for att in attachments:
            try:
                old_path = Path(att["stored_path"])
                if old_path.exists() and not old_path.is_symlink():
                    old_path.unlink()
            except OSError:
                pass  # best-effort, retryable

        return {"status": "moved", "task_id": task_id}

    except MoveError as exc:
        return {"status": "rejected", "reason": str(exc)}
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    finally:
        if conn is not None:
            try:
                conn.execute("DETACH DATABASE tgt")
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
        # If we switched to DELETE but didn't restore WAL yet, attempt
        # restoration in the finally block as a safety net. This handles
        # the case where an exception prevented the post-commit restore.
        if not wal_restored["src"] and src_orig_mode and src_orig_mode.lower() == "wal":
            try:
                _restore_wal_mode(source_path, "source")
            except Exception as exc:
                _log.warning("finally: source WAL restore failed: %s", exc)
        if not wal_restored["tgt"] and tgt_orig_mode and tgt_orig_mode.lower() == "wal":
            try:
                _restore_wal_mode(target_path, "target")
            except Exception as exc:
                _log.warning("finally: target WAL restore failed: %s", exc)
        _release_locks(lock_handles)


def _recompute_target_task_locked(conn, task_id, now):
    """Transaction-aware recompute for the moved task on the target (tgt.*) schema.

    Called inside the attached move transaction.  Mirrors
    :func:`_recompute_ready_locked` but queries ``tgt.tasks`` and
    ``tgt.task_links`` instead of ``main.*``.  Preserves manual/sticky
    blocks and circuit-breaker blocks exactly — no generic
    blocked→ready promotion.
    """
    tgt_task = conn.execute(
        "SELECT status, block_kind, consecutive_failures, max_retries "
        "FROM tgt.tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if tgt_task is None:
        return
    if tgt_task["status"] not in ("todo", "blocked"):
        return

    # Preserve sticky/manual blocks: check last blocked/unblocked event
    tgt_sticky = conn.execute(
        "SELECT kind FROM tgt.task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    is_sticky = tgt_sticky and tgt_sticky["kind"] == "blocked"
    if is_sticky:
        return

    # Check if all parents are done/archived
    parents = conn.execute(
        "SELECT t.status FROM tgt.tasks t "
        "JOIN tgt.task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ?",
        (task_id,),
    ).fetchall()
    if not parents or all(p["status"] in ("done", "archived") for p in parents):
        if tgt_task["status"] == "blocked":
            # Check circuit breaker
            failures = int(tgt_task["consecutive_failures"] or 0)
            task_limit = tgt_task["max_retries"]
            effective_limit = (
                int(task_limit) if task_limit is not None
                else int(DEFAULT_FAILURE_LIMIT)
            )
            if failures >= effective_limit:
                return
            conn.execute(
                "UPDATE tgt.tasks SET status = 'ready' "
                "WHERE id = ? AND status = 'blocked'",
                (task_id,),
            )
        else:
            conn.execute(
                "UPDATE tgt.tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                (task_id,),
            )
        conn.execute(
            "INSERT INTO tgt.task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'promoted', NULL, ?)",
            (task_id, now),
        )


def transition_task(
    conn: sqlite3.Connection,
    task_id: str,
    status: str,
    *,
    reason: Optional[str] = None,
    block_kind: Optional[str] = None,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    reviewer: Optional[str] = None,
    force_review: bool = False,
    actor: str = "project-kanban-host",
) -> bool:
    """Route a requested status through the canonical lifecycle verb."""
    normalized = (status or "").strip().lower()
    if normalized == "running":
        raise ValueError("running is dispatcher-managed")

    current = get_task(conn, task_id)
    if current is None:
        return False
    if current.status == normalized:
        return True

    if normalized == "done":
        return complete_task(
            conn,
            task_id,
            result=result,
            summary=summary,
            metadata=metadata,
        )
    if normalized == "blocked":
        return block_task(
            conn,
            task_id,
            reason=reason,
            kind=block_kind,
        )
    if normalized == "scheduled":
        return schedule_task(conn, task_id, reason=reason)
    if normalized == "review":
        return bool(
            request_review(
                conn,
                task_id,
                summary=summary,
                metadata=metadata,
                reviewer=reviewer,
                force=force_review,
            )
        )
    if normalized == "archived":
        return archive_task(conn, task_id)
    if normalized == "ready":
        if current.status in {"blocked", "scheduled"}:
            return unblock_task(conn, task_id)
        if current.status == "review":
            return reopen_review_task(conn, task_id)
        return _set_task_status_direct(
            conn,
            task_id,
            normalized,
            actor=actor,
        )
    if normalized == "todo" and current.status == "review":
        return reopen_review_task(conn, task_id)
    if normalized in {"backlog", "triage", "todo"}:
        return _set_task_status_direct(
            conn,
            task_id,
            normalized,
            actor=actor,
        )
    raise ValueError(f"unknown status: {status!r}")


def archive_task(conn: sqlite3.Connection, task_id: str, *, signal_fn=None) -> bool:
    """Archive a task; a *running* task's host-local worker is terminated.

    Clearing ``worker_pid`` in the DB alone left the OS process running past its
    own archive — it kept executing (and pushing work) against a task nothing
    tracked anymore (#76196). Snapshot pid+claim inside the archive txn so the
    kill is contingent on THIS caller winning the archive transition (a losing
    concurrent archiver must never signal the pid); the kill itself runs after
    commit — ``_poll_worker_exit`` can wait ~5 s and must not hold the write
    lock. Post-release kill is safe here because ``archived`` is terminal: no
    dispatcher can spawn a duplicate worker off the released claim. The
    termination outcome lands as its own ``archive_worker_termination`` event so
    the ``archived`` event stays atomic with the status flip.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return False
        was_running = row["status"] == "running"
        prev_pid, prev_lock = row["worker_pid"], row["claim_lock"]
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
    if was_running:
        termination = _terminate_reclaimed_worker(prev_pid, prev_lock, signal_fn=signal_fn)
        with write_txn(conn):
            _append_event(conn, task_id, "archive_worker_termination", termination, run_id=run_id)
    # ``archived`` parents no longer block children; promote them now.
    recompute_ready(conn)
    # Reap the workspace on archive too (never-completed tasks kept it forever).
    _cleanup_workspace(conn, task_id)
    return True


def _task_pipeline_stage(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the pipeline_stage for a task, or None if not a pipeline task."""
    row = conn.execute(
        "SELECT pipeline_stage FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return None
    return row["pipeline_stage"] or None


def build_denji_report(conn: sqlite3.Connection, *, days: int = 7) -> dict:
    """Consume the pipeline governance signals into one report for Denji.

    This is the consumer side of the Denji wiring (design doc build #15):
    spawn-frequency promotion proposals, audit review signals, and express
    bypass-records over the rolling window. Denji's review-cycle cron calls
    this (via ``hermes feature denji-report``) instead of the signals sitting
    unread in the events table.
    """
    import json as _json
    cutoff = int(time.time()) - int(days) * 86400

    spawn = get_spawn_frequency(conn, days=days)
    threshold = _get_spawn_frequency_threshold()
    promotion_proposals = [
        {**r, "threshold": threshold}
        for r in spawn if r["spawn_count"] >= threshold
    ]

    def _load(kind: str) -> list[dict]:
        rows = conn.execute(
            "SELECT task_id, payload, created_at FROM task_events "
            "WHERE kind = ? AND created_at >= ? ORDER BY created_at DESC",
            (kind, cutoff),
        ).fetchall()
        items = []
        for r in rows:
            try:
                data = _json.loads(r[1]) if r[1] else {}
            except (TypeError, ValueError):
                data = {}
            items.append({"task_id": r[0], "created_at": r[2], **data})
        return items

    review_signals = _load("denji_review_signal")
    signal_counts: dict[str, int] = {}
    for s in review_signals:
        signal_counts[s.get("signal_type", "unknown")] = (
            signal_counts.get(s.get("signal_type", "unknown"), 0) + 1
        )
    bypasses = _load("bypass_record")

    return {
        "window_days": days,
        "spawn_frequency": spawn,
        "promotion_proposals": promotion_proposals,
        "review_signals": review_signals,
        "review_signal_counts": signal_counts,
        "bypass_records": bypasses,
        "bypass_count": len(bypasses),
    }


def schedule_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Park a task in ``scheduled`` so it is waiting on time, not human input.

    ``scheduled`` tasks are intentionally not dispatchable; an external cron,
    human action, or automation can later call ``unblock_task`` to re-gate them
    to ``ready`` (or ``todo`` if parents are still incomplete).

    Refuses to schedule a task currently in the ``decision-needed`` terminal
    state — that status means a human decision is gating the task and
    automation must not silently re-time it. The status whitelist below also
    excludes ``decision-needed``, so the UPDATE would no-op anyway; this guard
    is here so the refusal is *explicit* (log + audit event) rather than
    silent, matching the pattern in :func:`unblock_task`. Mirrors
    ``VALID_STATUSES`` intent: a task in a terminal blocked state is
    human-owned until a human moves it.
    """
    with write_txn(conn):
        # Defensive early-out: if the row is in ``decision-needed`` log a
        # refusal event and return False instead of relying on the UPDATE
        # whitelist below to no-op. Keeps the invariant visible to anyone
        # reading the function or the audit trail, and protects against a
        # future change that widens the whitelist.
        gated = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if gated is not None and gated["status"] == "decision-needed":
            _log.warning(
                "schedule_task refused: task %s is decision-gated "
                "(status='decision-needed'); human decision required",
                task_id,
            )
            _append_event(
                conn, task_id, "schedule_refused",
                {
                    "reason": "decision-gated",
                    "status": "decision-needed",
                },
            )
            return False
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
        cur = conn.execute(sql, params)
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="scheduled", status="scheduled",
            summary=reason,
        )
        if run_id is None and reason:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="scheduled",
                summary=reason,
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
    """Delete events older than the cutoff on done/archived tasks only; returns the count."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN "
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
    _sqlite_connect,
)
from hermes_cli.kanban_db_notify import (  # noqa: E402
    add_notify_sub,
)
from hermes_cli.kanban_db_workspace import (  # noqa: E402
    _cleanup_workspace,
    _is_managed_scratch_path,
    _managed_scratch_path_info,
    _scratch_workspace,
    # KENSEI CUSTOM (fork re-anchor): re-exports so dispatch's _kb.* refs resolve.
    _maybe_emit_scratch_tip,
    resolve_workspace,
    set_branch_name,
    set_workspace_path,
)
from hermes_cli.kanban_db_dispatch import (  # noqa: E402
    DEFAULT_FAILURE_LIMIT,
    DEFAULT_SPAWN_FAILURE_LIMIT,
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    DispatchResult,
    _clear_failure_counter,
    _defer_reclaim_for_live_worker,
    _pid_alive,
    _terminate_reclaimed_worker,
    _worker_survived_termination,
    _worker_terminal_timeout_env,
    # KENSEI CUSTOM (fork re-anchor): dispatcher/council/pipeline/forced-skill machinery.
    _attach_loop_diagnosis,
    _clear_stale_ready_claims,
    _consume_daily_spawn,
    _count_events,
    _create_audit_followup_task,
    _create_decompose_child_tasks,
    _decompose_children_event,
    _get_council_revise_count,
    _get_daily_spawn_count,
    _get_max_revise_loops,
    _get_sign_off_timeout_hours,
    _get_spawn_frequency_threshold,
    _get_stage_owner,
    _hours_since_last_event,
    _is_profile_spawnable,
    _kanban_setting,
    _kanban_worker_skill_available,
    _maybe_launch_council,
    _missing_worker_forced_skills,
    _block_missing_forced_skills,
    _pin_sticky_reviewers,
    _record_bypass_record,
    _record_council_revise,
    _record_denji_review_signal,
    _record_pipeline_spawn,
    _should_review,
    _validate_pipeline_runtime_state,
    _clear_council_verdict,
    _skill_visible_in_search_dirs,
    _worker_skill_enabled_in_home,
    build_denji_report,
    claim_pipeline_task,
    clear_stale_pipeline_claims,
    complete_pipeline_task,
    get_spawn_frequency,
    validate_forced_skills_visible,
)

# KENSEI CUSTOM (fork re-anchor): id(conn) -> db path registry. Populated by
# kanban_db_connect._sqlite_connect; write_txn resolves the cross-process lock
# file path from it.
_conn_paths: dict[int, str] = {}




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

def get_task_parent(conn: sqlite3.Connection, task_id: str) -> Optional[Task]:
    """Return the containing parent task, or ``None`` when not contained.

    Containment is non-blocking: the returned parent has no effect on the
    child's readiness or dispatch, unlike a dependency edge in ``task_links``.
    """
    row = conn.execute(
        "SELECT p.* FROM tasks p JOIN tasks c ON c.parent_task_id = p.id "
        "WHERE c.id = ?",
        (task_id,),
    ).fetchone()
    return Task.from_row(row) if row else None


def get_subtask_children(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Return the ids of tasks directly contained by ``task_id``.

    These are containment children (``parent_task_id``), NOT dependency
    children (``task_links``). Ordering is stable by id.
    """
    rows = conn.execute(
        "SELECT id FROM tasks WHERE parent_task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return [r["id"] for r in rows]


def set_task_parent(conn: sqlite3.Connection, task_id: str, parent_task_id: Optional[str]) -> bool:
    """Set (or reparent) a task's containment parent. Non-blocking.

    Validates same-board existence and rejects self-parenting and cycles
    (a task cannot be contained by one of its own descendants). Returns
    False when ``task_id`` does not exist.
    """
    parent_task_id = (parent_task_id or "").strip() or None
    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
            return False
        _validate_containment_parent(conn, task_id, parent_task_id)
        conn.execute(
            "UPDATE tasks SET parent_task_id = ? WHERE id = ?",
            (parent_task_id, task_id),
        )
    return True


def clear_task_parent(conn: sqlite3.Connection, task_id: str) -> bool:
    """Clear a task's containment parent. Returns False when task absent."""
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET parent_task_id = NULL WHERE id = ?", (task_id,)
        )
    return cur.rowcount > 0


def _validate_containment_parent(
    conn: sqlite3.Connection, task_id: str, parent_task_id: Optional[str]
) -> None:
    """Fail closed: same-board existence, no self-parent, no containment cycle."""
    if parent_task_id is None:
        return
    if parent_task_id == task_id:
        raise ValueError("a task cannot contain itself")
    if (
        conn.execute("SELECT 1 FROM tasks WHERE id = ?", (parent_task_id,)).fetchone()
        is None
    ):
        raise ValueError(
            f"unknown parent_task_id: {parent_task_id!r} "
            "(containment parent must exist on the same board)"
        )
    # Cycle check: walk containment ancestors of parent_task_id; if we reach
    # task_id, then task_id is already an ancestor of parent_task_id.
    seen: set[str] = set()
    cursor: Optional[str] = parent_task_id
    while cursor is not None and cursor not in seen:
        if cursor == task_id:
            raise ValueError(
                f"setting parent_task_id={parent_task_id!r} on {task_id!r} "
                "would create a containment cycle"
            )
        seen.add(cursor)
        row = conn.execute(
            "SELECT parent_task_id FROM tasks WHERE id = ?", (cursor,)
        ).fetchone()
        cursor = row["parent_task_id"] if row else None
    return


def collate_children(
    conn: sqlite3.Connection, task_id: str,
) -> list[dict]:
    """Return done child results for a parent task (WS-5 collation).

    Returns a list of dicts with keys: id, title, result, assignee.
    Used by integrator/parent tasks to produce a rolled-up deliverable
    from completed child work before calling kanban_complete.
    """
    rows = conn.execute(
        """
        SELECT t.id AS id, t.title AS title, t.result AS result,
               t.assignee AS assignee
        FROM tasks t
        JOIN task_links l ON l.child_id = t.id
        WHERE l.parent_id = ? AND t.status = 'done'
        ORDER BY t.completed_at ASC
        """,
        (task_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"],
            "result": r["result"],
            "assignee": r["assignee"],
        }
        for r in rows
    ]


def _find_task_board(task_id: str) -> Optional[str]:
    """Return the slug of the board ``task_id`` lives on, active board first."""
    active = get_current_board()
    with connect_closing(board=active) as conn:
        if get_task(conn, task_id) is not None:
            return active
    for meta in list_boards():
        slug = meta["slug"]
        if slug == active:
            continue
        with connect_closing(board=slug) as conn:
            if get_task(conn, task_id) is not None:
                return slug
    return None


def connect_for_task(task_id: str):
    """Open a connection scoped to whichever board ``task_id`` actually lives on.

    Task IDs are drawn independently per board (see ``_new_task_id`` —
    ``secrets.token_hex(4)`` with no cross-board collision check, and
    ``test_tasks_do_not_leak_across_boards`` enshrines per-board isolation
    as intentional), so a caller that only ever queries the "active" board
    (``get_current_board()``) can miss a task that lives elsewhere — e.g.
    after ``<root>/kanban/current`` is left pointing at a board switched to
    in an unrelated session. ``hermes feature sign-off/reject/advance``
    hit exactly this: task genuinely exists, active-board pointer is
    stale, lookup returns "Task not found".

    Tries the active board first (cheap, the common case), then falls
    back to scanning ``list_boards()``. Yields ``(conn, task)`` — ``task``
    is ``None``, and ``conn`` is on the active board, if not found on any
    board. Connection is always closed on exit.
    """
    board = _find_task_board(task_id) or get_current_board()
    with connect_closing(board=board) as conn:
        yield conn, get_task(conn, task_id)


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
