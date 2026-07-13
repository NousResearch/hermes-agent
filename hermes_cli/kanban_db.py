"""SQLite-backed Kanban board for multi-profile, multi-project collaboration.

In a fresh install the board lives at ``<root>/kanban.db`` where
``<root>`` is the **shared Hermes root** (the parent of any active
profile). Profiles intentionally collapse onto a shared board: it IS
the cross-profile coordination primitive. A worker spawned with
``hermes -p <profile>`` joins the same board as the dispatcher that
claimed the task. The same applies to ``<root>/kanban/workspaces/`` and
``<root>/kanban/logs/``.

**Multiple boards (projects):** users can create additional boards to
separate unrelated streams of work (e.g. one per project / repo / domain).
Each board is a directory under ``<root>/kanban/boards/<slug>/`` with
its own ``kanban.db``, ``workspaces/``, and ``logs/``. All boards share
the profile's Hermes home but are otherwise isolated: a worker spawned
for a task on board ``atm10-server`` sees only that board's tasks,
cannot enumerate other boards, and its dispatcher ticks don't touch
other boards' DBs.

The first (and for single-project users, only) board is ``default``.
For back-compat its on-disk DB is ``<root>/kanban.db`` (not
``boards/default/kanban.db``), so installs that predate the boards
feature keep working with zero migration. See :func:`kanban_db_path`.

Board resolution order (highest precedence first, all optional):

* ``board=`` argument passed directly to :func:`connect` / :func:`init_db`
  (explicit — used by the CLI ``--board`` flag and the dashboard
  ``?board=...`` query param).
* ``HERMES_KANBAN_BOARD`` env var (used by the dispatcher to pin workers
  to the board their task lives on — workers cannot see other boards).
* ``HERMES_KANBAN_DB`` env var (pins the DB file path directly — legacy
  override still honoured; highest precedence when the file path itself
  is what the caller wants to force).
* ``<root>/kanban/current`` — a one-line text file holding the slug of
  the "currently selected" board. Written by ``hermes kanban boards
  switch <slug>``. When absent, the active board is ``default``.

In standard installs ``<root>`` is ``~/.hermes``. In Docker / custom
deployments where ``HERMES_HOME`` points outside ``~/.hermes`` (e.g.
``/opt/hermes``), ``<root>`` is ``HERMES_HOME``. Legacy env-var
overrides still work:

* ``HERMES_KANBAN_DB`` — pin the database file path directly.
* ``HERMES_KANBAN_WORKSPACES_ROOT`` — pin the workspaces root directly.
* ``HERMES_KANBAN_HOME`` — pin the umbrella root that anchors kanban
  paths. Useful for tests and unusual deployments.

The dispatcher injects ``HERMES_KANBAN_DB``,
``HERMES_KANBAN_WORKSPACES_ROOT``, and ``HERMES_KANBAN_BOARD`` into
worker subprocess env so workers converge on the exact DB the
dispatcher used to claim their task — even under unusual symlink or
Docker layouts.

Schema is intentionally small: tasks, task_links, task_comments,
task_events.  The ``workspace_kind`` field decouples coordination from git
worktrees so that research / ops / digital-twin workloads work alongside
coding workloads.  See ``docs/hermes-kanban-v1-spec.pdf`` for the full
design specification.

Concurrency strategy: WAL mode + ``BEGIN IMMEDIATE`` for write
transactions + compare-and-swap (CAS) updates on ``tasks.status`` and
``tasks.claim_lock``.  SQLite serializes writers via its WAL lock, so at
most one claimer can win any given task.  Losers observe zero affected
rows and move on -- no retry loops, no distributed-lock machinery.
The CAS coordination is **per-board** — each board is a separate DB,
so multi-board installs get the same atomicity guarantees without any
new locking.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import random
import secrets
import shutil
import sqlite3
import subprocess
import sys
import threading
import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_cli.sqlite_util import add_column_if_missing as _add_column_if_missing
from toolsets import get_toolset_names

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"}
VALID_INITIAL_STATUSES = {"running", "blocked"}

# Typed block reasons. Distinguishes the two fundamentally different things a
# worker (or human) means by "blocked", so each can be routed differently
# instead of all landing in one undifferentiated ``blocked`` bucket that a cron
# unblocks → worker re-blocks → cron unblocks … forever.
#
#   * ``dependency``   — can't proceed until another task finishes. Routed to
#                        ``todo`` (NOT ``blocked``) so the existing
#                        parent-gating / ``recompute_ready`` machinery promotes
#                        it automatically once parents are done. No human, no
#                        cron, no retry storm.
#   * ``needs_input``  — needs a human decision/answer it cannot derive.
#   * ``capability``   — hit a hard wall (no access, missing creds, an action no
#                        AI agent can perform). Genuinely human-only.
#   * ``transient``    — a flaky/temporary failure that may clear on retry.
#
# ``needs_input`` and ``capability`` are "truly blocked": they go to ``blocked``
# for a human, and the unblock-loop breaker (see ``block_task`` /
# ``BLOCK_RECURRENCE_LIMIT``) escalates them to ``triage`` if a cron keeps
# unblocking them only to have the worker re-block for the same reason.
# ``None`` = legacy/un-typed block (treated as a generic human blocker).
VALID_BLOCK_KINDS = {"dependency", "needs_input", "capability", "transient"}

# PASS-loop persistence states for the review/completion convergence guard.
#
# ``tracking`` = the task has seen at least one counted PASS-loop cycle and the
# durable fingerprint/evidence payload is being carried forward.
# ``halted`` = the threshold was reached and automation should stay stopped
# until a human changes the proof surface or otherwise intervenes.
# ``clear`` = loop memory was deliberately reset after meaningful progress, but
# the last reset reason / evidence can still remain in the JSON snapshot for
# auditability.
VALID_PASS_LOOP_STATUSES = {"clear", "tracking", "halted"}
PASS_LOOP_REASON_CODE = "pass-loop-detected"
PASS_LOOP_DEFAULT_THRESHOLD = 2

# After a task has been blocked, unblocked, and re-blocked this many times for
# the same (truly-blocked) reason, the unblock-loop breaker stops trusting the
# unblocker (usually a cron) and routes the task to ``triage`` instead of back
# to ``blocked`` — breaking the infinite unblock↔re-block loop and forcing a
# human-in-the-loop decision. Mirrors the dispatcher's ``DEFAULT_FAILURE_LIMIT``
# spirit (default 2) but counts a different signal: manual unblock recurrences,
# not dispatcher spawn/crash/timeout failures.
BLOCK_RECURRENCE_LIMIT = 2
VALID_WORKSPACE_KINDS = {"scratch", "worktree", "dir"}
KNOWN_TOOLSET_NAMES = frozenset(name.casefold() for name in get_toolset_names())
_IS_WINDOWS = sys.platform == "win32"


def _fire_kanban_lifecycle_hook(event: str, task_id: str, **fields: Any) -> None:
    """Fire a kanban lifecycle plugin hook, fully best-effort.

    Called by the claim/complete/block transitions AFTER their write txn has
    committed, so plugin code never runs while a SQLite write lock is held and
    always observes durable board state. Any failure (plugins unavailable,
    a plugin raising, import error) is swallowed — a misbehaving observer must
    never break a board state transition.

    ``profile_name`` is resolved from the active HERMES_HOME so dispatcher- and
    worker-side hooks both carry the right profile without the caller plumbing
    it through.
    """
    try:
        from hermes_cli.plugins import invoke_hook
        from hermes_cli.profiles import get_active_profile_name
        try:
            profile_name = get_active_profile_name()
        except Exception:
            profile_name = "default"
        invoke_hook(event, task_id=task_id, profile_name=profile_name, **fields)
    except Exception as exc:  # pragma: no cover - defensive
        _log.debug("kanban lifecycle hook %s failed: %s", event, exc)


# A running task's claim is valid for 15 minutes by default; after that the
# next dispatcher tick reclaims it. Workers that outlive this window should
# call ``heartbeat_claim(task_id)`` periodically. In practice most kanban
# workloads either finish within 15m, set a longer claim explicitly, or use
# ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` to raise the default claim window for
# long single-call MCP workflows.
DEFAULT_CLAIM_TTL_SECONDS = 15 * 60

# If a worker's PID is still alive but its ``last_heartbeat_at`` is
# older than this when ``release_stale_claims`` runs, treat the worker
# as wedged and reclaim regardless of PID liveness (#29747 gap 3).
# This catches the logic-loop case where the process is technically
# running but not making observable progress.  ``_touch_activity``
# bridges chunk-level liveness into ``last_heartbeat_at`` via #31752,
# so any genuinely active worker keeps its heartbeat fresh as a side
# effect of normal API traffic.
DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS = 60 * 60

# Grace added to a claim when a reclaim is deferred because the previous
# host-local worker is still alive after a termination attempt. Releasing the
# claim in that state would spawn a duplicate alongside the surviving worker —
# the runaway seen when a cgroup memory.high throttle parks a worker in
# uninterruptible (D) state, where a pending SIGKILL cannot be delivered until
# the throttle lifts. Holding the claim a short grace and retrying next tick
# stops the duplication; once no duplicate is spawned the pressure eases, the
# signal lands, and the following tick reclaims cleanly.
RECLAIM_DEFER_GRACE_SECONDS = 120


def _resolve_claim_ttl_seconds(ttl_seconds: Optional[int] = None) -> int:
    """Return the effective claim TTL, honoring the kanban env override.

    Explicit call-site values win. Otherwise a positive integer from
    ``HERMES_KANBAN_CLAIM_TTL_SECONDS`` overrides the built-in default.
    Invalid or non-positive env values fall back silently so existing
    installs keep working.
    """
    if ttl_seconds is not None:
        return max(1, int(ttl_seconds))

    raw = os.environ.get("HERMES_KANBAN_CLAIM_TTL_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed

    return DEFAULT_CLAIM_TTL_SECONDS


def live_worker_workspace_snapshot(task) -> dict[str, str]:
    """Return live workspace overrides for a worker task row.

    The dispatcher writes the intended workspace state into the DB, but for the
    currently running worker we prefer the live checkout on disk as the source
    of truth. We only expose live metadata for actual worktree tasks so scratch
    and dir workspaces never get an invented branch name -- EXCEPT for the
    worker's own currently-running task (``HERMES_KANBAN_TASK`` matches the row
    id), where the live checkout on disk is authoritative even if the persisted
    ``workspace_kind`` is stale/scratch. The git-worktree probe below still
    gates any branch name on the workspace actually being a linked worktree, so
    a non-worktree active workspace still returns no overrides.
    """
    is_active_task = (
        os.environ.get("HERMES_KANBAN_TASK", "").strip() == getattr(task, "id", None)
    )
    if not is_active_task and getattr(task, "workspace_kind", None) != "worktree":
        return {}

    workspace = os.environ.get("HERMES_KANBAN_WORKSPACE", "").strip()
    if not workspace:
        return {}

    path = Path(workspace).expanduser()
    git_dir = path / ".git"
    if not git_dir.is_file():
        return {}
    try:
        if not git_dir.read_text(encoding="utf-8").startswith("gitdir:"):
            return {}
    except OSError:
        return {}

    result = subprocess.run(
        ["git", "-C", str(path), "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return {}
    branch_name = (result.stdout or "").strip()
    if not branch_name:
        return {}

    try:
        resolved = str(path.resolve(strict=False))
    except OSError:
        resolved = str(path)
    return {
        "workspace_kind": "worktree",
        "workspace_path": resolved,
        "branch_name": branch_name,
    }


# Grace period after a task transitions to ``running`` during which
# ``detect_crashed_workers`` skips the ``_pid_alive`` check. Covers the
# fork() → /proc-visibility window where liveness can transiently report
# False for a freshly-spawned worker. The 15-minute claim TTL still
# catches genuinely-crashed workers; this only suppresses false positives
# during the launch window.
DEFAULT_CRASH_GRACE_SECONDS = 30


# Sentinel exit code a kanban worker uses to signal "I bailed because the
# provider rate-limited / exhausted quota, not because the task failed."
# The dispatcher's reap classifier maps this to a ``rate_limited`` exit kind
# so ``detect_crashed_workers`` can release the task back to ``ready``
# WITHOUT counting a failure (the circuit breaker must never trip on a
# transient throttle). 75 == BSD ``EX_TEMPFAIL`` (sysexits.h) — the
# conventional "temporary failure, retry later" code, and well clear of the
# 0/1/2 codes the worker uses for success / generic failure / usage error.
KANBAN_RATE_LIMIT_EXIT_CODE = 75


def _resolve_crash_grace_seconds() -> int:
    """Return the crash-detection grace period in seconds.

    Reads ``HERMES_KANBAN_CRASH_GRACE_SECONDS`` from the environment;
    falls back to ``DEFAULT_CRASH_GRACE_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 restores immediate-reclaim
    behaviour (useful for tests).
    """
    raw = os.environ.get("HERMES_KANBAN_CRASH_GRACE_SECONDS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_CRASH_GRACE_SECONDS


def _resolve_rate_limit_cooldown_seconds() -> int:
    """Return the rate-limit requeue cooldown in seconds.

    Reads ``HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS`` from the environment;
    falls back to ``DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS`` when absent, empty,
    non-integer, or negative. A value of 0 disables the cooldown (re-spawn on
    the next tick) — useful for tests that want to assert the task becomes
    spawnable again immediately.
    """
    raw = os.environ.get(
        "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", ""
    ).strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = -1
        if parsed >= 0:
            return parsed
    return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS


# Worker-context caps so build_worker_context() stays bounded on
# pathological boards (retry-heavy tasks, comment storms, giant
# summaries). Values chosen to fit a typical 100k-char LLM prompt with
# plenty of headroom. Each constant is tuned independently so users
# who need to relax one don't have to relax all of them.
_CTX_MAX_PRIOR_ATTEMPTS = 10      # most recent N prior runs shown in full
_CTX_MAX_COMMENTS       = 30      # most recent N comments shown in full
_CTX_MAX_FIELD_BYTES    = 4 * 1024   # 4 KB per summary/error/metadata/result
_CTX_MAX_BODY_BYTES     = 8 * 1024   # 8 KB per task.body (opening post)
_CTX_MAX_COMMENT_BYTES  = 2 * 1024   # 2 KB per comment


def _relative_age(ts: Optional[int], now: Optional[int] = None) -> str:
    """Render the age of an epoch-seconds timestamp as a coarse, human-
    readable string like ``just now``, ``18h ago``, ``3d ago``.

    Workers read parent handoffs, comments, and prior-attempt summaries as
    if they describe *current* state. A bare absolute timestamp
    (``2026-06-25 14:30``) doesn't make an LLM reason about staleness — it
    reads the content as fact regardless of how old it is. A relative age
    ("18h ago") is the signal that prompts the worker to re-verify against
    the live source before acting on stale sibling work. Returns an empty
    string for missing/invalid timestamps so callers can append
    unconditionally.
    """
    if ts is None:
        return ""
    try:
        ts = int(ts)
    except (TypeError, ValueError):
        return ""
    if now is None:
        now = int(time.time())
    delta = now - ts
    if delta < 0:
        # Clock skew across machines/profiles — don't claim "in the future".
        return "just now"
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = delta // 60
        return f"{m}m ago"
    if delta < 86400:
        h = delta // 3600
        return f"{h}h ago"
    d = delta // 86400
    return f"{d}d ago"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_BOARD = "default"
_CURRENT_BOARD_OVERRIDE: ContextVar[str | None] = ContextVar(
    "hermes_kanban_current_board_override",
    default=None,
)


@contextlib.contextmanager
def scoped_current_board(slug: str):
    """Temporarily pin the active board for the current context only."""
    token: Token[str | None] = _CURRENT_BOARD_OVERRIDE.set(slug)
    try:
        yield
    finally:
        _CURRENT_BOARD_OVERRIDE.reset(token)

# Slug validator: lowercase alphanumerics, digits, hyphens; 1–64 chars.
# Strict enough to stop traversal (`..`) and embedded path separators, loose
# enough that kebab-case names like ``atm10-server`` or ``hermes-agent``
# pass without fuss. Board names with display formatting (spaces, emoji)
# live in ``board.json``; the slug is just the directory name.
_BOARD_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,63}$")


def _normalize_board_slug(slug: Optional[str]) -> Optional[str]:
    """Lowercase + strip a slug; validate; return ``None`` for empty."""
    if slug is None:
        return None
    s = str(slug).strip().lower()
    if not s:
        return None
    if not _BOARD_SLUG_RE.match(s):
        raise ValueError(
            f"invalid board slug {slug!r}: must be 1-64 chars, lowercase "
            f"alphanumerics / hyphens / underscores, not starting with '-' or '_'"
        )
    return s


def kanban_home() -> Path:
    """Return the shared Hermes root that anchors the kanban board.

    Resolution order:

    1. ``HERMES_KANBAN_HOME`` env var when set and non-empty (explicit
       override for tests and unusual deployments).
    2. ``get_default_hermes_root()``, which already returns ``<root>``
       when ``HERMES_HOME`` is ``<root>/profiles/<name>``, and returns
       ``HERMES_HOME`` directly for Docker / custom deployments.

    The kanban board is shared across profiles **by design** (see the
    module docstring). Resolving the kanban paths through the active
    profile's ``HERMES_HOME`` would silently fork the board per profile,
    which breaks the dispatcher / worker handoff.
    """
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def boards_root() -> Path:
    """Return ``<root>/kanban/boards`` — the parent of non-default board dirs.

    ``default`` is intentionally NOT under this directory — its DB lives at
    ``<root>/kanban.db`` for back-compat with pre-boards installs. This
    function returns the directory where *additional* named boards live,
    used by :func:`list_boards` to enumerate them.
    """
    return kanban_home() / "kanban" / "boards"


def current_board_path() -> Path:
    """Return the path to ``<root>/kanban/current``.

    One-line text file written by ``hermes kanban boards switch <slug>``
    to persist the user's board selection across CLI invocations. Absent
    by default (meaning: active board is ``default``).
    """
    return kanban_home() / "kanban" / "current"


def get_current_board() -> str:
    """Return the active board slug, honouring the resolution chain.

    Order (highest precedence first):

    1. ``HERMES_KANBAN_BOARD`` env var (set by the dispatcher on worker
       spawn, or manually for ad-hoc overrides).
    2. ``<root>/kanban/current`` on disk (set by ``hermes kanban boards
       switch``), but only when that board still exists.
    3. ``DEFAULT_BOARD`` (``"default"``).

    A malformed or stale slug at any step falls through to the next layer
    with a best-effort warning — the dispatcher must never crash because a
    user hand-edited a file or removed a board directory.
    """
    scoped = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    if scoped:
        try:
            normed = _normalize_board_slug(scoped)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass

    env = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    if env:
        try:
            normed = _normalize_board_slug(env)
            if normed and board_exists(normed):
                return normed
        except ValueError:
            pass
    try:
        f = current_board_path()
        if f.exists():
            val = f.read_text(encoding="utf-8").strip()
            if val:
                try:
                    normed = _normalize_board_slug(val)
                    if normed and board_exists(normed):
                        return normed
                except ValueError:
                    pass
    except OSError:
        pass
    return DEFAULT_BOARD


def set_current_board(slug: str) -> Path:
    """Persist ``slug`` as the active board. Returns the file written.

    Writes ``<root>/kanban/current``. The caller should validate the slug
    exists first (via :func:`board_exists`) — this function does not —
    so that ``hermes kanban boards switch <typo>`` returns an error
    instead of silently pointing at nothing.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    path = current_board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normed + "\n", encoding="utf-8")
    return path


def clear_current_board() -> None:
    """Remove ``<root>/kanban/current`` so the active board reverts to ``default``."""
    try:
        current_board_path().unlink()
    except FileNotFoundError:
        pass


def board_dir(board: Optional[str] = None) -> Path:
    """Return the on-disk directory for ``board``.

    ``default`` is ``<root>/kanban/boards/default/`` **for metadata only**
    (board.json + workspaces/ + logs/). Its DB file stays at
    ``<root>/kanban.db`` for back-compat — see :func:`kanban_db_path`.

    All other boards live at ``<root>/kanban/boards/<slug>/`` with
    everything inside that directory including the ``kanban.db``.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return boards_root() / slug


def board_exists(board: Optional[str] = None) -> bool:
    """Return True if the board has persisted metadata or a DB on disk.

    ``default`` is considered to always exist — its DB is created
    on first :func:`connect` and there's no way for it to be missing
    in a configuration where the kanban feature is usable at all.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    if slug == DEFAULT_BOARD:
        return True
    d = board_dir(slug)
    return (d / "board.json").exists() or (d / "kanban.db").exists()


def _scoped_board_override_slug() -> Optional[str]:
    """Return the normalized in-process board override, if one is active.

    ``scoped_current_board(...)`` is the CLI's first-class explicit-board
    override path. When present it must outrank ambient worker DB/workspace
    pins, otherwise ``hermes kanban --board <slug> ...`` can open the wrong
    sqlite file even though the command is explicitly scoped.
    """
    scoped = (_CURRENT_BOARD_OVERRIDE.get() or "").strip()
    if not scoped:
        return None
    try:
        return _normalize_board_slug(scoped)
    except ValueError:
        return None


def kanban_db_path(board: Optional[str] = None) -> Path:
    """Return the path to the ``kanban.db`` for ``board``.

    Resolution (highest precedence first):

    1. Explicit board intent — either the ``board`` arg or an active
       ``scoped_current_board(...)`` override.
    2. ``HERMES_KANBAN_DB`` env var — pins the path directly when there is no
       explicit board override. Honoured for back-compat and for the
       dispatcher→worker handoff (defense in depth: dispatcher injects this
       into worker env so workers are immune to any path-resolution
       disagreement).
    3. When neither explicit-board path above is present, the active board
       from :func:`get_current_board` is used.
    4. Board ``default`` → ``<root>/kanban.db`` (back-compat path).
       Other boards → ``<root>/kanban/boards/<slug>/kanban.db``.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = _scoped_board_override_slug()
    if slug is None:
        override = os.environ.get("HERMES_KANBAN_DB", "").strip()
        if override:
            return Path(override).expanduser()
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban.db"
    return board_dir(slug) / "kanban.db"


def workspaces_root(board: Optional[str] = None) -> Path:
    """Return the directory under which ``scratch`` workspaces are created.

    Anchored per-board so workspaces don't leak between projects.
    Explicit board intent (``board=`` or ``scoped_current_board(...)``)
    outranks ambient worker pins so ``--board <slug>`` commands stay on their
    requested board. Without an explicit board,
    ``HERMES_KANBAN_WORKSPACES_ROOT`` still pins the path directly — the
    dispatcher injects this into worker env.

    ``default`` keeps the legacy path ``<root>/kanban/workspaces/`` so
    that existing scratch workspaces from before the boards feature are
    preserved. Other boards use ``<root>/kanban/boards/<slug>/workspaces/``.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = _scoped_board_override_slug()
    if slug is None:
        override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
        if override:
            return Path(override).expanduser()
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "workspaces"
    return board_dir(slug) / "workspaces"


def attachments_root(board: Optional[str] = None) -> Path:
    """Return the directory under which task file attachments are stored.

    Mirrors :func:`worker_logs_dir` / :func:`workspaces_root`: anchored
    per-board so attachments don't leak between projects. Each task gets
    its own ``<root>/.../attachments/<task_id>/`` subdirectory.

    ``HERMES_KANBAN_ATTACHMENTS_ROOT`` pins the path directly (highest
    precedence) for tests and unusual deployments.

    ``default`` uses ``<root>/kanban/attachments/``; other boards use
    ``<root>/kanban/boards/<slug>/attachments/``.

    Workers (which run with full file-tool access) read attached files
    by the absolute path surfaced in :func:`build_worker_context`. On the
    local terminal backend — the default for kanban — that path resolves
    directly. Remote backends (Docker/Modal) need this directory mounted;
    see the kanban docs.
    """
    override = os.environ.get("HERMES_KANBAN_ATTACHMENTS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "attachments"
    return board_dir(slug) / "attachments"


def task_attachments_dir(task_id: str, board: Optional[str] = None) -> Path:
    """Return the per-task attachment directory ``<root>/<task_id>/``."""
    return attachments_root(board=board) / task_id


def worker_logs_dir(board: Optional[str] = None) -> Path:
    """Return the directory under which per-task worker logs are written.

    ``default`` keeps the legacy path ``<root>/kanban/logs/``. Other
    boards use ``<root>/kanban/boards/<slug>/logs/``. Logs follow the
    board — makes ``hermes kanban log`` unambiguous even when multiple
    boards have tasks with the same id.
    """
    slug = _normalize_board_slug(board)
    if slug is None:
        slug = get_current_board()
    if slug == DEFAULT_BOARD:
        return kanban_home() / "kanban" / "logs"
    return board_dir(slug) / "logs"


def board_metadata_path(board: Optional[str] = None) -> Path:
    """Return the path to ``board.json`` for ``board``.

    Stores display metadata (display name, description, icon, color,
    created_at). The on-disk slug is the canonical identity; this file
    is purely for presentation in the CLI / dashboard.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    return board_dir(slug) / "board.json"


def _default_board_display_name(slug: str) -> str:
    """Turn a slug into a reasonable default display name.

    ``atm10-server`` → ``Atm10 Server``. Users can override via
    ``board.json`` but the default should look presentable in the
    dashboard without any follow-up editing.
    """
    return " ".join(part.capitalize() for part in slug.replace("_", "-").split("-") if part) or slug


def read_board_metadata(board: Optional[str] = None) -> dict:
    """Return ``board.json`` contents (or synthesized defaults).

    Never raises — a missing / malformed ``board.json`` falls back to a
    synthesised entry so the dashboard always has something to render.
    Includes the canonical ``slug`` and ``db_path`` so the caller
    doesn't need to reconstruct them.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta: dict[str, Any] = {
        "slug": slug,
        "name": _default_board_display_name(slug),
        "description": "",
        "icon": "",
        "color": "",
        "default_workdir": None,
        "worktree_base_ref": None,
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
    board: Optional[str],
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    archived: Optional[bool] = None,
    default_workdir: Optional[str] = None,
    worktree_base_ref: Optional[str] = None,
) -> dict:
    """Create / update ``board.json`` for ``board``.

    Preserves any existing fields not mentioned in the call. Sets
    ``created_at`` on first write. Returns the resulting metadata dict.
    """
    slug = _normalize_board_slug(board) or DEFAULT_BOARD
    meta = read_board_metadata(slug)
    # Preserve existing DB-derived fields — they get re-computed each
    # read but shouldn't be written into board.json.
    meta.pop("db_path", None)
    if name is not None:
        meta["name"] = str(name).strip() or _default_board_display_name(slug)
    if description is not None:
        meta["description"] = str(description)
    if icon is not None:
        meta["icon"] = str(icon)
    if color is not None:
        meta["color"] = str(color)
    if archived is not None:
        meta["archived"] = bool(archived)
    if default_workdir is not None:
        meta["default_workdir"] = str(default_workdir) if default_workdir else None
    if worktree_base_ref is not None:
        meta["worktree_base_ref"] = str(worktree_base_ref) if worktree_base_ref else None
    if not meta.get("created_at"):
        meta["created_at"] = int(time.time())
    path = board_metadata_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    meta["db_path"] = str(kanban_db_path(slug))
    return meta


def create_board(
    slug: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    icon: Optional[str] = None,
    color: Optional[str] = None,
    default_workdir: Optional[str] = None,
    worktree_base_ref: Optional[str] = None,
) -> dict:
    """Create a new board directory + DB + metadata. Idempotent.

    Returns the resulting metadata. Raises :class:`ValueError` for a
    malformed slug; returns the existing metadata (not an error) if the
    board already exists — matching ``mkdir -p`` semantics.
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    meta = write_board_metadata(
        normed,
        name=name,
        description=description,
        icon=icon,
        color=color,
        default_workdir=default_workdir,
        worktree_base_ref=worktree_base_ref,
    )
    # Touch the DB so list_boards() sees it immediately.
    init_db(board=normed)
    return meta


def list_boards(*, include_archived: bool = True) -> list[dict]:
    """Enumerate all boards that exist on disk.

    Always includes ``default`` (even when the ``boards/default/``
    metadata dir doesn't exist, because its DB is at the legacy path).
    Other boards are discovered by scanning ``boards/`` for subdirectories
    that either contain a ``kanban.db`` or a ``board.json``.

    Returns a list of metadata dicts, sorted with ``default`` first and
    the rest alphabetically.
    """
    entries: list[dict] = []
    seen: set[str] = set()

    # Default board is always first.
    entries.append(read_board_metadata(DEFAULT_BOARD))
    seen.add(DEFAULT_BOARD)

    root = boards_root()
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not child.is_dir():
                continue
            slug = child.name
            # Keep slug normalisation soft for discovery — but skip dirs
            # that don't parse as valid slugs so we don't surface junk.
            try:
                normed = _normalize_board_slug(slug)
            except ValueError:
                continue
            if not normed or normed in seen:
                continue
            has_db = (child / "kanban.db").exists()
            has_meta = (child / "board.json").exists()
            if not (has_db or has_meta):
                continue
            meta = read_board_metadata(normed)
            if meta.get("archived") and not include_archived:
                continue
            entries.append(meta)
            seen.add(normed)
    return entries


def remove_board(slug: str, *, archive: bool = True) -> dict:
    """Remove or archive a board.

    ``archive=True`` (default) moves the board's directory to
    ``<root>/kanban/boards/_archived/<slug>-<timestamp>/`` so the data
    is recoverable. ``archive=False`` deletes the directory outright.

    The ``default`` board cannot be removed — raises :class:`ValueError`.
    Returns a summary dict describing what happened (``{"slug", "action",
    "new_path"}``).
    """
    normed = _normalize_board_slug(slug)
    if not normed:
        raise ValueError("board slug is required")
    if normed == DEFAULT_BOARD:
        raise ValueError("the 'default' board cannot be removed")
    d = board_dir(normed)
    if not d.exists():
        raise ValueError(f"board {normed!r} does not exist")

    # If the user removed the currently-active board, revert to default.
    if get_current_board() == normed:
        clear_current_board()

    # A concurrent connect(board=normed) after the rename/delete recreates
    # an empty sqlite file via mkdir(exist_ok=True); the cache entry must be
    # dropped first so the schema init pass re-runs on that fresh file.
    _INITIALIZED_PATHS.discard(str((d / "kanban.db").resolve()))

    if archive:
        archive_root = boards_root() / "_archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        target = archive_root / f"{normed}-{ts}"
        # Avoid collision on rapid double-archives.
        suffix = 1
        while target.exists():
            target = archive_root / f"{normed}-{ts}-{suffix}"
            suffix += 1
        d.rename(target)
        return {"slug": normed, "action": "archived", "new_path": str(target)}
    else:
        import shutil
        shutil.rmtree(d)
        return {"slug": normed, "action": "deleted", "new_path": ""}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

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
    brand: Optional[str] = None
    branch_name: Optional[str] = None
    project_id: Optional[str] = None
    workspace_base_ref: Optional[str] = None
    workspace_base_commit: Optional[str] = None
    result: Optional[str] = None
    delivery_state: Optional[dict[str, Any]] = None
    pass_loop_state: Optional[dict[str, Any]] = None
    pass_loop_status: Optional[str] = None
    pass_loop_count: int = 0
    pass_loop_reason_code: Optional[str] = None
    idempotency_key: Optional[str] = None
    # Unified non-success counter. Incremented on any of:
    #   * spawn failure (dispatcher couldn't launch the worker)
    #   * timed_out outcome (worker exceeded max_runtime_seconds)
    #   * crashed outcome (worker PID vanished)
    # Reset to 0 only on a successful completion. See
    # ``_record_task_failure`` for the circuit-breaker trip rule.
    # (Pre-rename column: ``spawn_failures``.)
    consecutive_failures: int = 0
    worker_pid: Optional[int] = None
    # Short excerpt of the last failure's error text (any outcome, not
    # just spawn). Pre-rename column: ``last_spawn_error``.
    last_failure_error: Optional[str] = None
    max_runtime_seconds: Optional[int] = None
    last_heartbeat_at: Optional[int] = None
    current_run_id: Optional[int] = None
    workflow_template_id: Optional[str] = None
    current_step_key: Optional[str] = None
    # Force-loaded skills for the worker on this task (passed via
    # --skills). Stored as a JSON array of skill names. None = use only
    # the defaults; empty list = explicitly no extra skills.
    skills: Optional[list] = None
    model_override: Optional[str] = None
    # Per-task override for the consecutive-failure circuit breaker.
    # The value is the failure count at which the breaker trips — e.g.
    # ``max_retries=1`` blocks on the first failure (zero retries),
    # ``max_retries=3`` blocks on the third (two retries allowed).
    # ``None`` (the common case) falls through to the dispatcher-level
    # ``kanban.failure_limit`` config, and then to ``DEFAULT_FAILURE_LIMIT``.
    # Name matches the ``--max-retries`` CLI flag on ``kanban create``.
    max_retries: Optional[int] = None
    # When True, the dispatched worker runs in a Ralph-style goal loop
    # (the same engine behind the ``/goal`` slash command): after each
    # turn an auxiliary judge model evaluates the worker's response
    # against this card's title/body (treated as the goal). If the judge
    # says "not done" and budget remains, the worker is fed a
    # continuation prompt IN THE SAME SESSION and keeps working until the
    # judge agrees, the goal-turn budget is exhausted (→ kanban_block),
    # or the worker explicitly blocks/completes. ``False`` (default) =
    # the classic single-shot worker. ``goal_max_turns`` bounds the loop.
    goal_mode: bool = False
    # Goal-loop turn budget for ``goal_mode`` workers. ``None`` falls
    # through to the goals engine default (``goals.DEFAULT_MAX_TURNS``).
    goal_max_turns: Optional[int] = None
    # Originating chat/agent session id, when the task was created from
    # within an agent loop that propagated ``HERMES_SESSION_ID``. NULL for
    # tasks created from the CLI, the dashboard, or any path that doesn't
    # set the env var. Lets clients render a per-session board without
    # relying on tenant + time-window heuristics.
    session_id: Optional[str] = None
    # True when the workspace kind was explicitly chosen at creation
    # (e.g. ``--workspace scratch``). Pinned workspaces are never
    # auto-upgraded to a worktree at create or dispatch time.
    workspace_pinned: bool = False
    # Typed block reason (one of VALID_BLOCK_KINDS) or None for legacy/un-typed
    # blocks. Set by ``block_task``; preserved across unblock so a re-block for
    # the same kind is recognisable as an unblock↔re-block loop.
    block_kind: Optional[str] = None
    # Unblock-loop counter. See the column comment in SCHEMA_SQL and
    # ``BLOCK_RECURRENCE_LIMIT``. Reset only on successful completion.
    block_recurrences: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        keys = set(row.keys())
        # Parse skills JSON blob if present
        skills_value: Optional[list] = None
        if "skills" in keys and row["skills"]:
            try:
                parsed = json.loads(row["skills"])
                if isinstance(parsed, list):
                    skills_value = [str(s) for s in parsed if s]
            except Exception:
                skills_value = None
        delivery_state_value: Optional[dict[str, Any]] = None
        if "delivery_state" in keys and row["delivery_state"]:
            try:
                parsed_delivery_state = json.loads(row["delivery_state"])
                if isinstance(parsed_delivery_state, dict):
                    delivery_state_value = parsed_delivery_state
            except Exception:
                delivery_state_value = None
        pass_loop_state_value: Optional[dict[str, Any]] = None
        if "pass_loop_state" in keys and row["pass_loop_state"]:
            try:
                parsed_pass_loop_state = json.loads(row["pass_loop_state"])
                if isinstance(parsed_pass_loop_state, dict):
                    pass_loop_state_value = parsed_pass_loop_state
            except Exception:
                pass_loop_state_value = None
        return cls(
            id=row["id"],
            title=row["title"],
            body=row["body"],
            assignee=row["assignee"],
            status=row["status"],
            priority=row["priority"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            workspace_kind=row["workspace_kind"],
            workspace_path=row["workspace_path"],
            branch_name=row["branch_name"] if "branch_name" in keys else None,
            project_id=row["project_id"] if "project_id" in keys else None,
            workspace_base_ref=(
                row["workspace_base_ref"] if "workspace_base_ref" in keys else None
            ),
            workspace_base_commit=(
                row["workspace_base_commit"] if "workspace_base_commit" in keys else None
            ),
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            tenant=row["tenant"] if "tenant" in keys else None,
            brand=row["brand"] if "brand" in keys else None,
            result=row["result"] if "result" in keys else None,
            delivery_state=delivery_state_value,
            pass_loop_state=pass_loop_state_value,
            pass_loop_status=(
                row["pass_loop_status"]
                if "pass_loop_status" in keys and row["pass_loop_status"]
                else None
            ),
            pass_loop_count=(
                int(row["pass_loop_count"])
                if "pass_loop_count" in keys and row["pass_loop_count"] is not None
                else 0
            ),
            pass_loop_reason_code=(
                row["pass_loop_reason_code"]
                if "pass_loop_reason_code" in keys and row["pass_loop_reason_code"]
                else None
            ),
            idempotency_key=row["idempotency_key"] if "idempotency_key" in keys else None,
            consecutive_failures=(
                row["consecutive_failures"] if "consecutive_failures" in keys
                # Pre-migration fallback: ``_migrate_add_optional_columns`` always
                # adds ``consecutive_failures`` now, so this branch is only reachable
                # on a DB that was never opened since pre-#20410 code ran. Keep for
                # belt-and-suspenders safety; in practice it is dead code post-migration.
                else (row["spawn_failures"] if "spawn_failures" in keys else 0)
            ),
            worker_pid=row["worker_pid"] if "worker_pid" in keys else None,
            last_failure_error=(
                row["last_failure_error"] if "last_failure_error" in keys
                # Same belt-and-suspenders fallback as consecutive_failures above.
                else (row["last_spawn_error"] if "last_spawn_error" in keys else None)
            ),
            max_runtime_seconds=(
                row["max_runtime_seconds"] if "max_runtime_seconds" in keys else None
            ),
            last_heartbeat_at=(
                row["last_heartbeat_at"] if "last_heartbeat_at" in keys else None
            ),
            current_run_id=(
                row["current_run_id"] if "current_run_id" in keys else None
            ),
            workflow_template_id=(
                row["workflow_template_id"] if "workflow_template_id" in keys else None
            ),
            current_step_key=(
                row["current_step_key"] if "current_step_key" in keys else None
            ),
            skills=skills_value,
            model_override=row["model_override"] if "model_override" in keys and row["model_override"] else None,
            max_retries=(
                row["max_retries"] if "max_retries" in keys else None
            ),
            goal_mode=(
                bool(row["goal_mode"]) if "goal_mode" in keys and row["goal_mode"] else False
            ),
            goal_max_turns=(
                row["goal_max_turns"] if "goal_max_turns" in keys and row["goal_max_turns"] else None
            ),
            session_id=(
                row["session_id"] if "session_id" in keys else None
            ),
            workspace_pinned=(
                bool(row["workspace_pinned"])
                if "workspace_pinned" in keys and row["workspace_pinned"]
                else False
            ),
            block_kind=(
                row["block_kind"] if "block_kind" in keys and row["block_kind"] else None
            ),
            block_recurrences=(
                int(row["block_recurrences"])
                if "block_recurrences" in keys and row["block_recurrences"] is not None
                else 0
            ),
        )


@dataclass
class Run:
    """In-memory view of a ``task_runs`` row.

    A run is one attempt to execute a task — created on claim, closed
    on complete/block/crash/timeout/spawn_failure/reclaim. Multiple runs
    per task when retries happen. Carries the claim machinery, PID,
    heartbeat, and the structured handoff summary that downstream workers
    read via ``build_worker_context``.
    """

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
        try:
            meta = json.loads(row["metadata"]) if row["metadata"] else None
        except Exception:
            meta = None
        return cls(
            id=int(row["id"]),
            task_id=row["task_id"],
            profile=row["profile"],
            step_key=row["step_key"],
            status=row["status"],
            claim_lock=row["claim_lock"],
            claim_expires=row["claim_expires"],
            worker_pid=row["worker_pid"],
            max_runtime_seconds=row["max_runtime_seconds"],
            last_heartbeat_at=row["last_heartbeat_at"],
            started_at=int(row["started_at"]),
            ended_at=(int(row["ended_at"]) if row["ended_at"] is not None else None),
            outcome=row["outcome"],
            summary=row["summary"],
            metadata=meta,
            error=row["error"],
        )


@dataclass
class Comment:
    id: int
    task_id: str
    author: str
    body: str
    created_at: int


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


@dataclass
class Event:
    id: int
    task_id: str
    kind: str
    payload: Optional[dict]
    created_at: int
    run_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Delivery-state helpers (pilot: machine-checkable software-delivery truth)
# ---------------------------------------------------------------------------

DELIVERY_REVIEW_GATE_STAGES = {"implementation", "qa"}
DELIVERY_WORKTREE_PROVENANCE_STAGES = {"implementation", "review", "converge"}


def _delivery_ref_path(ref: Optional[dict[str, Any]]) -> Optional[str]:
    if not isinstance(ref, dict):
        return None
    raw = ref.get("path") or ref.get("stored_path")
    if raw is None:
        return None
    path = str(raw).strip()
    return path or None


def delivery_artifact_readable(ref: Optional[dict[str, Any]]) -> bool:
    """Return True when the delivery artifact ref resolves to a readable file.

    The pilot intentionally treats file readability as first-class delivery
    truth. Unsupported or incomplete refs stay false rather than pretending
    delivery is green.
    """

    path = _delivery_ref_path(ref)
    if not path:
        return False
    try:
        return Path(path).expanduser().is_file()
    except OSError:
        return False


def _delivery_workspace_snapshot(task: Task) -> dict[str, Any]:
    return {
        "kind": task.workspace_kind,
        "path": task.workspace_path,
        "branch_name": task.branch_name,
        "base_ref": task.workspace_base_ref,
        "base_commit": task.workspace_base_commit,
    }


def _deep_merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_pass_loop_state(task: Task, state: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized PASS-loop persistence snapshot for ``task``."""

    now = int(time.time())
    normalized = dict(state)
    normalized["schema_version"] = int(normalized.get("schema_version") or 1)
    normalized["task_id"] = task.id
    normalized["task_status"] = task.status
    normalized["assignee_profile"] = task.assignee
    normalized["block_kind"] = task.block_kind

    raw_status = str(normalized.get("status") or "tracking").strip().lower()
    status = raw_status or "tracking"
    if status not in VALID_PASS_LOOP_STATUSES:
        raise ValueError(
            f"pass-loop status must be one of {sorted(VALID_PASS_LOOP_STATUSES)}"
        )
    normalized["status"] = status

    try:
        count = int(normalized.get("count") or 0)
    except (TypeError, ValueError):
        count = 0
    normalized["count"] = max(0, count)

    try:
        threshold = int(
            normalized.get("threshold") or PASS_LOOP_DEFAULT_THRESHOLD
        )
    except (TypeError, ValueError):
        threshold = PASS_LOOP_DEFAULT_THRESHOLD
    normalized["threshold"] = max(PASS_LOOP_DEFAULT_THRESHOLD, threshold)

    reason_code = normalized.get("reason_code")
    if reason_code is not None:
        reason_code = str(reason_code).strip() or None
    if status == "halted" and not reason_code:
        reason_code = PASS_LOOP_REASON_CODE
    normalized["reason_code"] = reason_code

    fingerprint = normalized.get("fingerprint")
    normalized["fingerprint"] = dict(fingerprint) if isinstance(fingerprint, dict) else {}

    evidence = normalized.get("evidence")
    normalized["evidence"] = dict(evidence) if isinstance(evidence, dict) else {}

    resets = normalized.get("resets")
    normalized["resets"] = list(resets) if isinstance(resets, list) else []

    normalized["updated_at"] = now
    normalized.setdefault("first_recorded_at", now)
    return normalized


def derive_delivery_verdict(snapshot: Optional[dict[str, Any]]) -> tuple[str, str]:
    """Reduce a structured delivery-state snapshot into a verdict + reason."""

    if not isinstance(snapshot, dict) or not snapshot:
        return ("unknown", "delivery_state missing")

    required = [
        "stage",
        "workflow_stream_id",
        "artifact",
        "workspace",
        "proof",
        "review",
        "merge",
        "release",
    ]
    missing = [key for key in required if snapshot.get(key) is None]
    if missing:
        return ("unknown", f"missing delivery_state fields: {', '.join(missing)}")

    stage = str(snapshot.get("stage") or "").strip()
    if not stage:
        return ("unknown", "stage missing")

    artifact = snapshot.get("artifact") or {}
    primary_ref = artifact.get("primary_ref")
    if primary_ref is None:
        return ("unknown", "primary artifact missing")
    if artifact.get("readable") is False:
        return ("blocked", "primary artifact unreadable")

    proof = snapshot.get("proof") or {}
    proof_status = str(proof.get("proof_status") or "").strip()
    if proof_status == "failed":
        return ("failed", "proof status failed")

    workspace = snapshot.get("workspace") or {}
    if (
        stage in DELIVERY_WORKTREE_PROVENANCE_STAGES
        and workspace.get("kind") == "worktree"
        and (not workspace.get("branch_name") or not workspace.get("base_ref"))
    ):
        return ("unknown", "worktree delivery stage missing branch/base provenance")

    task_status = str(snapshot.get("task_status") or "").strip()
    if task_status == "blocked":
        return ("blocked", "task is blocked")

    review = snapshot.get("review") or {}
    merge = snapshot.get("merge") or {}
    release = snapshot.get("release") or {}
    review_status = str(review.get("status") or "").strip()
    merge_status = str(merge.get("status") or "").strip()
    release_status = str(release.get("status") or "").strip()

    if release_status == "released":
        return ("released", "release evidence recorded")
    if merge_status == "merged" and release_status not in {"released", "not_applicable"}:
        return ("merged_not_released", "merge verified but release not proven")
    if stage in DELIVERY_REVIEW_GATE_STAGES and review_status not in {"approved", "not_applicable"}:
        return ("needs_review", "stage output awaits explicit review")
    if review_status == "approved" and merge_status not in {"merged", "not_applicable"}:
        return ("verified_not_merged", "review approved but merge not proven")
    if task_status in {"running", "ready", "todo", "scheduled"}:
        return ("in_progress", "delivery evidence still being collected")
    if task_status == "done":
        return ("in_progress", "task completed but downstream delivery truth is still partial")
    return ("unknown", "delivery state incomplete")


def _normalize_delivery_state(task: Task, state: dict[str, Any]) -> dict[str, Any]:
    now = int(time.time())
    normalized = dict(state)
    normalized["schema_version"] = int(normalized.get("schema_version") or 1)
    normalized["task_id"] = task.id
    normalized["task_status"] = task.status
    normalized["assignee_profile"] = task.assignee

    workspace = normalized.get("workspace")
    if isinstance(workspace, dict):
        normalized["workspace"] = _deep_merge_dicts(_delivery_workspace_snapshot(task), workspace)
    else:
        normalized["workspace"] = _delivery_workspace_snapshot(task)

    artifact_raw = normalized.get("artifact")
    artifact: dict[str, Any] = dict(artifact_raw) if isinstance(artifact_raw, dict) else {}
    artifact.setdefault("primary_ref", None)
    refs = artifact.get("refs")
    artifact["refs"] = list(refs) if isinstance(refs, list) else []
    artifact["readable"] = delivery_artifact_readable(artifact.get("primary_ref"))
    artifact["last_checked_at"] = now
    normalized["artifact"] = artifact

    proof_raw = normalized.get("proof")
    proof: dict[str, Any] = dict(proof_raw) if isinstance(proof_raw, dict) else {}
    proof.setdefault("tests_required", {"count": 0, "items": []})
    proof.setdefault("tests_run", {"count": 0, "items": []})
    proof.setdefault("tests_passed", {"count": 0, "items": []})
    proof.setdefault("test_evidence_refs", [])
    proof.setdefault("proof_status", "not_started")
    normalized["proof"] = proof

    review_raw = normalized.get("review")
    review: dict[str, Any] = dict(review_raw) if isinstance(review_raw, dict) else {}
    review.setdefault("status", "not_requested")
    review.setdefault("reviewer_identity", None)
    review.setdefault("evidence_ref", None)
    normalized["review"] = review

    merge_raw = normalized.get("merge")
    merge: dict[str, Any] = dict(merge_raw) if isinstance(merge_raw, dict) else {}
    merge.setdefault("status", "not_applicable")
    merge.setdefault("target", None)
    merge.setdefault("commit", None)
    merge.setdefault("evidence_ref", None)
    normalized["merge"] = merge

    release_raw = normalized.get("release")
    release: dict[str, Any] = dict(release_raw) if isinstance(release_raw, dict) else {}
    release.setdefault("status", "not_applicable")
    release.setdefault("target", None)
    release.setdefault("evidence_ref", None)
    normalized["release"] = release

    normalized["risk_class"] = str(normalized.get("risk_class") or "medium")
    verdict, reason = derive_delivery_verdict(normalized)
    normalized["delivery_verdict"] = verdict
    normalized["delivery_verdict_reason"] = reason
    normalized["last_verified_at"] = now
    return normalized


def build_delivery_state(
    task: Task,
    *,
    stage: str,
    workflow_stream_id: str,
    artifact_ref: Optional[dict[str, Any]] = None,
    artifact_refs: Optional[Iterable[dict[str, Any]]] = None,
    risk_class: str = "medium",
    proof: Optional[dict[str, Any]] = None,
    review: Optional[dict[str, Any]] = None,
    merge: Optional[dict[str, Any]] = None,
    release: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    base_state: dict[str, Any] = {
        "schema_version": 1,
        "task_id": task.id,
        "workflow_stream_id": workflow_stream_id,
        "stage": stage,
        "task_status": task.status,
        "assignee_profile": task.assignee,
        "artifact": {
            "primary_ref": artifact_ref,
            "refs": list(artifact_refs) if artifact_refs is not None else [],
        },
        "workspace": _delivery_workspace_snapshot(task),
        "proof": proof or {},
        "review": review or {},
        "merge": merge or {},
        "release": release or {},
        "risk_class": risk_class,
    }
    return _normalize_delivery_state(task, base_state)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

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
    workspace_base_ref   TEXT,
    workspace_base_commit TEXT,
    claim_lock           TEXT,
    claim_expires        INTEGER,
    tenant               TEXT,
    brand                TEXT,
    result               TEXT,
    delivery_state       TEXT,
    -- PASS-loop breaker persistence snapshot. Carries the latest durable
    -- threshold-tracking state (fingerprint, evidence refs, reset notes) while
    -- preserving the detailed event/run history elsewhere.
    pass_loop_state      TEXT,
    -- Cheap query surface for the current PASS-loop lifecycle state. See
    -- VALID_PASS_LOOP_STATUSES.
    pass_loop_status     TEXT,
    -- Current PASS-loop count for the latest unchanged fingerprint.
    pass_loop_count      INTEGER NOT NULL DEFAULT 0,
    -- Machine-readable reason code for the current halted/tracking loop
    -- state (for the first implementation this is PASS_LOOP_REASON_CODE).
    pass_loop_reason_code TEXT,
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
    -- 1 when the workspace kind was explicitly chosen at creation
    -- (e.g. ``--workspace scratch``). Pinned workspaces are never
    -- auto-upgraded to a worktree at create or dispatch time.
    workspace_pinned     INTEGER NOT NULL DEFAULT 0,
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
    notifier_profile TEXT,
    created_at    INTEGER NOT NULL,
    last_event_id INTEGER NOT NULL DEFAULT 0,
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


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

_INITIALIZED_PATHS: set[str] = set()
_INIT_LOCK = threading.RLock()
_SQLITE_HEADER = b"SQLite format 3\x00"
DEFAULT_BUSY_TIMEOUT_MS = 120_000

# Bounded acquire for the cross-process init lock (#36644). The original bare
# blocking flock had no timeout, so a wedged holder blocked the dispatcher's
# next-tick connect forever. We retry a non-blocking acquire up to this
# deadline, polling at this interval, then proceed without the cross-process
# lock (the in-process _INIT_LOCK + idempotent init remain the backstop).
_INIT_LOCK_TIMEOUT_SECONDS = 10.0
_INIT_LOCK_POLL_SECONDS = 0.05


def _resolve_busy_timeout_ms() -> int:
    """Return the SQLite busy timeout for Kanban connections.

    Kanban is the shared cross-profile dispatch bus, so worker stampedes are
    expected.  A long busy timeout lets SQLite serialize writers via WAL rather
    than surfacing transient ``database is locked`` failures during bursts.
    """
    raw = os.environ.get("HERMES_KANBAN_BUSY_TIMEOUT_MS", "").strip()
    if raw:
        try:
            parsed = int(raw)
        except ValueError:
            parsed = 0
        if parsed > 0:
            return parsed
    return DEFAULT_BUSY_TIMEOUT_MS


def _sqlite_connect(path: Path) -> sqlite3.Connection:
    """Open a Kanban SQLite connection with consistent lock waiting."""
    busy_timeout_ms = _resolve_busy_timeout_ms()
    conn = sqlite3.connect(
        str(path),
        isolation_level=None,
        timeout=busy_timeout_ms / 1000.0,
    )
    # ``sqlite3.connect(timeout=...)`` normally maps to busy_timeout, but set
    # the PRAGMA explicitly so it is observable and survives future wrapper
    # changes. Parameter binding is not supported for PRAGMA assignments.
    conn.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
    return conn


@contextlib.contextmanager
def _cross_process_init_lock(path: Path):
    """Serialize first-connect WAL/schema/integrity setup across processes.

    ``_INIT_LOCK`` only protects threads inside one Python process. During a
    dispatcher burst, many worker processes can all hit a fresh/legacy board at
    once and each process has an empty ``_INITIALIZED_PATHS`` cache. This file
    lock keeps header validation, integrity probing, WAL activation, and
    additive migrations single-file/single-writer across the whole host while
    leaving normal post-init DB usage concurrent under SQLite WAL.

    The acquire is **bounded** (issue #36644): the original bare blocking
    ``flock(LOCK_EX)`` had no timeout, so a single process stalled inside the
    critical section (or a stale lock held by a wedged worker) blocked every
    other ``connect()`` — including the long-lived gateway dispatcher's
    next-tick connect — forever, with no traceback and no recovery short of a
    restart. We now retry a non-blocking acquire up to a deadline; on timeout
    we log a WARNING and proceed WITHOUT the cross-process lock. That is safe:
    the in-process ``_INIT_LOCK`` still serializes same-process threads, and
    the init work itself is idempotent (``CREATE TABLE IF NOT EXISTS`` +
    additive migrations), so the worst case of two processes racing first-init
    is redundant work, not corruption. A bounded "proceed anyway" beats an
    unbounded hang that silently stops the board.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".init.lock")
    handle = lock_path.open("a+b")
    acquired = False
    try:
        deadline = time.monotonic() + _INIT_LOCK_TIMEOUT_SECONDS
        if _IS_WINDOWS:
            import msvcrt

            locking = getattr(msvcrt, "locking")
            nb_lock = getattr(msvcrt, "LK_NBLCK")
            while True:
                try:
                    handle.seek(0)
                    locking(handle.fileno(), nb_lock, 1)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except (BlockingIOError, OSError):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_INIT_LOCK_POLL_SECONDS)
        if not acquired:
            _log.warning(
                "kanban init lock for %s not acquired within %.0fs — proceeding "
                "without the cross-process lock (in-process lock + idempotent "
                "init are the correctness backstop). A stuck holder is no longer "
                "able to block this connect indefinitely (#36644).",
                lock_path, _INIT_LOCK_TIMEOUT_SECONDS,
            )
        yield
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    locking = getattr(msvcrt, "locking")
                    unlock_mode = getattr(msvcrt, "LK_UNLCK")
                    locking(handle.fileno(), unlock_mode, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@contextlib.contextmanager
def _dispatch_tick_lock(db_path: Path):
    """Non-blocking single-writer guard around one dispatcher tick.

    Yields ``True`` when this process holds the board's dispatch lock and
    may proceed with the tick, or ``False`` when another process already
    holds it (the caller should skip the tick this round).

    Motivation (issue #35240): a ``hermes gateway run --replace`` /
    ``gateway restart`` invoked from a shell on a systemd/launchd host can
    leave an orphan gateway whose dispatcher escapes the service cgroup,
    survives ``systemctl restart``, and becomes a *second* long-lived
    writer on the same ``kanban.db``. Two dispatchers that each believe
    they own the file both pass SQLite ``busy_timeout`` and then race on
    WAL frames — the documented root cause of multi-writer corruption.
    The startup guard (``_guard_supervised_gateway_conflict``) blocks the
    common way an orphan is born, but this lock is the defense-in-depth
    that prevents two dispatchers from ever writing concurrently
    *regardless of how the second one got there*.

    The lock is **non-blocking** on purpose: the gateway's async watcher
    must never stall on a held lock. A losing dispatcher simply skips its
    tick (the winner is making progress on the same board), and tries
    again next interval.

    Board-scoped: the lock file is a ``.dispatch.lock`` sibling of the
    board's ``kanban.db``, so unrelated boards tick independently. On
    platforms without ``fcntl``/``msvcrt`` the guard degrades to a no-op
    (yields ``True``) — single-writer enforcement is best-effort and the
    orphan-dispatcher scenario is specific to POSIX service managers.
    """
    lock_path = db_path.with_name(db_path.name + ".dispatch.lock")
    handle = None
    acquired = False
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        if _IS_WINDOWS:
            try:
                import msvcrt

                handle.seek(0)
                locking = getattr(msvcrt, "locking")
                # LK_NBLCK = non-blocking exclusive byte-range lock.
                nb_lock = getattr(msvcrt, "LK_NBLCK")
                locking(handle.fileno(), nb_lock, 1)
                acquired = True
            except (OSError, AttributeError):
                acquired = False
        else:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (BlockingIOError, OSError):
                acquired = False
    except OSError:
        # Could not even open the lock file (permissions, read-only FS).
        # Degrade to a no-op so a probe failure never blocks dispatch.
        acquired = True
        handle = None
    try:
        yield acquired
    finally:
        if handle is not None:
            try:
                if acquired:
                    if _IS_WINDOWS:
                        import msvcrt

                        handle.seek(0)
                        locking = getattr(msvcrt, "locking")
                        unlock_mode = getattr(msvcrt, "LK_UNLCK")
                        locking(handle.fileno(), unlock_mode, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except (OSError, AttributeError):
                pass
            finally:
                handle.close()


def _looks_like_tls_record_at(data: bytes, offset: int) -> bool:
    """Return True for a TLS record header at ``data[offset:]``."""
    if len(data) < offset + 5:
        return False
    content_type = data[offset]
    major = data[offset + 1]
    minor = data[offset + 2]
    length = int.from_bytes(data[offset + 3:offset + 5], "big")
    return (
        content_type in {0x14, 0x15, 0x16, 0x17}
        and major == 0x03
        and minor in {0x00, 0x01, 0x02, 0x03, 0x04}
        and 0 < length <= 18432
    )


def _validate_sqlite_header(path: Path) -> None:
    """Fail early with an actionable error for non-SQLite Kanban DB files.

    ``sqlite3.connect()`` creates missing and zero-byte files, so those are
    allowed. Existing non-empty files must have the SQLite header before we
    hand them to SQLite/WAL setup. This keeps corrupted page-0 failures from
    being collapsed into a generic PRAGMA error and lets the gateway's corrupt
    board handling identify the board by fingerprint.
    """
    try:
        stat = path.stat()
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.st_size == 0:
        return
    try:
        with path.open("rb") as handle:
            head = handle.read(64)
    except OSError:
        return
    if head.startswith(_SQLITE_HEADER):
        return
    signature = ""
    if head.startswith(b"SQLit") and _looks_like_tls_record_at(head, 5):
        signature = " (TLS record header detected at byte offset 5)"
    elif _looks_like_tls_record_at(head, 0):
        signature = " (TLS record header detected at byte offset 0)"
    raise sqlite3.DatabaseError(
        "file is not a database: invalid SQLite header for "
        f"{path}{signature}; first_32={head[:32].hex(' ')}"
    )


class KanbanDbCorruptError(RuntimeError):
    """Raised when an existing kanban DB file fails integrity checks.

    Fail-closed guard against silent recreation of a corrupt board file,
    which would otherwise destroy the user's tasks. Carries both the
    original path and the timestamped backup we made before refusing.
    """

    def __init__(self, db_path: Path, backup_path: Optional[Path], reason: str):
        self.db_path = db_path
        self.backup_path = backup_path
        self.reason = reason
        backup_str = str(backup_path) if backup_path is not None else "<backup failed>"
        super().__init__(
            f"Refusing to open corrupt kanban DB at {db_path}: {reason}. "
            f"Original preserved; backup at {backup_str}."
        )


def _backup_corrupt_db(path: Path) -> Optional[Path]:
    """Copy a corrupt DB (and its WAL/SHM sidecars) to a content-addressed backup.

    The backup filename is deterministic in the main DB's sha256, so repeated
    quarantines of the same corrupt bytes (gateway restarts, dispatcher retries,
    multi-profile fleets all hitting the same shared DB) reuse one backup
    instead of amplifying disk usage by N. If the corrupt bytes actually
    change between attempts — e.g. a partial repair or further damage — the
    fingerprint changes and a separate backup is preserved.

    Returns the backup path of the main DB file, or ``None`` if the copy
    itself failed (the caller still raises loudly in that case).

    Writes are confined to the original DB's parent directory. The backup
    basename is derived purely from ``path.name`` and a content hash, never
    from caller-supplied directory segments — no traversal is possible.
    """
    # Resolve once and pin the parent so subsequent path operations cannot
    # escape it. ``Path.resolve()`` collapses any ``..`` segments and
    # symlinks, and we only ever write inside ``parent``.
    resolved = path.resolve()
    parent = resolved.parent
    base_name = resolved.name  # basename only
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    token = digest.hexdigest()[:16]
    candidate = parent / f"{base_name}.corrupt.{token}.bak"
    # Defensive: candidate must still be inside parent after construction.
    if candidate.parent != parent:
        return None
    if not candidate.exists():
        try:
            shutil.copy2(resolved, candidate)
        except OSError:
            return None
    for suffix in ("-wal", "-shm"):
        sidecar = parent / (base_name + suffix)
        if sidecar.parent != parent or not sidecar.exists():
            continue
        sidecar_backup = parent / (candidate.name + suffix)
        if sidecar_backup.parent != parent or sidecar_backup.exists():
            continue
        try:
            shutil.copy2(sidecar, sidecar_backup)
        except OSError:
            pass
    return candidate


def _guard_existing_db_is_healthy(path: Path) -> None:
    """Run ``PRAGMA integrity_check`` on an existing non-empty DB file.

    Opens the probe in read/write mode so SQLite can recover or
    checkpoint a healthy WAL/hot-journal DB before we declare it
    corrupt. If the file is malformed, copy it (and any WAL/SHM
    sidecars) to a timestamped backup and raise
    :class:`KanbanDbCorruptError` so callers cannot silently recreate
    the schema on top of a damaged DB.

    Transient lock/busy errors (``sqlite3.OperationalError``) are NOT
    treated as corruption; they propagate raw so the caller sees a
    normal lock failure and no spurious ``.corrupt`` backup is made.

    No-op for missing files, zero-byte files (treated as fresh), and
    paths already proven healthy this process (cache hit).

    Path-trust note: ``path`` arrives via :func:`connect`, which itself
    resolves it from an explicit ``db_path`` argument, the
    :func:`kanban_db_path` env-var chain, or the kanban-home default —
    all sources Hermes treats as user-controlled-but-trusted on the
    user's own machine. We additionally resolve the path here and
    confine all filesystem writes to its parent directory so any
    accidental ``..`` segments are collapsed before any I/O happens.
    """
    # Resolve before any I/O. ``Path.resolve()`` normalizes ``..`` and
    # symlinks, giving us a canonical path whose parent dir we can pin.
    try:
        resolved = path.resolve()
    except OSError:
        return
    try:
        if not resolved.exists() or resolved.stat().st_size == 0:
            return
    except OSError:
        return
    if str(resolved) in _INITIALIZED_PATHS:
        return
    reason: Optional[str] = None
    try:
        probe = _sqlite_connect(resolved)
        try:
            row = probe.execute("PRAGMA integrity_check").fetchone()
        finally:
            probe.close()
        if not row or (row[0] or "").lower() != "ok":
            reason = f"integrity_check returned {row[0] if row else '<no row>'!r}"
    except sqlite3.OperationalError:
        # Lock contention, busy, transient IO — not corruption. Let it propagate.
        raise
    except sqlite3.DatabaseError as exc:
        reason = f"sqlite refused to open file: {exc}"
    if reason is None:
        return
    backup = _backup_corrupt_db(resolved)
    raise KanbanDbCorruptError(resolved, backup, reason)


def connect(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> sqlite3.Connection:
    """Open (and initialize if needed) the kanban DB.

    WAL mode is enabled on every connection; it's a no-op after the first
    time but keeps the code robust if the DB file is ever re-created.

    The first connection to a given path auto-runs :func:`init_db` so
    fresh installs and test harnesses that construct `connect()`
    directly don't have to remember a separate init step. Subsequent
    connections skip the schema check via a module-level path cache.

    Path resolution:

    * ``db_path`` explicit → used as-is (legacy callers, tests).
    * ``board`` explicit → resolves to that board's DB.
    * Neither → :func:`kanban_db_path` resolves via
      ``HERMES_KANBAN_DB`` env → ``HERMES_KANBAN_BOARD`` env →
      ``<root>/kanban/current`` → ``default``.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: once THIS process has initialized this path, the expensive
    # first-open work (header validation, integrity probe, schema + additive
    # migrations) is already done and cached in _INITIALIZED_PATHS. Acquiring
    # the cross-process init lock on every connect is what let a single stalled
    # holder (e.g. an external `hermes kanban list` mid-integrity-probe) block
    # the long-lived gateway dispatcher's next-tick connect() forever — an
    # unbounded flock with no timeout, no LOCK_NB, no recovery (#36644). On the
    # steady-state path there is nothing for the cross-process lock to protect
    # (no schema/migration writes run), so skip it entirely and just open the
    # connection with WAL/pragmas under the cheap in-process _INIT_LOCK.
    resolved = str(path.resolve())
    if resolved in _INITIALIZED_PATHS:
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA wal_autocheckpoint=100")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute("PRAGMA secure_delete=ON")
                conn.execute("PRAGMA cell_size_check=ON")
        except Exception:
            conn.close()
            raise
        return conn

    with _cross_process_init_lock(path):
        # Cheap byte-level check first — catches the #29507 TLS-overwrite shape
        # and other invalid-header cases without opening a sqlite connection.
        _validate_sqlite_header(path)
        # Full integrity probe — catches corruption past the header (malformed
        # pages, broken internal metadata). Cached per-path after first success
        # via _INITIALIZED_PATHS so it only runs once per process per path.
        _guard_existing_db_is_healthy(path)
        resolved = str(path.resolve())
        conn = _sqlite_connect(path)
        try:
            conn.row_factory = sqlite3.Row
            with _INIT_LOCK:
                # WAL activation can take an exclusive lock while SQLite creates the
                # sidecar files for a fresh database. Keep it in the same process-local
                # critical section as schema initialization so concurrent gateway
                # startup threads do not race before _INITIALIZED_PATHS is populated.
                # WAL doesn't work on network filesystems (NFS/SMB/FUSE). Shared helper
                # falls back to DELETE with one WARNING so kanban stays usable there.
                # See hermes_state._WAL_INCOMPAT_MARKERS for detection logic.
                from hermes_state import apply_wal_with_fallback
                apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
                # FULL (was NORMAL): fsync before each checkpoint to narrow the
                # crash window that can leave a b-tree page header torn.
                conn.execute("PRAGMA synchronous=FULL")
                conn.execute("PRAGMA wal_autocheckpoint=100")
                conn.execute("PRAGMA foreign_keys=ON")
                # Zero freed pages so a later torn write cannot expose stale
                # cell content; persisted in the DB header for new DBs.
                conn.execute("PRAGMA secure_delete=ON")
                # Surface corrupt cells as read errors instead of silent
                # wrong-data returns.
                conn.execute("PRAGMA cell_size_check=ON")
                needs_init = resolved not in _INITIALIZED_PATHS
                if needs_init:
                    # Idempotent: runs CREATE TABLE IF NOT EXISTS + the additive
                    # migrations. Cached so subsequent connect() calls in the same
                    # process are cheap. The lock prevents same-process dispatcher
                    # threads from racing through the additive ALTER TABLE pass with
                    # stale PRAGMA snapshots during gateway startup.
                    conn.executescript(SCHEMA_SQL)
                    _migrate_add_optional_columns(conn)
                    _INITIALIZED_PATHS.add(resolved)
        except Exception:
            conn.close()
            raise
    return conn


@contextlib.contextmanager
def connect_closing(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
):
    """Open a kanban DB connection and guarantee it is closed on exit.

    Use this instead of ``with kb.connect() as conn:`` — sqlite3's
    built-in connection context manager only commits/rollbacks the
    transaction; it does NOT close the file descriptor. In long-lived
    processes (gateway, dashboard) that route every kanban operation
    through ``connect()`` (e.g. ``run_slash`` dispatching ``/kanban …``
    commands, ``decompose_task_endpoint`` calling
    ``kanban_decompose.decompose_task``), the unclosed connections
    accumulate as open FDs to ``kanban.db`` and ``kanban.db-wal``. After
    enough operations the process hits the kernel FD limit and dies
    with ``[Errno 24] Too many open files``.

    See #33159 for the production incident.

    The ``connect()`` function itself remains unchanged so callers that
    intentionally manage the connection lifetime (tests, long-lived
    callers) continue to work.
    """
    conn = connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
) -> Path:
    """Create the schema if it doesn't exist; return the path used.

    Kept as a public entry point so CLI ``hermes kanban init`` and the
    daemon have something explicit to call. Unlike :func:`connect`'s
    first-time auto-init (which caches by path), ``init_db`` always
    re-runs the migration pass. Callers that know the on-disk schema
    may have drifted — tests that write legacy event kinds directly,
    external tools that upgrade an old DB file — can call this to
    force re-migration.
    """
    if db_path is not None:
        path = db_path
    else:
        path = kanban_db_path(board=board)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(path.resolve())
    # Clear the cache entry so the underlying connect() re-runs the
    # schema + migration pass unconditionally.
    with _INIT_LOCK:
        _INITIALIZED_PATHS.discard(resolved)
    with contextlib.closing(connect(path)):
        pass
    return path


def _migrate_add_optional_columns(conn: sqlite3.Connection) -> None:
    """Add columns that were introduced after v1 release to legacy DBs.

    Called by ``init_db`` so opening an old DB is always safe.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
    if "tenant" not in cols:
        _add_column_if_missing(conn, "tasks", "tenant", "tenant TEXT")
    if "brand" not in cols:
        _add_column_if_missing(conn, "tasks", "brand", "brand TEXT")
    if "result" not in cols:
        _add_column_if_missing(conn, "tasks", "result", "result TEXT")
    if "branch_name" not in cols:
        _add_column_if_missing(conn, "tasks", "branch_name", "branch_name TEXT")
    if "project_id" not in cols:
        _add_column_if_missing(conn, "tasks", "project_id", "project_id TEXT")
    if "workspace_base_ref" not in cols:
        _add_column_if_missing(conn, "tasks", "workspace_base_ref", "workspace_base_ref TEXT")
    if "workspace_base_commit" not in cols:
        _add_column_if_missing(conn, "tasks", "workspace_base_commit", "workspace_base_commit TEXT")
    if "idempotency_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "idempotency_key", "idempotency_key TEXT"
        )
    # ``idx_tasks_idempotency`` is created unconditionally below alongside
    # the other additive-column indexes — see the block after the
    # legacy-column migration. Creating it here too would be redundant.

    # Refresh after early additive migrations above. Some existing DBs were
    # partially migrated in older releases and can already contain the later
    # columns (for example ``consecutive_failures``) even when this function's
    # initial snapshot did not. Re-snapshot here so the legacy-column migration
    # below is truly idempotent and never re-adds columns that already exist.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}

    # Legacy column migration: ``spawn_failures`` → ``consecutive_failures``
    # and ``last_spawn_error`` → ``last_failure_error``.
    #
    # Avoid ``ALTER TABLE ... RENAME COLUMN`` for two reasons:
    #   1. Primary: very old DBs may never have had ``spawn_failures`` at
    #      all, so RENAME raises OperationalError: no such column (the crash
    #      reported in issue #20842 after the #20410 update).
    #   2. Secondary: SQLite reparses the whole schema on any RENAME, which
    #      fails if related objects (views, triggers) reference the old name.
    #
    # ADD-first-then-copy is tolerant of both shapes and preserves
    # historical counter values when the legacy columns do exist.
    if "consecutive_failures" not in cols:
        added = _add_column_if_missing(
            conn,
            "tasks",
            "consecutive_failures",
            "consecutive_failures INTEGER NOT NULL DEFAULT 0",
        )
        if added and "spawn_failures" in cols:
            conn.execute(
                "UPDATE tasks SET consecutive_failures = COALESCE(spawn_failures, 0)"
            )
    if "worker_pid" not in cols:
        _add_column_if_missing(conn, "tasks", "worker_pid", "worker_pid INTEGER")
    if "last_failure_error" not in cols:
        added = _add_column_if_missing(
            conn, "tasks", "last_failure_error", "last_failure_error TEXT"
        )
        if added and "last_spawn_error" in cols:
            conn.execute(
                "UPDATE tasks SET last_failure_error = last_spawn_error"
            )
    if "max_runtime_seconds" not in cols:
        _add_column_if_missing(
            conn, "tasks", "max_runtime_seconds", "max_runtime_seconds INTEGER"
        )
    if "last_heartbeat_at" not in cols:
        _add_column_if_missing(
            conn, "tasks", "last_heartbeat_at", "last_heartbeat_at INTEGER"
        )
    if "current_run_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_run_id", "current_run_id INTEGER"
        )
    if "workflow_template_id" not in cols:
        _add_column_if_missing(
            conn, "tasks", "workflow_template_id", "workflow_template_id TEXT"
        )
    if "current_step_key" not in cols:
        _add_column_if_missing(
            conn, "tasks", "current_step_key", "current_step_key TEXT"
        )
    if "skills" not in cols:
        # JSON array of skill names the dispatcher force-loads into the
        # worker via --skills. NULL is fine for existing rows.
        _add_column_if_missing(conn, "tasks", "skills", "skills TEXT")

    if "max_retries" not in cols:
        # Per-task override for the consecutive-failure circuit breaker.
        # NULL = fall through to the dispatcher-level ``kanban.failure_limit``
        # config, then ``DEFAULT_FAILURE_LIMIT``. Existing rows get NULL,
        # which is the correct default (they keep the global behaviour
        # they were getting before the column existed).
        _add_column_if_missing(conn, "tasks", "max_retries", "max_retries INTEGER")

    if "model_override" not in cols:
        conn.execute("ALTER TABLE tasks ADD COLUMN model_override TEXT")

    if "goal_mode" not in cols:
        # Ralph-style goal loop toggle for the dispatched worker. 0 (the
        # default) = classic single-shot worker, preserving the behaviour
        # existing rows had before the column existed.
        _add_column_if_missing(
            conn, "tasks", "goal_mode", "goal_mode INTEGER NOT NULL DEFAULT 0"
        )

    if "goal_max_turns" not in cols:
        # Per-task goal-loop turn budget. NULL = goals-engine default.
        _add_column_if_missing(
            conn, "tasks", "goal_max_turns", "goal_max_turns INTEGER"
        )

    if "workspace_pinned" not in cols:
        # Explicit workspace choice marker. 0 (the default) preserves the
        # behaviour existing rows had before the column existed: eligible
        # for auto-upgrade to a worktree on git-backed boards.
        _add_column_if_missing(
            conn, "tasks", "workspace_pinned",
            "workspace_pinned INTEGER NOT NULL DEFAULT 0"
        )

    if "session_id" not in cols:
        # Originating agent/chat session id, populated when the task is
        # created from within an agent loop that propagated
        # ``HERMES_SESSION_ID`` (e.g. ACP). NULL on legacy rows and on any
        # creation path that doesn't set the env var (CLI, dashboard).
        _add_column_if_missing(
            conn, "tasks", "session_id", "session_id TEXT"
        )

    if "block_kind" not in cols:
        # Typed block reason (VALID_BLOCK_KINDS) or NULL for legacy/un-typed
        # blocks. Existing blocked rows get NULL, which is treated as a
        # generic human blocker — same behaviour they had before the column.
        _add_column_if_missing(conn, "tasks", "block_kind", "block_kind TEXT")

    if "block_recurrences" not in cols:
        # Unblock-loop counter. Existing rows start at 0, so the loop breaker
        # only begins counting from the first re-block after this migration.
        _add_column_if_missing(
            conn,
            "tasks",
            "block_recurrences",
            "block_recurrences INTEGER NOT NULL DEFAULT 0",
        )

    if "delivery_state" not in cols:
        # Canonical task-level machine-readable delivery-truth snapshot for
        # the structured software-delivery pilot. Existing rows keep NULL
        # until a bounded backfill or an explicit stage handoff writes one.
        _add_column_if_missing(
            conn, "tasks", "delivery_state", "delivery_state TEXT"
        )

    if "pass_loop_state" not in cols:
        # Durable PASS-loop fingerprint/evidence snapshot. Existing rows stay
        # NULL until the convergence guard writes one.
        _add_column_if_missing(
            conn, "tasks", "pass_loop_state", "pass_loop_state TEXT"
        )

    if "pass_loop_status" not in cols:
        _add_column_if_missing(
            conn, "tasks", "pass_loop_status", "pass_loop_status TEXT"
        )

    if "pass_loop_count" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "pass_loop_count",
            "pass_loop_count INTEGER NOT NULL DEFAULT 0",
        )

    if "pass_loop_reason_code" not in cols:
        _add_column_if_missing(
            conn,
            "tasks",
            "pass_loop_reason_code",
            "pass_loop_reason_code TEXT",
        )

    # Indexes over additive ``tasks`` columns must be created after the
    # columns exist. Keeping them in SCHEMA_SQL breaks legacy boards: SQLite
    # parses each statement in ``executescript`` against the live schema, so a
    # ``CREATE INDEX`` over a missing column aborts initialization before the
    # additive ``ALTER TABLE`` migrations below can run. Re-running them here
    # is cheap thanks to ``IF NOT EXISTS`` and stays correct on fresh DBs
    # (where the columns already exist from SCHEMA_SQL).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_tenant ON tasks(tenant)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_brand ON tasks(brand)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_idempotency ON tasks(idempotency_key)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_session_id ON tasks(session_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_pass_loop_status ON tasks(pass_loop_status)"
    )

    # task_events gained a run_id column; back-fill it as NULL for
    # historical events (they predate runs and can't be attributed).
    ev_cols = {row["name"] for row in conn.execute("PRAGMA table_info(task_events)")}
    if "run_id" not in ev_cols:
        _add_column_if_missing(conn, "task_events", "run_id", "run_id INTEGER")

    # Same ordering rule as the additive ``tasks`` indexes above: create the
    # index after the additive column migration so legacy ``task_events``
    # tables don't fail during SCHEMA_SQL execution before ``run_id`` exists.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_events_run "
        "ON task_events(run_id, id)"
    )

    notify_table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='kanban_notify_subs'"
    ).fetchone() is not None
    if notify_table_exists:
        notify_cols = {
            row["name"] for row in conn.execute("PRAGMA table_info(kanban_notify_subs)")
        }
        if "notifier_profile" not in notify_cols:
            _add_column_if_missing(
                conn, "kanban_notify_subs", "notifier_profile", "notifier_profile TEXT"
            )

    # One-shot backfill: any task that is 'running' before runs existed
    # had its claim_lock / claim_expires / worker_pid on the task row.
    # Synthesize a matching task_runs row so subsequent end-run / heartbeat
    # calls have something to write to. Wrapped in write_txn to serialize
    # against any concurrent dispatcher, and the per-row UPDATE uses
    # ``current_run_id IS NULL`` as a CAS guard so a racing claim can't
    # produce an orphaned row if it interleaves with the backfill pass.
    runs_exist = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='task_runs'"
    ).fetchone() is not None
    if runs_exist:
        with write_txn(conn):
            inflight = conn.execute(
                "SELECT id, assignee, claim_lock, claim_expires, worker_pid, "
                "       max_runtime_seconds, last_heartbeat_at, started_at "
                "FROM tasks "
                "WHERE status = 'running' AND current_run_id IS NULL"
            ).fetchall()
            for row in inflight:
                started = row["started_at"] or int(time.time())
                cur = conn.execute(
                    """
                    INSERT INTO task_runs (
                        task_id, profile, status,
                        claim_lock, claim_expires, worker_pid,
                        max_runtime_seconds, last_heartbeat_at,
                        started_at
                    ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["id"], row["assignee"], row["claim_lock"],
                        row["claim_expires"], row["worker_pid"],
                        row["max_runtime_seconds"], row["last_heartbeat_at"],
                        started,
                    ),
                )
                # CAS: only install the pointer if nothing else claimed
                # the task between our SELECT and here (shouldn't happen
                # under the write_txn, but belt-and-suspenders). If the
                # CAS fails we've got an orphan run_row — mark it
                # reclaimed so it doesn't look in-flight.
                upd = conn.execute(
                    "UPDATE tasks SET current_run_id = ? "
                    "WHERE id = ? AND current_run_id IS NULL",
                    (cur.lastrowid, row["id"]),
                )
                if upd.rowcount != 1:
                    conn.execute(
                        "UPDATE task_runs SET status = 'reclaimed', "
                        "    outcome = 'reclaimed', ended_at = ? "
                        "WHERE id = ?",
                        (int(time.time()), cur.lastrowid),
                    )

    # One-shot event-kind rename pass. The old names ("ready", "priority",
    # "spawn_auto_blocked") still worked but were awkward on the wire;
    # rename them in-place so existing DBs migrate cleanly. Fires once
    # per DB because after the UPDATE no rows match the old kinds.
    _EVENT_RENAMES = (
        # (old, new)
        ("ready",              "promoted"),
        ("priority",           "reprioritized"),
        ("spawn_auto_blocked", "gave_up"),
    )
    for old, new in _EVENT_RENAMES:
        conn.execute(
            "UPDATE task_events SET kind = ? WHERE kind = ?",
            (new, old),
        )

    _rebuild_drifted_tables(conn)


# Legacy DBs defined these tables with a ``TEXT PRIMARY KEY`` id (or, for
# ``kanban_notify_subs``, a nullable ``TEXT last_event_id``). The current
# schema uses ``INTEGER PRIMARY KEY AUTOINCREMENT`` / ``INTEGER NOT NULL
# DEFAULT 0``. ``CREATE TABLE IF NOT EXISTS`` skips existing tables
# regardless of schema and ``_add_column_if_missing`` only adds columns, so
# neither can fix a drifted column type — the table must be rebuilt. See
# #35096.
#
# Each entry pairs the canonical CREATE TABLE with the CREATE INDEX
# statements that DROP TABLE would otherwise take down with it (including
# ``idx_events_run``, added by the additive pass above). To guard against
# this list drifting from SCHEMA_SQL, ``test_rebuilt_schema_matches_fresh``
# asserts a rebuilt legacy DB is byte-identical to a fresh one.
_REBUILD_SPECS = {
    "task_events": (
        "CREATE TABLE task_events ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, run_id INTEGER, kind TEXT NOT NULL,"
        " payload TEXT, created_at INTEGER NOT NULL)",
        (
            "CREATE INDEX idx_events_task ON task_events(task_id, created_at)",
            "CREATE INDEX idx_events_run ON task_events(run_id, id)",
        ),
    ),
    "task_comments": (
        "CREATE TABLE task_comments ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, author TEXT NOT NULL, body TEXT NOT NULL,"
        " created_at INTEGER NOT NULL)",
        ("CREATE INDEX idx_comments_task ON task_comments(task_id, created_at)",),
    ),
    "task_runs": (
        "CREATE TABLE task_runs ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " task_id TEXT NOT NULL, profile TEXT, step_key TEXT,"
        " status TEXT NOT NULL, claim_lock TEXT, claim_expires INTEGER,"
        " worker_pid INTEGER, max_runtime_seconds INTEGER,"
        " last_heartbeat_at INTEGER, started_at INTEGER NOT NULL,"
        " ended_at INTEGER, outcome TEXT, summary TEXT, metadata TEXT,"
        " error TEXT)",
        (
            "CREATE INDEX idx_runs_task ON task_runs(task_id, started_at)",
            "CREATE INDEX idx_runs_status ON task_runs(status)",
        ),
    ),
    "kanban_notify_subs": (
        "CREATE TABLE kanban_notify_subs ("
        " task_id TEXT NOT NULL, platform TEXT NOT NULL, chat_id TEXT NOT NULL,"
        " thread_id TEXT NOT NULL DEFAULT '', user_id TEXT,"
        " notifier_profile TEXT, created_at INTEGER NOT NULL,"
        " last_event_id INTEGER NOT NULL DEFAULT 0,"
        " PRIMARY KEY (task_id, platform, chat_id, thread_id))",
        ("CREATE INDEX idx_notify_task ON kanban_notify_subs(task_id)",),
    ),
}


def _table_has_drifted(conn: sqlite3.Connection, table: str) -> bool:
    """True when ``table`` still carries the legacy (pre-AUTOINCREMENT) shape."""
    info = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if not info:
        return False  # table absent — nothing to rebuild
    if table == "kanban_notify_subs":
        lei = next((c for c in info if c["name"] == "last_event_id"), None)
        return lei is not None and (lei["type"] or "").upper() != "INTEGER"
    # task_events / task_comments / task_runs: id must be INTEGER and a PK.
    id_col = next((c for c in info if c["name"] == "id"), None)
    if id_col is None:
        return False
    return not ((id_col["type"] or "").upper() == "INTEGER" and id_col["pk"])


def _rebuild_drifted_tables(conn: sqlite3.Connection) -> None:
    """Rebuild any kanban table whose column types drifted from SCHEMA_SQL.

    Old boards crash the gateway notifier (``int(None)`` on a NULL id in
    ``unseen_events_for_sub``) and never match the ``id > cursor`` filter, so
    every kanban notification is silently lost (#35096). Each affected table is
    rebuilt with the standard SQLite pattern — CREATE new → INSERT shared
    columns → DROP old → RENAME — recreating its indexes too (DROP TABLE takes
    them down). The legacy TEXT ids are dropped (they aren't valid integers);
    AUTOINCREMENT assigns fresh ones and ``last_event_id`` cursors reset to 0,
    so the first post-migration tick replays a task's event history once —
    the safe failure mode for a feature that was already fully broken.

    The whole pass runs in one transaction so an interruption can't leave a
    table half-renamed, and under ``connect()``'s init locks so nothing races
    it. Idempotent: a correctly-typed DB skips every table and returns without
    opening a transaction.
    """
    drifted = [t for t in _REBUILD_SPECS if _table_has_drifted(conn, t)]
    if not drifted:
        return

    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in drifted:
            create_sql, index_sqls = _REBUILD_SPECS[table]
            old_cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})")]
            _log.info("kanban migration: rebuilding %s to match current schema", table)
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
            conn.execute(create_sql)
            new_cols = {c["name"] for c in conn.execute(f"PRAGMA table_info({table})")}
            if table == "kanban_notify_subs":
                # Cast the legacy TEXT cursor to INTEGER; NULL / non-numeric → 0.
                shared = [c for c in old_cols if c in new_cols and c != "last_event_id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}, last_event_id) "
                    f"SELECT {cols_csv}, COALESCE(CAST(last_event_id AS INTEGER), 0) "
                    f"FROM {table}_legacy"
                )
            else:
                # Drop the legacy TEXT id; AUTOINCREMENT reassigns it.
                shared = [c for c in old_cols if c in new_cols and c != "id"]
                cols_csv = ", ".join(shared)
                conn.execute(
                    f"INSERT INTO {table} ({cols_csv}) "
                    f"SELECT {cols_csv} FROM {table}_legacy"
                )
            conn.execute(f"DROP TABLE {table}_legacy")
            for index_sql in index_sqls:
                conn.execute(index_sql)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass
        raise


def _check_file_length_invariant(conn: sqlite3.Connection) -> None:
    """Read the SQLite header page_count and compare against actual file size.

    Raises sqlite3.DatabaseError if the file is shorter than the header claims
    (torn-extend corruption).
    """
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        if row is None:
            return
        path_str = row[2]  # column 2 is the file path; empty for in-memory DBs
        if not path_str:
            return  # in-memory or unnamed DB; skip
        path = path_str
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        file_size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(28)
            header_bytes = f.read(4)
        if len(header_bytes) < 4:
            return  # can't read header; skip
        header_page_count = int.from_bytes(header_bytes, "big")
        if header_page_count == 0:
            return  # new/empty DB; skip
        actual_pages = file_size // page_size
        if actual_pages < header_page_count:
            raise sqlite3.DatabaseError(
                f"torn-extend detected: page count mismatch on {path}: "
                f"header claims {header_page_count} pages, "
                f"file has {actual_pages} pages "
                f"(missing {header_page_count - actual_pages} pages, "
                f"file_size={file_size}, page_size={page_size})"
            )
    except sqlite3.DatabaseError:
        raise
    except Exception:
        pass  # I/O errors during check are non-fatal; let normal ops continue


# SQLite's own busy_timeout uses a near-deterministic backoff, so concurrent
# writers re-collide in lockstep under a stampede. A jittered retry on the
# transaction boundary breaks that convoy. Mirrors state.db's _execute_write:
# a fixed 20-150ms jitter band (a 20ms floor prevents a near-zero retry from
# busy-spinning back into the collision). Only BEGIN IMMEDIATE and COMMIT are
# retried -- both are idempotent re-issues that touch no transaction body, so a
# CAS inside write_txn is never replayed. kanban keeps fewer retries than
# state.db (5 vs 15) because its 120s busy_timeout already absorbs most waits;
# the retry is the backstop for the tail SQLite returns BUSY on immediately.
_BUSY_MAX_RETRIES = 5
_BUSY_RETRY_MIN_S = 0.020  # 20ms
_BUSY_RETRY_MAX_S = 0.150  # 150ms


def _is_busy_error(exc: BaseException) -> bool:
    return isinstance(exc, sqlite3.OperationalError) and (
        "database is locked" in str(exc).lower()
        or "database is busy" in str(exc).lower()
    )


def _execute_boundary_with_retry(conn: sqlite3.Connection, sql: str) -> None:
    for attempt in range(_BUSY_MAX_RETRIES + 1):
        try:
            conn.execute(sql)
            return
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt == _BUSY_MAX_RETRIES:
                raise
            time.sleep(random.uniform(_BUSY_RETRY_MIN_S, _BUSY_RETRY_MAX_S))


@contextlib.contextmanager
def write_txn(conn: sqlite3.Connection):
    """Context manager for an IMMEDIATE write transaction.

    Use for any multi-statement write (creating a task + link, claiming a
    task + recording an event, etc.).  A claim CAS inside this context is
    atomic -- at most one concurrent writer can succeed.

    The explicit ROLLBACK on exception is wrapped in try/except so that
    a SQLite auto-rollback (which leaves no active transaction) does not
    shadow the original exception with a spurious rollback error.
    """
    _execute_boundary_with_retry(conn, "BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            # SQLite has already auto-rolled-back the transaction (typical
            # under EIO, lock contention, or corruption). Nothing to undo;
            # do not let this secondary failure shadow the real one.
            pass
        raise
    else:
        try:
            _execute_boundary_with_retry(conn, "COMMIT")
        except Exception:
            # COMMIT exhausted retries with the txn still open; roll back so the
            # connection isn't poisoned for the next BEGIN IMMEDIATE.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        # Post-commit file-length check: header page_count must match actual file pages.
        # A discrepancy means a torn-extend — raise now rather than silently corrupt.
        _check_file_length_invariant(conn)


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------

def _new_task_id() -> str:
    """Generate a short, URL-safe task id.

    4 hex bytes = ~4.3B possibilities. At 10k tasks the collision
    probability is ~1.2e-5; at 100k it's ~1.2e-3. Previously we used 2
    hex bytes (65k possibilities) which hit the birthday paradox hard:
    ~5% collision probability at 1k tasks, ~50% at 10k. Callers that
    care about idempotency should pass ``idempotency_key`` to
    :func:`create_task` rather than rely on id uniqueness.
    """
    return "t_" + secrets.token_hex(4)


def _claimer_id() -> str:
    """Return a ``host:pid`` string that identifies this claimer."""
    import socket
    try:
        host = socket.gethostname() or "unknown"
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


# ---------------------------------------------------------------------------
# Task creation / mutation
# ---------------------------------------------------------------------------

def _canonical_assignee(assignee: Optional[str]) -> Optional[str]:
    """Lowercase-assignee normalization for Kanban rows (dashboard/CLI parity)."""
    if assignee is None:
        return None
    from hermes_cli.profiles import normalize_profile_name

    return normalize_profile_name(assignee)


def _canonical_brand(brand: Optional[str], board: Optional[str]) -> Optional[str]:
    """Return a stable brand tag for the task row.

    Explicit ``brand`` wins. Otherwise we fall back to the board slug so every
    task row carries a machine-checkable brand tag, even when the caller only
    knows which board/database it is writing to.
    """
    if brand is not None:
        brand = str(brand).strip()
        return brand or None
    try:
        board_slug = _normalize_board_slug(board) if board is not None else None
    except Exception:
        board_slug = None
    if board_slug:
        return board_slug
    try:
        return get_current_board() or None
    except Exception:
        return None


def create_task(
    conn: sqlite3.Connection,
    *,
    title: str,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    created_by: Optional[str] = None,
    workspace_kind: str = "scratch",
    workspace_path: Optional[str] = None,
    branch_name: Optional[str] = None,
    tenant: Optional[str] = None,
    brand: Optional[str] = None,
    priority: int = 0,
    parents: Iterable[str] = (),
    triage: bool = False,
    idempotency_key: Optional[str] = None,
    max_runtime_seconds: Optional[int] = None,
    skills: Optional[Iterable[str]] = None,
    max_retries: Optional[int] = None,
    goal_mode: bool = False,
    goal_max_turns: Optional[int] = None,
    initial_status: str = "running",
    session_id: Optional[str] = None,
    board: Optional[str] = None,
    project_id: Optional[str] = None,
    workspace_pinned: bool = False,
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
    """
    assignee = _canonical_assignee(assignee)
    if not title or not title.strip():
        raise ValueError("title is required")
    if initial_status not in VALID_INITIAL_STATUSES:
        raise ValueError(
            f"initial_status must be one of {sorted(VALID_INITIAL_STATUSES)}"
        )
    if workspace_kind not in VALID_WORKSPACE_KINDS:
        raise ValueError(
            f"workspace_kind must be one of {sorted(VALID_WORKSPACE_KINDS)}, "
            f"got {workspace_kind!r}"
        )
    if branch_name is not None:
        branch_name = str(branch_name).strip() or None
    if branch_name and workspace_kind != "worktree":
        raise ValueError("branch_name is only valid for worktree workspaces")

    # Resolve an optional first-class Project link. A project-linked task is
    # anchored to the project's primary repo as a git worktree, so its branch
    # can be named deterministically (project slug + task id) instead of the
    # random ``wt/<task-id>`` fallback the worker skill applies when no branch
    # is set. Projects live in the creator's per-profile projects.db; the repo
    # path is absolute (profile-independent) and the branch name is pure, so the
    # cross-profile dispatcher needs no projects.db access at dispatch time.
    project_obj = None
    # Primary repo of a project-linked worktree task whose path we still need to
    # derive (a fresh worktree dir under the repo, computed once task_id exists).
    project_repo: Optional[str] = None
    if project_id is not None:
        project_id = str(project_id).strip() or None
    if project_id:
        try:
            from hermes_cli import projects_db as _pdb

            with _pdb.connect_closing() as _pconn:
                project_obj = _pdb.get_project(_pconn, project_id)
        except Exception:
            project_obj = None
        if project_obj is None:
            # A project id/slug that doesn't resolve must not crash task
            # creation or persist a dangling reference — drop the link and
            # create the task as an ordinary (scratch) task.
            project_id = None
        else:
            # Canonicalise (a slug may have been passed) and anchor the
            # worktree under the project's primary repo.
            project_id = project_obj.id
            if workspace_kind == "scratch" and project_obj.primary_path:
                workspace_kind = "worktree"
            if (
                workspace_kind == "worktree"
                and workspace_path is None
                and project_obj.primary_path
            ):
                # Defer the concrete path to the insert loop: it's a fresh
                # ``<repo>/.worktrees/<task-id>`` dir keyed on the new task id.
                project_repo = str(project_obj.primary_path)

    # Board-linked worktree upgrade. When the task has no explicit project
    # link and would otherwise be an ephemeral ``scratch`` task, but the
    # target board is bound to a git repo (its ``default_workdir`` — set by
    # ``hermes project bind-board`` / ``project create --board``), materialize
    # the work as a real linked git worktree on that repo instead of a
    # throwaway scratch dir. This keeps board-routed builds on their real
    # branch/repo instead of evaporating when the scratch workspace is deleted
    # on completion. ``default_workdir`` lives in the *shared* board metadata,
    # so this resolves correctly no matter which profile creates the card —
    # no cross-profile projects.db access is needed.
    #
    # goal_mode roots stay scratch: they run coordination loops, produce no
    # code of their own, and the worktree completion gate (real CI on a code
    # branch) would block them forever. An explicit ``--workspace worktree``
    # still wins for the rare goal card that genuinely edits code.
    board_repo: Optional[str] = None
    if (
        project_repo is None
        and workspace_path is None
        and workspace_kind == "scratch"
        and not goal_mode
        and not workspace_pinned
    ):
        try:
            _board_slug = board if board else get_current_board()
            _board_default = (
                read_board_metadata(_board_slug).get("default_workdir") or ""
            ).strip()
            if _board_default:
                _repo_root = _repo_root_for_worktree_target(
                    Path(_board_default).expanduser()
                )
                if _repo_root is not None:
                    # Upgrade to a linked worktree anchored on the bound repo.
                    # The concrete ``<repo>/.worktrees/<task-id>`` path is
                    # deferred to the insert loop (keyed on the new task id),
                    # mirroring the project-linked path above.
                    workspace_kind = "worktree"
                    board_repo = str(_repo_root)
        except Exception:
            # Never let board resolution crash task creation — an unmounted
            # external repo, a torn git dir, or a transient git failure must
            # degrade gracefully to an ordinary scratch task.
            board_repo = None

    brand = _canonical_brand(brand, board)
    parents = tuple(p for p in parents if p)

    # Normalise + validate skills: strip whitespace, drop empties, dedupe
    # (preserving order). Refuse commas inside a single name so we don't
    # invisibly splatter a comma-joined string into one argv slot — the
    # `hermes --skills X,Y` comma syntax is handled in the dispatcher,
    # not here.
    skills_list: Optional[list[str]] = None
    if skills is not None:
        cleaned: list[str] = []
        seen: set[str] = set()
        # Collect all toolset-name confusions up front so the user sees the
        # whole list at once. Raising on the first hit is friendly when the
        # input has one mistake, but agents that confuse skills with toolsets
        # usually pass several at once (`skills=["web", "browser", "terminal"]`)
        # and serial-correcting one per failure round-trips wastes tokens.
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
        skills_list = cleaned

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
        and board_repo is None
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
            with write_txn(conn):
                # Determine task status from parent status, unless the caller
                # parks it directly in blocked for human-ops review or in
                # triage for a specifier.
                if initial_status == "blocked":
                    task_status = "blocked"
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

                # Project- or board-linked worktree: a fresh worktree dir under
                # the anchor repo, ``<repo>/.worktrees/<task-id>``, keyed on the
                # new task id. This kills the unanchored ``.worktrees/<id>``
                # under the dispatcher's cwd. A project link additionally gets a
                # deterministic branch (project slug + task id); a board link
                # falls back to the worker's ``wt/<task-id>`` branch.
                anchor_repo = project_repo or board_repo
                if workspace_kind == "worktree" and anchor_repo and not workspace_path:
                    workspace_path = os.path.join(
                        anchor_repo, ".worktrees", task_id
                    )
                if (
                    project_obj is not None
                    and workspace_kind == "worktree"
                    and not branch_name
                ):
                    # _pdb was imported above when project_obj was resolved.
                    try:
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
                        branch_name, project_id, tenant, brand, idempotency_key,
                        max_runtime_seconds,
                        skills, max_retries, goal_mode, goal_max_turns, session_id,
                        workspace_pinned
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        brand,
                        idempotency_key,
                        int(max_runtime_seconds) if max_runtime_seconds is not None else None,
                        json.dumps(skills_list) if skills_list is not None else None,
                        int(max_retries) if max_retries is not None else None,
                        1 if goal_mode else 0,
                        int(goal_max_turns) if goal_max_turns is not None else None,
                        session_id,
                        1 if workspace_pinned else 0,
                    ),
                )
                for pid in parents:
                    conn.execute(
                        "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
                        (pid, task_id),
                    )
                _append_event(
                    conn,
                    task_id,
                    "created",
                    {
                        "assignee": assignee,
                        "status": task_status,
                        "parents": list(parents),
                        "tenant": tenant,
                        "brand": brand,
                        "branch_name": branch_name,
                        "skills": list(skills_list) if skills_list else None,
                        "goal_mode": bool(goal_mode) or None,
                    },
                )
            return task_id
        except sqlite3.IntegrityError:
            if attempt == 1:
                raise
            # Retry with a fresh id.
            continue
    raise RuntimeError("unreachable")


def _find_missing_parents(conn: sqlite3.Connection, parents: Iterable[str]) -> list[str]:
    parents = list(parents)
    if not parents:
        return []
    placeholders = ",".join("?" * len(parents))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        parents,
    ).fetchall()
    present = {r["id"] for r in rows}
    return [p for p in parents if p not in present]


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
    conn: sqlite3.Connection,
    *,
    assignee: Optional[str] = None,
    status: Optional[str] = None,
    tenant: Optional[str] = None,
    brand: Optional[str] = None,
    session_id: Optional[str] = None,
    include_archived: bool = False,
    limit: Optional[int] = None,
    order_by: Optional[str] = None,
    workflow_template_id: Optional[str] = None,
    current_step_key: Optional[str] = None,
) -> list[Task]:
    query = "SELECT * FROM tasks WHERE 1=1"
    params: list[Any] = []
    if assignee is not None:
        query += " AND assignee = ?"
        params.append(_canonical_assignee(assignee))
    if status is not None:
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        query += " AND status = ?"
        params.append(status)
    if tenant is not None:
        query += " AND tenant = ?"
        params.append(tenant)
    if brand is not None:
        query += " AND brand = ?"
        params.append(brand)
    if session_id is not None:
        query += " AND session_id = ?"
        params.append(session_id)
    if workflow_template_id is not None:
        query += " AND workflow_template_id = ?"
        params.append(workflow_template_id)
    if current_step_key is not None:
        query += " AND current_step_key = ?"
        params.append(current_step_key)
    if not include_archived and status != "archived":
        query += " AND status != 'archived'"
    if order_by is not None:
        order_by = order_by.strip().lower()
        if order_by not in VALID_SORT_ORDERS:
            raise ValueError(
                f"order_by must be one of {sorted(VALID_SORT_ORDERS.keys())}"
            )
        query += f" ORDER BY {VALID_SORT_ORDERS[order_by]}"
    else:
        query += " ORDER BY priority DESC, created_at ASC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, params).fetchall()
    return [Task.from_row(r) for r in rows]


def assign_task(conn: sqlite3.Connection, task_id: str, profile: Optional[str]) -> bool:
    """Assign or reassign a task.  Returns True on success.

    Refuses to reassign a task that's currently running (claim_lock set).
    Reassign after the current run completes if needed.
    """
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
            # The retry guard is scoped to the task/profile combination. A
            # human reassigning the task is an explicit recovery action, so the
            # new profile should not inherit the previous profile's streak.
            conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, "
                "last_failure_error = NULL WHERE id = ?",
                (profile, task_id),
            )
        else:
            conn.execute("UPDATE tasks SET assignee = ? WHERE id = ?", (profile, task_id))
        _append_event(conn, task_id, "assigned", {"assignee": profile})
        return True


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------

def link_tasks(conn: sqlite3.Connection, parent_id: str, child_id: str) -> None:
    if parent_id == child_id:
        raise ValueError("a task cannot depend on itself")
    with write_txn(conn):
        missing = _find_missing_parents(conn, [parent_id, child_id])
        if missing:
            raise ValueError(f"unknown task(s): {', '.join(missing)}")
        if _would_cycle(conn, parent_id, child_id):
            raise ValueError(
                f"linking {parent_id} -> {child_id} would create a cycle"
            )
        conn.execute(
            "INSERT OR IGNORE INTO task_links (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )
        # If child was ready but parent is not yet done, demote child to todo.
        parent_status = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (parent_id,)
        ).fetchone()["status"]
        if parent_status != "done":
            conn.execute(
                "UPDATE tasks SET status = 'todo' WHERE id = ? AND status = 'ready'",
                (child_id,),
            )
        _append_event(
            conn, child_id, "linked",
            {"parent": parent_id, "child": child_id},
        )


def _would_cycle(conn: sqlite3.Connection, parent_id: str, child_id: str) -> bool:
    """Return True if adding parent->child creates a cycle.

    A cycle exists iff ``parent_id`` is already a descendant of
    ``child_id`` via existing parent->child links.  We walk downward
    from ``child_id`` and check whether we reach ``parent_id``.
    """
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
            "DELETE FROM task_links WHERE parent_id = ? AND child_id = ?",
            (parent_id, child_id),
        )
        if cur.rowcount:
            _append_event(
                conn, child_id, "unlinked",
                {"parent": parent_id, "child": child_id},
            )
        removed = cur.rowcount > 0
    if removed:
        # Dependency edge removed — re-evaluate promotion eligibility for the
        # child immediately.  Matches the contract of complete_task and
        # unblock_task; without this the child stays stuck in todo until the
        # next dispatcher tick or a manual `hermes kanban recompute` (issue #22459).
        recompute_ready(conn)
    return removed


def parent_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    return [r["parent_id"] for r in rows]


def child_ids(conn: sqlite3.Connection, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT child_id FROM task_links WHERE parent_id = ? ORDER BY child_id",
        (task_id,),
    ).fetchall()
    return [r["child_id"] for r in rows]


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


# ---------------------------------------------------------------------------
# Comments & events
# ---------------------------------------------------------------------------

def add_comment(
    conn: sqlite3.Connection, task_id: str, author: str, body: str
) -> int:
    if not body or not body.strip():
        raise ValueError("comment body is required")
    if not author or not author.strip():
        raise ValueError("comment author is required")
    now = int(time.time())
    with write_txn(conn):
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


def list_comments(conn: sqlite3.Connection, task_id: str) -> list[Comment]:
    rows = conn.execute(
        "SELECT * FROM task_comments WHERE task_id = ? ORDER BY created_at ASC",
        (task_id,),
    ).fetchall()
    return [
        Comment(
            id=r["id"],
            task_id=r["task_id"],
            author=r["author"],
            body=r["body"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

def add_attachment(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    filename: str,
    stored_path: str,
    content_type: Optional[str] = None,
    size: int = 0,
    uploaded_by: Optional[str] = None,
) -> int:
    """Record a file attachment for a task. Returns the new attachment id.

    The caller is responsible for writing the blob to ``stored_path``
    first (under :func:`task_attachments_dir`); this only persists the
    metadata row and appends an ``attached`` event.
    """
    if not filename or not filename.strip():
        raise ValueError("attachment filename is required")
    if not stored_path or not stored_path.strip():
        raise ValueError("attachment stored_path is required")
    now = int(time.time())
    with write_txn(conn):
        if not conn.execute(
            "SELECT 1 FROM tasks WHERE id = ?", (task_id,)
        ).fetchone():
            raise ValueError(f"unknown task {task_id}")
        cur = conn.execute(
            "INSERT INTO task_attachments "
            "(task_id, filename, stored_path, content_type, size, uploaded_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                filename.strip(),
                stored_path,
                content_type,
                int(size),
                uploaded_by,
                now,
            ),
        )
        _append_event(
            conn,
            task_id,
            "attached",
            {"filename": filename.strip(), "size": int(size), "by": uploaded_by},
        )
        return int(cur.lastrowid or 0)


def list_attachments(conn: sqlite3.Connection, task_id: str) -> list[Attachment]:
    rows = conn.execute(
        "SELECT * FROM task_attachments WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    return [
        Attachment(
            id=r["id"],
            task_id=r["task_id"],
            filename=r["filename"],
            stored_path=r["stored_path"],
            content_type=r["content_type"],
            size=r["size"] or 0,
            uploaded_by=r["uploaded_by"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def get_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    r = conn.execute(
        "SELECT * FROM task_attachments WHERE id = ?", (attachment_id,)
    ).fetchone()
    if r is None:
        return None
    return Attachment(
        id=r["id"],
        task_id=r["task_id"],
        filename=r["filename"],
        stored_path=r["stored_path"],
        content_type=r["content_type"],
        size=r["size"] or 0,
        uploaded_by=r["uploaded_by"],
        created_at=r["created_at"],
    )


def delete_attachment(conn: sqlite3.Connection, attachment_id: int) -> Optional[Attachment]:
    """Delete an attachment row and its on-disk blob. Returns the removed row.

    Returns ``None`` when no row matched. The blob is removed best-effort
    (a missing file is not an error); the metadata row is the source of
    truth for whether an attachment "exists".
    """
    with write_txn(conn):
        att = get_attachment(conn, attachment_id)
        if att is None:
            return None
        conn.execute("DELETE FROM task_attachments WHERE id = ?", (attachment_id,))
        _append_event(
            conn, att.task_id, "attachment_removed", {"filename": att.filename}
        )
    try:
        p = Path(att.stored_path)
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    return att


def list_events(conn: sqlite3.Connection, task_id: str) -> list[Event]:
    rows = conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY created_at ASC, id ASC",
        (task_id,),
    ).fetchall()
    out = []
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(
            Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
                run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
            )
        )
    return out


def write_delivery_state(
    conn: sqlite3.Connection,
    task_id: str,
    state: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    """Persist a normalized task-level delivery-truth snapshot."""

    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    normalized = _normalize_delivery_state(task, state)
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET delivery_state = ? WHERE id = ?",
            (json.dumps(normalized, ensure_ascii=False), task_id),
        )
        if emit_event:
            _append_event(
                conn,
                task_id,
                "delivery_state_updated",
                {
                    "stage": normalized.get("stage"),
                    "workflow_stream_id": normalized.get("workflow_stream_id"),
                    "delivery_verdict": normalized.get("delivery_verdict"),
                    "delivery_verdict_reason": normalized.get("delivery_verdict_reason"),
                    "artifact_readable": ((normalized.get("artifact") or {}).get("readable")),
                },
                run_id=run_id,
            )
    return normalized


def write_pass_loop_state(
    conn: sqlite3.Connection,
    task_id: str,
    state: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    """Persist a normalized PASS-loop convergence snapshot on the task row."""

    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    normalized = _normalize_pass_loop_state(task, state)
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET pass_loop_state = ?, pass_loop_status = ?, pass_loop_count = ?, pass_loop_reason_code = ? WHERE id = ?",
            (
                json.dumps(normalized, ensure_ascii=False),
                normalized.get("status"),
                int(normalized.get("count") or 0),
                normalized.get("reason_code"),
                task_id,
            ),
        )
        if emit_event:
            _append_event(
                conn,
                task_id,
                "pass_loop_state_updated",
                {
                    "status": normalized.get("status"),
                    "count": normalized.get("count"),
                    "threshold": normalized.get("threshold"),
                    "reason_code": normalized.get("reason_code"),
                    "block_kind": normalized.get("block_kind"),
                },
                run_id=run_id,
            )
    return normalized


def patch_task_pass_loop_state(
    conn: sqlite3.Connection,
    task_id: str,
    patch: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    base = task.pass_loop_state or {}
    merged = _deep_merge_dicts(base, patch)
    return write_pass_loop_state(conn, task_id, merged, run_id=run_id, emit_event=emit_event)


def init_task_delivery_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    stage: str,
    workflow_stream_id: str,
    artifact_ref: Optional[dict[str, Any]] = None,
    artifact_refs: Optional[Iterable[dict[str, Any]]] = None,
    risk_class: str = "medium",
    proof: Optional[dict[str, Any]] = None,
    review: Optional[dict[str, Any]] = None,
    merge: Optional[dict[str, Any]] = None,
    release: Optional[dict[str, Any]] = None,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    state = build_delivery_state(
        task,
        stage=stage,
        workflow_stream_id=workflow_stream_id,
        artifact_ref=artifact_ref,
        artifact_refs=artifact_refs,
        risk_class=risk_class,
        proof=proof,
        review=review,
        merge=merge,
        release=release,
    )
    return write_delivery_state(conn, task_id, state, run_id=run_id, emit_event=emit_event)


def patch_task_delivery_state(
    conn: sqlite3.Connection,
    task_id: str,
    patch: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    base = task.delivery_state or {}
    merged = _deep_merge_dicts(base, patch)
    return write_delivery_state(conn, task_id, merged, run_id=run_id, emit_event=emit_event)


def update_delivery_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    primary_ref: Optional[dict[str, Any]] = None,
    refs: Optional[Iterable[dict[str, Any]]] = None,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    artifact_patch: dict[str, Any] = {}
    if primary_ref is not None:
        artifact_patch["primary_ref"] = primary_ref
    if refs is not None:
        artifact_patch["refs"] = list(refs)
    return patch_task_delivery_state(
        conn,
        task_id,
        {"artifact": artifact_patch},
        run_id=run_id,
        emit_event=emit_event,
    )


def update_delivery_proof(
    conn: sqlite3.Connection,
    task_id: str,
    proof: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    return patch_task_delivery_state(
        conn, task_id, {"proof": proof}, run_id=run_id, emit_event=emit_event
    )


def update_delivery_review(
    conn: sqlite3.Connection,
    task_id: str,
    review: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    return patch_task_delivery_state(
        conn, task_id, {"review": review}, run_id=run_id, emit_event=emit_event
    )


def update_delivery_merge(
    conn: sqlite3.Connection,
    task_id: str,
    merge: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    return patch_task_delivery_state(
        conn, task_id, {"merge": merge}, run_id=run_id, emit_event=emit_event
    )


def update_delivery_release(
    conn: sqlite3.Connection,
    task_id: str,
    release: dict[str, Any],
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> dict[str, Any]:
    return patch_task_delivery_state(
        conn, task_id, {"release": release}, run_id=run_id, emit_event=emit_event
    )


def write_run_delivery_evidence(
    conn: sqlite3.Connection,
    run_id: int,
    delivery_evidence: dict[str, Any],
    *,
    merge: bool = True,
) -> dict[str, Any]:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown run {run_id}")
    try:
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
    except Exception:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    current = metadata.get("delivery_evidence")
    if merge and isinstance(current, dict):
        metadata["delivery_evidence"] = _deep_merge_dicts(current, delivery_evidence)
    else:
        metadata["delivery_evidence"] = dict(delivery_evidence)
    with write_txn(conn):
        conn.execute(
            "UPDATE task_runs SET metadata = ? WHERE id = ?",
            (json.dumps(metadata, ensure_ascii=False), run_id),
        )
    return metadata


def refresh_task_delivery_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> Optional[dict[str, Any]]:
    """Re-normalize an existing delivery-state snapshot against live task truth.

    This keeps fields derived from the task row itself (notably
    ``task_status`` and workspace provenance) aligned after lifecycle
    transitions such as complete/block/unblock.
    """

    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    if not isinstance(task.delivery_state, dict) or not task.delivery_state:
        return None
    return write_delivery_state(
        conn,
        task_id,
        task.delivery_state,
        run_id=run_id,
        emit_event=emit_event,
    )


def refresh_task_pass_loop_state(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    run_id: Optional[int] = None,
    emit_event: bool = True,
) -> Optional[dict[str, Any]]:
    """Re-normalize an existing PASS-loop snapshot against live task truth."""

    task = get_task(conn, task_id)
    if task is None:
        raise ValueError(f"unknown task {task_id}")
    if not isinstance(task.pass_loop_state, dict) or not task.pass_loop_state:
        return None
    return write_pass_loop_state(
        conn,
        task_id,
        task.pass_loop_state,
        run_id=run_id,
        emit_event=emit_event,
    )


def _is_review_required_reason(reason: Optional[str]) -> bool:
    text = str(reason or "").strip().lower()
    return bool(text) and "review-required" in text


def _is_pass_loop_approval_comment(body: object) -> bool:
    text = str(body or "").lower()
    return any(
        marker in text
        for marker in (
            '"verdict": "approve"',
            '"review_decision": "approve"',
            '"approved": true',
            '"router_action": "completed',
            '"follow_up_action": "merge',
            'auto-approved and completed',
            'review verdict: approve',
            'reviewer approve',
            'reviewer-approved',
            'approved and pushed',
        )
    )


def _pass_loop_branch_head_sha(task: Task) -> Optional[str]:
    workspace = str(task.workspace_path or "").strip()
    if not workspace:
        return None
    ws = Path(workspace).expanduser()
    if not ws.exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", str(ws), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    head = str(proc.stdout or "").strip()
    return head or None


def _latest_pass_loop_candidate(
    conn: sqlite3.Connection, task: Task
) -> Optional[dict[str, Any]]:
    blocked_event = conn.execute(
        """
        SELECT id, kind, payload, created_at
        FROM task_events
        WHERE task_id = ? AND kind LIKE 'completion_blocked_%'
        ORDER BY id DESC
        LIMIT 1
        """,
        (task.id,),
    ).fetchone()
    if blocked_event is None:
        return None
    try:
        payload = json.loads(blocked_event["payload"] or "{}") or {}
    except Exception:
        payload = {}
    blocked_created_at = int(blocked_event["created_at"] or 0)
    approval_signal = None
    for row in conn.execute(
        """
        SELECT id, body, created_at
        FROM task_comments
        WHERE task_id = ?
        ORDER BY id DESC
        """,
        (task.id,),
    ).fetchall():
        created_at = int(row["created_at"] or 0)
        if blocked_created_at and created_at > blocked_created_at:
            continue
        if _is_pass_loop_approval_comment(row["body"]):
            approval_signal = row
            break
    if approval_signal is None:
        return None
    carry_commits = payload.get("carry_commits") or []
    if not isinstance(carry_commits, list):
        carry_commits = [str(carry_commits)]
    fingerprint_payload = {
        "task_id": task.id,
        "assignee": str(task.assignee or "").strip().lower(),
        "branch_name": str(task.branch_name or "").strip(),
        "branch_head_sha": _pass_loop_branch_head_sha(task),
        "completion_block_kind": str(blocked_event["kind"] or "").strip(),
        "source_commit": payload.get("source_commit"),
        "proof_mode": payload.get("proof_mode"),
        "carry_commits": [
            str(commit).strip() for commit in carry_commits if str(commit).strip()
        ],
    }
    return {
        "approval_comment_id": int(approval_signal["id"]),
        "approval_comment_created_at": int(approval_signal["created_at"] or 0),
        "approval_excerpt": str(approval_signal["body"] or "").strip()[:280],
        "blocked_event_id": int(blocked_event["id"]),
        "blocked_event_created_at": blocked_created_at,
        "blocked_event_kind": str(blocked_event["kind"] or "").strip(),
        "blocked_message": str(payload.get("message") or "").strip(),
        "fingerprint": fingerprint_payload,
    }


def _derive_pass_loop_state(task: Task, candidate: dict[str, Any]) -> dict[str, Any]:
    previous = task.pass_loop_state if isinstance(task.pass_loop_state, dict) else {}
    previous_fingerprint = previous.get("fingerprint")
    if not isinstance(previous_fingerprint, dict):
        previous_fingerprint = {}
    previous_evidence = previous.get("evidence")
    if not isinstance(previous_evidence, dict):
        previous_evidence = {}
    previous_count = int(previous.get("count") or 0)
    resets = (
        list(previous.get("resets") or [])
        if isinstance(previous.get("resets"), list)
        else []
    )

    same_fingerprint = previous_fingerprint == candidate["fingerprint"]
    same_approval = int(previous_evidence.get("approval_comment_id") or 0) == int(
        candidate["approval_comment_id"]
    )
    same_block = int(previous_evidence.get("completion_block_event_id") or 0) == int(
        candidate["blocked_event_id"]
    )

    if same_fingerprint and not same_approval and not same_block:
        count = previous_count + 1 if previous_count > 0 else 1
    elif same_fingerprint and same_approval and same_block and previous_count > 0:
        count = previous_count
    else:
        count = 1
        if (
            previous_count > 0
            and previous_fingerprint
            and previous_fingerprint != candidate["fingerprint"]
        ):
            resets.append(
                {
                    "at": int(time.time()),
                    "from_count": previous_count,
                    "from_fingerprint": previous_fingerprint,
                    "to_fingerprint": candidate["fingerprint"],
                    "reason": "meaningful_progress_fingerprint_changed",
                    "trigger": {
                        "approval_comment_id": int(candidate["approval_comment_id"]),
                        "completion_block_event_id": int(candidate["blocked_event_id"]),
                    },
                }
            )

    status = "halted" if count >= PASS_LOOP_DEFAULT_THRESHOLD else "tracking"
    return {
        "status": status,
        "count": count,
        "threshold": PASS_LOOP_DEFAULT_THRESHOLD,
        "reason_code": PASS_LOOP_REASON_CODE if status == "halted" else None,
        "fingerprint": candidate["fingerprint"],
        "evidence": {
            "signal_path": "review_approval_comment_then_completion_block_event",
            "approval_comment_id": int(candidate["approval_comment_id"]),
            "approval_comment_created_at": int(candidate["approval_comment_created_at"]),
            "approval_excerpt": candidate["approval_excerpt"],
            "completion_block_event_id": int(candidate["blocked_event_id"]),
            "completion_block_event_created_at": int(candidate["blocked_event_created_at"]),
            "completion_block_kind": candidate["blocked_event_kind"],
            "completion_block_message": candidate["blocked_message"],
            "evidence_refs": {
                "task_comment_ids": [int(candidate["approval_comment_id"])],
                "task_event_ids": [int(candidate["blocked_event_id"])],
            },
        },
        "resets": resets,
    }


def backfill_delivery_states(
    conn: sqlite3.Connection,
    task_specs: Iterable[dict[str, Any]],
    *,
    emit_events: bool = True,
) -> list[dict[str, Any]]:
    """Bounded helper for pilot-chain backfills from verified current data."""

    written: list[dict[str, Any]] = []
    for spec in task_specs:
        task_id = str(spec.get("task_id") or "").strip()
        stage = str(spec.get("stage") or "").strip()
        workflow_stream_id = str(spec.get("workflow_stream_id") or "").strip()
        if not task_id or not stage or not workflow_stream_id:
            raise ValueError("each delivery-state backfill spec needs task_id, stage, and workflow_stream_id")
        written.append(
            init_task_delivery_state(
                conn,
                task_id,
                stage=stage,
                workflow_stream_id=workflow_stream_id,
                artifact_ref=spec.get("artifact_ref"),
                artifact_refs=spec.get("artifact_refs"),
                risk_class=str(spec.get("risk_class") or "medium"),
                proof=spec.get("proof"),
                review=spec.get("review"),
                merge=spec.get("merge"),
                release=spec.get("release"),
                run_id=spec.get("run_id"),
                emit_event=emit_events,
            )
        )
    return written


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
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_id, run_id, kind, pl, now),
    )


def _end_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    status: Optional[str] = None,
) -> Optional[int]:
    """Close the currently-active run for ``task_id`` and clear the pointer.

    ``outcome`` is the semantic result (completed / blocked / crashed /
    timed_out / spawn_failed / gave_up / reclaimed). ``status`` is the
    run-row status (usually just ``outcome``, but callers can pass it
    explicitly). Returns the closed run_id or ``None`` if no active run
    existed (e.g. a CLI user calling ``hermes kanban complete`` on a
    task that was never claimed).
    """
    now = int(time.time())
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if not row or not row["current_run_id"]:
        return None
    run_id = int(row["current_run_id"])
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
        (
            status or outcome,
            outcome,
            summary,
            error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now,
            run_id,
        ),
    )
    conn.execute(
        "UPDATE tasks SET current_run_id = NULL WHERE id = ?", (task_id,),
    )
    return run_id


def _current_run_id(conn: sqlite3.Connection, task_id: str) -> Optional[int]:
    row = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return int(row["current_run_id"]) if row and row["current_run_id"] else None


def _synthesize_ended_run(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    outcome: str,
    summary: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """Insert a zero-duration, already-closed run row.

    Used when a terminal transition happens on a task that was never
    claimed (CLI user calling ``hermes kanban complete <ready-task>
    --summary X``, or dashboard "mark done" on a ready task). Without
    this, the handoff fields (summary / metadata / error) would be
    silently dropped: ``_end_run`` is a no-op because there's no
    current run.

    The synthetic run has ``started_at == ended_at == now`` so it
    shows up in attempt history as "instant" and doesn't skew elapsed
    stats. Caller is responsible for leaving ``current_run_id`` NULL
    (or for clearing it elsewhere in the same txn) since this
    function does NOT touch the tasks row.
    """
    now = int(time.time())
    trow = conn.execute(
        "SELECT assignee, current_step_key FROM tasks WHERE id = ?",
        (task_id,),
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
            task_id, profile, step_key,
            outcome, outcome,
            summary, error,
            json.dumps(metadata, ensure_ascii=False) if metadata else None,
            now, now,
        ),
    )
    return int(cur.lastrowid or 0)


# ---------------------------------------------------------------------------
# Dependency resolution (todo -> ready)
# ---------------------------------------------------------------------------

def _dependency_state_snapshot(conn: sqlite3.Connection, task_id: str) -> dict[str, Any]:
    """Return a compact, comparable snapshot of the task's current parent state.

    Used by dependency-wait parking to distinguish a legitimate upstream
    transition (parent graph/status changed since the worker parked) from an
    unchanged state that should stay dormant instead of re-promoting into a new
    run.
    """
    parents = conn.execute(
        "SELECT t.id, t.status FROM tasks t "
        "JOIN task_links l ON l.parent_id = t.id "
        "WHERE l.child_id = ? ORDER BY t.id",
        (task_id,),
    ).fetchall()
    snapshot: list[list[str]] = []
    unresolved_parent_ids: list[str] = []
    for row in parents:
        parent_id = str(row["id"])
        status = str(row["status"])
        snapshot.append([parent_id, status])
        if status not in ("done", "archived"):
            unresolved_parent_ids.append(parent_id)
    return {
        "parents": snapshot,
        "unresolved_parent_ids": unresolved_parent_ids,
    }


def _last_dependency_wait_state(conn: sqlite3.Connection, task_id: str) -> Optional[dict[str, Any]]:
    """Return the most recent dependency-wait snapshot, if one was recorded."""
    row = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'dependency_wait' "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if not row or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("dependency_state")
    return state if isinstance(state, dict) else None


def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """Return True when ``task_id`` is sticky-blocked by an explicit
    worker/operator ``kanban_block`` call (#28712).

    A ``blocked`` status can come from two very different sources:

    * **Worker- or operator-initiated** — a worker called
      ``kanban_block(reason="review-required: ...")`` (or somebody ran
      ``hermes kanban block <id>``).  This is a deliberate handoff that
      should stay blocked until an operator unblocks it.  The block tool
      emits a ``"blocked"`` event row in ``task_events``.

    * **Circuit-breaker** — ``_record_task_failure`` tripped after
      repeated crashes / spawn failures / timeouts.  This emits
      ``"gave_up"``, *not* ``"blocked"``, and is meant to recover
      automatically once the underlying conditions change (e.g. parents
      finish, transient infra error clears).

    The cheapest signal that distinguishes the two is the most recent
    ``"blocked"`` / ``"unblocked"`` event for the task.  If the most
    recent one is ``"blocked"`` (or there is a ``"blocked"`` event and
    no ``"unblocked"`` event has fired since), the task is sticky and
    ``recompute_ready`` must *not* auto-promote it.

    Returns ``False`` when there is no such event at all (e.g. the task
    was set to ``status='blocked'`` by the circuit breaker or by direct
    DB manipulation) — preserves the pre-#28712 auto-recover semantics
    for that path.
    """
    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked') "
        "ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return bool(row) and row["kind"] == "blocked"


def recompute_ready(
    conn: sqlite3.Connection, failure_limit: int = None,
) -> int:
    """Promote ``todo`` tasks to ``ready`` when all parents are ``done`` or ``archived``.

    Returns the number of tasks promoted.  Safe to call inside or outside
    an existing transaction; it opens its own IMMEDIATE txn.

    ``blocked`` tasks are also considered for promotion (so a task
    blocked purely by a parent dependency unblocks itself when the
    parent completes), *except* in two cases:

    1. The most recent block event was a worker-initiated
       ``kanban_block`` — those stay blocked until an explicit
       ``kanban_unblock`` (#28712).

    2. The task's ``consecutive_failures`` has reached the effective
       failure limit.  This prevents infinite retry loops when a task
       repeatedly exhausts its iteration budget: without this guard the
       counter would reset on every recovery cycle and the circuit
       breaker could never trip (#35072).

    The effective failure limit resolves in the same order as the
    circuit breaker in ``_record_task_failure`` so the two never
    disagree about when a task is permanently blocked:

      1. per-task ``max_retries`` if set
      2. caller-supplied ``failure_limit`` (the dispatcher passes the
         ``kanban.failure_limit`` config value through ``dispatch_once``)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    promoted = 0
    with write_txn(conn):
        todo_rows = conn.execute(
            "SELECT id, status, consecutive_failures, max_retries, block_kind "
            "FROM tasks WHERE status IN ('todo', 'blocked')"
        ).fetchall()
        for row in todo_rows:
            task_id = row["id"]
            cur_status = row["status"]
            if cur_status == "blocked" and _has_sticky_block(conn, task_id):
                # Worker / operator asked for human review — do not
                # silently auto-recover.  ``unblock_task`` is the only
                # legitimate exit (it emits ``"unblocked"`` which flips
                # this predicate back).
                continue
            parent_state = _dependency_state_snapshot(conn, task_id)
            parents = parent_state["parents"]
            if cur_status == "todo" and row["block_kind"] == "dependency":
                last_dependency_state = _last_dependency_wait_state(conn, task_id)
                if last_dependency_state == parent_state:
                    _log.info(
                        "kanban dependency promotion suppressed: task=%s reason=unchanged_dependency_state parents=%s",
                        task_id,
                        parent_state["parents"],
                    )
                    continue
            if all(status in ("done", "archived") for _, status in parents):
                if cur_status == "blocked":
                    # Don't auto-recover tasks that have hit the
                    # circuit-breaker failure limit.  Without this
                    # guard, a task that repeatedly exhausts its
                    # iteration budget would cycle forever:
                    # block → auto-recover → respawn → budget
                    # exhausted → block → …  The counter must also
                    # be preserved so the breaker can accumulate
                    # across recovery cycles.
                    failures = int(row["consecutive_failures"] or 0)
                    task_limit = row["max_retries"]
                    effective_limit = (
                        int(task_limit) if task_limit is not None
                        else int(failure_limit)
                    )
                    if failures >= effective_limit:
                        continue
                    conn.execute(
                        "UPDATE tasks SET status = 'ready' "
                        "WHERE id = ? AND status = 'blocked'",
                        (task_id,),
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = 'ready' WHERE id = ? AND status = 'todo'",
                        (task_id,),
                    )
                _append_event(conn, task_id, "promoted", None)
                promoted += 1
    return promoted


# ---------------------------------------------------------------------------
# Claim / complete / block
# ---------------------------------------------------------------------------

def claim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
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
        # Structural invariant: never transition ready -> running while any
        # parent is not yet 'done'. This is the single enforcement point
        # regardless of which writer (create_task, link_tasks, unblock_task,
        # release_stale_claims, manual SQL) set status='ready'. If a racy
        # writer promoted a task with undone parents, demote it back to
        # 'todo' here — recompute_ready will re-promote when the parents
        # actually finish. See RCA at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        undone = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.status NOT IN ('done', 'archived') LIMIT 1",
            (task_id,),
        ).fetchone()
        if undone:
            conn.execute(
                "UPDATE tasks SET status = 'todo' "
                "WHERE id = ? AND status = 'ready'",
                (task_id,),
            )
            _append_event(
                conn, task_id, "claim_rejected",
                {"reason": "parents_not_done"},
            )
            return None
        # Defensive: if a prior run somehow leaked (invariant violation from
        # an unknown code path), close it as 'reclaimed' so we don't strand
        # it when the CAS resets the pointer below. No-op when the invariant
        # holds (the common case).
        stale = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ? AND status = 'ready'",
            (task_id,),
        ).fetchone()
        if stale and stale["current_run_id"]:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'reclaimed', outcome = 'reclaimed',
                       summary = COALESCE(summary, 'invariant recovery on re-claim'),
                       ended_at = ?,
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
                 WHERE id = ? AND ended_at IS NULL
                """,
                (now, int(stale["current_run_id"])),
            )
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'ready'
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        # Look up the current task row so we can populate the run with
        # its assignee / step / runtime cap.
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
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
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id},
            run_id=run_id,
        )
        claimed = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_claimed",
        task_id,
        board=get_current_board(),
        assignee=claimed.assignee if claimed else None,
        run_id=run_id,
    )
    return claimed


def claim_review_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> Optional[Task]:
    """Atomically transition ``review -> running``.

    Returns the claimed ``Task`` on success, ``None`` if the task was
    already claimed (or is not in ``review`` status).

    Unlike ``claim_task`` (which handles ``ready -> running``), this
    does NOT check parent dependencies — the task already passed that
    gate on its original ``todo -> ready -> running`` transition.

    Creates a new run entry so the review agent's lifecycle is tracked
    independently from the original worker run.
    """
    now = int(time.time())
    lock = claimer or _claimer_id()
    expires = now + _resolve_claim_ttl_seconds(ttl_seconds)
    with write_txn(conn):
        cur = conn.execute(
            """
            UPDATE tasks
               SET status        = 'running',
                   claim_lock    = ?,
                   claim_expires = ?,
                   started_at    = COALESCE(started_at, ?)
             WHERE id = ?
               AND status = 'review'
               AND claim_lock IS NULL
            """,
            (lock, expires, now, task_id),
        )
        if cur.rowcount != 1:
            return None
        trow = conn.execute(
            "SELECT assignee, max_runtime_seconds, current_step_key "
            "FROM tasks WHERE id = ?",
            (task_id,),
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
                task_id,
                trow["assignee"] if trow else None,
                trow["current_step_key"] if trow else None,
                lock,
                expires,
                trow["max_runtime_seconds"] if trow else None,
                now,
            ),
        )
        run_id = run_cur.lastrowid
        conn.execute(
            "UPDATE tasks SET current_run_id = ? WHERE id = ?",
            (run_id, task_id),
        )
        _append_event(
            conn, task_id, "claimed",
            {"lock": lock, "expires": expires, "run_id": run_id,
             "source_status": "review"},
            run_id=run_id,
        )
        return get_task(conn, task_id)


def heartbeat_claim(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    ttl_seconds: Optional[int] = None,
    claimer: Optional[str] = None,
) -> bool:
    """Extend a running claim.  Returns True if we still own it.

    Workers that know they'll exceed 15 minutes should call this every
    few minutes to keep ownership.
    """
    expires = int(time.time()) + _resolve_claim_ttl_seconds(ttl_seconds)
    lock = claimer or _claimer_id()
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET claim_expires = ? "
            "WHERE id = ? AND status = 'running' AND claim_lock = ?",
            (expires, task_id, lock),
        )
        if cur.rowcount == 1:
            run_id = _current_run_id(conn, task_id)
            if run_id is not None:
                conn.execute(
                    "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                    (expires, run_id),
                )
            return True
        return False


def release_stale_claims(
    conn: sqlite3.Connection,
    *,
    signal_fn=None,
) -> int:
    """Reset any ``running`` task whose claim has expired.

    A stale-by-TTL claim whose host-local worker PID is still alive is
    *extended* (with a ``claim_extended`` event) instead of being
    reclaimed. Reclaiming a live worker mid-flight produces the spawn-
    then-immediately-reclaim loop seen on slow models that spend longer
    than ``DEFAULT_CLAIM_TTL_SECONDS`` inside a single tool-free LLM
    call (#23025): no tool calls means no ``kanban_heartbeat``, even
    though the subprocess is healthy.

    Backstop (#29747 gap 3): if the worker's PID is still alive but its
    ``last_heartbeat_at`` is stale by more than
    ``DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS`` (1h), the worker has
    been making no observable progress and we reclaim anyway — even if
    ``_pid_alive`` is still true. This catches the wedged-in-a-logic-loop
    case where the process is technically running but accomplishing
    nothing. ``_touch_activity`` (run_agent.py) bridges chunk-level
    liveness into ``last_heartbeat_at`` via #31752, so any genuinely
    active worker keeps its heartbeat fresh as a side effect of normal
    API traffic. ``enforce_max_runtime`` and ``detect_crashed_workers``
    remain the upper bounds for genuinely wedged or dead workers.

    Returns the number of stale claims actually reclaimed (live-pid
    extensions don't count). Safe to call often.
    """
    now = int(time.time())
    reclaimed = 0
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
    stale = conn.execute(
        "SELECT id, claim_lock, worker_pid, claim_expires, last_heartbeat_at "
        "FROM tasks "
        "WHERE status = 'running' AND claim_expires IS NOT NULL "
        "  AND claim_expires < ?",
        (now,),
    ).fetchall()
    for row in stale:
        lock = row["claim_lock"] or ""
        host_local = lock.startswith(host_prefix)
        hb = row["last_heartbeat_at"]
        # Heartbeat staleness backstop: if we have a heartbeat at all
        # and it's older than the max-stale threshold, the worker is
        # not making observable progress.  Reclaim instead of extending,
        # even if the PID is still alive (it's likely in a logic loop).
        heartbeat_stale = (
            hb is not None
            and (now - int(hb)) > DEFAULT_CLAIM_HEARTBEAT_MAX_STALE_SECONDS
        )
        if (
            host_local
            and row["worker_pid"]
            and _pid_alive(row["worker_pid"])
            and not heartbeat_stale
        ):
            new_expires = now + _resolve_claim_ttl_seconds()
            with write_txn(conn):
                cur = conn.execute(
                    "UPDATE tasks SET claim_expires = ? "
                    "WHERE id = ? AND status = 'running' "
                    "  AND claim_lock IS ? "
                    "  AND claim_expires IS NOT NULL "
                    "  AND claim_expires < ?",
                    (new_expires, row["id"], row["claim_lock"], now),
                )
                if cur.rowcount != 1:
                    continue
                run_id = _current_run_id(conn, row["id"])
                if run_id is not None:
                    conn.execute(
                        "UPDATE task_runs SET claim_expires = ? WHERE id = ?",
                        (new_expires, run_id),
                    )
                _append_event(
                    conn, row["id"], "claim_extended",
                    {
                        "reason": "pid_alive",
                        "worker_pid": int(row["worker_pid"]),
                        "claim_lock": row["claim_lock"],
                        "claim_expires_was": int(row["claim_expires"]),
                        "claim_expires_now": new_expires,
                        "last_heartbeat_at": (
                            int(row["last_heartbeat_at"])
                            if row["last_heartbeat_at"] is not None
                            else None
                        ),
                    },
                    run_id=run_id,
                )
            continue

        termination = _terminate_reclaimed_worker(
            row["worker_pid"], row["claim_lock"], signal_fn=signal_fn,
        )
        # Never release a claim while our own worker is still alive: that would
        # spawn a duplicate beside it. Hold the claim and retry next tick.
        if _worker_survived_termination(termination):
            _defer_reclaim_for_live_worker(
                conn, row["id"], row["claim_lock"], now, termination,
                reason="ttl_expired_worker_alive",
            )
            continue
        with write_txn(conn):
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' AND claim_lock IS ? "
                "AND claim_expires IS NOT NULL AND claim_expires < ?",
                (row["id"], row["claim_lock"], now),
            )
            if cur.rowcount != 1:
                continue
            run_id = _end_run(
                conn, row["id"],
                outcome="reclaimed", status="reclaimed",
                error=f"stale_lock={row['claim_lock']}",
                metadata=termination,
            )
            payload = {
                "stale_lock": row["claim_lock"],
                "worker_pid": (
                    int(row["worker_pid"])
                    if row["worker_pid"] is not None else None
                ),
                "claim_expires": int(row["claim_expires"]),
                "last_heartbeat_at": (
                    int(row["last_heartbeat_at"])
                    if row["last_heartbeat_at"] is not None else None
                ),
                "now": now,
                "host_local": host_local,
                "heartbeat_stale": bool(heartbeat_stale),
            }
            payload.update(termination)
            _append_event(
                conn, row["id"], "reclaimed",
                payload,
                run_id=run_id,
            )
            reclaimed += 1
    return reclaimed


def reclaim_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    reason: Optional[str] = None,
    signal_fn=None,
) -> bool:
    """Operator-driven reclaim: release the claim and reset to ``ready``.

    Unlike :func:`release_stale_claims` which only acts on tasks whose
    ``claim_expires`` has passed, this function reclaims immediately
    regardless of TTL. Intended for the dashboard/CLI recovery flow
    when an operator wants to abort a running worker without waiting
    for the TTL to expire (e.g. after seeing a hallucination warning).

    Returns True if a reclaim happened, False if the task isn't in a
    reclaimable state (not running, or doesn't exist).
    """
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return False
    if row["status"] != "running" and row["claim_lock"] is None:
        # Nothing to reclaim — already ready / blocked / done.
        return False
    prev_lock = row["claim_lock"]
    termination = _terminate_reclaimed_worker(
        row["worker_pid"], prev_lock, signal_fn=signal_fn,
    )
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status IN ('running', 'ready', 'blocked') "
            "AND claim_lock IS ?",
            (task_id, prev_lock),
        )
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            error=(
                f"manual_reclaim: {reason}" if reason
                else f"manual_reclaim lock={prev_lock}"
            ),
            metadata=termination,
        )
        payload = {
            "manual": True,
            "reason": reason,
            "prev_lock": prev_lock,
        }
        payload.update(termination)
        _append_event(
            conn, task_id, "reclaimed",
            payload,
            run_id=run_id,
        )
    # Operator intervention — they've looked at the task, so the
    # consecutive-failures counter is now stale. Give the next retry
    # a fresh budget. (_clear_failure_counter opens its own write_txn,
    # so it runs after the enclosing one commits.)
    _clear_failure_counter(conn, task_id)
    return True


def reassign_task(
    conn: sqlite3.Connection,
    task_id: str,
    profile: Optional[str],
    *,
    reclaim_first: bool = False,
    reason: Optional[str] = None,
) -> bool:
    """Reassign a task, optionally reclaiming a stuck running worker first.

    This is the recovery path for "this profile's model is broken, try
    a different one". If ``reclaim_first`` is True, any active claim is
    released (via :func:`reclaim_task`) before the reassign happens;
    otherwise the function refuses to reassign a currently-running task
    and returns False (caller can retry with ``reclaim_first=True``).

    Returns True if the reassign landed. ``profile`` may be ``None`` to
    unassign entirely.
    """
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
    conn: sqlite3.Connection,
    completing_task_id: str,
    claimed_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Partition ``claimed_ids`` into (verified, phantom).

    A card is "verified" iff a row exists in ``tasks`` AND at least one
    of the following holds:

    * ``created_by`` matches the completing task's ``assignee`` profile
      (the common case: worker A spawns a card via ``kanban_create``,
      which stamps ``created_by=A``).
    * ``created_by`` matches the completing task's id (edge case where
      a worker passed its own task id as the ``created_by`` value).
    * The card is linked as a ``task_links.child`` of the completing
      task — i.e. the worker explicitly called ``kanban_create`` with
      ``parents=[<current_task>]``. This accepts cards created through
      the dashboard/CLI by a different principal but then attached to
      the completing task by the worker.

    ``phantom`` returns ids that either don't exist at all, or exist
    but don't satisfy any of the three trust conditions. The caller
    decides what to do with each bucket; this helper never mutates.
    """
    claimed = [str(x).strip() for x in (claimed_ids or []) if str(x).strip()]
    if not claimed:
        return [], []
    # Dedupe while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for cid in claimed:
        if cid not in seen:
            seen.add(cid)
            ordered.append(cid)

    row = conn.execute(
        "SELECT assignee FROM tasks WHERE id = ?", (completing_task_id,),
    ).fetchone()
    if row is None:
        # Completing task not found — nothing resolves.
        return [], ordered
    completing_assignee = row["assignee"]

    # Batch-fetch existence + created_by in one query.
    placeholders = ",".join(["?"] * len(ordered))
    rows = conn.execute(
        f"SELECT id, created_by FROM tasks WHERE id IN ({placeholders})",
        tuple(ordered),
    ).fetchall()
    found = {r["id"]: r["created_by"] for r in rows}

    # Pull the set of cards linked as children of the completing task.
    # Cheap: one query, indexed on parent_id.
    linked_children: set[str] = set(child_ids(conn, completing_task_id))

    verified: list[str] = []
    phantom: list[str] = []
    for cid in ordered:
        created_by = found.get(cid)
        if created_by is None:
            phantom.append(cid)
            continue
        # Accept if any of the three trust conditions holds.
        if completing_assignee and created_by == completing_assignee:
            verified.append(cid)
        elif created_by == completing_task_id:
            verified.append(cid)
        elif cid in linked_children:
            verified.append(cid)
        else:
            phantom.append(cid)
    return verified, phantom


# Task-id pattern used both by ``kanban_create`` (``t_<12 hex>``) and
# ``_new_task_id`` below. Kept permissive on length for forward compat:
# accept 8+ hex chars after the ``t_`` prefix.
_TASK_ID_PROSE_RE = re.compile(r"\bt_[a-f0-9]{8,}\b")


def _scan_prose_for_phantom_ids(
    conn: sqlite3.Connection,
    text: str,
) -> list[str]:
    """Regex-scan free-form text for ``t_<hex>`` references; return the
    ones that don't exist in ``tasks``.

    Used as a non-blocking advisory check on completion summaries. An
    empty return means "no suspicious references found" — either the
    text had no IDs at all, or every ID it mentioned resolves to a real
    task. Duplicates are deduped.
    """
    if not text:
        return []
    matches = _TASK_ID_PROSE_RE.findall(text)
    if not matches:
        return []
    # Dedupe preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    placeholders = ",".join(["?"] * len(unique))
    rows = conn.execute(
        f"SELECT id FROM tasks WHERE id IN ({placeholders})",
        tuple(unique),
    ).fetchall()
    existing = {r["id"] for r in rows}
    return [m for m in unique if m not in existing]


class HallucinatedCardsError(ValueError):
    """Raised by ``complete_task`` when ``created_cards`` contains ids
    that don't exist or weren't created by the completing worker.

    The phantom list is attached as ``.phantom`` for callers that want
    structured access. Kept as ``ValueError`` subclass so existing
    tool-error handlers treat it as a recoverable user error.
    """

    def __init__(self, phantom: list[str], completing_task_id: str):
        self.phantom = list(phantom)
        self.completing_task_id = completing_task_id
        super().__init__(
            f"completion blocked: claimed created_cards that do not exist "
            f"or were not created by this worker: {', '.join(phantom)}"
        )


class CompletionGateError(ValueError):
    """Raised when a task completion fails the reality-based gate.

    The message is designed for direct tool-error surfacing. Callers
    may inspect ``.details`` for structured diagnostics.
    """

    def __init__(self, message: str, *, details: Optional[dict] = None):
        self.details = dict(details or {})
        super().__init__(message)


def _ci_step_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("passed"), bool):
            return bool(value["passed"])
        if isinstance(value.get("ok"), bool):
            return bool(value["ok"])
        if isinstance(value.get("success"), bool):
            return bool(value["success"])
        if isinstance(value.get("returncode"), int):
            return value["returncode"] == 0
        if isinstance(value.get("exit_code"), int):
            return value["exit_code"] == 0
        status = str(value.get("status") or value.get("verdict") or "").strip().casefold()
        if status:
            return status in {"pass", "passed", "ok", "success", "green"}
    return False


def _completion_ci_green(metadata: Optional[dict]) -> tuple[bool, str]:
    if not isinstance(metadata, dict):
        return False, "completion blocked: worktree tasks must include metadata['ci'] with real CI results"

    ci = metadata.get("ci")
    if ci is None:
        return False, "completion blocked: worktree tasks must include metadata['ci'] with real CI results"

    if isinstance(ci, bool):
        return (
            ci,
            "completion blocked: CI verdict was red" if not ci else "",
        )

    if not isinstance(ci, dict):
        return False, "completion blocked: metadata['ci'] must be a boolean or object with typecheck/lint/tests verdicts"

    if isinstance(ci.get("passed"), bool):
        passed = bool(ci["passed"])
        return (
            passed,
            "completion blocked: CI verdict was red" if not passed else "",
        )

    required = {"typecheck": ci.get("typecheck"), "lint": ci.get("lint"), "tests": ci.get("tests")}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        return False, (
            "completion blocked: metadata['ci'] must include verdicts for "
            + ", ".join(missing)
        )

    failed = [name for name, value in required.items() if not _ci_step_passed(value)]
    if failed:
        return False, (
            "completion blocked: CI failed for " + ", ".join(failed)
        )

    return True, ""


def _block_completion(conn: sqlite3.Connection, task_id: str, kind: str, message: str, details: dict) -> None:
    payload = {"message": message, **details}
    with write_txn(conn):
        _append_event(conn, task_id, f"completion_blocked_{kind}", payload)
    raise CompletionGateError(message, details=payload)


_NOCODE_WAIVER_SUFFIXES = (".md", ".markdown", ".txt", ".rst")
_CHECKPOINT_COMMIT_PREFIX = "wip: kanban worker checkpoint"


def _completion_nocode_waiver(
    repo_root: Path, branch_name: str, base_ref: str
) -> Optional[dict]:
    """Detect a no-code task branch and return waiver details, else None.

    The CI + merge completion gates exist to stop unverified *code*
    handoffs. Research/spec/ops cards routed through a worktree produce
    no reviewable code: their branch carries at most the auto commit(s)
    made by the pre-handoff checkpoint guard, touching documentation
    files only. Blocking those cards forces operators to fabricate
    vacuous CI verdicts, so they are waived instead — with an audit
    event recording exactly what was skipped.

    A branch qualifies only when BOTH hold:
    - every commit in ``base_ref..branch`` is an auto checkpoint
      (``wip: kanban worker checkpoint …``), i.e. the worker authored
      no commits of its own; and
    - the cumulative diff vs the merge base touches only documentation
      files (``_NOCODE_WAIVER_SUFFIXES``).
    An already-merged branch with zero extra commits qualifies
    trivially (empty commit list, empty diff).
    """
    subjects = subprocess.run(
        [
            "git", "-C", str(repo_root),
            "log", "--format=%s", f"{base_ref}..{branch_name}",
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if subjects.returncode != 0:
        return None
    commit_subjects = [s for s in (subjects.stdout or "").splitlines() if s.strip()]
    if any(not s.startswith(_CHECKPOINT_COMMIT_PREFIX) for s in commit_subjects):
        return None
    files = subprocess.run(
        [
            "git", "-C", str(repo_root),
            "diff", "--name-only", f"{base_ref}...{branch_name}",
        ],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if files.returncode != 0:
        return None
    changed = [f for f in (files.stdout or "").splitlines() if f.strip()]
    if any(not f.casefold().endswith(_NOCODE_WAIVER_SUFFIXES) for f in changed):
        return None
    return {
        "checkpoint_commits": len(commit_subjects),
        "changed_files": changed,
    }


def complete_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str] = None,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
    created_cards: Optional[Iterable[str]] = None,
    expected_run_id: Optional[int] = None,
) -> bool:
    """Transition ``running|ready -> done`` and record ``result``.

    Accepts a task that is merely ``ready`` too, so a manual CLI
    completion (``hermes kanban complete <id>``) works without requiring
    a claim/start/complete sequence.

    ``summary`` and ``metadata`` are stored on the closing run (if any)
    and surfaced to downstream children via :func:`build_worker_context`.
    When ``summary`` is omitted we fall back to ``result`` so single-run
    callers do not have to pass both. ``metadata`` is a free-form dict
    (e.g. ``{"changed_files": [...], "tests_run": [...]}``) — workers
    are encouraged to use it for structured handoff facts.

    Worktree tasks are gated on the live git checkout: the branch must be
    merged into the base ref (``workspace_base_ref`` or ``main``) and the
    completion metadata must include a green ``metadata['ci']`` verdict
    for typecheck, lint, and tests. This keeps the gate anchored to the
    branch reality instead of prose quality.

    ``created_cards`` is an optional list of task ids the completing
    worker claims to have created. Each id is verified against
    ``tasks.created_by``. If any id is phantom (does not exist or was
    not created by this worker's assignee profile), completion is blocked
    with a ``HallucinatedCardsError`` and a
    ``completion_blocked_hallucination`` event is emitted so the rejected
    attempt is auditable. When all ids verify, they are recorded on the
    ``completed`` event payload.

    After a successful completion, ``summary`` and ``result`` are scanned
    for prose references like ``t_deadbeefcafe`` that do not resolve.
    Any suspected phantom references are recorded as a
    ``suspected_hallucinated_references`` event. This pass is advisory
    and never blocks.
    """
    now = int(time.time())

    # Gate: verify created_cards BEFORE the main write txn. A rejected
    # completion still needs an auditable event, so we emit it in a
    # tiny dedicated txn, then raise. The caller is responsible for
    # surfacing HallucinatedCardsError to the worker; this function
    # never mutates task state on a phantom-card rejection.
    if created_cards:
        verified_cards, phantom_cards = _verify_created_cards(
            conn, task_id, created_cards
        )
        if phantom_cards:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "completion_blocked_hallucination",
                    {
                        "phantom_cards": phantom_cards,
                        "verified_cards": verified_cards,
                        "summary_preview": (
                            (summary or result or "").strip().splitlines()[0][:200]
                            if (summary or result)
                            else None
                        ),
                    },
                )
            raise HallucinatedCardsError(phantom_cards, task_id)
    else:
        verified_cards = []

    task = get_task(conn, task_id)
    if task is None:
        return False

    workspace_snapshot = live_worker_workspace_snapshot(task)
    workspace_kind = workspace_snapshot.get("workspace_kind") or getattr(task, "workspace_kind", None)
    if workspace_kind == "worktree":
        workspace_path = workspace_snapshot.get("workspace_path") or getattr(task, "workspace_path", None)
        branch_name = workspace_snapshot.get("branch_name") or getattr(task, "branch_name", None) or f"wt/{task.id}"
        if not workspace_path:
            _block_completion(
                conn,
                task_id,
                "unverifiable_workspace",
                "completion blocked: could not locate the live worktree path for this task",
                {
                    "workspace_kind": workspace_kind,
                    "branch_name": branch_name,
                },
            )
        worktree_path = Path(str(workspace_path)).expanduser()
        if not _is_linked_worktree_checkout(worktree_path):
            _block_completion(
                conn,
                task_id,
                "unverifiable_workspace",
                "completion blocked: task workspace is not an inspectable git worktree checkout",
                {
                    "workspace_path": str(worktree_path),
                    "branch_name": branch_name,
                },
            )
        # Resolve the repo from the linked worktree checkout itself, not its
        # parent directory. Kanban worktrees often live in a board workspace
        # directory outside the repo root, so `worktree_path.parent` is just a
        # container folder with no git metadata. The checkout's own `.git`
        # file points back to the common git dir, which is enough to recover
        # the real repo root even for externally-located linked worktrees.
        repo_root = _repo_root_for_worktree_target(worktree_path)
        if repo_root is None:
            _block_completion(
                conn,
                task_id,
                "unverifiable_workspace",
                "completion blocked: could not determine the git repo for this worktree",
                {
                    "workspace_path": str(worktree_path),
                    "branch_name": branch_name,
                },
            )
        assert repo_root is not None
        base_ref = (getattr(task, "workspace_base_ref", None) or "main").strip() or "main"
        nocode_waiver = _completion_nocode_waiver(repo_root, branch_name, base_ref)
        if nocode_waiver is not None and isinstance(metadata, dict) and metadata.get("ci") is not None:
            # The waiver excuses MISSING verdicts on no-code branches; it
            # never overrides a worker-reported CI verdict. An explicitly
            # red verdict still blocks below.
            ci_ok, _ = _completion_ci_green(metadata)
            if not ci_ok:
                nocode_waiver = None
        if nocode_waiver is not None:
            # No worker-authored code on the branch: the CI + merge gates
            # are not applicable. Record the waiver so the skip is
            # auditable, then fall through to normal completion.
            with write_txn(conn):
                _append_event(
                    conn, task_id, "completion_gate_waived_nocode",
                    {
                        "workspace_path": str(worktree_path),
                        "branch_name": branch_name,
                        "base_ref": base_ref,
                        **nocode_waiver,
                    },
                )
        else:
            if not _git_is_ancestor(repo_root, branch_name, base_ref):
                _block_completion(
                    conn,
                    task_id,
                    "unmerged_branch",
                    f"completion blocked: branch {branch_name!r} is not merged into {base_ref!r} yet",
                    {
                        "workspace_path": str(worktree_path),
                        "branch_name": branch_name,
                        "base_ref": base_ref,
                    },
                )
            ci_ok, ci_message = _completion_ci_green(metadata)
            if not ci_ok:
                _block_completion(
                    conn,
                    task_id,
                    "ci_failure",
                    ci_message,
                    {
                        "workspace_path": str(worktree_path),
                        "branch_name": branch_name,
                        "base_ref": base_ref,
                    },
                )

    with write_txn(conn):
        if expected_run_id is None:
            cur = conn.execute(
                """
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
                   AND status IN ('running', 'ready', 'blocked')
                """,
                (result, now, task_id),
            )
        else:
            cur = conn.execute(
                """
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
                   AND status IN ('running', 'ready', 'blocked')
                   AND current_run_id = ?
                """,
                (result, now, task_id, int(expected_run_id)),
            )
        if cur.rowcount != 1:
            return False
        run_id = _end_run(
            conn, task_id,
            outcome="completed", status="done",
            summary=summary if summary is not None else result,
            metadata=metadata,
        )
        # If complete_task was called on a never-claimed task (ready or
        # blocked → done with no run in flight), synthesize a
        # zero-duration run so the handoff fields are persisted in
        # attempt history instead of silently lost.
        if run_id is None and (summary or metadata or result):
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=summary if summary is not None else result,
                metadata=metadata,
            )
        # Carry the handoff summary in the event payload so gateway
        # notifiers and dashboard WS consumers can render it without a
        # second SQL round-trip. First line only, 400 char cap — the
        # full summary stays on the run row.
        ev_summary = (summary if summary is not None else result) or ""
        ev_summary = ev_summary.strip().splitlines()[0][:400] if ev_summary else ""
        completed_payload: dict = {
            "result_len": len(result) if result else 0,
            "summary": ev_summary or None,
        }
        if verified_cards:
            completed_payload["verified_cards"] = verified_cards
        # Carry artifact paths in the event payload so the gateway
        # notifier can upload them as native attachments alongside the
        # completion message. Workers pass these via
        # ``kanban_complete(artifacts=[...])`` which stashes the list in
        # ``metadata["artifacts"]`` — we promote it onto the event so
        # consumers don't have to fetch the run row to find it.
        if isinstance(metadata, dict):
            md_artifacts = metadata.get("artifacts")
            if isinstance(md_artifacts, (list, tuple)):
                cleaned_artifacts = [
                    str(p).strip() for p in md_artifacts if isinstance(p, str) and str(p).strip()
                ]
                if cleaned_artifacts:
                    completed_payload["artifacts"] = cleaned_artifacts
        _append_event(
            conn, task_id, "completed",
            completed_payload,
            run_id=run_id,
        )
    refresh_task_delivery_state(conn, task_id, run_id=run_id)
    refresh_task_pass_loop_state(conn, task_id, run_id=run_id)
    # Prose-scan the summary + result for t_<hex> references that do
    # not resolve. Advisory — does not block the completion. Runs in
    # its own txn so the completion itself is already durable by the
    # time we emit the warning.
    scan_text = " ".join(filter(None, [summary, result]))
    if scan_text:
        phantom_refs = _scan_prose_for_phantom_ids(conn, scan_text)
        # Drop any phantom refs that were already flagged as verified
        # above (shouldn't happen — verified means they exist — but
        # belt-and-suspenders).
        phantom_refs = [p for p in phantom_refs if p not in set(verified_cards)]
        if phantom_refs:
            with write_txn(conn):
                _append_event(
                    conn, task_id, "suspected_hallucinated_references",
                    {
                        "phantom_refs": phantom_refs,
                        "source": "completion_summary",
                    },
                    run_id=run_id,
                )
    # Successful completion — wipe the consecutive-failures counter.
    # Failure history stays on the event log for audit; the counter
    # just tracks "is there a current pathology the breaker should
    # care about", and a success resets that question.
    _clear_failure_counter(conn, task_id)
    # Recompute ready status for dependents (separate txn so children see done).
    recompute_ready(conn)
    # Clean up the scratch workspace and any stale tmux session for the worker.
    _cleanup_workspace(conn, task_id)
    _done_task = get_task(conn, task_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_completed",
        task_id,
        board=get_current_board(),
        assignee=_done_task.assignee if _done_task else None,
        run_id=run_id,
        summary=(summary if summary is not None else result),
    )
    return True


# ---------------------------------------------------------------------------
# Workspace / tmux cleanup
# ---------------------------------------------------------------------------

def _is_managed_scratch_path(p: Path) -> bool:
    """Return True iff *p* is a strict descendant of a kanban-managed scratch root.

    A managed root is exclusively a ``workspaces/`` directory — never the
    broader kanban home, a board root, or sibling subtrees like ``logs/`` or
    ``boards/<slug>/`` itself. Allowed roots:

    * ``HERMES_KANBAN_WORKSPACES_ROOT`` when set (worker-side override
      injected by the dispatcher).
    * ``<kanban_home>/kanban/workspaces`` — legacy default-board scratch root.
    * ``<kanban_home>/kanban/boards/<slug>/workspaces`` for each board slug
      that currently exists on disk.

    The check requires strict descendancy: a path equal to one of these
    roots is NOT managed (deleting the workspaces root would wipe every
    task's scratch dir at once), and a path that resolves to ``<kanban_home>
    /kanban`` itself, ``<kanban_home>/kanban/logs``, or
    ``<kanban_home>/kanban/boards/<slug>`` is rejected because those
    subtrees hold Hermes' own DB, metadata, and logs, not task workspaces.

    Used by :func:`_cleanup_workspace` to refuse to ``shutil.rmtree`` paths
    outside Hermes-managed storage. A board ``default_workdir`` pointing at a
    real source tree can otherwise pair with ``workspace_kind='scratch'`` and
    cause task completion to delete user data (#28818).
    """
    try:
        p_abs = p.resolve(strict=False)
    except OSError:
        return False
    roots: list[Path] = []
    override = os.environ.get("HERMES_KANBAN_WORKSPACES_ROOT", "").strip()
    if override:
        try:
            roots.append(Path(override).expanduser().resolve(strict=False))
        except OSError:
            pass
    try:
        home = kanban_home()
    except OSError:
        home = None
    if home is not None:
        try:
            roots.append((home / "kanban" / "workspaces").resolve(strict=False))
        except OSError:
            pass
        try:
            boards_parent = (home / "kanban" / "boards").resolve(strict=False)
        except OSError:
            boards_parent = None
        if boards_parent is not None:
            try:
                entries = list(boards_parent.iterdir())
            except OSError:
                entries = []
            for entry in entries:
                try:
                    if not entry.is_dir():
                        continue
                except OSError:
                    continue
                try:
                    roots.append((entry / "workspaces").resolve(strict=False))
                except OSError:
                    continue
    for root in roots:
        if p_abs == root:
            continue
        try:
            if p_abs.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def _cleanup_workspace(conn: sqlite3.Connection, task_id: str) -> None:
    """Remove a task's scratch workspace dir and kill its stale tmux session.

    Called from :func:`complete_task` after the DB transaction commits.
    Best-effort — any error is swallowed so cleanup never blocks task completion.
    Only ``scratch`` workspaces are removed; ``worktree`` and ``dir`` workspaces
    are intentionally preserved.
    """
    try:
        row = conn.execute(
            "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row:
            return
        kind: Optional[str] = row["workspace_kind"]
        path: Optional[str] = row["workspace_path"]
        if kind != "scratch" or not path:
            # This task's own workspace isn't a removable scratch dir, but its
            # completion may still unblock a deferred parent scratch cleanup
            # (e.g. a 'dir' child whose scratch parent was waiting on it). #33774
            _try_cleanup_parent_workspaces(conn, task_id)
            return
        # Check if this task has children that still need the workspace.
        # If any child is not yet done/archived, defer cleanup so the
        # child can read handoff artifacts from the scratch dir (#33774).
        _active_children = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks t ON t.id = l.child_id "
            "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
            "LIMIT 1",
            (task_id,),
        ).fetchone()
        if _active_children:
            _log.debug(
                "Deferring scratch workspace cleanup for task %s: "
                "active children still need workspace at %s",
                task_id, path,
            )
            return
        import shutil
        wp = Path(path)
        if wp.is_dir():
            # Containment guard (#28818): a board's ``default_workdir`` can
            # pair ``workspace_kind='scratch'`` with a user-supplied path
            # pointing at a real source tree. Without this check, task
            # completion would unconditionally ``shutil.rmtree`` that path
            # and silently delete the user's source data.
            if _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Removed scratch workspace: %s", wp)
            else:
                _log.warning(
                    "Refusing to remove out-of-scratch workspace for task %s: %s "
                    "(workspace_kind='scratch' but path is outside any "
                    "kanban-managed workspaces root)",
                    task_id, wp,
                )
        # Also kill the tmux session for the worker that owned this task,
        # if the tmux session is now dead (worker process exited).
        _cleanup_worker_tmux(conn, task_id)
        # After cleaning up this task's workspace, check if any parent
        # tasks now have all children done — their deferred cleanup can
        # proceed (#33774).
        _try_cleanup_parent_workspaces(conn, task_id)
    except Exception:
        pass  # best-effort — never block completion


def _try_cleanup_parent_workspaces(conn: sqlite3.Connection, task_id: str) -> None:
    """Clean up parent scratch workspaces now that *task_id* completed.

    When a parent task's cleanup was deferred because it had active children,
    this function is called after each child completes.  If all children of a
    parent are now done/archived/failed/cancelled, the parent's scratch
    workspace is removed (#33774).
    """
    try:
        parents = conn.execute(
            "SELECT parent_id FROM task_links WHERE child_id = ?",
            (task_id,),
        ).fetchall()
        for (parent_id,) in parents:
            row = conn.execute(
                "SELECT workspace_kind, workspace_path FROM tasks WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
                continue
            # Check if ALL children of this parent are terminal
            active = conn.execute(
                "SELECT 1 FROM task_links l "
                "JOIN tasks t ON t.id = l.child_id "
                "WHERE l.parent_id = ? AND t.status NOT IN ('done', 'archived', 'failed', 'cancelled') "
                "LIMIT 1",
                (parent_id,),
            ).fetchone()
            if active:
                continue  # still has active children
            # All children done — safe to clean up parent workspace
            import shutil
            wp = Path(row["workspace_path"])
            if wp.is_dir() and _is_managed_scratch_path(wp):
                shutil.rmtree(wp, ignore_errors=True)
                _log.debug("Deferred cleanup: removed parent %s scratch workspace: %s", parent_id, wp)
    except Exception:
        pass  # best-effort


def _cleanup_worker_tmux(conn: sqlite3.Connection, task_id: str) -> None:
    """Kill the tmux session associated with a task's assignee, if dead."""
    try:
        row = conn.execute(
            "SELECT assignee FROM tasks WHERE id = ?", (task_id,)
        ).fetchone()
        if not row or not row["assignee"]:
            return
        assignee: str = row["assignee"]
        # Workers named swarm1-12 use tmux sessions named swarm-swarm1 etc.
        session = f"swarm-{assignee}"
        # Check if session exists and pane is dead before killing
        out = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
            capture_output=True, text=True, timeout=5,
        )
        if out.stdout.strip() == "1":
            subprocess.run(
                ["tmux", "kill-session", "-t", session],
                capture_output=True, timeout=5,
            )
            _log.debug("Killed stale tmux session: %s", session)
    except Exception:
        pass  # best-effort — never block completion


# ---------------------------------------------------------------------------
# First-use tip for scratch workspaces
# ---------------------------------------------------------------------------
#
# Scratch workspaces are intentionally ephemeral — ``_cleanup_workspace``
# removes them as soon as ``complete_task`` runs.  New users often don't
# realize that and lose worker output (community report, May 2026).  The
# behavior is right; the lack of warning is the bug.
#
# On the FIRST scratch workspace materialization across the whole install
# we:
#   1. Log a warning line on the dispatcher logger.
#   2. Append a ``tip_scratch_workspace`` event on the task so it's visible
#      via ``hermes kanban show <id>`` and the dashboard.
#   3. Touch a sentinel file under ``kanban_home() / '.scratch_tip_shown'``
#      so we don't repeat the tip — once you know, you know.
#
# Scope is per-install, not per-board: a user creating a second board
# already learned the lesson on board #1.

_SCRATCH_TIP_SENTINEL_NAME = ".scratch_tip_shown"

_SCRATCH_TIP_MESSAGE = (
    "scratch workspaces are ephemeral — they're deleted when the task "
    "completes. Use --workspace worktree: (git worktree) or "
    "--workspace dir:/abs/path (existing dir) to preserve worker output."
)


def _scratch_tip_sentinel_path() -> Path:
    """Path to the per-install scratch-workspace-tip sentinel file."""
    return kanban_home() / _SCRATCH_TIP_SENTINEL_NAME


def _scratch_tip_shown() -> bool:
    """True iff the scratch-workspace tip has already been emitted on this
    install. Best-effort — any error means we re-emit, which is the safer
    failure mode for a help message."""
    try:
        return _scratch_tip_sentinel_path().exists()
    except OSError:
        return False


def _mark_scratch_tip_shown() -> None:
    """Touch the sentinel so future scratch workspaces stay silent.

    Best-effort: a failure here just means the tip might appear once more,
    which is preferable to crashing dispatch over a help message.
    """
    try:
        path = _scratch_tip_sentinel_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    except OSError:
        pass


def _maybe_emit_scratch_tip(
    conn: sqlite3.Connection,
    task_id: str,
    workspace_kind: Optional[str],
) -> None:
    """Emit the first-use scratch-workspace tip exactly once per install.

    Called from the dispatcher right after a scratch workspace is
    materialized. No-op for ``worktree`` / ``dir`` workspaces (they're
    preserved by design) and no-op after the sentinel exists.
    """
    if (workspace_kind or "scratch") != "scratch":
        return
    if _scratch_tip_shown():
        return
    try:
        _log.warning("kanban: %s (task %s)", _SCRATCH_TIP_MESSAGE, task_id)
        with write_txn(conn):
            _append_event(
                conn, task_id, "tip_scratch_workspace",
                {"message": _SCRATCH_TIP_MESSAGE},
            )
    except Exception:
        # Best-effort — never block the spawn loop over a help message.
        pass
    finally:
        _mark_scratch_tip_shown()


def edit_completed_task_result(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: str,
    summary: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> bool:
    """Backfill the user-visible result for an already completed task."""
    handoff_summary = summary if summary is not None else result
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if not row or row["status"] != "done":
            return False
        conn.execute(
            "UPDATE tasks SET result = ? WHERE id = ?",
            (result, task_id),
        )
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
        run_id = int(run["id"]) if run else None
        if run_id is None:
            run_id = _synthesize_ended_run(
                conn, task_id,
                outcome="completed",
                summary=handoff_summary,
                metadata=metadata,
            )
        else:
            conn.execute(
                "UPDATE task_runs SET summary = ? WHERE id = ?",
                (handoff_summary, run_id),
            )
            if metadata is not None:
                conn.execute(
                    "UPDATE task_runs SET metadata = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False), run_id),
                )
        ev_summary = (
            handoff_summary.strip().splitlines()[0][:400]
            if handoff_summary else ""
        )
        _append_event(
            conn, task_id, "edited",
            {
                "fields": (
                    ["result", "summary"]
                    + (["metadata"] if metadata is not None else [])
                ),
                "result_len": len(result) if result else 0,
                "summary": ev_summary or None,
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

    Returns True on any successful transition (to ``blocked``, ``todo``, or
    ``triage``), False when the task wasn't in a blockable state.
    """
    if kind is not None and kind not in VALID_BLOCK_KINDS:
        raise ValueError(
            f"block kind must be one of {sorted(VALID_BLOCK_KINDS)} or None"
        )
    routed_to = "blocked"
    recurrences = 0
    run_id: Optional[int] = None
    pass_loop_state_to_write: Optional[dict[str, Any]] = None
    with write_txn(conn):
        cur_row = conn.execute(
            "SELECT status, block_kind, block_recurrences FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if cur_row is None:
            return False
        prev_kind = cur_row["block_kind"] if "block_kind" in cur_row.keys() else None
        prev_recurrences = (
            int(cur_row["block_recurrences"])
            if "block_recurrences" in cur_row.keys()
            and cur_row["block_recurrences"] is not None
            else 0
        )

        # Dependency blocks never enter the human ``blocked`` bucket — they
        # wait in ``todo`` and let ``recompute_ready`` gate on parents. Routing
        # here (rather than ``blocked``) is what keeps a cron from ever seeing
        # a dependency-wait as something to "unblock".
        if kind == "dependency":
            dependency_state = _dependency_state_snapshot(conn, task_id)
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
                    "dependency_state": dependency_state,
                },
                run_id=run_id,
            )
            if not dependency_state["unresolved_parent_ids"]:
                _append_event(
                    conn,
                    task_id,
                    "dependency_wait_without_unmet_parent",
                    {
                        "reason": reason,
                        "kind": kind,
                        "dependency_state": dependency_state,
                    },
                    run_id=run_id,
                )
            routed_to = "todo"
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
            pass_loop_candidate = None
            pass_loop_task = None
            if _is_review_required_reason(reason):
                pass_loop_task = get_task(conn, task_id)
                if pass_loop_task is not None:
                    pass_loop_candidate = _latest_pass_loop_candidate(conn, pass_loop_task)

            if pass_loop_candidate is not None and pass_loop_task is not None:
                pass_loop_state_to_write = _derive_pass_loop_state(
                    pass_loop_task, pass_loop_candidate
                )
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """
                    + (
                        ""
                        if expected_run_id is None
                        else " AND current_run_id = ?"
                    ),
                    (kind, recurrences, task_id)
                    if expected_run_id is None
                    else (kind, recurrences, task_id, int(expected_run_id)),
                )
                if cur.rowcount != 1:
                    return False
                run_id = _end_run(
                    conn,
                    task_id,
                    outcome="blocked",
                    status="blocked",
                    summary=reason,
                )
                if run_id is None and reason:
                    run_id = _synthesize_ended_run(
                        conn, task_id, outcome="blocked", summary=reason,
                    )
                _append_event(
                    conn,
                    task_id,
                    "block_loop_detected",
                    {
                        "reason": reason,
                        "kind": kind,
                        "recurrences": recurrences,
                        "limit": BLOCK_RECURRENCE_LIMIT,
                        "pass_loop": {
                            "signal_path": "review_approval_comment_then_completion_block_event",
                            "status": pass_loop_state_to_write.get("status"),
                            "count": pass_loop_state_to_write.get("count"),
                            "threshold": pass_loop_state_to_write.get("threshold"),
                            "reason_code": pass_loop_state_to_write.get("reason_code"),
                            "approval_comment_id": pass_loop_candidate["approval_comment_id"],
                            "completion_block_event_id": pass_loop_candidate["blocked_event_id"],
                            "completion_block_kind": pass_loop_candidate["blocked_event_kind"],
                        },
                    },
                    run_id=run_id,
                )
                routed_to = "blocked"
            else:
                # Loop detected — stop letting the unblocker spin this task. Route
                # to triage for a human-in-the-loop decision instead of blocked.
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'triage',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """ + ("" if expected_run_id is None else " AND current_run_id = ?"),
                    (kind, recurrences, task_id) if expected_run_id is None
                    else (kind, recurrences, task_id, int(expected_run_id)),
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
                    },
                    run_id=run_id,
                )
                routed_to = "triage"
        else:
            if expected_run_id is None:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                    """,
                    (kind, recurrences, task_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET status        = 'blocked',
                           claim_lock    = NULL,
                           claim_expires = NULL,
                           worker_pid    = NULL,
                           block_kind    = ?,
                           block_recurrences = ?
                     WHERE id = ?
                       AND status IN ('running', 'ready')
                       AND current_run_id = ?
                    """,
                    (kind, recurrences, task_id, int(expected_run_id)),
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
                {"reason": reason, "kind": kind, "recurrences": recurrences},
                run_id=run_id,
            )
        _blocked_task = get_task(conn, task_id)
    if pass_loop_state_to_write is not None:
        write_pass_loop_state(
            conn,
            task_id,
            pass_loop_state_to_write,
            run_id=run_id,
        )
    refresh_task_delivery_state(conn, task_id, run_id=run_id)
    refresh_task_pass_loop_state(conn, task_id, run_id=run_id)
    _fire_kanban_lifecycle_hook(
        "kanban_task_blocked",
        task_id,
        board=get_current_board(),
        assignee=_blocked_task.assignee if _blocked_task else None,
        run_id=run_id,
        reason=reason,
    )
    return True



def promote_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    actor: str,
    reason: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[bool, Optional[str]]:
    """Manually promote a `todo` or `blocked` task to `ready`.

    Mirrors the automatic promotion done by ``recompute_ready`` but
    drives it from a deliberate operator action with an audit-trail
    entry. Refuses to promote if any parent dep is not in a terminal
    state (`done`/`archived`) unless ``force=True``. Does NOT change
    assignee or claim state. Returns ``(True, None)`` on success and
    ``(False, reason)`` if refused. ``dry_run=True`` validates the
    promotion would succeed without mutating state.
    """
    row = conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return False, f"task {task_id} not found"

    cur_status = row["status"]
    if cur_status not in ("todo", "blocked"):
        return False, (
            f"task {task_id} is {cur_status!r}; promote only applies to "
            f"'todo' or 'blocked'"
        )

    if not force:
        parents = conn.execute(
            "SELECT t.id, t.status FROM tasks t "
            "JOIN task_links l ON l.parent_id = t.id "
            "WHERE l.child_id = ?",
            (task_id,),
        ).fetchall()
        unsatisfied = [
            p["id"] for p in parents
            if p["status"] not in ("done", "archived")
        ]
        if unsatisfied:
            return False, (
                f"unsatisfied parent dependencies: "
                f"{', '.join(unsatisfied)} (use --force to override)"
            )

    if dry_run:
        return True, None

    with write_txn(conn):
        upd = conn.execute(
            "UPDATE tasks SET status = 'ready' "
            "WHERE id = ? AND status IN ('todo', 'blocked')",
            (task_id,),
        )
        if upd.rowcount != 1:
            return False, f"task {task_id} status changed during promotion"
        _append_event(
            conn,
            task_id,
            "promoted_manual",
            {"actor": actor, "reason": reason, "forced": force},
        )

    return True, None


def unblock_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Transition ``blocked``/``scheduled`` -> ready or todo.

    Defensively closes any stale ``current_run_id`` pointer before flipping
    status. In the common path (``block_task`` closed the run already) this
    is a no-op. If a future or external write left the pointer dangling,
    the leaked run is closed as ``reclaimed`` inside the same txn so the
    runs invariant (``current_run_id IS NULL`` ⇔ run row in terminal
    state) holds for the rest of this function's lifetime.
    """
    now = int(time.time())
    with write_txn(conn):
        stale = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ? AND status IN ('blocked', 'scheduled')",
            (task_id,),
        ).fetchone()
        if stale and stale["current_run_id"]:
            conn.execute(
                """
                UPDATE task_runs
                   SET status = 'reclaimed', outcome = 'reclaimed',
                       summary = COALESCE(summary, 'invariant recovery on unblock'),
                       ended_at = ?,
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL
                 WHERE id = ? AND ended_at IS NULL
                """,
                (now, int(stale["current_run_id"])),
            )
        # Re-gate on parent completion before flipping 'blocked' back to
        # 'ready'. Unconditionally setting status='ready' here bypasses the
        # parent-completion invariant (the dispatcher trusts that column);
        # if parents are still in progress the task must wait in 'todo'
        # until recompute_ready picks it up. RCA: Bug 2 at
        # kanban/boards/cookai/workspaces/t_a6acd07d/root-cause.md.
        undone_parents = conn.execute(
            "SELECT 1 FROM task_links l "
            "JOIN tasks p ON p.id = l.parent_id "
            "WHERE l.child_id = ? AND p.status != 'done' LIMIT 1",
            (task_id,),
        ).fetchone()
        new_status = "todo" if undone_parents else "ready"
        # NOTE: deliberately does NOT touch ``block_recurrences`` or
        # ``block_kind``. Resetting the recurrence counter on unblock is exactly
        # the amnesia that let a cron unblock → worker re-block loop run
        # unbounded (Dale's report). The counter survives the unblock so that a
        # subsequent same-cause ``block_task`` can detect the loop and route to
        # triage at ``BLOCK_RECURRENCE_LIMIT``. It is reset to 0 only on a
        # successful completion (see ``complete_task``). ``consecutive_failures``
        # (the *dispatcher* spawn/crash/timeout counter — a different signal) is
        # still reset here, which is correct: a deliberate unblock is a fresh
        # start for the dispatcher's retry budget.
        cur = conn.execute(
            "UPDATE tasks SET status = ?, current_run_id = NULL, "
            "consecutive_failures = 0, last_failure_error = NULL "
            "WHERE id = ? AND status IN ('blocked', 'scheduled')",
            (new_status, task_id),
        )
        if cur.rowcount != 1:
            return False
        _append_event(
            conn, task_id, "unblocked",
            {"status": new_status} if new_status != "ready" else None,
        )
    refresh_task_delivery_state(conn, task_id)
    refresh_task_pass_loop_state(conn, task_id)
    return True


def specify_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    title: Optional[str] = None,
    body: Optional[str] = None,
    assignee: Optional[str] = None,
    author: Optional[str] = None,
    event_payload: Optional[dict] = None,
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
        payload: Optional[dict] = {"changed_fields": changed_fields} if changed_fields else None
        if event_payload:
            payload = {**(payload or {}), **event_payload}
        _append_event(conn, task_id, "specified", payload)
    # Outside the write_txn above, so we don't nest BEGIN IMMEDIATE — the
    # ready-promotion pass opens its own IMMEDIATE txn. This runs the same
    # logic the dispatcher would on its next tick, so a specified task
    # with no open parents flips straight to 'ready' here instead of
    # idling in 'todo' until the next sweep.
    recompute_ready(conn)
    return True


def decompose_triage_task(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    root_assignee: Optional[str],
    children: list[dict],
    author: Optional[str] = None,
    auto_promote: bool = True,
    rationale: Optional[str] = None,
    roster_snapshot: Optional[list[dict]] = None,
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
            "SELECT id, status, tenant, brand, workspace_kind, workspace_path "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if root_row is None:
            return None
        if root_row["status"] != "triage":
            return None
        tenant = root_row["tenant"]
        brand = root_row["brand"]
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
                # no lock. Leave the path unset so dispatch materializes a
                # fresh <repo>/.worktrees/<child-id> per child from the
                # board anchor.
                child_ws_path = None
            elif child_ws_kind == root_ws_kind:
                child_ws_path = root_ws_path
            else:
                child_ws_path = None
            conn.execute(
                "INSERT INTO tasks "
                "(id, title, body, assignee, status, workspace_kind, "
                " workspace_path, tenant, brand, created_at, created_by) "
                "VALUES (?, ?, ?, ?, 'todo', ?, ?, ?, ?, ?, ?)",
                (
                    new_id,
                    title,
                    body if isinstance(body, str) else None,
                    assignee,
                    child_ws_kind,
                    child_ws_path,
                    tenant,
                    brand,
                    now,
                    (author or "decomposer"),
                ),
            )
            _append_event(
                conn, new_id, "created",
                {
                    "by": author or "decomposer",
                    "from_decompose_of": task_id,
                    "tenant": tenant,
                    "brand": brand,
                },
            )
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
                "tenant": tenant,
                "brand": brand,
                "rationale": rationale,
                "roster_snapshot": roster_snapshot,
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


def archive_task(conn: sqlite3.Connection, task_id: str) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE tasks SET status = 'archived', "
            "    claim_lock = NULL, claim_expires = NULL, worker_pid = NULL "
            "WHERE id = ? AND status != 'archived'",
            (task_id,),
        )
        if cur.rowcount != 1:
            return False
        # If archive happened while a run was still in flight (e.g. user
        # archived a running task from the dashboard), close that run with
        # outcome='reclaimed' so attempt history isn't orphaned.
        run_id = _end_run(
            conn, task_id,
            outcome="reclaimed", status="reclaimed",
            summary="task archived with run still active",
        )
        _append_event(conn, task_id, "archived", None, run_id=run_id)
    # ``archived`` parents no longer block children, same as ``done``.
    # Promote newly-unblocked dependents immediately instead of waiting
    # for a later dispatcher tick.
    recompute_ready(conn)
    return True


def delete_archived_task(conn: sqlite3.Connection, task_id: str) -> bool:
    """Permanently remove an already-archived task and its related rows.

    Safety guard: only archived tasks can be deleted. Active / blocked / done
    tasks must be explicitly archived first so accidental data loss requires a
    second deliberate action.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT status FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if not row or row["status"] != "archived":
            return False
        conn.execute(
            "DELETE FROM task_links WHERE parent_id = ? OR child_id = ?",
            (task_id, task_id),
        )
        conn.execute("DELETE FROM task_comments WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_events WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))
        conn.execute("DELETE FROM kanban_notify_subs WHERE task_id = ?", (task_id,))
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


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------

def _git_toplevel(path: Path) -> Optional[Path]:
    """Return the git toplevel containing ``path``, or ``None`` if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
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
            text=True,
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
            text=True,
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
            text=True,
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
            text=True,
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
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"git worktree add failed for {target} on branch {branch_name}: {stderr}"
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
        p, _branch_name, _base_ref, _base_commit = _resolve_worktree_workspace(task, board=board)
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


def set_workspace_kind(
    conn: sqlite3.Connection, task_id: str, workspace_kind: str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_kind = ? WHERE id = ?",
            (str(workspace_kind), task_id),
        )


def set_branch_name(
    conn: sqlite3.Connection, task_id: str, branch_name: str
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET branch_name = ? WHERE id = ?",
            (str(branch_name), task_id),
        )


def set_workspace_base(
    conn: sqlite3.Connection,
    task_id: str,
    base_ref: Optional[str],
    base_commit: Optional[str],
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE tasks SET workspace_base_ref = ?, workspace_base_commit = ? WHERE id = ?",
            (base_ref, base_commit, task_id),
        )


def _git_ref_commit(path: Path, ref: str) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", ref],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0


def _git_remotes(repo_root: Path) -> set:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "remote"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {ln.strip() for ln in (result.stdout or "").splitlines() if ln.strip()}


def _resolve_worktree_base_commit(repo_root: Path, base_ref: str) -> str:
    """Resolve ``base_ref`` to a commit, fetching FIRST only if it is a
    remote-tracking ref (``<remote>/<branch>`` for a configured remote).

    A local ref (e.g. ``main``) is used as-is and NEVER triggers a network
    fetch. This is deliberate: for installs whose source of truth is local
    ``main`` and whose ``origin`` is a public fork the fleet must not contact,
    the base ref stays ``main`` and no remote is ever touched. Boards that
    genuinely track a remote can opt into ``origin/main`` via board config.
    """
    remote = base_ref.split("/", 1)[0] if "/" in base_ref else None
    if remote and remote in _git_remotes(repo_root):
        branch = base_ref.split("/", 1)[1]
        result = subprocess.run(
            ["git", "-C", str(repo_root), "fetch", remote, branch],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"git fetch {remote} {branch} failed in {repo_root}: {stderr}")
    commit = _git_ref_commit(repo_root, base_ref)
    if not commit:
        raise RuntimeError(f"worktree base ref {base_ref!r} is unavailable in {repo_root}")
    return commit


def _worktree_base_ref(board: Optional[str]) -> str:
    """The git ref new worktrees branch from. Per-board ``worktree_base_ref``
    config, defaulting to local ``main`` — workers build on the local default
    branch (the source of truth for patch-maintained installs), not a remote.
    """
    if board is not None:
        ref = (read_board_metadata(board).get("worktree_base_ref") or "").strip()
        if ref:
            return ref
    return "main"


def _board_is_git_backed(board: Optional[str]) -> bool:
    """True when the board's ``default_workdir`` resolves to a git repo root.

    Git-backed boards run workers in worktrees (branch-based, verifiable
    handoffs); other boards keep scratch/dir workspaces so research / ops
    workloads still dispatch (kinds coexist by design — see module header).
    """
    if board is None:
        return False
    wd = read_board_metadata(board).get("default_workdir")
    if not wd:
        return False
    return _repo_root_for_worktree_target(Path(str(wd)).expanduser()) is not None


def _git_toplevel(path: Path) -> Optional[Path]:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return None
    top = (result.stdout or "").strip()
    return Path(top).resolve(strict=False) if top else None


def _git_current_branch(path: Path) -> Optional[str]:
    result = subprocess.run(
        ["git", "-C", str(path), "branch", "--show-current"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return None
    branch = (result.stdout or "").strip()
    return branch or None


def _is_linked_worktree_checkout(path: Path) -> bool:
    git_dir = path / ".git"
    if not git_dir.exists():
        return False
    try:
        return git_dir.is_file() and git_dir.read_text(encoding="utf-8").startswith("gitdir:")
    except OSError:
        return False


def _repo_root_for_worktree_target(path: Path) -> Optional[Path]:
    cur = path.resolve(strict=False)
    result = subprocess.run(
        ["git", "-C", str(cur), "rev-parse", "--git-common-dir"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode == 0:
        common_dir = (result.stdout or "").strip()
        if common_dir:
            common_path = Path(common_dir)
            if not common_path.is_absolute():
                common_path = (cur / common_path).resolve(strict=False)
            return common_path.parent.resolve(strict=False)
    while True:
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            return None
        cur = cur.parent


def _ensure_git_worktree(repo_root: Path, target: Path, branch_name: str, base_ref: str) -> None:
    base_commit = _resolve_worktree_base_commit(repo_root, base_ref)
    if target.exists():
        if _is_linked_worktree_checkout(target):
            actual_branch = _git_current_branch(target)
            if actual_branch != branch_name:
                raise RuntimeError(
                    f"existing worktree {target} is on branch {actual_branch or '(detached)'} "
                    f"but expected {branch_name}"
                )
            if not _git_is_ancestor(repo_root, base_ref, branch_name):
                raise RuntimeError(
                    f"existing branch {branch_name} is not based on fresh {base_ref} {base_commit}"
                )
            return
        raise RuntimeError(f"worktree target {target} already exists but is not a linked git worktree")
    target.parent.mkdir(parents=True, exist_ok=True)
    if _git_ref_commit(repo_root, branch_name):
        if not _git_is_ancestor(repo_root, base_ref, branch_name):
            raise RuntimeError(
                f"existing branch {branch_name} is not based on fresh {base_ref} {base_commit}"
            )
        cmd = ["git", "-C", str(repo_root), "worktree", "add", str(target), branch_name]
    else:
        cmd = [
            "git", "-C", str(repo_root), "worktree", "add", "-b", branch_name, str(target), base_ref
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git worktree add failed for {target} on branch {branch_name}: {stderr}")


def _resolve_worktree_workspace(task: Task, *, board: Optional[str] = None) -> tuple[Path, str, str, str]:
    branch_name = (task.branch_name or "").strip() or f"wt/{task.id}"
    base_ref = _worktree_base_ref(board)

    def _finish(target_path: Path, resolved_branch: str, repo_root: Path) -> tuple[Path, str, str, str]:
        base_commit = _git_ref_commit(repo_root, base_ref)
        if not base_commit:
            raise RuntimeError(f"worktree base ref {base_ref!r} is unavailable in {repo_root}")
        return target_path, resolved_branch, base_ref, base_commit

    if not task.workspace_path:
        repo_root = None
        if board is not None:
            board_meta = read_board_metadata(board)
            board_default = board_meta.get("default_workdir")
            if board_default:
                repo_root = _repo_root_for_worktree_target(Path(str(board_default)).expanduser())
        if repo_root is None:
            raise ValueError(
                f"task {task.id} has workspace_kind=worktree but no workspace_path, "
                "and no git repo could be discovered from the board default_workdir"
            )
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name, base_ref)
        return _finish(target, branch_name, repo_root)

    requested = Path(task.workspace_path).expanduser()
    if not requested.is_absolute():
        raise ValueError(
            f"task {task.id} has non-absolute worktree path {task.workspace_path!r}; use an absolute path"
        )
    requested_resolved = requested.resolve(strict=False)
    if requested.exists() and _is_linked_worktree_checkout(requested):
        actual_branch = _git_current_branch(requested)
        if actual_branch != branch_name:
            # Another task's checkout occupies this path (decompose children
            # used to inherit the root's workspace_path verbatim, so siblings
            # collide here; stale rows still carry shared paths). Rather than
            # failing the spawn — or silently reusing the other task's
            # branch, which cross-contaminates provenance — fall back to a
            # fresh worktree of our own under the same repo.
            fallback_root = _repo_root_for_worktree_target(requested)
            fallback = (
                fallback_root / ".worktrees" / task.id
                if fallback_root is not None
                else None
            )
            if (
                fallback is None
                or fallback.resolve(strict=False) == requested_resolved
            ):
                # No repo to anchor a fallback on, or the occupied path IS
                # this task's own canonical worktree — that is real
                # corruption; keep failing loudly.
                raise ValueError(
                    f"task {task.id} worktree path {task.workspace_path!r} is already on branch {actual_branch or '(detached)'} but expected {branch_name}"
                )
            _ensure_git_worktree(fallback_root, fallback, branch_name, base_ref)
            return _finish(fallback.resolve(strict=False), branch_name, fallback_root)
        repo_root = _repo_root_for_worktree_target(requested.parent)
        if repo_root is None:
            raise ValueError(
                f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo"
            )
        _ensure_git_worktree(repo_root, requested, branch_name, base_ref)
        return _finish(requested_resolved, actual_branch or branch_name, repo_root)
    repo_root = _git_toplevel(requested)
    if repo_root is not None and requested_resolved == repo_root:
        target = repo_root / ".worktrees" / task.id
        _ensure_git_worktree(repo_root, target, branch_name, base_ref)
        return _finish(target, branch_name, repo_root)
    repo_root = _repo_root_for_worktree_target(requested.parent)
    if repo_root is None:
        raise ValueError(
            f"task {task.id} worktree path {task.workspace_path!r} is not inside a git repo and does not point at a git repo root"
        )
    _ensure_git_worktree(repo_root, requested, branch_name, base_ref)
    return _finish(requested_resolved, branch_name, repo_root)


# ---------------------------------------------------------------------------
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
    """
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
    route_watchdog_hits: list[tuple[str, str, str]] = field(default_factory=list)
    """Route-watchdog decisions recorded during dispatch, as
    ``(task_id, kind, action)`` triples. The watchdog is fail-open; this is
    telemetry and review signal, not a hard dependency on spawning."""


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
    * ``"nonzero_exit"`` — ``WIFEXITED`` with non-zero status. Real error.
    * ``"signaled"`` — ``WIFSIGNALED`` (OOM killer, SIGKILL, etc). Real crash.
    * ``"unknown"`` — pid was not in the reap registry (either reaped by
      something else, or died between reap tick and liveness check). Fall
      back to existing crashed-counter behavior.

    ``code`` is the exit status (for ``clean_exit`` / ``rate_limited`` /
    ``nonzero_exit``) or the signal number (for ``signaled``), or ``None``
    for ``unknown``.
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
                text=True,
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
    ``timed_out`` event and drops the task back to ``ready`` so the next
    dispatcher tick re-spawns it — unless the spawn-failure circuit
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
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (tid, pid, row["claim_lock"]),
            )
            if cur.rowcount == 1:
                payload = {
                    "pid": pid,
                    "elapsed_seconds": int(elapsed),
                    "limit_seconds": int(row["max_runtime_seconds"]),
                    "sigkill": killed,
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
        # breaker trips, this flips the task ``ready → blocked`` and
        # emits a ``gave_up`` event on top of the ``timed_out`` we
        # already emitted.
        if cur.rowcount == 1:
            _record_task_failure(
                conn, tid,
                error=f"elapsed {int(elapsed)}s > limit {int(row['max_runtime_seconds'])}s",
                outcome="timed_out",
                release_claim=False,
                end_run=False,
                event_payload_extra={"pid": pid, "sigkill": killed},
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

    On reclaim the task is reset to ``ready``, the run is closed with
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
    host_prefix = f"{_claimer_id().split(':', 1)[0]}:"
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
            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL, "
                "last_heartbeat_at = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND claim_lock IS ?",
                (tid, row["claim_lock"]),
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
        # going straight back to ``ready`` for re-dispatch. Counting it as
        # a worker failure would let two legitimately-long-running tasks
        # (>4h without explicit heartbeat) trip the circuit breaker and
        # auto-block, even though no worker actually failed. The 'stale'
        # event already lives in task_events for auditability; that's the
        # right surface for "this happened" without conflating with the
        # spawn_failed / timed_out / crashed counters.

    return reclaimed


def _error_fingerprint(error_text: str) -> str:
    """Normalize an error message for grouping identical failures.

    Strips host-specific details (PIDs, timestamps) so that errors
    with the same root cause produce the same fingerprint.
    """
    fp = re.sub(r'\bpid \d+\b', 'pid N', error_text[:80])
    fp = re.sub(r'\b\d{10,}\b', '<TS>', fp)
    return fp.lower().strip()


def detect_crashed_workers(conn: sqlite3.Connection) -> list[str]:
    """Reclaim ``running`` tasks whose worker PID is no longer alive.

    Appends a ``crashed`` event and drops the task back to ``ready``.
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
    to ``ready`` WITHOUT counting a failure (so a long quota window can't
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
    # clean-exit-but-still-running case so we can trip the breaker
    # immediately instead of incrementing by 1.
    crash_details: list[tuple[str, int, str, bool, str]] = []
    # (task_id, pid, claimer, protocol_violation, error_text)
    with write_txn(conn):
        rows = conn.execute(
            "SELECT id, worker_pid, claim_lock, started_at FROM tasks "
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
                # ``kanban_complete`` / ``kanban_block``. Retrying won't
                # help.
                protocol_violation = True
                error_text = (
                    "worker exited cleanly (rc=0) without calling "
                    "kanban_complete or kanban_block — protocol violation"
                )
                event_kind = "protocol_violation"
                event_payload = {
                    "pid": pid,
                    "claimer": row["claim_lock"],
                    "exit_code": code,
                }
            elif kind == "rate_limited":
                # Worker bailed because the provider rate-limited / exhausted
                # quota (EX_TEMPFAIL sentinel). This is NOT a task failure —
                # the task is fine, the account just hit a wall. Release it
                # back to ``ready`` so the respawn guard defers it until the
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
                else:
                    error_text = f"pid {pid} not alive"
                event_kind = "crashed"
                event_payload = {"pid": pid, "claimer": row["claim_lock"]}
                if code is not None and kind != "unknown":
                    event_payload["exit_kind"] = kind
                    event_payload["exit_code"] = code

            cur = conn.execute(
                "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                "claim_expires = NULL, worker_pid = NULL "
                "WHERE id = ? AND status = 'running' "
                "  AND worker_pid = ? AND claim_lock IS ?",
                (row["id"], pid, row["claim_lock"]),
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
                    crashed.append(row["id"])
                    crash_details.append(
                        (row["id"], pid, row["claim_lock"],
                         protocol_violation, error_text)
                    )
    # Outside the main txn: increment the unified failure counter for
    # each crashed task. If the breaker trips, the task transitions
    # ready → blocked with a ``gave_up`` event on top of the ``crashed``
    # event we already emitted.
    #
    # Protocol-violation crashes force an immediate trip (failure_limit=1)
    # because clean-exit-without-transition is deterministic: the next
    # respawn will do exactly the same thing. Better to surface to a
    # human with a clear reason than to loop ``DEFAULT_FAILURE_LIMIT``
    # times first.
    auto_blocked: list[str] = []
    if crash_details:
        # Fingerprint errors to detect systemic failures.
        _fp_counts: dict[str, int] = {}
        for _, _, _, _, err_text in crash_details:
            fp = _error_fingerprint(err_text)
            _fp_counts[fp] = _fp_counts.get(fp, 0) + 1
        for tid, pid, claimer, protocol_violation, error_text in crash_details:
            fp = _error_fingerprint(error_text)
            is_systemic = (
                not protocol_violation
                and _fp_counts.get(fp, 0) >= 3
            )
            tripped = _record_task_failure(
                conn, tid,
                error=error_text,
                outcome="crashed",
                failure_limit=1 if (protocol_violation or is_systemic) else None,
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
    return crashed


def _record_task_failure(
    conn: sqlite3.Connection,
    task_id: str,
    error: str,
    *,
    outcome: str,
    failure_limit: int = None,
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
      it back to ``ready`` (or ``blocked`` when the breaker trips),
      releases the claim, and closes the run with ``outcome=<outcome>``.

    * ``release_claim=False, end_run=False`` — timeout/crash path.
      Caller has ALREADY flipped the task to ``ready`` and closed the
      run with the appropriate outcome. This just increments the
      counter; if the breaker trips, the task is re-transitioned
      ``ready → blocked`` and a ``gave_up`` event is emitted.

    ``event_payload_extra`` merges into the ``gave_up`` event payload
    when the breaker trips, so callers can include outcome-specific
    context (e.g. pid on crash, elapsed on timeout).

    Resolution order for the effective threshold:
      1. per-task ``max_retries`` if set (nothing else overrides)
      2. caller-supplied ``failure_limit`` (gateway passes the config
         value from ``kanban.failure_limit``; tests pass fixed values)
      3. ``DEFAULT_FAILURE_LIMIT``
    """
    if failure_limit is None:
        failure_limit = DEFAULT_FAILURE_LIMIT
    blocked = False
    with write_txn(conn):
        row = conn.execute(
            "SELECT consecutive_failures, status, max_retries "
            "FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None:
            return False
        failures = int(row["consecutive_failures"]) + 1
        cur_status = row["status"]

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

        if failures >= effective_limit:
            # Trip the breaker.
            if release_claim:
                # Spawn path: still running, also clear claim state.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('running', 'ready')",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: task is already at ``ready``
                # with claim cleared; just flip to blocked + update
                # counter fields.
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status IN ('ready', 'running')",
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
                    },
                )
            payload = {
                "failures": failures,
                "effective_limit": effective_limit,
                "limit_source": limit_source,
                "error": error[:500],
                "trigger_outcome": outcome,
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
                # Spawn path: transition running → ready + clear claim.
                conn.execute(
                    "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
                    "claim_expires = NULL, worker_pid = NULL, "
                    "consecutive_failures = ?, last_failure_error = ? "
                    "WHERE id = ? AND status = 'running'",
                    (failures, error[:500], task_id),
                )
            else:
                # Timeout/crash path: task is already at ``ready`` via
                # its own UPDATE. Just bookkeep the counter + last error.
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
                    metadata={"failures": failures},
                )
                _append_event(
                    conn, task_id, outcome,
                    {"error": error[:500], "failures": failures},
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


def check_respawn_guard(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready task in ``dispatch_once`` before any claim attempt.
    Returning a reason defers the spawn this tick; the task stays in
    ``ready`` and gets another chance on the next dispatcher tick.

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
        seconds.  Useful work already succeeded for this task; wait for
        human review rather than immediately re-spawning.

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

    # 3. Completed run within guard window — proof of recent success.
    cutoff = now - _RESPAWN_GUARD_SUCCESS_WINDOW
    if conn.execute(
        "SELECT id FROM task_runs "
        "WHERE task_id = ? AND outcome = 'completed' AND ended_at >= ?",
        (task_id, cutoff),
    ).fetchone():
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
    route_watchdog: Optional[dict] = None,
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
        return _dispatch_once_locked(
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
            route_watchdog=route_watchdog,
        )
    with _dispatch_tick_lock(db_path) as held:
        if not held:
            return DispatchResult(skipped_locked=True)
        return _dispatch_once_locked(
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
            route_watchdog=route_watchdog,
        )


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
    route_watchdog: Optional[dict] = None,
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

    ``spawn_fn`` defaults to ``_default_spawn``. Tests pass a stub.
    ``board`` pins workspace/log/db resolution for this tick to a specific
    board. When omitted, the current-board resolution chain is used.
    """
    # Reap zombie children from previously spawned workers. See
    # reap_worker_zombies() for the full rationale.
    reap_worker_zombies()

    result = DispatchResult()
    result.reclaimed = release_stale_claims(conn)
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
    if max_spawn is not None:
        running_count = int(
            conn.execute(
                "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
            ).fetchone()[0]
        )

    route_watchdog_cfg = None
    route_watchdog_eval = None
    route_watchdog_board_meta = None
    if route_watchdog is not None:
        try:
            from hermes_cli.kanban_route_watchdog import evaluate_route as route_watchdog_eval_fn
            from hermes_cli.kanban_route_watchdog import load_watchdog_config

            route_watchdog_cfg = load_watchdog_config(route_watchdog)
            if route_watchdog_cfg.enabled:
                route_watchdog_eval = route_watchdog_eval_fn
                route_watchdog_board_meta = read_board_metadata(board)
        except Exception:
            route_watchdog_cfg = None
            route_watchdog_eval = None
            route_watchdog_board_meta = None

    def _apply_route_watchdog(task_id: str) -> bool:
        """Best-effort route watchdog hook.

        Returns True when the watchdog already held the task and the normal
        spawn path should stop. Never raises.
        """
        if route_watchdog_cfg is None or route_watchdog_eval is None:
            return False
        try:
            task = get_task(conn, task_id)
            if task is None:
                return False
            child_ids = [
                row["child_id"]
                for row in conn.execute(
                    "SELECT child_id FROM task_links WHERE parent_id = ?",
                    (task_id,),
                )
            ]
            decision = route_watchdog_eval(
                task,
                config=route_watchdog_cfg,
                board_meta=route_watchdog_board_meta,
                child_ids=child_ids,
            )
            if decision is None:
                return False
            action = "commented"
            if not dry_run:
                try:
                    add_comment(conn, task_id, "route-watchdog", decision.comment_body())
                except Exception:
                    pass
                if route_watchdog_cfg.mode == "hold":
                    try:
                        if block_task(conn, task_id, reason=decision.review_reason()):
                            action = "blocked"
                    except Exception:
                        pass
            else:
                action = "would_block" if route_watchdog_cfg.mode == "hold" else "would_comment"
            result.route_watchdog_hits.append((task_id, decision.kind, action))
            if dry_run:
                return route_watchdog_cfg.mode == "hold"
            return route_watchdog_cfg.mode == "hold" and action == "blocked"
        except Exception:
            return False

    ready_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'ready' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    # Honour kanban.max_in_progress: if the board already has enough running
    # tasks, skip spawning this tick so slow workers (local LLMs,
    # resource-constrained hosts) can finish what they have before more tasks
    # pile up and time out.
    if max_in_progress is not None and ready_rows:
        in_progress = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
        if in_progress >= max_in_progress:
            return result
        # Only spawn enough to reach the cap, respecting max_spawn too.
        remaining = max_in_progress - in_progress
        if max_spawn is None or max_spawn > remaining:
            max_spawn = remaining
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
        if max_spawn is not None and running_count + spawned >= max_spawn:
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
        if _apply_route_watchdog(row["id"]):
            continue
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
        # Git-backed boards (default_workdir resolves to a repo) run workers in
        # worktrees for branch-based, verifiable handoffs; a worktree-kind task
        # always does. Other boards keep scratch/dir so research/ops still runs.
        # goal_mode roots are coordination loops (they fan out / wake on child
        # completion) and produce no code of their own, so they stay scratch
        # even on a git-backed board — otherwise the worktree completion gate
        # (which requires real CI on a code branch) blocks them forever. An
        # explicit workspace_kind=worktree still wins for the rare goal card
        # that genuinely edits code.
        use_worktree = claimed.workspace_kind == "worktree" or (
            _board_is_git_backed(board)
            and not getattr(claimed, "goal_mode", False)
            and not getattr(claimed, "workspace_pinned", False)
        )
        try:
            if use_worktree:
                workspace, resolved_branch_name, _base_ref, _base_commit = _resolve_worktree_workspace(
                    claimed, board=board
                )
            else:
                workspace = resolve_workspace(claimed, board=board)
                resolved_branch_name = _base_ref = _base_commit = None
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
        if use_worktree:
            set_workspace_kind(conn, claimed.id, "worktree")
            set_branch_name(conn, claimed.id, resolved_branch_name)
            set_workspace_base(conn, claimed.id, _base_ref, _base_commit)
        _maybe_emit_scratch_tip(
            conn, claimed.id, "worktree" if use_worktree else (claimed.workspace_kind or "scratch")
        )
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
    # sdlc-review skill) that verifies the PR and either merges (→ done)
    # or rejects (→ back to running for the worker to fix).
    #
    # Same concurrency model as ready dispatch: review spawns count
    # against max_spawn alongside ready tasks, so the total number of
    # running workers stays bounded.
    review_rows = conn.execute(
        "SELECT id, assignee FROM tasks "
        "WHERE status = 'review' AND claim_lock IS NULL "
        "ORDER BY priority DESC, created_at ASC"
    ).fetchall()
    for row in review_rows:
        if max_spawn is not None and running_count + spawned >= max_spawn:
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
        if dry_run:
            result.spawned.append((row["id"], row["assignee"], ""))
            continue
        claimed = claim_review_task(conn, row["id"], ttl_seconds=ttl_seconds)
        if claimed is None:
            continue
        # goal_mode roots are coordination loops (they fan out / wake on child
        # completion) and produce no code of their own, so they stay scratch
        # even on a git-backed board — otherwise the worktree completion gate
        # (which requires real CI on a code branch) blocks them forever. An
        # explicit workspace_kind=worktree still wins for the rare goal card
        # that genuinely edits code.
        use_worktree = claimed.workspace_kind == "worktree" or (
            _board_is_git_backed(board)
            and not getattr(claimed, "goal_mode", False)
            and not getattr(claimed, "workspace_pinned", False)
        )
        try:
            if use_worktree:
                workspace, resolved_branch_name, _base_ref, _base_commit = _resolve_worktree_workspace(
                    claimed, board=board
                )
            else:
                workspace = resolve_workspace(claimed, board=board)
                resolved_branch_name = _base_ref = _base_commit = None
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
        if use_worktree:
            set_workspace_kind(conn, claimed.id, "worktree")
            set_branch_name(conn, claimed.id, resolved_branch_name)
            set_workspace_base(conn, claimed.id, _base_ref, _base_commit)
        _maybe_emit_scratch_tip(
            conn, claimed.id, "worktree" if use_worktree else (claimed.workspace_kind or "scratch")
        )
        # Force-load the sdlc-review skill for review agents — it carries
        # the review logic (AC verification, merge, etc.). The mandatory
        # kanban lifecycle is already injected into every worker's system
        # prompt via KANBAN_GUIDANCE, so this is the only extra skill the
        # review agent needs.
        claimed.skills = ["sdlc-review"]
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
            result.spawned.append((claimed.id, claimed.assignee or "", str(workspace)))
            spawned += 1
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

    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
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
    worker_toolsets = _resolve_worker_cli_toolsets(env.get("HERMES_HOME"))
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend([
        "chat",
        "-q", prompt,
    ])
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
            with contextlib.closing(connect()) as conn:
                res = dispatch_once(
                    conn,
                    max_spawn=max_spawn,
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
    """Return the full text a worker should read to understand its task.

    Order:
      1. Task title (mandatory).
      2. Task body (optional opening post, capped at 8 KB).
      3. Prior attempts on THIS task (most recent ``_CTX_MAX_PRIOR_ATTEMPTS``
         shown; older attempts collapsed into a one-line summary).
         Each attempt's ``summary`` / ``error`` / ``metadata`` capped at
         ``_CTX_MAX_FIELD_BYTES`` each.
      4. Structured handoff results of every done parent task. Prefers
         ``run.summary`` / ``run.metadata`` when the parent was executed
         via a run; falls back to ``task.result`` for older data. Same
         per-field cap.
      5. Cross-task role history for the assignee (most recent 5
         completed runs on other tasks).
      6. Comment thread (most recent ``_CTX_MAX_COMMENTS`` shown, older
         collapsed).

    All caps exist so worker prompts stay bounded even on pathological
    boards (retry-heavy tasks, comment storms). The per-field char cap
    prevents a single 1 MB summary from dominating context.
    """
    task = get_task(conn, task_id)
    if not task:
        raise ValueError(f"unknown task {task_id}")

    # Single clock reading shared by every relative-age stamp below, so all
    # ages in one rendering are consistent ("3h ago" / "3h ago", not drifting
    # by the seconds it takes to build the block).
    _now = int(time.time())

    live_workspace = os.environ.get("HERMES_KANBAN_WORKSPACE")
    workspace_kind = task.workspace_kind
    workspace_path = task.workspace_path
    branch_name = task.branch_name
    if live_workspace:
        live_path = Path(live_workspace).expanduser()
        if live_path.exists() and _is_linked_worktree_checkout(live_path):
            workspace_kind = "worktree"
            workspace_path = str(live_path)
            branch_name = _git_current_branch(live_path) or branch_name

    def _cap(s: Optional[str], limit: int = _CTX_MAX_FIELD_BYTES) -> str:
        """Truncate a string to `limit` chars with a visible ellipsis."""
        if not s:
            return ""
        s = s.strip()
        if len(s) <= limit:
            return s
        return s[:limit] + f"… [truncated, {len(s) - limit} chars omitted]"

    lines: list[str] = []
    lines.append(f"# Kanban task {task.id}: {task.title}")
    lines.append("")
    lines.append(f"Assignee: {task.assignee or '(unassigned)'}")
    lines.append(f"Status:   {task.status}")
    if task.tenant:
        lines.append(f"Tenant:   {task.tenant}")
    lines.append(f"Workspace: {workspace_kind} @ {workspace_path or '(unresolved)'}")
    if task.max_runtime_seconds is not None:
        terminal_timeout = _worker_terminal_timeout_env(
            task.max_runtime_seconds,
            os.environ.get("TERMINAL_TIMEOUT"),
        )
        effective_terminal_timeout = terminal_timeout or os.environ.get("TERMINAL_TIMEOUT")
        lines.append(f"Max runtime: {task.max_runtime_seconds}s")
        if effective_terminal_timeout:
            lines.append(f"Terminal timeout: {effective_terminal_timeout}s")
    if branch_name:
        lines.append(f"Branch:   {branch_name}")
    lines.append("")

    if task.body and task.body.strip():
        lines.append("## Body")
        lines.append(_cap(task.body, _CTX_MAX_BODY_BYTES))
        lines.append("")

    if task.delivery_state:
        lines.append("## Delivery state")
        try:
            state_text = json.dumps(task.delivery_state, ensure_ascii=False, sort_keys=True)
        except Exception:
            state_text = str(task.delivery_state)
        lines.append(f"`{_cap(state_text)}`")
        lines.append("")

    # Attachments — files uploaded to this task (PDFs, source docs,
    # images). Surface the absolute on-disk path so the worker, which has
    # full file-tool access, can read them directly (read_file, terminal
    # `pdftotext`, etc.). On the local terminal backend the path resolves
    # as-is; remote backends need the kanban attachments dir mounted.
    attachments = list_attachments(conn, task_id)
    if attachments:
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

    # Prior attempts — show closed runs so a retrying worker sees the
    # history. Skip the currently-active run (that's this worker).
    # Cap at _CTX_MAX_PRIOR_ATTEMPTS most-recent closed runs; older
    # attempts get collapsed into a one-line marker so the worker knows
    # more exist without bloating the prompt.
    all_prior = [r for r in list_runs(conn, task_id) if r.ended_at is not None]
    # list_runs returns ascending by started_at; "most recent" = last N
    if len(all_prior) > _CTX_MAX_PRIOR_ATTEMPTS:
        omitted = len(all_prior) - _CTX_MAX_PRIOR_ATTEMPTS
        shown = all_prior[-_CTX_MAX_PRIOR_ATTEMPTS:]
        first_shown_idx = omitted + 1
    else:
        omitted = 0
        shown = all_prior
        first_shown_idx = 1
    if shown:
        lines.append("## Prior attempts on this task")
        if omitted:
            lines.append(
                f"_({omitted} earlier attempt{'s' if omitted != 1 else ''} "
                f"omitted; showing most recent {len(shown)})_"
            )
        for offset, run in enumerate(shown):
            idx = first_shown_idx + offset
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(run.started_at))
            age = _relative_age(run.started_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            profile = run.profile or "(unknown)"
            outcome = run.outcome or run.status
            lines.append(f"### Attempt {idx} — {outcome} ({profile}, {ts_disp})")
            if run.summary and run.summary.strip():
                lines.append(_cap(run.summary))
            if run.error and run.error.strip():
                lines.append(f"_error_: {_cap(run.error)}")
            if run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.append("")

    # Parents: prefer the most-recent 'completed' run's summary + metadata,
    # fall back to ``task.result`` when no run rows exist (legacy DBs,
    # or tasks completed before the runs table landed).
    parent_rows = conn.execute(
        "SELECT parent_id FROM task_links WHERE child_id = ? ORDER BY parent_id",
        (task_id,),
    ).fetchall()
    parent_ids = [r["parent_id"] for r in parent_rows]

    if parent_ids:
        wrote_header = False
        for pid in parent_ids:
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

            # When did this parent's result get produced? Prefer the
            # completed run's end time; fall back to the task's completed_at.
            done_ts = None
            if run is not None and getattr(run, "ended_at", None):
                done_ts = run.ended_at
            elif pt.completed_at:
                done_ts = pt.completed_at
            age = _relative_age(done_ts, _now)
            lines.append(f"### {pid}" + (f" (completed {age})" if age else ""))

            body_lines: list[str] = []
            if run is not None and run.summary and run.summary.strip():
                body_lines.append(_cap(run.summary))
            elif pt.result:
                body_lines.append(_cap(pt.result))
            else:
                body_lines.append("(no result recorded)")

            if run is not None and run.metadata:
                try:
                    meta_str = json.dumps(run.metadata, ensure_ascii=False, sort_keys=True)
                    body_lines.append(f"_metadata_: `{_cap(meta_str)}`")
                except Exception:
                    pass
            lines.extend(body_lines)
            lines.append("")

    # Cross-task role history: what else has THIS assignee completed
    # recently? Gives the worker implicit continuity — "I'm the reviewer
    # and my last three reviews focused on security" — without forcing
    # the user to wire anything into SOUL.md / MEMORY.md. Bounded to the
    # most recent 5 completed runs, excluding this task so the retry
    # section above isn't duplicated. Safe on assignee=None (skipped).
    if task.assignee:
        role_rows = conn.execute(
            "SELECT t.id, t.title, r.summary, r.ended_at "
            "FROM task_runs r JOIN tasks t ON r.task_id = t.id "
            "WHERE r.profile = ? AND r.task_id != ? "
            "  AND r.outcome = 'completed' "
            "ORDER BY r.ended_at DESC LIMIT 5",
            (task.assignee, task_id),
        ).fetchall()
        if role_rows:
            lines.append(f"## Recent work by @{task.assignee}")
            for row in role_rows:
                ts = time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(int(row["ended_at"]))
                )
                age = _relative_age(row["ended_at"], _now)
                ts_disp = f"{ts}, {age}" if age else ts
                s = (row["summary"] or "").strip().splitlines()
                first = s[0][:200] if s else "(no summary)"
                lines.append(f"- {row['id']} — {row['title']} ({ts_disp}): {first}")
            lines.append("")

    # Comments: cap at the most-recent _CTX_MAX_COMMENTS so
    # comment-storm tasks don't blow out the worker's prompt. Older
    # comments summarised in a one-line marker like prior attempts.
    all_comments = list_comments(conn, task_id)
    if len(all_comments) > _CTX_MAX_COMMENTS:
        omitted_c = len(all_comments) - _CTX_MAX_COMMENTS
        shown_c = all_comments[-_CTX_MAX_COMMENTS:]
    else:
        omitted_c = 0
        shown_c = all_comments
    if shown_c:
        lines.append("## Comment thread")
        if omitted_c:
            lines.append(
                f"_({omitted_c} earlier comment{'s' if omitted_c != 1 else ''} "
                f"omitted; showing most recent {len(shown_c)})_"
            )
        for c in shown_c:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(c.created_at))
            age = _relative_age(c.created_at, _now)
            ts_disp = f"{ts}, {age}" if age else ts
            # Render author with explicit "comment from worker" framing so
            # operator-controlled HERMES_PROFILE values like "hermes-system"
            # or "operator" can't be misread by the next worker as a system
            # directive above the (attacker-influenceable) comment body.
            # Defense-in-depth — the LLM-controlled author-forgery surface
            # was already closed in #22435. See #22452.
            safe_author = (c.author or "").replace("`", "")
            lines.append(f"comment from worker `{safe_author}` at {ts_disp}:")
            lines.append(_cap(c.body, _CTX_MAX_COMMENT_BYTES))
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Stats + SLA helpers
# ---------------------------------------------------------------------------

def board_stats(conn: sqlite3.Connection) -> dict:
    """Per-status + per-assignee counts, plus the oldest ``ready`` age in
    seconds (the clearest staleness signal for a router or HUD).
    """
    by_status: dict[str, int] = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' GROUP BY status"
    ):
        by_status[row["status"]] = int(row["n"])

    by_assignee: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        by_assignee.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

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


def _to_epoch(val) -> Optional[int]:
    """Normalise a timestamp to unix epoch seconds.

    Accepts ints (pass-through), numeric strings, and ISO-8601 strings.
    Returns ``None`` for ``None`` / empty values.
    """
    if val is None:
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, float):
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
    age_since_created = now - _c if _c is not None else None
    age_since_started = now - _s if _s is not None else None
    time_to_complete = (
        _co - (_s or _c) if _co is not None else None
    )
    return {
        "created_age_seconds": age_since_created,
        "started_age_seconds": age_since_started,
        "time_to_complete_seconds": time_to_complete,
    }


# ---------------------------------------------------------------------------
# Notification subscriptions (used by the gateway kanban-notifier)
# ---------------------------------------------------------------------------

def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
) -> None:
    """Register a gateway source that wants terminal-state notifications
    for ``task_id``. Idempotent on (task, platform, chat, thread)."""
    now = int(time.time())
    with write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_notify_subs
                (task_id, platform, chat_id, thread_id, user_id, notifier_profile, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (task_id, platform, chat_id, thread_id or "", user_id, notifier_profile, now),
        )
        if notifier_profile:
            # Self-heal legacy rows that predate notifier ownership by
            # backfilling only when the existing value is unset.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET notifier_profile = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                   AND (notifier_profile IS NULL OR notifier_profile = '')
                """,
                (notifier_profile, task_id, platform, chat_id, thread_id or ""),
            )


def list_notify_subs(
    conn: sqlite3.Connection, task_id: Optional[str] = None,
) -> list[dict]:
    if task_id is not None:
        rows = conn.execute(
            "SELECT * FROM kanban_notify_subs WHERE task_id = ?", (task_id,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM kanban_notify_subs").fetchall()
    return [dict(r) for r in rows]


def remove_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id = ? "
            "AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        )
    return cur.rowcount > 0


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` for a given subscription.

    Only events with ``id > last_event_id`` are returned. The subscription's
    cursor is NOT advanced here; call :func:`advance_notify_cursor` after
    the gateway has successfully delivered the notifications.
    """
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs "
        "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
        (task_id, platform, chat_id, thread_id or ""),
    ).fetchone()
    if row is None:
        return 0, []
    cursor = int(row["last_event_id"])
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, cursor]
    if kind_list:
        params.extend(kind_list)
    rows = conn.execute(q, params).fetchall()
    out: list[Event] = []
    max_id = cursor
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(Event(
            id=r["id"], task_id=r["task_id"], kind=r["kind"],
            payload=payload, created_at=r["created_at"],
            run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
        ))
        max_id = max(max_id, int(r["id"]))
    return max_id, out


def claim_unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, int, list[Event]]:
    """Atomically claim unseen notification events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``. When events are returned,
    ``kanban_notify_subs.last_event_id`` has already been advanced to
    ``new_cursor`` inside a ``BEGIN IMMEDIATE`` transaction. That makes the
    notifier's read/claim step single-owner across multiple gateway watcher
    processes pointed at the same board DB: concurrent watchers serialize on
    SQLite's writer lock, and only the first process sees and claims a given
    event range.

    Callers should send the claimed events, then either leave the cursor at
    ``new_cursor`` on success or call :func:`rewind_notify_cursor` if delivery
    failed before any terminal unsubscribe removed the row.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        ).fetchone()
        if row is None:
            return 0, 0, []
        old_cursor = int(row["last_event_id"])
        new_cursor, events = unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or "", int(old_cursor)),
        )
        return old_cursor, new_cursor, events


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or ""),
        )


def rewind_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    claimed_cursor: int,
    old_cursor: int,
) -> bool:
    """Undo a notification claim when delivery fails.

    The CAS guard only rewinds if no later notifier advanced the row after our
    claim. This keeps retry behavior for transient send failures without
    clobbering newer progress.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (
                int(old_cursor), task_id, platform, chat_id, thread_id or "",
                int(claimed_cursor),
            ),
        )
    return cur.rowcount > 0


# ---------------------------------------------------------------------------
# Retention + garbage collection
# ---------------------------------------------------------------------------

def gc_events(
    conn: sqlite3.Connection, *, older_than_seconds: int = 30 * 24 * 3600,
) -> int:
    """Delete task_events rows older than ``older_than_seconds`` for tasks
    in a terminal state (``done`` or ``archived``). Returns the number of
    rows deleted. Running / ready / blocked tasks keep their full event
    history."""
    cutoff = int(time.time()) - int(older_than_seconds)
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM task_events WHERE created_at < ? AND task_id IN "
            "(SELECT id FROM tasks WHERE status IN ('done', 'archived'))",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def gc_worker_logs(
    *, older_than_seconds: int = 30 * 24 * 3600,
    board: Optional[str] = None,
) -> int:
    """Delete worker log files older than ``older_than_seconds``. Returns
    the number of files removed. Kept separate from ``gc_events`` because
    log files live on disk, not in SQLite. Scoped to ``board`` (defaults
    to the active board) — per-board isolation means deleting logs from
    board A cannot touch board B's logs."""
    log_dir = worker_logs_dir(board=board)
    if not log_dir.exists():
        return 0
    cutoff = time.time() - older_than_seconds
    removed = 0
    for p in log_dir.iterdir():
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink()
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Worker log accessor
# ---------------------------------------------------------------------------

def worker_log_path(task_id: str, *, board: Optional[str] = None) -> Path:
    """Return the path to a worker's log file. The file may not exist
    (task never spawned, or log already GC'd).

    When ``board`` is None, resolves via the active board (env var →
    current-board file → default). The dispatcher always passes the
    board explicitly to avoid any resolution ambiguity when multiple
    boards exist."""
    return worker_logs_dir(board=board) / f"{task_id}.log"


def read_worker_log(
    task_id: str, *, tail_bytes: Optional[int] = None,
    board: Optional[str] = None,
) -> Optional[str]:
    """Read the worker log for ``task_id``. Returns None if the file
    doesn't exist. If ``tail_bytes`` is set, only the last N bytes are
    returned (useful for the dashboard drawer which shouldn't page megabytes)."""
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
                # Skip a partial line if we tailed mid-line. But if the
                # window has no newline at all (one giant log line),
                # readline() would eat everything — in that case don't
                # skip and return the raw tail.
                probe = f.tell()
                partial = f.readline()
                if not partial.endswith(b"\n") and f.tell() >= size:
                    f.seek(probe)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Assignee enumeration (known profiles + per-profile board stats)
# ---------------------------------------------------------------------------

def list_profiles_on_disk() -> list[str]:
    """Return the set of assignee/profile names discovered on disk.

    Includes:
    - named profiles under ``<default-root>/profiles/<name>/config.yaml``
    - the implicit ``default`` profile when the default Hermes root exists

    Reads profile paths directly so this module has no import dependency on
    ``hermes_cli.profiles`` (which pulls in a large chunk of the CLI startup
    path).
    """
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
            for entry in sorted(profiles_dir.iterdir()):
                if not entry.is_dir():
                    continue
                if (entry / "config.yaml").is_file():
                    names.add(entry.name)
        except OSError:
            pass

    return sorted(names)


def known_assignees(conn: sqlite3.Connection) -> list[dict]:
    """Return every assignee name known to the board or on disk.

    Each entry is ``{"name": str, "on_disk": bool, "counts": {status: n}}``.
    A name is included when it's a configured profile on disk OR when
    any non-archived task has it as the assignee. Used by:

    - ``hermes kanban assignees`` for the terminal.
    - The dashboard assignee dropdown (so a fresh profile appears in
      the picker even before it's been given any task).
    - Router-profile heuristics ("who's overloaded?") without scanning
      the whole board.
    """
    on_disk = set(list_profiles_on_disk())

    # Count tasks per (assignee, status), excluding archived.
    counts: dict[str, dict[str, int]] = {}
    for row in conn.execute(
        "SELECT assignee, status, COUNT(*) AS n FROM tasks "
        "WHERE status != 'archived' AND assignee IS NOT NULL "
        "GROUP BY assignee, status"
    ):
        counts.setdefault(row["assignee"], {})[row["status"]] = int(row["n"])

    names = sorted(on_disk | set(counts.keys()))
    return [
        {
            "name": name,
            "on_disk": name in on_disk,
            "counts": counts.get(name, {}),
        }
        for name in names
    ]


# ---------------------------------------------------------------------------
# Runs (attempt history on a task)
# ---------------------------------------------------------------------------

def list_runs(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    include_active: bool = True,
    state_type: Optional[str] = None,
    state_name: Optional[str] = None,
) -> list[Run]:
    """Return all runs for ``task_id`` in start order.

    ``include_active=True`` (default) includes the currently-running
    attempt if any. Set False to return only closed runs (useful for
    "how many prior attempts have there been?" checks).

    When ``state_type`` and ``state_name`` are set, restrict to rows
    where that column equals ``state_name`` (``state_type`` is
    ``status`` or ``outcome``). Both must be passed together.
    """
    if (state_type is None) ^ (state_name is None):
        raise ValueError("state_type and state_name must both be set or both omitted")
    if state_type is not None:
        if state_type not in ("status", "outcome"):
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
    row = conn.execute(
        "SELECT * FROM task_runs WHERE id = ?", (int(run_id),),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_run(conn: sqlite3.Connection, task_id: str) -> Optional[Run]:
    """Return the most recent run regardless of outcome (active or closed)."""
    row = conn.execute(
        "SELECT * FROM task_runs WHERE task_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return Run.from_row(row) if row else None


def latest_summary(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the latest non-null ``task_runs.summary`` for ``task_id``.

    The worker writes its handoff to ``task_runs.summary``
    via ``complete_task(summary=...)``; ``tasks.result`` is left empty
    unless the caller passes ``result=`` explicitly. Dashboards and CLI
    "show" views need this value to surface what a worker actually did
    — without it, ``tasks.result`` is NULL and the task looks like a
    no-op even when the run completed.

    Picks the most recent run by ``ended_at`` (falling back to ``id``
    for ties or unfinished rows). Returns None if no run has a summary.
    """
    row = conn.execute(
        "SELECT summary FROM task_runs "
        "WHERE task_id = ? AND summary IS NOT NULL AND summary != '' "
        "ORDER BY COALESCE(ended_at, started_at) DESC, id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    return row["summary"] if row else None


def latest_summaries(
    conn: sqlite3.Connection, task_ids: Iterable[str]
) -> dict[str, str]:
    """Batch-fetch latest non-null summaries for a list of task ids.

    Used by the dashboard board endpoint to attach ``latest_summary`` to
    every card in a single SQL query, avoiding the N+1 pattern of
    calling :func:`latest_summary` per task. Returns a dict mapping
    ``task_id`` → summary string, omitting tasks with no summary.

    Approach: a window function picks the newest non-null-summary row
    per ``task_id``; works against SQLite ≥ 3.25 (default on every
    supported platform).
    """
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
