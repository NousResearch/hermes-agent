"""Kanban worker launcher — subprocess spawn path for dispatched tasks.

Extracted from ``hermes_cli/kanban_db.py`` (PR2 of the kanban remediation
plan) to:

* keep the 6 k-line DB module focused on storage concerns,
* make the spawn path independently testable without a live SQLite connection,
* enable pre-spawn environment validation that prevents process-level errors
  from silently cycling through the failure counter,
* give future PRs a clean seam to add alternative execution routes
  (Docker, remote SSH, Modal, …) without touching the DB layer.

Architecture
------------
The launcher exposes three levels of abstraction:

1. **Helper layer** — ``_resolve_hermes_argv``, ``_kanban_worker_skill_available``,
   ``_worker_terminal_timeout_env``, log-rotation helpers.  Pure functions,
   no external state.

2. **Preparation layer** — ``validate_spawn_env``, ``build_worker_env``,
   ``build_worker_cmd``, ``prepare_launch``.  Takes a resolved ``Task`` plus
   already-resolved board paths and returns a ``LaunchContext`` dataclass.
   No ``subprocess`` calls; safe to call from tests without spawning anything.

3. **Execution layer** — ``execute_launch`` (Popen wrapper), ``_default_spawn``
   (thin orchestrator that resolves board paths then delegates to layers 1–2).

Circular-import note
--------------------
``kanban_db.py`` imports ``_default_spawn`` from this module.  To avoid a
circular import, ``_default_spawn`` resolves kanban-path functions
(``kanban_db_path``, ``workspaces_root``, etc.) via a *lazy* import inside
the function body, executed only at call time (after all module-level
imports have completed).
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

# Max bytes to keep in a single worker log file before rotation on next spawn.
DEFAULT_LOG_ROTATE_BYTES = 2 * 1024 * 1024   # 2 MiB
DEFAULT_LOG_BACKUP_COUNT = 1

# Grace seconds so the worker can call kanban_block/complete before max_runtime
# kills the process.
KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS = 30


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LaunchContext:
    """Fully-prepared spawn parameters for a kanban worker.

    Created by :func:`prepare_launch` — holds every value that
    :func:`execute_launch` (or a future alternative executor) needs to start a
    worker process.  The separation allows callers to inspect or mutate the
    context before execution, and lets tests exercise the preparation phase
    without actually spawning a subprocess.
    """

    task_id: str
    profile: str
    workspace: str
    env: dict = field(default_factory=dict)
    cmd: list = field(default_factory=list)
    log_path: str = ""
    board: str = "default"


class SpawnValidationError(ValueError):
    """Raised by :func:`validate_spawn_env` when a pre-spawn check fails.

    Carries a ``task_id`` attribute so callers can surface it without
    parsing the message.
    """

    def __init__(self, message: str, *, task_id: str = "") -> None:
        super().__init__(message)
        self.task_id = task_id


# ---------------------------------------------------------------------------
# Integer coercion helper (shared by log-rotation + timeout helpers)
# ---------------------------------------------------------------------------

def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


# ---------------------------------------------------------------------------
# Hermes argv resolution
# ---------------------------------------------------------------------------

def _module_hermes_argv() -> list:
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


def _path_search_names(command: str) -> list:
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


def _hermes_path_argv(path: str) -> list:
    """Return argv for a resolved Hermes executable path.

    Windows batch shims (`.cmd` / `.bat`) are not safe as argv[0] for
    worker launches because the argument vector includes task-derived
    values. Prefer the interpreter-bound module form whenever the resolved
    executable is only a shell shim.
    """
    if _IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list:
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


# ---------------------------------------------------------------------------
# Worker skill and timeout helpers
# ---------------------------------------------------------------------------

def _kanban_worker_skill_available(hermes_home: Optional[str]) -> bool:
    """True if the bundled ``kanban-worker`` skill resolves for the home the
    spawned worker will run under.

    The dispatcher injects ``--skills kanban-worker`` into every worker. When
    the worker activates a profile (``hermes -p <name>``), its ``SKILLS_DIR``
    becomes ``<profile_home>/skills`` — which on many profiles does NOT contain
    the bundled skill (it ships in the *default* root home, not every
    profile-scoped skills dir). Preloading a missing skill is fatal at CLI
    startup (``ValueError: Unknown skill(s): kanban-worker``), aborting the
    worker before the agent loop runs. Gate the flag on actual resolvability;
    the kanban lifecycle contract is still injected via ``KANBAN_GUIDANCE``, so
    omitting the flag only drops the supplementary pattern library.
    """
    # An unset HERMES_HOME means the worker falls back to the default root
    # home (``~/.hermes``), which ships the bundled skill.
    base = Path(hermes_home) if hermes_home else (Path.home() / ".hermes")
    skills_root = base / "skills"
    if not skills_root.is_dir():
        return False
    # Canonical bundled location first (cheap), then a bounded scan for
    # profiles that have it nested elsewhere.
    if (skills_root / "devops" / "kanban-worker" / "SKILL.md").is_file():
        return True
    try:
        for skill_md in skills_root.rglob("kanban-worker/SKILL.md"):
            if skill_md.is_file():
                return True
    except OSError:
        pass
    return False


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


# ---------------------------------------------------------------------------
# Log rotation helpers
# ---------------------------------------------------------------------------

def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple:
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


# ---------------------------------------------------------------------------
# Pre-spawn validation
# ---------------------------------------------------------------------------

def validate_spawn_env(task: "Task", workspace: str) -> None:
    """Assert that ``task`` is ready to spawn a worker.

    Raises :class:`SpawnValidationError` on the first failure found so the
    dispatcher can record a structured failure reason rather than letting a
    cryptic ``FileNotFoundError`` or ``ValueError`` bubble through the
    ``_record_spawn_failure`` path.

    Checks (in order):
    1. Task has a non-empty ``id``.
    2. Task has a non-empty ``assignee``.

    Note: workspace directory existence is intentionally not validated here
    because scratch workspaces may be created on demand, and the Popen call
    already handles missing directories via ``cwd=None`` fallback.
    """
    if not task.id:
        raise SpawnValidationError("task has no id", task_id="")
    if not task.assignee:
        raise SpawnValidationError(
            f"task {task.id!r} has no assignee — cannot spawn worker",
            task_id=task.id,
        )


# ---------------------------------------------------------------------------
# Preparation layer
# ---------------------------------------------------------------------------

def build_worker_env(
    task: "Task",
    workspace: str,
    *,
    resolved_db_path: str,
    resolved_workspaces_root: str,
    resolved_board: str,
    base_env: Optional[dict] = None,
) -> dict:
    """Build the environment dict for a worker subprocess.

    Parameters
    ----------
    task:
        The task about to be spawned.
    workspace:
        Resolved workspace path string (may be empty for scratch tasks).
    resolved_db_path:
        Absolute path to the board's ``kanban.db``.
    resolved_workspaces_root:
        Absolute path to the board's ``workspaces/`` directory.
    resolved_board:
        Canonical board slug (already normalised).
    base_env:
        Starting environment; defaults to ``os.environ`` when ``None``.
    """
    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    env = dict(base_env if base_env is not None else os.environ)
    profile_arg = normalize_profile_name(task.assignee)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml
    # (fallback_providers, toolsets, agent settings, etc.) instead of the root
    # config.  Without this, `env = dict(os.environ)` copies only the parent's
    # env, and when the child process starts `hermes -p <name>` the
    # _apply_profile_override() runs *before* hermes_constants is imported.
    # If HERMES_HOME is absent from the child's env, get_hermes_home() falls
    # back to Path.home() / ".hermes" (the DEFAULT profile root), ignoring the
    # profile-specific config entirely.
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

    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock

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
    # dispatcher's.
    env["HERMES_KANBAN_DB"] = resolved_db_path
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = resolved_workspaces_root
    # Board slug — the final defense-in-depth pin.
    env["HERMES_KANBAN_BOARD"] = resolved_board
    # HERMES_PROFILE is the author the kanban_comment tool defaults to.
    env["HERMES_PROFILE"] = profile_arg

    return env


def build_worker_cmd(task: "Task", hermes_home: Optional[str] = None) -> list:
    """Build the ``hermes`` argv for a worker subprocess.

    Parameters
    ----------
    task:
        The task about to be spawned.
    hermes_home:
        The worker's ``HERMES_HOME`` (after profile resolution). Used to
        determine whether the ``kanban-worker`` skill is available.
    """
    from hermes_cli.profiles import normalize_profile_name

    profile_arg = normalize_profile_name(task.assignee)
    prompt = f"work kanban task {task.id}"

    cmd = [
        *_resolve_hermes_argv(),
        "-p", profile_arg,
        # Worker subprocesses switch to a profile-scoped HERMES_HOME above,
        # so they see that profile's shell-hook allowlist instead of the
        # dispatcher's root allowlist. Pass --accept-hooks explicitly so
        # profile-local worker sessions still register configured hooks.
        "--accept-hooks",
    ]
    if _kanban_worker_skill_available(hermes_home):
        cmd.extend(["--skills", "kanban-worker"])
    if task.skills:
        for sk in task.skills:
            if sk and sk != "kanban-worker":
                cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
    cmd.extend(["chat", "-q", prompt])
    return cmd


def prepare_launch(
    task: "Task",
    workspace: str,
    *,
    resolved_db_path: str,
    resolved_workspaces_root: str,
    resolved_board: str,
    resolved_log_dir: Path,
) -> LaunchContext:
    """Build a :class:`LaunchContext` without spawning any process.

    Validates the environment, builds the env dict and command list, and
    prepares the log path (including rotation of any previous log file).
    Returns a :class:`LaunchContext` that :func:`execute_launch` can execute
    without any further I/O beyond writing to the log.

    Parameters
    ----------
    task:
        The task to spawn a worker for.
    workspace:
        Resolved absolute path to the workspace directory (may be empty for
        scratch workspaces not yet created).
    resolved_db_path / resolved_workspaces_root / resolved_board:
        Board-specific paths already resolved by the caller (avoids a
        circular import between this module and ``kanban_db``).
    resolved_log_dir:
        Directory for worker log files; created if absent.
    """
    validate_spawn_env(task, workspace)

    env = build_worker_env(
        task,
        workspace,
        resolved_db_path=resolved_db_path,
        resolved_workspaces_root=resolved_workspaces_root,
        resolved_board=resolved_board,
    )
    cmd = build_worker_cmd(task, hermes_home=env.get("HERMES_HOME"))

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    _log.info(
        "kanban launch prepared: task=%s profile=%s board=%s workspace=%s",
        task.id,
        env.get("HERMES_PROFILE", ""),
        resolved_board,
        workspace or "(scratch)",
    )

    return LaunchContext(
        task_id=task.id,
        profile=env.get("HERMES_PROFILE", ""),
        workspace=workspace,
        env=env,
        cmd=cmd,
        log_path=str(log_path),
        board=resolved_board,
    )


# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------

def execute_launch(ctx: LaunchContext) -> int:
    """Spawn a worker subprocess from a :class:`LaunchContext`.

    Returns the child PID. The caller is responsible for recording the PID
    in the database and handling spawn failures.

    Raises ``RuntimeError`` when the ``hermes`` executable cannot be found.
    """
    import subprocess

    log_path = Path(ctx.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = ctx.workspace
    log_f = open(log_path, "ab")  # noqa: WPS515 — intentional: child inherits FD
    try:
        proc = subprocess.Popen(  # noqa: S603 — argv built from fixed list
            ctx.cmd,
            cwd=workspace if workspace and os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=ctx.env,
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
    _log.info(
        "kanban worker spawned: task=%s pid=%s profile=%s board=%s",
        ctx.task_id,
        proc.pid,
        ctx.profile,
        ctx.board,
    )
    return proc.pid


# ---------------------------------------------------------------------------
# Top-level spawn entry point (used by dispatch_once in kanban_db)
# ---------------------------------------------------------------------------

def _default_spawn(
    task: "Task",
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

    Board-path resolution is deferred to call time via a lazy import from
    ``kanban_db`` to avoid a module-level circular import.
    """
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    # Lazy imports from kanban_db to break the circular-import cycle:
    # kanban_db imports _default_spawn from this module at module level;
    # we resolve kanban paths only at call time.
    from hermes_cli.kanban_db import (  # noqa: PLC0415
        _normalize_board_slug,
        get_current_board,
        kanban_db_path,
        worker_logs_dir,
        workspaces_root,
    )

    resolved_board = _normalize_board_slug(board) or get_current_board()

    ctx = prepare_launch(
        task,
        workspace,
        resolved_db_path=str(kanban_db_path(board=board)),
        resolved_workspaces_root=str(workspaces_root(board=board)),
        resolved_board=resolved_board,
        resolved_log_dir=worker_logs_dir(board=board),
    )
    return execute_launch(ctx)
