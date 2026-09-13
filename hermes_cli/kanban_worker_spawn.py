"""Kanban worker launch, log rotation and executable resolution.

Scope launch integrated from PR #101911; current main owns profile and grant setup.
"""

from __future__ import annotations

import hermes_cli.kanban_db as _owner_kanban_db

import hermes_cli.kanban_db_boards as _owner_kanban_boards
import os
import contextlib
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Optional
from hermes_cli.kanban_db_models import Task
from hermes_cli import kanban_db as _kb
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def worker_log_rotation_config(kanban_cfg: Optional[dict] = None) -> tuple[int, int]:
    """Return ``(rotate_bytes, backup_count)`` for worker log rotation.
    Defaults: rotate at 2 MiB, keep one backup (``.log.1``); both overridable
    from ``config.yaml``.
    """
    if kanban_cfg is None:
        try:
            from hermes_cli.config import load_config

            kanban_cfg = load_config().get("kanban") or {}
        except Exception:
            kanban_cfg = {}
    kanban_cfg = kanban_cfg or {}
    max_bytes = _positive_int(
        kanban_cfg.get("worker_log_rotate_bytes"),
        _kanban_db_dispatch.DEFAULT_LOG_ROTATE_BYTES,
        minimum=1,
    )
    backup_count = _positive_int(
        kanban_cfg.get("worker_log_backup_count"),
        _kanban_db_dispatch.DEFAULT_LOG_BACKUP_COUNT,
        minimum=0,
    )
    return max_bytes, backup_count


def _rotated_log_path(log_path: Path, generation: int) -> Path:
    return log_path.with_suffix(log_path.suffix + f".{generation}")


def _rotate_worker_log(
    log_path: Path,
    max_bytes: int,
    backup_count: int = _kanban_db_dispatch.DEFAULT_LOG_BACKUP_COUNT,
) -> None:
    """Rotate ``<log>`` when it exceeds ``max_bytes``: ``<log>`` → ``<log>.1``,
    older generations shift up to ``backup_count``.
    """
    try:
        if not log_path.exists() or log_path.stat().st_size <= max_bytes:
            return
        backup_count = _positive_int(
            backup_count, _kanban_db_dispatch.DEFAULT_LOG_BACKUP_COUNT, minimum=0
        )
        if backup_count == 0:
            log_path.unlink()
            return
        oldest = _rotated_log_path(log_path, backup_count)
        with contextlib.suppress(OSError):
            if oldest.exists():
                oldest.unlink()
        for generation in range(backup_count - 1, 0, -1):
            src = _rotated_log_path(log_path, generation)
            if not src.exists():
                continue
            with contextlib.suppress(OSError):
                src.rename(_rotated_log_path(log_path, generation + 1))
        log_path.rename(_rotated_log_path(log_path, 1))
    except OSError:
        pass


def _module_hermes_argv() -> list[str]:
    """Interpreter-bound Hermes CLI invocation (``hermes_cli.main`` is the
    console-script target — there is no top-level ``hermes`` package)."""
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
    if not _kb._IS_WINDOWS or os.path.splitext(command)[1]:
        return [command]
    raw = os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD"
    return [command + ext for ext in raw.split(";") if ext]


def _safe_which_no_cwd(command: str) -> Optional[str]:
    """Resolve a bare command from PATH without implicit current-dir search.

    On Windows ``shutil.which`` may search the current directory before PATH
    for bare names — unsafe for a dispatcher. Only explicit PATH entries are
    considered; empty / ``.`` entries are skipped.
    """
    for raw_dir in os.environ.get("PATH", "").split(os.pathsep):
        if not raw_dir or raw_dir == ".":
            continue
        directory = os.path.expanduser(raw_dir)
        for name in _path_search_names(command):
            candidate = os.path.join(directory, name)
            if os.path.isfile(candidate) and (
                _kb._IS_WINDOWS or os.access(candidate, os.X_OK)
            ):
                return candidate
    return None


def _hermes_path_argv(path: str) -> list[str]:
    """argv for a resolved Hermes executable path. Windows batch shims
    (``.cmd``/``.bat``) are unsafe as argv[0] because the argument vector
    includes task-derived values; prefer the module form."""
    if _kb._IS_WINDOWS and _is_windows_batch_shim(path):
        return _module_hermes_argv()
    return [_absolute_hermes_path(path)]


def _resolve_hermes_argv() -> list[str]:
    """Resolve the ``hermes`` invocation as argv for ``Popen``: ``$HERMES_BIN``
    (path-like -> absolute; bare names keep PATH semantics, never a
    same-directory file), then ``which("hermes")`` (Windows: safe PATH search,
    batch shims fall back to the module form), then ``sys.executable -m
    hermes_cli.main`` for shim-less environments (cron, systemd ``User=``,
    launchd). Mirrors ``gateway.run._resolve_hermes_bin``; local because
    ``hermes_cli`` sits below ``gateway`` in the dependency order.
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

    hermes_bin = (
        _safe_which_no_cwd("hermes") if _kb._IS_WINDOWS else shutil.which("hermes")
    )
    if hermes_bin:
        return _hermes_path_argv(hermes_bin)
    return _module_hermes_argv()


def _worker_terminal_timeout_env(
    max_runtime_seconds: Optional[int],
    current_timeout: Optional[str],
) -> Optional[str]:
    """Return a worker-scoped TERMINAL_TIMEOUT override, if needed.

    When ``max_runtime_seconds`` exceeds the terminal tool's default timeout,
    raise only the child's default so a long command isn't killed by the
    generic terminal default first.
    """
    if max_runtime_seconds is None:
        return None
    try:
        runtime = int(max_runtime_seconds)
    except (TypeError, ValueError):
        return None
    if runtime <= 0:
        return None

    desired = max(
        1, runtime - _kanban_db_dispatch.KANBAN_TERMINAL_TIMEOUT_GRACE_SECONDS
    )
    try:
        existing = int(str(current_timeout).strip()) if current_timeout else 0
    except (TypeError, ValueError):
        existing = 0
    if existing >= desired:
        return None
    return str(desired)


def _resolve_worker_cli_toolsets(hermes_home: Optional[str]) -> Optional[list[str]]:
    """Return the assigned profile's effective CLI toolsets for a worker.

    Resolved at dispatch time and passed as an explicit ``--toolsets`` pin so
    worker startup cannot fall back to a stale root/active-profile config or a
    profile whose top-level ``toolsets`` is only the kanban orchestrator
    surface. ``model_tools`` still appends the task-scoped kanban lifecycle
    tools when ``HERMES_KANBAN_TASK`` is set.
    """
    if not hermes_home:
        return None
    try:
        from agent.secret_scope import (
            build_profile_secret_scope,
            is_multiplex_active,
            reset_secret_scope,
            set_secret_scope,
        )
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        token = set_hermes_home_override(hermes_home)
        # Toolset availability probes read credentials (``get_secret``); under multiplex an
        # unscoped read raises and the pin was silently dropped for every worker.
        secret_token = (
            set_secret_scope(build_profile_secret_scope(Path(hermes_home)))
            if is_multiplex_active()
            else None
        )
        try:
            cfg = load_config()
            toolsets = sorted(_get_platform_tools(cfg, "cli"))
        finally:
            if secret_token is not None:
                reset_secret_scope(secret_token)
            reset_hermes_home_override(token)
        return toolsets or None
    except Exception as exc:
        _kb._log.debug(
            "kanban worker: could not resolve CLI toolsets for HERMES_HOME=%r (%s)",
            hermes_home,
            exc,
        )
        return None


_retagged_workspace_roots: set[str] = set()


def _retag_legacy_worker_sessions(workspaces_root_path: str) -> None:
    """Reclaim pre-tag worker rows in state.db so they leave the session lists.

    Best-effort: the durable gate is ``state_meta`` in
    ``retag_kanban_worker_sessions``; the in-process set avoids reopening
    state.db on every spawn. A tick must never fail because a session DB was
    busy or missing.
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
        _kb._log.debug("kanban worker: legacy session retag skipped (%s)", exc)


def _worker_argv(task: Task, profile_arg: str, hermes_home: Optional[str]) -> list[str]:
    """Build the ``hermes -p <profile> --cli ... chat -q ...`` worker command."""
    cmd = [
        *_resolve_hermes_argv(),
        "-p",
        profile_arg,
        # A worker must NEVER boot the interactive TUI: its no-TTY bail-out
        # exits 0 without doing the task → "protocol violation" every attempt.
        "--cli",
        # Workers run under a profile-scoped HERMES_HOME and so see that
        # profile's shell-hook allowlist; pass --accept-hooks explicitly so
        # configured hooks still register.
        "--accept-hooks",
    ]
    # One `--skills X` pair per name: easier to read in `ps` and avoids quoting
    # ambiguity if a skill name contains unusual chars.
    for sk in task.skills or ():
        if sk:
            cmd.extend(["--skills", sk])
    if task.model_override:
        cmd.extend(["-m", task.model_override])
        # Pin the provider too so the worker resolves the model against the
        # intended backend (model X with provider Y is the classic board-stall).
        if task.provider_override:
            cmd.extend(["--provider", task.provider_override])
    # Independent of the model override — a task can run the profile's own
    # model at a different depth.
    if task.reasoning_effort:
        cmd.extend(["--reasoning", task.reasoning_effort])
    worker_toolsets = _resolve_worker_cli_toolsets(hermes_home)
    if worker_toolsets:
        cmd.extend(["--toolsets", ",".join(worker_toolsets)])
    cmd.extend(["chat", "-q", f"work kanban task {task.id}"])
    if task.goal_mode:
        # The kanban goal-loop hook only runs in cli.py's fully-quiet branch.
        # Without -Q the worker gets one turn, prints text, exits rc=0, and the
        # dispatcher records a protocol violation.
        cmd.append("-Q")
    return cmd


def _open_worker_log(task: Task, board: Optional[str]):
    """Append-mode per-task log (a re-run on unblock appends, never overwrites),
    rotated first. Anchored at the board root (not the shared kanban root) so
    `hermes kanban log` reads its own file and boards sharing task ids don't
    collide."""
    log_dir = _owner_kanban_boards.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)
    return open(log_path, "ab")


def _restart_safe_worker_argv(
    task: Task, command: list[str], *, board: Optional[str] = None
) -> list[str]:
    """Wrap a managed-gateway worker in the shared restart-safe scope.

    Kanban workers are long-lived agentic runs, so they never take cron's
    degraded mode: ``require_restart_safe_scope=True`` makes the helper raise.
    """
    from tools.process_registry import restart_safe_gateway_child_argv

    if task.current_run_id is None:
        # Outside managed systemd this is harmless, but a managed dispatch must
        # never mint an untraceable worker.  Check topology through the shared
        # helper first, using a placeholder suffix that cannot be launched.
        dispatch = restart_safe_gateway_child_argv(
            command,
            unit_suffix=f"kanban-{_kanban_worker_scope._scope_task_key(task.id, board=board)}-run-missing",
            require_restart_safe_scope=True,
        )
        if dispatch.mode != "in_process":
            raise RuntimeError(
                "cannot create restart-safe systemd scope for Kanban worker: "
                "the claimed task has no current run id"
            )
        return command

    return restart_safe_gateway_child_argv(
        command,
        unit_suffix=f"kanban-{_kanban_worker_scope._scope_task_key(task.id, board=board)}-run-{task.current_run_id}",
        require_restart_safe_scope=True,
    ).argv


class WorkerLaunchUnconfirmed(RuntimeError):
    """A launch may own a live scope; retry is forbidden until it stops."""

    def __init__(self, message, pid, scope_unit):
        super().__init__(message)
        self.pid = int(pid)
        self.scope_unit = scope_unit


class _SpawnedWorkerPid(int):
    """Worker PID annotated with the scope unit the spawn actually created.

    Per-call channel replacing the old process-global
    ``_default_spawn._last_scope_unit`` function attribute: per-board
    dispatches run concurrently, so a global let one board's dispatch
    record another board's unit on its task row (Gate B review, finding
    F). Subclassing ``int`` keeps every existing caller contract —
    ``if pid:``, ``int(pid)``, ``str(pid)``, and plain-int returns from
    custom spawn stubs (whose unit reads as ``""``) — unchanged.
    """

    scope_unit: str = ""

    def __new__(cls, pid: int, scope_unit: str = ""):
        self = super().__new__(cls, pid)
        self.scope_unit = scope_unit
        return self


def _default_spawn(
    task: Task, workspace: str, *, board: Optional[str] = None
) -> Optional[int]:
    """Fire-and-forget ``hermes -p <profile> chat -q ...`` subprocess.

    Returns the child's PID so the dispatcher can detect crashes before the
    claim TTL expires; completion is still observed via the worker's own
    ``complete`` / ``block`` transitions. ``board`` pins the child's
    ``HERMES_KANBAN_DB`` / ``HERMES_KANBAN_BOARD`` / workspaces_root to the
    board the task was claimed from, so workers cannot see other boards.
    """
    if not task.assignee:
        raise ValueError(f"task {task.id} has no assignee")

    from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

    profile_arg = normalize_profile_name(task.assignee)

    from agent.secret_scope import is_multiplex_active
    from tools.environments.local import build_subprocess_env, strip_launch_profile_env

    env = build_subprocess_env(
        scrub_secrets=is_multiplex_active(),
        inherit_profile_home=True,
    )
    # The dispatcher is detached from every conversation; its worker must never
    # inherit routing mirrored by a previous gateway turn.
    from gateway.session_context import _VAR_MAP

    for key in _VAR_MAP:
        env.pop(key, None)

    # Inject HERMES_HOME so the worker reads the profile-scoped config.yaml:
    # without it the child's get_hermes_home() falls back to the DEFAULT
    # profile root because `hermes -p` applies its override before
    # hermes_constants is imported.
    try:
        env["HERMES_HOME"] = resolve_profile_env(profile_arg)
        # A multiplexer dispatching for another profile must not hand it the launch
        # profile's .env settings / TERMINAL_* policy — a standalone dispatcher never would.
        strip_launch_profile_env(env, env["HERMES_HOME"])
    except FileNotFoundError:
        # No profile dir (isolated test fixtures) — the CLI resolves it from
        # HERMES_PROFILE (set below) instead.
        pass
    if task.tenant:
        env["HERMES_TENANT"] = task.tenant
    env["HERMES_KANBAN_TASK"] = task.id
    env["HERMES_KANBAN_WORKSPACE"] = workspace
    # Tag the session `kanban` so session-browsing surfaces filter it out by
    # source instead of rendering one sidebar row per attempt.
    env["HERMES_SESSION_SOURCE"] = "kanban"
    # TERMINAL_CWD takes precedence over process cwd in file_tools and
    # build_context_files_prompt; without it relative writes land in the gateway
    # user's home and workers load the gateway's AGENTS.md. file_tools rejects
    # relative / sentinel values, so only set a real absolute directory.
    # Pin TERMINAL_CWD to the task's workspace so the worker's file tools and context-file loader anchor on
    # the workspace, not whatever cwd the dispatching gateway happened to export. The worker subprocess is
    # already launched with cwd=workspace, but TERMINAL_CWD takes precedence over the process cwd in both
    # file_tools._resolve_base_dir (#41312 — relative write_file paths were landing in the gateway user's
    # home) and build_context_files_prompt (#34619 — workers loaded the dispatching gateway's AGENTS.md
    # instead of the task's). Setting it to the workspace fixes both: the workspace is where the task's work
    # actually happens.
    if workspace and os.path.isabs(workspace) and os.path.isdir(workspace):
        env["TERMINAL_CWD"] = workspace
    if task.branch_name:
        env["HERMES_KANBAN_BRANCH"] = task.branch_name
    if task.current_run_id is not None:
        env["HERMES_KANBAN_RUN_ID"] = str(task.current_run_id)
    if task.claim_lock:
        env["HERMES_KANBAN_CLAIM_LOCK"] = task.claim_lock
    # Goal-loop mode (Ralph-style /goal judge loop in cli.py quiet-mode path).
    # Only set when enabled so non-goal tasks keep a clean env.
    if task.goal_mode:
        env["HERMES_KANBAN_GOAL_MODE"] = "1"
        if task.goal_max_turns is not None:
            env["HERMES_KANBAN_GOAL_MAX_TURNS"] = str(int(task.goal_max_turns))
    for var in ("TERMINAL_TIMEOUT", "TERMINAL_MAX_FOREGROUND_TIMEOUT"):
        override = _worker_terminal_timeout_env(task.max_runtime_seconds, env.get(var))
        if override is not None:
            env[var] = override
    # Pin the board DB + workspaces root so the worker's kanban paths still
    # match after `hermes -p` rewrites HERMES_HOME (symlink / Docker layouts).
    env["HERMES_KANBAN_DB"] = str(_owner_kanban_db.kanban_db_path(board=board))
    env["HERMES_KANBAN_WORKSPACES_ROOT"] = str(
        _owner_kanban_boards.workspaces_root(board=board)
    )
    _retag_legacy_worker_sessions(env["HERMES_KANBAN_WORKSPACES_ROOT"])
    # Board slug — defense-in-depth pin if a path is resolved without the
    # DB / workspaces env vars.
    env["HERMES_KANBAN_BOARD"] = (
        _owner_kanban_boards._normalize_board_slug(board)
        or _owner_kanban_boards.get_current_board()
    )
    # kanban_comment reads HERMES_PROFILE for its default author; `-p` alone
    # doesn't set the env var.
    env["HERMES_PROFILE"] = profile_arg
    # This is the grant boundary: the dispatcher assigned this new worker's task.
    from agent.delegation_context import DELEGATED_CHILD_ENV_MARKER

    env.pop(DELEGATED_CHILD_ENV_MARKER, None)
    # `--cli` is the highest-precedence TUI override; dropping HERMES_TUI covers
    # older hermes builds on PATH that predate the flag's precedence.
    env.pop("HERMES_TUI", None)

    cmd = _worker_argv(task, profile_arg, env.get("HERMES_HOME"))
    return _launch_worker(task, workspace, env, cmd, board=board)


def _launch_worker(task, workspace, env, cmd, *, board=None):
    # A worker spawned by a managed systemd gateway must leave the gateway's
    # cgroup before startup; otherwise restarting the service kills the worker
    # that is performing the handoff.  That wrap is applied HERE, per launch
    # path, rather than unconditionally to ``cmd``: worker isolation below
    # wraps the same argv in its own transient scope through the same
    # ``systemd-run --user --scope`` mechanism, so an isolated worker has
    # already left the gateway cgroup.  Wrapping twice would nest one
    # transient scope inside another — the inner unit adopts the process and
    # the outer one is left empty — and it would also defeat the
    # ``wrapped[0] != cmd[0]`` launch check below, which reads a systemd-run
    # argv0 on both sides as "the scope wrap did not happen".  Every path
    # that launches WITHOUT an isolation scope of its own goes through this
    # helper instead, so the restart-safe guarantee still holds on each.
    #
    # It returns ``(argv, scope_unit)``: on a managed gateway "plain" is a
    # misnomer — the argv IS scoped, just under the restart-safe wrap's own
    # unit name — and the caller must record THAT unit as the run's
    # ``worker_scope``.  Reporting "" there left a real scope untracked: the
    # row counted as registered immediately, no stop was ever requested for
    # it, and the audit sweep could not see it.
    def _plain_launch_argv(log_handle=None) -> tuple[list[str], str]:
        try:
            argv = _restart_safe_worker_argv(task, cmd, board=board)
        except RuntimeError as exc:
            # Fail closed, but record the cause on the spawn-error channel
            # so the dispatcher classifies it as spawn_failed with real
            # text. ``log_handle`` is passed by the call sites that run
            # before the spawn try/except owns the handle, matching the
            # close-before-raise discipline of the strict-mode refusals.
            spawn_error = str(exc)
            if log_handle is not None:
                log_handle.close()
            raise
        if argv is cmd or argv[0] == cmd[0]:
            return argv, ""  # genuinely plain: no wrap was applied
        return argv, _kanban_worker_scope._scope_unit_from_argv(argv)

    # Topology is read ONCE per spawn and carried to every decision in
    # it. ``_managed_gateway_dispatch`` re-derives the answer on each
    # call and its probe swallows failures into False, so re-asking
    # after the launch could flip a managed gateway to "unmanaged"
    # mid-spawn on a transient error — and the fallback branch below
    # would then plain-spawn a duplicate worker under a scope wrap it
    # believed was absent. One snapshot makes every branch of this
    # spawn agree by construction.
    managed_gateway = _kanban_worker_scope._managed_gateway_dispatch()

    # Both launch paths name their unit after the ATTEMPT, so a managed
    # dispatch of a claimed task with no current run id is refused HERE,
    # once, above the isolation branch. ``_restart_safe_worker_argv`` makes
    # the same refusal, but the isolation path never reaches it and would
    # otherwise mint the attempt-free ``hermes-kanban-<task>.scope`` — an
    # untraceable name a retry can collide with.
    if task.current_run_id is None and managed_gateway:
        spawn_error = (
            "cannot create restart-safe systemd scope for Kanban worker: "
            "the claimed task has no current run id"
        )
        raise RuntimeError(spawn_error)

    # Redirect output to a per-task log under <board-root>/logs/.
    # Anchored at the board root (not the shared kanban root), so
    # `hermes kanban log` on a specific board reads its own file and
    # logs don't collide across boards that happen to share task ids.
    log_dir = _owner_kanban_boards.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.id}.log"
    rotate_bytes, backup_count = worker_log_rotation_config()
    _rotate_worker_log(log_path, rotate_bytes, backup_count)

    # Use 'a' so a re-run on unblock appends rather than overwrites.
    log_f = open(log_path, "ab")

    # Worker isolation: wrap the argv in its own transient systemd user
    # scope (``hermes-kanban-<task>-r<run_id>.scope``) so the worker's
    # cgroup — and every dev server / browser / database it spawns — is
    # independent of the gateway's, bounded by MemoryMax + MemorySwapMax,
    # and survives gateway restarts. The unit name embeds the run id, so
    # a respawn after a failed attempt can never collide with a lingering
    # half-dead scope from the previous one. ``start_new_session`` below
    # still applies inside the scope (the systemd-run wrapper execs the
    # command; the scope, not the session, is what isolates the cgroup).
    # Unavailable systemd / isolation 'none' leaves ``cmd`` byte-identical
    # to today. STRICT mode (worker_isolation: systemd-scope) never
    # degrades: an unusable probe or a vanished systemd-run fails the
    # spawn loudly as spawn_failed instead (Gate B review, finding H) —
    # an operator who pinned strict chose "no worker" over "unisolated
    # worker", and only 'auto' may fall back to the plain spawn.
    launch_cmd = cmd
    scope_unit = ""
    spawn_err_path = None
    strict_scope = _kanban_worker_scope._resolve_worker_isolation() == "systemd-scope"
    scope_enabled = _kanban_worker_scope._kanban_worker_scope_enabled()
    if strict_scope and not scope_enabled:
        spawn_error = (
            f"worker_isolation=systemd-scope is configured but "
            f"systemd-run --user --scope is unavailable on this host — "
            f"refusing to spawn task {task.id} without isolation "
            f"(set worker_isolation: auto to allow the unisolated "
            f"fallback)"
        )
        log_f.close()
        raise RuntimeError(spawn_error)
    if scope_enabled:
        from tools.process_registry_scope import _build_systemd_scope_argv

        scope_unit = _kanban_worker_scope._kanban_worker_scope_unit(
            task.id, task.current_run_id, board=board
        )
        memory_max = _kanban_worker_scope._kanban_worker_memory_bytes()
        wrapped = _build_systemd_scope_argv(
            cmd,
            task.id,
            unit_name=scope_unit,
            description=f"Hermes kanban worker {task.id}: {task.title}"[:160],
            memory_max_bytes=memory_max,
            memory_swap_max_bytes=memory_max,
            # No --collect: a fast NONZERO worker exit must leave the
            # failed unit LOADED (failed transient units are kept for
            # inspection unless collected), so the launch probe below
            # can tell "the wrapped command ran and exited" from
            # "systemd-run refused the launch".  With --collect the unit
            # could be unloaded before the probe reads it, a ran worker
            # looked like a refused launch, and the auto fallback
            # plain-spawned a DUPLICATE beside the exited one.  The unit
            # is collected explicitly once the run is terminal
            # (_collect_kanban_scope / the verified-stop path).
            collect=False,
        )
        if wrapped[0] != cmd[0]:
            launch_cmd = wrapped
            # The systemd-run client's own stderr is the only place a
            # refused launch surfaces (user bus gone, unit name rejected,
            # property refused); the worker's stdout/stderr go to the
            # normal per-task log. Capture it separately so a launch
            # failure classifies as spawn_failed with real text instead
            # of surfacing later as an unexplained "worker crashed".
            spawn_err_path = log_dir / f"{task.id}.spawn.err"
        else:
            # systemd-run vanished between the availability probe and the
            # build — the helper returned the argv unwrapped. In strict
            # mode that is the same refusal as a failed probe (finding H):
            # the operator pinned systemd-scope, so "no worker" beats an
            # unisolated worker. Only auto continues, honestly recording
            # whichever scope the launch actually got (the restart-safe
            # wrap's on a managed gateway, none anywhere else).
            if strict_scope:
                spawn_error = (
                    f"worker_isolation=systemd-scope is configured but "
                    f"systemd-run disappeared between the availability "
                    f"probe and the launch of task {task.id} — refusing "
                    f"to spawn without isolation"
                )
                log_f.close()
                raise RuntimeError(spawn_error)
            launch_cmd, scope_unit = _plain_launch_argv(log_f)
    else:
        launch_cmd, scope_unit = _plain_launch_argv(log_f)
    from tools.process_registry import systemd_user_bus_env

    env = systemd_user_bus_env(env)
    if scope_unit:
        env["HERMES_KANBAN_SCOPE"] = scope_unit

    return _execute_worker_launch(
        task,
        workspace,
        env,
        launch_cmd,
        scope_unit,
        spawn_err_path,
        log_f,
        managed_gateway,
        _plain_launch_argv,
    )


def _execute_worker_launch(
    task,
    workspace,
    env,
    launch_cmd,
    scope_unit,
    spawn_err_path,
    log_f,
    managed_gateway,
    _plain_launch_argv,
):
    def _spawn(stderr_target):
        return subprocess.Popen(  # noqa: S603 -- argv is a fixed list built above
            launch_cmd,
            cwd=workspace if os.path.isdir(workspace) else None,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=stderr_target,
            env=env,
            start_new_session=True,
            creationflags=subprocess.CREATE_NO_WINDOW if _kanban_db._IS_WINDOWS else 0,
        )

    def _read_spawn_err() -> str:
        if spawn_err_path is None:
            return ""
        try:
            return spawn_err_path.read_text("utf-8", "replace").strip()
        except OSError:
            return ""

    spawn_err_f = open(spawn_err_path, "wb") if spawn_err_path else None
    try:
        try:
            proc = _spawn(
                spawn_err_f if spawn_err_f is not None else subprocess.STDOUT,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "`hermes` executable not found on PATH. "
                "Install Hermes Agent or activate its venv before running the kanban dispatcher."
            )
        if spawn_err_f is not None:
            # The systemd-run client exits almost immediately when the
            # launch is refused; watch it for a short bounded window so
            # those failures classify NOW as spawn_failed with the
            # captured stderr. A healthy ``--scope`` launcher lives as
            # long as the worker, so the window is a BOUND, not a wait:
            # the probe also ends the moment the transient unit appears
            # (the launch-confirmation signal). The worker registers its
            # own pid from inside the scope (see register_worker_pid).
            deadline = (
                time.monotonic() + _kanban_worker_scope._worker_spawn_probe_seconds()
            )
            next_unit_check = 0.0
            while proc.poll() is None:
                if time.monotonic() >= deadline:
                    break
                if time.monotonic() >= next_unit_check:
                    next_unit_check = time.monotonic() + 0.2
                    if _kanban_worker_scope._scope_unit_created(scope_unit):
                        break  # unit appeared — launch confirmed
                time.sleep(0.05)
            if proc.poll() is not None:
                # The launcher exited inside the window. That is a LAUNCH
                # failure only when the systemd-run client itself failed:
                # non-zero rc AND the transient unit never came to life.
                # A launcher that exits rc=0, or with the unit created,
                # RAN the worker — the worker just exited fast — and the
                # next tick's exit classification owns that outcome.
                # Plain-spawning a "replacement" here duplicated the
                # work (the review's critical spawn bug).
                unit_created = _kanban_worker_scope._scope_unit_created(scope_unit)
                if proc.returncode == 0 or unit_created:
                    _kanban_db._log.info(
                        "kanban dispatch: task %s worker exited within "
                        "the launch probe (launcher rc=%s, unit %s) — "
                        "leaving it to exit classification",
                        task.id,
                        proc.returncode,
                        "created" if unit_created else "not created",
                    )
                    spawn_err_f.close()
                    spawn_err_f = None
                    try:
                        os.unlink(spawn_err_path)
                    except OSError:
                        pass
                    return _SpawnedWorkerPid(proc.pid, scope_unit)
                err_text = _read_spawn_err()
                # Verified cleanup of whatever half-formed unit the
                # failed launch left behind. If even that cannot be
                # confirmed, do NOT plain-spawn a replacement: a fresh
                # worker beside an unkillable half-created scope is
                # worse than a failed attempt — record spawn_failed.
                if scope_unit and not _kanban_worker_scope._stop_kanban_worker_scope(
                    scope_unit
                ):
                    spawn_error = (
                        f"systemd-run launch failed for task {task.id} "
                        f"(rc={proc.returncode}): "
                        f"{err_text or 'no stderr captured'}; the "
                        f"half-created unit {scope_unit} could not be "
                        f"verified stopped — refusing to spawn a "
                        f"replacement beside it"
                    )
                    raise WorkerLaunchUnconfirmed(spawn_error, proc.pid, scope_unit)
                if (
                    _kanban_worker_scope._resolve_worker_isolation() != "systemd-scope"
                    and not managed_gateway
                ):
                    # auto on a host that can genuinely run the worker
                    # unwrapped: fall back to a plain spawn for THIS run
                    # and say so once — a missing user bus must not stall
                    # the board. Reset EVERYTHING the scope wrap changed:
                    # launching the wrapped argv again would re-enter
                    # systemd-run (and "succeed" this time, hiding the
                    # degraded mode), and the scope env would mislabel
                    # an unisolated worker.
                    _kanban_db._log.warning(
                        "kanban dispatch: task %s systemd-run launch failed "
                        "(rc=%s: %s) — retrying WITHOUT scope isolation "
                        "(unmanaged host: the retry runs the bare worker "
                        "argv, unisolated, and records no scope unit)",
                        task.id,
                        proc.returncode,
                        err_text or "no stderr captured",
                    )
                    launch_cmd, scope_unit = _plain_launch_argv()
                    env.pop("HERMES_KANBAN_SCOPE", None)
                    proc = _spawn(subprocess.STDOUT)
                    # NOTE: log_f stays open for the child (see below).
                    return _SpawnedWorkerPid(proc.pid, scope_unit)
                if managed_gateway:
                    # A systemd-MANAGED gateway has no unisolated fallback
                    # to degrade to: the "plain" argv is itself a
                    # systemd-run scope wrap (the restart-safe one), so
                    # retrying here would re-enter systemd-run under a
                    # DIFFERENT unit name, with no launch probe and no
                    # stderr capture, while the row recorded scope "" for
                    # a worker that is in fact scoped. Fail closed on the
                    # refusal's own stderr, exactly as strict mode does.
                    _kanban_db._log.warning(
                        "kanban dispatch: task %s systemd-run launch failed "
                        "(rc=%s: %s) — this gateway is systemd-managed, so "
                        "there is no unisolated fallback to retry with; "
                        "recording spawn_failed",
                        task.id,
                        proc.returncode,
                        err_text or "no stderr captured",
                    )
                # systemd-scope mode (and every managed gateway) never
                # degrades silently: fail loudly so the dispatcher records
                # spawn_failed with the cause.
                spawn_error = err_text
                raise RuntimeError(
                    f"systemd-run launch failed for task {task.id} "
                    f"(rc={proc.returncode}): "
                    f"{err_text or 'no stderr captured'}"
                )
            # Launcher alive at the deadline or the unit appeared — the
            # run is live. Drop the spawn-stderr capture so successful
            # runs leave no stale .spawn.err artifacts behind.
            spawn_err_f.close()
            spawn_err_f = None
            try:
                os.unlink(spawn_err_path)
            except OSError:
                pass
    except BaseException:
        log_f.close()
        raise
    finally:
        log_f.close()
        if spawn_err_f is not None:
            spawn_err_f.close()
    # NOTE: we intentionally do NOT close log_f here — we want Popen's
    # child process to keep writing after this function returns.  The
    # handle is kept alive by the child's inheritance.  The parent's
    # reference goes out of scope and is GC'd, but the OS-level FD stays
    # open in the child until the child exits.
    _kanban_db._log.info(
        "kanban dispatch: task %s worker pid %s running %s (isolation=%s)",
        task.id,
        proc.pid,
        f"in systemd scope {scope_unit}" if scope_unit else "unisolated",
        _kanban_worker_scope._resolve_worker_isolation(),
    )
    return _SpawnedWorkerPid(proc.pid, scope_unit)


from hermes_cli import kanban_db as _kanban_db
from hermes_cli import kanban_db_dispatch as _kanban_db_dispatch
from hermes_cli import kanban_worker_scope as _kanban_worker_scope
