"""One detached-process primitive; callers must first prove owner/service absence."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from hermes_cli._subprocess_compat import windows_detach_popen_kwargs, _WINDOWS_GATEWAY_BREAKAWAY_ENV
from hermes_cli.gateway_runtime_service import RuntimeStartError, remaining


# One launch's policy and pins, never the shared daemon's: hook consent would become daemon-lifetime
# auto-approval for messaging/cron, and a Kanban worker's or terminal's pins would retarget every later
# session's board, cwd and source. HERMES_INFERENCE_* fallbacks stay (the daemon's own defaults).
DAEMON_STRIPPED_ENV = (
    "HERMES_PROFILE", "HERMES_YOLO_MODE", "HERMES_IGNORE_RULES", "HERMES_SAFE_MODE",
    "HERMES_IGNORE_USER_CONFIG", "HERMES_ACCEPT_HOOKS", "HERMES_SESSION_SOURCE", "TERMINAL_CWD",
    "HERMES_KANBAN_TASK", "HERMES_KANBAN_WORKSPACE", "HERMES_KANBAN_BRANCH", "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK", "HERMES_KANBAN_CLAIM_TTL_SECONDS", "HERMES_KANBAN_GOAL_MODE",
    "HERMES_KANBAN_GOAL_MAX_TURNS", "HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_BOARD", "HERMES_TENANT", "HERMES_SESSION_SOURCE_EXPLICIT", "HERMES_TURN_AUTHOR",
    # Human-presence/one-shot markers: inherited, every messaging/cron turn would read as attended or -q.
    "HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_SINGLE_QUERY_SESSION",
)


def spawn_unmanaged_gateway(profile_home: Path, *, deadline: float, idle_exit: bool = False) -> subprocess.Popen:
    """Request a daemon, not readiness. Refuse Windows no-breakaway fallback.

    No --replace, persistence, login changes, elevation, shell, or inherited stdio.
    The gateway runtime's own exclusive ownership fence arbitrates racing starts.
    ``idle_exit``: the daemon ends itself once idle (a chat/TUI/cron client's own start; never
    Desktop's ``gateway ensure``, which keeps its attached gateway for the app's lifetime).
    """
    home = profile_home.resolve()
    root = Path(__file__).resolve().parent.parent
    # A root home is only pinned by an explicit selector: without one the child's
    # _apply_profile_override follows the sticky active_profile and boots the wrong
    # profile's daemon. A <root>/profiles/<name> home is already trusted as-is.
    selector = [] if home.parent.name == "profiles" else ["--profile", "default"]
    command = [sys.executable, "-m", "hermes_cli.main", *selector, "gateway", "run", "--quiet",
               *(["--idle-exit"] if idle_exit else [])]
    env = dict(os.environ)
    if sys.platform == "win32":
        from hermes_cli.gateway_windows import windowless_gateway_restart_spec
        command, _, overlay = windowless_gateway_restart_spec(command)
        if not overlay:
            raise RuntimeStartError("windows_interpreter_unavailable")
        env.update(overlay)
        env[_WINDOWS_GATEWAY_BREAKAWAY_ENV] = "1"
    env.update(HERMES_HOME=str(home), HERMES_GATEWAY_DETACHED="1", PYTHONIOENCODING="utf-8")
    # Profile selection is the explicit home, not the invoking client's display
    # name. Launch-only approval/context/config flags ride the launching session's frozen
    # policy, not the shared daemon or every later session.
    from gateway.session_context import _VAR_MAP  # the launching turn's session identity, too
    for key in (*DAEMON_STRIPPED_ENV, *_VAR_MAP):
        env.pop(key, None)
    remaining(deadline)
    logs = home / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    try:
        with (logs / "gateway-stdio.log").open("ab", buffering=0) as output:
            remaining(deadline)
            return subprocess.Popen(command, cwd=root, env=env, close_fds=True,
                                    stdin=subprocess.DEVNULL, stdout=output, stderr=output,
                                    **windows_detach_popen_kwargs())
    except OSError as exc:
        reason = "windows_breakaway_unavailable" if sys.platform == "win32" else "spawn_failed"
        raise RuntimeStartError(reason) from exc


def stdio_log_path(profile_home: Path) -> Path:
    return profile_home / "logs" / "gateway-stdio.log"


def stdio_log_size(profile_home: Path) -> int:
    """Where this launch's output will start in the append-only stdio log."""
    try:
        return stdio_log_path(profile_home).stat().st_size
    except OSError:
        return 0


_BANNER_PREFIXES = ("┌", "│", "├", "└")
_TAIL_LINES = 15
_TAIL_BYTES = 64 * 1024


def startup_failure_report(profile_home: Path, offset: int, status: int, pid: int | None = None) -> str:
    """What a client prints when the gateway it started died before serving: the exit status, the
    redacted tail of THIS launch's stdio (the traceback lives there, written before logging is up and
    before the detached-stdio redactor is installed) and the command that shows the whole log."""
    path = stdio_log_path(profile_home)
    try:
        with path.open("rb") as log:
            log.seek(max(offset, path.stat().st_size - _TAIL_BYTES))
            text = log.read().decode("utf-8", errors="replace")
    except OSError:
        text = ""
    lines = [line.rstrip() for line in text.splitlines()
             if line.strip() and not line.lstrip().startswith(_BANNER_PREFIXES)][-_TAIL_LINES:]
    if not lines:
        # A clean refusal (config/credential verdict, exit 78) is logged, not printed: the runtime
        # status file names it.
        from gateway.status import read_runtime_status
        status_file = read_runtime_status(profile_home / "gateway_state.json") or {}
        if status_file.get("exit_reason") and pid is not None and status_file.get("pid") == pid:
            lines = [str(status_file["exit_reason"])[:2000]]
            path = profile_home / "logs" / "gateway.log"
    from agent.redact import redact_sensitive_text
    tail = redact_sensitive_text("\n".join(lines), force=True, redact_url_credentials=True)
    see = f'type "{path}"' if sys.platform == "win32" else f"tail -n 200 '{path}'"
    body = "\n".join("  " + line for line in tail.splitlines()) if tail else "  (it wrote no output)"
    return f"the gateway exited with status {status} before it could serve:\n{body}\nFull log: {see}"
