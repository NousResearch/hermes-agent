"""Environment and shell line for the detached ``/restart`` helper (#152714).

The helper runs ``<python> -m hermes_cli.main gateway restart`` after the gateway exits, so it
must import the same Hermes the gateway runs, and its failure must leave a trace somewhere.
"""

import os
import shlex
from pathlib import Path
from typing import Optional


def _running_checkout_root() -> Optional[str]:
    """Root of the source checkout ``hermes_cli`` was imported from, or None for an install in
    site-packages (the interpreter finds that one without PYTHONPATH)."""
    from hermes_cli._startup_fast import project_root_str

    root = Path(project_root_str())
    if any(part in ("site-packages", "dist-packages") for part in root.parts):
        return None
    return str(root)


def keep_running_checkout_importable(watcher_env: dict) -> dict:
    """Put the running checkout back on the watcher's PYTHONPATH.

    The child-env builders strip Hermes-owned PYTHONPATH entries so ordinary children never
    inherit the gateway's checkout. The restart helper is the one child that must re-run THIS
    install: a gateway started through PYTHONPATH (``scripts/hermes-python``) otherwise respawns
    a helper that dies on ``No module named 'hermes_cli'``."""
    root = _running_checkout_root()
    if root:
        existing = [p for p in watcher_env.get("PYTHONPATH", "").split(os.pathsep) if p]
        if root not in existing:
            watcher_env["PYTHONPATH"] = os.pathsep.join([root, *existing])
    return watcher_env


def _detached_restart_log_path() -> Optional[Path]:
    """``<HERMES_HOME>/logs/detached-restart.log``, or None when the logs dir cannot be made."""
    from hermes_constants import get_hermes_home

    log_path = get_hermes_home() / "logs" / "detached-restart.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return log_path


def detached_restart_shell(cmd: str, current_pid: int, restart_after_s: float) -> str:
    """POSIX shell line: wait for ``current_pid`` to exit (bounded), then run ``<cmd> gateway
    restart``. Its output and exit status go to ``logs/detached-restart.log``: the gateway that
    could log a failed respawn is already gone."""
    wait = (
        f"deadline=$(( $(date +%s) + {int(restart_after_s)} )); "
        f"while kill -0 {current_pid} 2>/dev/null && [ $(date +%s) -lt $deadline ]; do sleep 0.2; done; "
    )
    log_path = _detached_restart_log_path()
    if log_path is None:
        return f"{wait}{cmd} gateway restart"
    log_arg = shlex.quote(str(log_path))
    # An unopenable log must not skip the restart: bash drops a command whose redirect fails.
    return (
        f"{wait}if : >> {log_arg} 2>/dev/null; then "
        f"{cmd} gateway restart >> {log_arg} 2>&1; "
        f"echo \"detached restart exited with status $?\" >> {log_arg}; "
        f"else {cmd} gateway restart; fi"
    )
