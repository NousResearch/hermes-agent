"""Source of the detached gateway-restart watcher (``hermes_cli.gateway._spawn_gateway_restart_watcher``).

The watcher runs as ``python -c <source> <old_pid> <respawn argv…>`` on whatever interpreter the
updater runs on (the bare store Python after the package-manager handoff), so the source imports
only stdlib-safe Hermes modules. It respawns the gateway once the old one is gone: on POSIX it
waits as long as that takes (a busy gateway's work is never killed on a timer), judging "gone" by
the old process's ``(pid, start time)`` so a reused PID never holds it.
"""

from __future__ import annotations

import json
import textwrap

# Module level: the watcher is built after the update moved the checkout (post-swap imports test).
from hermes_cli._subprocess_compat import process_start_time_stdlib

_SOURCE = textwrap.dedent(
    """
        import sys
        if sys.path[:1] == [""]: del sys.path[0]  # -c puts the cwd first: never import from it
        import os
        import subprocess
        import time
        # Stdlib-only imports: after the package-manager handoff this is the bare store Python;
        # name the checkout explicitly.
        sys.path.insert(0, {project_root_literal})
        from hermes_cli._subprocess_compat import (
            _WINDOWS_GATEWAY_BREAKAWAY_ENV, pid_exists_stdlib, process_start_time_stdlib,
            windows_detach_flags, windows_detach_flags_without_breakaway,
        )
        from hermes_cli.gateway_launchd_runtime import start_gateway_job

        pid = int(sys.argv[1])
        cmd = sys.argv[2:]
        _respawn_cwd = {respawn_cwd_literal}
        _respawn_env_overlay = {respawn_env_literal}
        deadline = time.monotonic() + {watcher_timeout_literal} if sys.platform == "win32" else None
        # The old gateway is (pid, start time): a PID reused by another process between polls ends
        # the wait like an exit does. ``None`` (no stdlib start-time read here): PID existence only.
        _start_time = {start_time_literal}
        while deadline is None or time.monotonic() < deadline:
            # ``os.kill(pid, 0)`` is not a no-op on Windows — use the cross-platform existence check.
            if not pid_exists_stdlib(pid):
                break
            if _start_time is not None and process_start_time_stdlib(pid) != _start_time:
                break
            time.sleep(0.2)

        # Respawned stdout/stderr go to _spawn_detached's sidecar log: with DEVNULL a gateway killed moments
        # after respawn (Job Object teardown when breakaway is denied) left ZERO trace. Best-effort.
        _stdio_target = subprocess.DEVNULL
        _stdio_fh = None
        try:
            from hermes_constants import get_hermes_home
            from pathlib import Path
            _log_dir = Path(get_hermes_home()) / "logs"
            _log_dir.mkdir(parents=True, exist_ok=True)
            _stdio_fh = open(_log_dir / "gateway-stdio.log", "ab", buffering=0)
            _stdio_target = _stdio_fh
        except Exception:
            pass

        # Detach the respawn: POSIX setsid; Windows needs CREATE_BREAKAWAY_FROM_JOB, since the watcher may sit in a job
        # object (Electron/Tauri parent) whose teardown would otherwise kill it (_subprocess_compat.windows_detach_flags).
        _popen_kwargs = {{"stdout": _stdio_target, "stderr": _stdio_target}}
        # Stable cwd + env overlay (VIRTUAL_ENV/PYTHONPATH/HERMES_HOME) the windowless interpreter needs; empty on POSIX.
        if _respawn_cwd:
            _popen_kwargs["cwd"] = _respawn_cwd
        _base_env = {{**os.environ, **_respawn_env_overlay}}
        try:
            if sys.platform == "win32":
                try:
                    _popen_kwargs["creationflags"] = windows_detach_flags()
                    # Stamp the breakaway state exactly like gateway_windows._spawn_detached so the
                    # respawned gateway's exit-diag / lifecycle records show whether it escaped the
                    # parent Job Object (a job-teardown kill is otherwise indistinguishable).
                    _popen_kwargs["env"] = {{**_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "1"}}
                    subprocess.Popen(cmd, **_popen_kwargs)
                except OSError:
                    # CREATE_BREAKAWAY_FROM_JOB is rejected with ERROR_ACCESS_DENIED when the parent's
                    # job object refuses breakaway; retry without it (mirrors _spawn_detached).
                    _popen_kwargs["creationflags"] = windows_detach_flags_without_breakaway()
                    _popen_kwargs["env"] = {{**_base_env, _WINDOWS_GATEWAY_BREAKAWAY_ENV: "0"}}
                    subprocess.Popen(cmd, **_popen_kwargs)
            elif sys.platform == "darwin" and _stdio_fh and start_gateway_job(
                    cmd, env=_base_env, cwd=os.getcwd(), log_path=_log_dir / "gateway-stdio.log"):
                pass  # a launchd job: its LAN access outlives whoever ran this one
            else:
                if _respawn_env_overlay:
                    _popen_kwargs["env"] = _base_env
                _popen_kwargs["start_new_session"] = True
                subprocess.Popen(cmd, **_popen_kwargs)
        finally:
            if _stdio_fh is not None:
                try:
                    _stdio_fh.close()
                except OSError:
                    pass
    """
).strip()


def watcher_source(old_pid: int, *, respawn_cwd: str, respawn_env_overlay: dict[str, str],
                   timeout_s: float, project_root: str) -> str:
    """The watcher program for *old_pid*, whose start time is read now, while the parent knows it
    is the gateway being replaced. Literals ride the source, not argv."""
    return _SOURCE.format(respawn_cwd_literal=json.dumps(respawn_cwd),
                          respawn_env_literal=json.dumps(respawn_env_overlay),
                          watcher_timeout_literal=json.dumps(timeout_s),
                          project_root_literal=json.dumps(project_root),
                          # repr, not json: None (no /proc on macOS) must stay a Python literal.
                          start_time_literal=repr(process_start_time_stdlib(old_pid)))
