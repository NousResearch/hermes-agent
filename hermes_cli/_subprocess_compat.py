"""Frozen-updater compatibility facade for subprocess helpers.

Current Hermes code imports the runtime owners directly. This module remains only because an
already-running pre-package-manager updater can lazy-import these historical symbols after the
checkout is replaced.
"""

from __future__ import annotations

from typing import NoReturn

from runtime.git_subprocess import (
    NO_DRIVER_DIFF_FLAGS,
    NO_LAZY_FETCH_ENV,
    bounded_git_probe,
    expose_pm_git,
    harden_git_argv,
    noninteractive_git_env,
    selected_git_env,
)
from runtime.process_identity import pid_exists_stdlib, pid_is_hermes
from runtime.processes import kill_popen_process_tree as kill_process_tree
from runtime.subprocess_compat import (
    IS_WINDOWS,
    bounded_probe_run,
    resolve_node_command,
    restore_ambient_pythonpath,
    split_command_line,
    suppress_platform_ver_console,
    windows_detach_flags,
    windows_detach_flags_without_breakaway,
    windows_detach_popen_kwargs,
    windows_hide_flags,
)


def run(cmd, **kwargs) -> NoReturn:
    """Stop an old updater at the handoff boundary instead of starting its installer."""
    from hermes_cli._old_updater import stop_for_relaunch

    stop_for_relaunch()


__all__ = [
    "IS_WINDOWS",
    "NO_DRIVER_DIFF_FLAGS",
    "NO_LAZY_FETCH_ENV",
    "bounded_git_probe",
    "bounded_probe_run",
    "expose_pm_git",
    "harden_git_argv",
    "kill_process_tree",
    "noninteractive_git_env",
    "pid_exists_stdlib",
    "pid_is_hermes",
    "resolve_node_command",
    "restore_ambient_pythonpath",
    "run",
    "selected_git_env",
    "split_command_line",
    "suppress_platform_ver_console",
    "windows_detach_flags",
    "windows_detach_flags_without_breakaway",
    "windows_detach_popen_kwargs",
    "windows_hide_flags",
]
