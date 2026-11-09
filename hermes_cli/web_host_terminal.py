"""Host-terminal policy and process setup for the browser-hosted Desktop UI."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Optional

META_PREFIX = "\0HERMES_TERMINAL_META:"


def request_allowed() -> bool:
    """Allow host shells only for Webapp on loopback or behind authentication."""
    from hermes_cli.web_server import app
    from hermes_cli.web_server_chat import _LOOPBACK_HOSTS
    from hermes_cli.web_server_surface import policy

    if not policy(app.state).host_terminal:
        return False
    if getattr(app.state, "auth_required", False):
        return True
    return (getattr(app.state, "bound_host", "") or "").strip().lower() in _LOOPBACK_HOSTS


def shell_command(candidate: str) -> Optional[str]:
    """Return an executable shell path for ``candidate``, or None.

    A relative path is refused: the PTY chdirs into the workspace before exec,
    so it would name a different file than the one checked here.
    """
    from hermes_platform.resolver import locate_command

    raw = (candidate or "").strip()
    found = locate_command(raw).command if raw else ()
    return found[0] if found else None


def shell_spec() -> tuple[list[str], str]:
    """Resolve the interactive-shell ladder the native Desktop uses (electron/terminal-ipc.ts).

    POSIX honours ``$SHELL``, then /bin/zsh, /bin/bash, /bin/sh. Windows ignores ``$SHELL``
    (usually a stray MSYS/Git path) and takes PowerShell 7, Windows PowerShell, ``COMSPEC``,
    then cmd.exe. Electron's ``HERMES_DESKTOP_SHELL`` override is not read here: settings do
    not ride on ``HERMES_*`` environment variables.
    """
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
        windows_powershell = Path(system_root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        ladder = ("pwsh.exe", "pwsh", str(windows_powershell), "powershell.exe", os.environ.get("COMSPEC", ""))
        fallback = "cmd.exe"
    else:
        ladder = (os.environ.get("SHELL", ""), "/bin/zsh", "/bin/bash", "/bin/sh")
        fallback = "/bin/sh"
    command = next(filter(None, map(shell_command, ladder)), fallback)

    name = Path(command).name.lower()
    if name.startswith(("pwsh", "powershell")):
        args = ["-NoLogo"]
    elif name.startswith("cmd"):
        args = []
    elif "zsh" in name or "bash" in name:
        args = ["-il"]
    else:
        args = ["-i"]
    return [command, *args], name


def safe_cwd(requested: Optional[str]) -> str:
    fallback = Path.home()
    try:
        candidate = Path((requested or "").strip() or fallback).expanduser().resolve()
        if candidate.is_dir():
            return str(candidate)
        if candidate.is_file():
            return str(candidate.parent)
    except (OSError, RuntimeError, ValueError):
        pass
    return str(fallback)


def terminal_lc_ctype(env: Mapping[str, str], platform: str) -> Optional[str]:
    """Return the LC_CTYPE a host shell's ``env`` needs added, or None to leave it alone.

    macOS accepts the bare charset name ``UTF-8`` as a locale; glibc rejects it,
    and LC_CTYPE outranks LANG, so forcing it on Linux breaks even a valid LANG.
    There a set LANG is left in charge (copying it would pin a value a login rc
    may still change) and only a missing one falls back to glibc's C.UTF-8.
    Windows shells take their code page from the console, not LC_CTYPE. Desktop's
    terminalLcCtype (electron/terminal-ipc.ts) shares the darwin and C.UTF-8 choices
    but copies LANG into LC_CTYPE and does not defer to LC_ALL.
    """
    if env.get("LC_ALL") or env.get("LC_CTYPE") or platform == "win32":
        return None
    if platform == "darwin":
        return "UTF-8"
    return None if env.get("LANG") else "C.UTF-8"


def resolve_argv(
    *, home: Path, requested_cwd: Optional[str] = None,
) -> tuple[list[str], str, dict[str, str], str]:
    """Return argv/cwd/env/name for Webapp's authenticated host terminal in ``home``."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli import __version__
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_constants import (
        get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )
    from hermes_platform.host.facts import os_family
    from tools.environments.local import build_subprocess_env, served_profile_child_env
    from tools.terminal_scope import enforce_no_refusal, get_terminal_scope
    from tui_gateway.launch_profile_policy import (
        activate_multi_profile_hosting, launch_profile_runtime_scope,
    )

    if home.resolve() != get_process_hermes_home().resolve():
        activate_multi_profile_hosting()
        scope = _profile_runtime_scope(home)
    else:
        scope = launch_profile_runtime_scope(home)
    # The startup eager-multiplex guard applies even before a secondary's first
    # request. Home alone cannot authorize passthrough reads; both shell paths
    # need the same complete scope, including the frozen launch environment.
    override_token = set_hermes_home_override(str(home))
    try:
        with scope:
            enforce_no_refusal()
            base_env = served_profile_child_env(target_home=home)
            for env_var in TERMINAL_CONFIG_ENV_MAP.values():
                base_env.pop(env_var, None)
            terminal_scope = get_terminal_scope()
            assert terminal_scope is not None  # both runtime scopes bind the complete policy
            base_env.update(terminal_scope)
            env = build_subprocess_env(base=base_env, scrub_secrets=True)
    finally:
        reset_hermes_home_override(override_token)

    for key in list(env):
        if key == "npm_config_prefix" or key.startswith(("npm_config_", "npm_package_")):
            env.pop(key, None)
    for key in ("NO_COLOR", "FORCE_COLOR", "COLORFGBG"):
        env.pop(key, None)
    env["COLORTERM"] = "truecolor"
    env["TERM"] = "xterm-256color"
    env["TERM_PROGRAM"] = "Hermes"
    env["TERM_PROGRAM_VERSION"] = __version__
    env["HERMES_DESKTOP_TERMINAL"] = "1"
    if lc_ctype := terminal_lc_ctype(env, os_family()):
        env["LC_CTYPE"] = lc_ctype

    argv, shell_name = shell_spec()
    return argv, safe_cwd(requested_cwd), env, shell_name


def query_dimension(raw: Optional[str], default: int, maximum: int) -> int:
    try:
        value = int(raw or default)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(2, min(maximum, value))
