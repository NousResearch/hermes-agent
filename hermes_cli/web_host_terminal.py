"""Host-terminal policy and process setup for the browser-hosted Desktop UI."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Set
from pathlib import Path
from typing import Optional


def request_allowed(
    *,
    ui_surface: str,
    auth_required: bool,
    bound_host: str,
    loopback_hosts: Set[str],
) -> bool:
    """Allow host shells only for Webapp on loopback or behind authentication."""
    if ui_surface != "webapp":
        return False
    if auth_required:
        return True
    return bound_host.strip().lower() in loopback_hosts


def shell_command(candidate: str) -> Optional[str]:
    """Return an executable shell path for ``candidate``, or None."""
    raw = (candidate or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    try:
        if path.is_absolute() and path.is_file() and (
            os.name == "nt" or os.access(path, os.X_OK)
        ):
            return str(path)
    except OSError:
        pass
    return shutil.which(raw)


def shell_spec(
    resolve_shell: Callable[[str], Optional[str]] = shell_command,
) -> tuple[list[str], str]:
    """Resolve the same interactive-shell ladder the native Desktop uses."""
    override = (os.environ.get("HERMES_DESKTOP_SHELL") or "").strip()
    if os.name != "nt":
        override = override or (os.environ.get("SHELL") or "").strip()
    command = resolve_shell(override)

    if os.name == "nt":
        if not command:
            command = resolve_shell("pwsh.exe") or resolve_shell("pwsh")
        if not command:
            system_root = (
                os.environ.get("SystemRoot")
                or os.environ.get("windir")
                or r"C:\Windows"
            )
            command = resolve_shell(
                str(
                    Path(system_root)
                    / "System32"
                    / "WindowsPowerShell"
                    / "v1.0"
                    / "powershell.exe"
                )
            )
        command = command or resolve_shell("powershell.exe")
        command = command or resolve_shell(os.environ.get("COMSPEC", "")) or "cmd.exe"
    elif not command:
        command = next(
            (
                resolved
                for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh")
                if (resolved := resolve_shell(candidate))
            ),
            "/bin/sh",
        )

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


def resolve_argv(
    *,
    profile: Optional[str],
    requested_cwd: Optional[str],
    resolve_profile_dir: Callable[[str], Path],
    resolve_shell_spec: Callable[[], tuple[list[str], str]],
    resolve_cwd: Callable[[Optional[str]], str],
    version: str,
) -> tuple[list[str], str, dict[str, str], str]:
    """Return argv/cwd/env/name for Webapp's authenticated host terminal."""
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP, apply_terminal_config_to_env
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.environments.local import build_subprocess_env

    requested_profile = (profile or "").strip()
    profile_dir = None
    if requested_profile and requested_profile.lower() != "current":
        profile_dir = resolve_profile_dir(requested_profile)

    override_token = (
        set_hermes_home_override(str(profile_dir)) if profile_dir is not None else None
    )
    try:
        base_env = os.environ.copy()
        if profile_dir is not None:
            base_env["HERMES_HOME"] = str(profile_dir)
            for env_var in TERMINAL_CONFIG_ENV_MAP.values():
                base_env.pop(env_var, None)
        apply_terminal_config_to_env(env=base_env)
        env = build_subprocess_env(base=base_env, scrub_secrets=True)
    finally:
        if override_token is not None:
            reset_hermes_home_override(override_token)

    for key in list(env):
        if key == "npm_config_prefix" or key.startswith(("npm_config_", "npm_package_")):
            env.pop(key, None)
    for key in ("NO_COLOR", "FORCE_COLOR", "COLORFGBG"):
        env.pop(key, None)
    env["COLORTERM"] = "truecolor"
    env["TERM"] = "xterm-256color"
    env["TERM_PROGRAM"] = "Hermes"
    env["TERM_PROGRAM_VERSION"] = version
    env["HERMES_DESKTOP_TERMINAL"] = "1"
    env.setdefault("LC_CTYPE", "UTF-8")

    argv, shell_name = resolve_shell_spec()
    return argv, resolve_cwd(requested_cwd), env, shell_name


def query_dimension(raw: Optional[str], default: int, maximum: int) -> int:
    try:
        value = int(raw or default)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(2, min(maximum, value))
