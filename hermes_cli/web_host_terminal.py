"""Host-terminal policy and process setup for the browser-hosted Desktop UI."""

from __future__ import annotations

import os
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
    """Resolve the same interactive-shell ladder the native Desktop uses."""
    override = (os.environ.get("HERMES_DESKTOP_SHELL") or "").strip()
    if os.name != "nt":
        override = override or (os.environ.get("SHELL") or "").strip()
    command = shell_command(override)

    if os.name == "nt":
        if not command:
            command = shell_command("pwsh.exe") or shell_command("pwsh")
        if not command:
            system_root = (
                os.environ.get("SystemRoot")
                or os.environ.get("windir")
                or r"C:\Windows"
            )
            command = shell_command(
                str(
                    Path(system_root)
                    / "System32"
                    / "WindowsPowerShell"
                    / "v1.0"
                    / "powershell.exe"
                )
            )
        command = command or shell_command("powershell.exe")
        command = command or shell_command(os.environ.get("COMSPEC", "")) or "cmd.exe"
    elif not command:
        command = next(
            (
                resolved
                for candidate in ("/bin/zsh", "/bin/bash", "/bin/sh")
                if (resolved := shell_command(candidate))
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
    *, home: Path, requested_cwd: Optional[str] = None,
) -> tuple[list[str], str, dict[str, str], str]:
    """Return argv/cwd/env/name for Webapp's authenticated host terminal in ``home``."""
    from gateway.run import _profile_runtime_scope
    from hermes_cli import __version__
    from hermes_cli.config import TERMINAL_CONFIG_ENV_MAP
    from hermes_constants import (
        get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override,
    )
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
    env.setdefault("LC_CTYPE", "UTF-8")

    argv, shell_name = shell_spec()
    return argv, safe_cwd(requested_cwd), env, shell_name


def query_dimension(raw: Optional[str], default: int, maximum: int) -> int:
    try:
        value = int(raw or default)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(2, min(maximum, value))
