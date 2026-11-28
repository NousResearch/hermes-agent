"""Run an explicit user-scope gateway restart in the named account's context."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_USER_RESTART_TIMEOUT_SECONDS = 3600



def _gateway_module():
    from hermes_cli import gateway

    return gateway


def _lookup_user(username: str):
    import pwd

    return pwd.getpwnam(username)


def _explicit_profile_name() -> str | None:
    try:
        from hermes_cli.main import explicit_cli_profile
    except (ImportError, AttributeError):
        return None
    return explicit_cli_profile()


def _fail(gateway, message: str, code: int = 1) -> NoReturn:
    gateway.print_error(message)
    raise SystemExit(code)


def _target_environment(username: str, home: Path, shell: str, gateway) -> dict[str, str]:
    env = {
        "HOME": str(home),
        "USER": username,
        "LOGNAME": username,
        "SHELL": shell or "/bin/sh",
        "PATH": _SYSTEM_PATH,
        "HERMES_HOME": str(home / ".hermes"),
        "PYTHONPATH": str(gateway.PROJECT_ROOT),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "TERM": os.environ.get("TERM", "dumb"),
    }
    # This selects the XDG systemd user-unit directory; preserve it when explicitly set.
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg_config_home:
        env["XDG_CONFIG_HOME"] = xdg_config_home
    return env


def _child_argv() -> list[str]:
    profile = _explicit_profile_name()
    args = list(sys.argv[1:])
    return ["--profile", profile, *args] if profile else args


def _require_user_unit(gateway, username: str) -> None:
    unit_path = gateway.get_systemd_unit_path(system=False)
    if not unit_path.exists():
        _fail(
            gateway,
            f"No user-scope gateway unit for account {username}: {unit_path}\n"
            f"Install it as that account with `hermes gateway install`; refusing system-scope fallback.",
        )


def dispatch_user_target_restart(args) -> bool:
    """Handle ``gateway restart --user ACCOUNT``; return False for the target user's normal restart path."""
    username = getattr(args, "user", None)
    if username is None:
        return False

    gateway = _gateway_module()
    if getattr(args, "system", False):
        _fail(gateway, "Choose either --system or --user ACCOUNT, not both.", code=2)
    if not gateway.is_linux() or gateway.is_termux():
        _fail(gateway, "--user ACCOUNT is supported only for Linux systemd user services.", code=2)
    if not gateway.supports_systemd_services():
        _fail(gateway, "systemctl is unavailable; cannot target a user-scope gateway service.")

    try:
        account = _lookup_user(username)
    except KeyError:
        _fail(gateway, f"Unknown account for --user: {username}", code=2)

    euid = os.geteuid()  # windows-footgun: ok — the Linux-only guard above rejects this option first
    if euid not in (0, account.pw_uid):
        _fail(gateway, f"Targeting account {username} requires root unless it is the current user.")
    if euid == account.pw_uid:
        _require_user_unit(gateway, username)
        return False

    setpriv = shutil.which("setpriv", path=_SYSTEM_PATH)
    if not setpriv:
        _fail(gateway, "setpriv is required to safely switch to another account; no service was changed.")

    home = Path(account.pw_dir)
    if not home.is_absolute():
        _fail(gateway, f"Cannot resolve the home directory for account {username}: {account.pw_dir!r}")

    command = [
        setpriv,
        f"--reuid={account.pw_uid}",
        f"--regid={account.pw_gid}",
        "--init-groups",
        "--no-new-privs",
        "--",
        sys.executable,
        "-m",
        "hermes_cli.main",
        *_child_argv(),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=str(gateway.PROJECT_ROOT),
            env=_target_environment(username, home, account.pw_shell, gateway),
            check=False,
            timeout=_USER_RESTART_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        _fail(
            gateway,
            f"Restart as {username} timed out after {_USER_RESTART_TIMEOUT_SECONDS} seconds; "
            "check service status before retrying.",
        )
    except OSError as exc:
        _fail(gateway, f"Could not launch the gateway command as {username}: {exc}")
    if result.returncode:
        raise SystemExit(result.returncode)
    return True
