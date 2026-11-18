"""``hermes dashboard password``: set or change the dashboard's username and password.

The bundled ``basic`` sign-in provider (``plugins/dashboard_auth/basic``) reads
``dashboard.basic_auth.{username,password_hash,secret}`` from ``config.yaml``, and every listener that
mounts the dashboard app reads it there: ``hermes dashboard`` / ``hermes serve`` and the gateway's own
listener that serves the Webapp (``gateway.run_api.GatewayDashboardMount``), each when it starts. Until
now the only way to write it was the TTY prompt ``hermes dashboard`` offers when its sign-in gate finds no
provider. This command writes the same fields the same way (:func:`store_basic_auth`, shared with that
prompt): an scrypt hash only, never the password; a signing secret when ``config.yaml`` has none, so
sign-ins survive a restart. It reads the password from a TTY prompt, or with ``--password-stdin`` from
standard input for scripts and UIs, never from argv, and never prints or logs it.

``HERMES_DASHBOARD_BASIC_AUTH_{USERNAME,PASSWORD,PASSWORD_HASH}`` win over ``config.yaml`` when set, so
the ones in this home's ``.env`` are removed (named, never shown) and ones set by the environment the
command runs in are named as still overriding.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from typing import Callable, NoReturn, Optional

#: The environment names the provider prefers over ``dashboard.basic_auth`` (``basic._settings``).
SHADOWING_ENV = ("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD",
                 "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD_HASH")

#: Longest password ``--password-stdin`` accepts, in bytes of its UTF-8 encoding.
MAX_PASSWORD_BYTES = 1024

EXIT_INVALID = 2


class DashboardPasswordError(Exception):
    """The credential could not be stored; ``str()`` says why (never the password)."""


def store_basic_auth(username: str, password: str, *, on_reenabled: Optional[Callable[[], None]] = None) -> bool:
    """Write *username* and the scrypt hash of *password* under ``dashboard.basic_auth`` in this home's
    ``config.yaml``, clear any plaintext ``password`` there and add a signing secret when the section has
    none (returns whether it did). Re-enable the bundled ``basic`` plugin when ``plugins.disabled`` parks
    it; *on_reenabled* runs then, before the file is written. Raises :class:`DashboardPasswordError`."""
    try:
        from plugins.dashboard_auth.basic import hash_password
    except Exception as exc:
        raise DashboardPasswordError(f"Could not load the password provider: {exc}") from exc

    password_hash = hash_password(password)
    # A stable token-signing secret so sessions survive a dashboard restart.
    secret = secrets.token_urlsafe(32)

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins_cmd import ensure_basic_auth_plugin_enabled_in_config
        cfg = load_config()
        basic = cfg.setdefault("dashboard", {}).setdefault("basic_auth", {})
        basic["username"] = username
        basic["password_hash"] = password_hash
        basic["password"] = ""  # never persist plaintext
        created_secret = not str(basic.get("secret", "") or "").strip()
        if created_secret:
            basic["secret"] = secret
        # The bundled basic provider is a backend plugin that honours plugins.disabled; unblock it.
        if ensure_basic_auth_plugin_enabled_in_config(cfg) and on_reenabled is not None:
            on_reenabled()
        save_config(cfg)
    except Exception as exc:
        raise DashboardPasswordError(f"Failed to write config.yaml: {exc}") from exc
    return created_secret


def _refuse(message: str) -> NoReturn:
    print(f"✗ {message}; nothing was changed.", file=sys.stderr)
    raise SystemExit(EXIT_INVALID)


def _read_password_stdin() -> str:
    """The first line of standard input without its ``\\n`` or ``\\r\\n``, as UTF-8 text. Refuses a terminal:
    what is typed there is shown, so the hidden prompt (no ``--password-stdin``) is the way to type it."""
    if sys.stdin.isatty():
        _refuse("--password-stdin reads a pipe or a file, and standard input is a terminal that would show "
                "the password: run without --password-stdin to type it at a hidden prompt")
    line = sys.stdin.buffer.readline(MAX_PASSWORD_BYTES + 3)
    line = line[:-2] if line.endswith(b"\r\n") else line.removesuffix(b"\n")
    if len(line) > MAX_PASSWORD_BYTES:
        _refuse(f"The password is longer than {MAX_PASSWORD_BYTES} bytes")
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        _refuse("The password on standard input is not UTF-8 text")


def _read_password(args) -> str:
    """The password from stdin (``--password-stdin``) or a hidden TTY prompt asked twice. Raises
    :class:`SystemExit` with :data:`EXIT_INVALID` when there is neither."""
    if getattr(args, "password_stdin", False):
        return _read_password_stdin()
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("✗ No terminal to ask for the password: pass --password-stdin and write it to standard input.",
              file=sys.stderr)
        raise SystemExit(EXIT_INVALID)
    import getpass
    try:
        password = getpass.getpass("  Password: ")
        confirm = getpass.getpass("  Confirm password: ")
    except (EOFError, KeyboardInterrupt):
        print("\n  Cancelled.", file=sys.stderr)
        raise SystemExit(1) from None
    if password != confirm:
        _refuse("Passwords don't match")
    return password


def _current_username() -> str:
    from hermes_cli.config import load_config_readonly
    from hermes_cli.env_loader import resolve_home_dotenv
    from hermes_constants import get_hermes_home
    local_username = resolve_home_dotenv(get_hermes_home()).get("HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "").strip()
    if local_username:
        return local_username
    section = ((load_config_readonly().get("dashboard") or {}).get("basic_auth") or {})
    return str(section.get("username") or "").strip() if isinstance(section, dict) else ""


def _shadow_refusal() -> Optional[str]:
    """Why the overriding names cannot be cleared (an administrator-pinned one), or None. Asked before
    config.yaml is written: ``remove_env_value`` refuses by returning False."""
    from hermes_cli.config import _ENV_REF_RE, _env_ref_var_name, load_env, read_raw_config, require_env_writable
    for name in SHADOWING_ENV:
        try:
            require_env_writable(name, "remove")
        except ValueError as exc:
            return f"{exc} It overrides config.yaml"
    raw = ((read_raw_config().get("dashboard") or {}).get("basic_auth") or {})
    raw_user = raw.get("username") if isinstance(raw, dict) else None
    if isinstance(raw_user, str):
        removing = set(load_env()).intersection(SHADOWING_ENV)
        if any(_env_ref_var_name(ref.strip()) in removing for ref in _ENV_REF_RE.findall(raw_user)):
            return "Set dashboard.basic_auth.username to a literal first: it references an override this command removes"
    return None


def _inherited_shadows() -> list[str]:
    """The names in :data:`SHADOWING_ENV` the environment this command was started in sets (not this
    home's ``.env``, which the CLI loads over it and this command clears). Read before that clearing."""
    from hermes_cli.env_loader import environ_before_dotenv
    return [name for name in SHADOWING_ENV if (environ_before_dotenv(name) or "").strip()]


def _clear_dotenv_shadows() -> tuple[list[str], list[str]]:
    """Remove the names in :data:`SHADOWING_ENV` from this home's ``.env``; return (removed, still there)."""
    from hermes_cli.config import load_env, remove_env_value
    in_dotenv = [name for name in SHADOWING_ENV if name in load_env()]
    removed, left = [], []
    for name in in_dotenv:
        try:
            gone = remove_env_value(name)
        except OSError:
            gone = False
        (removed if gone else left).append(name)
    return removed, left


def _restart_lines() -> list[str]:
    """Which running listeners must restart to use the new sign-in (they read it when they start)."""
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway import _profile_arg
    from hermes_cli.web_server_port_holder import running_gateway
    hermes = " ".join(("hermes", _profile_arg())).strip()
    lines = []
    if running_gateway() is not None:
        lines.append(f"  The gateway is running: restart it to use the new sign-in ({hermes} gateway restart).")
    try:
        from hermes_cli.main_dashboard import _find_stale_dashboard_pids
        backends = _find_stale_dashboard_pids(scope_home=str(get_hermes_home()))
    except (OSError, subprocess.SubprocessError):
        backends = []  # a failed scan only drops this hint
    if backends:
        pids = ", ".join(str(pid) for pid in backends)
        lines.append(f"  hermes dashboard / hermes serve is running (PID {pids}): restart it to use the new sign-in.")
    if not lines:
        lines.append("  A gateway or hermes dashboard / hermes serve started from now on uses it.")
    return lines


def cmd_dashboard_password(args) -> int:
    from hermes_cli.config import is_managed, managed_error
    if is_managed():
        managed_error("set the dashboard password")
        return 1
    username = (getattr(args, "username", None) or "").strip() or _current_username() or "admin"
    if any(ch.isspace() for ch in username):
        print("✗ The username cannot contain spaces.", file=sys.stderr)
        return EXIT_INVALID
    password = _read_password(args)
    if not password:
        print("✗ Empty password; nothing was changed.", file=sys.stderr)
        return EXIT_INVALID
    refusal = _shadow_refusal()
    if refusal:
        print(f"✗ {refusal}; nothing was changed.", file=sys.stderr)
        return 1
    inherited = _inherited_shadows()
    reenabled: list[bool] = []
    try:
        created_secret = store_basic_auth(username, password, on_reenabled=lambda: reenabled.append(True))
    except DashboardPasswordError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1
    removed, left = _clear_dotenv_shadows()
    if left:
        from hermes_cli.config import get_env_path
        done = f" and {', '.join(removed)} removed from .env" if removed else ""
        print(f"✗ config.yaml was updated{done}, but {', '.join(left)} could not be removed from {get_env_path()} "
              "and still overrides it: remove it there by hand, then restart.", file=sys.stderr)
        return 1
    print(f"✓ Dashboard sign-in set for user '{username}' (config.yaml, dashboard.basic_auth: a password hash, "
          "never the password).")
    if created_secret:
        print("  Created a signing secret (dashboard.basic_auth.secret) so sign-ins survive a restart.")
    if reenabled:
        print("  Re-enabled the bundled 'basic' auth plugin (it was in plugins.disabled).")
    if removed:
        print(f"  Removed {', '.join(removed)} from .env: it overrode config.yaml.")
    if inherited:
        print(f"  ⚠ {', '.join(inherited)} is set in the environment this command runs in and overrides "
              "config.yaml: a gateway or backend started from it (or from a service with the same "
              "Environment=) still signs in with that value, not the new one. Unset it there.")
    print("  Sign-ins made with the old password end once a listener restarts with the new one. The same "
          "sign-in works on hermes dashboard / hermes serve and on the gateway's Webapp; it is asked for on a "
          "non-loopback address or a public URL, while a loopback-only listener opens without it.")
    for line in _restart_lines():
        print(line)
    return 0
