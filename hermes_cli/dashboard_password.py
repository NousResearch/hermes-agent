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
command runs in are named as still overriding. On Windows, where environment names ignore case, a ``.env``
line with one of those names in another case refuses the change until it is deleted or renamed.
"""

from __future__ import annotations

import os
import secrets
import subprocess
import sys
from pathlib import Path
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
    from hermes_cli.config import _ENV_REF_RE, _env_ref_var_name, read_raw_config, require_env_writable
    for name in SHADOWING_ENV:
        try:
            require_env_writable(name, "remove")
        except ValueError as exc:
            return f"{exc} It overrides config.yaml"
    from hermes_cli import managed_scope
    removing = set(_dotenv_shadows())
    for config in (read_raw_config(), managed_scope.load_managed_config()):
        raw = ((config.get("dashboard") or {}).get("basic_auth") or {})
        raw_user = raw.get("username") if isinstance(raw, dict) else None
        if isinstance(raw_user, str) and any(
                _env_ref_var_name(ref.strip()) in removing for ref in _ENV_REF_RE.findall(raw_user)):
            return "Set dashboard.basic_auth.username to a literal first: it references an override this command removes"
    return None


def _checkout_refusal(project_env: Path, home: Optional[Path] = None) -> Optional[str]:
    """Why *project_env* (the checkout's ``.env``, which the CLI and the gateway load after the home's
    own on every start) would bring an override back, or None: evaluated like the loader does it
    (``${VAR}`` expansion, fill-gaps-only when the home has a ``.env``) against the environment as it
    is once this command removed the overrides from *home*'s ``.env``. Never edits that file: it is
    shared by every Hermes home run from the checkout."""
    from hermes_cli.env_loader import resolve_project_dotenv
    from hermes_constants import get_hermes_home
    supplied = resolve_project_dotenv(project_env, Path(home) if home is not None else get_hermes_home(),
                                      removed=frozenset(SHADOWING_ENV))
    found = [name for name in SHADOWING_ENV if supplied.get(name, "").strip()]
    if not found:
        return None
    return (f"{', '.join(found)} is set in {project_env} (this checkout's .env, loaded on every start "
            "after the home's) and overrides config.yaml: remove it there, then run this again")


def _unquoted(key: str) -> str:
    """*key* as python-dotenv (the loader) reads it: a name in single quotes (``'KEY'=value``) is ``KEY``."""
    return key[1:-1] if len(key) > 1 and key[0] == key[-1] == "'" else key


def _case_variant_refusal(env_files, *, is_windows: Optional[bool] = None) -> Optional[str]:
    """On Windows, why a line in one of *env_files* or in the administrator-managed ``.env`` (loaded last,
    with override) would still override the new sign-in, or None: its key is a :data:`SHADOWING_ENV` name in
    another case (``hermes_dashboard_basic_auth_password``), which Windows reads as that name, and which the
    rest of this command (exact names) neither removes nor judges. Names the file, line and key as spelled,
    never the value. Lines are read one at a time with the ``.env`` tokenizer ``load_env`` uses, so a
    look-alike line inside a quoted multi-line value counts too: the refusal is safe. Off Windows
    (*is_windows* defaults to this host) such a key is another variable."""
    from agent.secret_scope import _decode_env_bytes, _parse_env_text
    from hermes_cli import managed_scope
    from hermes_cli.config import _IS_WINDOWS
    if not (_IS_WINDOWS if is_windows is None else is_windows):
        return None

    def scan(paths) -> tuple[list[str], list[str]]:
        found, read_as = [], []
        for path in paths:
            try:
                lines = _decode_env_bytes(Path(path).read_bytes()).splitlines()
            except OSError:
                continue
            hits = [(number, key) for number, line in enumerate(lines, 1) for key in _parse_env_text(line)
                    if _unquoted(key).upper() in SHADOWING_ENV and _unquoted(key) not in SHADOWING_ENV]
            if hits:
                found.append(f"{path} " + ", ".join(f"line {number} ({key})" for number, key in hits))
                read_as += [_unquoted(key).upper() for _, key in hits]
        return found, read_as

    def overrides(read_as) -> str:
        return (f"Windows reads environment names in any case, so this overrides config.yaml as "
                f"{', '.join(name for name in SHADOWING_ENV if name in read_as)}")

    managed_dir = managed_scope.get_managed_dir()
    reasons = []
    found, read_as = scan(env_files)
    if found:
        which = "that line" if len(read_as) == 1 else "those lines"
        reasons.append(f"{'; '.join(found)}: {overrides(read_as)}, and this command handles only the upper-case "
                       f"spelling: delete or rename {which}, then run this command again")
    found, read_as = scan([managed_dir / ".env"] if managed_dir else [])
    if found:
        which = "that line" if len(read_as) == 1 else "those lines"
        reasons.append(f"{'; '.join(found)} (the administrator-managed .env, loaded last): {overrides(read_as)}, "
                       f"and editing your own .env cannot change it: an administrator has to delete or rename "
                       f"{which}")
    return "; ".join(reasons) or None


def _managed_config_refusal(username: str) -> Optional[str]:
    """Why the administrator's managed config.yaml keeps the new sign-in from taking effect, or None:
    a pinned ``password_hash``, or a pinned ``username`` other than *username*. ``save_config`` drops
    pinned keys with only a note, so this is asked before anything is written."""
    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars
    pinned = managed_scope.managed_config_keys()
    if not pinned:
        return None
    source = managed_scope.get_managed_dir() / "config.yaml"

    def is_pinned(key: str) -> bool:
        return any(key == item or key.startswith(item + ".") for item in pinned)

    if is_pinned("dashboard.basic_auth.password_hash"):
        return f"dashboard.basic_auth.password_hash is managed by your administrator ({source})"
    section = (managed_scope.load_managed_config().get("dashboard") or {}).get("basic_auth") or {}
    # The value in effect, expanded by the config loader's own rules (a ``${VAR}`` pin is not a rename).
    pinned_user = str(_expand_env_vars(section.get("username") or "")).strip() if isinstance(section, dict) else ""
    if is_pinned("dashboard.basic_auth.username") and pinned_user != username:
        return (f"dashboard.basic_auth.username is managed by your administrator ({source}) as "
                f"'{pinned_user}'; leave out --username to keep it")
    return None


def _inherited_shadows() -> list[str]:
    """The names in :data:`SHADOWING_ENV` the environment this command was started in sets (not this
    home's ``.env``, which the CLI loads over it and this command clears). Read before that clearing."""
    from hermes_cli.env_loader import environ_before_dotenv
    return [name for name in SHADOWING_ENV if (environ_before_dotenv(name) or "").strip()]


def _multiline_refusal() -> Optional[str]:
    """Why the overrides cannot be removed from this home's ``.env`` without changing other bytes, or None:
    it needs whole-assignment removal, is not UTF-8 text (which the rewrite would re-encode) and has an
    override assignment to remove. With none to remove, the file is left alone, as on main."""
    from hermes_cli.config import get_env_path
    from hermes_cli.env_loader import _quoted_name, dotenv_assigned, dotenv_multiline
    try:
        raw = get_env_path().read_bytes()
        raw.decode("utf-8")
        return None
    except OSError:
        return None
    except UnicodeDecodeError:
        assigned = set(SHADOWING_ENV) & set(dotenv_assigned(raw))
        if not assigned or not (dotenv_multiline(raw) or any(_quoted_name(raw, key) for key in assigned)):
            return None
    return (f"{get_env_path()} is not UTF-8 text, so "
            f"{', '.join(SHADOWING_ENV)} cannot be removed from it safely: remove them there by hand")


def _dotenv_shadows() -> list[str]:
    """The names in :data:`SHADOWING_ENV` this home's ``.env`` sets: the line reader's, plus the ones python-dotenv
    (the loader) assigns under a quoted name (``'KEY'=value``), which the line reader keeps quoted. The refusal
    before the write and the cleanup after it both use this list."""
    from hermes_cli.config import get_env_path, load_env
    from hermes_cli.env_loader import dotenv_assigned
    try:
        assigned = set(dotenv_assigned(get_env_path().read_bytes()))
    except OSError:
        assigned = set()
    return [name for name in SHADOWING_ENV if name in load_env() or name in assigned]


def _clear_dotenv_shadows() -> tuple[list[str], list[str]]:
    """Remove the names in :data:`SHADOWING_ENV` from this home's ``.env``; return (removed, still there).
    ``remove_env_value`` removes whole assignments from a ``.env`` with a quoted value over several lines, so a
    name the line reader saw only inside another value there is not one to remove."""
    from hermes_cli.config import get_env_path, remove_env_value
    from hermes_cli.env_loader import dotenv_assigned, dotenv_multiline
    in_dotenv = _dotenv_shadows()
    removed, left = [], []
    for name in in_dotenv:
        try:
            gone = remove_env_value(name)
        except OSError:
            gone = False
        (removed if gone else left).append(name)
    if left:
        try:
            raw = get_env_path().read_bytes()
        except OSError:
            return removed, left
        if dotenv_multiline(raw):
            left = [name for name in left if name in dotenv_assigned(raw)]
    return removed, left


def _hermes_command(home) -> str:
    """``hermes`` pinned to *home*, for a command this one prints: a bare ``hermes`` follows a sticky
    ``hermes profile use``, so the default home is named ``-p default`` and a profile ``--profile NAME``;
    a home that is neither gets ``HERMES_HOME=...``."""
    import shlex

    from hermes_cli.gateway import _profile_arg
    from hermes_constants import get_default_hermes_root
    if Path(home).resolve() == get_default_hermes_root().resolve():
        return "hermes -p default"
    profile_arg = _profile_arg(str(home))
    return f"hermes {profile_arg}" if profile_arg else f"HERMES_HOME={shlex.quote(str(home))} hermes"


def _profile_scope_lines(home) -> list[str]:
    """For a named profile: which listeners read its own sign-in. Dashboard sign-in belongs to a
    listener's launch home, and ``hermes -p NAME dashboard`` / ``serve`` open the machine dashboard
    unless ``--isolated``, so a profile's credentials reach only the listeners it launches itself."""
    from hermes_cli.gateway import _profile_arg
    from hermes_constants import get_default_hermes_root
    profile_arg = _profile_arg(str(home))
    if not profile_arg:
        return []
    name, own, machine = profile_arg.split()[-1], _hermes_command(home), _hermes_command(get_default_hermes_root())
    return [f"  This is profile '{name}''s own sign-in: {own} dashboard --isolated (or serve) and a gateway this "
            f"profile runs by itself use it. {own} dashboard without --isolated opens the machine dashboard, "
            "and a gateway serving several profiles signs in with its own home's credentials: set those with "
            f"{machine} dashboard password."]


def _restart_lines() -> list[str]:
    """Which running listeners must restart to use the new sign-in (they read it when they start)."""
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server_port_holder import running_gateway
    home = get_hermes_home()
    lines = _profile_scope_lines(home)
    restart = []
    gateway = running_gateway()
    if gateway is not None and Path(gateway.control_home).resolve() != Path(home).resolve():
        lines.append(f"  ⚠ The running gateway serves this profile from {gateway.control_home} and signs in with "
                     f"that home's credentials, not these: change them with "
                     f"{_hermes_command(gateway.control_home)} dashboard password.")
    elif gateway is not None:
        restart.append("  The gateway is running: restart it to use the new sign-in "
                       f"({_hermes_command(home)} gateway restart).")
    try:
        from hermes_cli.main_dashboard import _find_stale_dashboard_pids
        backends = _find_stale_dashboard_pids(scope_home=str(home))
    except (OSError, subprocess.SubprocessError):
        backends = []  # a failed scan only drops this hint
    if backends:
        pids = ", ".join(str(pid) for pid in backends)
        restart.append(f"  hermes dashboard / hermes serve is running (PID {pids}): restart it to use the new sign-in.")
    if not restart:
        restart.append("  A gateway or hermes dashboard / hermes serve started from now on uses it.")
    return lines + restart


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
    from hermes_cli._startup_fast import project_root_str
    from hermes_cli.config import get_env_path
    project_env = Path(project_root_str()) / ".env"
    refusal = (_case_variant_refusal([get_env_path(), project_env]) or _shadow_refusal() or _multiline_refusal()
               or _managed_config_refusal(username) or _checkout_refusal(project_env))
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
