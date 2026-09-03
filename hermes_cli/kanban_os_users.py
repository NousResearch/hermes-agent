"""Profile-to-Linux-user launch for Kanban workers.

Optional ``kanban.profile_os_users`` maps a canonical Hermes profile id to a
POSIX account. Missing or empty mapping preserves trusted-local-user spawn
(the dispatcher remains the gateway UID).

Mapped workers are launched as an argv list with no shell::

    sudo -n -H -E -u <user> -- <hermes argv...>

Privilege drop is fail-closed: a configured mapping never falls back to the
gateway UID. Same-UID mappings are rejected and must not be reported as
isolation. Root is refused.

Optional companion ``kanban.profile_os_homes`` maps a profile id to that
user's Hermes *runtime root* (``{pw_dir}/.hermes`` by default). OS ``HOME``
is always the passwd home directory and is never silently overloaded with
the Hermes root.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


# Portable POSIX username: leading letter or underscore, then alnum/_/- .
# Caps at 32 characters (Linux login.defs default). No dots, slashes, or
# shell metacharacters — those cannot appear because the regex forbids them.
_POSIX_USERNAME_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_FORBIDDEN_OS_USERS = frozenset({"root", "toor"})

_HERMES_ENV_KEEP_PREFIXES = ("HERMES_", "TERMINAL_")
_SECRET_KEY_SUFFIXES = (
    "_TOKEN",
    "_SECRET",
    "_API_KEY",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
)
_SECRET_KEY_PREFIXES = ("SSH_", "SUDO_", "CURSOR_")
_SECRET_BASENAMES = frozenset({
    ".env",
    "auth.json",
    "id_rsa",
    "id_ecdsa",
    "id_ed25519",
    "id_dsa",
    "credentials.json",
})

DEFAULT_GROUP = "hermes-kanban"
DEFAULT_SUDOERS_PATH = "/etc/sudoers.d/hermes-kanban-os-users"
DEFAULT_SUDO_BIN = "/usr/bin/sudo"
DEFAULT_ID_BIN = "/usr/bin/id"
DEFAULT_TEST_BIN = "/usr/bin/test"
DEFAULT_STATE_PATH = "/var/lib/hermes/kanban-os-users-state.json"
REQUIRED_MAPPED_HOME_FILES = ("config.yaml", ".env", "SOUL.md")
REQUIRED_MAPPED_HOME_DIRS = ("skills",)
DEFAULT_DEV_WORKSPACE = "/home/matt/Documents/WorkoutTracker"
DEFAULT_FLUTTER_SDK = "/home/matt/flutter"
DEFAULT_ANDROID_SDK = "/home/matt/Android/Sdk"
DEFAULT_JDK_HOME = "/home/matt/.local/opt/jdk-17"
DEFAULT_GH_BIN = "/usr/bin/gh"
DEFAULT_GIT_BIN = "/usr/bin/git"
VERSIONED_RUNTIME_ROOT = "/opt/hermes/kanban-os-users"
GITHUB_LS_REMOTE_URL = "https://github.com/NousResearch/hermes-agent.git"

# Env vars the mapped worker must inherit through sudo env_reset.
SUDO_ENV_KEEP = (
    "HERMES_HOME",
    "HERMES_PROFILE",
    "HERMES_BIN",
    "HERMES_TENANT",
    "HERMES_SESSION_SOURCE",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_WORKSPACES_ROOT",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_BRANCH",
    "HERMES_KANBAN_GOAL_MODE",
    "HERMES_KANBAN_GOAL_MAX_TURNS",
    "TERMINAL_CWD",
    "TERMINAL_TIMEOUT",
    "TERMINAL_MAX_FOREGROUND_TIMEOUT",
    "PATH",
    "LANG",
    "LC_ALL",
    "TZ",
    "PUB_CACHE",
    "GRADLE_USER_HOME",
    "ANDROID_SDK_ROOT",
    "ANDROID_HOME",
    "JAVA_HOME",
    "FLUTTER_ROOT",
)


class MappedLaunchError(RuntimeError):
    """A configured profile_os_users mapping cannot be honoured.

    Callers must not fall back to the gateway UID when this is raised.
    """


class IncompatiblePrincipalError(RuntimeError):
    """A pre-existing user/group does not match the planned account design."""


@dataclass
class PasswdEntry:
    pw_name: str
    pw_uid: int
    pw_gid: int
    pw_dir: str


@dataclass
class GroupEntry:
    gr_name: str
    gr_gid: int
    gr_mem: tuple[str, ...] = ()


@dataclass
class LaunchHooks:
    """Injectable boundary for tests. Production uses pwd / subprocess."""

    getpwnam: Optional[Callable[[str], PasswdEntry]] = None
    getgrnam: Optional[Callable[[str], GroupEntry]] = None
    geteuid: Optional[Callable[[], int]] = None
    run: Optional[Callable[..., Any]] = None
    sudo_bin: str = DEFAULT_SUDO_BIN
    id_bin: str = DEFAULT_ID_BIN
    test_bin: str = DEFAULT_TEST_BIN
    is_windows: Optional[bool] = None


# ---------------------------------------------------------------------------
# Parsing / validation
# ---------------------------------------------------------------------------


def validate_os_username(name: str) -> str:
    """Return a canonical POSIX username or raise ValueError."""
    if not isinstance(name, str):
        raise ValueError(f"OS username must be a string, got {type(name).__name__}")
    stripped = name.strip()
    if not stripped:
        raise ValueError("OS username cannot be empty")
    if stripped != name:
        # Leading/trailing whitespace is an injection / confusion hazard.
        raise ValueError(f"OS username {name!r} has surrounding whitespace")
    if stripped in _FORBIDDEN_OS_USERS or stripped.casefold() == "root":
        raise ValueError(
            f"Refusing to map a Kanban specialist to {stripped!r} — root is not allowed"
        )
    if not _POSIX_USERNAME_RE.match(stripped):
        raise ValueError(
            f"Invalid OS username {stripped!r}. Must match "
            f"[a-z_][a-z0-9_-]{{0,31}} (no dots, slashes, or metacharacters)"
        )
    return stripped


def _canonical_profile_id(name: str) -> str:
    from hermes_cli.profiles import normalize_profile_name, validate_profile_name

    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    return canon


def parse_profile_os_users(raw: Any) -> dict[str, str]:
    """Parse ``kanban.profile_os_users``.

    ``None``, missing, or ``{}`` yields an empty dict (trusted-local-user
    behaviour). Invalid types, profile ids, or usernames raise ValueError.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"kanban.profile_os_users must be a mapping of profile -> username, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"kanban.profile_os_users[{key!r}] must be a username string, "
                f"got {type(value).__name__}"
            )
        if not value.strip():
            raise ValueError(
                f"kanban.profile_os_users[{key!r}] is empty; omit the key instead"
            )
        profile = _canonical_profile_id(str(key))
        username = validate_os_username(value)
        out[profile] = username
    return out


def parse_profile_os_homes(raw: Any) -> dict[str, str]:
    """Parse optional ``kanban.profile_os_homes`` (Hermes runtime roots).

    Values must be absolute paths. This is *not* OS HOME.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"kanban.profile_os_homes must be a mapping of profile -> absolute path, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"kanban.profile_os_homes[{key!r}] must be a path string, "
                f"got {type(value).__name__}"
            )
        profile = _canonical_profile_id(str(key))
        path = value.strip()
        if not os.path.isabs(path):
            raise ValueError(
                f"kanban.profile_os_homes[{profile!r}] must be an absolute path, "
                f"got {path!r}"
            )
        out[profile] = path
    return out


def load_profile_os_user_config(
    cfg: Optional[Mapping[str, Any]] = None,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(profile_os_users, profile_os_homes)`` from Kanban config."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config

            loaded = load_config()
        except Exception:
            loaded = {}
        cfg = loaded if isinstance(loaded, Mapping) else {}
    kanban = cfg.get("kanban", {}) if isinstance(cfg, Mapping) else {}
    if not isinstance(kanban, Mapping):
        kanban = {}
    users = parse_profile_os_users(kanban.get("profile_os_users"))
    homes = parse_profile_os_homes(kanban.get("profile_os_homes"))
    return users, homes


def lookup_mapped_os_user(
    profile: str,
    mapping: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the mapped username for *profile*, or None if unmapped."""
    if mapping is None:
        mapping, _homes = load_profile_os_user_config()
    if not mapping:
        return None
    canon = _canonical_profile_id(profile)
    return mapping.get(canon)


# ---------------------------------------------------------------------------
# Argv / env construction (no shell)
# ---------------------------------------------------------------------------


def build_sudo_argv(
    username: str,
    inner_argv: Sequence[str],
    *,
    sudo_bin: str = DEFAULT_SUDO_BIN,
) -> list[str]:
    """Build ``sudo -n -H -E -u <user> -- <inner...>`` as a list.

    Never joins into a shell string. ``--`` terminates sudo options so a
    hostile inner argv cannot inject sudo flags.
    """
    user = validate_os_username(username)
    if not inner_argv:
        raise ValueError("inner argv must not be empty")
    inner = [str(part) for part in inner_argv]
    if any(part is None for part in inner_argv):
        raise ValueError("inner argv must not contain None")
    sudo = sudo_bin or DEFAULT_SUDO_BIN
    if not sudo or sudo != str(sudo):
        raise ValueError("sudo_bin must be a non-empty string")
    return [str(sudo), "-n", "-H", "-E", "-u", user, "--", *inner]


def is_sudo_wrapped(argv: Sequence[str]) -> bool:
    if len(argv) < 8:
        return False
    try:
        u_idx = list(argv).index("-u")
    except ValueError:
        return False
    return (
        os.path.basename(str(argv[0])) == "sudo"
        and "-n" in argv[: u_idx + 1]
        and "--" in argv
    )


def resolve_mapped_hermes_home(
    profile: str,
    pw_dir: str,
    *,
    homes: Optional[Mapping[str, str]] = None,
) -> str:
    """Hermes runtime home for a mapped profile (not OS HOME).

    Default: ``{pw_dir}/.hermes/profiles/{profile}`` for named profiles,
    ``{pw_dir}/.hermes`` for ``default``. ``kanban.profile_os_homes`` overrides
    the Hermes *root* only.
    """
    canon = _canonical_profile_id(profile)
    root = None
    if homes:
        root = homes.get(canon)
    if not root:
        root = str(Path(pw_dir) / ".hermes")
    root_path = Path(root)
    if canon == "default":
        return str(root_path)
    if root_path.name == canon and root_path.parent.name == "profiles":
        return str(root_path)
    return str(root_path / "profiles" / canon)


def _is_inherited_secret_key(key: str) -> bool:
    upper = key.upper()
    if any(upper.startswith(p) for p in _SECRET_KEY_PREFIXES):
        return True
    if any(upper.endswith(s) for s in _SECRET_KEY_SUFFIXES):
        return True
    return False


def build_mapped_env(
    env: Mapping[str, str],
    *,
    username: str,
    home: str,
    hermes_home: str,
) -> dict[str, str]:
    """Copy dispatcher env, pin HOME/HERMES_HOME, drop inherited secrets.

    Specialists must load credentials from their private ``HERMES_HOME/.env``,
    not from the gateway process environment (which may hold the operator's
    Cursor key, SSH agent, etc.).
    """
    out = {str(k): str(v) for k, v in env.items()}
    for key in list(out):
        if _is_inherited_secret_key(key):
            out.pop(key, None)
    out["HOME"] = home
    out["USER"] = username
    out["LOGNAME"] = username
    out["HERMES_HOME"] = hermes_home
    return out


# ---------------------------------------------------------------------------
# Preflight / fail-closed launch
# ---------------------------------------------------------------------------


def _default_getpwnam(name: str) -> PasswdEntry:
    import pwd

    try:
        pw = pwd.getpwnam(name)
    except KeyError as exc:
        raise MappedLaunchError(
            f"Mapped OS user {name!r} does not exist. "
            f"Create it with: sudo hermes kanban os-users setup --apply"
        ) from exc
    return PasswdEntry(pw.pw_name, int(pw.pw_uid), int(pw.pw_gid), pw.pw_dir)


def _default_run(argv: Sequence[str], **kwargs: Any) -> Any:
    import subprocess

    return subprocess.run(  # noqa: S603 — argv is a validated list, no shell
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=kwargs.get("timeout", 15),
        env=kwargs.get("env"),
    )


def _default_getgrnam(name: str) -> GroupEntry:
    import grp

    try:
        gr = grp.getgrnam(name)
    except KeyError as exc:
        raise KeyError(name) from exc
    return GroupEntry(gr.gr_name, int(gr.gr_gid), tuple(gr.gr_mem))


def _hooks_or_defaults(hooks: Optional[LaunchHooks]) -> LaunchHooks:
    h = hooks or LaunchHooks()
    if h.getpwnam is None:
        h.getpwnam = _default_getpwnam
    if h.getgrnam is None:
        h.getgrnam = _default_getgrnam
    if h.geteuid is None:
        h.geteuid = os.geteuid
    if h.run is None:
        h.run = _default_run
    if h.is_windows is None:
        h.is_windows = sys.platform.startswith("win")
    if not h.test_bin:
        h.test_bin = DEFAULT_TEST_BIN
    return h


def preflight_mapped_user(
    username: str,
    *,
    hooks: Optional[LaunchHooks] = None,
    hermes_home: Optional[str] = None,
    workspace: Optional[str] = None,
    board_db: Optional[str] = None,
    require_paths: bool = True,
) -> PasswdEntry:
    """Validate the mapped account and sudo -n before any useful work."""
    h = _hooks_or_defaults(hooks)
    if h.is_windows:
        raise MappedLaunchError(
            "kanban.profile_os_users is Linux-only; unset the mapping on Windows"
        )
    user = validate_os_username(username)
    assert h.getpwnam is not None
    pw = h.getpwnam(user)
    if pw.pw_uid == 0 or user in _FORBIDDEN_OS_USERS:
        raise MappedLaunchError(
            f"Refusing to launch a Kanban worker as {user!r} (uid {pw.pw_uid})"
        )
    assert h.geteuid is not None
    euid = int(h.geteuid())
    if pw.pw_uid == euid:
        raise MappedLaunchError(
            f"kanban.profile_os_users maps to {user!r} which is the same UID "
            f"as the dispatcher ({euid}). This is not isolation — refusing to "
            f"report it as such. Use a distinct Linux account or omit the mapping."
        )
    sudo_bin = h.sudo_bin or DEFAULT_SUDO_BIN
    if not os.path.isabs(sudo_bin):
        raise MappedLaunchError(
            f"sudo binary must be an absolute path, got {sudo_bin!r}"
        )
    if h.run is _default_run and not os.path.isfile(sudo_bin):
        raise MappedLaunchError(
            f"sudo binary {sudo_bin} is missing; cannot drop privileges for {user!r}"
        )
    id_argv = build_sudo_argv(user, [h.id_bin, "-u"], sudo_bin=sudo_bin)
    assert h.run is not None
    try:
        proc = h.run(id_argv, timeout=15)
    except FileNotFoundError as exc:
        raise MappedLaunchError(
            f"Non-interactive sudo is unavailable ({exc}). "
            f"Install sudoers via: sudo hermes kanban os-users setup --apply"
        ) from exc
    except Exception as exc:
        raise MappedLaunchError(f"sudo -n -u {user} -- id -u failed: {exc}") from exc
    rc = int(getattr(proc, "returncode", 1))
    stdout = str(getattr(proc, "stdout", None) or "")
    stderr = str(getattr(proc, "stderr", None) or "")
    if rc != 0:
        detail = (stderr or stdout).strip() or f"exit {rc}"
        raise MappedLaunchError(
            f"Non-interactive sudo denied for {user!r}: {detail}. "
            f"Install the drop-in from `hermes kanban os-users sudoers` "
            f"and do not enable profile_os_users until check passes."
        )
    reported = stdout.strip().splitlines()[0].strip() if stdout.strip() else ""
    if reported != str(pw.pw_uid):
        raise MappedLaunchError(
            f"sudo -n -u {user} reported uid {reported!r}, expected {pw.pw_uid}. "
            f"Refusing to launch (fail closed)."
        )
    if require_paths:
        _preflight_paths(
            pw,
            hermes_home=hermes_home,
            workspace=workspace,
            board_db=board_db,
            hooks=h,
        )
    return pw


def probe_user_access(
    username: str,
    path: str,
    *,
    mode: str,
    hooks: LaunchHooks,
    expect_ok: bool = True,
) -> tuple[bool, str]:
    """Fail-closed sudo -n probe via /usr/bin/test as *username*.

    *mode* is one of ``r``, ``w``, ``x``. When *expect_ok* is False the probe
    passes only if the target UID is denied (cross-home isolation).
    """
    if mode not in {"r", "w", "x"}:
        raise ValueError(f"unsupported probe mode {mode!r}")
    h = _hooks_or_defaults(hooks)
    test_bin = h.test_bin or DEFAULT_TEST_BIN
    argv = build_sudo_argv(username, [test_bin, f"-{mode}", path], sudo_bin=h.sudo_bin)
    assert h.run is not None
    try:
        proc = h.run(argv, timeout=15)
    except Exception as exc:
        return False, f"probe {username} {mode} {path} failed: {exc}"
    rc = int(getattr(proc, "returncode", 1))
    allowed = rc == 0
    if expect_ok:
        if allowed:
            return True, f"{username} can {mode} {path}"
        return False, f"{username} cannot {mode} {path} (exit {rc})"
    if not allowed:
        return True, f"{username} denied {mode} {path}"
    return False, f"{username} unexpectedly can {mode} {path}"


def prove_sqlite_wal_lifecycle(db_path: str) -> None:
    """Open *db_path* in WAL mode and prove sidecar create/write/unlink."""
    import sqlite3

    path = Path(db_path)
    if not path.parent.is_dir():
        raise MappedLaunchError(f"Shared Kanban DB parent {path.parent} is missing")
    con = sqlite3.connect(str(path), timeout=5)
    try:
        mode = con.execute("PRAGMA journal_mode=WAL").fetchone()
        if not mode or str(mode[0]).lower() != "wal":
            raise MappedLaunchError(f"Could not enable WAL on {path}: {mode!r}")
        con.execute("CREATE TABLE IF NOT EXISTS _os_users_probe (id INTEGER)")
        con.execute("INSERT INTO _os_users_probe (id) VALUES (1)")
        con.execute("DELETE FROM _os_users_probe")
        con.commit()
    finally:
        con.close()
    wal = path.parent / f"{path.name}-wal"
    shm = path.parent / f"{path.name}-shm"
    if not wal.exists() and not shm.exists():
        # Some sqlite builds unlink sidecars on close; directory write is enough.
        probe = path.parent / f".{path.name}.os-users-wal-probe"
        probe.write_bytes(b"probe")
        probe.unlink()


def mapped_home_ready(hermes_home: str | Path) -> tuple[bool, str]:
    """Required files/dirs must exist in mapped HERMES_HOME. Never reads secrets."""
    root = Path(hermes_home)
    missing: list[str] = []
    for name in REQUIRED_MAPPED_HOME_FILES:
        if not (root / name).is_file():
            missing.append(name)
    for name in REQUIRED_MAPPED_HOME_DIRS:
        if not (root / name).is_dir():
            missing.append(f"{name}/")
    if missing:
        return False, f"mapped HERMES_HOME missing required {missing}"
    return True, "config.yaml, .env, SOUL.md, skills/ present"


def _preflight_paths(
    pw: PasswdEntry,
    *,
    hermes_home: Optional[str],
    workspace: Optional[str],
    board_db: Optional[str],
    hooks: Optional[LaunchHooks] = None,
) -> None:
    h = _hooks_or_defaults(hooks) if hooks is not None else None
    if hermes_home:
        home_path = Path(hermes_home)
        if not home_path.is_dir():
            raise MappedLaunchError(
                f"Mapped HERMES_HOME {hermes_home} does not exist or is not a directory. "
                f"Provision it with: sudo hermes kanban os-users setup --apply"
            )
        try:
            st = home_path.stat()
        except OSError as exc:
            raise MappedLaunchError(
                f"Cannot stat mapped HERMES_HOME {hermes_home}: {exc}"
            ) from exc
        if stat.S_IMODE(st.st_mode) & 0o077:
            raise MappedLaunchError(
                f"Mapped HERMES_HOME {hermes_home} is group/world-accessible "
                f"(mode {stat.S_IMODE(st.st_mode):04o}); expected 0700"
            )
        if st.st_uid != pw.pw_uid:
            raise MappedLaunchError(
                f"Mapped HERMES_HOME {hermes_home} is owned by uid {st.st_uid}, "
                f"expected {pw.pw_uid} ({pw.pw_name})"
            )
        if h is not None:
            ok, detail = probe_user_access(
                pw.pw_name, hermes_home, mode="x", hooks=h, expect_ok=True
            )
            if not ok:
                raise MappedLaunchError(detail)
    if workspace:
        ws = Path(workspace)
        if not ws.is_dir():
            raise MappedLaunchError(
                f"Task workspace {workspace} is not an accessible directory"
            )
        if h is not None:
            ok, detail = probe_user_access(
                pw.pw_name, workspace, mode="x", hooks=h, expect_ok=True
            )
            if not ok:
                raise MappedLaunchError(detail)
    if board_db:
        db_path = Path(board_db)
        parent = db_path.parent
        if not parent.is_dir():
            raise MappedLaunchError(f"Shared Kanban DB parent {parent} is missing")
        if h is not None:
            ok_x, detail_x = probe_user_access(
                pw.pw_name, str(parent), mode="x", hooks=h, expect_ok=True
            )
            if not ok_x:
                raise MappedLaunchError(detail_x)
            ok_w, detail_w = probe_user_access(
                pw.pw_name, str(parent), mode="w", hooks=h, expect_ok=True
            )
            if not ok_w:
                raise MappedLaunchError(detail_w)


def apply_mapped_worker_launch(
    *,
    profile: str,
    argv: Sequence[str],
    env: Mapping[str, str],
    workspace: str = "",
    board_db: Optional[str] = None,
    mapping: Optional[Mapping[str, str]] = None,
    homes: Optional[Mapping[str, str]] = None,
    hooks: Optional[LaunchHooks] = None,
    preflight: bool = True,
) -> tuple[list[str], dict[str, str]]:
    """Maybe wrap *argv*/*env* for a mapped profile.

    Unmapped profiles return ``(list(argv), dict(env))`` unchanged.
    Mapped profiles always return sudo-wrapped argv; on any failure this
    raises ``MappedLaunchError`` instead of returning the inner argv.
    """
    inner = [str(p) for p in argv]
    out_env = dict(env)
    if mapping is None and homes is None:
        mapping, homes = load_profile_os_user_config()
    elif mapping is None:
        mapping, _ignored = load_profile_os_user_config()
    username = lookup_mapped_os_user(profile, mapping)
    if username is None:
        return inner, out_env
    h = _hooks_or_defaults(hooks)
    pw = (
        preflight_mapped_user(
            username,
            hooks=h,
            hermes_home=None,
            workspace=None,
            board_db=None,
            require_paths=False,
        )
        if preflight
        else _hooks_or_defaults(h).getpwnam(username)  # type: ignore[misc]
    )
    hermes_home = resolve_mapped_hermes_home(profile, pw.pw_dir, homes=homes or {})
    if preflight:
        _preflight_paths(
            pw,
            hermes_home=hermes_home,
            workspace=workspace,
            board_db=board_db,
            hooks=h,
        )
    wrapped = build_sudo_argv(username, inner, sudo_bin=h.sudo_bin)
    mapped_env = build_mapped_env(
        out_env,
        username=username,
        home=pw.pw_dir,
        hermes_home=hermes_home,
    )
    if not is_sudo_wrapped(wrapped):
        raise MappedLaunchError("internal error: sudo wrap produced unwrapped argv")
    if wrapped == inner:
        raise MappedLaunchError(
            "internal error: refusing to launch mapped worker unwrapped"
        )
    return wrapped, mapped_env


# ---------------------------------------------------------------------------
# Host setup / check / rollback (dry-run first; never prints secrets)
# ---------------------------------------------------------------------------


@dataclass
class SetupStep:
    title: str
    argv: list[str]
    privileged: bool = True
    creates: str = ""  # user:name | group:name | acl:path
    optional: bool = False  # skip when the install source path is missing


def _quote_cmd(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in argv)


def default_mapping_example() -> dict[str, str]:
    return {"dev": "hermes-dev", "sysadmin": "hermes-sysadmin"}


def _gateway_user() -> str:
    try:
        import pwd

        return pwd.getpwuid(os.geteuid()).pw_name
    except Exception:
        return os.environ.get("USER") or "matt"


def resolve_setup_targets(
    mapping: Optional[Mapping[str, str]] = None,
) -> dict[str, str]:
    if mapping:
        return dict(mapping)
    try:
        loaded, _homes = load_profile_os_user_config()
    except Exception:
        loaded = {}
    return loaded or default_mapping_example()


def shared_board_paths() -> dict[str, Path]:
    from hermes_cli.kanban_db import kanban_db_path, kanban_home, workspaces_root

    db = Path(kanban_db_path())
    root = Path(kanban_home())
    return {
        "hermes_root": root,
        "kanban_db": db,
        "kanban_dir": root / "kanban",
        "workspaces": Path(workspaces_root()),
        "logs": db.parent / "kanban" / "logs"
        if db.name != "kanban.db"
        else root / "kanban" / "logs",
    }


def render_sudoers(
    *,
    gateway_user: Optional[str] = None,
    mapping: Optional[Mapping[str, str]] = None,
    hermes_argv: Optional[Sequence[str]] = None,
    extra_bins: Optional[Sequence[str]] = None,
) -> str:
    gw = gateway_user or _gateway_user()
    users = sorted(set(resolve_setup_targets(mapping).values()))
    runas = ",".join(users) if users else "hermes-dev, hermes-sysadmin"
    keep = " ".join(SUDO_ENV_KEEP)
    if hermes_argv is None:
        try:
            from hermes_cli.kanban_db import _resolve_hermes_argv

            hermes_argv = _resolve_hermes_argv()
        except Exception:
            hermes_argv = ["/usr/bin/hermes"]
    cmd = _quote_cmd(hermes_argv)
    id_cmd = DEFAULT_ID_BIN
    test_cmd = DEFAULT_TEST_BIN
    lines = [
        f"# Hermes Kanban profile_os_users — generated, review before installing",
        f"# Install: sudo install -m 0440 this-file {DEFAULT_SUDOERS_PATH}",
        f"# Then: sudo visudo -c -f {DEFAULT_SUDOERS_PATH}",
        f"Defaults:{gw} !requiretty",
        f'Defaults:{gw} env_keep += "{keep}"',
        f"{gw} ALL=({runas}) NOPASSWD:SETENV: {id_cmd}",
        f"{gw} ALL=({runas}) NOPASSWD:SETENV: {test_cmd}",
        f"{gw} ALL=({runas}) NOPASSWD:SETENV: {cmd}",
    ]
    for bin_path in extra_bins or ():
        quoted = _quote_cmd([str(bin_path)])
        lines.append(f"{gw} ALL=({runas}) NOPASSWD:SETENV: {quoted}")
    lines.append("")
    return "\n".join(lines)


def _acl_traverse_ancestors(path: Path) -> list[Path]:
    """Ancestors of *path* from nearest parent up to (not including) / or /home."""
    out: list[Path] = []
    cur = Path(path)
    if not cur.is_absolute():
        cur = Path("/") / cur
    parent = cur.parent
    while str(parent) not in {"/", "/home", ""} and parent != cur:
        out.append(parent)
        nxt = parent.parent
        if nxt == parent:
            break
        cur, parent = parent, nxt
    out.reverse()
    return out


def empty_os_users_state() -> dict[str, Any]:
    return {
        "created_users": [],
        "created_groups": [],
        "preexisting_users": [],
        "preexisting_groups": [],
        "acl_paths": [],
    }


def load_os_users_state(path: str | Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    p = Path(path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return empty_os_users_state()
    if not isinstance(raw, dict):
        return empty_os_users_state()
    out = empty_os_users_state()
    for key in out:
        val = raw.get(key, [])
        if isinstance(val, list):
            out[key] = [str(x) for x in val]
    return out


def save_os_users_state(
    state: Mapping[str, Any], path: str | Path = DEFAULT_STATE_PATH
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    merged = empty_os_users_state()
    for key in merged:
        val = state.get(key, [])
        if isinstance(val, list):
            merged[key] = sorted({str(x) for x in val})
    p.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    os.chmod(p, 0o600)


def plan_setup_steps(
    *,
    mapping: Optional[Mapping[str, str]] = None,
    gateway_user: Optional[str] = None,
    group: str = DEFAULT_GROUP,
    dev_workspace: Optional[str] = None,
    hermes_argv: Optional[Sequence[str]] = None,
    board_paths: Optional[Mapping[str, Path]] = None,
    flutter_sdk: Optional[str] = None,
    android_sdk: Optional[str] = None,
    jdk_home: Optional[str] = None,
    include_db_migration: bool = False,
) -> list[SetupStep]:
    targets = resolve_setup_targets(mapping)
    gw = gateway_user or _gateway_user()
    steps: list[SetupStep] = [
        SetupStep(
            "create shared group",
            ["groupadd", "--system", group],
            creates=f"group:{group}",
        ),
    ]
    seen_users: set[str] = set()
    for profile, user in targets.items():
        if user in seen_users:
            continue
        seen_users.add(user)
        home = f"/home/{user}"
        steps.extend([
            SetupStep(
                f"create private primary group {user}",
                ["groupadd", "--system", user],
                creates=f"group:{user}",
            ),
            SetupStep(
                f"create {user}",
                [
                    "useradd",
                    "--system",
                    "--create-home",
                    "--home-dir",
                    home,
                    "--shell",
                    "/usr/sbin/nologin",
                    "-g",
                    user,
                    "-G",
                    group,
                    user,
                ],
                creates=f"user:{user}",
            ),
            SetupStep(
                f"private Hermes root for {user}",
                [
                    "install",
                    "-d",
                    "-m",
                    "0700",
                    "-o",
                    user,
                    "-g",
                    user,
                    f"{home}/.hermes",
                ],
            ),
            SetupStep(
                f"private profile dir for {profile}",
                [
                    "install",
                    "-d",
                    "-m",
                    "0700",
                    "-o",
                    user,
                    "-g",
                    user,
                    f"{home}/.hermes/profiles/{profile}",
                ],
            ),
        ])
    paths = board_paths
    if paths is None:
        try:
            paths = shared_board_paths()
        except Exception:
            paths = None
    if paths:
        from hermes_cli.kanban_os_users_rollout import (
            migrate_db_argv,
            reject_write_acl_on_hermes_root,
            self_hermes_argv,
            shared_board_acl_layout,
        )

        layout = shared_board_acl_layout(paths)
        writable_dir = Path(layout["writable_dir"])
        target_db = Path(layout["target_db"])
        live_db = Path(layout["live_db"])
        kdir = Path(layout["kanban_dir"])
        hermes_root = layout.get("hermes_root")
        argv_self = self_hermes_argv(hermes_argv)
        steps.append(
            SetupStep(
                "dedicated shared kanban dir (group wx lives here, never on ~/.hermes)",
                ["install", "-d", "-m", "2770", "-o", gw, "-g", group, str(kdir)],
            )
        )
        # Traverse-only on Hermes root and ancestors. Write ACLs only on the
        # dedicated kanban directory so specialists cannot create/rename files
        # in the default Hermes root.
        traverse_targets: list[Path] = []
        if hermes_root is not None:
            traverse_targets.append(Path(hermes_root))
            traverse_targets.extend(_acl_traverse_ancestors(Path(hermes_root)))
        traverse_targets.extend(_acl_traverse_ancestors(writable_dir))
        seen_traverse: set[str] = set()
        for ancestor in traverse_targets:
            key = str(ancestor)
            if key in seen_traverse:
                continue
            seen_traverse.add(key)
            if hermes_root is not None and str(ancestor) == str(writable_dir):
                continue
            steps.append(
                SetupStep(
                    f"traverse ACL on {ancestor} for {group} (execute only; no listdir, no write)",
                    ["setfacl", "-m", f"g:{group}:--x", str(ancestor)],
                    creates=f"acl:{ancestor}",
                )
            )
        steps.extend([
            SetupStep(
                f"ACL on dedicated DB parent {writable_dir} (group wx for WAL/shm; no listdir)",
                ["setfacl", "-m", f"g:{group}:wx", str(writable_dir)],
                creates=f"acl:{writable_dir}",
            ),
            SetupStep(
                f"default ACL on dedicated DB parent {writable_dir} for new WAL/shm files",
                ["setfacl", "-d", "-m", f"g:{group}:rw", str(writable_dir)],
                creates=f"acl:{writable_dir}",
            ),
            SetupStep(
                "ACL on dedicated kanban.db (SQLite needs sibling -wal/-shm)",
                ["setfacl", "-m", f"g:{group}:rw", str(target_db)],
                creates=f"acl:{target_db}",
                optional=True,
            ),
        ])
        if include_db_migration:
            steps.append(
                SetupStep(
                    f"sqlite backup {live_db} -> {target_db} (never cp a live DB/WAL/SHM)",
                    migrate_db_argv(live_db, target_db, hermes_argv=argv_self),
                    optional=True,
                )
            )
        ws = (
            str(paths["workspaces"])
            if "workspaces" in paths
            else str(kdir / "workspaces")
        )
        steps.append(
            SetupStep(
                "shared workspaces root (group rwx, no world)",
                ["install", "-d", "-m", "2770", "-o", gw, "-g", group, ws],
            )
        )
        if hermes_root is not None:
            reject_write_acl_on_hermes_root(steps, Path(hermes_root))
    if dev_workspace:
        dev_user = targets.get("dev", "hermes-dev")
        ws_path = Path(dev_workspace)
        for ancestor in _acl_traverse_ancestors(ws_path):
            steps.append(
                SetupStep(
                    f"dev-only traverse ACL on {ancestor} (not confidentiality; world-readable trees are not isolation)",
                    ["setfacl", "-m", f"u:{dev_user}:--x", str(ancestor)],
                    creates=f"acl:{ancestor}",
                )
            )
        steps.extend([
            SetupStep(
                f"dev-only recursive ACL on {dev_workspace} (sysadmin not granted; world-readable is not isolation)",
                ["setfacl", "-R", "-m", f"u:{dev_user}:rwx", str(ws_path)],
                creates=f"acl:{ws_path}",
            ),
            SetupStep(
                f"dev-only default ACL on {dev_workspace} for newly created files",
                ["setfacl", "-d", "-m", f"u:{dev_user}:rwx", str(ws_path)],
                creates=f"acl:{ws_path}",
            ),
        ])
    if flutter_sdk or android_sdk or jdk_home:
        dev_user = targets.get("dev", "hermes-dev")
        for label, raw in (
            ("flutter", flutter_sdk),
            ("android-sdk", android_sdk),
            ("jdk", jdk_home),
        ):
            if not raw:
                continue
            tool = Path(raw)
            for ancestor in _acl_traverse_ancestors(tool):
                steps.append(
                    SetupStep(
                        f"dev-only traverse ACL on {ancestor} for {label} (not a grant of /home/matt)",
                        ["setfacl", "-m", f"u:{dev_user}:--x", str(ancestor)],
                        creates=f"acl:{ancestor}",
                    )
                )
            steps.extend([
                SetupStep(
                    f"dev-only read/execute ACL on {tool} (no write; not recursive on /home/matt)",
                    ["setfacl", "-R", "-m", f"u:{dev_user}:r-x", str(tool)],
                    creates=f"acl:{tool}",
                    optional=True,
                ),
                SetupStep(
                    f"dev-only default r-x ACL on {tool}",
                    ["setfacl", "-d", "-m", f"u:{dev_user}:r-x", str(tool)],
                    creates=f"acl:{tool}",
                    optional=True,
                ),
            ])
        for cache in (
            ".cache/flutter",
            ".cache/pub",
            ".cache/gradle",
            ".cache/android",
        ):
            steps.append(
                SetupStep(
                    f"private {cache} for {dev_user}",
                    [
                        "install",
                        "-d",
                        "-m",
                        "0700",
                        "-o",
                        dev_user,
                        "-g",
                        dev_user,
                        f"/home/{dev_user}/{cache}",
                    ],
                )
            )
    steps.append(
        SetupStep(
            f"add {gw} to {group} for admin visibility",
            ["usermod", "-aG", group, gw],
        )
    )
    sudoers_body = render_sudoers(
        gateway_user=gw,
        mapping=targets,
        hermes_argv=hermes_argv,
    )
    # install via visudo check; body is written to a temp path by the CLI
    steps.append(
        SetupStep(
            "install sudoers drop-in (after visudo -c)",
            [
                "install",
                "-m",
                "0440",
                "-o",
                "root",
                "-g",
                "root",
                "<generated-sudoers>",
                DEFAULT_SUDOERS_PATH,
            ],
        )
    )
    _ = sudoers_body  # generated by the CLI printer, not interpolated here
    return steps


def plan_rollback_steps(
    *,
    mapping: Optional[Mapping[str, str]] = None,
    group: str = DEFAULT_GROUP,
    state: Optional[Mapping[str, Any]] = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    dev_users: Optional[Sequence[str]] = None,
) -> list[SetupStep]:
    targets = resolve_setup_targets(mapping)
    st = dict(state) if state is not None else load_os_users_state(state_path)
    created_users = list(st.get("created_users") or [])
    created_groups = list(st.get("created_groups") or [])
    preexisting_users = list(st.get("preexisting_users") or [])
    acl_paths = list(st.get("acl_paths") or [])
    steps: list[SetupStep] = [
        SetupStep("remove sudoers drop-in", ["rm", "-f", DEFAULT_SUDOERS_PATH]),
    ]
    acl_users = (
        list(dev_users) if dev_users is not None else [targets.get("dev", "hermes-dev")]
    )
    for path in acl_paths:
        steps.append(
            SetupStep(
                f"remove group ACL on {path}",
                ["setfacl", "-x", f"g:{group}", path],
            )
        )
        for user in acl_users:
            steps.append(
                SetupStep(
                    f"remove user ACL for {user} on {path}",
                    ["setfacl", "-x", f"u:{user}", path],
                )
            )
        steps.append(
            SetupStep(f"remove default ACL on {path}", ["setfacl", "-k", path])
        )
    has_state = bool(created_users or created_groups or preexisting_users or acl_paths)
    if not has_state:
        # No proof we created accounts — never userdel/groupdel pre-existing principals.
        return steps
    for user in created_users:
        steps.append(
            SetupStep(f"remove user {user} (created by setup)", ["userdel", "-r", user])
        )
    private = [g for g in created_groups if g != group]
    for gname in sorted(private):
        steps.append(
            SetupStep(f"remove group {gname} (created by setup)", ["groupdel", gname])
        )
    if group in created_groups:
        steps.append(
            SetupStep(f"remove group {group} (created by setup)", ["groupdel", group])
        )
    return steps


def format_plan(steps: Sequence[SetupStep], *, heading: str) -> str:
    lines = [
        heading,
        "",
        "All commands are argv (no shell). Review, then run as root.",
        "",
    ]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step.title}")
        lines.append(f"   {_quote_cmd(step.argv)}")
    lines.append("")
    lines.append("Do not cat/print .env, auth.json, gh tokens, or SSH keys.")
    lines.append(
        "MANUAL GATE: setup --apply does not copy profile files unless --migrate-profile-files."
    )
    lines.append(
        "Skills use bounded copy-tree (no per-file flood). Shared DB uses sqlite backup, not cp."
    )
    lines.append(
        "Required in mapped HERMES_HOME before check can report ready: "
        "config.yaml, .env, SOUL.md, skills/."
    )
    return "\n".join(lines)


def _source_hermes_root(source_root: Optional[Path] = None) -> Path:
    if source_root is not None:
        return Path(source_root)
    try:
        from hermes_constants import get_default_hermes_root

        return get_default_hermes_root()
    except Exception:
        return Path.home() / ".hermes"


def plan_migrate_steps(
    mapping: Mapping[str, str],
    *,
    source_root: Optional[Path] = None,
    hermes_argv: Optional[Sequence[str]] = None,
) -> list[SetupStep]:
    from hermes_cli.kanban_os_users_rollout import copy_tree_argv, self_hermes_argv

    root = _source_hermes_root(source_root)
    argv_self = self_hermes_argv(hermes_argv)
    steps: list[SetupStep] = []
    for profile, user in mapping.items():
        src = root / "profiles" / profile
        dst = Path(f"/home/{user}") / ".hermes" / "profiles" / profile
        for name in REQUIRED_MAPPED_HOME_FILES:
            steps.append(
                SetupStep(
                    f"migrate {name} for {profile} (never prints contents)",
                    [
                        "install",
                        "-m",
                        "0600",
                        "-o",
                        user,
                        "-g",
                        user,
                        str(src / name),
                        str(dst / name),
                    ],
                    optional=True,
                )
            )
        steps.append(
            SetupStep(
                f"bounded copy-tree skills for {profile} (rejects symlinks; never prints contents)",
                copy_tree_argv(
                    src / "skills",
                    dst / "skills",
                    owner=user,
                    group=user,
                    hermes_argv=argv_self,
                ),
                optional=True,
            )
        )
    return steps


def migrate_profile_files_commands(
    mapping: Mapping[str, str],
    *,
    source_root: Optional[Path] = None,
    hermes_argv: Optional[Sequence[str]] = None,
) -> list[str]:
    """Summarize planned copy roots/counts; never dump file contents or per-file argv."""
    from hermes_cli.kanban_os_users_rollout import summarize_tree

    root = _source_hermes_root(source_root)
    steps = plan_migrate_steps(
        mapping, source_root=source_root, hermes_argv=hermes_argv
    )
    lines = [
        "# Credential/config migration (contents are never printed)",
        "# MANUAL GATE: setup --apply does not copy these unless --migrate-profile-files.",
        "# Review each source path. Skip files that should stay gateway-only.",
        "# Required in mapped HERMES_HOME before check can report ready:",
        "#   config.yaml, .env, SOUL.md, skills/",
        "# Dry-run summarizes counts/roots; it does not emit per-file install(1) lines.",
    ]
    for profile, user in mapping.items():
        src = root / "profiles" / profile
        dst = Path(f"/home/{user}") / ".hermes" / "profiles" / profile
        skills = summarize_tree(src / "skills")
        lines.append(
            f"# {profile}: {src} -> {dst}; skills files={skills['files']} "
            f"dirs={skills['dirs']} rejected_symlinks={skills['symlinks']}"
        )
    for step in steps:
        lines.append(_quote_cmd(step.argv))
    return lines


@dataclass
class AuditItem:
    name: str
    ok: bool
    detail: str
    isolation: bool = False


def _mode_ok(path: Path, *, max_other: int = 0) -> tuple[bool, str]:
    try:
        st = path.stat()
    except OSError as exc:
        return False, f"missing or unreadable: {exc}"
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o007 > max_other:
        return False, f"{path} mode {mode:04o} allows world access"
    return True, f"{path} mode {mode:04o} uid={st.st_uid}"


def prove_sqlite_wal_as_user(
    username: str,
    db_path: str,
    *,
    hooks: Optional[LaunchHooks] = None,
    hermes_argv: Optional[Sequence[str]] = None,
) -> None:
    """Fail-closed: run the WAL lifecycle probe as *username* via sudo -n."""
    h = _hooks_or_defaults(hooks)
    if hermes_argv is None:
        try:
            from hermes_cli.kanban_db import _resolve_hermes_argv

            hermes_argv = _resolve_hermes_argv()
        except Exception:
            hermes_argv = None
    inner = [str(p) for p in (hermes_argv or ["hermes"])] + [
        "kanban",
        "os-users",
        "probe",
        "--kind",
        "wal",
        "--path",
        db_path,
    ]
    argv = build_sudo_argv(username, inner, sudo_bin=h.sudo_bin)
    assert h.run is not None
    try:
        proc = h.run(argv, timeout=30)
    except Exception as exc:
        raise MappedLaunchError(f"WAL probe as {username} failed: {exc}") from exc
    rc = int(getattr(proc, "returncode", 1))
    stderr = str(getattr(proc, "stderr", None) or "").strip()
    stdout = str(getattr(proc, "stdout", None) or "").strip()
    if rc != 0:
        detail = stderr or stdout or f"exit {rc}"
        raise MappedLaunchError(f"WAL probe as {username} failed: {detail}")


def audit_mapping(
    *,
    mapping: Optional[Mapping[str, str]] = None,
    homes: Optional[Mapping[str, str]] = None,
    hooks: Optional[LaunchHooks] = None,
    board_paths: Optional[Mapping[str, Path]] = None,
    wal_prover: Optional[Callable[[str, str], None]] = None,
    host_gates: bool = False,
    hermes_argv: Optional[Sequence[str]] = None,
    flutter_sdk: Optional[str] = None,
    android_sdk: Optional[str] = None,
    jdk_home: Optional[str] = None,
    dev_workspace: Optional[str] = None,
) -> list[AuditItem]:
    items: list[AuditItem] = []
    if mapping is None:
        try:
            mapping, loaded_homes = load_profile_os_user_config()
        except Exception:
            mapping, loaded_homes = {}, {}
        if homes is None:
            homes = loaded_homes
    targets = dict(mapping or {})
    if not targets:
        items.append(
            AuditItem(
                "mapping",
                True,
                "empty mapping — trusted-local-user (not isolation)",
            )
        )
        return items
    h = _hooks_or_defaults(hooks)
    pw_by_user: dict[str, PasswdEntry] = {}
    homes_by_profile: dict[str, str] = {}
    for profile, user in targets.items():
        try:
            pw = preflight_mapped_user(
                user,
                hooks=h,
                require_paths=False,
            )
            pw_by_user[user] = pw
            items.append(
                AuditItem(
                    f"user:{user}",
                    True,
                    f"uid={pw.pw_uid} home={pw.pw_dir} sudo -n ok",
                    isolation=True,
                )
            )
        except (MappedLaunchError, ValueError) as exc:
            items.append(AuditItem(f"user:{user}", False, str(exc)))
            continue
        hermes_home = resolve_mapped_hermes_home(profile, pw.pw_dir, homes=homes or {})
        homes_by_profile[profile] = hermes_home
        ok, detail = _mode_ok(Path(hermes_home), max_other=0)
        items.append(AuditItem(f"home:{profile}", ok, detail, isolation=ok))
        ok_x, detail_x = probe_user_access(
            user, hermes_home, mode="x", hooks=h, expect_ok=True
        )
        items.append(
            AuditItem(f"home-access:{profile}", ok_x, detail_x, isolation=ok_x)
        )
        ready, ready_detail = mapped_home_ready(hermes_home)
        items.append(
            AuditItem(f"home-ready:{profile}", ready, ready_detail, isolation=ready)
        )
        env_path = Path(hermes_home) / ".env"
        if env_path.exists():
            ok_e, detail_e = _mode_ok(env_path, max_other=0)
            items.append(
                AuditItem(f"secrets:{profile}", ok_e, detail_e, isolation=ok_e)
            )
    homes_resolved = [(profile, path) for profile, path in homes_by_profile.items()]
    for i, (p_a, h_a) in enumerate(homes_resolved):
        for p_b, h_b in homes_resolved[i + 1 :]:
            same = os.path.realpath(h_a) == os.path.realpath(h_b)
            items.append(
                AuditItem(
                    f"cross:{p_a}!={p_b}",
                    not same,
                    "distinct runtime homes" if not same else f"COLLISION {h_a}",
                    isolation=not same,
                )
            )
            if same:
                continue
            user_a = targets.get(p_a)
            user_b = targets.get(p_b)
            if user_a and user_b:
                ok_ab, d_ab = probe_user_access(
                    user_a, h_b, mode="r", hooks=h, expect_ok=False
                )
                items.append(
                    AuditItem(
                        f"cross-deny:{user_a}->{p_b}",
                        ok_ab,
                        d_ab,
                        isolation=ok_ab,
                    )
                )
                ok_ba, d_ba = probe_user_access(
                    user_b, h_a, mode="r", hooks=h, expect_ok=False
                )
                items.append(
                    AuditItem(
                        f"cross-deny:{user_b}->{p_a}",
                        ok_ba,
                        d_ba,
                        isolation=ok_ba,
                    )
                )
    try:
        from hermes_cli.kanban_os_users_rollout import shared_board_acl_layout

        paths = board_paths if board_paths is not None else shared_board_paths()
        layout = shared_board_acl_layout(paths)
        db = Path(layout["target_db"])
        parent = Path(layout["writable_dir"])
        hermes_root = layout.get("hermes_root")
        if hermes_root is not None:
            try:
                same_root = parent.resolve() == Path(hermes_root).resolve()
            except OSError:
                same_root = str(parent) == str(hermes_root)
            items.append(
                AuditItem(
                    "board-not-hermes-root",
                    not same_root,
                    (
                        f"dedicated shared dir {parent}"
                        if not same_root
                        else f"WRITE PARENT IS HERMES ROOT {hermes_root}"
                    ),
                    isolation=not same_root,
                )
            )
        parent_ok = parent.is_dir()
        parent_bits = [f"board parent {parent}"]
        if not parent_ok:
            parent_bits.append("missing directory")
        for user in pw_by_user:
            ok_x, d_x = probe_user_access(
                user, str(parent), mode="x", hooks=h, expect_ok=True
            )
            ok_w, d_w = probe_user_access(
                user, str(parent), mode="w", hooks=h, expect_ok=True
            )
            parent_bits.append(d_x)
            parent_bits.append(d_w)
            if not ok_x or not ok_w:
                parent_ok = False
        items.append(AuditItem("board-parent", parent_ok, "; ".join(parent_bits)))
        if db.exists():
            items.append(AuditItem("board-db", True, f"{db} present"))
            prover = wal_prover or (
                lambda u, p: prove_sqlite_wal_as_user(u, p, hooks=h)
            )
            for user in pw_by_user:
                try:
                    prover(user, str(db))
                    items.append(
                        AuditItem(
                            f"board-wal:{user}",
                            True,
                            f"WAL lifecycle as {user} ok",
                            isolation=True,
                        )
                    )
                except Exception as exc:
                    items.append(AuditItem(f"board-wal:{user}", False, str(exc)))
        else:
            items.append(
                AuditItem("board-db", False, f"{db} missing — run hermes kanban init")
            )
    except Exception as exc:
        items.append(AuditItem("board", False, str(exc)))
    if host_gates:
        items.extend(
            _host_continuity_gates(
                targets=targets,
                pw_by_user=pw_by_user,
                homes_by_profile=homes_by_profile,
                hooks=h,
                hermes_argv=hermes_argv,
                flutter_sdk=flutter_sdk,
                android_sdk=android_sdk,
                jdk_home=jdk_home,
                dev_workspace=dev_workspace,
            )
        )
    return items


def _host_continuity_gates(
    *,
    targets: Mapping[str, str],
    pw_by_user: Mapping[str, PasswdEntry],
    homes_by_profile: Mapping[str, str],
    hooks: LaunchHooks,
    hermes_argv: Optional[Sequence[str]] = None,
    flutter_sdk: Optional[str] = None,
    android_sdk: Optional[str] = None,
    jdk_home: Optional[str] = None,
    dev_workspace: Optional[str] = None,
) -> list[AuditItem]:
    from hermes_cli.kanban_os_users_rollout import (
        DEFAULT_ANDROID_SDK,
        DEFAULT_DEV_WORKSPACE,
        DEFAULT_FLUTTER_SDK,
        DEFAULT_GH_BIN,
        DEFAULT_GIT_BIN,
        DEFAULT_JDK_HOME,
        GITHUB_LS_REMOTE_URL,
        feature_source_sha,
        hermes_argv_covers_feature,
        self_hermes_argv,
    )

    items: list[AuditItem] = []
    argv = list(hermes_argv) if hermes_argv else self_hermes_argv()
    covers = hermes_argv_covers_feature(argv)
    sha = feature_source_sha() or "(unknown)"
    items.append(
        AuditItem(
            "runtime-sha",
            covers,
            f"reviewed sha {sha}; argv={argv!r}. Isolation is false until the live dispatcher uses this commit.",
            isolation=covers,
        )
    )

    def _cmd_ok(username: str, inner: list[str]) -> tuple[bool, str]:
        a = build_sudo_argv(username, inner, sudo_bin=hooks.sudo_bin)
        assert hooks.run is not None
        try:
            proc = hooks.run(a, timeout=30)
        except Exception as exc:
            return False, f"{inner[0]} probe failed: {exc}"
        rc = int(getattr(proc, "returncode", 1))
        if rc == 0:
            return True, f"{username} {inner[0]} ok (stdout omitted)"
        return False, f"{username} {inner[0]} failed (exit {rc}; stdout omitted)"

    dev_user = targets.get("dev")
    if dev_user and dev_user in pw_by_user:
        ok_gh, d_gh = _cmd_ok(dev_user, [DEFAULT_GH_BIN, "api", "user"])
        items.append(AuditItem("github-api", ok_gh, d_gh, isolation=False))
        ok_git, d_git = _cmd_ok(
            dev_user, [DEFAULT_GIT_BIN, "ls-remote", GITHUB_LS_REMOTE_URL, "HEAD"]
        )
        items.append(AuditItem("github-ls-remote", ok_git, d_git, isolation=False))
        ok_ssh, d_ssh = probe_user_access(
            dev_user, "/home/matt/.ssh", mode="r", hooks=hooks, expect_ok=False
        )
        items.append(AuditItem("deny-matt-ssh", ok_ssh, d_ssh, isolation=ok_ssh))
        for key_path in ("/home/matt/.ssh/id_ed25519", "/home/matt/.ssh/id_rsa"):
            if Path(key_path).exists():
                ok_k, d_k = probe_user_access(
                    dev_user, key_path, mode="r", hooks=hooks, expect_ok=False
                )
                items.append(
                    AuditItem(
                        f"deny-key:{Path(key_path).name}", ok_k, d_k, isolation=ok_k
                    )
                )
        sys_user = targets.get("sysadmin")
        if sys_user and "dev" in homes_by_profile:
            envp = str(Path(homes_by_profile["dev"]) / ".env")
            ok_sc, d_sc = probe_user_access(
                sys_user, envp, mode="r", hooks=hooks, expect_ok=False
            )
            items.append(
                AuditItem("deny-sysadmin-dev-env", ok_sc, d_sc, isolation=ok_sc)
            )
        ws = Path(dev_workspace or DEFAULT_DEV_WORKSPACE)
        if ws.exists():
            try:
                world = bool(stat.S_IMODE(ws.stat().st_mode) & 0o004)
            except OSError:
                world = False
            items.append(
                AuditItem(
                    "workspace-modes",
                    True,
                    f"{ws} world-readable={world}; not an isolation claim",
                    isolation=False,
                )
            )
            ok_ws, d_ws = probe_user_access(
                dev_user, str(ws), mode="x", hooks=hooks, expect_ok=True
            )
            items.append(
                AuditItem("dev-workspace-traverse", ok_ws, d_ws, isolation=False)
            )
        for label, raw in (
            (
                "flutter",
                flutter_sdk
                or (
                    DEFAULT_FLUTTER_SDK if Path(DEFAULT_FLUTTER_SDK).exists() else None
                ),
            ),
            (
                "android-sdk",
                android_sdk
                or (
                    DEFAULT_ANDROID_SDK if Path(DEFAULT_ANDROID_SDK).exists() else None
                ),
            ),
            (
                "jdk",
                jdk_home
                or (DEFAULT_JDK_HOME if Path(DEFAULT_JDK_HOME).exists() else None),
            ),
        ):
            if not raw:
                continue
            ok_x, d_x = probe_user_access(
                dev_user, str(raw), mode="x", hooks=hooks, expect_ok=True
            )
            items.append(AuditItem(f"toolchain-{label}", ok_x, d_x, isolation=False))
        cache = f"/home/{dev_user}/.cache"
        ok_w, d_w = probe_user_access(
            dev_user, cache, mode="w", hooks=hooks, expect_ok=True
        )
        items.append(AuditItem("dev-private-cache", ok_w, d_w, isolation=False))
    return items


def format_audit(items: Sequence[AuditItem]) -> str:
    lines = ["Kanban profile_os_users audit", ""]
    failed = 0
    for item in items:
        mark = "PASS" if item.ok else "FAIL"
        iso = " isolation" if item.isolation and item.ok else ""
        if not item.ok:
            failed += 1
        lines.append(f"  [{mark}] {item.name}{iso}: {item.detail}")
    lines.append("")
    if failed:
        lines.append(
            f"{failed} check(s) failed. Do not enable kanban.profile_os_users yet."
        )
    else:
        lines.append(
            "All checks passed. Enable mappings in the *gateway* config.yaml only after review."
        )
        lines.append(
            "Do not restart the gateway from this worker; Matt enables mappings manually."
        )
    return "\n".join(lines)


def existing_user_compatible(
    user: str,
    *,
    group: str = DEFAULT_GROUP,
    hooks: Optional[LaunchHooks] = None,
) -> None:
    """Fail closed if a pre-existing account does not match the planned design."""
    h = _hooks_or_defaults(hooks)
    assert h.getpwnam is not None
    try:
        pw = h.getpwnam(user)
    except (KeyError, MappedLaunchError) as exc:
        raise IncompatiblePrincipalError(
            f"user {user!r} missing after useradd already-exists rc"
        ) from exc
    expected_home = f"/home/{user}"
    if str(pw.pw_dir).rstrip("/") != expected_home:
        raise IncompatiblePrincipalError(
            f"{user} home is {pw.pw_dir!r}, expected {expected_home!r}"
        )
    assert h.getgrnam is not None
    try:
        private = h.getgrnam(user)
    except (KeyError, MappedLaunchError) as exc:
        raise IncompatiblePrincipalError(
            f"{user} has no private primary group {user!r}"
        ) from exc
    if int(pw.pw_gid) != int(private.gr_gid):
        raise IncompatiblePrincipalError(
            f"{user} primary gid {pw.pw_gid} is not private group {user} "
            f"(gid {private.gr_gid}); refusing accounts whose primary group "
            f"is {group} or any other shared group"
        )
    try:
        shared = h.getgrnam(group)
    except (KeyError, MappedLaunchError) as exc:
        raise IncompatiblePrincipalError(f"shared group {group!r} missing") from exc
    if user not in set(shared.gr_mem):
        raise IncompatiblePrincipalError(
            f"{user} is not a supplemental member of {group}"
        )


def execute_setup_plan(
    steps: Sequence[SetupStep],
    *,
    run: Optional[Callable[..., Any]] = None,
    hooks: Optional[LaunchHooks] = None,
    state_path: str | Path = DEFAULT_STATE_PATH,
    group: str = DEFAULT_GROUP,
    sudoers_text: str = "",
    sudoers_tmp: str = "/tmp/hermes-kanban-os-users.sudoers",
) -> int:
    """Run planned argv lists. Records created vs pre-existing principals."""
    h = _hooks_or_defaults(hooks)
    runner = run or h.run
    assert runner is not None
    state = load_os_users_state(state_path)
    created_users = set(state.get("created_users") or [])
    created_groups = set(state.get("created_groups") or [])
    preexisting_users = set(state.get("preexisting_users") or [])
    preexisting_groups = set(state.get("preexisting_groups") or [])
    acl_paths = set(state.get("acl_paths") or [])

    for step in steps:
        argv = list(step.argv)
        if step.optional:
            src = ""
            if "copy-tree" in argv and "--src" in argv:
                src = str(argv[argv.index("--src") + 1])
            elif "migrate-db" in argv and "--from" in argv:
                src = str(argv[argv.index("--from") + 1])
            elif (
                argv and os.path.basename(str(argv[0])) == "install" and len(argv) >= 2
            ):
                src = str(argv[-2])
            if src.startswith("/") and not Path(src).exists():
                print(f"# skip optional missing source {src}")
                continue
        if "<generated-sudoers>" in argv:
            tmp = Path(sudoers_tmp)
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(sudoers_text, encoding="utf-8")
            os.chmod(tmp, 0o600)
            check = runner(["visudo", "-c", "-f", str(tmp)])
            if int(getattr(check, "returncode", 1)) != 0:
                err = str(getattr(check, "stderr", None) or "")
                print(f"visudo -c failed: {err}", file=sys.stderr)
                return 1
            argv = [
                "install",
                "-m",
                "0440",
                "-o",
                "root",
                "-g",
                "root",
                str(tmp),
                DEFAULT_SUDOERS_PATH,
            ]
        print(f"# {step.title}")
        proc = runner(argv)
        rc = int(getattr(proc, "returncode", 1))
        if rc != 0:
            try:
                ok = _idempotent_ok(argv, rc, hooks=h, group=group)
            except IncompatiblePrincipalError as exc:
                print(f"incompatible principal: {exc}", file=sys.stderr)
                return 1
            if not ok:
                print(
                    f"command failed ({rc}): {_quote_cmd(argv)}",
                    file=sys.stderr,
                )
                return 1
            if step.creates.startswith("user:"):
                preexisting_users.add(step.creates.split(":", 1)[1])
            elif step.creates.startswith("group:"):
                preexisting_groups.add(step.creates.split(":", 1)[1])
        else:
            if step.creates.startswith("user:"):
                created_users.add(step.creates.split(":", 1)[1])
            elif step.creates.startswith("group:"):
                created_groups.add(step.creates.split(":", 1)[1])
        if step.creates.startswith("acl:"):
            acl_paths.add(step.creates.split(":", 1)[1])

    save_os_users_state(
        {
            "created_users": sorted(created_users),
            "created_groups": sorted(created_groups),
            "preexisting_users": sorted(preexisting_users),
            "preexisting_groups": sorted(preexisting_groups),
            "acl_paths": sorted(acl_paths),
        },
        state_path,
    )
    return 0


def run_os_users_cli(
    args: Any,
    *,
    hooks: Optional[LaunchHooks] = None,
    state_path: Optional[str | Path] = None,
    board_paths: Optional[Mapping[str, Path]] = None,
) -> int:
    action = getattr(args, "os_users_action", None) or "check"
    mapping = None
    homes = None
    try:
        mapping, homes = load_profile_os_user_config()
    except ValueError as exc:
        print(f"kanban os-users: invalid config: {exc}", file=sys.stderr)
        return 2
    resolved_state = (
        Path(state_path) if state_path is not None else Path(DEFAULT_STATE_PATH)
    )

    if action == "migrate-db":
        from hermes_cli.kanban_os_users_rollout import sqlite_backup_copy

        src = getattr(args, "migrate_from", None)
        dst = getattr(args, "migrate_to", None)
        if not src or not dst:
            print(
                "kanban os-users migrate-db: --from and --to are required",
                file=sys.stderr,
            )
            return 2
        try:
            sqlite_backup_copy(str(src), str(dst))
        except Exception as exc:
            print(f"kanban os-users migrate-db failed: {exc}", file=sys.stderr)
            return 1
        print(f"sqlite backup ok {src} -> {dst} (WAL/SHM not raw-copied)")
        return 0

    if action == "copy-tree":
        from hermes_cli.kanban_os_users_rollout import copy_tree_reject_symlinks

        src = getattr(args, "copy_src", None)
        dst = getattr(args, "copy_dst", None)
        owner = getattr(args, "copy_owner", None) or "root"
        group = getattr(args, "copy_group", None) or owner
        if not src or not dst:
            print(
                "kanban os-users copy-tree: --src and --dst are required",
                file=sys.stderr,
            )
            return 2
        try:
            stats = copy_tree_reject_symlinks(
                str(src), str(dst), owner=owner, group=group
            )
        except Exception as exc:
            print(f"kanban os-users copy-tree failed: {exc}", file=sys.stderr)
            return 1
        print(
            f"copy-tree ok files={stats['files']} dirs={stats['dirs']} "
            f"rejected_symlinks={stats['rejected_symlinks']}"
        )
        return 0

    if action == "probe":
        kind = getattr(args, "probe_kind", None) or "wal"
        path = getattr(args, "probe_path", None)
        if not path:
            print("kanban os-users probe: --path is required", file=sys.stderr)
            return 2
        if kind != "wal":
            print(f"kanban os-users probe: unknown kind {kind!r}", file=sys.stderr)
            return 2
        try:
            prove_sqlite_wal_lifecycle(str(path))
        except Exception as exc:
            print(f"kanban os-users probe failed: {exc}", file=sys.stderr)
            return 1
        print(f"WAL lifecycle ok for {path}")
        return 0

    if action == "check":
        from hermes_cli.kanban_os_users_rollout import self_hermes_argv

        items = audit_mapping(
            mapping=mapping or None,
            homes=homes,
            hooks=hooks,
            board_paths=board_paths,
            host_gates=True,
            hermes_argv=self_hermes_argv(),
            flutter_sdk=getattr(args, "flutter_sdk", None),
            android_sdk=getattr(args, "android_sdk", None),
            jdk_home=getattr(args, "jdk_home", None),
            dev_workspace=getattr(args, "dev_workspace", None),
        )
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "items": [
                            {
                                "name": i.name,
                                "ok": i.ok,
                                "detail": i.detail,
                                "isolation": i.isolation,
                            }
                            for i in items
                        ],
                        "ok": all(i.ok for i in items),
                    },
                    indent=2,
                )
            )
        else:
            print(format_audit(items))
        return 0 if all(i.ok for i in items) else 1

    if action == "sudoers":
        print(render_sudoers(mapping=mapping or None))
        return 0

    if action == "rollback":
        steps = plan_rollback_steps(
            mapping=mapping or None,
            state_path=resolved_state,
        )
        print(
            format_plan(steps, heading="Kanban profile_os_users rollback (destructive)")
        )
        st = load_os_users_state(resolved_state)
        if not (
            st.get("created_users") or st.get("created_groups") or st.get("acl_paths")
        ):
            print(
                "No setup state file — rollback will not userdel/groupdel "
                "pre-existing principals. Remove ACLs on recorded paths only."
            )
        return 0

    if action == "setup":
        from hermes_cli.kanban_os_users_rollout import (
            apply_command_hint,
            default_toolchain_if_present,
            extra_sudoers_bins,
            format_rollout_and_rollback,
            github_manual_gate_lines,
            self_hermes_argv,
        )

        gw = getattr(args, "gateway_user", None)
        dev_ws = getattr(args, "dev_workspace", None)
        if not dev_ws and Path(DEFAULT_DEV_WORKSPACE).exists():
            dev_ws = DEFAULT_DEV_WORKSPACE
        migrate = bool(getattr(args, "migrate_profile_files", False))
        include_db = bool(getattr(args, "migrate_shared_db", False))
        toolchain = default_toolchain_if_present()
        flutter = getattr(args, "flutter_sdk", None) or toolchain.get("flutter_sdk")
        android = getattr(args, "android_sdk", None) or toolchain.get("android_sdk")
        jdk = getattr(args, "jdk_home", None) or toolchain.get("jdk_home")
        argv_self = self_hermes_argv()
        extra_bins = extra_sudoers_bins(
            flutter_sdk=flutter, android_sdk=android, jdk_home=jdk
        )
        targets = mapping or default_mapping_example()
        steps = plan_setup_steps(
            mapping=mapping or None,
            gateway_user=gw,
            dev_workspace=dev_ws,
            board_paths=board_paths,
            hermes_argv=argv_self,
            flutter_sdk=flutter,
            android_sdk=android,
            jdk_home=jdk,
            include_db_migration=include_db,
        )
        print(format_plan(steps, heading="Kanban profile_os_users setup (dry-run)"))
        print("")
        print(format_rollout_and_rollback(hermes_argv=argv_self))
        print("")
        print("\n".join(migrate_profile_files_commands(targets, hermes_argv=argv_self)))
        print("")
        print("\n".join(github_manual_gate_lines(targets.get("dev", "hermes-dev"))))
        print("")
        print("Sudoers snippet:")
        sudoers_text = render_sudoers(
            mapping=mapping or None,
            gateway_user=gw,
            hermes_argv=argv_self,
            extra_bins=extra_bins,
        )
        print(sudoers_text)
        hint = apply_command_hint(argv_self)
        if migrate:
            steps = list(steps) + plan_migrate_steps(targets, hermes_argv=argv_self)
            print(
                "Will copy profile files (--migrate-profile-files). "
                "Contents are never printed."
            )
        else:
            print(
                "MANUAL GATE: profile files were NOT copied. Re-run with "
                "--migrate-profile-files after review, or copy with copy-tree."
            )
            print(
                "Check cannot report ready without config.yaml, .env, SOUL.md, skills/."
            )
        apply = bool(getattr(args, "apply", False))
        if not apply:
            print("Dry-run only. Re-run as root with --apply after review.")
            print(f"  {hint}")
            print("Do not use `sudo hermes`; that may invoke the old installed CLI.")
            return 0
        if os.geteuid() != 0:
            print(
                "kanban os-users: --apply requires euid 0. Not prompting for a password.",
                file=sys.stderr,
            )
            print(f"  {hint}", file=sys.stderr)
            return 1
        rc = execute_setup_plan(
            steps,
            hooks=hooks,
            state_path=resolved_state,
            sudoers_text=sudoers_text,
        )
        if rc == 0:
            print("Apply complete. Run: hermes kanban os-users check")
            print("Do not enable profile_os_users until check passes.")
            print(
                "Check will FAIL until mapped HERMES_HOME has "
                "config.yaml, .env, SOUL.md, skills/."
            )
        return rc

    print(f"kanban os-users: unknown action {action!r}", file=sys.stderr)
    return 2


def _idempotent_ok(
    argv: Sequence[str],
    rc: int,
    *,
    hooks: Optional[LaunchHooks] = None,
    group: str = DEFAULT_GROUP,
) -> bool:
    """Treat useradd/groupadd already-exists as success only if compatible."""
    if rc == 0:
        return True
    cmd = os.path.basename(str(argv[0])) if argv else ""
    if cmd == "useradd" and rc in {9, 4}:
        user = str(argv[-1])
        existing_user_compatible(user, group=group, hooks=hooks)
        return True
    if cmd == "groupadd" and rc in {9, 4}:
        name = str(argv[-1])
        h = _hooks_or_defaults(hooks)
        assert h.getgrnam is not None
        try:
            h.getgrnam(name)
        except (KeyError, MappedLaunchError) as exc:
            raise IncompatiblePrincipalError(
                f"group {name!r} missing after already-exists rc"
            ) from exc
        return True
    return False
