"""Shared constants for Hermes Agent.

Import-safe, stdlib-only — importable from anywhere without circular-import risk.
"""

import contextlib
import os
import re
import shutil
import stat
import sys
from contextvars import ContextVar, Token
from pathlib import Path

_profile_fallback_warned: bool = False
_UNSET = object()
_HERMES_HOME_OVERRIDE: ContextVar[str | object] = ContextVar("_HERMES_HOME_OVERRIDE", default=_UNSET)

# TUI busy-indicator styles (CLI /indicator, TUI gateway config, /help registry).
# Keep in sync with INDICATOR_STYLES / DEFAULT_INDICATOR_STYLE in ui-tui/src/app/interfaces.ts.
INDICATOR_STYLES: tuple[str, ...] = ("ascii", "emoji", "kaomoji", "unicode")
DEFAULT_INDICATOR_STYLE: str = "kaomoji"


def set_hermes_home_override(path: str | Path | None) -> Token:
    """Set a context-local Hermes home override and return its reset token.

    Deliberately does not mutate ``os.environ`` (shared by every thread in the process).
    """
    value: str | object = _UNSET if path is None else str(path)
    return _HERMES_HOME_OVERRIDE.set(value)


def reset_hermes_home_override(token: Token) -> None:
    """Restore the previous context-local Hermes home override."""
    _HERMES_HOME_OVERRIDE.reset(token)


def get_hermes_home_override() -> str | None:
    """Return the active context-local Hermes home override, if any."""
    override = _HERMES_HOME_OVERRIDE.get()
    return str(override) if override is not _UNSET and override else None


def _get_platform_default_hermes_home() -> Path:
    """Return the platform default with the literal data-directory suffix."""
    suffix = os.environ.get("HERMES_DATA_DIR_SUFFIX", "")
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / ("hermes" + suffix)
    return Path.home() / (".hermes" + suffix)


def sudo_invoker_default_home() -> Path | None:
    """The invoking user's native ``~/.hermes`` when this process is root under ``sudo``, else None.

    sudo strips HERMES_HOME and sets HOME=/root, so the process's own default is root's; the profile
    store and the system service being operated on belong to SUDO_USER.
    """
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return None
    sudo_user = os.environ.get("SUDO_USER", "").strip()
    if not sudo_user or sudo_user == "root":
        return None
    import pwd

    try:
        return Path(pwd.getpwnam(sudo_user).pw_dir) / ".hermes"
    except KeyError:  # SUDO_USER not in passwd (chroot/container)
        return None


def _warn_profile_fallback_once() -> None:
    """Warn once when HERMES_HOME is unset but a non-default profile is sticky-active (wrong fallback)."""
    global _profile_fallback_warned
    if _profile_fallback_warned:
        return
    try:
        fallback_home = _get_platform_default_hermes_home()
        active_path = fallback_home / "active_profile"
        active = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else ""
    except (UnicodeDecodeError, OSError):
        active = ""
    if active and active != "default":
        _profile_fallback_warned = True
        # Direct stderr, not logging: runs at import time (often before logging is
        # configured) and root-logger propagation would double-emit.
        msg = (
            f"[HERMES_HOME fallback] HERMES_HOME is unset but active "
            f"profile is {active!r}. Falling back to {fallback_home}, which "
            f"is the DEFAULT profile — not {active!r}. Any data this "
            f"process writes will land in the wrong profile. The "
            f"subprocess spawner should pass HERMES_HOME explicitly "
            f"(see issue #18594)."
        )
        with contextlib.suppress(Exception):
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()


def get_hermes_home() -> Path:
    """Hermes home: context-local override → ``HERMES_HOME`` env var → platform default."""
    override = get_hermes_home_override()
    if override:
        return Path(override)
    if not os.environ.get("HERMES_HOME", "").strip():
        _warn_profile_fallback_once()
    return get_process_hermes_home()


# Resolved keys, keyed by the path string that was handed in. Path.resolve()
# is a filesystem call, and this function sits under every ToolRegistry
# lookup through current_scope_key(), so without this the registry pays a
# syscall per lookup. A process only ever sees a handful of home paths, so
# the dict stays tiny. Only paths that really exist are stored, see below.
_HOME_KEY_CACHE: dict[str, str] = {}


def hermes_home_key(path: str | Path | None = None) -> str:
    """Stable registry key for a Hermes home/profile dir.

    ``strict=False`` so profiles whose directories don't exist yet still get a key.

    The resolved value is remembered per input path. A directory that does
    not exist yet is resolved without touching the cache, because the answer
    can change once it is created (e.g. part of the path turns out to be a symlink).
    """
    candidate = Path(path) if path is not None else get_hermes_home()
    raw = str(candidate)
    cached = _HOME_KEY_CACHE.get(raw)
    if cached is not None:
        return cached
    expanded = candidate.expanduser()
    try:
        resolved = expanded.resolve(strict=True)
    except OSError:
        # Not on disk yet: lenient resolve, not stored, so the real answer is
        # picked up once the directory appears.
        return os.path.normcase(str(expanded.resolve(strict=False)))
    key = os.path.normcase(str(resolved))
    _HOME_KEY_CACHE[raw] = key
    return key


def reset_hermes_home_key_cache() -> None:
    """Forget every remembered home key (for tests that move a home dir on disk)."""
    _HOME_KEY_CACHE.clear()


def get_process_hermes_home() -> Path:
    """Hermes home of the running process, ignoring task overrides.

    For process-level assets (theme YAML, dashboard plugin manifests) that must stay visible while a
    request is scoped to another profile (e.g. embedded ``/chat`` under ``--open-profile``).
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else _get_platform_default_hermes_home()


# get_default_hermes_root() memo keyed on (native home, HERMES_HOME) so it stays
# fresh when a test or plugin mutates HERMES_HOME; saves ~80us/call at 31+ sites.
_default_hermes_root_memo: "tuple[str, str, Path] | None" = None


def get_default_hermes_root(*, home: str | Path | None = None) -> Path:
    """Root of an explicit home, or the process home when none is supplied."""
    global _default_hermes_root_memo
    native_home = _get_platform_default_hermes_home()
    env_home = str(home) if home is not None else os.environ.get("HERMES_HOME", "")
    memo = _default_hermes_root_memo
    if memo is not None and memo[:2] == (str(native_home), env_home):
        return memo[2]
    result = native_home
    if env_home:
        env_path = Path(env_home)
        try:
            env_path.resolve().relative_to(native_home.resolve())  # under ~/.hermes (normal or profile mode)
        except ValueError:  # Docker/custom root: <root>/profiles/<name> -> <root>, else HERMES_HOME itself
            result = env_path.parent.parent if env_path.parent.name == "profiles" else env_path
    _default_hermes_root_memo = (str(native_home), env_home, result)
    return result


# Tombstone lives beside the profile dir (not inside) so a stale mkdir or rmtree cannot erase it.
_DELETED_PROFILES_DIR = ".deleted"
# Files marking a real Hermes home; arbitrary dirs with a ``profiles`` segment lack them.
_HERMES_HOME_MARKERS = ("config.yaml", ".env", "state.db")


def _is_hermes_profiles_root(profiles_dir: Path) -> bool:
    """True when *profiles_dir* is provably ``<hermes-home>/profiles``.

    Accepts the classic ``~/.hermes`` layout, a root carrying Hermes-home marker files, a
    ``profiles/.deleted`` tombstone dir (only ``profile delete`` creates it), or the default root.
    """
    root = profiles_dir.parent
    if root.name == ".hermes":
        return True
    try:
        if (profiles_dir / _DELETED_PROFILES_DIR).is_dir() or any(
            (root / marker).exists() for marker in _HERMES_HOME_MARKERS
        ):
            return True
    except OSError:
        pass
    try:
        return root.resolve(strict=False) == get_default_hermes_root().resolve(strict=False)
    except OSError:
        return False


def named_profile_home(path: str | Path) -> Path | None:
    """Return ``<root>/profiles/<name>`` when *path* is that home or under it.

    Requires ``<name>`` not to start with ``.`` and the ``profiles`` parent to be a real Hermes home;
    a default home whose path merely contains a ``profiles`` segment is not a named profile.
    """
    current = Path(path)
    for candidate in (current, *current.parents):
        if (candidate.parent.name == "profiles" and not candidate.name.startswith(".")
                and _is_hermes_profiles_root(candidate.parent)):
            return candidate
        if candidate.name == ".hermes":  # default home: a coincidental profiles/ ancestor is not a root
            return None
    return None


def profile_name_for_home(path: str | Path | None) -> str | None:
    """Return the canonical profile id owning *path*, or ``None`` when it is not a profile home.

    The default home is the Hermes root itself, so its basename is an installation detail (``.hermes``
    on POSIX and commonly ``hermes`` on Windows), not the profile id ``default``.
    """
    if path is None or not str(path).strip():
        return None
    current = Path(path).expanduser()
    try:
        default_root = get_default_hermes_root()
        for candidate in (current, current.resolve(strict=False)):
            if candidate == default_root or candidate == default_root.resolve(strict=False):
                return "default"
            named = named_profile_home(candidate)
            if named is not None:
                return named.name
            # A stored profile home is authoritative: its owner already resolved it, so the
            # <root>/profiles/<name> shape names the profile even when <root> carries no markers.
            if candidate.parent.name == "profiles" and not candidate.name.startswith("."):
                return candidate.name
    except (OSError, RuntimeError, ValueError):
        return None
    return None


def profile_tombstone_path(profile_home: Path) -> Path:
    return profile_home.parent / _DELETED_PROFILES_DIR / profile_home.name


def named_profile_is_deleted(profile_home: str | Path) -> bool:
    return profile_tombstone_path(Path(profile_home)).exists()


def mark_named_profile_deleted(profile_home: str | Path) -> None:
    marker = profile_tombstone_path(Path(profile_home))
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("deleted\n", encoding="utf-8")


def clear_named_profile_deleted(profile_home: str | Path) -> None:
    profile_tombstone_path(Path(profile_home)).unlink(missing_ok=True)


def assert_named_profile_home_live(path: str | Path) -> None:
    """Refuse missing or tombstoned named profile homes."""
    home = named_profile_home(path)
    if home is not None and (named_profile_is_deleted(home) or not home.exists()):
        raise FileNotFoundError(
            f"Named profile home does not exist: {home}. "
            "Create the profile explicitly before using it."
        )


def mkdir_under_hermes_home(path: str | Path) -> Path:
    """Create *path*, but never materialize a deleted/missing named profile."""
    target = Path(path)
    assert_named_profile_home_live(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _packaged_dir(env_var: str, default: Path | None, subdir: str) -> Path:
    """Resolve a package-manager-relocatable directory.

    Order: *env_var* (Nix wrapper / explicit override) → caller ``default`` (source checkout) →
    ``<HERMES_HOME>/<subdir>``.
    """
    override = os.getenv(env_var, "").strip()
    return Path(override) if override else default if default is not None else get_hermes_home() / subdir


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers."""
    return _packaged_dir("HERMES_OPTIONAL_SKILLS", default, "optional-skills")


def get_optional_mcps_dir(default: Path | None = None) -> Path:
    """Return the optional-mcps directory, honoring package-manager wrappers (``HERMES_OPTIONAL_MCPS``)."""
    return _packaged_dir("HERMES_OPTIONAL_MCPS", default, "optional-mcps")


def get_bundled_skills_dir(default: Path | None = None) -> Path:
    """Return the bundled skills directory, honoring package-manager wrappers (``HERMES_BUNDLED_SKILLS``)."""
    return _packaged_dir("HERMES_BUNDLED_SKILLS", default, "skills")


def get_hermes_dir(new_subpath: str, old_name: str, *, home: Path | None = None) -> Path:
    """Resolve a Hermes subdirectory, honouring a populated legacy ``<old_name>/`` (no migration).

    An empty legacy dir does NOT count (install scaffolds, manual mkdir) so it cannot shadow the new path.

    A bare empty ``<old_name>/`` directory does **not** count as "the legacy install is in use" — install
    scaffolds, manual ``mkdir`` work, and cleared-then-abandoned locations all create empty stubs that would
    otherwise silently shadow real data populated at ``<new_subpath>/``. See #27602 for the pairing-store
    regression where a dormant empty ``pairing/`` orphaned approved-user data in ``platforms/pairing/``.
    """
    home = home or get_hermes_home()
    old_path = home / old_name
    return old_path if _legacy_path_has_content(old_path) else home / new_subpath


def iter_hermes_node_dirs(home: Path | None = None) -> list[Path]:
    """Historical layout for old-updater diagnostics, never runtime selection."""
    node_dir = (home or get_hermes_home()) / "node"
    return [node_dir, node_dir / "bin"] if sys.platform == "win32" else [node_dir / "bin", node_dir]


_WINDOWS_NODE_SHIMS = {
    "npm": ["npm.cmd", "npm.exe", "npm"], "npx": ["npx.cmd", "npx.exe", "npx"], "node": ["node.exe", "node"],
}


def _candidate_node_command_names(command: str) -> list[str]:
    base = Path(command).name
    if sys.platform != "win32" or "." in base:
        return [base]
    # Prefer npm.cmd: PowerShell may block npm.ps1 by policy; CreateProcess cannot launch a bare .ps1.
    return _WINDOWS_NODE_SHIMS.get(base.lower(), [f"{base}.cmd", f"{base}.exe", base])

def _run_version_probe(argv: list[str], **kwargs):
    """Run a hidden ``--version`` probe; ``None`` when it cannot run."""
    import subprocess
    try:
        from hermes_cli._subprocess_compat import windows_hide_flags
        return subprocess.run(
            argv, capture_output=True, timeout=10, creationflags=windows_hide_flags(), **kwargs
        )
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


def _version_probe_ok(path: str) -> bool:
    """True when ``<path> --version`` exits 0 under the Hermes-managed Node PATH."""
    result = _run_version_probe([path, "--version"], env=with_hermes_node_path())
    return result is not None and result.returncode == 0


# Install tree root (this file lives at <install_root>/hermes_constants.py). Used by secure_parent_dir() to
# skip chmod on the install dir — chmodding it 0700 breaks hermes-user traversal in Docker (UID 10000). See
# #25821, #93050.
_INSTALL_ROOT = Path(__file__).resolve().parent


def _is_executable_file(path: str) -> bool:
    """``exists()`` follows symlinks, so a dangling link never spawns a probe."""
    return os.path.exists(path) and os.access(path, os.X_OK)


def node_tool_runnable(path: str | None) -> bool:
    """True only when *path* is a Node/npm/npx binary that actually runs (``--version`` probe)."""
    if not path:
        return False
    present = Path(path).is_file() if sys.platform == "win32" else _is_executable_file(path)
    return present and _version_probe_ok(path)


def hermes_managed_node_tree_present(home: Path | None = None) -> bool:
    """Read-only legacy artifact detection for already-running old updaters."""
    names = [n for c in ("node", "npm", "npx") for n in _candidate_node_command_names(c)]
    return any((directory / name).is_file() for directory in iter_hermes_node_dirs(home) for name in names)


def find_node_executable(command: str) -> str | None:
    """Read PM's selected Node/npm/npx, then a user-owned PATH toolchain.

    Explicit executable paths remain caller-owned. Discovery never installs,
    probes, repairs, or activates the retired ``HERMES_HOME/node`` layout.
    """
    command = str(command)
    if any(sep in command for sep in ("/", "\\")):
        if sys.platform == "win32":
            return command if Path(command).is_file() else None
        return shutil.which(command)
    base = command.lower()
    for suffix in (".cmd", ".exe", ".ps1"):
        base = base.removesuffix(suffix)
    package_name = {"node": "node", "npm": "npm", "npx": "npm"}.get(base)
    if package_name is not None:
        from pm import installed_package

        installed = installed_package(package_name)
        if installed is not None and installed.binary is not None:
            if base != "npx":
                return str(installed.binary)
            for name in _candidate_node_command_names("npx"):
                candidate = installed.binary.parent / name
                if candidate.is_file():
                    return str(candidate)
            return None
    if sys.platform != "win32":
        return shutil.which(command)
    directories = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    for name in _candidate_node_command_names(command):
        for directory in directories:
            candidate = Path(directory) / name
            if candidate.is_file():
                return str(candidate)
    return None


def with_hermes_node_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Compose installed PM npm and its Node dependency without provisioning."""
    from pm import env_for

    return env_for("npm", base_env=env)


def agent_browser_runnable(path: str | None) -> bool:
    """True when *path* is an agent-browser CLI that runs (``--version`` exits 0).

    Dead/wrong-arch/hung binaries are rejected so callers try the next candidate.

    A bare presence check (``shutil.which`` / ``Path.exists``) is not enough: agent-browser's npm
    ``postinstall`` re-points a *global* install symlink (e.g. ``/opt/homebrew/bin/agent-browser``) at our
    local ``node_modules/agent-browser/bin/...`` binary, which then disappears on the next ``hermes update``
    — leaving a **dangling symlink** that ``which`` still reports but exec fails on with exit 127 (issue
    #48521). Callers that trust such a path silently break every browser tool.
    """
    if not path:
        return False
    return _is_executable_file(path) and _version_probe_ok(path)


def _legacy_path_has_content(path: Path) -> bool:
    """True iff *path* is a non-directory file or a populated directory.

    Non-not-found ``OSError`` means "assume occupied" (never orphan legacy data). Symlinks are
    judged on their target; a dangling one does NOT count.
    """
    try:
        st = path.lstat()
        if stat.S_ISLNK(st.st_mode):
            st = path.stat()  # judge a symlink on its target; dangling → FileNotFoundError
    except FileNotFoundError:
        return False
    except OSError:  # e.g. PermissionError on a parent: assume occupied
        return True
    if not stat.S_ISDIR(st.st_mode):
        return True
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        pass
    return True


def display_hermes_home() -> str:
    """User-facing ``~/`` display string for HERMES_HOME (``~/.hermes/profiles/coder``)."""
    home = get_hermes_home()
    try:  # as_posix(): str() on Windows yields chimeras like ~/AppData\Local\hermes/skills/
        return "~/" + home.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(home)


def profile_cli_selector() -> str:
    """``-p <name> `` (trailing space) pinning copy-pasteable ``hermes ...`` guidance to the
    active NAMED profile, else ``""``: a bare ``hermes`` follows the sticky ``active_profile``
    file, which can name a different database than the one that failed (#105887). A custom
    home outside the profile tree has no selector (only HERMES_HOME names it)."""
    name = profile_name_for_home(get_hermes_home())
    return f"-p {name} " if name and name != "default" else ""


def secure_parent_dir(path: Path) -> None:
    """Chmod ``0o700`` on *path*'s parent, refusing ``/`` and top-level dirs (misresolved HERMES_HOME)."""
    parent = path.parent.resolve()
    if parent == Path("/") or len(parent.parts) < 3:
        return
    # Refuse the install tree: chmod 0700 breaks hermes-user traversal in Docker (UID 10000).
    # A credential file here means HERMES_HOME misresolved; surface it (caused production lockouts).
    # See #25821, #93050.
    if parent == _INSTALL_ROOT or _INSTALL_ROOT in parent.parents:
        import logging

        logging.getLogger(__name__).warning(
            "Not restricting permissions on %s: it is inside the "
            "hermes-agent install directory (%s). Credential files are "
            "normally stored under the hermes home directory instead.", parent, _INSTALL_ROOT,
        )
        return
    with contextlib.suppress(OSError):
        os.chmod(parent, 0o700)


def _norm_home_path(path: str | None) -> str:
    """Return a comparable absolute path string, or ``""`` for empty input."""
    raw = (path or "").strip()
    if not raw:
        return ""
    try:
        return os.path.normcase(os.path.abspath(os.path.expanduser(raw)))
    except Exception:
        return os.path.normcase(raw)


def _profile_home_path(env: dict[str, str] | None = None) -> str | None:
    """Return ``{HERMES_HOME}/home`` when the profile-home directory exists."""
    hermes_home = get_hermes_home_override() or (env or {}).get("HERMES_HOME") or os.getenv("HERMES_HOME")
    if not hermes_home:
        return None
    profile_home = os.path.join(hermes_home, "home")
    return profile_home if os.path.isdir(profile_home) else None


def _is_profile_home(candidate: str | None, profile_home: str | None) -> bool:
    return bool(candidate and profile_home and _norm_home_path(candidate) == _norm_home_path(profile_home))


def _env_get(env: dict[str, str], key: str, default: str = "") -> str:
    """Stripped *key* from *env*, falling back to the process environment."""
    return str(env.get(key) or os.getenv(key, default)).strip()


def _iter_real_home_candidates(env: dict[str, str] | None = None) -> list[str]:
    """Return likely OS-user home candidates in trust order."""
    env = env or {}
    candidates = [_env_get(env, "HERMES_REAL_HOME"), _env_get(env, "HOME")]
    with contextlib.suppress(Exception):
        import pwd
        candidates.append(pwd.getpwuid(os.getuid()).pw_dir.strip())  # windows-footgun: ok — POSIX-only module inside try/except
    candidates.append(_env_get(env, "USERPROFILE"))
    drive, path = _env_get(env, "HOMEDRIVE"), _env_get(env, "HOMEPATH")
    if drive and path:
        candidates.append(f"{drive}{path}" if path.startswith(("\\", "/")) else os.path.join(drive, path))
    expanded = os.path.expanduser("~")
    if expanded != "~":
        candidates.append(expanded)
    return [c for c in candidates if c]


def get_real_home(env: dict[str, str] | None = None) -> str:
    """The OS user's real home, avoiding the Hermes profile HOME.

    ``HOME`` belongs to the OS account and external CLIs keeping credentials under ``~``; a parent
    already running with ``HOME={HERMES_HOME}/home`` is repaired back when possible.
    """
    profile_home = _profile_home_path(env)
    seen: set[str] = set()
    for candidate in _iter_real_home_candidates(env):
        key = _norm_home_path(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if not _is_profile_home(candidate, profile_home):
            return candidate
    return "/tmp"


_HOME_MODE_ALIASES = {"isolated": "profile", "profile_home": "profile", "profile-home": "profile",
                      "host": "real", "user": "real", "real_home": "real", "real-home": "real"}


def get_subprocess_home(env: dict[str, str] | None = None) -> str | None:
    """Subprocess ``HOME`` override, or ``None``.

    ``auto``: hosts keep real HOME (repairing a profile-home parent), containers use
    ``{HERMES_HOME}/home``; ``real``: always real HOME; ``profile``: always the profile home.
    """
    env = env or {}
    profile_home = _profile_home_path(env)
    mode = _env_get(env, "TERMINAL_HOME_MODE", "auto").lower() or "auto"
    mode = _HOME_MODE_ALIASES.get(mode, mode)

    if mode == "profile":
        return profile_home
    real_home = get_real_home(env)
    current_home = _env_get(env, "HOME")
    repaired = real_home if _norm_home_path(real_home) != _norm_home_path(current_home) else None
    if mode == "real":
        return repaired

    if profile_home and is_container():
        return profile_home
    if _is_profile_home(current_home, profile_home):
        return repaired
    return None


def apply_subprocess_home_env(env: dict[str, str]) -> None:
    """Apply Hermes' subprocess HOME contract to *env* in-place."""
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
    home = get_subprocess_home(env)
    if home:
        env["HOME"] = home


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def parse_reasoning_effort(effort) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    ``None`` for empty/unrecognized input (caller uses the default); ``{"enabled": False}`` for
    "none"/"false"/"disabled"/YAML False — ``reasoning_effort: false`` must mean disabled.
    """
    if effort is None or effort is True:
        return None
    effort = str(effort).strip().lower()  # False -> "false" -> disabled; "" matches neither set
    if effort in {"none", "false", "disabled"}:
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


def _canonical_model_variants(model: str) -> list[str]:
    """Spelling variants for tolerant override matching, exact first, deduped in order.

    Dot/dash recovery runs on EACH base form so ``x-4.5``, ``x-4-5`` and ``x.4.5`` share one variant set.
    """
    _dash_to_dot = lambda s: re.sub(r'(\d)-(\d)', r'\1.\2', s)
    _dot_to_dash = lambda s: re.sub(r'(\d)\.(\d)', r'\1-\2', s)
    seen: set[str] = set()
    variants: list[str] = []

    def _add(*values):
        for v in values:
            if v and v not in seen:
                seen.add(v)
                variants.append(v)

    def _add_with_derivatives(s):
        dashed, dotted = s.replace('.', '-'), s.replace('-', '.')
        _add(s, dashed, dotted, _dash_to_dot(s), _dot_to_dash(s), _dash_to_dot(dashed), _dot_to_dash(dotted))
    _add_with_derivatives(model)
    parts = model.split('/')
    if len(parts) >= 2:  # bare model (strip provider/aggregator prefix)
        _add_with_derivatives(parts[-1])
    if len(parts) >= 3:  # strip aggregator only: "openrouter/anthropic/x" → "anthropic/x"
        _add_with_derivatives('/'.join(parts[1:]))
    known_providers = (
        'anthropic', 'openai', 'google', 'openrouter', 'groq', 'mistral',
        'xai', 'cohere', 'perplexity', 'together', 'fireworks', 'deepseek',
    )
    for v in [v for v in variants if '/' not in v]:
        _add(*(f"{provider}/{v}" for provider in known_providers))
    known_aggregators = ('openrouter', 'opencode', 'fireworks', 'groq', 'together')
    for v in [v for v in variants if v.count('/') == 1]:
        _add(*(f"{agg}/{v}" for agg in known_aggregators))
    return variants


def resolve_per_model_reasoning_effort(model: str, overrides: dict | None) -> dict | None:
    """Per-model reasoning_effort override with spelling tolerance; first non-None parse wins.

    Order: exact → dots↔dashes → provider stripped → aggregator stripped → known prefixes added.
    """
    if not overrides or not isinstance(overrides, dict) or not model:
        return None
    for variant in _canonical_model_variants(model):
        if variant in overrides:
            result = parse_reasoning_effort(overrides[variant])
            if result is not None:
                return result
    return None


def resolve_per_model_provider_routing(model: str, models: dict | None) -> dict:
    """``provider_routing.models.<id>`` entry for *model*, spelling-tolerant like
    ``reasoning_overrides``; ``{}`` when none matches. Only the keys a user sets per model
    are returned so unset ones fall through to the flat ``provider_routing`` values."""
    if not model or not isinstance(models, dict):
        return {}
    for variant in _canonical_model_variants(model):
        entry = models.get(variant)
        if isinstance(entry, dict):
            return entry
    return {}


def resolve_reasoning_config(cfg: dict | None, model: str = "") -> dict | None:
    """Effective reasoning config for *model*: per-model override, then global ``agent.reasoning_effort``.

    Single chokepoint for every surface (CLI, gateway, TUI, cron, ``/model``, fallback activation).
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    agent_cfg = cfg.get("agent") if isinstance(cfg.get("agent"), dict) else {}

    if not model:
        model_cfg = cfg.get("model")
        if isinstance(model_cfg, dict):
            model_cfg = model_cfg.get("default") or model_cfg.get("model") or ""
        model = model_cfg.strip() if isinstance(model_cfg, str) else ""
    per_model = resolve_per_model_reasoning_effort(model, agent_cfg.get("reasoning_overrides") or {})
    if per_model is not None:
        return per_model

    # Keep the raw value: ``or ""`` would turn a YAML False into "" and silently re-enable thinking.
    effort = agent_cfg.get("reasoning_effort", "")
    result = parse_reasoning_effort(effort)
    if effort and str(effort).strip() and result is None:
        import logging
        logging.getLogger(__name__).warning("Unknown reasoning_effort '%s', using default (medium)", effort)
    return result


def is_termux() -> bool:
    """True inside Termux (Android): ``TERMUX_VERSION`` or the Termux-specific ``PREFIX`` path."""
    prefix = os.getenv("PREFIX", "")
    return bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)


_wsl_detected: bool | None = None


def is_wsl() -> bool:
    """True inside WSL1/WSL2 (``microsoft`` marker in ``/proc/version``); cached per process."""
    global _wsl_detected
    if _wsl_detected is None:
        try:
            with open("/proc/version", "r", encoding="utf-8") as f:
                _wsl_detected = "microsoft" in f.read().lower()
        except Exception:
            _wsl_detected = False
    return _wsl_detected


def windows_path_to_wsl(path: str) -> str | None:
    """Convert a Windows drive path (``C:\\...``) to its ``/mnt/<drive>/...`` form."""
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", str(path or "").strip())
    return f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}" if match else None


def wsl_unc_path_to_posix(path: str) -> str | None:
    """Convert a ``\\\\wsl.localhost\\<distro>\\...`` (or legacy ``\\\\wsl$``) UNC path to POSIX."""
    normalized = str(path or "").strip().replace("/", "\\")
    match = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+\\(.*)$", normalized, re.IGNORECASE)
    return "/" + match.group(1).replace("\\", "/") if match else None


def translate_cwd_for_wsl_backend(cwd: str) -> str:
    """Map a Windows-host cwd (drive path or ``\\\\wsl.localhost\\`` UNC) to POSIX when Hermes runs in WSL.

    No-op off WSL and for paths already POSIX.
    """
    if not is_wsl():
        return cwd
    for translator in (wsl_unc_path_to_posix, windows_path_to_wsl):
        translated = translator(cwd)
        if translated is not None:
            return translated
    return cwd


_container_detected: bool | None = None


def is_container() -> bool:
    """True inside a container (Docker/Podman/LXC/Kubernetes markers); cached per process.

    See: NousResearch/hermes-agent#47111
    """
    global _container_detected
    if _container_detected is None:
        _container_detected = _detect_container()
    return _container_detected


def _read_proc(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _proc_file_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    content = _read_proc(path)
    return any(marker in content for marker in markers)


def _detect_container() -> bool:
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
        or _proc_file_has_marker("/proc/1/cgroup", ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio"))
    ):
        return True
    # cgroup v2: /proc/1/cgroup is just "0::/"; the runtime still shows in mountinfo — but ONLY on
    # the root ("/") mount line. A host that merely *runs* containers exposes every container's
    # overlay lowerdir (``lowerdir=/var/lib/containerd/...``) at non-root mount points, which a
    # whole-file scan misread as "inside a container" and flipped subprocess HOME (#58135).
    return _root_mount_has_marker("/proc/self/mountinfo", ("kubepods", "containerd", "crio"))


def _root_mount_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    """mountinfo field 5 (index 4) is the mount point; only the root ("/") line is the process's own rootfs."""
    root_lines = [line for line in _read_proc(path).splitlines() if len(f := line.split()) >= 5 and f[4] == "/"]
    return any(marker in line for line in root_lines for marker in markers)


def get_config_path() -> Path:
    """Return the path to ``config.yaml`` under HERMES_HOME."""
    return get_hermes_home() / "config.yaml"


def get_skills_dir() -> Path:
    """Return the path to the skills directory under HERMES_HOME."""
    return get_hermes_home() / "skills"


def get_env_path() -> Path:
    """Return the path to the ``.env`` file under HERMES_HOME."""
    return get_hermes_home() / ".env"


def apply_ipv4_preference(force: bool = False) -> None:
    """Monkey-patch ``socket.getaddrinfo`` to prefer IPv4 when *force* is True.

    Broken-IPv6 hosts hang on AAAA for the full TCP timeout; ``AF_UNSPEC`` resolves as ``AF_INET``,
    falling back to full resolution when no A record exists (pure-IPv6 hosts still work).
    """
    if not force:
        return

    import socket

    if getattr(socket.getaddrinfo, "_hermes_ipv4_patched", False):
        return
    _original_getaddrinfo = socket.getaddrinfo

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if family == 0:  # AF_UNSPEC — caller didn't request a specific family
            try:
                return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)
            except socket.gaierror:  # no A record — pure-IPv6 host
                return _original_getaddrinfo(host, port, family, type, proto, flags)
        return _original_getaddrinfo(host, port, family, type, proto, flags)
    _ipv4_getaddrinfo._hermes_ipv4_patched = True  # type: ignore[attr-defined]
    socket.getaddrinfo = _ipv4_getaddrinfo  # type: ignore[assignment]


PARTIAL_STREAM_STUB_ID = "partial-stream-stub"
FINISH_REASON_LENGTH = "length"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

# OpenRouter request-time routing variants (docs: guides/routing/model-variants).
# These suffixes are per-request routing modifiers valid on ANY model id —
# ":nitro" sorts the endpoint pool by throughput and admits priority-tier
# endpoints, ":floor" sorts by price and admits flex-tier endpoints, ":exacto"
# applies quality-first provider sorting, ":online" attaches the web plugin.
# They are never separate catalog entries: /models lists only the base id, so
# every catalog lookup must key on the BASE while the suffixed id stays on the
# wire.
# NOT in this set: ":free", ":batch", ":thinking", ":extended" — those ARE
# distinct catalog SKUs with their own /models entries (and their own context
# windows), so stripping them would resolve the wrong window.
OPENROUTER_VARIANT_SUFFIXES: frozenset[str] = frozenset(
    {"nitro", "floor", "exacto", "online"}
)


def openrouter_variant_base(model_id: str) -> str | None:
    """Return the base model id when ``model_id`` carries a recognized
    OpenRouter routing-variant suffix (e.g. ``x-ai/grok-4:nitro`` →
    ``x-ai/grok-4``), else ``None``.

    Lives here rather than in ``hermes_cli.models`` so the metadata layer
    (``agent.model_metadata``) can share one definition without importing the
    CLI — this module is dependency-free by contract.

    >>> openrouter_variant_base("x-ai/grok-4:nitro")
    'x-ai/grok-4'
    >>> openrouter_variant_base("x-ai/grok-4:free") is None
    True
    >>> openrouter_variant_base("x-ai/grok-4") is None
    True
    """
    base, sep, suffix = (model_id or "").rpartition(":")
    if not sep or not base:
        return None
    if suffix.lower() in OPENROUTER_VARIANT_SUFFIXES:
        return base
    return None


AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"


def venv_bin_dir(venv_dir, *, windows: bool | None = None) -> Path:
    """Venv executable dir (``Scripts``/``bin``); *windows* lets tests exercise Windows paths on Linux.

    Returned unconditionally — callers differ on whether a missing venv is an error.

    Canonical helper for venv layout. This was open-coded in seven places across four ``hermes_cli`` modules
    using three different Windows predicates (``platform.system()``, ``is_windows()``, ``_is_windows()``);
    each new call site had to re-derive it, and #76091 shipped an eighth copy because the correct behaviour
    lived 2400 lines away in another function. A few sites outside ``hermes_cli``
    (``tools/code_execution_tool.py``, ``agent/lsp/install.py``, ``agent/lsp/servers.py``) still hand-roll
    it — convert them as they are touched.
    """
    windows = sys.platform == "win32" if windows is None else windows
    return Path(venv_dir) / ("Scripts" if windows else "bin")


def project_venv_dir(project_root) -> Path | None:
    """The project's ``venv`` or ``.venv`` dir when one exists (``uv venv`` defaults to ``.venv``).

    ``uv venv`` defaults to ``.venv`` while our installers create ``venv``, so both layouts are in the wild.
    Call sites that only knew about ``venv`` silently no-oped on a ``.venv`` install — that is how the
    Windows shim-lock preflight skipped itself entirely (#79542). ``venv`` wins when both exist, matching
    what the installers write.
    """
    root = Path(project_root)
    return next((root / n for n in ("venv", ".venv") if (root / n).is_dir()), None)


def venv_python_path(venv_dir, *, windows: bool | None = None) -> Path:
    """Path to the Python interpreter inside *venv_dir* (may not exist)."""
    bin_dir = venv_bin_dir(venv_dir, windows=windows)
    return bin_dir / ("python.exe" if bin_dir.name == "Scripts" else "python")


# First-party roots: an ImportError naming one means our own tree is inconsistent. The
# update post-probe shares this set so the guard that BLOCKS and the hint that EXPLAINS agree.
FIRST_PARTY_MODULE_ROOTS = frozenset({
    "agent", "acp_adapter", "cli", "cron", "gateway", "model_tools", "plugins",
    "providers", "tools", "toolsets", "run_agent", "tui_gateway", "utils",
})


def is_first_party_module(name: str | None) -> bool:
    """True when *name* ships with Hermes (exact first segment; ``startswith`` would claim ``agentops``)."""
    root = str(name).split(".")[0] if name else ""
    return bool(root) and (root in FIRST_PARTY_MODULE_ROOTS or root.startswith("hermes_"))


def partial_update_hint(exc: BaseException) -> list[str]:
    """Recovery guidance lines when *exc* looks like a half-updated tree, else ``[]``."""
    # A missing third-party dep (bad venv, missing extra) is a different problem.
    if (not isinstance(exc, ImportError) or isinstance(exc, ModuleNotFoundError)
            or not is_first_party_module(getattr(exc, "name", None))):
        return []
    return [
        "",
        "This looks like a partially-updated install: one module was refreshed "
        "and a related one was not.",
        "Re-run the update to bring the whole tree to the same version:",
        "    hermes update",
        "If that also fails, reinstall: https://hermes-agent.nousresearch.com",
    ]


def emit_partial_update_hint(exc: BaseException, *, file=None) -> bool:
    """Print recovery guidance for a half-updated tree."""
    lines = partial_update_hint(exc)
    if not lines:
        return False
    for line in (f"Error: {exc}", *lines):
        print(line, file=sys.stderr if file is None else file)
    return True


def normalize_scope(scope: str | Path | None) -> str | None:
    """Normalize a WRITE-side registry scope key, preserving ``None``.

    Two different contracts live on the same registries — do not unify them:

    * **Write / slot paths** (``register_*``, ``snapshot_registration``,
      ``restore_registration``, tool-registry slot lookup): ``None`` means
      the process-global layer and must stay ``None``. Use this function.
    * **Read paths** (``list_providers``, ``get_provider``): ``None`` means
      "the active home's scope" and must go through :func:`hermes_home_key`
      (falsy input resolves to the active default home). Using this
      function there hides every scoped registration — the exact bug
      fixed after e66a627aa5.

    Both normalize non-None values identically (resolved absolute path,
    normcase on Windows) so writes and reads agree on the key.
    """
    return hermes_home_key(scope) if scope is not None else None

