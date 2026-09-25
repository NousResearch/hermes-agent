"""Shared constants for Hermes Agent.

Import-safe, stdlib-only — importable from anywhere without circular-import risk.
"""

import contextlib
import os
import re
import shutil
import stat
import sys
from collections.abc import MutableMapping
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


def _expand_hermes_home(path: str) -> Path:
    """Expand environment and user-home syntax in a Hermes home path."""
    return Path(os.path.expanduser(os.path.expandvars(path)))


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
    # Latch on the FIRST check regardless of outcome (one-shot contract). Previously the latch
    # was only set on the warning branch, so with no active_profile (or "default") the stat +
    # read re-ran on every get_hermes_home() call (#90065).
    _profile_fallback_warned = True
    try:
        fallback_home = _get_platform_default_hermes_home()
        active_path = fallback_home / "active_profile"
        active = active_path.read_text(encoding="utf-8").strip() if active_path.exists() else ""
    except (UnicodeDecodeError, OSError):
        active = ""
    if active and active != "default":
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
        return _expand_hermes_home(override)
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
    request is scoped to another profile (e.g. embedded ``/chat`` under ``--open-profile``). Follows
    ``HERMES_HOME`` live on purpose: routed-profile DECISIONS compare against
    :func:`get_routing_process_hermes_home` instead (#119242).
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    return _expand_hermes_home(val) if val else _get_platform_default_hermes_home()


# Host-pinned identity of the profile this process serves as its own (None: follow HERMES_HOME).
_PINNED_PROCESS_HERMES_HOME: str | None = None


def pin_process_hermes_home(path: str | Path | None) -> None:
    """Pin the home this process serves as its own profile, for "is this task routed?" decisions.

    An embedding host that serves several profiles and mirrors the active turn's profile into
    ``os.environ["HERMES_HOME"]`` for legacy readers (Hermes WebUI) otherwise makes every turn's own
    profile look like the launch profile: ``agent.secret_scope.serves_routed_profile()`` turns
    False and that turn's MCP connections fall back to bare, cross-profile names; the sibling
    launch-home checks (``secret_scope._is_process_home``, ``tools.environments.local._is_routed_home``,
    ``hermes_cli.env_loader._process_hermes_home``) misjudge the same way. ``None`` clears the pin.

    Process-global on purpose: it names the process's own identity, not a per-task value. It is NOT
    folded into :func:`get_process_hermes_home`: :func:`get_hermes_home` falls back to that for
    tasks carrying no override, and the host's mirror exists precisely so those readers see the
    served profile. Hosts that never mutate ``HERMES_HOME`` need not call this (no-op).
    """
    global _PINNED_PROCESS_HERMES_HOME
    _PINNED_PROCESS_HERMES_HOME = None if path is None else str(path)


def process_hermes_home_is_pinned() -> bool:
    return _PINNED_PROCESS_HERMES_HOME is not None


def get_routing_process_hermes_home() -> Path:
    """Launch home for routed-profile decisions: the pinned home, else :func:`get_process_hermes_home`."""
    pinned = _PINNED_PROCESS_HERMES_HOME
    return _expand_hermes_home(pinned) if pinned else get_process_hermes_home()


# Hermes-managed runtime downloads at the root of a home (GGUF models, llama.cpp runtimes,
# managed Node): re-downloadable on demand and routinely tens to hundreds of GB. Shared by
# ``hermes backup`` (excludes them) and ``profile create --clone-all`` (skips them from the
# default profile) so the two lists cannot drift apart.
LOCAL_RUNTIME_ROOT_DIRS: frozenset[str] = frozenset({"models", "runtimes", "node"})

# get_default_hermes_root() memo keyed on (native home, expanded HERMES_HOME) so it stays
# fresh when a test or plugin mutates either input; saves ~80us/call at 31+ sites.
_default_hermes_root_memo: "tuple[str, str, Path] | None" = None


def get_default_hermes_root(*, home: str | Path | None = None) -> Path:
    """Root of an explicit home, or the process home when none is supplied."""
    global _default_hermes_root_memo
    native_home = _get_platform_default_hermes_home()
    env_home = str(home).strip() if home is not None else os.environ.get("HERMES_HOME", "").strip()
    env_path = _expand_hermes_home(env_home) if env_home else None
    memo_key = (str(native_home), str(env_path) if env_path is not None else "")
    memo = _default_hermes_root_memo
    if memo is not None and memo[:2] == memo_key:
        return memo[2]
    result = native_home
    if env_path is not None:
        try:
            env_path.resolve().relative_to(native_home.resolve())  # under ~/.hermes (normal or profile mode)
        except ValueError:  # Docker/custom root: <root>/profiles/<name> -> <root>, else HERMES_HOME itself
            result = env_path.parent.parent if env_path.parent.name == "profiles" else env_path
    _default_hermes_root_memo = (*memo_key, result)
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


# A directory under profiles/ is a profile only when something identifies it as one.
# Runtime side-effects (cron heartbeats, log rotation, caches) create dirs that carry
# none of these; a pre-tombstone ghost shell or a stray infrastructure dir must never be
# listed, served, ticked, or seeded with the default install's credentials.
_PROFILE_IDENTITY_MARKERS = ("config.yaml", ".env", "SOUL.md", "profile.yaml", "auth.json", "state.db")
# Canonical named-profile id grammar; every profile-directory gate imports this one object.
PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def named_profile_has_identity(profile_home: str | Path) -> bool:
    # A dangling symlinked marker (clone/migration leftover) is still an identity claim:
    # ``is_file()`` follows links, so it alone would make such a profile unlistable.
    home = Path(profile_home)
    return any((home / marker).is_file() or (home / marker).is_symlink() for marker in _PROFILE_IDENTITY_MARKERS)


def named_profile_has_servable_identity(profile_home: str | Path) -> bool:
    """Stricter than :func:`named_profile_has_identity`: is this dir a profile a host should change
    its own posture for?

    An EMPTY ``.env`` is all a crashed ``hermes profile create`` leaves behind, and it is enough for
    ``named_profile_has_identity``. Listing such a shell is harmless; counting it as a second tenant
    is not — it flips the whole host's credential reads fail-closed at the next boot. Every other
    marker, and a non-empty or symlinked ``.env``, still counts.
    """
    home = Path(profile_home)
    for marker in _PROFILE_IDENTITY_MARKERS:
        path = home / marker
        if path.is_symlink():
            return True
        try:
            if path.is_file() and (marker != ".env" or path.stat().st_size > 0):
                return True
        except OSError:
            continue
    return False


def named_profile_is_live(profile_home: str | Path) -> bool:
    """A resolvable named profile: an existing dir with identity that has not been deleted.
    ``-p``/``--profile`` resolution and ``profile_exists`` share this so a stale ghost shell can
    never be started as a backend (whose ``ensure_hermes_home`` would rebuild the full tree)."""
    home = Path(profile_home)
    return home.is_dir() and named_profile_has_identity(home) and not named_profile_is_deleted(home)


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


def display_hermes_home(home: Path | None = None) -> str:
    """User-facing ``~/`` display string for HERMES_HOME (``~/.hermes/profiles/coder``).

    ``home`` overrides the lookup for callers that run before the CLI has applied the sticky
    ``active_profile`` (``get_hermes_home()`` would emit the wrong-profile fallback warning there).
    """
    if home is None:
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
    profile_home = str(_expand_hermes_home(hermes_home) / "home")
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
    import tempfile
    try:
        return tempfile.gettempdir()
    except (RuntimeError, OSError):
        # no HOME/USERPROFILE at all (env-less child on Windows): tempfile cannot expand ``~``
        return "/tmp"  # no-tmp: ok — last-resort fallback for an env with no home; not a write target we choose


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
    if not current_home or _is_profile_home(current_home, profile_home):
        return repaired
    return None


def apply_subprocess_home_env(env: MutableMapping[str, str]) -> None:
    """Apply Hermes' subprocess HOME contract to *env* in-place: ``HOME``/``HERMES_REAL_HOME``
    per the home mode, and the temp vars re-pointed at ``env["HERMES_HOME"]``'s scratch dir."""
    real_home = get_real_home(env)
    if real_home:
        env["HERMES_REAL_HOME"] = real_home
    home = get_subprocess_home(env)
    if home:
        env["HOME"] = home
    apply_scratch_tmp_env(env)


# --- Scratch dir: Hermes' own temp space, never the system /tmp ---
# System temp is tmpfs on most Linux distros and containers, so browser profiles, PTY probes,
# download spools and every ``tempfile.mkdtemp()`` a Hermes-launched script performs eat RAM
# and vanish on reboot. ``HERMES_HOME/cache/scratch`` is real storage with an IDLE retention:
# an entry lives while anything inside it is still being written and goes 24h after the last
# write anywhere in its subtree. A fixed age was wrong both ways — a directory's own mtime only
# moves when a direct child is added or removed, so a lane writing deep inside a tree looked
# untouched and lost its worktree at the deadline, while finished trees (a 7 GB clone with its
# own venv per campaign lane) sat for three days and filled the disk.
SCRATCH_TMP_ENV_VARS = ("TMPDIR", "TMP", "TEMP")
SCRATCH_DIR_MARKER_ENV = "HERMES_SCRATCH_DIR"
SCRATCH_MAX_IDLE_HOURS = 24
_SCRATCH_PRUNE_STAMP = ".last_prune"
_SCRATCH_PRUNE_INTERVAL_SECONDS = 3600
_scratch_pruned_once = False

# AF_UNIX socket paths cap at 104 bytes (macOS) / 108 (Linux). Chrome appends
# ``com.google.Chrome.XXXXXX/SingletonSocket`` (~45) and the code kernel
# ``hermes_rpc_<32 hex>.sock`` (~49) to the temp root, so a root longer than this budget
# makes the bind fail (Chrome: "Socket path too long" at startup).
SOCKET_TMPDIR_MAX_LEN = 50


def socket_safe_tmpdir(max_len: int | None = None) -> str:
    """Temp root short enough for AF_UNIX sockets. The scratch dir usually fits; macOS
    ``TMPDIR`` never does and a deep profile home may not, so those fall back to the OS
    default root for sockets only (everything else stays in the scratch dir).
    ``max_len`` tightens the budget for callers that append more than the ~49-byte
    ``hermes_rpc_<32hex>.sock`` suffix the default is tuned for."""
    import tempfile

    if sys.platform == "darwin":
        return "/tmp"  # no-tmp: ok — AF_UNIX 104-byte socket path limit on darwin
    budget = SOCKET_TMPDIR_MAX_LEN if max_len is None else max_len
    candidate = tempfile.gettempdir()
    if len(candidate) <= budget or not os.path.isdir(
        "/tmp"  # no-tmp: ok — probe, not a write target
    ):
        return candidate
    return "/tmp"  # no-tmp: ok — AF_UNIX 108-byte socket path limit on Linux


# ---- Managed mode (NixOS declarative config) ----
# Canonical home of "is this install package-manager managed": ``hermes_cli.config`` re-exports
# these, and :func:`apply_secure_dir_policy` below reads them. Lives here because constants
# must stay import-safe from the CLI.
_MANAGED_TRUE_VALUES = ("true", "1", "yes")
# Only the NixOS module ever wrote a bare "true" or an empty marker.
_LEGACY_MANAGED_SYSTEM = "nixos"
# Homebrew is no longer a supported distribution: these markers fall through to git/unknown
# detection instead of blocking config writes.
_IGNORED_MANAGED_VALUES = frozenset({"brew", "homebrew"})
# Explicit opt-out (``HERMES_MANAGED=false``): without this a bool-shaped value became a package
# manager literally named "false" and is_managed() blocked `hermes update` (#12864).
_MANAGED_FALSE_VALUES = frozenset({"false", "0", "no", "off"})


def get_managed_system(home: str | Path | None = None) -> str | None:
    """Return the package manager owning this install, if any.
    Signals: HERMES_MANAGED env var (systemd service) or a ``.managed`` marker file in
    HERMES_HOME (NixOS activation script — interactive shells don't see the service env).
    An unreadable or empty marker still counts as managed (the legacy NixOS shape).

    ``home`` names the home whose marker file is read, for callers that already resolved it
    (:func:`get_scratch_dir` at boot, before ``--profile`` re-homes the process).
    Defaults to the effective home."""
    marker = os.getenv("HERMES_MANAGED", "").strip().lower() or None
    managed_marker = (Path(home) if home is not None else get_hermes_home()) / ".managed"
    if marker is None and managed_marker.exists():
        try:
            marker = managed_marker.read_text(encoding="utf-8", errors="replace").strip().lower()
        except OSError:
            marker = ""
    if marker is None or marker in _IGNORED_MANAGED_VALUES or marker in _MANAGED_FALSE_VALUES:
        return None
    if marker == "" or marker in _MANAGED_TRUE_VALUES:
        return _LEGACY_MANAGED_SYSTEM
    return marker


def _container_or_chmod_skipped() -> bool:
    """Container/chmod-skip detection: the ``HERMES_CONTAINER``/``HERMES_SKIP_CHMOD`` operator
    overrides on top of the canonical :func:`_detect_container` signals (same breadth as
    :func:`is_container` — Docker/Podman/LXC/Kubernetes, cgroup and root-mountinfo). The cached
    :func:`is_container` itself is deliberately avoided: it ignores these env overrides and is
    computed only once per process, so tests could not flip it. Volume-mounted config is not
    forced to owner-only in containers: gateway and dashboard may run as different UIDs, or
    the mount itself needs broader permissions."""
    if os.environ.get("HERMES_CONTAINER") or os.environ.get("HERMES_SKIP_CHMOD"):
        return True
    return _detect_container()


def _resolve_hermes_uid_gid() -> tuple[int | None, int | None]:
    """Read HERMES_UID / HERMES_GID (set by Docker deployments); (None, None) if unset/invalid/Windows.
    The entrypoint chowns HERMES_HOME once, but subdirs created at runtime (``profiles/<name>/``)
    need the same chown or they land root:root and block later uid-mapped workers.

    Docker containers running Hermes commonly set these to map the in-container user to a host user so
    volume-mounted state files end up with the right ownership. See #34107.
    """
    if sys.platform == "win32":
        return None, None

    def _env_int(name: str) -> int | None:
        try:
            return int(os.environ.get(name, "").strip() or None)
        except (TypeError, ValueError):
            return None

    return _env_int("HERMES_UID"), _env_int("HERMES_GID")


def _chown_to_hermes_uid(path) -> None:
    """Chown ``path`` to ``HERMES_UID:HERMES_GID`` when set; EPERM/ENOENT are non-fatal (the
    entrypoint's startup chown -R fixes ownership on the next restart).

    Used by :func:`apply_secure_dir_policy` to keep ownership consistent across all directories
    created by ``ensure_hermes_home`` on Docker deployments. See #34107.
    """
    uid, gid = _resolve_hermes_uid_gid()
    if uid is None and gid is None:
        return
    try:
        os.chown(path, uid if uid is not None else -1, gid if gid is not None else -1)
    except (OSError, AttributeError, NotImplementedError):
        pass


def apply_secure_dir_policy(path, *, home: str | Path | None = None) -> None:
    """Apply the canonical Hermes home-directory permission policy to *path*.

    Owner-only ``0700`` by default, but the operator's explicit and managed sharing choices
    win (#117347): managed installs are left exactly as the package manager / activation
    script set them (#77579); in a container only an explicit ``HERMES_HOME_MODE`` is applied
    (a bind-mounted data dir is often shared with sibling containers, #10757); elsewhere
    ``HERMES_HOME_MODE`` (e.g. ``0701``, ``2770``) is passed to the host's ``chmod``. Special
    bits may be cleared by the host filesystem (macOS commonly clears setgid on directories
    whose group the caller does not belong to). ``HERMES_UID`` / ``HERMES_GID`` ownership is
    applied when those env vars are set (#34107).

    ``home`` (keyword-only: both arguments are path-likes) names the home whose managed-mode
    marker is read, for callers that already resolved it; without it the effective home is
    consulted.

    Import-safe twin of ``hermes_cli.config._secure_dir`` (which delegates here), so callers
    outside the CLI package — like :func:`get_scratch_dir` — share one policy implementation.
    """
    if get_managed_system(home) is not None:
        return
    explicit_mode = os.environ.get("HERMES_HOME_MODE", "").strip()
    if _container_or_chmod_skipped() and not explicit_mode:
        _chown_to_hermes_uid(path)
        return
    try:
        mode = int(explicit_mode or "700", 8)
    except ValueError:
        mode = 0o700
    try:
        os.chmod(path, mode)
    except (OSError, NotImplementedError):
        pass
    _chown_to_hermes_uid(path)


def get_scratch_dir(home: str | Path | None = None, *, prune: bool = True) -> Path:
    """``<home>/cache/scratch`` (created, owner-only); *home* defaults to the active Hermes home.

    Every Hermes process and child gets ``TMPDIR``/``TMP``/``TEMP`` pointed here at boot (see
    :func:`export_scratch_tmp_env`), so ``tempfile`` defaults land here without call sites
    knowing. Entries idle for ``SCRATCH_MAX_IDLE_HOURS`` are pruned at most once per process
    and once per hour across processes (stamp file), so a fan-out of children stays cheap.

    Permissions follow :func:`apply_secure_dir_policy`, so an explicit ``HERMES_HOME_MODE`` or
    a managed/shared home is honored instead of a blanket ``0700`` (#117347).
    """
    base = Path(home) if home is not None else get_hermes_home()
    scratch = base / "cache" / "scratch"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            # The caller's home decides the policy: re-reading the effective home here would
            # warn about a profile the CLI has not switched to yet (boot scratch setup).
            apply_secure_dir_policy(scratch, home=base)
    except OSError:
        pass
    if prune:
        _prune_scratch_dir_once(scratch)
    return scratch


def prune_scratch_dir(scratch: Path | None = None, max_idle_hours: float = SCRATCH_MAX_IDLE_HOURS) -> int:
    """Delete top-level scratch entries with no write anywhere in their subtree for
    *max_idle_hours*, reaping processes and git worktree registrations rooted in them
    first (``hermes_constants_scratch``); return the count removed."""
    from hermes_constants_scratch import prune_idle_entries

    root = scratch if scratch is not None else get_scratch_dir(prune=False)
    return prune_idle_entries(root, max_idle_hours, frozenset({_SCRATCH_PRUNE_STAMP}))


def _prune_scratch_dir_once(scratch: Path) -> None:
    global _scratch_pruned_once
    if _scratch_pruned_once:
        return
    _scratch_pruned_once = True
    import time
    stamp = scratch / _SCRATCH_PRUNE_STAMP
    try:
        if time.time() - stamp.stat().st_mtime < _SCRATCH_PRUNE_INTERVAL_SECONDS:
            return
    except OSError:
        pass
    with contextlib.suppress(Exception):
        stamp.touch()
        prune_scratch_dir(scratch)


def scratch_dir_usage_bytes(scratch: Path | None = None) -> int:
    """Total bytes under the scratch dir (for ``hermes doctor``); 0 when unreadable."""
    root = scratch if scratch is not None else get_scratch_dir(prune=False)
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        for name in filenames:
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, name)).st_size
    return total


def apply_scratch_tmp_env(env: MutableMapping[str, str]) -> bool:
    """Point ``TMPDIR``/``TMP``/``TEMP`` in *env* at the scratch dir of ``env["HERMES_HOME"]``.

    A temp var the user (or the OS: macOS ``/var/folders``, Windows ``%TEMP%``) set is
    respected and nothing changes. A value Hermes itself exported earlier — recognisable
    because it equals ``HERMES_SCRATCH_DIR`` — is re-derived, so a child running under another
    profile's home gets that home's scratch dir rather than its parent's. Returns True when
    the vars were (re)written.
    """
    ours = env.get(SCRATCH_DIR_MARKER_ENV, "")
    for key in SCRATCH_TMP_ENV_VARS:
        value = env.get(key, "").strip()
        if value and value != ours:
            return False
    home = env.get("HERMES_HOME", "").strip()
    try:
        scratch = str(get_scratch_dir(_expand_hermes_home(home) if home else get_process_hermes_home()))
    except (RuntimeError, OSError):
        # No HERMES_HOME and no resolvable user home (a child env built from nothing on
        # Windows): there is no scratch dir to point at; the child keeps the OS default.
        return False
    for key in SCRATCH_TMP_ENV_VARS:
        env[key] = scratch
    env[SCRATCH_DIR_MARKER_ENV] = scratch
    return True


def export_scratch_tmp_env() -> bool:
    """Boot hook: apply :func:`apply_scratch_tmp_env` to this process and reset ``tempfile``'s
    cached default so ``tempfile.gettempdir()`` follows. Call again after anything that
    re-homes the process (``--profile`` resolution); a user-set temp var is never overridden."""
    changed = apply_scratch_tmp_env(os.environ)
    if changed:
        import tempfile
        tempfile.tempdir = None
    return changed


VALID_REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh", "max", "ultra")


def parse_reasoning_effort(effort) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    ``None`` for empty/unrecognized input (caller uses the default); ``{"enabled": False}`` for
    "none"/"false"/"disabled"/YAML False — ``reasoning_effort: false`` must mean disabled.

    The dict form ``{"enabled": true, "effort": "<level>"}`` passes ``effort`` through verbatim so
    providers with bespoke thinking tiers (``fast``/``thinking`` relays) can be asked for their real
    level; bare strings stay strict so a typo like ``hgih`` never reaches the wire. The wire layer
    already tolerates unknown names (``agent.reasoning_effort.clamp_effort``).
    """
    if effort is None or effort is True:
        return None
    if isinstance(effort, dict):
        if effort.get("enabled", True) is False:
            return {"enabled": False}
        # ``or ""``: a falsy effort (0/False) is "no level", never the string "0" on the wire.
        level = str(effort.get("effort") or "").strip()
        return {"enabled": True, "effort": level} if level else None
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

    Order: exact → dots↔dashes → provider stripped → aggregator stripped → known prefixes added →
    reverse lookup of prefixed keys whose stripped forms match (custom provider slugs are not
    enumerable, so a key like ``ollama-local/qwen3.6:27b`` must still match the bare
    ``qwen3.6:27b`` model string a fallback swap feeds after stripping the prefix).
    """
    if not overrides or not isinstance(overrides, dict) or not model:
        return None
    variants = _canonical_model_variants(model)
    for variant in variants:
        if variant in overrides:
            result = parse_reasoning_effort(overrides[variant])
            if result is not None:
                return result
    # Reverse lookup: the key may carry a custom-provider prefix the model string lost
    # (fallback entries and custom-provider resolution feed the bare slug, while the
    # documented key spelling keeps the ``provider/model`` form). Direct and variant
    # matches above still win, so provider-qualified keys stay most specific.
    variant_set = set(variants)
    for key, raw in overrides.items():
        if not isinstance(key, str) or "/" not in key:
            continue
        parts = key.split("/")
        key_forms = _canonical_model_variants(parts[-1])
        if len(parts) >= 3:
            key_forms += _canonical_model_variants("/".join(parts[1:]))
        if any(form in variant_set for form in key_forms):
            result = parse_reasoning_effort(raw)
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
    """Delegate Termux detection without making the bootstrap constants module depend on the full package."""
    from hermes_platform.host.runtime import is_termux as detect

    return detect()


def is_wsl() -> bool:
    """Delegate WSL detection without making the bootstrap constants module depend on the full package."""
    from hermes_platform.host.runtime import is_wsl as detect

    return detect()


def is_container() -> bool:
    """Delegate container detection without making the bootstrap constants module depend on the full package."""
    from hermes_platform.host.runtime import is_container as detect

    return detect()


def _detect_container() -> bool:
    """Keep the historical test seam while delegating canonical detection."""
    from hermes_platform.host.runtime import _detect_container as detect

    return detect()


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
    """Frozen updater surface: pre-PM updaters import this name; pm.environments owns it."""
    from pm.environments import venv_bin_dir as resolve

    return resolve(venv_dir, windows=windows)


def project_venv_dir(project_root) -> Path | None:
    """The project's ``venv`` or ``.venv`` dir when one exists (``uv venv`` defaults to ``.venv``);
    for an install whose interpreter lives outside the checkout, the running interpreter's venv.

    ``uv venv`` defaults to ``.venv`` while our installers create ``venv``, so both layouts are in the wild.
    Call sites that only knew about ``venv`` silently no-oped on a ``.venv`` install — that is how the
    Windows shim-lock preflight skipped itself entirely (#79542). ``venv`` wins when both exist, matching
    what the installers write.

    Installers that keep the interpreter out of the checkout (``$HERMES_HOME/venvs/<name>``, the layout the
    shipped Windows launchers assume) have neither, and the ``project_venv_dir(root) or root / "venv"``
    idiom those call sites share then handed ``uv`` a ``VIRTUAL_ENV`` that does not exist: that one invented
    path skipped the import probe, reclassified every ``hermes tools`` dependency as missing and failed the
    reinstall with interpreter errors (#116148). The interpreter running this module is the only truthful
    answer to "which venv is live", so fall back to it — but only for the checkout it was loaded from. A
    foreign root (test temp dir, another clone) still resolves to ``None``: handing it someone else's venv
    would point the callers' writes at the wrong environment.
    """
    root = Path(project_root)
    in_tree = next((root / n for n in ("venv", ".venv") if (root / n).is_dir()), None)
    if in_tree is not None:
        return in_tree
    # Out-of-tree install: the path is real by construction (never invented), and non-venv installs
    # keep today's ``None`` so the ``or root / "venv"`` fallback cannot install into a base interpreter.
    running = Path(sys.prefix)
    if (Path(__file__).resolve().parent == root.resolve()
            and sys.prefix != sys.base_prefix
            and venv_python_path(running).is_file()
            and _venv_installs_checkout(running, root)):
        return running
    return None


def _venv_installs_checkout(venv: Path, root: Path) -> bool:
    """Is *venv*'s own ``hermes-agent`` installed from *root*?

    Where this module was loaded from does not answer that: ``PYTHONPATH=<checkout>
    <other install>/bin/python`` runs one checkout's code on another install's interpreter,
    and adopting that venv made a dev checkout's update rewrite the Desktop install's venv
    into an editable install of the dev tree. Every install of a checkout into a venv
    (installers, ``uv sync``) records the source tree in ``direct_url.json``.
    """
    import json
    from importlib.metadata import distributions
    from urllib.parse import urlparse
    from urllib.request import url2pathname

    from pm.environments import site_packages

    for dist in distributions(name="hermes-agent", path=[str(site_packages(venv))]):
        try:
            raw = dist.read_text("direct_url.json")  # windows-footgun: ok — importlib.metadata API, reads utf-8, no encoding=
            url = json.loads(raw or "{}").get("url", "")
        except ValueError:
            continue
        if url.startswith("file:") and Path(url2pathname(urlparse(url).path)).resolve() == root.resolve():
            return True
    return False


def venv_python_path(venv_dir, *, windows: bool | None = None) -> Path:
    """Frozen updater surface: pre-PM updaters import this name; pm.environments owns it."""
    from pm.environments import venv_python

    return venv_python(venv_dir, windows=windows)


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

