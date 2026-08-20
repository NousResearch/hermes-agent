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
    """Return the platform-native default Hermes home path."""
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


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


def get_default_hermes_root() -> Path:
    """Root Hermes dir for profile-level ops: ``<root>`` when ``HERMES_HOME=<root>/profiles/<name>``."""
    global _default_hermes_root_memo
    native_home = _get_platform_default_hermes_home()
    env_home = os.environ.get("HERMES_HOME", "")
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
    """Hermes-managed Node dirs in lookup order; both Windows and POSIX shapes so migrated installs work.

    Keep in sync with hermesManagedNodePathEntries() in apps/desktop/electron/backend-env.ts.
    """
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


def _iter_managed_node_candidates(names: list[str], home: Path | None = None):
    """Yield existing (and on POSIX, executable) ``<node-dir>/<name>`` files."""
    for directory in iter_hermes_node_dirs(home):
        for name in names:
            candidate = directory / name
            if candidate.is_file() and (sys.platform == "win32" or os.access(candidate, os.X_OK)):
                yield candidate


def _first_runnable_managed(names: list[str]) -> tuple[str | None, bool]:
    """Return ``(first runnable candidate, saw a broken one)``."""
    broken = False
    for candidate in map(str, _iter_managed_node_candidates(names)):
        if node_tool_runnable(candidate):
            return candidate, broken
        broken = True
    return None, broken


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


_HERMES_NODE_TARGET_MAJOR = int(os.environ.get("HERMES_NODE_TARGET_MAJOR", "22"))
_managed_node_heal_attempted = False
_NODE_BOOTSTRAP_SCRIPT = Path(__file__).resolve().parent / "scripts" / "lib" / "node-bootstrap.sh"

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
    """Return True when any Hermes-managed node/npm/npx shim exists on disk."""
    names = [n for c in ("node", "npm", "npx") for n in _candidate_node_command_names(c)]
    return next(_iter_managed_node_candidates(names, home), None) is not None


def _path_under_any(path: str, roots: list[str]) -> bool:
    """True when *path* sits inside one of *roots* (same drive).

    Compared via ``normcase``: psutil and env vars can disagree on Windows drive-letter casing.
    """
    path_norm = os.path.normcase(os.path.normpath(path))
    for root in roots:
        root_norm = os.path.normcase(os.path.normpath(root))
        try:
            if os.path.commonpath([path_norm, root_norm]) == root_norm:
                return True
        except ValueError:  # different drives on Windows
            continue
    return False


def managed_node_tree_in_use(home: Path | None = None) -> bool:
    """True when a running process executes from the managed Node tree.

    Windows locks running executables against delete/overwrite, so the updater must not rewrite
    ``%HERMES_HOME%\\node`` while the desktop app holds it (``[WinError 5]`` on ``npm.cmd``).

    Always ``False`` on POSIX, which has no equivalent lock semantics. See #80926.
    """
    if sys.platform != "win32":
        return False
    try:
        import psutil
    except Exception:
        return False
    dirs: list[str] = []
    for directory in iter_hermes_node_dirs(home):
        try:
            dirs.append(str(Path(directory).resolve()))
        except OSError:
            continue
    if not dirs:
        return False
    try:
        procs = psutil.process_iter(["exe", "cmdline"])
    except Exception:
        return False
    for proc in procs:
        try:
            info = proc.info
        except Exception:
            continue
        exe = info.get("exe")
        if exe:
            try:
                exe = str(Path(exe).resolve())
            except (OSError, ValueError):
                exe = str(exe)
        if any(_path_under_any(p, dirs) for p in ([exe] if exe else []) + list(info.get("cmdline") or [])):
            return True
    return False


_managed_node_in_use_notice_printed = False


def _print_managed_node_in_use_notice() -> None:
    """Print the managed-Node deferral notice once per process."""
    global _managed_node_in_use_notice_printed
    if _managed_node_in_use_notice_printed:
        return
    _managed_node_in_use_notice_printed = True
    print(
        "→ Hermes-managed Node.js is in use by a running app; deferring its "
        "upgrade until the app is closed (re-run `hermes update` afterwards).", flush=True,
    )


def _fetch_url(url: str, timeout: int) -> bytes | None:
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except OSError:
        return None


def _stage_windows_node_zip(home: Path, node_arch: str) -> Path | None:
    """Download the target-major portable Node zip into a sibling ``node.new-*`` dir.

    A sibling makes the later swap a same-volume rename. ``None`` on any failure.
    """
    import tempfile
    import uuid
    import zipfile

    index_url = f"https://nodejs.org/dist/latest-v{_HERMES_NODE_TARGET_MAJOR}.x/"
    index_bytes = _fetch_url(index_url, 60)
    if index_bytes is None:
        return None
    pattern = rf"node-v{_HERMES_NODE_TARGET_MAJOR}\.\d+\.\d+-win-{node_arch}\.zip"
    match = re.search(pattern, index_bytes.decode("utf-8", errors="replace"))
    if not match:
        return None
    zip_name = match.group(0)
    zip_bytes = _fetch_url(f"{index_url}{zip_name}", 300)
    if zip_bytes is None:
        return None
    staged = home / f"node.new-{uuid.uuid4().hex[:8]}"
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / zip_name).write_bytes(zip_bytes)
            extract_dir = tmp_path / "extract"
            extract_dir.mkdir()
            with zipfile.ZipFile(tmp_path / zip_name) as archive:
                archive.extractall(extract_dir)
            extracted = next(extract_dir.glob("node-v*"), None)
            if extracted is None or not extracted.is_dir():
                return None
            shutil.move(str(extracted), str(staged))
    except OSError:
        return None
    return staged


def _swap_node_tree(target: Path, staged: Path) -> bool | None:
    """Rename the live tree aside (``node.old-*``) and *staged* into place.

    ``None`` when the OS refuses to move the live tree (in use; untouched, retried next time),
    ``False`` when the staged tree cannot be moved in (live tree rolled back).
    """
    backup = target.parent / f"node.old-{staged.name.removeprefix('node.new-')}"
    had_live = target.exists()
    if had_live:
        try:
            os.replace(str(target), str(backup))
        except OSError:
            _print_managed_node_in_use_notice()
            shutil.rmtree(staged, ignore_errors=True)
            return None
        # Rename preserves mtime: touch the backup (best-effort) so a concurrent heal's
        # litter sweep never removes it mid-swap.
        with contextlib.suppress(OSError):
            os.utime(backup, None)
    try:
        os.replace(str(staged), str(target))
    except OSError:
        if had_live:  # roll the live tree back
            with contextlib.suppress(OSError):
                os.replace(str(backup), str(target))
        shutil.rmtree(staged, ignore_errors=True)
        return False
    if had_live:
        # Locked files may keep the old tree on disk until the next heal; safe.
        shutil.rmtree(backup, ignore_errors=True)
    return True


def _heal_managed_node_windows(home: Path | None = None) -> bool | None:
    """Redownload the portable Node zip into ``%HERMES_HOME%\\node`` on Windows.

    ``True`` on success, ``False`` on genuine failure (offline, bad archive), ``None`` when the
    tree is in use and the heal is deferred — callers must not record the once-per-process attempt
    for ``None``. Staging-first (extract to ``node.new-*``, rename live aside, rename staged in) so
    an interrupted heal cannot gut the install; a refused rename *is* the in-use signal.

    The replacement is staging-first: the new tree is fully downloaded and extracted to a sibling
    ``node.new-*`` directory, then the live tree is renamed aside (``node.old-*``) and the staged tree
    renamed into place. The live tree is never deleted before its replacement is ready, so an interrupted
    heal cannot gut the running installation. Windows allows renaming a tree whose executables are running
    (images are mapped with ``FILE_SHARE_DELETE`` — the same mechanism as the hermes.exe quarantine); when
    the OS refuses the rename, that refusal *is* the in-use signal and the heal defers instead of forcing
    the write and crashing with ``PermissionError: [WinError 5]`` on ``npm.cmd`` (#80926).
    """
    import time

    arch = (os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get("PROCESSOR_ARCHITECTURE", "")).lower()
    node_arch = {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "x86": "x86"}.get(arch)
    if node_arch is None:
        return False
    home = home or get_hermes_home()
    target = home / "node"
    # Cheap pre-check; the rename-based swap is the authoritative guard.
    if managed_node_tree_in_use(home):
        _print_managed_node_in_use_notice()
        return None
    # Sweep litter from interrupted runs; only dirs >10 min old so a concurrent in-flight swap survives.
    cutoff = time.time() - 600
    for stale in (*home.glob("node.old-*"), *home.glob("node.new-*")):
        try:
            if stale.stat().st_mtime < cutoff:
                shutil.rmtree(stale, ignore_errors=True)
        except OSError:
            continue
    staged = _stage_windows_node_zip(home, node_arch)
    if staged is None:
        return False
    return _swap_node_tree(target, staged) and node_tool_runnable(str(target / "node.exe"))


def _run_node_bootstrap(func: str, *, timeout: int, **extra_env: str) -> bool:
    """Source ``scripts/lib/node-bootstrap.sh`` and run shell function *func*."""
    if not _NODE_BOOTSTRAP_SCRIPT.is_file():
        return False
    import subprocess
    try:
        result = subprocess.run(
            ["bash", "-c", f'source "{_NODE_BOOTSTRAP_SCRIPT}" && {func}'],
            env={**os.environ, "HERMES_HOME": str(get_hermes_home()), **extra_env},
            capture_output=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def bootstrap_hermes_managed_node() -> str | None:
    """Install a Hermes-managed Node tree under ``$HERMES_HOME/node`` and return its npm path.

    Hermes never modifies a user-owned toolchain (system, nvm, brew, Nix) that fails ``engines``.
    """
    existing = find_hermes_node_executable("npm")
    if existing:
        return existing
    if sys.platform == "win32":
        ok = _heal_managed_node_windows()
    else:
        # HERMES_NODE_SKIP_LINKS=1 keeps node/npm/npx out of ~/.local/bin: never shadow the user toolchain.
        ok = _run_node_bootstrap("_nb_install_bundled_node", timeout=600, HERMES_NODE_SKIP_LINKS="1")
    if not ok:
        return None
    return _first_runnable_managed(_candidate_node_command_names("npm"))[0]


def heal_hermes_managed_node() -> bool:
    """Redownload Hermes-managed Node when the tree exists but is broken; at most once per process.

    A Windows in-use deferral does NOT record the attempt so a later call can heal once free.

    POSIX installs shell out to ``heal_managed_node`` in ``scripts/lib/node-bootstrap.sh``; Windows
    downloads the portable zip directly (same source as ``install.ps1``). See #80926.
    """
    global _managed_node_heal_attempted
    if _managed_node_heal_attempted or not hermes_managed_node_tree_present():
        return False
    if sys.platform == "win32":
        result = _heal_managed_node_windows()
    else:
        result = _run_node_bootstrap("heal_managed_node", timeout=300)
    if result is None:  # in-use deferral: leave the attempt flag clear
        return False
    _managed_node_heal_attempted = True
    return bool(result)


def _managed_node_tree_outdated(home: Path | None = None) -> bool:
    """True when the managed node runs but is below the target major (heals like a broken tree)."""
    for candidate in _iter_managed_node_candidates(_candidate_node_command_names("node"), home):
        result = _run_version_probe([str(candidate), "--version"])
        if result is None:
            return False  # broken, not outdated — the runnable probe handles it
        try:
            version = result.stdout.decode().strip().lstrip("v")
            major = int(version.split(".")[0])
        except (ValueError, IndexError):
            return False
        # A pre-release is outdated whatever its major: nodejs.org publishes headers only for
        # final releases, so node-gyp cannot build node-pty. Mirrors node_satisfies_build() in install.sh.
        if "-" in version:
            return True
        return major < _HERMES_NODE_TARGET_MAJOR
    return False


def find_hermes_node_executable(command: str) -> str | None:
    """Hermes-managed Node/npm path, healing broken/outdated trees; heal failure still returns old Node."""
    names = _candidate_node_command_names(command)
    resolved, broken_present = _first_runnable_managed(names)
    needs_heal = broken_present or (resolved is not None and _managed_node_tree_outdated())
    if needs_heal and heal_hermes_managed_node():
        healed, _ = _first_runnable_managed(names)
        if healed:
            return healed
    return resolved


def find_node_executable_on_path(command: str) -> str | None:
    """Node/npm from PATH; on Windows prefer ``.cmd``/``.exe`` (CreateProcess cannot run the bare shim)."""
    if sys.platform != "win32":
        return shutil.which(command)
    command_str = str(command)
    if any(sep and sep in command_str for sep in (os.sep, os.altsep, "/", "\\")):
        return command_str if Path(command_str).is_file() else None
    directories = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    for name in _candidate_node_command_names(command_str):
        for directory in directories:
            if (Path(directory) / name).is_file():
                return str(Path(directory) / name)
    return None


def find_node_executable(command: str) -> str | None:
    """Resolve a Node command, preferring a healthy managed install.

    A managed tree that exists but cannot be healed yields ``None`` rather than system Node.
    """
    managed = find_hermes_node_executable(command)
    if managed:
        return managed
    if hermes_managed_node_tree_present():
        return None
    return find_node_executable_on_path(command)


def with_hermes_node_path(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return *env* with Hermes-managed Node directories prepended to PATH."""
    merged = dict(os.environ if env is None else env)
    parts = [p for p in merged.get("PATH", "").split(os.pathsep) if p]
    for entry in reversed([str(path) for path in iter_hermes_node_dirs() if path.is_dir()]):
        if entry not in parts:
            parts.insert(0, entry)
    merged["PATH"] = os.pathsep.join(parts)
    return merged


def agent_browser_runnable(path: str | None) -> bool:
    """True when *path* is an agent-browser CLI that runs (``--version`` exits 0) or the npx fallback.

    Dead/wrong-arch/hung binaries are rejected so callers try the next candidate.

    A bare presence check (``shutil.which`` / ``Path.exists``) is not enough: agent-browser's npm
    ``postinstall`` re-points a *global* install symlink (e.g. ``/opt/homebrew/bin/agent-browser``) at our
    local ``node_modules/agent-browser/bin/...`` binary, which then disappears on the next ``hermes update``
    — leaving a **dangling symlink** that ``which`` still reports but exec fails on with exit 127 (issue
    #48521). Callers that trust such a path silently break every browser tool.
    """
    if not path:
        return False
    # The npx fallback is a two-token command string, not a path; npx validates at run time.
    if " " in path and path.split()[0].endswith("npx"):
        return True
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


def _root_mount_has_container_runtime(mountinfo: str) -> bool:
    """Return whether the process root mount names a container runtime."""
    runtime_markers = (
        "docker",
        "kubepods",
        "containerd",
        "crio",
        "/lxc/",
        "containers/storage",
    )
    mountinfo_escapes = {
        r"\040": " ",
        r"\011": "\t",
        r"\012": "\n",
        r"\134": "\\",
    }
    for line in mountinfo.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-", 6)
        except (ValueError, IndexError):
            continue
        if separator + 3 >= len(fields):
            continue
        mount_point = fields[4]
        for escaped, unescaped in mountinfo_escapes.items():
            mount_point = mount_point.replace(escaped, unescaped)
        if mount_point != "/":
            continue
        # After the "-" separator the fields are [fstype, mount source,
        # super options...]. A host root mount's source can name a backing
        # device such as /dev/mapper/docker--vg-root whose text merely happens
        # to contain a runtime marker, so match the mount point, the root
        # path (field 3, e.g. /containerd/.../rootfs inside the container),
        # the fstype, and the super options — never the mount source, and
        # never the pre-separator device fields.
        match_fields = [
            mount_point,
            fields[3],
            fields[separator + 1],
            *fields[separator + 3 :],
        ]
        if any(
            marker in field.lower()
            for field in match_fields
            for marker in runtime_markers
        ):
            return True
    return False


def is_container() -> bool:
    """True inside a container (Docker/Podman/LXC/Kubernetes markers); cached per process.

    See: NousResearch/hermes-agent#47111
    """
    global _container_detected
    if _container_detected is None:
        _container_detected = _detect_container()
    return _container_detected


def _proc_file_has_marker(path: str, markers: tuple[str, ...]) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return False
    return any(marker in content for marker in markers)


def _detect_container() -> bool:
    if (
        os.path.exists("/.dockerenv")
        or os.path.exists("/run/.containerenv")
        or os.environ.get("KUBERNETES_SERVICE_HOST")
        or _proc_file_has_marker("/proc/1/cgroup", ("docker", "podman", "/lxc/", "kubepods", "containerd", "crio"))
    ):
        return True
    # cgroup v2: /proc/1/cgroup is just "0::/"; the runtime still shows in mountinfo.
    # Scan only the root mount's structural fields (never the mount source — a
    # host root device like /dev/mapper/docker--vg-root would false-positive).
    try:
        with open("/proc/self/mountinfo", "r", encoding="utf-8") as f:
            mountinfo = f.read()
        if _root_mount_has_container_runtime(mountinfo):
            return True
    except OSError:
        pass
    return _proc_file_has_marker("/proc/self/mountinfo", ("kubepods", "containerd", "crio"))


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
