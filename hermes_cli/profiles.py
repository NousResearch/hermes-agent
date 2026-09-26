"""Profile management for multiple isolated Hermes instances."""

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_cli.archive_safe import archive_root_dirs, make_targz, normalize_archive_parts, safe_extract_targz
from hermes_cli.home_data_layout import PM_RUNTIME_ROOT_DIRS
from hermes_constants import (
    LOCAL_RUNTIME_ROOT_DIRS, PROFILE_ID_RE, clear_named_profile_deleted, mark_named_profile_deleted,
    named_profile_has_identity, named_profile_is_deleted, named_profile_is_live,
)

logger = logging.getLogger(__name__)

_PROFILE_ID_RE = PROFILE_ID_RE  # legacy alias; hermes_constants.PROFILE_ID_RE is canonical

# Directories bootstrapped inside every new profile. ``home`` is the back-compat/Docker
# HOME for tool subprocesses (host subprocesses keep the real HOME so CLI credentials
# stay visible; containers persist HOME state here). See hermes_constants.get_subprocess_home().
_PROFILE_DIRS = ["memories", "sessions", "skills", "skins", "logs", "plans", "workspace", "cron", "home"]

# Files copied during --clone (if they exist in the source).
_CLONE_CONFIG_FILES = ["config.yaml", ".env", "SOUL.md"]
# Subdirectory files copied during --clone: memory files are part of the agent's curated
# identity, as important as SOUL.md for continuity.
_CLONE_SUBDIR_FILES = ["memories/MEMORY.md", "memories/USER.md"]

# Runtime files stripped after --clone-all. A post-copy step rather than an ignore filter
# because they are created dynamically and may be absent at copy time.
_CLONE_ALL_STRIP: list[str] = [
    "gateway.pid", "gateway_state.json", "processes.json",
    # Bot Desktop runtime identity: pid + create_time of the SOURCE's launcher, its DISPLAY/XAUTHORITY and
    # lease. Copied verbatim, `screen stop` on the clone would kill the source's X server. Browser user data
    # stays, but Chromium's process markers must not let the clone attach to the source's live browser.
    "bot-desktop/launcher.pid", "bot-desktop/env", "bot-desktop/rfb.sock",
    "bot-desktop/lease.json", "bot-desktop/lease.lock",
    "bot-desktop/browser-profile/DevToolsActivePort",
    "bot-desktop/browser-profile/SingletonLock",
    "bot-desktop/browser-profile/SingletonCookie",
    "bot-desktop/browser-profile/SingletonSocket",
]

# Infrastructure excluded from --clone-all ONLY when the source is the default profile
# (``~/.hermes``): git checkout (+ ~3 GB venv), worktrees, sibling profiles, shared bins,
# npm packages, and the managed local-models trees — GGUF weights (tens of GB), the
# llama.cpp runtime binaries and the managed Node install, all re-downloadable on demand
# and resolved from the default root only. Named profiles never hold these at root, so the
# gate avoids silently dropping user data from a named-profile source. Export uses a root
# allow-list instead (``_DEFAULT_EXPORT_INCLUDE_ROOT``): an archive is a portable snapshot,
# a clone must run. The runtime trio is ``LOCAL_RUNTIME_ROOT_DIRS``, shared with
# ``hermes_cli.backup._EXCLUDED_ROOT_DIRS`` so the two lists cannot drift.
_CLONE_ALL_DEFAULT_EXCLUDE_ROOT: frozenset[str] =  frozenset({
    "hermes-agent",
    ".worktrees",
    "profiles",
    "bin",
    "node_modules",
    # Managed runtime trees — install artifacts, never profile state
    # (install/profile bucket split; see test_install_bucket_separation).
    ".hermes-runtime",
    "node",
}) | LOCAL_RUNTIME_ROOT_DIRS

# Per-profile history excluded from --clone-all for ANY source: SQLite session store
# (+wal/shm, can reach many GB), session dirs, `hermes backup` archives, quick-backup
# snapshots, checkpoints. Inheriting them is never useful (restoring one inside the
# clone would resurrect the SOURCE profile's state) and can balloon the copy by tens of GB.
# ``cron`` is scheduled work bound to the source profile and its origin channel: a clone
# that inherits jobs.json runs every job twice (two gateways, same job ids, double spend,
# duplicate deliveries) the moment its gateway starts. The empty dir is recreated below.
_CLONE_ALL_HISTORY_EXCLUDE_ROOT: frozenset[str] = frozenset({
    "state.db", "state.db-wal", "state.db-shm", "sessions", "backups", "state-snapshots", "checkpoints",
    "cron",
})

# Marker written by `hermes profile create --no-skills`. When present at a profile root,
# seed_profile_skills() callers (fresh-create, `hermes update` all-profile sync, the
# dashboard) skip bundled-skill seeding. Delete the file to opt back in.
NO_BUNDLED_SKILLS_MARKER = ".no-bundled-skills"

# ``profile.yaml`` ``role`` values. A role grants backend capabilities (the setup toolset), so
# only the backend writes one, and a copy of a profile (clone-all, import) never inherits it.
SETUP_ROLE = "setup"
PROFILE_ROLES = frozenset({SETUP_ROLE})

# Header seeded into a profile's empty .env so it owns a credentials file from day one.
_PLACEHOLDER_ENV = (
    "# Per-profile secrets for this Hermes profile.\n"
    "# API keys and tokens set here override the shell environment.\n"
    "# Behavioral settings belong in config.yaml, not here.\n"
)


def _non_exportable_entries(directory: str, contents: list) -> set:
    """Entries under *directory* that must never be copied out of a profile: bytecode caches,
    ``*.sock``/``*.tmp`` names, and anything that is not a regular file, directory, or symlink.
    :func:`shutil.copytree` cannot copy special files, so a single live Unix socket without a
    ``.sock`` name (or a FIFO, or a device node) would abort the whole export or clone with
    ``[Errno 6] No such device or address``. Symlinks survive — copytree recreates them."""
    ignored: set = set()
    for entry in contents:
        if entry == "__pycache__" or entry.endswith((".sock", ".tmp", ".pyc", ".pyo")):
            ignored.add(entry)
            continue
        try:
            mode = os.lstat(os.path.join(directory, entry)).st_mode
        except OSError:
            ignored.add(entry)  # vanished mid-walk — copytree would fail on it anyway
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            ignored.add(entry)
    return ignored


def _clone_all_copytree_ignore(source_dir: Path):
    """copytree ignore for --clone-all: history artifacts for any source, infrastructure
    only when the source is the default profile (see the two exclude sets above)."""
    source_resolved = source_dir.resolve()
    root_exclude = set(_CLONE_ALL_HISTORY_EXCLUDE_ROOT)
    if source_resolved == _get_default_hermes_home().resolve():
        root_exclude |= _CLONE_ALL_DEFAULT_EXCLUDE_ROOT

    def _ignore(directory: str, names: List[str]) -> set:
        try:
            at_root = Path(directory).resolve() == source_resolved
        except (OSError, ValueError):
            # resolve() can fail on odd FS layouts (broken symlinks, missing parents).
            # Fail open — better to over-copy than silently drop user data.
            at_root = False
        ignored = _non_exportable_entries(directory, names)
        if at_root:
            ignored.update(root_exclude & set(names))
        return ignored

    return _ignore


# Directories/files to exclude when exporting the default (~/.hermes) profile.
# The default profile contains infrastructure (repo checkout, worktrees, DBs,
# caches, binaries) that named profiles don't have.  We exclude those so the
# export is a portable, reasonable-size archive of actual profile data.
# The ONE exclude-list authority for "infrastructure / runtime state that is
# not profile data". The export path (``_default_export_ignore``) consumes it
# directly; the distribution path (``profile_distribution.USER_OWNED_EXCLUDE``)
# layers its distribution-specific additions on top of it. Keep new root
# artifacts here — not in a downstream copy.
_DEFAULT_EXPORT_EXCLUDE_ROOT = DEFAULT_EXPORT_EXCLUDE_ROOT = frozenset({
    # Infrastructure
    "hermes-agent",         # repo checkout (multi-GB)
    ".worktrees",           # git worktrees
    "profiles",             # other profiles — never recursive-export
    "bin",                  # installed binaries (tirith, etc.)
    "node_modules",         # npm packages
    ".hermes-runtime",      # managed runtime tree (install artifact)
    "node",                 # legacy pre-split managed Node tree
    # Databases & runtime state
    "state.db", "state.db-shm", "state.db-wal",
    "hermes_state.db",
    "response_store.db", "response_store.db-shm", "response_store.db-wal",
    "gateway.pid", "gateway_state.json", "processes.json",
    "auth.json",            # API keys, OAuth tokens, credential pools
    ".env",                 # API keys (dotenv)
    "auth.lock", "active_profile", ".update_check", "source-checks",
    "errors.log",
    ".hermes_history",
    # Caches (regenerated on use)
    "image_cache", "audio_cache", "document_cache",
    "browser_screenshots", "checkpoints",
    "sandboxes",
    "logs",                 # gateway logs
}) | PM_RUNTIME_ROOT_DIRS

# Allow-list for ``export_profile("default")``: when HERMES_HOME equals the
# cwd (Docker/custom deployments), the default profile home is the working
# directory and contains arbitrary user files that should NOT be bundled
# into the export. The set below identifies the *known Hermes profile
# artifacts* at the root of HERMES_HOME; everything else is excluded.
# Sensitive runtime infrastructure (``state.db``, ``logs/``, ``auth.*``,
# other profiles) is intentionally *not* in this list so the export stays
# a portable, credential-free snapshot of the user-facing surface
# (#58394). Add new artifacts here when introduced in ``hermes_constants``.
_DEFAULT_EXPORT_INCLUDE_ROOT = frozenset({
    # Configuration / persona
    "config.yaml", "SOUL.md", "MEMORY.md", "USER.md", "todo.json",
    "system_prompt.md", "AGENTS.md", "CLAUDE.md", ".cursorrules",
    # Desktop appearance overlay (written/applied by the desktop app's export/import).
    "desktop.json",
    # User-facing skill, cron, and session artifacts
    "skills", "cron", "scripts", "sessions",
    # Plugin / memory surfaces (per-profile overrides live here)
    "plugins", "memories", "knowledge", "preferences",
})

# Names that cannot be used as profile aliases
_RESERVED_NAMES = frozenset({"hermes", "default", "test", "tmp", "root", "sudo"})

# Hermes subcommands that cannot be used as profile names/aliases
_HERMES_SUBCOMMANDS = frozenset({
    "chat", "model", "gateway", "setup", "whatsapp", "login", "logout",
    "status", "cron", "doctor", "dump", "config", "pairing", "skills", "tools",
    "mcp", "sessions", "insights", "version", "update", "uninstall", "profile", "plugins", "honcho", "acp",
})


# Path helpers

def _get_profiles_root() -> Path:
    """Named-profiles root, anchored to the hermes root (NOT the current HERMES_HOME, which
    may itself be a profile) so ``coder profile list`` sees all profiles."""
    return _get_default_hermes_home() / "profiles"


def _get_default_hermes_home() -> Path:
    """Default (pre-profile) HERMES_HOME: ``~/.hermes``, or HERMES_HOME itself in
    Docker/custom deployments (e.g. ``/opt/data``)."""
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root()


def _get_active_profile_path() -> Path:
    return _get_default_hermes_home() / "active_profile"


def _get_wrapper_dir() -> Path:
    return Path.home() / ".local" / "bin"


def _wrapper_path(alias: str) -> Path:
    """Wrapper script path for *alias*: ``<alias>.bat`` on Windows, bare name elsewhere."""
    return _get_wrapper_dir() / (f"{alias}.bat" if sys.platform == "win32" else alias)


def _is_our_wrapper(path: Path) -> bool:
    """True when *path* reads as a Hermes-generated wrapper (contains ``hermes -p``)."""
    try:
        return "hermes -p" in path.read_text(encoding="utf-8-sig")
    except Exception:
        return False


def _missing_profile_error(canon: str) -> FileNotFoundError:
    return FileNotFoundError(f"Profile '{canon}' does not exist. Create it with: hermes profile create {canon}")


def _unknown_profile_error(canon: str) -> FileNotFoundError:
    """For delete/rename/export of a name that matches no profile (likely a typo)."""
    return FileNotFoundError(f"No profile named '{canon}'. See your profiles with: hermes profile list")


def _profile_exists_error(canon: str) -> FileExistsError:
    return FileExistsError(
        f"A profile named '{canon}' already exists. Switch to it with `hermes profile use {canon}`, "
        "see all profiles with `hermes profile list`, or choose a different name."
    )


_PROFILE_NAME_RULE = (
    "Use lowercase letters, numbers, '-' or '_', starting with a letter or number, "
    "up to 64 characters"
)


def _suggest_profile_name(name: str) -> str:
    """Best-effort valid id derived from *name* (``'My Work'`` -> ``'my-work'``); ``my-work`` if nothing usable."""
    candidate = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-_")[:64]
    return candidate if _PROFILE_ID_RE.match(candidate) else "my-work"


def _invalid_profile_name_error(name: str) -> ValueError:
    suggestion = _suggest_profile_name(name)
    return ValueError(
        f"{name!r} is not a valid profile name. {_PROFILE_NAME_RULE} (for example: {suggestion}). "
        f"Then run `hermes profile create {suggestion}`."
    )


# Validation

def normalize_profile_name(name: str) -> str:
    """Canonical profile id used on disk and in ``-p`` argv: lowercase, ``default`` matched
    case-insensitively. Dashboards/tools may pass title-cased labels — normalize before
    validation, assignment, and subprocess spawn.

    Named profiles are stored lowercase under ``profiles/<id>/``. See #18498.
    """
    if not isinstance(name, str):
        name = str(name)
    stripped = name.strip()
    if not stripped:
        raise ValueError("profile name cannot be empty")
    if stripped.casefold() == "default":
        return "default"
    return stripped.lower()


def validate_profile_name(name: str) -> None:
    """Raise ``ValueError`` unless *name* is a valid profile id (strict as-given lowercase —
    normalize mixed-case input first) and not in ``_RESERVED_NAMES``; ``default`` passes.

    Callers that accept mixed-case or title-cased input from users (dashboard UI, CLI args) should call
    :func:`normalize_profile_name` first. This separation keeps validate honest about what the on-disk
    directory name must look like, while ingress-point normalization handles UX flexibility (see #18498).
    """
    if name == "default":
        return  # special alias for ~/.hermes
    if not _PROFILE_ID_RE.match(name):
        raise _invalid_profile_name_error(name)
    if name in _RESERVED_NAMES:
        raise ValueError(
            f"Profile name {name!r} is reserved — it collides with either "
            f"the Hermes installation itself or a common system binary.  "
            f"Pick a different name."
        )


def validate_alias_name(name: str) -> None:
    """Raise ``ValueError`` unless *name* is a safe wrapper filename: it is used verbatim
    under ``~/.local/bin``, so ``../../.bashrc`` must never escape the wrapper dir."""
    if not _PROFILE_ID_RE.match(name):
        raise ValueError(f"Invalid alias name {name!r}. {_PROFILE_NAME_RULE}.")


def _canon_valid(name: str) -> str:
    """normalize + validate in one step; returns the canonical id."""
    canon = normalize_profile_name(name)
    validate_profile_name(canon)
    return canon


def _existing_profile_dir(name: str) -> Tuple[str, Path]:
    """``(canon, profile_dir)`` for an existing profile; FileNotFoundError otherwise."""
    canon = _canon_valid(name)
    profile_dir = get_profile_dir(canon)
    if not profile_dir.is_dir():
        raise _unknown_profile_error(canon)
    return canon, profile_dir


def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its HERMES_HOME directory."""
    canon = normalize_profile_name(name)
    if canon == "default":
        return _get_default_hermes_home()
    # The name becomes a path component under profiles/; refuse anything that
    # is not a valid profile id so every caller (WS params, /p/<profile>/
    # prefixes, tool args) fails closed instead of escaping the root. The
    # regex only, not _RESERVED_NAMES: a pre-reserved-list dir like
    # profiles/hermes may still exist and must keep resolving.
    if not _PROFILE_ID_RE.match(canon):
        raise _invalid_profile_name_error(canon)
    return _get_profiles_root() / canon


def profile_exists(name: str) -> bool:
    """Check whether a live (non-tombstoned) profile directory exists."""
    try:
        canon = normalize_profile_name(name)
        profile_dir = get_profile_dir(canon)
    except ValueError:
        return False
    if canon == "default":
        return True
    return named_profile_is_live(profile_dir)


def profile_matches_home(name: str, home: "Path | None" = None) -> bool:
    """True when *name* refers to the profile served from *home* (default: current home).

    Lets single-profile gateways decide whether a ``/p/<profile>/`` URL prefix is
    self-referential (safe on the bare route) or names a different profile, which must fail
    closed rather than silently resolve the owner's config. Invalid names return False."""
    try:
        target = get_profile_dir(name)
        if home is None:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        return Path(target).expanduser().resolve(strict=False) == Path(home).expanduser().resolve(strict=False)
    except Exception:
        return False


def _iter_named_profile_dirs(*, live_only: bool = True) -> List[Path]:
    """Sorted named-profile dirs (valid ids, never ``default``); ``live_only`` skips tombstones.

    A dir is a profile only when it carries an identity marker (``named_profile_has_identity``):
    cron/logging side-effects and pre-tombstone ghost shells leave marker-less dirs that must
    not be listed, served, ticked, or ``.env``-seeded — that seeding is what turned a ghost
    shell into a "real" profile on the next ``hermes update`` (#95188, #94823, #99392)."""
    profiles_root = _get_profiles_root()
    if not profiles_root.is_dir():
        return []
    return [
        entry for entry in sorted(profiles_root.iterdir())
        if entry.is_dir()
        and entry.name != "default"
        and _PROFILE_ID_RE.match(entry.name)
        and named_profile_has_identity(entry)
        and not (live_only and named_profile_is_deleted(entry))
    ]


def list_profile_names() -> List[str]:
    """Cheap name-only listing (``default`` + LIVE profile dirs). Unlike :func:`list_profiles` this
    reads NO per-profile config — safe for hot paths (cron target listings, create validation).
    Tombstoned shells are skipped like everywhere else: a stale process that re-mkdirs a deleted
    profile's directory must not resurface it as a ``bot-chat:<name>`` cron target."""
    names = ["default"]
    with contextlib.suppress(OSError):
        names.extend(entry.name for entry in _iter_named_profile_dirs())
    return names


# Alias / wrapper script management

def check_alias_collision(name: str) -> Optional[str]:
    """Return a human-readable collision message, or None if the name is safe."""
    canon = normalize_profile_name(name)
    try:
        validate_alias_name(canon)
    except ValueError as exc:
        return str(exc)
    if canon in _RESERVED_NAMES:
        return f"'{canon}' is a reserved name"
    if canon in _HERMES_SUBCOMMANDS:
        return f"'{canon}' conflicts with a hermes subcommand"
    try:
        result = subprocess.run(
            ["where" if sys.platform == "win32" else "which", canon],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=5,
        )
        if result.returncode == 0:
            existing_path = result.stdout.strip().splitlines()[0]
            expected = _wrapper_path(canon)
            if existing_path == str(expected) and _is_our_wrapper(expected):
                return None  # our own wrapper, safe to overwrite
            return f"'{canon}' conflicts with an existing command ({existing_path})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None  # safe


def _is_wrapper_dir_in_path() -> bool:
    return str(_get_wrapper_dir()) in os.environ.get("PATH", "").split(os.pathsep)


def create_wrapper_script(name: str, target: Optional[str] = None) -> Optional[Path]:
    """Create ``~/.local/bin/<name>`` activating profile *target* (default: *name*), so a
    custom alias can point at a differently-named profile without a post-hoc rewrite."""
    canon = normalize_profile_name(name)
    profile = normalize_profile_name(target) if target else canon
    validate_alias_name(canon)  # alias is a verbatim filename: no traversal
    wrapper_dir = _get_wrapper_dir()
    try:
        wrapper_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠ Could not create {wrapper_dir}: {e}")
        return None
    wrapper_path = _wrapper_path(canon)
    try:
        if sys.platform == "win32":
            wrapper_path.write_text(f"@echo off\r\nhermes -p {profile} %*\r\n", encoding="utf-8")
        else:
            hermes_exe = shutil.which("hermes") or "hermes"
            wrapper_path.write_text(f'#!/bin/sh\nexec {shlex.quote(hermes_exe)} -p {profile} "$@"\n', encoding="utf-8")
            wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return wrapper_path
    except OSError as e:
        print(f"⚠ Could not create wrapper at {wrapper_path}: {e}")
        return None


def remove_wrapper_script(name: str) -> bool:
    """Remove the wrapper script for a profile. Returns True if removed."""
    canon = normalize_profile_name(name)
    # A traversal-shaped name could point unlink() outside the wrapper dir; refuse it.
    try:
        validate_alias_name(canon)
    except ValueError:
        return False

    # Both the extensionless path (POSIX) and .bat (Windows)
    candidates = [_get_wrapper_dir() / canon]
    if sys.platform == "win32":
        candidates.insert(0, _get_wrapper_dir() / f"{canon}.bat")
    for wrapper_path in candidates:
        if wrapper_path.exists() and _is_our_wrapper(wrapper_path):
            with contextlib.suppress(Exception):
                wrapper_path.unlink()
                return True
    return False


def _migrate_profile_config_if_outdated(profile_dir: Path) -> None:
    """Migrate a copied config.yaml to the current schema (non-interactive, scoped to the new
    profile); otherwise the first desktop/doctor view shows a scary ``v0 -> latest`` warning."""
    from hermes_cli.config_backend import config_exists
    if not config_exists(profile_dir / "config.yaml"):
        return
    # Creation must not fail over an unmigratable old config; `hermes doctor --fix` surfaces
    # the detailed error in the target profile.
    with contextlib.suppress(Exception):
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from hermes_cli.config import check_config_version, migrate_config
        token = set_hermes_home_override(str(profile_dir))
        try:
            current_ver, latest_ver = check_config_version()
            if current_ver < latest_ver:
                migrate_config(interactive=False, quiet=True)
        finally:
            reset_hermes_home_override(token)


def find_alias_for_profile(profile_name: str) -> Optional[str]:
    """Alias name of the wrapper activating *profile_name*, or None. For listing ALL profiles
    prefer :func:`build_alias_map`: per-profile calls re-read every wrapper N times (O(N*M)),
    which on a ``~/.local/bin`` full of large binaries meant multi-second ``list_profiles``."""
    return build_alias_map().get(normalize_profile_name(profile_name))


# Cap on how much of a wrapper file is read when reverse-looking-up its profile. Real
# wrappers are a few hundred bytes with the ``hermes -p X`` needle near the top; the wrapper
# dir commonly also holds large binaries (ffmpeg, node, …) whose whole-file reads, N times,
# dominated ``list_profiles`` (~4.5s).
_WRAPPER_READ_LIMIT = 8192


def build_alias_map() -> dict[str, str]:
    """Single-pass reverse map ``{canonical_profile -> alias_name}``.

    Scans the wrapper dir ONCE, reading only a head slice of each candidate and skipping
    binaries. A custom alias (file name != profile) wins over the profile-named wrapper;
    deterministic via sorted iteration."""
    wrapper_dir = _get_wrapper_dir()
    result: dict[str, str] = {}
    if not wrapper_dir.is_dir():
        return result
    is_windows = sys.platform == "win32"
    prefix = "hermes -p "
    for entry in sorted(wrapper_dir.iterdir()):
        if not entry.is_file():
            continue
        # Our wrappers are named after the alias and (on Windows only) carry .bat.
        if is_windows and entry.suffix != ".bat":
            continue
        if not is_windows and entry.suffix:
            continue
        try:
            with open(entry, "r", encoding="utf-8-sig", errors="strict") as f:
                content = f.read(_WRAPPER_READ_LIMIT)
        except (OSError, UnicodeDecodeError):
            continue  # UnicodeDecodeError = a binary on PATH, not a wrapper
        idx = content.find(prefix)
        if idx == -1:
            continue
        rest = content[idx + len(prefix):]
        # Profile id is the first whitespace-delimited token after the flag.
        canon = rest.split(None, 1)[0].strip() if rest.strip() else ""
        if not canon:
            continue
        canon = normalize_profile_name(canon)
        alias = entry.stem if is_windows else entry.name
        if alias == canon:
            result.setdefault(canon, alias)  # never overwrite a custom alias already found
        else:
            result[canon] = alias
    return result


# ProfileInfo

@dataclass
class ProfileInfo:
    """Summary information about a profile."""
    name: str
    path: Path
    is_default: bool
    gateway_running: bool
    model: Optional[str] = None
    provider: Optional[str] = None
    has_env: bool = False
    skill_count: int = 0
    alias_path: Optional[Path] = None
    # Custom alias (wrapper file name) when it differs from ``name``; ``name`` when a
    # profile-named wrapper exists; None if no wrapper points here.
    alias_name: Optional[str] = None
    # Distribution metadata (None if the profile wasn't installed from a distribution).
    distribution_name: Optional[str] = None
    distribution_version: Optional[str] = None
    distribution_source: Optional[str] = None
    # 1-2 sentence role description from ``profile.yaml``; empty when never described.
    # Surfaced to the kanban decomposer so it routes work by role rather than name.
    description: str = ""
    # True when ``description`` was LLM-generated and not yet user-confirmed (dashboard
    # shows a "review" badge).
    description_auto: bool = False
    # Presentation-only display name; resolution/comparison/spawn always use ``name``.
    display_name: str = ""
    # Bot Mode title (``profile.yaml`` ``ui_meta['hermes-bots'].title``) — the name
    # the Bots roster shows. Presentation-only, like ``display_name``.
    bot_title: str = ""
    # Canonical ids this profile was previously known by (``hermes profile rename``
    # appends here). Lets Bot Mode group chats re-link persisted member
    # descriptors to the renamed live profile (#110200).
    previous_names: List[str] = field(default_factory=list)
    # Backend-assigned role (``SETUP_ROLE`` or None). Only ``hermes_cli.setup_profile`` writes it.
    role: Optional[str] = None


def _load_yaml_dict(path: Path) -> Optional[dict]:
    """Return the mapping in a YAML file, or None when missing/unreadable/not a mapping."""
    if not path.is_file():
        return None
    try:
        import hermes_yaml as yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _load_config_dict(profile_dir: Path) -> Optional[dict]:
    """:func:`_load_yaml_dict` for a profile's config.yaml, read through the config backend."""
    from hermes_cli.config_backend import config_exists, read_config_doc
    path = profile_dir / "config.yaml"
    if not config_exists(path):
        return None
    try:
        data = read_config_doc(path) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


# (path, kind) -> (file signature, the small derived value). `list_profiles` re-reads three YAML
# files PER PROFILE, and it is the shared body of `GET /api/profiles` and `profiles.list`, which the
# Bots roster polls every 5s per connection — so an installer-seeded config.yaml (the annotated
# template, ~119KB) was re-parsed for every bot every five seconds to yield the same two strings.
# Only DERIVED values are cached, never a document a caller could write back: the raw readers
# (`read_user_config_raw`, `_load_yaml_dict`) keep their uncached contract. See #117378.
_PROFILE_FILE_CACHE: Dict[tuple, tuple] = {}
_PROFILE_FILE_CACHE_MAX = 512


def _profile_file_signature(path: Path) -> Optional[tuple]:
    """``(mtime_ns, size, inode)``, or None when the file is absent. The atomic writers rename a
    temp file into place, so a rewrite always lands a new inode even within one mtime tick."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _cached_profile_read(path: Path, kind: str, compute):
    """``compute()``'s value, reused while *path* has not changed. A missing file is never cached:
    reading it costs nothing, and one created later must be picked up."""
    signature = _profile_file_signature(path)
    if signature is None:
        return compute()
    key = (str(path), kind)
    cached = _PROFILE_FILE_CACHE.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = compute()
    if len(_PROFILE_FILE_CACHE) >= _PROFILE_FILE_CACHE_MAX:
        _PROFILE_FILE_CACHE.clear()
    _PROFILE_FILE_CACHE[key] = (signature, value)
    return value


def _read_distribution_meta(profile_dir: Path) -> tuple:
    """``(name, version, source)`` from ``distribution.yaml``; ``(None, None, None)`` if absent."""
    def _read() -> tuple:
        data = _load_yaml_dict(profile_dir / "distribution.yaml")
        if data is None:
            return None, None, None
        return data.get("name"), data.get("version"), data.get("source")

    return _cached_profile_read(profile_dir / "distribution.yaml", "distribution", _read)


def _read_config_model(profile_dir: Path) -> tuple:
    """Read model/provider from a profile's config.yaml. Returns (model, provider)."""
    from hermes_cli.config_backend import config_exists
    config_path = profile_dir / "config.yaml"
    if not config_exists(config_path):
        return None, None

    def _read() -> tuple:
        try:
            # load_config() targets the ACTIVE profile's home; read THIS profile's file raw.
            from hermes_cli.config import read_user_config_raw
            model_cfg = read_user_config_raw(config_path).get("model", {})
            if isinstance(model_cfg, str):
                return model_cfg, None
            if isinstance(model_cfg, dict):
                return model_cfg.get("default") or model_cfg.get("model"), model_cfg.get("provider")
        except Exception:
            pass
        return None, None

    return _cached_profile_read(config_path, "config-model", _read)


def launch_model_seed(source_cfg: dict) -> dict:
    """The config a fresh profile needs to run the launch profile's model: its ``model`` block plus,
    when that block points at a custom ``providers:`` gateway (self-hosted / local endpoint), that
    provider's definition — ``model.provider: my-gateway`` alone is "Unknown provider" on the first
    turn. ``{}`` when the launch profile has no model."""
    model_cfg = source_cfg.get("model")
    if not model_cfg:
        return {}
    seed = {"model": model_cfg}
    providers = source_cfg.get("providers")
    name = model_cfg.get("provider") if isinstance(model_cfg, dict) else None
    if isinstance(providers, dict) and name in providers:
        seed["providers"] = {name: providers[name]}
    return seed


def _seed_model_config(profile_dir: Path) -> None:
    """Copy (not link) the active profile's model block into a fresh profile so it is usable;
    profiles stay independent islands afterwards."""
    from hermes_cli.config_backend import config_exists
    config_path = profile_dir / "config.yaml"
    if config_exists(config_path):
        return
    with contextlib.suppress(Exception):  # creation must not fail over this; `hermes model` sets it later
        from hermes_constants import get_hermes_home
        from hermes_cli.config import atomic_config_write, read_user_config_raw
        source = get_hermes_home() / "config.yaml"
        seed = launch_model_seed(read_user_config_raw(source)) if config_exists(source) else {}
        if seed:
            atomic_config_write(config_path, seed)


def _check_gateway_running(profile_dir: Path) -> bool:
    """Gateway liveness for a profile dir, never mutating HERMES_HOME.

    Primary signal is ``gateway.pid`` verified against the runtime lock (fails closed when
    the lock isn't held by *this* reader: dashboard as a separate s6 service, launch-service
    gateways with no live PID file); fallback validates the PID in ``gateway_state.json``
    against the process table, matching ``/api/status``."""
    from gateway.status import get_running_pid, resolve_gateway_liveness
    # cleanup_stale=False: a status probe for ANOTHER profile must never unlink its PID file.
    return resolve_gateway_liveness(
        profile_dir=profile_dir, use_cache=False,
        pid_probe=lambda path: get_running_pid(path, cleanup_stale=False)).running


def _served_by_running_multiplexer(profile_name: str) -> bool:
    """True when the live default gateway multiplexes ``profile_name`` (such a profile has no
    gateway.pid of its own, so ``_check_gateway_running`` alone reports it stopped).

    Single shared lookup with the named-profile start guard and cron liveness (#97120).
    """
    try:
        from hermes_cli.gateway import named_profile_served_by_running_multiplexer
        return named_profile_served_by_running_multiplexer(profile_name)
    except Exception:
        return False


# In-process skill-count cache. Counting walks every skill's sub-tree (~4 fs calls per
# skill); ``list_profiles`` counts EVERY profile, and its two remaining Desktop callers
# (``GET /api/profiles``, ``profiles.list``) are POLLED every few seconds. Keyed by skills
# dir; a walk is repeated only when the tree signature changes (skill add/remove) or after
# the TTL (deep edits). Polled callers never walk: ``lazy_skill_count`` serves the last known
# value and refreshes stale entries on a background thread, at most once per recheck window
# per profile — the TTL is decoupled from the poll rate (#114041).
_SKILL_COUNT_CACHE: dict[str, tuple[float, float, int]] = {}
_SKILL_COUNT_TTL_SECONDS = 600.0
_SKILL_COUNT_RECHECK_SECONDS = 60.0
_SKILL_COUNT_NEXT_CHECK: dict[str, float] = {}
_SKILL_COUNT_LOCK = threading.Lock()
# One scan lock per skills dir so concurrent cold scans of the SAME profile share one walk
# while different profiles' scans still run in parallel.
_SKILL_COUNT_SCAN_LOCKS: dict[str, threading.Lock] = {}


def _skills_dir_signature(skills_dir: Path) -> float:
    """Max mtime of ``skills_dir`` and its immediate children (adding/removing a category
    bumps the root, a skill bumps its category). One scandir: O(#categories), not O(#files)."""
    try:
        sig = skills_dir.stat().st_mtime
    except OSError:
        return 0.0
    try:
        with os.scandir(skills_dir) as it:
            for entry in it:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        sig = max(sig, entry.stat(follow_symlinks=False).st_mtime)
                except OSError:
                    continue
    except OSError:
        pass
    return sig


def _walk_skill_count(skills_dir: Path) -> int:
    """One ``os.walk`` over the skills tree (prunes ``.git``/``node_modules``/support dirs
    instead of statting them). Best-effort: a subtree that vanishes mid-walk (a concurrent
    skill install/update) is skipped, never raised — ``os.walk`` (``onerror=None``) swallows
    scandir errors itself, so one profile's churn cannot abort the whole enumeration."""
    from agent.skill_utils import iter_skill_index_files
    return sum(1 for _ in iter_skill_index_files(skills_dir, "SKILL.md"))


def _count_skills(profile_dir: Path) -> int:
    """Count installed skills in a profile (cached by skills-dir signature + TTL). Walks
    synchronously when stale — detail surfaces (``hermes profile info``, ``profiles.describe``)
    want the fresh number; polled lists go through :func:`_cached_skill_count`."""
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return 0
    key = str(skills_dir)
    signature = _skills_dir_signature(skills_dir)
    cached = _SKILL_COUNT_CACHE.get(key)
    if cached is not None and cached[0] == signature and (time.time() - cached[1]) < _SKILL_COUNT_TTL_SECONDS:
        return cached[2]
    with _SKILL_COUNT_LOCK:
        scan_lock = _SKILL_COUNT_SCAN_LOCKS.setdefault(key, threading.Lock())
    with scan_lock:
        # Re-check under the lock: a concurrent caller may have just finished this walk.
        signature = _skills_dir_signature(skills_dir)
        cached = _SKILL_COUNT_CACHE.get(key)
        if cached is not None and cached[0] == signature and (time.time() - cached[1]) < _SKILL_COUNT_TTL_SECONDS:
            return cached[2]
        count = _walk_skill_count(skills_dir)
        # Stamp AFTER the walk: a scan longer than the TTL must not publish an already-expired entry.
        _SKILL_COUNT_CACHE[key] = (signature, time.time(), count)
        return count


def _cached_skill_count(profile_dir: Path) -> int:
    """Last known skill count with ZERO skill-tree I/O on the calling thread. A never-counted
    or aged entry schedules one background :func:`_count_skills` per profile per recheck
    window (which itself re-walks only on signature change / TTL); the next poll picks the
    result up. ``0`` until the first refresh lands."""
    key = str(profile_dir / "skills")
    now = time.time()
    with _SKILL_COUNT_LOCK:
        due = _SKILL_COUNT_NEXT_CHECK.get(key, 0.0) <= now
        if due:
            _SKILL_COUNT_NEXT_CHECK[key] = now + _SKILL_COUNT_RECHECK_SECONDS
    if due:
        threading.Thread(target=_count_skills, args=(profile_dir,),
                         name="hermes-skill-count", daemon=True).start()
    cached = _SKILL_COUNT_CACHE.get(key)
    return cached[2] if cached is not None else 0


# profile.yaml — per-profile metadata (description, role, etc.)
# Deliberately tiny and separate from ``config.yaml`` (user-facing Hermes config, ~5000
# lines of defaults): this is metadata ABOUT the profile. Missing file -> empty defaults,
# never an error; the kanban decomposer falls back to the profile name.


def read_profile_meta(profile_dir: Path) -> dict:
    """Read ``profile.yaml`` -> ``{description, description_auto, display_name,
    previous_names}`` (empty defaults when missing/unreadable). Never raises — a
    corrupt file on one profile must not break ``hermes profile list``."""
    def _read() -> dict:
        data = _load_yaml_dict(profile_dir / "profile.yaml") or {}
        ui_meta = data.get("ui_meta")
        bot_title = ""
        if isinstance(ui_meta, dict):
            hermes_bots = ui_meta.get("hermes-bots")
            if isinstance(hermes_bots, dict):
                bot_title = str(hermes_bots.get("title") or "").strip()
        return {
            "description": str(data.get("description") or "").strip(),
            "description_auto": bool(data.get("description_auto", False)),
            "display_name": str(data.get("display_name") or "").strip(),
            "bot_title": bot_title,
            "previous_names": _clean_previous_names(data.get("previous_names")),
            "role": data.get("role") if data.get("role") in PROFILE_ROLES else None,
        }

    # A copy per caller (list included): the cached value is shared, and a caller that mutates
    # its result must not poison the next reader.
    meta = dict(_cached_profile_read(profile_dir / "profile.yaml", "profile-meta", _read))
    meta["previous_names"] = list(meta["previous_names"])
    return meta


def _clean_previous_names(raw) -> List[str]:
    """Normalize the ``previous_names`` list from ``profile.yaml``: strings only,
    stripped, de-duplicated preserving order. Never raises."""
    if not isinstance(raw, list):
        return []
    cleaned: List[str] = []
    seen = set()
    for item in raw:
        name = str(item or "").strip()
        if name and name not in seen:
            seen.add(name)
            cleaned.append(name)
    return cleaned


def write_profile_meta(
    profile_dir: Path, *, description: Optional[str] = None, description_auto: Optional[bool] = None,
    display_name: Optional[str] = None, previous_names: Optional[List[str]] = None,
    role: Optional[str] = None,
) -> None:
    """Update ``profile.yaml`` in place: only passed fields are overwritten; the file is
    created if missing. The profile directory itself must exist. ``role`` grants backend
    capabilities, so no client-facing writer passes it through."""
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory does not exist: {profile_dir}")
    if role is not None and role not in PROFILE_ROLES:
        raise ValueError(f"unknown profile role: {role!r}")
    path = profile_dir / "profile.yaml"
    existing: dict = _load_yaml_dict(path) or {}
    if role is not None:
        existing["role"] = role
    if description is not None:
        existing["description"] = description.strip()
    if description_auto is not None:
        existing["description_auto"] = bool(description_auto)
    if display_name is not None:
        # Empty string clears the key (falls back to the canonical id).
        if display_name.strip():
            existing["display_name"] = display_name.strip()
        else:
            existing.pop("display_name", None)
    if previous_names is not None:
        # Rename history for group-chat member sync (#110200): consumers match
        # stale persisted handles against the live roster via these names.
        cleaned = _clean_previous_names(previous_names)
        if cleaned:
            existing["previous_names"] = cleaned
        else:
            existing.pop("previous_names", None)
    # Atomic write: bare open("w") truncates before the dump, and the read path swallows
    # parse errors as {}, so a crashed write would silently drop unspecified fields.
    # See #51356.
    from utils import atomic_yaml_write
    atomic_yaml_write(path, existing, sort_keys=False)


def drop_profile_role(profile_dir: Path) -> None:
    """Remove ``role`` from a copied ``profile.yaml``: a copy is an ordinary profile."""
    path = profile_dir / "profile.yaml"
    existing = _load_yaml_dict(path)
    if not existing or "role" not in existing:
        return
    existing.pop("role")
    from utils import atomic_yaml_write
    atomic_yaml_write(path, existing, sort_keys=False)


def format_profile_label(name: str, display_name: Optional[str]) -> str:
    """``display_name (canonical_id)``, or the bare id when no display name is set (or it
    equals the id) — byte-for-byte the pre-feature rendering."""
    dn = (display_name or "").strip()
    return f"{dn} ({name})" if dn and dn != name else name


def set_profile_display_name(profile_name: str, display_name: str) -> str:
    """Set (or clear, with ``""``) a presentation-only display name. Returns the stored value;
    raises ``ValueError`` over 64 chars."""
    canon, profile_dir = _existing_profile_dir(profile_name)
    cleaned = (display_name or "").strip()
    if len(cleaned) > 64:
        raise ValueError(f"Display name too long ({len(cleaned)} chars, max 64).")
    write_profile_meta(profile_dir, display_name=cleaned)
    return cleaned


# CRUD operations

def _profile_info(name: str, path: Path, *, is_default: bool, alias_name: Optional[str] = None,
                  lazy_skill_count: bool = False) -> ProfileInfo:
    """Build one :class:`ProfileInfo` from a profile directory."""
    model, provider = _read_config_model(path)
    dist_name, dist_version, dist_source = _read_distribution_meta(path)
    meta = read_profile_meta(path)
    alias_path = _wrapper_path(alias_name) if alias_name else None
    if alias_path is not None and not alias_path.exists():
        alias_path = None
    gateway_running = _check_gateway_running(path)
    if not is_default:
        gateway_running = gateway_running or _served_by_running_multiplexer(name)
    skill_count = _cached_skill_count(path) if lazy_skill_count else _count_skills(path)
    return ProfileInfo(
        name=name, path=path, is_default=is_default, gateway_running=gateway_running, model=model,
        provider=provider, has_env=(path / ".env").exists(), skill_count=skill_count,
        alias_path=alias_path, alias_name=alias_name, distribution_name=dist_name,
        distribution_version=dist_version, distribution_source=dist_source,
        **meta,
    )


def list_profiles(*, lazy_skill_count: bool = False) -> List[ProfileInfo]:
    """Return info for all profiles, including the default.

    ``lazy_skill_count=True`` is for POLLED callers (``GET /api/profiles``, ``profiles.list``):
    ``skill_count`` is the last known value, refreshed off-request, so the request never walks
    a skill tree (#114041). Default ``False`` counts synchronously (CLI, detail views)."""
    profiles = []
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        profiles.append(_profile_info("default", default_home, is_default=True,
                                      lazy_skill_count=lazy_skill_count))
    named = _iter_named_profile_dirs()
    if named:
        alias_map = build_alias_map()  # ONCE, not per profile (was the dominant cost)
        for entry in named:
            alias_name = alias_map.get(normalize_profile_name(entry.name))
            profiles.append(_profile_info(entry.name, entry, is_default=False, alias_name=alias_name,
                                          lazy_skill_count=lazy_skill_count))
    return profiles


#: One signature/result per home: the webhook and
#: api-server callers run :func:`profiles_to_serve` per inbound request, so the reader
#: must not re-parse a profile's config.yaml every time.
_STANDALONE_MEMO: Dict[str, Tuple[Optional[tuple], Optional[bool]]] = {}
_STANDALONE_WARNED = False

_STANDALONE_DEFAULT_WARNING = (
    "gateway.standalone is ignored on the default profile: it is the host gateway")


def profile_is_standalone(home: Path) -> bool:
    """Does ``home``'s own config.yaml opt this profile out of the host multiplexer
    (``gateway.standalone: true``)? Memoised by file signature. The DEFAULT profile is
    never standalone — it IS the host — and warns once per process if the key is set there."""
    global _STANDALONE_WARNED
    from hermes_yaml import YAMLError
    from hermes_cli.config_backend import config_version

    home = Path(home)
    cfg_path = home / "config.yaml"
    key = str(home)
    try:
        signature = config_version(cfg_path)
    except FileNotFoundError:
        signature = None
    except OSError as exc:
        signature = ("stat-error", exc.errno)
        if _STANDALONE_MEMO.get(key) != (signature, False):
            logger.warning("Cannot read gateway.standalone from %s (%s); treating as not standalone",
                           cfg_path, type(exc).__name__)
        _STANDALONE_MEMO[key] = (signature, False)
        return False
    cached = _STANDALONE_MEMO.get(key)
    if cached is not None and cached[0] == signature and cached[1] is not None:
        return cached[1]
    value = None
    if signature is not None:
        from hermes_cli.config import read_user_config_raw
        try:
            cfg = read_user_config_raw(cfg_path) or {}
        except (YAMLError, OSError, UnicodeError) as exc:
            if cached is None or cached[0] != signature:
                logger.warning("Cannot read gateway.standalone from %s (%s); treating as not standalone",
                               cfg_path, type(exc).__name__)
            # Access can recover without changing the signature. None retains the
            # warning receipt without caching a transient failure as config.
            _STANDALONE_MEMO[key] = (signature, False if isinstance(exc, YAMLError) else None)
            return False
        if isinstance(cfg.get("gateway"), dict):
            value = cfg["gateway"].get("standalone")
    result = _standalone_truthy(value)
    if home == _get_default_hermes_home():
        if result and not _STANDALONE_WARNED:
            logger.warning(_STANDALONE_DEFAULT_WARNING)
            _STANDALONE_WARNED = True
        result = False
    _STANDALONE_MEMO[key] = (signature, result)
    return result


def _standalone_truthy(value: object) -> bool:
    """``gateway.standalone`` truthiness via the shared bool parser; only the
    ``gateway:`` section's ``standalone`` key is read (no top-level alias)."""
    from gateway.config import _bool_token
    if isinstance(value, str):
        return _bool_token(value) is True
    return bool(value)


def parked_marker_path(home: Path) -> Path:
    return Path(home) / "gateway.parked"


def profile_is_parked(home: Path) -> bool:
    """Marker contents are deliberately irrelevant, including for provisioning."""
    return parked_marker_path(home).exists()


_parked_default_warned: set[Path] = set()


def profiles_to_serve(multiplex: bool, *, include_standalone: bool = False,
                      include_parked: bool = False) -> List[Tuple[str, Path]]:
    """``(profile_name, hermes_home)`` pairs a gateway should serve — the single chokepoint
    for "which profiles does the inbound gateway handle".

    ``multiplex=False``: exactly one entry for the *active* profile (byte-for-byte the
    historical single-profile behavior; name is ``"default"`` or the named profile's id).
    ``multiplex=True``: default plus every live named profile under ``profiles/`` (tombstoned
    and parked profiles skipped). Pure directory read: never creates a profile dir (#94590).

    Named profiles that authored ``gateway.standalone: true`` are skipped because they opted
    out of the host multiplexer; a ``gateway.parked`` marker (``hermes -p X gateway stop``) skips
    a profile the host would otherwise serve. Callers enumerating INSTALLED profiles pass
    ``include_standalone=True, include_parked=True``; serving/ticking callers pass neither."""
    active = get_active_profile_name() or "default"
    default = _get_default_hermes_home()
    if profile_is_parked(default) and default not in _parked_default_warned:
        logger.warning("Ignoring gateway.parked for the default profile; stop the host gateway instead")
        _parked_default_warned.add(default)
    if not multiplex:
        return [(active, get_profile_dir(active))]
    serve: List[Tuple[str, Path]] = [("default", default)]
    serve.extend((entry.name, entry) for entry in _iter_named_profile_dirs()
                 if (include_standalone or not profile_is_standalone(entry))
                 and (include_parked or not profile_is_parked(entry)))
    return serve


def _resolve_clone_source(clone_from: Optional[str]) -> Path:
    """Directory to clone from: the named profile, or the active profile when ``None``."""
    from hermes_cli.config_backend import require_file_tooling
    require_file_tooling("Profile clone")
    if clone_from is None:
        from hermes_constants import get_hermes_home
        source_dir = get_hermes_home()
    else:
        clone_from = _canon_valid(clone_from)
        source_dir = get_profile_dir(clone_from)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source profile '{clone_from or 'active'}' does not exist at {source_dir}")
    return source_dir


def _seed_file_if_missing(path: Path, text: str, mode: Optional[int] = None) -> None:
    """Best-effort: write *text* to *path* unless it already exists; never raises."""
    if path.exists():
        return
    with contextlib.suppress(OSError):
        path.write_text(text, encoding="utf-8")
        if mode is not None:
            os.chmod(str(path), mode)


def _clone_file(source_dir: Path, profile_dir: Path, relpath: str) -> None:
    """Copy one profile-relative file if it exists. ``.env`` is tightened to owner-only:
    ``copy2`` preserves source mode bits, so a loose source (umask 0o644) would leak."""
    src = source_dir / relpath
    if not src.exists():
        return
    dst = profile_dir / relpath
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if relpath == ".env":
        with contextlib.suppress(OSError):
            os.chmod(str(dst), 0o600)


# Files a clone edits in place after copying. A ``--clone-all`` copy preserves symlinks
# (``symlinks=True``), so a symlinked source ``.env`` would otherwise be edited THROUGH the link and
# the channel stripping would mutate the SOURCE profile. These are materialized as real files first.
_CLONE_MATERIALIZE = (".env", "config.yaml", "auth.json", "SOUL.md")


def _materialize_symlinked_files(profile_dir: Path) -> List[str]:
    """Replace symlinked root files the clone will edit with private copies of their targets (a
    dangling link is dropped). Returns the relative names materialized."""
    done: List[str] = []
    for name in _CLONE_MATERIALIZE:
        path = profile_dir / name
        if not path.is_symlink():
            continue
        target = Path(os.path.realpath(path))
        path.unlink()
        if target.is_file():
            shutil.copy2(target, path)
        done.append(name)
    return done


def _junction_target(path: str) -> Optional[str]:
    """Target of an NTFS directory junction, else ``None``. A junction is a reparse point, not a
    symlink: ``os.path.islink()`` is False and ``shutil.copytree(symlinks=True)`` descends into it."""
    if os.name != "nt":
        return None
    try:
        if os.lstat(path).st_reparse_tag != stat.IO_REPARSE_TAG_MOUNT_POINT:
            return None
        target = os.readlink(path)
    except OSError:
        return None
    # readlink hands back the substitute name; CreateJunction rejects the ``\\?\`` spelling.
    if target.startswith("\\\\?\\UNC\\"):
        return "\\" + target[7:]
    return target[4:] if target.startswith("\\\\?\\") else target


def _copytree_keep_junctions(src: Path, dst: Path, ignore, dirs_exist_ok: bool = False) -> None:
    """``shutil.copytree(symlinks=True)`` that re-creates NTFS junctions as junctions instead of
    traversing them. A ``skills/foo`` junction into a ``skills.external_dirs`` root copied as a
    physical tree is a second same-named candidate and ``_locate_skill`` refuses to guess (#113471).
    A junction whose target is gone is skipped with a warning, never a crash."""
    junctions: Dict[str, str] = {}

    def _ignore(directory: str, names: List[str]) -> set:
        ignored = set(ignore(directory, names))
        for name in names:
            target = _junction_target(os.path.join(directory, name))
            if target is not None:
                junctions[os.path.join(directory, name)] = target
                ignored.add(name)
        return ignored

    shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=dirs_exist_ok, ignore=_ignore)
    if junctions:
        import _winapi  # Windows-only stdlib module; only reachable once a junction was seen
    for link, target in junctions.items():
        try:
            _winapi.CreateJunction(target, os.path.join(dst, os.path.relpath(link, src)))
        except OSError as exc:
            logger.warning("clone: skipped junction %s -> %s (%s)", link, target, exc)


def _clone_all_into(source_dir: Path, profile_dir: Path, canon: str) -> None:
    """--clone-all: full copytree minus infrastructure/history, then strip runtime files,
    the backend-assigned role, and cloned single-use OAuth grants."""
    _copytree_keep_junctions(source_dir, profile_dir, _clone_all_copytree_ignore(source_dir))
    drop_profile_role(profile_dir)
    materialized = _materialize_symlinked_files(profile_dir)
    if materialized:
        logger.info("profile %s: materialized symlinked %s so the clone never writes through to %s",
                    canon, materialized, source_dir)
    # Excluded history dirs (sessions/, cron/) must still exist as empty dirs so the clone runs.
    for subdir in _PROFILE_DIRS:
        (profile_dir / subdir).mkdir(parents=True, exist_ok=True)
    for stale in _CLONE_ALL_STRIP:
        (profile_dir / stale).unlink(missing_ok=True)
    # auth.json / .anthropic_oauth.json copied verbatim fork single-use OAuth grants
    # (Anthropic / Codex / xAI): one credential with two owners, and the first profile to
    # refresh revokes the pair for every sibling. Drop the copies; the clone reads the root
    # grant through the credential-pool fallback.
    from hermes_cli.auth import strip_cloned_single_use_oauth_grants
    stripped = strip_cloned_single_use_oauth_grants(profile_dir)
    if any(stripped.values()):
        logger.info(
            "profile %s: dropped cloned single-use OAuth grants %s "
            "(inherits the root grant instead)", canon, stripped,
        )


def _bootstrap_profile_dir(profile_dir: Path, source_dir: Optional[Path],
                           sync_imports: bool = False) -> None:
    """Fresh layout: bootstrap dirs, then either seed a model block (no source) or clone
    config files, installed skills (the dashboard's "clone from default" must keep bundled
    AND user-installed skills), and memory/identity files from *source_dir*.

    ``sync_imports`` also copies the source's ``import-sync.json`` (the ``hermes import-agent``
    manifest) so the clone stays registered against the same external Claude Code / Codex trees
    and ``hermes -p <clone> import-agent --sync`` keeps pulling from them. The link is to the
    external tree, never to the source profile: both profiles stay independent islands."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    for subdir in _PROFILE_DIRS:
        (profile_dir / subdir).mkdir(parents=True, exist_ok=True)
    if source_dir is None:
        _seed_model_config(profile_dir)
        return
    for relpath in _CLONE_CONFIG_FILES:
        _clone_file(source_dir, profile_dir, relpath)
    source_skills = source_dir / "skills"
    if source_skills.is_dir():
        _copytree_keep_junctions(source_skills, profile_dir / "skills", _non_exportable_entries, dirs_exist_ok=True)
    for relpath in _CLONE_SUBDIR_FILES:
        _clone_file(source_dir, profile_dir, relpath)
    from hermes_cli.profile_memory_config import active_memory_provider, clone_memory_provider_config
    clone_memory_provider_config(source_dir, profile_dir, active_memory_provider(_load_config_dict(source_dir)))
    if sync_imports:
        from hermes_cli.agent_import_sync import SYNC_MANIFEST_NAME  # lazy: keeps yaml/utils off the hot startup path
        _clone_file(source_dir, profile_dir, SYNC_MANIFEST_NAME)


def create_profile(
    name: str, clone_from: Optional[str] = None, clone_all: bool = False, clone_config: bool = False,
    no_alias: bool = False, no_skills: bool = False, description: Optional[str] = None,
    clone_channels: bool = False, sync_imports: bool = False,
) -> Path:
    """Create a new profile directory and return its path.

    ``clone_from`` defaults to the active profile when cloning. ``clone_all`` copies all state;
    ``clone_config`` copies config.yaml/.env/SOUL.md, installed skills, and identity files.
    Either clone strips the source's messaging channels — bot tokens, allowlists, platform
    sections, pairing/session state — unless ``clone_channels`` opts in: a copied bot credential
    makes two gateways fight over one bot (``hermes_cli.profile_channels``; callers list what
    was left behind with ``channel_platforms_configured(source_dir)``).
    ``no_skills`` creates an empty profile and writes a marker so ``hermes update`` skips
    re-seeding its skills; it is mutually exclusive with the clone options, which copy skills.
    ``sync_imports`` (``--clone`` only; ``--clone-all`` copies the file anyway) also copies the
    ``import-agent`` sync manifest so the clone can keep pulling the same external agent trees."""
    if no_skills and (clone_from is not None or clone_config or clone_all):
        raise ValueError(
            "--no-skills is mutually exclusive with --clone / --clone-from / --clone-all "
            "(cloning explicitly copies skills from the source profile)."
        )
    if sync_imports and not (clone_config or clone_all):
        raise ValueError("--sync-imports requires --clone or --clone-from (there is no import "
                         "manifest to carry over without a source profile).")
    cloning = clone_from is not None or clone_all or clone_config
    if clone_channels and not cloning:
        raise ValueError("--clone-channels only applies to a clone (--clone, --clone-from or --clone-all).")
    canon = _canon_valid(name)
    if canon == "default":
        raise ValueError("Cannot create a profile named 'default' — it is the built-in profile (~/.hermes).")
    profile_dir = get_profile_dir(canon)
    if profile_dir.exists() and not named_profile_has_identity(profile_dir):
        if named_profile_is_deleted(profile_dir):
            # Empty shell left by a post-delete mkdir: invisible to ``profile list``, safe to replace.
            shutil.rmtree(profile_dir)
        else:
            # A live marker-less dir is invisible to ``profile list`` but may still hold user
            # files (skills/, memories/, cron/jobs.json): fail closed and name it, never rmtree.
            raise FileExistsError(
                f"Cannot create profile '{canon}': {profile_dir} exists but carries no profile identity "
                "file, so it is not listed as a profile. Move or remove that directory first."
            )
    if profile_dir.exists():
        raise _profile_exists_error(canon)
    source_dir = _resolve_clone_source(clone_from) if cloning else None
    if source_dir is not None and clone_channels:
        from hermes_cli.profile_channels import clone_channels_refusal
        refusal = clone_channels_refusal(source_dir, clone_from or get_active_profile_name() or "default")
        if refusal:
            raise ValueError(refusal)
    clear_named_profile_deleted(profile_dir)
    # Build in a hidden sibling and publish with one rename: a running multiplexer rescans profiles/
    # on every create and every 30 s, and ``_iter_named_profile_dirs`` only lists valid ids (no leading
    # dot), so it can never adopt the half-copied tree and start adapters on credentials the strip
    # below has not removed yet.
    staging = _clone_staging_dir(profile_dir)
    try:
        if clone_all and source_dir:
            _clone_all_into(source_dir, staging, canon)
        else:
            _bootstrap_profile_dir(staging, source_dir, sync_imports=sync_imports)
        if source_dir is not None and not clone_channels:
            from hermes_cli.profile_channels import strip_channel_settings
            stripped = strip_channel_settings(staging, include_state=clone_all, source_dir=source_dir)
            if stripped:
                logger.info("profile %s: cloned without messaging channels %s", canon, stripped)
        _finish_profile_layout(staging, no_skills=no_skills, clone_all=clone_all, description=description)
        os.rename(staging, profile_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Inside a container under s6, register the gateway as a runtime s6 service so
    # `hermes -p <profile> gateway start` supervises via `s6-svc -u` instead of a bare
    # process. No-op on host (systemd/launchd/windows unit generation handles lifecycle).
    _maybe_register_gateway_service(canon)
    # A running multiplexer enumerates profiles/ at boot: ask it to serve this one now (it also
    # rescans periodically, so a missed signal only delays serving).
    _notify_multiplexer(canon)
    return profile_dir


def _clone_staging_dir(profile_dir: Path) -> Path:
    """Fresh ``profiles/.<name>.staging-<pid>`` beside the final dir (same filesystem, so the publish
    rename is atomic). A leftover from a crashed create is discarded."""
    staging = profile_dir.parent / f".{profile_dir.name}.staging-{os.getpid()}"
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    if staging.is_symlink() or staging.is_file():
        staging.unlink()
    elif staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)
    return staging


def _finish_profile_layout(profile_dir: Path, *, no_skills: bool, clone_all: bool,
                           description: Optional[str]) -> None:
    """Seed files a fresh profile owns from day one; runs on the staging tree before publish."""
    # Seed an empty .env so the profile owns a credentials file from day one. Without it,
    # profile-scoped env writes (dashboard Channels/Keys pages, `hermes -p <name> auth add`)
    # had no file until first write and the profile silently inherited shell API keys —
    # read by users as "the new profile reads the root .env". Skipped when a clone copied one.
    _seed_file_if_missing(profile_dir / ".env", _PLACEHOLDER_ENV, 0o600)

    # Default SOUL.md to customize immediately (skipped when a clone already provided one).
    with contextlib.suppress(Exception):  # best-effort — don't fail profile creation over this
        from hermes_cli.default_soul import DEFAULT_SOUL_MD
        _seed_file_if_missing(profile_dir / "SOUL.md", DEFAULT_SOUL_MD)

    # Opt-out marker read by seed_profile_skills() and `hermes update`'s all-profile sync
    # (the feature still works via the empty skills/ dir if this fails).
    if no_skills:
        _seed_file_if_missing(
            profile_dir / NO_BUNDLED_SKILLS_MARKER,
            "This profile opted out of bundled-skill seeding (`hermes profile create --no-skills`).\n"
            "Delete this file to re-enable sync on the next `hermes update`.\n",
        )

    # Migrate config-only clones now so desktop/status don't warn that a just-created
    # profile is v0/outdated; --clone-all snapshots stay byte-for-byte apart from the
    # explicit runtime/history stripping above.
    if not clone_all:
        _migrate_profile_config_if_outdated(profile_dir)

    # Description last, so a partial-create failure doesn't strand a description file.
    if description and description.strip():
        with contextlib.suppress(Exception):  # non-fatal — `hermes profile describe` works later
            write_profile_meta(profile_dir, description=description.strip(), description_auto=False)


def _notify_multiplexer(canon: str) -> None:
    from hermes_cli.gateway_multiplex_served import notify_multiplexer_profiles_changed
    notify_multiplexer_profiles_changed(canon)


def _purge_identity(canon: str) -> bool:
    """Settle a deleted profile's durable identity (``profile_identity.purge_profile_identity``).

    False means the filesystem delete happened but the identity settlement did not — the caller
    reports that as a pending settlement rather than a clean success."""
    from hermes_cli.profile_identity import purge_profile_identity
    return purge_profile_identity(canon)


def _live_default_multiplexer() -> bool:
    """True when a live default gateway has recorded a served-profile set: every dir under
    profiles/ is then served by it, so a profile-identity change must be unrouted first."""
    try:
        from hermes_cli.gateway_multiplex_served import recorded_served_profiles
        return recorded_served_profiles() is not None
    except Exception:
        return False


def seed_profile_skills(profile_dir: Path, quiet: bool = False) -> Optional[dict]:
    """Seed bundled skills into a profile via subprocess (sync_skills() caches HERMES_HOME at
    module level). Returns the sync result dict, or None on failure. ``--no-skills`` profiles
    still run the sync: ``sync_skills()`` detects the marker and seeds only essentials."""
    project_root = Path(__file__).parent.parent.resolve()
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; from tools.skills_sync import sync_skills; "
             "r = sync_skills(quiet=True); print(json.dumps(r))"],
            env={**os.environ, "HERMES_HOME": str(profile_dir)},
            cwd=str(project_root),
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
        if not quiet:
            print(f"⚠ Skill seeding returned exit code {result.returncode}")
            if result.stderr.strip():
                print(f"  {result.stderr.strip()[:200]}")
        return None
    except subprocess.TimeoutExpired:
        if not quiet:
            print("⚠ Skill seeding timed out (60s)")
        return None
    except Exception as e:
        if not quiet:
            print(f"⚠ Skill seeding failed: {e}")
        return None


def backfill_profile_envs(quiet: bool = False) -> List[str]:
    """Give every named profile predating per-profile ``.env`` one (copy of the default's, or
    the placeholder header). Never overwrites an existing profile ``.env``.

    Profiles created before the dashboard/CLI started seeding a ``.env`` (PR #44792) have none, so once the
    Channels/Keys endpoints became profile-scoped those profiles stopped inheriting the root install's
    credentials and showed everything as unconfigured. To avoid breaking anyone on update, copy the DEFAULT
    install's ``.env`` into each named profile that lacks one — that preserves the effective credentials
    those profiles were already running with (they previously read the root ``.env`` via the process
    environment). Users can then diverge per profile from there.
    """
    backfilled: List[str] = []
    default_env = _get_default_hermes_home() / ".env"
    for entry in _iter_named_profile_dirs():
        env_path = entry / ".env"
        if env_path.exists():
            continue
        try:
            if default_env.is_file():
                shutil.copy2(default_env, env_path)
            else:
                env_path.write_text(_PLACEHOLDER_ENV, encoding="utf-8")
            os.chmod(str(env_path), 0o600)
            backfilled.append(entry.name)
        except OSError as e:
            if not quiet:
                print(f"⚠ Could not seed .env for profile '{entry.name}': {e}")
    return backfilled


_BACKEND_TOKENS = frozenset({"serve", "dashboard", "gateway"})
_HERMES_ARGV_MARKERS = ("hermes_cli.main", "hermes-gateway", "tui_gateway")
# python / python3 / python3.12 / pythonw(.exe): the interpreter basenames a
# `#!/…/python3` console-script shim is exec'd through when something (e.g. Electron's
# `findOnPath('hermes')`) spawns the shim by handing the interpreter its path — then the
# OS-reported argv[0] is the interpreter, not "hermes".
_PYTHON_INTERPRETER_RE = re.compile(r"^python[\d.]*w?(\.exe)?$")
# Console-script entry points this project ships (pyproject.toml [project.scripts]).
# argv[1] is matched against exact names, not ``startswith("hermes")``: with a bare
# interpreter argv[0], argv[1] can be ANY user script ("hermes-notes.py").
_HERMES_CONSOLE_SCRIPT_NAMES = frozenset({"hermes", "hermes-agent", "hermes-acp"})


def _is_hermes_argv(argv: list) -> bool:
    """True for a Hermes process: entrypoint marker in argv, executable named ``hermes*``,
    or a python interpreter directly exec'ing a known ``hermes`` console-script shim."""
    joined = " ".join(argv)
    exe_name = os.path.basename(argv[0]).lower()
    if any(marker in joined for marker in _HERMES_ARGV_MARKERS) or exe_name.startswith("hermes"):
        return True
    if len(argv) >= 2 and _PYTHON_INTERPRETER_RE.match(exe_name):
        script_name = os.path.basename(str(argv[1])).lower()
        return script_name.rsplit(".", 1)[0] in _HERMES_CONSOLE_SCRIPT_NAMES
    return False


def _argv_profile_selectors(argv: list):
    """Yield every profile name selected via ``-p X`` / ``--profile X`` / ``--profile=X``."""
    for i, tok in enumerate(argv):
        if tok in {"--profile", "-p"} and i + 1 < len(argv):
            yield argv[i + 1]
        elif tok.startswith("--profile="):
            yield tok.split("=", 1)[1]


def _profile_bound_backend_pids(canon: str, profile_dir: Path) -> list[int]:
    """PIDs of running Hermes *backends* bound to this profile (``gateway.pid`` only tracks
    the messaging gateway). Tightly scoped: current-user processes, backend subcommands only
    (never an interactive ``chat``/``tui``), never this process or its ancestors. Empty when
    ``psutil`` can't inspect anything."""
    try:
        import psutil  # type: ignore
    except Exception:
        return []
    try:
        resolved_dir = profile_dir.resolve()
    except OSError:
        resolved_dir = profile_dir

    # Never terminate ourselves or a parent (`hermes -p <canon> profile delete` runs under
    # the very profile it's deleting).
    skip: set[int] = {os.getpid()}
    with contextlib.suppress(Exception):
        parent = psutil.Process(os.getpid()).parent()
        while parent is not None:
            skip.add(parent.pid)
            parent = parent.parent()
    try:
        current_user = psutil.Process(os.getpid()).username()
    except Exception:
        current_user = None
    pids: list[int] = []
    for proc in psutil.process_iter(["pid", "name", "username", "cmdline"]):
        try:
            info = proc.info
            pid = info.get("pid")
            if pid is None or pid in skip:
                continue
            if current_user is not None and info.get("username") != current_user:
                continue
            argv = info.get("cmdline") or []
            if not argv or not _is_hermes_argv(argv):
                continue
            if not ({tok.lower() for tok in argv} & _BACKEND_TOKENS):
                continue

            # Bound to THIS profile by selector flag, or by HERMES_HOME pointing at its dir.
            bound = any(normalize_profile_name(sel) == canon for sel in _argv_profile_selectors(argv))
            if not bound:
                with contextlib.suppress(Exception):  # environ() can raise AccessDenied even same-user
                    env_home = (proc.environ() or {}).get("HERMES_HOME", "")
                    bound = bool(env_home) and Path(env_home).resolve() == resolved_dir
            if bound:
                pids.append(pid)
        except Exception:
            continue  # NoSuchProcess / AccessDenied / ZombieProcess and anything else
    return pids


def _wait_then_force_kill(pids: List[int], start_times: dict, *, wait: float = 10.0) -> bool:
    """After a graceful ``terminate_pid``, wait up to *wait* seconds (0.5s polls) for *pids*
    to exit, then force-kill stragglers. True when every pid exited gracefully.
    ``start_times`` pins each force kill to the same process incarnation (PID reuse guard)."""
    from gateway.status import _pid_exists, get_process_start_time, terminate_pid
    for _ in range(int(wait / 0.5)):
        time.sleep(0.5)
        if not any(_pid_exists(pid) for pid in pids):
            return True
    for pid in pids:
        if _pid_exists(pid):
            with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
                terminate_pid(pid, force=True, expected_start_time=start_times.get(pid, get_process_start_time(pid)))
    return False


def _stop_profile_backends(canon: str, profile_dir: Path) -> None:
    """Terminate Desktop-spawned / stray backends bound to this profile. Complements
    ``_stop_gateway_process`` (which only knows ``gateway.pid``): a live ``serve``/``dashboard``
    keeps creating files while ``rmtree`` walks, so the final rmdir fails ENOTEMPTY."""
    pids = _profile_bound_backend_pids(canon, profile_dir)
    if not pids:
        return
    try:
        from gateway.status import terminate_pid
    except Exception:
        return
    for pid in pids:
        try:
            terminate_pid(pid)  # graceful first
        except (ProcessLookupError, PermissionError, OSError):
            continue
    _wait_then_force_kill(pids, {})
    print(f"✓ Stopped {len(pids)} profile backend process(es)")


def _stop_bot_desktop(profile_dir: Path) -> None:
    """Stop the profile's Bot Desktop (Xvnc + Xfce launcher) before its directory is removed or renamed;
    gateway shutdown does not reach it (its own session, its own pid file). Scoped through the hermes-home
    override so the runtime reads THIS profile's bot-desktop/ state, whichever profile invoked the op.
    A failure here is logged, never fatal: the profile op is what the user asked for."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.bot_desktop import runtime
    if not runtime.is_supported_host():
        return
    token = set_hermes_home_override(profile_dir)
    try:
        if runtime.stop():
            print("✓ Bot Desktop stopped")
            # The screen a human's exclusion protected is gone; on rename the directory (lease.json
            # included) moves with the profile, and a human lease for a dead viewer would fence the
            # agent out of the renamed profile's next screen until someone force-released it.
            from tools.bot_desktop import lease
            lease.release()
    except Exception as e:
        logger.warning("Could not stop the Bot Desktop of %s: %s", profile_dir, e)
    finally:
        reset_hermes_home_override(token)


def _rmtree_make_writable(func, path, exc):
    """onexc/onerror handler: add +w on PermissionError so rmtree can proceed. Covers NixOS-
    style read-only copies where the path itself (0444) or its parent (0555) isn't writable."""
    # onexc(func, path, exc_instance) on 3.12+; onerror(func, path, exc_info_tuple) on 3.11.
    if isinstance(exc, tuple):
        exc = exc[1]
    if not isinstance(exc, PermissionError):
        raise
    for target in (path, os.path.dirname(path)):  # parent needed for unlink/rmdir
        if target:
            with contextlib.suppress(OSError):
                os.chmod(target, os.stat(target).st_mode | stat.S_IWUSR)
    func(path)


def _rmtree_with_retry(profile_dir: Path, onexc_handler) -> None:
    """``shutil.rmtree`` with a short retry loop: a just-terminated process can leave in-flight
    writes (SQLite -wal/-shm checkpoints, sandbox temp files) landing after rmtree walked
    past a directory — ENOTEMPTY on POSIX, transient PermissionError on Windows."""
    attempts = 3
    last_exc: OSError | None = None
    for attempt in range(attempts):
        try:
            try:
                shutil.rmtree(profile_dir, onexc=onexc_handler)
            except TypeError:  # ``onexc`` is 3.12+; 3.11 has ``onerror``
                shutil.rmtree(profile_dir, onerror=onexc_handler)
            return
        except OSError as e:
            last_exc = e
            if not profile_dir.exists():
                return
            if attempt < attempts - 1:
                time.sleep(0.3 * (attempt + 1))
    if last_exc is not None:
        raise last_exc


def _print_delete_summary(canon: str, profile_dir: Path, gw_running: bool, wrapper_path: Optional[Path]) -> None:
    """Show what ``delete_profile`` is about to remove."""
    model, provider = _read_config_model(profile_dir)
    skill_count = _count_skills(profile_dir)
    dist_name, dist_version, dist_source = _read_distribution_meta(profile_dir)
    print(f"\nProfile: {canon}")
    print(f"Path:    {profile_dir}")
    if model:
        print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
    if skill_count:
        print(f"Skills:  {skill_count}")
    if dist_name:
        print(f"Distribution: {dist_name}@{dist_version or '?'}")
        if dist_source:
            print(f"Installed from: {dist_source}")
    print("\nThis will permanently delete:")
    print("  • All config, API keys, memories, sessions, skills, cron jobs")
    if wrapper_path is not None:
        print(f"  • Command alias ({wrapper_path})")
    if gw_running:
        print("  ⚠ Gateway is running — it will be stopped.")


class ProfileIdentitySettlementPending(RuntimeError):
    """The profile's directory was deleted, but its durable session/routing identity was not
    settled (``hermes_cli.profile_identity.purge_profile_identity`` returned False).

    Subclasses ``RuntimeError`` so delete-failure handling that treats the error as fatal (the
    CLI's ``hermes profile delete``) keeps working unchanged; surfaces that can report a partial
    success (the dashboard's ``DELETE /api/profiles/{name}``) catch this type — it carries the
    profile, its now-removed path, and the retry command — instead of matching on the message.
    """

    def __init__(self, profile: str, path: Path):
        self.profile = profile
        self.path = path
        self.retry_command = f"hermes profile purge-identity {profile}"
        super().__init__(
            f"Profile '{profile}' was deleted, but its session/routing identity settlement is "
            f"still pending — run: {self.retry_command}")


def delete_profile(name: str, yes: bool = False) -> Path:
    """Delete a profile, its wrapper script, and its gateway service (service disabled first
    to prevent auto-restart, gateway stopped if running)."""
    canon = normalize_profile_name(name)
    if canon == "default":
        raise ValueError("Cannot delete the default profile (~/.hermes).\nTo remove everything, use: hermes uninstall")
    canon, profile_dir = _existing_profile_dir(canon)
    gw_running = _check_gateway_running(profile_dir)
    wrapper_path = _get_wrapper_dir() / canon
    has_wrapper = wrapper_path.exists()
    _print_delete_summary(canon, profile_dir, gw_running, wrapper_path if has_wrapper else None)
    if not yes:
        print()
        try:
            confirm = input(f"Type '{canon}' to confirm: ").strip()
        except (KeyboardInterrupt, EOFError):
            confirm = None
            print()
        if confirm != canon:
            print("Cancelled.")
            return profile_dir

    # 1. Disable service (prevents auto-restart); drop the s6 slot on container (host no-op).
    _cleanup_gateway_service(canon, profile_dir)
    _maybe_unregister_gateway_service(canon)

    # 2. Stop the gateway, then other backends bound to this profile (Desktop-spawned
    # serve/dashboard the pid file never names): they hold the SQLite connection open and
    # keep writing, which made rmtree fail ENOTEMPTY and resurrected the deleted tree.
    if gw_running:
        _stop_gateway_process(profile_dir)
    _stop_profile_backends(canon, profile_dir)
    _stop_bot_desktop(profile_dir)

    # Tombstone before rmtree so a stale serve/logging mkdir cannot relist this name live.
    mark_named_profile_deleted(profile_dir)
    # The multiplexer sees the tombstone, stops this profile's adapters and releases its handles
    # into the directory before we remove it. Identity settlement is a separate delete-only
    # operation below: an ordinary unserve must preserve identity, because a rename's old name
    # leaves the served set exactly like a deleted one does (#111926, delete side).
    _notify_multiplexer(canon)
    identity_settled = _purge_identity(canon)

    # The main serve process survives this deletion. Stop only this profile's MCP
    # transports and release cached stderr handles, including completed probes.
    from hermes_constants import hermes_home_key
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers(scope=hermes_home_key(profile_dir))

    # Release this process's holographic memory-store connections into the profile. The
    # Desktop's main serve process opens memory_store.db for every profile and is
    # deliberately not stopped above; on Windows its handles fail rmtree with WinError 32.
    # Inside serve (DELETE /api/profiles/<name>) the handles live here; from the CLI no-op.
    with contextlib.suppress(Exception):  # best-effort: never block the delete on the release path
        # 2c. See #88347.
        from plugins.memory import import_provider_module
        _released = import_provider_module("holographic", "store").MemoryStore.release_all_under(profile_dir)
        if _released:
            print(f"✓ Released {_released} memory-store connection(s) held by this process")
    with contextlib.suppress(Exception):
        from hermes_state_registry import close_all_under as _close_session_dbs_under
        _closed = _close_session_dbs_under(profile_dir)
        if _closed:
            print(f"✓ Released {_closed} session database connection(s) held by this process")

    # The Desktop serve process routes its agent/errors logs for every profile through one
    # QueueListener. On Windows those ConcurrentRotatingFileHandler instances retain their
    # ``.__*.lock`` files until explicitly closed, so rmtree otherwise fails with WinError 32.
    with contextlib.suppress(Exception):
        from hermes_logging import release_profile_log_handlers
        _released_logs = release_profile_log_handlers(profile_dir)
        if _released_logs:
            print(f"✓ Released {_released_logs} profile log handler(s) held by this process")

    # 3. Remove wrapper script
    if has_wrapper and remove_wrapper_script(canon):
        print(f"✓ Removed {wrapper_path}")

    # 4. Remove profile directory
    remove_error: Exception | None = None
    try:
        _rmtree_with_retry(profile_dir, _rmtree_make_writable)
        print(f"✓ Removed {profile_dir}")
    except Exception as e:
        print(f"⚠ Could not remove {profile_dir}: {e}")
        remove_error = e

    # 5. Clear active_profile if it pointed to this profile
    _retarget_active_profile(canon, "default", "✓ Active profile reset to default")
    if remove_error is not None:
        raise RuntimeError(f"Could not remove profile directory {profile_dir}: {remove_error}") from remove_error
    print(f"\nProfile '{canon}' deleted.")
    if not identity_settled:
        # Filesystem work and runtime teardown are done; the durable identity is not. Report the
        # partial settlement as a typed failure (still a RuntimeError for the CLI's handler)
        # instead of a clean success; the type carries the path and the retry for surfaces that
        # can report a partial success.
        raise ProfileIdentitySettlementPending(canon, profile_dir)
    return profile_dir


def _s6_runtime_manager():
    """The s6 service manager inside the container, else None. Silent on host: a failing/
    absent detector must never print a confusing s6 warning to non-container users."""
    try:
        from hermes_cli.service_manager import detect_service_manager, get_service_manager
        if detect_service_manager() != "s6":
            return None
        mgr = get_service_manager()
    except Exception:
        return None
    return mgr if mgr.supports_runtime_registration() else None


def _maybe_register_gateway_service(profile_name: str) -> None:
    """Register a profile's gateway with s6 inside the container. Best-effort: profile
    creation must not fail over a supervision-tree hiccup; `gateway start` re-registers.

    Port selection: each supervised profile gateway loads its own ``HERMES_HOME`` and binds the port
    resolved by ``gateway/config.py`` from that profile's environment — ``API_SERVER_PORT`` (or
    ``platforms.api_server.extra.port`` in the profile's ``config.yaml``), defaulting to 8642. There is no
    ``[gateway] port`` key and no Python-side allocator (PR #30136 review item I5 retired the
    SHA-256-derived range [9200, 9800) as dead code), so two profiles that both leave the port at its
    default will both try to bind 8642 — give each profile a distinct ``API_SERVER_PORT`` in its ``.env``.
    """
    mgr = _s6_runtime_manager()
    if mgr is None:
        return
    try:
        mgr.register_profile_gateway(profile_name, start_now=False)
    except ValueError:
        pass  # already registered (e.g. the container-boot reconciler brought up a stale slot)
    except Exception as exc:
        print(f"⚠ Could not register s6 gateway service: {exc}")


def _maybe_unregister_gateway_service(profile_name: str) -> None:
    """Tear down a profile's s6 gateway service inside the container; host no-op, idempotent."""
    mgr = _s6_runtime_manager()
    if mgr is None:
        return
    try:
        mgr.unregister_profile_gateway(profile_name)
    except Exception as exc:
        print(f"⚠ Could not unregister s6 gateway service: {exc}")


def _cleanup_gateway_service(name: str, profile_dir: Path) -> None:
    """Disable and remove systemd/launchd service for a profile."""
    import platform as _platform

    # The service name follows get_hermes_home(): bind the override (the seam a multiplexed
    # dashboard/tui-gateway process reads) and mirror the env for identity-file readers, so a
    # DELETE from a multi-profile dashboard names THIS profile's unit, never the host's bare one.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    old_home = os.environ.get("HERMES_HOME")
    home_token = set_hermes_home_override(str(profile_dir))
    try:
        os.environ["HERMES_HOME"] = str(profile_dir)
        from hermes_cli.gateway import get_service_name, get_launchd_plist_path, user_systemd_unit_dir

        def _run(*cmd: str) -> None:
            subprocess.run(list(cmd), capture_output=True, check=False, timeout=10)

        system = _platform.system()
        if system == "Linux":
            svc_name = get_service_name()
            svc_file = user_systemd_unit_dir() / f"{svc_name}.service"
            if svc_file.exists():
                _run("systemctl", "--user", "disable", svc_name)
                _run("systemctl", "--user", "stop", svc_name)
                svc_file.unlink(missing_ok=True)
                _run("systemctl", "--user", "daemon-reload")
                print(f"✓ Service {svc_name} removed")
        elif system == "Darwin":
            plist_path = get_launchd_plist_path()
            if plist_path.exists():
                _run("launchctl", "unload", str(plist_path))
                plist_path.unlink(missing_ok=True)
                print("✓ Launchd service removed")
    except Exception as e:
        print(f"⚠ Service cleanup: {e}")
    finally:
        reset_hermes_home_override(home_token)
        os.environ.pop("HERMES_HOME", None)
        if old_home is not None:
            os.environ["HERMES_HOME"] = old_home


def _stop_gateway_process(profile_dir: Path) -> None:
    """Stop a running gateway process via its PID file."""
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return
    try:
        raw = pid_file.read_text(encoding="utf-8-sig").strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        # Cross-profile kill refusal: the record's hermes_home stamp names the gateway's TRUE
        # owner. A poisoned gateway.pid in this dir can point at another profile's live
        # gateway — killing it starts a mutual SIGTERM restart loop.
        from gateway.status import get_process_start_time, recorded_gateway_home_conflicts, terminate_pid
        if recorded_gateway_home_conflicts(data, expected_home=profile_dir):
            print(
                f"✗ Refusing to stop PID {pid}: its recorded HERMES_HOME "
                f"belongs to a different profile than {profile_dir} "
                "(stale/poisoned PID record, #89315)."
            )
            return
        # terminate_pid picks the Windows primitive (taskkill /T cascades to children; raw
        # os.kill with SIGKILL fails at import on Windows).
        expected_start_time = data.get("start_time")
        if expected_start_time is None:
            expected_start_time = get_process_start_time(pid)
        terminate_pid(pid)  # graceful first
        if _wait_then_force_kill([pid], {pid: expected_start_time}):
            print(f"✓ Gateway stopped (PID {pid})")
        else:
            print(f"✓ Gateway force-stopped (PID {pid})")
    except (ProcessLookupError, PermissionError):
        print("✓ Gateway already stopped")
    except Exception as e:
        print(f"⚠ Could not stop gateway: {e}")


# Active profile (sticky default)

def get_active_profile(root: Path | None = None) -> str:
    """Read the sticky active profile name (of *root*, default: this process's Hermes root)."""
    path = root / "active_profile" if root is not None else _get_active_profile_path()
    try:
        name = path.read_text(encoding="utf-8-sig").strip()
        if not name:
            return "default"
        return name
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return "default"


def set_active_profile(name: str) -> None:
    """Set the sticky active profile (``default`` = remove the file)."""
    canon = _canon_valid(name)
    if canon != "default" and not profile_exists(canon):
        raise _missing_profile_error(canon)
    path = _get_active_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if canon == "default":
        path.unlink(missing_ok=True)
    else:
        tmp = path.with_suffix(".tmp")  # atomic write
        tmp.write_text(canon + "\n", encoding="utf-8")
        tmp.replace(path)


def _retarget_active_profile(old: str, new: str, message: str) -> None:
    """If the sticky active profile is *old*, point it at *new* and print *message*. Never raises."""
    with contextlib.suppress(Exception):
        if get_active_profile() == old:
            set_active_profile(new)
            print(message)


def get_active_profile_name() -> str:
    """Profile name inferred from HERMES_HOME: ``"default"`` when unset or ``~/.hermes``, the
    name under ``~/.hermes/profiles/<name>``, ``"custom"`` for any other path."""
    from hermes_constants import get_hermes_home
    resolved = get_hermes_home().resolve()
    if resolved == _get_default_hermes_home().resolve():
        return "default"
    profiles_root = _get_profiles_root().resolve()
    try:
        parts = resolved.relative_to(profiles_root).parts
        if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
            return parts[0]
    except ValueError:
        pass
    return "custom"


def current_profile_name(default: str | None = None) -> str | None:
    """Identity of the profile the current task runs FOR: the ``HERMES_HOME`` override when one is
    bound (a multiplexed cron tick, a routed gateway turn), else a launcher-pinned
    ``HERMES_PROFILE_NAME``/``HERMES_PROFILE`` (the kanban dispatcher pins it on its workers), else
    the name derived from the process's ``HERMES_HOME``; *default* when nothing names a profile.

    The env pin is read only outside an override: ``os.environ`` is the LAUNCH profile's, so under
    an override it would re-label a served profile's board writes with the host's identity.
    """
    from hermes_constants import get_hermes_home_override
    if get_hermes_home_override() is None:
        for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
            value = (os.environ.get(env_name) or "").strip()
            if value:
                return value
    try:
        return get_active_profile_name() or default
    except Exception:
        return default


# Export / Import

def _inside_git_checkout(path: Path) -> bool:
    """True when *path* lies inside a Git checkout. Walks the path's OWN resolved ancestry
    (not cwd) so the check holds when HERMES_HOME sits in a checkout but the process runs
    elsewhere (cron, service manager). Resolution failure reports True (fail closed)."""
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):  # RuntimeError: symlink loops on Python <= 3.12
        return True
    return any((candidate / ".git").exists() for candidate in (resolved, *resolved.parents))


def _profile_export_directory() -> Path:
    """Choose an export directory that cannot become source-tree input."""
    import tempfile
    export_dir = _get_default_hermes_home() / "profile-exports"
    if not _inside_git_checkout(export_dir):
        return export_dir

    # A custom deployment may point HERMES_HOME at its source checkout: use a sibling store,
    # falling back to the OS temp dir only when the user's home itself is a checkout (dotfiles
    # repo). Per-uid temp name: a fixed /tmp/hermes-profile-exports is a predictable shared
    # path another local user could pre-create (or symlink) first.
    uid_suffix = f"-{os.getuid()}" if hasattr(os, "getuid") else ""
    candidates = (
        Path.home() / ".hermes-profile-exports", Path(tempfile.gettempdir()) / f"hermes-profile-exports{uid_suffix}"
    )
    for candidate in candidates:
        if not _inside_git_checkout(candidate):
            return candidate
    # Fail closed: writing a secret-bearing archive into a source tree is the incident this
    # helper prevents; a stderr warning would not stop a scripted export.
    raise ValueError(
        # See #92457.
        "No safe automatic export destination: every candidate directory is "
        "inside a Git checkout. Provide an explicit output path outside the "
        "checkout (CLI: -o /path/outside/repo/profile.tar.gz)."
    )


def get_profile_export_path(name: str, *, timestamp: Optional[str] = None) -> Path:
    """Managed destination for an export with no explicit output — outside the cwd and every
    profile, since a ``<name>.tar.gz`` default in a source checkout got committed by accident."""
    canon = _canon_valid(name)
    export_dir = _profile_export_directory()
    export_dir.mkdir(parents=True, exist_ok=True)
    # exist_ok=True silently accepts a directory (or symlink) another local user pre-created
    # at a predictable path; refuse to write a secret-bearing archive anywhere we don't own.
    if export_dir.is_symlink():
        raise ValueError(
            f"Export directory {export_dir} is a symlink; refusing to write "
            "a profile archive through it. Provide an explicit output path."
        )
    if hasattr(os, "getuid") and export_dir.stat().st_uid != os.getuid():
        raise ValueError(
            f"Export directory {export_dir} is owned by another user; "
            "refusing to write a profile archive there. Provide an explicit output path."
        )
    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
    return export_dir / f"{canon}-{stamp}.tar.gz"


def _default_export_ignore(root_dir: Path):
    """copytree ignore for the default-profile export: root-level allow-list
    (``_DEFAULT_EXPORT_INCLUDE_ROOT``) plus universal exclusions. Surviving text files are
    then force-redacted by :func:`_scrub_export_secrets`.

    * **Root-level allow-list** — only entries whose name appears in ``_DEFAULT_EXPORT_INCLUDE_ROOT``
    survive. Everything else (such as an unrelated ``x11-dev/`` directory in a Docker deployment where
    HERMES_HOME equals the cwd) is excluded. Blacklisting was tried first and proved unable to anticipate
    every non-Hermes file the user may have lying alongside HERMES_HOME (#58394). * **Universal exclusions
    at any depth** — ``__pycache__``, sockets and other special files, temp files
    (:func:`_non_exportable_entries`); plus npm lockfiles, which may appear at the root.
    """

    def _ignore(directory: str, contents: list) -> set:
        # Universal exclusions (any depth) plus npm lockfiles that can appear at root.
        ignored = _non_exportable_entries(directory, contents)
        ignored.update({"package.json", "package-lock.json"} & set(contents))
        if Path(directory) == root_dir:
            ignored.update(entry for entry in contents if entry not in _DEFAULT_EXPORT_INCLUDE_ROOT)
        return ignored

    return _ignore


# Credential files dropped from named-profile exports. ``bot-desktop`` is the screen's runtime state:
# its persistent Chromium profile (Cookies, Login Data — the bot's live web sessions), Xauthority, sockets.
_EXPORT_CREDENTIAL_FILES = frozenset({"auth.json", ".env", "bot-desktop"})

# Text/config suffixes secret-scrubbed on export; binary DBs, images etc. are left alone.
_EXPORT_REDACT_SUFFIXES = frozenset({
    ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".ini", ".cfg", ".conf", ".py", ".sh",
    ".bash", ".zsh", ".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".xml", ".csv",
})
# ``Path(".cursorrules").suffix`` is "" — name-match; ``*.env.example`` uses endswith.
_EXPORT_REDACT_NAMES = frozenset({".cursorrules"})


def _should_redact_export_file(path: Path) -> bool:
    name = path.name
    return (
        name in _EXPORT_REDACT_NAMES
        or name.lower().endswith(".env.example")
        or path.suffix.lower() in _EXPORT_REDACT_SUFFIXES
    )


def _scrub_export_secrets(staged: Path) -> None:
    """Force-redact secret-shaped strings in a staged export tree (same pass as ``hermes
    sessions export --redact``). Runs on the staged copy only; symlinks to text files are
    materialized when content changes so redaction never follows a link back into the source."""
    from agent.redact import redact_sensitive_text
    for path in staged.rglob("*"):
        try:
            is_link = path.is_symlink()
            if not path.is_file():  # broken links, symlinked dirs, non-files
                continue
        except OSError:
            continue
        if not _should_redact_export_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        redacted = redact_sensitive_text(text, force=True)
        if redacted == text:
            continue
        if is_link:
            path.unlink()
        path.write_text(redacted, encoding="utf-8")


def export_profile(name: str, output_path: str, extra_files: Optional[Dict[str, str]] = None) -> Path:
    """Export a profile to a tar.gz archive; credential files are excluded and staged text is
    force-redacted first. Returns the output file path."""
    import tempfile
    canon, profile_dir = _existing_profile_dir(name)
    # Archive base name without extension (.tar.gz appended by the writer).
    base = str(Path(output_path)).removesuffix(".tar.gz").removesuffix(".tgz")

    # The default profile IS ~/.hermes (dir name ".hermes"), so both paths stage a filtered
    # copy under a temp dir named after the canonical id: root allow-list for default,
    # credential exclusion for named profiles.
    def _ignore_credentials(directory: str, contents: list) -> set:
        ignored = _non_exportable_entries(directory, contents)
        ignored.update(_EXPORT_CREDENTIAL_FILES & set(contents))
        if Path(directory) == profile_dir:
            ignored |= PM_RUNTIME_ROOT_DIRS & set(contents)
        return ignored

    ignore = _default_export_ignore(profile_dir) if canon == "default" else _ignore_credentials
    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / canon
        shutil.copytree(profile_dir, staged, symlinks=True, ignore=ignore)
        for rel, content in (extra_files or {}).items():
            target = staged.joinpath(*normalize_archive_parts(rel))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        _scrub_export_secrets(staged)
        return Path(make_targz(base, tmpdir, canon))


def import_profile(archive_path: str, name: Optional[str] = None) -> Path:
    """Import a profile from a tar.gz archive."""
    import tempfile
    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")
    top_dirs = archive_root_dirs(archive)
    archive_root = top_dirs.pop() if len(top_dirs) == 1 else None
    inferred_name = name or archive_root
    if not inferred_name:
        raise ValueError(
            "Cannot determine profile name from archive. "
            "Specify it explicitly: hermes profile import <archive> --name <name>"
        )
    if archive_root is None:
        raise ValueError("Profile archive must contain exactly one top-level directory.")

    # Default-profile archives have "default/" at top level; importing as "default" would
    # target ~/.hermes itself.
    canon = _canon_valid(inferred_name)
    if canon == "default":
        raise ValueError(
            "Cannot import as 'default' — that is the built-in root profile (~/.hermes). "
            "Specify a different name: hermes profile import <archive> --name <name>"
        )
    profile_dir = get_profile_dir(canon)
    if profile_dir.exists():
        raise _profile_exists_error(canon)
    _get_profiles_root().mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hermes_profile_import_") as tmpdir:
        staging_root = Path(tmpdir)
        safe_extract_targz(archive, staging_root)
        extracted = staging_root / archive_root
        if not extracted.is_dir():
            raise ValueError(f"Profile archive root is missing or invalid: {archive_root}")
        final_source = extracted
        if archive_root != canon:
            final_source = staging_root / canon
            extracted.rename(final_source)
        # Remove foreign runtime state before publishing, including file-shaped roots.
        # A failed removal must abort import rather than install a stale PM selection.
        for child in final_source.iterdir():
            if child.name in PM_RUNTIME_ROOT_DIRS:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        drop_profile_role(final_source)
        shutil.move(str(final_source), str(profile_dir))
    return profile_dir


# Rename

def _atomic_write_json(path: Path, data: dict) -> bool:
    """Atomic rewrite of a third-party JSON config; False on OSError (nothing partially written)."""
    from utils import atomic_json_write
    try:
        atomic_json_write(path, data)
        return True
    except OSError:
        return False


def _migrate_honcho_profile_host(old_name: str, new_name: str, new_dir: Path) -> None:
    """Rename Honcho host blocks for a renamed profile without changing peers."""
    old_host = f"hermes_{old_name}"
    legacy_old_host = f"hermes.{old_name}"
    new_host = f"hermes_{new_name}"
    candidates = [
        new_dir / "honcho.json", _get_default_hermes_home() / "honcho.json", Path.home() / ".honcho" / "config.json"
    ]
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        hosts = raw.get("hosts")
        if not isinstance(hosts, dict):
            continue
        source_host = old_host if old_host in hosts else legacy_old_host
        if source_host not in hosts:
            continue
        if new_host in hosts:
            print(f"⚠ Honcho host block not migrated: {new_host} already exists in {path}")
            continue
        block = hosts[source_host]
        if isinstance(block, dict) and "aiPeer" not in block:
            block["aiPeer"] = old_name  # source_host is ``hermes_<old>`` or legacy ``hermes.<old>``
        hosts[new_host] = hosts.pop(source_host)
        if _atomic_write_json(path, raw):
            print(f"✓ Honcho host updated: {source_host} → {new_host}")


def _record_profile_rename(new_dir: Path, old_canon: str) -> None:
    """Append ``old_canon`` to the renamed profile's ``previous_names`` history.
    Best-effort: never raises, so a metadata write failure cannot fail the rename.

    Only reached for a real slug change — ``rename_profile`` returns early for the
    default profile (display-name only) and refuses ``old == new`` (target exists)."""
    try:
        history = read_profile_meta(new_dir).get("previous_names") or []
        if old_canon not in history:
            history = [*history, old_canon]
        write_profile_meta(new_dir, previous_names=history)
    except Exception as exc:  # unwritable / corrupt profile.yaml — history is advisory
        logger.debug("profile rename: could not record previous name %r in %s: %s", old_canon, new_dir, exc)


def rename_profile(old_name: str, new_name: str) -> Path:
    """Rename a profile: directory, wrapper script, service, active_profile. The default
    profile's home IS the installation root, so "renaming" it sets a presentation-only
    ``display_name`` instead — the canonical id stays ``default``."""
    old_canon = _canon_valid(old_name)
    if old_canon == "default":
        if not (new_name or "").strip():
            raise ValueError("Display name cannot be empty.")
        cleaned = set_profile_display_name("default", new_name)
        print(f"✓ Display name set: {cleaned} (canonical id remains 'default')")
        return _get_default_hermes_home()
    new_canon = _canon_valid(new_name)
    if new_canon == "default":
        raise ValueError("Cannot rename to 'default' — it is reserved.")
    old_dir = get_profile_dir(old_canon)
    new_dir = get_profile_dir(new_canon)
    if not old_dir.is_dir():
        raise _unknown_profile_error(old_canon)
    if new_dir.exists():
        raise _profile_exists_error(new_canon)

    # 1. Stop gateway if running, and the screen whose launcher holds paths under the old name
    if _check_gateway_running(old_dir):
        _cleanup_gateway_service(old_canon, old_dir)
        _stop_gateway_process(old_dir)
    _stop_bot_desktop(old_dir)

    # 1b. Unroute the old name from a live multiplexer BEFORE the rename (same protocol as
    # delete_profile). A multiplexed secondary has no gateway.pid of its own, so the check above
    # reports it stopped while the default gateway still holds its adapters, cron ticker, logging
    # and SQLite handles; those re-``mkdir`` the old home the moment it moves (no tombstone →
    # ``mkdir_under_hermes_home`` does not refuse it) and the periodic reconcile re-adopts the
    # resurrected dir as a ghost served profile (#109267).
    live_mux = _live_default_multiplexer()
    if live_mux:
        mark_named_profile_deleted(old_dir)
        _notify_multiplexer(old_canon)

    # 1c. Release this process's cached MCP stderr handle into the old home (same as
    # delete_profile): Windows refuses to rename a directory holding an open file, and the
    # handle would otherwise stay cached under the old key after the move.
    from hermes_constants import hermes_home_key
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    shutdown_mcp_servers(scope=hermes_home_key(old_dir))

    # 2. Rename directory. If the move fails (cross-device EXDEV, permissions, a racing writer),
    # undo the unroute so the profile is never stranded tombstoned-but-present.
    try:
        old_dir.rename(new_dir)
    except Exception:
        if live_mux:
            clear_named_profile_deleted(old_dir)
            _notify_multiplexer(old_canon)
        raise
    print(f"✓ Renamed {old_dir.name} → {new_dir.name}")
    # The tombstone lives at profiles/.deleted/<old_name>; old_dir is gone so nothing can
    # resurrect it, and a future profile reusing the old name must not read as deleted.
    if live_mux:
        clear_named_profile_deleted(old_dir)

    # 2b. Record the rename so Bot Mode group chats can re-link persisted
    # member descriptors to the new slug (#110200). Best-effort: a metadata
    # write failure must never fail the rename itself.
    _record_profile_rename(new_dir, old_canon)

    # 3. Update profile-scoped Honcho host blocks, preserving aiPeer identity
    _migrate_honcho_profile_host(old_canon, new_canon, new_dir)

    # 4. Update wrapper script
    remove_wrapper_script(old_canon)
    collision = check_alias_collision(new_canon)
    if not collision:
        create_wrapper_script(new_canon)
        print(f"✓ Alias updated: {new_canon}")
    else:
        print(f"⚠ Cannot create alias '{new_canon}' — {collision}")

    # 5. Update active_profile if it pointed to old name
    _retarget_active_profile(old_canon, new_canon, f"✓ Active profile updated: {new_canon}")

    # 6. Migrate profile-name-keyed session/routing state (session keys, profile_name, heartbeats,
    # delivery + routing index) from the old name to the new one. A stale ``agent:<old>:*`` routing
    # key otherwise resolves to a profile that no longer exists on every inbound event.
    from hermes_cli.profile_identity import _migrate_profile_identity
    _migrate_profile_identity(old_canon, new_canon, live_mux)

    # 7. Hot-serve the renamed profile now (mirrors create; a missed signal only delays it).
    if live_mux:
        _notify_multiplexer(new_canon)
    return new_dir


# Profile env resolution (called from _apply_profile_override)

def profile_root_for_env_home(env_home: str, default_root: Path) -> Path:
    """Hermes root named by an exported ``HERMES_HOME``: the grandparent of a profile-shaped value
    (``<root>/profiles/<name>``, mirrors ``get_default_hermes_root()``), the value itself otherwise,
    *default_root* when unset. Pure: callers pass any process's env, not only ``os.environ``."""
    env_home = env_home.strip()
    if not env_home:
        return default_root
    env_path = Path(env_home)
    return env_path.parent.parent if env_path.parent.name == "profiles" else env_path


def resolve_profile_env(profile_name: str) -> str:
    """Resolve a profile name to a HERMES_HOME path string. Called early in the CLI entry
    point, before hermes modules are imported, to set HERMES_HOME.

    When HERMES_HOME is already set, the configured spelling IS the launch root (it may be a
    junction/symlink alias of the platform default). Keep that spelling so profile re-home does not destroy
    the launcher's lexical provenance -- the subprocess sanitizer needs it to match Hermes-owned PYTHONPATH
    entries written in the same spelling (#82581 junction follow-up). Physically the paths are identical
    (junction-transparent); only the spelling is preserved.
    """
    canon = _canon_valid(profile_name)
    root = profile_root_for_env_home(os.environ.get("HERMES_HOME", ""), _get_default_hermes_home())
    if canon == "default":
        return str(root)
    profile_dir = root / "profiles" / canon
    if not named_profile_is_live(profile_dir):
        raise _missing_profile_error(canon)
    return str(profile_dir)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def has_bundled_skills_opt_out(profile_dir: Path) -> bool:
    """Return True if the profile opted out of bundled-skill seeding."""
    try:
        return (profile_dir / NO_BUNDLED_SKILLS_MARKER).exists()
    except OSError:
        return False
# ---- END PLUGIN-COMPAT ----
