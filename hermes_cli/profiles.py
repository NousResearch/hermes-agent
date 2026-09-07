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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.skill_utils import is_excluded_skill_path
from hermes_cli.archive_safe import archive_root_dirs, make_targz, normalize_archive_parts, safe_extract_targz
from hermes_constants import clear_named_profile_deleted, mark_named_profile_deleted, named_profile_is_deleted

logger = logging.getLogger(__name__)

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_WARNED_MISSING_ALLOWLIST_ENTRIES: set[tuple[str, ...]] = set()

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
_CLONE_ALL_STRIP: list[str] = ["gateway.pid", "gateway_state.json", "processes.json"]

# Infrastructure excluded from --clone-all ONLY when the source is the default profile
# (``~/.hermes``): git checkout (+ ~3 GB venv), worktrees, sibling profiles, shared bins,
# npm packages. Named profiles never hold these at root, so the gate avoids silently
# dropping user data from a named-profile source. Export uses a root allow-list instead
# (``_DEFAULT_EXPORT_INCLUDE_ROOT``): an archive is a portable snapshot, a clone must run.
_CLONE_ALL_DEFAULT_EXCLUDE_ROOT: frozenset[str] = frozenset({
    "hermes-agent", ".worktrees", "profiles", "bin", "node_modules",
})

# Per-profile history excluded from --clone-all for ANY source: SQLite session store
# (+wal/shm, can reach many GB), session dirs, `hermes backup` archives, quick-backup
# snapshots, checkpoints. Inheriting them is never useful (restoring one inside the
# clone would resurrect the SOURCE profile's state) and can balloon the copy by tens of GB.
_CLONE_ALL_HISTORY_EXCLUDE_ROOT: frozenset[str] = frozenset({
    "state.db", "state.db-wal", "state.db-shm", "sessions", "backups", "state-snapshots", "checkpoints",
})

# Marker written by `hermes profile create --no-skills`. When present at a profile root,
# seed_profile_skills() callers (fresh-create, `hermes update` all-profile sync, the
# dashboard) skip bundled-skill seeding. Delete the file to opt back in.
NO_BUNDLED_SKILLS_MARKER = ".no-bundled-skills"

# Header seeded into a profile's empty .env so it owns a credentials file from day one.
_PLACEHOLDER_ENV = (
    "# Per-profile secrets for this Hermes profile.\n"
    "# API keys and tokens set here override the shell environment.\n"
    "# Behavioral settings belong in config.yaml, not here.\n"
)


def _clone_all_copytree_ignore(source_dir: Path):
    """copytree ignore for --clone-all: history artifacts for any source, infrastructure
    only when the source is the default profile (see the two exclude sets above)."""
    source_resolved = source_dir.resolve()
    root_exclude = set(_CLONE_ALL_HISTORY_EXCLUDE_ROOT)
    if source_resolved == _get_default_hermes_home().resolve():
        root_exclude |= _CLONE_ALL_DEFAULT_EXCLUDE_ROOT

    def _ignore(directory: str, names: List[str]) -> List[str]:
        try:
            at_root = Path(directory).resolve() == source_resolved
        except (OSError, ValueError):
            # resolve() can fail on odd FS layouts (broken symlinks, missing parents).
            # Fail open — better to over-copy than silently drop user data.
            at_root = False
        return [
            entry for entry in names
            if entry == "__pycache__"
            or entry.endswith((".pyc", ".pyo", ".sock", ".tmp"))
            or (at_root and entry in root_exclude)
        ]

    return _ignore


# Allow-list for ``export_profile("default")``: when HERMES_HOME equals the cwd
# (Docker/custom deployments) the default home holds arbitrary user files that must NOT
# be bundled. Only known Hermes profile artifacts at the root survive; sensitive runtime
# infrastructure (``state.db``, ``logs/``, ``auth.*``, other profiles) is deliberately
# absent so the export stays a portable, credential-free snapshot. Add new artifacts here
# when introduced in ``hermes_constants``.
# See #58394.
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
        return "hermes -p" in path.read_text(encoding="utf-8")
    except Exception:
        return False


def _missing_profile_error(canon: str) -> FileNotFoundError:
    return FileNotFoundError(f"Profile '{canon}' does not exist. Create it with: hermes profile create {canon}")


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
        raise ValueError(f"Invalid profile name {name!r}. Must match [a-z0-9][a-z0-9_-]{{0,63}}")
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
        raise ValueError(f"Invalid alias name {name!r}. Must match [a-z0-9][a-z0-9_-]{{0,63}}")


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
        raise FileNotFoundError(f"Profile '{canon}' does not exist.")
    return canon, profile_dir


def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its HERMES_HOME directory."""
    canon = normalize_profile_name(name)
    if canon == "default":
        return _get_default_hermes_home()
    return _get_profiles_root() / canon


def profile_exists(name: str) -> bool:
    """Check whether a live (non-tombstoned) profile directory exists."""
    canon = normalize_profile_name(name)
    if canon == "default":
        return True
    profile_dir = get_profile_dir(canon)
    return profile_dir.is_dir() and not named_profile_is_deleted(profile_dir)


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
    """Sorted named-profile dirs (valid ids, never ``default``); ``live_only`` skips tombstones."""
    profiles_root = _get_profiles_root()
    if not profiles_root.is_dir():
        return []
    return [
        entry for entry in sorted(profiles_root.iterdir())
        if entry.is_dir()
        and entry.name != "default"
        and _PROFILE_ID_RE.match(entry.name)
        and not (live_only and named_profile_is_deleted(entry))
    ]


def list_profile_names() -> List[str]:
    """Cheap name-only listing (``default`` + profile dirs). Unlike :func:`list_profiles` this
    reads NO per-profile config — safe for hot paths (cron target listings, create validation)."""
    names = ["default"]
    with contextlib.suppress(OSError):
        names.extend(entry.name for entry in _iter_named_profile_dirs(live_only=False))
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
    if not (profile_dir / "config.yaml").exists():
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
            with open(entry, "r", encoding="utf-8", errors="strict") as f:
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


def _load_yaml_dict(path: Path) -> Optional[dict]:
    """Return the mapping in a YAML file, or None when missing/unreadable/not a mapping."""
    if not path.is_file():
        return None
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_distribution_meta(profile_dir: Path) -> tuple:
    """``(name, version, source)`` from ``distribution.yaml``; ``(None, None, None)`` if absent."""
    data = _load_yaml_dict(profile_dir / "distribution.yaml")
    if data is None:
        return None, None, None
    return data.get("name"), data.get("version"), data.get("source")


def _read_config_model(profile_dir: Path) -> tuple:
    """Read model/provider from a profile's config.yaml. Returns (model, provider)."""
    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return None, None
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


def _seed_model_config(profile_dir: Path) -> None:
    """Copy (not link) the active profile's model block into a fresh profile so it is usable;
    profiles stay independent islands afterwards."""
    config_path = profile_dir / "config.yaml"
    if config_path.exists():
        return
    with contextlib.suppress(Exception):  # creation must not fail over this; `hermes model` sets it later
        import yaml
        from hermes_constants import get_hermes_home
        from hermes_cli.config import read_user_config_raw
        source = get_hermes_home() / "config.yaml"
        model_cfg = read_user_config_raw(source).get("model") if source.is_file() else None
        if model_cfg:
            config_path.write_text(yaml.safe_dump({"model": model_cfg}, sort_keys=False), encoding="utf-8")


def _check_gateway_running(profile_dir: Path) -> bool:
    """Gateway liveness for a profile dir, never mutating HERMES_HOME.

    Primary signal is ``gateway.pid`` verified against the runtime lock (fails closed when
    the lock isn't held by *this* reader: dashboard as a separate s6 service, launch-service
    gateways with no live PID file); fallback validates the PID in ``gateway_state.json``
    against the process table, matching ``/api/status``."""
    try:
        from gateway.status import get_running_pid, get_runtime_status_running_pid, read_runtime_status
        if get_running_pid(profile_dir / "gateway.pid", cleanup_stale=False) is not None:
            return True
    except Exception:
        pass
    try:
        runtime = read_runtime_status(profile_dir / "gateway_state.json")
        return get_runtime_status_running_pid(runtime, expected_home=profile_dir) is not None
    except Exception:
        return False


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


# In-process skill-count cache. ``rglob("SKILL.md")`` walks every skill's sub-trees; the
# default profile alone has ~270 skills and ``list_profiles`` counts EVERY profile (16+), so
# an uncached scan costs ~6s — enough for the desktop's per-request calls to time out and
# the sidebar to render "全部智能体 0". Keyed by skills dir, invalidated when the tree
# signature changes (skill add/remove) or after a short TTL (deep edits).
_SKILL_COUNT_CACHE: dict[str, tuple[float, float, int]] = {}
_SKILL_COUNT_TTL_SECONDS = 30.0


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


def _count_skills(profile_dir: Path) -> int:
    """Count installed skills in a profile (cached by skills-dir signature)."""
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return 0
    key = str(skills_dir)
    signature = _skills_dir_signature(skills_dir)
    now = time.time()
    cached = _SKILL_COUNT_CACHE.get(key)
    if cached is not None and cached[0] == signature and (now - cached[1]) < _SKILL_COUNT_TTL_SECONDS:
        return cached[2]
    count = sum(1 for md in skills_dir.rglob("SKILL.md") if not is_excluded_skill_path(md))
    _SKILL_COUNT_CACHE[key] = (signature, now, count)
    return count


# profile.yaml — per-profile metadata (description, role, etc.)
# Deliberately tiny and separate from ``config.yaml`` (user-facing Hermes config, ~5000
# lines of defaults): this is metadata ABOUT the profile. Missing file -> empty defaults,
# never an error; the kanban decomposer falls back to the profile name.


def read_profile_meta(profile_dir: Path) -> dict:
    """Read ``profile.yaml`` -> ``{description, description_auto, display_name}`` (empty
    defaults when missing/unreadable). Never raises — a corrupt file on one profile must not
    break ``hermes profile list``."""
    data = _load_yaml_dict(profile_dir / "profile.yaml") or {}
    return {
        "description": str(data.get("description") or "").strip(),
        "description_auto": bool(data.get("description_auto", False)),
        "display_name": str(data.get("display_name") or "").strip(),
    }


def write_profile_meta(
    profile_dir: Path, *, description: Optional[str] = None, description_auto: Optional[bool] = None,
    display_name: Optional[str] = None,
) -> None:
    """Update ``profile.yaml`` in place: only passed fields are overwritten; the file is
    created if missing. The profile directory itself must exist."""
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"profile directory does not exist: {profile_dir}")
    path = profile_dir / "profile.yaml"
    existing: dict = _load_yaml_dict(path) or {}
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
    # Atomic write: bare open("w") truncates before the dump, and the read path swallows
    # parse errors as {}, so a crashed write would silently drop unspecified fields.
    # See #51356.
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

def _profile_info(name: str, path: Path, *, is_default: bool, alias_name: Optional[str] = None) -> ProfileInfo:
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
    return ProfileInfo(
        name=name, path=path, is_default=is_default, gateway_running=gateway_running, model=model,
        provider=provider, has_env=(path / ".env").exists(), skill_count=_count_skills(path),
        alias_path=alias_path, alias_name=alias_name, distribution_name=dist_name,
        distribution_version=dist_version, distribution_source=dist_source,
        **meta,
    )


def list_profiles() -> List[ProfileInfo]:
    """Return info for all profiles, including the default."""
    profiles = []
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        profiles.append(_profile_info("default", default_home, is_default=True))
    named = _iter_named_profile_dirs()
    if named:
        alias_map = build_alias_map()  # ONCE, not per profile (was the dominant cost)
        for entry in named:
            alias_name = alias_map.get(normalize_profile_name(entry.name))
            profiles.append(_profile_info(entry.name, entry, is_default=False, alias_name=alias_name))
    return profiles


def profiles_to_serve(multiplex: bool, profile_allowlist: Optional[List[str]] = None) -> List[Tuple[str, Path]]:
    """``(profile_name, hermes_home)`` pairs a gateway should serve — the single chokepoint
    for "which profiles does the inbound gateway handle".

    ``multiplex=False``: exactly one entry for the *active* profile (byte-for-byte the
    historical single-profile behavior; name is ``"default"`` or the named profile's id).
    ``multiplex=True``: default plus every live named profile, optionally filtered by
    *profile_allowlist* (invalid entries skipped, missing ones warned once)."""
    active = get_active_profile_name() or "default"
    if not multiplex:
        return [(active, get_profile_dir(active))]
    serve: List[Tuple[str, Path]] = [("default", _get_default_hermes_home())]
    allowed: Optional[set[str]] = None
    if profile_allowlist is not None:
        allowed = set()
        for entry in profile_allowlist:
            if not isinstance(entry, str):
                continue
            try:
                name = _canon_valid(entry)
            except ValueError:
                continue
            if name != "default":
                allowed.add(name)
    for entry in _iter_named_profile_dirs():
        if allowed is None or entry.name in allowed:
            serve.append((entry.name, entry))
    if allowed is not None:
        missing = tuple(sorted(allowed - {name for name, _ in serve}))
        if missing and missing not in _WARNED_MISSING_ALLOWLIST_ENTRIES:
            _WARNED_MISSING_ALLOWLIST_ENTRIES.add(missing)
            logger.warning("Skipping missing gateway.multiplex_profile_allowlist profile(s): %s", ", ".join(missing))
    return serve


def _resolve_clone_source(clone_from: Optional[str]) -> Path:
    """Directory to clone from: the named profile, or the active profile when ``None``."""
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


def _clone_all_into(source_dir: Path, profile_dir: Path, canon: str) -> None:
    """--clone-all: full copytree minus infrastructure/history, then strip runtime files
    and cloned single-use OAuth grants."""
    shutil.copytree(source_dir, profile_dir, symlinks=True, ignore=_clone_all_copytree_ignore(source_dir))
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


def _bootstrap_profile_dir(profile_dir: Path, source_dir: Optional[Path]) -> None:
    """Fresh layout: bootstrap dirs, then either seed a model block (no source) or clone
    config files, installed skills (the dashboard's "clone from default" must keep bundled
    AND user-installed skills), and memory/identity files from *source_dir*."""
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
        shutil.copytree(source_skills, profile_dir / "skills", symlinks=True, dirs_exist_ok=True)
    for relpath in _CLONE_SUBDIR_FILES:
        _clone_file(source_dir, profile_dir, relpath)


def create_profile(
    name: str, clone_from: Optional[str] = None, clone_all: bool = False, clone_config: bool = False,
    no_alias: bool = False, no_skills: bool = False, description: Optional[str] = None,
) -> Path:
    """Create a new profile directory and return its path.

    ``clone_from`` defaults to the active profile when cloning. ``clone_all`` copies all state;
    ``clone_config`` copies config.yaml/.env/SOUL.md, installed skills, and identity files.
    ``no_skills`` creates an empty profile and writes a marker so ``hermes update`` skips
    re-seeding its skills; it is mutually exclusive with the clone options, which copy skills."""
    if no_skills and (clone_from is not None or clone_config or clone_all):
        raise ValueError(
            "--no-skills is mutually exclusive with --clone / --clone-from / --clone-all "
            "(cloning explicitly copies skills from the source profile)."
        )
    canon = _canon_valid(name)
    if canon == "default":
        raise ValueError("Cannot create a profile named 'default' — it is the built-in profile (~/.hermes).")
    profile_dir = get_profile_dir(canon)
    if profile_dir.exists() and named_profile_is_deleted(profile_dir):
        # Empty shells left by post-delete mkdir may be replaced. Identity files mean the
        # leftover is not a shell — fail closed, no rmtree.
        if (profile_dir / "config.yaml").exists() or (profile_dir / ".env").exists():
            raise FileExistsError(f"Profile '{canon}' already exists at {profile_dir}")
        shutil.rmtree(profile_dir)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{canon}' already exists at {profile_dir}")
    clear_named_profile_deleted(profile_dir)
    source_dir = None
    if clone_from is not None or clone_all or clone_config:
        source_dir = _resolve_clone_source(clone_from)
    if clone_all and source_dir:
        _clone_all_into(source_dir, profile_dir, canon)
    else:
        _bootstrap_profile_dir(profile_dir, source_dir)

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

    # Inside a container under s6, register the gateway as a runtime s6 service so
    # `hermes -p <profile> gateway start` supervises via `s6-svc -u` instead of a bare
    # process. No-op on host (systemd/launchd/windows unit generation handles lifecycle).
    _maybe_register_gateway_service(canon)
    return profile_dir


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

    # Tombstone before rmtree so a stale serve/logging mkdir cannot relist this name live.
    mark_named_profile_deleted(profile_dir)

    # Release this process's holographic memory-store connections into the profile. The
    # Desktop's main serve process opens memory_store.db for every profile and is
    # deliberately not stopped above; on Windows its handles fail rmtree with WinError 32.
    # Inside serve (DELETE /api/profiles/<name>) the handles live here; from the CLI no-op.
    with contextlib.suppress(Exception):  # best-effort: never block the delete on the release path
        # 2c. See #88347.
        from plugins.memory.holographic.store import MemoryStore as _MemoryStore
        _released = _MemoryStore.release_all_under(profile_dir)
        if _released:
            print(f"✓ Released {_released} memory-store connection(s) held by this process")

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

    # HERMES_HOME is set temporarily so _profile_suffix resolves the service name.
    old_home = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = str(profile_dir)
        from hermes_cli.gateway import get_service_name, get_launchd_plist_path

        def _run(*cmd: str) -> None:
            subprocess.run(list(cmd), capture_output=True, check=False, timeout=10)

        system = _platform.system()
        if system == "Linux":
            svc_name = get_service_name()
            svc_file = Path.home() / ".config" / "systemd" / "user" / f"{svc_name}.service"
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
        os.environ.pop("HERMES_HOME", None)
        if old_home is not None:
            os.environ["HERMES_HOME"] = old_home


def _stop_gateway_process(profile_dir: Path) -> None:
    """Stop a running gateway process via its PID file."""
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
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

def get_active_profile() -> str:
    """Read the sticky active profile name."""
    path = _get_active_profile_path()
    try:
        return path.read_text(encoding="utf-8").strip() or "default"
    except (UnicodeDecodeError, OSError):
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
    at any depth** — ``__pycache__``, sockets, temp files; plus npm lockfiles, which may appear at the root.
    """

    def _ignore(directory: str, contents: list) -> set:
        # Universal exclusions (any depth) plus npm lockfiles that can appear at root.
        ignored: set = {
            entry for entry in contents
            if entry == "__pycache__"
            or entry.endswith((".sock", ".tmp"))
            or entry in {"package.json", "package-lock.json"}
        }
        if Path(directory) == root_dir:
            ignored.update(entry for entry in contents if entry not in _DEFAULT_EXPORT_INCLUDE_ROOT)
        return ignored

    return _ignore


# Credential files dropped from named-profile exports.
_EXPORT_CREDENTIAL_FILES = frozenset({"auth.json", ".env"})

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
            text = path.read_text(encoding="utf-8")
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
        return _EXPORT_CREDENTIAL_FILES & set(contents)

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
        raise FileExistsError(f"Profile '{canon}' already exists at {profile_dir}")
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
        shutil.move(str(final_source), str(profile_dir))
    return profile_dir


# Rename

def _atomic_write_json(path: Path, data: dict) -> bool:
    """Write *data* to *path* via a sibling ``.tmp`` + rename. Returns False (tmp cleaned) on OSError."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
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
            raw = json.loads(path.read_text(encoding="utf-8"))
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
        raise FileNotFoundError(f"Profile '{old_canon}' does not exist.")
    if new_dir.exists():
        raise FileExistsError(f"Profile '{new_canon}' already exists.")

    # 1. Stop gateway if running
    if _check_gateway_running(old_dir):
        _cleanup_gateway_service(old_canon, old_dir)
        _stop_gateway_process(old_dir)

    # 2. Rename directory
    old_dir.rename(new_dir)
    print(f"✓ Renamed {old_dir.name} → {new_dir.name}")

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
    return new_dir


# Profile env resolution (called from _apply_profile_override)

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
    env_home = os.environ.get("HERMES_HOME", "").strip()
    if env_home:
        env_path = Path(env_home)
        # A profile-shaped env value means the root is the grandparent (mirrors
        # get_default_hermes_root()).
        root = env_path.parent.parent if env_path.parent.name == "profiles" else env_path
    else:
        root = _get_default_hermes_home()
    if canon == "default":
        return str(root)
    profile_dir = root / "profiles" / canon
    if not profile_dir.is_dir() or named_profile_is_deleted(profile_dir):
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
