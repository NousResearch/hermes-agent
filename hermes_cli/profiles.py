"""
Profile management for multiple isolated Hermes instances.

Each profile is a fully independent HERMES_HOME directory with its own
config.yaml, .env, memory, sessions, skills, gateway, cron, and logs.
Profiles live under ``~/.hermes/profiles/<name>/`` by default.

The "default" profile is ``~/.hermes`` itself — backward compatible,
zero migration needed.

Usage::

    hermes profile create coder          # fresh profile + bundled skills
    hermes profile create coder --clone  # also copy config, .env, SOUL.md
    hermes profile create coder --clone-all  # full copy of source profile
    coder chat                           # use via wrapper alias
    hermes -p coder chat                 # or via flag
    hermes profile use coder             # set as sticky default
    hermes profile delete coder          # remove profile + alias + service
"""

import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import List, Optional

_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# Directories bootstrapped inside every new profile
_PROFILE_DIRS = [
    "memories",
    "sessions",
    "skills",
    "skins",
    "logs",
    "plans",
    "workspace",
    "cron",
]

# Files copied during --clone (if they exist in the source)
_CLONE_CONFIG_FILES = [
    "config.yaml",
    ".env",
    "SOUL.md",
]

# Subdirectory files copied during --clone (path relative to profile root).
# Memory files are part of the agent's curated identity — just as important
# as SOUL.md for continuity when cloning a profile.
_CLONE_SUBDIR_FILES = [
    "memories/MEMORY.md",
    "memories/USER.md",
]

# Runtime files stripped after --clone-all (shouldn't carry over)
_CLONE_ALL_STRIP = [
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
]

# Directories/files to exclude when exporting the default (~/.hermes) profile.
# The default profile contains infrastructure (repo checkout, worktrees, DBs,
# caches, binaries) that named profiles don't have.  We exclude those so the
# export is a portable, reasonable-size archive of actual profile data.
_DEFAULT_EXPORT_EXCLUDE_ROOT = frozenset({
    # Infrastructure
    "hermes-agent",         # repo checkout (multi-GB)
    ".worktrees",           # git worktrees
    "profiles",             # other profiles — never recursive-export
    "bin",                  # installed binaries (tirith, etc.)
    "node_modules",         # npm packages
    # Databases & runtime state
    "state.db", "state.db-shm", "state.db-wal",
    "hermes_state.db",
    "response_store.db", "response_store.db-shm", "response_store.db-wal",
    "gateway.pid", "gateway_state.json", "processes.json",
    "auth.json",            # API keys, OAuth tokens, credential pools
    ".env",                 # API keys (dotenv)
    "auth.lock", "active_profile", ".update_check",
    "errors.log",
    ".hermes_history",
    # Caches (regenerated on use)
    "image_cache", "audio_cache", "document_cache",
    "browser_screenshots", "checkpoints",
    "sandboxes",
    "logs",                 # gateway logs
})

# Names that cannot be used as profile aliases
_RESERVED_NAMES = frozenset({
    "hermes", "default", "test", "tmp", "root", "sudo",
})

# Hermes subcommands that cannot be used as profile names/aliases
_HERMES_SUBCOMMANDS = frozenset({
    "chat", "model", "gateway", "setup", "whatsapp", "login", "logout",
    "status", "cron", "doctor", "config", "pairing", "skills", "tools",
    "mcp", "sessions", "insights", "version", "update", "uninstall",
    "profile", "plugins", "honcho", "acp",
})


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_profiles_root() -> Path:
    """Return the directory where named profiles are stored.

    Always ``~/.hermes/profiles/`` — anchored to the user's home,
    NOT to the current HERMES_HOME (which may itself be a profile).
    This ensures ``coder profile list`` can see all profiles.
    """
    return Path.home() / ".hermes" / "profiles"


def _get_default_hermes_home() -> Path:
    """Return the default (pre-profile) HERMES_HOME path."""
    return Path.home() / ".hermes"


def _get_active_profile_path() -> Path:
    """Return the path to the sticky active_profile file."""
    return _get_default_hermes_home() / "active_profile"


def _get_wrapper_dir() -> Path:
    """Return the directory for wrapper scripts."""
    return Path.home() / ".local" / "bin"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_profile_name(name: str) -> None:
    """Raise ``ValueError`` if *name* is not a valid profile identifier."""
    if name == "default":
        return  # special alias for ~/.hermes
    if not _PROFILE_ID_RE.match(name):
        raise ValueError(
            f"Invalid profile name {name!r}. Must match "
            f"[a-z0-9][a-z0-9_-]{{0,63}}"
        )


def get_profile_dir(name: str) -> Path:
    """Resolve a profile name to its HERMES_HOME directory."""
    if name == "default":
        return _get_default_hermes_home()
    return _get_profiles_root() / name


def profile_exists(name: str) -> bool:
    """Check whether a profile directory exists."""
    if name == "default":
        return True
    return get_profile_dir(name).is_dir()


# ---------------------------------------------------------------------------
# Alias / wrapper script management
# ---------------------------------------------------------------------------

def check_alias_collision(name: str) -> Optional[str]:
    """Return a human-readable collision message, or None if the name is safe.

    Checks: reserved names, hermes subcommands, existing binaries in PATH.
    """
    if name in _RESERVED_NAMES:
        return f"'{name}' is a reserved name"
    if name in _HERMES_SUBCOMMANDS:
        return f"'{name}' conflicts with a hermes subcommand"

    # Check existing commands in PATH
    wrapper_dir = _get_wrapper_dir()
    try:
        result = subprocess.run(
            ["which", name], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            existing_path = result.stdout.strip()
            # Allow overwriting our own wrappers
            if existing_path == str(wrapper_dir / name):
                try:
                    content = (wrapper_dir / name).read_text()
                    if "hermes -p" in content:
                        return None  # it's our wrapper, safe to overwrite
                except Exception:
                    pass
            return f"'{name}' conflicts with an existing command ({existing_path})"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None  # safe


def _is_wrapper_dir_in_path() -> bool:
    """Check if ~/.local/bin is in PATH."""
    wrapper_dir = str(_get_wrapper_dir())
    return wrapper_dir in os.environ.get("PATH", "").split(os.pathsep)


def create_wrapper_script(name: str) -> Optional[Path]:
    """Create a shell wrapper script at ~/.local/bin/<name>.

    Returns the path to the created wrapper, or None if creation failed.
    """
    wrapper_dir = _get_wrapper_dir()
    try:
        wrapper_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"⚠ Could not create {wrapper_dir}: {e}")
        return None

    wrapper_path = wrapper_dir / name
    try:
        wrapper_path.write_text(f'#!/bin/sh\nexec hermes -p {name} "$@"\n')
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        return wrapper_path
    except OSError as e:
        print(f"⚠ Could not create wrapper at {wrapper_path}: {e}")
        return None


def remove_wrapper_script(name: str) -> bool:
    """Remove the wrapper script for a profile. Returns True if removed."""
    wrapper_path = _get_wrapper_dir() / name
    if wrapper_path.exists():
        try:
            # Verify it's our wrapper before removing
            content = wrapper_path.read_text()
            if "hermes -p" in content:
                wrapper_path.unlink()
                return True
        except Exception:
            pass
    return False


# ---------------------------------------------------------------------------
# ProfileInfo
# ---------------------------------------------------------------------------

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


def _read_config_model(profile_dir: Path) -> tuple:
    """Read model/provider from a profile's config.yaml. Returns (model, provider)."""
    config_path = profile_dir / "config.yaml"
    if not config_path.exists():
        return None, None
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {})
        if isinstance(model_cfg, str):
            return model_cfg, None
        if isinstance(model_cfg, dict):
            return model_cfg.get("default") or model_cfg.get("model"), model_cfg.get("provider")
        return None, None
    except Exception:
        return None, None


def _check_gateway_running(profile_dir: Path) -> bool:
    """Check if a gateway is running for a given profile directory."""
    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return False
    try:
        raw = pid_file.read_text().strip()
        if not raw:
            return False
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        os.kill(pid, 0)  # existence check
        return True
    except (json.JSONDecodeError, KeyError, ValueError, TypeError,
            ProcessLookupError, PermissionError, OSError):
        return False


def _count_skills(profile_dir: Path) -> int:
    """Count installed skills in a profile."""
    skills_dir = profile_dir / "skills"
    if not skills_dir.is_dir():
        return 0
    count = 0
    for md in skills_dir.rglob("SKILL.md"):
        if "/.hub/" not in str(md) and "/.git/" not in str(md):
            count += 1
    return count


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def list_profiles() -> List[ProfileInfo]:
    """Return info for all profiles, including the default."""
    profiles = []
    wrapper_dir = _get_wrapper_dir()

    # Default profile
    default_home = _get_default_hermes_home()
    if default_home.is_dir():
        model, provider = _read_config_model(default_home)
        profiles.append(ProfileInfo(
            name="default",
            path=default_home,
            is_default=True,
            gateway_running=_check_gateway_running(default_home),
            model=model,
            provider=provider,
            has_env=(default_home / ".env").exists(),
            skill_count=_count_skills(default_home),
        ))

    # Named profiles
    profiles_root = _get_profiles_root()
    if profiles_root.is_dir():
        for entry in sorted(profiles_root.iterdir()):
            if not entry.is_dir():
                continue
            name = entry.name
            if not _PROFILE_ID_RE.match(name):
                continue
            model, provider = _read_config_model(entry)
            alias_path = wrapper_dir / name
            profiles.append(ProfileInfo(
                name=name,
                path=entry,
                is_default=False,
                gateway_running=_check_gateway_running(entry),
                model=model,
                provider=provider,
                has_env=(entry / ".env").exists(),
                skill_count=_count_skills(entry),
                alias_path=alias_path if alias_path.exists() else None,
            ))

    return profiles


def create_profile(
    name: str,
    clone_from: Optional[str] = None,
    clone_all: bool = False,
    clone_config: bool = False,
    no_alias: bool = False,
) -> Path:
    """Create a new profile directory.

    Parameters
    ----------
    name:
        Profile identifier (lowercase, alphanumeric, hyphens, underscores).
    clone_from:
        Source profile to clone from. If ``None`` and clone_config/clone_all
        is True, defaults to the currently active profile.
    clone_all:
        If True, do a full copytree of the source (all state).
    clone_config:
        If True, copy only config files (config.yaml, .env, SOUL.md).
    no_alias:
        If True, skip wrapper script creation.

    Returns
    -------
    Path
        The newly created profile directory.
    """
    validate_profile_name(name)

    if name == "default":
        raise ValueError(
            "Cannot create a profile named 'default' — it is the built-in profile (~/.hermes)."
        )

    profile_dir = get_profile_dir(name)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{name}' already exists at {profile_dir}")

    # Resolve clone source
    source_dir = None
    if clone_from is not None or clone_all or clone_config:
        if clone_from is None:
            # Default: clone from active profile
            from hermes_constants import get_hermes_home
            source_dir = get_hermes_home()
        else:
            validate_profile_name(clone_from)
            source_dir = get_profile_dir(clone_from)
        if not source_dir.is_dir():
            raise FileNotFoundError(
                f"Source profile '{clone_from or 'active'}' does not exist at {source_dir}"
            )

    if clone_all and source_dir:
        # Full copy of source profile
        shutil.copytree(source_dir, profile_dir)
        # Strip runtime files
        for stale in _CLONE_ALL_STRIP:
            (profile_dir / stale).unlink(missing_ok=True)
    else:
        # Bootstrap directory structure
        profile_dir.mkdir(parents=True, exist_ok=True)
        for subdir in _PROFILE_DIRS:
            (profile_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Clone config files from source
        if source_dir is not None:
            for filename in _CLONE_CONFIG_FILES:
                src = source_dir / filename
                if src.exists():
                    shutil.copy2(src, profile_dir / filename)

            # Clone memory and other subdirectory files
            for relpath in _CLONE_SUBDIR_FILES:
                src = source_dir / relpath
                if src.exists():
                    dst = profile_dir / relpath
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

    return profile_dir


def seed_profile_skills(profile_dir: Path, quiet: bool = False) -> Optional[dict]:
    """Seed bundled skills into a profile via subprocess.

    Uses subprocess because sync_skills() caches HERMES_HOME at module level.
    Returns the sync result dict, or None on failure.
    """
    project_root = Path(__file__).parent.parent.resolve()
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import json; from tools.skills_sync import sync_skills; "
             "r = sync_skills(quiet=True); print(json.dumps(r))"],
            env={**os.environ, "HERMES_HOME": str(profile_dir)},
            cwd=str(project_root),
            capture_output=True, text=True, timeout=60,
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


def delete_profile(name: str, yes: bool = False) -> Path:
    """Delete a profile, its wrapper script, and its gateway service.

    Stops the gateway if running. Disables systemd/launchd service first
    to prevent auto-restart.

    Returns the path that was removed.
    """
    validate_profile_name(name)

    if name == "default":
        raise ValueError(
            "Cannot delete the default profile (~/.hermes).\n"
            "To remove everything, use: hermes uninstall"
        )

    profile_dir = get_profile_dir(name)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")

    # Show what will be deleted
    model, provider = _read_config_model(profile_dir)
    gw_running = _check_gateway_running(profile_dir)
    skill_count = _count_skills(profile_dir)

    print(f"\nProfile: {name}")
    print(f"Path:    {profile_dir}")
    if model:
        print(f"Model:   {model}" + (f" ({provider})" if provider else ""))
    if skill_count:
        print(f"Skills:  {skill_count}")

    items = [
        "All config, API keys, memories, sessions, skills, cron jobs",
    ]

    # Check for service
    from hermes_cli.gateway import _profile_suffix, get_service_name
    wrapper_path = _get_wrapper_dir() / name
    has_wrapper = wrapper_path.exists()
    if has_wrapper:
        items.append(f"Command alias ({wrapper_path})")

    print(f"\nThis will permanently delete:")
    for item in items:
        print(f"  • {item}")
    if gw_running:
        print(f"  ⚠ Gateway is running — it will be stopped.")

    # Confirmation
    if not yes:
        print()
        try:
            confirm = input(f"Type '{name}' to confirm: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return profile_dir
        if confirm != name:
            print("Cancelled.")
            return profile_dir

    # 1. Disable service (prevents auto-restart)
    _cleanup_gateway_service(name, profile_dir)

    # 2. Stop running gateway
    if gw_running:
        _stop_gateway_process(profile_dir)

    # 3. Remove wrapper script
    if has_wrapper:
        if remove_wrapper_script(name):
            print(f"✓ Removed {wrapper_path}")

    # 4. Remove profile directory
    try:
        shutil.rmtree(profile_dir)
        print(f"✓ Removed {profile_dir}")
    except Exception as e:
        print(f"⚠ Could not remove {profile_dir}: {e}")

    # 5. Clear active_profile if it pointed to this profile
    try:
        active = get_active_profile()
        if active == name:
            set_active_profile("default")
            print("✓ Active profile reset to default")
    except Exception:
        pass

    print(f"\nProfile '{name}' deleted.")
    return profile_dir


def _cleanup_gateway_service(name: str, profile_dir: Path) -> None:
    """Disable and remove systemd/launchd service for a profile."""
    import platform as _platform

    # Derive service name for this profile
    # Temporarily set HERMES_HOME so _profile_suffix resolves correctly
    old_home = os.environ.get("HERMES_HOME")
    try:
        os.environ["HERMES_HOME"] = str(profile_dir)
        from hermes_cli.gateway import get_service_name, get_launchd_plist_path

        if _platform.system() == "Linux":
            svc_name = get_service_name()
            svc_file = Path.home() / ".config" / "systemd" / "user" / f"{svc_name}.service"
            if svc_file.exists():
                subprocess.run(
                    ["systemctl", "--user", "disable", svc_name],
                    capture_output=True, check=False, timeout=10,
                )
                subprocess.run(
                    ["systemctl", "--user", "stop", svc_name],
                    capture_output=True, check=False, timeout=10,
                )
                svc_file.unlink(missing_ok=True)
                subprocess.run(
                    ["systemctl", "--user", "daemon-reload"],
                    capture_output=True, check=False, timeout=10,
                )
                print(f"✓ Service {svc_name} removed")

        elif _platform.system() == "Darwin":
            plist_path = get_launchd_plist_path()
            if plist_path.exists():
                subprocess.run(
                    ["launchctl", "unload", str(plist_path)],
                    capture_output=True, check=False, timeout=10,
                )
                plist_path.unlink(missing_ok=True)
                print(f"✓ Launchd service removed")
    except Exception as e:
        print(f"⚠ Service cleanup: {e}")
    finally:
        if old_home is not None:
            os.environ["HERMES_HOME"] = old_home
        elif "HERMES_HOME" in os.environ:
            del os.environ["HERMES_HOME"]


def _stop_gateway_process(profile_dir: Path) -> None:
    """Stop a running gateway process via its PID file."""
    import signal as _signal
    import time as _time

    pid_file = profile_dir / "gateway.pid"
    if not pid_file.exists():
        return

    try:
        raw = pid_file.read_text().strip()
        data = json.loads(raw) if raw.startswith("{") else {"pid": int(raw)}
        pid = int(data["pid"])
        os.kill(pid, _signal.SIGTERM)
        # Wait up to 10s for graceful shutdown
        for _ in range(20):
            _time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                print(f"✓ Gateway stopped (PID {pid})")
                return
        # Force kill
        try:
            os.kill(pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
        print(f"✓ Gateway force-stopped (PID {pid})")
    except (ProcessLookupError, PermissionError):
        print("✓ Gateway already stopped")
    except Exception as e:
        print(f"⚠ Could not stop gateway: {e}")


# ---------------------------------------------------------------------------
# Active profile (sticky default)
# ---------------------------------------------------------------------------

def get_active_profile() -> str:
    """Read the sticky active profile name.

    Returns ``"default"`` if no active_profile file exists or it's empty.
    """
    path = _get_active_profile_path()
    try:
        name = path.read_text().strip()
        if not name:
            return "default"
        return name
    except (FileNotFoundError, UnicodeDecodeError, OSError):
        return "default"


def set_active_profile(name: str) -> None:
    """Set the sticky active profile.

    Writes to ``~/.hermes/active_profile``. Use ``"default"`` to clear.
    """
    validate_profile_name(name)
    if name != "default" and not profile_exists(name):
        raise FileNotFoundError(
            f"Profile '{name}' does not exist. "
            f"Create it with: hermes profile create {name}"
        )

    path = _get_active_profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if name == "default":
        # Remove the file to indicate default
        path.unlink(missing_ok=True)
    else:
        # Atomic write
        tmp = path.with_suffix(".tmp")
        tmp.write_text(name + "\n")
        tmp.replace(path)


def get_active_profile_name() -> str:
    """Infer the current profile name from HERMES_HOME.

    Returns ``"default"`` if HERMES_HOME is not set or points to ``~/.hermes``.
    Returns the profile name if HERMES_HOME points into ``~/.hermes/profiles/<name>``.
    Returns ``"custom"`` if HERMES_HOME is set to an unrecognized path.
    """
    from hermes_constants import get_hermes_home
    hermes_home = get_hermes_home()
    resolved = hermes_home.resolve()

    default_resolved = _get_default_hermes_home().resolve()
    if resolved == default_resolved:
        return "default"

    profiles_root = _get_profiles_root().resolve()
    try:
        rel = resolved.relative_to(profiles_root)
        parts = rel.parts
        if len(parts) == 1 and _PROFILE_ID_RE.match(parts[0]):
            return parts[0]
    except ValueError:
        pass

    return "custom"


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

def _default_export_ignore(root_dir: Path):
    """Return an *ignore* callable for :func:`shutil.copytree`.

    At the root level it excludes everything in ``_DEFAULT_EXPORT_EXCLUDE_ROOT``.
    At all levels it excludes ``__pycache__``, sockets, and temp files.
    """

    def _ignore(directory: str, contents: list) -> set:
        ignored: set = set()
        for entry in contents:
            # Universal exclusions (any depth)
            if entry == "__pycache__" or entry.endswith((".sock", ".tmp")):
                ignored.add(entry)
            # npm lockfiles can appear at root
            elif entry in ("package.json", "package-lock.json"):
                ignored.add(entry)
        # Root-level exclusions
        if Path(directory) == root_dir:
            ignored.update(c for c in contents if c in _DEFAULT_EXPORT_EXCLUDE_ROOT)
        return ignored

    return _ignore


def export_profile(name: str, output_path: str) -> Path:
    """Export a profile to a tar.gz archive.

    Returns the output file path.
    """
    import tempfile

    validate_profile_name(name)
    profile_dir = get_profile_dir(name)
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile '{name}' does not exist.")

    output = Path(output_path)
    # shutil.make_archive wants the base name without extension
    base = str(output).removesuffix(".tar.gz").removesuffix(".tgz")

    if name == "default":
        # The default profile IS ~/.hermes itself — its parent is ~/ and its
        # directory name is ".hermes", not "default".  We stage a clean copy
        # under a temp dir so the archive contains ``default/...``.
        with tempfile.TemporaryDirectory() as tmpdir:
            staged = Path(tmpdir) / "default"
            shutil.copytree(
                profile_dir,
                staged,
                ignore=_default_export_ignore(profile_dir),
            )
            result = shutil.make_archive(base, "gztar", tmpdir, "default")
            return Path(result)

    # Named profiles — stage a filtered copy to exclude credentials
    with tempfile.TemporaryDirectory() as tmpdir:
        staged = Path(tmpdir) / name
        _CREDENTIAL_FILES = {"auth.json", ".env"}
        shutil.copytree(
            profile_dir,
            staged,
            ignore=lambda d, contents: _CREDENTIAL_FILES & set(contents),
        )
        result = shutil.make_archive(base, "gztar", tmpdir, name)
        return Path(result)


def _normalize_profile_archive_parts(member_name: str) -> List[str]:
    """Return safe path parts for a profile archive member."""
    normalized_name = member_name.replace("\\", "/")
    posix_path = PurePosixPath(normalized_name)
    windows_path = PureWindowsPath(member_name)

    if (
        not normalized_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
    ):
        raise ValueError(f"Unsafe archive member path: {member_name}")

    parts = [part for part in posix_path.parts if part not in ("", ".")]
    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe archive member path: {member_name}")
    return parts


def _safe_extract_profile_archive(archive: Path, destination: Path) -> None:
    """Extract a profile archive without allowing path escapes or links."""
    import tarfile

    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _normalize_profile_archive_parts(member.name)
            target = destination.joinpath(*parts)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise ValueError(
                    f"Unsupported archive member type: {member.name}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise ValueError(f"Cannot read archive member: {member.name}")

            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)

            try:
                os.chmod(target, member.mode & 0o777)
            except OSError:
                pass


def import_profile(archive_path: str, name: Optional[str] = None) -> Path:
    """Import a profile from a tar.gz archive.

    If *name* is not given, infers it from the archive's top-level directory.
    Returns the imported profile directory.
    """
    import tarfile

    archive = Path(archive_path)
    if not archive.exists():
        raise FileNotFoundError(f"Archive not found: {archive}")

    # Peek at the archive to find the top-level directory name
    with tarfile.open(archive, "r:gz") as tf:
        top_dirs = {
            parts[0]
            for member in tf.getmembers()
            for parts in [_normalize_profile_archive_parts(member.name)]
            if len(parts) > 1 or member.isdir()
        }
        if not top_dirs:
            top_dirs = {
                _normalize_profile_archive_parts(member.name)[0]
                for member in tf.getmembers()
                if member.isdir()
            }

    inferred_name = name or (top_dirs.pop() if len(top_dirs) == 1 else None)
    if not inferred_name:
        raise ValueError(
            "Cannot determine profile name from archive. "
            "Specify it explicitly: hermes profile import <archive> --name <name>"
        )

    # Archives exported from the default profile have "default/" as top-level
    # dir.  Importing as "default" would target ~/.hermes itself — disallow
    # that and guide the user toward a named profile.
    if inferred_name == "default":
        raise ValueError(
            "Cannot import as 'default' — that is the built-in root profile (~/.hermes). "
            "Specify a different name: hermes profile import <archive> --name <name>"
        )

    validate_profile_name(inferred_name)
    profile_dir = get_profile_dir(inferred_name)
    if profile_dir.exists():
        raise FileExistsError(f"Profile '{inferred_name}' already exists at {profile_dir}")

    profiles_root = _get_profiles_root()
    profiles_root.mkdir(parents=True, exist_ok=True)

    _safe_extract_profile_archive(archive, profiles_root)

    # If the archive extracted under a different name, rename
    extracted = profiles_root / (top_dirs.pop() if top_dirs else inferred_name)
    if extracted != profile_dir and extracted.exists():
        extracted.rename(profile_dir)

    return profile_dir


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------

def rename_profile(old_name: str, new_name: str) -> Path:
    """Rename a profile: directory, wrapper script, service, active_profile.

    Returns the new profile directory.
    """
    validate_profile_name(old_name)
    validate_profile_name(new_name)

    if old_name == "default":
        raise ValueError("Cannot rename the default profile.")
    if new_name == "default":
        raise ValueError("Cannot rename to 'default' — it is reserved.")

    old_dir = get_profile_dir(old_name)
    new_dir = get_profile_dir(new_name)

    if not old_dir.is_dir():
        raise FileNotFoundError(f"Profile '{old_name}' does not exist.")
    if new_dir.exists():
        raise FileExistsError(f"Profile '{new_name}' already exists.")

    # 1. Stop gateway if running
    if _check_gateway_running(old_dir):
        _cleanup_gateway_service(old_name, old_dir)
        _stop_gateway_process(old_dir)

    # 2. Rename directory
    old_dir.rename(new_dir)
    print(f"✓ Renamed {old_dir.name} → {new_dir.name}")

    # 3. Update wrapper script
    remove_wrapper_script(old_name)
    collision = check_alias_collision(new_name)
    if not collision:
        create_wrapper_script(new_name)
        print(f"✓ Alias updated: {new_name}")
    else:
        print(f"⚠ Cannot create alias '{new_name}' — {collision}")

    # 4. Update active_profile if it pointed to old name
    try:
        if get_active_profile() == old_name:
            set_active_profile(new_name)
            print(f"✓ Active profile updated: {new_name}")
    except Exception:
        pass

    return new_dir


# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------

def generate_bash_completion() -> str:
    """Generate a bash completion script for hermes profile names."""
    return '''# Hermes Agent profile completion
# Add to ~/.bashrc: eval "$(hermes completion bash)"

_hermes_profiles() {
    local profiles_dir="$HOME/.hermes/profiles"
    local profiles="default"
    if [ -d "$profiles_dir" ]; then
        profiles="$profiles $(ls "$profiles_dir" 2>/dev/null)"
    fi
    echo "$profiles"
}

_hermes_completion() {
    local cur prev
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Complete profile names after -p / --profile
    if [[ "$prev" == "-p" || "$prev" == "--profile" ]]; then
        COMPREPLY=($(compgen -W "$(_hermes_profiles)" -- "$cur"))
        return
    fi

    # Complete profile subcommands
    if [[ "${COMP_WORDS[1]}" == "profile" ]]; then
        case "$prev" in
            profile)
                COMPREPLY=($(compgen -W "list use create delete show alias rename export import" -- "$cur"))
                return
                ;;
            use|delete|show|alias|rename|export)
                COMPREPLY=($(compgen -W "$(_hermes_profiles)" -- "$cur"))
                return
                ;;
        esac
    fi

    # Top-level subcommands
    if [[ "$COMP_CWORD" == 1 ]]; then
        local commands="chat model gateway setup status cron doctor config skills tools mcp sessions profile update version"
        COMPREPLY=($(compgen -W "$commands" -- "$cur"))
    fi
}

complete -F _hermes_completion hermes
'''


def generate_zsh_completion() -> str:
    """Generate a zsh completion script for hermes profile names."""
    return '''#compdef hermes
# Hermes Agent profile completion
# Add to ~/.zshrc: eval "$(hermes completion zsh)"

_hermes() {
    local -a profiles
    profiles=(default)
    if [[ -d "$HOME/.hermes/profiles" ]]; then
        profiles+=("${(@f)$(ls $HOME/.hermes/profiles 2>/dev/null)}")
    fi

    _arguments \\
        '-p[Profile name]:profile:($profiles)' \\
        '--profile[Profile name]:profile:($profiles)' \\
        '1:command:(chat model gateway setup status cron doctor config skills tools mcp sessions profile update version)' \\
        '*::arg:->args'

    case $words[1] in
        profile)
            _arguments '1:action:(list use create delete show alias rename export import)' \\
                        '2:profile:($profiles)'
            ;;
    esac
}

_hermes "$@"
'''


# ---------------------------------------------------------------------------
# Profile env resolution (called from _apply_profile_override)
# ---------------------------------------------------------------------------

def resolve_profile_env(profile_name: str) -> str:
    """Resolve a profile name to a HERMES_HOME path string.

    Called early in the CLI entry point, before any hermes modules
    are imported, to set the HERMES_HOME environment variable.
    """
    validate_profile_name(profile_name)
    profile_dir = get_profile_dir(profile_name)

    if profile_name != "default" and not profile_dir.is_dir():
        raise FileNotFoundError(
            f"Profile '{profile_name}' does not exist. "
            f"Create it with: hermes profile create {profile_name}"
        )

    return str(profile_dir)
