"""
Core utilities for Hermes migration commands.

Shared between export, import, verify, and doctor subcommands.
"""

import json
import os
import platform
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from hermes_cli.colors import Colors, color
from hermes_constants import get_hermes_home


HERMES_HOME = get_hermes_home()  # cached at import time for backwards compat
# All internal code uses get_hermes_home() (below) to pick up the env var at
# runtime. This is required so that tests can patch get_hermes_home() and get
# a fresh value — patching the cached module-level constant breaks under
# pytest-xdist parallel workers because module state is shared per-worker.
BUNDLE_VERSION = "1.0"

# Files/directories that are NEVER migrated (platform-specific or runtime)
_EXCLUDE_ALWAYS = frozenset({
    "hermes-agent",
    ".git",
    ".worktrees",
    "node_modules",
    "state.db",
    "state.db-shm",
    "state.db-wal",
    "hermes_state.db",
    "response_store.db",
    "response_store.db-shm",
    "response_store.db-wal",
    "gateway.pid",
    "gateway_state.json",
    "processes.json",
    "auth.lock",
    ".update_check",
    "errors.log",
    ".hermes_history",
    "__pycache__",
})

_SECRET_FILES = frozenset({".env", "auth.json"})

_PLATFORM_SKIP = {
    # windows: kept (Windows scripts migrate normally)
    "linux": {".ps1", ".bat", ".cmd"},
    "macos": {".ps1", ".bat", ".cmd"},
    "wsl": {".ps1", ".bat", ".cmd"},
}

# External tool dependencies that may need manual setup on target
_EXTERNAL_TOOLS = frozenset({
    "docker", "git", "node", "npm", "python3", "pip3",
    "ffmpeg", "ssh", "scp", "rsync", "curl", "wget",
})


@dataclass
class MigrationReport:
    """Structured report of a migration operation."""
    migrated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    needs_reauth: list[str] = field(default_factory=list)
    incompatible: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.migrated, self.skipped,
            self.needs_reauth, self.incompatible
        ])


def _log_warning(msg: str) -> None:
    """Log a warning during migration operations."""
    print(color(f"  ⚠ {msg}", Colors.YELLOW))


def _log_error(msg: str) -> None:
    """Log an error during migration operations."""
    print(color(f"  ✗ {msg}", Colors.RED))


def detect_platform() -> dict:
    """Detect current platform and home directory."""
    system = platform.system().lower()
    if system == "windows":
        return {
            "os": "windows",
            "home": Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))),
        }
    elif system == "darwin":
        return {
            "os": "macos",
            "home": get_home(),
        }
    else:
        kernel_release = platform.release().lower()
        if "microsoft" in kernel_release or "wsl" in kernel_release:
            home = Path(os.environ.get("HOME", os.path.expanduser("~")))
            return {"os": "wsl", "home": home}
        return {
            "os": "linux",
            "home": get_home(),
        }


def get_home() -> Path:
    """Get current user's home directory."""
    return Path(os.path.expanduser("~"))


def create_manifest(preset: str, source_platform: dict) -> dict:
    """Create migration manifest metadata."""
    return {
        "version": BUNDLE_VERSION,
        "bundle_created_at": datetime.now(timezone.utc).isoformat(),
        "hermes_version": _get_hermes_version(),
        "source_os": source_platform["os"],
        "source_home": str(source_platform["home"]),
        "preset": preset,
        "includes_secrets": preset == "full",
    }


def _get_hermes_version() -> str:
    """Get installed Hermes version."""
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:
        return "unknown"


def _should_skip_dir(name: str, os_type: str) -> bool:
    """Check if directory should be skipped during migration."""
    if name in _EXCLUDE_ALWAYS:
        return True
    return False


def _should_skip_file(name: str, os_type: str) -> bool:
    """Check if file should be skipped during migration."""
    if name.startswith(".") and name not in _SECRET_FILES:
        return True
    if name in _EXCLUDE_ALWAYS:
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext in _PLATFORM_SKIP.get(os_type, set()):
        return True
    return False


def _is_secret(rel_path: str, preset: str) -> bool:
    """Return True if the path resolves to a secret file under safe preset."""
    if preset != "safe":
        return False
    return Path(rel_path).name in _SECRET_FILES


def _is_text_file(name: str) -> bool:
    """Check if file is a text file that may contain paths."""
    text_extensions = {".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".env", ".toml", ".ini", ".cfg"}
    return os.path.splitext(name)[1].lower() in text_extensions or name in {".env", "config.yaml", "SOUL.md"}


def _remap_content(content: str, source_home: Path, target_home: Path) -> str:
    """Remap home directory paths in text content.

    Uses word-boundary regex replacement to avoid replacing the path prefix
    when it appears inside strings or comments.
    """
    source_str = str(source_home).replace("\\", "/")
    target_str = str(target_home).replace("\\", "/")
    # Normalize content to forward slashes too, so Windows paths match
    normalized_content = content.replace("\\", "/")
    if source_str == target_str or source_str not in normalized_content:
        return content
    # Match at a path boundary: / (separator), end of string, whitespace,
    # or quote character. Using a positive lookahead instead of negative
    # (?!\w) avoids false positives on paths like /home/user-backup where
    # the hyphen is not a word character and would incorrectly trigger
    # replacement of /home/user inside it.
    boundary = re.escape(source_str) + r"(?=/|$|\s|[:'\"])"
    result = re.sub(boundary, target_str, normalized_content)
    # Restore original backslashes in content that wasn't remapped
    if result == normalized_content:
        return content
    return result


def _collect_migration_items(preset: str) -> dict:
    """Collect list of items to migrate with their status."""
    items = {}
    hermes_home = get_hermes_home()

    for dirname in ["memories", "sessions", "skills", "profiles", "hooks", "cron"]:
        dest = hermes_home / dirname
        items[dirname] = {
            "type": "directory",
            "status": "migrated" if dest.exists() else "skipped",
            "reason": "not found" if not dest.exists() else None,
        }

    for fname in ["config.yaml", "SOUL.md"]:
        dest = hermes_home / fname
        items[fname] = {
            "type": "file",
            "status": "migrated" if dest.exists() else "skipped",
            "reason": "not found" if not dest.exists() else None,
        }

    for fname in [".env", "auth.json"]:
        dest = hermes_home / fname
        if fname in _SECRET_FILES and preset != "full":
            items[fname] = {"type": "file", "status": "skipped", "reason": "secrets excluded (use --preset full)"}
        else:
            items[fname] = {
                "type": "file",
                "status": "migrated" if dest.exists() else "skipped",
                "reason": "not found" if not dest.exists() else None,
            }

    return items
