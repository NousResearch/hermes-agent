from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from .paths import AppPaths, build_paths, ensure_app_dirs


DEFAULT_INCLUDE_PATTERNS = ["**/*"]
DEFAULT_EXCLUDE_PATTERNS = [
    "audio_cache/**",
    "cron/output/**",
    "**/*.log",
    "**/__pycache__/**",
    "**/tmp/**",
    # Python virtual environments — extremely large, contain thousands of
    # platform-specific binary packages; snapshotting them is数十GB for no reason.
    ".venv/**",
    "venv/**",
    ".virtualenv/**",
    "**/site-packages/**",
    # Node modules — similarly huge and irrelevant to Hermes state.
    "**/node_modules/**",
    # Python build artifacts and egg-info.
    "**/*.egg-info/**",
    "**/build/**",
    "**/dist/**",
    # Editor and OS metadata — machine-specific, not part of Hermes state.
    ".vscode/**",
    ".idea/**",
    ".DS_Store",
    "Thumbs.db",
    # Version control — .git is massive and Hermes uses its own SQLite store.
    # Use **/.git/** (not .git/**) so it matches nested paths like src/.git/.
    "**/.git/**",
]

# Patterns matched against the relative path of a file. Matching files are
# skipped immediately without retry during snapshot creation.  Use this to
# skip known-problematic paths such as network mounts, socket files, or
# FUSE filesystems that are known to hang.
DEFAULT_SKIP_STUCK_PATTERNS: list[str] = []

# Default max consecutive stuck-file skips before copy aborts.
DEFAULT_MAX_CONSECUTIVE_SKIPPED = 20

# Default list of filesystem types (fstype from /proc/mounts) to skip during
# traversal.  Network-backed and virtual filesystems are excluded by default
# because they can hang on unresponsive servers or contain no real files.
DEFAULT_SKIP_FS_TYPES: list[str] = [
    "proc", "sysfs", "devpts", "devtmpfs", "cgroup", "cgroup2",
    "autofs", "pstore", "securityfs", "debugfs", "hugetlbfs",
    "mqueue", "fusectl", "configfs", "binfmt_misc",
    "nfs", "nfs4", "cifs", "smb", "smb3", "sshfs",
    "fuse", "fuseblk", "overlay", "aufs", "btrfs",
    "container", "containerd", "crio", "runtime",
    # WSL / Plan 9 shared-folder protocol
    "9p",
]

# Default mount-point prefixes to skip during traversal (independent of
# filesystem type).  These are almost always virtual / system mounts.
# /tmp is intentionally excluded because it may live on the same filesystem
# as the user's source root; add it explicitly if you want to skip it.
DEFAULT_SKIP_MOUNT_POINTS: list[str] = [
    "/proc", "/sys", "/dev", "/dev/pts", "/dev/shm",
    "/run", "/run/lock",
    "/mnt", "/media", "/snap",
]

# Default: do NOT cross device/mount boundaries during traversal.
# Setting this to True can cause huge or infinite traversals when
# a bind mount points outside the source tree.
DEFAULT_CROSS_DEVICE = False


@dataclass
class SnapshotManagerSettings:
    source_root: str
    snapshot_root: str
    schedule_enabled: bool = False
    schedule_cron: str = "0 */6 * * *"
    retention_hourly: int = 24
    retention_daily: int = 30
    retention_weekly: int = 12
    include_patterns: list[str] = field(default_factory=lambda: DEFAULT_INCLUDE_PATTERNS.copy())
    exclude_patterns: list[str] = field(default_factory=lambda: DEFAULT_EXCLUDE_PATTERNS.copy())
    skip_stuck_patterns: list[str] = field(default_factory=lambda: DEFAULT_SKIP_STUCK_PATTERNS.copy())
    max_consecutive_skipped: int = DEFAULT_MAX_CONSECUTIVE_SKIPPED
    skip_fs_types: list[str] = field(default_factory=lambda: DEFAULT_SKIP_FS_TYPES.copy())
    skip_mount_points: list[str] = field(default_factory=lambda: DEFAULT_SKIP_MOUNT_POINTS.copy())
    cross_device: bool = DEFAULT_CROSS_DEVICE

    @classmethod
    def from_paths(cls, paths: AppPaths) -> "SnapshotManagerSettings":
        return cls(source_root=str(paths.source_root), snapshot_root=str(paths.snapshot_root))


def _merge_unique(existing: list[str] | None, defaults: list[str]) -> list[str]:
    merged = list(existing or [])
    for item in defaults:
        if item not in merged:
            merged.append(item)
    return merged


def load_settings(paths: AppPaths | None = None) -> SnapshotManagerSettings:
    app_paths = paths or build_paths()
    ensure_app_dirs(app_paths)
    settings_path = app_paths.settings_path
    if not settings_path.exists():
        settings = SnapshotManagerSettings.from_paths(app_paths)
        save_settings(settings, app_paths)
        return settings

    try:
        raw = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, JSONDecodeError):
        settings = SnapshotManagerSettings.from_paths(app_paths)
        save_settings(settings, app_paths)
        return settings
    defaults = asdict(SnapshotManagerSettings.from_paths(app_paths))
    merged = {**defaults, **raw}

    # Migrate older settings files forward by appending newly introduced safe
    # defaults instead of freezing the exclude/skip lists forever at the moment
    # the file was first created.
    merged["exclude_patterns"] = _merge_unique(raw.get("exclude_patterns"), DEFAULT_EXCLUDE_PATTERNS)
    merged["skip_fs_types"] = _merge_unique(raw.get("skip_fs_types"), DEFAULT_SKIP_FS_TYPES)
    merged["skip_mount_points"] = _merge_unique(raw.get("skip_mount_points"), DEFAULT_SKIP_MOUNT_POINTS)

    settings = SnapshotManagerSettings(**merged)
    save_settings(settings, app_paths)
    return settings


def save_settings(settings: SnapshotManagerSettings, paths: AppPaths | None = None) -> None:
    app_paths = paths or build_paths(Path(settings.source_root), Path(settings.snapshot_root).parent)
    ensure_app_dirs(app_paths)
    app_paths.settings_path.write_text(json.dumps(asdict(settings), indent=2, sort_keys=True), encoding="utf-8")


def update_settings(data: dict[str, Any], paths: AppPaths | None = None) -> SnapshotManagerSettings:
    app_paths = paths or build_paths()
    current = load_settings(app_paths)
    merged = {**asdict(current), **data}
    settings = SnapshotManagerSettings(**merged)
    save_settings(settings, app_paths)
    return settings
