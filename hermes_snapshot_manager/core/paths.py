from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    source_root: Path
    app_root: Path
    snapshot_root: Path
    export_root: Path
    log_root: Path
    db_path: Path
    settings_path: Path
    lock_path: Path
    operation_status_path: Path


def default_source_root() -> Path:
    return Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")).expanduser().resolve()


def default_app_root() -> Path:
    return Path(os.environ.get("HERMES_SNAPSHOT_HOME", "/mnt/d/hermes_snapshot")).expanduser().resolve()


def build_paths(source_root: Path | None = None, app_root: Path | None = None) -> AppPaths:
    source = (source_root or default_source_root()).expanduser().resolve()
    root = (app_root or default_app_root()).expanduser().resolve()
    return AppPaths(
        source_root=source,
        app_root=root,
        snapshot_root=root / "snapshots",
        export_root=root / "exports",
        log_root=root / "logs",
        db_path=root / "catalog.db",
        settings_path=root / "settings.json",
        lock_path=root / ".snapshot.lock",
        operation_status_path=root / "operation-status.json",
    )


def ensure_app_dirs(paths: AppPaths) -> None:
    for path in (paths.app_root, paths.snapshot_root, paths.export_root, paths.log_root):
        path.mkdir(parents=True, exist_ok=True)
