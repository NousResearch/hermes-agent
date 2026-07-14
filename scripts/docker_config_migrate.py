#!/usr/bin/env python3
"""Run Docker boot-time config migrations safely."""
from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hermes_cli.config import (
    check_config_version,
    get_config_path,
    get_env_path,
    migrate_config,
)
from utils import env_var_enabled


def _backup_path(path: Path, stamp: str) -> Path:
    base = path.with_name(f"{path.name}.bak-{stamp}")
    if not base.exists():
        return base
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}.bak-{stamp}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not choose a backup path for {path}")


def _backup_existing(paths: Iterable[Path]) -> dict[Path, Path]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backups: dict[Path, Path] = {}
    for path in paths:
        if not path.is_file():
            continue
        dest = _backup_path(path, stamp)
        shutil.copy2(path, dest)
        backups[path] = dest
    return backups


def _restore_backups(backups: dict[Path, Path]) -> list[Path]:
    restored: list[Path] = []
    for original, backup in backups.items():
        if not backup.is_file():
            continue
        shutil.copy2(backup, original)
        restored.append(original)
    return restored


def main() -> int:
    if env_var_enabled("HERMES_SKIP_CONFIG_MIGRATION"):
        print("[config-migrate] HERMES_SKIP_CONFIG_MIGRATION is set; skipping config migration")
        return 0

    current_ver, latest_ver = check_config_version()
    if current_ver >= latest_ver:
        return 0

    backups = _backup_existing((get_config_path(), get_env_path()))
    backup_text = ", ".join(str(path) for path in backups.values()) if backups else "none"
    print(
        f"[config-migrate] Migrating config schema {current_ver} -> {latest_ver}; "
        f"backups: {backup_text}"
    )
    try:
        migrate_config(interactive=False, quiet=False)
    except Exception:
        restored = _restore_backups(backups)
        if restored:
            print(
                "[config-migrate] Migration failed; restored "
                + ", ".join(str(path) for path in restored)
            )
        raise

    post_ver, _ = check_config_version()
    if post_ver < latest_ver:
        restored = _restore_backups(backups)
        restored_text = ", ".join(str(path) for path in restored) if restored else "none"
        raise RuntimeError(
            f"migration did not advance config version to {latest_ver} "
            f"(still {post_ver}); restored: {restored_text}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[config-migrate] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
