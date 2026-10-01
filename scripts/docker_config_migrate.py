#!/usr/bin/env python3
"""Run Docker boot-time config migrations safely."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

from hermes_cli.config import (
    InvalidUserConfigError,
    _read_config_version_stamp,
    check_config_version,
    get_config_path,
    get_env_path,
    migrate_config,
)
from hermes_cli.config_backups import backup_config, list_config_backups
from hermes_cli.config_migrations import (
    SUPPORT_FLOOR_VERSION,
    support_floor_message,
)
from utils import env_var_enabled


def _backup_existing(paths: Iterable[Path]) -> dict[Path, Path]:
    """Snapshot each file into backups/config/; an identical existing snapshot is reused."""
    backups: dict[Path, Path] = {}
    for path in paths:
        dest = backup_config(path, "pre-docker-migrate") or next(
            iter(list_config_backups(path, "pre-docker-migrate")), None)
        if dest is not None:
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

    # Strict read: malformed YAML or a non-mapping root is left alone with a warning and the
    # boot continues, instead of running the backup/migrate dance that migrate_config() would
    # refuse anyway.
    try:
        stamp, latest_ver = _read_config_version_stamp(raise_on_parse_error=True)
    except InvalidUserConfigError as exc:
        print(f"[config-migrate] WARNING: {exc}; leaving config.yaml untouched", file=sys.stderr)
        return 0
    current_ver = 0 if stamp is None else stamp
    if current_ver >= latest_ver:
        return 0

    # Below the auto-migration support floor: migrate_config() refuses (and
    # leaves the file untouched), so don't run the backup/verify dance that
    # would raise "did not advance config version" and block the boot.
    # Warn-and-continue matches the CLI's fail-safe posture. A config with no
    # _config_version (stamp None: a volume seeded from the template) is not
    # below the floor: migrate_config() stamps it.
    if stamp is not None and current_ver < SUPPORT_FLOOR_VERSION:
        print(
            f"[config-migrate] WARNING: {support_floor_message()}",
            file=sys.stderr,
        )
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
        raise SystemExit(1)
