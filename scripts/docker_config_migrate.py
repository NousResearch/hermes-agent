#!/usr/bin/env python3
"""Run Docker boot-time config migrations safely."""
from __future__ import annotations

import errno
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable

from hermes_cli.config import (
    InvalidUserConfigError,
    get_config_path,
    get_env_path,
    migrate_config,
)
from hermes_cli.config_version_stamp import check_config_version, read_config_version_stamp, stamp_missing_config
from hermes_cli.config_backups import backup_config, list_config_backups
from hermes_cli.config_migrations import (
    SUPPORT_FLOOR_VERSION,
    support_floor_message,
)
from utils import env_var_enabled


try:
    _EROFS = errno.EROFS
except AttributeError:  # Windows: the CRT has no read-only-filesystem errno
    _EROFS = None


def _write_blocker(path: Path) -> str | None:
    """Short reason *path* cannot be opened for writing, or None when it can.

    Probes with O_RDWR (no creat/trunc): a read-only bind mount or read-only
    filesystem fails here without touching a byte, where os.access() would lie
    (it reports the mode bits, and passes for root even on EROFS mounts).
    A missing file is not a blocker — the migration creates it.
    """
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if _EROFS is not None and exc.errno == _EROFS:
            return f"read-only file system ({exc})"
        if exc.errno in (errno.EACCES, errno.EPERM):
            return f"not writable by this user ({exc})"
        return str(exc)
    os.close(fd)
    return None


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
        try:
            shutil.copy2(backup, original)
        except OSError as exc:
            # The failure that triggered the rollback (e.g. a read-only mount)
            # usually breaks the restore the same way — report it, but never
            # let it mask the original migration error below.
            print(
                f"[config-migrate] WARNING: could not restore {original} "
                f"from {backup}: {exc}",
                file=sys.stderr,
            )
            continue
        restored.append(original)
    return restored


def main() -> int:
    if env_var_enabled("HERMES_SKIP_CONFIG_MIGRATION"):
        print("[config-migrate] HERMES_SKIP_CONFIG_MIGRATION is set; skipping config migration")
        return 0

    # Strict read: malformed YAML or a non-mapping root is left alone with a warning and the
    # boot continues, instead of running the backup/migrate dance that migrate_config() would
    # refuse anyway. A missing config.yaml reads as current, so its legacy browser settings get their
    # first stamp before that read.
    stamp_missing_config()
    try:
        stamp, latest_ver = read_config_version_stamp(raise_on_parse_error=True)
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

    config_path = get_config_path()
    blocker = _write_blocker(config_path) if config_path.is_file() else None
    if blocker is not None:
        # A read-only config.yaml (a `:ro` file bind, or a Kubernetes
        # ConfigMap/secret subPath mount, which is always read-only) fails
        # every persisting step plus the final stamp — and the old path still
        # printed "Config version: X → Y" although nothing was persisted.
        # Skip loudly instead: the gateway deep-merges defaults at read time,
        # so the stale schema still boots.
        print(
            f"[config-migrate] WARNING: skipping migration {current_ver} -> {latest_ver}: "
            f"{config_path} is not writable ({blocker}). The gateway will run with the "
            f"on-disk schema. Remount the file writable, or set "
            f"HERMES_SKIP_CONFIG_MIGRATION=1 if the schema is managed externally.",
            file=sys.stderr,
        )
        return 0

    backups = _backup_existing((config_path, get_env_path()))
    backup_text = ", ".join(str(path) for path in backups.values()) if backups else "none"
    print(
        f"[config-migrate] Migrating config schema {current_ver} -> {latest_ver}; "
        f"backups: {backup_text}"
    )
    try:
        results = migrate_config(interactive=False, quiet=False) or {}
    except Exception:
        restored = _restore_backups(backups)
        if restored:
            print(
                "[config-migrate] Migration failed; restored "
                + ", ".join(str(path) for path in restored)
            )
        raise

    post_ver, _ = check_config_version()
    failed = results.get("failed_step")
    if failed is not None:
        # The steps before it stay; the next container start retries the rest.
        print(f"[config-migrate] WARNING: migration stopped at v{failed}; kept {current_ver} -> {post_ver}",
              file=sys.stderr)
        return 0
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
