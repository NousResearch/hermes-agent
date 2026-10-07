"""Fail-closed full backups for explicitly unattended update attempts."""

from __future__ import annotations

import stat
from pathlib import Path

REQUIRED_BACKUP_EXIT = 11


def _directory(path: Path, *, missing_ok: bool = False) -> bool:
    try:
        mode = path.stat().st_mode
    except FileNotFoundError:
        if missing_ok and not path.is_symlink():
            return False
        raise
    if not stat.S_ISDIR(mode):
        raise NotADirectoryError(str(path))
    return True


def _affected_homes() -> list[Path]:
    """Read the live profile roster without the normal best-effort enumeration fallback."""
    from hermes_constants import get_hermes_home, profile_tombstone_path
    from hermes_cli.profiles import (
        _get_default_hermes_home, _get_profiles_root, _iter_named_profile_dirs, _PROFILE_ID_RE,
    )

    root, active = _get_default_hermes_home(), get_hermes_home()
    _directory(root)
    _directory(active)
    profiles = _get_profiles_root()
    if _directory(profiles, missing_ok=True):
        # Path.is_dir()/exists() can hide access errors. Probe candidate directories and
        # tombstones before the canonical identity/roster predicate makes its decision.
        for entry in sorted(profiles.iterdir()):
            if entry.name == "default" or not _PROFILE_ID_RE.fullmatch(entry.name):
                continue
            if stat.S_ISDIR(entry.stat().st_mode):
                list(entry.iterdir())
                tombstone = profile_tombstone_path(entry)
                try:
                    tombstone.stat()
                except FileNotFoundError:
                    pass
    homes = [root, active, *_iter_named_profile_dirs()]
    return list(dict.fromkeys(home.resolve(strict=True) for home in homes))


def _record_backups(paths: dict[str, str], *, error: str = "") -> None:
    from hermes_cli import update_receipt

    update_receipt.record_fact("required_backups", paths)
    update_receipt.record_step("required_backup", not error, error or "; ".join(paths.values()))
    update_receipt.record_stage("required_backup", "failed" if error else "success")
    if error:
        return
    correlation = update_receipt.current_correlation_id()
    stored = update_receipt.read_run_record(correlation) if correlation else None
    if stored is None or stored[1].get("required_backups") != paths:
        raise OSError("could not persist the required-backup receipt")


def require_pre_update_backups(args) -> dict[str, str]:
    """Require complete full archives before the normal quick-snapshot/recovery phase."""
    from hermes_cli.backup import create_pre_update_backup
    from hermes_cli.update_cmd import _updates_config

    paths: dict[str, str] = {}
    try:
        if getattr(args, "no_backup", False):
            raise ValueError("--require-backup cannot be combined with --no-backup")
        homes = _affected_homes()
        keep = max(1, int(_updates_config().get("backup_keep", 5)))
        for home in homes:
            archive = create_pre_update_backup(hermes_home=home, keep=keep, strict=True)
            if archive is None or not archive.is_file():
                raise OSError(f"required full backup missing or incomplete for {home}")
            paths[str(home)] = str(archive)
        if _affected_homes() != homes:
            raise OSError("profile roster changed during required backups; retry the update")
        _record_backups(paths)
    except Exception as exc:
        _record_backups(paths, error=str(exc))
        print(f"✗ Required pre-update backup failed: {exc}. No update was applied.")
        raise SystemExit(REQUIRED_BACKUP_EXIT) from exc
    print(f"◆ Required full backup(s): {', '.join(paths.values())}")
    return paths
