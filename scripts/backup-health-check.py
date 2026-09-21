#!/usr/bin/env python3
"""Backup health checker — verifies the most recent backup archive.

Checks: archive exists, is recent (<36h), manifest matches, checksums valid,
         archive is extractable, critical files present.

Silent when healthy. Alerts to #ops on failure.
"""
import os, json, tarfile, hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path

BACKUP_ROOT = Path(os.environ.get("BACKUP_ROOT", os.path.expanduser("~/backups/daily")))
# P13 dry-run: when BACKUP_HEALTH_DRY_RUN=1, skip archive verification
# (opening tarfiles, sha256 computation, manifest parsing) so a disposable
# run never opens a live backup archive. Only the path resolution runs.
_DRY_RUN = os.environ.get("BACKUP_HEALTH_DRY_RUN", "") not in ("", "0", "false")
MAX_AGE_HOURS = 36  # Alert if newest backup is older than this
# Files that must be present — auth.json excluded (secrets need separate encrypted backup)
CRITICAL_FILES = ["config.yaml", "kanban.db"]

def parse_backup_ts(archive: Path) -> str:
    """Extract timestamp from archive name: kensei-YYYYMMDD-HHMM.tar.gz -> YYYYMMDD-HHMM"""
    name = archive.name
    # Remove .tar.gz or .tar suffix
    for suffix in [".tar.gz", ".tar"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name.replace("kensei-", "")

def find_latest_backup() -> tuple[Path, Path] | None:
    """Return (archive_path, manifest_path) of the most recent backup."""
    archives = sorted(BACKUP_ROOT.glob("kensei-*.tar.gz"), reverse=True)
    if not archives:
        return None
    latest = archives[0]
    # Manifest name: kensei-YYYYMMDD-HHMM.manifest.json (no .tar)
    manifest_stem = latest.name.replace(".tar.gz", "").replace(".tar", "")
    manifest = BACKUP_ROOT / f"{manifest_stem}.manifest.json"
    return latest, manifest

def check_age(archive: Path) -> str | None:
    """Return error if archive is too old."""
    ts_str = parse_backup_ts(archive)
    try:
        ts = datetime.strptime(ts_str, "%Y%m%d-%H%M").replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - ts
        if age > timedelta(hours=MAX_AGE_HOURS):
            return f"Archive is {age.total_seconds()/3600:.0f}h old (max {MAX_AGE_HOURS}h)"
    except ValueError:
        return f"Cannot parse timestamp from filename: {archive.name}"
    return None

def check_manifest(archive: Path, manifest_path: Path) -> str | None:
    """Return error if manifest is missing, mismatched, or checksum fails."""
    if not manifest_path.is_file():
        return f"Manifest missing: {manifest_path.name}"
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except json.JSONDecodeError:
        return f"Manifest corrupt: {manifest_path.name}"
    
    # Verify file count matches archive contents
    expected_count = manifest.get("file_count", 0)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            actual_count = len(names)
    except tarfile.TarError:
        return "Cannot open archive for count verification"
    
    if actual_count != expected_count:
        return f"File count mismatch: manifest={expected_count}, archive={actual_count}"
    
    # Verify checksum: compute sha256 of the archive and compare
    import hashlib
    manifest_checksum = manifest.get("sha256")
    if not manifest_checksum:
        return "Manifest has no sha256 checksum — archive integrity unverifiable"
    sha = hashlib.sha256()
    with open(archive, "rb") as af:
        for chunk in iter(lambda: af.read(8192), b""):
            sha.update(chunk)
    actual_checksum = sha.hexdigest()
    if actual_checksum != manifest_checksum:
        return f"Checksum mismatch: manifest={manifest_checksum[:16]}..., actual={actual_checksum[:16]}..."
    
    return None

def check_extractable(archive: Path) -> str | None:
    """Return error if archive cannot be extracted."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
            for cf in CRITICAL_FILES:
                if cf not in names:
                    return f"Critical file missing from archive: {cf}"
    except tarfile.TarError as e:
        return f"Cannot extract archive: {e}"
    return None

def main():
    result = find_latest_backup()
    if result is None:
        if _DRY_RUN:
            print(f"DRY-RUN: no backup archives found in {BACKUP_ROOT} (verification skipped)")
        else:
            print(f"ALERT: No backup archives found in {BACKUP_ROOT}")
        return 0

    archive, manifest_path = result

    if _DRY_RUN:
        print(f"DRY-RUN: would verify {archive.name} (age, manifest, checksum, extractable — all skipped)")
        return 0

    errors = []
    
    err = check_age(archive)
    if err:
        errors.append(err)
    
    err = check_manifest(archive, manifest_path)
    if err:
        errors.append(err)
    
    err = check_extractable(archive)
    if err:
        errors.append(err)
    
    if errors:
        print(f"ALERT: Backup health check failed for {archive.name}")
        for e in errors:
            print(f"  - {e}")
        return 0

    # Silent when healthy
    return 0

if __name__ == "__main__":
    main()
