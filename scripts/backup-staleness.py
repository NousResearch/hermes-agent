#!/usr/bin/env python3
"""Backup Staleness Monitor - alerts when backups are old, missing, or shrinking.

Silent when healthy (exit 0, empty stdout).
Outputs alert text when backup age > 48h, size shrinking, or no backups found.
Also performs cleanup: removes tarballs older than 7 days from daily/ and
removes static snapshot directories older than 30 days.
"""
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BACKUP_DIR = Path(os.environ.get("BACKUP_STALENESS_DIR", "/home/kensei/backups"))
DAILY_DIR = BACKUP_DIR / "daily"
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
STATE_FILE = _HERMES_HOME / "scripts" / ".backup-staleness-state.json"
MAX_AGE_HOURS = 48
TZ = ZoneInfo("Europe/London")
CLEANUP_DAILY_DAYS = 7
CLEANUP_SNAPSHOT_DAYS = 30


def get_latest_backup(backup_dir=None):
    """Find the most recent file in the backup directory.

    Monitors daily/ (the actual rotation output) rather than the top level,
    which contains one-off snapshots and would mask a dead rotation.
    """
    backup_dir = backup_dir if backup_dir is not None else BACKUP_DIR / "daily"
    if not backup_dir.exists():
        return None, 0, []
    files = sorted(backup_dir.iterdir(), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None, 0, []
    latest = files[0]
    age_hours = (datetime.now().timestamp() - latest.stat().st_mtime) / 3600
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    return latest, age_hours, files


def load_state(state_file=None):
    """Load previous state for size comparison."""
    state_file = state_file if state_file is not None else STATE_FILE
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except Exception:
            pass
    return {"last_size": 0, "last_check": None}


def save_state(state, state_file=None):
    """Save current state."""
    state_file = state_file if state_file is not None else STATE_FILE
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


def cleanup_old_daily(backup_dir=None, *, dry_run=False):
    """Remove tarballs in daily/ older than CLEANUP_DAILY_DAYS."""
    deleted = []
    backup_dir = backup_dir if backup_dir is not None else BACKUP_DIR
    daily_dir = backup_dir / "daily"
    if not daily_dir.exists():
        return deleted
    cutoff = datetime.now().timestamp() - (CLEANUP_DAILY_DAYS * 86400)
    for item in daily_dir.iterdir():
        if item.is_file() and item.suffix == ".gz":
            if item.stat().st_mtime < cutoff:
                try:
                    size = item.stat().st_size
                    if not dry_run:
                        item.unlink()
                    deleted.append((item.name, size))
                except Exception as e:
                    print(f"Failed to delete {item}: {e}")
    return deleted


def cleanup_old_snapshots(backup_dir=None, *, dry_run=False):
    """Remove static snapshot directories older than CLEANUP_SNAPSHOT_DAYS."""
    deleted = []
    backup_dir = backup_dir if backup_dir is not None else BACKUP_DIR
    if not backup_dir.exists():
        return deleted
    cutoff = datetime.now().timestamp() - (CLEANUP_SNAPSHOT_DAYS * 86400)
    # Patterns for static snapshots we consider safe to clean
    patterns = ("kensei-phase0-*", "profiles-bak-voicewire-*")
    for pattern in patterns:
        for item in backup_dir.glob(pattern):
            if item.is_dir() and item.stat().st_mtime < cutoff:
                try:
                    # Get size before deletion for logging
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    if not dry_run:
                        shutil.rmtree(item)
                    deleted.append((item.name, size))
                except Exception as e:
                    print(f"Failed to delete snapshot {item}: {e}")
    return deleted


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan cleanup only; never delete files")
    parser.add_argument("--backup-dir", type=Path, default=None,
                        help="Override backup directory (default: env BACKUP_STALENESS_DIR or /home/kensei/backups)")
    parser.add_argument("--state-file", type=Path, default=None,
                        help="Override state file path (default: HERMES_HOME/scripts/.backup-staleness-state.json)")
    parser.add_argument("--hermes-home", type=Path, default=None,
                        help="Override HERMES_HOME used to derive the default state file")
    args = parser.parse_args(argv)
    dry_run = args.dry_run
    backup_dir = args.backup_dir if args.backup_dir is not None else BACKUP_DIR
    if args.hermes_home is not None:
        state_file = args.hermes_home / "scripts" / ".backup-staleness-state.json"
    elif args.state_file is not None:
        state_file = args.state_file
    else:
        state_file = STATE_FILE

    alerts = []
    latest, age_hours, files = get_latest_backup(backup_dir)
    state = load_state(state_file)
    now = datetime.now(TZ)

    if latest is None:
        alerts.append("**No backups found** in " + str(backup_dir) + "/")
    else:
        # Check age
        if age_hours > MAX_AGE_HOURS:
            age_days = age_hours / 24
            alerts.append(f"**Latest backup is {age_days:.1f} days old** - {latest.name} (max: {MAX_AGE_HOURS/24:.0f} days)")
        # Check size trend (shrinking = possible corruption)
        current_size = sum(f.stat().st_size for f in files if f.is_file())
        if state["last_size"] > 0 and current_size > 0:
            if current_size < state["last_size"] * 0.5:  # Shrank by >50%
                alerts.append(f"**Backup size shrinking** - was {state['last_size']/(1024*1024):.1f}MB, now {current_size/(1024*1024):.1f}MB")
        # Check backup count
        if len(files) < 3:
            alerts.append(f"**Only {len(files)} backup files** - expected multiple rotation copies")

    # Perform cleanup
    deleted_daily = cleanup_old_daily(backup_dir, dry_run=dry_run)
    deleted_snapshots = cleanup_old_snapshots(backup_dir, dry_run=dry_run)
    cleanup_msgs = []
    if deleted_daily:
        total_size = sum(sz for _, sz in deleted_daily)
        cleanup_msgs.append(f"Cleaned up {len(deleted_daily)} daily backup tarball(s) (> {CLEANUP_DAILY_DAYS} days), freed {total_size/(1024**3):.2f} GB")
    if deleted_snapshots:
        total_size = sum(sz for _, sz in deleted_snapshots)
        cleanup_msgs.append(f"Cleaned up {len(deleted_snapshots)} static snapshot directory(ies) (> {CLEANUP_SNAPSHOT_DAYS} days), freed {total_size/(1024**3):.2f} GB")

    # Save state for next run (skip in dry-run so state is untouched)
    if not dry_run:
        save_state({
            "last_size": sum(f.stat().st_size for f in files if f.is_file()) if files else 0,
            "last_check": now.isoformat()
        }, state_file)

    if not alerts and not cleanup_msgs:
        sys.exit(0)  # Silent when healthy

    # Output alerts and cleanup info
    if alerts:
        print(f"**🟡 Backup Staleness Alert - {now.strftime('%d/%m/%y %H:%M')}**\n")
        for alert in alerts:
            print(f"• {alert}")
        print()
    if cleanup_msgs:
        print(f"**🔧 Backup Cleanup - {now.strftime('%d/%m/%y %H:%M')}**\n")
        for msg in cleanup_msgs:
            print(f"• {msg}")
        print()

    print("**Action:** Verify backup cron and /home/kensei/backups/ contents.")
    sys.exit(0)


if __name__ == "__main__":
    main()