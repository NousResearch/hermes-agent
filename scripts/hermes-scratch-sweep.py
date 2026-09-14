#!/usr/bin/env python3
"""Weekly age-based cleanup of ~/../var/tmp/hermes-scratch/ (the shared ephemeral scratch
convention for pytest runs, agent scratch work, and e2e tests -- see Backlog.md, "X").

Deletes by ALLOW-LIST, never by exclusion: only ever touches paths under the three named
subdirectories below, and only entries older than the age threshold. Never reads or deletes
anything else in /var/tmp -- that directory also holds systemd-private-* sockets and other
host-level content this script must never see as fair game. A missing keep-marker must never
mean data loss, so there is no keep-marker at all -- age is the only signal, and anything
genuinely persistent belongs in ~/.cache/hermes-shared/ instead, never here.

Silent when nothing is old enough to remove (matches the no_agent cron convention: empty
stdout = no notification). Reports what it deleted otherwise.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from foundation_cron_common import age_hours, local_now

DEFAULT_SCRATCH_ROOT = Path("/var/tmp/hermes-scratch")
ALLOWED_SUBDIRS = ("pytest", "agents", "e2e")
MAX_AGE_DAYS = 7.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--max-age-days", type=float, default=MAX_AGE_DAYS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _candidates(scratch_root: Path) -> list[Path]:
    """Entries eligible for deletion: only top-level items directly under one of the
    three allow-listed subdirectories -- never the scratch root itself, never a sibling
    of it, never anything under a subdirectory not in ALLOWED_SUBDIRS."""
    found: list[Path] = []
    for name in ALLOWED_SUBDIRS:
        sub = scratch_root / name
        if not sub.is_dir():
            continue
        found.extend(p for p in sub.iterdir())
    return found


def main() -> int:
    args = parse_args()
    if not args.scratch_root.is_dir():
        return 0

    now = local_now()
    removed: list[tuple[str, float]] = []
    for path in _candidates(args.scratch_root):
        age_days = age_hours(path, now) / 24
        if age_days < args.max_age_days:
            continue
        size_mb = 0.0
        try:
            if path.is_dir():
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            pass
        if not args.dry_run:
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"  failed to remove {path}: {exc}")
                continue
        removed.append((str(path.relative_to(args.scratch_root)), size_mb))

    if not removed:
        return 0

    total_mb = sum(size for _, size in removed)
    verb = "would remove" if args.dry_run else "removed"
    print(f"hermes-scratch-sweep {verb} {len(removed)} entries, {total_mb / 1024:.2f} GB "
          f"older than {args.max_age_days:g}d:")
    for name, size_mb in sorted(removed, key=lambda e: -e[1]):
        print(f"  {size_mb / 1024:6.2f} GB  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
