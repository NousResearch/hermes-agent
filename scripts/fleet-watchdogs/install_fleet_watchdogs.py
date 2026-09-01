#!/usr/bin/env python3
"""Deploy-time installer for fleet watchdog scripts (copy-on-merge only).

The fleet watchdogs' source of truth is scripts/fleet-watchdogs/ in the
hermes-agent repo, but the cron scheduler actually runs them from the live
scripts dir (~/.hermes/scripts/). That live dir is outside the repo's auto
merge+restart path, so a watchdog change merged to main would sit uninstalled
until it was copied by hand — exactly the 09-01 bob-blocker-watch gap.

The deploy (merge + restart gateway) path calls this script for each watchdog
file that changed in the merged range. Guarantees:

  * Copy-on-merge only — never a symlink (the scheduler rejects symlinks).
  * Only the files explicitly listed are touched in the destination. It never
    creates, deletes, or otherwise touches any file the merge did not change.
  * Idempotent: if a listed source already matches the destination, nothing
    happens for that file (no re-copy, no extra backup).
  * A file gets a timestamped .bak-predeploy-<ts> backup only when it is
    actually replaced with differing content.

Usage (from a deploy path; paths are repo-relative watchdog files changed in
the merged range, e.g. from `git diff --name-only main...<branch> -- scripts/fleet-watchdogs/`):

    python3 install_fleet_watchdogs.py --dest ~/.hermes/scripts <changed...>

Exit code 0 on success; nonzero on any error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

DEFAULT_DEST = Path.home() / ".hermes" / "scripts"

# Repo-relative root of the fleet watchdog scripts (their source of truth).
WATCHDOG_DIR = "scripts/fleet-watchdogs/"


def select_watchdog_files(diff_names):
    """Filter a git change-list to the ones under the watchdog dir.

    ``diff_names`` is an iterable of repo-relative paths (e.g. the stdout of
    ``git diff --name-only``). Returns the sorted subset whose path sits under
    ``WATCHDOG_DIR``. Used so a deploy installs *only* the watchdog files that
    actually changed in the merged range, and installs nothing when the range
    contains no watchdog changes.
    """
    out = []
    for name in diff_names:
        name = (name or "").strip().replace("\\", "/")
        # Match both repo-relative ("scripts/fleet-watchdogs/...") and absolute
        # (".../repo/scripts/fleet-watchdogs/...") paths that sit under the
        # watchdog dir. The trailing slash in WATCHDOG_DIR keeps a same-named
        # sibling file (e.g. ".../fleet-watchdogs.txt") from matching.
        if WATCHDOG_DIR in name:
            out.append(name)
    return sorted(set(out))


def install_file(src: Path, dest_dir: Path):
    """Copy ``src`` into ``dest_dir/<basename>`` with a pre-deploy backup.

    Returns a tuple ``(status, dest, backup)``:
      * ("installed", dest, backup|None) — source differed and was copied.
      * ("skipped", dest, None)          — destination already matches source.

    Raises FileNotFoundError if the source is missing, ValueError for a
    basename that is not a plain file name.
    """
    src_p = Path(src).expanduser().resolve()
    dest_dir_p = Path(dest_dir).expanduser().resolve()

    if not src_p.is_file():
        raise FileNotFoundError(f"source not a file: {src_p}")

    name = src_p.name
    if name in ("", ".", ".."):
        raise ValueError(f"refusing basename {name!r}")

    dest_p = dest_dir_p / name
    backup: Path | None = None

    if dest_p.is_file():
        if dest_p.read_bytes() == src_p.read_bytes():
            # Idempotent: already at the target revision. Reconcile the exec
            # bit anyway — cron needs it, and a hand-edited live copy may have
            # lost +x while staying byte-identical.
            src_mode = src_p.stat().st_mode
            if os.stat(dest_p).st_mode & 0o777 != src_mode & 0o777:
                os.chmod(dest_p, src_mode & 0o777)
            return ("skipped", dest_p, None)
        ts = int(time.time())
        backup = dest_dir_p / f"{dest_p.name}.bak-predeploy-{ts}"
        n = 1
        while backup.exists():
            backup = dest_dir_p / f"{dest_p.name}.bak-predeploy-{ts}-{n}"
            n += 1

    dest_dir_p.mkdir(parents=True, exist_ok=True)
    if backup is not None:
        shutil.copy2(dest_p, backup)
    shutil.copy2(src_p, dest_p)
    return ("installed", dest_p, backup)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Install changed fleet watchdog scripts into the live scripts dir."
    )
    ap.add_argument(
        "files",
        nargs="*",
        help="absolute or repo paths of changed files (watchdog files only are installed)",
    )
    ap.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="live scripts dir (default ~/.hermes/scripts)",
    )
    args = ap.parse_args(argv)

    dest_dir = Path(args.dest).expanduser().resolve()

    # Scope the input to watchdog files only (self-enforced, not left to the
    # caller's git pathspec) and treat an empty change-list as a clean no-op:
    # a no-watchdog-change merge must not exit nonzero.
    to_install = select_watchdog_files(args.files)
    if not to_install:
        print("no watchdog changes to install")
        return 0

    installed: list[tuple[str, str | None]] = []
    for f in to_install:
        status, dst, backup = install_file(Path(f), dest_dir)
        if status == "installed":
            bak_name = backup.name if backup is not None else None
            installed.append((dst.name, bak_name))
            if bak_name:
                print(f"installed {dst.name} (backup {bak_name})")
            else:
                print(f"installed {dst.name}")
        else:
            print(f"skipped {dst.name} (identical)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
