"""
Backup and import commands for hermes CLI.

`hermes backup` creates a zip archive of the entire ~/.hermes/ directory
(excluding the hermes-agent repo and transient files).

`hermes import` restores from a backup zip, overlaying onto the current
HERMES_HOME root.
"""

import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

from hermes_constants import get_default_hermes_root, display_hermes_home


# ---------------------------------------------------------------------------
# Exclusion rules
# ---------------------------------------------------------------------------

# Directory names to skip entirely (matched against each path component)
_EXCLUDED_DIRS = {
    "hermes-agent",     # the codebase repo — re-clone instead
    "__pycache__",      # bytecode caches — regenerated on import
    ".git",             # nested git dirs (profiles shouldn't have these, but safety)
    "node_modules",     # js deps if website/ somehow leaks in
}

# File-name suffixes to skip
_EXCLUDED_SUFFIXES = (
    ".pyc",
    ".pyo",
)

# File names to skip (runtime state that's meaningless on another machine)
_EXCLUDED_NAMES = {
    "gateway.pid",
    "cron.pid",
}


def _should_exclude(rel_path: Path) -> bool:
    """Return True if *rel_path* (relative to hermes root) should be skipped."""
    parts = rel_path.parts

    # Any path component matches an excluded dir name
    for part in parts:
        if part in _EXCLUDED_DIRS:
            return True

    name = rel_path.name

    if name in _EXCLUDED_NAMES:
        return True

    if name.endswith(_EXCLUDED_SUFFIXES):
        return True

    return False


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def _format_size(nbytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def run_backup(args) -> None:
    """Create a zip backup of the Hermes home directory."""
    hermes_root = get_default_hermes_root()

    if not hermes_root.is_dir():
        print(f"Error: Hermes home directory not found at {hermes_root}")
        sys.exit(1)

    # Determine output path
    if args.output:
        out_path = Path(args.output).expanduser().resolve()
        # If user gave a directory, put the zip inside it
        if out_path.is_dir():
            stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            out_path = out_path / f"hermes-backup-{stamp}.zip"
    else:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        out_path = Path.home() / f"hermes-backup-{stamp}.zip"

    # Ensure the suffix is .zip
    if out_path.suffix.lower() != ".zip":
        out_path = out_path.with_suffix(out_path.suffix + ".zip")

    # Ensure parent directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files
    print(f"Scanning {display_hermes_home()} ...")
    files_to_add: list[tuple[Path, Path]] = []  # (absolute, relative)
    skipped_dirs = set()

    for dirpath, dirnames, filenames in os.walk(hermes_root, followlinks=False):
        dp = Path(dirpath)
        rel_dir = dp.relative_to(hermes_root)

        # Prune excluded directories in-place so os.walk doesn't descend
        orig_dirnames = dirnames[:]
        dirnames[:] = [
            d for d in dirnames
            if d not in _EXCLUDED_DIRS
        ]
        for removed in set(orig_dirnames) - set(dirnames):
            skipped_dirs.add(str(rel_dir / removed))

        for fname in filenames:
            fpath = dp / fname
            rel = fpath.relative_to(hermes_root)

            if _should_exclude(rel):
                continue

            # Skip the output zip itself if it happens to be inside hermes root
            try:
                if fpath.resolve() == out_path.resolve():
                    continue
            except (OSError, ValueError):
                pass

            files_to_add.append((fpath, rel))

    if not files_to_add:
        print("No files to back up.")
        return

    # Create the zip
    file_count = len(files_to_add)
    print(f"Backing up {file_count} files ...")

    total_bytes = 0
    errors = []
    t0 = time.monotonic()

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for i, (abs_path, rel_path) in enumerate(files_to_add, 1):
            try:
                zf.write(abs_path, arcname=str(rel_path))
                total_bytes += abs_path.stat().st_size
            except (PermissionError, OSError) as exc:
                errors.append(f"  {rel_path}: {exc}")
                continue

            # Progress every 500 files
            if i % 500 == 0:
                print(f"  {i}/{file_count} files ...")

    elapsed = time.monotonic() - t0
    zip_size = out_path.stat().st_size

    # Summary
    print()
    print(f"Backup complete: {out_path}")
    print(f"  Files:       {file_count}")
    print(f"  Original:    {_format_size(total_bytes)}")
    print(f"  Compressed:  {_format_size(zip_size)}")
    print(f"  Time:        {elapsed:.1f}s")

    if skipped_dirs:
        print(f"\n  Excluded directories:")
        for d in sorted(skipped_dirs):
            print(f"    {d}/")

    if errors:
        print(f"\n  Warnings ({len(errors)} files skipped):")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    print(f"\nRestore with: hermes import {out_path.name}")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def _validate_backup_zip(zf: zipfile.ZipFile) -> tuple[bool, str]:
    """Check that a zip looks like a Hermes backup.

    Returns (ok, reason).
    """
    names = zf.namelist()
    if not names:
        return False, "zip archive is empty"

    # Look for telltale files that a hermes home would have
    markers = {"config.yaml", ".env", "hermes_state.db", "memory_store.db"}
    found = set()
    for n in names:
        # Could be at the root or one level deep (if someone zipped the directory)
        basename = Path(n).name
        if basename in markers:
            found.add(basename)

    if not found:
        return False, (
            "zip does not appear to be a Hermes backup "
            "(no config.yaml, .env, or state databases found)"
        )

    return True, ""


def _detect_prefix(zf: zipfile.ZipFile) -> str:
    """Detect if the zip has a common directory prefix wrapping all entries.

    Some tools zip as `.hermes/config.yaml` instead of `config.yaml`.
    Returns the prefix to strip (empty string if none).
    """
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        return ""

    # Find common prefix
    parts_list = [Path(n).parts for n in names]

    # Check if all entries share a common first directory
    first_parts = {p[0] for p in parts_list if len(p) > 1}
    if len(first_parts) == 1:
        prefix = first_parts.pop()
        # Only strip if it looks like a hermes dir name
        if prefix in (".hermes", "hermes"):
            return prefix + "/"

    return ""


def run_import(args) -> None:
    """Restore a Hermes backup from a zip file."""
    zip_path = Path(args.zipfile).expanduser().resolve()

    if not zip_path.is_file():
        print(f"Error: File not found: {zip_path}")
        sys.exit(1)

    if not zipfile.is_zipfile(zip_path):
        print(f"Error: Not a valid zip file: {zip_path}")
        sys.exit(1)

    hermes_root = get_default_hermes_root()

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Validate
        ok, reason = _validate_backup_zip(zf)
        if not ok:
            print(f"Error: {reason}")
            sys.exit(1)

        prefix = _detect_prefix(zf)
        members = [n for n in zf.namelist() if not n.endswith("/")]
        file_count = len(members)

        print(f"Backup contains {file_count} files")
        print(f"Target: {display_hermes_home()}")

        if prefix:
            print(f"Detected archive prefix: {prefix!r} (will be stripped)")

        # Check for existing installation
        has_config = (hermes_root / "config.yaml").exists()
        has_env = (hermes_root / ".env").exists()

        if (has_config or has_env) and not args.force:
            print()
            print("Warning: Target directory already has Hermes configuration.")
            print("Importing will overwrite existing files with backup contents.")
            print()
            try:
                answer = input("Continue? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)
            if answer not in ("y", "yes"):
                print("Aborted.")
                return

        # Extract
        print(f"\nImporting {file_count} files ...")
        hermes_root.mkdir(parents=True, exist_ok=True)

        errors = []
        restored = 0
        t0 = time.monotonic()

        for member in members:
            # Strip prefix if detected
            if prefix and member.startswith(prefix):
                rel = member[len(prefix):]
            else:
                rel = member

            if not rel:
                continue

            target = hermes_root / rel

            # Security: reject absolute paths and traversals
            try:
                target.resolve().relative_to(hermes_root.resolve())
            except ValueError:
                errors.append(f"  {rel}: path traversal blocked")
                continue

            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                restored += 1
            except (PermissionError, OSError) as exc:
                errors.append(f"  {rel}: {exc}")

            if restored % 500 == 0:
                print(f"  {restored}/{file_count} files ...")

        elapsed = time.monotonic() - t0

        # Summary
        print()
        print(f"Import complete: {restored} files restored in {elapsed:.1f}s")
        print(f"  Target: {display_hermes_home()}")

        if errors:
            print(f"\n  Warnings ({len(errors)} files skipped):")
            for e in errors[:10]:
                print(e)
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

        # Post-import: restore profile wrapper scripts
        profiles_dir = hermes_root / "profiles"
        restored_profiles = []
        if profiles_dir.is_dir():
            try:
                from hermes_cli.profiles import (
                    create_wrapper_script, check_alias_collision,
                    _is_wrapper_dir_in_path, _get_wrapper_dir,
                )
                for entry in sorted(profiles_dir.iterdir()):
                    if not entry.is_dir():
                        continue
                    profile_name = entry.name
                    # Only create wrappers for directories with config
                    if not (entry / "config.yaml").exists() and not (entry / ".env").exists():
                        continue
                    collision = check_alias_collision(profile_name)
                    if collision:
                        print(f"  Skipped alias '{profile_name}': {collision}")
                        restored_profiles.append((profile_name, False))
                    else:
                        wrapper = create_wrapper_script(profile_name)
                        restored_profiles.append((profile_name, wrapper is not None))

                if restored_profiles:
                    created = [n for n, ok in restored_profiles if ok]
                    skipped = [n for n, ok in restored_profiles if not ok]
                    if created:
                        print(f"\n  Profile aliases restored: {', '.join(created)}")
                    if skipped:
                        print(f"  Profile aliases skipped:  {', '.join(skipped)}")
                    if not _is_wrapper_dir_in_path():
                        print(f"\n  Note: {_get_wrapper_dir()} is not in your PATH.")
                        print('  Add to your shell config (~/.bashrc or ~/.zshrc):')
                        print('    export PATH="$HOME/.local/bin:$PATH"')
            except ImportError:
                # hermes_cli.profiles might not be available (fresh install)
                if any(profiles_dir.iterdir()):
                    print(f"\n  Profiles detected but aliases could not be created.")
                    print(f"  Run: hermes profile list  (after installing hermes)")

        # Guidance
        print()
        if not (hermes_root / "hermes-agent").is_dir():
            print("Note: The hermes-agent codebase was not included in the backup.")
            print("  If this is a fresh install, run: hermes update")

        if restored_profiles:
            gw_profiles = [n for n, _ in restored_profiles]
            print("\nTo re-enable gateway services for profiles:")
            for pname in gw_profiles:
                print(f"  hermes -p {pname} gateway install")

        print("Done. Your Hermes configuration has been restored.")
