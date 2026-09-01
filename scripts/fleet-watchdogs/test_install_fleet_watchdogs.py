#!/usr/bin/env python3
"""Executable tests for scripts/fleet-watchdogs/install_fleet_watchdogs.py.

Zero-LLM, stdlib-only, runnable standalone:

    python3 test_install_fleet_watchdogs.py

The module under test is imported by file path so this runs identically whether
the script lives in the repo worktree or in the live scripts dir.

Covers the copy-on-merge guarantees (2026-09-01):
1. Changed watchdog file is installed with a timestamped .bak-predeploy-<ts> backup.
2. Unrelated scripts in the live dir are untouched.
3. Idempotent: re-installing an already-current file copies nothing and makes no backup.
4. Installing a file the dest already matches = skipped, no backup.
5. Copy-on-merge only: the scheduler rejects symlinks, so install must be a real copy
   (dest is a regular file, not a symlink).
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "install_fleet_watchdogs.py")
sys.path.insert(0, _HERE)

_spec = importlib.util.spec_from_file_location("ifw", _TARGET)
assert _spec is not None and _spec.loader is not None, f"cannot load {_TARGET}"
ifw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ifw)

FAILURES = []


def check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'} {name}: got={got!r} want={want!r}")
    if not ok:
        FAILURES.append((name, got, want))
    return ok


def test_changed_file_installed_with_backup():
    """A watchdog source differing from the live copy is installed, and the
    replaced live copy gets a .bak-predeploy-<ts> backup."""
    print("test_changed_file_installed_with_backup")
    with tempfile.TemporaryDirectory() as d:
        src_dir = os.path.join(d, "src")
        dest_dir = os.path.join(d, "live")
        os.makedirs(src_dir)
        os.makedirs(dest_dir)

        src = os.path.join(src_dir, "watch.sh")
        dest = os.path.join(dest_dir, "watch.sh")
        with open(src, "w", encoding='utf-8') as f:
            f.write("#!/bin/sh\n# NEW version\n")
        with open(dest, "w", encoding='utf-8') as f:
            f.write("#!/bin/sh\n# OLD version\n")

        status, dst, backup = ifw.install_file(src, dest_dir)

        check("status installed", status, "installed")
        check("same dest path", str(dst), str(Path(dest).resolve()))
        check("backup parent dir", str(Path(backup).parent), str(Path(dest_dir).resolve()))
        check("dest has new content", open(dest, 'r', encoding='utf-8').read(), "#!/bin/sh\n# NEW version\n")
        check(
            "backup has old content",
            open(str(backup), 'r', encoding='utf-8').read(),
            "#!/bin/sh\n# OLD version\n",
        )
        check("backup name has predeploy marker", "bak-predeploy-" in os.path.basename(str(backup)), True)


def test_idempotent_no_resinstall_no_backup():
    """Re-installing a file whose live copy already matches = skipped, no backup,
    and mtime/content untouched."""
    print("test_idempotent_no_resinstall_no_backup")
    with tempfile.TemporaryDirectory() as d:
        src_dir = os.path.join(d, "src")
        dest_dir = os.path.join(d, "live")
        os.makedirs(src_dir)
        os.makedirs(dest_dir)

        src = os.path.join(src_dir, "watch.py")
        dest = os.path.join(dest_dir, "watch.py")
        content = "print('current')\n"
        with open(src, "w", encoding='utf-8') as f:
            f.write(content)
        with open(dest, "w", encoding='utf-8') as f:
            f.write(content)

        status, dst, backup = ifw.install_file(src, dest_dir)

        check("status skipped when identical", status, "skipped")
        check("no backup on skip", backup, None)
        # No extra backup file was created alongside dest.
        siblings = [n for n in os.listdir(dest_dir) if n != "watch.py"]
        check("no stray backup files", siblings, [])


def test_unrelated_scripts_untouched():
    """Only the listed file is ever created/modified in the live dir. An
    unrelated script present before install is byte-identical afterward and no
    extra files appear."""
    print("test_unrelated_scripts_untouched")
    with tempfile.TemporaryDirectory() as d:
        src_dir = os.path.join(d, "src")
        dest_dir = os.path.join(d, "live")
        os.makedirs(src_dir)
        os.makedirs(dest_dir)

        src = os.path.join(src_dir, "watch.py")
        unrelated = os.path.join(dest_dir, "other-watch.py")
        with open(src, "w", encoding='utf-8') as f:
            f.write("new\n")
        with open(unrelated, "w", encoding='utf-8') as f:
            f.write("untouched\n")
        before = open(unrelated, 'r', encoding='utf-8').read()
        before_mtime = os.path.getmtime(unrelated)

        ifw.install_file(src, dest_dir)

        check("unrelated still same content", open(unrelated, 'r', encoding='utf-8').read(), before)
        check("unrelated mtime unchanged", os.path.getmtime(unrelated), before_mtime)
        listing = sorted(os.listdir(dest_dir))
        check("only dest watch.py + unrelated present", listing, ["other-watch.py", "watch.py"])


def test_copy_not_symlink():
    """Copy-on-merge only: install produces a regular file, never a symlink
    (the scheduler rejects symlinks)."""
    print("test_copy_not_symlink")
    with tempfile.TemporaryDirectory() as d:
        src_dir = os.path.join(d, "src")
        dest_dir = os.path.join(d, "live")
        os.makedirs(src_dir)
        os.makedirs(dest_dir)

        src = os.path.join(src_dir, "watch.sh")
        with open(src, "w", encoding='utf-8') as f:
            f.write("#!/bin/sh\n")

        status, dst, _ = ifw.install_file(src, dest_dir)
        check("status installed", status, "installed")
        check("dest is regular file not symlink", os.path.islink(str(dst)), False)


def test_no_watchdog_changes_installs_nothing():
    """A diff with no watchdog files selects nothing, so a deploy with no
    watchdog change installs zero files (and the step is a no-op)."""
    print("test_no_watchdog_changes_installs_nothing")
    diff = [
        "hermes_cli/kanban_db.py",
        "tools/kanban_tools.py",
        "scripts/other/setup.sh",
    ]
    selected = ifw.select_watchdog_files(diff)
    check("no watchdog files selected", selected, [])


def test_select_only_watchdog_subtree():
    """Only files under scripts/fleet-watchdogs/ are selected, deduped/sorted,
    even when the diff mixes watchdog and non-watchdog paths."""
    print("test_select_only_watchdog_subtree")
    diff = [
        "scripts/fleet-watchdogs/bob-blocker-watch.py",
        "tools/kanban_tools.py",
        "scripts/fleet-watchdogs/test_bob_blocker_watch.py",
        "scripts/fleet-watchdogs/bob-blocker-watch.py",
    ]
    selected = ifw.select_watchdog_files(diff)
    check(
        "only watchdog files, deduped+sorted",
        selected,
        [
            "scripts/fleet-watchdogs/bob-blocker-watch.py",
            "scripts/fleet-watchdogs/test_bob_blocker_watch.py",
        ],
    )


def test_missing_source_raises():
    """A missing source is a hard error, not a silent no-op."""
    print("test_missing_source_raises")
    with tempfile.TemporaryDirectory() as d:
        dest_dir = os.path.join(d, "live")
        os.makedirs(dest_dir)
        try:
            ifw.install_file(os.path.join(d, "nope.py"), dest_dir)
            check("missing source raised", False, True)
        except FileNotFoundError:
            check("missing source raised", True, True)


def main():
    print("=== install_fleet_watchdogs executable tests ===")
    for fn in (
        test_changed_file_installed_with_backup,
        test_idempotent_no_resinstall_no_backup,
        test_unrelated_scripts_untouched,
        test_copy_not_symlink,
        test_no_watchdog_changes_installs_nothing,
        test_select_only_watchdog_subtree,
        test_missing_source_raises,
    ):
        fn()
    print("")
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for name, got, want in FAILURES:
            print(f"  - {name}: got={got!r} want={want!r}")
        sys.exit(1)
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    # Import reference only — keeps the module importable-for-reflection clean.
    _ = ifw
    sys.exit(main())