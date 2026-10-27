"""Deferred Desktop promotion: an update never stops or swaps files under a running Desktop.

A terminal, cron or gateway ``hermes update`` rebuilds the packaged Desktop while the user may
still have it open from ``apps/desktop/release``. Renaming that tree away (or stopping the app to
do it) crashes or kills a live Electron: its renderer lazy-loads hashed chunks by path, and its
binary pages come from the replaced files (#139235, #138502, #109643). Instead the verified
staged pack is parked under ``apps/desktop/.pending-release`` with a ready marker written last,
and promoted by the existing two-rename swap once no process runs from ``release/``: when the
Desktop quits (a detached waiter), on the next ``hermes desktop`` launch, or by the next update.
Windows keeps its stop-before-pack path: its exe lock means the pack cannot be built under a
live app in the first place.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hermes_cli.main")

PENDING_DIRNAME = ".pending-release"
READY_MARKER = "hermes-pending.json"
_WAIT_POLL_S = 2.0


def pending_dir(desktop_dir: Path) -> Path:
    """Sibling of ``release/`` (same filesystem, so promotion is a rename), outside its globs."""
    return desktop_dir / PENDING_DIRNAME


def _ready_pending(desktop_dir: Path) -> Optional[Path]:
    """The parked pack when its ready marker exists (written only after the rename landed)."""
    pending = pending_dir(desktop_dir)
    return pending if (pending / READY_MARKER).is_file() else None


def discard_pending(desktop_dir: Path) -> None:
    shutil.rmtree(pending_dir(desktop_dir), ignore_errors=True)


def live_desktop_pids(desktop_dir: Path) -> list[int]:
    """PIDs running from this build's ``release/`` that an update must not swap under.

    Excludes the Desktop driving this process (and its helpers): that one has already quit
    for a hand-off or owns the relaunch, exactly as the stop path spares it."""
    from hermes_cli.main_desktop import _desktop_processes_from_build

    return sorted(int(proc.pid) for proc in _desktop_processes_from_build(desktop_dir))


def defer_promotion_while_desktop_runs(desktop_dir: Path, staging_dir: Path) -> bool:
    """Park the verified *staging_dir* instead of swapping it under a live POSIX Desktop.

    True when parked (the caller must not swap). False when nothing runs from ``release/``:
    the caller swaps now, and any older parked pack is superseded by this build."""
    if sys.platform == "win32":
        return False
    pids = live_desktop_pids(desktop_dir)
    if not pids:
        discard_pending(desktop_dir)
        return False
    pending = pending_dir(desktop_dir)
    discard_pending(desktop_dir)
    os.rename(staging_dir, pending)
    marker = pending / READY_MARKER
    tmp = marker.with_suffix(".tmp")
    tmp.write_text(json.dumps({"staged_at": int(time.time()), "deferred_for_pids": pids}), encoding="utf-8")
    os.replace(tmp, marker)
    waiter = spawn_promotion_waiter(desktop_dir)
    print(f"  ⏸ Hermes Desktop is running (pid {', '.join(map(str, pids))}); its files were left untouched.")
    print("    The rebuilt app is staged and installs automatically when you quit Hermes Desktop,")
    print("    or the next time you start it with `hermes desktop`.")
    logger.info("desktop promotion deferred for live pids %s; waiter %s", pids, waiter)
    return True


def promote_pending_desktop_app(desktop_dir: Path) -> Optional[Path]:
    """Swap a ready parked pack into ``release/`` when no Desktop runs from it; the live exe, or None.

    Callers hold the desktop build lock (``hermes desktop``, the update build, the waiter)."""
    pending = _ready_pending(desktop_dir)
    if pending is None or live_desktop_pids(desktop_dir):
        return None
    from hermes_cli.main_desktop import _swap_staged_desktop_app

    exe = _swap_staged_desktop_app(desktop_dir, pending)
    if exe is not None:
        print(f"  ✓ Installed the Hermes Desktop update staged while it was running: {exe}")
        # The update refreshed owned installed copies (/Applications/Hermes.app) from the OLD
        # release/ while this pack was parked; Finder and the Dock launch those copies.
        from hermes_cli.main_desktop import _refresh_installed_desktop_apps

        _refresh_installed_desktop_apps(desktop_dir)
    return exe


def pending_current_for_head(desktop_dir: Path, project_root: Path) -> bool:
    """True when the parked pack was built from HEAD and its inputs are unchanged since."""
    pending = _ready_pending(desktop_dir)
    if pending is None:
        return False
    from hermes_cli.main_desktop import _desktop_packaged_executable_in
    from hermes_cli.source_build import source_product_current
    from hermes_cli.steward import read_install_stamp
    from hermes_cli.version_info import _run_git

    exe = _desktop_packaged_executable_in(pending)
    if exe is None:
        return False
    resources = exe.parent.parent / "Resources" if sys.platform == "darwin" else exe.parent / "resources"
    commit = read_install_stamp(resources).get("commit")
    if not isinstance(commit, str) or commit != _run_git(project_root, "rev-parse", "HEAD"):
        return False
    return source_product_current(project_root, "desktop", resources / "app.asar.unpacked" / "dist")


def settle_pending_desktop_update(desktop_dir: Path, project_root: Path) -> bool:
    """Finish an update parked earlier instead of rebuilding it: True when a pack current for HEAD
    is parked, installed now if no Desktop runs, else kept with a fresh waiter (the earlier one
    may not have survived a reboot). Callers hold the desktop build lock."""
    if not pending_current_for_head(desktop_dir, project_root):
        return False
    if promote_pending_desktop_app(desktop_dir) is None and _ready_pending(desktop_dir) is not None:
        spawn_promotion_waiter(desktop_dir)
        print("  ⏸ A Desktop update is already staged; it installs when you quit Hermes Desktop.")
    return True


def _scoped_argv(argv: list[str]) -> tuple[list[str], dict]:
    """Run *argv* in its own transient user scope when systemd can make one, so stopping the
    gateway or terminal unit that ran this update cannot take the waiter with it (the same
    cgroup-escape rule and scope builder as restart-safe workers)."""
    if not sys.platform.startswith("linux"):
        return argv, dict(os.environ)
    from tools.process_registry import (
        _build_systemd_scope_argv, _systemd_run_user_scope_available, systemd_user_bus_env)

    env = systemd_user_bus_env()
    if not _systemd_run_user_scope_available():
        return argv, env
    return _build_systemd_scope_argv(argv, f"desktop-promotion-{os.getpid()}-{int(time.time())}"), env


def spawn_promotion_waiter(desktop_dir: Path) -> Optional[int]:
    """Start a detached ``--wait`` promoter for *desktop_dir*; its PID, or None when it could not start."""
    from hermes_constants import get_hermes_home

    log_dir = get_hermes_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    argv, env = _scoped_argv(
        [sys.executable, "-m", "hermes_cli.main_desktop_pending", "--wait", "--desktop-dir", str(desktop_dir)])
    try:
        with open(log_dir / "desktop-promotion.log", "ab") as log:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True, close_fds=True)
    except OSError as exc:
        logger.warning("could not start the deferred desktop promotion waiter: %s", exc)
        return None
    return proc.pid


def _waiter_slot_path(desktop_dir: Path) -> Path:
    """One waiter per checkout: a lock beside the desktop build lock (outside the checkout)."""
    from hermes_cli.desktop_build_lock import DesktopBuildLock

    build_lock = DesktopBuildLock(desktop_dir.parents[1]).path
    return build_lock.with_name(build_lock.stem + "-promotion-waiter.lock")


def wait_then_promote(desktop_dir: Path) -> Optional[Path]:
    """Block until no process runs from ``release/``, then promote the parked pack under the build lock.

    Exits at once when another waiter owns this checkout: every deferred update and every launch
    that keeps a parked pack spawns one, and the owner already follows whichever pack is parked.
    The owner leaves only after seeing no parked pack WHILE holding the build lock (every park
    and promotion holds it), so a pack parked by a waiter that bowed out is never orphaned."""
    from gateway.status import _release_file_lock, _try_acquire_file_lock
    from hermes_cli.desktop_build_lock import DesktopBuildLock

    slot_path = _waiter_slot_path(desktop_dir)
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    with open(slot_path, "a+", encoding="utf-8") as slot:
        if not _try_acquire_file_lock(slot):
            return None
        while True:
            if _ready_pending(desktop_dir) is not None and live_desktop_pids(desktop_dir):
                time.sleep(_WAIT_POLL_S)
                continue
            lock = DesktopBuildLock(desktop_dir.parents[1])
            lock.acquire(wait=True)
            try:
                if _ready_pending(desktop_dir) is None:
                    _release_file_lock(slot)
                    return None
                promoted = promote_pending_desktop_app(desktop_dir)
                # None with a Desktop running again: it relaunched before the lock; keep waiting.
                # None with none running: the swap failed and the next launch or update retries.
                if promoted is not None or not live_desktop_pids(desktop_dir):
                    _release_file_lock(slot)
                    return promoted
            finally:
                lock.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Promote a Desktop update staged while the app was running")
    parser.add_argument("--desktop-dir", type=Path, required=True)
    parser.add_argument("--wait", action="store_true", help="wait for the running Desktop to quit first")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    target = args.desktop_dir.resolve()
    promoted = wait_then_promote(target) if args.wait else promote_pending_desktop_app(target)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} deferred desktop promotion: {promoted or 'nothing promoted'}")
