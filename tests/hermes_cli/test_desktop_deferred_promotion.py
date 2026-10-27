"""An update never swaps files under (or stops) a running POSIX Desktop: it parks the verified
pack and installs it once the app has quit (#139235, #138502)."""

import json
import shutil
import subprocess
import pytest

from hermes_cli import main_desktop, main_desktop_pending
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("linux")


def _pack(base, tag):
    unpacked = base / "linux-unpacked"
    (unpacked / "resources").mkdir(parents=True)
    shutil.copy2("/bin/bash", unpacked / "hermes")  # a real executable living in the build tree
    (unpacked / "resources" / "install-stamp.json").write_text(json.dumps({"commit": tag}), encoding="utf-8")
    return unpacked


def _stamp(desktop_dir):
    return json.loads((desktop_dir / "release/linux-unpacked/resources/install-stamp.json").read_text(encoding="utf-8-sig"))["commit"]


@pytest.fixture
def waiters(monkeypatch):
    spawned = []
    monkeypatch.setattr(main_desktop_pending, "spawn_promotion_waiter", spawned.append)
    return spawned


@pytest.fixture
def desktop_dir(tmp_path):
    desktop = tmp_path / "repo" / "apps" / "desktop"
    _pack(desktop / "release", "OLD")
    return desktop


def test_update_parks_the_build_while_desktop_runs_and_installs_it_after_quit(desktop_dir, waiters):
    running = subprocess.Popen([str(desktop_dir / "release/linux-unpacked/hermes"), "-c", "sleep 60; :"])
    try:
        wait_until(lambda: main_desktop_pending.live_desktop_pids(desktop_dir) == [running.pid], timeout=10)
        staging = main_desktop._desktop_staging_dir(desktop_dir)
        _pack(staging, "NEW")

        assert main_desktop._promote_staged_desktop_app(desktop_dir, staging, integrity_check=lambda p: None) is None
        assert running.poll() is None, "the running Desktop was stopped"
        assert _stamp(desktop_dir) == "OLD"
        assert waiters == [desktop_dir]
        assert main_desktop_pending.promote_pending_desktop_app(desktop_dir) is None  # still running
    finally:
        running.terminate()
        running.wait()

    assert main_desktop_pending.wait_then_promote(desktop_dir) == desktop_dir / "release/linux-unpacked/hermes"
    assert _stamp(desktop_dir) == "NEW"
    assert not main_desktop_pending.pending_dir(desktop_dir).exists()


def test_update_with_no_desktop_running_swaps_now_and_supersedes_a_parked_build(desktop_dir, waiters):
    parked = main_desktop_pending.pending_dir(desktop_dir)
    _pack(parked, "PARKED")
    (parked / main_desktop_pending.READY_MARKER).write_text("{}", encoding="utf-8")
    staging = main_desktop._desktop_staging_dir(desktop_dir)
    _pack(staging, "NEW")

    assert main_desktop._promote_staged_desktop_app(
        desktop_dir, staging, integrity_check=lambda p: None) == desktop_dir / "release/linux-unpacked/hermes"
    assert _stamp(desktop_dir) == "NEW"
    assert not parked.exists()
    assert waiters == []


def _park(desktop_dir, tag):
    parked = main_desktop_pending.pending_dir(desktop_dir)
    _pack(parked, tag)
    (parked / main_desktop_pending.READY_MARKER).write_text("{}", encoding="utf-8")
    return parked


def test_a_second_waiter_leaves_the_parked_build_to_the_one_already_waiting(desktop_dir):
    from gateway.status import _release_file_lock, _try_acquire_file_lock

    parked = _park(desktop_dir, "NEW")
    slot = main_desktop_pending._waiter_slot_path(desktop_dir)
    slot.parent.mkdir(parents=True, exist_ok=True)
    with open(slot, "a+", encoding="utf-8") as owner:
        assert _try_acquire_file_lock(owner)
        try:
            assert main_desktop_pending.wait_then_promote(desktop_dir) is None
            assert parked.exists() and _stamp(desktop_dir) == "OLD"
        finally:
            _release_file_lock(owner)
    assert main_desktop_pending.wait_then_promote(desktop_dir) == desktop_dir / "release/linux-unpacked/hermes"
    assert _stamp(desktop_dir) == "NEW"


def test_a_deferred_install_refreshes_the_installed_app_copies(desktop_dir, monkeypatch):
    refreshed = []
    monkeypatch.setattr(main_desktop, "_refresh_installed_desktop_apps", refreshed.append)
    _park(desktop_dir, "NEW")

    assert main_desktop_pending.wait_then_promote(desktop_dir) is not None
    assert refreshed == [desktop_dir]


@pytest.mark.parametrize("product", ["desktop", "webapp"])
def test_parking_a_build_keeps_the_desktop_build_receipts_current(tmp_path, monkeypatch, product):
    from hermes_cli.source_build import source_product_current
    from tests.hermes_cli.test_source_build import copy_freshness_scripts, stamp_product, use_host_node_as_pm_node

    root = tmp_path / "repo"
    (root / "apps/desktop/src").mkdir(parents=True)
    (root / "apps/desktop/src/main.ts").write_text("export {}\n", encoding="utf-8")
    copy_freshness_scripts(root)
    use_host_node_as_pm_node(monkeypatch)
    out = root / "apps/desktop/release/linux-unpacked/resources/app.asar.unpacked/dist"
    out.mkdir(parents=True)
    stamp_product(root, product, out)
    assert source_product_current(root, product, out)

    _park(root / "apps/desktop", "NEW")
    assert source_product_current(root, product, out)


def test_a_desktop_reopened_before_the_swap_keeps_the_waiter_waiting(desktop_dir, monkeypatch):
    """The Desktop quits, the waiter heads for the swap, and the user reopens Hermes before the waiter
    holds the build lock: the swap must not land under the reopened app; it lands once that one quits."""
    import threading

    from hermes_cli import desktop_build_lock

    _park(desktop_dir, "NEW")
    monkeypatch.setattr(main_desktop_pending, "_WAIT_POLL_S", 0.05)
    exe = desktop_dir / "release/linux-unpacked/hermes"
    reopened = []

    class ReopenWhileTakingTheLock(desktop_build_lock.DesktopBuildLock):
        def acquire(self, *, wait=False):
            if not reopened:  # the gap between "nothing runs" and holding the lock
                reopened.append(subprocess.Popen([str(exe), "-c", "sleep 60; :"]))
                wait_until(lambda: main_desktop_pending.live_desktop_pids(desktop_dir) == [reopened[0].pid], timeout=10)
            return super().acquire(wait=wait)

    monkeypatch.setattr(desktop_build_lock, "DesktopBuildLock", ReopenWhileTakingTheLock)
    result = []
    waiter = threading.Thread(target=lambda: result.append(main_desktop_pending.wait_then_promote(desktop_dir)), daemon=True)
    waiter.start()
    try:
        wait_until(lambda: bool(reopened), timeout=10)
        waiter.join(timeout=1.0)
        assert waiter.is_alive(), "the waiter gave up while the reopened Desktop still runs"
        assert reopened[0].poll() is None and _stamp(desktop_dir) == "OLD"
        assert main_desktop_pending.pending_dir(desktop_dir).exists()
    finally:
        for proc in reopened:
            proc.terminate()
            proc.wait()

    waiter.join(timeout=30)
    assert not waiter.is_alive()
    assert result == [exe]
    assert _stamp(desktop_dir) == "NEW"
