"""A client's lock probe never refuses a gateway boot.

``gateway ensure`` (and every discovery) asks "is a reservation pending?" by taking
``gateway.lock``'s flock for an instant (``gateway.status._probe_lock_file``). A multiplexer whose
non-blocking boot reserve lands in that instant used to exit 1 with ``Cannot reserve gateway
profiles``, so the very client that started it read ``runtime_exited``. These tests hold a real
flock exactly where a probe holds it and boot through the real reserve.
"""
import asyncio
from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from gateway.run_bootstrap import _browse_reconcile_reserve_wait_s, _reserve_waiting_out_browse_reconcile
from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
from gateway.runtime_ownership import OwnershipConflict, ProfileOwnership, process_ownership

pytestmark = pytest.mark.platforms("linux", "macos")


def _homes(tmp_path):
    root = tmp_path / "state"
    boot = root / "profiles" / "boot"
    for home in (root, boot):
        home.mkdir(parents=True, mode=0o700, exist_ok=True)
        (home / "gateway.lock").touch(mode=0o600)
    return root, boot


def _probe_hold(home):
    """What ``_probe_lock_file`` holds between its acquire and release: an exclusive flock on its
    own open file description, with no record written."""
    handle = open(home / "gateway.lock", "r+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_boot_reserve_survives_a_probe_on_a_served_profile(tmp_path):
    """The probe holds boot's flock during the reserve and lets go before the re-check."""
    root, boot = _homes(tmp_path)

    class ProbedDuringFirstReserve(ProfileOwnership):
        attempts = 0

        def reserve(self, homes, **kwargs):
            self.attempts += 1
            if self.attempts > 1:
                return super().reserve(homes, **kwargs)
            with _probe_hold(boot):
                return super().reserve(homes, **kwargs)

    booting = ProbedDuringFirstReserve()
    try:
        _reserve_waiting_out_browse_reconcile(booting, [root, boot], wait_s=10)
        assert booting.owns(root) and booting.owns(boot)
        assert booting.attempts == 2
    finally:
        booting.close()


def test_boot_reserve_survives_a_probe_during_the_recheck(tmp_path):
    """The rolled-back attempt left this process's own record on root; a probe holding root while
    the boot re-checks must not read that record as another owner."""
    root, boot = _homes(tmp_path)
    held = []

    class ProbedAcrossRecheck(ProfileOwnership):
        attempts = 0

        def reserve(self, homes, **kwargs):
            self.attempts += 1
            while held:
                held.pop().close()
            if self.attempts > 1:
                return super().reserve(homes, **kwargs)
            try:
                with _probe_hold(boot):
                    return super().reserve(homes, **kwargs)
            finally:
                held.append(_probe_hold(root))  # still held when the boot re-checks who conflicts

    booting = ProbedAcrossRecheck()
    try:
        _reserve_waiting_out_browse_reconcile(booting, [root, boot], wait_s=10)
        assert booting.owns(root) and booting.owns(boot)
    finally:
        booting.close()
        for handle in held:
            handle.close()


_HOLDER = """
import fcntl, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
home = Path(sys.argv[2])
if sys.argv[1] == "owner":
    from gateway.runtime_ownership import ProfileOwnership
    owner = ProfileOwnership()
    owner.reserve([home])
else:  # what _probe_lock_file holds: the flock on its own description, no record written
    handle = open(home / "gateway.lock", "r+", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
print("held", flush=True)
if sys.stdin.readline().strip() == "release" and sys.argv[1] == "probe":
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    print("released", flush=True)
sys.stdin.readline()
"""


@contextmanager
def _holder(kind, home):
    """A separate process holding *home*'s flock as a client probe (``probe``) or a gateway
    (``owner``, its record naming that live process)."""
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, kind, str(home), str(Path(__file__).resolve().parents[2])],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "held"
        yield proc
    finally:
        proc.kill()
        proc.wait()


def _release(proc):
    proc.stdin.write("release\n")
    proc.stdin.flush()
    assert proc.stdout.readline().strip() == "released"


def test_a_live_foreign_owner_still_refuses_at_once(tmp_path):
    """Control: a holder whose record names another live process is an owner, not a probe."""
    root, boot = _homes(tmp_path)
    wait_s = 10
    with _holder("owner", boot):
        booting = ProfileOwnership()
        started = time.monotonic()
        with pytest.raises(OwnershipConflict):
            _reserve_waiting_out_browse_reconcile(booting, [root, boot], wait_s=wait_s)
        assert time.monotonic() - started < wait_s / 2
        assert not booting.owns(root)


def test_a_persistent_probe_flicker_still_refuses_at_the_deadline(tmp_path):
    """A probe that is back on every attempt is retried with a backoff, never spun on, and the
    reserve still refuses once the deadline passes."""
    root, boot = _homes(tmp_path)
    wait_s = 0.3

    class AlwaysProbed(ProfileOwnership):
        refused_at = []
        gaps = []

        def reserve(self, homes, **kwargs):
            if self.refused_at:
                self.gaps.append(time.monotonic() - self.refused_at[-1])
            try:
                with _probe_hold(boot):
                    return super().reserve(homes, **kwargs)
            finally:
                self.refused_at.append(time.monotonic())

    booting = AlwaysProbed()
    started = time.monotonic()
    with pytest.raises(OwnershipConflict):
        _reserve_waiting_out_browse_reconcile(booting, [root, boot], wait_s=wait_s)
    assert time.monotonic() - started >= wait_s
    assert booting.gaps and min(booting.gaps) >= 0.01, booting.gaps  # backed off, never spun
    assert not booting.owns(root)


class _HotServeRunner(GatewayProfileReconcileMixin):
    """An adapters-only multiplexer (no authority registry): serving grows the reservation alone."""
    session_authorities = None

    def __init__(self, root):
        self.config = SimpleNamespace(_runtime_profile_homes=(("default", root),))


def test_hot_serve_of_a_new_profile_survives_a_probe(tmp_path, monkeypatch):
    """A client probe on a new profile's lock while the running multiplexer serves it: the profile
    is served, not parked as owned by another gateway."""
    root, fresh = _homes(tmp_path)
    attempts = []
    with _holder("probe", fresh) as probe:
        def reserve(homes, **kwargs):
            attempts.append(homes)
            try:
                return ProfileOwnership.reserve(process_ownership, homes, **kwargs)
            except OwnershipConflict:
                _release(probe)  # the probe lets go before the re-check, as _probe_lock_file does
                raise

        monkeypatch.setattr(process_ownership, "reserve", reserve)
        runner = _HotServeRunner(root)
        try:
            assert asyncio.run(runner._serve_profile_runtime("boot", fresh)) is None
            assert process_ownership.owns(fresh) and len(attempts) == 2
            assert ("boot", fresh.resolve()) in runner.config._runtime_profile_homes
        finally:
            process_ownership.release(fresh)


def test_hot_serve_still_parks_a_profile_another_gateway_owns(tmp_path):
    """Control: a live gateway on the new profile's home parks it at once."""
    root, fresh = _homes(tmp_path)
    runner = _HotServeRunner(root)
    with _holder("owner", fresh):
        started = time.monotonic()
        reason = asyncio.run(runner._serve_profile_runtime("boot", fresh))
        assert reason and reason.startswith("home owned by another gateway")
        assert time.monotonic() - started < _browse_reconcile_reserve_wait_s() / 2
        assert not process_ownership.owns(fresh)
        assert runner.config._runtime_profile_homes == (("default", root),)


def test_cancelled_hot_serve_never_takes_the_profile_after_cleanup(tmp_path, monkeypatch):
    """The gateway stop cancels the reconcile mid-wait, then releases ownership. The reservation
    worker must not outlive that cancel: a worker still retrying took the flock once the holder let
    go, after cleanup, with no coroutine left to serve or release it."""
    from gateway import runtime_ownership

    root, fresh = _homes(tmp_path)
    retrying, worker_done = threading.Event(), threading.Event()
    real_conflicts = runtime_ownership._conflicting_homes
    real_reserve = runtime_ownership.reserve_waiting_out_transient_holders

    def conflicts(homes):
        found = real_conflicts(homes)
        retrying.set()  # the worker refused once and is waiting the holder out
        return found

    def reserve(*args, **kwargs):
        try:
            return real_reserve(*args, **kwargs)
        finally:
            worker_done.set()

    monkeypatch.setattr(runtime_ownership, "_conflicting_homes", conflicts)
    monkeypatch.setattr(runtime_ownership, "reserve_waiting_out_transient_holders", reserve)

    async def scenario(probe):
        task = asyncio.ensure_future(_HotServeRunner(root)._serve_profile_runtime("boot", fresh))
        assert await asyncio.to_thread(retrying.wait, 30)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        process_ownership.release(fresh)  # the canceller's ownership cleanup
        _release(probe)  # the holder lets go after cleanup
        assert await asyncio.to_thread(worker_done.wait, 30)

    try:
        with _holder("probe", fresh) as probe:
            asyncio.run(scenario(probe))
        assert not process_ownership.owns(fresh)
    finally:
        process_ownership.release(fresh)
