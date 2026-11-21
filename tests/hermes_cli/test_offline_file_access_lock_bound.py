"""Board #741: the connection-lifecycle lock must not let one wedged reader park every
other reader in the process.

Every terminal guard call runs ``cron.lifecycle_guard._read_referenced_script`` ->
``hermes_cli.sqlite_safe_read.offline_file_access``, which holds ``_live_lock`` across
caller I/O. When one reader wedges inside that window, every later guard call blocks on
the acquire with no bound -- visible in ``gateway.error.log`` as 27 of 27 abandoned
``deadline-terminal.pre-exec-guard`` workers parked at the ``with _live_lock`` line.

The contract pinned here: a waiter gives up (fail closed, LiveConnectionError) instead of
waiting forever, and the lock is still released correctly on every path.

Every wait runs in a helper thread so a regression fails the assertion instead of hanging
the suite.
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hermes_cli import sqlite_safe_read as ssr  # noqa: E402

# A waiter must answer within this many seconds once the lock's own bound has expired.
PATIENCE_S = 8.0


@pytest.fixture(autouse=True)
def _clean_registry():
    """Leave the module's registry and lock exactly as found."""
    with ssr._live_lock:
        ssr._live_connections.clear()
    yield
    with ssr._live_lock:
        ssr._live_connections.clear()


def _hold_lock_in_a_thread(started, release):
    def _holder():
        with ssr._live_lock:
            started.set()
            release.wait(30)

    threading.Thread(target=_holder, daemon=True).start()
    assert started.wait(5), "holder thread never took the lock"


def _attempt(**kwargs):
    """Enter offline_file_access in a thread. Returns (outcome, seconds).

    ``outcome`` is "entered", "refused" (LiveConnectionError), or "wedged" -- the last
    only reachable without the bound, which is exactly the regression this pins.
    """
    box = {}

    def _w():
        try:
            with ssr.offline_file_access(Path("/tmp/probe-not-a-db"), what="serve", **kwargs):
                box["outcome"] = "entered"
        except ssr.LiveConnectionError:
            box["outcome"] = "refused"
        except BaseException as exc:  # noqa: BLE001 - surface anything unexpected
            box["outcome"] = f"raised {type(exc).__name__}: {exc}"

    t0 = time.monotonic()
    th = threading.Thread(target=_w, daemon=True)
    th.start()
    th.join(PATIENCE_S)
    elapsed = time.monotonic() - t0
    if th.is_alive():
        return "wedged", elapsed
    return box.get("outcome", "no outcome"), elapsed


def test_waiter_fails_closed_instead_of_waiting_forever():
    """CONTROL: with the bound a waiter answers; without it, it never returns."""
    started, release = threading.Event(), threading.Event()
    _hold_lock_in_a_thread(started, release)
    try:
        outcome, elapsed = _attempt()
        assert outcome == "refused", (
            f"a waiter parked behind a wedged reader returned {outcome!r} after "
            f"{elapsed:.1f}s; it must fail closed with LiveConnectionError")
        assert elapsed < PATIENCE_S, f"waited {elapsed:.1f}s"
    finally:
        release.set()


def test_default_bound_is_applied_without_an_explicit_timeout():
    """The shipped default (not just an injected one) must bound the wait."""
    assert ssr._LIVE_LOCK_TIMEOUT_S > 0
    assert ssr._LIVE_LOCK_TIMEOUT_S <= 10.0, "a guard call cannot wait 10s+ for a lock"
    started, release = threading.Event(), threading.Event()
    _hold_lock_in_a_thread(started, release)
    try:
        t0 = time.monotonic()
        outcome, _ = _attempt()
        elapsed = time.monotonic() - t0
        assert outcome == "refused", f"default bound did not apply: {outcome!r}"
        assert elapsed <= ssr._LIVE_LOCK_TIMEOUT_S + 2.0, (
            f"waited {elapsed:.1f}s, longer than the {ssr._LIVE_LOCK_TIMEOUT_S}s default")
    finally:
        release.set()


def test_lock_is_released_after_a_refused_wait():
    """A timed-out waiter must not leak the lock: the next caller still gets in."""
    started, release = threading.Event(), threading.Event()
    _hold_lock_in_a_thread(started, release)
    outcome, _ = _attempt(timeout=0.1)
    assert outcome == "refused"
    release.set()
    assert ssr._live_lock.acquire(timeout=5.0), "lock leaked after a refused wait"
    ssr._live_lock.release()


def test_normal_path_still_works_and_releases():
    """No holder: the context manager behaves exactly as before."""
    with ssr.offline_file_access(Path("/tmp/probe-not-a-db"), what="read"):
        pass
    assert ssr._live_lock.acquire(timeout=5.0), "lock not released after a normal read"
    ssr._live_lock.release()


def test_live_connection_still_refuses(tmp_path):
    """The original fail-closed rule is untouched."""
    target = tmp_path / "live.db"
    target.write_bytes(b"x")
    conn = ssr.connect_tracked(target)
    try:
        with pytest.raises(ssr.LiveConnectionError) as excinfo:
            with ssr.offline_file_access(target, what="read"):
                pass  # pragma: no cover
        assert "still open" in str(excinfo.value)
    finally:
        conn.close()


def test_exception_inside_the_block_still_releases():
    """A raising body must not strand the lock."""
    with pytest.raises(RuntimeError):
        with ssr.offline_file_access(Path("/tmp/probe-not-a-db"), what="read"):
            raise RuntimeError("boom")
    assert ssr._live_lock.acquire(timeout=5.0), "lock stranded by a raising body"
    ssr._live_lock.release()
