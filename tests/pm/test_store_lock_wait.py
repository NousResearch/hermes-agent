"""A writer queued behind the store lock says so instead of blocking silently."""
import os
import threading
import time

from pm.filesystem import lock_fd
from pm.store import Store


def test_install_lock_reports_what_it_waits_on(tmp_path, capsys):
    store = Store(tmp_path / "store")
    store.root.mkdir()
    holder = os.open(store.root / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(holder, wait=False)
    entered = threading.Event()

    def contend():
        with store.install_lock():
            entered.set()

    waiter = threading.Thread(target=contend, daemon=True)
    waiter.start()
    deadline = time.monotonic() + 15
    err = ""
    while "waiting for" not in err and time.monotonic() < deadline:
        time.sleep(0.1)
        err += capsys.readouterr().err
    assert str(store.root / ".install.lock") in err
    assert not entered.is_set()
    os.close(holder)
    assert entered.wait(timeout=10), "the message must not replace the wait"
    waiter.join(timeout=5)


def _hold(store: Store) -> int:
    """Take the install lock from this process and return the holding fd."""
    store.root.mkdir(parents=True, exist_ok=True)
    fd = os.open(store.root / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(fd, wait=False), "the fixture never took the lock"
    return fd


def test_a_bounded_wait_raises_instead_of_queueing(tmp_path):
    """The bounded form is fail-closed: a caller that must not hang gets an error.

    ``hermes doctor --fix`` is the caller this exists for — it may not sit behind a
    long PM build. The default (``timeout=None``) still waits, which the test above
    pins; this one pins the opposite contract, so neither can be satisfied by
    accident.
    """
    store = Store(tmp_path / "store")
    holder = _hold(store)
    try:
        started = time.monotonic()
        try:
            with store.install_lock(timeout=0.2):
                raise AssertionError("a held lock must not be entered under a bounded wait")
        except TimeoutError:
            pass
        else:
            raise AssertionError("bounded wait returned instead of raising TimeoutError")
        elapsed = time.monotonic() - started
        assert elapsed < 10, f"the bounded wait took {elapsed:.1f}s; the timeout did not apply"
    finally:
        os.close(holder)


def test_a_bounded_wait_still_enters_a_free_lock(tmp_path):
    """The timeout bounds the WAIT, it does not require a contended lock to proceed."""
    store = Store(tmp_path / "store")
    store.root.mkdir(parents=True, exist_ok=True)
    with store.install_lock(timeout=5):
        pass


def test_a_bounded_wait_does_not_announce_a_wait_it_will_not_make(tmp_path, capsys):
    """A budget too short to wait is a one-try probe, so it must not claim to wait.

    "waiting for <lock> (another PM operation holds it)" followed by an immediate
    raise would describe something that never happened, which is how a user ends up
    looking for a PM operation that was never running.
    """
    store = Store(tmp_path / "store")
    holder = _hold(store)
    try:
        try:
            with store.install_lock(timeout=0.01):
                pass
        except TimeoutError:
            pass
        assert "waiting for" not in capsys.readouterr().err
    finally:
        os.close(holder)
