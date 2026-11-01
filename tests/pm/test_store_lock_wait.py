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
