"""Tests for the TERMINAL_CWD readers-writer lock in cron/scheduler.py.

Workdir cron jobs override the process-global ``os.environ["TERMINAL_CWD"]``
for their whole agent run.  Workdir-less jobs run concurrently on a separate
pool and read that same global (via the terminal / file / code-exec tools), so
without serialization they execute commands in another job's workdir.

``_ReadWriteLock`` models workdir jobs as writers (exclusive) and workdir-less
jobs as readers (concurrent with each other, excluded from a writer's run).
These tests assert that contract.
"""

import threading


def _lock():
    import cron.scheduler as sched

    return sched._ReadWriteLock()


def test_multiple_readers_run_concurrently():
    """Workdir-less jobs (readers) hold the lock simultaneously."""
    lock = _lock()
    # Barrier of 3 only releases if both reader threads hold the read lock at
    # the same time as the main thread waits — proving readers are concurrent.
    barrier = threading.Barrier(3, timeout=5)

    def reader():
        lock.acquire_read()
        try:
            barrier.wait()
        finally:
            lock.release_read()

    threads = [threading.Thread(target=reader) for _ in range(2)]
    for t in threads:
        t.start()

    # Does not raise BrokenBarrierError -> both readers were holding at once.
    barrier.wait(timeout=5)
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()


def test_writer_waits_for_active_reader():
    """A workdir job (writer) cannot acquire while a reader holds the lock."""
    lock = _lock()
    order = []
    reader_holding = threading.Event()
    let_reader_go = threading.Event()

    def reader():
        lock.acquire_read()
        try:
            reader_holding.set()
            let_reader_go.wait(timeout=5)
            order.append("reader-release")
        finally:
            lock.release_read()

    def writer():
        reader_holding.wait(timeout=5)
        lock.acquire_write()
        try:
            order.append("writer-acquire")
        finally:
            lock.release_write()

    rt = threading.Thread(target=reader)
    wt = threading.Thread(target=writer)
    rt.start()
    wt.start()

    # Give the writer time to try (and block) while the reader still holds.
    reader_holding.wait(timeout=5)
    let_reader_go.set()

    rt.join(timeout=5)
    wt.join(timeout=5)
    assert not rt.is_alive() and not wt.is_alive()
    # The writer only ran after the reader released — never alongside it.
    assert order == ["reader-release", "writer-acquire"]


def test_reader_never_observes_writer_override():
    """Regression: the cross-pool TERMINAL_CWD corruption.

    A workdir job (writer) overriding the shared cwd must never be observed by
    a concurrent workdir-less job (reader).  ``shared["cwd"]`` stands in for
    ``os.environ["TERMINAL_CWD"]``: the reader, even though it starts while the
    writer holds the override, must block until the writer restores the value.
    """
    lock = _lock()
    shared = {"cwd": "<scheduler>"}
    observations = []
    writer_holding = threading.Event()
    release_writer = threading.Event()

    def writer():
        lock.acquire_write()
        try:
            shared["cwd"] = "/project/A"
            writer_holding.set()
            release_writer.wait(timeout=5)
        finally:
            shared["cwd"] = "<scheduler>"
            lock.release_write()

    def reader():
        # Start only once the writer holds the lock and has applied the
        # override — the exact window the old code corrupted.
        writer_holding.wait(timeout=5)
        lock.acquire_read()
        try:
            observations.append(shared["cwd"])
        finally:
            lock.release_read()

    wt = threading.Thread(target=writer)
    rt = threading.Thread(target=reader)
    wt.start()
    rt.start()

    # The reader is now blocked on the writer; let the writer finish.
    writer_holding.wait(timeout=5)
    release_writer.set()

    wt.join(timeout=5)
    rt.join(timeout=5)
    assert not wt.is_alive() and not rt.is_alive()
    # The reader saw the restored value, never the writer's /project/A override.
    assert observations == ["<scheduler>"]
