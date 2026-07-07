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


def test_run_job_releases_cwd_lock_when_body_raises(tmp_path):
    """A workdir job whose run_job body raises must still RELEASE the writer lock.

    Regression for the leak that made the fix "still broken": the acquire was
    placed before the try whose finally releases, so an exception in the
    unprotected window (or anywhere in the body) leaked the writer lock and
    deadlocked the whole scheduler. This asserts the lock is free again after a
    raising run — acquire_write() must not block.
    """
    from unittest.mock import MagicMock, patch
    import cron.scheduler as sched

    workdir = tmp_path / "proj"
    workdir.mkdir()
    job = {"id": "boom-job", "name": "boom", "prompt": "hi", "workdir": str(workdir)}

    # Force a raise in the WINDOW BETWEEN acquire and the try body — the exact
    # spot the buggy placement left unprotected. With the fix these statements
    # are inside the try (finally releases); with the bug the lock leaks.
    # logger.info(...) fires right after os.environ["TERMINAL_CWD"] is set for a
    # workdir job, in that window, so making it raise exercises the leak path.
    real_info = sched.logger.info

    def _raise_on_workdir_log(msg, *args, **kwargs):
        if isinstance(msg, str) and "using workdir" in msg:
            raise RuntimeError("boom")
        return real_info(msg, *args, **kwargs)

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch.object(sched.logger, "info", side_effect=_raise_on_workdir_log), \
         patch("hermes_state.SessionDB", return_value=MagicMock()):
        # run_job catches its own body exceptions and returns (False, ...);
        # it must not propagate, and it must release the lock either way.
        success, _out, _final, _err = sched.run_job(job)

    assert success is False

    # If the writer lock leaked, this acquire would block forever. Prove it's
    # free by acquiring as a writer from another thread under a short timeout.
    acquired = threading.Event()

    def try_acquire():
        sched._terminal_cwd_lock.acquire_write()
        try:
            acquired.set()
        finally:
            sched._terminal_cwd_lock.release_write()

    t = threading.Thread(target=try_acquire, daemon=True)
    t.start()
    assert acquired.wait(timeout=5), "writer lock was leaked by run_job on exception"
    t.join(timeout=5)


def test_writer_active_flag_set_before_lock_release():
    """Regression: the writer-active flag must be set *inside* the condition
    lock's critical section, never in the gap between releasing the lock and
    setting the flag.

    ``_ReadWriteLock`` models a workdir cron job as a writer that excludes
    workdir-less (reader) jobs from observing its ``TERMINAL_CWD`` override.
    If ``_writer_active`` is published only *after* the condition lock is
    released, a concurrent reader can pass its guard in that window and run in
    the writer's workdir — exactly the corruption the lock exists to prevent.
    """
    import cron.scheduler as sched

    class _ProbeCondition:
        """Condition stand-in that records whether a reader could have slipped
        through the lock-release window during ``acquire_write``.

        The real ``threading.Condition`` is used for actual locking; the probe
        only inspects the lock's published state at the exact moment the
        ``with self._cond`` block in ``acquire_write`` is about to release.
        """

        def __init__(self, lock):
            self._lock = lock
            self._real = threading.Condition(threading.Lock())
            self.violation = False
            self.checking = False

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, *exc):
            # Only the ``acquire_write`` critical section is in scope (it is the
            # only call we wrap with ``checking``).  At its ``with`` exit the
            # condition lock is about to be released: if the writer-active flag
            # is still False and no writer is queued, a reader's guard would pass
            # and run concurrently with this writer.
            if self.checking and (
                not self._lock._writer_active and self._lock._writers_waiting == 0
            ):
                self.violation = True
            return self._real.__exit__(*exc)

        def wait(self):
            return self._real.wait()

        def notify_all(self):
            return self._real.notify_all()

    lock = sched._ReadWriteLock()
    probe = _ProbeCondition(lock)
    lock._cond = probe
    lock._readers = 0
    lock._writer_active = False
    lock._writers_waiting = 0

    probe.checking = True
    lock.acquire_write()
    probe.checking = False

    assert not probe.violation, (
        "a workdir-less reader could acquire the read lock concurrently with "
        "the writer because _writer_active was published outside the condition "
        "lock's critical section (TERMINAL_CWD isolation breach)"
    )

    lock.release_write()
