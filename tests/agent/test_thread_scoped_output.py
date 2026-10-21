"""Tests for agent.thread_scoped_output.thread_scoped_silence.

Behaviour contract: a thread inside ``thread_scoped_silence()`` has its
stdout/stderr routed to devnull, while every OTHER thread keeps writing to the
real stream — even concurrently, while the first thread is still inside the
context.  This is the property the old process-global
``contextlib.redirect_stdout(devnull)`` violated (issue #55769 / #55925).
"""

import contextlib
import io
import sys
import threading
import time

import agent.thread_scoped_output as thread_output
from agent.thread_scoped_output import thread_scoped_silence


def _run_with_real_stream(fn):
    """Bind a StringIO as the real stdout, run fn, return what reached it."""
    real_out = io.StringIO()
    orig = sys.stdout
    sys.stdout = real_out
    try:
        fn()
    finally:
        sys.stdout = orig
    return real_out.getvalue()






def test_stderr_is_also_routed_per_thread():
    real_err = io.StringIO()
    orig = sys.stderr
    sys.stderr = real_err
    try:
        with thread_scoped_silence():
            sys.stderr.write("err-dropped\n")
        sys.stderr.write("err-kept\n")
    finally:
        sys.stderr = orig
    out = real_err.getvalue()
    assert "err-dropped" not in out
    assert "err-kept" in out






def test_many_concurrent_silenced_and_loud_threads():
    """Stress: interleaved silenced/loud threads keep their respective fates."""
    start = threading.Event()

    def silenced(i):
        start.wait(timeout=2.0)
        with thread_scoped_silence():
            print(f"S{i}")
            time.sleep(0.05)

    def loud(i):
        start.wait(timeout=2.0)
        time.sleep(0.02)
        print(f"L{i}")

    def body():
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=silenced, args=(i,)))
            threads.append(threading.Thread(target=loud, args=(i,)))
        for t in threads:
            t.start()
        start.set()
        for t in threads:
            t.join(timeout=15.0)
        assert not any(t.is_alive() for t in threads), "straggler thread would truncate captured output"

    captured = _run_with_real_stream(body)
    for i in range(5):
        assert f"S{i}" not in captured, f"silenced S{i} leaked"
        assert f"L{i}" in captured, f"loud L{i} swallowed"


def test_repeated_contexts_never_write_to_a_closed_sink():
    """The installed proxy must survive later silenced workers."""
    original = sys.stdout
    try:
        for _ in range(3):
            with thread_scoped_silence():
                sys.stdout.write("hidden\n")
            sys.stdout.fileno()
    finally:
        sys.stdout = original


def test_temporary_global_redirects_do_not_allocate_new_sinks(monkeypatch):
    """A displaced proxy is temporary, not a reason to leak another FD pair."""
    opened_sinks = []

    def fake_open(*_args, **_kwargs):
        sink = io.StringIO()
        opened_sinks.append(sink)
        return sink

    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "open", fake_open, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        original_proxies = dict(thread_output._installed)

        for _ in range(20):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with thread_scoped_silence():
                    print("hidden")

        with thread_scoped_silence():
            pass
        assert len(opened_sinks) == 2
        assert thread_output._installed == original_proxies
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def test_agent_builds_keep_the_installed_routing_proxy(monkeypatch):
    """Agent builds between silences must not stack wrappers on sys.stdout/stderr (#139687)."""
    from agent.process_bootstrap import _install_safe_stdio

    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    _install_safe_stdio()
    with thread_scoped_silence():
        pass
    routers = (sys.stdout, sys.stderr)

    for _ in range(3):
        _install_safe_stdio()
        with thread_scoped_silence():
            pass

    assert (sys.stdout, sys.stderr) == routers


class _RecordingLock:
    """Lock proxy recording acquire/release, injected in place of the shared lock."""

    def __init__(self, inner):
        self._inner = inner
        self.events = []

    def __enter__(self):
        self.events.append("acquire")
        self._inner.acquire()
        return self

    def __exit__(self, *exc_info):
        self.events.append("release")
        self._inner.release()
        return False


def test_safe_stdio_transaction_shares_the_router_install_lock(monkeypatch):
    """The safe-stdio read/check/publish transaction must hold the shared install lock.

    A snapshot taken outside the lock can be assigned after a worker published
    its router, replacing it with a SafeWriter around the pre-silence stream:
    the worker's silence stays registered but nothing consults it anymore.
    """
    from agent.process_bootstrap import _install_safe_stdio

    proxy = _RecordingLock(threading.Lock())
    monkeypatch.setattr(thread_output, "_install_lock", proxy)
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    _install_safe_stdio()

    assert proxy.events, "safe-stdio install must participate in the shared install lock"
    assert proxy.events[-1] == "release", "the transaction must release the shared lock"


def test_redaction_install_shares_the_router_install_lock(monkeypatch):
    """The detached-redaction installer must hold the same shared install lock."""
    import hermes_logging

    proxy = _RecordingLock(threading.Lock())
    monkeypatch.setattr(thread_output, "_install_lock", proxy)
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    hermes_logging.redact_detached_stdio()

    assert proxy.events, "redaction install must participate in the shared install lock"
    assert proxy.events[-1] == "release", "the transaction must release the shared lock"


def test_build_overlapping_a_silence_keeps_the_worker_hidden(monkeypatch):
    """A build running while a worker is silenced must not leak that worker's output.

    With a global redirect active, the silenced worker's writes must stay absent
    from the redirect destination while a loud writer still reaches it — the
    stale-snapshot interleaving replaced the router with a SafeWriter around the
    pre-silence stream and sent worker output into a /curator capture.
    """
    from agent.process_bootstrap import _install_safe_stdio

    monkeypatch.setattr(thread_output, "_installed", {})
    capture_out, capture_err = io.StringIO(), io.StringIO()
    original = (sys.stdout, sys.stderr)
    sys.stdout, sys.stderr = capture_out, capture_err
    entered = threading.Event()
    try:
        def worker():
            with thread_scoped_silence():
                entered.set()
                sys.stdout.write("hidden\n")
                sys.stderr.write("hidden-err\n")

        thread = threading.Thread(target=worker)
        thread.start()
        assert entered.wait(timeout=10)

        _install_safe_stdio()  # an agent build overlapping the silence
        sys.stdout.write("loud\n")
        sys.stderr.write("loud-err\n")
        thread.join(timeout=10)
        assert not thread.is_alive()

        assert "hidden" not in capture_out.getvalue(), "silenced worker leaked into the capture"
        assert "hidden-err" not in capture_err.getvalue(), "silenced worker leaked into the capture"
        assert "loud\n" in capture_out.getvalue()
        assert "loud-err\n" in capture_err.getvalue()
    finally:
        sys.stdout, sys.stderr = original


def test_silence_survives_redirect_restoring_an_older_proxy(monkeypatch):
    """Silencing is stream-wide, even when a redirect swaps proxy generations."""
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    passthrough = io.StringIO()
    sys.stdout = passthrough
    entered = threading.Event()
    release = threading.Event()

    try:
        with thread_scoped_silence():
            pass

        def worker():
            with thread_scoped_silence():
                entered.set()
                assert release.wait(timeout=10)
                print("must-stay-silenced")

        redirected = io.StringIO()
        with contextlib.redirect_stdout(redirected):
            thread = threading.Thread(target=worker)
            thread.start()
            assert entered.wait(timeout=10)

        release.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert "must-stay-silenced" not in passthrough.getvalue()
        assert "must-stay-silenced" not in redirected.getvalue()
    finally:
        release.set()
        sys.stdout, sys.stderr = original_stdout, original_stderr
