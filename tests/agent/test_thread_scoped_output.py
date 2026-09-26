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
import pytest
from agent.process_bootstrap import _SafeWriter, _install_safe_stdio
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
    results_lock = threading.Lock()

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


class _BrokenPipe(io.StringIO):
    """A stdout whose writes fail the way a closed pipe does."""

    def write(self, _data):
        raise OSError(5, "Input/output error")

    def flush(self):
        raise OSError(5, "Input/output error")


def _close_sinks():
    """Release the /dev/null sinks this test made silence install."""
    for sink in list(thread_output._sinks.values()):
        with contextlib.suppress(Exception):
            sink.close()


@pytest.mark.parametrize("safe_first", [True, False])
def test_repeated_installs_keep_the_proxy_live_and_output_flowing(monkeypatch, safe_first):
    """``_install_safe_stdio`` runs on every agent build and again per subagent.

    It must not wrap the thread-scoped routing proxy. A wrapped proxy is no longer
    adopted by ``_ensure_installed``, so each later build+silence cycle stacks two more
    layers on the stdout chain; past the interpreter's recursion limit the proxy's own
    ``_forward`` swallows the RecursionError and console output silently stops arriving.
    """
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "_routing_states", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    cycles = sys.getrecursionlimit()  # measured: output stops after 332 cycles at 1000
    try:
        real, real_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = real, real_err
        if safe_first:
            _install_safe_stdio()
        with thread_scoped_silence():
            pass
        routed, routed_err = sys.stdout, sys.stderr
        for _ in range(cycles):
            _install_safe_stdio()
            with thread_scoped_silence():
                print("silenced")

        assert sys.stdout is routed, "the routing proxy is no longer the live stdout"
        assert sys.stderr is routed_err, "the routing proxy is no longer the live stderr"
        print("kept")
        print("kept-err", file=sys.stderr)
        assert "silenced" not in real.getvalue()
        assert real.getvalue().count("kept") == 1, "console output stopped reaching the terminal"
        assert real_err.getvalue().count("kept-err") == 1
    finally:
        _close_sinks()
        sys.stdout, sys.stderr = original_stdout, original_stderr


@pytest.mark.parametrize("safe_first", [True, False])
def test_a_dead_pipe_never_raises_through_print(monkeypatch, safe_first):
    """A broken pipe must be swallowed before and after the proxy is installed."""
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {}, raising=False)
    monkeypatch.setattr(thread_output, "_routing_states", {}, raising=False)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = _BrokenPipe(), io.StringIO()
        if safe_first:
            _install_safe_stdio()
        with thread_scoped_silence():
            pass
        for _ in range(3):
            _install_safe_stdio()
        print("must not raise")
        sys.stdout.write("still must not raise\n")
        sys.stdout.flush()
    finally:
        _close_sinks()
        sys.stdout, sys.stderr = original_stdout, original_stderr
