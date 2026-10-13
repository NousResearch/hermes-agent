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

import pytest

import agent.thread_scoped_output as thread_output
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



def test_safe_stdio_and_silence_alternation_never_grows_the_chain(monkeypatch):
    """``_install_safe_stdio`` runs on every AIAgent build; silenced worker threads
    install the routing proxy.  Each wrapper only guards against re-wrapping itself,
    so alternating the two used to stack proxy/safe-writer pairs until any attribute
    miss (``hermes_logging``'s per-build ``line_buffering`` read) died with
    RecursionError and swallowed console writes in long-lived gateway processes
    (gateway.error.log, 2026-10-09).  However often they interleave, the delegation
    chain must stay bounded and routing must keep working."""
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_sinks", {})
    monkeypatch.setattr(thread_output, "_routing_states", {})
    original_stdout, original_stderr = sys.stdout, sys.stderr
    real_out, real_err = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = real_out, real_err

    def chain_depth(stream):
        depth = 0
        while isinstance(stream, (thread_output._ThreadRoutingStream, _SafeWriter)):
            stream = stream._passthrough if isinstance(stream, thread_output._ThreadRoutingStream) else stream._inner
            depth += 1
        return depth

    try:
        _install_safe_stdio()
        with thread_scoped_silence():
            pass
        settled = chain_depth(sys.stdout)

        for _ in range(300):
            _install_safe_stdio()
            with thread_scoped_silence():
                pass

        assert chain_depth(sys.stdout) == settled <= 4
        assert chain_depth(sys.stderr) == settled
        # The exact read that killed gateway agent builds must resolve.
        assert getattr(sys.stdout, "line_buffering", False) is False
        # Routing still holds: silenced writes vanish, unsilenced writes arrive.
        with thread_scoped_silence():
            sys.stdout.write("hidden\n")
        sys.stdout.write("kept\n")
        assert real_out.getvalue() == "kept\n"
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr


def _delegation_cycle():
    """The incident's hand-built shape: the bootstrap _SafeWriter wrapping the
    thread-routing proxy, whose passthrough delegates back into that writer."""
    state = thread_output._RoutingState(sink=io.StringIO())
    routing = thread_output._ThreadRoutingStream(passthrough=None, state=state)
    safe = _SafeWriter(routing)
    routing._passthrough = safe
    return safe, routing


def test_delegation_cycle_misses_as_attribute_error_never_recursion():
    """An intentional two-proxy cycle (bootstrap proxy wrapping the thread-scoped
    proxy wrapping it back) must make attribute misses look like plain misses —
    AttributeError, so ``getattr(obj, name, default)`` yields the default —
    instead of recursing to RecursionError, which ``getattr``'s default cannot
    catch and which killed every agent build in the long-lived gateway
    (gateway.error.log, 2026-10-09).  A self-wrapping _SafeWriter must
    terminate the same way."""
    safe, routing = _delegation_cycle()
    for proxy in (safe, routing):
        with pytest.raises(AttributeError):
            proxy.line_buffering
        assert getattr(proxy, "line_buffering", False) is False

    self_loop = _SafeWriter(None)
    object.__setattr__(self_loop, "_inner", self_loop)
    with pytest.raises(AttributeError):
        self_loop.line_buffering
    assert getattr(self_loop, "line_buffering", False) is False


def test_logging_buffering_probes_survive_a_delegation_cycled_stdio(monkeypatch):
    """``setup_logging`` runs on every AIAgent build and probes sys.stdout for
    ``line_buffering``/``reconfigure`` (and sys.stderr for ``encoding``); with a
    delegation-cycled proxy bound there the probes must see a miss and move on,
    never crash setup (the 2026-10-09 gateway incident's fatal read)."""
    import hermes_logging

    safe, _routing = _delegation_cycle()
    monkeypatch.setattr(hermes_logging, "_line_buffered_stdout", None)
    monkeypatch.setattr(sys, "stdout", safe)
    hermes_logging._line_buffer_piped_stdout()  # must not raise
    assert hermes_logging._line_buffered_stdout is None

    safe_err, _routing_err = _delegation_cycle()
    monkeypatch.setattr(sys, "stderr", safe_err)
    assert hermes_logging._safe_stderr() is safe_err
