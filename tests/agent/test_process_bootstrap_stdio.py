"""Regression tests for agent.process_bootstrap._install_safe_stdio (#139687).

Behaviour contract: interleaving agent builds (``_install_safe_stdio``) with
silenced workers (``thread_scoped_output``) must never grow the ``sys.stdout``
proxy chain. ``_ThreadRoutingStream._forward`` already swallows the
OSError/ValueError ``_SafeWriter`` guards against, so wrapping a routing proxy
stacked two more links onto the chain on every cycle until attribute probes
(``getattr(stream, "line_buffering", ...)`` in ``hermes_logging``) blew the
recursion limit on long-lived gateway processes.

The installers also share one installation lock with the routing installer:
a stream snapshot taken outside it could publish over a router a silenced
worker just installed, sending its output to the pre-silence destination.
"""

import io
import sys
import threading

import agent.process_bootstrap as bootstrap
import agent.thread_scoped_output as thread_output
from agent.thread_scoped_output import thread_scoped_silence


def _chain_links(stream, cap=64):
    """Count proxy hops below ``stream`` down to the real underlying stream."""
    links = 0
    cur = stream
    while True:
        if isinstance(cur, thread_output._ThreadRoutingStream):
            cur = cur._passthrough
        elif isinstance(cur, bootstrap._SafeWriter):
            cur = cur._inner
        else:
            return links
        links += 1
        assert links <= cap, f"stdio proxy chain grew to {links} links"


def test_build_and_silence_cycle_keeps_the_stdio_chain_short(monkeypatch):
    """Alternating agent builds and silences must not stack the two wrappers."""
    monkeypatch.setattr(thread_output, "_installed", {})
    real_out = io.StringIO()
    real_err = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_out)
    monkeypatch.setattr(sys, "stderr", real_err)
    for _ in range(40):
        bootstrap._install_safe_stdio()  # an agent build
        with thread_scoped_silence():  # a worker silencing itself
            sys.stdout.write("hidden\n")
        sys.stdout.write("loud\n")
    assert _chain_links(sys.stdout) <= 2
    assert "hidden" not in real_out.getvalue()
    assert real_out.getvalue().count("loud") == 40


def test_safe_stdio_leaves_the_installed_routing_proxy_alone(monkeypatch):
    """Once the routing proxy owns sys.stdout, a build must not re-wrap it."""
    monkeypatch.setattr(thread_output, "_installed", {})
    real_out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", real_out)
    with thread_scoped_silence():
        pass
    installed = sys.stdout
    assert isinstance(installed, thread_output._ThreadRoutingStream)
    bootstrap._install_safe_stdio()
    assert sys.stdout is installed
    sys.stdout.write("post-build\n")
    assert "post-build\n" in real_out.getvalue()


class _RecordingLock:
    """Lock proxy recording acquire/release for the transaction test below."""

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

    def acquire(self, *args, **kwargs):
        self.events.append("acquire")
        return self._inner.acquire(*args, **kwargs)

    def release(self):
        self.events.append("release")
        self._inner.release()


def test_safe_stdio_transaction_shares_the_router_install_lock(monkeypatch):
    """The whole read/check/publish transaction must hold the shared install lock.

    A snapshot taken outside the lock can replace a router a silenced worker
    just published (its silencing state stays registered but nothing consults
    it). The lock is the coordination point between the two installers.
    """
    proxy = _RecordingLock(threading.Lock())
    monkeypatch.setattr(thread_output, "_install_lock", proxy)
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    bootstrap._install_safe_stdio()

    assert proxy.events, "safe-stdio install must participate in the shared install lock"
    assert proxy.events[-1] == "release", "the transaction must release the shared lock"


def test_build_does_not_bypass_an_active_silence_under_a_redirect(monkeypatch):
    """A build running while a worker is silenced must not leak that worker's output.

    With a global redirect active, the silenced worker's writes must stay absent
    from the redirect destination while a loud writer still reaches it — on both
    streams.
    """
    monkeypatch.setattr(thread_output, "_installed", {})
    capture_out, capture_err = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", capture_out)
    monkeypatch.setattr(sys, "stderr", capture_err)
    entered = threading.Event()

    def worker():
        with thread_scoped_silence():
            entered.set()
            sys.stdout.write("hidden\n")
            sys.stderr.write("hidden-err\n")

    thread = threading.Thread(target=worker)
    thread.start()
    assert entered.wait(timeout=10)

    bootstrap._install_safe_stdio()  # an agent build overlapping the silence
    sys.stdout.write("loud\n")
    sys.stderr.write("loud-err\n")
    thread.join(timeout=10)
    assert not thread.is_alive()

    assert "hidden" not in capture_out.getvalue(), "silenced worker leaked into the capture"
    assert "hidden-err" not in capture_err.getvalue(), "silenced worker leaked into the capture"
    assert "loud\n" in capture_out.getvalue()
    assert "loud-err\n" in capture_err.getvalue()
