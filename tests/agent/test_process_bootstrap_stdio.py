"""Regression tests for agent.process_bootstrap._install_safe_stdio (#139687).

Behaviour contract: interleaving agent builds (``_install_safe_stdio``) with
silenced workers (``thread_scoped_output``) must never grow the ``sys.stdout``
proxy chain. ``_ThreadRoutingStream._forward`` already swallows the
OSError/ValueError ``_SafeWriter`` guards against, so wrapping a routing proxy
stacked two more links onto the chain on every cycle until attribute probes
(``getattr(stream, "line_buffering", ...)`` in ``hermes_logging``) blew the
recursion limit on long-lived gateway processes.
"""

import io
import sys

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
