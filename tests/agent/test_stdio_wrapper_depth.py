"""Regression: the stdio wrapper chain must stay bounded across agent builds.

Every AIAgent construction calls ``_install_safe_stdio()`` and every silenced
worker enters ``thread_scoped_silence()``. Each used to wrap whatever
``sys.stdout``/``sys.stderr`` currently was without looking inside, so the two
installers alternated ``_SafeWriter(_ThreadRoutingStream(...))`` layers — two
per native run — until attribute delegation through the chain blew the
recursion limit in long-lived gateways, before AIAgent construction.
"""

import sys

import agent.thread_scoped_output as thread_output
from agent.process_bootstrap import _SafeWriter, _install_safe_stdio
from agent.thread_scoped_output import _ThreadRoutingStream, thread_scoped_silence

# A stable state is one routing proxy, optionally over a safe writer; anything
# beyond that means the installers are still stacking generations.
_MAX_STABLE_DEPTH = 4
_NATIVE_RUN_CYCLES = 50


def _wrapper_depth(stream) -> int:
    depth = 0
    seen = set()
    while isinstance(stream, (_SafeWriter, _ThreadRoutingStream)) and id(stream) not in seen:
        seen.add(id(stream))
        depth += 1
        stream = stream._inner if isinstance(stream, _SafeWriter) else stream._passthrough
    return depth


def test_interleaved_stdio_installers_keep_wrapper_depth_bounded():
    original = (sys.stdout, sys.stderr)
    installed, routing_states = dict(thread_output._installed), dict(thread_output._routing_states)
    try:
        thread_output._installed.clear()
        thread_output._routing_states.clear()
        sys.stdout = sys.__stdout__ or sys.stdout
        sys.stderr = sys.__stderr__ or sys.stderr

        for _ in range(_NATIVE_RUN_CYCLES):
            _install_safe_stdio()
            with thread_scoped_silence():
                pass

        assert _wrapper_depth(sys.stdout) <= _MAX_STABLE_DEPTH
        assert _wrapper_depth(sys.stderr) <= _MAX_STABLE_DEPTH
        # Capability lookups through whatever is installed must stay cheap and
        # total — this is the lookup that crashed gateway startup.
        assert getattr(sys.stdout, "line_buffering", False) in (True, False)
        assert getattr(sys.stderr, "line_buffering", False) in (True, False)
    finally:
        sys.stdout, sys.stderr = original
        thread_output._installed.clear()
        thread_output._installed.update(installed)
        thread_output._routing_states.clear()
        thread_output._routing_states.update(routing_states)
