"""Regression: the stdio wrapper chain must stay bounded across agent builds.

Every AIAgent construction calls ``_install_safe_stdio()`` and every silenced
worker enters ``thread_scoped_silence()``. Each used to wrap whatever
``sys.stdout``/``sys.stderr`` currently was without looking inside, so the two
installers alternated ``_SafeWriter(_ThreadRoutingStream(...))`` layers — two
per native run — until attribute delegation through the chain blew the
recursion limit in long-lived gateways, before AIAgent construction.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import agent.thread_scoped_output as thread_output
from agent.process_bootstrap import _install_safe_stdio
from agent.stdio_wrappers import _SafeWriter
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


def test_silence_path_never_pulls_the_bootstrap_chain():
    """``_ensure_installed`` must not import agent.process_bootstrap — even lazily.

    A function-level ``from agent.process_bootstrap import ...`` inside
    ``_ensure_installed`` made this hot per-thread stdio path trigger the
    module-scope ``hermes_bootstrap`` import on first call in a fresh process:
    import-time dependency activation, real-HERMES_HOME filesystem I/O and
    sys.path mutation at an arbitrary runtime moment (and an import-lock
    deadlock risk under threads). Import-order-dependent, hence the subprocess.
    """
    repo_root = Path(thread_output.__file__).resolve().parent.parent
    probe = (
        "import json, sys\n"
        "import agent.thread_scoped_output as t\n"
        "t._ensure_installed('stdout', sys.__stdout__ or sys.stdout)\n"
        "print(json.dumps({'hb': 'hermes_bootstrap' in sys.modules,"
        " 'pb': 'agent.process_bootstrap' in sys.modules}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env=dict(os.environ, PYTHONPATH=str(repo_root)),
        check=True,
        timeout=120,
    )
    flags = json.loads(result.stdout.strip().splitlines()[-1])
    assert flags == {"hb": False, "pb": False}
