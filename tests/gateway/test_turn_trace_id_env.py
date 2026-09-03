"""Turn-level trace correlation: HERMES_LANGFUSE_TRACE_ID reaches subprocesses.

The bundled observability/langfuse plugin opens one Langfuse trace per turn and
publishes its id via ``gateway.session_context.set_current_turn_trace_id``. A
subprocess spawned during that turn (the terminal tool's claude shim,
execute_code) must see the id in its environment so the child's own telemetry
can be stamped with it and joined back to the exact spawning turn —
``HERMES_SESSION_ID`` says which conversation, this says which turn inside it.

The id rides the same ContextVar + subprocess-env-bridge machinery as the
session identity vars (``_VAR_MAP`` →
``tools.environments.local._inject_session_context_env``) rather than a bare
``os.environ`` write, because ``os.environ`` is process-global: under a
concurrent multi-session host two turns would race one slot and a child could be
stamped with a sibling turn's trace. Companion coverage:
tests/tools/test_local_env_session_leak.py (the strip-on-unset guard),
tests/gateway/test_session_context_inheritance.py (the inheritance leak),
tests/plugins/test_langfuse_plugin.py (the publishing half).
"""

import asyncio
import os
import subprocess
import sys

import pytest

import gateway.session_context as sc
from gateway.session_context import (
    _UNSET,
    _VAR_MAP,
    get_session_env,
    reset_session_vars,
    set_current_turn_trace_id,
    set_session_vars,
)
from tools.environments.local import _make_run_env, _sanitize_subprocess_env

TRACE_VAR = "HERMES_LANGFUSE_TRACE_ID"
SESSION_VARS = list(_VAR_MAP.keys())


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + os.environ + engaged-latch slate per test, restored."""
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    saved_engaged = sc._session_context_engaged
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    for name in SESSION_VARS:
        os.environ.pop(name, None)
    sc._session_context_engaged = True  # a concurrent multi-session host
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        sc._session_context_engaged = saved_engaged
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _echo_in_subprocess(env: dict) -> str:
    """What a real child process reads for the trace var under *env*."""
    result = subprocess.run(
        [sys.executable, "-c",
         f"import os; print(os.environ.get({TRACE_VAR!r}, '<absent>'))"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


# --------------------------------------------------------------------------- #
# The feature: a bound turn trace id reaches the child
# --------------------------------------------------------------------------- #

def test_bound_turn_trace_id_reaches_a_real_subprocess():
    set_session_vars(session_key="agent:main:cli", session_id="sess-1", source="cli")
    set_current_turn_trace_id("0123456789abcdef0123456789abcdef")

    assert _echo_in_subprocess(_make_run_env({})) == "0123456789abcdef0123456789abcdef"


def test_background_spawn_path_also_carries_the_id():
    """process_registry.spawn_local builds its env via _sanitize_subprocess_env."""
    set_session_vars(session_key="agent:main:cli", session_id="sess-1", source="cli")
    set_current_turn_trace_id("bgtrace0000")

    sanitized = _sanitize_subprocess_env({"PATH": os.environ.get("PATH", "")})

    assert sanitized.get(TRACE_VAR) == "bgtrace0000"


def test_get_session_env_reads_the_published_id():
    set_current_turn_trace_id("hexhexhex")
    assert get_session_env(TRACE_VAR, "") == "hexhexhex"


# --------------------------------------------------------------------------- #
# No leak past the turn
# --------------------------------------------------------------------------- #

def test_cleared_turn_trace_id_does_not_reach_a_subprocess():
    """What _finish_trace does: the finished turn's id must be gone."""
    set_session_vars(session_key="agent:main:cli", session_id="sess-1", source="cli")
    set_current_turn_trace_id("finished-turn-trace")
    set_current_turn_trace_id("")

    assert _echo_in_subprocess(_make_run_env({})) in ("", "<absent>")
    assert os.environ.get(TRACE_VAR) is None, (
        "clearing must drop the os.environ mirror, not blank it — a CLI/cron "
        "process whose bridge falls back to the process env would still see it"
    )


def test_fresh_session_bind_does_not_inherit_a_sibling_id():
    """set_session_vars starts a turn with no trace published yet."""
    set_current_turn_trace_id("sibling-turn-trace")
    set_session_vars(session_key="agent:main:cli", session_id="sess-2", source="cli")

    assert get_session_env(TRACE_VAR, "") == ""
    assert _make_run_env({}).get(TRACE_VAR) == ""


def test_reset_session_vars_strips_the_id_from_the_child_env():
    """The handler-entry reset drops an inherited id; engaged ⇒ strip, not inherit."""
    set_current_turn_trace_id("inherited-turn-trace")
    reset_session_vars()

    assert TRACE_VAR not in _make_run_env({}), (
        "an unbound turn under an engaged host must strip the var, not inherit "
        "the process-global mirror of whichever turn wrote it last"
    )


# --------------------------------------------------------------------------- #
# Concurrency: task-local, and delegated children don't clobber the parent
# --------------------------------------------------------------------------- #

def test_concurrent_turns_do_not_cross_contaminate():
    """Two turns publishing at once each keep their own id (ContextVar, not global)."""
    async def _turn(trace_id, ready, go):
        set_session_vars(session_key=f"k:{trace_id}", session_id=trace_id, source="cli")
        set_current_turn_trace_id(trace_id)
        ready.set()
        await go.wait()  # both turns have published; the global holds one of them
        return _make_run_env({}).get(TRACE_VAR)

    async def _main():
        ready_a, ready_b, go = asyncio.Event(), asyncio.Event(), asyncio.Event()
        task_a = asyncio.create_task(_turn("trace-A", ready_a, go))
        task_b = asyncio.create_task(_turn("trace-B", ready_b, go))
        await ready_a.wait()
        await ready_b.wait()
        go.set()
        return await asyncio.gather(task_a, task_b)

    seen_a, seen_b = asyncio.run(_main())

    assert seen_a == "trace-A"
    assert seen_b == "trace-B"


def test_delegated_child_does_not_clobber_the_parent_mirror():
    """A delegate_task child runs in-process; its id must stay task-local.

    Mirrors set_current_session_id: writing the child's id to the process-global
    os.environ would replace the parent turn's id for every parent subprocess
    spawned after the child was built.
    """
    from agent.delegation_context import delegated_child_context

    set_current_turn_trace_id("parent-turn-trace")
    assert os.environ.get(TRACE_VAR) == "parent-turn-trace"

    with delegated_child_context():
        set_current_turn_trace_id("child-turn-trace")
        # The child's own tools resolve the child id through the ContextVar …
        assert get_session_env(TRACE_VAR, "") == "child-turn-trace"
        assert _make_run_env({}).get(TRACE_VAR) == "child-turn-trace"
        # … while the process-global mirror still belongs to the parent turn.
        assert os.environ.get(TRACE_VAR) == "parent-turn-trace"


# --------------------------------------------------------------------------- #
# The shared bash snapshot must not persist a turn's id
# --------------------------------------------------------------------------- #

def test_snapshot_dump_unsets_the_turn_trace_id():
    """One long-lived backend serves many turns; the snapshot is sourced by all.

    Persisting the FIRST turn's id would make every later turn's command source
    a stale trace, overriding the correct per-command Popen env.
    """
    from tools.environments.base import _export_dump_excluding_session_vars

    assert TRACE_VAR in _export_dump_excluding_session_vars('"$tmp"')
