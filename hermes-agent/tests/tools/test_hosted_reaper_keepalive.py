"""The idle reaper keeps a hosted sandbox while any process carrying its key runs, even when the
environment object cached under that key is not the one the process was spawned through; once the
process is gone, the sandbox is suspended, not destroyed."""

from __future__ import annotations

from contextlib import contextmanager

from tests.tools.test_sandbox_scope_conformance import A
from tests.tools.test_sandbox_scope_conformance import provider as _conformance_provider
from tools.environments import sandbox_conformance as c

provider = _conformance_provider  # the recording-sandbox fixture, by its name


@contextmanager
def _running_under_replaced_env(provider):
    """Start a hosted background process through the terminal tool, then cache a second environment
    of the same sandbox (created and wrapped as the hosted cache holds it) under the sandbox key."""
    from tools import terminal_tool
    from tools.environments.sandbox_scope import create_hosted_environment
    from tools.process_registry import process_registry
    with c._bind(A):
        started = c.call_tool("terminal", {"command": "sleep 600", "background": True}, task_id="t")
    session_id = started.get("session_id")
    assert session_id, f"the background spawn failed: {started}"
    key = c._identity(A).key
    try:
        original = terminal_tool._active_environments[key]
        with c._bind(A):
            replacement = create_hosted_environment(c.STUB_BACKEND, provider, cwd="/workspace", timeout=60)
        assert replacement is not original and replacement.sandbox == original.sandbox
        terminal_tool._active_environments[key] = replacement
        # The state the key rule exists for: the process is no longer matched by environment object.
        assert not process_registry.has_running_in_env(key, replacement)
        yield key, session_id
    finally:
        with c._bind(A):
            c.call_tool("process_manage", {"action": "kill", "session_id": session_id}, task_id="t")
        terminal_tool._active_environments.pop(key, None)
        terminal_tool._last_activity.pop(key, None)


def test_hosted_sandbox_is_kept_while_a_process_carrying_its_key_runs(provider):
    """Pins the hosted keep-alive rule: a sandbox is kept while a process carrying its key runs, even
    when its cached environment is not the object that process was spawned through. No production
    path produces that state today; the cache entry is replaced directly."""
    from tools import terminal_tool
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    with _running_under_replaced_env(provider) as (key, _session_id):
        c._age_sandbox(A)
        seen = len(provider.calls)
        _cleanup_inactive_envs(lifetime_seconds=300)
        assert key in terminal_tool._active_environments, "a sandbox with running work was reaped"
        assert not [call.op for call in provider.calls[seen:] if call.op in ("cleanup", "stop_idle")]


def test_hosted_sandbox_is_suspended_once_that_process_has_exited(provider):
    """Control: in the same replaced-environment state, once the process is killed the next reaper pass
    suspends the sandbox (``stop_idle``); it is not destroyed."""
    from tools import terminal_tool
    from tools.terminal_tool_lifecycle import _cleanup_inactive_envs
    with _running_under_replaced_env(provider) as (key, session_id):
        with c._bind(A):
            killed = c.call_tool("process_manage", {"action": "kill", "session_id": session_id}, task_id="t")
        assert killed.get("status") in ("killed", "already_exited"), killed
        c._age_sandbox(A)
        seen = len(provider.calls)
        _cleanup_inactive_envs(lifetime_seconds=300)
        assert key not in terminal_tool._active_environments, "an idle sandbox was kept"
        lifecycle = [call.op for call in provider.calls[seen:] if call.op in ("cleanup", "stop_idle")]
        assert lifecycle == ["stop_idle"], f"the idle sandbox was not suspended: {lifecycle}"
