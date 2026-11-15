"""``async_rewake`` shell hooks (Claude Code ``asyncRewake``): the hook runs in the background, the
event never waits for it, and exit 2 wakes the session that fired it.

Real hook processes, no mocked spawn: background lifetime, process-group kills and the routed
profile environment are what is under test, and none of them can be faked.
"""

from __future__ import annotations

import os
import queue
import time
from pathlib import Path

import pytest

from agent import shell_hooks, shell_hooks_rewake
from tests.waiting import wait_until

pytestmark = pytest.mark.platforms("posix")  # bash hooks, POSIX process groups


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    from hermes_cli import plugins
    from tools.process_registry import process_registry
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    plugins._plugin_manager = plugins.PluginManager()
    shell_hooks.reset_for_tests()
    _drain(process_registry.completion_queue)
    yield
    shell_hooks.reset_for_tests()
    _drain(process_registry.completion_queue)


def _drain(q: queue.Queue) -> list:
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


def _hook(tmp_path: Path, body: str) -> Path:
    """A hook that records its pid, then runs ``body``."""
    path = tmp_path / f"hook{len(list(tmp_path.glob('hook*.sh')))}.sh"
    path.write_text(f"#!/usr/bin/env bash\ncat >/dev/null\necho $$ > {path}.pid\n{body}\n")
    path.chmod(0o755)
    return path


def _pid(script: Path) -> int:
    marker = Path(f"{script}.pid")
    return int(wait_until(lambda: marker.exists() and marker.read_text().strip(), what=f"{script.name} started"))


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _register(event: str, script: Path, **entry) -> None:
    cfg = {"hooks": {event: [{"command": str(script), "async_rewake": True, **entry}]}}
    assert len(shell_hooks.register_from_config(cfg, accept_hooks=True)) == 1


def _fire(event: str, session_id: str = "sess-1", session_key: str = "agent:main:telegram:dm:123",
          manager=None, **kwargs):
    from gateway.session_context import clear_session_vars, set_session_vars
    from hermes_cli import plugins
    manager = manager or plugins.get_plugin_manager()
    tokens = set_session_vars(platform="telegram", chat_id="123", chat_type="dm", user_id="u1",
                              session_key=session_key, session_id=session_id)
    try:
        return manager.invoke_hook(event, session_id=session_id, **kwargs)
    finally:
        clear_session_vars(tokens)


def _queued() -> list:
    from tools.process_registry import process_registry
    return _drain(process_registry.completion_queue)


def test_exit_2_wakes_the_firing_session_without_holding_up_the_event(tmp_path):
    """The event returns while the hook is still waiting; its exit 2 queues a wake routed to the
    firing chat, carrying stderr and labelled as hook output. The hook runs in the profile that
    fired it, as a synchronous hook would."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools.process_registry_notifications import HOOK_WAKE_PREFIX, format_process_notification
    release = tmp_path / "release"
    script = _hook(tmp_path, f'until [ -e {release} ]; do sleep 0.05; done\n'
                             'echo "2 new messages from lex ($HERMES_HOME)" >&2\nexit 2')
    _register("post_llm_call", script)
    from hermes_cli import plugins
    manager = plugins.get_plugin_manager()  # the hooks' manager: managers are per profile home
    routed = tmp_path / "routed"
    routed.mkdir()
    token = set_hermes_home_override(str(routed))
    try:
        t0 = time.monotonic()
        results = _fire("post_llm_call", manager=manager)
        returned_after = time.monotonic() - t0
    finally:
        reset_hermes_home_override(token)
    assert results == [] and returned_after < 2
    _pid(script)  # running, not done
    assert _queued() == []

    release.touch()
    events = wait_until(_queued, what="hook wake queued")
    (event,) = events
    assert event["type"] == "hook_wake"
    assert (event["session_key"], event["platform"], event["chat_id"], event["parent_session_id"]) == (
        "agent:main:telegram:dm:123", "telegram", "123", "sess-1")
    assert event["output"] == f"2 new messages from lex ({routed})"
    text = format_process_notification(event)
    assert text.startswith(HOOK_WAKE_PREFIX)
    assert str(script) in text and "not by the user" in text and event["output"] in text
    assert shell_hooks_rewake.live_run_count() == 0


@pytest.mark.parametrize("body, timeout", [("exit 0", 60), ("echo boom >&2; exit 1", 60), ("sleep 30", 1)])
def test_only_exit_2_wakes(tmp_path, body, timeout):
    """Exit 0 (no longer needed), another failure and a timeout end the run quietly; the timeout
    still takes the hook down."""
    script = _hook(tmp_path, body)
    _register("post_llm_call", script, timeout=timeout)
    _fire("post_llm_call")
    pid = _pid(script)
    wait_until(lambda: shell_hooks_rewake.live_run_count() == 0, what="run finished")
    assert not _alive(pid)
    assert _queued() == []


def test_a_new_firing_replaces_the_waiting_run_of_the_same_session(tmp_path):
    """A hook on a per-turn event must not pile up one waiting process per turn: the next firing
    for the same session stops the previous run, which then never wakes. Other sessions keep theirs."""
    script = _hook(tmp_path, "sleep 30; exit 2")
    _register("post_llm_call", script)
    _fire("post_llm_call")
    first = _pid(script)
    Path(f"{script}.pid").unlink()
    _fire("post_llm_call")
    second = _pid(script)
    Path(f"{script}.pid").unlink()
    _fire("post_llm_call", session_id="sess-2", session_key="agent:main:telegram:dm:456")
    other = _pid(script)

    wait_until(lambda: not _alive(first), what="replaced run stopped")
    assert _alive(second) and _alive(other)
    assert shell_hooks_rewake.live_run_count() == 2
    assert _queued() == []


def test_session_teardown_stops_its_runs_and_nothing_else(tmp_path):
    """No orphan outlives its session, and a hook stopped this way never wakes the successor."""
    script = _hook(tmp_path, "sleep 30; exit 2")
    _register("on_session_start", script)
    _fire("on_session_start")
    mine = _pid(script)
    Path(f"{script}.pid").unlink()
    _fire("on_session_start", session_id="sess-2", session_key="agent:main:telegram:dm:456")
    other = _pid(script)

    from hermes_cli import plugins
    plugins.get_plugin_manager().invoke_hook("on_session_finalize", session_id="sess-1", platform="telegram")

    wait_until(lambda: not _alive(mine), what="finalized session's hook stopped")
    assert _alive(other)
    assert shell_hooks_rewake.live_run_count() == 1
    shell_hooks_rewake.cancel_all()  # interpreter exit
    wait_until(lambda: not _alive(other), what="remaining hook stopped at exit")
    assert _queued() == []


def test_revoking_the_hook_stops_its_waiting_run(tmp_path):
    script = _hook(tmp_path, "sleep 30; exit 2")
    _register("post_llm_call", script)
    _fire("post_llm_call")
    pid = _pid(script)
    shell_hooks.revoke(str(script))
    shell_hooks.sync_from_config(lambda: {"hooks": {"post_llm_call": [{"command": str(script), "async_rewake": True}]}})
    wait_until(lambda: not _alive(pid), what="revoked hook stopped")
    assert _queued() == []


def test_never_started_where_no_wake_can_arrive(tmp_path):
    """A delegated child has no idle loop, and a one-shot / stateless session is gone after its turn."""
    from agent.delegation_context import delegated_child_context
    from gateway.session_context import clear_session_vars, set_session_vars
    script = _hook(tmp_path, "exit 2")
    _register("post_llm_call", script)
    with delegated_child_context("child-1"):
        _fire("post_llm_call", session_id="child-1")
    assert shell_hooks_rewake.live_run_count() == 0
    tokens = set_session_vars(session_key="api-1", session_id="api-1", async_delivery=False)
    try:
        from hermes_cli import plugins
        plugins.get_plugin_manager().invoke_hook("post_llm_call", session_id="api-1")
    finally:
        clear_session_vars(tokens)
    assert shell_hooks_rewake.live_run_count() == 0
    assert not Path(f"{script}.pid").exists()


@pytest.mark.parametrize("entry, rewake, timeout", [
    ({"async_rewake": True}, True, shell_hooks.DEFAULT_REWAKE_TIMEOUT_SECONDS),
    ({"asyncRewake": True, "timeout": 7200}, True, 7200),  # Claude Code spelling; past the 300 s sync cap
    ({"async_rewake": True, "timeout": 10**6}, True, shell_hooks.MAX_REWAKE_TIMEOUT_SECONDS),
    ({"async_rewake": True, "fail_closed": True}, True, shell_hooks.DEFAULT_REWAKE_TIMEOUT_SECONDS),
    ({"async_rewake": "yes"}, False, shell_hooks.DEFAULT_TIMEOUT_SECONDS),  # not a boolean: stays synchronous
    ({"timeout": 7200}, False, shell_hooks.MAX_TIMEOUT_SECONDS),  # synchronous hooks keep their cap
])
def test_async_rewake_has_its_own_timeout_bounds(entry, rewake, timeout):
    """A background hook cannot block the action it was fired for, so fail_closed never sticks."""
    (spec,) = shell_hooks._parse_hooks_block({"pre_tool_call": [{"command": "/bin/true", **entry}]})
    assert (spec.async_rewake, spec.timeout, spec.fail_closed) == (rewake, timeout, False)


def test_the_wake_never_counts_as_words_the_user_typed():
    """Compression keeps the user's own messages verbatim; a hook's text riding a user-role row
    must be recognised as scaffolding, like a background-process notice."""
    from agent.context_compressor import _synthetic_user_row
    from agent.conversation_compression_scaffolding import is_user_role_scaffolding
    from tools.process_registry_notifications import format_process_notification
    text = format_process_notification({"type": "hook_wake", "command": "agent-bus ready",
                                        "hook_event": "post_llm_call", "output": "ping from lex", "elapsed": 90})
    assert _synthetic_user_row(text)
    assert is_user_role_scaffolding({"role": "user", "content": text})


def test_hooks_test_shows_the_wake_it_would_deliver(tmp_path):
    """``hermes hooks test`` runs an async_rewake hook in the foreground and reports the wake."""
    script = _hook(tmp_path, 'echo "you have mail" >&2; exit 2')
    spec = shell_hooks.ShellHookSpec(event="post_llm_call", command=str(script), async_rewake=True)
    assert shell_hooks.run_once(spec, {"session_id": "s"})["parsed"] == {"wake": "you have mail"}
    quiet = shell_hooks.ShellHookSpec(event="post_llm_call", command=str(_hook(tmp_path, "exit 0")), async_rewake=True)
    assert shell_hooks.run_once(quiet, {"session_id": "s"})["parsed"] == {"wake": None}
