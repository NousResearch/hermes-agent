"""Behavior contracts for best-effort final agent-result observation."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import hermes_cli.plugins as plugins_module
from agent.result_observer import (
    MAX_ERROR_CHARS,
    MAX_IDENTIFIER_CHARS,
    MAX_LINEAGE_HASH_CHARS_PER_FIELD,
    MAX_OUTPUT_CHARS,
    POST_AGENT_RESULT_HOOK,
    build_agent_result_event,
)
from hermes_cli.plugins import (
    LoadedPlugin,
    PluginContext,
    PluginManager,
    PluginManifest,
)
from run_agent import AIAgent


EXPECTED_EVENT_KEYS = {
    "schema_version",
    "event",
    "actor",
    "role",
    "session_id",
    "task_id",
    "turn_id",
    "turn_started",
    "subagent_id",
    "parent_session_id",
    "parent_turn_id",
    "parent_subagent_id",
    "platform",
    "identity_complete",
    "identity_truncated_fields",
    "lineage_sha256",
    "lineage_hash_input_complete",
    "lineage_hash_input_truncated_fields",
    "result_kind",
    "completed",
    "failed",
    "partial",
    "interrupted",
    "response_transformed",
    "output",
    "output_type",
    "output_chars",
    "output_truncated",
    "output_excerpt_sha256",
    "error",
    "error_chars",
    "error_truncated",
    "exception_type",
}


def _bare_agent(*, subagent: bool = False) -> AIAgent:
    agent = object.__new__(AIAgent)
    agent.session_id = "child-session" if subagent else "root-session"
    agent.platform = "subagent" if subagent else "telegram"
    agent._parent_session_id = "root-session" if subagent else None
    agent._current_task_id = "old-task"
    agent._current_turn_id = "old-turn"
    agent._subagent_id = "sa-7" if subagent else ""
    agent._parent_subagent_id = "sa-parent" if subagent else ""
    agent._parent_turn_id = "parent-turn" if subagent else ""
    agent._delegate_role = "leaf" if subagent else ""
    return agent


def _returning_stub(result: Any, *, turn_id: str = "new-turn"):
    def _run(
        agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        persist_user_timestamp=None,
        moa_config=None,
    ):
        agent._current_task_id = task_id or "generated-task"
        agent._current_turn_id = turn_id
        return result

    return _run


def _register_observer(monkeypatch, callback, *, manager=None) -> PluginManager:
    manager = manager or PluginManager()
    context = PluginContext(PluginManifest(name="result-observer-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, callback)
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
    return manager


def _provider_text_response(text: str) -> SimpleNamespace:
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_forwarder_emits_allowlisted_transformed_root_result(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()
    result = {
        "final_response": "transformed answer",
        "completed": True,
        "response_transformed": True,
        "messages": [{"role": "user", "content": "TRANSCRIPT-SECRET"}],
        "conversation_history": ["HISTORY-SECRET"],
        "tool_calls": ["TOOL-SECRET"],
        "provider": "PROVIDER-SECRET",
    }

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result),
    ):
        returned = agent.run_conversation("request", task_id="root-task")

    event = captured.get(timeout=2)
    assert returned is result
    assert set(event) == EXPECTED_EVENT_KEYS
    assert event["event"] == "agent.result"
    assert event["actor"] == "root"
    assert event["role"] == "root"
    assert event["session_id"] == "root-session"
    assert event["task_id"] == "root-task"
    assert event["turn_id"] == "new-turn"
    assert event["turn_started"] is True
    assert event["result_kind"] == "returned"
    assert event["completed"] is True
    assert event["response_transformed"] is True
    assert event["output"] == "transformed answer"
    assert (
        not {
            "messages",
            "conversation_history",
            "tool_calls",
            "provider",
        }
        & event.keys()
    )
    assert "SECRET" not in repr(event)


def test_discovered_plugin_observes_real_forwarder_contract(
    tmp_path: Path, monkeypatch
):
    assert POST_AGENT_RESULT_HOOK in plugins_module.VALID_HOOKS

    home = tmp_path / "hermes"
    plugin_dir = home / "plugins" / "result-observer-e2e"
    plugin_dir.mkdir(parents=True)
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - result-observer-e2e\n",
        encoding="utf-8",
    )
    (plugin_dir / "plugin.yaml").write_text(
        "name: result-observer-e2e\nversion: 0.1.0\n",
        encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "import threading\n"
        "EVENTS = []\n"
        "DONE = threading.Event()\n"
        "def _capture(*, event):\n"
        "    EVENTS.append(event)\n"
        "    DONE.set()\n"
        "def register(ctx):\n"
        "    ctx.register_hook('post_agent_result', _capture)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)
    loaded = manager._plugins["result-observer-e2e"]
    assert loaded.enabled is True
    assert loaded.module is not None

    agent = _bare_agent()
    result = {"final_response": "discovered observer", "completed": True}
    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result, turn_id="discovered-turn"),
    ):
        returned = agent.run_conversation("request", task_id="discovered-task")

    assert returned is result
    assert loaded.module.DONE.wait(timeout=2)
    assert len(loaded.module.EVENTS) == 1
    event = loaded.module.EVENTS[0]
    assert set(event) == EXPECTED_EVENT_KEYS
    assert event["task_id"] == "discovered-task"
    assert event["turn_id"] == "discovered-turn"


def test_real_gateway_root_and_native_delegate_share_observer_contract(
    tmp_path: Path, monkeypatch
):
    """Drive real AIAgent loops and delegate construction; stub only provider I/O."""
    captured: queue.Queue = queue.Queue()
    manager = _register_observer(monkeypatch, lambda *, event: captured.put(event))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    provider_platforms = []

    def _provider_boundary(agent, api_kwargs):
        provider_platforms.append(agent.platform)
        text = (
            "native child result"
            if agent.platform == "subagent"
            else "gateway root result"
        )
        return _provider_text_response(text)

    # Force the ordinary non-streaming provider boundary for both the gateway
    # root and the genuinely constructed native child. No conversation-loop,
    # AIAgent constructor, delegate builder, or child run method is stubbed.
    monkeypatch.setattr(AIAgent, "_disable_streaming", True, raising=False)
    monkeypatch.setattr(AIAgent, "_interruptible_api_call", _provider_boundary)

    root = AIAgent(
        model="test/model",
        provider="openrouter",
        api_mode="chat_completions",
        api_key="test-key",
        base_url="http://127.0.0.1:9/v1",
        max_iterations=2,
        enabled_toolsets=[],
        quiet_mode=True,
        platform="telegram",
        session_id="gateway-root-session",
        gateway_session_key="agent:main:telegram:dm:user-7",
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        root_result = root.run_conversation(
            "gateway request", task_id="gateway-root-task"
        )
        assert root_result["final_response"] == "gateway root result"

        from tools.delegate_tool import delegate_task

        delegated = json.loads(
            delegate_task(goal="native delegated request", parent_agent=root)
        )
        assert delegated["results"][0]["summary"] == "native child result"
        # Exactly the root and child provider calls occurred. Observation did
        # not invoke a model or create an additional agent run.
        assert provider_platforms == ["telegram", "subagent"]

        events = [captured.get(timeout=5), captured.get(timeout=5)]
        assert [event["actor"] for event in events] == ["root", "subagent"]
        root_event, child_event = events
        assert root_event["session_id"] == "gateway-root-session"
        assert root_event["task_id"] == "gateway-root-task"
        assert root_event["platform"] == "telegram"
        assert child_event["role"] == "leaf"
        assert child_event["parent_session_id"] == "gateway-root-session"
        assert child_event["parent_turn_id"] == root_event["turn_id"]
        assert child_event["subagent_id"].startswith("sa-0-")
        assert child_event["task_id"] == child_event["subagent_id"]
        assert manager.observer_health()["degraded"] is False
    finally:
        root.close()


def test_event_caps_output_error_and_identity_without_spreading_result():
    agent = _bare_agent(subagent=True)
    setattr(agent, "session_id", "s" * (MAX_IDENTIFIER_CHARS + 50))
    setattr(agent, "_current_task_id", "t" * (MAX_IDENTIFIER_CHARS + 50))
    setattr(agent, "_current_turn_id", "turn-new")
    output = "A" * MAX_OUTPUT_CHARS + "MIDDLE-SECRET" + "Z" * 400
    error = "E" * MAX_ERROR_CHARS + "ERROR-TAIL"
    result = {
        "final_response": output,
        "error": error,
        "partial": True,
        "messages": ["TRANSCRIPT-SECRET"],
        "unknown": "UNKNOWN-SECRET",
    }

    event = build_agent_result_event(
        agent,
        result=result,
        previous_turn_id="turn-old",
        requested_task_id="ignored-request-task",
    )

    assert set(event) == EXPECTED_EVENT_KEYS
    assert len(event["session_id"]) == MAX_IDENTIFIER_CHARS
    assert len(event["task_id"]) == MAX_IDENTIFIER_CHARS
    assert event["identity_complete"] is False
    assert event["identity_truncated_fields"] == "session_id,task_id"
    assert len(event["lineage_sha256"]) == 64
    assert event["lineage_hash_input_complete"] is True
    assert event["lineage_hash_input_truncated_fields"] == ""
    assert len(event["output"]) <= MAX_OUTPUT_CHARS
    assert event["output_chars"] == len(output)
    assert event["output_truncated"] is True
    assert event["output"].startswith("A")
    assert event["output"].endswith("Z" * 100)
    assert len(event["error"]) <= MAX_ERROR_CHARS
    assert event["error_chars"] == len(error)
    assert event["error_truncated"] is True
    assert (
        event["output_excerpt_sha256"]
        == hashlib.sha256(event["output"].encode("utf-8")).hexdigest()
    )
    assert "TRANSCRIPT-SECRET" not in repr(event)
    assert "UNKNOWN-SECRET" not in repr(event)


def test_lineage_hash_input_work_is_capped_and_loss_is_disclosed(monkeypatch):
    agent = _bare_agent()
    huge_session_id = (
        "A" * MAX_LINEAGE_HASH_CHARS_PER_FIELD
        + "OMITTED-MIDDLE" * 10_000
        + "Z" * MAX_LINEAGE_HASH_CHARS_PER_FIELD
    )
    setattr(agent, "session_id", huge_session_id)
    encoded_inputs = []
    original_dumps = json.dumps

    def _record_dumps(value, *args, **kwargs):
        encoded_inputs.append(value)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr("agent.result_observer.json.dumps", _record_dumps)
    event = build_agent_result_event(agent, result={"final_response": "ok"})

    assert len(encoded_inputs) == 1
    hash_identity = encoded_inputs[0]
    assert all(
        len(item["excerpt"]) <= MAX_LINEAGE_HASH_CHARS_PER_FIELD
        for item in hash_identity.values()
    )
    assert hash_identity["session_id"]["chars"] == len(huge_session_id)
    assert "OMITTED-MIDDLE" not in hash_identity["session_id"]["excerpt"]
    assert event["identity_complete"] is False
    assert "session_id" in event["identity_truncated_fields"].split(",")
    assert event["lineage_hash_input_complete"] is False
    assert event["lineage_hash_input_truncated_fields"] == "session_id"
    assert len(event["lineage_sha256"]) == 64
    assert "OMITTED-MIDDLE" not in repr(event)


def test_lineage_digest_binds_complete_identity_beyond_public_preview():
    agent = _bare_agent()
    shared_preview = "s" * MAX_IDENTIFIER_CHARS
    setattr(agent, "session_id", shared_preview + "-first")
    first = build_agent_result_event(agent, result={"final_response": "ok"})
    setattr(agent, "session_id", shared_preview + "-second")
    second = build_agent_result_event(agent, result={"final_response": "ok"})

    assert first["session_id"] == second["session_id"] == shared_preview
    assert first["identity_complete"] is False
    assert second["identity_complete"] is False
    assert first["lineage_hash_input_complete"] is True
    assert second["lineage_hash_input_complete"] is True
    assert first["lineage_sha256"] != second["lineage_sha256"]


def test_native_subagent_result_keeps_parent_and_child_lineage(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent(subagent=True)
    result = {
        "final_response": "bounded child failure",
        "completed": False,
        "partial": True,
        "error": "child provider stopped",
    }

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result, turn_id="child-turn"),
    ):
        returned = agent.run_conversation("child request", task_id="child-task")

    event = captured.get(timeout=2)
    assert returned is result
    assert event["actor"] == "subagent"
    assert event["role"] == "leaf"
    assert event["session_id"] == "child-session"
    assert event["task_id"] == "child-task"
    assert event["turn_id"] == "child-turn"
    assert event["subagent_id"] == "sa-7"
    assert event["parent_session_id"] == "root-session"
    assert event["parent_turn_id"] == "parent-turn"
    assert event["parent_subagent_id"] == "sa-parent"
    assert event["partial"] is True
    assert event["error"] == "child provider stopped"


@pytest.mark.parametrize(
    ("flag", "error"),
    [
        ("interrupted", "operator interrupted"),
        ("failed", "provider failed"),
    ],
)
def test_returned_terminal_failure_flags_are_observed(monkeypatch, flag, error):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()
    result = {
        "final_response": "bounded terminal result",
        flag: True,
        "error": error,
    }

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result),
    ):
        returned = agent.run_conversation("request", task_id="terminal-task")

    event = captured.get(timeout=2)
    assert returned is result
    assert event["result_kind"] == "returned"
    assert event[flag] is True
    assert event["error"] == error


def test_raised_error_is_observed_then_reraised_unchanged(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()
    expected = RuntimeError("provider exploded")

    def _raise(
        runtime_agent,
        user_message,
        system_message,
        conversation_history,
        task_id,
        stream_callback,
        persist_user_message,
        persist_user_timestamp=None,
        moa_config=None,
    ):
        runtime_agent._current_task_id = task_id
        runtime_agent._current_turn_id = "error-turn"
        raise expected

    with patch("agent.conversation_loop.run_conversation", side_effect=_raise):
        with pytest.raises(RuntimeError) as caught:
            agent.run_conversation("request", task_id="error-task")

    event = captured.get(timeout=2)
    assert caught.value is expected
    assert event["result_kind"] == "raised"
    assert event["task_id"] == "error-task"
    assert event["turn_id"] == "error-turn"
    assert event["exception_type"] == "RuntimeError"
    assert event["error"] == "provider exploded"
    assert event["output"] == ""


def test_malformed_result_mapping_cannot_replace_returned_result(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()

    class ExplodingMapping(dict):
        def get(self, key, default=None):
            raise RuntimeError("malformed mapping")

    result = ExplodingMapping(final_response="must still return")
    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result),
    ):
        returned = agent.run_conversation("request", task_id="malformed-task")

    assert returned is result
    with pytest.raises(queue.Empty):
        captured.get(timeout=0.1)


def test_prologue_error_is_not_attributed_to_previous_turn(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()

    def _raise_before_turn(*args, **kwargs):
        raise RuntimeError("prologue failed")

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_raise_before_turn,
    ):
        with pytest.raises(RuntimeError, match="prologue failed"):
            agent.run_conversation("request", task_id="requested-task")

    event = captured.get(timeout=2)
    assert event["task_id"] == "requested-task"
    assert event["turn_id"] == ""
    assert event["turn_started"] is False


def test_no_listener_returns_without_creating_plugin_state(monkeypatch):
    monkeypatch.setattr(plugins_module, "_plugin_manager", None)
    agent = _bare_agent()
    result = {"final_response": object(), "completed": True}

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result),
    ):
        returned = agent.run_conversation("request", task_id="root-task")

    assert returned is result
    assert plugins_module._plugin_manager is None


def test_explicitly_exempt_internal_fork_is_not_misattributed_as_root(monkeypatch):
    captured: queue.Queue = queue.Queue()
    _register_observer(monkeypatch, lambda *, event: captured.put(event))
    agent = _bare_agent()
    setattr(agent, "_result_observer_exempt", True)
    result = {"final_response": "internal review output", "completed": True}

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=_returning_stub(result),
    ):
        returned = agent.run_conversation("internal review", task_id="aux-task")

    assert returned is result
    with pytest.raises(queue.Empty):
        captured.get(timeout=0.1)


def test_observer_registration_failure_does_not_leave_false_active_hook(
    monkeypatch,
):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="failed-observer"), manager)

    def _fail_start(_thread):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(threading.Thread, "start", _fail_start)
    context.register_hook(POST_AGENT_RESULT_HOOK, lambda *, event: None)

    assert manager._hooks.get(POST_AGENT_RESULT_HOOK, []) == []
    assert manager.has_active_observer_hook(POST_AGENT_RESULT_HOOK) is False
    assert manager.observer_health()["registration_failure"] is True


def test_slow_observer_cannot_delay_or_mutate_result(monkeypatch):
    observer_started = threading.Event()
    observer_release = threading.Event()
    observer_finished = threading.Event()
    call_finished = threading.Event()
    result = {"final_response": "answer", "completed": True}
    returned = []

    def _slow_observer(*, event):
        observer_started.set()
        observer_release.wait(timeout=5)
        event["output"] = "attempted rewrite"
        observer_finished.set()
        return {"final_response": "replacement"}

    _register_observer(monkeypatch, _slow_observer)
    agent = _bare_agent()

    def _call_agent():
        with patch(
            "agent.conversation_loop.run_conversation",
            side_effect=_returning_stub(result),
        ):
            returned.append(agent.run_conversation("request", task_id="root-task"))
        call_finished.set()

    runner = threading.Thread(target=_call_agent, daemon=True)
    runner.start()
    try:
        assert call_finished.wait(timeout=2)
        assert observer_started.wait(timeout=2)
        assert observer_finished.is_set() is False
        assert returned == [result]
        assert returned[0] is result
        assert result["final_response"] == "answer"
    finally:
        observer_release.set()
        runner.join(timeout=2)
    assert observer_finished.wait(timeout=2)
    assert result["final_response"] == "answer"


def test_observer_queue_drops_overload_instead_of_blocking(monkeypatch):
    monkeypatch.setattr(plugins_module, "_OBSERVER_QUEUE_MAX", 1)
    manager = PluginManager()
    observer_started = threading.Event()
    observer_release = threading.Event()

    def _blocked_observer(*, event):
        observer_started.set()
        observer_release.wait(timeout=5)

    context = PluginContext(PluginManifest(name="queue-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, _blocked_observer)
    try:
        assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
        assert observer_started.wait(timeout=2)
        assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 2})
        assert (
            manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 3}) is False
        )
        worker = manager._observer_runtime.workers[POST_AGENT_RESULT_HOOK][0]
        assert worker._queue.qsize() == 1
        health = manager.observer_health()
        assert health["degraded"] is True
        assert health["listeners"][POST_AGENT_RESULT_HOOK][0]["drop_observed"] is True
    finally:
        observer_release.set()


def test_hung_observer_cannot_head_of_line_block_another_listener(monkeypatch):
    monkeypatch.setattr(plugins_module, "_OBSERVER_QUEUE_MAX", 1)
    manager = PluginManager()
    slow_started = threading.Event()
    slow_release = threading.Event()
    fast_events: queue.Queue = queue.Queue()

    def _hung(*, event):
        slow_started.set()
        slow_release.wait(timeout=5)

    context = PluginContext(PluginManifest(name="isolation-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, _hung)
    context.register_hook(
        POST_AGENT_RESULT_HOOK, lambda *, event: fast_events.put(event["sequence"])
    )
    try:
        assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
        assert slow_started.wait(timeout=2)
        assert fast_events.get(timeout=2) == 1
        assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 2})
        assert fast_events.get(timeout=2) == 2
        # The slow listener's queue is now full. Its third copy drops, while
        # the independent listener continues to receive the same event.
        assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 3})
        assert fast_events.get(timeout=2) == 3
        started = time.monotonic()
        assert manager.drain_observer_hooks(timeout=0.01) is False
        assert time.monotonic() - started < 0.2
        health = manager.observer_health()
        assert health["degraded"] is True
        assert health["drain_timeout_observed"] is True
    finally:
        slow_release.set()
    assert manager.drain_observer_hooks(timeout=2)


def test_callback_failure_is_isolated_and_visible_in_health():
    manager = PluginManager()
    failed = threading.Event()
    surviving_events: queue.Queue = queue.Queue()

    def _broken(*, event):
        failed.set()
        raise RuntimeError("observer failed")

    context = PluginContext(PluginManifest(name="failure-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, _broken)
    context.register_hook(
        POST_AGENT_RESULT_HOOK,
        lambda *, event: surviving_events.put(event["sequence"]),
    )

    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
    assert failed.wait(timeout=2)
    assert surviving_events.get(timeout=2) == 1
    assert manager.drain_observer_hooks(timeout=2)
    health = manager.observer_health()
    assert health["degraded"] is True
    assert health["callback_failure_observed"] is True
    listeners = health["listeners"][POST_AGENT_RESULT_HOOK]
    assert listeners[0]["failure_observed"] is True
    assert listeners[1]["failure_observed"] is False


def test_retirement_race_purges_late_enqueue_and_records_drop(monkeypatch):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="retire-race-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, lambda *, event: None)
    worker = manager._observer_runtime.workers[POST_AGENT_RESULT_HOOK][0]
    original_put = worker._queue.put_nowait
    before_put = threading.Event()
    allow_put = threading.Event()
    emission_result = []

    def _delayed_put(item):
        before_put.set()
        assert allow_put.wait(timeout=2)
        original_put(item)

    monkeypatch.setattr(worker._queue, "put_nowait", _delayed_put)
    emitter = threading.Thread(
        target=lambda: emission_result.append(
            manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
        ),
        daemon=True,
    )
    emitter.start()
    assert before_put.wait(timeout=2)

    assert manager.shutdown_observer_hooks(timeout=0.2, drain=False)
    allow_put.set()
    emitter.join(timeout=2)

    assert emitter.is_alive() is False
    assert emission_result == [False]
    assert worker._queue.qsize() == 0
    assert worker._queue.unfinished_tasks == 0
    health = manager.observer_health()
    assert health["degraded"] is True
    assert health["drop_observed"] is True


def test_force_reload_retires_generation_and_purges_queued_callbacks(monkeypatch):
    manager = PluginManager()
    old_started = threading.Event()
    old_release = threading.Event()
    old_second = threading.Event()
    old_seen = []

    def _old(*, event):
        old_seen.append(event["sequence"])
        if event["sequence"] == 1:
            old_started.set()
            old_release.wait(timeout=5)
        else:
            old_second.set()

    context = PluginContext(PluginManifest(name="old-generation"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, _old)
    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
    assert old_started.wait(timeout=2)
    health = manager.observer_health()
    assert health["listeners"][POST_AGENT_RESULT_HOOK][0]["callbacks_in_flight"] == 1
    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 2})

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    started = time.monotonic()
    manager.discover_and_load(force=True)
    assert time.monotonic() - started < 0.5
    assert manager.has_active_observer_hook(POST_AGENT_RESULT_HOOK) is False
    assert manager.observer_health()["retired_generation_degraded"] is True

    # A callback already running cannot be killed, but its queued generation is
    # purged and cannot survive a disable/reload.
    old_release.set()
    assert old_second.wait(timeout=0.2) is False
    assert old_seen == [1]

    new_events: queue.Queue = queue.Queue()
    PluginContext(PluginManifest(name="new-generation"), manager).register_hook(
        POST_AGENT_RESULT_HOOK, lambda *, event: new_events.put(event["sequence"])
    )
    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 3})
    assert new_events.get(timeout=2) == 3
    assert old_seen == [1]


def test_safe_mode_force_reload_preserves_non_observer_plugin_state(monkeypatch):
    manager = PluginManager()
    callback = lambda **kwargs: None
    manager._hooks["pre_llm_call"] = [callback]
    sentinel_plugin = LoadedPlugin(PluginManifest(name="existing"))
    manager._plugins["existing"] = sentinel_plugin

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    manager.discover_and_load(force=True)

    assert manager._hooks["pre_llm_call"] == [callback]
    assert manager._plugins["existing"] is sentinel_plugin


def test_force_reload_rejects_dequeued_but_unclaimed_old_callback(
    monkeypatch,
):
    manager = PluginManager()
    callback_called = threading.Event()
    context = PluginContext(PluginManifest(name="claim-race-test"), manager)
    context.register_hook(
        POST_AGENT_RESULT_HOOK, lambda *, event: callback_called.set()
    )
    worker = manager._observer_runtime.workers[POST_AGENT_RESULT_HOOK][0]
    original_claim = worker._claim_callback
    claim_reached = threading.Event()
    allow_claim = threading.Event()

    def _delayed_claim(generation):
        claim_reached.set()
        assert allow_claim.wait(timeout=2)
        return original_claim(generation)

    monkeypatch.setattr(worker, "_claim_callback", _delayed_claim)
    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1})
    assert claim_reached.wait(timeout=2)

    # The worker has dequeued the event but has not claimed callback execution.
    # Force-disable must retire that generation before returning, even though
    # the daemon itself cannot exit until the test releases the artificial gap.
    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    started = time.monotonic()
    manager.discover_and_load(force=True)
    assert time.monotonic() - started < 0.5
    allow_claim.set()

    assert worker.join(timeout=2)
    assert callback_called.wait(timeout=0.2) is False
    assert manager.observer_health()["retired_generation_degraded"] is True


def test_generic_hook_invocation_preserves_async_observer_contract():
    manager = PluginManager()
    observer_started = threading.Event()
    observer_release = threading.Event()

    def _blocked_observer(*, event):
        observer_started.set()
        observer_release.wait(timeout=5)
        return "must be ignored"

    context = PluginContext(PluginManifest(name="invoke-test"), manager)
    context.register_hook(POST_AGENT_RESULT_HOOK, _blocked_observer)
    try:
        assert manager.invoke_hook(POST_AGENT_RESULT_HOOK, event={"sequence": 1}) == []
        assert observer_started.wait(timeout=2)
    finally:
        observer_release.set()


def _write_observer_plugin(
    home: Path, name: str, *, raise_after_register: bool = False
) -> Path:
    """Create a real user plugin whose register() subscribes to the observer
    hook, optionally raising afterward to simulate a failed load."""
    plugin_dir = home / "plugins" / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        f"name: {name}\nversion: 0.1.0\n", encoding="utf-8"
    )
    failure_line = (
        "    raise RuntimeError('boom during load')\n"
        if raise_after_register
        else ""
    )
    (plugin_dir / "__init__.py").write_text(
        "import threading\n"
        "EVENTS = []\n"
        "DONE = threading.Event()\n"
        "def _capture(*, event):\n"
        "    EVENTS.append(event)\n"
        "    DONE.set()\n"
        "def register(ctx):\n"
        "    ctx.register_hook('post_agent_result', _capture)\n"
        + failure_line,
        encoding="utf-8",
    )
    return plugin_dir


def _enable_plugins(home: Path, names: list[str]) -> None:
    lines = "".join(f"    - {name}\n" for name in names)
    (home / "config.yaml").write_text(
        f"plugins:\n  enabled:\n{lines}", encoding="utf-8"
    )


def _active_observer_listeners(manager: PluginManager) -> int:
    return sum(
        1
        for items in manager.observer_health()["listeners"].values()
        for item in items
        if item["active"]
    )


def test_failed_plugin_load_rolls_back_observer_registration(
    tmp_path: Path, monkeypatch
):
    """A plugin that raises during register() must not keep observing:
    its callback is deregistered, its worker retires, and the result path
    stays a no-op — while bounded drain/shutdown lifecycle keeps working."""
    home = tmp_path / "hermes"
    _write_observer_plugin(home, "rollback-bad-observer", raise_after_register=True)
    _enable_plugins(home, ["rollback-bad-observer"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    threads_before = set(threading.enumerate())
    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    loaded = manager._plugins["rollback-bad-observer"]
    assert loaded.enabled is False
    assert "boom during load" in (loaded.error or "")

    # The disabled plugin is fully detached: no registered callback, no
    # active worker, no acceptance, no delivery, no lingering daemon.
    assert POST_AGENT_RESULT_HOOK not in manager._hooks
    assert plugins_module.has_observer_hook(POST_AGENT_RESULT_HOOK) is False
    from agent.result_observer import result_observer_enabled

    assert result_observer_enabled(_bare_agent()) is False
    assert (
        manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"probe": True}) is False
    )
    assert loaded.module.DONE.wait(timeout=0.3) is False
    assert loaded.module.EVENTS == []
    assert _active_observer_listeners(manager) == 0
    health = manager.observer_health()
    assert health["degraded"] is False
    assert not [
        thread
        for thread in threading.enumerate()
        if thread not in threads_before
        and thread.name.startswith("hermes-observer-")
        and thread.is_alive()
    ]

    # Rollback must not corrupt the shared lifecycle helpers.
    assert manager.drain_observer_hooks(timeout=0.2) is True
    assert manager.shutdown_observer_hooks(timeout=0.5, drain=True) is True


def test_failed_plugin_rollback_is_isolated_from_loaded_plugin(
    tmp_path: Path, monkeypatch
):
    """Rolling back the failed plugin must not disturb a successfully loaded
    listener — in the same sweep and across a force reload."""
    home = tmp_path / "hermes"
    _write_observer_plugin(home, "rollback-good-observer")
    _write_observer_plugin(home, "rollback-bad-observer", raise_after_register=True)
    _enable_plugins(home, ["rollback-good-observer", "rollback-bad-observer"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(plugins_module, "_plugin_manager", manager)

    good = manager._plugins["rollback-good-observer"]
    bad = manager._plugins["rollback-bad-observer"]
    assert good.enabled is True
    assert bad.enabled is False

    assert plugins_module.has_observer_hook(POST_AGENT_RESULT_HOOK) is True
    assert len(manager._hooks[POST_AGENT_RESULT_HOOK]) == 1
    assert _active_observer_listeners(manager) == 1

    assert (
        manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 1}) is True
    )
    assert good.module.DONE.wait(timeout=2)
    assert len(good.module.EVENTS) == 1
    assert bad.module.EVENTS == []

    # Force reload re-runs the same sweep: the good listener re-registers
    # under the new generation, the bad plugin fails and rolls back again.
    good.module.DONE.clear()
    manager.discover_and_load(force=True)
    good = manager._plugins["rollback-good-observer"]
    assert good.enabled is True
    assert manager._plugins["rollback-bad-observer"].enabled is False
    assert _active_observer_listeners(manager) == 1
    assert (
        manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"sequence": 2}) is True
    )
    assert good.module.DONE.wait(timeout=2)
    assert manager.shutdown_observer_hooks(timeout=0.5, drain=True) is True


def test_failed_plugin_rollback_respects_safe_mode_reload(
    tmp_path: Path, monkeypatch
):
    """After a failed load rolled back, a safe-mode force reload must stay a
    clean no-listener state with the bounded lifecycle intact."""
    home = tmp_path / "hermes"
    _write_observer_plugin(home, "rollback-bad-observer", raise_after_register=True)
    _enable_plugins(home, ["rollback-bad-observer"])
    monkeypatch.setenv("HERMES_HOME", str(home))

    manager = PluginManager()
    manager.discover_and_load()
    assert _active_observer_listeners(manager) == 0

    monkeypatch.setenv("HERMES_SAFE_MODE", "1")
    manager.discover_and_load(force=True)
    assert _active_observer_listeners(manager) == 0
    assert manager.emit_observer_hook(POST_AGENT_RESULT_HOOK, {"probe": True}) is False
    assert manager.shutdown_observer_hooks(timeout=0.5, drain=True) is True
