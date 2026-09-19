"""Behavior contract for metadata-only plugin agent-turn context."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _plugin_context() -> PluginContext:
    return PluginContext(PluginManifest(name="context-probe"), PluginManager())


def test_agent_context_is_none_unbound_and_snapshot_is_frozen():
    from agent.plugin_agent_context import AgentContext, bind_agent_context

    ctx = _plugin_context()
    assert ctx.agent_context() is None

    snapshot = AgentContext(
        session_id="session-1",
        task_id="task-1",
        turn_id="turn-1",
        platform="cli",
        source="tool",
        parent_session_id=None,
    )
    with bind_agent_context(snapshot):
        assert ctx.agent_context() is snapshot
        with pytest.raises(FrozenInstanceError):
            snapshot.turn_id = "changed"

    assert ctx.agent_context() is None


def test_discovered_plugin_hook_and_tool_share_turn_identity(monkeypatch):
    import hermes_cli.plugins as plugins_mod
    import model_tools
    from gateway.session_context import clear_session_vars, set_session_vars
    from run_agent import AIAgent

    hermes_home = Path(__import__("os").environ["HERMES_HOME"])
    plugin_dir = hermes_home / "plugins" / "agent-context-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text("name: agent-context-probe\n", encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "import dataclasses\n"
        "import json\n"
        "HOOK_CONTEXT = None\n"
        "def register(ctx):\n"
        "    def observe(**kwargs):\n"
        "        global HOOK_CONTEXT\n"
        "        value = ctx.agent_context()\n"
        "        HOOK_CONTEXT = dataclasses.asdict(value) if value else None\n"
        "    ctx.register_hook('post_tool_call', observe)\n"
        "    ctx.register_tool(\n"
        "        name='agent_context_probe', toolset='context-probe',\n"
        "        schema={'name': 'agent_context_probe', 'description': 'probe', "
        "'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kwargs: json.dumps(\n"
        "            dataclasses.asdict(ctx.agent_context()) if ctx.agent_context() else None),\n"
        "    )\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - agent-context-probe\n",
        encoding="utf-8",
    )
    manager = plugins_mod.PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    manager.discover_and_load()
    loaded = manager._plugins["agent-context-probe"]

    # Import before replacing the coordinator: module registration targets the real coordinator.
    import hermes_cli.observability.relay_shared_metrics as metrics

    monkeypatch.setattr(metrics, "start_task_run", lambda **kwargs: None)
    monkeypatch.setattr(metrics, "finish_task_run", lambda **kwargs: None)

    class _Coordinator:
        def acquire_conversation(self, **kwargs):
            return SimpleNamespace()

        def begin_turn(self, lease, **kwargs):
            return SimpleNamespace(relay_enabled=False)

        def finish_logical_calls(self, *args, **kwargs):
            pass

        def end_turn(self, *args, **kwargs):
            pass

        def release_conversation(self, *args, **kwargs):
            pass

    monkeypatch.setattr("agent.relay_runtime.SESSION_COORDINATOR", _Coordinator())
    monkeypatch.setattr("agent.background_review.cancel_background_review_for_live_turn", lambda agent: None)
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda config: (None, 1.0)
    )

    class _TimerHandle:
        def cancel(self, *, wait):
            pass

    monkeypatch.setattr("agent.periodic_scheduler.schedule", lambda callback, interval: _TimerHandle())

    class _RotatingDb:
        def get_session(self, session_id):
            return {"id": session_id}

        def acquire_session_turn_lease(self, session_id, holder, *, on_wait, **kwargs):
            on_wait(0.1)
            return True

        def resolve_resume_session_id(self, session_id):
            return "rotated-session"

        def get_messages_as_conversation(self, session_id, **kwargs):
            assert session_id == "rotated-session"
            return []

        def release_session_turn_lease(self, session_id, holder):
            pass

    observed = {}

    def agent_stub(*, session_id, platform, parent_session_id=None, session_db=None):
        return SimpleNamespace(
            session_id=session_id,
            platform=platform,
            model="test-model",
            _parent_session_id=parent_session_id,
            _session_db=session_db,
            _persist_disabled=False,
            _session_turn_lease_refresh_interval=60.0,
            _interrupt_requested=False,
            _emit_status=lambda message: None,
            _touch_activity=lambda message: None,
            _liveness_activity_lock=threading.Lock,
            _conversation_root_id=lambda: parent_session_id or session_id,
            _reset_activity_labels_after_turn=lambda: None,
        )

    def fake_loop(agent, user_message, *args, **kwargs):
        if agent.platform == "subagent":
            observed["child"] = _plugin_context().agent_context()
            return {"final_response": "child", "completed": True, "interrupted": False}
        observed["parent_before_child"] = _plugin_context().agent_context()
        child = agent_stub(
            session_id="child-session",
            platform="subagent",
            parent_session_id="session-1",
        )
        AIAgent.run_conversation(child, "child work", task_id="child-task")
        observed["parent_after_child"] = _plugin_context().agent_context()
        plugins_mod.invoke_hook("post_tool_call", tool_name="before-dispatch")
        observed["tool"] = json.loads(model_tools.handle_function_call(
            "agent_context_probe",
            {},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
            skip_pre_tool_call_hook=True,
        ))
        return {"final_response": "done", "completed": True, "interrupted": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fake_loop)
    agent = agent_stub(session_id="session-1", platform="desktop", session_db=_RotatingDb())
    tokens = set_session_vars(platform="desktop", source="tool", session_id="session-1")
    try:
        result = AIAgent.run_conversation(agent, "hello", task_id="task-1")
    finally:
        clear_session_vars(tokens)

    assert result["final_response"] == "done"
    assert observed["tool"] is not None
    turn_id = observed["tool"]["turn_id"]
    assert turn_id.startswith("session-1:task-1:")
    expected = {
        "session_id": "rotated-session",
        "task_id": "task-1",
        "turn_id": turn_id,
        "platform": "desktop",
        "source": "tool",
        "parent_session_id": None,
    }
    assert observed["tool"] == expected
    assert loaded.module.HOOK_CONTEXT == expected
    assert observed["parent_before_child"] == observed["parent_after_child"]
    assert observed["parent_before_child"].turn_id == turn_id
    assert observed["child"].session_id == "child-session"
    assert observed["child"].task_id == "child-task"
    assert observed["child"].platform == "subagent"
    assert observed["child"].source == "subagent"
    assert observed["child"].parent_session_id == "session-1"
    assert observed["child"].turn_id != turn_id
    assert _plugin_context().agent_context() is None

    def fail_loop(*args, **kwargs):
        assert _plugin_context().agent_context() is not None
        raise RuntimeError("turn failed")

    monkeypatch.setattr("agent.conversation_loop.run_conversation", fail_loop)
    tokens = set_session_vars(platform="desktop", source="tool", session_id="failed-session")
    try:
        with pytest.raises(RuntimeError, match="turn failed"):
            AIAgent.run_conversation(
                agent_stub(session_id="failed-session", platform="desktop"),
                "fail",
                task_id="failed-task",
            )
    finally:
        clear_session_vars(tokens)
    assert _plugin_context().agent_context() is None


def test_nested_concurrent_and_copied_worker_contexts_are_isolated():
    from agent.plugin_agent_context import AgentContext, bind_agent_context, get_agent_context
    from tools.thread_context import propagate_context_to_thread

    def snapshot(label: str) -> AgentContext:
        return AgentContext(label, f"task-{label}", f"turn-{label}", "cli", "cli", None)

    outer = snapshot("outer")
    inner = snapshot("inner")
    copied = queue.Queue()
    with bind_agent_context(outer):
        worker = threading.Thread(
            target=propagate_context_to_thread(lambda: copied.put(get_agent_context()))
        )
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert copied.get_nowait() is outer
        with bind_agent_context(inner):
            assert get_agent_context() is inner
        assert get_agent_context() is outer

    barrier = threading.Barrier(2)
    concurrent = queue.Queue()

    def observe(value: AgentContext) -> None:
        with bind_agent_context(value):
            barrier.wait(timeout=5)
            concurrent.put(get_agent_context())

    left, right = snapshot("left"), snapshot("right")
    threads = [threading.Thread(target=observe, args=(value,)) for value in (left, right)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert {concurrent.get_nowait(), concurrent.get_nowait()} == {left, right}
    assert get_agent_context() is None


def test_binding_cleanup_on_exception():
    from agent.plugin_agent_context import AgentContext, bind_agent_context, get_agent_context

    snapshot = AgentContext("s", "task", "turn", "cli", "cli", None)
    with pytest.raises(RuntimeError, match="boom"):
        with bind_agent_context(snapshot):
            raise RuntimeError("boom")
    assert get_agent_context() is None
