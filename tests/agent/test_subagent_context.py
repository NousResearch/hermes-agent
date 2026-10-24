"""Forks share a conversation boundary, never live tool state or mutable history."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.subagent_context import bind_fork_source


def _history():
    return [
        {"role": "user", "content": "Earlier request", "_row_id": 4, "_db_persisted": True},
        {"role": "assistant", "content": "Earlier answer", "finish_reason": "stop", "reasoning": "private"},
        {"role": "assistant", "content": "Compaction handoff", "_compressed_summary": True},
        {"role": "user", "content": [{"type": "text", "text": "Current request"}]},
        {"role": "assistant", "content": "Working...", "finish_reason": "incomplete"},
        {"role": "assistant", "content": "Spawning A and B", "tool_calls": [
            {"id": "a", "function": {"name": "delegate_task", "arguments": "assignment A"}},
            {"id": "b", "function": {"name": "delegate_task", "arguments": "assignment B"}},
        ]},
        {"role": "tool", "tool_call_id": "a", "content": "sibling id"},
    ]


def test_parallel_snapshot_is_unformatted_detached_and_shared_boundary():
    from agent.subagent_context import capture_fork_boundary
    history = _history()
    original = deepcopy(history)
    parent = SimpleNamespace(_session_messages=[{"role": "user", "content": "stale"}])
    with bind_fork_source(parent, history):
        contexts = [copy_context() for _ in range(3)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            children = list(pool.map(lambda ctx: ctx.run(capture_fork_boundary, parent), contexts))
        assert children[0] == children[1] == children[2] == original
        children[0][3]["content"][0]["text"] = "Child A mutation"
        assert children[1] == original
        history.append({"role": "tool", "tool_call_id": "b", "content": "late sibling"})
        assert capture_fork_boundary(parent) == original
    assert history[:-1] == original
    assert contexts[0].run(capture_fork_boundary, parent) == parent._session_messages


@pytest.fixture
def fork_formatter(monkeypatch):
    from agent.subagent_extension import SubagentExtension
    class UserHistory(SubagentExtension):
        def format_fork_history(self, messages, *, task):
            return [message for message in messages if message.get("role") == "user"]
    monkeypatch.setattr("agent.subagent_extension.get_subagent_extension", lambda: UserHistory())


@pytest.mark.parametrize("surface", ["tool", "plugin"])
@pytest.mark.parametrize("inherit,has_history", [(True, True), (False, True), (True, False)])
def test_real_child_prologue_receives_and_persists_fork(tmp_path, monkeypatch, surface, inherit, has_history, fork_formatter):
    from agent.subagent_lifecycle import SubagentLaunchRequest, SubagentLifecycleService
    from agent.turn_context import build_turn_context
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from tools.delegate_tool_child_run import _ChildRun
    from tools.registry import registry

    # Real construction, delegation, worker-thread forwarding and turn prologue;
    # only the provider client and post-prologue model execution are replaced.
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "AGENTS.md").write_text("PROJECT_RULE: Preserve existing changes.")
    (tmp_path / "SOUL.md").write_text("SHARED_PERSONA: Be thoughtful and precise.")
    (tmp_path / "instructions").mkdir()
    (tmp_path / "instructions" / "main.md").write_text("MAIN_ONLY: Coordinate the overall task.")
    (tmp_path / "instructions" / "subagent.md").write_text("CHILD_ONLY: Report findings to your parent.")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    cfg = {"compression": {"enabled": False}, "delegation": {"max_concurrent_children": 3},
           "terminal": {"cwd": str(workspace)}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda *a, **kw: cfg)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda *a, **kw: cfg)
    observed = []
    expected_fork = inherit and has_history

    def run_conversation(child, user_message, conversation_history=None, task_id=None, **kwargs):
        from agent import conversation_loop as loop
        ctx = build_turn_context(
            child, user_message, None, conversation_history, task_id, None, None,
            restore_or_build_system_prompt=loop._restore_or_build_system_prompt,
            install_safe_stdio=loop._install_safe_stdio,
            sanitize_surrogates=loop._sanitize_surrogates,
            summarize_user_message_for_log=loop._summarize_user_message_for_log,
            set_session_context=loop.set_session_context,
            set_current_write_origin=loop.set_current_write_origin,
            ra=loop._ra,
        )
        observed.append((child.session_id, deepcopy(ctx.messages), child.context_compressor))
        prompt = ctx.active_system_prompt + "\n" + child.ephemeral_system_prompt
        assert "PROJECT_RULE" in prompt
        assert child._delegate_forked_context is expected_fork
        return {"final_response": "done", "completed": True, "messages": ctx.messages, "api_calls": 0}

    def run_child(task_index, goal, child, parent_agent, **kwargs):
        run = _ChildRun(child, parent_agent, task_index, goal, child._subagent_id, None)
        try:
            result, error, _ = run.await_child()
            assert error is None, error
            return {"status": "completed", "summary": result["final_response"], "api_calls": 0}
        finally:
            child.close()

    def run_batch(batch, background):
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(copy_context().run, run_child, i, task["goal"], child, batch.parent_agent)
                       for i, task, child in batch.children]
            for future in futures:
                future.result()
        return "{}"

    monkeypatch.setattr(AIAgent, "run_conversation", run_conversation)
    monkeypatch.setattr("tools.delegate_tool._run_batch", run_batch)
    monkeypatch.setattr("tools.delegate_tool._run_single_child", run_child)
    with patch("agent.process_bootstrap.OpenAI"):
        db = SessionDB(db_path=tmp_path / "state.db")
        parent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1", model="test/model",
            provider="custom", enabled_toolsets=[], quiet_mode=True,
            skip_context_files=True, skip_memory=True, session_db=db, session_id="fork-parent",
        )
        parent._ensure_db_session()
        history = _history() if has_history else []
        original = deepcopy(history)
        try:
            with bind_fork_source(parent, history):
                if surface == "tool":
                    tasks = [{"goal": f"Independent assignment {i}", "fork_context": inherit} for i in range(3)]
                    if not inherit:
                        for task in tasks:
                            task["fork_context"] = False
                    registry.get_entry("delegate_task").handler({"tasks": tasks}, parent_agent=parent)
                else:
                    service = SubagentLifecycleService(lambda: parent)
                    request = {"goal": "Independent plugin assignment", "fork_context": inherit}
                    if not inherit:
                        request["fork_context"] = False
                    handle = service.launch(SubagentLaunchRequest(**request))
                    terminal = service.wait(handle, timeout_seconds=30)
                    assert terminal.completed and terminal.state.value == "SUCCEEDED"
            assert len(observed) == (3 if surface == "tool" else 1)
            for sid, messages, compressor in observed:
                assert messages[-1]["content"].startswith("Independent")
                assert bool(messages[:-1]) is expected_fork
                assert compressor is not parent.context_compressor
                if expected_fork:
                    assert messages[-3]["content"][0]["text"] == "Current request"
                    assert messages[-2]["role"] == "assistant"
                    stored = db.get_messages_as_conversation(sid)
                    assert stored[0]["content"] == "Earlier request"
                    assert stored[-1]["content"].startswith("Independent")
                    assert all(not m.get("tool_calls") and m["role"] != "tool" for m in stored)
            assert len({id(item[2]) for item in observed}) == len(observed)
            assert history == original
            assert db.get_messages_as_conversation(parent.session_id) == []
        finally:
            parent.close()
            db.close()
