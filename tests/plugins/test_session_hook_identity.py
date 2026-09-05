"""Native hooks bind the dispatch owner before lazy builds or tool execution."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest


def install_probe(home):
    plugin = home / "plugins" / "identity-probe"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: identity-probe\nversion: 1.0.0\n")
    (home / "config.yaml").write_text("plugins:\n  enabled: [identity-probe]\n")
    (plugin / "__init__.py").write_text('''
import json
from hermes_constants import get_hermes_home

def register(ctx):
    home = get_hermes_home()
    def receive(event, **kwargs):
        with (home / "receipt.jsonl").open("a") as out:
            out.write(json.dumps(dict(event=event, active_home=str(get_hermes_home()), **kwargs), default=str) + "\\n")
    for event in ("on_session_identity", "on_session_start", "on_session_end", "on_session_finalize", "pre_tool_call"):
        ctx.register_hook(event, lambda _event=event, **kw: receive(_event, **kw))
    # Narrow signatures must keep working as identity fields are added.
    ctx.register_hook("on_session_identity", lambda session_id: receive("legacy", session_id=session_id))
''')


def receipts(home, event):
    path = home / "receipt.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [row for row in rows if row["event"] == event]


def test_identity_is_published_for_two_new_lazy_sessions_before_rpc_returns(tmp_path, monkeypatch):
    from tui_gateway import server
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ambient = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    homes = [ambient / "profiles" / name for name in ("a", "b")]
    for home in homes:
        install_probe(home)

    def create(home):
        response = server._methods["session.create"](home.name, {"source": "desktop", "profile": home.name})
        assert "error" not in response, response
        result = response["result"]
        observed = receipts(home, "on_session_identity")
        assert observed, "ownership must be published before the client receives its runtime ID"
        payload = observed[-1]
        assert payload["runtime_session_id"] == result["session_id"]
        assert payload["stored_session_id"] == result["stored_session_id"]
        assert payload["session_id"] == result["stored_session_id"]
        assert payload["task_id"] is None
        assert payload["profile"] == home.name
        assert payload["hermes_home"] == payload["active_home"] == str(home)
        assert payload["source"] == payload["surface"] == "desktop"
        assert receipts(home, "legacy")[-1]["session_id"] == payload["session_id"]
        assert server._sessions[result["session_id"]]["agent"] is None
        return payload

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(create, homes))
    assert left["runtime_session_id"] != right["runtime_session_id"]
    assert left["stored_session_id"] != right["stored_session_id"]
    assert get_hermes_home() == ambient
    for home, owner in zip(homes, (left, right)):
        record = server._sessions[owner["runtime_session_id"]]
        server._finalize_session(record)
        final = receipts(home, "on_session_finalize")[-1]
        for field in ("runtime_session_id", "stored_session_id", "session_id",
                      "task_id", "profile", "hermes_home", "source", "surface"):
            assert final[field] == owner[field]
        assert not receipts(home, "on_session_end"), "an unused draft has no turn to end"


@pytest.mark.parametrize("resume_mode", [{"lazy": True}, {}, {"eager_build": True}])
def test_resumed_sessions_keep_owner_through_tools_compression_and_finalization(tmp_path, monkeypatch, resume_mode):
    from tui_gateway import server
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent.tool_executor import _pre_tool_block, _ToolCallRef
    from agent.conversation_loop import _restore_or_build_system_prompt
    from agent.turn_context import _bind_turn_identity
    from agent.turn_finalizer import finalize_turn

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ambient = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_start_session_services", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_wire_session_agent", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)

    def make_agent(sid, key, session_db=None, **kwargs):
        # Only external model/tool discovery is stubbed; construction, persistence,
        # hook discovery, callback dispatch, and session registration are real.
        with patch("model_tools.get_tool_definitions", return_value=[]), patch("model_tools.check_toolset_requirements", return_value={}):
            return AIAgent(model="test-model", api_key="test-key", base_url="http://127.0.0.1:1/v1",
                           provider="custom", enabled_toolsets=[], quiet_mode=True,
                           skip_context_files=True, skip_memory=True, skip_background_review=True,
                           session_db=session_db, session_id=key, platform="desktop")

    monkeypatch.setattr(server, "_make_agent_in_context", make_agent)
    owners = []
    for name in ("a", "b"):
        home = ambient / "profiles" / name
        install_probe(home)
        key = "stored-" + name
        with SessionDB(db_path=home / "state.db") as db:
            db.create_session(key, source="desktop")
        response = server._methods["session.resume"](name, dict(session_id=key, profile=name, source="desktop", **resume_mode))
        assert "error" not in response, response
        sid = response["result"]["session_id"]
        record = server._sessions[sid]
        identity = receipts(home, "on_session_identity")
        assert identity, "resumed runtime must announce identity before returning"
        assert identity[-1]["runtime_session_id"] == sid
        assert identity[-1]["stored_session_id"] == key
        if record.get("agent") is None:
            db = SessionDB(db_path=home / "state.db")
            agent = make_agent(sid, key, db)
            server._attach_built_agent(record, agent)
        else:
            agent = record["agent"]
        owners.append((home, sid, key, record, agent))

    try:
        for home, sid, key, record, agent in owners:
            task, turn = _bind_turn_identity(agent, key, None, None, None, None)
            monkeypatch.setattr(agent, "_build_system_prompt", lambda _: "A test prompt")
            _restore_or_build_system_prompt(agent, None, [])
            block, args = _pre_tool_block(agent, _ToolCallRef("terminal", {"command": "true"}, task, "call-1", []))
            assert block is None
            payload = receipts(home, "pre_tool_call")[-1]
            for field, value in dict(runtime_session_id=sid, stored_session_id=key,
                                     session_id=key, task_id=task, profile=home.name,
                                     hermes_home=str(home), source="desktop", surface="desktop").items():
                assert payload[field] == value
                assert receipts(home, "on_session_start")[-1][field] == value
            assert payload["active_home"] == str(home)
            agent._invoke_tool("todo", {"todos": []}, task, tool_call_id="inline-call")
            inline = receipts(home, "pre_tool_call")[-1]
            assert inline["tool_call_id"] == "inline-call"
            assert inline["runtime_session_id"] == sid

            # A real stored compression edge and the production recovery/adoption path.
            from agent.conversation_compression import recover_rotated_compression_session
            child = key + "-compressed"
            agent._session_db.publish_compression_child(
                parent_session_id=key, child_session_id=child, source="desktop", model=agent.model,
                model_config={}, system_prompt="compressed", require_compression_lease=False,
                messages=[{"role": "user", "content": "Compressed handoff"}])
            assert recover_rotated_compression_session(agent)
            assert agent.session_id == child
            _pre_tool_block(agent, _ToolCallRef("terminal", {}, task, "call-2", []))
            compressed = receipts(home, "pre_tool_call")[-1]
            assert compressed["session_id"] == child
            assert compressed["stored_session_id"] == key
            assert compressed["runtime_session_id"] == sid
            assert compressed["task_id"] == task
            server._sync_session_key_after_compress(sid, record, restart_slash_worker=False)
            assert receipts(home, "on_session_identity")[-1]["stored_session_id"] == child

            finalize_turn(agent, final_response="done", api_call_count=1, interrupted=False,
                          failed=False, messages=[], conversation_history=[], effective_task_id=task,
                          turn_id=turn, user_message="hello", original_user_message="hello",
                          _should_review_memory=False, _turn_exit_reason="text_response(test)")
            ended = receipts(home, "on_session_end")[-1]
            assert ended["session_id"] == child
            assert ended["runtime_session_id"] == sid
            assert ended["task_id"] == task
            assert not receipts(home, "on_session_finalize"), "turn end must not finalize the runtime"
            server._finalize_session(record, end_reason="tui_close")
            final = receipts(home, "on_session_finalize")[-1]
            assert final["runtime_session_id"] == sid
            assert final["stored_session_id"] == final["session_id"] == child
            assert final["profile"] == home.name
            assert final["hermes_home"] == final["active_home"] == str(home)
            count = len(receipts(home, "on_session_finalize"))
            server._finalize_session(record)
            assert len(receipts(home, "on_session_finalize")) == count
    finally:
        for _, _, _, _, agent in owners:
            agent.close()
            agent._session_db.close()


def test_library_identity_and_cli_finalization_never_borrow_ambient_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agent.agent_init import _init_session_state
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.session_hook_context import agent_session_identity, hook_profile_scope

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ambient = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(ambient))
    owners = []
    for name in ("a", "b"):
        home = ambient / "profiles" / name
        install_probe(home)
        agent = SimpleNamespace(platform="cli", max_iterations=1)
        with hook_profile_scope(home):
            _init_session_state(agent, None, None, None, None, None, False, 1, 1, 1)
        owners.append((home, agent))

    monkeypatch.setenv("HERMES_SESSION_ID", "not-an-owner")
    for home, agent in owners:
        identity = agent_session_identity(agent)
        assert identity["runtime_session_id"] is None
        assert identity["task_id"] is None
        assert identity["session_id"] == identity["stored_session_id"] == agent.session_id
        assert identity["hermes_home"] == home
        cli = SimpleNamespace(agent=agent, platform="cli")
        CLISessionMixin._notify_session_boundary(cli, "on_session_finalize")
        final = receipts(home, "on_session_finalize")
        assert final, "CLI finalization must dispatch in the agent's captured home"
        assert final[-1]["session_id"] == agent.session_id
        assert final[-1]["runtime_session_id"] is None
        from cli import _notify_single_query_session_finalize
        _notify_single_query_session_finalize(cli, reason="query_complete")
        assert len(receipts(home, "on_session_finalize")) == len(final) + 1
        from cli import _invoke_interrupted_session_end
        _invoke_interrupted_session_end(agent, agent.session_id, "keyboard_interrupt", task_id="interrupted-task")
        interrupted = receipts(home, "on_session_end")
        assert interrupted, "interrupt dispatch must keep the agent's identity"
        assert interrupted[-1]["task_id"] == "interrupted-task"
        assert interrupted[-1]["stored_session_id"] == agent.session_id
        from run_agent import AIAgent
        agent.session_id = "resumed-" + home.name
        agent._current_task_id = "previous-turn"
        agent._transition_context_engine_session = lambda **kw: None
        AIAgent.reset_session_state(agent)
        rebound = agent_session_identity(agent)
        assert rebound["stored_session_id"] == agent.session_id
        assert rebound["task_id"] is None
        assert rebound["hermes_home"] == home

    unknown = agent_session_identity(SimpleNamespace(session_id="external-conversation"))
    assert unknown["session_id"] == "external-conversation"
    assert all(value is None for key, value in unknown.items() if key != "session_id")
