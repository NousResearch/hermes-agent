"""Opted-in context engines receive detached, profile-owned host state (#133644)."""

import copy
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.conversation_compression import _resolve_compress_call
from agent.message_metadata import without_persistence_fields
from agent.conversation_loop import _apply_context_engine_selection
from hermes_cli.goals import GoalContract, GoalState
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools.todo_tool import TodoStore


def _bind_call(agent):
    return _resolve_compress_call(agent, approx_tokens=0, focus_topic=None, force=False,
                                  memory_context="", bypass_cooldown=False)


def _mutate(state):
    state["todos"][0]["content"] = "engine changed todo"
    state["goal"]["text"] = "engine changed goal"
    state["goal"]["subgoals"].append("engine changed subgoal")
    state["goal"]["contract"]["verification"] = "engine changed contract"


@pytest.mark.parametrize("launch_name", ["default", "launch"])
def test_compaction_and_selection_receive_fresh_detached_profile_state(tmp_path, monkeypatch, launch_name):
    from run_agent import AIAgent

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default = tmp_path / ".hermes"
    homes = {"default": default, "launch": default / "profiles" / "launch",
             "a": default / "profiles" / "a", "b": default / "profiles" / "b"}
    for home in homes.values():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model:\n  default: test/model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(homes[launch_name]))
    agents, dbs, goals = {}, {}, {}
    try:
        for name in ("a", "b"):
            token = set_hermes_home_override(homes[name])
            try:
                db = SessionDB(db_path=homes[name] / "state.db")
                dbs[name] = db
                db.create_session("shared-session", source="test")
                goal = GoalState(goal="finish " + name, status="active" if name == "a" else "paused",
                                 subgoals=["verify " + name], contract=GoalContract(verification="run " + name))
                goals[name] = goal
                db.set_meta("goal:shared-session", goal.to_json())
                agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                                quiet_mode=True, session_db=db, session_id="shared-session",
                                skip_context_files=True, skip_memory=True)
                agent._compression_feasibility_checked = True
                agent.compression_in_place = True
                agent._todo_store.write([{"id": "1", "content": "todo " + name, "status": "pending"}])
                agents[name] = agent
            finally:
                reset_hermes_home_override(token)
        for name in ("a", "b", "a"):
            token = set_hermes_home_override(homes[name])
            try:
                agent = agents[name]
                seen = []

                def compress(messages, current_tokens=None, *, host_state=None):
                    assert host_state is not None
                    seen.append(copy.deepcopy(host_state))
                    _mutate(host_state)
                    agent.context_compressor.compression_count += 1
                    return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, messages[-1]]

                agent.context_compressor.compress = compress
                messages = [{"role": "user", "content": "turn " + str(i)} for i in range(20)]
                compressed, system = agent._compress_context(messages, "frozen system", approx_tokens=120_000)
                assert compressed[0]["content"] == "[CONTEXT COMPACTION] summary"
                expected = {"todos": [{"id": "1", "content": "todo " + name, "status": "pending"}],
                            "goal": {"text": goals[name].goal, "status": goals[name].status,
                                     "contract": goals[name].contract.to_dict(), "subgoals": goals[name].subgoals},
                            "plan_path": None}
                assert seen == [expected]

                selected = []

                def select(request_messages, *, conversation_messages=None, incoming_message=None,
                           budget_tokens=0, host_state=None):
                    selected.append(copy.deepcopy(host_state))
                    assert host_state == expected
                    _mutate(host_state)
                    return request_messages

                agent.context_compressor.select_context = select
                request = [{"role": "system", "content": system}] + [without_persistence_fields(m) for m in compressed]
                assert _apply_context_engine_selection(agent, request, compressed, compressed[-1],
                                                        logger=logging.getLogger(__name__)) is request
                assert selected == [expected]
                assert request[0]["content"] == system
                assert agent._todo_store.read() == expected["todos"]
                assert GoalState.from_json(dbs[name].get_meta("goal:shared-session")) == goals[name]
            finally:
                reset_hermes_home_override(token)
        assert not (homes[launch_name] / "state.db").exists()
    finally:
        for db in dbs.values():
            db.close()


@pytest.mark.parametrize("kind", ["legacy", "forwarder", "builtin", "explicit-kwargs", "db-error", "done"])
def test_host_state_opt_in_preserves_legacy_dispatch_and_fails_open(tmp_path, monkeypatch, kind):
    from agent.context_compressor import ContextCompressor

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = TodoStore()
    store.write([{"id": "1", "content": "todo", "status": "pending"}])
    builtin = ContextCompressor(model="test/model", config_context_length=10000, quiet_mode=True)
    calls = []
    database_reads = []
    goal = GoalState(goal="done", status="done")

    class Database:
        def get_meta(self, key):
            database_reads.append(key)
            if kind == "db-error":
                raise OSError("database unavailable")
            assert kind == "done", "legacy engines must not read host state"
            return goal.to_json()

    def legacy(messages, current_tokens=None, focus_topic=None):
        calls.append({"current_tokens": current_tokens, "focus_topic": focus_topic})
        return messages

    def forwarder(messages, **kwargs):
        assert "host_state" not in kwargs
        calls.append(kwargs)
        return builtin.compress(messages, **kwargs)

    def opted(messages, *, host_state=None, **kwargs):
        assert host_state == {"todos": store.read(), "goal": None, "plan_path": None}
        calls.append(kwargs)
        return messages

    compressor = {"legacy": legacy, "forwarder": forwarder, "builtin": builtin.compress,
                  "explicit-kwargs": opted, "db-error": opted, "done": opted}[kind]
    database = Database() if kind != "explicit-kwargs" else None
    agent = SimpleNamespace(context_compressor=SimpleNamespace(compress=compressor),
                            _session_db=database, session_id="shared-session", _todo_store=store)
    if kind == "explicit-kwargs":
        agent.session_id = ""
    fn, kwargs = _bind_call(agent)
    assert ("host_state" in kwargs) == (kind in {"explicit-kwargs", "db-error", "done"})
    messages = []
    assert fn(messages, **kwargs) is messages
    assert kwargs.get("current_tokens") == 0
    if kind == "legacy":
        assert calls == [{"current_tokens": 0, "focus_topic": None}]
    if kind in {"builtin", "forwarder"}:
        assert kwargs == {"current_tokens": 0, "focus_topic": None, "force": False}

    def legacy_select(request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0):
        calls.append("selected")
        return request_messages

    agent.context_compressor.select_context = legacy_select
    request = [{"role": "user", "content": "hello"}]
    assert _apply_context_engine_selection(agent, request, request, request[-1],
                                          logger=logging.getLogger(__name__)) is request
    assert calls[-1] == "selected"
    assert database_reads == (["goal:shared-session"] if kind in {"db-error", "done"} else [])
