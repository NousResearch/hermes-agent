"""Installed middleware consumes current persisted input in real native turns."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from gateway.run import _profile_runtime_scope
from hermes_cli.plugins import get_plugin_manager
from hermes_state import SessionDB
from run_agent import AIAgent


def _install_consumer(home: Path) -> None:
    plugin = home / "plugins" / "input-consumer"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: input-consumer\nversion: 0.1.0\n")
    (home / "config.yaml").write_text(json.dumps({
        "plugins": {"enabled": ["input-consumer"]},
        "model": {"context_length": 256000},
        "auxiliary": {"title_generation": {"enabled": False}},
    }))
    (plugin / "__init__.py").write_text('''
import json
from hermes_constants import get_hermes_home

def register(ctx):
    def consume(**context):
        descriptor = context.get("native_user_message")
        record = {key: context.get(key) for key in (
            "session_id", "original_user_message", "native_user_message")}
        record["descriptor_present"] = "native_user_message" in context
        with (get_hermes_home() / "input-observations.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\\n")
        request = context["request"]
        if descriptor is None:
            request["messages"][-1]["content"] = "No verified persisted input."
        else:
            # Metadata belongs to this observer, not the live row or provider payload.
            descriptor["content"] = "observer-local change"
        return {"request": request}
    ctx.register_middleware("llm_request", consume)
''')


def _agent(home: Path, client: MagicMock, *, durable: bool) -> AIAgent:
    with (
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key", base_url="http://127.0.0.1:1/v1",
            provider="openai", model="test/model", api_mode="chat_completions",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            enabled_toolsets=[], max_iterations=2, session_id="shared-session",
            session_db=SessionDB(home / "state.db") if durable else None,
        )
    if not durable:
        agent._session_db = None
    agent._cached_system_prompt = "Neutral system prompt."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _client() -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None),
                                 finish_reason="stop")],
        model="test/model", usage=None,
    )
    return client


def _observations(home: Path) -> list[dict]:
    return [json.loads(line) for line in (home / "input-observations.jsonl").read_text().splitlines()]


def test_real_turn_input_provenance_follows_profiles_a_b_a(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch-profile"))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-plugins"))
    homes = {name: tmp_path / name for name in ("a", "b")}
    clients = {name: _client() for name in homes}
    agents = {}
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for name, home in homes.items():
            _install_consumer(home)
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                get_plugin_manager().discover_and_load()
                agents[name] = _agent(home, clients[name], durable=True)
        turns = (("a", "first profile A input", "first profile A input"),
                 ("b", "profile B input", "profile B input"),
                 ("a", "runtime wrapper for profile A", "second profile A input"))
        for name, message, original in turns:
            with _profile_runtime_scope(homes[name], prepared_secret_scope={}):
                result = agents[name].run_conversation(message, persist_user_message=original)
                assert result["completed"] is True
                record = _observations(homes[name])[-1]
                descriptor = record["native_user_message"]
                assert record["descriptor_present"] is True
                assert record["original_user_message"] == original
                assert descriptor["content"] == original
                stored = agents[name]._session_db.get_messages(record["session_id"])
                assert next(row for row in stored if row["id"] == descriptor["_row_id"])["content"] == original
                wire = clients[name].chat.completions.create.call_args.kwargs
                assert any(row.get("content") == message for row in wire["messages"])
                assert "native_user_message" not in wire and "original_user_message" not in wire
                assert all("_row_id" not in row for row in wire["messages"])
        assert [row["original_user_message"] for row in _observations(homes["a"])] == [
            "first profile A input", "second profile A input"]
        assert [row["original_user_message"] for row in _observations(homes["b"])] == ["profile B input"]
    finally:
        for name, agent in agents.items():
            with _profile_runtime_scope(homes[name], prepared_secret_scope={}):
                agent.close()
        set_multiplex_active(previous)


def test_missing_persistence_still_reaches_request_middleware(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-plugins"))
    home = tmp_path / "profile"
    _install_consumer(home)
    client = _client()
    with _profile_runtime_scope(home, prepared_secret_scope={}):
        get_plugin_manager().discover_and_load()
        agent = _agent(home, client, durable=False)
        try:
            result = agent.run_conversation("input without durable provenance")
            assert result["completed"] is True
            record = _observations(home)[-1]
            assert record["descriptor_present"] is True
            assert record["native_user_message"] is None
            assert record["original_user_message"] == "input without durable provenance"
            wire = client.chat.completions.create.call_args.kwargs
            assert wire["messages"][-1]["content"] == "No verified persisted input."
        finally:
            agent.close()
