"""Installed consumers reconcile explicit rows after native and gateway settlement."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.secret_scope import is_multiplex_active, set_multiplex_active
from gateway.run import _profile_runtime_scope
from gateway.session_state import SessionState, TurnState
from hermes_cli.plugins import get_plugin_manager
from hermes_state import SessionDB
from run_agent import AIAgent


def _install(home):
    plugin = home / "plugins" / "settled-consumer"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: settled-consumer\nversion: 0.1.0\n")
    (home / "config.yaml").write_text(json.dumps({
        "plugins": {"enabled": ["settled-consumer"]},
        "model": {"context_length": 256000},
        "auxiliary": {"title_generation": {"enabled": False}},
    }))
    (plugin / "__init__.py").write_text('''
import asyncio
import json
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

def record(kind, receipt):
    with (get_hermes_home() / "settlement.jsonl").open("a") as stream:
        stream.write(json.dumps({"kind": kind, "receipt": receipt}) + "\\n")

def selected(db, session_id):
    rows = db.get_messages(session_id)
    # The synthetic consumer owns these explicit fixture inputs only.
    ids = [row["id"] for row in rows if row["role"] == "user"
           and str(row["content"]).startswith("owned payload ")]
    return db.get_message_redaction_snapshot(session_id, ids), rows[-1]["id"]

def register(ctx):
    def native(session_id, **_):
        with SessionDB(get_hermes_home() / "state.db") as db:
            assert db.try_acquire_session_turn_lease(session_id, "consumer-check")
            db.release_session_turn_lease(session_id, "consumer-check")
            expected, watermark = selected(db, session_id)
            record("native", db.redact_message_payloads(
                session_id, expected, expected_message_watermark=watermark))

    def gateway(session_id, session_key, gateway, **_):
        db = gateway.session_store._db_for_key(session_key)
        expected, watermark = selected(db, session_id)
        async def reconcile():
            receipt = await gateway.redact_native_message_payloads(
                session_key, session_id, expected, expected_message_watermark=watermark)
            record("gateway", receipt)
        asyncio.get_running_loop().create_task(reconcile())

    ctx.register_hook("on_native_turn_settled", native)
    ctx.register_hook("on_gateway_turn_settled", gateway)
''')


def _records(home):
    path = home / "settlement.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _agent(home, client):
    with (
        patch("agent.process_bootstrap.OpenAI", return_value=client),
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
    ):
        agent = AIAgent(api_key="test-key", base_url="http://127.0.0.1:1/v1",
                        provider="openai", model="test/model", api_mode="chat_completions",
                        quiet_mode=True, skip_context_files=True, skip_memory=True,
                        enabled_toolsets=[], max_iterations=2, session_id="shared-session",
                        session_db=SessionDB(home / "state.db"))
    agent._cached_system_prompt = "Neutral system prompt."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def test_installed_native_consumer_erases_after_real_turns_in_profiles_a_b_a(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "launch"))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-plugins"))
    homes = {name: tmp_path / name for name in ("a", "b")}
    agents, clients, histories = {}, {}, {}
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for name, home in homes.items():
            _install(home)
            client = MagicMock()
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="done", tool_calls=None),
                                         finish_reason="stop")], model="test/model", usage=None)
            clients[name] = client
            with _profile_runtime_scope(home, prepared_secret_scope={}):
                get_plugin_manager().discover_and_load()
                agents[name] = _agent(home, client)
        for name, suffix in (("a", "a1"), ("b", "b1"), ("a", "a2")):
            with _profile_runtime_scope(homes[name], prepared_secret_scope={}):
                result = agents[name].run_conversation(
                    "owned payload " + suffix, conversation_history=histories.get(name))
                assert result["completed"] is True
                histories[name] = result["messages"]
                records = _records(homes[name])
                assert records[-1]["kind"] == "native"
                assert records[-1]["receipt"]["status"] == "redacted"
                stored = agents[name]._session_db.get_messages("shared-session")
                assert all("owned payload" not in str(row["content"]) for row in stored)
                assert stored[-1]["content"] == "done"
                if suffix == "a2":
                    wire = clients[name].chat.completions.create.call_args.kwargs
                    assert "owned payload a1" not in str(wire["messages"])
        assert len(_records(homes["a"])) == 2
        assert len(_records(homes["b"])) == 1
    finally:
        for name, agent in agents.items():
            with _profile_runtime_scope(homes[name], prepared_secret_scope={}):
                agent.close()
        set_multiplex_active(previous)


def test_installed_gateway_consumer_runs_after_local_lease_and_clears_cached_copy(tmp_path, monkeypatch):
    from agent import secret_scope
    from tests.gateway.test_transcript_redaction import _runner
    from tui_gateway import launch_profile_policy

    home = tmp_path / ".hermes"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "empty-plugins"))
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", False)
    monkeypatch.setattr(launch_profile_policy, "_snapshot", None)
    _install(home)
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text("{}\n")
    launch_profile_policy.activate_multi_profile_hosting()
    runner, db = _runner(tmp_path)
    runner.session_store.config.multiplex_profiles = False
    target = db.append_message("child", "user", "owned payload gateway")
    agent = runner._agent_cache["route:child"][0]

    async def scenario():
        with _profile_runtime_scope(home, prepared_secret_scope={}):
            get_plugin_manager().discover_and_load()
            token = await runner._turn_leases.acquire("child", owner_key="route:child", generation=7)
            state = SessionState(turn=TurnState(lease_tokens={7: token}))
            runner._peek_session_state = lambda key: state if key == "route:child" else None
            # A standalone gateway can share its process with a hosted profile.
            # Settlement still belongs to its launch profile after that scope exits.
            with _profile_runtime_scope(secondary, prepared_secret_scope={}):
                assert runner._release_turn_lease("route:child", 7) is True
            async with asyncio.timeout(5):
                while not _records(home):
                    await asyncio.sleep(.01)
            receipt = _records(home)[0]
            assert receipt["kind"] == "gateway"
            assert receipt["receipt"]["redacted_ids"] == [target]
            assert agent._session_messages == [] and agent._db_flush_scan_prefix is None
            assert "route:unrelated" in runner._agent_cache
            assert runner._release_turn_lease("route:child", 7) is False
            assert len(_records(home)) == 1
            assert _records(secondary) == []
    try:
        asyncio.run(scenario())
    finally:
        db.close()
