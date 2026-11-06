"""Opt-in task sessions for #118208; default peer traffic still uses Bot Chat.

Real peer HTTP client, profile middleware, Runs API, SQLite and agent turn admission.
Only agent construction and the model loop are replaced; each task takes its own
durable lease and writes its transcript while the canonical lease remains held.
"""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import secret_scope
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.subcommands import peer
from hermes_constants import get_hermes_home
from hermes_state import SessionDB
from run_agent import AIAgent


def _task_agent(db, session_id):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = session_id
    agent.platform = "api_server"
    agent.model = "test-model"
    agent._session_db = db
    agent._session_db_created = False
    agent._persist_disabled = False
    agent._parent_session_id = None
    agent._session_init_model_config = {}
    agent._cached_system_prompt = None
    agent._relay_pending_turn_id = None
    agent._reset_activity_labels_after_turn = lambda: None
    agent._conversation_root_id = lambda: session_id
    agent.log_prefix = ""
    agent._vprint = lambda *args, **kwargs: None
    agent.status_callback = None
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._pending_redirect = None
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    return agent


@pytest.mark.asyncio
async def test_new_peer_tasks_are_independent_and_replay_in_their_own_profile(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    keys = {"alice": "a" * 40, "bob": "b" * 40}
    stores = {}
    opened = []
    holder = f"pid={os.getpid()}:turn=canonical-holder:platform=api_server"
    for name, key in keys.items():
        profile_home = home / "profiles" / name
        profile_home.mkdir(parents=True)
        (profile_home / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")
        db = stores[name] = SessionDB(profile_home / "state.db")
        db.create_session("canonical", "gateway_botmode", profile_name=name)
        assert db.set_session_title("canonical", "Bot Chat")
        assert db.try_acquire_session_turn_lease("canonical", holder, ttl_seconds=300)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 2.0)
    secret_scope.set_multiplex_active(True)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "d" * 40}))
    adapter.gateway_runner = SimpleNamespace(config=GatewayConfig(multiplex_profiles=True))

    def create_agent(**kwargs):
        db = SessionDB(get_hermes_home() / "state.db")
        opened.append(db)
        return _task_agent(db, kwargs["session_id"])

    def model_turn(agent, message, _system, history, *_args, **_kwargs):
        lease_holder = agent._active_session_turn_lease_holder
        assert lease_holder, "the new task bypassed durable lease admission"
        agent._ensure_db_session()
        for role, content in (("user", message), ("assistant", f"reply:{message}")):
            agent._session_db.append_message(
                agent.session_id, role, content, turn_lease_holder=lease_holder)
        return {"final_response": f"reply:{message}", "messages": history}

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    monkeypatch.setattr("agent.conversation_loop.run_conversation", model_turn)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    app["api_server_adapter"] = adapter

    try:
        async with TestClient(TestServer(app)) as client:
            monkeypatch.setattr(peer, "_load_peers", lambda: {"spark": {"url": str(client.make_url("/"))}})

            async def call(profile, action="run", **kwargs):
                monkeypatch.setattr(peer, "_peer_secret", lambda _name: keys[profile])
                args = SimpleNamespace(peer_action=action, target=f"spark/{profile}", json=True,
                                       message="task", new=True, **kwargs)
                assert await asyncio.to_thread(peer.cmd_peer, args) == 0
                return json.loads(capsys.readouterr().out)

            async def completed(profile, run_id):
                async with asyncio.timeout(15):
                    while True:
                        status = await call(profile, "status", run_id=run_id)
                        if status["status"] not in {"started", "queued", "running"}:
                            assert status["status"] == "completed", status
                            return status
                        await asyncio.sleep(0.02)

            first = await call("alice", idempotency_key="task-retry")
            assert "session_id" not in first  # acceptance has not reported one
            first_status = await completed("alice", first["run_id"])
            sid = first_status["session_id"]
            assert sid != "canonical"
            assert stores["alice"].message_count(sid) == 2

            # A -> B -> A: the same retry key is scoped to its authenticated profile.
            bob = await call("bob", idempotency_key="task-retry")
            bob_status = await completed("bob", bob["run_id"])
            replay = await call("alice", idempotency_key="task-retry")
            assert replay["replayed"] and replay["run_id"] == first["run_id"]
            assert (await completed("alice", replay["run_id"]))["session_id"] == sid
            assert stores["alice"].message_count(sid) == 2
            assert bob_status["session_id"] != sid
            assert stores["bob"].get_session(sid) is None
            assert stores["alice"].get_session(bob_status["session_id"]) is None

            fresh = await call("alice")  # generated key means a different task
            fresh_status = await completed("alice", fresh["run_id"])
            assert fresh["run_id"] != first["run_id"]
            assert fresh_status["session_id"] != sid
            assert fresh["idempotency_key"] != first["idempotency_key"]
            stopped = await call("alice", "stop", run_id=first["run_id"])
            assert stopped["run_id"] == first["run_id"]
            assert (await completed("alice", fresh["run_id"]))["session_id"] == fresh_status["session_id"]

            for name, db in stores.items():
                assert db.message_count("canonical") == 0
                assert db.get_session_by_title("Bot Chat")["id"] == "canonical"
                assert db.refresh_session_turn_lease("canonical", holder, ttl_seconds=300)
                task_id = sid if name == "alice" else bob_status["session_id"]
                assert db.get_session(task_id)["profile_name"] == name
    finally:
        await adapter.disconnect()
        secret_scope.set_multiplex_active(False)
        for db in opened:
            db.close()
        for db in stores.values():
            db.release_session_turn_lease("canonical", holder)
            db.close()
