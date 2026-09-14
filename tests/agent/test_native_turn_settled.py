"""Settled observers can reconcile native rows after the relevant writer leases exit."""

import asyncio
import threading
from pathlib import Path

import pytest

from hermes_cli import plugins
from gateway.session_state import SessionState, TurnState
from hermes_cli.plugins import PluginContext, PluginManager
from hermes_cli.plugins_manifest import PluginManifest
from hermes_state import SessionDB
from run_agent import AIAgent
from tests.gateway.test_transcript_redaction import _runner
from tests.agent.test_cross_process_turn_lease import _agent_with_db


def _context(monkeypatch):
    manager = PluginManager()
    manager._discovered = True
    monkeypatch.setattr(plugins, "_delivery_manager", lambda: manager)
    return PluginContext(PluginManifest(name="settled-test"), manager)


def test_native_settled_can_erase_after_persistence_and_release_including_first_turn(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda _: (None, 1))
    ctx = _context(monkeypatch)
    db = SessionDB(tmp_path / "state.db")
    rows, receipts = {}, []
    current_thread = threading.get_ident()

    def reconcile(session_id, task_id, turn_id, platform):
        assert threading.get_ident() == current_thread
        # The row is persisted and the ordinary turn's durable lease has been released.
        assert db.try_acquire_session_turn_lease(session_id, "settled-reader")
        db.release_session_turn_lease(session_id, "settled-reader")
        expected = db.get_message_redaction_snapshot(session_id, [rows[session_id]])
        result = db.redact_message_payloads(session_id, expected)
        receipts.append((session_id, task_id, turn_id, platform, result["status"]))

    ctx.register_hook("on_native_turn_settled", reconcile)
    failure = None

    def loop(agent, _message, _system, history, *_args, **_kwargs):
        if db.get_session(agent.session_id) is None:
            db.create_session(agent.session_id, source="cli")
        rows[agent.session_id] = db.append_message(
            agent.session_id, "user", "selected owned payload",
            turn_lease_holder=getattr(agent, "_active_session_turn_lease_holder", None),
        )
        if failure is not None:
            raise failure("original turn failure")
        return {"final_response": "settled", "messages": history, "failed": False}

    monkeypatch.setattr("agent.conversation_loop.run_conversation", loop)
    try:
        for sid, failure in (("existing", None), ("first", None),
                             ("failed", RuntimeError), ("interrupted", KeyboardInterrupt)):
            if sid != "first":
                db.create_session(sid, source="cli")
            agent = _agent_with_db(db, session_id=sid, platform="cli")
            if failure is None:
                result = AIAgent.run_conversation(agent, "work", task_id="accepted-task")
                assert result["final_response"] == "settled"
            else:
                with pytest.raises(failure, match="original turn failure"):
                    AIAgent.run_conversation(agent, "work", task_id="accepted-task")
            assert receipts[-1][0:2] == (sid, "accepted-task")
            assert receipts[-1][2].startswith(f"{sid}:accepted-task:")
            assert receipts[-1][3:] == ("cli", "redacted")
            assert "selected owned payload" not in str(db.get_messages(sid))
        assert len(receipts) == 4

        # Rejected admission never entered a turn and must not announce settlement.
        agent = _agent_with_db(db, session_id="existing", platform="cli")
        agent._interrupt_requested = True
        result = AIAgent.run_conversation(agent, "not admitted")
        assert result["interrupted"] is True
        assert len(receipts) == 4
        # Persistence-disabled internal forks do not become native session events.
        failure = None
        agent = _agent_with_db(db, session_id="detached", platform="cli")
        agent._persist_disabled = True
        AIAgent.run_conversation(agent, "internal turn")
        assert len(receipts) == 4
    finally:
        db.close()


def test_gateway_settled_runs_on_loop_after_release_and_schedules_real_redaction(tmp_path, monkeypatch):
    ctx = _context(monkeypatch)
    runner, db = _runner(tmp_path)
    tasks, receipts = [], []
    message_id = db.get_messages("source")[0]["id"]
    expected = db.get_message_redaction_snapshot("source", [message_id])

    def reconcile(session_id, session_key, run_generation, gateway):
        loop = asyncio.get_running_loop()  # This observer must never run on a timeout worker.
        assert gateway is runner
        assert not runner._turn_leases._leases[session_id].lock.locked()
        tasks.append(loop.create_task(
            gateway.redact_native_message_payloads(session_key, session_id, expected)
        ))
        receipts.append((session_id, session_key, run_generation))

    ctx.register_hook("on_gateway_turn_settled", reconcile)

    async def scenario():
        token = await runner._turn_leases.acquire("source", owner_key="route:source", generation=7)
        state = SessionState(turn=TurnState(lease_tokens={7: token}))
        runner._peek_session_state = lambda key: state if key == "route:source" else None
        assert runner._release_turn_lease("route:source", 6) is False
        assert receipts == []
        assert runner._release_turn_lease("route:source", 7) is True
        assert runner._release_turn_lease("route:source", 7) is False
        assert receipts == [("source", "route:source", 7)]
        result = await tasks[0]
        assert result["status"] == "redacted"
        assert "original source payload" not in str(db.get_messages("source"))
        assert set(runner._agent_cache) == {"route:unrelated"}

    try:
        asyncio.run(scenario())
    finally:
        db.close()


def test_settled_hooks_select_owning_profile_and_scheduled_work_keeps_that_scope(tmp_path, monkeypatch):
    from gateway.run import _profile_runtime_scope
    from hermes_constants import get_hermes_home

    home = tmp_path / ".hermes"
    secondary = home / "profiles" / "secondary"
    secondary.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("agent.turn_liveness.resolve_turn_liveness_settings", lambda _: (None, 1))
    runner, db = _runner(tmp_path)
    runner.session_store.config.multiplex_profiles = True
    immediate, deferred, native, tasks, managers = [], [], [], [], {}

    def register(name, profile_home):
        with _profile_runtime_scope(profile_home, hydrate_secrets=False):
            manager = plugins.get_plugin_manager()
            manager._discovered = True
            managers[name] = manager
            ctx = PluginContext(PluginManifest(name="settled-profile-test"), manager)

            def gateway_settled(session_key, **_kwargs):
                immediate.append((name, get_hermes_home(), session_key))

                async def reconciliation():
                    await asyncio.sleep(0)
                    deferred.append((name, get_hermes_home(), plugins.get_plugin_manager() is manager))

                tasks.append(asyncio.create_task(reconciliation()))

            ctx.register_hook("on_gateway_turn_settled", gateway_settled)
            ctx.register_hook("on_native_turn_settled", lambda **_: native.append((name, get_hermes_home())))

    register("default", home)
    register("secondary", secondary)
    assert managers["default"] is not managers["secondary"]

    async def release_turn(key, generation):
        token = await runner._turn_leases.acquire("source", owner_key=key, generation=generation)
        state = SessionState(turn=TurnState(lease_tokens={generation: token}))
        runner._peek_session_state = lambda candidate: state if candidate == key else None
        assert runner._release_turn_lease(key, generation) is True

    async def scenario():
        secondary_key = "agent:secondary:telegram:dm:owner"
        default_key = "agent:main:telegram:dm:owner"
        # Both releases happen after their agent worker's profile scope has exited.
        assert get_hermes_home() == home
        await release_turn(secondary_key, 1)
        await release_turn(default_key, 2)
        await asyncio.gather(*tasks)
        assert immediate == [("secondary", secondary, secondary_key), ("default", home, default_key)]
        assert deferred == [("secondary", secondary, True), ("default", home, True)]
        assert get_hermes_home() == home
        # Missing named profiles must not dispatch through the ambient default manager.
        await release_turn("agent:missing:telegram:dm:owner", 3)
        assert len(immediate) == 2

    try:
        asyncio.run(scenario())
        monkeypatch.setattr("agent.conversation_loop.run_conversation",
                            lambda *args, **kwargs: {"final_response": "fixture", "messages": [], "failed": False})
        # CLI facade settlement still runs within its caller's profile; no gateway scope needed.
        with _profile_runtime_scope(secondary, hydrate_secrets=False):
            agent = _agent_with_db(db, session_id="source", platform="cli")
            AIAgent.run_conversation(agent, "settle")
        assert native == [("secondary", secondary)]
        assert get_hermes_home() == home
    finally:
        db.close()
