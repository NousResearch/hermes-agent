"""Exact owned-copy redaction preserves unrelated work and settles on cancellation."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.session import SessionStore
from gateway.turn_lease import SessionTurnLeaseRegistry
from hermes_state import SessionDB


class _Runner(GatewayAgentCacheMixin):
    def _running_agent_ids(self):
        return set()

    def _peek_session_state(self, key):
        return None

    def _spawn_release_thread(self, target, args, name, *, inline_fallback):
        self.client_cleanups.append((target, args))


def _runner(tmp_path):
    db = SessionDB(tmp_path / "profile" / "state.db")
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    runner = _Runner()
    runner.session_store = store
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {}
    runner.client_cleanups = []
    runner._turn_leases = SessionTurnLeaseRegistry()
    for sid, parent in (("source", None), ("child", "source"), ("unrelated", None)):
        if parent:
            db.end_session(parent, "compression")
        db.create_session(sid, source="test", parent_session_id=parent)
        db.append_message(sid, "user", content=f"original {sid} payload")
        key = f"route:{sid}"
        agent = SimpleNamespace(
            session_id=sid, _session_messages=[{"content": f"cached {sid}"}],
            _db_flush_scan_prefix=[{"content": f"cached {sid}"}],
            sandbox=object(), clients_released=False,
        )
        agent.release_clients = lambda agent=agent: setattr(agent, "clients_released", True)
        runner._agent_cache[key] = (agent, "sig", 1, sid)
        store._entries[key] = SimpleNamespace(session_id=sid)
    return runner, db


def test_busy_and_recovery_copies_defer_then_idle_redaction_preserves_other_work(tmp_path):
    runner, db = _runner(tmp_path)
    source = db.get_messages("source")
    snapshot = db.get_message_redaction_snapshot("source", [source[0]["id"]])
    unaffected = db.get_messages("unrelated")
    old_agents = {key: entry[0] for key, entry in runner._agent_cache.items()}
    sandboxes = {key: agent.sandbox for key, agent in old_agents.items()}

    async def scenario():
        registry = runner._turn_leases
        token = await registry.acquire("source", owner_key="ordinary-turn", generation=1)
        assert registry.rebind(token, "child")
        result = await runner.redact_native_message_payloads("route:source", "source", snapshot)
        assert result["status"] == "pending"
        assert db.get_messages("source") == source
        registry.release(token)

        runner.session_store._dirty_transcripts["child"] = [{"content": "pending recovery"}]
        result = await runner.redact_native_message_payloads("route:source", "source", snapshot)
        assert result["status"] == "pending"
        runner.session_store._dirty_transcripts.clear()

        spool = db.db_path.parent / "pending_messages" / "pending-trial.json"
        spool.parent.mkdir()
        spool.write_text(json.dumps({"session_id": "source", "messages": source}))
        result = await runner.redact_native_message_payloads("route:source", "source", snapshot)
        assert result["status"] == "pending"
        assert spool.exists() and db.get_messages("source") == source
        spool.unlink()

        assert db.try_acquire_session_turn_lease("child", "other-process")
        result = await runner.redact_native_message_payloads("route:source", "source", snapshot)
        assert result["status"] == "pending"
        db.release_session_turn_lease("child", "other-process")

        result = await runner.redact_native_message_payloads("route:source", "source", snapshot)
        assert result["status"] == "redacted"
        assert result["redacted_ids"] == [source[0]["id"]]
        assert "original source payload" not in json.dumps(db.get_messages("source"))
        assert db.get_messages("unrelated") == unaffected
        assert set(runner._agent_cache) == {"route:unrelated"}
        assert runner._agent_cache["route:unrelated"][0] is old_agents["route:unrelated"]
        # The receipt precedes daemon client cleanup, but no selected history buffers remain.
        assert not any(agent.clients_released for agent in old_agents.values())
        for key in ("route:source", "route:child"):
            assert old_agents[key]._session_messages == []
            assert old_agents[key]._db_flush_scan_prefix is None
        assert all(agent.sandbox is sandboxes[key] for key, agent in old_agents.items())
        for target, args in runner.client_cleanups:
            target(*args)
        assert old_agents["route:source"].clients_released
        assert old_agents["route:child"].clients_released
        assert not old_agents["route:unrelated"].clients_released
        # Both local alias leases and the shared compression durable lease are released.
        tokens = await registry.try_acquire_many(["source", "child"], owner_key="next", generation=2)
        assert tokens is not None
        for token in tokens:
            registry.release(token)
        for sid in ("source", "child"):
            assert db.try_acquire_session_turn_lease(sid, "next")
            db.release_session_turn_lease(sid, "next")

    try:
        asyncio.run(scenario())
    finally:
        db.close()


def test_cancellation_keeps_leases_until_real_worker_and_cache_eviction_settle(tmp_path, monkeypatch):
    runner, db = _runner(tmp_path)
    source = db.get_messages("source")
    snapshot = db.get_message_redaction_snapshot("source", [source[0]["id"]])
    entered, release = threading.Event(), threading.Event()
    redact = db.redact_message_payloads

    def delayed(*args, **kwargs):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test worker was not released")
        return redact(*args, **kwargs)

    monkeypatch.setattr(db, "redact_message_payloads", delayed)

    async def scenario():
        task = asyncio.create_task(
            runner.redact_native_message_payloads("route:source", "source", snapshot)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            # Yield through the cancellation handler, then verify the real worker is fenced.
            await asyncio.sleep(0)
            assert not task.done()
            assert not db.try_acquire_session_turn_lease("source", "competitor")
            assert await runner._turn_leases.try_acquire_many(
                ["source"], owner_key="competitor", generation=2,
            ) is None
            assert "route:source" in runner._agent_cache
            # A boundary command can install unrelated work under a formerly affected
            # routing key while the DB worker runs. Never evict the replacement.
            runner._agent_cache["route:source"] = runner._agent_cache["route:unrelated"]
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert "original source payload" not in json.dumps(db.get_messages("source"))
        assert set(runner._agent_cache) == {"route:source", "route:unrelated"}
        assert runner._agent_cache["route:source"] is runner._agent_cache["route:unrelated"]
        assert db.try_acquire_session_turn_lease("source", "after-settlement")
        db.release_session_turn_lease("source", "after-settlement")

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        db.close()
