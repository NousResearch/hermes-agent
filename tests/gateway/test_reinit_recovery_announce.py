"""Track A — the unified re-init/restore recovery announce at the gateway
(_announce_and_persist_served_route), keyed on the FINAL served route.

This is the single site that emits BOTH recovery legs:
  • "new turn" — cache-warm restore (agent restored at the turn prologue)
  • "re-init"  — the agent cache was rebuilt → a fresh agent inited on the
                 config default; its silent snap-back is the bug this fixes.

Covers: A1 (re-init snap-back announces), A3 (correct rider / one line),
A4 (silent no-op re-init), A6 (sink written regardless of chat gate), and the
persist of the served route for the next comparison.
"""

import json
import threading
import types
from datetime import datetime, timezone

import gateway.run as gateway_run
from gateway.session import SessionEntry, SessionStore


def _entry(session_key="agent:main:discord:c1:c1", session_id="s1", last_served=None):
    now = datetime.now(timezone.utc)
    e = SessionEntry(session_key=session_key, session_id=session_id, created_at=now, updated_at=now)
    e.last_served_identity = last_served
    return e


def _store(tmp_path):
    store = object.__new__(SessionStore)
    store._entries = {}
    store.sessions_dir = tmp_path
    store._lock = threading.RLock()
    store._loaded = True
    store._record_gateway_session_peer = lambda *a, **k: None
    return store


def _runner(store):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.session_store = store
    return runner


def _agent():
    a = types.SimpleNamespace()
    a._announced = []
    a._emit_status = lambda m: a._announced.append(m)
    a._last_fallback_announced = None
    return a


def _sink_lines(home):
    import pathlib
    p = pathlib.Path(home) / "state" / "model-route-changes.log"
    if not p.exists():
        return []
    return [ln for ln in p.read_text().splitlines() if ln.strip()]


def _call(runner, agent, key, provider, model, was_reinit):
    runner._announce_and_persist_served_route(
        agent=agent,
        session_key=key,
        served_provider=provider,
        served_model=model,
        was_reinit=was_reinit,
    )


class TestReinitRecoveryAnnounce:
    def test_reinit_snapback_announces(self, tmp_path, monkeypatch):
        """A1: prev served = opus, this turn (fresh agent) serves fable →
        '🔄 Model recovery (re-init): .../opus → .../fable' + a sink line."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = _store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(
            key, last_served={"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"}
        )
        runner = _runner(store)
        agent = _agent()
        _call(runner, agent, key, "yunwu", "claude-fable-5", was_reinit=True)
        assert len(agent._announced) == 1
        assert agent._announced[0] == (
            "🔄 Model recovery (re-init): "
            "claude-api-proxy-f3/claude-opus-4-8 → yunwu/claude-fable-5"
        )
        # Persisted the new served route for next comparison.
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_restore_uses_new_turn_rider(self, tmp_path, monkeypatch):
        """A2: a cache-warm restore (was_reinit=False) reads '(new turn)'."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = _store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(
            key, last_served={"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"}
        )
        runner = _runner(store)
        agent = _agent()
        _call(runner, agent, key, "yunwu", "claude-fable-5", was_reinit=False)
        assert agent._announced[0].startswith("🔄 Model recovery (new turn): ")

    def test_same_model_reinit_is_silent(self, tmp_path, monkeypatch):
        """A4 / INV-3: a re-init that serves the SAME model it last served
        emits nothing (route-tuple no-op)."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = _store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(
            key, last_served={"provider": "yunwu", "model": "claude-fable-5"}
        )
        runner = _runner(store)
        agent = _agent()
        _call(runner, agent, key, "yunwu", "claude-fable-5", was_reinit=True)
        assert agent._announced == []
        # Still persists (idempotent — same value).
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_no_prior_served_route_is_silent_but_persists(self, tmp_path, monkeypatch):
        """First turn ever (last_served_identity=None): nothing to recover from,
        so silent — but the served route is persisted for next time."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: true\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = _store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key, last_served=None)
        runner = _runner(store)
        agent = _agent()
        _call(runner, agent, key, "yunwu", "claude-fable-5", was_reinit=True)
        assert agent._announced == []
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_sink_written_when_chat_gate_off(self, tmp_path, monkeypatch):
        """A6 / INV-5: with announce_recovery OFF, no chat emit but the durable
        sink line IS written (so the sink stops showing zero recovery lines)."""
        monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path, raising=False)
        (tmp_path / "config.yaml").write_text("model:\n  announce_recovery: false\n")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        store = _store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(
            key, last_served={"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"}
        )
        runner = _runner(store)
        agent = _agent()
        _call(runner, agent, key, "yunwu", "claude-fable-5", was_reinit=True)
        assert agent._announced == []  # chat gate off
        lines = _sink_lines(str(tmp_path))
        assert len(lines) == 1 and " recovery " in lines[0], lines

    def test_missing_agent_or_key_is_noop(self, tmp_path):
        """Degenerate inputs never raise and never persist."""
        store = _store(tmp_path)
        runner = _runner(store)
        # No agent
        runner._announce_and_persist_served_route(
            agent=None, session_key="k", served_provider="p", served_model="m", was_reinit=True,
        )
        # No served model
        agent = _agent()
        runner._announce_and_persist_served_route(
            agent=agent, session_key="k", served_provider="p", served_model=None, was_reinit=True,
        )
        assert agent._announced == []
