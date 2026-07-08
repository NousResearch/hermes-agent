"""P3a — reasoning override persistence across gateway restart.

The in-memory _session_reasoning_overrides dict is lost on a gateway restart,
silently reverting /reasoning high to the config default. P3a persists the
override onto the SessionEntry (sessions.json) and rehydrates it on boot — while
still clearing it on /new, /reset, /reasoning reset (durability, not scope).
"""

import json
import pytest
from unittest.mock import MagicMock

import gateway.run as gateway_run
from gateway.session import SessionEntry, SessionStore, SessionSource
from gateway.config import Platform
from datetime import datetime, timezone


def _entry(session_key="agent:main:discord:c1:c1", session_id="s1"):
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_key=session_key, session_id=session_id, created_at=now, updated_at=now,
    )


class TestSessionEntryPersistenceRoundtrip:
    def test_reasoning_override_roundtrips(self):
        e = _entry()
        e.reasoning_override = {"enabled": True, "effort": "high"}
        d = e.to_dict()
        assert d["reasoning_override"] == {"enabled": True, "effort": "high"}
        e2 = SessionEntry.from_dict(d)
        assert e2.reasoning_override == {"enabled": True, "effort": "high"}

    def test_old_sessions_json_without_field_loads(self):
        # A dict lacking the new keys (a pre-P3 sessions.json) must load, field None.
        d = _entry().to_dict()
        d.pop("reasoning_override", None)
        d.pop("model_override_identity", None)
        e = SessionEntry.from_dict(d)
        assert e.reasoning_override is None
        assert e.model_override_identity is None

    def test_from_dict_ignores_unknown_keys(self):
        # Rollback tolerance: an unknown persisted key is ignored on load (data.get,
        # not **splat).
        d = _entry().to_dict()
        d["some_future_field_from_a_newer_version"] = {"x": 1}
        e = SessionEntry.from_dict(d)  # must not raise
        assert e.session_key


def _make_store(tmp_path):
    store = object.__new__(SessionStore)
    store._entries = {}
    store.sessions_dir = tmp_path
    store._lock = __import__("threading").RLock()
    return store


class TestSetSessionReasoningOverrideWriteThrough:
    def _runner_with_store(self, store):
        runner = object.__new__(gateway_run.GatewayRunner)
        runner._session_reasoning_overrides = {}
        runner.session_store = store
        return runner

    def test_persists_to_entry_and_saves(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        runner = self._runner_with_store(store)

        runner._set_session_reasoning_override(key, {"enabled": True, "effort": "high"})
        assert store._entries[key].reasoning_override == {"enabled": True, "effort": "high"}
        # sessions.json written with the field
        saved = json.load(open(tmp_path / "sessions.json"))
        assert saved[key]["reasoning_override"] == {"enabled": True, "effort": "high"}

    def test_clear_nulls_entry_field(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.reasoning_override = {"enabled": True, "effort": "high"}
        store._entries[key] = e
        runner = self._runner_with_store(store)

        runner._set_session_reasoning_override(key, None)  # /new, /reset, /reasoning reset path
        assert store._entries[key].reasoning_override is None
        assert key not in runner._session_reasoning_overrides


class TestBootRehydrate:
    def test_reasoning_override_rehydrated_on_boot(self, tmp_path):
        # Simulate a restart: an entry persisted with a reasoning override → the
        # fresh runner's in-memory dict is repopulated on boot.
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.reasoning_override = {"enabled": True, "effort": "high"}
        store._entries[key] = e

        runner = object.__new__(gateway_run.GatewayRunner)
        runner._session_reasoning_overrides = {}
        runner._session_model_overrides = {}
        runner.session_store = store
        # _ensure_loaded is a no-op here (entries already in memory)
        store._ensure_loaded = lambda: None

        runner._rehydrate_session_overrides()
        assert runner._session_reasoning_overrides[key] == {"enabled": True, "effort": "high"}

    def test_rehydrate_skips_non_dict_reasoning(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.reasoning_override = "corrupt-not-a-dict"  # type: ignore
        store._entries[key] = e

        runner = object.__new__(gateway_run.GatewayRunner)
        runner._session_reasoning_overrides = {}
        runner._session_model_overrides = {}
        runner.session_store = store
        store._ensure_loaded = lambda: None

        runner._rehydrate_session_overrides()  # must not raise
        assert key not in runner._session_reasoning_overrides
