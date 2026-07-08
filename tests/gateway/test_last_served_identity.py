"""Track A — SessionEntry.last_served_identity persistence plumbing.

last_served_identity records the (provider, model) a session last SERVED a turn
on, so a re-init'd agent's snap-back can be announced (see the unified recovery
announce in gateway/run.py _run_agent_inner). Identity-only ({provider, model}),
never api_key/base_url — mirroring model_override_identity. Cleared on
/new,/reset,auto-reset by SessionEntry reconstruction.
"""

import json
import threading
from datetime import datetime, timezone

from gateway.session import SessionEntry, SessionStore


def _entry(session_key="agent:main:discord:c1:c1", session_id="s1"):
    now = datetime.now(timezone.utc)
    return SessionEntry(
        session_key=session_key, session_id=session_id, created_at=now, updated_at=now
    )


def _make_store(tmp_path):
    store = object.__new__(SessionStore)
    store._entries = {}
    store.sessions_dir = tmp_path
    store._lock = threading.RLock()
    store._loaded = True
    # update_session records a peer index entry after saving; no-op it here.
    store._record_gateway_session_peer = lambda *a, **k: None
    return store


class TestLastServedIdentityRoundTrip:
    def test_to_dict_from_dict_preserves_identity(self):
        e = _entry()
        e.last_served_identity = {"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"}
        d = e.to_dict()
        assert d["last_served_identity"] == {
            "provider": "claude-api-proxy-f3", "model": "claude-opus-4-8",
        }
        back = SessionEntry.from_dict(d)
        assert back.last_served_identity == e.last_served_identity

    def test_from_dict_non_dict_coerces_to_none(self):
        e = _entry()
        d = e.to_dict()
        d["last_served_identity"] = "not-a-dict"
        assert SessionEntry.from_dict(d).last_served_identity is None

    def test_default_is_none(self):
        assert _entry().last_served_identity is None


class TestNoSecrets:
    def test_persisted_identity_has_no_api_key_or_base_url(self, tmp_path):
        """A5 / INV-2: only {provider, model} is ever persisted."""
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        store.update_session(
            key,
            served_identity={"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"},
        )
        saved_text = (tmp_path / "sessions.json").read_text()
        ident = json.loads(saved_text)[key]["last_served_identity"]
        assert ident == {"provider": "claude-api-proxy-f3", "model": "claude-opus-4-8"}
        assert "api_key" not in ident and "base_url" not in ident
        # And nothing secret-shaped leaked into the whole record.
        assert "api_key" not in saved_text and "base_url" not in saved_text


class TestUpdateSessionGuards:
    def test_served_identity_persisted(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        store.update_session(key, served_identity={"provider": "yunwu", "model": "claude-fable-5"})
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_telemetry_only_call_does_not_blank_existing(self, tmp_path):
        """INV-8-adjacent: a telemetry-only update (no served_identity, e.g. the
        compression path's last_prompt_tokens=0) must NOT wipe the persisted
        route, mirroring the had_any_turn latch."""
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        store.update_session(key, served_identity={"provider": "yunwu", "model": "claude-fable-5"})
        # A later telemetry-only call must preserve it.
        store.update_session(key, last_prompt_tokens=0)
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }

    def test_empty_served_identity_does_not_blank(self, tmp_path):
        store = _make_store(tmp_path)
        key = "agent:main:discord:c1:c1"
        store._entries[key] = _entry(key)
        store.update_session(key, served_identity={"provider": "yunwu", "model": "claude-fable-5"})
        store.update_session(key, served_identity={})  # falsy → no overwrite
        assert store._entries[key].last_served_identity == {
            "provider": "yunwu", "model": "claude-fable-5",
        }


class TestResetClearsByConstruction:
    def test_reset_session_drops_last_served_identity(self, tmp_path):
        """A7 / INV-6: reset_session builds a fresh SessionEntry that does NOT
        carry last_served_identity forward — cleared by construction, single
        door, no per-site '= None'."""
        store = _make_store(tmp_path)
        store._db = None
        key = "agent:main:discord:c1:c1"
        e = _entry(key)
        e.last_served_identity = {"provider": "yunwu", "model": "claude-fable-5"}
        store._entries[key] = e
        store._ensure_loaded_locked = lambda: None
        new_entry = store.reset_session(key)
        assert new_entry is not None
        assert new_entry.last_served_identity is None
        assert store._entries[key].last_served_identity is None
