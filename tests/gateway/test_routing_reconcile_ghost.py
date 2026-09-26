"""Regression tests for _reconcile_recovered_routing_locked.

After a fallback-only startup (state.db unreadable, sessions.json legacy mirror
carried the index), the router must reconcile its in-memory index with the now-
recovered authoritative state.db rows. The reconcile contract we pin here:

  * a key created while on fallback WINS over a DB-only key (the live index is
    newer than what the DB row predates);
  * a key the fallback load saw but that was then DELETED while still on fallback
    stays deleted AND its stale state.db row is removed (a delete is a real
    mutation, not a view-only affordance);
  * an untouched fallback key YIELDS to the authoritative DB copy.

The third branch existed before. The second is the regression this suite guards:
previously `key not in current` just `continue`d and left the stale state.db
routing row behind, so the deleted key "came back" on the next healthy load as a
resurrected routing entry / pending session.
"""
from __future__ import annotations

import json

import hermes_state
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
import gateway.session_persistence as gp


def _source(user_id: str = "user-1") -> SessionSource:
    return SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id=user_id,
    )


def _make_store(tmp_path, monkeypatch, **config_kwargs) -> SessionStore:
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: str(tmp_path))
    return SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(**config_kwargs),
    )


def _routing_rows(store: SessionStore) -> dict:
    db = store._db
    if db is None:
        return {}
    return db.load_gateway_routing_entries(scope=store._routing_scope())


def _bootstrap_entry(store: SessionStore, user: str = "user-1"):
    return store.get_or_create_session(_source(user))


def _seed_fallback_index(store: SessionStore, entries) -> None:
    """Place the store into the post-fallback-only-load state the reconcile
    branch reads, using ``entries`` as the fallback-loaded index and recording it
    as the baseline, WITHOUT relying on a genuinely broken SQLite fixture."""
    with store._lock:
        store._entries = dict(entries)
    store._loaded = True
    store._routing_db_loaded = False
    store._routing_fallback_baseline = store._entries_as_dicts()



class TestFallbackDeleteStaysDeleted:
    def test_deleted_during_fallback_does_not_resurrect(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = _bootstrap_entry(store, "user-1")
        key = entry.session_key
        store._db.close()

        # Simulate the fallback-only startup state: the index was loaded from
        # the legacy mirror and recorded as the baseline, DB reported unreadable.
        _seed_fallback_index(store, {key: entry})
        assert key in store._routing_fallback_baseline

        # Delete the key while still on fallback (e.g. an age/expiry prune that
        # could not reach the broken DB).
        with store._lock:
            store._entries.pop(key, None)
        assert key not in store._entries

        # DB recovers: re-open a live handle that has a stale row for key.
        import hermes_state_registry
        store._db = hermes_state_registry.acquire(tmp_path / "state.db")
        # The stale row is already present in state.db from bootstrap.
        assert key in store._db.load_gateway_routing_entries(scope=store._routing_scope())

        store._routing_db_loaded = False
        store._reconcile_recovered_routing_locked()

        # Deleted key must not reappear in the live index...
        assert key not in store._entries
        # ...and its stale state.db row must be dropped.
        assert key not in store._db.load_gateway_routing_entries(scope=store._routing_scope())

        # A fresh healthy store therefore never sees the ghost key.
        store3 = _make_store(tmp_path, monkeypatch)
        store3._ensure_loaded()
        assert key not in store3._entries


class TestFallbackCreateWins:
    def test_created_during_fallback_wins_over_db_only(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        _bootstrap_entry(store, "user-1")
        store._db.close()
        store._db = None
        store._ensure_loaded()
        baseline = store._routing_fallback_baseline or {}

        fresh = store.get_or_create_session(_source("fresh-user"))
        fresh_key = fresh.session_key
        assert fresh_key not in baseline

        import hermes_state_registry
        store._db = hermes_state_registry.acquire(tmp_path / "state.db")

        store._routing_db_loaded = False
        store._reconcile_recovered_routing_locked()

        assert fresh_key in store._entries
        assert fresh_key in _routing_rows(store)


class TestUntouchedFallbackYields:
    def test_untouched_fallback_key_yields_to_db_authority(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = _bootstrap_entry(store, "user-1")
        key = entry.session_key
        store._db.close()

        # Fallback-only startup state.
        _seed_fallback_index(store, {key: entry})
        assert key in store._routing_fallback_baseline

        import hermes_state_registry
        db = hermes_state_registry.acquire(tmp_path / "state.db")

        # The DB's authoritative row has a different (newer) session_id than the
        # fallback mirror; reconcile must yield to it. Build a fully-populated
        # different entry to inject as the DB authority.
        authority_entry = store._entries[key]
        auth_dict = authority_entry.to_dict()
        auth_dict["session_id"] = "db-authoritative-id"
        db.save_gateway_routing_entry(
            key, json.dumps(auth_dict), scope=store._routing_scope()
        )

        store._db = db
        store._routing_db_loaded = False
        store._reconcile_recovered_routing_locked()

        assert store._entries[key].session_id == "db-authoritative-id"
