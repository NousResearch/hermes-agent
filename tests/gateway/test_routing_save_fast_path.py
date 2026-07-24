"""Single-row routing save fast path.

Metadata-only per-turn writes (get_or_create_session's healthy-path
``updated_at`` bump, ``update_session``) persist through a single-row
UPSERT instead of rewriting the whole routing index. These tests prove
the write-skipping never loses a NEEDED write: changed values always
land in state.db, and restart rebinding works even when the legacy
sessions.json mirror lagged behind (or never existed).
"""
from __future__ import annotations

import json
import threading

import hermes_state
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


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
    return SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(**config_kwargs),
    )


def _routing_row(store: SessionStore, session_key: str) -> dict:
    rows = store._db.load_gateway_routing_entries(scope=store._routing_scope())
    return json.loads(rows[session_key])


class TestChangedValuesAlwaysPersist:
    def test_update_session_persists_last_prompt_tokens_to_db(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        store.update_session(entry.session_key, last_prompt_tokens=54321)

        durable = _routing_row(store, entry.session_key)
        assert durable["last_prompt_tokens"] == 54321
        store._db.close()

    def test_healthy_path_bump_persists_updated_at_to_db(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        before = _routing_row(store, entry.session_key)["updated_at"]

        # Second lookup takes the healthy-path metadata-only save.
        again = store.get_or_create_session(_source())
        assert again.session_id == entry.session_id

        after = _routing_row(store, entry.session_key)["updated_at"]
        assert after >= before
        # The row reflects the in-memory bump, not a stale copy.
        assert after == again.updated_at.isoformat()
        store._db.close()

    def test_fast_path_survives_restart_across_stores(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        store.update_session(entry.session_key, last_prompt_tokens=777)
        store._db.close()

        restarted = _make_store(tmp_path, monkeypatch)
        rebound = restarted.get_or_create_session(_source())
        assert rebound.session_id == entry.session_id
        assert rebound.last_prompt_tokens == 777
        restarted._db.close()


class TestRestartRebindWithoutMirror:
    def test_rebind_works_when_mirror_lagged_fast_path_writes(
        self, tmp_path, monkeypatch
    ):
        """The fast path skips sessions.json; state.db alone must rebind."""
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        sessions_json = tmp_path / "sessions" / "sessions.json"
        assert sessions_json.exists()  # structural save wrote the mirror

        # Remove the mirror, then do fast-path-only writes: they must NOT
        # recreate it (proves the fast path ran) and must not need it.
        sessions_json.unlink()
        store.update_session(entry.session_key, last_prompt_tokens=42)
        again = store.get_or_create_session(_source())
        assert again.session_id == entry.session_id
        assert not sessions_json.exists()
        store._db.close()

        restarted = _make_store(tmp_path, monkeypatch)
        rebound = restarted.get_or_create_session(_source())
        assert rebound.session_id == entry.session_id
        assert rebound.last_prompt_tokens == 42
        restarted._db.close()

    def test_compression_heal_takes_full_path_and_updates_mirror(
        self, tmp_path, monkeypatch
    ):
        """A heal rewrites session_id, so it must bypass the fast path.

        The fast path persists state.db only; if a heal took it, the
        sessions.json mirror would keep the ended parent id until the
        next structural save (indefinitely on a healthy key).
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        healed_id = entry.session_id + "_child"
        monkeypatch.setattr(
            store,
            "_compression_tip_for_session_id",
            lambda sid: healed_id if sid == entry.session_id else sid,
        )

        again = store.get_or_create_session(_source())
        assert again.session_id == healed_id

        assert _routing_row(store, entry.session_key)["session_id"] == healed_id
        sessions_json = tmp_path / "sessions" / "sessions.json"
        data = json.loads(sessions_json.read_text(encoding="utf-8"))
        assert data[entry.session_key]["session_id"] == healed_id
        store._db.close()

    def test_structural_save_still_rewrites_mirror(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        first = store.get_or_create_session(_source())
        fresh = store.get_or_create_session(_source(), force_new=True)
        assert fresh.session_id != first.session_id

        sessions_json = tmp_path / "sessions" / "sessions.json"
        data = json.loads(sessions_json.read_text(encoding="utf-8"))
        assert data[fresh.session_key]["session_id"] == fresh.session_id
        store._db.close()


class TestFallbacks:
    def test_no_db_falls_back_to_full_rewrite(self, tmp_path, monkeypatch):
        """DB-less installs keep sessions.json durable every turn."""
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        store._db.close()
        store._db = None

        store.update_session(entry.session_key, last_prompt_tokens=99)

        sessions_json = tmp_path / "sessions" / "sessions.json"
        data = json.loads(sessions_json.read_text(encoding="utf-8"))
        assert data[entry.session_key]["last_prompt_tokens"] == 99

    def test_failed_upsert_falls_back_to_full_rewrite(
        self, tmp_path, monkeypatch
    ):
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        def boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(store._db, "save_gateway_routing_entry", boom)
        store.update_session(entry.session_key, last_prompt_tokens=1234)

        # The full rewrite carried the change to both stores.
        sessions_json = tmp_path / "sessions" / "sessions.json"
        data = json.loads(sessions_json.read_text(encoding="utf-8"))
        assert data[entry.session_key]["last_prompt_tokens"] == 1234
        assert _routing_row(store, entry.session_key)["last_prompt_tokens"] == 1234
        store._db.close()


class TestPeerRecordConsistency:
    def test_update_session_records_peer_fields_snapshotted_under_lock(
        self, tmp_path, monkeypatch
    ):
        """Peer fields must come from one lock-held snapshot, not late reads.

        The peer record runs outside ``_lock``; a concurrent reset that
        rewrites the entry in that window must not produce a torn
        old/new field mix in the recorded row.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())
        original_id = entry.session_id

        recorded = []
        monkeypatch.setattr(
            store,
            "_record_gateway_session_peer",
            lambda sid, key, origin, display_name=None: recorded.append(
                (sid, key, display_name)
            ),
        )
        orig_save_entry = store._save_entry

        def swap_then_save(key):
            # Simulate a concurrent reset landing after update_session
            # released _lock but before the peer record runs.
            with store._lock:
                store._entries[key].session_id = "reset-rewrote-me"
            orig_save_entry(key)

        monkeypatch.setattr(store, "_save_entry", swap_then_save)
        store.update_session(entry.session_key, last_prompt_tokens=5)

        assert recorded == [
            (original_id, entry.session_key, entry.display_name)
        ]
        store._db.close()


class TestGenerationOrdering:
    def test_upsert_skipped_when_newer_full_snapshot_persisted(
        self, tmp_path, monkeypatch
    ):
        """A full snapshot taken after our serialize point must win.

        Its copy of the key is same-or-newer, so writing ours would
        regress it. Simulated by advancing _persisted_routing_generation
        past the generation captured at serialize time.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        calls = []
        monkeypatch.setattr(
            store._db,
            "save_gateway_routing_entry",
            lambda *a, **k: calls.append((a, k)),
        )
        store._persisted_routing_generation = store._routing_generation + 1

        store._save_entry(entry.session_key)

        assert calls == []
        store._db.close()

    def test_restart_rebind_after_skipped_idempotent_write(
        self, tmp_path, monkeypatch
    ):
        """A skipped fast-path write never orphans the session on restart.

        The skip only fires when a newer FULL snapshot — which contains
        this key — already persisted, so state.db can always rebind.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        calls = []
        monkeypatch.setattr(
            store._db,
            "save_gateway_routing_entry",
            lambda *a, **k: calls.append(a),
        )
        store._persisted_routing_generation = store._routing_generation + 1
        store._save_entry(entry.session_key)
        assert calls == []  # the idempotent write was skipped
        store._db.close()

        restarted = _make_store(tmp_path, monkeypatch)
        rebound = restarted.get_or_create_session(_source())
        assert rebound.session_id == entry.session_id
        restarted._db.close()

    def test_upsert_proceeds_at_current_generation(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        calls = []
        monkeypatch.setattr(
            store._db,
            "save_gateway_routing_entry",
            lambda key, entry_json, **k: calls.append((key, entry_json, k)),
        )

        store._save_entry(entry.session_key)

        assert len(calls) == 1
        key, entry_json, kwargs = calls[0]
        assert key == entry.session_key
        assert json.loads(entry_json)["session_id"] == entry.session_id
        assert kwargs["scope"] == store._routing_scope()
        store._db.close()

    def test_missing_key_is_a_noop(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path, monkeypatch)
        store._ensure_loaded()

        calls = []
        monkeypatch.setattr(
            store._db,
            "save_gateway_routing_entry",
            lambda *a, **k: calls.append(a),
        )

        store._save_entry("agent:main:local:nope")

        assert calls == []
        store._db.close()


class _GatedSaveLock:
    """``_save_lock`` wrapper that parks one thread at lock entry.

    The parked thread has already serialized its entry (and taken its
    revision) under ``_lock``, so this deterministically reproduces a
    delayed write: the writer sits between its serialize point and the
    durable-write section while other writers run to completion.
    """

    def __init__(self, inner: threading.Lock) -> None:
        self._inner = inner
        self.gated_thread: threading.Thread | None = None
        self.reached = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        if threading.current_thread() is self.gated_thread:
            self.reached.set()
            assert self.release.wait(timeout=5), "gated writer never released"
        return self._inner.__enter__()

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)


class TestDelayedWriteOrdering:
    def test_reverse_completion_fast_saves_keep_newer_entry(
        self, tmp_path, monkeypatch
    ):
        """Two same-key fast saves completing in reverse order.

        The older save serializes first but its UPSERT is delayed past
        the newer save's. Its revision is below the newer one recorded
        in ``_fast_persisted_entries``, so it must skip — the newer
        entry_json stays in state.db.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        gate = _GatedSaveLock(store._save_lock)
        store._save_lock = gate
        older = threading.Thread(
            target=store.update_session,
            args=(entry.session_key,),
            kwargs={"last_prompt_tokens": 1},
        )
        gate.gated_thread = older
        older.start()
        # The older save has serialized last_prompt_tokens=1 and parked
        # before its UPSERT; the newer save now runs to completion.
        assert gate.reached.wait(timeout=5)
        store.update_session(entry.session_key, last_prompt_tokens=2)

        gate.release.set()
        older.join(timeout=5)
        assert not older.is_alive()

        assert _routing_row(store, entry.session_key)["last_prompt_tokens"] == 2
        store._db.close()

    def test_delayed_full_rewrite_folds_in_newer_fast_save(
        self, tmp_path, monkeypatch
    ):
        """A full rewrite landing after a later-serialized fast save.

        The rewrite's snapshot predates the fast save, so replaying it
        verbatim would regress the key. ``_persist_routing_data`` must
        fold the newer fast record into the rewrite — in state.db and in
        the sessions.json mirror.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        with store._lock:
            data, generation = store._snapshot_routing_locked()
        store.update_session(entry.session_key, last_prompt_tokens=7)

        # The delayed full rewrite lands last.
        store._persist_routing_data(data, generation)

        assert _routing_row(store, entry.session_key)["last_prompt_tokens"] == 7
        sessions_json = tmp_path / "sessions" / "sessions.json"
        mirror = json.loads(sessions_json.read_text(encoding="utf-8"))
        assert mirror[entry.session_key]["last_prompt_tokens"] == 7
        store._db.close()

    def test_delayed_fast_save_skips_after_newer_full_rewrite(
        self, tmp_path, monkeypatch
    ):
        """A fast save delayed past a full rewrite serialized after it.

        The rewrite's snapshot contains a newer copy of the key, so the
        parked UPSERT must skip instead of regressing it.
        """
        store = _make_store(tmp_path, monkeypatch)
        entry = store.get_or_create_session(_source())

        upserts = []
        real_saver = store._db.save_gateway_routing_entry

        def counting_saver(session_key, entry_json, **kwargs):
            upserts.append(session_key)
            real_saver(session_key, entry_json, **kwargs)

        monkeypatch.setattr(
            store._db, "save_gateway_routing_entry", counting_saver
        )
        gate = _GatedSaveLock(store._save_lock)
        store._save_lock = gate
        older = threading.Thread(
            target=store.update_session,
            args=(entry.session_key,),
            kwargs={"last_prompt_tokens": 1},
        )
        gate.gated_thread = older
        older.start()
        assert gate.reached.wait(timeout=5)

        # A full rewrite serialized after the parked save persists first.
        with store._lock:
            store._entries[entry.session_key].last_prompt_tokens = 2
        store._save_entries()

        gate.release.set()
        older.join(timeout=5)
        assert not older.is_alive()

        assert upserts == []  # the delayed UPSERT was skipped
        assert _routing_row(store, entry.session_key)["last_prompt_tokens"] == 2
        store._db.close()
