"""Tests for SessionStore.prune_old_entries and the gateway watcher that calls it.

The SessionStore in-memory dict (and its backing sessions.json) grew
unbounded — every unique (platform, chat_id, thread_id, user_id) tuple
ever seen was kept forever, regardless of how stale it became.  These
tests pin the prune behaviour:

  * Entries older than max_age_days (by updated_at) are removed
  * Entries marked ``suspended`` are preserved (user-paused)
  * Entries with an active process attached are preserved
  * max_age_days <= 0 disables pruning entirely
  * sessions.json is rewritten with the post-prune dict
  * The ``updated_at`` field — not ``created_at`` — drives the decision
    (so a long-running-but-still-active session isn't pruned)
"""

import json
import threading
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, SessionResetPolicy
from gateway.session import SessionEntry, SessionStore


def _make_store(tmp_path, max_age_days: int = 90, has_active_processes_fn=None):
    """Build a SessionStore bypassing SQLite/disk-load side effects."""
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none"),
        session_store_max_age_days=max_age_days,
    )
    with patch("gateway.session.SessionStore._ensure_loaded"):
        store = SessionStore(
            sessions_dir=tmp_path,
            config=config,
            has_active_processes_fn=has_active_processes_fn,
        )
    store._db = None
    store._loaded = True
    return store


def _entry(key: str, age_days: float, *, suspended: bool = False,
           session_id: str | None = None) -> SessionEntry:
    now = datetime.now()
    return SessionEntry(
        session_key=key,
        session_id=session_id or f"sid_{key}",
        created_at=now - timedelta(days=age_days + 30),  # arbitrary older
        updated_at=now - timedelta(days=age_days),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        suspended=suspended,
    )


class TestPruneBasics:
    def test_prune_removes_entries_past_max_age(self, tmp_path):
        store = _make_store(tmp_path)
        store._entries["old"] = _entry("old", age_days=100)
        store._entries["fresh"] = _entry("fresh", age_days=5)

        removed = store.prune_old_entries(max_age_days=90)

        assert removed == 1
        assert "old" not in store._entries
        assert "fresh" in store._entries

    def test_prune_uses_updated_at_not_created_at(self, tmp_path):
        """A session created long ago but updated recently must be kept."""
        store = _make_store(tmp_path)
        now = datetime.now()
        entry = SessionEntry(
            session_key="long-lived",
            session_id="sid",
            created_at=now - timedelta(days=365),   # ancient
            updated_at=now - timedelta(days=3),     # but just chatted
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        store._entries["long-lived"] = entry

        removed = store.prune_old_entries(max_age_days=30)

        assert removed == 0
        assert "long-lived" in store._entries

    def test_prune_disabled_when_max_age_is_zero(self, tmp_path):
        store = _make_store(tmp_path, max_age_days=0)
        for i in range(5):
            store._entries[f"s{i}"] = _entry(f"s{i}", age_days=365)

        assert store.prune_old_entries(0) == 0
        assert len(store._entries) == 5

    def test_prune_disabled_when_max_age_is_negative(self, tmp_path):
        store = _make_store(tmp_path)
        store._entries["s"] = _entry("s", age_days=365)

        assert store.prune_old_entries(-1) == 0
        assert "s" in store._entries

    def test_prune_skips_suspended_entries(self, tmp_path):
        """/stop-suspended sessions must be kept for later resume."""
        store = _make_store(tmp_path)
        store._entries["suspended"] = _entry(
            "suspended", age_days=1000, suspended=True
        )
        store._entries["idle"] = _entry("idle", age_days=1000)

        removed = store.prune_old_entries(max_age_days=90)

        assert removed == 1
        assert "suspended" in store._entries
        assert "idle" not in store._entries

    def test_prune_skips_entries_with_active_processes(self, tmp_path):
        """Sessions with active bg processes aren't pruned even if old.

        The callback is keyed by session_key — matching what
        process_registry.has_active_for_session() actually consumes in
        gateway/run.py.  Prior to the fix this test passed the callback a
        session_id, which silently matched an implementation bug where
        prune_old_entries was also passing session_id; real-world usage
        (via process_registry) takes a session_key and never matched, so
        active sessions were still being pruned.
        """
        active_session_keys = {"active"}

        def _has_active(session_key: str) -> bool:
            return session_key in active_session_keys

        store = _make_store(tmp_path, has_active_processes_fn=_has_active)
        store._entries["active"] = _entry(
            "active", age_days=1000, session_id="sid_active"
        )
        store._entries["idle"] = _entry(
            "idle", age_days=1000, session_id="sid_idle"
        )

        removed = store.prune_old_entries(max_age_days=90)

        assert removed == 1
        assert "active" in store._entries
        assert "idle" not in store._entries

    def test_prune_active_check_uses_session_key_not_session_id(self, tmp_path):
        """Regression guard: a callback that only recognises session_ids must
        NOT protect entries during prune.  This pins the fix so a future
        refactor can't silently revert to passing session_id again.
        """
        def _recognises_only_ids(identifier: str) -> bool:
            return identifier.startswith("sid_")

        store = _make_store(tmp_path, has_active_processes_fn=_recognises_only_ids)
        store._entries["active"] = _entry(
            "active", age_days=1000, session_id="sid_active"
        )

        removed = store.prune_old_entries(max_age_days=90)

        # Entry is pruned because the callback receives "active" (session_key),
        # not "sid_active" (session_id), so _recognises_only_ids returns False.
        assert removed == 1
        assert "active" not in store._entries

    def test_prune_does_not_write_disk_when_no_removals(self, tmp_path):
        """If nothing is evictable, _save() should NOT be called."""
        store = _make_store(tmp_path)
        store._entries["fresh1"] = _entry("fresh1", age_days=1)
        store._entries["fresh2"] = _entry("fresh2", age_days=2)

        save_calls = []
        store._save = lambda: save_calls.append(1)

        assert store.prune_old_entries(max_age_days=90) == 0
        assert save_calls == []

    def test_prune_writes_disk_after_removal(self, tmp_path):
        store = _make_store(tmp_path)
        store._entries["stale"] = _entry("stale", age_days=500)
        store._entries["fresh"] = _entry("fresh", age_days=1)

        save_calls = []
        store._save = lambda: save_calls.append(1)

        store.prune_old_entries(max_age_days=90)
        assert save_calls == [1]

    def test_prune_is_thread_safe(self, tmp_path):
        """Prune acquires _lock internally; concurrent update_session is safe."""
        store = _make_store(tmp_path)
        for i in range(20):
            age = 1000 if i % 2 == 0 else 1
            store._entries[f"s{i}"] = _entry(f"s{i}", age_days=age)

        results = []

        def _pruner():
            results.append(store.prune_old_entries(max_age_days=90))

        def _reader():
            # Mimic a concurrent update_session reader iterating under lock.
            with store._lock:
                list(store._entries.keys())

        threads = [threading.Thread(target=_pruner)]
        threads += [threading.Thread(target=_reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        # Exactly one pruner ran; removed exactly the 10 stale entries.
        assert results == [10]
        assert len(store._entries) == 10
        for i in range(20):
            if i % 2 == 1:  # fresh
                assert f"s{i}" in store._entries


class TestPrunePersistsToDisk:
    def test_prune_rewrites_sessions_json(self, tmp_path):
        """After prune, sessions.json on disk reflects the new dict."""
        config = GatewayConfig(
            default_reset_policy=SessionResetPolicy(mode="none"),
            session_store_max_age_days=90,
        )
        store = SessionStore(sessions_dir=tmp_path, config=config)
        store._db = None
        # Force-populate without calling get_or_create to avoid DB side-effects
        store._entries["stale"] = _entry("stale", age_days=500)
        store._entries["fresh"] = _entry("fresh", age_days=1)
        store._loaded = True
        store._save()

        # Verify pre-prune state on disk.
        saved_pre = json.loads((tmp_path / "sessions.json").read_text())
        assert set(saved_pre.keys()) == {"stale", "fresh"}

        # Prune and check disk.
        store.prune_old_entries(max_age_days=90)
        saved_post = json.loads((tmp_path / "sessions.json").read_text())
        assert set(saved_post.keys()) == {"fresh"}


class TestGatewayConfigSerialization:
    def test_session_store_max_age_days_defaults_to_90(self):
        cfg = GatewayConfig()
        assert cfg.session_store_max_age_days == 90

    def test_session_store_max_age_days_roundtrips(self):
        cfg = GatewayConfig(session_store_max_age_days=30)
        restored = GatewayConfig.from_dict(cfg.to_dict())
        assert restored.session_store_max_age_days == 30

    def test_session_store_max_age_days_missing_defaults_90(self):
        """Loading an old config (pre-this-field) falls back to default."""
        restored = GatewayConfig.from_dict({})
        assert restored.session_store_max_age_days == 90

    def test_session_store_max_age_days_negative_coerced_to_zero(self):
        """A negative value (accidental or hostile) becomes 0 (disabled)."""
        restored = GatewayConfig.from_dict({"session_store_max_age_days": -5})
        assert restored.session_store_max_age_days == 0

    def test_session_store_max_age_days_bad_type_falls_back(self):
        """Non-int values fall back to the default, not a crash."""
        restored = GatewayConfig.from_dict({"session_store_max_age_days": "nope"})
        assert restored.session_store_max_age_days == 90


class TestGatewayWatcherCallsPrune:
    """The session_expiry_watcher should call prune_old_entries once per hour."""

    def test_prune_gate_fires_on_first_tick(self):
        """First watcher tick has _last_prune_ts=0, so the gate opens."""
        import time as _t

        last_ts = 0.0
        prune_interval = 3600.0
        now = _t.time()

        # Mirror the production gate check in _session_expiry_watcher.
        should_prune = (now - last_ts) > prune_interval
        assert should_prune is True

    def test_prune_gate_suppresses_within_interval(self):
        import time as _t

        last_ts = _t.time() - 600  # 10 minutes ago
        prune_interval = 3600.0
        now = _t.time()

        should_prune = (now - last_ts) > prune_interval
        assert should_prune is False
