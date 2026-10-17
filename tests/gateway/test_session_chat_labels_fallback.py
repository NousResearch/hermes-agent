"""Label fallback publication and replay preserve the newest observation (#140259)."""

import json

import pytest

import gateway.session_persistence as persistence
from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore


@pytest.mark.parametrize("fault", ["missing-db", "write", "partial-replay", "cleanup", "disk"])
def test_label_journal_retries_without_losing_or_resurrecting_observations(tmp_path, monkeypatch, fault):
    config = GatewayConfig(write_sessions_json=False)
    sessions = tmp_path / "sessions"
    store = SessionStore(sessions, config)
    source = SessionSource(platform=Platform.DISCORD, scope_id="g1", chat_id="ch1",
                           chat_name="Before", chat_topic="Before topic")
    assert store.record_chat_labels(source)
    db = store._routing_db
    source.chat_name, source.chat_topic = "Observed offline", "Offline topic"

    with monkeypatch.context() as outage:
        if fault == "write":
            def fail_write(*args, **kwargs):
                raise OSError("SQLite write unavailable")

            outage.setattr(db, "set_meta", fail_write)
        else:
            store._db = None
        if fault == "disk":
            with monkeypatch.context() as full_disk:
                def fail_publish(*args, **kwargs):
                    raise OSError("disk full")

                full_disk.setattr(persistence, "atomic_json_write", fail_publish)
                with pytest.raises(OSError, match="disk full"):
                    store.record_chat_labels(source)
        assert store.record_chat_labels(source)
    store._db = None
    # Same channel ID, different guild/platform: clearing a peer must not clear this chat.
    peer = SessionSource(platform=Platform.DISCORD, scope_id="g2", chat_id="ch1")
    other = SessionSource(platform=Platform.SLACK, scope_id="g1", chat_id="ch1",
                          chat_name="Separate platform", chat_topic="Separate topic")
    assert store.record_chat_labels(peer)
    assert store.record_chat_labels(other)
    assert store.chat_labels(Platform.DISCORD, "g2", "ch1") == (None, None)
    with pytest.raises(LookupError):
        store.chat_labels(Platform.DISCORD, "unknown", "ch1")
    store.close_all_db_handles()

    store = SessionStore(sessions, config)
    db = store._routing_db
    with monkeypatch.context() as recovering:
        if fault == "missing-db":
            store._db = None
        elif fault == "partial-replay":
            real_set = db.set_meta

            def partially_commit(key, value):
                if key == store._chat_labels_meta_key(Platform.DISCORD, "g2", "ch1"):
                    raise OSError("second replay write failed")
                real_set(key, value)

            recovering.setattr(db, "set_meta", partially_commit)
        elif fault == "cleanup":
            publish = persistence.atomic_json_write

            def fail_cleanup(path, data, **kwargs):
                if not data:
                    raise OSError("journal cleanup failed")
                publish(path, data, **kwargs)

            recovering.setattr(persistence, "atomic_json_write", fail_cleanup)
        assert store.chat_labels(Platform.DISCORD, "g1", "ch1") == (source.chat_name, source.chat_topic)
        # A newer observation must reach the journal before SQLite, even if cleanup keeps failing.
        source.chat_name, source.chat_topic = "Newest", None
        assert store.record_chat_labels(source)
    store.close_all_db_handles()

    store = SessionStore(sessions, config)
    expected = {(Platform.DISCORD, "g1"): ("Newest", None),
                (Platform.DISCORD, "g2"): (None, None),
                (Platform.SLACK, "g1"): (other.chat_name, other.chat_topic)}
    for (platform, guild), labels in expected.items():
        assert store.chat_labels(platform, guild, "ch1") == labels
        key = store._chat_labels_meta_key(platform, guild, "ch1")
        assert json.loads(store._routing_db.get_meta(key)) == list(labels)
    # Once replay commits, an old fallback copy must not shadow a later SQLite update.
    source.chat_name = "Later database observation"
    assert store.record_chat_labels(source)
    store.close_all_db_handles()
    store = SessionStore(sessions, config)
    assert store.chat_labels(Platform.DISCORD, "g1", "ch1") == (source.chat_name, None)
    store.close_all_db_handles()
