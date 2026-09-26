from __future__ import annotations

import pytest

from plugins.agentops.control.store import StoreIntegrityError, inspect_store, open_store


def test_store_uses_wal_and_event_idempotency(make_event, write_config):
    from plugins.agentops.control.config import load_agentops_config

    store = open_store(load_agentops_config(write_config()))
    event = make_event()

    first = store.append_event(event)
    second = store.append_event(event)

    assert first.inserted is True
    assert second.inserted is False
    assert store.journal_mode() == "wal"
    assert store.event_count() == 1


def test_backup_restore_returns_to_verified_snapshot(make_event, write_config):
    from plugins.agentops.control.config import load_agentops_config

    store = open_store(load_agentops_config(write_config()))
    store.append_event(make_event("evt-a"))
    backup = store.backup_to()
    store.append_event(make_event("evt-b"))

    store.restore_from(backup)

    assert store.event_count() == 1
    assert store.verify_audit_chain() is True


def test_same_event_id_with_different_content_is_not_treated_as_a_duplicate(make_event, write_config):
    from plugins.agentops.control.config import load_agentops_config

    store = open_store(load_agentops_config(write_config()))
    store.append_event(make_event("evt-collision", payload={"status": "first"}))

    with pytest.raises(StoreIntegrityError):
        store.append_event(make_event("evt-collision", payload={"status": "second"}))

    assert store.event_count() == 1


def test_read_only_inspection_does_not_create_missing_database(tmp_path):
    state_db = tmp_path / "missing.db"

    result = inspect_store(state_db)

    assert result.exists is False
    assert result.schema_version is None
    assert not state_db.exists()
