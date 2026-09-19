from __future__ import annotations

import json

import pytest

from plugins.agentops.control.events import EventSpool, EventValidationError
from plugins.agentops.control.store import open_store


def test_event_hash_is_stable_when_mapping_order_changes(make_event):
    first = make_event(payload={"b": 2, "a": 1})
    second = make_event(payload={"a": 1, "b": 2})

    assert first.content_hash == second.content_hash


def test_secret_payload_is_rejected_before_spooling_or_storage(tmp_path, make_event):
    with pytest.raises(EventValidationError):
        make_event(payload={"token": "sk-test-canary-secret"})

    spool = EventSpool(tmp_path / "spool")
    assert not spool.pending_paths()


def test_spool_replay_is_idempotent_and_deletes_committed_file(make_event, write_config):
    from plugins.agentops.control.config import load_agentops_config

    config = load_agentops_config(write_config())
    spool = EventSpool(config.spool_dir)
    store = open_store(config)
    event = make_event()
    spool.write(event)

    first = spool.replay(store)
    second = spool.replay(store)

    assert first.appended == 1
    assert first.duplicates == 0
    assert second.appended == 0
    assert store.event_count() == 1
    assert not spool.pending_paths()


def test_unknown_schema_is_quarantined_without_entering_store(write_config):
    from plugins.agentops.control.config import load_agentops_config

    config = load_agentops_config(write_config())
    spool = EventSpool(config.spool_dir)
    store = open_store(config)
    spool.root.mkdir(parents=True)
    (spool.root / "unknown.json").write_text(
        json.dumps({"schema_version": 99, "event_id": "evt-unknown"}),
        encoding="utf-8",
    )

    result = spool.replay(store)

    assert result.quarantined == 1
    assert store.event_count() == 0
    assert (spool.quarantine_dir / "unknown.json").exists()


def test_secret_in_corrupt_spool_is_replaced_by_redacted_quarantine_metadata(write_config):
    from plugins.agentops.control.config import load_agentops_config

    config = load_agentops_config(write_config())
    spool = EventSpool(config.spool_dir)
    store = open_store(config)
    spool.root.mkdir(parents=True)
    (spool.root / "unsafe.json").write_text(
        '{"token":"sk-test-canary-secret","schema_version":999}',
        encoding="utf-8",
    )

    result = spool.replay(store)

    assert result.quarantined == 1
    content = (spool.quarantine_dir / "unsafe.json").read_text(encoding="utf-8")
    assert "sk-test-canary-secret" not in content
    assert '"redacted":true' in content
