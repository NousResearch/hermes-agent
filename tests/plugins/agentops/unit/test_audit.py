from __future__ import annotations

import sqlite3

import pytest

from plugins.agentops.control.audit import AuditValidationError
from plugins.agentops.control.models import AuditEvent
from plugins.agentops.control.store import open_store


def make_audit(action: str = "event.append"):
    return AuditEvent.create(
        actor_type="system",
        actor_id="agentopsd",
        action=action,
        object_type="event",
        object_id="evt-0001",
        timestamp="2026-08-09T12:00:00+00:00",
        metadata={"source": "test"},
    )


def test_audit_chain_verifies_and_detects_tampering(write_config):
    from plugins.agentops.control.config import load_agentops_config

    store = open_store(load_agentops_config(write_config()))
    store.append_audit(make_audit("event.append"))
    store.append_audit(make_audit("event.replayed"))

    assert store.verify_audit_chain() is True

    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE audit_events SET entry_hash = 'tampered' WHERE sequence = 2")
        conn.commit()

    assert store.verify_audit_chain() is False


def test_secret_audit_metadata_is_rejected_before_persistence(write_config):
    from plugins.agentops.control.config import load_agentops_config

    store = open_store(load_agentops_config(write_config()))
    with pytest.raises(AuditValidationError):
        store.append_audit(
            AuditEvent.create(
                actor_type="system",
                actor_id="agentopsd",
                action="event.append",
                object_type="event",
                object_id="evt-0001",
                timestamp="2026-08-09T12:00:00+00:00",
                metadata={"api_key": "sk-test-canary-secret"},
            )
        )

    assert store.audit_count() == 0
