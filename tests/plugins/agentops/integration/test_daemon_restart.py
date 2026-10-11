from __future__ import annotations

from plugins.agentops.control.config import load_agentops_config
from plugins.agentops.control.daemon import start_daemon_thread
from plugins.agentops.control.events import EventSpool
from plugins.agentops.control.models import AuditEvent
from plugins.agentops.control.store import open_store


def test_restart_replays_spool_once_and_keeps_observe_only(tmp_path, make_event, write_config):
    config_path = write_config()
    config = load_agentops_config(config_path)
    EventSpool(config.spool_dir, max_bytes=config.event_spool_max_bytes).write(make_event())

    first = start_daemon_thread(config_path)
    first.stop()
    second = start_daemon_thread(config_path)
    try:
        health = second.health()
        assert health["event_count"] == 1
        assert health["authority_mode"] == "observe_only"
        assert health["global_write_enabled"] is False
    finally:
        second.stop()


def test_migration_failure_serves_safe_observe_only_health(tmp_path, monkeypatch, write_config):
    import plugins.agentops.control.daemon as daemon
    from plugins.agentops.control.store import StoreMigrationError

    config_path = write_config()

    def fail_open_store(_config):
        raise StoreMigrationError("synthetic migration failure")

    monkeypatch.setattr(daemon, "open_store", fail_open_store)
    handle = daemon.start_daemon_thread(config_path)
    try:
        health = handle.health()
        assert health["authority_mode"] == "observe_only"
        assert health["store_available"] is False
        assert "store_migration_failed" in health["safe_start_reasons"]
    finally:
        handle.stop()


def test_invalid_audit_chain_serves_safe_observe_only_health(tmp_path, write_config):
    config_path = write_config()
    config = load_agentops_config(config_path)
    store = open_store(config)
    store.append_audit(
        AuditEvent.create(
            actor_type="system",
            actor_id="agentopsd",
            action="test.audit",
            object_type="test",
            object_id="audit-1",
            timestamp="2026-08-09T12:00:00+00:00",
            metadata={"source": "test"},
        )
    )
    store._connection.execute("UPDATE audit_events SET entry_hash = 'tampered' WHERE sequence = 1")
    store._connection.commit()
    store.close()

    handle = start_daemon_thread(config_path)
    try:
        health = handle.health()
        assert health["authority_mode"] == "observe_only"
        assert health["audit_chain_valid"] is False
        assert "audit_chain_invalid" in health["safe_start_reasons"]
    finally:
        handle.stop()
