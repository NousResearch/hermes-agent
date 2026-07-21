"""Gateway dispatcher behavior for persisted Kanban DB health circuits."""

import asyncio
import logging

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin
import hermes_cli.config as config_module
import hermes_cli.kanban_db as kb


def _configure_dispatcher(monkeypatch):
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda: {
            "kanban": {
                "dispatch_in_gateway": True,
                "dispatch_interval_seconds": 1,
            }
        },
    )
    monkeypatch.setattr(
        kb,
        "list_boards",
        lambda include_archived=False: [{"slug": kb.DEFAULT_BOARD}],
    )
    monkeypatch.setattr(kb, "read_board_metadata", lambda slug: {"slug": slug})
    monkeypatch.setattr(kb, "reap_worker_zombies", lambda *args, **kwargs: 0)


def test_dispatcher_open_circuit_never_reopens_sqlite(monkeypatch, caplog):
    """One persisted incident pauses all dispatcher and ready-probe opens."""
    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    health = {
        "incident_id": "kbd-test-incident",
        "classification": "corruption",
        "manifest_path": "/tmp/kanban.incident.json",
    }
    _configure_dispatcher(monkeypatch)
    monkeypatch.setattr(kb, "get_db_health", lambda board=None: health)
    monkeypatch.setattr(
        kb,
        "connect",
        lambda *args, **kwargs: pytest.fail("open circuit attempted SQLite reopen"),
    )

    calls = {"to_thread": 0}

    async def fake_to_thread(fn, *args, **kwargs):
        calls["to_thread"] += 1
        result = fn(*args, **kwargs)
        if calls["to_thread"] >= 6:
            runner._running = False
        return result

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", fake_sleep)

    with caplog.at_level(logging.ERROR, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0)
        )

    messages = [record.getMessage() for record in caplog.records]
    assert sum("kbd-test-incident" in message for message in messages) == 1
    assert not any("tick failed on board" in message for message in messages)
    assert not any(record.exc_info for record in caplog.records)


def test_dispatcher_resumes_after_circuit_clears(monkeypatch, caplog):
    """Replacing the DB clears health state and resumes without a restart."""
    runner = GatewayKanbanWatchersMixin()
    runner._running = True
    health = {
        "incident_id": "kbd-test-incident",
        "classification": "fatal_storage",
        "manifest_path": "/tmp/kanban.incident.json",
    }
    health_reads = iter([health, health, None, None, None, None])
    _configure_dispatcher(monkeypatch)
    monkeypatch.setattr(
        kb,
        "get_db_health",
        lambda board=None: next(health_reads, None),
    )

    class Connection:
        def close(self):
            return None

    calls = {"connect": 0, "dispatch": 0, "to_thread": 0}

    def connect(*args, **kwargs):
        calls["connect"] += 1
        return Connection()

    def dispatch_once(*args, **kwargs):
        calls["dispatch"] += 1
        return None

    async def fake_to_thread(fn, *args, **kwargs):
        calls["to_thread"] += 1
        result = fn(*args, **kwargs)
        if calls["to_thread"] >= 6:
            runner._running = False
        return result

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(kb, "connect", connect)
    monkeypatch.setattr(kb, "dispatch_once", dispatch_once)
    monkeypatch.setattr(kb, "has_spawnable_ready", lambda conn: False)
    monkeypatch.setattr(kb, "has_spawnable_review", lambda conn: False)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", fake_sleep)

    with caplog.at_level(logging.INFO, logger="gateway.run"):
        asyncio.run(
            asyncio.wait_for(runner._kanban_dispatcher_watcher(), timeout=3.0)
        )

    messages = [record.getMessage() for record in caplog.records]
    assert calls["dispatch"] == 1
    assert calls["connect"] >= 1
    assert any("database health restored; resuming" in message for message in messages)
