from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from gateway import readiness
from gateway.readiness import collect_runtime_readiness


def test_probe_state_db_closes_sqlite_connection(tmp_path, monkeypatch):
    """The state-db readiness probe must release its SQLite descriptors."""
    home = tmp_path / ".hermes"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")

    real_connect = sqlite3.connect
    opened = []

    class CloseTrackingConnection:
        def __init__(self, connection):
            self._connection = connection
            self.closed = False

        def execute(self, *args, **kwargs):
            return self._connection.execute(*args, **kwargs)

        def close(self):
            self.closed = True
            self._connection.close()

        # sqlite3's context manager commits or rolls back but does not close.
        # Preserve that behavior so this fails if the probe stops using
        # contextlib.closing().
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    def connect(*args, **kwargs):
        connection = CloseTrackingConnection(real_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(readiness.sqlite3, "connect", connect)

    for _ in range(25):
        assert readiness._probe_state_db(home)["status"] == "ok"

    assert len(opened) == 25
    assert all(connection.closed for connection in opened)


def test_collect_runtime_readiness_reports_healthy_local_runtime(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  model: test/model\n",
        encoding="utf-8",
    )
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        readiness.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100, used=10, free=90),
    )

    result = collect_runtime_readiness(
        configured_model="test/model",
        runtime_status={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "updated_at": "2026-07-09T00:00:00Z",
        },
        active_api_runs=2,
    )

    assert result["status"] == "ok"
    assert result["checks"]["state_db"]["status"] == "ok"
    assert result["checks"]["config"]["status"] == "ok"
    assert result["checks"]["model"]["status"] == "ok"
    assert result["checks"]["gateway"]["status"] == "ok"
    assert result["checks"]["background_queues"]["active_api_runs"] == 2
    assert result["checks"]["disk"]["status"] in {"ok", "degraded"}


def test_collect_runtime_readiness_degrades_on_invalid_config_and_stopped_gateway(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model: [unterminated", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = collect_runtime_readiness(
        configured_model="",
        runtime_status={"gateway_state": "stopped", "platforms": {}},
    )

    assert result["status"] == "degraded"
    assert result["checks"]["config"]["status"] == "degraded"
    assert result["checks"]["model"]["status"] == "degraded"
    assert result["checks"]["gateway"]["status"] == "degraded"
    # Readiness is diagnostic data, not an exception or a destructive repair.
    assert (home / "config.yaml").read_text(encoding="utf-8") == "model: [unterminated"
