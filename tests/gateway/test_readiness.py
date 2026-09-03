from __future__ import annotations

import json
import os
import sqlite3
from collections import namedtuple
from pathlib import Path

import pytest

from gateway.readiness import collect_runtime_readiness


@pytest.mark.parametrize("used, expected_status", [(50, "ok"), (95, "degraded")])
def test_collect_runtime_readiness_reports_disk_health(
    tmp_path, monkeypatch, used, expected_status
):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "gateway.readiness.shutil.disk_usage",
        lambda _path: usage(100, used, 100 - used),
    )
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  model: test/model\n",
        encoding="utf-8",
    )
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = collect_runtime_readiness(
        configured_model="test/model",
        runtime_status={
            "gateway_state": "running",
            "platforms": {"telegram": {"state": "connected"}},
            "updated_at": "2026-07-09T00:00:00Z",
        },
        active_api_runs=2,
    )

    assert result["status"] == expected_status
    assert result["checks"]["state_db"]["status"] == "ok"
    assert result["checks"]["session_store"]["status"] == "ok"
    assert result["checks"]["config"]["status"] == "ok"
    assert result["checks"]["model"]["status"] == "ok"
    assert result["checks"]["gateway"]["status"] == "ok"
    assert result["checks"]["background_queues"]["active_api_runs"] == 2
    assert result["checks"]["disk"]["status"] == expected_status


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


def test_readiness_uses_running_session_store_state_over_independent_probe(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    with sqlite3.connect(home / "state.db") as conn:
        conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
    monkeypatch.setenv("HERMES_HOME", str(home))

    unavailable = collect_runtime_readiness(
        configured_model="test/model",
        runtime_status={
            "gateway_state": "running",
            "platforms": {},
            "session_store": {"status": "unavailable"},
        },
    )

    assert unavailable["checks"]["state_db"]["status"] == "ok"
    assert unavailable["checks"]["session_store"] == {"status": "unavailable"}
    assert unavailable["status"] == "degraded"

    recovered = collect_runtime_readiness(
        configured_model="test/model",
        runtime_status={
            "gateway_state": "running",
            "platforms": {},
            "session_store": {"status": "ok"},
        },
    )
    assert recovered["checks"]["session_store"] == {"status": "ok"}


