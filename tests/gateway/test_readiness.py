from __future__ import annotations

import json
import os
import asyncio
import threading
from pathlib import Path

import gateway.readiness as readiness_mod
from gateway.readiness import collect_runtime_readiness


def test_collect_runtime_readiness_reports_healthy_local_runtime(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n  provider: openrouter\n  model: test/model\n",
        encoding="utf-8",
    )
    from hermes_state import SessionDB

    db = SessionDB.for_home(home)
    db.close()
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


def test_collect_runtime_readiness_marks_corrupt_state_db_degraded(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("{}\n", encoding="utf-8")
    (home / "state.db").write_bytes(b"not sqlite")
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = collect_runtime_readiness(configured_model="configured-model", runtime_status={})

    assert result["status"] == "degraded"
    assert result["checks"]["state_db"]["status"] == "degraded"
    assert "detail" in result["checks"]["state_db"]


def test_collect_runtime_readiness_never_exposes_config_values(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    secret = "do-not-return-this-value"
    (home / "config.yaml").write_text(
        f"model:\n  provider: openrouter\nprivate_value: {secret}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    result = collect_runtime_readiness(configured_model="model", runtime_status={})

    assert secret not in json.dumps(result)
    assert str(home) not in json.dumps(result)
    assert result["checks"]["config"]["status"] == "ok"


def test_collect_runtime_readiness_uses_active_profile_home(tmp_path, monkeypatch):
    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    result = collect_runtime_readiness(configured_model="model", runtime_status={})

    assert result["checks"]["config"]["status"] == "ok"
    assert not (tmp_path / ".hermes" / "state.db").exists()
    assert os.environ["HERMES_HOME"] == str(profile_home)


def test_postgres_readiness_reports_capabilities_without_dsn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    raw_dsn = "postgresql://user:secret@db.example/hermes"
    (home / "config.yaml").write_text(
        """
sessions:
  state:
    backend: postgres
    postgres:
      dsn_env: HERMES_STATE_POSTGRES_DSN
      schema: hermes_state
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_POSTGRES_DSN", raw_dsn)

    class FakePostgresDB:
        capabilities = {"core_schema": True, "full_text_search": True}

        def list_gateway_sessions(self, *, active_only):
            raise AssertionError("readiness must not materialize gateway sessions")

        def close(self):
            pass

    from hermes_state import SessionDB

    monkeypatch.setattr(
        SessionDB,
        "for_home",
        staticmethod(lambda *_args, **_kwargs: FakePostgresDB()),
    )

    result = readiness_mod._probe_state_db(home)

    assert result["status"] == "ok"
    assert result["backend"] == "postgres"
    assert result["schema"] == "hermes_state"
    assert result["capabilities"]["core_schema"] is True
    assert raw_dsn not in json.dumps(result)


def test_readiness_defers_state_probe_while_an_event_loop_is_running(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    started = threading.Event()
    release = threading.Event()

    def fake_probe(probe_home):
        assert probe_home == home
        started.set()
        release.wait(timeout=1)
        return {"status": "ok", "backend": "sqlite"}

    monkeypatch.setattr(readiness_mod, "_probe_state_db", fake_probe)
    monkeypatch.setattr(readiness_mod, "_STATE_PROBE_HOME", None)
    monkeypatch.setattr(readiness_mod, "_STATE_PROBE_RESULT", None)
    monkeypatch.setattr(readiness_mod, "_STATE_PROBE_FUTURE", None)

    async def collect():
        return readiness_mod._probe_state_db_without_blocking_event_loop(home)

    result = asyncio.run(collect())

    assert result["detail"] == "probe pending"
    assert started.wait(timeout=1)
    release.set()
