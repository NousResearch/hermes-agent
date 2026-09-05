from __future__ import annotations

import importlib.util
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _load_watchdog():
    path = Path(__file__).parents[2] / "ops" / "coder-fallback-watch.py"
    spec = importlib.util.spec_from_file_location("coder_fallback_watch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _db(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE tasks(id TEXT PRIMARY KEY, assignee TEXT, status TEXT, title TEXT, body TEXT);"
            "CREATE TABLE task_runs(id INTEGER PRIMARY KEY, task_id TEXT, profile TEXT, outcome TEXT, ended_at INTEGER);"
        )


def test_watchdog_does_not_restore_primary_before_provider_declared_reset(tmp_path, monkeypatch):
    from agent.provider_health import ProviderHealthStore, ProviderRoute

    watchdog = _load_watchdog()
    db_path = tmp_path / "kanban.db"
    state_path = tmp_path / "state.json"
    coder_home = tmp_path / "coder"
    _db(db_path)
    activated = datetime(2026, 9, 5, tzinfo=timezone.utc)
    reset = datetime(2026, 9, 11, tzinfo=timezone.utc)
    state_path.write_text(json.dumps({
        "mode": "fallback",
        "fallback_until": int(activated.timestamp()) + 3600,
        "primary": {"provider": "zai", "model": "glm-5.3-flash"},
    }))
    ProviderHealthStore(coder_home).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        f"Weekly Limit Exhausted; reset_at={reset.isoformat()}",
        source="agent:worker",
        now=activated,
    )
    restored = []
    monkeypatch.setattr(watchdog, "_restore_primary", lambda primary: restored.append(primary))
    monkeypatch.setenv("CODER_FALLBACK_DB", str(db_path))
    monkeypatch.setenv("CODER_FALLBACK_STATE", str(state_path))
    monkeypatch.setenv("CODER_FALLBACK_PROFILE_HOME", str(coder_home))
    monkeypatch.setenv("CODER_FALLBACK_NOW", str(int(activated.timestamp()) + 7200))

    assert watchdog.main() == 0
    assert restored == []
    persisted = json.loads(state_path.read_text())
    assert persisted["mode"] == "fallback"
    assert persisted["fallback_until"] == int(reset.timestamp())


def test_watchdog_restores_primary_once_when_it_owns_post_reset_probe(tmp_path, monkeypatch):
    from agent.provider_health import ProviderHealthStore, ProviderRoute

    watchdog = _load_watchdog()
    db_path = tmp_path / "kanban.db"
    state_path = tmp_path / "state.json"
    coder_home = tmp_path / "coder"
    _db(db_path)
    activated = datetime(2026, 9, 5, tzinfo=timezone.utc)
    reset = datetime(2026, 9, 11, tzinfo=timezone.utc)
    state_path.write_text(json.dumps({
        "mode": "fallback",
        "fallback_until": int(activated.timestamp()) + 3600,
        "primary": {"provider": "zai", "model": "glm-5.3-flash"},
    }))
    ProviderHealthStore(coder_home).record_failure(
        ProviderRoute("zai", "glm-5.3-flash"),
        f"Weekly Limit Exhausted; reset_at={reset.isoformat()}",
        source="agent:worker",
        now=activated,
    )
    restored = []
    monkeypatch.setattr(watchdog, "_restore_primary", lambda primary: restored.append(primary))
    monkeypatch.setattr(watchdog, "_retry_auto_blocked", lambda conn: [])
    monkeypatch.setenv("CODER_FALLBACK_DB", str(db_path))
    monkeypatch.setenv("CODER_FALLBACK_STATE", str(state_path))
    monkeypatch.setenv("CODER_FALLBACK_PROFILE_HOME", str(coder_home))
    monkeypatch.setenv("CODER_FALLBACK_NOW", str(int(reset.timestamp())))

    assert watchdog.main() == 0
    assert restored == [{"provider": "zai", "model": "glm-5.3-flash"}]
    assert json.loads(state_path.read_text())["mode"] == "primary"
    health = ProviderHealthStore(coder_home).get(ProviderRoute("zai", "glm-5.3-flash"))
    assert health is not None
    assert health.probe_owner is None
