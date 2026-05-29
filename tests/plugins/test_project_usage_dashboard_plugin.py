"""Tests for the Project Usage dashboard plugin and ledger backfill."""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban_db as kb
from hermes_cli import project_usage_ledger as ledger


SESSION_COLUMNS_SQL = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT
);
"""


def _load_plugin_router():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_file = repo_root / "plugins" / "project_usage" / "dashboard" / "plugin_api.py"
    assert plugin_file.exists(), f"plugin file missing: {plugin_file}"
    spec = importlib.util.spec_from_file_location(
        "hermes_dashboard_plugin_project_usage_test", plugin_file,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.router


@pytest.fixture
def usage_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Avoid module-level kanban init cache carrying paths across tests.
    kb._INITIALIZED_PATHS.clear()
    yield home
    kb._INITIALIZED_PATHS.clear()


def _seed_state_db(home: Path):
    conn = sqlite3.connect(home / "state.db")
    conn.execute(SESSION_COLUMNS_SQL)
    conn.execute(
        """
        INSERT INTO sessions (
            id, source, user_id, model, started_at, ended_at, title,
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
            reasoning_tokens, billing_provider, billing_mode,
            estimated_cost_usd, actual_cost_usd, cost_status,
            api_call_count, tool_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sess-worker-1", "kanban", "u1", "test/model", 1000.0, 1060.0,
            "Worker session", 120, 45, 10, 5, 7, "openrouter", "estimated",
            0.0123, None, "estimated", 2, 3,
        ),
    )
    conn.execute(
        """
        INSERT INTO sessions (
            id, source, user_id, model, started_at, ended_at, title,
            input_tokens, output_tokens, estimated_cost_usd, api_call_count, tool_call_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sess-unassigned", "cli", "u1", "test/model", 900.0, 930.0,
            "Loose session", 10, 4, 0.001, 1, 0,
        ),
    )
    conn.commit()
    conn.close()


def _seed_board():
    kb.create_board("proj", name="Project One")
    conn = kb.connect(board="proj")
    try:
        task_id = kb.create_task(conn, title="Implement thing", assignee="default")
        conn.execute(
            """
            INSERT INTO task_runs (
                task_id, profile, status, started_at, ended_at, outcome, metadata
            ) VALUES (?, 'default', 'done', 1000, 1060, 'completed', ?)
            """,
            (task_id, '{"worker_session_id":"sess-worker-1"}'),
        )
        return task_id
    finally:
        conn.close()


def test_backfill_creates_ledger_and_correlates_board_task_usage(usage_home):
    _seed_state_db(usage_home)
    task_id = _seed_board()

    summary = ledger.get_summary(refresh=True)

    assert Path(summary["ledger_path"]).exists()
    project = next(b for b in summary["boards"] if b["board_slug"] == "proj")
    assert project["board_name"] == "Project One"
    assert project["tasks"] == 1
    assert project["sessions"] == 1
    assert project["input_tokens"] == 120
    assert project["output_tokens"] == 45
    assert project["estimated_cost_usd"] == pytest.approx(0.0123)

    unassigned = next(b for b in summary["boards"] if b["board_slug"] == "__unassigned__")
    assert unassigned["sessions"] == 1

    drilldown = ledger.get_summary(board="proj", task_id=task_id, refresh=False)
    assert drilldown["totals"]["tasks"] == 1
    assert drilldown["runs"][0]["session_id"] == "sess-worker-1"


def test_project_usage_plugin_summary_endpoint(usage_home):
    _seed_state_db(usage_home)
    task_id = _seed_board()
    app = FastAPI()
    app.include_router(_load_plugin_router(), prefix="/api/plugins/project_usage")
    client = TestClient(app)

    r = client.get(f"/api/plugins/project_usage/summary?board=proj&task_id={task_id}")

    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["totals"]["input_tokens"] == 120
    assert payload["tasks"][0]["task_id"] == task_id
    assert payload["runs"][0]["session_id"] == "sess-worker-1"


def test_project_usage_dashboard_bundle_registers_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    bundle = repo_root / "plugins" / "project_usage" / "dashboard" / "dist" / "index.js"
    js = bundle.read_text()

    assert 'registry.register("project_usage", ProjectUsagePage)' in js
    assert "/api/plugins/project_usage/summary" in js
    assert "Per-board totals" in js
    assert "Per-task drilldown" in js
