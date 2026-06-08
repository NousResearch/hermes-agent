"""Tests for the bounded Hermes reliability sentinel."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import yaml

from hermes_cli import kanban_db as kb
from hermes_cli import reliability


def test_reliability_report_flags_config_policy_and_writes_json(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        reliability,
        "_check_url",
        lambda _url, timeout=2.0: {"ok": True, "status": 200, "body": {"ok": True}},
    )
    monkeypatch.setattr(
        reliability,
        "_process_snapshot",
        lambda: {"ok": True, "zombies": 0},
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "fallback_providers": [],
            "skills": {"guard_agent_created": False, "disabled": ["spotify"]},
            "known_plugin_toolsets": {"cli": ["spotify"]},
        }),
        encoding="utf-8",
    )
    kb.init_db()

    report = reliability.build_reliability_report(home)

    issue_ids = {issue["id"] for issue in report["issues"]}
    assert "fallback_not_configured" in issue_ids
    assert "skills_guard_disabled" in issue_ids
    assert "spotify_toolset_registered" in issue_ids

    path = reliability.write_reliability_report(report, home)
    assert path == home / "health" / "reliability.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["profile_home"] == str(home)


def test_apply_bounded_repairs_archives_empty_sessions_and_dedupes_cards(
    tmp_path,
    monkeypatch,
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (home / "config.yaml").write_text("skills:\n  guard_agent_created: true\n", encoding="utf-8")
    conn = sqlite3.connect(str(home / "state.db"))
    conn.execute(
        "CREATE TABLE sessions ("
        "id TEXT PRIMARY KEY, archived INTEGER DEFAULT 0, "
        "message_count INTEGER DEFAULT 0, ended_at INTEGER, end_reason TEXT)"
    )
    conn.execute("INSERT INTO sessions (id, archived, message_count) VALUES ('empty', 0, 0)")
    conn.execute("INSERT INTO sessions (id, archived, message_count) VALUES ('live', 0, 2)")
    conn.commit()
    conn.close()
    kb.init_db()
    report = {
        "profile_home": str(home),
        "issues": [
            {
                "id": "dashboard_status_unreachable",
                "severity": "error",
                "title": "Dashboard status endpoint is unreachable",
                "detail": "boom",
                "repair": "kanban_card",
                "data": {},
            }
        ],
    }

    first = reliability.apply_bounded_repairs(report, home)
    second = reliability.apply_bounded_repairs(report, home)

    assert Path(first["backup_dir"]).exists()
    assert first["archived_empty_sessions"] == 1
    assert second["archived_empty_sessions"] == 0
    assert first["kanban_cards"] == second["kanban_cards"]

    conn = sqlite3.connect(str(home / "state.db"))
    rows = dict(conn.execute("SELECT id, archived FROM sessions").fetchall())
    conn.close()
    assert rows["empty"] == 1
    assert rows["live"] == 0

    with kb.connect() as board:
        count = board.execute(
            "SELECT COUNT(*) FROM tasks WHERE idempotency_key=?",
            ("reliability:dashboard_status_unreachable",),
        ).fetchone()[0]
    assert count == 1
