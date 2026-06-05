from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.self_improvement import log_session_telemetry as telemetry


def _make_state_db(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        """
        create table sessions (
            id text primary key,
            source text not null,
            model text,
            started_at real not null,
            ended_at real,
            end_reason text,
            message_count integer default 0,
            tool_call_count integer default 0,
            api_call_count integer default 0,
            input_tokens integer default 0,
            output_tokens integer default 0,
            cache_read_tokens integer default 0,
            cache_write_tokens integer default 0,
            reasoning_tokens integer default 0,
            estimated_cost_usd real,
            actual_cost_usd real,
            cost_status text,
            cost_source text
        );
        create table messages (
            id integer primary key,
            session_id text not null,
            role text not null,
            content text,
            tool_name text
        );
        """
    )
    con.execute(
        """
        insert into sessions (
            id, source, model, started_at, ended_at, end_reason,
            message_count, tool_call_count, api_call_count,
            input_tokens, output_tokens, cache_read_tokens,
            cache_write_tokens, reasoning_tokens,
            estimated_cost_usd, actual_cost_usd, cost_status, cost_source
        ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "ended-discord",
            "discord",
            "gpt-test",
            10.0,
            20.0,
            "session_reset",
            4,
            2,
            3,
            100,
            20,
            300,
            10,
            5,
            0.12,
            0.0,
            "estimated",
            "fixture",
        ),
    )
    con.execute(
        """
        insert into sessions (id, source, model, started_at, ended_at)
        values ('open-discord', 'discord', 'gpt-open', 99.0, null)
        """
    )
    con.executemany(
        "insert into messages (session_id, role, content, tool_name) values (?, ?, ?, ?)",
        [
            ("ended-discord", "user", "please do thing", None),
            ("ended-discord", "assistant", "working", None),
            ("ended-discord", "tool", "x" * 12_000, "skill_view"),
            ("ended-discord", "tool", "ok", "terminal"),
        ],
    )
    con.commit()
    con.close()


def test_dry_run_prints_task_run_without_writing_jsonl(tmp_path, capsys):
    db = tmp_path / "state.db"
    out_dir = tmp_path / "self-improvement-log"
    _make_state_db(db)

    rc = telemetry.main(["--db", str(db), "--root", str(out_dir), "--dry-run"])

    assert rc == 0
    assert not (out_dir / "task_runs.jsonl").exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["kind"] == "task_run_telemetry"
    assert payload["session_id"] == "ended-discord"
    assert payload["input_tokens"] == 100
    assert payload["tool_stats"] == ["skill_view:1", "terminal:1"]
    assert payload["largest_context_items"][0] == "tool:skill_view:12000"


def test_appends_to_task_runs_jsonl_not_events_by_default(tmp_path):
    db = tmp_path / "state.db"
    out_dir = tmp_path / "self-improvement-log"
    _make_state_db(db)

    rc = telemetry.main(["--db", str(db), "--root", str(out_dir)])

    assert rc == 0
    task_runs = out_dir / "task_runs.jsonl"
    assert task_runs.exists()
    assert not (out_dir / "events.jsonl").exists()
    rows = [json.loads(line) for line in task_runs.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "ended-discord"
    assert rows[0]["cost_status"] == "estimated"


def test_duplicate_session_is_not_appended_without_force(tmp_path, capsys):
    db = tmp_path / "state.db"
    out_dir = tmp_path / "self-improvement-log"
    _make_state_db(db)

    assert telemetry.main(["--db", str(db), "--root", str(out_dir)]) == 0
    assert telemetry.main(["--db", str(db), "--root", str(out_dir)]) == 0

    task_runs = out_dir / "task_runs.jsonl"
    assert len(task_runs.read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["status"] == "already_logged"

    assert telemetry.main(["--db", str(db), "--root", str(out_dir), "--force"]) == 0
    assert len(task_runs.read_text(encoding="utf-8").splitlines()) == 2


def test_latest_session_defaults_to_ended_sessions_unless_include_open(tmp_path):
    db = tmp_path / "state.db"
    _make_state_db(db)
    con = sqlite3.connect(db)

    assert telemetry.latest_session_id(con, source="discord") == "ended-discord"
    assert telemetry.latest_session_id(con, source="discord", include_open=True) == "open-discord"


def test_append_event_writes_compact_event_with_measures(tmp_path):
    db = tmp_path / "state.db"
    out_dir = tmp_path / "self-improvement-log"
    _make_state_db(db)

    rc = telemetry.main(["--db", str(db), "--root", str(out_dir), "--append-event"])

    assert rc == 0
    event_rows = [json.loads(line) for line in (out_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(event_rows) == 1
    assert event_rows[0]["route"] == "workflow_improvement"
    assert event_rows[0]["measures"]["session_id"] == "ended-discord"


@pytest.mark.parametrize("argv", [["--source", "telegram"], ["--session-id", "missing"]])
def test_missing_session_exits_cleanly(tmp_path, argv):
    db = tmp_path / "state.db"
    out_dir = tmp_path / "self-improvement-log"
    _make_state_db(db)

    with pytest.raises(SystemExit):
        telemetry.main(["--db", str(db), "--root", str(out_dir), *argv])
