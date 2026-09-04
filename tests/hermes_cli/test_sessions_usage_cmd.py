"""Tests for `hermes sessions usage <id>` (per-session usage breakdown).

Inspired by Amp's `amp threads usage <thread-id> --details` (Explain Usage,
Aug 2026): a per-thread token/cost drill-down with machine-readable output.
"""

import json
import time
from types import SimpleNamespace

import pytest

from hermes_cli.sessions_cmd import cmd_sessions


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_state

    db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    sid = "usagetest-0123456789abcdef"
    db._conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', ?)",
        (sid, time.time()),
    )
    db._conn.commit()
    db.update_token_counts(
        sid,
        input_tokens=1000,
        output_tokens=250,
        cache_read_tokens=400,
        cache_write_tokens=50,
        reasoning_tokens=30,
        model="test/model-x",
        billing_provider="openrouter",
        billing_base_url="https://openrouter.ai/api/v1",
    )
    db.flush_token_counts()
    db.close()

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    orig = hermes_state.SessionDB

    def _open(*a, **k):
        k.setdefault("db_path", tmp_path / "state.db")
        return orig(*a, **k)

    monkeypatch.setattr(hermes_state, "SessionDB", _open)
    return sid


def _args(session_id, as_json=False):
    return SimpleNamespace(sessions_action="usage", session_id=session_id, json=as_json)


def test_usage_text_output(seeded_db, capsys):
    cmd_sessions(_args("usagetest"))
    out = capsys.readouterr().out
    assert "test/model-x" in out
    assert "1,000" in out  # input tokens
    assert "1,700" in out  # total tokens
    assert "Per-route breakdown" in out
    assert "openrouter" in out


def test_usage_json_output(seeded_db, capsys):
    cmd_sessions(_args("usagetest", as_json=True))
    data = json.loads(capsys.readouterr().out)
    assert data["session_id"] == seeded_db
    assert data["totals"]["input_tokens"] == 1000
    assert data["totals"]["total_tokens"] == 1700
    assert data["totals"]["reasoning_tokens"] == 30
    assert len(data["by_model"]) == 1
    route = data["by_model"][0]
    assert route["model"] == "test/model-x"
    assert route["provider"] == "openrouter"


def test_usage_session_not_found(seeded_db, capsys):
    cmd_sessions(_args("does-not-exist"))
    out = capsys.readouterr().out
    assert "Session not found" in out
