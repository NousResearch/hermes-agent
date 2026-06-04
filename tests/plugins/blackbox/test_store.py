from __future__ import annotations

import logging
import sqlite3
import time
from decimal import Decimal

import pytest

from plugins.blackbox.record import TurnRecord
from plugins.blackbox import store


def make_record(turn_id: str, **overrides) -> TurnRecord:
    values = {
        "turn_id": turn_id,
        "parent_turn_id": "parent-1",
        "is_subagent": True,
        "ts_start": 100.0,
        "ts_end": 104.5,
        "profile": "default",
        "provider": "openai",
        "model": "gpt-test",
        "platform": "telegram",
        "chat_id": "chat-1",
        "chat_name": "Test Chat",
        "api_calls": 2,
        "tools": ["exec", "exec", "read"],
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_read_tokens": 3,
        "cache_write_tokens": 4,
        "reasoning_tokens": 5,
        "context_used": 1000,
        "context_length": 2000,
        "cost_usd": Decimal("0.1234"),
        "cost_status": "estimated",
        "interrupted": True,
        "alerted": False,
        "user_text": "hello",
        "final_text": "world",
        "tool_calls": [
            {
                "name": "exec",
                "args_preview": '{"cmd": "echo hi"}',
                "result_preview": "hi",
            },
            {
                "name": "read",
                "args_preview": "file.txt",
                "result_preview": "contents",
            },
        ],
    }
    values.update(overrides)
    return TurnRecord(**values)


def test_insert_and_get_round_trips_all_fields_and_tool_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = make_record("turn-1")

    store.insert_turn(record)

    row = store.get_turn("turn-1")
    assert row is not None
    assert row["turn_id"] == record.turn_id
    assert row["parent_turn_id"] == record.parent_turn_id
    assert row["is_subagent"] is True
    assert row["ts_start"] == record.ts_start
    assert row["ts_end"] == record.ts_end
    assert row["profile"] == record.profile
    assert row["provider"] == record.provider
    assert row["model"] == record.model
    assert row["platform"] == record.platform
    assert row["chat_id"] == record.chat_id
    assert row["chat_name"] == record.chat_name
    assert row["api_calls"] == record.api_calls
    assert row["tools"] == record.tools
    assert row["tools_summary"] == "exec×2, read"
    assert row["input_tokens"] == record.input_tokens
    assert row["output_tokens"] == record.output_tokens
    assert row["cache_read_tokens"] == record.cache_read_tokens
    assert row["cache_write_tokens"] == record.cache_write_tokens
    assert row["reasoning_tokens"] == record.reasoning_tokens
    assert row["context_used"] == record.context_used
    assert row["context_length"] == record.context_length
    assert row["cost_usd"] == float(record.cost_usd)
    assert row["cost_status"] == record.cost_status
    assert row["interrupted"] is True
    assert row["alerted"] is False
    assert row["user_text"] == record.user_text
    assert row["final_text"] == record.final_text

    calls = store.get_tool_calls("turn-1")
    assert calls == [
        {
            "seq": 0,
            "name": "exec",
            "args_preview": '{"cmd": "echo hi"}',
            "result_preview": "hi",
        },
        {
            "seq": 1,
            "name": "read",
            "args_preview": "file.txt",
            "result_preview": "contents",
        },
    ]


def test_mark_alerted_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store.insert_turn(make_record("turn-alert"))

    assert store.mark_alerted("turn-alert") is True
    assert store.mark_alerted("turn-alert") is False
    assert store.get_turn("turn-alert")["alerted"] is True


def test_get_last_turn_resolves_by_platform_and_chat_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store.insert_turn(make_record("turn-a", platform="slack", chat_id="c1", ts_end=1))
    store.insert_turn(make_record("turn-b", platform="slack", chat_id="c2", ts_end=2))
    store.insert_turn(make_record("turn-c", platform="slack", chat_id="c1", ts_end=3))

    assert store.get_last_turn("slack", "c1")["turn_id"] == "turn-c"
    assert store.get_last_turn("slack", "c2")["turn_id"] == "turn-b"
    assert store.get_last_turn("discord", "c1") is None


def test_scrub_before_truncate_masks_token_crossing_boundary():
    token = "sk-" + ("A" * 50)
    text = ("x" * 1986) + " " + token + " tail"

    scrubbed = store.scrub_and_truncate(text, n=2000)

    assert len(scrubbed) == 2000
    assert token[:10] not in scrubbed
    assert scrubbed.endswith("sk-AAA...AAAA")


def test_sweep_deletes_old_rows_respects_cap_and_writes_sentinel(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    for idx in range(3):
        store.insert_turn(
            make_record(
                f"old-{idx}",
                ts_start=now - 10 * 86400,
                ts_end=now - (10 * 86400) + idx,
                tool_calls=[{"name": "exec", "args_preview": "a", "result_preview": "r"}],
            )
        )
    store.insert_turn(make_record("new", ts_start=now, ts_end=now))

    assert store.sweep(retention_days=7, max_deletes=2) == 2
    assert store.get_turn("old-0") is None
    assert store.get_turn("old-1") is None
    assert store.get_turn("old-2") is not None
    assert store.get_turn("new") is not None
    assert store.get_tool_calls("old-0") == []

    conn = sqlite3.connect(str(tmp_path / "blackbox" / "turns.db"))
    try:
        sentinel = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_sweep_date'"
        ).fetchone()
    finally:
        conn.close()
    assert sentinel is not None
    assert sentinel[0] == time.strftime("%Y-%m-%d", time.gmtime())
    assert store.sweep(retention_days=7, max_deletes=2) == 0
    assert store.get_turn("old-2") is not None


def test_insert_turn_on_broken_db_path_logs_and_does_not_raise(tmp_path, monkeypatch, caplog):
    broken_home = tmp_path / "home-file"
    broken_home.write_text("not a directory")
    monkeypatch.setenv("HERMES_HOME", str(broken_home))

    with caplog.at_level(logging.WARNING, logger="plugins.blackbox.store"):
        store.insert_turn(make_record("broken"))

    assert "blackbox telemetry insert failed" in caplog.text


def test_session_rollup_and_top_turns(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    store.insert_turn(make_record("cheap", ts_end=now - 3, cost_usd=0.10))
    store.insert_turn(make_record("mid", ts_end=now - 2, cost_usd=0.20))
    store.insert_turn(make_record("other-chat", chat_id="chat-2", ts_end=now - 1, cost_usd=9.0))
    store.insert_turn(make_record("expensive", ts_end=now, cost_usd=0.30))
    store.insert_turn(make_record("old", ts_end=now - 40 * 86400, cost_usd=10.0))

    rollup = store.session_rollup("telegram", "chat-1", limit=3)
    assert rollup["count"] == 3
    assert rollup["total_usd"] == pytest.approx(0.60)
    assert rollup["avg_usd"] == pytest.approx(0.20)
    assert rollup["max_turn"]["turn_id"] == "expensive"

    top = store.top_turns(n=2, since_days=30)
    assert [row["turn_id"] for row in top] == ["other-chat", "expensive"]


def test_debug_stats_reports_counts_and_paths(tmp_path, monkeypatch):
    """Real-DB operational snapshot for /cost debug — no mocks."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    # 2 top-level + 1 subagent; mark one alerted.
    store.insert_turn(make_record("t-top", is_subagent=False, ts_end=now - 5))
    store.insert_turn(make_record("t-sub", is_subagent=True, ts_end=now - 3))
    store.insert_turn(make_record("t-alert", is_subagent=False, ts_end=now))
    store.mark_alerted("t-alert")

    stats = store.debug_stats()

    assert stats["db_exists"] is True
    assert stats["db_path"].endswith("blackbox/turns.db")
    assert stats["turns"] == 3
    assert stats["subagent_turns"] == 1
    assert stats["alerted"] == 1
    # Each make_record carries 2 tool_calls → 3 turns × 2 = 6 side rows.
    assert stats["tool_calls"] == 6
    assert stats["newest_ts"] == pytest.approx(now)
    assert "error" not in stats


def test_debug_stats_on_missing_db_is_graceful(tmp_path, monkeypatch):
    """Before any turn is recorded, debug must still report (db absent)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    stats = store.debug_stats()
    # _connect() creates the schema on first touch, so the DB exists but is
    # empty; the key contract is: no exception, zero counts.
    assert stats["turns"] == 0
    assert stats["tool_calls"] == 0
    assert "error" not in stats
