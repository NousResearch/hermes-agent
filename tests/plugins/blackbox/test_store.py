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


def test_session_rollup_splits_subagent_spend(tmp_path, monkeypatch):
    """session_rollup must report subagent_count/subagent_usd as a SUBSET of the
    total (subagent turns are real rows already included in total_usd)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    store.insert_turn(make_record("main-1", is_subagent=False, ts_end=now - 3, cost_usd=1.00))
    store.insert_turn(make_record("sub-1", is_subagent=True, ts_end=now - 2, cost_usd=0.25))
    store.insert_turn(make_record("sub-2", is_subagent=True, ts_end=now - 1, cost_usd=0.15))

    rollup = store.session_rollup("telegram", "chat-1", limit=50)
    assert rollup["count"] == 3
    assert rollup["total_usd"] == pytest.approx(1.40)        # includes subagents
    assert rollup["subagent_count"] == 2
    assert rollup["subagent_usd"] == pytest.approx(0.40)


def test_subagent_rollup_aggregates_by_channel_with_unpriced(tmp_path, monkeypatch):
    """subagent_rollup sums cost/tokens of is_subagent rows in a channel, counts
    unpriced (cost_usd IS NULL) separately, and ignores other channels + main
    turns."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    # 2 priced + 1 unpriced subagent in chat-1; a main turn and a foreign-chat
    # subagent that must NOT be counted.
    store.insert_turn(make_record("s1", is_subagent=True, ts_end=now - 4, cost_usd=0.10, model="gpt-5.5"))
    store.insert_turn(make_record("s2", is_subagent=True, ts_end=now - 3, cost_usd=0.20, model="gpt-5.5"))
    store.insert_turn(make_record("s3-unpriced", is_subagent=True, ts_end=now - 2, cost_usd=None, model="claude-opus-4-8"))
    store.insert_turn(make_record("main", is_subagent=False, ts_end=now - 1, cost_usd=5.0))
    store.insert_turn(make_record("s-other", is_subagent=True, chat_id="chat-2", ts_end=now, cost_usd=9.0))

    roll = store.subagent_rollup("telegram", "chat-1", limit=200)
    assert roll["count"] == 3
    assert roll["total_usd"] == pytest.approx(0.30)   # unpriced excluded from sum
    assert roll["unpriced"] == 1
    assert set(roll["models"]) == {"gpt-5.5", "claude-opus-4-8"}


def test_subagent_rollup_empty_for_blank_channel(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert store.subagent_rollup("", "") == {
        "count": 0, "total_usd": 0.0, "unpriced": 0,
        "input_tokens": 0, "output_tokens": 0, "models": [],
    }


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


# --------------------------------------------------------------------------- #
# Last-call cache split (window decomposition) — columns + migration
# --------------------------------------------------------------------------- #
def test_last_call_split_round_trips_and_sums_to_context_used(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # last split sums to context_used by construction (invariant from the loop).
    record = make_record(
        "turn-split",
        context_used=206941,
        last_cache_read_tokens=3100,
        last_cache_write_tokens=203839,
        last_uncached_tokens=2,
    )
    store.insert_turn(record)
    row = store.get_turn("turn-split")
    assert row["last_cache_read_tokens"] == 3100
    assert row["last_cache_write_tokens"] == 203839
    assert row["last_uncached_tokens"] == 2
    assert (
        row["last_cache_read_tokens"]
        + row["last_cache_write_tokens"]
        + row["last_uncached_tokens"]
    ) == row["context_used"]


def test_last_call_split_defaults_to_null_when_absent(tmp_path, monkeypatch):
    """Old/unsplit records must persist NULL (not 0) so the renderer can fall back."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store.insert_turn(make_record("turn-nosplit"))  # no last_* overrides
    row = store.get_turn("turn-nosplit")
    assert row["last_cache_read_tokens"] is None
    assert row["last_cache_write_tokens"] is None
    assert row["last_uncached_tokens"] is None


def test_schema_migration_adds_columns_to_preexisting_table(tmp_path, monkeypatch):
    """A DB created WITHOUT the split columns gets them added idempotently.

    Simulates a real legacy DB: a full pre-split `turns` table (all the original
    columns, just missing the three last_* ones). _ensure_schema must ALTER them
    in, and a freshly inserted record must then round-trip through them.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db_path = store._db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Full pre-split schema (everything except last_cache_read/write/uncached).
    legacy = sqlite3.connect(str(db_path))
    legacy.execute(
        """
        CREATE TABLE turns (
            turn_id TEXT PRIMARY KEY, parent_turn_id TEXT, is_subagent INT,
            ts_start REAL, ts_end REAL, profile TEXT, provider TEXT, model TEXT,
            platform TEXT, chat_id TEXT, chat_name TEXT, api_calls INT, tools TEXT,
            input_tokens INT, output_tokens INT, cache_read INT, cache_write INT,
            reasoning INT, context_used INT, context_length INT, cost_usd REAL,
            cost_status TEXT, interrupted INT, alerted INT DEFAULT 0,
            user_text TEXT, final_text TEXT
        )
        """
    )
    legacy.commit()
    legacy.close()

    # Re-open through the store: _ensure_schema must ALTER in the new columns.
    with store._connect() as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(turns)").fetchall()}
    assert {"last_cache_read", "last_cache_write", "last_uncached"} <= cols

    # A new record round-trips through the migrated columns.
    store.insert_turn(make_record("post-migrate", last_cache_read_tokens=7,
                                  last_cache_write_tokens=8, last_uncached_tokens=1))
    row = store.get_turn("post-migrate")
    assert row["last_cache_read_tokens"] == 7
    assert row["last_cache_write_tokens"] == 8
    assert row["last_uncached_tokens"] == 1

    # Idempotent: connecting again must not raise (columns already present).
    with store._connect() as conn:
        conn.execute("SELECT last_cache_read FROM turns").fetchall()


# ---------------------------------------------------------------------------
# SPEC-C Phase 3 — per-class cost columns persist + migrate (additive, NULL-safe)
# ---------------------------------------------------------------------------
def test_perclass_cost_columns_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rec = make_record(
        "turn-perclass",
        cost_usd=1.50,
        cost_status="estimated",
        cost_uncached_usd=0.50,
        cost_cache_read_usd=0.30,
        cost_cache_write_usd=0.20,
        cost_output_usd=0.50,
    )
    store.insert_turn(rec)
    row = store.get_turn("turn-perclass")
    assert row is not None
    assert row["cost_uncached_usd"] == 0.50
    assert row["cost_cache_read_usd"] == 0.30
    assert row["cost_cache_write_usd"] == 0.20
    assert row["cost_output_usd"] == 0.50
    # the four parts reconcile to the stored total
    parts = (row["cost_uncached_usd"] + row["cost_cache_read_usd"]
             + row["cost_cache_write_usd"] + row["cost_output_usd"])
    assert abs(parts - row["cost_usd"]) < 1e-9


def test_perclass_none_persists_as_sql_null(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # a partial turn carries a real total but NO split (D-9) → NULL, not 0.
    store.insert_turn(make_record("turn-partial", cost_usd=2.0, cost_status="partial"))
    row = store.get_turn("turn-partial")
    assert row["cost_usd"] == 2.0
    assert row["cost_uncached_usd"] is None
    assert row["cost_cache_read_usd"] is None
    assert row["cost_cache_write_usd"] is None
    assert row["cost_output_usd"] is None


def test_migration_adds_perclass_columns_to_old_db(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import sqlite3
    db = tmp_path / "blackbox" / "turns.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    # Prove a fresh DB has the four perclass columns, then simulate an OLDER db
    # missing exactly those columns and prove the guarded ALTER adds them.
    c0 = store._connect(); c0.close()
    c1 = store._connect()
    cols = {r[1] for r in c1.execute("PRAGMA table_info(turns)").fetchall()}
    c1.close()
    for col in ("cost_uncached_usd", "cost_cache_read_usd",
                "cost_cache_write_usd", "cost_output_usd"):
        assert col in cols, f"{col} not present after schema ensure"
    # Now simulate an OLDER db missing exactly the perclass columns and prove
    # the guarded ALTER adds them. Drop+recreate isn't available, so rebuild a
    # turns table without them, then re-open via _connect (runs the migration).
    conn = sqlite3.connect(str(db))
    conn.execute("DROP TABLE turns")
    # full schema minus the 4 perclass columns
    conn.execute(
        """
        CREATE TABLE turns (
            turn_id TEXT PRIMARY KEY, parent_turn_id TEXT, is_subagent INT,
            ts_start REAL, ts_end REAL, profile TEXT, provider TEXT, model TEXT,
            platform TEXT, chat_id TEXT, chat_name TEXT, api_calls INT, tools TEXT,
            input_tokens INT, output_tokens INT, cache_read INT, cache_write INT,
            reasoning INT, context_used INT, context_length INT,
            last_cache_read INT, last_cache_write INT, last_uncached INT,
            comp_calls_json TEXT, cost_usd REAL, cost_status TEXT,
            interrupted INT, alerted INT DEFAULT 0, user_text TEXT, final_text TEXT
        )
        """)
    conn.commit(); conn.close()
    c2 = store._connect()  # runs _ensure_schema → guarded ALTERs
    cols2 = {r[1] for r in c2.execute("PRAGMA table_info(turns)").fetchall()}
    c2.close()
    for col in ("cost_uncached_usd", "cost_cache_read_usd",
                "cost_cache_write_usd", "cost_output_usd"):
        assert col in cols2, f"{col} not migrated into the old-shape DB"
    # idempotent: a further open must not raise
    store._connect().close()
