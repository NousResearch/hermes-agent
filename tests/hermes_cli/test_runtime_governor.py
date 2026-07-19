import sqlite3
import time

from hermes_cli.runtime_governor import generate_runtime_report, format_runtime_report


def _create_db(path, *, minimal=False):
    conn = sqlite3.connect(path)
    if minimal:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
    else:
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL,
                title TEXT,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                api_call_count INTEGER DEFAULT 0,
                estimated_cost_usd REAL,
                actual_cost_usd REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL
            )
            """
        )
    conn.commit()
    return conn


def _add_session(conn, session_id, now, *, started_days_ago=1, input_tokens=0, output_tokens=0,
                 cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=0,
                 api_call_count=0, estimated_cost_usd=None, actual_cost_usd=None,
                 title=None):
    conn.execute(
        """
        INSERT INTO sessions (
            id, source, started_at, title, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, reasoning_tokens, api_call_count,
            estimated_cost_usd, actual_cost_usd
        ) VALUES (?, 'cli', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            now - started_days_ago * 86400,
            title,
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            api_call_count,
            estimated_cost_usd,
            actual_cost_usd,
        ),
    )


def _add_message(conn, session_id, content, *, role="assistant", offset=0, tool_name=None):
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, tool_name, time.time() + offset),
    )


def test_missing_cost_fields_are_marked_unreliable_and_not_reported_as_spend(tmp_path):
    db_path = tmp_path / "state.db"
    conn = _create_db(db_path)
    now = time.time()
    _add_session(
        conn,
        "s1",
        now,
        input_tokens=100_000,
        output_tokens=20_000,
        cache_read_tokens=1_000_000,
        api_call_count=12,
        estimated_cost_usd=0.0,
        actual_cost_usd=None,
    )
    conn.commit()
    conn.close()

    report = generate_runtime_report(state_db_path=db_path, days=30, now=now)
    rendered = format_runtime_report(report)

    assert report["cost"]["reliable"] is False
    assert report["totals"]["sessions"] == 1
    assert "cost telemetry unreliable" in rendered.lower()
    assert "receipt spend is" not in rendered.lower()


def test_receipt_tagged_sessions_are_upper_bounds_and_loops_are_flagged(tmp_path):
    db_path = tmp_path / "state.db"
    conn = _create_db(db_path)
    now = time.time()
    _add_session(
        conn,
        "receipt_loop",
        now,
        input_tokens=60_000,
        output_tokens=10_000,
        cache_read_tokens=500_000,
        cache_write_tokens=5_000,
        api_call_count=9,
        title="receipt finalization churn",
    )
    _add_message(conn, "receipt_loop", "Write the receipt to .hermes/receipts/r.md", offset=1)
    _add_message(conn, "receipt_loop", "Read the receipt back with read_file", offset=2)
    _add_message(conn, "receipt_loop", "Summarize the receipt again for the final response", offset=3)
    conn.commit()
    conn.close()

    report = generate_runtime_report(state_db_path=db_path, days=30, now=now)
    rendered = format_runtime_report(report)

    assert report["receipt_overhead"]["receipt_tagged_sessions"] == 1
    assert report["receipt_overhead"]["suspected_loops"] == 1
    assert report["receipt_overhead"]["write_mentions"] == 1
    assert report["receipt_overhead"]["read_mentions"] == 1
    assert "upper bound" in rendered.lower()
    assert "write/read/summarize" in rendered.lower()


def test_schema_tolerance_with_minimal_tables(tmp_path):
    db_path = tmp_path / "state.db"
    conn = _create_db(db_path, minimal=True)
    now = time.time()
    conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('minimal', 'cli', ?)", (now,))
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('minimal', 'assistant', 'No receipt here', ?)",
        (now,),
    )
    conn.commit()
    conn.close()

    report = generate_runtime_report(state_db_path=db_path, days=1, now=now)

    assert report["totals"]["sessions"] == 1
    assert report["totals"]["api_calls"] == 0
    assert report["cost"]["reliable"] is False


def test_outlier_sessions_and_policy_recommendations_are_deterministic(tmp_path):
    db_path = tmp_path / "state.db"
    conn = _create_db(db_path)
    now = time.time()
    _add_session(conn, "normal", now, input_tokens=1_000, output_tokens=500, api_call_count=2)
    _add_session(
        conn,
        "burner",
        now,
        input_tokens=300_000,
        output_tokens=50_000,
        cache_read_tokens=2_000_000,
        cache_write_tokens=25_000,
        api_call_count=75,
    )
    _add_message(conn, "burner", "Saved final receipt", offset=1)
    _add_message(conn, "burner", "Read the full receipt again", offset=2)
    conn.commit()
    conn.close()

    report = generate_runtime_report(state_db_path=db_path, days=30, now=now)

    assert report["outliers"]["by_api_calls"][0]["session_id"] == "burner"
    assert report["recommendations"]
    assert any("stat/checksum/excerpt" in item["recommendation"] for item in report["recommendations"])
