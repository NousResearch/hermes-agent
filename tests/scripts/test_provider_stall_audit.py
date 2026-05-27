import json
import sqlite3
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "provider_stall_audit.py"


def run_audit(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_provider_stall_audit_aggregates_logs_without_raw_payloads(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    secretish_payload = "user prompt says provider 300s audit meta discussion"
    (logs / "errors.log").write_text(
        "\n".join(
            [
                "2026-05-25 01:02:03,000 WARNING [20260525_aaa] agent.chat_completion_helpers: Non-streaming API call stale for 300s (threshold 300s). model=gpt-5.5 context=~123 tokens. Killing connection.",
                "2026-05-25 01:03:03,000 WARNING tools.delegate_tool: Subagent 12 timed out after 600.0s",
                "2026-05-26 01:04:03,000 WARNING agent.conversation_compression: No auxiliary LLM provider for compression — summaries will be unavailable.",
                f"2026-05-26 01:05:03,000 INFO gateway.run: inbound message: platform=discord msg='{secretish_payload}'",
            ]
        ),
        encoding="utf-8",
    )
    (logs / "gateway.log").write_text(
        "\n".join(
            [
                "2026-05-25 02:02:03,000 INFO gateway.run: response ready: platform=discord chat=123 time=61.0s api_calls=2 response=9 chars",
                "2026-05-25 02:03:03,000 WARNING gateway.platforms.discord: [Discord] Failed to send follow-up chunk to forum thread 123: rate limited",
                "2026-05-26 02:02:03,000 INFO gateway.run: response ready: platform=discord chat=123 time=59.9s api_calls=1 response=9 chars",
            ]
        ),
        encoding="utf-8",
    )

    output = run_audit("--logs-dir", str(logs), "--start-date", "2026-05-25", "--end-date", "2026-05-26")

    assert output["totals_by_category"] == {
        "compression_summary_failures": 1,
        "direct_provider_stalls": 1,
        "discord_send_failures": 1,
        "discord_slow_responses": 1,
        "timeout_lines": 1,
    }
    assert output["counts"]["direct_provider_stalls"]["2026-05-25"]["errors_log"] == 1
    assert output["counts"]["discord_slow_responses"]["2026-05-25"]["gateway_log"] == 1
    serialized = json.dumps(output)
    assert secretish_payload not in serialized
    assert "Non-streaming API call stale" not in serialized


def test_provider_stall_audit_keeps_session_meta_mentions_separate(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "agent.log").write_text("", encoding="utf-8")
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    try:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                started_at REAL NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL
            );
            """
        )
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', 1779667200)")
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('s2', 'discord', 1779667200)")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
            ("s1", "assistant", "audit report says provider_300s mentions can be meta", None, 1779667200),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
            ("s2", "tool", "provider stale line observed in a tool payload", "terminal", 1779667200),
        )
        conn.commit()
    finally:
        conn.close()

    output = run_audit(
        "--logs-dir",
        str(logs),
        "--state-db",
        str(db),
        "--include-session-content-mentions",
        "--start-date",
        "2026-05-25",
        "--end-date",
        "2026-05-25",
    )

    assert output["totals_by_category"] == {
        "session_meta_discussion_mentions": 1,
        "session_tool_log_mentions": 1,
    }
    assert output["counts"]["session_meta_discussion_mentions"]["2026-05-25"]["cli"] == 1
    assert output["counts"]["session_tool_log_mentions"]["2026-05-25"]["discord"] == 1
    serialized = json.dumps(output)
    assert "audit report says" not in serialized
    assert "provider stale line" not in serialized
