import json
import sqlite3
from pathlib import Path

from agent.self_evolution_bridge import export_state_db_sessions


def _create_test_db(path: Path) -> None:
    conn = sqlite3.connect(path)
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
                timestamp REAL NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            [
                ("discord-session", "discord", 20.0),
                ("cli-session", "cli", 10.0),
                ("tool-only", "cli", 5.0),
            ],
        )
        conn.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            [
                ("cli-session", "user", "first task", 1.0),
                ("cli-session", "assistant", "first answer", 2.0),
                ("discord-session", "system", "sys", 1.0),
                ("discord-session", "user", "[SYSTEM: skipped wrapper]", 1.5),
                ("discord-session", "user", "second task", 2.0),
                ("discord-session", "assistant", "second answer", 3.0),
                ("tool-only", "tool", "{}", 1.0),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_export_state_db_sessions_writes_legacy_json(tmp_path):
    db_path = tmp_path / "state.db"
    output_dir = tmp_path / "exported"
    _create_test_db(db_path)

    summary = export_state_db_sessions(output_dir=output_dir, db_path=db_path)

    assert summary.sessions_scanned == 3
    assert summary.sessions_exported == 2
    assert summary.sessions_skipped == 1
    assert summary.messages_exported == 5

    exported_files = sorted(path.name for path in output_dir.glob("*.json"))
    assert exported_files == ["cli-session.json", "discord-session.json"]

    discord_payload = json.loads((output_dir / "discord-session.json").read_text())
    assert discord_payload["session_id"] == "discord-session"
    assert discord_payload["source"] == "discord"
    assert [message["role"] for message in discord_payload["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert all("[SYSTEM:" not in message["content"] for message in discord_payload["messages"])


def test_export_state_db_sessions_respects_source_filter_and_limit(tmp_path):
    db_path = tmp_path / "state.db"
    output_dir = tmp_path / "filtered"
    _create_test_db(db_path)

    summary = export_state_db_sessions(
        output_dir=output_dir,
        db_path=db_path,
        sources=["discord"],
        limit_sessions=1,
    )

    assert summary.sessions_scanned == 1
    assert summary.sessions_exported == 1
    exported_files = list(output_dir.glob("*.json"))
    assert len(exported_files) == 1
    payload = json.loads(exported_files[0].read_text())
    assert payload["session_id"] == "discord-session"
