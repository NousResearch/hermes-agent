from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from hegi.collector import HermesSQLiteCollector, deduplicate_messages, extract_content
from hegi.config import AgentSourceConfig
from hegi.models import SourceMessage
from hegi.state import StateStore


def _make_db(path: Path, rows: list[tuple]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, chat_id TEXT, chat_type TEXT
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            timestamp REAL, platform_message_id TEXT, active INTEGER, compacted INTEGER
        );
        INSERT INTO sessions VALUES ('session-1', '-1001', 'group');
        """
    )
    connection.executemany(
        "INSERT INTO messages VALUES (?, 'session-1', ?, ?, ?, ?, ?, ?)", rows
    )
    connection.commit()
    connection.close()


def test_extract_content_excludes_tool_and_reasoning_blocks():
    content = json.dumps(
        [
            {"type": "text", "text": "첫 발언"},
            {"type": "tool_call", "content": "비밀 도구 인자"},
            {"type": "reasoning", "text": "숨은 추론"},
            {"type": "output_text", "text": "둘째 발언"},
        ],
        ensure_ascii=False,
    )
    assert extract_content(content) == "첫 발언 둘째 발언"


def test_collector_reads_active_chat_messages_and_persists_cursor(tmp_path):
    source_db = tmp_path / "source.db"
    _make_db(
        source_db,
        [
            (1, "user", "질문", 100.0, "p-1", 1, 0),
            (2, "assistant", '[{"type":"text","text":"답변"}]', 101.0, "p-2", 1, 0),
            (3, "assistant", "비활성", 102.0, "p-3", 0, 0),
        ],
    )
    state = StateStore(tmp_path / "state.db")
    collector = HermesSQLiteCollector(
        [AgentSourceConfig("헤헤", source_db)], state
    )

    messages = collector.collect("-1001")

    assert [message.content for message in messages] == ["질문", "답변"]
    assert state.get_cursor(str(source_db)) == (101.0, 2)
    assert [message.content for message in collector.collect("-1001")] == ["질문", "답변"]
    state.consume_messages([(str(source_db), 1), (str(source_db), 2)])
    assert collector.collect("-1001") == []


def test_dedup_keeps_shared_user_once_but_agents_distinct():
    base = dict(
        source_db="/tmp/db",
        message_id=1,
        session_id="s",
        platform_message_id="platform-1",
        chat_id="-1",
        chat_type="group",
        content="같은 내용",
        timestamp=100.0,
    )
    messages = [
        SourceMessage(source_agent="헤헤", role="user", **base),
        SourceMessage(source_agent="헤코", role="user", **base),
        SourceMessage(source_agent="헤헤", role="assistant", **base),
        SourceMessage(source_agent="헤코", role="assistant", **base),
    ]
    merged = deduplicate_messages(messages)
    assert len(merged) == 3
    assert {item.source_agent for item in merged if item.role == "assistant"} == {
        "헤헤",
        "헤코",
    }
