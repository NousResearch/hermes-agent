from __future__ import annotations

import json
import time

from agent.insights import InsightsEngine
from conversation_store import ConversationRevision, ConversationStore
from hermes_cli.approvals_suggest import scan_approval_history
from hermes_state import SessionDB


class SurfaceStore(ConversationStore):
    def __init__(self):
        now = time.time()
        self.messages = {
            "s1": [
                {"id": 101, "session_id": "s1", "role": "user", "content": "go", "timestamp": now},
                {
                    "id": 102, "session_id": "s1", "role": "assistant", "content": "", "timestamp": now,
                    "tool_calls": [
                        {"id": "terminal-1", "function": {"name": "terminal", "arguments": json.dumps({"command": "git push --force origin main"})}},
                        {"id": "search-1", "function": {"name": "search_files", "arguments": "{}"}},
                        {"id": "skill-1", "function": {"name": "skill_view", "arguments": json.dumps({"name": "github-pr-workflow"})}},
                    ],
                },
                {"id": 103, "session_id": "s1", "role": "tool", "content": "ok", "tool_call_id": "terminal-1", "tool_name": "terminal", "timestamp": now},
                {"id": 104, "session_id": "s1", "role": "tool", "content": "matches", "tool_call_id": "search-1", "tool_name": "search_files", "timestamp": now},
            ]
        }

    @property
    def name(self):
        return "surface-store"

    def is_available(self):
        return True

    def ensure_conversation(self, conversation):
        return ConversationRevision(1)

    def list_conversations(self, **filters):
        rows = [{"id": "s1", "source": "cli", "started_at": time.time(), "archived": 0}]
        offset = int(filters.get("offset") or 0)
        limit = filters.get("limit")
        return rows[offset:] if limit is None else rows[offset:offset + int(limit)]

    def list_messages(self, conversation_id, **filters):
        return [dict(row) for row in self.messages.get(conversation_id, [])]


def _db(tmp_path):
    store = SurfaceStore()
    db = SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store)
    db.create_session("s1", source="cli", model="test-model")
    return db, store


def test_external_insights_read_canonical_transcript_not_sqlite_messages(tmp_path):
    db, _store = _db(tmp_path)
    try:
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
        engine = InsightsEngine(db)
        usage = engine.get_usage_breakdown(days=30)
        tools = {row["tool"]: row["count"] for row in usage["tools"]}
        assert tools["terminal"] == 1
        assert tools["search_files"] == 1
        assert tools["skill_view"] == 1
        skills = {row["skill"]: row for row in usage["skills"]["top_skills"]}
        assert skills["github-pr-workflow"]["view_count"] == 1

        report = engine.generate(days=30)
        assert report["overview"]["user_messages"] == 1
        assert report["overview"]["assistant_messages"] == 1
        assert report["overview"]["tool_messages"] == 2
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        db.close()


def test_default_approval_scan_can_mine_external_canonical_history(tmp_path):
    db, _store = _db(tmp_path)
    try:
        records = scan_approval_history(days=0, session_db=db)
        assert len(records) == 1
        assert records[0][0] == "git push --force origin main"
        assert db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    finally:
        db.close()
