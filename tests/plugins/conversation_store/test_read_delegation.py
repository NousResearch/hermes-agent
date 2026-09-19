from __future__ import annotations

from pathlib import Path

from conversation_store import ConversationStore
from hermes_state import SessionDB
from tools.session_search_tool import _get_message_storage_state
from tui_gateway.methods_profiles import _latest_message_preview


class ReadStore(ConversationStore):
    def __init__(self):
        self.calls = []

    @property
    def name(self):
        return "read-store"

    def is_available(self):
        return True

    def _record(self, name, **kwargs):
        self.calls.append((name, kwargs))

    def ensure_conversation(self, conversation):
        from conversation_store import ConversationRevision
        self._record("ensure_conversation", conversation=conversation)
        return ConversationRevision(1)

    def get_conversation(self, conversation_id):
        self._record("get_conversation", conversation_id=conversation_id)
        return {"id": conversation_id, "title": "External", "title_source": "user"}

    def resolve_conversation_id(self, value):
        self._record("resolve_conversation_id", value=value)
        return "external-session"

    def list_conversations(self, **filters):
        self._record("list_conversations", **filters)
        return [{"id": "external-session", "preview": "provider"}]

    def search_conversations(self, **filters):
        self._record("search_conversations", **filters)
        return [{"id": "external-session", "last_active": 9}]

    def count_conversations(self, **filters):
        self._record("count_conversations", **filters)
        return 7

    def find_conversation_by_title(self, title):
        self._record("find_conversation_by_title", title=title)
        return {"id": "external-session", "title": title}

    def resolve_conversation_by_title(self, title):
        self._record("resolve_conversation_by_title", title=title)
        return "external-session"

    def list_messages(self, conversation_id, **filters):
        self._record("list_messages", conversation_id=conversation_id, **filters)
        count = 3 if conversation_id == "external-large" else 1
        rows = [
            {"id": 41 + i, "session_id": conversation_id, "role": "user", "content": "external"}
            for i in range(count)
        ]
        limit = filters.get("limit")
        return rows if limit is None else rows[:limit]

    def messages_around(self, conversation_id, message_id, *, window=5):
        self._record("messages_around", conversation_id=conversation_id, message_id=message_id, window=window)
        return {"window": [{"id": message_id, "role": "user"}], "messages_before": 1, "messages_after": 2}

    def conversation_history(self, conversation_id, **filters):
        self._record("conversation_history", conversation_id=conversation_id, **filters)
        return [{"role": "user", "content": "hello", "_row_id": 41}, {"role": "assistant", "content": "hi"}]

    def resume_histories(self, conversation_id):
        self._record("resume_histories", conversation_id=conversation_id)
        return ([{"role": "user", "content": "hello"}], [{"role": "user", "content": "hello"}])

    def resolve_resume_conversation_id(self, conversation_id):
        self._record("resolve_resume_conversation_id", conversation_id=conversation_id)
        return "external-tip"

    def resume_message_count(self, conversation_id, *, tip_only=False):
        self._record("resume_message_count", conversation_id=conversation_id, tip_only=tip_only)
        return 3 if tip_only else 5

    def count_messages(self, conversation_id=None):
        self._record("count_messages", conversation_id=conversation_id)
        return 11

    def search_messages(self, query, **filters):
        self._record("search_messages", query=query, **filters)
        return [{"id": 41, "session_id": "external-session", "snippet": "external"}]

    def anchored_view(self, conversation_id, message_id, **filters):
        self._record("anchored_view", conversation_id=conversation_id, message_id=message_id, **filters)
        return {"window": [{"id": message_id}], "bookend_start": [], "bookend_end": []}

    def recent_user_messages(self, conversation_id, **filters):
        self._record("recent_user_messages", conversation_id=conversation_id, **filters)
        return [{"id": 41, "timestamp": 1.0, "preview": "external"}]

    def message_storage_state(self, message_id):
        self._record("message_storage_state", message_id=message_id)
        return {"session_id": "external-session", "active": 1, "compacted": 0}

    def latest_message_preview(self, conversation_id):
        self._record("latest_message_preview", conversation_id=conversation_id)
        return "external preview"


def _db(tmp_path: Path):
    store = ReadStore()
    return SessionDB(db_path=tmp_path / "shadow.db", conversation_store=store), store


def test_session_listing_and_title_reads_use_external_store(tmp_path):
    db, store = _db(tmp_path)
    try:
        db.create_session("external-session", source="cli")
        session = db.get_session("external-session")
        assert session["title"] == "External"
        assert session["source"] == "cli"  # local operational shadow survives canonical overlay
        assert db.get_session_title("external-session") == "External"
        assert db.get_session_title_source("external-session") == "user"
        assert db.resolve_session_id("external") == "external-session"
        assert db.get_session_by_title("External")["id"] == "external-session"
        assert db.resolve_session_by_title("External") == "external-session"
        assert db.list_sessions_rich(limit=3)[0]["preview"] == "provider"
        assert db.search_sessions(source="cli", limit=2)[0]["id"] == "external-session"
        assert db.session_count(source="cli") == 7
        assert any(name == "list_conversations" and args["limit"] == 3 for name, args in store.calls)
    finally:
        db.close()


def test_message_and_resume_reads_use_external_store(tmp_path):
    db, store = _db(tmp_path)
    try:
        assert db.get_messages("external-session", latest=True)[0]["id"] == 41
        assert db.get_messages_around("external-session", 41, window=2)["messages_after"] == 2
        history = db.get_messages_as_conversation("external-session", include_row_ids=True)
        assert history[0]["_db_persisted"] is True
        assert history[0]["_row_id"] == 41
        model, display = db.get_resume_conversations("external-session")
        assert model[0]["_db_persisted"] is True
        assert display[0]["_db_persisted"] is True
        assert db.resolve_resume_session_id("external-session") == "external-tip"
        assert db.get_resume_message_count("external-session") == 5
        assert db.get_resume_message_count("external-session", tip_only=True) == 3
        assert db.message_count("external-session") == 11
        from hermes_state import SessionExportTooLargeError
        try:
            db.assert_export_safe("external-large", max_messages=2)
        except SessionExportTooLargeError:
            pass
        else:
            raise AssertionError("external transcript must obey the export-size guard")
    finally:
        db.close()


def test_search_views_and_direct_sql_escape_hatches_use_external_store(tmp_path):
    db, store = _db(tmp_path)
    try:
        assert db.search_messages("needle")[0]["snippet"] == "external"
        assert db.get_anchored_view("external-session", 41)["window"][0]["id"] == 41
        assert db.list_recent_user_messages("external-session")[0]["preview"] == "external"
        assert _get_message_storage_state(db, 41)["session_id"] == "external-session"
        assert _latest_message_preview(db, "external-session") == "external preview"
        names = [name for name, _ in store.calls]
        assert "message_storage_state" in names
        assert "latest_message_preview" in names
    finally:
        db.close()
