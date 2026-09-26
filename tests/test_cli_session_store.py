"""Unit contracts for the PostgreSQL CLI session-store facade."""

from cli_session_store import PostgreSQLCLISessionStore


def test_common_cli_history_page_hydrates_only_the_requested_postgresql_rows():
    """Bare /resume must not hydrate an arbitrary large PostgreSQL history."""
    class SummaryStore:
        def __init__(self):
            self.summary_calls = []
            self.message_record_calls = []
            self.rows = [
                {"id": f"history-{index:03d}", "source": "cli", "started_at": index,
                 "last_active": index, "message_count": 1}
                for index in range(50, 0, -1)
            ]

        def list_session_summaries(self, **kwargs):
            self.summary_calls.append(kwargs)
            start = kwargs["offset"]
            return self.rows[start:start + kwargs["limit"]]

        def get_compression_tip(self, session_id):
            return session_id

        def get_session(self, session_id):
            index = int(session_id.rsplit("-", 1)[1])
            return {"id": session_id, "title": f"History {index}", "started_at": index,
                    "last_active": index}

        def get_message_records(self, session_id):
            self.message_record_calls.append(session_id)
            return [{"role": "user", "content": f"preview {session_id}"}]

    backend = SummaryStore()
    rows = PostgreSQLCLISessionStore(backend).list_sessions_rich(
        source="cli", limit=10, order_by_last_active=True)

    assert backend.summary_calls == [{
        "source": "cli", "exclude_sources": (), "limit": 10, "offset": 0,
        "include_archived": False, "archived_only": False, "include_hidden": False,
        "include_pinned": False,
    }]
    assert backend.message_record_calls == [f"history-{index:03d}" for index in range(50, 40, -1)]
    assert [row["id"] for row in rows] == [f"history-{index:03d}" for index in range(50, 40, -1)]
    assert [row["preview"] for row in rows] == [f"preview history-{index:03d}" for index in range(50, 40, -1)]
