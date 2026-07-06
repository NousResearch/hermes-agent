import asyncio

from hermes_cli import web_server
from hermes_state import SessionDB


def test_get_messages_include_compacted_keeps_compression_history_but_not_rewound(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("telegram_session", source="telegram")
    db.append_message("telegram_session", role="user", content="original prompt")
    db.append_message("telegram_session", role="assistant", content="original answer")

    db.archive_and_compact(
        "telegram_session",
        [
            {"role": "user", "content": "current prompt"},
            {"role": "assistant", "content": "current answer"},
        ],
    )
    rewound_id = db.append_message("telegram_session", role="user", content="undone prompt")
    db._execute_write(
        lambda conn: conn.execute(
            "UPDATE messages SET active = 0, compacted = 0 WHERE id = ?",
            (rewound_id,),
        )
    )

    live = [m["content"] for m in db.get_messages("telegram_session")]
    history = [
        m["content"]
        for m in db.get_messages("telegram_session", include_compacted=True)
        if m["role"] in {"user", "assistant"}
    ]
    audit = [m["content"] for m in db.get_messages("telegram_session", include_inactive=True)]

    assert live == ["current prompt", "current answer"]
    assert history == [
        "original prompt",
        "original answer",
        "current prompt",
        "current answer",
    ]
    assert "undone prompt" in audit
    assert "undone prompt" not in history

    db.close()


class _FakeDB:
    def __init__(self):
        self.calls = []
        self.closed = False

    def resolve_session_id(self, session_id):
        return f"resolved-{session_id}"

    def resolve_resume_session_id(self, session_id):
        return f"tip-{session_id}"

    def get_messages(self, session_id, **kwargs):
        self.calls.append((session_id, kwargs))
        return [{"role": "user", "content": "hello"}]

    def close(self):
        self.closed = True


def test_web_session_messages_endpoint_accepts_compacted_history_flag(monkeypatch):
    fake = _FakeDB()
    monkeypatch.setattr(web_server, "_open_session_db_for_profile", lambda profile=None: fake)

    response = asyncio.run(
        web_server.get_session_messages("abc", include_compacted=True)
    )

    assert response == {
        "session_id": "tip-resolved-abc",
        "messages": [{"role": "user", "content": "hello"}],
    }
    assert fake.calls == [
        (
            "tip-resolved-abc",
            {"include_inactive": False, "include_compacted": True},
        )
    ]
    assert fake.closed is True
