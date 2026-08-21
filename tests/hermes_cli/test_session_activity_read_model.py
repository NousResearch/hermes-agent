from __future__ import annotations

from contextlib import contextmanager

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from hermes_state import SessionDB
from hermes_cli import web_server
from hermes_cli.web_routers import sessions as session_routes


class _FakeSessionDB:
    opened: list[tuple[str | None, bool]] = []
    closed = 0
    message_call: dict | None = None

    def resolve_session_id(self, session_id: str):
        return "session-root" if session_id == "short" else None

    def resolve_resume_session_id(self, session_id: str):
        assert session_id == "session-root"
        return "session-tip"

    def get_session(self, session_id: str):
        assert session_id == "session-tip"
        return {
            "id": session_id,
            "source": "telegram",
            "started_at": 100.0,
            "last_activity_at": 200.0,
            "ended_at": None,
            "system_prompt": "SYSTEM PROMPT MUST NOT LEAK",
            "model_config": {"api_key": "MODEL CONFIG MUST NOT LEAK"},
            "cwd": "/secret/path",
        }

    def get_activity_messages(self, session_id: str, **kwargs):
        assert session_id == "session-tip"
        type(self).message_call = kwargs
        return [
            {
                "id": 1,
                "role": "user",
                "content": "token ghp_1234567890abcdefghijklmnopqrstuvwxyz",
                "timestamp": 101.0,
                "reasoning": "USER REASONING MUST NOT LEAK",
            },
            {
                "id": 2,
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Visible answer"},
                    {"type": "image_url", "image_url": {"url": "https://secret.invalid"}},
                    {"type": "reasoning", "text": "PRIVATE CHAIN MUST NOT LEAK"},
                    {"type": "tool_result", "content": "PRIVATE RESULT MUST NOT LEAK"},
                    {"type": "unknown", "text": "PRIVATE UNKNOWN MUST NOT LEAK"},
                    {"type": "text", "text": "x" * 600},
                ],
                "timestamp": 102.0,
                "tool_calls": [{"function": {"name": "terminal", "arguments": "secret"}}],
                "reasoning_content": "PRIVATE REASONING MUST NOT LEAK",
            },
            {
                "id": 3,
                "role": "tool",
                "content": "TOOL OUTPUT MUST NOT LEAK",
                "timestamp": 103.0,
            },
            {
                "id": 4,
                "role": "system",
                "content": "SYSTEM MESSAGE MUST NOT LEAK",
                "timestamp": 104.0,
            },
            {"id": 5, "role": "assistant", "content": "", "timestamp": 105.0},
            {
                "id": 6,
                "role": "user",
                "content": (
                    "POSIX /home/alice/private/project "
                    "WINDOWS C:\\Users\\Alice\\secret.txt "
                    "HOME ~/private/key FILE file:///root/private/key"
                ),
                "timestamp": 106.0,
            },
        ]

    def close(self):
        type(self).closed += 1


def _client(monkeypatch) -> TestClient:
    _FakeSessionDB.opened = []
    _FakeSessionDB.closed = 0
    _FakeSessionDB.message_call = None

    def _open(profile):
        _FakeSessionDB.opened.append((profile, True))
        return _FakeSessionDB()

    monkeypatch.setattr(web_server, "_open_session_db_strict_read_only", _open)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_session_activity_capability_discovery(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/api/sessions/capabilities")

    assert response.status_code == 200
    assert response.json() == {
        "capabilities": ["sessions.activity.read_model.v1"]
    }
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-hermes-session-capabilities"] == (
        "sessions.activity.read_model.v1"
    )
    assert _FakeSessionDB.opened == []


def test_session_activity_openapi_publishes_closed_response_schemas():
    schema = web_server.app.openapi()
    capability_schema = schema["paths"]["/api/sessions/capabilities"]["get"][
        "responses"
    ]["200"]["content"]["application/json"]["schema"]
    activity_schema = schema["paths"]["/api/sessions/{session_id}/activity"][
        "get"
    ]["responses"]["200"]["content"]["application/json"]["schema"]
    head_response = schema["paths"]["/api/sessions/{session_id}/activity"][
        "head"
    ]["responses"]["200"]

    assert capability_schema["$ref"].endswith(
        "/SessionActivityCapabilitiesResponse"
    )
    assert activity_schema["$ref"].endswith("/SessionActivityResponse")
    assert "content" not in head_response

    components = schema["components"]["schemas"]
    activity = components["SessionActivityResponse"]
    assert activity["additionalProperties"] is False
    assert set(activity["properties"]) == {
        "schema",
        "capabilities",
        "requested_session_id",
        "session",
        "messages",
        "window",
    }
    assert set(activity["required"]) == set(activity["properties"])


def test_session_activity_snapshot_is_bounded_sanitized_and_read_only(monkeypatch):
    client = _client(monkeypatch)

    response = client.get(
        "/api/sessions/short/activity?profile=default&limit=10"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "sessions.activity.read_model.v1"
    assert body["capabilities"] == ["sessions.activity.read_model.v1"]
    assert body["requested_session_id"] == "session-root"
    assert body["session"] == {
        "session_id": "session-tip",
        "profile_id": "default",
        "source": "telegram",
        "started_at": 100.0,
        "last_activity_at": 200.0,
        "ended_at": None,
        "state": "open",
    }
    assert body["window"] == {
        "limit": 10,
        "order": "latest",
        "source_rows": 6,
        "returned": 3,
    }
    assert [message["role"] for message in body["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert [set(message) for message in body["messages"]] == [
        {"message_id", "role", "content", "occurred_at", "truncated"},
        {"message_id", "role", "content", "occurred_at", "truncated"},
        {"message_id", "role", "content", "occurred_at", "truncated"},
    ]
    sanitized = body["messages"][0]["content"]
    secret = "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
    assert secret not in sanitized
    assert sanitized.startswith("token ")
    assert len(sanitized) < len(f"token {secret}")
    assert body["messages"][0]["truncated"] is False
    assert body["messages"][1]["content"].startswith("Visible answer\n")
    assert len(body["messages"][1]["content"]) == 500
    assert body["messages"][1]["truncated"] is True
    assert body["messages"][2]["content"] == "[REDACTED: path-like content]"

    serialized = response.text
    for forbidden in (
        "SYSTEM PROMPT MUST NOT LEAK",
        "MODEL CONFIG MUST NOT LEAK",
        "/secret/path",
        "PRIVATE REASONING MUST NOT LEAK",
        "PRIVATE CHAIN MUST NOT LEAK",
        "PRIVATE RESULT MUST NOT LEAK",
        "PRIVATE UNKNOWN MUST NOT LEAK",
        "USER REASONING MUST NOT LEAK",
        "TOOL OUTPUT MUST NOT LEAK",
        "SYSTEM MESSAGE MUST NOT LEAK",
        "terminal",
        "https://secret.invalid",
        "/home/alice/private/project",
        "C:\\Users\\Alice\\secret.txt",
        "~/private/key",
        "file:///root/private/key",
    ):
        assert forbidden not in serialized

    assert _FakeSessionDB.opened == [("default", True)]
    assert _FakeSessionDB.closed == 1
    assert _FakeSessionDB.message_call == {
        "limit": 10,
        "content_bytes": 8192,
    }
    assert response.headers["cache-control"] == "private, no-store"


def test_session_activity_head_runs_same_authorized_read_without_body(monkeypatch):
    client = _client(monkeypatch)

    get_response = client.get("/api/sessions/short/activity?limit=7")
    response = client.head("/api/sessions/short/activity?limit=7")

    assert response.status_code == 200
    assert response.content == b""
    assert response.headers["content-length"] == get_response.headers["content-length"]
    assert response.headers["content-type"] == get_response.headers["content-type"]
    assert response.headers["x-hermes-session-capabilities"] == (
        "sessions.activity.read_model.v1"
    )
    assert _FakeSessionDB.opened == [(None, True), (None, True)]
    assert _FakeSessionDB.message_call["limit"] == 7
    assert _FakeSessionDB.closed == 2


def test_session_activity_rejects_unknown_duplicate_and_out_of_range_query(monkeypatch):
    client = _client(monkeypatch)

    unknown = client.get("/api/sessions/short/activity?unexpected=1")
    duplicate = client.get("/api/sessions/short/activity?limit=1&limit=2")
    out_of_range = client.get("/api/sessions/short/activity?limit=201")

    assert unknown.status_code == 400
    assert duplicate.status_code == 400
    assert out_of_range.status_code == 422
    assert _FakeSessionDB.opened == []


def test_session_activity_returns_404_and_closes_database(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/api/sessions/missing/activity")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session not found"
    assert _FakeSessionDB.opened == [(None, True)]
    assert _FakeSessionDB.closed == 1


def test_session_activity_redaction_failure_is_fail_closed(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(session_routes, "redact_sensitive_text", _raise)

    message = session_routes._session_activity_message(
        {
            "id": 9,
            "role": "assistant",
            "content": "secret material",
            "timestamp": 300.0,
        }
    )

    assert message == {
        "message_id": "9",
        "role": "assistant",
        "content": "[REDACTED: content unavailable]",
        "occurred_at": 300.0,
        "truncated": False,
    }


def test_session_activity_redaction_preserves_no_secret_fragments():
    secret = "ghp_" + ("A" * 24) + "TAIL"

    message = session_routes._session_activity_message(
        {
            "id": 10,
            "role": "assistant",
            "content": f"token {secret}",
            "timestamp": 301.0,
        }
    )

    assert message is not None
    assert secret not in message["content"]
    assert secret[:6] not in message["content"]
    assert secret[-4:] not in message["content"]
    assert "«redacted:ghp_…»" in message["content"]


@pytest.mark.parametrize(
    "content",
    [
        "/workspace",
        "/data",
        "/custom/file",
        "relative/private.txt",
        "./secret/file",
        "../secret/file",
        "path:/home/alice/private",
        "/home/alice/My Project/key.txt",
        r"C:\Users\Alice\secret.txt",
        r"\\server\share\secret.txt",
        "file:///root/key",
        "https://example.com/private/path",
    ],
)
def test_path_like_content_fails_closed(content):
    assert session_routes._session_activity_redact_paths(content) == (
        "[REDACTED: path-like content]"
    )


def test_plain_public_text_is_not_changed_by_path_redaction():
    assert session_routes._session_activity_redact_paths("public answer") == (
        "public answer"
    )


def test_activity_projection_never_reads_oversized_blob(monkeypatch):
    class _OversizedBlob:
        closed = False

        def __len__(self):
            return 9000

        def read(self, _size=-1):
            raise AssertionError("oversized content must not be read")

        def close(self):
            self.closed = True

    blob = _OversizedBlob()

    class _Connection:
        def execute(self, _sql, _params):
            return self

        def fetchall(self):
            return [
                {
                    "id": 1,
                    "role": "assistant",
                    "timestamp": 1.0,
                    "content_is_null": 0,
                }
            ]

        def blobopen(self, table, column, rowid, *, readonly):
            assert (table, column, rowid, readonly) == (
                "messages",
                "content",
                1,
                True,
            )
            return blob

    @contextmanager
    def _read_ctx():
        yield _Connection()

    db = SessionDB.__new__(SessionDB)
    monkeypatch.setattr(db, "_read_ctx", _read_ctx)

    rows = db.get_activity_messages("session", content_bytes=8192)

    assert rows == [
        {
            "id": 1,
            "role": "assistant",
            "content": None,
            "content_oversized": 1,
            "timestamp": 1.0,
        }
    ]
    assert blob.closed is True


def test_session_activity_omits_unknown_content_shapes():
    message = session_routes._session_activity_message(
        {
            "id": 10,
            "role": "user",
            "content": {"image_url": "https://private.invalid/image.png"},
            "timestamp": 301.0,
        }
    )

    assert message is None


def test_session_activity_omits_messages_without_explicit_id():
    message = session_routes._session_activity_message(
        {
            "role": "assistant",
            "content": "answer without evidence id",
            "timestamp": 302.0,
        }
    )

    assert message is None


def test_session_activity_reads_real_sqlite_without_mutating_it(monkeypatch, tmp_path):
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session("real-session", "desktop")
    db.append_message(
        "real-session",
        "user",
        "visible token in /home/alice/private/project",
    )
    db.append_message(
        "real-session",
        "tool",
        "private tool output",
        tool_name="terminal",
    )
    db.append_message(
        "real-session",
        "assistant",
        [
            {"type": "text", "text": "public answer"},
            {"type": "reasoning", "text": "private chain"},
            {"type": "tool_result", "content": "private result"},
        ],
    )
    db.append_message("real-session", "assistant", "z" * 9000)
    db.close()

    projection_db = SessionDB(db_path=db_path, read_only=True)
    projected = projection_db.get_activity_messages(
        "real-session",
        limit=10,
        content_bytes=8192,
    )
    projection_db.close()
    assert [set(row) for row in projected] == [
        {"id", "role", "content", "content_oversized", "timestamp"},
        {"id", "role", "content", "content_oversized", "timestamp"},
        {"id", "role", "content", "content_oversized", "timestamp"},
    ]
    assert [row["role"] for row in projected] == ["user", "assistant", "assistant"]
    assert projected[-1]["content"] is None
    assert projected[-1]["content_oversized"] == 1
    before = db_path.stat().st_mtime_ns

    def _open(_profile):
        return SessionDB(db_path=db_path, read_only=True)

    monkeypatch.setattr(web_server, "_open_session_db_strict_read_only", _open)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN

    response = client.get("/api/sessions/real-session/activity?limit=10")

    assert response.status_code == 200
    body = response.json()
    assert [message["role"] for message in body["messages"]] == [
        "user",
        "assistant",
        "assistant",
    ]
    assert "public answer" in response.text
    assert "private tool output" not in response.text
    assert "private chain" not in response.text
    assert "private result" not in response.text
    assert "terminal" not in response.text
    assert "/home/alice/private/project" not in response.text
    assert body["messages"][-1]["content"] == (
        "[REDACTED: message exceeds public read limit]"
    )
    assert body["messages"][-1]["truncated"] is True
    assert db_path.stat().st_mtime_ns == before


def test_strict_read_only_open_does_not_bootstrap_missing_or_empty_store(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        web_server,
        "_cron_profile_home",
        lambda profile: (profile, str(tmp_path)),
    )
    db_path = tmp_path / "state.db"
    initial_entries = {path.name for path in tmp_path.iterdir()}

    with pytest.raises(HTTPException) as missing:
        web_server._open_session_db_strict_read_only("ghost")
    assert missing.value.status_code == 503
    assert not db_path.exists()

    db_path.write_bytes(b"")
    before = db_path.stat().st_mtime_ns
    with pytest.raises(HTTPException) as empty:
        web_server._open_session_db_strict_read_only("ghost")
    assert empty.value.status_code == 503
    assert db_path.stat().st_size == 0
    assert db_path.stat().st_mtime_ns == before
    assert {path.name for path in tmp_path.iterdir()} == initial_entries | {"state.db"}
