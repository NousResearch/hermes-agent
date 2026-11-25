"""Activity privacy and HTTP contracts over real isolated persistence."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest
from fastapi.testclient import TestClient

from agent.think_scrubber import THINK_TAG_NAMES
from hermes_cli import web_server
from hermes_cli.web_routers import sessions as routes
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope


@pytest.fixture
def activity_store(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.create(NewSession("session-root", "desktop"))
        yield store, home / "state.db"


def client():
    result = TestClient(web_server.app)
    result.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return result


def test_capabilities_do_not_open_store(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("discovery must not access a store")
    monkeypatch.setattr(routes, "_open_activity_store", forbidden)
    response = client().get("/api/sessions/capabilities")
    assert response.status_code == 200
    assert response.json() == {"capabilities": ["sessions.activity.read_model.v1"]}
    assert response.headers["cache-control"] == "private, no-store"


def test_real_projection_sanitized_and_bounded(activity_store):
    store, _ = activity_store
    for message in (NewMessage("user", "path /private/key"),
                    NewMessage("tool", "PRIVATE TOOL", tool_name="terminal"),
                    NewMessage("assistant", [{"type": "text", "text": "public answer"},
                        {"type": "reasoning", "text": "PRIVATE REASONING"},
                        {"type": "tool_result", "content": "PRIVATE RESULT"},
                        {"type": "image_url", "image_url": {"url": "https://private.invalid"}}],
                        reasoning="PRIVATE COLUMN"),
                    NewMessage("assistant", "x" * 600), NewMessage("assistant", "z" * 9000)):
        store.messages.append("session-root", message)
    response = client().get("/api/sessions/session-root/activity?limit=200")
    assert response.status_code == 200
    data = response.json()
    assert data["schema"] == "sessions.activity.read_model.v1"
    assert data["requested_session_id"] == "session-root"
    assert data["session"]["source"] == "desktop"
    assert data["window"] == {"limit": 200, "order": "latest", "source_rows": 4, "returned": 4}
    assert [m["role"] for m in data["messages"]] == ["user", "assistant", "assistant", "assistant"]
    assert data["messages"][0]["content"] == "[REDACTED: path-like content]"
    assert data["messages"][1]["content"] == "public answer"
    assert len(data["messages"][2]["content"]) == 500
    assert data["messages"][2]["truncated"] is True
    assert data["messages"][3]["content"] == "[REDACTED: message exceeds public read limit]"
    assert data["messages"][3]["truncated"] is True
    assert "PRIVATE" not in response.text
    assert "terminal" not in response.text
    assert "/private" not in response.text
    for message in data["messages"]:
        assert set(message) == {"message_id", "role", "content", "occurred_at", "truncated"}


@pytest.mark.parametrize("endpoint", ["/api/sessions/capabilities", "/api/sessions/session-root/activity"])
def test_head_representation_headers_match_get(activity_store, endpoint):
    activity_store[0].messages.append("session-root", NewMessage("assistant", "public answer"))
    get, head = client().get(endpoint), client().head(endpoint)
    assert get.status_code == head.status_code == 200
    assert head.content == b""
    for key in ("content-length", "content-type", "cache-control", "x-hermes-session-capabilities"):
        assert get.headers[key] == head.headers[key]


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("query,status", [("?unexpected=1", 400), ("?limit=1&limit=2", 400),
    ("?profile=default&profile=other", 400), ("?limit=201", 422), ("?limit=0", 422)])
def test_snapshot_query_is_closed(method, query, status):
    assert client().request(method, "/api/sessions/session-root/activity" + query).status_code == status


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("endpoint", ["/api/sessions/capabilities", "/api/sessions/session-root/activity"])
def test_activity_requires_existing_auth(method, endpoint):
    assert TestClient(web_server.app).request(method, endpoint).status_code == 401


def test_unknown_session_is_not_found(activity_store):
    assert client().get("/api/sessions/missing/activity").status_code == 404


def test_openapi_models_are_closed():
    schema = web_server.app.openapi()
    components = schema["components"]["schemas"]
    for name in ("SessionActivityResponse", "SessionActivitySessionResponse", "SessionActivityMessageResponse",
                 "SessionActivityCapabilitiesResponse", "SessionActivityWindowResponse"):
        assert components[name]["additionalProperties"] is False
    assert set(components["SessionActivityResponse"]["properties"]) == {
        "schema", "capabilities", "requested_session_id", "session", "messages", "window"}
    for endpoint in ("/api/sessions/{session_id}/activity", "/api/sessions/capabilities"):
        assert "content" not in schema["paths"][endpoint]["head"]["responses"]["200"]


@pytest.mark.parametrize("content", ["/workspace", "relative/private.txt", "./secret", "../secret",
    r"C:\Users\Alice\secret", r"\\server\share", "file:///root/key", "https://example.com/private/path"])
def test_path_like_content_fails_closed(content):
    assert routes._session_activity_redact_paths(content) == "[REDACTED: path-like content]"


def test_plain_public_text_retained():
    assert routes._session_activity_redact_paths("public answer") == "public answer"


def test_secrets_fully_redacted():
    secret = "ghp_" + "A" * 24 + "TAIL"
    message = routes._session_activity_message({"id": 1, "role": "assistant", "content": "token " + secret})
    assert message is not None
    assert secret[:6] not in message["content"]
    assert secret[-4:] not in message["content"]


_OPAQUE = "opaqueSecretValue12345"


@pytest.mark.parametrize("content", [
    f"my DB_PASSWORD={_OPAQUE} ok",
    '{"api_key": "' + _OPAQUE + '"}',
    f"api_key: {_OPAQUE}",
    f"export MY_SERVICE_TOKEN={_OPAQUE}",
])
def test_opaque_assignment_values_become_non_reusable_sentinels(content):
    """A prefix-less credential assigned to a credential-shaped key is masked, never echoed or head/tail-kept."""
    message = routes._session_activity_message({"id": 1, "role": "user", "content": content})
    assert message is not None
    assert _OPAQUE not in message["content"]
    assert _OPAQUE[:6] not in message["content"] and _OPAQUE[-4:] not in message["content"]
    assert "«redacted" in message["content"]


_OPAQUE_BEARER = "q7Zr2LmX9vKp4TnW8sYc3HdJ6fBg1NaE5uRo0iQt"


@pytest.mark.parametrize("content", [
    'headers = {"Authorization": "Bearer ' + _OPAQUE_BEARER + '"}',
    '{"headers": {"Authorization": "Bearer ' + _OPAQUE_BEARER + '"}}',
    f"my token: Bearer {_OPAQUE_BEARER}",
    "fetch(url, {headers: {Authorization: `Bearer " + _OPAQUE_BEARER + "`}})",
    "axios.get(url, {headers: {'Authorization': 'Bearer " + _OPAQUE_BEARER + "'}})",
    f"Authorization: Bearer {_OPAQUE_BEARER}",
])
@pytest.mark.parametrize("role", ["user", "assistant"])
def test_opaque_bearer_tokens_never_survive(content, role):
    """A prefix-less ``Bearer <opaque>`` credential carries no vendor shape; no byte of it may survive."""
    message = routes._session_activity_message({"id": 1, "role": role, "content": content})
    assert message is not None
    assert _OPAQUE_BEARER[:6] not in message["content"]
    assert _OPAQUE_BEARER[-4:] not in message["content"]


def test_opaque_assignment_redacted_over_http(activity_store):
    store, _ = activity_store
    store.messages.append("session-root", NewMessage("user", f"my DB_PASSWORD={_OPAQUE} ok"))
    response = client().get("/api/sessions/session-root/activity")
    assert response.status_code == 200
    assert _OPAQUE[:6] not in response.text and _OPAQUE[-4:] not in response.text


def test_redactor_failure_fails_closed(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(routes, "redact_sensitive_text", broken)
    message = routes._session_activity_message({"id": 1, "role": "user", "content": "secret"})
    assert message["content"] == "[REDACTED: content unavailable]"


@pytest.mark.parametrize("content", [{"image_url": "private"}, [{"type": "unknown", "text": "private"}]])
def test_unknown_content_omitted(content):
    assert routes._session_activity_message({"id": 1, "role": "user", "content": content}) is None


@pytest.mark.parametrize("state", ["missing", "empty", "corrupt", "old-schema"])
def test_snapshot_never_bootstraps_or_heals(tmp_path, monkeypatch, state):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    path = home / "state.db"
    if state == "empty":
        path.write_bytes(b"")
    elif state == "corrupt":
        path.write_bytes(b"not a sqlite database")
    elif state == "old-schema":
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    before = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
              for p in home.iterdir() if p.is_file()}
    response = client().get("/api/sessions/session-root/activity")
    assert response.status_code == 503
    after = {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
             for p in home.iterdir() if p.is_file()}
    assert before == after
    assert str(home) not in response.text


def test_storage_activity_caps_rows_and_bytes(activity_store):
    store, path = activity_store
    for i in range(202):
        store.messages.append("session-root", NewMessage("assistant", str(i)))
    store.messages.append("session-root", NewMessage("assistant", "z" * 8193))
    with open_persistence_reader(path) as reader:
        rows = reader.messages.read_activity("session-root", limit=900)
    assert len(rows) == 200
    assert rows[0].content == "3"
    assert rows[-1].content is None
    assert rows[-1].content_oversized is True


@pytest.mark.parametrize("content", ['path="/root/private"', '`/root/private`', "key=/root/private", "x,/root/private"])
def test_delimited_paths_fail_closed(content):
    assert routes._session_activity_redact_paths(content) == "[REDACTED: path-like content]"


def test_metadata_cannot_bypass_sanitizer(activity_store):
    store, _ = activity_store
    secret = "ghp_" + "B" * 24 + "TAIL"
    store.sessions.create(NewSession(secret, "path=/root/private"))
    response = client().get(f"/api/sessions/{secret}/activity")
    assert response.status_code == 200
    assert secret[:6] not in response.text
    assert secret[-4:] not in response.text
    assert "/root/private" not in response.text


@pytest.mark.parametrize("content", [[{"type": [], "text": "private"}], [{"type": {}, "text": "private"}]])
def test_malformed_block_types_are_omitted(content):
    assert routes._session_activity_message({"id": 1, "role": "assistant", "content": content}) is None


@pytest.mark.parametrize("tag", ["<think>", "<analysis>", "<reasoning>"])
def test_inline_reasoning_fails_closed(tag):
    message = routes._session_activity_message({"id": 1, "role": "assistant", "content": tag + "PRIVATE REASONING"})
    assert "PRIVATE REASONING" not in message["content"]


_REASONING_TAGS = (*THINK_TAG_NAMES, "analysis")


@pytest.mark.parametrize("name", _REASONING_TAGS)
@pytest.mark.parametrize("shape", ["{open}PRIVATE REASONING", "PRIVATE REASONING{close}public answer"])
def test_every_reasoning_tag_fails_closed(name, shape):
    """Open-only and orphan-close reasoning of every shared tag name redacts as reasoning, not as a path."""
    content = shape.format(open=f"<{name}>", close=f"</{name}>")
    message = routes._session_activity_message({"id": 1, "role": "assistant", "content": content})
    assert message is not None
    assert message["content"] == "[REDACTED: reasoning content]"


def test_hidden_scaffolds_and_compaction_never_project(activity_store):
    store, _ = activity_store
    for message in (NewMessage("user", "PRIVATE HIDDEN", display_kind="hidden"),
                    NewMessage("assistant", "PRIVATE SUMMARY", compressed_summary=True),
                    NewMessage("assistant", "public answer")):
        store.messages.append("session-root", message)
    response = client().get("/api/sessions/session-root/activity")
    assert response.status_code == 200
    assert "PRIVATE" not in response.text
    assert [m["content"] for m in response.json()["messages"]] == ["public answer"]


_LEGACY_SYNTHETIC_ROWS = (
    ("user", "[IMPORTANT: Background process proc_1 completed (exit code 0).\nCommand: echo PRIVATE-BG-CMD\n"
             "Output:\nPRIVATE-BG-OUTPUT]"),
    ("user", "[ASYNC DELEGATION COMPLETE — d1]\nPRIVATE-DELEGATION-RESULT"),
    ("user", "A background subagent you dispatched earlier has finished. PRIVATE-SUBAGENT"),
    ("user", "[System: The active model for this chat has changed to PRIVATE-MODEL.]"),
    ("user", "[System note: Your previous turn was interrupted mid-run — the app stopped. "
             "The interrupted request was:]\n\nPRIVATE-AUTO-CONTINUE"),
    ("user", "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted. PRIVATE-LEGACY-SUMMARY"),
    ("user", "[CONTEXT SUMMARY]: PRIVATE-OLDEST-SUMMARY"),
    ("user", "[Your active task list was preserved across context compression]\n- PRIVATE-TODO"),
    ("assistant", "[CONTEXT COMPACTION — REFERENCE ONLY] PRIVATE-ASSISTANT-SUMMARY"),
)


def _append_legacy_untyped(store, path, rows):
    for role, content in rows:
        store.messages.append("session-root", NewMessage(role, content))
    # Rows written before display typing / the _compressed_summary column carry neither.
    with sqlite3.connect(path) as db:
        db.execute("UPDATE messages SET display_kind = NULL, _compressed_summary = 0")


def test_legacy_untyped_synthetic_rows_never_project(activity_store):
    store, path = activity_store
    _append_legacy_untyped(store, path, (("user", "public question"), *_LEGACY_SYNTHETIC_ROWS,
                                         ("assistant", "public answer")))
    response = client().get("/api/sessions/session-root/activity")
    assert response.status_code == 200
    assert "PRIVATE" not in response.text
    assert [(m["role"], m["content"]) for m in response.json()["messages"]] == [
        ("user", "public question"), ("assistant", "public answer")]


def test_store_activity_read_excludes_legacy_untyped_synthetic_rows(activity_store):
    store, path = activity_store
    _append_legacy_untyped(store, path, (*_LEGACY_SYNTHETIC_ROWS, ("user", "public question")))
    with open_persistence_reader(path) as reader:
        rows = reader.messages.read_activity("session-root", limit=200)
    assert [(row.role, row.content) for row in rows] == [("user", "public question")]


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("query", ["?unexpected=secret", "?profile=default&profile=other", "?limit=1"])
def test_discovery_query_is_closed(method, query):
    assert client().request(method, "/api/sessions/capabilities" + query).status_code == 400


@pytest.mark.parametrize("query", ["?limit=ghp_" + "C" * 24 + "TAIL", "?ghp_" + "C" * 24 + "TAIL=1"])
def test_invalid_query_does_not_echo_secret(query):
    response = client().get("/api/sessions/session-root/activity" + query)
    assert response.status_code in (400, 422)
    assert "ghp_" not in response.text
    assert "TAIL" not in response.text


@pytest.mark.parametrize("raw", ["%EF%BC%95", "%D9%A5", "1_0", "%2B5", "%205", "5%20", "0x5"])
def test_limit_accepts_only_ascii_decimal_digits(raw):
    """``int()`` accepts Unicode digits, underscores, signs and padding; the snapshot does not, and never echoes them."""
    response = client().get("/api/sessions/session-root/activity?limit=" + raw)
    assert response.status_code == 422
    assert response.json() == {"detail": "invalid_limit"}


@pytest.fixture
def storeless_home(tmp_path, monkeypatch):
    home = tmp_path / "storeless"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("endpoint,status", [
    ("/api/sessions/session-root/activity?bad=1", 400), ("/api/sessions/session-root/activity?limit=0", 422),
    ("/api/sessions/missing/activity", 404), ("/api/sessions/capabilities?bad=1", 400)])
def test_error_responses_are_not_cacheable(activity_store, method, endpoint, status):
    response = client().request(method, endpoint)
    assert response.status_code == status
    assert response.headers.get("cache-control") == "private, no-store"


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
def test_reserved_capability_mutation_refusal_is_not_cacheable(method):
    response = client().request(method, "/api/sessions/capabilities")
    assert response.status_code == 405
    assert response.headers.get("allow") == "GET, HEAD"
    assert response.headers.get("cache-control") == "private, no-store"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_unavailable_store_response_is_not_cacheable(storeless_home, method):
    response = client().request(method, "/api/sessions/session-root/activity")
    assert response.status_code == 503
    assert response.headers.get("cache-control") == "private, no-store"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_profile_error_response_is_not_cacheable(method):
    response = client().request(method, "/api/sessions/session-root/activity?profile=no-such-profile-zz")
    assert response.status_code in (400, 404)
    assert response.headers.get("cache-control") == "private, no-store"


@pytest.mark.parametrize("method", ["GET", "HEAD"])
def test_profile_errors_do_not_echo_private_names(method):
    response = client().request(method, "/api/sessions/session-root/activity?profile=ghp_" + "c" * 24 + "tail")
    assert response.status_code in (400, 404)
    assert "ghp_" not in response.text
    assert "c" * 12 not in response.text


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"])
@pytest.mark.parametrize("endpoint", ["/api/sessions/capabilities", "/api/sessions/session-root/activity"])
def test_activity_method_allowlist_is_closed(method, endpoint):
    response = client().request(method, endpoint)
    assert response.status_code == 405


def test_malformed_message_timestamp_is_not_an_egress_field(activity_store):
    store, path = activity_store
    store.messages.append("session-root", NewMessage("assistant", "public answer"))
    secret = "ghp_" + "D" * 24 + "TAIL"
    with sqlite3.connect(path) as db:
        db.execute("UPDATE messages SET timestamp = ?", (secret,))
    response = client().get("/api/sessions/session-root/activity")
    assert response.status_code == 200
    assert response.json()["messages"][0]["occurred_at"] is None
    assert secret not in response.text


def test_activity_profile_selection_never_reads_launch_profile(activity_store):
    launch, path = activity_store
    launch.messages.append("session-root", NewMessage("assistant", "launch answer"))
    home = path.parent / "profiles" / "alpha"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("{}\n")
    with open_persistence(profile_scope(home=home)) as other:
        other.sessions.create(NewSession("session-root", "desktop"))
        other.messages.append("session-root", NewMessage("assistant", "alpha answer"))
    response = client().get("/api/sessions/session-root/activity?profile=alpha")
    assert response.status_code == 200
    assert response.json()["session"]["profile_id"] == "alpha"
    assert [m["content"] for m in response.json()["messages"]] == ["alpha answer"]
    assert "launch answer" not in response.text
