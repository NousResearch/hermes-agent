"""Stored chat search and exact bounded jumps, including compression history (#132677)."""

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    homes = {"default": tmp_path / ".hermes", "other": tmp_path / ".hermes/profiles/other"}
    databases = {}
    for name, home in homes.items():
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test/model\n")
        databases[name] = SessionDB(db_path=home / "state.db")
    monkeypatch.setenv("HERMES_HOME", str(homes["default"]))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", homes["default"] / "state.db")
    from hermes_cli.web_routers.sessions import manage_router
    app = FastAPI()
    app.include_router(manage_router)
    with TestClient(app) as client:
        yield databases, homes, client
    for db in databases.values():
        db.close()


@pytest.mark.parametrize("launch", ["default", "other"])
@pytest.mark.parametrize("query", ["needle", "记忆", '"two words"', "needle OR 记忆"])
def test_search_and_jump_share_the_whole_owned_display_transcript(stores, monkeypatch, launch, query):
    databases, homes, client = stores
    monkeypatch.setenv("HERMES_HOME", str(homes[launch]))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", homes[launch] / "state.db")
    for name, db in databases.items():
        db.create_session(session_id="root", source="desktop", profile_name=name)
        db.append_messages_batch("root", [
            {"role": "user", "content": f"{name} needle 记忆 two words archived", "timestamp": 1},
            {"role": "assistant", "content": f"{name} needle 记忆 two words answer", "timestamp": 2},
        ])
        db.end_session("root", end_reason="compression")
        db.create_session(session_id="tip", source="desktop", parent_session_id="root", profile_name=name)
        db.append_messages_batch("tip", [
            {"role": "user", "content": f"{name} recent prompt", "timestamp": 3},
            {"role": "assistant", "content": f"{name} recent answer", "timestamp": 4},
            {"role": "tool", "content": "x" * 80000 + f" {name} needle 记忆 two words tool tail", "timestamp": 5},
        ])
        db.archive_and_compact("tip", db.get_messages("tip"))
        db.create_session(session_id="unrelated", source="desktop", profile_name=name)
        db.append_message("unrelated", role="user", content="needle 记忆 two words unrelated")
        db.create_session(session_id="child", source="tool", parent_session_id="tip", profile_name=name)
        db.append_message("child", role="user", content="needle 记忆 two words child")
        hidden = db.append_message("tip", role="user", content="needle 记忆 two words hidden", display_kind="hidden")
        rewound = db.append_message("tip", role="assistant", content="needle 记忆 two words rewound")
        db._write_sql("UPDATE messages SET active = 0, compacted = 0 WHERE id = ?", (rewound,))
        assert hidden != rewound
    for name in ("default", "other", "default"):
        hits = []
        offset = 0
        while True:
            response = client.get("/api/sessions/root/messages/search", params={
                **({"profile": name} if name != launch else {}), "q": query, "limit": 1, "offset": offset})
            assert response.status_code == 200, response.text
            page = response.json()
            assert page["session_id"] == "tip" and page["profile"] == name
            hits.extend(page["results"])
            if not page["pagination"]["has_more"]:
                break
            offset = page["pagination"]["next_offset"]
        assert len(hits) == 3
        assert len({hit["row_id"] for hit in hits}) == 3
        assert all(name in hit["snippet"] for hit in hits)
        assert all("unrelated" not in hit["snippet"] and "child" not in hit["snippet"] for hit in hits)
        assert len(response.content) < 10000
        for hit in hits:
            jump = client.get("/api/sessions/root/messages/match", params={
                **({"profile": name} if name != launch else {}), "row_id": hit["row_id"], "limit": 2})
            assert jump.status_code == 200, jump.text
            payload = jump.json()
            assert len(payload["messages"]) <= 2
            assert any(message["id"] == hit["row_id"] for message in payload["messages"])
            assert payload["profile"] == name
            assert payload["pagination"]["total"] == 5
        assert client.get("/api/sessions/root/messages/search", params={
            "profile": name, "q": "absent phrase"}).json()["results"] == []


@pytest.mark.parametrize("legacy", [False, True])
def test_search_uses_current_representatives_and_rejects_foreign_jump_rows(stores, legacy):
    databases, _, client = stores
    db = databases["default"]
    db.create_session(session_id="chat", source="desktop", profile_name="default")
    db.append_messages_batch("chat", [
        {"role": "user", "content": "retained needle", "timestamp": 1},
        {"role": "assistant", "content": "needle answer", "timestamp": 2},
        *[{"role": "assistant", "content": f"step {i}", "timestamp": i + 3} for i in range(150)],
    ])
    original = db.get_messages("chat")[0]["id"]
    db.archive_and_compact("chat", db.get_messages("chat"))
    from agent.context_compressor import SUMMARY_PREFIX, _SUMMARY_END_MARKER
    db.append_messages_batch("chat", [
        {"role": "user", "content": SUMMARY_PREFIX + "unseen-handoff", "_compressed_summary": True},
        {"role": "user", "content": SUMMARY_PREFIX + "unseen-handoff" + _SUMMARY_END_MARKER + "\nlive needle ask",
         "display_kind": "hidden", "_compressed_summary": True},
        {"role": "user", "content": [
            {"type": "text", "text": "needle caption"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,binary-only-data"}},
        ]},
    ])
    if legacy:
        db._write_sql("UPDATE messages SET display_identity = NULL, display_order = NULL WHERE session_id = ?", ("chat",))
    response = client.get("/api/sessions/chat/messages/search?q=needle")
    assert response.status_code == 200, response.text
    hits = response.json()["results"]
    assert len(hits) == 4 and hits[0]["row_id"] != original
    assert client.get("/api/sessions/chat/messages/search?q=unseen-handoff").json()["results"] == []
    assert client.get("/api/sessions/chat/messages/search?q=binary-only-data").json()["results"] == []
    live = client.get("/api/sessions/chat/messages/search?q=live").json()["results"]
    assert len(live) == 1 and "live needle ask" in live[0]["snippet"]
    current = hits[0]["row_id"]
    jump = client.get(f"/api/sessions/chat/messages/match?row_id={original}")
    assert jump.status_code == 200
    payload = jump.json()
    assert payload["messages"][0]["id"] == current
    assert len(payload["messages"]) == 120
    assert payload["pagination"]["has_newer"]
    db.create_session(session_id="foreign", source="desktop", profile_name="default")
    foreign = db.append_message("foreign", role="user", content="foreign needle")
    assert client.get(f"/api/sessions/chat/messages/match?row_id={foreign}").status_code == 404
    assert client.get("/api/sessions/missing/messages/search?q=needle").status_code == 404
    assert client.get("/api/sessions/chat/messages/search?limit=101&q=needle").status_code == 422
    assert client.get(f"/api/sessions/chat/messages/match?row_id={current}&limit=121").status_code == 422
