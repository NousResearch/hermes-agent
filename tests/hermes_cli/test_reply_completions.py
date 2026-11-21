"""The completion feed uses dashboard auth, exact profile scope, and a forward-only cursor."""

import hashlib
import json
import secrets
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope


def _seed_replies(home, report_id):
    with open_persistence(profile_scope(home=home)) as store:
        for session_id in ("silent-chat", "visible-chat", "read-chat", report_id):
            store.sessions.create(NewSession(session_id=session_id, source="desktop"))
        epoch = store.runtime.begin_epoch(instance_id="test-owner")
        for session_id, response in (
            ("silent-chat", "[SILENT]"),
            ("visible-chat", "A private reply"),
            ("read-chat", "An already-read reply"),
        ):
            store.admissions.admit(epoch=epoch, principal_id="person", session_id=session_id,
                                   request_id="prompt", payload={"text": "Should this notify?"})
            claim = store.admissions.claim(epoch=epoch, session_id=session_id)
            store.admissions.settle(
                epoch=epoch, admission_id=claim.admission_id, generation=claim.generation,
                outcome="completed", result={"result": {"final_response": response, "messages": []}},
            )
        store.sessions.set_read("read-chat")
        store.sessions.end("visible-chat", "compression")
        store.sessions.create(NewSession(session_id="continuation", source="desktop", parent_session_id="visible-chat"))
        store.messages.append_completion_note(report_id, "report", "A private report", "internal_notification",
                                              {"completion_note": ["cron:job:run"]})


@pytest.mark.parametrize("gated", [False, True])
def test_reply_feed_is_authenticated_profile_scoped_and_skips_silence(tmp_path, monkeypatch, gated):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    research = home / "profiles" / "research"
    research.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_replies(home, "default-report")
    _seed_replies(research, "research-report")

    from hermes_cli.web_routers.reply_completions import register_auth
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    register_auth()
    monkeypatch.setattr(app.state, "auth_required", gated, raising=False)
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    grant = {"profile": "research", "expires_at": int(time.time()) + 60}
    grants_file = tmp_path / "reply-grants.json"
    grants_file.write_text(json.dumps({digest: grant}), encoding="utf-8")
    monkeypatch.setenv("HERMES_REPLY_FEED_GRANTS_FILE", str(grants_file))

    endpoint = "/api/sessions/reply-completions"
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get(endpoint, params={"profile": "research"}).status_code == 401
        assert client.get(endpoint, params={"profile": "research"},
                          headers={_SESSION_HEADER_NAME: _SESSION_TOKEN}).status_code == 401
        client.headers["Authorization"] = f"Bearer {token}"
        assert client.get("/api/sessions", params={"profile": "research"}).status_code == 401
        assert client.post("/api/gateway/drain", json={"action": "drain"}).status_code == 401
        first = client.get(endpoint, params={"profile": "research", "limit": 1})
        assert first.status_code == 200
        assert first.headers["Cache-Control"] == "no-store"
        page = first.json()
        assert page == {"profile": "research", "events": [], "next_cursor": 1, "has_more": True}

        second = client.get(endpoint, params={"profile": "research", "after": page["next_cursor"]})
        assert second.status_code == 200
        events = second.json()["events"]
        assert all(set(event) == {"sequence", "session_id", "conversation_id", "completed_at", "kind"} for event in events)
        assert [(event["sequence"], event["session_id"], event["conversation_id"], event["kind"]) for event in events] == [
            (2, "continuation", "visible-chat", "turn"), (4, "research-report", "research-report", "note"),
        ]
        assert second.json()["next_cursor"] == 4
        assert "A private reply" not in second.text
        assert "A private report" not in second.text
        default = client.get(endpoint, params={"profile": "default"})
        assert default.status_code == 403
        assert client.get(endpoint, params=[("profile", "default"), ("profile", "research")]).status_code == 403
        assert client.get(endpoint, params={"profile": "research", "after": -1}).status_code == 422
        assert client.get(endpoint).status_code == 422
        grants_file.write_text(json.dumps({digest: {**grant, "expires_at": 0}}), encoding="utf-8")
        assert client.get(endpoint, params={"profile": "research"}).status_code == 401
        grants_file.unlink()
        assert client.get(endpoint, params={"profile": "research"}).status_code == 401
