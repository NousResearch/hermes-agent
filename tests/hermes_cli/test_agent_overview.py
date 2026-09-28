"""Real SQLite + HTTP contract for read-only Desktop fleet discovery."""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_state import SessionDB
from hermes_cli.web_routers import profiles


def test_unbilled_history_and_canonical_provider_come_from_their_own_config(tmp_path, monkeypatch):
    with SessionDB(tmp_path / 'state.db') as db:
        db.create_session(session_id='ordinary', source='desktop', model='model-a', model_config={'provider': 'openai', 'api_key': 'SECRET'})
        db.create_session(session_id='bot', source='desktop', model='model-b', model_config={'provider': 'anthropic', 'api_key': 'SECRET'})
        db.set_session_title('bot', 'Bot Chat')
        db.set_session_hidden('bot', True)
    monkeypatch.setattr(profiles, '_profile_targets', lambda *a, **k: [('worker', tmp_path)])
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    body = TestClient(app).get('/api/profiles/agent-overview').json()
    assert body['sessions'][0]['provider'] == 'openai'
    assert body['canonical'][0]['provider'] == 'anthropic'
    assert 'SECRET' not in str(body)


def test_overview_pages_past_recency_cap_and_keeps_canonical_chat(tmp_path, monkeypatch):
    home = tmp_path / "worker"
    home.mkdir()
    db = SessionDB(home / "state.db")
    try:
        for n in range(507):
            db.create_session(session_id=f"session-{n:04}", source="desktop", model="provider/model")
        db.create_session(session_id="canonical", source="desktop", model="other/model")
        db.set_session_title("canonical", "Bot Chat")
        db.set_session_hidden("canonical", True)
    finally:
        db.close()
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: [("worker", home)])
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    client = TestClient(app)
    pages = [client.get(f"/api/profiles/agent-overview?offset={offset}&limit=100") for offset in range(0, 600, 100)]
    assert all(page.status_code == 200 for page in pages)
    bodies = [page.json() for page in pages]
    ids = [row["id"] for body in bodies for row in body["sessions"]]
    assert len(ids) == len(set(ids)) == bodies[0]["total"] == 507
    assert bodies[0]["canonical"][0]["id"] == "canonical"
    assert bodies[0]["canonical"][0]["profile"] == "worker"
    assert all(row["profile"] == "worker" for body in bodies for row in body["sessions"])
    assert all("is_active" not in row for body in bodies for row in body["sessions"])


def _overview_client(monkeypatch, home):
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: [("worker", home)])
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    return TestClient(app)


def test_old_hidden_canonical_chat_carries_persisted_activity_without_live_runtime(tmp_path, monkeypatch):
    """The canonical row is resolved by exact title, not recency, so its activity must be projected
    from persisted messages: otherwise a Bot Chat created long ago but used a minute ago renders as
    stale (started_at fallback) and drops out of Recent activity."""
    import time
    home = tmp_path / "worker"
    home.mkdir()
    with SessionDB(home / "state.db") as db:
        db.create_session(session_id="bot", source="desktop", model="m")
        db.set_session_title("bot", "Bot Chat")
        db.set_session_hidden("bot", True)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (time.time() - 3600, "bot"))
        db._conn.commit()
        db.set_session_read("bot")
        time.sleep(0.01)
        message_ts = time.time()
        db.append_message(session_id="bot", role="user", content="Deploy the fix please", timestamp=message_ts)
    body = _overview_client(monkeypatch, home).get("/api/profiles/agent-overview").json()
    assert body["sessions"] == []  # hidden: the rich list cannot be the activity source
    (row,) = body["canonical"]
    assert row["id"] == "bot"
    assert row["last_active"] >= message_ts
    assert row["preview"].startswith("Deploy the fix")
    assert row["unread"] is True
    assert row["message_count"] == 1


def test_compressed_canonical_chat_projects_activity_from_live_tip(tmp_path, monkeypatch):
    import time
    home = tmp_path / "worker"
    home.mkdir()
    with SessionDB(home / "state.db") as db:
        db.create_session(session_id="root", source="desktop", model="m")
        db.end_session("root", "compression")
        db.create_session(session_id="tip", source="desktop", model="m", parent_session_id="root")
        db.set_session_title("root", "Bot Chat")
        db.set_session_hidden("root", True)
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id IN (?, ?)", (time.time() - 3600, "root", "tip"))
        db._conn.commit()
        message_ts = time.time()
        db.append_message(session_id="tip", role="user", content="Continue after compression", timestamp=message_ts)
    body = _overview_client(monkeypatch, home).get("/api/profiles/agent-overview").json()
    (row,) = body["canonical"]
    assert (row["id"], row["resolved_id"], row["title"]) == ("root", "tip", "Bot Chat")
    assert row["last_active"] >= message_ts
    assert row["preview"].startswith("Continue after compression")
    assert row["message_count"] == 1
    assert row["unread"] is False  # never tracked = read; the field must still be present


def test_live_rows_are_owner_scoped_and_independent_of_history_page(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tui_gateway import server
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: [("a", home_a), ("b", home_b)])
    monkeypatch.setattr(server, "_sessions", {
        "runtime-a": {"session_key": "same-id", "profile_home": str(home_a), "running": True,
                      "pending_title": "Older running task", "agent": SimpleNamespace(model="model-a", provider="provider-a")},
        "runtime-b": {"session_key": "same-id", "profile_home": str(home_b), "running": False,
                      "agent": SimpleNamespace(model="model-b", provider="provider-b")},
    })
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    body = TestClient(app).get("/api/profiles/agent-overview?limit=1&offset=500").json()
    assert body["sessions"] == []
    live = {row["profile"]: row for row in body["live"]}
    assert live["a"]["runtime_id"] == "runtime-a"
    assert live["a"]["status"] == "working"
    assert live["b"]["status"] == "idle"
    assert live["a"]["id"] == live["b"]["id"] == "same-id"
    assert live["a"]["provider"] == "provider-a"
    assert body["live_coverage"] == "process"


def test_live_only_refresh_does_not_open_history_databases(tmp_path, monkeypatch):
    import hermes_state
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: [("worker", tmp_path)])
    (tmp_path / "state.db").touch()
    def refuse(*args, **kwargs):
        raise AssertionError("history DB read on live-only refresh")
    monkeypatch.setattr(hermes_state, "SessionDB", refuse)
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    body = TestClient(app).get("/api/profiles/agent-overview?live_only=true").json()
    assert body["errors"] == []
    assert body["sessions"] == []


def test_live_activity_preview_is_bounded_and_profile_metadata_is_safe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tui_gateway import server
    home = tmp_path / "worker"
    home.mkdir()
    (home / "profile.yaml").write_text("display_name: Builder\ndescription: Builds things\nprivate: SECRET\n")
    monkeypatch.setattr(profiles, "_profile_targets", lambda *a, **k: [("worker", home)])
    session = {"profile_home": str(home), "session_key": "task", "running": True,
               "agent": SimpleNamespace(model="m", provider="p"),
               "history": [{"role": "assistant", "content": "Inspecting the tests " * 100}]}
    monkeypatch.setattr(server, "_sessions", {"runtime": session})
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    client = TestClient(app)
    body = client.get("/api/profiles/agent-overview").json()
    assert body["profiles"] == [{"name": "worker", "display_name": "Builder", "description": "Builds things"}]
    assert body["live"][0]["preview"].startswith("Inspecting the tests")
    assert len(body["live"][0]["preview"]) <= 160
    session["history"][-1]["content"] = "Tests passed"
    assert client.get("/api/profiles/agent-overview?live_only=true").json()["live"][0]["preview"] == "Tests passed"
    assert "SECRET" not in str(body)
    assert not (home / "state.db").exists()
