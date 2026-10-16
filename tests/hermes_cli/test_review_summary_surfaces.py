"""Read-only receipt queries and CLI rehydrate must not construct live agents."""
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.review_summary import publish_review_summary
from hermes_state import SessionDB


def test_receipt_query_pages_receipts_not_chat_and_never_crosses_profiles(tmp_path, monkeypatch):
    from hermes_cli.web_routers import sessions
    stores = {name: SessionDB(tmp_path / f"{name}.db") for name in ("room", "other")}
    for name, db in stores.items():
        db.create_session("group-sid", "desktop")
        db.create_session("bot-chat", "desktop")
        agent = SimpleNamespace(_session_db=db, session_id="group-sid", _persist_disabled=False,
            _safe_print=lambda *_: None, background_review_callback=None)
        publish_review_summary(agent, [f"Skill '{name}' patched"])
        agent.session_id = "bot-chat"
        publish_review_summary(agent, ["PRIVATE BOT CHAT"])
        db.append_messages_batch("group-sid", [
            {"role": "assistant", "content": f"ordinary message {i}"} for i in range(600)])
    before = stores["room"].get_resume_conversations("group-sid")[0]
    reads = []
    def with_db(profile, fn, **kwargs):
        reads.append((profile, kwargs))
        return fn(stores[profile])
    monkeypatch.setattr(sessions, "_with_db", with_db)
    app = FastAPI()
    app.include_router(sessions.manage_router)
    with TestClient(app) as client:
        for profile in stores:
            result = client.get(f"/api/sessions/group-sid/review-summaries?profile={profile}")
            assert result.status_code == 200
            rows = result.json()["messages"]
            assert len(rows) == 1
            assert f"'{profile}'" in rows[0]["content"]
            assert "PRIVATE" not in rows[0]["content"]
        assert client.get("/api/sessions/missing/review-summaries?profile=room").status_code == 404
    assert all(options == {"read_only": True} for _, options in reads)
    assert stores["room"].get_resume_conversations("group-sid")[0] == before
    # Archive in place: the read rail survives while the model tail stays small.
    stores["room"].archive_and_compact("group-sid", [{"role": "assistant", "content": "summary"}], tail_count=1)
    assert len(stores["room"].get_review_summaries("group-sid")) == 1
    for db in stores.values():
        db.close()


def test_cli_resume_renders_receipts_as_events_without_replaying_in_model(tmp_path):
    from hermes_cli.cli_agent_setup_mixin import _collect_resume_entries
    db = SessionDB(tmp_path / "state.db")
    db.create_session("chat", "cli")
    db.append_message("chat", "user", "task")
    agent = SimpleNamespace(_session_db=db, session_id="chat", _persist_disabled=False,
        _safe_print=lambda *_: None, background_review_callback=None)
    publish_review_summary(agent, ["Skill 'canary' patched"])
    model, display = db.get_resume_conversations("chat")
    entries, _, _ = _collect_resume_entries(display, {}, lambda s: s)
    assert any(role == "event" and "canary" in text for role, text in entries)
    assert not any("canary" in str(row.get("content", "")) for row in model)
    db.close()
