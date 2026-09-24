"""Native compaction is accounting metadata, not assistant prose (#121301)."""
from agent.transports.codex_event_projector import CodexEventProjector
from agent.transports.codex_app_server_session import CodexAppServerSession
from hermes_state import SessionDB
from tests.agent.transports.test_codex_app_server_session import FakeClient


def test_compaction_preserves_pending_reasoning_and_unknown_item_fallback():
    projector = CodexEventProjector()

    def project(kind, **fields):
        return projector.project({"method": "item/completed", "params": {
            "item": {"type": kind, "id": kind, **fields}}})

    project("reasoning", summary=["reasoning before compaction"])
    boundary = project("contextCompaction")
    assert boundary.messages == []
    assert boundary.final_text is None and not boundary.is_tool_iteration
    reply = project("agentMessage", text="after compaction")
    assert reply.messages == [{"role": "assistant", "content": "after compaction",
                               "reasoning": "reasoning before compaction"}]
    # This is a known control item, not a reason to discard unknown protocol items.
    unknown = project("futureItem", data="keep me")
    assert "keep me" in unknown.messages[0]["content"]


def test_session_compaction_survives_accounting_without_polluting_persisted_replay(tmp_path):
    client = FakeClient()
    for kind, fields in [
        ("agentMessage", {"text": "C-ONE"}),
        ("contextCompaction", {}),
        ("agentMessage", {"text": "C-TWO"}),
    ]:
        client.queue_notification("item/completed", threadId="th", turnId="tu1",
                                  item={"type": kind, "id": kind, **fields})
    client.queue_notification("turn/completed", threadId="th",
                              turn={"id": "tu1", "status": "completed"})
    events = []
    session = CodexAppServerSession(cwd=str(tmp_path), client_factory=lambda **kw: client,
                                   on_event=events.append)
    try:
        turn = session.run_turn("compact me")
        assert turn.error is None and turn.compacted
        assert turn.final_text == "C-TWO" and turn.tool_iterations == 0
        assert any(n.get("params", {}).get("item", {}).get("type") == "contextCompaction"
                   for n in events)
        # Persist the actual session projection, then reopen the durable transcript.
        path = tmp_path / "state.db"
        db = SessionDB(path)
        try:
            db.create_session(session_id="projection", source="cli", model="codex")
            for message in turn.projected_messages:
                db.append_message("projection", role=message["role"], content=message["content"])
        finally:
            db.close()
        db = SessionDB(path)
        try:
            assert [m["content"] for m in db.get_messages("projection")] == ["C-ONE", "C-TWO"]
        finally:
            db.close()
    finally:
        session.close()
