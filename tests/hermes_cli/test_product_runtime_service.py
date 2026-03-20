from fastapi.testclient import TestClient

from hermes_cli.product_runtime_service import create_product_runtime_app


class FakeAgent:
    def __init__(self):
        self.session_id = "product_admin_123"
        self.reasoning_callback = None

    def run_conversation(self, user_message, conversation_history=None, stream_callback=None, sync_honcho=None):
        if self.reasoning_callback is not None:
            self.reasoning_callback("thinking")
        if stream_callback is not None:
            stream_callback("answer")
        history = list(conversation_history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "done"})
        return {"final_response": "done", "messages": history}


def test_product_runtime_session_and_turn(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "SOUL.md").write_text("Runtime identity", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("MYNAH_PRODUCT_SESSION_ID", "product_admin_123")
    monkeypatch.setattr("hermes_cli.product_runtime_service.build_runtime_agent", lambda db, session_id, reasoning_callback=None: FakeAgent())
    monkeypatch.setattr(
        "hermes_cli.product_runtime_service._load_session_messages",
        lambda db, session_id: [{"role": "assistant", "content": "earlier"}],
    )

    client = TestClient(create_product_runtime_app())
    session = client.get("/runtime/session")
    assert session.status_code == 200
    assert session.json()["session_id"] == "product_admin_123"
    assert session.json()["runtime_mode"] == "product"
    assert session.json()["runtime_toolsets"] == ["memory", "session_search"]

    turn = client.post("/runtime/turn", json={"user_message": "hello"})
    assert turn.status_code == 200
    assert turn.json()["final_response"] == "done"


def test_product_runtime_stream_emits_reasoning_and_final(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / "SOUL.md").write_text("Runtime identity", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("MYNAH_PRODUCT_SESSION_ID", "product_admin_123")
    monkeypatch.setattr("hermes_cli.product_runtime_service.build_runtime_agent", lambda db, session_id, reasoning_callback=None: FakeAgent())
    monkeypatch.setattr("hermes_cli.product_runtime_service._load_session_messages", lambda db, session_id: [])

    client = TestClient(create_product_runtime_app())
    with client.stream("POST", "/runtime/turn/stream", json={"user_message": "hello"}) as response:
        assert response.status_code == 200
        payload = "\n".join(response.iter_text())

    assert "event: reasoning" in payload
    assert "\"delta\": \"thinking\"" in payload
    assert "event: final" in payload
    assert "\"final_response\": \"done\"" in payload
    assert "\"runtime_toolsets\": [\"memory\", \"session_search\"]" in payload
