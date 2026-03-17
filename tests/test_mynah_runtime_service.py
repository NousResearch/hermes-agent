from fastapi.testclient import TestClient

from mynah_runtime.service import create_runtime_app


class FakeAgent:
    def __init__(self):
        self.session_id = "session-123"

    def run_conversation(self, user_message, system_message=None, conversation_history=None):
        history = list(conversation_history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "runtime ok"})
        return {
            "response": "runtime ok",
            "messages": history,
        }


class FakeFactory:
    def create(self):
        return FakeAgent()


def test_runtime_health_and_turn(monkeypatch):
    monkeypatch.setenv("MYNAH_RUNTIME_PROFILE", "tier1")
    monkeypatch.setenv("MYNAH_RUNTIME_TOOLSET", "mynah-tier1")
    monkeypatch.setenv("HERMES_HOME", "/srv/mynah/user/hermes")
    monkeypatch.setenv("MYNAH_INFERENCE_MODEL", "qwen3.5-9b-local")

    client = TestClient(create_runtime_app(FakeFactory()))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["runtime_toolset"] == "mynah-tier1"

    turn = client.post(
        "/runtime/turn",
        json={
            "user_message": "hello",
            "conversation_history": [{"role": "system", "content": "base"}],
        },
    )
    assert turn.status_code == 200
    body = turn.json()
    assert body["final_response"] == "runtime ok"
    assert body["session_id"] == "session-123"
    assert body["messages"][-1]["content"] == "runtime ok"
