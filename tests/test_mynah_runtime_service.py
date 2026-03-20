from fastapi.testclient import TestClient

from hermes_cli.product_config import load_product_config, save_product_config
from mynah_runtime.service import (
    _load_runtime_soul,
    create_runtime_app,
)


class FakeAgent:
    def __init__(self):
        self.session_id = "session-123"

    def run_conversation(self, user_message, system_message=None, conversation_history=None, stream_callback=None):
        history = list(conversation_history or [])
        if getattr(self, "reasoning_callback", None):
            self.reasoning_callback("thinking one")
            self.reasoning_callback(" thinking two")
        if stream_callback is not None:
            stream_callback("runtime ")
            stream_callback("ok")
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "runtime ok"})
        return {
            "final_response": "runtime ok",
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


def test_runtime_turn_accepts_legacy_response_key():
    class LegacyAgent:
        def __init__(self):
            self.session_id = "legacy-session"

        def run_conversation(self, user_message, system_message=None, conversation_history=None, stream_callback=None):
            history = list(conversation_history or [])
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": "legacy ok"})
            return {
                "response": "legacy ok",
                "messages": history,
            }

    class LegacyFactory:
        def create(self):
            return LegacyAgent()

    client = TestClient(create_runtime_app(LegacyFactory()))
    turn = client.post("/runtime/turn", json={"user_message": "hello"})

    assert turn.status_code == 200
    assert turn.json()["final_response"] == "legacy ok"


def test_runtime_turn_stream_emits_reasoning_answer_and_final(monkeypatch):
    monkeypatch.setenv("MYNAH_RUNTIME_PROFILE", "tier1")
    monkeypatch.setenv("MYNAH_RUNTIME_TOOLSET", "mynah-tier1")

    client = TestClient(create_runtime_app(FakeFactory()))

    with client.stream("POST", "/runtime/turn/stream", json={"user_message": "hello"}) as response:
        assert response.status_code == 200
        payload = "\n".join(response.iter_text())

    assert "event: reasoning" in payload
    assert "\"delta\": \"thinking one\"" in payload
    assert "event: answer" in payload
    assert "\"delta\": \"runtime \"" in payload
    assert "event: final" in payload
    assert "\"final_response\": \"runtime ok\"" in payload


def test_runtime_identity_prompt_loads_soul(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "SOUL.md").write_text("I am your MYNAH assistant.", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    assert _load_runtime_soul() == "I am your MYNAH assistant."


def test_runtime_identity_prompt_requires_soul(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    try:
        _load_runtime_soul()
    except RuntimeError as exc:
        assert "SOUL.md" in str(exc)
    else:
        raise AssertionError("expected runtime to require SOUL.md")


def test_runtime_health_uses_product_config_defaults(monkeypatch, tmp_path):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("MYNAH_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("MYNAH_RUNTIME_TOOLSET", raising=False)
    monkeypatch.delenv("MYNAH_INFERENCE_MODEL", raising=False)

    config = load_product_config()
    config["runtime"]["default_profile"] = "tier2"
    config["runtime"]["default_toolset"] = "mynah-tier2"
    config["models"]["default_route"]["model"] = "qwen-test-model"
    save_product_config(config)

    client = TestClient(create_runtime_app(FakeFactory()))

    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["runtime_profile"] == "tier2"
    assert health.json()["runtime_toolset"] == "mynah-tier2"
    assert health.json()["model"] == "qwen-test-model"
