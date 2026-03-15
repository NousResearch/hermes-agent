import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


class _JsonRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def test_load_gateway_config_enables_webhook_from_env(monkeypatch):
    monkeypatch.setenv("WEBHOOK_PORT", "4568")

    config = load_gateway_config()

    assert Platform.WEBHOOK in config.platforms
    assert config.platforms[Platform.WEBHOOK].enabled is True
    assert config.platforms[Platform.WEBHOOK].extra["port"] == 4568


def test_runner_creates_webhook_adapter(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBHOOK_PORT", "4568")
    config = load_gateway_config()
    config.sessions_dir = tmp_path / "sessions"

    runner = GatewayRunner(config)
    adapter = runner._create_adapter(Platform.WEBHOOK, config.platforms[Platform.WEBHOOK])

    assert adapter is not None
    assert adapter.platform == Platform.WEBHOOK


def test_webhook_messages_are_always_authorized(tmp_path):
    runner = GatewayRunner(config=load_gateway_config())
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="hermes-1",
        chat_type="dm",
        user_id="agent-1",
        user_name="bridge",
    )

    assert runner._is_user_authorized(source) is True


def test_gateway_max_tokens_loads_from_config(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  max_tokens: 1024\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("HERMES_MAX_TOKENS", raising=False)

    assert GatewayRunner._load_gateway_max_tokens() == 1024


def test_gateway_max_tokens_env_overrides_config(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    config_path = tmp_path / "config.yaml"
    config_path.write_text("agent:\n  max_tokens: 1024\n", encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setenv("HERMES_MAX_TOKENS", "768")

    assert GatewayRunner._load_gateway_max_tokens() == 768


@pytest.mark.asyncio
async def test_run_agent_passes_gateway_max_tokens_to_ai_agent(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.tools = []
            self.model = kwargs.get("model")
            self.session_id = kwargs.get("session_id")

        def run_conversation(self, message, conversation_history=None, task_id=None):
            return {"final_response": "ok", "messages": [], "api_calls": 1}

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = FakeAIAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "openrouter/hunter-alpha")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("display:\n  tool_progress: off\nagent:\n  max_tokens: 1024\n", encoding="utf-8")

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.WEBHOOK: MagicMock()}
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._voice_mode = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._provider_routing = {}
    runner._session_db = None
    runner._fallback_model = None
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._get_or_create_gateway_honcho = lambda _session_key: (None, None)

    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="hermes-1:room",
        chat_type="dm",
        user_id="visitor",
        user_name="visitor",
    )

    result = await runner._run_agent(
        message="hello",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-1",
        session_key="agent:main:webhook:dm:hermes-1:room",
    )

    assert result["final_response"] == "ok"
    assert captured["max_tokens"] == 1024
    assert captured["skip_context_files"] is True


@pytest.mark.asyncio
async def test_webhook_post_returns_last_response_for_chat_id(monkeypatch):
    monkeypatch.setenv("WEBHOOK_PORT", "4568")

    from gateway.platforms.webhook import WebhookAdapter

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 4568}))
    seen = []

    async def fake_handle_message(event):
        seen.append(build_session_key(event.source))
        await adapter.send(event.source.chat_id, "intermediate")
        await adapter.send(event.source.chat_id, "final")

    adapter.handle_message = fake_handle_message

    response = await adapter._handle_post(_JsonRequest({
        "chat_id": "hermes-1",
        "message": "hello",
        "from": "bridge",
        "user_id": "agent-1",
    }))

    body = json.loads(response.text)
    assert body["ok"] is True
    assert body["response"] == "final"
    assert seen[0].endswith("webhook:dm:hermes-1")


@pytest.mark.asyncio
async def test_webhook_session_key_stays_stable_for_same_chat_id(monkeypatch):
    monkeypatch.setenv("WEBHOOK_PORT", "4568")

    from gateway.platforms.webhook import WebhookAdapter

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 4568}))
    seen = []

    async def fake_handle_message(event):
        seen.append(build_session_key(event.source))
        await adapter.send(event.source.chat_id, "ok")

    adapter.handle_message = fake_handle_message

    for _ in range(2):
        await adapter._handle_post(_JsonRequest({
            "chat_id": "hermes-1",
            "message": "hello",
        }))

    assert len(seen) == 2
    assert seen[0] == seen[1]
    assert seen[0].endswith("webhook:dm:hermes-1")


@pytest.mark.asyncio
async def test_webhook_post_returns_response_even_if_active_session_lingers(monkeypatch):
    monkeypatch.setenv("WEBHOOK_PORT", "4568")

    from gateway.platforms.webhook import WebhookAdapter

    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"port": 4568}))

    async def fake_handle_message(event):
        session_key = build_session_key(event.source)
        adapter._active_sessions[session_key] = SimpleNamespace(is_set=lambda: False)
        await adapter.send(event.source.chat_id, "still-returns")

    adapter.handle_message = fake_handle_message

    response = await adapter._handle_post(_JsonRequest({
        "chat_id": "hermes-1",
        "message": "hello",
    }))

    body = json.loads(response.text)
    assert body["ok"] is True
    assert body["response"] == "still-returns"
