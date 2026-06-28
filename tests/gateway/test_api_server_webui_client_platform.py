"""API server client-platform selection for Hermes WebUI callers."""

import asyncio
import json

from gateway.platforms.api_server import APIServerAdapter


class _Request:
    def __init__(self, headers):
        self.headers = headers


class _JsonRequest(_Request):
    def __init__(self, headers, body, session_id="sess-1"):
        super().__init__(headers)
        self._body = body
        self.match_info = {"session_id": session_id}

    async def json(self):
        return self._body


def test_client_platform_header_selects_webui_without_session_key():
    request = _Request({"X-Hermes-Client": "webui"})

    assert APIServerAdapter._client_platform_from_request(request) == "webui"


def test_client_platform_falls_back_to_api_server_for_generic_clients():
    request = _Request({"X-Hermes-Client": "curl"})

    assert APIServerAdapter._client_platform_from_request(request) == "api_server"


def test_webui_session_key_still_selects_webui_when_present():
    request = _Request({})

    assert APIServerAdapter._client_platform_from_request(request, "webui:session-123") == "webui"


def test_bind_api_server_session_keeps_async_delivery_off_for_webui(monkeypatch):
    from gateway import session_context

    captured = {}

    def fake_set_session_vars(**kwargs):
        captured.update(kwargs)
        return ["token"]

    monkeypatch.setattr(session_context, "set_session_vars", fake_set_session_vars)

    tokens = APIServerAdapter._bind_api_server_session(
        chat_id="sess-1",
        session_key="webui:sess-1",
        session_id="sess-1",
        client_platform="webui",
    )

    assert tokens == ["token"]
    assert captured["platform"] == "api_server"
    assert captured["async_delivery"] is False


def test_run_agent_uses_webui_presentation_but_api_server_session_context(monkeypatch):
    from gateway.session_context import get_session_env

    captured = {}

    class FakeAgent:
        session_id = "sess-1"
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def run_conversation(self, user_message, conversation_history, task_id):
            captured["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            captured["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            return {"final_response": "ok", "messages": []}

    def fake_create_agent(**kwargs):
        captured["client_platform"] = kwargs.get("client_platform")
        return FakeAgent()

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._inflight_agent_runs = 0
    adapter._active_session_agents = {}
    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = asyncio.run(
        adapter._run_agent(
            "hi",
            [],
            session_id="sess-1",
            gateway_session_key="webui:sess-1",
            client_platform="webui",
        )
    )

    assert result["final_response"] == "ok"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert captured["client_platform"] == "webui"
    assert captured["context_platform"] == "api_server"
    assert captured["context_session_key"] == "webui:sess-1"
    assert adapter._active_session_agents == {}


def test_session_steer_routes_to_active_webui_session_key():
    calls = []

    class FakeAgent:
        def steer(self, text):
            calls.append(text)
            return True

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._api_key = "sk-secret"
    adapter._active_session_agents = {"webui:sess-1": FakeAgent()}
    request = _JsonRequest(
        {
            "Authorization": "Bearer sk-secret",
            "X-Hermes-Session-Key": "webui:sess-1",
        },
        {"text": "keep going"},
    )

    response = asyncio.run(adapter._handle_session_steer(request))
    payload = json.loads(response.text)

    assert payload == {"accepted": True, "fallback": None, "session_id": "sess-1"}
    assert calls == ["keep going"]


def test_session_steer_reports_not_running_without_queue_or_interrupt():
    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._api_key = "sk-secret"
    adapter._active_session_agents = {}
    request = _JsonRequest(
        {
            "Authorization": "Bearer sk-secret",
            "X-Hermes-Session-Key": "webui:sess-1",
        },
        {"text": "keep going"},
    )

    response = asyncio.run(adapter._handle_session_steer(request))
    payload = json.loads(response.text)

    assert payload == {"accepted": False, "fallback": "not_running", "session_id": "sess-1"}


def test_create_agent_uses_webui_platform_but_api_server_toolsets(monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli import tools_config
    import run_agent

    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)
    monkeypatch.setattr(gateway_run, "_current_max_iterations", lambda: 3)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "test-model")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"platform_toolsets": {"api_server": ["web"]}})
    monkeypatch.setattr(gateway_run.GatewayRunner, "_load_reasoning_config", staticmethod(lambda: None))
    monkeypatch.setattr(gateway_run.GatewayRunner, "_load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr(tools_config, "_get_platform_tools", lambda cfg, platform: {f"toolset:{platform}"})

    adapter = APIServerAdapter.__new__(APIServerAdapter)
    adapter._session_db = None

    agent = adapter._create_agent(session_id="sess-1", client_platform="webui")

    assert isinstance(agent, FakeAgent)
    assert captured["platform"] == "webui"
    assert captured["enabled_toolsets"] == ["toolset:api_server"]
