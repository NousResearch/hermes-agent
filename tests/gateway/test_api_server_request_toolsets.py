import asyncio
import sys
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server


def test_request_toolsets_absent_keeps_platform_default():
    override, error = api_server._resolve_request_toolset_override(
        {},
        platform_enabled_toolsets=["web", "terminal"],
    )

    assert override is None
    assert error is None


@pytest.mark.parametrize(
    "body",
    [
        {"tools": []},
        {"tool_choice": "none"},
        {"enabled_toolsets": []},
        {"toolsets": []},
    ],
)
def test_request_can_disable_all_tools(body):
    override, error = api_server._resolve_request_toolset_override(
        body,
        platform_enabled_toolsets=["web", "terminal"],
    )

    assert override == []
    assert error is None


def test_request_can_restrict_to_enabled_subset_only():
    override, error = api_server._resolve_request_toolset_override(
        {"enabled_toolsets": ["web"]},
        platform_enabled_toolsets=["web", "terminal"],
    )

    assert override == ["web"]
    assert error is None


def test_api_stream_progress_event_keeps_reasoning_separate_from_tool_events():
    mapped = api_server._api_stream_progress_event(
        "reasoning.available",
        message_id="msg_1",
        tool_name="_thinking",
        preview="thinking text",
    )

    assert mapped == (
        "reasoning.available",
        {"message_id": "msg_1", "delta": "thinking text"},
    )


def test_api_stream_progress_event_preserves_real_tool_events():
    mapped = api_server._api_stream_progress_event(
        "tool.started",
        message_id="msg_1",
        tool_name="terminal",
        preview="ls",
        args={"command": "ls"},
    )

    assert mapped == (
        "tool.started",
        {
            "message_id": "msg_1",
            "tool_name": "terminal",
            "preview": "ls",
            "args": {"command": "ls"},
        },
    )


def test_request_toolset_override_rejects_expansion_beyond_platform_surface():
    override, error = api_server._resolve_request_toolset_override(
        {"enabled_toolsets": ["web", "terminal"]},
        platform_enabled_toolsets=["web"],
    )

    assert override is None
    assert error is not None
    assert "not enabled for api_server" in error


def test_request_toolset_override_rejects_invalid_shape():
    override, error = api_server._resolve_request_toolset_override(
        {"enabled_toolsets": "web"},
        platform_enabled_toolsets=["web"],
    )

    assert override is None
    assert error is not None
    assert "must be an array" in error


class FakeResponse:
    def __init__(self, data=None, *, status=200, headers=None):
        self.data = data
        self.status = status
        self.headers = headers or {}


class FakeWeb:
    @staticmethod
    def json_response(data=None, *, status=200, headers=None):
        return FakeResponse(data, status=status, headers=headers)


class FakeRequest:
    def __init__(self, body, *, headers=None, match_info=None):
        self._body = body
        self.headers = headers or {}
        self.match_info = match_info or {}
        self.transport = None
        self.remote = "127.0.0.1"
        self.method = "POST"
        self.path_qs = "/test"

    async def json(self):
        return self._body


def _minimal_adapter(monkeypatch):
    monkeypatch.setattr(api_server, "web", FakeWeb)
    adapter = object.__new__(api_server.APIServerAdapter)
    adapter._api_key = ""
    adapter._model_name = "test-model"
    adapter._background_tasks = set()
    adapter._get_platform_enabled_toolsets = lambda: ["web", "terminal"]
    adapter._parse_session_key_header = lambda request: (None, None)
    return adapter


def test_chat_completions_handler_passes_empty_tool_override(monkeypatch):
    async def run_case():
        captured = {}
        adapter = _minimal_adapter(monkeypatch)

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)
            return {"final_response": "ok", "completed": True, "session_id": "sid"}, {}

        adapter._run_agent = fake_run_agent
        request = FakeRequest({
            "model": "test-model",
            "messages": [{"role": "user", "content": "probe"}],
            "tools": [],
        })

        response = await adapter._handle_chat_completions(request)

        assert response.status == 200
        assert captured["enabled_toolsets_override"] == []

    asyncio.run(run_case())


def test_responses_handler_passes_empty_tool_override(monkeypatch):
    async def run_case():
        captured = {}
        adapter = _minimal_adapter(monkeypatch)

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)
            return {"final_response": "ok", "completed": True, "messages": []}, {}

        adapter._run_agent = fake_run_agent
        request = FakeRequest({
            "model": "test-model",
            "input": "probe",
            "store": False,
            "enabled_toolsets": [],
        })

        response = await adapter._handle_responses(request)

        assert response.status == 200
        assert captured["enabled_toolsets_override"] == []

    asyncio.run(run_case())


def test_session_chat_handler_passes_empty_tool_override(monkeypatch):
    async def run_case():
        captured = {}
        adapter = _minimal_adapter(monkeypatch)
        adapter._get_existing_session_or_404 = lambda session_id: ({"id": session_id}, None)
        adapter._conversation_history_for_session = lambda session_id: []

        async def fake_run_agent(**kwargs):
            captured.update(kwargs)
            return {"final_response": "ok", "completed": True, "session_id": "sid"}, {}

        adapter._run_agent = fake_run_agent
        request = FakeRequest(
            {"message": "probe", "toolsets": []},
            match_info={"session_id": "sid"},
        )

        response = await adapter._handle_session_chat(request)

        assert response.status == 200
        assert captured["enabled_toolsets_override"] == []

    asyncio.run(run_case())


def test_runs_handler_passes_empty_tool_override_to_created_agent(monkeypatch):
    async def run_case():
        captured = {}
        adapter = _minimal_adapter(monkeypatch)
        adapter._response_store = SimpleNamespace(get=lambda response_id: None)
        adapter._run_streams = {}
        adapter._run_streams_created = {}
        adapter._active_run_agents = {}
        adapter._active_run_tasks = {}
        adapter._run_statuses = {}
        adapter._run_approval_sessions = {}

        class FakeAgent:
            def run_conversation(self, **kwargs):
                return {"final_response": "ok", "completed": True, "messages": []}

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return FakeAgent()

        adapter._create_agent = fake_create_agent
        request = FakeRequest({"input": "probe", "tool_choice": "none"})

        response = await adapter._handle_runs(request)
        await asyncio.sleep(0.05)

        assert response.status == 202
        assert captured["enabled_toolsets_override"] == []

    asyncio.run(run_case())


def _install_fake_gateway_run(monkeypatch):
    class FakeGatewayRunner:
        @staticmethod
        def _load_reasoning_config():
            return None

        @staticmethod
        def _load_fallback_model():
            return None

    monkeypatch.setitem(
        sys.modules,
        "gateway.run",
        SimpleNamespace(
            _resolve_runtime_agent_kwargs=lambda: {"api_key": "key", "base_url": "http://example.test"},
            _resolve_gateway_model=lambda: "test-model",
            GatewayRunner=FakeGatewayRunner,
        ),
    )


def test_create_agent_uses_request_local_toolset_override(monkeypatch):
    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=FakeAIAgent))
    _install_fake_gateway_run(monkeypatch)

    adapter = object.__new__(api_server.APIServerAdapter)
    adapter._ensure_session_db = lambda: None
    adapter._get_platform_enabled_toolsets = lambda: ["web", "terminal"]

    adapter._create_agent(enabled_toolsets_override=[])

    assert captured["enabled_toolsets"] == []


def test_create_agent_defaults_to_platform_toolsets_when_no_override(monkeypatch):
    captured = {}

    class FakeAIAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=FakeAIAgent))
    _install_fake_gateway_run(monkeypatch)

    adapter = object.__new__(api_server.APIServerAdapter)
    adapter._ensure_session_db = lambda: None
    adapter._get_platform_enabled_toolsets = lambda: ["web", "terminal"]

    adapter._create_agent()

    assert captured["enabled_toolsets"] == ["web", "terminal"]
