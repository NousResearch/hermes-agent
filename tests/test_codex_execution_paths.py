import asyncio
import sys
import types
from types import SimpleNamespace


sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import cron.scheduler as cron_scheduler
import gateway.run as gateway_run
import run_agent
from gateway.config import Platform
from gateway.session import SessionSource


def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        run_agent,
        "get_tool_definitions",
        lambda **kwargs: [
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "description": "Run shell commands.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda: {})


def _codex_message_response(text: str):
    return SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        ],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
        status="completed",
        model="gpt-5-codex",
    )


class _UnauthorizedError(RuntimeError):
    def __init__(self):
        super().__init__("Error code: 401 - unauthorized")
        self.status_code = 401


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def close(self):
        return None


class _Codex401ThenSuccessAgent(run_agent.AIAgent):
    refresh_attempts = 0
    last_init = {}

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_context_files", True)
        kwargs.setdefault("skip_memory", True)
        kwargs.setdefault("max_iterations", 4)
        type(self).last_init = dict(kwargs)
        super().__init__(*args, **kwargs)
        self._cleanup_task_resources = lambda task_id: None
        self._persist_session = lambda messages, history=None: None
        self._save_trajectory = lambda messages, user_message, completed: None
        self._save_session_log = lambda messages: None

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        type(self).refresh_attempts += 1
        return True

    def run_conversation(self, user_message: str, conversation_history=None):
        calls = {"api": 0}

        def _fake_api_call(api_kwargs):
            calls["api"] += 1
            if calls["api"] == 1:
                raise _UnauthorizedError()
            return _codex_message_response("Recovered via refresh")

        self._interruptible_api_call = _fake_api_call
        return super().run_conversation(user_message, conversation_history=conversation_history)


def test_cron_run_job_codex_path_handles_internal_401_refresh(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run_agent, "AIAgent", _Codex401ThenSuccessAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
        },
    )
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))

    _Codex401ThenSuccessAgent.refresh_attempts = 0
    _Codex401ThenSuccessAgent.last_init = {}

    success, output, final_response, error = cron_scheduler.run_job(
        {"id": "job-1", "name": "Codex Refresh Test", "prompt": "ping"}
    )

    assert success is True
    assert error is None
    assert final_response == "Recovered via refresh"
    assert "Recovered via refresh" in output
    assert _Codex401ThenSuccessAgent.refresh_attempts == 1
    assert _Codex401ThenSuccessAgent.last_init["provider"] == "openai-codex"
    assert _Codex401ThenSuccessAgent.last_init["api_mode"] == "codex_responses"


def test_gateway_run_agent_codex_path_handles_internal_401_refresh(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr(run_agent, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run_agent, "AIAgent", _Codex401ThenSuccessAgent)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
        },
    )
    monkeypatch.setenv("HERMES_TOOL_PROGRESS", "false")

    _Codex401ThenSuccessAgent.refresh_attempts = 0
    _Codex401ThenSuccessAgent.last_init = {}

    runner = gateway_run.GatewayRunner.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._running_agents = {}
    from unittest.mock import MagicMock, AsyncMock
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None

    source = SessionSource(
        platform=Platform.LOCAL,
        chat_id="cli",
        chat_name="CLI",
        chat_type="dm",
        user_id="user-1",
    )

    result = asyncio.run(
        runner._run_agent(
            message="ping",
            context_prompt="",
            history=[],
            source=source,
            session_id="session-1",
            session_key="agent:main:local:dm",
        )
    )

    assert result["final_response"] == "Recovered via refresh"
    assert _Codex401ThenSuccessAgent.refresh_attempts == 1
    assert _Codex401ThenSuccessAgent.last_init["provider"] == "openai-codex"
    assert _Codex401ThenSuccessAgent.last_init["api_mode"] == "codex_responses"
