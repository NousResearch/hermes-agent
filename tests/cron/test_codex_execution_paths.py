import sys
import types
from types import SimpleNamespace
from pathlib import Path
import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import cron.scheduler as cron_scheduler
import run_agent

def _patch_agent_bootstrap(monkeypatch):
    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
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
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})

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

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        type(self).refresh_attempts += 1
        return True

    def run_conversation(self, user_message: str, conversation_history=None, task_id=None):
        calls = {"api": 0}

        def _fake_api_call(api_kwargs):
            calls["api"] += 1
            if calls["api"] == 1:
                raise _UnauthorizedError()
            return _codex_message_response("Recovered via refresh")

        self._interruptible_api_call = _fake_api_call
        return super().run_conversation(user_message, conversation_history=conversation_history, task_id=task_id)

def test_cron_run_job_codex_path_handles_internal_401_refresh(monkeypatch):
    _patch_agent_bootstrap(monkeypatch)
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run_agent, "AIAgent", _Codex401ThenSuccessAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kwargs: {
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
        {"id": "job-1", "name": "Codex Refresh Test", "prompt": "ping", "model": "gpt-5.3-codex"}
    )

    assert success is True
    assert error is None
    assert final_response == "Recovered via refresh"
    assert "Recovered via refresh" in output
    assert _Codex401ThenSuccessAgent.refresh_attempts == 1
    assert _Codex401ThenSuccessAgent.last_init["provider"] == "openai-codex"
    assert _Codex401ThenSuccessAgent.last_init["api_mode"] == "codex_responses"
class _RetryCaptureAgent(run_agent.AIAgent):
    """Captures the effective ``_api_max_retries`` at run time, after the
    scheduler has applied any per-job override."""

    captured_retries = None

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("skip_context_files", True)
        kwargs.setdefault("skip_memory", True)
        kwargs.setdefault("max_iterations", 4)
        super().__init__(*args, **kwargs)
        self._cleanup_task_resources = lambda task_id: None
        self._persist_session = lambda messages, history=None: None
        self._save_trajectory = lambda messages, user_message, completed: None

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        type(self).captured_retries = getattr(self, "_api_max_retries", None)
        return {"final_response": "ok", "messages": []}


def _patch_codex_runtime(monkeypatch):
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    monkeypatch.setattr(run_agent, "AIAgent", _RetryCaptureAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **kw: {
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
        },
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc)
    )


@pytest.mark.parametrize("inherited, requested, expected", [(3, 6, 6), (3, 1, 3), (8, 6, 8)])
def test_cron_per_job_api_max_retries_override(monkeypatch, inherited, requested, expected):
    """A job that sets ``api_max_retries`` raises the agent's retry budget so
    transient backend hangs are retried more times on the requested model
    before the fallback chain swaps models."""
    _patch_agent_bootstrap(monkeypatch)
    _patch_codex_runtime(monkeypatch)

    from hermes_constants import get_hermes_home
    (Path(get_hermes_home()) / "config.yaml").write_text(f"agent:\n  api_max_retries: {inherited}\n")
    _RetryCaptureAgent.captured_retries = None
    success, output, final_response, error = cron_scheduler.run_job(
        {
            "id": "digest-1",
            "name": "Morning Digest",
            "prompt": "ping",
            "model": "gpt-5.5",
            "provider": "openai-codex",
            "api_max_retries": requested,
        }
    )

    assert success is True, error
    assert _RetryCaptureAgent.captured_retries == expected


def test_cron_without_override_inherits_agent_default(monkeypatch):
    """Without a per-job ``api_max_retries`` the agent keeps its configured
    default (no global behavior change for other jobs)."""
    _patch_agent_bootstrap(monkeypatch)
    _patch_codex_runtime(monkeypatch)

    _RetryCaptureAgent.captured_retries = None
    success, output, final_response, error = cron_scheduler.run_job(
        {
            "id": "digest-2",
            "name": "No Override",
            "prompt": "ping",
            "model": "gpt-5.5",
            "provider": "openai-codex",
        }
    )

    assert success is True, error
    # Agent default is 3 (config.yaml agent.api_max_retries, default 3).
    assert _RetryCaptureAgent.captured_retries == 3
