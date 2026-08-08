"""Runtime regressions for deterministic NVIDIA worker quota exhaustion."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


class WorkerQuotaError(Exception):
    status_code = 429

    def __init__(self):
        message = (
            "Upstream error from Nvidia: ResourceExhausted: "
            "Worker local total request limit reached (267/32)"
        )
        super().__init__(message)
        self.body = {"error": {"code": "resource_exhausted", "message": message}}
        self.response = SimpleNamespace(headers={})


def _response(content):
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_details=None,
        reasoning_content=None,
        audio=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="fallback/model",
        usage=None,
    )


def _make_agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode="chat_completions",
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run(agent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("answer once")


def test_worker_quota_without_fallback_does_not_retry():
    agent = _make_agent()
    agent._fallback_chain = []
    agent.client.chat.completions.create.side_effect = WorkerQuotaError()

    result = _run(agent)

    assert agent.client.chat.completions.create.call_count == 1
    assert result["failed"] is True
    assert "worker local total request limit" in result["error"].lower()


def test_worker_quota_activates_configured_fallback():
    agent = _make_agent()
    agent._fallback_chain = [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"}
    ]
    agent._fallback_index = 0
    agent.client.chat.completions.create.side_effect = [
        WorkerQuotaError(),
        _response("Recovered via fallback"),
    ]

    def activate_fallback(*, reason):
        agent._fallback_index = 1
        agent.model = "anthropic/claude-sonnet-4"
        return True

    with patch.object(agent, "_try_activate_fallback", side_effect=activate_fallback):
        result = _run(agent)

    assert agent.client.chat.completions.create.call_count == 2
    assert result["completed"] is True
    assert result["final_response"] == "Recovered via fallback"
