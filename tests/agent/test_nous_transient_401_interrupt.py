"""A /stop during the transient-401 backoff ends the turn without another request."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai

from agent import credential_auth_recovery
from run_agent import AIAgent


def _agent() -> AIAgent:
    with (
        patch("hermes_agent.model_tools.get_tool_definitions", return_value=[]),
        patch("hermes_agent.model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.compression_enabled = False
    a.save_trajectories = False
    return a


def _auth_401() -> openai.AuthenticationError:
    req = httpx.Request("POST", "https://inference-api.nousresearch.com/v1/chat/completions")
    return openai.AuthenticationError(
        "authentication required", response=httpx.Response(401, request=req),
        body={"error": "authentication required"})


def test_stop_during_transient_401_backoff_sends_no_further_request(monkeypatch):
    agent = _agent()
    requests: list[dict] = []

    def _send(api_kwargs, *_a, **_k):
        requests.append(api_kwargs)
        if len(requests) > 3:  # bound the broken loop so the assertion, not a hang, reports it
            raise InterruptedError("Agent interrupted during API call")
        raise _auth_401()

    monkeypatch.setattr(credential_auth_recovery, "TRANSIENT_AUTH_BACKOFF_S", (5.0,))

    def _recover_with_pool(**kwargs):
        # The episode's one forced refresh already happened, so this 401 takes the backoff wait,
        # and the user's /stop lands while it waits.
        agent._interrupt_requested = True
        episode = {"refreshes": 1, "waits": 0, "last": time.monotonic()}
        return credential_auth_recovery._retry_after_backoff(
            agent, SimpleNamespace(entries=list), episode, failed_key=agent.api_key,
            entry_id="e1", credential_id="e1", has_retried_429=kwargs["has_retried_429"])

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("run_agent._set_interrupt"),
        patch("agent.interrupt_control._set_interrupt"),
        patch.object(agent, "_interruptible_api_call", side_effect=_send),
        patch.object(agent, "_recover_with_credential_pool", side_effect=_recover_with_pool),
    ):
        started = time.monotonic()
        result = agent.run_conversation("hello")

    assert len(requests) == 1, f"/stop during the 401 backoff still sent {len(requests) - 1} more request(s)"
    assert result["interrupted"] is True
    assert time.monotonic() - started < 4.0, "the interrupted backoff waited out its full delay"
