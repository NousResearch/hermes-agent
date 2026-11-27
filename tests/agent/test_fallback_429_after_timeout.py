"""Regression test for #32646: fallback_providers not activated when
HTTP 429 follows a successful primary-transport recovery.

Reproduces the (timeout x N -> recover -> 429 -> no fallback) sequence
reported against zai/glm-5.1 -> zai/glm-4.7 on the Telegram gateway.

Scenario:
  1. ``_try_recover_primary_transport()`` succeeds after 3 timeouts and
     resets ``retry_count = 0`` so the rebuilt primary client gets one
     more attempt.
  2. The next attempt hits HTTP 429.
  3. Before this fix, an eager-fallback attempt that lost its race with
     a concurrent session mutating the on-disk credential pool could
     leave ``_fallback_index`` advanced past the chain length without
     setting ``_fallback_activated`` to True.  The subsequent 429s then
     short-circuited the eager-fallback gate (``_fallback_index >=
     len(_fallback_chain)``), so the retry budget burned on the primary
     model with no fallback ever attempted.
  4. The fix resets ``_fallback_index`` / ``_fallback_activated`` /
     ``TurnRetryState.has_retried_429`` after transport recovery so the post-recovery
     429 always gets a fresh fallback-chain attempt.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent

def _make_tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

def _make_agent_with_fallback(fb_chain):
    """Build a minimal AIAgent with the given fallback chain configured."""
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs()),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="primary-key-abcdef12",
            base_url="https://open.bigmodel.cn/api/coding/paas/v4",
            provider="zai",
            model="glm-5.1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fb_chain,
        )
        agent.client = MagicMock()
        return agent

def _mock_response(content: str):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="fallback/model", usage=None)

class ReadTimeout(Exception):
    pass

class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}

# Regression: post-recovery reset of fallback-chain state

class TestFallbackChainResetOnTransportRecovery:
    """The bug surfaced when a stale ``_fallback_index`` survived the
    transport-recovery cycle.  These tests exercise the reset directly
    via the same call sequence the conversation loop performs, without
    needing to drive the full ``run_conversation`` loop."""

    def test_run_conversation_fallbacks_on_429_after_timeout_recovery(self):
        """Full loop regression for #32646.

        Start the turn with the fallback chain already burned, matching
        the stale state reported in the issue. Two transient timeouts
        exhaust the retry loop and trigger primary transport recovery.
        The next primary attempt returns 429. The conversation loop must
        reset the stale fallback-chain state during recovery so that the
        post-recovery 429 activates the configured fallback provider.
        """
        fb_chain = [
            {
                "provider": "zai",
                "model": "glm-4.7",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain)
        agent._api_max_retries = 2

        calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            attempt = len(calls)
            if attempt == 1:
                agent._fallback_index = len(agent._fallback_chain)
                agent._fallback_activated = False
            if attempt <= 2:
                raise ReadTimeout("read timed out")
            if attempt == 3:
                raise RateLimitError()
            return _mock_response("Recovered via fallback")

        mock_fb_client = MagicMock()
        mock_fb_client.api_key = "primary-key-abcdef12"
        mock_fb_client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        mock_fb_client._custom_headers = None
        mock_fb_client.default_headers = None

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ) as mock_resolve,
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is True
        assert result["final_response"] == "Recovered via fallback"
        assert calls == [
            ("zai", "glm-5.1"),
            ("zai", "glm-5.1"),
            ("zai", "glm-5.1"),
            ("zai", "glm-4.7"),
        ]
        mock_resolve.assert_called_once()
        assert agent._fallback_activated is True
        assert agent.model == "glm-4.7"

# Defensive: pure-timeout cycle without 429 still works
