"""Regression: /stop must halt a turn that is looping across model fallbacks.

Motivating incident (2026-07-14, Discord session 20260705_194244_a6a3ab50):
Ace /stop'd a turn; the UI said "Stopped" but the turn kept running for 4.5
minutes, bouncing across rate-limited fallback providers
(claude-apx-9 -> apx-1 -> apx-2) and appending 28 transcript rows AFTER the
stop -- clobbering three /undo commands.

Root cause: `agent.interrupt()` sets a COOPERATIVE flag (`_interrupt_requested`)
that is only checked between tool calls and in the API-error branch. The
fallback/retry loop (`while retry_count < max_retries:` in
agent/conversation_loop.py) resets `retry_count = 0` and `continue`s on each
`_try_activate_fallback(...)` with NO interrupt check at the loop top -- so a
turn churning through fallbacks never notices the stop until it completes.

Fix (B): an `_interrupt_requested` check at the TOP of the fallback loop that
returns an interrupted result before issuing the next model call -- which also
stops the turn from appending more rows (the incident's actual harm).

These tests drive the REAL `run_conversation` loop (not a mock that shortcuts
the fallback edge) so they exercise the actual checkpoint. RED-proof: without
the checkpoint, the loop issues a SECOND model call after the interrupt.
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
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
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


def _mock_response(content: str, finish_reason: str = "stop"):
    msg = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="fallback/model", usage=None)


class RateLimitError(Exception):
    status_code = 429

    def __init__(self):
        super().__init__("Error code: 429 - rate limit exceeded")
        self.response = SimpleNamespace(headers={})
        self.body = {"error": {"message": "rate limit exceeded"}}


_FB_CHAIN = [
    {
        "provider": "zai",
        "model": "glm-4.7",
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
    }
]


class TestStopHaltsFallbackLoopingTurn:
    """AC1 / AC1b / AC2: a /stop mid-fallback-loop halts before the next
    model call, runs the checkpoint without raising, and appends no further
    rows."""

    def _run_with_interrupt_on_first_fallback(self):
        """Shared driver: first call returns an EMPTY/malformed response ->
        eager fallback activates (retry_count=0, continue) -> the flag is set
        -> the loop MUST hit the top-of-loop checkpoint and stop before the
        next model call.

        The empty/malformed eager-fallback path (conversation_loop.py ~L1594)
        is the real gap: unlike the API-ERROR branch (~L3197), it does NOT
        check the interrupt flag before `continue`, so a /stop landing here is
        invisible until the turn otherwise completes. This is the path the
        loop-top checkpoint (B) is written to close."""
        agent = _make_agent_with_fallback(_FB_CHAIN)
        agent._api_max_retries = 5

        calls = []
        persist_calls = []

        def fake_api_call(api_kwargs):
            calls.append((agent.provider, agent.model))
            attempt = len(calls)
            if attempt == 1:
                # Simulate the /stop landing while we're mid-turn: the gateway
                # sets the cooperative flag via agent.interrupt().
                agent._interrupt_requested = True
                # Return a content_filter refusal -> the safety-refusal path
                # activates the fallback and `continue`s back to the inner-loop
                # top WITHOUT checking the interrupt flag (the real gap). The
                # new checkpoint at the loop top must then halt the turn.
                return _mock_response("", finish_reason="content_filter")
            # If the checkpoint works, we NEVER reach here. If it's missing,
            # the fallback loop issues this second call (RED).
            return _mock_response("zombie fallback response")

        mock_fb_client = MagicMock()
        mock_fb_client.api_key = "primary-key-abcdef12"
        mock_fb_client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        mock_fb_client._custom_headers = None
        mock_fb_client.default_headers = None

        def fake_persist(messages, conversation_history=None, *a, **k):
            persist_calls.append(len(messages))

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session", side_effect=fake_persist),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            result = agent.run_conversation("hello")
        return agent, result, calls, persist_calls

    def test_stop_halts_fallback_looping_turn(self):
        """AC1: no SECOND model call is issued after the interrupt; the loop
        returns an interrupted (completed=False) result."""
        agent, result, calls, _ = self._run_with_interrupt_on_first_fallback()

        # Exactly ONE model call happened -- the checkpoint stopped the loop
        # before it could issue the fallback call. Without the fix, len==2.
        assert len(calls) == 1, (
            f"expected the fallback loop to halt at the interrupt checkpoint "
            f"after 1 call, but it issued {len(calls)} calls: {calls}"
        )
        assert result.get("completed") is False, (
            "an interrupted turn must not report completed=True"
        )

    def test_checkpoint_runs_without_error(self):
        """AC1b: the checkpoint EXECUTES (no NameError from a verbatim copy of
        the error-branch block, whose text uses undefined error_type/api_error
        at loop-top). The turn returns a well-formed interrupted dict."""
        agent, result, calls, _ = self._run_with_interrupt_on_first_fallback()

        # A non-raising return with the interrupted contract present.
        assert isinstance(result, dict)
        assert "final_response" in result
        assert result.get("completed") is False
        # The interrupt text must NOT claim an API-error context we're not in
        # (the empty/malformed path is a pre-request halt, not error handling).
        assert "handling API error" not in str(result.get("final_response", ""))

    def test_stop_suppresses_fallback_row_writes(self):
        """AC2 (the incident's actual harm): once the checkpoint fires, the
        stopped turn does not keep persisting rows via additional model calls.

        Proxy for "no further transcript rows": the number of model calls is
        bounded at 1, so no further assistant/tool rows are generated by the
        fallback loop. (append_message rows are produced per model-call turn;
        halting the loop halts the row production.)"""
        agent, result, calls, persist_calls = (
            self._run_with_interrupt_on_first_fallback()
        )
        assert len(calls) == 1, (
            "the stopped turn must not issue further model calls that would "
            f"append more rows; got {len(calls)} calls"
        )
        # The truncated transcript IS persisted once on the interrupt unwind
        # (so the halt is durable), but no further per-call persists happen.
        assert len(persist_calls) >= 1, (
            "the interrupt unwind must persist the truncated transcript"
        )

    def test_stop_cleans_up_task_resources(self):
        """Greptile P1: an early interrupt return bypasses finalize_turn, so the
        shared unwind must tear down task-scoped resources (open VMs / browser
        sessions / remote agents) itself — otherwise a /stop mid-turn leaks them.
        Assert _cleanup_task_resources is called on the checkpoint path."""
        agent = _make_agent_with_fallback(_FB_CHAIN)
        agent._api_max_retries = 5
        cleanup_calls = []

        def fake_api_call(api_kwargs):
            agent._interrupt_requested = True
            return _mock_response("", finish_reason="content_filter")

        mock_fb_client = MagicMock()
        mock_fb_client.api_key = "primary-key-abcdef12"
        mock_fb_client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        mock_fb_client._custom_headers = None
        mock_fb_client.default_headers = None

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(
                agent,
                "_cleanup_task_resources",
                side_effect=lambda tid: cleanup_calls.append(tid),
            ),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch("agent.agent_runtime_helpers.time.sleep"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch("agent.model_metadata.get_model_context_length", return_value=200000),
        ):
            agent.run_conversation("hello")

        assert cleanup_calls, (
            "the interrupt checkpoint must tear down task-scoped resources "
            "(it bypasses finalize_turn, which normally does this)"
        )
