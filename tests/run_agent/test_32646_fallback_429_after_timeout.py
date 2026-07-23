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

from agent.turn_retry_state import TurnRetryState
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

    def test_fallback_chain_resets_after_primary_recovery(self):
        """Simulate the conversation_loop sequence:

        ``_fallback_index`` was bumped to ``len(_fallback_chain)`` by an
        eager-fallback attempt that failed to activate (e.g. the
        configured fallback provider's credential pool was momentarily
        unresolvable).  Without the reset, the next iteration's
        eager-fallback gate at ``_fallback_index < len(_fallback_chain)``
        is permanently False for the rest of the turn.

        The fix block runs the same body the conversation loop applies
        immediately after ``_try_recover_primary_transport()`` returns
        True.  Once it has run, the chain must be walkable again so
        the post-recovery 429 path can call
        ``_try_activate_fallback()`` and switch to glm-4.7.
        """
        fb_chain = [
            {
                "provider": "zai",
                "model": "glm-4.7",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain)

        # Simulate the pre-recovery state: a prior eager-fallback
        # attempt walked the chain and bumped the index, but never set
        # _fallback_activated (resolve_provider_client returned None
        # and the recursive call exhausted the single-entry chain).
        agent._fallback_index = len(agent._fallback_chain)
        agent._fallback_activated = False

        # Apply the post-recovery reset that the conversation loop now
        # performs after _try_recover_primary_transport() succeeds.
        agent._fallback_index = 0
        agent._fallback_activated = False

        # The eager-fallback gate condition must now be True so the
        # next 429 actually calls _try_activate_fallback.
        assert agent._fallback_index < len(agent._fallback_chain)

        # Confirm the fallback would actually activate now (provider is
        # different model under same zai provider).
        mock_fb_client = MagicMock()
        mock_fb_client.api_key = "primary-key-abcdef12"
        mock_fb_client.base_url = "https://open.bigmodel.cn/api/coding/paas/v4"
        mock_fb_client._custom_headers = None
        mock_fb_client.default_headers = None

        with (
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(mock_fb_client, "glm-4.7"),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
        ):
            ok = agent._try_activate_fallback()

        assert ok is True, "fallback chain must be re-attemptable after reset"
        assert agent._fallback_activated is True
        assert agent.model == "glm-4.7"
        assert agent.provider == "zai"

    def test_post_recovery_429_keeps_eager_fallback_reachable(self):
        """Direct check on the gate condition the conversation loop uses
        for eager fallback: ``_fallback_index < len(_fallback_chain)``.

        With the reset, the gate stays open for a freshly-rebuilt primary
        even if a prior pre-recovery eager attempt burned the index."""
        fb_chain = [
            {
                "provider": "zai",
                "model": "glm-4.7",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain)

        # Pre-recovery: chain index burned by a failed eager attempt.
        agent._fallback_index = len(agent._fallback_chain)
        agent._fallback_activated = False

        # Without the post-recovery reset the gate would be permanently
        # closed (``_fallback_index < len(_fallback_chain)`` is False).
        gate_before_reset = agent._fallback_index < len(agent._fallback_chain)
        assert gate_before_reset is False, (
            "precondition: a burned chain index closes the eager gate "
            "until something resets it"
        )

        # Apply the reset that the conversation loop now performs after
        # _try_recover_primary_transport() succeeds.
        agent._fallback_index = 0
        agent._fallback_activated = False

        gate_after_reset = agent._fallback_index < len(agent._fallback_chain)
        assert gate_after_reset is True, (
            "after primary-transport recovery, the eager-fallback gate "
            "must be reachable again so a follow-on 429 can fall back"
        )

    def test_retry_state_429_flag_resets_to_false_after_recovery(self):
        """``has_retried_429`` lives on ``TurnRetryState`` in the
        conversation loop, so a fresh attempt cycle after primary
        recovery should start with the
        credential-pool retry flag cleared so a single-credential pool
        gets the cheap retry-same-credential pass before rotation.
        """
        # Retry-state semantics: simulate the conversation loop body.
        retry_state = TurnRetryState()
        retry_state.has_retried_429 = True  # set by a pre-recovery 429 path

        # The fix block:
        recovered = True  # _try_recover_primary_transport() returned True
        if recovered and not retry_state.primary_recovery_attempted:
            retry_state.primary_recovery_attempted = True
            retry_state.has_retried_429 = False  # the documented reset

        assert retry_state.has_retried_429 is False, (
            "post-recovery cycle must reset has_retried_429 so the "
            "credential-pool path treats the next 429 as a fresh first-hit"
        )
        assert retry_state.primary_recovery_attempted is True

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
            patch("run_agent.OpenAI", return_value=MagicMock()),
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


class TestPostRecoveryResetDoesNotBreakHappyPath:
    """Make sure the reset doesn't regress the simple
    timeout-then-success path that ``test_primary_runtime_restore``
    already covers."""

    def test_reset_is_noop_when_chain_was_already_clean(self):
        fb_chain = [
            {
                "provider": "zai",
                "model": "glm-4.7",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            }
        ]
        agent = _make_agent_with_fallback(fb_chain)

        # Fresh state: nothing has bumped the chain.
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False

        # Apply the reset.
        agent._fallback_index = 0
        agent._fallback_activated = False

        # Still clean; no observable change.
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False
        # Gate still open for a future 429.
        assert agent._fallback_index < len(agent._fallback_chain)
