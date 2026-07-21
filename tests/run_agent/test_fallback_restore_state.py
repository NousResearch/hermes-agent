"""Restore-time fallback state: streaming preference, cooldown gating, and
session-level unavailable-key suppression.

Covers the re-scoped #58984 behavior:

* ``_disable_streaming`` is agent-wide, so a fallback provider that rejects
  streaming would otherwise leave the restored PRIMARY stuck non-streaming for
  the rest of the session.  ``_try_activate_fallback`` snapshots the primary's
  value on the primary→fallback transition and ``restore_primary_runtime``
  puts that exact value back — so a disable the primary earned itself (e.g. a
  Bedrock IAM streaming denial) stays sticky, while a fallback-only disable is
  undone.

* While the primary's rate-limit cooldown is armed, ``restore_primary_runtime``
  must stay gated WITHOUT resetting ``_fallback_index`` — the exhausted index
  is the #24996 anti-storm throttle (see
  tests/run_agent/test_24996_fallback_exhaustion_cooldown.py).  Once the
  cooldown expires, restoration resets the index as usual.

* ``_unavailable_fallback_keys`` records chain entries that are locally
  unusable (missing key/env, unconfigured provider) for SESSION-level
  suppression; restoration must not clear it.
"""

import time
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent(fallback_model=None):
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://api.openai.com/v1", api_key="fb-key-1234"):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = api_key
    return client


def _activate(agent):
    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), None),
    ):
        assert agent._try_activate_fallback() is True


def _restore(agent):
    with patch("run_agent.OpenAI", return_value=MagicMock()):
        return agent._restore_primary_runtime()


_FB = {"provider": "openai", "model": "gpt-4o"}
_FB_CHAIN = [
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "openai", "model": "gpt-4o-mini"},
]


class TestStreamingPreferenceRestore:
    def test_fallback_stream_rejection_does_not_stick_to_primary(self):
        agent = _make_agent(fallback_model=_FB)
        assert not getattr(agent, "_disable_streaming", False)

        _activate(agent)
        # The fallback provider rejects streaming mid-turn.
        agent._disable_streaming = True

        assert _restore(agent) is True
        assert agent._disable_streaming is False

    def test_primary_earned_disable_stays_sticky_across_fallback(self):
        agent = _make_agent(fallback_model=_FB)
        # The PRIMARY itself disabled streaming (e.g. Bedrock IAM denial of
        # InvokeModelWithResponseStream) — that is session-sticky by design.
        agent._disable_streaming = True

        _activate(agent)
        assert _restore(agent) is True
        assert agent._disable_streaming is True

    def test_chain_switch_does_not_overwrite_primary_snapshot(self):
        agent = _make_agent(fallback_model=_FB_CHAIN)
        assert not getattr(agent, "_disable_streaming", False)

        _activate(agent)                    # primary → fallback[0]
        agent._disable_streaming = True     # fallback[0] rejects streaming
        _activate(agent)                    # fallback[0] → fallback[1] (chain switch)

        # The chain switch must not re-snapshot the (now fallback-tainted)
        # flag as the "primary" value.
        assert _restore(agent) is True
        assert agent._disable_streaming is False

    def test_restore_without_prior_fallback_leaves_flag_alone(self):
        agent = _make_agent(fallback_model=_FB)
        agent._disable_streaming = True
        # No fallback activation this turn: restore is a no-op and must not
        # touch the primary's own streaming state.
        assert _restore(agent) is False
        assert agent._disable_streaming is True


class TestCooldownGatingPreservesThrottleState:
    def test_restore_during_cooldown_keeps_index_and_fallback(self):
        agent = _make_agent(fallback_model=_FB)
        _activate(agent)
        fallback_model = agent.model
        # Walk the chain to exhaustion, then arm the primary cooldown (as a
        # rate-limit failover would).
        agent._fallback_index = len(agent._fallback_chain)
        agent._rate_limited_until = time.monotonic() + 60

        assert _restore(agent) is False
        # Gated restore must not reset the exhausted index — it is the
        # cross-turn replay throttle for #24996 — nor leave the fallback.
        assert agent._fallback_index == len(agent._fallback_chain)
        assert agent._fallback_activated is True
        assert agent.model == fallback_model

    def test_restore_after_cooldown_expiry_resets_index(self):
        agent = _make_agent(fallback_model=_FB)
        _activate(agent)
        agent._fallback_index = len(agent._fallback_chain)
        agent._rate_limited_until = time.monotonic() - 1

        assert _restore(agent) is True
        assert agent._fallback_index == 0
        assert agent._fallback_activated is False


class TestUnavailableKeySuppressionIsSessionScoped:
    def test_restore_preserves_unavailable_fallback_keys(self):
        agent = _make_agent(fallback_model=_FB_CHAIN)
        _activate(agent)
        poisoned = {("openai", "gpt-4o-mini", "")}
        agent._unavailable_fallback_keys = set(poisoned)

        assert _restore(agent) is True
        # Locally-unusable entries are suppressed for the whole session;
        # only a fallback-config reload may clear the memo.
        assert agent._unavailable_fallback_keys == poisoned
