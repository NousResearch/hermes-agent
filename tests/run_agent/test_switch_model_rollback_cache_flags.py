"""Regression: switch_model() rollback must cover the prompt-cache flags.

Why: ``switch_model`` snapshots model/provider/client/etc. and restores them
in an ``except`` block when a swap step fails, so the live agent is never left
with a new model name paired with the old client. But the two prompt-cache
flags -- ``_use_prompt_caching`` and ``_use_native_cache_layout`` -- were
re-evaluated OUTSIDE the protected ``try`` block (after the rollback/``raise``)
and were absent from the snapshot. Two consequences:

  1. If the flag re-evaluation itself raises, it happens *past* the rollback
     block, so the agent is left HALF-SWAPPED -- model/provider/client already
     committed to the new values, operation failed, no rollback at all.
  2. Because the flags aren't snapshotted, no rollback path can restore them.

The fix moves the flag re-evaluation INSIDE the ``try`` (before the client
rebuild) and adds both flags to ``_snapshot``, so any failure during the swap
restores model + provider + client AND the cache flags atomically.

What: forces a swap failure and asserts the agent is fully restored.

Test: both cases below go RED before the fix (case 1 leaves the agent
half-swapped; case 2 leaves drifted flags) and GREEN after.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names):
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


def _make_agent(provider="custom", base_url="https://my-llm.example.com/v1"):
    """Minimal openai-mode AIAgent suitable for driving switch_model()."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url=base_url,
            provider=provider,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def test_cache_policy_failure_rolls_back_everything():
    """If the prompt-cache policy re-eval raises, the whole swap rolls back.

    Before the fix the policy was evaluated OUTSIDE the try, so this raise
    left the agent half-swapped (new model/provider committed, no rollback).
    After the fix the policy is inside the try, so model/provider/client/flags
    are all restored.
    """
    agent = _make_agent(provider="custom")

    old_model = agent.model
    old_provider = agent.provider
    old_client = agent.client
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False

    with (
        # Client rebuild succeeds...
        patch.object(agent, "_create_openai_client", return_value=MagicMock()),
        # ...but the cache-policy re-eval blows up.
        patch.object(
            agent,
            "_anthropic_prompt_cache_policy",
            side_effect=RuntimeError("policy boom"),
        ),
    ):
        try:
            agent.switch_model(
                "some/new-model",
                "openrouter",
                api_key="new-key-abcdefgh",
                base_url="https://bad.example.invalid/v1",
                api_mode="openai",
            )
        except RuntimeError:
            pass  # re-raised after rollback -- expected.

    # No half-swap: model/provider/client restored.
    assert agent.model == old_model
    assert agent.provider == old_provider
    assert agent.client is old_client
    # Flags untouched / restored.
    assert agent._use_prompt_caching is False
    assert agent._use_native_cache_layout is False


def test_client_build_failure_restores_cache_flags():
    """A failed client rebuild restores model/provider AND the cache flags.

    The swap recomputes the flags to the OPPOSITE policy before the client
    rebuild fails; rollback must put the pre-swap flag values back, not leave
    the new policy bolted onto the restored old model.
    """
    agent = _make_agent(provider="custom")

    agent._use_prompt_caching = True
    agent._use_native_cache_layout = True
    old_model = agent.model
    old_provider = agent.provider

    with (
        patch.object(
            agent,
            "_anthropic_prompt_cache_policy",
            return_value=(False, False),
        ),
        patch.object(
            agent,
            "_create_openai_client",
            side_effect=RuntimeError("connection refused"),
        ),
    ):
        try:
            agent.switch_model(
                "some/other-model",
                "openrouter",
                api_key="new-key-abcdefgh",
                base_url="https://bad.example.invalid/v1",
                api_mode="openai",
            )
        except RuntimeError:
            pass

    assert agent.model == old_model
    assert agent.provider == old_provider
    assert agent._use_prompt_caching is True
    assert agent._use_native_cache_layout is True
