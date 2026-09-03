"""Regression test: switch_model() must roll back to the pre-swap state if the
LM Studio preload / context resolution raises.

The switch_model transaction has three failure windows:

  Window A (client-rebuild phase): wrapped in try/except → _restore_snapshot()
    (covered by test_switch_model_rollback.py)

  Window B (LM Studio preload + context resolution phase):
    ``_ensure_lmstudio_runtime_loaded`` is a real network I/O call to the LM
    Studio management API. Before the fix it sat BETWEEN the Window A
    try/except and the Window C try/except, OUTSIDE any rollback. A failure
    there left the agent with a NEW host route (model/provider/client already
    committed in Window A) but the OLD engine pre-cap and OLD _primary_runtime
    (both updated only AFTER Window B) — the exact incoherence class the
    switch_model rollback exists to prevent.

  Window C (engine transition phase): wrapped in try/except → _restore_snapshot()

This test drives the REAL ``agent.agent_runtime_helpers.switch_model`` with
``_ensure_lmstudio_runtime_loaded`` raising (simulating a network error / HTTP
error in the LM Studio management API call), and asserts that the previous
coherent route is FULLY preserved: model, provider, client, pre-cap, engine
route, and _primary_runtime.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_agent_lmstudio() -> AIAgent:
    """Agent on lmstudio (openai-compatible) with a coherent initial state.

    The initial state is a COMPLETE coherent route: model + provider + client
    + pre-cap + engine route + _primary_runtime all agree. After a failed
    LM Studio preload, every one of these must be restored to this state.
    """
    agent = AIAgent.__new__(AIAgent)

    # OLD (pre-swap) coherent route
    agent.provider = "lmstudio"
    agent.model = "lmstudio/old-model"
    agent.requested_provider = "lmstudio"
    agent.base_url = "http://localhost:1234/v1"
    agent.api_key = "lm-key-original"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock(name="OriginalLMStudioClient")
    agent._client_kwargs = {
        "api_key": "lm-key-original",
        "base_url": "http://localhost:1234/v1",
    }
    agent._config_context_length = 131072
    agent._pre_cap_context_length = 131072
    agent._primary_runtime = {
        "model": "lmstudio/old-model",
        "provider": "lmstudio",
        "base_url": "http://localhost:1234/v1",
        "api_mode": "chat_completions",
        "compressor_context_length": 131072,
    }

    # A mock context_compressor so the engine transition (Window C) is
    # REACHABLE but will NOT be reached because the LM Studio preload
    # (Window B) fails first.
    cc = MagicMock(name="ContextCompressor")
    cc.model = "lmstudio/old-model"
    cc.provider = "lmstudio"
    cc.base_url = "http://localhost:1234/v1"
    cc.api_key = "lm-key-original"
    cc.api_mode = "chat_completions"
    cc.context_length = 131072
    cc.pre_cap_context_length = 131072
    agent.context_compressor = cc

    # Other fields that switch_model touches
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = []
    agent._fallback_model = None
    agent._credential_pool = None
    agent._credential_pool_entry_id = None
    agent._reasoning_echo_flag = False
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent.reasoning_config = None
    agent._custom_providers = None

    return agent


def test_lmstudio_preload_failure_rolls_back_to_original_state():
    """When _ensure_lmstudio_runtime_loaded raises, the previous coherent route
    (model, provider, client, pre-cap, engine route, _primary_runtime) must be
    FULLY preserved — not left half-switched to the new model."""
    agent = _make_agent_lmstudio()

    original_client = agent.client
    original_primary_runtime = dict(agent._primary_runtime)
    original_pre_cap = agent._pre_cap_context_length
    original_engine = agent.context_compressor

    # Make the LM Studio preload raise (simulates a network error / HTTP error
    # in the LM Studio management API call).
    def boom(*_a, **_kw):
        raise RuntimeError("simulated LM Studio preload failure")

    agent._ensure_lmstudio_runtime_loaded = boom

    # Make the client rebuild SUCCEED (so we get past Window A and reach
    # Window B where the failure occurs).
    new_client = MagicMock(name="NewLMStudioClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        with pytest.raises(RuntimeError, match="simulated LM Studio preload failure"):
            agent.switch_model(
                new_model="lmstudio/new-model",
                new_provider="lmstudio",
                api_key="lm-key-new",
                base_url="http://localhost:1234/v1",
                api_mode="chat_completions",
            )

    # Core invariant: the previous coherent route is FULLY preserved.
    assert agent.model == "lmstudio/old-model", (
        f"model must be restored to old value, got {agent.model!r}"
    )
    assert agent.provider == "lmstudio", (
        f"provider must be restored, got {agent.provider!r}"
    )
    assert agent.client is original_client, (
        "client must be restored to the original object"
    )
    assert agent._pre_cap_context_length == original_pre_cap, (
        f"pre-cap must be restored, got {agent._pre_cap_context_length!r}"
    )
    assert agent._primary_runtime == original_primary_runtime, (
        f"_primary_runtime must be restored, got {agent._primary_runtime!r}"
    )
    # Engine route must be unchanged (the engine was NOT transitioned because
    # the failure happened before the engine transition).
    assert agent.context_compressor is original_engine, (
        "context_compressor must be the same object"
    )
    assert agent.context_compressor.model == "lmstudio/old-model", (
        "engine model must be restored"
    )
    assert agent.context_compressor.provider == "lmstudio", (
        "engine provider must be restored"
    )


def test_successful_switch_still_works_after_lmstudio_rollback_refactor():
    """Sanity check: the try/except wrapper around the LM Studio preload phase
    hasn't broken the happy path (LM Studio preload succeeds → switch completes)."""
    agent = _make_agent_lmstudio()

    new_client = MagicMock(name="NewLMStudioClient")
    agent._create_openai_client = lambda *_a, **_kw: new_client
    # LM Studio preload succeeds (returns a load result with context_length).
    agent._ensure_lmstudio_runtime_loaded = lambda *_a, **_kw: MagicMock(
        context_length=65536, rejected=False, load_attempted=True
    )
    # _lmstudio_load_was_unverified should return False for a successful load.
    agent._lmstudio_load_was_unverified = lambda *_a, **_kw: False
    # _effective_lmstudio_context_length should return a valid int.
    agent._effective_lmstudio_context_length = (
        lambda *a, **kw: 65536
    )

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="lmstudio/new-model",
            new_provider="lmstudio",
            api_key="lm-key-new",
            base_url="http://localhost:1234/v1",
            api_mode="chat_completions",
        )

    assert agent.model == "lmstudio/new-model"
    assert agent.provider == "lmstudio"
    assert agent.client is new_client


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
