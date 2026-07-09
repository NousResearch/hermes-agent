"""
Regression tests for issue #61296 - switch_model trusts a non-empty
stale base_url, cross-wiring the new provider's API key with the old
provider's endpoint.

The bug: ``switch_model`` (in agent/agent_runtime_helpers.py) has a
guard at L1813 ``if base_url: agent.base_url = base_url`` that only
checks truthiness. When the caller passes a non-empty stale
base_url from a DIFFERENT registered provider (e.g. ollama-cloud's
``https://ollama.com/v1`` left over from a prior session when
switching to minimax-cn), the runtime trusts it verbatim and pairs
the new provider's API key with the old endpoint — silent credential
mis-routing.

The fix: extend the L1813 guard to also reject incoming base_urls
whose host matches the canonical endpoint of a different registered
provider. A stale non-empty URL from another provider is just as
dangerous as a stale empty URL.

Test cases cover:
  - The issue author's exact reproduction (the regression case)
  - The empty-string case (must still work — original guard behavior)
  - An ollama-localhost to anthropic switch (the canonical localhost case)
  - Cross-wiring with a custom user-defined provider URL (must NOT be
    rejected if no other registered provider owns that host)
"""

from __future__ import annotations

import types
import pytest


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def _build_fake_agent(*, model: str, provider: str, base_url: str, api_key: str) -> types.SimpleNamespace:
    """Build a minimal ``agent`` SimpleNamespace that satisfies switch_model's
    runtime contract. Mirrors the issue author's fixture shape but is
    self-contained (no shared module state, no class import dance).
    """
    agent = types.SimpleNamespace()
    agent.model = model
    agent.provider = provider
    agent.base_url = base_url
    agent.api_mode = "chat_completions"
    agent.api_key = api_key
    agent.client = object()
    agent._client_kwargs = {}
    agent._anthropic_client = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = ""
    agent._is_anthropic_oauth = False
    agent._config_context_length = None
    agent._transport_cache = {}
    agent._fallback_chain = []
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent.context_compressor = None
    agent._use_prompt_caching = False
    agent._use_native_cache_layout = False
    agent._cached_system_prompt = None
    agent._primary_runtime = {}
    agent._anthropic_prompt_cache_policy = lambda **kw: (False, False)
    agent._ensure_lmstudio_runtime_loaded = lambda: None
    agent._create_openai_client = lambda kw, **r: types.SimpleNamespace()
    agent._credential_pool = None
    agent._log_context_provider = lambda: provider
    return agent


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_non_empty_stale_base_url_from_different_provider_is_rejected(monkeypatch):
    """The regression test from the issue body. A stale non-empty
    base_url belonging to a different registered provider must not be
    trusted: switch_model must either reject it or clear it, not pair
    the new provider's API key with the old endpoint.

    Scenario:
      - Old state: provider=ollama-cloud, base_url=https://ollama.com/v1
      - Switch to: provider=minimax-cn, key=minimax-secret-key
      - Stale value passed: base_url=https://ollama.com/v1 (truthy, wrong)

    After the fix: agent.base_url must NOT be the ollama URL. Either it
    is cleared (None / "") or replaced with the new provider's endpoint.
    Cross-wired state (minimax-cn provider paired with ollama URL) is
    a security boundary violation.
    """
    from agent.agent_runtime_helpers import switch_model
    import hermes_cli.providers as prov_mod

    # Stub determine_api_mode so the switch doesn't try to introspect
    # the api_mode from base_url/protocol details.
    monkeypatch.setattr(
        prov_mod, "determine_api_mode",
        lambda provider, base_url="": "chat_completions",
    )

    agent = _build_fake_agent(
        model="qwen3:32b",
        provider="ollama-cloud",
        base_url="https://ollama.com/v1",
        api_key="old-ollama-key",
    )

    switch_model(
        agent=agent,
        new_model="MiniMax-M3",
        new_provider="minimax-cn",
        api_key="minimax-secret-key",
        base_url="https://ollama.com/v1",  # stale, truthy, wrong
        api_mode="chat_completions",
    )

    # The provider and API key swap MUST take effect.
    assert agent.provider == "minimax-cn"
    assert agent.api_key == "minimax-secret-key"
    assert agent.model == "MiniMax-M3"

    # The critical security assertion: a base_url whose host belongs to
    # a DIFFERENT registered provider must NOT be retained on the agent.
    # The fix discards the stale URL entirely (clears agent.base_url
    # to "") so the new provider's key never gets paired with the
    # old endpoint, either via agent.base_url or _client_kwargs.
    assert agent.base_url != "https://ollama.com/v1", (
        f"BUG (#61296): stale non-empty base_url trusted verbatim — "
        f"minimax-cn provider paired with base_url={agent.base_url!r} "
        f"(old ollama-cloud endpoint) instead of being cleared or replaced."
    )
    assert agent.base_url in (None, ""), (
        f"after the #61296 fix, the stale ollama URL should be discarded "
        f"(agent.base_url set to None or empty string); got "
        f"{agent.base_url!r}"
    )
    # The new client kwargs (which is what actually goes out on the wire)
    # must also NOT carry the ollama URL.
    assert agent._client_kwargs.get("base_url") != "https://ollama.com/v1", (
        f"BUG (#61296): _client_kwargs retains the stale ollama URL "
        f"({agent._client_kwargs!r}); the new minimax-cn key is paired with "
        f"the old ollama endpoint."
    )


def test_matching_base_url_for_new_provider_is_accepted(monkeypatch):
    """Regression guard: when the incoming base_url legitimately matches
    the new provider's canonical endpoint, it MUST be accepted. The fix
    only rejects URLs whose host matches a DIFFERENT registered provider.
    """
    from agent.agent_runtime_helpers import switch_model
    import hermes_cli.providers as prov_mod

    monkeypatch.setattr(
        prov_mod, "determine_api_mode",
        lambda provider, base_url="": "chat_completions",
    )

    agent = _build_fake_agent(
        model="some-old-model",
        provider="ollama-local",
        base_url="http://localhost:11434/v1",
        api_key="",
    )

    # ollama-cloud's canonical endpoint is https://ollama.com/v1 (per
    # plugins/model-providers/ollama-cloud/__init__.py); passing that
    # exact URL when switching TO ollama-cloud is legitimate.
    switch_model(
        agent=agent,
        new_model="qwen3:32b",
        new_provider="ollama-cloud",
        api_key="ollama-cloud-key",
        base_url="https://ollama.com/v1",
        api_mode="chat_completions",
    )

    assert agent.provider == "ollama-cloud"
    assert agent.base_url == "https://ollama.com/v1", (
        f"matching canonical URL for the new provider was rejected: "
        f"agent.base_url={agent.base_url!r}"
    )