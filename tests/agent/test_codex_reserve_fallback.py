"""Tests for Luna Reserve (gpt-reserve) fallback wiring.

Covers: the fallback-chain prepend for Codex OAuth primaries (agent_init) and
the pool-cooldown bypass in the Codex token reader (auxiliary_client) so a
regular-allowance 429 can still reach the reserve quota on the same credential.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.agent_init import _init_fallback_chain
from agent.auxiliary_client import _read_codex_access_token


def _make_agent(provider="openai-codex"):
    return SimpleNamespace(
        provider=provider,
        _fallback_chain=[],
        _fallback_index=0,
        _fallback_activated=False,
        quiet_mode=True,
    )


def test_reserve_prepended_for_codex_primary():
    agent = _make_agent()
    _init_fallback_chain(agent, [{"provider": "nous", "model": "upstage/solar-pro4:free"}])
    assert agent._fallback_chain[0] == {"provider": "openai-codex", "model": "gpt-reserve"}
    assert agent._fallback_chain[1] == {"provider": "nous", "model": "upstage/solar-pro4:free"}


def test_reserve_not_duplicated_when_already_configured():
    agent = _make_agent()
    _init_fallback_chain(agent, [{"provider": "openai-codex", "model": "gpt-reserve"}])
    assert agent._fallback_chain.count({"provider": "openai-codex", "model": "gpt-reserve"}) == 1


def test_reserve_not_added_for_non_codex_primary():
    agent = _make_agent(provider="nous")
    _init_fallback_chain(agent, [{"provider": "openrouter", "model": "x/model"}])
    assert all(e.get("model") != "gpt-reserve" for e in agent._fallback_chain)


def test_reserve_keeps_configured_chain_order():
    agent = _make_agent()
    _init_fallback_chain(agent, [{"provider": "openrouter", "model": "x/model"}])
    assert [e["model"] for e in agent._fallback_chain] == ["gpt-reserve", "x/model"]


def test_reserve_rearmed_after_switch_to_codex():
    """A /model switch TO Codex must re-arm the reserve rung.

    The prune drops entries targeting the OLD provider (nous) — the user just
    rejected it — so the chain after the switch is the reserve rung alone.
    """
    from agent.agent_runtime_helpers import _finish_switch

    agent = _make_agent(provider="openai-codex")
    agent._fallback_chain = [{"provider": "nous", "model": "upstage/solar-pro4:free"}]
    _finish_switch(agent, "openai-codex", "nous", "openai-codex")
    assert agent._fallback_chain == [{"provider": "openai-codex", "model": "gpt-reserve"}]


def test_reserve_pruned_after_switch_away_from_codex():
    """A /model switch AWAY from Codex leaves the reserve pruned."""
    from agent.agent_runtime_helpers import _finish_switch

    agent = _make_agent(provider="nous")
    agent._fallback_chain = [
        {"provider": "openai-codex", "model": "gpt-reserve"},
        {"provider": "nous", "model": "upstage/solar-pro4:free"},
    ]
    _finish_switch(agent, "nous", "openai-codex", "nous")
    assert all(e.get("model") != "gpt-reserve" for e in agent._fallback_chain)


def test_read_codex_token_uses_pool_entry_in_cooldown():
    """A pool entry benched for the regular quota still serves the reserve model."""
    entry = SimpleNamespace(runtime_api_key="reserve-token", access_token="")
    pool = SimpleNamespace(entries=lambda: [entry])
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
        patch("agent.credential_pool.load_pool", return_value=pool),
    ):
        assert _read_codex_access_token(allow_cooldown=True) == "reserve-token"


def test_read_codex_token_does_not_bypass_cooldown_by_default():
    """Other callers (image_gen, aux) keep the old behavior: benched → None."""
    entry = SimpleNamespace(runtime_api_key="reserve-token", access_token="")
    pool = SimpleNamespace(entries=lambda: [entry])
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
        patch("agent.credential_pool.load_pool", return_value=pool),
    ):
        assert _read_codex_access_token() is None


def test_read_codex_token_skips_dead_pool_entries():
    """Revoked (DEAD) entries must not serve the reserve — their tokens are unusable."""
    dead = SimpleNamespace(runtime_api_key="dead-token", access_token="", last_status="dead")
    live = SimpleNamespace(runtime_api_key="live-token", access_token="", last_status="exhausted")
    pool = SimpleNamespace(entries=lambda: [dead, live])
    with (
        patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
        patch("agent.credential_pool.load_pool", return_value=pool),
    ):
        assert _read_codex_access_token(allow_cooldown=True) == "live-token"


def test_read_codex_token_prefers_available_pool_selection():
    entry = SimpleNamespace(runtime_api_key="selected-token", access_token="")
    with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, entry)):
        assert _read_codex_access_token() == "selected-token"


def _fallback_agent():
    """Minimal agent for try_activate_fallback: chain [gpt-reserve, openrouter]."""
    return SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.6-luna",
        base_url="",
        requested_provider="openai-codex",
        api_mode="codex_responses",
        _fallback_chain=[
            {"provider": "openai-codex", "model": "gpt-reserve"},
            {"provider": "openrouter", "model": "x/model"},
        ],
        _fallback_index=0,
        _fallback_activated=False,
        _unavailable_fallback_keys=None,
        _config_context_length=None,
        _reasoning_echo_flag=False,
        _provider_fallback_active=False,
        _provider_fallback_route=None,
        _pending_fallback_notice=None,
        _rate_limited_until=0,
        runtime_capabilities=None,
        quiet_mode=True,
        _buffer_status=lambda notice: None,
        _anthropic_prompt_cache_policy=lambda **kw: (False, False),
        _ensure_lmstudio_runtime_loaded=lambda: None,
    )


def test_reserve_exhausted_advances_to_next_provider():
    """When the reserve rung cannot resolve (quota gone / provider unconfigured),
    try_activate_fallback must advance to the next provider in the chain — the
    reserve must not block the fallback (bot review point 2)."""
    from agent.chat_completion_helpers import try_activate_fallback

    agent = _fallback_agent()
    client_mock = SimpleNamespace(base_url="https://openrouter.ai/v1")

    def _resolve(provider, model=None, **kw):
        if provider == "openai-codex":
            return None, None  # reserve unavailable → skip
        return client_mock, model

    with (
        patch("agent.fallback_cooldown._arm_rate_limit_cooldown", return_value=None),
        patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve),
        patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m),
        patch("agent.chat_completion_helpers._fallback_api_mode_resolved", return_value="chat_completions"),
        patch("agent.chat_completion_helpers._rebind_fallback_credential_pool"),
        patch("agent.client_lifecycle._swap_fallback_clients"),
        patch("agent.agent_runtime_helpers.sync_credential_pool_entry_id"),
        patch("agent.chat_completion_helpers._update_fallback_context_compressor"),
        patch("agent.chat_completion_helpers._reresolve_fallback_reasoning_config"),
        patch("agent.chat_completion_helpers._rescope_fallback_extra_body"),
        patch("agent.chat_completion_helpers.rewrite_prompt_model_identity"),
        patch("agent.chat_completion_helpers._reset_stale_streak"),
        patch("agent.native_compaction.resolve_native_compaction_capabilities", return_value={}),
    ):
        assert try_activate_fallback(agent) is True

    # Advanced past the reserve to the next provider.
    assert agent.provider == "openrouter"
    assert agent.model == "x/model"
    assert agent._fallback_index == 2
    assert agent._fallback_activated is True
    # The reserve rung was marked unavailable, not silently retried forever.
    assert ("openai-codex", "gpt-reserve", "") in agent._unavailable_fallback_keys
