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
