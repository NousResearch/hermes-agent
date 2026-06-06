"""Regression tests for MiniMax OAuth routing outside the main agent path."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock


MINIMAX_ANTHROPIC_URL = "https://api.minimax.io/anthropic"


def test_auxiliary_resolve_provider_client_supports_minimax_oauth(monkeypatch):
    """Auxiliary clients should use MiniMax OAuth instead of falling back.

    MiniMax OAuth has provider id ``minimax-oauth`` and auth type
    ``oauth_minimax``. The main runtime resolver understands that pair, but the
    auxiliary resolver previously treated the auth type as unknown and returned
    ``(None, None)``, causing auto-routing to fall through to OpenRouter.
    """
    from agent import auxiliary_client as aux

    real_client = MagicMock(name="anthropic-client")
    built = {}

    def fake_build_anthropic_client(api_key, base_url, **kwargs):
        built["api_key"] = api_key
        built["base_url"] = base_url
        built["kwargs"] = kwargs
        return real_client

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        lambda: {
            "api_key": "mm-oauth-token",
            "base_url": MINIMAX_ANTHROPIC_URL,
            "source": "oauth",
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        fake_build_anthropic_client,
    )

    client, model = aux.resolve_provider_client(
        "minimax-oauth",
        model="MiniMax-M3",
    )

    assert model == "MiniMax-M3"
    assert isinstance(client, aux.AnthropicAuxiliaryClient)
    assert client.base_url == MINIMAX_ANTHROPIC_URL
    assert client.api_key == "mm-oauth-token"
    assert built["api_key"] == "mm-oauth-token"
    assert built["base_url"] == MINIMAX_ANTHROPIC_URL


def test_auxiliary_resolve_provider_client_supports_minimax_oauth_async(monkeypatch):
    """Async auxiliary callers should get the async Anthropic wrapper too."""
    from agent import auxiliary_client as aux

    monkeypatch.setattr(
        "hermes_cli.auth.resolve_minimax_oauth_runtime_credentials",
        lambda: {
            "api_key": "mm-oauth-token",
            "base_url": MINIMAX_ANTHROPIC_URL,
            "source": "oauth",
        },
    )
    monkeypatch.setattr(
        "agent.anthropic_adapter.build_anthropic_client",
        lambda *args, **kwargs: MagicMock(name="anthropic-client"),
    )

    client, model = aux.resolve_provider_client(
        "minimax-oauth",
        model="MiniMax-M3",
        async_mode=True,
    )

    assert model == "MiniMax-M3"
    assert isinstance(client, aux.AsyncAnthropicAuxiliaryClient)
    assert client.base_url == MINIMAX_ANTHROPIC_URL
    assert client.api_key == "mm-oauth-token"


def test_delegate_child_pool_does_not_share_stale_parent_pool(monkeypatch):
    """Delegation must not lease a parent pool whose provider is stale.

    A live TUI model switch can update ``parent.provider`` to ``minimax-oauth``
    while ``parent._credential_pool`` still points at ``openai-codex``. If the
    child shares that stale pool, ``_run_single_child`` leases a Codex credential
    and ``_swap_credential`` rewrites the MiniMax child back to the Codex base
    URL before its first request.
    """
    from tools import delegate_tool

    stale_parent_pool = SimpleNamespace(
        provider="openai-codex",
        has_credentials=lambda: True,
    )
    minimax_pool = SimpleNamespace(
        provider="minimax-oauth",
        has_credentials=lambda: True,
    )
    parent = SimpleNamespace(
        provider="minimax-oauth",
        _credential_pool=stale_parent_pool,
    )

    monkeypatch.setattr(
        "agent.credential_pool.load_pool",
        lambda provider: minimax_pool if provider == "minimax-oauth" else None,
    )

    resolved = delegate_tool._resolve_child_credential_pool("minimax-oauth", parent)

    assert resolved is minimax_pool
    assert resolved is not stale_parent_pool


def test_delegate_credentials_provider_resolution_fills_minimax_runtime(monkeypatch):
    """Blank delegation.base_url should not require manual endpoint config."""
    from tools import delegate_tool

    seen = {}

    def fake_resolve_runtime_provider(*, requested=None, target_model=None, **kwargs):
        seen["requested"] = requested
        seen["target_model"] = target_model
        return {
            "provider": "minimax-oauth",
            "model": target_model,
            "api_mode": "anthropic_messages",
            "base_url": MINIMAX_ANTHROPIC_URL,
            "api_key": "mm-oauth-token",
            "source": "oauth",
        }

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        fake_resolve_runtime_provider,
    )

    creds = delegate_tool._resolve_delegation_credentials(
        {
            "model": "MiniMax-M3",
            "provider": "minimax-oauth",
            "base_url": "",
            "api_key": "",
            "api_mode": "",
        },
        parent_agent=SimpleNamespace(),
    )

    assert seen == {"requested": "minimax-oauth", "target_model": "MiniMax-M3"}
    assert creds["provider"] == "minimax-oauth"
    assert creds["model"] == "MiniMax-M3"
    assert creds["base_url"] == MINIMAX_ANTHROPIC_URL
    assert creds["api_mode"] == "anthropic_messages"
    assert creds["api_key"] == "mm-oauth-token"
