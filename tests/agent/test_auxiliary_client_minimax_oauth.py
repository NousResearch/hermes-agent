"""Tests for auxiliary client routing of the ``minimax-oauth`` provider.

Covers the dedicated branch in ``agent.auxiliary_client.resolve_provider_client``
that delegates to ``_try_minimax_oauth`` instead of falling through the
generic ``PROVIDER_REGISTRY`` auth_type dispatch, which has no case for
``oauth_minimax`` and previously dead-ended in the unhandled-auth_type
fallback (#49232, #22213).

MiniMax OAuth access tokens expire in ~15 minutes. The fix mirrors the
callable-token-provider pattern already proven for the primary-provider
switch (``agent_init.py`` / ``agent_runtime_helpers.py``) and for Azure
Foundry's Entra ID path (see ``test_auxiliary_client_azure_foundry.py``):
``resolve_minimax_oauth_runtime_credentials(as_token_provider=True)``
returns a zero-arg callable instead of a static string, and
``build_anthropic_client`` detects the callable and installs a
bearer-injecting httpx event hook so every request mints a fresh token.

Pinned scenarios:

  * Logged in → callable ``api_key`` reaches the OpenAI SDK constructor
    intact, then gets rewrapped into an Anthropic client with the
    bearer-injecting httpx hook (mirrors the Entra ID assertions).
  * Not logged in → ``AuthError`` is caught and the branch returns
    ``(None, None)`` cleanly.
  * ``resolve_provider_client`` takes the dedicated branch, not the
    generic ``PROVIDER_REGISTRY`` path (which has no ``oauth_minimax``
    case and would silently dead-end).
  * Default model resolves from the provider's live profile
    (``_get_aux_model_for_provider``), not a hardcoded literal — a
    regression guard for the exact staleness that got both #49232 and
    #22213 flagged (their test snapshots hardcoded a model name that
    later drifted from the profile default).
  * Async mode wraps the result in ``AsyncAnthropicAuxiliaryClient`` with
    the callable still intact.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fake_minimax_creds(monkeypatch):
    """Stand-in for ``resolve_minimax_oauth_runtime_credentials`` — keeps
    tests hermetic (no real OAuth state / token refresh) while still
    proving the callable, not a string, is what gets returned and
    propagated when ``as_token_provider=True``.
    """
    import hermes_cli.auth as _auth

    calls = {"count": 0, "as_token_provider": None}

    def _token_provider():
        calls["count"] += 1
        return f"fresh-minimax-token-{calls['count']}"

    def _resolve(*, min_token_ttl_seconds=None, as_token_provider=False):
        calls["as_token_provider"] = as_token_provider
        return {
            "provider": "minimax-oauth",
            "api_key": _token_provider if as_token_provider else "static-minimax-token",
            "base_url": "https://api.minimax.io/anthropic",
            "source": "oauth",
        }

    monkeypatch.setattr(_auth, "resolve_minimax_oauth_runtime_credentials", _resolve)
    return calls


@pytest.fixture
def fake_minimax_not_logged_in(monkeypatch):
    """``resolve_minimax_oauth_runtime_credentials`` raising as it does
    when there's no stored MiniMax OAuth state."""
    import hermes_cli.auth as _auth

    def _resolve(*, min_token_ttl_seconds=None, as_token_provider=False):
        raise _auth.AuthError(
            "Not logged into MiniMax OAuth. Run `hermes model` and select "
            "MiniMax (OAuth).",
            provider="minimax-oauth", code="not_logged_in", relogin_required=True,
        )

    monkeypatch.setattr(_auth, "resolve_minimax_oauth_runtime_credentials", _resolve)


# ---------------------------------------------------------------------------
# _try_minimax_oauth
# ---------------------------------------------------------------------------


class TestTryMinimaxOauth:
    def test_callable_api_key_reaches_anthropic_client_with_bearer_hook(
        self, monkeypatch, fake_minimax_creds,
    ):
        from agent import auxiliary_client as _aux
        from agent import anthropic_adapter as _anthropic

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received["openai"] = kwargs
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        class _FakeAnthropicSDK:
            class Anthropic:
                def __init__(self, **kwargs):
                    received["anthropic"] = kwargs

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)

        client, resolved = _aux._try_minimax_oauth(model="MiniMax-M2.7")

        assert client is not None
        assert resolved == "MiniMax-M2.7"
        assert fake_minimax_creds["as_token_provider"] is True

        # The OpenAI SDK constructor saw the callable untouched, exactly
        # like the Entra ID path — never stringified before hand-off.
        openai_kwargs = received.get("openai") or {}
        assert callable(openai_kwargs.get("api_key"))
        assert not isinstance(openai_kwargs["api_key"], str)

        # _maybe_wrap_anthropic rewrapped it: build_anthropic_client saw
        # the callable, installed the bearer-injecting httpx hook, and
        # used the (provider-agnostic, despite the name) sentinel token.
        anthropic_kwargs = received.get("anthropic") or {}
        assert "http_client" in anthropic_kwargs, (
            "build_anthropic_client must pass a custom http_client when "
            "given a callable api_key, otherwise the SDK cannot mint "
            "fresh tokens per request and the session 401s once the "
            "~15-minute MiniMax OAuth token expires"
        )
        assert anthropic_kwargs.get("auth_token") == "entra-id-bearer-via-http-hook"
        http_client = anthropic_kwargs["http_client"]
        hooks = getattr(http_client, "event_hooks", {})
        assert "request" in hooks and len(hooks["request"]) >= 1

    def test_default_model_comes_from_live_provider_profile_not_hardcoded(
        self, monkeypatch, fake_minimax_creds,
    ):
        """Regression guard for the exact bug that staled out both prior
        PRs: #22213's test hardcoded ``MiniMax-M2.7-highspeed`` and drifted
        from the profile default. Assert against the live profile instead
        of a literal so this test can't go stale the same way."""
        from agent import auxiliary_client as _aux

        monkeypatch.setattr(_aux, "OpenAI", lambda **kw: object())

        expected_default = _aux._get_aux_model_for_provider("minimax-oauth")
        assert expected_default, "test fixture assumption: minimax-oauth must have a default_aux_model"

        client, resolved = _aux._try_minimax_oauth(model=None)
        assert resolved == expected_default

    def test_not_logged_in_returns_none_cleanly(self, fake_minimax_not_logged_in):
        from agent.auxiliary_client import _try_minimax_oauth

        client, resolved = _try_minimax_oauth(model="MiniMax-M2.7")
        assert client is None
        assert resolved is None

    def test_network_error_during_refresh_returns_none_cleanly(self, monkeypatch):
        """Token refresh does a live HTTP call whose transport failures
        (timeout, DNS, connection refused) aren't wrapped in AuthError.
        A network blip on a side-channel task must not crash the caller."""
        import hermes_cli.auth as _auth
        from agent.auxiliary_client import _try_minimax_oauth

        def _resolve(*, min_token_ttl_seconds=None, as_token_provider=False):
            raise ConnectionError("connection refused")

        monkeypatch.setattr(_auth, "resolve_minimax_oauth_runtime_credentials", _resolve)

        client, resolved = _try_minimax_oauth(model="MiniMax-M2.7")
        assert client is None
        assert resolved is None

    def test_token_provider_mints_fresh_token_per_call(
        self, monkeypatch, fake_minimax_creds,
    ):
        """The whole point of the fix: calling the exact callable that
        reached the OpenAI SDK constructor twice must mint two different
        tokens (proves it's a live per-request provider, not a string
        frozen at client-build time)."""
        from agent import auxiliary_client as _aux
        from agent import anthropic_adapter as _anthropic

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received["openai"] = kwargs
                self.api_key = kwargs.get("api_key", "")

        class _FakeAnthropicSDK:
            class Anthropic:
                def __init__(self, **kwargs):
                    received["anthropic"] = kwargs

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)

        _aux._try_minimax_oauth(model="MiniMax-M2.7")

        token_provider = received["openai"]["api_key"]
        assert callable(token_provider)
        first = token_provider()
        second = token_provider()
        assert first != second
        assert fake_minimax_creds["count"] == 2


# ---------------------------------------------------------------------------
# resolve_provider_client → minimax-oauth dispatch
# ---------------------------------------------------------------------------


class TestResolveProviderClientMinimaxOauth:
    def test_dispatches_to_minimax_oauth_branch_not_generic_registry_path(
        self, monkeypatch, fake_minimax_creds,
    ):
        """End-to-end: ``resolve_provider_client`` must take the dedicated
        branch, not fall through PROVIDER_REGISTRY's auth_type dispatch —
        which has no ``oauth_minimax`` case and previously dead-ended in
        the unhandled-auth_type fallback, returning (None, None) even
        when the user was logged in."""
        from agent import auxiliary_client as _aux
        from agent import anthropic_adapter as _anthropic

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received["openai"] = kwargs
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        class _FakeAnthropicSDK:
            class Anthropic:
                def __init__(self, **kwargs):
                    received["anthropic"] = kwargs

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)

        client, resolved = _aux.resolve_provider_client("minimax-oauth", "MiniMax-M2.7")

        assert client is not None
        assert resolved == "MiniMax-M2.7"
        assert callable(received["openai"]["api_key"])

    def test_warns_and_returns_none_when_not_logged_in(
        self, fake_minimax_not_logged_in, caplog,
    ):
        import logging
        from agent.auxiliary_client import resolve_provider_client

        with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            client, resolved = resolve_provider_client("minimax-oauth")

        assert client is None
        assert resolved is None
        assert any(
            "minimax-oauth" in rec.message and "hermes model" in rec.message
            for rec in caplog.records
        )

    def test_async_mode_wraps_client_with_callable_still_intact(
        self, monkeypatch, fake_minimax_creds,
    ):
        """The async auxiliary path (used by e.g. background title
        generation) must go through the same callable-token branch, not
        a separate code path that regresses back to a static string."""
        from agent import auxiliary_client as _aux
        from agent import anthropic_adapter as _anthropic

        received = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                received["openai"] = kwargs
                self.api_key = kwargs.get("api_key", "")
                self.base_url = kwargs.get("base_url", "")

        class _FakeAnthropicSDK:
            class Anthropic:
                def __init__(self, **kwargs):
                    received["anthropic"] = kwargs

        monkeypatch.setattr(_aux, "OpenAI", _FakeOpenAI)
        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)

        client, resolved = _aux.resolve_provider_client(
            "minimax-oauth", "MiniMax-M2.7", async_mode=True,
        )

        assert resolved == "MiniMax-M2.7"
        assert isinstance(client, _aux.AsyncAnthropicAuxiliaryClient)
        assert callable(received["openai"]["api_key"])
        assert "http_client" in (received.get("anthropic") or {})
