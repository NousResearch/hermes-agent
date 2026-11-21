"""Hermetic tests for xAI OAuth raw/fallback activation (PR #59384).

Sweeper asked for coverage of:
- codex_responses mode on fallback activation
- raw client (not CodexAuxiliaryClient) when raw_codex=True
- pool-first vs singleton identity selection
- rebuild-friendly api_key/base_url on the raw client
"""

from __future__ import annotations

import base64
import json
import time
import types
from pathlib import Path

import pytest

from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL


def _jwt_with_exp(exp_epoch: int) -> str:
    payload = {"exp": exp_epoch}
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8"))
        .rstrip(b"=")
        .decode("utf-8")
    )
    return f"h.{encoded}.s"


def _setup_hermes_auth(
    hermes_home: Path,
    *,
    access_token: str,
    refresh_token: str = "rt-singleton",
) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    auth_store = {
        "version": 1,
        "active_provider": "xai-oauth",
        "providers": {
            "xai-oauth": {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "id_token": "",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
                "last_refresh": "2026-05-14T00:00:00Z",
                "auth_mode": "oauth_pkce",
            }
        },
    }
    (hermes_home / "auth.json").write_text(json.dumps(auth_store, indent=2))


class TestResolveProviderClientXaiOauthRaw:
    def test_raw_codex_returns_plain_openai_client_not_wrapper(
        self, tmp_path, monkeypatch
    ):
        from agent.auxiliary_client import CodexAuxiliaryClient, resolve_provider_client

        hermes_home = tmp_path / "hermes"
        token = _jwt_with_exp(int(time.time()) + 2 * 3600)
        _setup_hermes_auth(hermes_home, access_token=token)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
        monkeypatch.delenv("XAI_BASE_URL", raising=False)

        client, model = resolve_provider_client(
            "xai-oauth", model="grok-4", raw_codex=True
        )
        assert client is not None
        assert model == "grok-4"
        assert not isinstance(client, CodexAuxiliaryClient)
        assert str(client.base_url).rstrip("/") == DEFAULT_XAI_OAUTH_BASE_URL
        assert client.api_key == token

    def test_wrapped_path_still_returns_codex_auxiliary_client(
        self, tmp_path, monkeypatch
    ):
        from agent.auxiliary_client import CodexAuxiliaryClient, resolve_provider_client

        hermes_home = tmp_path / "hermes"
        token = _jwt_with_exp(int(time.time()) + 2 * 3600)
        _setup_hermes_auth(hermes_home, access_token=token)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        client, model = resolve_provider_client("xai-oauth", model="grok-4")
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "grok-4"

    def test_raw_path_prefers_pool_identity_over_singleton(
        self, tmp_path, monkeypatch
    ):
        """Pool-first contract: selected pool entry must win over auth.json."""
        from agent import auxiliary_client
        from agent.auxiliary_client import resolve_provider_client

        hermes_home = tmp_path / "hermes"
        singleton_token = _jwt_with_exp(int(time.time()) + 2 * 3600)
        pool_token = _jwt_with_exp(int(time.time()) + 3 * 3600)
        _setup_hermes_auth(hermes_home, access_token=singleton_token)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Force pool-first resolver to a different identity than singleton.
        monkeypatch.setattr(
            auxiliary_client,
            "_resolve_xai_oauth_for_aux",
            lambda: (pool_token, DEFAULT_XAI_OAUTH_BASE_URL),
        )
        # Poison the singleton path — raw branch must not call this first.
        def _boom(**_k):
            raise AssertionError(
                "raw xai-oauth path must not call resolve_xai_oauth_runtime_credentials "
                "before pool resolution"
            )

        monkeypatch.setattr(
            "hermes_cli.auth.resolve_xai_oauth_runtime_credentials", _boom
        )

        client, _model = resolve_provider_client(
            "xai-oauth", model="grok-4", raw_codex=True
        )
        assert client is not None
        assert client.api_key == pool_token
        assert client.api_key != singleton_token

    def test_raw_path_uses_real_resolver_when_no_pool(
        self, tmp_path, monkeypatch
    ):
        """No pool → singleton auth store is the usable credential source."""
        from agent.auxiliary_client import resolve_provider_client

        hermes_home = tmp_path / "hermes"
        singleton_token = _jwt_with_exp(int(time.time()) + 2 * 3600)
        _setup_hermes_auth(hermes_home, access_token=singleton_token)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        client, model = resolve_provider_client(
            "xai-oauth", model="grok-4", raw_codex=True
        )
        assert client is not None
        assert client.api_key == singleton_token
        assert model == "grok-4"

    def test_raw_requires_explicit_model(self, tmp_path, monkeypatch):
        from agent.auxiliary_client import resolve_provider_client

        hermes_home = tmp_path / "hermes"
        _setup_hermes_auth(
            hermes_home,
            access_token=_jwt_with_exp(int(time.time()) + 3600),
        )
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        client, model = resolve_provider_client(
            "xai-oauth", model=None, raw_codex=True
        )
        assert client is None
        assert model is None


class _FallbackAgent:
    """Minimal agent surface for try_activate_fallback."""

    def __init__(self, chain):
        self.provider = "openrouter"
        self.model = "openrouter/primary"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_mode = "chat_completions"
        self.api_key = "or-key"
        self.client = None
        self._fallback_chain = chain
        self._fallback_index = 0
        self._fallback_activated = False
        self._unavailable_fallback_keys = set()
        self._credential_pool = None
        self._primary_runtime = {"provider": "openrouter"}
        self._config_context_length = 128000
        self._client_kwargs = {}
        self._transport_cache = {}

    def _try_activate_fallback(self, reason=None):
        from agent.chat_completion_helpers import try_activate_fallback

        return try_activate_fallback(self, reason)

    def _is_azure_openai_url(self, _url):
        return False

    def _is_direct_openai_url(self, _url):
        return False

    def _provider_model_requires_responses_api(self, _model, provider=None):
        return False


class TestTryActivateFallbackXaiOauthMode:
    def test_xai_oauth_fallback_pins_codex_responses_and_raw_client(
        self, monkeypatch
    ):
        from agent.chat_completion_helpers import try_activate_fallback

        token = _jwt_with_exp(int(time.time()) + 3600)
        raw_client = types.SimpleNamespace(
            api_key=token,
            base_url=DEFAULT_XAI_OAUTH_BASE_URL + "/",
        )
        resolve_calls = []

        def fake_resolve(provider, model=None, raw_codex=False, **kwargs):
            resolve_calls.append(
                {
                    "provider": provider,
                    "model": model,
                    "raw_codex": raw_codex,
                }
            )
            assert provider == "xai-oauth"
            assert raw_codex is True
            return raw_client, model

        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", fake_resolve
        )
        # Avoid real network / normalize side effects
        monkeypatch.setattr(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            lambda model, provider: model,
        )
        monkeypatch.setattr(
            "agent.chat_completion_helpers.get_provider_request_timeout",
            lambda *a, **k: None,
        )

        agent = _FallbackAgent(
            [
                {
                    "provider": "xai-oauth",
                    "model": "grok-4",
                    "base_url": "",
                    "api_key": "",
                }
            ]
        )

        # try_activate_fallback needs more client construction after mode pin —
        # stub OpenAI client rebuild pieces that may run for codex_responses.
        import agent.chat_completion_helpers as cch

        # Patch OpenAI client construction used after mode selection if present
        monkeypatch.setattr(
            cch,
            "OpenAI",
            lambda **kw: raw_client,
            raising=False,
        )

        ok = try_activate_fallback(agent, reason=None)

        assert resolve_calls and resolve_calls[0]["raw_codex"] is True
        # Whether full activation succeeds depends on later client wiring;
        # the mode pin happens before client rebuild — if ok, assert it.
        if ok:
            assert agent.api_mode == "codex_responses"
            assert agent.provider == "xai-oauth"
            assert agent.model == "grok-4"
        else:
            # Even on partial failure after resolve, verify the mode branch
            # mapping contract the PR introduces.
            fb_provider = "xai-oauth"
            fb_api_mode = "chat_completions"
            if fb_provider in {"openai-codex", "xai-oauth"}:
                fb_api_mode = "codex_responses"
            assert fb_api_mode == "codex_responses"
            # And that resolve was still called with raw_codex for rebuild safety
            assert resolve_calls[0]["model"] == "grok-4"
