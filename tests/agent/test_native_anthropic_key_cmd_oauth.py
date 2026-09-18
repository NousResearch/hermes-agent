"""key_cmd Claude Code OAuth on native custom-provider routes keeps its identity (#114967).

A named custom provider that targets api.anthropic.com and mints a Claude Code OAuth
token through ``key_cmd`` used to lose its OAuth identity everywhere:

  * main-runtime flags assigned ``is_oauth=False`` because detection was
    ``isinstance(effective_key, str)`` and a ``CommandTokenSource`` is callable;
  * the wire client routed callables into the Entra bearer hook, which sends a
    bare Bearer without the Claude Code fingerprint Anthropic's OAuth routing
    requires (misleading HTTP 429 ``rate_limit_error: Error``);
  * auxiliary custom routes hardcoded ``AnthropicAuxiliaryClient(..., is_oauth=False)``;
  * ``_resolve_custom_runtime`` stomped the callable into "no-key-required".

The invariant under test: OAuth identity survives exactly when the route host is
api.anthropic.com (or the native default) AND the materialized token has the
OAuth shape. Third-party Anthropic-protocol endpoints never qualify.

All token strings below are dummy fixtures assembled from inert fragments; no
real credential appears in this file.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


_NATIVE = "https://api.anthropic.com"
# Dummy fixture values: prefix fragments joined so the shape tests stay meaningful.
_OAUTH_TOKEN = "sk-ant-" + "oat01-dummy-fixture-oauth-token"
_CONSOLE_KEY = "sk-ant-" + "api03-dummy-fixture-console-key"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_providers(tmp_path, providers: dict):
    import yaml

    (tmp_path / ".hermes" / "config.yaml").write_text(
        yaml.dump({
            "model": {"default": "m1", "provider": next(iter(providers))},
            "providers": providers,
        })
    )


# ---------------------------------------------------------------------------
# is_native_anthropic_oauth — the shared host + token-shape gate
# ---------------------------------------------------------------------------


class TestIsNativeAnthropicOauth:
    def test_static_oauth_token_on_native_host(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(_OAUTH_TOKEN, _NATIVE) is True

    def test_empty_base_url_is_the_native_default(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(_OAUTH_TOKEN, "") is True
        assert is_native_anthropic_oauth(_OAUTH_TOKEN, None) is True

    def test_console_api_key_is_never_oauth(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(_CONSOLE_KEY, _NATIVE) is False

    def test_callable_source_is_classified_by_shape(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(lambda: _OAUTH_TOKEN, _NATIVE) is True

    def test_mint_failure_classifies_non_oauth(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        def broken():
            raise RuntimeError("key_cmd exited 1")

        assert is_native_anthropic_oauth(broken, _NATIVE) is False

    def test_non_string_non_callable_key_is_not_oauth(self):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(None, _NATIVE) is False
        assert is_native_anthropic_oauth(1234, _NATIVE) is False

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.minimaxi.com/anthropic",
            "https://open.bigmodel.cn/api/anthropic",
            "https://litellm.internal.example.com/v1",
            "https://api.anthropic.com.evil.example.com",  # look-alike host must not pass
            "https://r.services.ai.azure.com/anthropic",
        ],
    )
    def test_third_party_hosts_never_oauth(self, base_url):
        from agent.anthropic_credentials import is_native_anthropic_oauth

        assert is_native_anthropic_oauth(_OAUTH_TOKEN, base_url) is False
        assert is_native_anthropic_oauth(lambda: _OAUTH_TOKEN, base_url) is False


# ---------------------------------------------------------------------------
# bearer-hook client — Claude Code identity headers on the native host
# ---------------------------------------------------------------------------


class TestBearerHookOAuthIdentityHeaders:
    def _build(self, monkeypatch, token_provider, base_url):
        from agent import anthropic_adapter as _anthropic
        from agent import azure_identity_adapter as _azure

        received = {}

        class _FakeAnthropicSDK:
            Omit = object

            class Anthropic:
                def __init__(self, **kwargs):
                    received["kwargs"] = kwargs

        monkeypatch.setattr(_anthropic, "_get_anthropic_sdk", lambda: _FakeAnthropicSDK)
        monkeypatch.setattr(
            _azure, "build_bearer_http_client", lambda provider, timeout: MagicMock()
        )
        client = _anthropic.build_anthropic_client(token_provider, base_url)
        return client, received["kwargs"]

    def test_oauth_callable_on_native_host_gets_claude_code_identity(self, monkeypatch):
        _, kwargs = self._build(monkeypatch, lambda: _OAUTH_TOKEN, _NATIVE)
        headers = kwargs.get("default_headers") or {}
        assert str(headers.get("user-agent", "")).startswith("claude-code/")
        assert headers.get("x-app") == "cli"
        beta = str(headers.get("anthropic-beta", ""))
        assert "oauth-2025-04-20" in beta, f"OAuth-only betas missing from {beta}"

    def test_non_oauth_callable_on_native_host_keeps_plain_headers(self, monkeypatch):
        _, kwargs = self._build(monkeypatch, lambda: _CONSOLE_KEY, _NATIVE)
        headers = kwargs.get("default_headers") or {}
        assert not str(headers.get("user-agent", "")).startswith("claude-code/")
        assert "oauth-2025-04-20" not in str(headers.get("anthropic-beta", ""))

    def test_third_party_host_callable_keeps_plain_headers(self, monkeypatch):
        _, kwargs = self._build(
            monkeypatch,
            lambda: _OAUTH_TOKEN,
            "https://r.services.ai.azure.com/anthropic",
        )
        headers = kwargs.get("default_headers") or {}
        assert "oauth-2025-04-20" not in str(headers.get("anthropic-beta", ""))

    def test_oauth_only_betas_disjoint_from_common_betas(self):
        # The OAuth branches concatenate the common betas with _OAUTH_ONLY_BETAS;
        # a beta present in both sets would be sent twice on the anthropic-beta line.
        from agent.anthropic_adapter import _OAUTH_ONLY_BETAS, _common_betas_for_base_url

        for base_url in (
            _NATIVE,
            None,
            "https://api.minimaxi.com/anthropic",
            "https://r.services.ai.azure.com/anthropic",
        ):
            for drop_1m in (False, True):
                common = set(
                    _common_betas_for_base_url(base_url, drop_context_1m_beta=drop_1m)
                )
                assert common.isdisjoint(_OAUTH_ONLY_BETAS), (
                    f"OAuth-only betas duplicated in the common set for {base_url!r}"
                )


# ---------------------------------------------------------------------------
# auxiliary named custom provider — api_mode=anthropic_messages + native host
# ---------------------------------------------------------------------------


class TestNamedCustomAuxiliaryOAuthFlag:
    @pytest.fixture(autouse=True)
    def _fake_anthropic_builder(self, monkeypatch):
        """The OAuth flag under test is set before any SDK client exists; a stub
        builder keeps these flag tests runnable without the anthropic package
        (the real-client path is covered by test_auxiliary_named_custom_providers)."""
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            lambda api_key, base_url, **kw: MagicMock(),
        )

    def test_static_oauth_key_on_native_host_is_oauth(self, tmp_path):
        _write_providers(
            tmp_path,
            {
                "enterprise-anthropic": {
                    "name": "enterprise-anthropic",
                    "base_url": _NATIVE,
                    "api_mode": "anthropic_messages",
                    "api_key": _OAUTH_TOKEN,
                    "default_model": "claude-opus-5",
                },
            },
        )
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
        )

        client, model = resolve_provider_client("enterprise-anthropic")
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert client.chat.completions._is_oauth is True

    def test_key_cmd_oauth_token_on_native_host_is_oauth(self, tmp_path):
        # The mint runs a real `printf` through the shell — hermetic, no network.
        _write_providers(
            tmp_path,
            {
                "enterprise-anthropic": {
                    "name": "enterprise-anthropic",
                    "base_url": _NATIVE,
                    "api_mode": "anthropic_messages",
                    "key_cmd": f"printf {_OAUTH_TOKEN}",
                    "default_model": "claude-opus-5",
                },
            },
        )
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
        )

        client, model = resolve_provider_client("enterprise-anthropic")
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert client.chat.completions._is_oauth is True

    def test_third_party_relay_with_oauth_shaped_key_stays_non_oauth(self, tmp_path):
        _write_providers(
            tmp_path,
            {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "api_mode": "anthropic_messages",
                    "api_key": _OAUTH_TOKEN,
                    "default_model": "claude-opus-4-7",
                },
            },
        )
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
        )

        client, _ = resolve_provider_client("myrelay")
        assert isinstance(client, AnthropicAuxiliaryClient)
        assert client.chat.completions._is_oauth is False


# ---------------------------------------------------------------------------
# auxiliary anonymous custom runtime reuse
# ---------------------------------------------------------------------------


class TestAnonymousCustomRuntimeOAuth:
    def _try_custom(self, monkeypatch, base, key):
        from agent import auxiliary_client as _aux

        monkeypatch.setattr(
            _aux, "_resolve_custom_runtime", lambda: (base, key, "anthropic_messages")
        )
        monkeypatch.setattr(
            "agent.anthropic_adapter.build_anthropic_client",
            lambda api_key, base_url, **kw: MagicMock(),
        )
        return _aux._try_custom_endpoint()

    def test_callable_key_cmd_source_survives_and_keeps_oauth(self, monkeypatch):
        client, _ = self._try_custom(monkeypatch, _NATIVE, lambda: _OAUTH_TOKEN)
        assert client is not None
        assert client.chat.completions._is_oauth is True

    def test_static_oauth_key_on_native_host_is_oauth(self, monkeypatch):
        client, _ = self._try_custom(monkeypatch, _NATIVE, _OAUTH_TOKEN)
        assert client is not None
        assert client.chat.completions._is_oauth is True

    def test_third_party_custom_host_stays_non_oauth(self, monkeypatch):
        client, _ = self._try_custom(
            monkeypatch, "https://relay.internal.example.com", lambda: _OAUTH_TOKEN
        )
        assert client is not None
        assert client.chat.completions._is_oauth is False

    def test_resolve_custom_runtime_no_longer_stomps_callables(self, monkeypatch):
        from agent import auxiliary_client as _aux

        provider = lambda: _OAUTH_TOKEN  # noqa: E731
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda requested: {
                "base_url": _NATIVE,
                "api_key": provider,
                "api_mode": "anthropic_messages",
            },
        )
        base, key, mode = _aux._resolve_custom_runtime()
        assert base == _NATIVE
        assert key is provider, (
            "callable key_cmd source must reach the wire client untouched"
        )
        assert mode == "anthropic_messages"


# ---------------------------------------------------------------------------
# main-runtime OAuth flag
# ---------------------------------------------------------------------------


class TestMainRuntimeOAuthFlag:
    def _flag(self, provider: str, base_url, token) -> bool:
        from run_agent import AIAgent

        ns = SimpleNamespace(provider=provider, _anthropic_base_url=base_url)
        return AIAgent._anthropic_oauth_flag(ns, token)

    def test_named_custom_native_host_static_oauth(self):
        assert self._flag("enterprise-anthropic", _NATIVE, _OAUTH_TOKEN) is True

    def test_named_custom_native_host_key_cmd_source(self):
        assert self._flag("enterprise-anthropic", _NATIVE, lambda: _OAUTH_TOKEN) is True

    def test_native_provider_default_base(self):
        assert self._flag("anthropic", None, _OAUTH_TOKEN) is True

    def test_native_provider_overridden_third_party_base(self):
        assert (
            self._flag("anthropic", "https://relay.internal.example.com", _OAUTH_TOKEN)
            is False
        )

    def test_third_party_provider_with_oauth_shaped_key(self):
        assert (
            self._flag("minimax", "https://api.minimaxi.com/anthropic", _OAUTH_TOKEN)
            is False
        )
