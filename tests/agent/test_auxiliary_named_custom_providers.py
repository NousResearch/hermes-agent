"""Tests for named custom provider and 'main' alias resolution in auxiliary_client."""

import json
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Write a minimal config so load_config doesn't fail
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestNormalizeVisionProvider:
    """_normalize_vision_provider should resolve 'main' to actual main provider."""


    def test_main_resolves_to_openrouter(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "openrouter"








class TestResolveProviderClientMainAlias:
    """resolve_provider_client('main', ...) should resolve to actual main provider."""

    def test_main_resolves_to_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "override-model")
        assert client is not None
        assert model == "override-model"
        assert "beans.local" in str(client.base_url)

    def test_main_with_custom_colon_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "test")
        assert client is not None
        assert "beans.local" in str(client.base_url)

    def test_main_resolves_github_copilot_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "gpt-5.4", "provider": "github-copilot"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "ghu_test_token",
                "base_url": "https://api.githubcopilot.com",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.called


class TestResolveProviderClientNamedCustom:
    """resolve_provider_client should resolve named custom providers directly."""

    def test_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans", "my-model")
        assert client is not None
        assert model == "my-model"
        assert "beans.local" in str(client.base_url)


        # no-key-required should be used

    def test_providers_dict_uses_durable_pool_when_no_inline_key(self, tmp_path):
        """Titles/compression/vision must read credential_pool.<key>, not a placeholder."""
        _write_config(tmp_path, {
            "providers": {
                "b-ai": {
                    "name": "B.AI",
                    "base_url": "https://api.b.ai/v1",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "b-ai": [
                    {
                        "id": "k1",
                        "label": "primary",
                        "auth_type": "api_key",
                        "priority": 0,
                        "source": "manual",
                        "access_token": "sk-real-b-ai-pool-key-12345",
                    }
                ]
            },
        }))
        from agent.auxiliary_client import resolve_provider_client
        client, _model = resolve_provider_client("b-ai", "b-ai-model")
        assert client is not None
        assert "api.b.ai" in str(client.base_url)
        assert client.api_key == "sk-real-b-ai-pool-key-12345"


    def test_main_and_auxiliary_prefer_declared_env_key_over_pool(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RELAY_API_KEY", "declared-env-key")
        _write_config(tmp_path, {
            "model": {"provider": "custom:relay", "default": "relay-model"},
            "providers": {
                "relay": {
                    "base_url": "https://relay.example/v1",
                    "key_env": "RELAY_API_KEY",
                    "default_model": "relay-model",
                },
            },
        })

        entry = MagicMock(runtime_api_key="pool-key", access_token="")
        pool = MagicMock()
        pool.has_credentials.return_value = True
        pool.select.return_value = entry
        with patch("hermes_cli.runtime_provider.load_pool", return_value=pool):
            from hermes_cli.runtime_provider import resolve_runtime_provider
            from agent.auxiliary_client import resolve_provider_client

            main = resolve_runtime_provider(requested="custom:relay")
            auxiliary, _model = resolve_provider_client("custom:relay", "relay-model")

        assert main["api_key"] == "declared-env-key"
        assert auxiliary.api_key == "declared-env-key"

    def test_missing_declared_key_fails_closed_but_loopback_stays_keyless(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RELAY_API_KEY", raising=False)
        config = {
            "model": {"provider": "custom:relay", "default": "relay-model"},
            "providers": {
                "relay": {
                    "base_url": "https://relay.example/v1",
                    "key_env": "RELAY_API_KEY",
                    "default_model": "relay-model",
                },
            },
        }
        _write_config(tmp_path, config)

        from hermes_cli.auth import AuthError
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from agent.auxiliary_client import resolve_provider_client

        with pytest.raises(AuthError, match="Set RELAY_API_KEY"):
            resolve_runtime_provider(requested="custom:relay")
        with pytest.raises(AuthError, match="Set RELAY_API_KEY"):
            resolve_provider_client("custom:relay", "relay-model")

        config["providers"]["relay"]["base_url"] = "http://127.0.0.1:8080/v1"
        _write_config(tmp_path, config)
        main = resolve_runtime_provider(requested="custom:relay")
        auxiliary, _model = resolve_provider_client("custom:relay", "relay-model")

        assert main["api_key"] == "no-key-required"
        assert auxiliary.api_key == "no-key-required"


class TestResolveProviderClientModelNormalization:
    """Direct-provider auxiliary routing should normalize models like main runtime."""

    def test_matching_native_prefix_is_stripped_for_main_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "zai/glm-5.1")

        assert client is not None
        assert model == "glm-5.1"


    def test_aggregator_vendor_slug_is_preserved(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "openrouter", "anthropic/claude-sonnet-4.6"
            )

        assert client is not None
        assert model == "anthropic/claude-sonnet-4.6"


class TestResolveVisionProviderClientModelNormalization:
    """Vision auto-routing should reuse the same provider-specific normalization."""

    def test_vision_auto_strips_matching_main_provider_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None
        assert model == "glm-5.3-flash"  # zai coding endpoints support this vision-capable fallback


class TestAutoClientCacheModelCompatibility:
    """Auto client cache should not keep OpenRouter-format model overrides on non-OR clients."""

    def test_first_auto_cache_miss_drops_openrouter_model_for_named_custom_runtime(self, tmp_path):
        from agent import auxiliary_client as ac

        ac._client_cache.clear()
        try:
            fake_client = MagicMock()
            fake_client.base_url = "https://aixj.vip/v1"
            fake_client.api_key = "test-key"

            runtime = {
                "provider": "custom:aixj.vip",
                "model": "gpt-5.4",
                "base_url": "https://aixj.vip/v1",
                "api_key": "***",
                "api_mode": "codex_responses",
            }

            with patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(fake_client, "gpt-5.4"),
            ) as mock_resolve:
                client, model = ac._get_cached_client(
                    "auto",
                    "google/gemini-3-flash-preview",
                    main_runtime=runtime,
                )

            assert client is fake_client
            assert model == "gpt-5.4"
            mock_resolve.assert_called_once()
        finally:
            ac._client_cache.clear()




class TestProvidersDictApiModeAnthropicMessages:
    """Regression guard for #15033.

    Named providers declared under the ``providers:`` dict with
    ``api_mode: anthropic_messages`` must route auxiliary calls through
    the Anthropic Messages API (via AnthropicAuxiliaryClient), not
    through an OpenAI chat-completions client.

    The bug had two halves: the providers-dict branch of
    ``_get_named_custom_provider`` dropped the ``api_mode`` field, and
    ``resolve_provider_client``'s named-custom branch never read it.
    """




    def test_resolve_provider_client_returns_anthropic_client(self, tmp_path, monkeypatch):
        """Named custom provider with api_mode=anthropic_messages must
        route through AnthropicAuxiliaryClient, carrying the entry's extra_headers
        like the OpenAI-wire arms do (#109595)."""
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                    "extra_headers": {"X-Gateway-Token": "gw-1"},
                },
            },
        })
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
            AsyncAnthropicAuxiliaryClient,
        )
        sync_client, sync_model = resolve_provider_client("myrelay", async_mode=False)
        assert isinstance(sync_client, AnthropicAuxiliaryClient), (
            f"expected AnthropicAuxiliaryClient, got {type(sync_client).__name__}"
        )
        assert sync_model == "claude-opus-4-7"
        sdk_headers = sync_client._real_client._custom_headers
        assert sdk_headers.get("X-Gateway-Token") == "gw-1"
        assert "anthropic-beta" in sdk_headers, "entry headers must merge onto, not replace, the builder's headers"

        async_client, async_model = resolve_provider_client("myrelay", async_mode=True)
        assert isinstance(async_client, AsyncAnthropicAuxiliaryClient), (
            f"expected AsyncAnthropicAuxiliaryClient, got {type(async_client).__name__}"
        )
        assert async_model == "claude-opus-4-7"




class TestCustomProviderAliasCollision:
    """A user-declared custom_providers entry whose name matches a built-in
    *alias* (not a canonical provider) must win over the built-in.

    Regression guard for #15743: users who defined fallback_model pointing at
    a custom_providers entry named ``kimi`` were having requests routed to
    the built-in kimi-coding endpoint because ``_normalize_aux_provider``
    rewrote ``kimi`` → ``kimi-coding`` before the named-custom lookup.
    """

    def test_custom_named_kimi_wins_over_builtin_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
            "custom_providers": [
                {
                    "name": "kimi",
                    "base_url": "https://my-custom-kimi.example.com/v1",
                    "api_key": "my-kimi-key",
                    "models": {"my-kimi-model": {"context_length": 200000}},
                },
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, model = resolve_provider_client("kimi", model="my-kimi-model", raw_codex=True)
        assert isinstance(client, OpenAI)
        assert "my-custom-kimi.example.com" in str(client.base_url)
        assert client.api_key == "my-kimi-key"
        assert model == "my-kimi-model"

    def test_bare_kimi_without_custom_still_routes_to_builtin(self, tmp_path, monkeypatch):
        """Regression guard: bare 'kimi' with no custom entry must still
        reach the built-in kimi-coding provider."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        client, _ = resolve_provider_client("kimi", model="kimi-k2-0905-preview", raw_codex=True)
        assert client is not None
        base_url = str(client.base_url)
        # Built-in kimi-coding points at api.moonshot.ai
        assert "moonshot" in base_url or "kimi" in base_url, f"unexpected base_url {base_url!r}"

    @pytest.mark.parametrize("provider", ["llamacpp", "custom:llamacpp"])
    def test_named_llamacpp_wins_over_local_server_alias(self, tmp_path, provider):
        """A ``providers:`` entry whose name is also a local-server alias (``llamacpp``) resolves to
        its configured base_url, not to the alias's generic ``custom`` branch (#115990)."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
            "providers": {
                "llamacpp": {
                    "base_url": "http://127.0.0.1:8081/v1",
                    "model": "local-model",
                },
            },
        })
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI

        client, model = resolve_provider_client(provider, model="local-model", raw_codex=True)

        assert isinstance(client, OpenAI)
        assert str(client.base_url).rstrip("/") == "http://127.0.0.1:8081/v1"
        assert model == "local-model"

    def test_explicit_overrides_applied_on_api_key_branch(self, tmp_path, monkeypatch):
        """Explicit base_url/api_key from the caller must override the
        registered provider's defaults on the API-key branch.  Used by
        _try_activate_fallback to route a fallback through a built-in
        provider name but targeting a user-supplied endpoint."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, _ = resolve_provider_client(
            "kimi-coding", model="kimi-k2", raw_codex=True,
            explicit_base_url="https://override.example.com",
            explicit_api_key="override-key",
        )
        assert isinstance(client, OpenAI)
        assert "override.example.com" in str(client.base_url)
        assert client.api_key == "override-key"


class TestResolveProviderClientMainRuntimeCustom:
    """When the main agent uses a named custom provider (custom:<name>),
    resolve_provider_client('custom', ..., main_runtime=...) must reuse the
    main_runtime's base_url + api_key instead of re-resolving from the bare
    'custom' provider name.  Re-resolution loses the provider name and falls
    back to OpenRouter or a wrong API-key provider. (#45472)"""

    def test_custom_provider_main_runtime_used_directly(self, tmp_path, monkeypatch):
        """main_runtime with base_url + api_key for a named custom provider
        is used directly, bypassing the _try_custom_endpoint / API-key
        fallback chain."""
        from agent.auxiliary_client import resolve_provider_client
        main_runtime = {
            "provider": "custom",
            "base_url": "https://my-gateway.example.com/v1",
            "api_key": "***",
            "model": "glm-5.1",
        }
        client, model = resolve_provider_client(
            "custom",
            model="explicit-glm-5.1",
            main_runtime=main_runtime,
        )
        assert client is not None
        assert model == "explicit-glm-5.1"
        assert "my-gateway.example.com" in str(client.base_url)
        assert client.api_key == "***"

    def test_custom_provider_main_runtime_no_credentials_falls_through(self, tmp_path, monkeypatch):
        """When main_runtime has no base_url or no api_key, the existing
        _try_custom_endpoint / _resolve_api_key_provider fallback chain is
        still tried."""
        # Ensure no env-provided credentials interfere
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from agent.auxiliary_client import resolve_provider_client
        # main_runtime with key but no base_url → must fall through
        client, model = resolve_provider_client(
            "custom",
            main_runtime={"api_key": "k", "base_url": ""},
        )
        # Should fall through to _try_custom_endpoint → return None,None
        # because no OPENAI_BASE_URL is set and no custom endpoint is configured
        assert client is None

    def test_custom_provider_main_runtime_respects_explicit_base_url(self, tmp_path):
        """explicit_base_url still wins over main_runtime — the caller's
        explicit argument is the strongest signal."""
        from agent.auxiliary_client import resolve_provider_client
        main_runtime = {
            "base_url": "https://main-runtime.example.com/v1",
            "api_key": "sk-main",
            "model": "ignored-model",
        }
        client, model = resolve_provider_client(
            "custom",
            model="explicit-model",
            explicit_base_url="https://explicit.example.com/v1",
            explicit_api_key="sk-explicit",
            main_runtime=main_runtime,
        )
        assert client is not None
        assert model == "explicit-model"
        assert "explicit.example.com" in str(client.base_url)
        assert client.api_key == "sk-explicit"


class TestBareNamedAuxCredentialSurvivesAsyncRebuild:
    """#109595: a bare-named ``providers:`` entry pinned by ``auxiliary.<task>`` must reach the wire
    with its key_cmd bearer AND its extra_headers on the async path too. The SDK parks a callable
    key in ``_api_key_provider`` and leaves ``.api_key`` empty, so a rebuild from the snapshot
    alone sends no Authorization header at all."""

    def _cfg(self):
        import sys
        return {
            "model": {"provider": "hermes-gw", "default": "main-model"},
            "providers": {
                "hermes-gw": {
                    "base_url": "http://127.0.0.1:1/openai/v1",
                    "api_mode": "chat_completions",
                    "key_cmd": f'"{sys.executable}" -c "print(\'vk-test-1234\')"',
                    "extra_headers": {"x-gw-session": "aux-session-tag"},
                },
            },
            "auxiliary": {"compression": {"provider": "hermes-gw", "model": "aux-model"}},
        }

    @staticmethod
    def _wire_headers(client):
        """Headers the SDK would put on a request (after its own auth refresh), not the config view."""
        from openai._models import FinalRequestOptions
        return client._build_headers(FinalRequestOptions(method="post", url="/chat/completions"))

    def test_async_task_client_carries_key_cmd_bearer_and_extra_headers(self, tmp_path):
        import asyncio
        _write_config(tmp_path, self._cfg())
        from agent.auxiliary_client import resolve_provider_client
        client, _model = resolve_provider_client("hermes-gw", "aux-model", async_mode=True, task="compression")
        assert client is not None
        asyncio.run(client._refresh_api_key())
        headers = self._wire_headers(client)
        assert headers["authorization"] == "Bearer vk-test-1234"
        assert headers["x-gw-session"] == "aux-session-tag"

    def test_sync_to_async_rebuild_keeps_credential_and_headers(self, tmp_path):
        """The fallback-candidate ladder rebuilds a resolved sync client through ``_to_async_client``."""
        import asyncio
        _write_config(tmp_path, self._cfg())
        from agent.auxiliary_client import _to_async_client, resolve_provider_client
        sync_client, model = resolve_provider_client("hermes-gw", "aux-model", task="compression")
        sync_client._refresh_api_key()  # what the SDK does in _prepare_options before each request
        assert self._wire_headers(sync_client)["authorization"] == "Bearer vk-test-1234"
        async_client, _ = _to_async_client(sync_client, model)
        asyncio.run(async_client._refresh_api_key())
        headers = self._wire_headers(async_client)
        assert headers["authorization"] == "Bearer vk-test-1234"
        assert headers["x-gw-session"] == "aux-session-tag"


class TestKeyedCustomProviderReasoningWire:
    """Aux calls to a keyed ``providers:`` entry take the ``custom`` profile's reasoning wire (#75089).

    Referenced by bare key or via ``main``, a keyed OpenAI-compatible endpoint must get top-level
    ``reasoning_effort`` (what the main path sends), never the aggregator-only nested
    ``extra_body.reasoning`` that strict gateways reject with 400.
    """

    _KEYED = {
        "model": {"default": "vendor/model", "provider": "groq"},
        "providers": {"groq": {"name": "groq", "api": "https://api.groq.com/openai/v1", "api_key": "k"}},
    }

    @pytest.mark.parametrize("provider", ["groq", "main", "custom:groq"])
    def test_keyed_entry_sends_top_level_reasoning_effort(self, tmp_path, provider):
        """api.groq.com takes top-level reasoning_effort only as 'none'/'default' (#75089), so the
        configured 'medium' is clamped to 'default' — the bare-key case goes through ``call_llm``."""
        _write_config(tmp_path, self._KEYED)
        from agent.auxiliary_client import _build_call_kwargs, call_llm
        common = dict(reasoning_config={"enabled": True, "effort": "medium"}, base_url="https://api.groq.com/openai/v1")
        if provider == "groq":
            client = MagicMock(base_url=common["base_url"])
            with patch("agent.auxiliary_client._get_cached_client", return_value=(client, "vendor/model")), \
                    patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _t, **_kw: resp):
                call_llm(provider=provider, model="vendor/model", messages=[{"role": "user", "content": "hi"}], **common)
            kwargs = client.chat.completions.create.call_args.kwargs
        else:
            kwargs = _build_call_kwargs(provider, "vendor/model", [{"role": "user", "content": "hi"}], **common)
        assert kwargs.get("reasoning_effort") == "default"
        assert "reasoning" not in (kwargs.get("extra_body") or {})

    def test_profile_backed_and_unknown_providers_keep_their_wire(self, tmp_path):
        _write_config(tmp_path, self._KEYED)
        from agent.auxiliary_client import _build_call_kwargs
        nested = {"reasoning": {"enabled": True, "effort": "medium"}}
        # Aggregator profile: nested extra_body.reasoning is its wire; unchanged.
        kwargs = _build_call_kwargs(
            "openrouter", "vendor/model", [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"}, base_url="https://openrouter.ai/api/v1",
        )
        assert kwargs["extra_body"] == nested and "reasoning_effort" not in kwargs
        # No keyed entry, no base_url, no profile: generic fallback, never the custom projection.
        kwargs = _build_call_kwargs(
            "someunknown", "vendor/model", [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
        )
        assert kwargs["extra_body"] == nested and "reasoning_effort" not in kwargs
