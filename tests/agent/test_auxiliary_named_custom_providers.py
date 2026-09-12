"""Tests for named custom provider and 'main' alias resolution in auxiliary_client."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


@pytest.fixture
def completion_server():
    """Local OpenAI-compatible endpoint recording the request destination and auth."""
    received = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            received["path"] = self.path
            received["authorization"] = self.headers.get("Authorization")
            received["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": "relay-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", received
    finally:
        server.shutdown()
        server.server_close()


class TestNormalizeVisionProvider:
    """_normalize_vision_provider should resolve 'main' to actual main provider."""


    def test_main_resolves_to_openrouter(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "openrouter"






    def test_auto_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("auto") == "auto"
        assert _normalize_vision_provider(None) == "auto"


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


    def test_named_custom_no_api_key_uses_fallback(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "local", "base_url": "http://localhost:8080/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("local", "test")
        assert client is not None
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
        assert model == "glm-5v-turbo"  # zai has dedicated vision model in _PROVIDER_VISION_MODELS


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


class TestVisionPathApiMode:
    """Vision path should propagate api_mode to _get_cached_client."""

    def test_explicit_provider_passes_api_mode(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "auxiliary": {"vision": {"api_mode": "chat_completions"}},
        })
        with patch("agent.auxiliary_client._get_cached_client") as mock_gcc:
            mock_gcc.return_value = (MagicMock(), "test-model")
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client(provider="deepseek")

        mock_gcc.assert_called_once()
        _, kwargs = mock_gcc.call_args
        assert kwargs.get("api_mode") == "chat_completions"


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

    def test_providers_dict_propagates_api_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("myrelay")
        assert entry is not None
        assert entry.get("api_mode") == "anthropic_messages"
        assert entry.get("base_url") == "https://example-relay.test/anthropic"
        assert entry.get("api_key") == "sk-test"



    def test_resolve_provider_client_returns_anthropic_client(self, tmp_path, monkeypatch):
        """Named custom provider with api_mode=anthropic_messages must
        route through AnthropicAuxiliaryClient."""
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
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

        async_client, async_model = resolve_provider_client("myrelay", async_mode=True)
        assert isinstance(async_client, AsyncAnthropicAuxiliaryClient), (
            f"expected AsyncAnthropicAuxiliaryClient, got {type(async_client).__name__}"
        )
        assert async_model == "claude-opus-4-7"




class TestCustomProviderAliasCollision:
    """Explicit custom identities win even when their name collides with a built-in."""

    def test_explicit_custom_openrouter_call_uses_custom_endpoint_and_pool(
        self, tmp_path, monkeypatch, completion_server,
    ):
        """The production call path must keep the custom endpoint and its key
        even when a different credential exists in the built-in OpenRouter pool."""
        base_url, received = completion_server
        _write_config(tmp_path, {
            "model": {"provider": "custom:openrouter", "default": "relay-model"},
            "providers": {
                "openrouter": {
                    "name": "openrouter",
                    "base_url": base_url,
                    "default_model": "relay-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter": [{
                    "id": "public", "label": "public", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "public-openrouter-key",
                }],
                "custom:openrouter": [{
                    "id": "private", "label": "private", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "private-router-key",
                }],
            },
        }))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from agent import auxiliary_client
        auxiliary_client.shutdown_cached_clients()
        try:
            response = auxiliary_client.call_llm(
                provider="custom:openrouter",
                model="relay-model",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=20,
                timeout=10,
            )
        finally:
            auxiliary_client.shutdown_cached_clients()

        assert response.choices[0].message.content == "ok"
        assert received["path"] == "/v1/chat/completions"
        assert received["authorization"] == "Bearer private-router-key"
        assert received["body"]["model"] == "relay-model"

    def test_explicit_custom_openrouter_uses_durable_provider_pool(
        self, tmp_path,
    ):
        """A keyed providers entry keeps its durable pool slug under explicit custom routing."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "openrouter",
                    "base_url": "https://relay.example.test/v1",
                    "default_model": "relay-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:openrouter": [{
                    "id": "relay", "label": "relay", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "durable-relay-key",
                }],
                "openrouter": [{
                    "id": "public", "label": "public", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "public-openrouter-key",
                }],
            },
        }))

        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "custom:openrouter", model="relay-model", raw_codex=True,
        )

        assert client is not None
        assert client.api_key == "durable-relay-key"
        assert "relay.example.test" in str(client.base_url)
        assert model == "relay-model"

    def test_custom_display_alias_of_builtin_collision_uses_canonical_custom_pool(
        self, tmp_path,
    ):
        """Every alias of providers.openrouter resolves to custom:openrouter credentials."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "Private Relay",
                    "base_url": "https://relay.example.test/v1",
                    "default_model": "relay-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter": [{
                    "id": "public", "label": "public", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "public-openrouter-key",
                }],
                "custom:openrouter": [{
                    "id": "private", "label": "private", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "private-router-key",
                }],
            },
        }))

        from agent.auxiliary_client import resolve_provider_client, resolve_vision_provider_client

        client, model = resolve_provider_client(
            "private-relay", model="relay-model", raw_codex=True,
        )
        explicit_client, explicit_model = resolve_provider_client(
            "custom:private-relay", model="relay-model", raw_codex=True,
        )
        vision_provider, vision_client, vision_model = resolve_vision_provider_client(
            provider="custom:private-relay", model="relay-model",
        )

        assert client is not None and client.api_key == "private-router-key"
        assert "relay.example.test" in str(client.base_url)
        assert model == "relay-model"
        assert explicit_client is not None and explicit_client.api_key == "private-router-key"
        assert "relay.example.test" in str(explicit_client.base_url)
        assert explicit_model == "relay-model"
        assert vision_provider == "custom:private-relay"
        assert vision_client is not None and vision_client.api_key == "private-router-key"
        assert "relay.example.test" in str(vision_client.base_url)
        assert vision_model == "relay-model"

    def test_colliding_custom_provider_rejects_ambiguous_durable_builtin_pool(
        self, tmp_path,
    ):
        """credential_pool.openrouter alone is built-in auth, never custom relay auth."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "openrouter",
                    "base_url": "https://relay.example.test/v1",
                    "default_model": "relay-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter": [{
                    "id": "ambiguous", "label": "ambiguous", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "built-in-or-ambiguous-key",
                }],
            },
        }))

        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "custom:openrouter", model="relay-model", raw_codex=True,
        )

        assert client is not None
        assert client.api_key == "no-key-required"
        assert client.api_key != "built-in-or-ambiguous-key"
        assert "relay.example.test" in str(client.base_url)
        assert model == "relay-model"

    def test_explicit_custom_openrouter_vision_uses_custom_endpoint_and_pool(
        self, tmp_path,
    ):
        """Vision preserves the explicit custom namespace instead of entering built-in routing."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "openrouter",
                    "base_url": "https://vision-relay.example.test/v1",
                    "default_model": "vision-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter": [{
                    "id": "public", "label": "public", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "public-openrouter-key",
                }],
                "custom:openrouter": [{
                    "id": "private", "label": "private", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "private-router-key",
                }],
            },
        }))

        from agent.auxiliary_client import resolve_vision_provider_client

        provider, client, model = resolve_vision_provider_client(
            provider="custom:openrouter", model="vision-model",
        )

        assert provider == "custom:openrouter"
        assert client is not None
        assert client.api_key == "private-router-key"
        assert "vision-relay.example.test" in str(client.base_url)
        assert model == "vision-model"

    def test_missing_explicit_custom_vision_does_not_fall_back(self, tmp_path, monkeypatch):
        """A missing custom vision identity cannot send the image to an unrelated provider."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "openai/gpt-4o-mini"},
        })
        from agent import auxiliary_client

        auto = MagicMock(side_effect=AssertionError("auto fallback must not run"))
        monkeypatch.setattr(auxiliary_client, "_vision_auto_route", auto)

        with pytest.raises(RuntimeError, match="refusing vision fallback"):
            auxiliary_client.call_llm(
                task="vision", provider="custom:missing", model="vision-model",
                messages=[{"role": "user", "content": "inspect"}],
                max_tokens=20, timeout=1,
            )

        auto.assert_not_called()

    def test_missing_explicit_custom_text_does_not_use_fallback_chain(self, tmp_path, monkeypatch):
        """A missing custom text identity cannot send prompt content to another provider."""
        _write_config(tmp_path, {
            "auxiliary": {
                "compression": {
                    "fallback_chain": [{"provider": "openrouter", "model": "fallback"}],
                },
            },
        })
        from agent import auxiliary_client

        fallback = MagicMock(side_effect=AssertionError("fallback chain must not run"))
        monkeypatch.setattr(auxiliary_client, "_try_configured_fallback_for_unavailable_client", fallback)

        with pytest.raises(RuntimeError, match="refusing text fallback"):
            auxiliary_client.call_llm(
                task="compression", provider="custom:missing", model="relay-model",
                messages=[{"role": "user", "content": "private context"}],
                max_tokens=20, timeout=1,
            )

        fallback.assert_not_called()

    def test_disabled_bare_custom_vision_alias_does_not_fall_back(self, tmp_path, monkeypatch):
        """A disabled custom display alias remains explicit intent and fails closed."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "openai/gpt-4o-mini"},
            "providers": {
                "openrouter": {
                    "name": "Private Relay",
                    "base_url": "https://relay.example.test/v1",
                    "enabled": False,
                },
            },
        })
        from agent import auxiliary_client

        auto = MagicMock(side_effect=AssertionError("auto fallback must not run"))
        monkeypatch.setattr(auxiliary_client, "_vision_auto_route", auto)

        with pytest.raises(RuntimeError, match="refusing vision fallback"):
            auxiliary_client.call_llm(
                task="vision", provider="private-relay", model="vision-model",
                messages=[{"role": "user", "content": "inspect"}],
                max_tokens=20, timeout=1,
            )

        auto.assert_not_called()

    def test_disabled_custom_alias_does_not_route_to_builtin(self, tmp_path):
        """Auxiliary raw alias intent fails closed before kimi alias normalization."""
        _write_config(tmp_path, {
            "providers": {
                "private-kimi": {
                    "name": "kimi",
                    "base_url": "https://relay.example.test/v1",
                    "enabled": False,
                },
            },
        })
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client("kimi", model="kimi-model")

        assert client is None
        assert model is None

    def test_custom_builtin_collision_does_not_use_builtin_auth_refresh(self, monkeypatch):
        """A custom:copilot/custom:anthropic 401 cannot activate built-in OAuth refresh."""
        from agent import auxiliary_client

        assert auxiliary_client._auth_refresh_provider_for_route(
            "custom:copilot", "https://relay.example.test/v1",
        ) == "custom:copilot"
        assert auxiliary_client._auth_refresh_provider_for_route(
            "custom:anthropic", "https://relay.example.test/v1",
        ) == "custom:anthropic"

        refresh = MagicMock(return_value=True)
        monkeypatch.setattr(auxiliary_client, "_refresh_provider_credentials", refresh)
        monkeypatch.setattr(auxiliary_client, "_recoverable_pool_provider", lambda *args, **kwargs: None)
        monkeypatch.setattr(auxiliary_client, "_is_auth_error", lambda exc: True)
        route = auxiliary_client._LadderRoute(
            client=MagicMock(api_key="private-key"), task="compression", tag="",
            resolved_provider="custom:copilot", resolved_model="relay-model",
            resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
            final_model="relay-model", base_info="https://relay.example.test/v1",
            main_runtime={}, async_mode=False, route_info=None,
        )
        ladder = auxiliary_client._ladder_credential_rungs(
            RuntimeError("unauthorized"), route, {}, False,
        )

        with pytest.raises(StopIteration):
            next(ladder)
        refresh.assert_not_called()

    def test_custom_auth_failure_rotates_only_custom_pool(self, tmp_path, monkeypatch):
        """A custom endpoint rejection cannot quarantine the colliding built-in pool."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "openrouter",
                    "base_url": "https://relay.example.test/v1",
                },
            },
        })
        from agent import auxiliary_client

        client = MagicMock()
        client.base_url = "https://relay.example.test/v1"
        custom_pool = MagicMock()
        custom_pool.has_credentials.return_value = True
        custom_pool.try_refresh_current.return_value = None
        custom_pool.mark_exhausted_and_rotate.return_value = MagicMock()
        builtin_pool = MagicMock()
        builtin_pool.has_credentials.return_value = True
        pools = {
            "custom:openrouter": custom_pool,
            "openrouter": builtin_pool,
        }
        class AuthError(Exception):
            status_code = 401

        err = AuthError("unauthorized")
        monkeypatch.setattr(auxiliary_client, "load_pool", lambda key: pools[key])
        monkeypatch.setattr(auxiliary_client, "_is_auth_error", lambda exc: exc is err)

        pool_key = auxiliary_client._recoverable_pool_provider(
            "custom:openrouter", client,
        )
        assert pool_key == "custom:openrouter"
        recovered = auxiliary_client._recover_provider_pool(
            pool_key, err, failed_api_key="private-router-key",
        )
        assert recovered is True
        custom_pool.mark_exhausted_and_rotate.assert_called_once()
        builtin_pool.mark_exhausted_and_rotate.assert_not_called()

    def test_named_custom_foreign_base_override_does_not_borrow_saved_key(self, tmp_path):
        """An explicit foreign endpoint must bring its own key on text and vision paths."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "Private Relay",
                    "base_url": "https://relay.example.test/v1",
                    "api_key": "saved-private-key",
                    "default_model": "relay-model",
                },
            },
        })
        from agent import auxiliary_client
        from agent.auxiliary_client import resolve_provider_client, resolve_vision_provider_client

        client, _ = resolve_provider_client(
            "custom:private-relay", model="relay-model", raw_codex=True,
            explicit_base_url="https://foreign.example.test/v1",
        )
        _, vision_client, _ = resolve_vision_provider_client(
            provider="custom:private-relay", model="relay-model",
            base_url="https://foreign.example.test/v1",
        )

        cached_client, _ = auxiliary_client._get_cached_client(
            "custom:private-relay", "relay-model",
            base_url="https://foreign.example.test/v1",
        )

        assert client is not None and client.api_key == "no-key-required"
        assert vision_client is not None and vision_client.api_key == "no-key-required"
        assert cached_client is not None and cached_client.api_key == "no-key-required"
        assert "foreign.example.test" in str(client.base_url)
        assert "foreign.example.test" in str(vision_client.base_url)

    def test_named_custom_recovery_rejects_foreign_endpoint_override(self, tmp_path):
        """A 401 from a foreign override cannot rotate or expose the configured custom pool."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter": {
                    "name": "Private Relay",
                    "base_url": "https://relay.example.test/v1",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "custom:openrouter": [{
                    "id": "private", "label": "private", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "private-key",
                }],
            },
        }))
        from agent import auxiliary_client

        foreign = MagicMock()
        foreign.base_url = "https://foreign.example.test/v1"
        foreign.api_key = "private-key"

        assert auxiliary_client._recoverable_pool_provider(
            "private-relay", foreign,
        ) is None
        assert auxiliary_client._recoverable_pool_provider(
            "custom:private-relay", foreign,
        ) is None

    def test_custom_pool_rotation_rebuilds_cached_client_with_next_key(self, tmp_path, monkeypatch):
        """After a custom 401, cache eviction must expose the rotated credential on retry."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter-relay": {
                    "name": "Relay",
                    "base_url": "https://relay.example.test/v1",
                    "default_model": "relay-model",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter-relay": [
                    {
                        "id": "first", "label": "first", "auth_type": "api_key",
                        "priority": 0, "source": "manual", "access_token": "private-key-1",
                    },
                    {
                        "id": "second", "label": "second", "auth_type": "api_key",
                        "priority": 1, "source": "manual", "access_token": "private-key-2",
                    },
                ],
            },
        }))
        from agent import auxiliary_client

        first_client, _ = auxiliary_client._get_cached_client(
            "custom:openrouter-relay", "relay-model",
        )
        assert first_client is not None
        assert first_client.api_key == "private-key-1"
        pool_key = auxiliary_client._recoverable_pool_provider(
            "custom:openrouter-relay", first_client,
        )
        assert pool_key == "openrouter-relay"

        class AuthError(Exception):
            status_code = 401

        err = AuthError("unauthorized")
        monkeypatch.setattr(auxiliary_client, "_is_auth_error", lambda exc: exc is err)
        assert auxiliary_client._recover_provider_pool(
            pool_key, err, failed_api_key="private-key-1",
        ) is True

        second_client, _ = auxiliary_client._get_cached_client(
            "custom:openrouter-relay", "relay-model",
        )
        assert second_client is not None
        assert second_client is not first_client
        assert second_client.api_key == "private-key-2"

    def test_bare_openrouter_still_uses_builtin_pool(self, tmp_path, monkeypatch):
        """The strict custom namespace must not change bare built-in OpenRouter routing."""
        _write_config(tmp_path, {
            "providers": {
                "openrouter-relay": {
                    "name": "openrouter",
                    "base_url": "https://relay.example.test/v1",
                },
            },
        })
        auth_path = tmp_path / ".hermes" / "auth.json"
        auth_path.write_text(json.dumps({
            "version": 1,
            "providers": {},
            "credential_pool": {
                "openrouter": [{
                    "id": "public", "label": "public", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "public-openrouter-key",
                }],
                "openrouter-relay": [{
                    "id": "private", "label": "private", "auth_type": "api_key",
                    "priority": 0, "source": "manual", "access_token": "private-router-key",
                }],
            },
        }))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "openrouter", model="openai/gpt-4o-mini", raw_codex=True,
        )

        assert client is not None
        assert client.api_key == "public-openrouter-key"
        assert "openrouter.ai" in str(client.base_url)
        assert model == "openai/gpt-4o-mini"

    def test_missing_explicit_custom_openrouter_does_not_fall_back_to_builtin(
        self, tmp_path, monkeypatch,
    ):
        """An absent explicit custom identity fails closed instead of using
        the built-in OpenRouter endpoint and its unrelated credential."""
        _write_config(tmp_path, {
            "model": {"provider": "custom:openrouter", "default": "relay-model"},
        })
        monkeypatch.setenv("OPENROUTER_API_KEY", "public-openrouter-key")

        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "custom:openrouter", model="relay-model", raw_codex=True,
        )

        assert client is None
        assert model is None

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
