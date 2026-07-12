"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import base64
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    get_text_auxiliary_client,
    get_available_vision_backends,
    resolve_vision_provider_client,
    resolve_provider_client,
    auxiliary_max_tokens_param,
    call_llm,
    async_call_llm,
    _build_call_kwargs,
    _read_codex_access_token,
    _get_provider_chain,
    _is_payment_error,
    _is_rate_limit_error,
    _is_model_not_found_error,
    _is_model_incompatible_error,
    _refresh_nous_recommended_model,
    _normalize_aux_provider,
    _try_payment_fallback,
    _try_openrouter,
    _OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    _resolve_auto,
    _resolve_task_provider_model,
    _resolve_xai_oauth_for_aux,
    _CodexCompletionsAdapter,
    _pool_runtime_base_url,
)


def _jwt_with_claims(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class _FakeAnthropicStream:
    def __init__(self, final_message):
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self._final_message


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)
    # Module-level unhealthy cache (10-min TTL) leaks between tests;
    # earlier tests that call _mark_provider_unhealthy() poison the
    # cache for later ones, causing _resolve_auto to skip providers
    # that the test patched to return valid clients.
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()


@pytest.fixture
def codex_auth_dir(tmp_path, monkeypatch):
    """Provide a writable ~/.codex/ directory with a valid auth.json."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {
            "access_token": "codex-test-token-abc123",
            "refresh_token": "codex-refresh-xyz",
        }
    }))
    monkeypatch.setattr(
        "agent.auxiliary_client._read_codex_access_token",
        lambda: "codex-test-token-abc123",
    )
    return codex_dir


class TestAuxiliaryMaxTokensParam:
    def test_uses_max_completion_tokens_for_github_copilot_custom_base(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}

    def test_uses_max_completion_tokens_for_github_copilot_custom_base_path(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com/chat/completions", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}


class TestResolveTaskProviderModel:
    @pytest.mark.parametrize(
        "provider",
        [
            "anthropic",
            "minimax-oauth",
            "nous",
            "openai-codex",
            "qwen-oauth",
            "xai-oauth",
        ],
    )
    def test_explicit_base_url_preserves_first_class_provider_identity(self, provider):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="moa_reference",
            provider=provider,
            model="test-model",
            base_url="https://provider.example/v1",
            api_key="resolved-token",
        )

        assert resolved_provider == provider
        assert model == "test-model"
        assert base_url == "https://provider.example/v1"
        assert api_key == "resolved-token"
        assert api_mode is None

    @pytest.mark.parametrize("provider", ["", "auto", "custom", "custom:local", "unknown-provider"])
    def test_explicit_base_url_without_first_class_provider_routes_as_custom(self, provider):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="moa_reference",
            provider=provider,
            model="test-model",
            base_url="https://provider.example/v1",
            api_key="resolved-token",
        )

        assert resolved_provider == "custom"
        assert model == "test-model"
        assert base_url == "https://provider.example/v1"
        assert api_key == "resolved-token"
        assert api_mode is None

    def test_direct_openai_alias_with_base_url_still_routes_as_custom(self):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="vision",
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://proxy.example/v1",
            api_key="sk-test",
        )

        assert resolved_provider == "custom"
        assert model == "gpt-4o-mini"
        assert base_url == "https://proxy.example/v1"
        assert api_key == "sk-test"
        assert api_mode is None

    def test_explicit_provider_adopts_configured_task_endpoint(self):
        """Explicit provider matching the configured one must not bypass
        auxiliary.<task>.base_url/api_key (#58515)."""
        task_config = {
            "provider": "custom",
            "model": "meta/llama-3.2-11b-vision-instruct",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "nvapi-secret",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
                model="meta/llama-3.2-11b-vision-instruct",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://integrate.api.nvidia.com/v1"
        assert api_key == "nvapi-secret"
        assert model == "meta/llama-3.2-11b-vision-instruct"
        assert api_mode is None

    def test_explicit_provider_adopts_endpoint_when_config_names_no_provider(self):
        task_config = {
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://nim.example/v1"
        assert api_key == "cfg-key"

    def test_explicit_first_class_provider_with_matching_config_keeps_identity(self):
        task_config = {
            "provider": "anthropic",
            "base_url": "https://anthropic-proxy.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="compression",
                provider="anthropic",
            )

        assert resolved_provider == "anthropic"
        assert base_url == "https://anthropic-proxy.example/v1"
        assert api_key == "cfg-key"

    def test_explicit_auto_provider_keeps_auto_resolution(self):
        """provider="auto" is a sentinel for "inherit / auto-detect" and must
        not adopt the configured endpoint — the auto chain owns resolution."""
        task_config = {
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="auto",
            )

        assert resolved_provider == "auto"
        assert base_url is None
        assert api_key is None

    def test_explicit_provider_differing_from_config_ignores_config_endpoint(self):
        """A caller forcing a different provider keeps full explicit-arg
        priority — the configured endpoint belongs to cfg_provider only."""
        task_config = {
            "provider": "custom",
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="nous",
            )

        assert resolved_provider == "nous"
        assert base_url is None
        assert api_key is None

    def test_explicit_provider_and_base_url_still_win_over_config(self):
        task_config = {
            "provider": "custom",
            "base_url": "https://configured.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
                base_url="https://explicit.example/v1",
                api_key="explicit-key",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://explicit.example/v1"
        assert api_key == "explicit-key"


class TestBuildCallKwargsMaxTokens:
    """_build_call_kwargs should not cap output by default (#34530).

    Most chat-completions providers treat an omitted max_tokens as "use the
    model max", which is what we want for auxiliary tasks. An explicit cap only
    risks truncation or a wire-format 400 (GitHub Copilot / GPT-5 reject
    max_tokens; ZAI vision rejects it entirely). The Anthropic Messages wire is
    the one exception — max_tokens is a mandatory field there.
    """

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("copilot", "gpt-5.4", "https://api.githubcopilot.com"),
            ("copilot", "gpt-5.5", "https://api.githubcopilot.com"),
            ("custom", "gpt-5", "https://api.openai.com/v1"),
            ("openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1"),
            ("nous", "hermes-4", "https://inference-api.nousresearch.com/v1"),
            ("custom", "qwen", "http://localhost:8080/v1"),
            ("zai", "glm-4v-flash", "https://open.bigmodel.cn/api/paas/v4"),
        ],
    )
    def test_omits_max_tokens_for_openai_compatible(self, provider, model, base_url):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1234,
            base_url=base_url,
        )
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("minimax", "minimax-m2", "https://api.minimax.io/v1"),
            ("custom", "claude", "https://proxy.example.com/anthropic/v1"),
        ],
    )
    def test_keeps_max_tokens_on_anthropic_wire(self, provider, model, base_url):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1234,
            base_url=base_url,
        )
        assert kwargs["max_tokens"] == 1234
        assert "max_completion_tokens" not in kwargs

    def test_keeps_max_tokens_for_nvidia_nim(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="nvidia",
            model="minimaxai/minimax-m3",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        assert kwargs["max_tokens"] == 4096


class TestNousTagsScoping:
    def test_tags_injected_when_provider_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", False)

        kwargs = aux._build_call_kwargs(
            provider="nous",
            model="hermes-4",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert kwargs["extra_body"]["tags"] == aux._nous_portal_tags()

    def test_tags_not_injected_for_gemini_when_main_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", True)

        kwargs = aux._build_call_kwargs(
            provider="gemini",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "extra_body" not in kwargs

    def test_tags_not_injected_for_openrouter_when_main_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", True)

        kwargs = aux._build_call_kwargs(
            provider="openrouter",
            model="openai/gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "extra_body" not in kwargs


class TestNormalizeAuxProvider:
    def test_maps_github_copilot_aliases(self):
        assert _normalize_aux_provider("github") == "copilot"
        assert _normalize_aux_provider("github-copilot") == "copilot"
        assert _normalize_aux_provider("github-models") == "copilot"

    def test_maps_github_copilot_acp_aliases(self):
        assert _normalize_aux_provider("github-copilot-acp") == "copilot-acp"
        assert _normalize_aux_provider("copilot-acp-agent") == "copilot-acp"


class TestReadCodexAccessToken:
    def test_valid_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "tok-123", "refresh_token": "r-456"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "tok-123"

    def test_pool_without_selected_entry_falls_back_to_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        valid_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjk5OTk5OTk5OTl9.sig"
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("hermes_cli.auth._read_codex_tokens", return_value={
                 "tokens": {"access_token": valid_jwt, "refresh_token": "refresh"}
             }):
            result = _read_codex_access_token()

        assert result == valid_jwt

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            result = _read_codex_access_token()
        assert result is None

    def test_empty_token_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "  ", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{bad json")
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_missing_tokens_key_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps({"other": "data"}))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None


    def test_expired_jwt_returns_none(self, tmp_path, monkeypatch):
        """Expired JWT tokens should be skipped so auto chain continues."""
        import base64
        import time as _time

        # Build a JWT with exp in the past
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            result = _read_codex_access_token()
        assert result is None, "Expired JWT should return None"

    def test_valid_jwt_returns_token(self, tmp_path, monkeypatch):
        """Non-expired JWT tokens should be returned."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) + 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        valid_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": valid_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == valid_jwt

    def test_non_jwt_token_passes_through(self, tmp_path, monkeypatch):
        """Non-JWT tokens (no dots) should be returned as-is."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "plain-token-no-jwt", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "plain-token-no-jwt"


class TestResolveXaiOAuthForAux:
    def test_uses_pool_backed_credentials_without_singleton(self, tmp_path, monkeypatch):
        """Auxiliary xAI OAuth must see pool-only credentials.

        ``hermes auth status`` already reports these as logged in; compression
        should not fall through to "no auxiliary provider configured" just
        because the singleton auth-store entry is absent.
        """
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
        monkeypatch.delenv("XAI_BASE_URL", raising=False)

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai123",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            DEFAULT_XAI_OAUTH_BASE_URL,
        )

    def test_pool_backed_credentials_honor_base_url_env_override(self, tmp_path, monkeypatch):
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://example.x.ai/v1/")

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai456",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            "https://example.x.ai/v1",
        )


class TestAnthropicOAuthFlag:
    """Test that OAuth tokens get is_oauth=True in auxiliary Anthropic client."""

    def test_oauth_token_sets_flag(self, monkeypatch):
        """OAuth tokens (sk-ant-oat01-*) should create client with is_oauth=True."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-token")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            # The adapter inside should have is_oauth=True
            adapter = client.chat.completions
            assert adapter._is_oauth is True

    def test_api_key_no_oauth_flag(self, monkeypatch):
        """Regular API keys (sk-ant-api-*) should create client with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-testkey1234"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_pool_entry_takes_priority_over_legacy_resolution(self):
        class _Entry:
            access_token = "sk-ant-oat01-pooled"
            base_url = "https://api.anthropic.com"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", side_effect=AssertionError("legacy path should not run")),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as mock_build,
        ):
            from agent.auxiliary_client import _try_anthropic

            client, model = _try_anthropic()

        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        assert mock_build.call_args.args[0] == "sk-ant-oat01-pooled"


class TestBuildCodexClient:
    def test_pool_without_selected_entry_falls_back_to_auth_store(self):
        with (
            patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
            patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-auth-token"),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.call_args.kwargs["api_key"] == "codex-auth-token"
        assert mock_openai.call_args.kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"

    def test_rejects_missing_model(self):
        """Callers must pass an explicit model; no hardcoded default."""
        from agent.auxiliary_client import _build_codex_client

        client, model = _build_codex_client("")
        assert client is None
        assert model is None

    def test_cached_codex_client_rebuilds_when_pool_entry_changes(self):
        import agent.auxiliary_client as aux

        class _Entry:
            def __init__(self, entry_id, token):
                self.id = entry_id
                self.runtime_api_key = token
                self.runtime_base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def __init__(self):
                self.entry = _Entry("cred-a", "tok-a")

            def has_credentials(self):
                return True

            def current(self):
                return self.entry

            def peek(self):
                return self.entry

            def select(self):
                return self.entry

        pool = _Pool()
        client_a = MagicMock(name="codex-client-a")
        client_b = MagicMock(name="codex-client-b")

        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client.OpenAI", side_effect=[client_a, client_b]) as mock_openai,
        ):
            aux.shutdown_cached_clients()
            try:
                first_client, first_model = aux._get_cached_client("openai-codex", "gpt-5.4")
                pool.entry = _Entry("cred-b", "tok-b")
                second_client, second_model = aux._get_cached_client("openai-codex", "gpt-5.4")
            finally:
                aux.shutdown_cached_clients()

        assert first_client is not second_client
        assert first_model == "gpt-5.4"
        assert second_model == "gpt-5.4"
        assert mock_openai.call_count == 2


class TestResolveProviderClientUniversalModelFallback:
    """resolve_provider_client() picks a sensible model when callers pass none (#31845).

    Aux tasks (title generation, vision, session search, etc.) routinely
    reach this function without an explicit model — the user's main
    provider was picked via ``hermes model``, no per-task override is
    set, and the expectation is "just use my main model for side tasks
    too."  The resolver fills in ``model`` from a 3-step universal
    fallback before any provider branch runs:

        1. ``model`` argument           (caller knew what they wanted)
        2. provider's catalog default   (cheap aux model, if registered)
        3. user's main model            (``model.model`` in config.yaml)

    Pre-fix the OAuth providers (xai-oauth, openai-codex) returned
    ``(None, None)`` on an empty model — both lack a catalog default
    because their accepted-model lists drift on the backend.  That
    silent failure caused ``_resolve_auto`` to drop to its Step-2
    fallback chain (OpenRouter / Nous / etc.), so aux tasks billed
    against the wrong subscription.
    """

    def test_empty_model_for_oauth_provider_falls_back_to_main_model(self):
        """xai-oauth: no catalog default → uses main model."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="grok-4.3",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",  # xai-oauth has no catalog default
            ),
            patch(
                "agent.auxiliary_client._build_xai_oauth_aux_client",
                return_value=(MagicMock(), "grok-4.3"),
            ) as mock_build,
        ):
            client, model = resolve_provider_client("xai-oauth", "")

        assert client is not None, (
            "should not fall through when main model is set"
        )
        assert model == "grok-4.3"
        # The builder receives the main-model fallback, never the empty
        # string the caller passed.
        assert mock_build.call_args.args[0] == "grok-4.3"

    def test_empty_model_for_codex_also_uses_main_model(self):
        """openai-codex: symmetric with xai-oauth — same universal fallback."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="gpt-5.4",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",  # openai-codex has no catalog default either
            ),
            patch(
                "agent.auxiliary_client._build_codex_client",
                return_value=(MagicMock(), "gpt-5.4"),
            ) as mock_build,
            patch(
                "agent.auxiliary_client._select_pool_entry",
                return_value=(True, None),
            ),
        ):
            client, model = resolve_provider_client("openai-codex", "")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_build.call_args.args[0] == "gpt-5.4"

    def test_empty_model_for_catalog_provider_uses_catalog_default(self):
        """anthropic / nous / openrouter / etc.: catalog default wins
        over main model when no explicit model is passed.

        This preserves the original \"cheap aux model for direct API
        providers\" behaviour — users on anthropic for their main chat
        still get claude-haiku-4-5 for title generation, NOT their
        expensive chat model.  Step 2 of the universal fallback chain.
        """
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                # Main model is the expensive opus; if this leaks into
                # aux it costs real money.
                return_value="claude-opus-4-6",
            ) as mock_read_main,
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="claude-haiku-4-5-20251001",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(),
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="sk-ant-***",
            ),
            patch(
                "agent.auxiliary_client._read_nous_auth", return_value=None
            ),
        ):
            client, model = resolve_provider_client("anthropic", "")

        # Catalog default takes precedence — main_model was a no-op
        # because step 2 of the fallback chain already produced a model.
        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        mock_read_main.assert_not_called()

    def test_explicit_model_takes_precedence_over_fallbacks(self):
        """Step 1: caller-passed model wins.  Per-task config
        (``auxiliary.<task>.model``) routes here — when the user
        explicitly picks gemini-3-flash for title generation, that's
        what runs, not their main model.
        """
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch("agent.auxiliary_client._read_main_model") as mock_read_main,
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="catalog-default-should-not-be-used",
            ),
            patch(
                "agent.auxiliary_client._build_xai_oauth_aux_client",
                return_value=(MagicMock(), "grok-4.20-multi-agent"),
            ) as mock_build,
        ):
            client, model = resolve_provider_client(
                "xai-oauth", "grok-4.20-multi-agent",
            )

        assert client is not None
        assert model == "grok-4.20-multi-agent"
        mock_read_main.assert_not_called()
        assert mock_build.call_args.args[0] == "grok-4.20-multi-agent"


class TestExpiredCodexFallback:
    """Test that expired Codex tokens don't block the auto chain."""

    def test_expired_codex_falls_through_to_next(self, tmp_path, monkeypatch):
        """When Codex token is expired, auto chain should skip it and try next provider."""
        import base64
        import time as _time

        # Expired Codex JWT
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Set up Anthropic as fallback
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-fallback")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
            # Should NOT be Codex, should be Anthropic (or another available provider)
            assert not isinstance(client, type(None)), "Should find a provider after expired Codex"


    def test_expired_codex_openrouter_wins(self, tmp_path, monkeypatch):
        """With expired Codex + OpenRouter key, OpenRouter should win (1st in chain)."""
        import base64
        import time as _time

        # Belt-and-suspenders: _try_openrouter marks openrouter unhealthy
        # when OPENROUTER_API_KEY is absent (which the preceding test in
        # this class exercises).  The file-level _clean_env autouse fixture
        # clears the cache, but fixture ordering with the conftest
        # _hermetic_environment autouse can leave a narrow window where
        # the mark reappears.  Explicitly clear here so this test is
        # independent of run order.
        import agent.auxiliary_client as _aux_mod
        _aux_mod._aux_unhealthy_until.clear()
        _aux_mod._aux_unhealthy_logged_at.clear()

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
            assert client is not None
            # OpenRouter is 1st in chain, should win
            mock_openai.assert_called()

    def test_expired_codex_custom_endpoint_wins(self, tmp_path, monkeypatch):
        """With expired Codex + custom endpoint (Ollama), custom should win (3rd in chain)."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Simulate Ollama or custom endpoint
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "sk-dummy")):
            with patch("agent.auxiliary_client.OpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                from agent.auxiliary_client import _resolve_auto
                client, model = _resolve_auto()
                assert client is not None


    def test_hermes_oauth_file_sets_oauth_flag(self, monkeypatch):
        """OAuth-style tokens should get is_oauth=*** (token is not sk-ant-api-*)."""
        # Mock resolve_anthropic_token to return an OAuth-style token
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat-hermes-token"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
            assert client is not None, "Should resolve token"
            adapter = client.chat.completions
            assert adapter._is_oauth is True, "Non-sk-ant-api token should set is_oauth=True"

    def test_jwt_missing_exp_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid JSON but no exp claim should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"sub": "user123"}).encode()  # no exp
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        no_exp_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": no_exp_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == no_exp_jwt, "JWT without exp should pass through"

    def test_jwt_invalid_json_payload_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid base64 but invalid JSON payload should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b"not-json-content").rstrip(b"=").decode()
        bad_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": bad_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == bad_jwt, "JWT with invalid JSON payload should pass through"

    def test_claude_code_oauth_env_sets_flag(self, monkeypatch):
        """CLAUDE_CODE_OAUTH_TOKEN env var should get is_oauth=True."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-cc-test-token")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is True


class TestExplicitProviderRouting:
    """Test explicit provider selection bypasses auto chain correctly."""

    def test_explicit_anthropic_api_key(self, monkeypatch):
        """provider='anthropic' + regular API key should work with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api-regular-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client("anthropic")
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_explicit_openrouter_pool_exhausted_logs_precise_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "credential pool has no usable entries" in record.message
            for record in caplog.records
        )
        assert not any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

    def test_explicit_openrouter_missing_env_keeps_not_set_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

    def test_try_openrouter_pool_exhausted_falls_back_to_env(self, monkeypatch):
        """Pool present but exhausted → fall through to OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-fallback")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client

            client, model = _try_openrouter()

        assert client is mock_client
        assert model == _OPENROUTER_MODEL
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "sk-or-env-fallback"
        assert mock_openai.call_args.kwargs["base_url"] == OPENROUTER_BASE_URL

    def test_try_openrouter_pool_exhausted_no_env_marks_unhealthy(self, monkeypatch):
        """Pool exhausted AND no env var → final failure marks provider unhealthy."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("agent.auxiliary_client._mark_provider_unhealthy") as mock_mark, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter()

        assert client is None
        assert model is None
        mock_openai.assert_not_called()
        mock_mark.assert_called_once_with("openrouter", ttl=60)

class TestGetTextAuxiliaryClient:
    """Test the full resolution chain for get_text_auxiliary_client."""

    def test_codex_pool_entry_takes_priority_over_auth_store(self):
        class _Entry:
            access_token = "pooled-codex-token"
            base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI"),
            patch("hermes_cli.auth._read_codex_tokens", side_effect=AssertionError("legacy codex store should not run")),
        ):
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        from agent.auxiliary_client import CodexAuxiliaryClient

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.4"

    def test_returns_none_when_nothing_available(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = get_text_auxiliary_client()
        assert client is None
        assert model is None

    def test_custom_endpoint_uses_codex_wrapper_when_runtime_requests_responses_api(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("https://api.openai.com/v1", "sk-test", "codex_responses")), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=None), \
             patch("agent.auxiliary_client._read_main_model", return_value="gpt-5.3-codex"), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()

        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.3-codex"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-test"


class TestVisionClientFallback:
    """Vision client auto mode resolves known-good multimodal backends."""

    def test_vision_auto_includes_active_provider_when_configured(self, monkeypatch):
        """Active provider appears in available backends when credentials exist."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-sonnet-4"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            backends = get_available_vision_backends()

        assert "anthropic" in backends

    def test_resolve_provider_client_returns_native_anthropic_wrapper(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            client, model = resolve_provider_client("anthropic")

        assert client is not None
        assert client.__class__.__name__ == "AnthropicAuxiliaryClient"
        assert model == "claude-haiku-4-5-20251001"

    def test_anthropic_auxiliary_client_aggregates_stream_response(self):
        from agent.auxiliary_client import AnthropicAuxiliaryClient

        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="streamed aux response")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
        messages_api = SimpleNamespace(
            stream=MagicMock(return_value=_FakeAnthropicStream(final_message)),
            create=MagicMock(return_value="raw event-stream text"),
        )
        real_client = SimpleNamespace(messages=messages_api)
        client = AnthropicAuxiliaryClient(
            real_client,
            "claude-sonnet-4-20250514",
            "sk-test",
            "https://sse-only.example/v1",
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "summarize"}],
            max_tokens=16,
        )

        messages_api.stream.assert_called_once()
        messages_api.create.assert_not_called()
        assert response.choices[0].message.content == "streamed aux response"
        assert response.usage.prompt_tokens == 3
        assert response.usage.completion_tokens == 4

    def test_anthropic_auxiliary_client_uses_model_output_limit_by_default(self):
        from agent.auxiliary_client import AnthropicAuxiliaryClient

        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="aux response")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
        messages_api = SimpleNamespace(create=MagicMock())
        real_client = SimpleNamespace(messages=messages_api)
        captured_kwargs = {}

        def fake_create_anthropic_message(_client, kwargs):
            captured_kwargs.update(kwargs)
            return final_message

        client = AnthropicAuxiliaryClient(
            real_client,
            "claude-opus-4-8",
            "sk-test",
            "https://api.anthropic.com",
        )

        with patch(
            "agent.anthropic_adapter.create_anthropic_message",
            side_effect=fake_create_anthropic_message,
        ):
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert response.choices[0].message.content == "aux response"
        # Behavior contract, not a frozen literal: a capless native-Anthropic
        # aux call must default to the model's native output ceiling (resolved
        # via _get_anthropic_max_output) rather than the old hidden 2000 cap.
        # Asserting against the resolver keeps this test alive across
        # model-table churn while still catching a regression to `or 2000`.
        from agent.anthropic_adapter import _get_anthropic_max_output

        expected_ceiling = _get_anthropic_max_output("claude-opus-4-8")
        assert expected_ceiling > 2000
        assert captured_kwargs["max_tokens"] == expected_ceiling


class TestAuxiliaryPoolAwareness:
    def test_try_nous_uses_pool_entry(self):
        pooled_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() + 3600),
        })

        class _Entry:
            access_token = "pooled-access-token"
            agent_key = pooled_token
            agent_key_expires_at = "2099-01-01T00:00:00+00:00"
            scope = "inference:invoke"
            inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None),
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        assert mock_openai.call_args.kwargs["api_key"] == pooled_token
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_try_nous_refreshes_stale_pool_entry(self):
        stale_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() - 60),
        })
        fresh_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() + 3600),
        })

        class _Entry:
            def __init__(self, token):
                self.access_token = "pooled-access-token"
                self.agent_key = token
                self.agent_key_expires_at = "2099-01-01T00:00:00+00:00"
                self.scope = "inference:invoke"
                self.inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            refreshed = False

            def has_credentials(self):
                return True

            def select(self):
                return _Entry(stale_token)

            def try_refresh_current(self):
                self.refreshed = True
                return _Entry(fresh_token)

        pool = _Pool()
        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None),
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert pool.refreshed is True
        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        assert mock_openai.call_args.kwargs["api_key"] == fresh_token
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_resolve_nous_runtime_api_rejects_stale_pool_entry_when_refresh_fails(self):
        stale_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() - 60),
        })

        class _Entry:
            access_token = "pooled-access-token"
            agent_key = stale_token
            agent_key_expires_at = "2099-01-01T00:00:00+00:00"
            scope = "inference:invoke"
            inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

            def try_refresh_current(self):
                return None

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch(
                "hermes_cli.auth.resolve_nous_runtime_credentials",
                side_effect=RuntimeError("no singleton auth"),
            ),
        ):
            from agent.auxiliary_client import _resolve_nous_runtime_api

            runtime = _resolve_nous_runtime_api()

        assert runtime is None

    def test_try_nous_uses_portal_recommendation_for_text(self):
        """When the Portal recommends a compaction model, _try_nous honors it."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="minimax/minimax-m2.7") as mock_rec,
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            from agent.auxiliary_client import _try_nous

            mock_openai.return_value = MagicMock()
            client, model = _try_nous(vision=False)

        assert client is not None
        assert model == "minimax/minimax-m2.7"
        assert mock_rec.call_args.kwargs["vision"] is False

    def test_try_nous_uses_portal_recommendation_for_vision(self):
        """Vision tasks should ask for the vision-specific recommendation."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="google/gemini-3-flash-preview") as mock_rec,
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous(vision=True)

        assert client is not None
        assert model == "google/gemini-3-flash-preview"
        assert mock_rec.call_args.kwargs["vision"] is True

    def test_try_nous_falls_back_when_recommendation_lookup_raises(self):
        """If the Portal lookup throws, we must still return a usable model."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", side_effect=RuntimeError("portal down")),
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous()

        assert client is not None
        assert model == "google/gemini-3-flash-preview"

    def test_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create.side_effect = _Auth401("stale nous key")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_client.chat.completions.create.return_value = {"ok": True}

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client.OpenAI", return_value=fresh_client),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_nous_after_free_tier_block_when_account_paid(self):
        from hermes_cli.nous_account import NousPortalAccountInfo

        class _Payment404(Exception):
            status_code = 404

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create.side_effect = _Payment404(
            "model_not_supported_on_free_tier: model is not available on the free tier"
        )

        fresh_client = MagicMock()
        fresh_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_client.chat.completions.create.return_value = {"ok": True}

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client.OpenAI", return_value=fresh_client),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
            patch(
                "hermes_cli.nous_account.get_nous_portal_account_info",
                return_value=NousPortalAccountInfo(
                    logged_in=True,
                    source="account_api",
                    fresh=True,
                    paid_service_access=True,
                ),
            ),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create = AsyncMock(side_effect=_Auth401("stale nous key"))

        fresh_async_client = MagicMock()
        fresh_async_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_async_client.chat.completions.create = AsyncMock(return_value={"ok": True})

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client._to_async_client", return_value=(fresh_async_client, "nous-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_async_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_nous_after_free_tier_block_when_account_paid(self):
        from hermes_cli.nous_account import NousPortalAccountInfo

        class _Payment404(Exception):
            status_code = 404

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create = AsyncMock(side_effect=_Payment404(
            "model_not_supported_on_free_tier: model is not available on the free tier"
        ))

        fresh_async_client = MagicMock()
        fresh_async_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_async_client.chat.completions.create = AsyncMock(return_value={"ok": True})

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client._to_async_client", return_value=(fresh_async_client, "nous-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
            patch(
                "hermes_cli.nous_account.get_nous_portal_account_info",
                return_value=NousPortalAccountInfo(
                    logged_in=True,
                    source="account_api",
                    fresh=True,
                    paid_service_access=True,
                ),
            ),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_async_client.chat.completions.create.await_count == 1

    def test_cached_gmi_client_keeps_explicit_slash_model_override(self):
        import agent.auxiliary_client as aux

        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fake_client, "google/gemini-3.1-flash-lite-preview"),
        ) as mock_resolve:
            aux.shutdown_cached_clients()
            try:
                client, model = aux._get_cached_client(
                    "gmi",
                    "google/gemini-3.1-flash-lite-preview",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
                assert client is fake_client
                assert model == "google/gemini-3.1-flash-lite-preview"

                client, model = aux._get_cached_client(
                    "gmi",
                    "openai/gpt-5.4-mini",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
            finally:
                aux.shutdown_cached_clients()

        assert client is fake_client
        assert model == "openai/gpt-5.4-mini"
        # A DIFFERENT model resolves its own client (model participates in the
        # cache key). This isolation is what stops two concurrent advisors on
        # the same provider/base_url/key (e.g. a MoA fan-out) from sharing — and
        # racing the lifecycle of — one cached client. Same-model reuse is still
        # a single resolve (verified elsewhere); distinct models => distinct
        # resolves.
        assert mock_resolve.call_count == 2


# ── Payment / credit exhaustion fallback ─────────────────────────────────


class TestIsPaymentError:
    """_is_payment_error detects 402 and credit-related errors."""

    def test_402_status_code(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_402_with_credits_message(self):
        exc = Exception("You requested up to 65535 tokens, but can only afford 8029")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_429_with_credits_message(self):
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_404_free_tier_model_block_is_payment(self):
        exc = Exception(
            "Model 'gpt-5' is not available on the Free Tier. "
            "Upgrade at https://portal.nousresearch.com or pick a free model."
        )
        exc.status_code = 404
        assert _is_payment_error(exc) is True

    def test_403_subscription_required_is_payment(self):
        exc = Exception(
            "this model requires a subscription, upgrade for access: "
            "https://ollama.com/upgrade"
        )
        setattr(exc, "status_code", 403)
        assert _is_payment_error(exc) is True

    def test_429_session_usage_limit_is_payment(self):
        exc = Exception(
            "you have reached your session usage limit, upgrade for higher limits"
        )
        setattr(exc, "status_code", 429)
        assert _is_payment_error(exc) is True

    def test_404_generic_not_found_is_not_payment(self):
        exc = Exception("Not Found")
        exc.status_code = 404
        assert _is_payment_error(exc) is False

    def test_429_without_credits_message_is_not_payment(self):
        """Normal rate limits should NOT be treated as payment errors."""
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_payment_error(exc) is False

    def test_generic_500_is_not_payment(self):
        exc = Exception("Internal server error")
        exc.status_code = 500
        assert _is_payment_error(exc) is False

    def test_no_status_code_with_billing_message(self):
        exc = Exception("billing: payment required for this request")
        assert _is_payment_error(exc) is True

    def test_no_status_code_no_message(self):
        exc = Exception("connection reset")
        assert _is_payment_error(exc) is False

    # ── Daily / monthly quota exhaustion (#26803) ────────────────────────────

    def test_429_quota_exceeded(self):
        """Cloud provider quota exhaustion (e.g. Vertex AI) is a payment error."""
        exc = Exception("RESOURCE_EXHAUSTED: quota exceeded for project")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_too_many_tokens_per_day(self):
        """Bedrock / LiteLLM daily token limit is a payment error."""
        exc = Exception("Too many tokens per day: 1000000 used, 1000000 limit")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_daily_limit_phrase(self):
        """Generic 'daily limit' phrasing is a payment error."""
        exc = Exception("You have exceeded your daily limit.")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_resource_exhausted_grpc(self):
        """Vertex AI gRPC RESOURCE_EXHAUSTED maps to payment error."""
        exc = Exception("resource exhausted")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_daily_quota_phrase(self):
        """'daily quota' phrasing is a payment error."""
        exc = Exception("Daily quota of 500 requests reached.")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_transient_rate_limit_not_quota(self):
        """Transient 429 rate limit without quota keywords is NOT a payment error."""
        exc = Exception("Rate limit exceeded. Retry after 10s.")
        exc.status_code = 429
        assert _is_payment_error(exc) is False


class TestIsModelNotFoundError:
    """_is_model_not_found_error detects stale/invalid model 404s, distinct
    from payment errors."""

    def test_nous_openrouter_catalog_404(self):
        """The exact incident error: a Portal-recommended model dropped from
        the Nous → OpenRouter catalog."""
        exc = Exception(
            "Model 'gpt-5.4-mini' not found. The requested model does not "
            "exist in our configuration or OpenRouter catalog."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is True

    def test_openai_style_model_does_not_exist(self):
        exc = Exception("The model `gpt-9-turbo` does not exist")
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is True

    def test_invalid_model_id_400(self):
        exc = Exception("openrouter/foo/bar is not a valid model ID")
        exc.status_code = 400
        assert _is_model_not_found_error(exc) is True

    def test_no_such_model(self):
        exc = Exception("no such model: phantom-v1")
        exc.status_code = 400
        assert _is_model_not_found_error(exc) is True

    def test_billing_404_is_not_model_not_found(self):
        """Free-tier / credit 404s belong to _is_payment_error, not here —
        the two predicates must not overlap."""
        exc = Exception(
            "Model 'gpt-5' is not available on the free tier. Upgrade."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is False
        assert _is_payment_error(exc) is True

    def test_out_of_funds_404_is_not_model_not_found(self):
        exc = Exception(
            "Your API key is blocked or out of funds. model_not_found"
        )
        exc.status_code = 404
        # billing keyword wins — payment owns it
        assert _is_model_not_found_error(exc) is False

    def test_rate_limit_is_not_model_not_found(self):
        exc = Exception("rate limit exceeded, retry after 5s")
        exc.status_code = 429
        assert _is_model_not_found_error(exc) is False

    def test_500_is_not_model_not_found(self):
        exc = Exception("model does not exist")  # right phrase, wrong status
        exc.status_code = 500
        assert _is_model_not_found_error(exc) is False


class TestIsModelIncompatibleError:
    """_is_model_incompatible_error detects 400s where the route cannot run
    the model at all (capability mismatch), distinct from not-found and
    payment errors."""

    def test_codex_chatgpt_account_model_gating(self):
        """The exact incident: an openai-codex/ChatGPT-account fallback asked
        to compress a glm-5.2 conversation."""
        exc = Exception(
            "Error code: 400 - {'detail': \"The 'glm-5.2' model is not "
            "supported when using Codex with a ChatGPT account.\"}"
        )
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_model_is_not_supported_phrasing(self):
        exc = Exception("This model is not supported for this endpoint")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_unsupported_model_keyword(self):
        exc = Exception("unsupported model for this account tier")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_not_found_is_not_incompatible(self):
        """A model-does-not-exist 400 belongs to _is_model_not_found_error —
        the two predicates must not overlap."""
        exc = Exception("openrouter/foo/bar is not a valid model ID")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False
        assert _is_model_not_found_error(exc) is True

    def test_payment_400_is_not_incompatible(self):
        """A billing 400 that also contains capability-ish phrasing must be
        rejected here — billing keywords win so the payment path owns it and
        the two buckets don't overlap."""
        exc = Exception("insufficient credits: model is not supported on free tier")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False

    def test_wrong_status_is_not_incompatible(self):
        exc = Exception("model is not supported")  # right phrase, wrong status
        exc.status_code = 500
        assert _is_model_incompatible_error(exc) is False

    def test_generic_400_is_not_incompatible(self):
        """A plain request-validation 400 without capability phrasing must not
        trigger fallback (we respect explicit-provider choice for those)."""
        exc = Exception("invalid value for parameter temperature")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False


class TestRefreshNousRecommendedModel:
    """_refresh_nous_recommended_model picks a fresh model after a stale 404."""

    def test_returns_fresh_portal_recommendation(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: "stepfun/step-3.7-flash:free",
        )
        out = _refresh_nous_recommended_model(
            vision=True, stale_model="openai/gpt-5.4-mini")
        assert out == "stepfun/step-3.7-flash:free"

    def test_falls_back_to_default_when_portal_matches_stale(self, monkeypatch):
        """If the Portal still recommends the model that just 404'd, fall back
        to the known-good default."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: "openai/gpt-5.4-mini",
        )
        out = _refresh_nous_recommended_model(
            vision=True, stale_model="openai/gpt-5.4-mini")
        assert out == "google/gemini-3-flash-preview"

    def test_falls_back_to_default_when_portal_unavailable(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("portal down")
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model", _boom)
        out = _refresh_nous_recommended_model(
            vision=False, stale_model="some/dead-model")
        assert out == "google/gemini-3-flash-preview"

    def test_returns_none_when_no_distinct_alternative(self, monkeypatch):
        """When the failed model IS the default and the Portal has nothing
        else, there's no usable alternative."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: "google/gemini-3-flash-preview",
        )
        out = _refresh_nous_recommended_model(
            vision=False, stale_model="google/gemini-3-flash-preview")
        assert out is None


class TestIsRateLimitError:
    """_is_rate_limit_error detects 429 rate-limit errors warranting fallback."""

    def test_429_with_rate_limit_message(self):
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_resets_in_message(self):
        """Nous-style 429: 'resets in 3508s'."""
        exc = Exception("Hold up for a bit, you've exceeded the rate limit on your API key")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_too_many_requests(self):
        exc = Exception("Too many requests")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_without_billing_keywords_is_rate_limit(self):
        """Generic 429 without billing keywords = likely a rate limit."""
        exc = Exception("Something went wrong")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_credits_message_is_not_rate_limit(self):
        """Billing-related 429 should NOT be classified as rate limit."""
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_429_with_billing_message_is_not_rate_limit(self):
        exc = Exception("you can only afford 1000 tokens")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_402_is_not_rate_limit(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_rate_limit_error(exc) is False

    def test_500_is_not_rate_limit(self):
        exc = Exception("Internal Server Error")
        exc.status_code = 500
        assert _is_rate_limit_error(exc) is False

    def test_openai_ratelimiterror_classname(self):
        """OpenAI SDK RateLimitError may omit .status_code — detect by class name."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("rate limit exceeded")
        # No status_code set, but class name matches
        assert _is_rate_limit_error(exc) is True

    def test_no_status_code_no_keywords_is_not_rate_limit(self):
        exc = Exception("connection reset")
        assert _is_rate_limit_error(exc) is False


class TestGetProviderChain:
    """_get_provider_chain() resolves functions at call time (testable)."""

    def test_returns_four_entries(self):
        chain = _get_provider_chain()
        assert len(chain) == 4
        labels = [label for label, _ in chain]
        assert labels == ["openrouter", "nous", "local/custom", "api-key"]
        # Codex is deliberately NOT in this chain — see _get_provider_chain
        # docstring. ChatGPT-account Codex has a shifting model allow-list;
        # guessing a model to fall back on breaks more often than it helps.
        assert "openai-codex" not in labels

    def test_picks_up_patched_functions(self):
        """Patches on _try_* functions must be visible in the chain."""
        sentinel = lambda: ("patched", "model")
        with patch("agent.auxiliary_client._try_openrouter", sentinel):
            chain = _get_provider_chain()
        assert chain[0] == ("openrouter", sentinel)


class TestTryPaymentFallback:
    """_try_payment_fallback skips the failed provider and tries alternatives."""

    @pytest.fixture(autouse=True)
    def _clear_unhealthy_cache(self):
        """Earlier tests in this file call _mark_provider_unhealthy() which
        pollutes the module-level ``_aux_unhealthy_until`` dict (10-min TTL).
        Without this cleanup the fallback chain skips providers we've patched
        to return valid clients — the patched function is never called.
        """
        from agent.auxiliary_client import _aux_unhealthy_until, _aux_unhealthy_logged_at
        _aux_unhealthy_until.clear()
        _aux_unhealthy_logged_at.clear()
        yield
        _aux_unhealthy_until.clear()
        _aux_unhealthy_logged_at.clear()

    def test_skips_failed_provider(self):
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(mock_client, "nous-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is mock_client
        assert model == "nous-model"
        assert label == "nous"

    def test_returns_none_when_no_fallback(self):
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert label == ""

    def test_codex_alias_maps_to_chain_label(self):
        """'codex' should map to 'openai-codex' in the skip set."""
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(mock_client, "or-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openai-codex"):
            client, model, label = _try_payment_fallback("openai-codex", task="vision")
        assert client is mock_client
        assert label == "openrouter"

    def test_codex_not_in_fallback_chain(self):
        """Codex is deliberately NOT a fallback rung (shifting model allow-list).

        When OR/Nous/custom/api-key all fail, payment-fallback returns None —
        Codex is never tried with a guessed model.
        """
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert model is None
        assert label == ""

    def test_policy_off_never_walks_builtin_remote_chain(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "off"},
        ), patch("agent.auxiliary_client._get_provider_chain") as chain:
            client, model, label = _try_payment_fallback("custom", task="vision")

        chain.assert_not_called()
        assert (client, model, label) == (None, None, "")

    def test_local_only_skips_remote_and_returns_local_endpoint(self):
        remote = MagicMock(base_url="https://api.anthropic.com")
        local = MagicMock(base_url="http://10.55.0.3:8000/v1")
        remote_resolver = MagicMock(return_value=(remote, "claude"))
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "local-only"},
        ), patch(
            "agent.auxiliary_client._get_provider_chain",
            return_value=[
                ("anthropic", remote_resolver),
                ("local/custom", lambda: (local, "qwen")),
            ],
        ), patch("agent.auxiliary_client._read_main_provider", return_value="custom"):
            client, model, label = _try_payment_fallback("failed", task="compression")

        assert client is local
        assert model == "qwen"
        assert label == "local/custom"
        remote_resolver.assert_not_called()


class TestCallLlmPaymentFallback:
    """call_llm() retries with a different provider on 402 / payment / rate-limit errors."""

    def _make_402_error(self, msg="Payment Required: insufficient credits"):
        exc = Exception(msg)
        exc.status_code = 402
        return exc

    def _make_429_rate_limit_error(self, msg="Rate limit exceeded, try again in 60 seconds"):
        exc = Exception(msg)
        exc.status_code = 429
        return exc

    def test_non_payment_error_not_caught(self, monkeypatch):
        """Non-payment/non-connection errors (500) should NOT trigger fallback."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        server_err = Exception("Internal Server Error")
        server_err.status_code = 500
        primary_client.chat.completions.create.side_effect = server_err

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "google/gemini-3-flash-preview", None, None, None)):
            with pytest.raises(Exception, match="Internal Server Error"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

    def test_429_rate_limit_triggers_fallback(self, monkeypatch):
        """429 rate-limit errors should trigger fallback to next provider."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        rate_err = self._make_429_rate_limit_error()
        primary_client.chat.completions.create.side_effect = rate_err

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="fallback response"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "xiaomi/mimo-v2-pro")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "xiaomi/mimo-v2-pro", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(fallback_client, "fallback-model", "openrouter")):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
            )
        # Fallback client should have been used
        assert fallback_client.chat.completions.create.called

    def test_401_auth_error_triggers_fallback_in_auto_mode(self, monkeypatch):
        """401 auth errors should trigger fallback in auto mode (#21165).

        When refresh is unavailable/fails and the user is on the auto chain,
        a 401 must fall back instead of silently dropping the aux task
        (which caused compression message loss).
        """
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.base_url = "https://api.minimax.chat/v1"
        primary_client.chat.completions.create.side_effect = _AuxAuth401("expired key")

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = _DummyResponse("fallback auth response")

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimax/minimax-m2.7")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", "minimax/minimax-m2.7", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(fallback_client, "fallback-model", "openrouter")) as mock_fb:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "fallback auth response"
        assert fallback_client.chat.completions.create.called
        # Labelled as an auth error, not mis-tagged as a connection error.
        assert mock_fb.call_args.kwargs.get("reason") == "auth error"

    def test_401_auth_error_no_fallback_with_explicit_provider(self, monkeypatch):
        """401 on an explicitly-configured provider must NOT silently switch.

        Auth is not a capacity error: the explicit-provider gate means a 401
        respects the user's choice and raises instead of falling back. This
        guards the deliberate design at the should_fallback/is_capacity gate.
        """
        primary_client = MagicMock()
        primary_client.base_url = "https://api.minimax.chat/v1"
        primary_client.chat.completions.create.side_effect = _AuxAuth401("expired key")

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimax/minimax-m2.7")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("minimax", "minimax/minimax-m2.7", None, None, None)), \
             patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
             patch("agent.auxiliary_client._try_payment_fallback") as mock_fb:
            with pytest.raises(_AuxAuth401):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )
        mock_fb.assert_not_called()


class TestStaleFallbackCandidateSkip:
    """A fallback candidate with a stale credential must not abort the task.

    Live case (mattalachia debug dump, Jul 2026): Codex compression timed out,
    the aux chain fell back to Anthropic using an expired ANTHROPIC_TOKEN, and
    the resulting 401 aborted compression with a 60s cooldown — five times in
    one session — even though refreshing or skipping the candidate would have
    let compression proceed.
    """

    def _timeout_err(self):
        # Class name carries "Timeout" — matches _is_connection_error's
        # type-name detection, like the real Codex stream-deadline error.
        class _AuxStreamTimeoutError(Exception):
            pass
        return _AuxStreamTimeoutError(
            "Codex auxiliary Responses stream exceeded 120.0s total timeout")

    def test_stale_anthropic_fallback_refreshes_and_retries(self, monkeypatch):
        """401 from the fallback candidate → refresh its creds → retry succeeds."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        stale_fb = MagicMock()
        stale_fb.base_url = "https://api.anthropic.com"
        stale_fb.chat.completions.create.side_effect = _AuxAuth401("Invalid bearer token")

        fresh_fb = MagicMock()
        fresh_fb.base_url = "https://api.anthropic.com"
        fresh_fb.chat.completions.create.return_value = _DummyResponse("fresh-fallback")

        def _cached_client(provider, model=None, **kw):
            if provider == "anthropic":
                return (fresh_fb, "claude-haiku-4-5-20251001")
            return (primary_client, "gpt-5.5")

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client", side_effect=_cached_client), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(stale_fb, "claude-haiku-4-5-20251001", "anthropic")), \
             patch("agent.auxiliary_client._refresh_provider_credentials",
                   return_value=True) as mock_refresh:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "fresh-fallback"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_fb.chat.completions.create.call_count == 1
        assert fresh_fb.chat.completions.create.call_count == 1

    def test_unrefreshable_stale_candidate_is_skipped_to_next(self, monkeypatch):
        """Refresh fails (expired setup token) → candidate quarantined, chain
        walked again, next candidate serves the request."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        stale_fb = MagicMock()
        stale_fb.base_url = "https://api.anthropic.com"
        stale_fb.chat.completions.create.side_effect = _AuxAuth401("Invalid bearer token")

        healthy_fb = MagicMock()
        healthy_fb.base_url = "https://openrouter.ai/api/v1"
        healthy_fb.chat.completions.create.return_value = _DummyResponse("openrouter-serves")

        fb_walks = [
            (stale_fb, "claude-haiku-4-5-20251001", "anthropic"),
            (healthy_fb, "fallback-model", "openrouter"),
        ]

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   side_effect=fb_walks) as mock_fb, \
             patch("agent.auxiliary_client._refresh_provider_credentials",
                   return_value=False), \
             patch("agent.auxiliary_client._mark_provider_unhealthy") as mock_mark:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "openrouter-serves"
        assert mock_fb.call_count == 2
        assert mock_fb.call_args_list[1].kwargs.get("reason") == "stale fallback credential"
        mock_mark.assert_called_once_with("anthropic")
        assert stale_fb.chat.completions.create.call_count == 1
        assert healthy_fb.chat.completions.create.call_count == 1

    def test_non_auth_fallback_error_still_raises(self, monkeypatch):
        """A non-auth error from the fallback candidate propagates unchanged."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        broken_fb = MagicMock()
        broken_fb.base_url = "https://api.anthropic.com"
        broken_fb.chat.completions.create.side_effect = ValueError("malformed response")

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(broken_fb, "claude-haiku-4-5-20251001", "anthropic")):
            with pytest.raises(ValueError, match="malformed response"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )


class TestAuxiliaryFallbackLayering:
    """Explicit-provider users get layered fallback: configured_chain → main agent → warn."""

    def _make_payment_err(self):
        exc = Exception("Payment Required: insufficient credits")
        exc.status_code = 402
        return exc

    def test_empty_choices_with_output_text_is_recovered_before_fallback(self, monkeypatch):
        """Responses-style output_text should be used before provider fallback."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            output_text="recovered title",
            model="minimaxai/minimax-m3",
        )

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain") as mock_chain:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "recovered title"
        mock_chain.assert_not_called()

    def test_empty_choices_with_output_items_is_recovered_before_fallback(self, monkeypatch):
        """Responses-style output message items should be normalized for aux callers."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="part one"),
                        {"type": "text", "text": "part two"},
                    ],
                )
            ],
            model="minimaxai/minimax-m3",
        )

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain") as mock_chain:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "part one\npart two"
        mock_chain.assert_not_called()

    def test_invalid_empty_choices_response_triggers_fallback(self, monkeypatch):
        """HTTP-200 malformed chat completions should not abort aux fallback."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = MagicMock(choices=[])

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain, \
             patch("agent.auxiliary_client._try_main_agent_model_fallback") as mock_main:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "from fallback chain"
        mock_chain.assert_called_once_with(
            "title_generation",
            "nvidia",
            reason="invalid provider response",
        )
        mock_main.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_invalid_empty_choices_response_triggers_fallback(self, monkeypatch):
        """Async aux calls use the same malformed-response fallback path."""
        primary_client = MagicMock()
        primary_client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[]))

        fallback_client = MagicMock()
        async_fallback_client = MagicMock()
        async_fallback_client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[
            MagicMock(message=MagicMock(content="from async fallback"))
        ]))

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain, \
             patch("agent.auxiliary_client._to_async_client",
                   return_value=(async_fallback_client, "gpt-5.4-mini")):
            result = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "from async fallback"
        mock_chain.assert_called_once_with(
            "compression",
            "nvidia",
            reason="invalid provider response",
        )

    def test_auto_provider_uses_task_then_main_chain_before_builtin_chain(self, monkeypatch):
        """Auto aux call failures try per-task then top-level fallback before built-ins."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        main_chain_client = MagicMock()
        main_chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from main fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "qwen/qwen3.5-122b-a10b")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")) as mock_task_chain, \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(main_chain_client, "inclusionai/ring-2.6-1t:free", "openrouter")) as mock_main_chain, \
             patch("agent.auxiliary_client._try_payment_fallback") as mock_builtin_chain:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert main_chain_client.chat.completions.create.called
        mock_task_chain.assert_called_once_with(
            "title_generation", "auto", reason="payment error")
        mock_main_chain.assert_called_once_with(
            "title_generation", "auto", reason="payment error")
        mock_builtin_chain.assert_not_called()

    def test_explicit_provider_uses_configured_chain_first(self, monkeypatch, caplog):
        """When a user has fallback_chain configured, it's tried BEFORE the main agent model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        chain_client = MagicMock()
        chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from configured chain"))
        ])

        main_called = MagicMock()

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(chain_client, "gpt-4o-mini", "fallback_chain[0](openai)")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   side_effect=main_called):
            result = call_llm(
                task="vision",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert chain_client.chat.completions.create.called
        # Main agent fallback should NOT have been consulted — chain succeeded first
        main_called.assert_not_called()

    def test_explicit_provider_falls_back_to_main_when_chain_exhausted(self, monkeypatch):
        """If configured fallback_chain returns nothing, main agent model is tried next."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        main_client = MagicMock()
        main_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from main agent"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   return_value=(main_client, "claude-sonnet-4", "main-agent(openrouter)")):
            result = call_llm(
                task="vision",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert main_client.chat.completions.create.called

    def test_explicit_provider_rate_limit_triggers_fallback(self, monkeypatch):
        """429 rate-limit on an explicit provider must trigger fallback (not be ignored).

        Regression test for #52228: rate limits were excluded from
        ``is_capacity_error``, so explicit-provider auxiliary calls never
        fell back on 429 — only auto-provider calls did.
        """
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        rate_err = Exception("Rate limit exceeded, try again in 60 seconds")
        rate_err.status_code = 429
        primary_client.chat.completions.create.side_effect = rate_err

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai-codex", "gpt-5.5", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "deepseek-v4-pro", "fallback_chain[0](opencode-go)")) as mock_chain, \
             patch("agent.auxiliary_client._try_main_agent_model_fallback") as mock_main:
            result = call_llm(
                task="kanban_decomposer",
                messages=[{"role": "user", "content": "decompose this"}],
            )

        # Fallback chain MUST be tried for rate-limit on explicit provider
        mock_chain.assert_called()
        assert fallback_client.chat.completions.create.called
        # Main agent fallback should NOT be needed when chain succeeds
        mock_main.assert_not_called()


    def test_warning_emitted_when_all_fallbacks_exhausted(self, monkeypatch, caplog):
        """When chain AND main model both fail, a user-visible warning fires before re-raise."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   return_value=(None, None, "")), \
             caplog.at_level("WARNING", logger="agent.auxiliary_client"):
            with pytest.raises(Exception, match="Payment Required"):
                call_llm(
                    task="vision",
                    messages=[{"role": "user", "content": "hello"}],
                )

        assert any(
            "all fallbacks exhausted" in r.message for r in caplog.records
        ), f"Expected exhaustion warning, got: {[r.message for r in caplog.records]}"

    def test_explicit_provider_no_client_uses_configured_chain_before_error(self, monkeypatch):
        """Missing primary credentials should still honor auxiliary fallback_chain."""
        chain_client = MagicMock()
        chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from configured chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(chain_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert chain_client.chat.completions.create.called
        assert result.choices[0].message.content == "from configured chain"
        mock_chain.assert_called_once_with(
            "compression",
            "ollama-cloud",
            reason="provider unavailable",
        )

    def test_explicit_provider_no_client_without_chain_keeps_clear_error(self, monkeypatch):
        """No fallback configured: keep the existing actionable missing-key error."""
        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")) as mock_chain:
            with pytest.raises(RuntimeError, match="Provider 'ollama-cloud'.*no API key"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

        mock_chain.assert_called_once_with(
            "compression",
            "ollama-cloud",
            reason="provider unavailable",
        )

    def test_fallback_entry_openai_codex_uses_oauth_pool_without_inline_key(self):
        """Configured Codex fallback resolves through Hermes auth / credential pool."""
        from agent.auxiliary_client import _resolve_fallback_entry

        pool_entry = MagicMock()
        pool_entry.id = "codex-pool-1"
        pool_entry.runtime_api_key = "codex-oauth-token"
        pool_entry.access_token = "codex-oauth-token"
        pool_entry.runtime_base_url = "https://chatgpt.com/backend-api/codex"

        real_client = MagicMock()
        real_client.api_key = "codex-oauth-token"
        real_client.base_url = "https://chatgpt.com/backend-api/codex"

        with patch("agent.auxiliary_client._select_pool_entry",
                   return_value=(True, pool_entry)), \
             patch("agent.auxiliary_client._read_codex_access_token",
                   side_effect=AssertionError("should use pool token")), \
             patch("agent.auxiliary_client.OpenAI", return_value=real_client) as mock_openai:
            client, model = _resolve_fallback_entry({
                "provider": "openai-codex",
                "model": "gpt-5.4-mini",
            })

        assert client is not None
        assert model == "gpt-5.4-mini"
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "codex-oauth-token"


class TestTryMainAgentModelFallback:
    """_try_main_agent_model_fallback resolves the user's main provider+model as a safety net."""

    def test_returns_none_when_main_provider_is_auto(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="auto"), \
             patch("agent.auxiliary_client._read_main_model", return_value="some-model"):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is None and model is None and label == ""

    def test_returns_none_when_failed_provider_equals_main(self):
        """If the thing that failed IS the main model, no point retrying it."""
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"):
            client, model, label = _try_main_agent_model_fallback("openrouter", task="vision")
        assert client is None and label == ""

    def test_resolves_main_provider_client(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        fake_client = MagicMock()
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(fake_client, "anthropic/claude-sonnet-4")):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is fake_client
        assert model == "anthropic/claude-sonnet-4"
        assert label == "main-agent(openrouter)"

    def test_skips_when_main_provider_is_unhealthy(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=True):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is None


# ---------------------------------------------------------------------------
# Gate: _resolve_api_key_provider must skip anthropic when not configured
# ---------------------------------------------------------------------------


def test_resolve_api_key_provider_skips_unconfigured_anthropic(monkeypatch):
    """_resolve_api_key_provider must not try anthropic when user never configured it."""
    from collections import OrderedDict
    from hermes_cli.auth import ProviderConfig

    # Build a minimal registry with only "anthropic" so the loop is guaranteed
    # to reach it without being short-circuited by earlier providers.
    fake_registry = OrderedDict({
        "anthropic": ProviderConfig(
            id="anthropic",
            name="Anthropic",
            auth_type="api_key",
            inference_base_url="https://api.anthropic.com",
            api_key_env_vars=("ANTHROPIC_API_KEY",),
        ),
    })

    called = []

    def mock_try_anthropic():
        called.append("anthropic")
        return None, None

    monkeypatch.setattr("agent.auxiliary_client._try_anthropic", mock_try_anthropic)
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda pid: False,
    )

    from agent.auxiliary_client import _resolve_api_key_provider
    _resolve_api_key_provider()

    assert "anthropic" not in called, \
        "_try_anthropic() should not be called when anthropic is not explicitly configured"


# ---------------------------------------------------------------------------
# model="default" elimination (#7512)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _try_payment_fallback reason parameter (#7512 bug 3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _is_connection_error coverage
# ---------------------------------------------------------------------------


class TestTransientTransportRetry:
    """call_llm retries ONCE on the same provider for a transient transport
    blip before escalating to the fallback chain.

    Salvaged from PR #16587 (@ARegalado1). The original fixed only the
    context-compression caller; this lives in call_llm so every auxiliary
    task (compression, memory flush, title-gen, session-search, vision)
    gets the same same-target retry, and the gate reuses the canonical
    _is_connection_error detector.
    """

    def _patches(self, client):
        return (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "some-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "some-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task: resp,
            ),
        )

    def test_retries_streaming_close_once_same_provider(self):
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            Exception(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            ),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        # Same client called twice — no provider fallback needed.
        assert client.chat.completions.create.call_count == 2

    def test_retries_5xx_once_same_provider(self):
        class _Err503(Exception):
            status_code = 503

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_Err503("upstream"), {"ok": True}]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_does_not_retry_non_transient_400(self):
        class _Err400(Exception):
            status_code = 400

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = _Err400("bad request")
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3, pytest.raises(_Err400):
            call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        # Non-transient: single attempt, no same-target retry.
        assert client.chat.completions.create.call_count == 1

    def test_second_transient_failure_escalates_to_fallback(self):
        """Two transient failures in a row exhaust the same-target retry and
        fall through to the existing connection-error provider fallback."""
        primary = MagicMock()
        primary.base_url = "https://openrouter.ai/api/v1"
        primary.chat.completions.create.side_effect = Exception(
            "peer closed connection without sending complete message body"
        )

        fb_client = MagicMock()
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch("agent.auxiliary_client._transient_retry_count", return_value=1),
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(None, None, ""),
            ),
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(fb_client, "fb-model", "openai"),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        # Primary tried twice (initial + one same-target retry), then fallback.
        assert primary.chat.completions.create.call_count == 2
        assert fb_client.chat.completions.create.call_count == 1

    def test_compression_skips_same_provider_retry_on_timeout(self):
        """A timeout on the critical compression path must NOT retry the same
        provider (that doubles the user-visible stall, issue #54465) — it
        falls straight through to the fallback chain instead.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        primary = MagicMock()
        primary.base_url = "https://openrouter.ai/api/v1"
        primary.chat.completions.create.side_effect = _Timeout("Request timed out.")

        fb_client = MagicMock()
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(None, None, ""),
            ),
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(fb_client, "fb-model", "openai"),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        # Primary tried ONCE only — no same-provider timeout retry — then fallback.
        assert primary.chat.completions.create.call_count == 1
        assert fb_client.chat.completions.create.call_count == 1

    def test_non_compression_still_retries_same_provider_on_timeout(self):
        """The timeout skip is scoped to compression only; other auxiliary
        tasks keep the single same-provider transient retry.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            _Timeout("Request timed out."),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_compression_still_retries_streaming_close_on_timeout_path(self):
        """A fast streaming-close (not a full-budget timeout) still retries
        same-provider even for compression — only timeouts are skipped.
        """
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            Exception(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            ),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2


class TestAuxClientNoSdkRetries:
    """Auxiliary OpenAI clients are constructed with SDK-internal retries
    disabled so Hermes owns the retry/timeout budget (issue #54465). The SDK
    default (max_retries=2 → 3 attempts) silently triples the effective wall
    time of every aux call against a slow/hung endpoint.
    """

    def test_sync_client_disables_sdk_retries(self):
        from agent import auxiliary_client as ac
        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(ac, "OpenAI", _FakeOpenAI), \
             patch.object(ac, "_openai_http_client_kwargs", return_value={}):
            ac._create_openai_client(api_key="k", base_url="https://x/v1")
        assert captured.get("max_retries") == 0

    def test_explicit_max_retries_override_wins(self):
        from agent import auxiliary_client as ac
        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(ac, "OpenAI", _FakeOpenAI), \
             patch.object(ac, "_openai_http_client_kwargs", return_value={}):
            ac._create_openai_client(api_key="k", base_url="https://x/v1", max_retries=5)
        assert captured.get("max_retries") == 5


class TestIsTimeoutError:
    """_is_timeout_error distinguishes a full-budget timeout from a fast
    connection drop."""

    def test_timed_out_string(self):
        from agent.auxiliary_client import _is_timeout_error
        assert _is_timeout_error(Exception("Request timed out.")) is True

    def test_timeout_typename(self):
        from agent.auxiliary_client import _is_timeout_error

        class ReadTimeout(Exception):
            pass

        assert _is_timeout_error(ReadTimeout("slow")) is True

    def test_streaming_close_is_not_timeout(self):
        from agent.auxiliary_client import _is_timeout_error
        err = Exception("peer closed connection (incomplete chunked read)")
        assert _is_timeout_error(err) is False

    def test_5xx_is_not_timeout(self):
        from agent.auxiliary_client import _is_timeout_error

        class _Err503(Exception):
            status_code = 503

        assert _is_timeout_error(_Err503("upstream")) is False


class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True

    def test_timeout(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Request timed out.")
        assert _is_connection_error(err) is True

    def test_dns_failure(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Name or service not known")
        assert _is_connection_error(err) is True

    def test_normal_api_error_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False

    def test_500_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Internal Server Error")
        err.status_code = 500
        assert _is_connection_error(err) is False


class TestKimiTemperatureOmitted:
    """Kimi/Moonshot models should have temperature OMITTED from API kwargs.

    The Kimi gateway selects the correct temperature server-side based on the
    active mode (thinking → 1.0, non-thinking → 0.6).  Sending any temperature
    value conflicts with gateway-managed defaults.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "kimi-for-coding",
            "kimi-k2.5",
            "kimi-k2.6",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
            "kimi-k2-instruct",
            "kimi-k2-instruct-0905",
            "moonshotai/kimi-k2.5",
            "moonshotai/Kimi-K2-Thinking",
            "moonshotai/Kimi-K2-Instruct",
        ],
    )
    def test_kimi_models_omit_temperature(self, model):
        """No kimi model should have a temperature key in kwargs."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert "temperature" not in kwargs

    def test_kimi_for_coding_no_temperature_when_none(self):
        """When caller passes temperature=None, still no temperature key."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        assert "temperature" not in kwargs

    def test_sync_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_async_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "gpt-5.4",
            "deepseek-chat",
        ],
    )
    def test_non_kimi_models_preserve_temperature(self, model):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="openrouter",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert kwargs["temperature"] == 0.3

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.moonshot.ai/v1",
            "https://api.moonshot.cn/v1",
            "https://api.kimi.com/coding/v1",
        ],
    )
    def test_kimi_k2_5_omits_temperature_regardless_of_endpoint(self, base_url):
        """Temperature is omitted regardless of which Kimi endpoint is used."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            base_url=base_url,
        )

        assert "temperature" not in kwargs


# ---------------------------------------------------------------------------
# async_call_llm payment / connection fallback (#7512 bug 2)
# ---------------------------------------------------------------------------


class TestStaleBaseUrlWarning:
    """_resolve_auto() warns when OPENAI_BASE_URL conflicts with config provider (#5161)."""

    def test_warns_when_openai_base_url_set_with_named_provider(self, monkeypatch, caplog):
        """Warning fires when OPENAI_BASE_URL is set but provider is a named provider."""
        import agent.auxiliary_client as mod
        # Reset the module-level flag so the warning fires
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Expected a warning about stale OPENAI_BASE_URL"
        assert mod._stale_base_url_warned is True


class TestAuxiliaryTaskExtraBody:
    def test_sync_call_merges_task_extra_body_from_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {
                        "enable_thinking": False,
                        "reasoning": {"effort": "none"},
                    }
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"metadata": {"source": "test"}},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is False
        assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}
        assert kwargs["extra_body"]["metadata"] == {"source": "test"}

    @pytest.mark.asyncio
    async def test_async_call_explicit_extra_body_overrides_task_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {"enable_thinking": False}
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"enable_thinking": True},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is True

    def test_no_warning_when_provider_is_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom' — OPENAI_BASE_URL is expected."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "test-key", None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai, \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            mock_openai.return_value = MagicMock()
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom'"

    def test_no_warning_when_provider_is_named_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom:myname' — base_url comes from config."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom:ollama-local"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(MagicMock(), "llama3")), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom:*'"

    def test_no_warning_when_openai_base_url_not_set(self, monkeypatch, caplog):
        """No warning when OPENAI_BASE_URL is absent."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when OPENAI_BASE_URL is not set"

# ---------------------------------------------------------------------------
# Anthropic-compatible image block conversion
# ---------------------------------------------------------------------------

class TestAnthropicCompatImageConversion:
    """Tests for _is_anthropic_compat_endpoint and _convert_openai_images_to_anthropic."""

    def test_known_providers_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("minimax", "")
        assert _is_anthropic_compat_endpoint("minimax-cn", "")

    def test_openrouter_not_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert not _is_anthropic_compat_endpoint("openrouter", "")
        assert not _is_anthropic_compat_endpoint("anthropic", "")

    def test_url_based_detection(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("custom", "https://api.minimax.io/anthropic")
        assert _is_anthropic_compat_endpoint("custom", "https://example.com/anthropic/v1")
        assert not _is_anthropic_compat_endpoint("custom", "https://api.openai.com/v1")

    def test_base64_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "iVBOR="

    def test_url_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.jpg"

    def test_text_only_messages_unchanged(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0] is messages[0]  # same object, not copied

    def test_jpeg_media_type_parsed(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["source"]["media_type"] == "image/jpeg"

    def test_base64_video_converted_to_video_block(self):
        # MiniMax M3's Anthropic-compatible endpoint expects type="video"
        # (not OpenAI's "video_url", not "input_video").
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What happens in this clip?"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        vid_block = result[0]["content"][1]
        assert vid_block["type"] == "video"
        assert vid_block["source"]["type"] == "base64"
        assert vid_block["source"]["media_type"] == "video/mp4"
        assert vid_block["source"]["data"] == "AAAA"

    def test_video_media_type_parsed_from_data_uri(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/quicktime;base64,QQ=="}}
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["source"]["media_type"] == "video/quicktime"

    def test_url_video_converted_to_video_block(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        vid_block = result[0]["content"][0]
        assert vid_block["type"] == "video"
        assert vid_block["source"] == {"type": "url", "url": "https://example.com/clip.mp4"}

    def test_mixed_image_and_video_both_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["type"] == "image"
        assert result[0]["content"][1]["type"] == "video"


class _AuxAuth401(Exception):
    status_code = 401

    def __init__(self, message="Provided authentication token is expired"):
        super().__init__(message)


class _DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [MagicMock(message=MagicMock(content=text))]


class _FailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("sync-ok")


class _AsyncFailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("async-ok")


class TestAuxiliaryAuthRefreshRetry:
    def test_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _FailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-sync")

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-sync"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_call_llm_refreshes_codex_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("stale codex token")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-non-vision")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-non-vision"
        mock_refresh.assert_called_once_with("openai-codex")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_copilot_when_auto_routes_to_copilot_on_401(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.githubcopilot.com"
        stale_client.chat.completions.create.side_effect = _AuxAuth401(
            "IDE token expired: unauthorized: token expired"
        )

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.githubcopilot.com"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-auto-copilot")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.5"), (fresh_client, "gpt-5.5")]) as mock_get_client,
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
            patch("agent.auxiliary_client._evict_cached_clients") as mock_evict,
        ):
            resp = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                main_runtime={"provider": "copilot", "model": "gpt-5.5"},
            )

        assert resp.choices[0].message.content == "fresh-auto-copilot"
        mock_refresh.assert_called_once_with("copilot")
        mock_evict.assert_called_once_with("auto")
        assert mock_get_client.call_args_list[0].args[0] == "auto"
        assert mock_get_client.call_args_list[1].args[0] == "copilot"
        assert mock_get_client.call_args_list[1].args[1] == "gpt-5.5"
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_codex_when_auto_routes_to_codex_on_401(self):
        # Preflight compression's exact failure (#23670): provider auto →
        # Codex OAuth backend 401s → before the fix, no refresh was attempted
        # because resolved_provider stayed "auto".
        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("User not found.")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-auto-codex")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.5"), (fresh_client, "gpt-5.5")]) as mock_get_client,
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
            patch("agent.auxiliary_client._evict_cached_clients") as mock_evict,
        ):
            resp = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                main_runtime={"provider": "openai-codex", "model": "gpt-5.5"},
            )

        assert resp.choices[0].message.content == "fresh-auto-codex"
        mock_refresh.assert_called_once_with("openai-codex")
        mock_evict.assert_called_once_with("auto")
        assert mock_get_client.call_args_list[1].args[0] == "openai-codex"
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("anthropic token expired")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-anthropic")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _AsyncFailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async"))

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_refresh_provider_credentials_force_refreshes_anthropic_oauth_and_evicts_cache(self, monkeypatch):
        stale_client = MagicMock()
        cache_key = ("anthropic", False, None, None, None)

        monkeypatch.setenv("ANTHROPIC_TOKEN", "")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        with (
            patch("agent.auxiliary_client._client_cache", {cache_key: (stale_client, "claude-haiku-4-5-20251001", None)}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value={
                "accessToken": "expired-token",
                "refreshToken": "refresh-token",
                "expiresAt": 0,
            }),
            patch("agent.anthropic_adapter.refresh_anthropic_oauth_pure", return_value={
                "access_token": "fresh-token",
                "refresh_token": "refresh-token-2",
                "expires_at_ms": 9999999999999,
            }) as mock_refresh_oauth,
            patch("agent.anthropic_adapter._write_claude_code_credentials") as mock_write,
        ):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("anthropic") is True

        mock_refresh_oauth.assert_called_once_with("refresh-token", use_json=False)
        mock_write.assert_called_once_with("fresh-token", "refresh-token-2", 9999999999999)
        stale_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create = AsyncMock(side_effect=_AuxAuth401("anthropic token expired"))

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async-anthropic"))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_client.chat.completions.create.await_count == 1


class TestAuxiliaryPoolRotationRetry:
    def test_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = [rate_err, rate_err]

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("rotated-sync")

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-sync"
        assert stale_client.chat.completions.create.call_count == 2
        assert fresh_client.chat.completions.create.call_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create = AsyncMock(side_effect=[rate_err, rate_err])

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("rotated-async"))

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-async"
        assert stale_client.chat.completions.create.await_count == 2
        assert fresh_client.chat.completions.create.await_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()


class TestCodexAdapterReasoningTranslation:
    """Verify _CodexCompletionsAdapter translates extra_body.reasoning
    into the Responses API's top-level reasoning + include fields, matching
    agent/transports/codex.py::build_kwargs() behavior.

    Regression for user feedback (Apr 26): auxiliary callers that configure
    reasoning via auxiliary.<task>.extra_body.reasoning had that config
    silently dropped because the adapter only forwarded messages/model/tools.
    """

    @staticmethod
    def _build_adapter():
        """Build a _CodexCompletionsAdapter with a mocked responses.create()."""
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        # The event-driven path consumes ``responses.create(stream=True)`` as a
        # raw iterable of SSE events.  Emit a minimal stream containing one
        # ``response.output_item.done`` (message) and a ``response.completed``
        # terminal frame.
        message_item = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed",
                    id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.3-codex")
        return adapter, captured_kwargs

    def test_reasoning_effort_medium_translated_to_top_level(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "medium"}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_minimal_clamped_to_low(self):
        """Codex backend rejects 'minimal'; adapter clamps to 'low' per main transport."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "minimal"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_low_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "low"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}

    def test_reasoning_effort_high_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "high"}},
        )
        assert captured.get("reasoning") == {"effort": "high", "summary": "auto"}

    def test_reasoning_disabled_omits_reasoning_and_include(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"enabled": False}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_reasoning_default_effort_when_only_enabled_flag(self):
        """extra_body={"reasoning": {}} (truthy enabled by omission) → default 'medium'."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_no_extra_body_means_no_reasoning_keys(self):
        """Baseline: without extra_body, no reasoning/include is sent (preserves
        current behavior for callers that don't opt in)."""
        adapter, captured = self._build_adapter()
        adapter.create(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_extra_body_without_reasoning_key_is_noop(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"metadata": {"source": "test"}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_non_dict_reasoning_value_is_ignored_gracefully(self):
        """Defensive: if a caller accidentally passes a string/None, we
        silently skip instead of crashing inside the adapter."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": "medium"},  # wrong shape — must not crash
        )
        assert "reasoning" not in captured

    def test_reasoning_effort_null_falls_back_to_medium(self):
        """Parity with agent/transports/codex.py::build_kwargs() — falsy
        ``effort`` (None / empty / 0) keeps the default ``medium`` instead
        of being forwarded to Codex.  Codex rejects ``{"effort": null}``
        with HTTP 400 (Invalid value for parameter `reasoning.effort`)."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": None}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_empty_string_falls_back_to_medium(self):
        """Empty-string effort (e.g. ``effort: ""`` in YAML) is falsy in
        the main-agent path's truthy check; mirror that here so the same
        config produces the same result."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": ""}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_zero_falls_back_to_medium(self):
        """Numeric ``0`` is also falsy — the docstring lists it explicitly,
        so cover the contract.  Codex would reject ``{"effort": 0}`` the
        same way it rejects ``null``."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": 0}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]


class TestCodexAdapterPromptCacheKey:
    """_CodexCompletionsAdapter emits a stable content-addressed prompt_cache_key
    on the Codex/Responses aux path, matching the main transport
    (agent/transports/codex.py). Regression for issue #53735: MoA acting-
    aggregator and other auxiliary Responses calls stayed cache-cold because
    the adapter never set prompt_cache_key.
    """

    @staticmethod
    def _build_adapter(base_url="https://chatgpt.com/backend-api/codex"):
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        message_item = SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed", id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.base_url = base_url
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.5")
        return adapter, captured_kwargs

    def test_cache_key_set_and_prefixed(self):
        adapter, captured = self._build_adapter()
        adapter.create(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ])
        key = captured.get("prompt_cache_key")
        assert isinstance(key, str) and key.startswith("pck_")

    def test_cache_key_stable_across_identical_prefix(self):
        """Same instructions + tools → same key (content-addressed, not per-call)."""
        a1, c1 = self._build_adapter()
        a1.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "first"},
        ])
        a2, c2 = self._build_adapter()
        a2.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "second — different user turn"},
        ])
        # User-turn content differs but the static prefix (instructions) matches,
        # so the routing key is identical → same warm cache bucket.
        assert c1["prompt_cache_key"] == c2["prompt_cache_key"]

    def test_cache_key_differs_on_different_instructions(self):
        a1, c1 = self._build_adapter()
        a1.create(messages=[{"role": "system", "content": "SYS-A"}, {"role": "user", "content": "x"}])
        a2, c2 = self._build_adapter()
        a2.create(messages=[{"role": "system", "content": "SYS-B"}, {"role": "user", "content": "x"}])
        assert c1["prompt_cache_key"] != c2["prompt_cache_key"]

    def test_cache_key_skipped_for_xai_host(self):
        """xAI Responses takes the key in extra_body, not top-level — skip here."""
        adapter, captured = self._build_adapter(base_url="https://api.x.ai/v1")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_key" not in captured

    def test_cache_key_skipped_for_github_copilot_host(self):
        """GitHub/Copilot Responses opts out of cache-key routing entirely."""
        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_key" not in captured


class TestCodexAdapterGithubResponsesMessageIdDrop:
    """_CodexCompletionsAdapter must drop codex_message_items ``id`` when
    talking to Copilot (githubcopilot.com), independent of the main
    transport's build_kwargs path. Auxiliary calls (context compression,
    flush_memories, MoA aggregation) route through this adapter instead of
    agent/transports/codex.py, so they need the same #32716 guard applied
    separately — Copilot binds replayed ids to a backend "connection" that
    doesn't survive credential rotation/gateway restarts, and rejects a
    stale id with HTTP 401 regardless of its length.
    """

    @staticmethod
    def _build_adapter(base_url):
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        message_item = SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed", id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.base_url = base_url
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.5")
        return adapter, captured_kwargs

    @staticmethod
    def _replay_messages():
        return [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "content": "pong",
                "codex_message_items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [{"type": "output_text", "text": "pong"}],
                        "id": "msg_short_but_connection_scoped",
                        "phase": "final_answer",
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

    def test_drops_message_id_for_github_copilot_host(self):
        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=self._replay_messages())
        message_item = next(
            item for item in captured["input"] if item.get("type") == "message"
        )
        assert "id" not in message_item
        assert message_item["phase"] == "final_answer"
        assert message_item["status"] == "in_progress"
        assert message_item["content"] == [{"type": "output_text", "text": "pong"}]

    def test_keeps_message_id_for_codex_backend_host(self):
        adapter, captured = self._build_adapter(
            base_url="https://chatgpt.com/backend-api/codex"
        )
        adapter.create(messages=self._replay_messages())
        message_item = next(
            item for item in captured["input"] if item.get("type") == "message"
        )
        assert message_item["id"] == "msg_short_but_connection_scoped"


class TestVisionAutoSkipsKimiCoding:
    """_resolve_auto vision branch skips providers that have no vision on
    their main endpoint (e.g. Kimi Coding Plan /coding) and falls through
    to the aggregator chain instead of handing back a client that will 404
    on every request (#17076).
    """

    def test_kimi_coding_skipped_falls_through_to_openrouter(self, monkeypatch):
        """kimi-coding as main + vision auto → OpenRouter (not kimi)."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        # Guard: if the skip doesn't fire, _resolve_strict_vision_backend
        # and resolve_provider_client both would try kimi-coding — detect
        # either via the main-provider call and fail loud.
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding "
            "on the vision auto path"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )

        def fake_strict(provider, model=None):
            if provider == "openrouter":
                return fake_or_client, "google/gemini-3-flash-preview"
            if provider == "nous":
                return None, None
            raise AssertionError(
                f"strict vision backend should not be called for {provider!r} "
                "when main provider is kimi-coding"
            )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            fake_strict,
        )

        provider, client, model = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client
        assert model == "google/gemini-3-flash-preview"

    def test_kimi_coding_cn_skipped_too(self, monkeypatch):
        """Same skip applies to the CN variant."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding-cn",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding-cn"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            lambda p, m=None: (fake_or_client, "gemini")
            if p == "openrouter"
            else (None, None),
        )

        provider, client, _ = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client

    def test_explicit_override_to_kimi_coding_still_honored(self, monkeypatch):
        """When a user *explicitly* requests kimi-coding for vision (e.g.
        they know what they're doing, or are running a future build that
        adds image_in capability to Kimi Code), the explicit path still
        routes to kimi-coding — only the auto branch applies the skip.
        """
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "openrouter",
        )
        fake_kimi_client = MagicMock(name="kimi_client")
        gcc_mock = MagicMock(return_value=(fake_kimi_client, "kimi-code"))
        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client", gcc_mock,
        )

        provider, client, model = resolve_vision_provider_client(
            provider="kimi-coding",
        )
        assert provider == "kimi-coding"
        assert client is fake_kimi_client
        gcc_mock.assert_called_once()

    def test_skip_set_covers_exactly_known_entries(self):
        """Guard against accidental widening of the skip list."""
        from agent.auxiliary_client import _PROVIDERS_WITHOUT_VISION
        assert _PROVIDERS_WITHOUT_VISION == frozenset({
            "kimi-coding",
            "kimi-coding-cn",
        })


class TestCodexAuxiliaryAdapterTimeout:
    def test_forwards_timeout_to_responses_create(self):
        message_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="summary")],
        )
        events = [
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed", id="r1", usage=None,
            )),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        response = adapter.create(
            messages=[{"role": "user", "content": "summarize this"}],
            timeout=12.5,
        )

        assert fake_client.responses.kwargs["timeout"] == 12.5
        assert fake_client.responses.kwargs["stream"] is True
        assert response.choices[0].message.content == "summary"

    def test_enforces_total_timeout_while_stream_keeps_emitting_events(self):
        class _SlowAliveCreateStream:
            def __iter__(self):
                for _ in range(5):
                    time.sleep(0.03)
                    yield SimpleNamespace(type="response.in_progress")

            def close(self): pass

        class FakeResponses:
            def create(self, **kwargs):
                return _SlowAliveCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses(), close=lambda: None)
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            adapter.create(
                messages=[{"role": "user", "content": "summarize this"}],
                timeout=0.05,
            )

        assert time.monotonic() - started < 0.14


class TestCodexAuxiliaryToolMessageConversion:
    """Regression for issue #5709.

    The auxiliary Codex adapter used to maintain its own chat->Responses
    conversion loop that forwarded every non-system message's ``role``
    verbatim into Responses ``input[]``. When ``flush_memories()`` /
    compression replayed real session history containing assistant
    ``tool_calls`` and ``role="tool"`` results, the tool messages leaked
    into the request and the Responses API rejected them with
    ``HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant',
    'system', 'developer', and 'user'.``

    The fix routes the auxiliary path through the SAME shared converter the
    main agent transport uses (``_chat_messages_to_responses_input``), so
    no Responses request ever includes a raw ``role="tool"`` input item.
    """

    def _capture_input(self, messages):
        from agent.auxiliary_client import _CodexCompletionsAdapter

        class _FakeCreateStream:
            def __iter__(self):
                return iter([
                    SimpleNamespace(type="response.created"),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text="ok")],
                        ),
                    ),
                    SimpleNamespace(type="response.completed", response=SimpleNamespace(
                        status="completed", id="r1", usage=None,
                    )),
                ])

            def close(self):
                pass

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")
        adapter.create(messages=messages, model="gpt-5.5")
        return fake_client.responses.kwargs

    def test_tool_history_never_leaks_role_tool(self):
        messages = [
            {"role": "system", "content": "You are a memory summarizer."},
            {"role": "user", "content": "What files did I touch?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "search_files", "arguments": '{"pattern":"foo"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Found 3 matches"},
            {"role": "assistant", "content": "You touched bar.py."},
        ]
        kwargs = self._capture_input(messages)
        input_items = kwargs["input"]

        # No raw role="tool" item reaches the Responses API (the 400 trigger).
        assert not any(it.get("role") == "tool" for it in input_items)

        # Assistant tool call -> function_call item with a call_id.
        function_calls = [it for it in input_items if it.get("type") == "function_call"]
        assert function_calls, "assistant tool_call must become a function_call item"
        assert function_calls[0]["call_id"] == "call_abc123"
        assert function_calls[0]["name"] == "search_files"

        # Tool result -> function_call_output with the matching call_id.
        outputs = [it for it in input_items if it.get("type") == "function_call_output"]
        assert outputs, "tool result must become a function_call_output item"
        assert outputs[0]["call_id"] == "call_abc123"

        # System message is hoisted to instructions, not left in input[].
        assert kwargs["instructions"] == "You are a memory summarizer."
        assert not any(it.get("role") == "system" for it in input_items)

    def test_plain_text_history_still_works(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        kwargs = self._capture_input(messages)
        input_items = kwargs["input"]
        roles = [it.get("role") for it in input_items]
        assert "user" in roles and "assistant" in roles
        assert not any(it.get("role") == "tool" for it in input_items)
        assert kwargs["instructions"] == "sys"


class TestCodexAuxiliaryAdapterNullOutputRecovery:
    def test_recovers_output_item_when_terminal_event_has_null_output(self):
        """Regression for #11179 in auxiliary calls.

        The wire shape that broke the SDK is ``response.completed`` with
        ``response.output = null``.  The event-driven path is structurally
        immune because it reconstructs from ``response.output_item.done``
        events and never reads the terminal event's ``output`` field for
        content.  Assert the auxiliary path returns the streamed item even
        when the terminal frame's output is ``null``.
        """
        output_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="aux survived")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=output_item),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed",
                id="resp_null_output",
                # This is the field the SDK helper would have iterated and crashed on:
                output=None,
                usage=None,
            )),
        ]

        class _NullOutputCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        class FakeResponses:
            def create(self, **kwargs):
                return _NullOutputCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        response = adapter.create(messages=[{"role": "user", "content": "summarize"}])

        assert response.choices[0].message.content == "aux survived"

    def test_handles_final_output_is_none_after_consumer(self):
        """Regression for #33368 — defense against ``final.output`` being ``None``.

        The event-driven consumer always sets ``final.output`` to a list, so this
        shape can't come from our own path. But a mocked client / compatibility
        shim that returns a typed Response with ``output=None`` directly (or a
        future code path that wraps a different consumer) would crash on
        ``for item in getattr(final, "output", [])`` because ``getattr`` returns
        ``None`` (not the default) when the attribute exists but is ``None``.
        Coerce with ``or []`` to handle this defensively.
        """
        # Stream that returns no items but a terminal with output=None.
        # The consumer assembles an empty list. We then mock the consumer's
        # return to simulate a third-party path that returns final.output=None.
        empty_events = [
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed", id="r", output=None, usage=None,
            )),
        ]

        class _Stream:
            def __iter__(self): return iter(empty_events)
            def close(self): pass

        # Monkey-patch the consumer to return a final whose .output is None
        # (mimics third-party shim behavior the defensive guard protects against).
        from agent import codex_runtime
        original_consume = codex_runtime._consume_codex_event_stream

        def _consume_returning_none_output(*args, **kwargs):
            return SimpleNamespace(
                output=None,  # the defensive guard target
                output_text="",
                usage=None,
                status="completed",
                id="r",
                model=kwargs.get("model"),
                incomplete_details=None,
                error=None,
            )

        codex_runtime._consume_codex_event_stream = _consume_returning_none_output
        try:
            class FakeResponses:
                def create(self, **kwargs):
                    return _Stream()

            fake_client = SimpleNamespace(responses=FakeResponses())
            adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

            # Should not raise TypeError: 'NoneType' object is not iterable
            response = adapter.create(messages=[{"role": "user", "content": "x"}])
            assert response.choices[0].message.content is None
            assert response.choices[0].finish_reason == "stop"
        finally:
            codex_runtime._consume_codex_event_stream = original_consume


# ---------------------------------------------------------------------------
# Issue #23432 — auxiliary timeout poisons cached client; later aux calls fail
# ---------------------------------------------------------------------------

class TestAuxiliaryClientPoisonedCacheEviction:
    """Connection/timeout errors must evict the cached aux client.

    Otherwise the next auxiliary call (compression retry, memory flush,
    background review) reuses the closed httpx transport and fails with
    ``Connection error`` even though the main provider route is healthy.
    See https://github.com/NousResearch/hermes-agent/issues/23432.
    """

    def test_evict_cached_client_instance_drops_direct_match(self):
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
        )

        target = MagicMock(name="target_client")
        other = MagicMock(name="other_client")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openrouter", False, None, None, None)] = (target, "x", None)
            _client_cache[("anthropic", False, None, None, None)] = (other, "y", None)
        try:
            assert _evict_cached_client_instance(target) is True
            assert ("openrouter", False, None, None, None) not in _client_cache
            assert ("anthropic", False, None, None, None) in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_walks_codex_wrapper(self):
        """Closing the underlying OpenAI client must evict the Codex shim."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (wrapper, "gpt-5.5", None)
        try:
            # Eviction by the inner OpenAI client must remove the wrapper entry.
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_handles_none_and_misses(self):
        from agent.auxiliary_client import _evict_cached_client_instance

        assert _evict_cached_client_instance(None) is False
        assert _evict_cached_client_instance(MagicMock()) is False

    def test_evict_cached_client_instance_walks_async_wrapper(self):
        """async_mode is part of the cache key so sync and async share the same
        underlying OpenAI client across two distinct cache entries. A single
        timeout that closes the leaf must evict BOTH — otherwise the async
        entry survives, keeps reusing the dead transport, and every async
        aux call (compression, vision, session_search) fails fast with
        'Connection error' until gateway restart even while the sync route
        recovers.

        Regression for the async-side gap left by #23482, which fixed the
        sync wrapper's _real_client walk but missed the async wrappers.
        """
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient, AsyncCodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        sync_wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        async_wrapper = AsyncCodexAuxiliaryClient(sync_wrapper)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (sync_wrapper, "gpt-5.5", None)
            _client_cache[("openai-codex", True, None, None, None)] = (async_wrapper, "gpt-5.5", None)
        try:
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
            assert ("openai-codex", True, None, None, None) not in _client_cache, (
                "async cache entry survived eviction — wrapper is missing _real_client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_codex_timeout_evicts_cached_wrapper(self):
        """The timeout closer evicts the cache entry that wraps the closed client."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock,
            _CodexCompletionsAdapter, CodexAuxiliaryClient,
        )

        class _SlowAliveCreateStream:
            def __iter__(self):
                for _ in range(20):
                    time.sleep(0.01)
                    yield SimpleNamespace(type="response.in_progress")

            def close(self): pass

        closed = {"flag": False}

        class FakeClient:
            def __init__(self):
                self.responses = SimpleNamespace(create=lambda **k: _SlowAliveCreateStream())
                self.api_key = "k"
                self.base_url = "https://chatgpt.com/backend-api/codex"

            def close(self):
                closed["flag"] = True

        fake_real = FakeClient()
        wrapper = CodexAuxiliaryClient(fake_real, "gpt-5.5")
        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (wrapper, "gpt-5.5", None)
        try:
            adapter = _CodexCompletionsAdapter(fake_real, "gpt-5.5")
            with pytest.raises(TimeoutError):
                adapter.create(
                    messages=[{"role": "user", "content": "x"}],
                    timeout=0.05,
                )
            assert closed["flag"] is True, "timeout closer must close inner client"
            assert cache_key not in _client_cache, (
                "timeout closer must evict cache entry that wraps the closed client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        """Connection error on an explicit provider must drop the cached client.

        Reporter scenario: ``auxiliary.compression.provider: main`` (resolves
        to ``openai-codex``).  After #26803, capacity errors (payment/quota/
        connection) DO trigger fallback even on explicit providers — so we
        also stub ``_try_payment_fallback`` to ``(None, None, "")`` so the
        connection error re-raises after eviction instead of escaping into
        a real network call.  The contract under test is cache eviction,
        not the fallback gate.
        """
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create.side_effect = ConnectionError("transport closed")

        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ), patch(
                "agent.auxiliary_client._try_payment_fallback",
                return_value=(None, None, ""),
            ):
                with pytest.raises(ConnectionError):
                    call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache, (
                "connection error must evict cached client so the next call rebuilds"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    @pytest.mark.asyncio
    async def test_async_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_async_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create = AsyncMock(side_effect=ConnectionError("transport closed"))

        cache_key = ("openai-codex", True, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ), patch(
                "agent.auxiliary_client._try_payment_fallback",
                return_value=(None, None, ""),
            ):
                with pytest.raises(ConnectionError):
                    await async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()


# ---------------------------------------------------------------------------
# _build_call_kwargs — tool dedup at API boundary
# ---------------------------------------------------------------------------

class TestBuildCallKwargsToolDedup:
    """_build_call_kwargs must deduplicate tool names before passing to API.

    Providers like Google Vertex, Azure, and Bedrock reject requests with
    duplicate tool names (HTTP 400).  This guard converts a hard failure into
    a warning log so agent turns succeed even if an upstream injection path
    regresses.  See: https://github.com/NousResearch/hermes-agent/issues/18478
    """

    def _make_tool(self, name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_unique_tools_pass_through_unchanged(self):
        tools = [self._make_tool("alpha"), self._make_tool("beta")]
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=tools,
        )
        assert len(kwargs["tools"]) == 2
        names = [t["function"]["name"] for t in kwargs["tools"]]
        assert names == ["alpha", "beta"]

    def test_duplicate_tool_names_are_deduplicated(self):
        """RED test — must fail until dedup guard is added."""
        tools = [
            self._make_tool("lcm_grep"),
            self._make_tool("lcm_describe"),
            self._make_tool("lcm_grep"),  # duplicate
            self._make_tool("lcm_expand"),
            self._make_tool("lcm_describe"),  # duplicate
        ]
        kwargs = _build_call_kwargs(
            provider="google", model="gemini-2.5-pro", messages=[], tools=tools,
        )
        result_tools = kwargs["tools"]
        names = [t["function"]["name"] for t in result_tools]
        # Must be deduplicated — no repeated names
        assert len(names) == len(set(names)), (
            f"Duplicate tool names found: {names}"
        )
        assert len(result_tools) == 3  # lcm_grep, lcm_describe, lcm_expand

    def test_empty_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=[],
        )
        assert kwargs.get("tools") == [] or "tools" not in kwargs

    def test_none_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=None,
        )
        assert "tools" not in kwargs


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean."""
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestNvidiaBillingHeaders:
    """NVIDIA NIM billing-origin headers are scoped to NVIDIA cloud."""

    def test_resolve_provider_client_cloud_adds_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs["default_headers"]
        assert headers["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"

    def test_resolve_provider_client_local_nim_skips_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-local-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        assert "X-BILLING-INVOKE-ORIGIN" not in headers


class TestOpenRouterExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_openrouter()."""

    def test_resolve_provider_client_passes_explicit_api_key_to_openrouter(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called with explicit_api_key for OpenRouter,
        the explicit key should be passed to the OpenAI client instead of falling back
        to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should NOT be used when explicit_api_key is provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key="explicit-pool-key",
            )

            # Verify a client was created
            assert client is not None
            # Verify the explicit key was used, not the env var fallback
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "explicit-pool-key", (
                f"Expected explicit_api_key to be passed, got: {call_kwargs['api_key']}"
            )
            assert call_kwargs["api_key"] != "env-fallback-key", (
                "Should NOT fall back to OPENROUTER_API_KEY when explicit_api_key is provided"
            )

    def test_resolve_provider_client_without_explicit_api_key_falls_back_to_env(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called WITHOUT explicit_api_key for OpenRouter,
        it should fall back to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should be used when explicit_api_key is NOT provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key=None,
            )

            # Verify a client was created
            assert client is not None
            # Verify the env var fallback was used
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "env-fallback-key", (
                f"Expected env fallback key to be used when explicit_api_key is None, got: {call_kwargs['api_key']}"
            )


def test_pool_runtime_base_url_uses_nous_env_override(monkeypatch):
    entry = SimpleNamespace(
        provider="nous",
        runtime_base_url="https://inference-api.nousresearch.com/v1",
        inference_base_url="https://inference-api.nousresearch.com/v1",
        base_url="https://inference-api.nousresearch.com/v1",
    )
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://ai.wildebeest-newton.ts.net/v1")

    assert _pool_runtime_base_url(entry) == "https://ai.wildebeest-newton.ts.net/v1"


class TestAnthropicExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_anthropic().

    Parity with the OpenRouter fix in #18768: resolve_provider_client() passes
    explicit_api_key to _try_openrouter(), but the anthropic branch was not
    updated — _try_anthropic() always fell back to resolve_anthropic_token()
    even when an explicit key was supplied (e.g. from a fallback_model entry).
    """

    def test_try_anthropic_uses_explicit_api_key_over_env(self):
        """_try_anthropic(explicit_api_key) must use the supplied key, not the env fallback."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic("explicit-pool-key")
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-pool-key", (
            f"Expected explicit_api_key to be passed, got: {mock_build.call_args.args[0]}"
        )
        assert mock_build.call_args.args[0] != "env-fallback-key"

    def test_try_anthropic_without_explicit_key_falls_back_to_resolve(self):
        """Without explicit_api_key, _try_anthropic falls back to resolve_anthropic_token."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
        assert client is not None
        assert mock_build.call_args.args[0] == "env-fallback-key"

    def test_resolve_provider_client_passes_explicit_api_key_to_anthropic(self):
        """resolve_provider_client(provider='anthropic', explicit_api_key=...) must propagate the key."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client(
                provider="anthropic",
                explicit_api_key="explicit-fallback-key",
            )
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-fallback-key", (
            "resolve_provider_client must forward explicit_api_key to _try_anthropic()"
        )


# ── Auxiliary unhealthy-provider TTL cache (issue #23570) ────────────────


class TestAuxUnhealthyCache:
    """Recently-402'd providers are skipped on subsequent aux calls.

    Without this, every compression / title-gen / session-search call on a
    long session retries a depleted OpenRouter (~1 RTT to 402) before
    falling back to the next provider. The TTL cache hides the unhealthy
    provider for ``_AUX_UNHEALTHY_TTL_SECONDS`` so the chain skips it.
    """

    def setup_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def teardown_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def test_mark_then_skip(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        assert _is_provider_unhealthy("openrouter") is False
        _mark_provider_unhealthy("openrouter")
        assert _is_provider_unhealthy("openrouter") is True

    def test_ttl_expiry_evicts(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
            _aux_unhealthy_until,
        )
        _mark_provider_unhealthy("openrouter", ttl=0.01)
        assert _is_provider_unhealthy("openrouter") is True
        import time
        time.sleep(0.02)
        # Lazy eviction: first lookup after expiry returns False AND removes the entry.
        assert _is_provider_unhealthy("openrouter") is False
        assert "openrouter" not in _aux_unhealthy_until

    def test_alias_normalization(self):
        """'codex' should normalize to 'openai-codex' so the cache lookup
        matches the chain label."""
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        _mark_provider_unhealthy("codex")
        assert _is_provider_unhealthy("openai-codex") is True

    def test_resolve_auto_skips_unhealthy_step2(self):
        """_resolve_auto Step-2 chain skips unhealthy providers."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark OpenRouter unhealthy → chain should skip it and pick nous.
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value=""), \
             patch("agent.auxiliary_client._read_main_model", return_value=""), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "nous-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        assert client is nous_client
        assert model == "nous-model"
        # The skipped provider's _try_* should NOT have been called at all.
        or_try.assert_not_called()

    def test_resolve_auto_skips_unhealthy_main_in_step1(self):
        """Step-1 also consults the unhealthy cache so a depleted main
        provider doesn't burn a 402 RTT every aux call. Falls through to
        Step-2 chain (which also respects the cache)."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4.6"), \
             patch("agent.auxiliary_client.resolve_provider_client") as step1, \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        # Step-1 was bypassed — resolve_provider_client never invoked
        step1.assert_not_called()
        # Step-2 also skipped openrouter and landed on nous
        or_try.assert_not_called()
        assert client is nous_client

    def test_payment_fallback_skips_unhealthy(self):
        """_try_payment_fallback also consults the unhealthy cache so a 402
        on OpenRouter doesn't cause a second OR call within the same chain
        iteration if it gets re-entered."""
        from agent.auxiliary_client import (
            _try_payment_fallback,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark BOTH the failed provider (openrouter) and a sibling (custom)
        # unhealthy. The chain should still find nous.
        _mark_provider_unhealthy("local/custom")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint") as custom_try, \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is nous_client
        assert label == "nous"
        # OR is skipped via skip_chain_labels (failed provider), custom via unhealthy cache.
        or_try.assert_not_called()
        custom_try.assert_not_called()

    def test_call_llm_marks_provider_unhealthy_on_402(self, monkeypatch):
        """A 402 from call_llm causes the provider to be marked unhealthy
        so the next call skips it instead of re-trying the same depleted
        endpoint."""
        from agent.auxiliary_client import (
            call_llm,
            _is_provider_unhealthy,
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        # base_url tells _recoverable_pool_provider() that this is OpenRouter
        # (resolved_provider="auto" doesn't carry that information by itself).
        primary_client.base_url = "https://openrouter.ai/api/v1/"
        err = Exception("Payment Required: insufficient credits")
        err.status_code = 402
        primary_client.chat.completions.create.side_effect = err

        nous_client = MagicMock()
        nous_resp = MagicMock()
        nous_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        nous_client.chat.completions.create.return_value = nous_resp

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "google/gemini-3-flash-preview")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "google/gemini-3-flash-preview", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(nous_client, "n-model", "nous")), \
             patch("agent.auxiliary_client._build_call_kwargs",
                    return_value={"model": "n-model", "messages": [{"role": "user", "content": "hi"}]}):
            assert _is_provider_unhealthy("openrouter") is False
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )
            # After the 402, OpenRouter is in the unhealthy cache.
            assert _is_provider_unhealthy("openrouter") is True


# ── auxiliary_max_tokens_param ──────────────────────────────────────────────


class TestAuxiliaryMaxTokensParam:
    """Verify the kwarg emitted by ``auxiliary_max_tokens_param`` across
    URL / provider / model-name combinations. Regression cover: a custom
    OpenAI-compatible endpoint serving ``gpt-5.x`` was silently getting
    ``max_tokens`` and 400-ing on ``unsupported_parameter``."""

    def test_direct_openai_returns_max_completion_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://api.openai.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_completion_tokens": 4096}

    def test_local_endpoint_without_model_uses_max_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="http://localhost:11434/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_tokens": 4096}

    def test_openrouter_api_key_present_keeps_max_tokens_without_model_hint(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://openrouter.ai/api/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_tokens": 4096}

    # Model-name fallback — this is the regression guard.

    def test_custom_endpoint_serving_gpt5_uses_max_completion_tokens(self):
        """Third-party gateway + gpt-5.x: name-based detection must kick in."""
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="gpt-5.4") == {
                "max_completion_tokens": 4096
            }

    def test_openrouter_serving_gpt4o_uses_max_completion_tokens(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://openrouter.ai/api/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="openai/gpt-4o-mini") == {
                "max_completion_tokens": 4096
            }

    def test_custom_endpoint_serving_classic_llama_keeps_max_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="llama3-70b") == {
                "max_tokens": 4096
            }

    def test_empty_model_falls_back_to_url_only(self):
        """No model hint → only the URL-based rule applies."""
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="") == {"max_tokens": 4096}
            assert auxiliary_max_tokens_param(4096, model=None) == {"max_tokens": 4096}


# ── Regression tests for issue #52392 ─────────────────────────────────────
# Compression fallback chain currently picks the first reachable candidate
# without checking whether the candidate's context window is large enough.
# When the chosen candidate is reachable but too small for the compression
# task, the call errors out instead of continuing through the chain.

class TestCompressionFallbackContextFilter:
    """Aux fallback chains must skip candidates whose context window is
    smaller than the task minimum, then continue to the next candidate.

    Layer coverage:
      L2: _try_configured_fallback_chain skips too-small candidates
      L3: _try_main_fallback_chain skips too-small candidates
      L4: candidates with unknown context (None) are passed through
      L5: backward compat — first viable candidate still wins
    """

    @staticmethod
    def _make_chain_entry(provider, model, base_url="https://example.com/v1",
                          api_key="k"):
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }

    def _mock_resolve(self, entry):
        """Mock _resolve_fallback_entry to return a (client, model) per entry."""
        client = MagicMock()
        client.base_url = entry.get("base_url", "")
        return client, entry["model"]

    # ── L2: configured fallback chain ─────────────────────────────────

    def test_configured_chain_skips_too_small_candidate_for_compression(self, monkeypatch):
        """When entry[0] is reachable but too small and entry[1] is large enough,
        _try_configured_fallback_chain must return entry[1], not entry[0]."""
        from agent.auxiliary_client import (
            _try_configured_fallback_chain,
        )

        small_client = MagicMock(name="small_client")
        large_client = MagicMock(name="large_client")
        entries = [
            self._make_chain_entry("small-provider", "tiny-8k"),
            self._make_chain_entry("big-provider", "huge-1m"),
        ]

        def fake_resolve(entry):
            if entry is entries[0]:
                return small_client, "tiny-8k"
            return large_client, "huge-1m"

        # tiny-8k resolves to 8K (below 64K floor); huge-1m resolves to 1M
        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"tiny-8k": 8192, "huge-1m": 1_048_576}.get(model, 256_000)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client, (
            f"Expected large_client (1M context), got {client}. "
            "L2 bug: chain returned the first reachable candidate without "
            "screening by context window.")
        assert model == "huge-1m"
        assert "big-provider" in label

    def test_configured_chain_continues_after_skipping_too_small(self, monkeypatch):
        """When all small candidates are skipped and only the last is large enough,
        the chain still returns it (does not stop after first filter)."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        small_client_a = MagicMock(name="small_a")
        small_client_b = MagicMock(name="small_b")
        large_client = MagicMock(name="large")
        entries = [
            self._make_chain_entry("p1", "small-a-32k"),
            self._make_chain_entry("p2", "small-b-48k"),
            self._make_chain_entry("p3", "large-512k"),
        ]

        def fake_resolve(entry):
            if entry is entries[0]:
                return small_client_a, "small-a-32k"
            if entry is entries[1]:
                return small_client_b, "small-b-48k"
            return large_client, "large-512k"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"small-a-32k": 32_000,
                    "small-b-48k": 48_000,
                    "large-512k": 512_000}.get(model, 256_000)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client
        assert model == "large-512k"

    # ── L3: main fallback chain ────────────────────────────────────────

    def test_main_chain_skips_too_small_candidate_for_compression(self, monkeypatch):
        """Same behaviour for the top-level main-agent fallback chain."""
        from agent.auxiliary_client import (
            _try_main_fallback_chain,
        )

        small_client = MagicMock(name="small_main")
        large_client = MagicMock(name="large_main")

        # Mock load_config + get_fallback_chain to return our controlled chain
        chain = [
            self._make_chain_entry("p-small", "tiny-16k"),
            self._make_chain_entry("p-large", "huge-1m"),
        ]

        def fake_resolve(entry):
            if entry is chain[0]:
                return small_client, "tiny-16k"
            return large_client, "huge-1m"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"tiny-16k": 16_384, "huge-1m": 1_048_576}.get(model, 256_000)

        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda cfg: chain,
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx), \
             patch("agent.auxiliary_client._is_provider_unhealthy",
                   return_value=False):
            client, model, label = _try_main_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client, (
            f"Expected large_client (1M), got {client}. "
            "L3 bug: main chain returned the first reachable candidate "
            "without screening by context window.")
        assert model == "huge-1m"

    # ── L4: unknown context passthrough ────────────────────────────────

    def test_configured_chain_passes_through_unknown_context(self, monkeypatch):
        """When get_model_context_length returns None (cannot probe),
        the candidate is NOT filtered — the existing behaviour of using
        the default 256K fallback in the resolver chain is preserved."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        unknown_client = MagicMock(name="unknown_client")
        entries = [self._make_chain_entry("unknown-provider", "unprobed-model")]

        def fake_resolve(entry):
            return unknown_client, "unprobed-model"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return None  # cannot determine context length

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is unknown_client, (
            "L4 bug: candidates with unknown context must be passed through, "
            "not blocked. Being unsure is not the same as being too small.")
        assert model == "unprobed-model"

    # ── L5: backward compat — non-compression tasks unchanged ──────────

    def test_non_compression_task_does_not_filter_by_context(self, monkeypatch):
        """For tasks without a context floor (e.g. title_generation, vision),
        the chain behaviour is unchanged: first reachable candidate wins."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        small_client = MagicMock(name="small")
        entries = [self._make_chain_entry("p", "tiny-4k")]

        def fake_resolve(entry):
            return small_client, "tiny-4k"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return 4_096  # small — but title_generation has no floor

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "title_generation" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="title_generation", failed_provider="auto")

        assert client is small_client, (
            "L5 regression: non-compression tasks must not be filtered "
            "by context window. The first reachable candidate should win.")
        assert model == "tiny-4k"

    # ── End-to-end: configured chain skips too-small for vision too ──
    # vision has its own implicit context requirements; test that the
    # compression-specific filter does NOT affect vision chains.

    def test_compression_task_uses_minimum_context_constant(self):
        """The task minimum for compression must equal MINIMUM_CONTEXT_LENGTH
        so the runtime fallback stays consistent with the startup feasibility
        check in agent/conversation_compression.py."""
        from agent.auxiliary_client import _task_minimum_context_length
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

        assert _task_minimum_context_length("compression") == MINIMUM_CONTEXT_LENGTH
        # Non-compression tasks have no minimum (None)
        assert _task_minimum_context_length("vision") is None
        assert _task_minimum_context_length("title_generation") is None
        assert _task_minimum_context_length("web_extract") is None
        assert _task_minimum_context_length("skills_hub") is None
        assert _task_minimum_context_length("mcp") is None
        assert _task_minimum_context_length("session_search") is None
        # Empty / unknown tasks have no minimum
        assert _task_minimum_context_length("") is None
        assert _task_minimum_context_length(None) is None


class TestCustomEndpointApiKeyInheritance:
    """Issue #9318: when an auxiliary task uses provider=custom with an
    explicit base_url but empty api_key, the custom_key fallback chain must
    inherit ``model.api_key`` from config.yaml before falling to the
    ``no-key-required`` placeholder.

    Without this fix, users on self-hosted gateways who share the same
    endpoint+credentials for both the main model and auxiliary tasks get 401
    auth errors because the placeholder key is sent instead of the real one.

    Inheritance is host-gated: the main key is only inherited when the aux
    base_url points at the same host as the main model's base_url, so a
    misconfigured aux endpoint cannot leak the main credential cross-host.
    """

    def test_inherits_main_api_key_when_aux_key_empty(self, monkeypatch):
        """RED→GREEN: explicit_api_key is None, OPENAI_API_KEY unset →
        model.api_key from config.yaml must be used (same-host gateway)."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        fake_config = {
            "model": {
                "api_key": "sk-main-config-key",
                "base_url": "https://gw.example.com/v1",
                "default": "main-model",
            }
        }
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "sk-main-config-key", (
            "Custom endpoint with empty api_key should inherit "
            "model.api_key from config, got: "
            + repr(captured.get("api_key"))
        )

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        """explicit_api_key wins over config model.api_key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {"api_key": "sk-main-config-key"}}
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key="sk-explicit",
            )

        assert captured.get("api_key") == "sk-explicit"

    def test_local_server_falls_to_no_key_required(self, monkeypatch):
        """When no key is available anywhere (explicit, env, config), fall
        back to ``no-key-required`` for local servers (Ollama, etc.)."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {}}  # no api_key configured
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="http://localhost:11434/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"

    def test_runtime_override_key_is_used(self, monkeypatch):
        """When _RUNTIME_MAIN_API_KEY is set (by set_runtime_main), it takes
        precedence over config.yaml for the custom endpoint key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch.object(ac, "_RUNTIME_MAIN_API_KEY", "sk-runtime-key"), \
             patch.object(ac, "_RUNTIME_MAIN_BASE_URL", "https://gw.example.com/v1"), \
             patch("hermes_cli.config.load_config", return_value={"model": {}}), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "sk-runtime-key"

    def test_cross_host_aux_endpoint_does_not_inherit_main_key(self, monkeypatch):
        """An aux base_url on a DIFFERENT host than the main model must NOT
        inherit model.api_key — that would leak the main credential to
        whatever host a misconfigured aux endpoint names. Falls back to the
        fail-safe no-key-required placeholder instead."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {
            "model": {
                "api_key": "sk-main-config-key",
                "base_url": "https://gw.example.com/v1",
            }
        }
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://other-host.example.net/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"

    def test_no_main_base_url_does_not_inherit_main_key(self, monkeypatch):
        """When the main model has no base_url (e.g. a first-class provider),
        there is no 'same gateway' to match — do not inherit the key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {"api_key": "sk-main-config-key"}}
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"
