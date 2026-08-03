"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import base64
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    _NOUS_MODEL,
    CodexAuxiliaryClient,
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
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
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
    pass



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






    def test_explicit_provider_moa_unwraps_to_aggregator(self, monkeypatch):
        """An *explicit* `provider="moa"` arg (e.g. a per-task model override
        naming a MoA preset) must resolve to the preset's aggregator, not the
        literal "moa" string — mirrors #53827's fix for the implicit
        "main provider is moa" case in _resolve_auto(), which this function
        never went through."""
        preset = {
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        }
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: preset,
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
            provider="moa",
            model="opus-gpt",
            base_url="moa://local",
            api_key="moa-virtual-provider",
        )

        assert resolved_provider == "openrouter"
        assert model == "anthropic/claude-opus-4.8"
        # The virtual moa:// endpoint must not be forwarded to the aggregator.
        assert base_url is None
        assert api_key is None

    def test_config_provider_moa_unwraps_to_aggregator(self, monkeypatch):
        """`auxiliary.<task>.provider: moa` in config.yaml — the same crash,
        reached via the config path instead of an explicit call-time arg.
        Before the fix this returned ("moa", ...) verbatim, and
        resolve_provider_client() would then look up "moa" in
        PROVIDER_REGISTRY (which has no such entry, it's not a real HTTP
        provider), fail, and surface a "MOA_API_KEY environment variable"
        error for a provider that was never meant to be reached over the wire."""
        preset = {
            "aggregator": {"provider": "anthropic", "model": "claude-opus-4.8"},
        }
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"provider": "moa", "model": "opus-gpt"} if task == "title_generation" else {},
        )
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: preset,
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
        )

        assert resolved_provider == "anthropic"
        assert model == "claude-opus-4.8"
        assert base_url is None
        assert api_key is None


    def test_provider_moa_falls_back_to_literal_when_preset_resolution_fails(self, monkeypatch):
        """If the MoA preset can't be resolved (e.g. renamed/deleted), the
        function must not raise — it degrades to the pre-fix behavior
        (literal "moa") rather than crash resolve_provider_client() harder."""
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: (_ for _ in ()).throw(KeyError("gone-preset")),
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
            provider="moa",
            model="gone-preset",
        )

        assert resolved_provider == "moa"
        assert model == "gone-preset"


    def test_explicit_model_auto_sentinel_is_normalized(self):
        """MoA slots (agent/moa_loop.py's _slot_runtime) forward a preset's
        `model:` field as the explicit `model` kwarg here, not through
        auxiliary.<task> config. Only cfg_model was normalized before, so a
        MoA reference/aggregator slot configured with `model: auto` sent the
        literal string "auto" to the wire as a model id."""
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            provider="anthropic",
            model="auto",
        )

        assert resolved_provider == "anthropic"
        assert model is None





class TestMoaAggregatorSharedResolution:
    """The shared MoA→aggregator helper and the layers that consume it.

    Real-config tests: write an actual config.yaml under a temp HERMES_HOME
    and exercise the genuine load_config() → resolve_moa_preset() boundary —
    no mocking of the configuration-resolution chain.
    """

    @staticmethod
    def _write_moa_config(tmp_path, monkeypatch, default_preset="opus-gpt"):
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "moa": {
                        "default_preset": default_preset,
                        "presets": {
                            "opus-gpt": {
                                "enabled": True,
                                "reference_models": [
                                    {"provider": "openrouter", "model": "openai/gpt-5.5"}
                                ],
                                "aggregator": {
                                    "provider": "openrouter",
                                    "model": "anthropic/claude-opus-4.8",
                                },
                            },
                            "nous-mix": {
                                "enabled": True,
                                "reference_models": [
                                    {"provider": "nous", "model": "hermes-4-70b"}
                                ],
                                "aggregator": {
                                    "provider": "nous",
                                    "model": "hermes-4-405b",
                                },
                            },
                        },
                    }
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_real_config_explicit_task_provider_moa(self, tmp_path, monkeypatch):
        """auxiliary.<task>.provider: moa in a REAL config.yaml resolves to the
        aggregator through the genuine load_config()/resolve_moa_preset() path."""
        import yaml

        home = self._write_moa_config(tmp_path, monkeypatch)
        cfg = yaml.safe_load((home / "config.yaml").read_text())
        cfg["auxiliary"] = {"title_generation": {"provider": "moa", "model": "opus-gpt"}}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
        )

        assert resolved_provider == "openrouter"
        assert model == "anthropic/claude-opus-4.8"
        assert base_url is None
        assert api_key is None






    def test_main_agent_fallback_uses_aggregator_for_moa_main(self, tmp_path, monkeypatch):
        """_try_main_agent_model_fallback with a moa main resolves the
        aggregator instead of asking for a literal "moa" client."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        self._write_moa_config(tmp_path, monkeypatch)
        with patch("agent.auxiliary_client._read_main_provider", return_value="moa"), \
             patch("agent.auxiliary_client._read_main_model", return_value="opus-gpt"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.8")

            client, model, label = _try_main_agent_model_fallback("anthropic", task="compression")

        assert client is mock_client
        assert model == "anthropic/claude-opus-4.8"
        assert label == "main-agent(openrouter)"
        assert mock_resolve.call_args.kwargs["provider"] == "openrouter"
        assert mock_resolve.call_args.kwargs["model"] == "anthropic/claude-opus-4.8"


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


    # ── MoA task should honor max_tokens on ALL providers (#reference_max_tokens) ──

    @pytest.mark.parametrize(
        "provider,model,base_url,expected_key",
        [
            ("zai", "glm-5.2", "https://api.z.ai/api/coding/paas/v4", "max_tokens"),
            ("openrouter", "deepseek/deepseek-v4-flash:nitro", "https://openrouter.ai/api/v1", "max_tokens"),
            ("copilot", "gpt-5.5", "https://api.githubcopilot.com", "max_completion_tokens"),
            ("nous", "hermes-4", "https://inference-api.nousresearch.com/v1", "max_tokens"),
        ],
    )
    def test_moa_task_sends_max_tokens_on_openai_compatible(self, provider, model, base_url, expected_key):
        """MoA reference tasks must honor max_tokens regardless of provider.

        The ``reference_max_tokens`` config option (PR #56756) caps advisor output
        to reduce turn latency.  Before the fix, ``_build_call_kwargs`` silently
        dropped the value for OpenAI-compatible providers (PR #34845), so the cap
        never reached the API.  With the ``task`` parameter threaded through,
        ``task == "moa_reference"`` includes the output cap in kwargs.

        Models that require ``max_completion_tokens`` (GPT-5 family, Copilot)
        get the correct parameter name via ``auxiliary_max_tokens_param()``.
        """
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=800,
            base_url=base_url,
            task="moa_reference",
        )
        assert kwargs[expected_key] == 800




    def test_moa_task_exact_match(self):
        """Only task == "moa_reference" triggers the cap — not the aggregator,
        not arbitrary 'moa_' prefixed tasks."""
        from agent.auxiliary_client import _build_call_kwargs

        # 'moa_reference' → honored
        kw = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_reference",
        )
        assert kw["max_tokens"] == 500

        # 'moa_aggregator' → dropped (aggregator is the acting model, not an advisor)
        kw2 = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_aggregator",
        )
        assert "max_tokens" not in kw2

        # 'moa_custom_future' → dropped (only moa_reference is whitelisted)
        kw3 = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_custom_future",
        )
        assert "max_tokens" not in kw3




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


class TestOpenRouterPaidLaneGuard:
    """Issue #75803: auxiliary auto-chain OpenRouter fallback must be
    configurable and never silently engage a PAID model."""

    def test_free_only_skips_paid_default_model(self, monkeypatch):
        """free_only=true + default (paid) model → OpenRouter skipped."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {"free_only": True}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter()
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

    def test_free_only_allows_free_model(self, monkeypatch):
        """free_only=true + :free model → OpenRouter used with that model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly",
                   return_value={"auxiliary": {"free_only": True,
                                              "openrouter_model": "nvidia/nemotron-3-ultra-550b-a55b:free"}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            client, model = _try_openrouter()
        assert client is mock_client
        assert model == "nvidia/nemotron-3-ultra-550b-a55b:free"

    def test_configured_model_overrides_hardcoded_default(self, monkeypatch):
        """auxiliary.openrouter_model replaces _OPENROUTER_MODEL."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly",
                   return_value={"auxiliary": {"openrouter_model": "some/vendor-model"}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            client, model = _try_openrouter()
        assert client is mock_client
        assert model == "some/vendor-model"

    def test_explicit_caller_model_respects_free_only(self, monkeypatch):
        """Auxiliary.<task>.model (explicit) is also gated by free_only."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {"free_only": True}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter(model="google/gemini-3.6-flash")
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

    def test_paid_lane_warns_once(self, monkeypatch, caplog):
        """Engaging the default paid model logs a WARNING (once per model)."""
        import logging
        from agent.auxiliary_client import _paid_lane_warned
        _paid_lane_warned.discard(_OPENROUTER_MODEL)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = _try_openrouter()
        assert client is mock_client
        assert model == _OPENROUTER_MODEL
        assert any("PAID lane engaged" in r.getMessage() for r in caplog.records)
        # Second call logs nothing new.
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                _try_openrouter()
        assert not any("PAID lane engaged" in r.getMessage() for r in caplog.records)
        _paid_lane_warned.discard(_OPENROUTER_MODEL)

    def test_is_free_model(self):
        from agent.auxiliary_client import _is_free_model
        assert _is_free_model("nvidia/nemotron-3-ultra-550b-a55b:free")
        assert not _is_free_model("google/gemini-3.6-flash")
        assert not _is_free_model("")
        assert not _is_free_model(None)


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



class TestAuxiliaryPoolAwareness:

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
        assert model == _NOUS_MODEL
        assert mock_openai.call_args.kwargs["api_key"] == fresh_token
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"





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
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1




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




    def test_403_subscription_required_is_payment(self):
        exc = Exception(
            "this model requires a subscription, upgrade for access: "
            "https://ollama.com/upgrade"
        )
        setattr(exc, "status_code", 403)
        assert _is_payment_error(exc) is True


    def test_404_generic_not_found_is_not_payment(self):
        exc = Exception("Not Found")
        exc.status_code = 404
        assert _is_payment_error(exc) is False





    # ── Daily / monthly quota exhaustion (#26803) ────────────────────────────








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




class TestRefreshNousRecommendedModel:
    """_refresh_nous_recommended_model picks a fresh model after a stale 404."""



    def test_falls_back_to_default_when_portal_unavailable(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("portal down")
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model", _boom)
        out = _refresh_nous_recommended_model(
            vision=False, stale_model="some/dead-model")
        assert out == _NOUS_MODEL

    def test_returns_none_when_no_distinct_alternative(self, monkeypatch):
        """When the failed model IS the default and the Portal has nothing
        else, there's no usable alternative."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: _NOUS_MODEL,
        )
        out = _refresh_nous_recommended_model(
            vision=False, stale_model=_NOUS_MODEL)
        assert out is None


class TestIsRateLimitError:
    """_is_rate_limit_error detects 429 rate-limit errors warranting fallback."""

    def test_429_with_rate_limit_message(self):
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True








    def test_openai_ratelimiterror_classname(self):
        """OpenAI SDK RateLimitError may omit .status_code — detect by class name."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("rate limit exceeded")
        # No status_code set, but class name matches
        assert _is_rate_limit_error(exc) is True



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
                side_effect=lambda resp, _task, **_kw: resp,
            ),
        )



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

    def test_timeout_forwards_failed_model_to_configured_chain(self):
        """A timeout is model-specific, so call_llm must forward the failed
        model to the configured chain (failed_model=<model>, not None). This
        lets a same-provider sibling in the chain be tried instead of the
        whole provider being skipped — the exact NVIDIA NIM bug's trigger.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        primary = MagicMock()
        primary.base_url = "https://integrate.api.nvidia.com/v1"
        primary.chat.completions.create.side_effect = _Timeout("Request timed out.")

        fb_client = MagicMock()
        fb_client.base_url = "https://integrate.api.nvidia.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(fb_client, "sibling-model", "fallback_chain[0](openrouter)"),
            ) as mock_chain,
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(None, None, ""),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        _, kwargs = mock_chain.call_args
        assert kwargs.get("failed_model") == "some-model", (
            "A timeout is model-specific — the failed model must be forwarded "
            "so a same-provider sibling can be tried, not skipped wholesale."
        )




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




class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True



    def test_normal_api_error_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False



class TestKimiTemperatureOmitted:
    """Kimi/Moonshot models should have temperature OMITTED from API kwargs.

    The Kimi gateway selects the correct temperature server-side based on the
    active mode (thinking → 1.0, non-thinking → 0.6).  Sending any temperature
    value conflicts with gateway-managed defaults.
    """




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

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
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

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
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






    @pytest.mark.parametrize("moa_task", ["moa_reference", "moa_aggregator"])
    def test_moa_tasks_reject_task_level_reasoning_effort(self, moa_task, caplog):
        """MoA reasoning is per-slot in the preset — the auxiliary-task
        shorthand is ignored with a warning pointing at the preset config."""
        from agent.auxiliary_client import _get_task_extra_body

        config = {"auxiliary": {moa_task: {"reasoning_effort": "xhigh"}}}
        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            result = _get_task_extra_body(moa_task)

        assert "reasoning" not in result
        assert any("per-slot" in rec.message for rec in caplog.records)


    def test_anthropic_aux_client_forwards_extra_body_reasoning(self):
        """_AnthropicCompletionsAdapter passes extra_body.reasoning into
        build_anthropic_kwargs as reasoning_config."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(MagicMock(), "claude-sonnet-4-6", is_oauth=False)

        with patch("agent.anthropic_adapter.build_anthropic_kwargs",
                   return_value={"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 64}) as mock_bak, \
             patch("agent.anthropic_adapter.create_anthropic_message") as mock_create, \
             patch("agent.transports.get_transport") as mock_gt:
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok", tool_calls=None, reasoning=None, finish_reason="stop",
                usage=None, provider_data=None,
            )
            adapter.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"reasoning": {"enabled": True, "effort": "low"}},
            )

        assert mock_bak.call_args.kwargs["reasoning_config"] == {
            "enabled": True, "effort": "low",
        }
        mock_create.assert_called_once()

    def _run_anthropic_adapter(self, *, call_extra_body=None, bak_result=None):
        """Drive _AnthropicCompletionsAdapter.create() with mocked SDK layers;
        return the api_kwargs handed to create_anthropic_message."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(MagicMock(), "claude-sonnet-4-6", is_oauth=False)
        bak_result = bak_result or {
            "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 64,
        }
        with patch("agent.anthropic_adapter.build_anthropic_kwargs",
                   return_value=dict(bak_result)), \
             patch("agent.anthropic_adapter.create_anthropic_message") as mock_create, \
             patch("agent.transports.get_transport") as mock_gt:
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok", tool_calls=None, reasoning=None, finish_reason="stop",
                usage=None, provider_data=None,
            )
            kwargs = {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
            }
            if call_extra_body is not None:
                kwargs["extra_body"] = call_extra_body
            adapter.create(**kwargs)
        return mock_create.call_args.args[1]

    def test_anthropic_aux_extra_body_passthrough(self):
        """Bug B (#37217): vendor fields in extra_body reach the Anthropic SDK."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={"thinking": {"type": "disabled"}, "metadata": {"user_id": "u1"}},
        )
        assert api_kwargs["extra_body"] == {
            "thinking": {"type": "disabled"}, "metadata": {"user_id": "u1"},
        }





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



# ---------------------------------------------------------------------------
# Anthropic-compatible image block conversion
# ---------------------------------------------------------------------------

class TestAnthropicCompatImageConversion:
    """Tests for _is_anthropic_compat_endpoint and _convert_openai_images_to_anthropic."""



    def test_url_based_detection(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("custom", "https://api.minimax.io/anthropic")
        assert _is_anthropic_compat_endpoint("custom", "https://example.com/anthropic/v1")
        assert not _is_anthropic_compat_endpoint("custom", "https://api.openai.com/v1")


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

    def test_refresh_provider_credentials_remints_vertex_token_and_evicts_cache(self):
        """Vertex tokens live ~1h; on a long-running gateway the cached
        auxiliary client's bearer token expires mid-session and 401s.
        _refresh_provider_credentials("vertex") must re-mint the token via
        the adapter (which refreshes in place when near expiry) and evict
        the stale cached client so the next call rebuilds with a fresh one —
        previously there was no "vertex" branch here at all, so this fell
        through to the final `return False` and the stale client (and its
        dead token) stayed cached until process restart."""
        stale_client = MagicMock()
        cache_key = ("vertex", False, None, None, None)

        with (
            patch("agent.auxiliary_client._client_cache", {cache_key: (stale_client, "google/gemini-3-flash-preview", None)}),
            patch(
                "agent.vertex_adapter.get_vertex_config",
                return_value=("ya29.FRESH", "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"),
            ) as mock_get_config,
        ):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("vertex") is True

        mock_get_config.assert_called_once()
        stale_client.close.assert_called_once()

    def test_refresh_provider_credentials_vertex_returns_false_when_unminted(self):
        """No usable token/base_url (e.g. ADC and the service-account file
        both failed) — refresh must report failure, not silently evict and
        pretend the client is fixed."""
        with patch("agent.vertex_adapter.get_vertex_config", return_value=(None, None)):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("vertex") is False


    def test_resolve_provider_client_vertex_none_when_no_credentials(self):
        with patch("agent.vertex_adapter.has_vertex_credentials", return_value=False):
            client, model = resolve_provider_client("vertex", "google/gemini-3-flash-preview")

        assert client is None
        assert model is None



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



class TestAnthropicAuxiliaryReasoningTranslation:
    """Native Anthropic aux adapters must receive normalized Hermes reasoning.

    MoA slot reasoning is carried through call_llm as a Hermes
    ``reasoning_config``. The native Anthropic Messages path cannot consume the
    generic OpenAI-style ``extra_body.reasoning`` fallback, so assert the final
    ``messages.create`` kwargs contain Anthropic's provider-aware wire shape.
    """

    @staticmethod
    def _build_adapter(model="claude-fable-5"):
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        captured = {}

        class _Messages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                )

        real_client = SimpleNamespace(messages=_Messages())
        return _AnthropicCompletionsAdapter(real_client, model), captured

    def test_reasoning_config_reaches_native_anthropic_wire_kwargs(self):
        adapter, captured = self._build_adapter()

        adapter.create(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            _reasoning_config={"enabled": True, "effort": "medium"},
        )

        assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert captured["output_config"] == {"effort": "medium"}
        assert "extra_body" not in captured

    def test_build_call_kwargs_private_reasoning_only_for_anthropic_messages(self):
        anthropic_kwargs = _build_call_kwargs(
            "anthropic",
            "claude-fable-5",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.anthropic.com/v1",
        )
        assert anthropic_kwargs["_reasoning_config"] == {"enabled": True, "effort": "medium"}

        proxy_kwargs = _build_call_kwargs(
            "custom",
            "claude-fable-5",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://example.test/anthropic/v1",
        )
        assert proxy_kwargs["_reasoning_config"] == {"enabled": True, "effort": "medium"}

        openai_wire_kwargs = _build_call_kwargs(
            "custom",
            "gpt-compatible",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://example.test/v1",
        )
        assert "_reasoning_config" not in openai_wire_kwargs


class TestAuxiliaryProviderProfileReasoning:
    """Auxiliary calls must reuse provider-profile reasoning wire shapes."""

    def test_kimi_reasoning_uses_top_level_effort(self):
        kwargs = _build_call_kwargs(
            "kimi-coding",
            "kimi-k2-turbo-preview",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.moonshot.ai/v1",
        )

        assert kwargs["reasoning_effort"] == "medium"
        assert "reasoning" not in kwargs.get("extra_body", {})
        assert "thinking" not in kwargs.get("extra_body", {})



    @pytest.mark.asyncio
    async def test_async_call_llm_preserves_profile_reasoning_kwargs(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            base_url="https://api.moonshot.ai/v1",
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(
                "kimi-coding",
                "kimi-k2-turbo-preview",
                "https://api.moonshot.ai/v1",
                "test-key",
                None,
            ),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-k2-turbo-preview"),
        ):
            result = await async_call_llm(
                provider="kimi-coding",
                model="kimi-k2-turbo-preview",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_config={"enabled": True, "effort": "high"},
            )

        assert result is response
        final_kwargs = create.call_args.kwargs
        assert final_kwargs["reasoning_effort"] == "high"
        assert "reasoning" not in final_kwargs.get("extra_body", {})


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



    def test_reasoning_effort_low_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "low"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}




    def test_no_extra_body_means_no_reasoning_keys(self):
        """Baseline: without extra_body, no reasoning/include is sent (preserves
        current behavior for callers that don't opt in)."""
        adapter, captured = self._build_adapter()
        adapter.create(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning" not in captured
        assert "include" not in captured



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




class TestCodexAdapterPromptCacheKey:
    """_CodexCompletionsAdapter emits a stable content-addressed prompt_cache_key
    on the Codex/Responses aux path, matching the main transport
    (agent/transports/codex.py). Regression for issue #53735: MoA acting-
    aggregator and other auxiliary Responses calls stayed cache-cold because
    the adapter never set prompt_cache_key.
    """

    @staticmethod
    def _build_adapter(base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.5"):
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
        adapter = _CodexCompletionsAdapter(real_client, model)
        return adapter, captured_kwargs






    @pytest.mark.parametrize("model", [
        "gpt-4.1",
        "gpt-5.1-codex-max",
        "openai.gpt-5.5-pro",
    ])
    def test_extended_cache_models_set_prompt_cache_retention(self, model):
        adapter, captured = self._build_adapter(
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
            model=model,
        )
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert captured["prompt_cache_retention"] == "24h"

    def test_prompt_cache_retention_skipped_for_codex_backend(self):
        adapter, captured = self._build_adapter()
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com/v1",
        "https://example.services.ai.azure.com/openai/v1",
        "https://responses.example.com/v1",
    ])
    def test_prompt_cache_retention_skipped_for_other_compatible_endpoints(self, base_url):
        adapter, captured = self._build_adapter(base_url=base_url)
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

    def test_prompt_cache_retention_skipped_for_xai_and_github_hosts(self):
        adapter, captured = self._build_adapter(base_url="https://api.x.ai/v1")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured



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


class TestCodexAuxiliaryAdapterCompletedResponse:
    def test_accepts_completed_response_when_stream_was_requested(self):
        completed = SimpleNamespace(
            status="completed",
            id="resp_completed",
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(
                    type="output_text",
                    text="completed response",
                )],
            )],
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=3,
                total_tokens=14,
            ),
        )

        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["stream"] is True
                return completed

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.6-terra")

        response = adapter.create(
            messages=[{"role": "user", "content": "review this"}],
        )

        assert response.choices[0].message.content == "completed response"
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 14


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
            ), patch(
                "agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0
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



    def test_openrouter_api_key_present_keeps_max_tokens_without_model_hint(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://openrouter.ai/api/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_tokens": 4096}

    # Model-name fallback — this is the regression guard.




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


    # ── same-provider, different-model chain entries ────────────────────
    # A configured fallback_chain may legitimately list several models
    # under the *same* provider (e.g. two more NVIDIA NIM models after the
    # primary NIM model). failed_provider alone must not skip those
    # sibling entries — only failed_model narrows the skip to the exact
    # (provider, model) pair that just failed.


    def test_same_provider_same_model_still_skipped(self, monkeypatch):
        """The exact (provider, model) pair that just failed is still
        skipped — failed_model narrows the skip, it doesn't disable it."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        entries = [
            self._make_chain_entry("nvidia", "deepseek-ai/deepseek-v4-pro"),
        ]

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=AssertionError("must not be resolved")):
            client, model, label = _try_configured_fallback_chain(
                task="compression",
                failed_provider="nvidia",
                failed_model="deepseek-ai/deepseek-v4-pro",
            )

        assert client is None
        assert model is None
        assert label == ""


    # ── L3: main fallback chain ────────────────────────────────────────


    # ── L4: unknown context passthrough ────────────────────────────────


    # ── L5: backward compat — non-compression tasks unchanged ──────────


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

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
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

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key="sk-explicit",
            )

        assert captured.get("api_key") == "sk-explicit"


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
             patch("hermes_cli.config.load_config", return_value={"model": {}}), patch("hermes_cli.config.load_config_readonly", return_value={"model": {}}), \
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

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://other-host.example.net/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"


class TestMoaAggregatorStreamingBypass:
    def test_moa_aggregator_stream_bypasses_relay_for_codex_auxiliary_client(self, monkeypatch):
        """The MoA facade owns the streaming contract. For Codex Responses-shim
        clients (openai-codex, xai-oauth), call_llm must return the provider's
        direct create() result instead of routing through Relay's managed
        stream, which cannot iterate a completed SimpleNamespace (#74903).
        """

        completed = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        real_client = SimpleNamespace(
            api_key="test-key",
            base_url="https://chatgpt.com/backend-api/codex/",
            close=lambda: None,
        )
        client = CodexAuxiliaryClient(real_client, "gpt-5.6-sol")
        direct_create = MagicMock(return_value=completed)
        monkeypatch.setattr(client.chat.completions, "create", direct_create)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client",
            lambda *args, **kwargs: (client, "gpt-5.6-sol"),
        )
        relay_stream = MagicMock(side_effect=AssertionError("_relay_sync_stream must not be used"))
        monkeypatch.setattr("agent.auxiliary_client._relay_sync_stream", relay_stream)

        result = call_llm(
            task="moa_aggregator",
            provider="openai-codex",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "只回答 OK"}],
            stream=True,
        )

        assert result is completed
        direct_create.assert_called_once()
        relay_stream.assert_not_called()


class TestSynchronousFallbackCachePlans:
    @staticmethod
    def _run_configured_fallback(monkeypatch, entry):
        from agent.auxiliary_client import (
            _call_fallback_candidate_sync,
            _try_configured_fallback_chain,
        )

        client = MagicMock()
        client.base_url = entry["base_url"]
        client.chat.completions.create.return_value = _DummyResponse()
        resolved_calls = []

        def resolve(provider, model=None, **kwargs):
            resolved_calls.append((provider, model, kwargs))
            return client, model

        monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": [entry]},
        )
        fallback_client, fallback_model, label = _try_configured_fallback_chain(
            task="moa_aggregator",
            failed_provider="primary",
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        _call_fallback_candidate_sync(
            fallback_client,
            fallback_model,
            label,
            task="moa_aggregator",
            messages=[
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": "lookup"},
            ],
            temperature=None,
            max_tokens=None,
            tools=tools,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )
        return client, resolved_calls, tools

    def test_direct_anthropic_fallback_uses_entry_destination_for_tool_marker(self, monkeypatch):
        client, resolved_calls, tools = self._run_configured_fallback(monkeypatch, {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        })

        assert resolved_calls == [(
            "anthropic",
            "claude-sonnet-4-6",
            {
                "explicit_base_url": "https://api.anthropic.com",
                "explicit_api_key": None,
                "api_mode": "anthropic_messages",
            },
        )]
        wire_tools = client.chat.completions.create.call_args.kwargs["tools"]
        assert "cache_control" in wire_tools[-1]
        assert "cache_control" not in tools[-1]

    def test_third_party_anthropic_fallback_keeps_message_markers_without_tool_marker(self, monkeypatch):
        client, resolved_calls, tools = self._run_configured_fallback(monkeypatch, {
            "provider": "custom",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.minimax.io/anthropic",
            "api_mode": "anthropic_messages",
        })

        assert resolved_calls[0][2]["explicit_base_url"] == "https://api.minimax.io/anthropic"
        assert resolved_calls[0][2]["api_mode"] == "anthropic_messages"
        wire_request = client.chat.completions.create.call_args.kwargs
        assert "cache_control" not in wire_request["tools"][-1]
        assert "cache_control" not in tools[-1]
        assert any(
            isinstance(part, dict) and "cache_control" in part
            for message in wire_request["messages"]
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        )



class TestAsynchronousFallbackCachePlans:
    @pytest.mark.asyncio
    async def test_async_fallback_replans_cache_sections_like_sync(self, monkeypatch):
        """Async mirror parity: per-destination cache replan, not verbatim pass-through."""
        from agent.auxiliary_client import (
            _call_fallback_candidate_async,
            _try_configured_fallback_chain,
        )

        entry = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        }
        client = MagicMock()
        client.base_url = entry["base_url"]

        async def _create(**kwargs):
            return _DummyResponse()

        client.chat.completions.create = MagicMock(side_effect=_create)
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model=None, **kwargs: (client, model),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": [entry]},
        )
        fallback_client, fallback_model, label = _try_configured_fallback_chain(
            task="moa_aggregator",
            failed_provider="primary",
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        await _call_fallback_candidate_async(
            fallback_client,
            fallback_model,
            label,
            task="moa_aggregator",
            messages=[
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": "lookup"},
            ],
            temperature=None,
            max_tokens=None,
            tools=tools,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )

        wire_tools = client.chat.completions.create.call_args.kwargs["tools"]
        assert "cache_control" in wire_tools[-1]
        assert "cache_control" not in tools[-1]
