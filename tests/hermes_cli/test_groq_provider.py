"""Focused tests for Groq first-class provider wiring."""

from __future__ import annotations

import contextlib
import io
import sys
import types
from argparse import Namespace
from unittest.mock import patch

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import resolve_provider
from hermes_cli.config import load_config
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_LABELS,
    _PROVIDER_MODELS,
    normalize_provider,
    provider_model_ids,
)
from agent.auxiliary_client import resolve_provider_client


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "MINIMAX_API_KEY",
        "GMI_API_KEY",
        "GROQ_API_KEY",
        "GROQ_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestGroqAliases:
    @pytest.mark.parametrize("alias", ["groq", "groqcloud", "groq-cloud"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        assert resolve_provider(alias) == "groq"

    def test_models_normalize_provider(self):
        assert normalize_provider("groqcloud") == "groq"
        assert normalize_provider("groq-cloud") == "groq"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("groqcloud") == "groq"
        assert normalize_provider_in_providers("groq-cloud") == "groq"


class TestGroqConfigRegistry:
    def test_optional_env_vars_include_groq(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "GROQ_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GROQ_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GROQ_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["GROQ_API_KEY"]["url"] == "https://console.groq.com/keys"

        assert "GROQ_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GROQ_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GROQ_BASE_URL"]["password"] is False


class TestGroqModelCatalog:
    def test_static_model_fallback_exists(self):
        assert "groq" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["groq"]
        assert "llama-3.3-70b-versatile" in models
        assert "llama-3.1-8b-instant" in models
        assert "mixtral-8x7b-32768" in models
        assert "meta-llama/llama-4-maverick-17b-128e-instruct" in models

    def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "groq" in slugs

    def test_provider_model_ids_prefers_live_api(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "groq-live-key",
                "base_url": "https://api.groq.com/openai/v1",
                "source": "GROQ_API_KEY",
            },
        )
        # The profile-based generic fetch path calls profile.fetch_models(),
        # not fetch_api_models().  Mock at the profile level.
        groq_mod = sys.modules["plugins.model_providers.groq"]
        monkeypatch.setattr(groq_mod, "_CACHE", None)
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda self, *, api_key=None, timeout=8.0: [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
            ],
        )

        result = provider_model_ids("groq")
        assert "llama-3.3-70b-versatile" in result
        assert "llama-3.1-8b-instant" in result

    def test_provider_model_ids_falls_back_when_no_api_key(self, monkeypatch):
        # When no credentials are available, falls through to profile
        # fallback_models or static _PROVIDER_MODELS.
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "",
                "base_url": "https://api.groq.com/openai/v1",
                "source": "",
            },
        )
        groq_mod = sys.modules["plugins.model_providers.groq"]
        monkeypatch.setattr(groq_mod, "_CACHE", None)
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda self, *, api_key=None, timeout=8.0: None,
        )

        result = provider_model_ids("groq")
        # Should return fallback_models from the profile
        assert len(result) > 0
        assert "llama-3.3-70b-versatile" in result


class TestGroqProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "groq" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["groq"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "https://api.groq.com/openai/v1"
        assert overlay.base_url_env_var == "GROQ_BASE_URL"
        assert not overlay.is_aggregator

    def test_provider_label(self):
        assert _PROVIDER_LABELS["groq"] == "Groq"


class TestGroqDoctor:
    def test_provider_env_hints_include_groq(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "GROQ_API_KEY" in _PROVIDER_ENV_HINTS

    def test_run_doctor_checks_groq_models_endpoint(self, monkeypatch, tmp_path):
        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        (home / ".env").write_text("GROQ_API_KEY=***\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

        for env_name in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_TOKEN",
            "GLM_API_KEY",
            "ZAI_API_KEY",
            "Z_AI_API_KEY",
            "KIMI_API_KEY",
            "KIMI_CN_API_KEY",
            "ARCEEAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "HF_TOKEN",
            "DASHSCOPE_API_KEY",
            "MINIMAX_API_KEY",
            "MINIMAX_CN_API_KEY",
            "AI_GATEWAY_API_KEY",
            "KILOCODE_API_KEY",
            "OPENCODE_ZEN_API_KEY",
            "OPENCODE_GO_API_KEY",
            "XIAOMI_API_KEY",
            "GMI_API_KEY",
        ):
            monkeypatch.delenv(env_name, raising=False)

        fake_model_tools = types.SimpleNamespace(
            check_tool_availability=lambda *a, **kw: ([], []),
            TOOLSET_REQUIREMENTS={},
        )
        monkeypatch.setitem(sys.modules, "model_tools", fake_model_tools)

        try:
            from hermes_cli import auth as _auth_mod

            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})
        except Exception:
            pass

        calls = []

        def fake_get(url, headers=None, timeout=None):
            calls.append((url, headers, timeout))
            return types.SimpleNamespace(status_code=200)

        import httpx

        monkeypatch.setattr(httpx, "get", fake_get)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "API key or custom endpoint configured" in out
        assert "Groq" in out
        assert any(url == "https://api.groq.com/openai/v1/models" for url, _, _ in calls)


class TestGroqModelMetadata:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.groq.com") == "groq"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "groq" in _PROVIDER_PREFIXES
        assert "groqcloud" in _PROVIDER_PREFIXES
        assert "groq-cloud" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.groq.com/openai/v1") == "groq"


class TestGroqAuxiliary:
    def test_aux_default_model(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("groq") == "llama-3.1-8b-instant"

    def test_resolve_provider_client_uses_groq_aux_default(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("groq")

        assert client is not None
        assert model == "llama-3.1-8b-instant"
        assert mock_openai.call_args.kwargs["api_key"] == "groq-test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.groq.com/openai/v1"

    def test_resolve_provider_client_accepts_groq_alias(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("groqcloud")

        assert client is not None
        assert model == "llama-3.1-8b-instant"


class TestGroqProviderProfile:
    def test_profile_exists(self):
        from providers import get_provider_profile

        profile = get_provider_profile("groq")
        assert profile is not None
        assert profile.name == "groq"
        assert profile.display_name == "Groq"
        assert profile.base_url == "https://api.groq.com/openai/v1"
        assert "GROQ_API_KEY" in profile.env_vars

    def test_profile_is_groq_subclass(self):
        from providers import get_provider_profile

        profile = get_provider_profile("groq")
        assert type(profile).__name__ == "GroqProfile"

    def test_fetch_models_filters_non_chat(self, monkeypatch):
        """Non-chat models (whisper, TTS, etc.) are filtered from live results."""
        from providers import get_provider_profile

        profile = get_provider_profile("groq")
        raw_models = [
            "llama-3.3-70b-versatile",
            "whisper-large-v3",
            "distil-whisper-large-v3-en",
            "llama-3.1-8b-instant",
            "playai-tts",
            "llama-guard-3-8b",
        ]

        # Patch the parent fetch_models to return our raw list
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda self, *, api_key=None, timeout=8.0: list(raw_models),
        )
        # Clear any cached result
        groq_mod = sys.modules["plugins.model_providers.groq"]
        monkeypatch.setattr(groq_mod, "_CACHE", None)

        result = profile.fetch_models()
        assert result == ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
        assert "whisper-large-v3" not in result
        assert "playai-tts" not in result
        assert "llama-guard-3-8b" not in result

    def test_fetch_models_caches_result(self, monkeypatch):
        """Second call returns cached result without re-fetching."""
        from providers import get_provider_profile

        profile = get_provider_profile("groq")
        call_count = 0

        def counting_fetch(self, *, api_key=None, timeout=8.0):
            nonlocal call_count
            call_count += 1
            return ["llama-3.3-70b-versatile"]

        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            counting_fetch,
        )
        groq_mod = sys.modules["plugins.model_providers.groq"]
        monkeypatch.setattr(groq_mod, "_CACHE", None)

        result1 = profile.fetch_models()
        result2 = profile.fetch_models()
        assert result1 == result2 == ["llama-3.3-70b-versatile"]
        assert call_count == 1  # only one HTTP call


class TestGroqMainFlow:
    def test_chat_parser_accepts_groq_provider(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
        monkeypatch.setattr(
            "hermes_cli.main.cmd_chat",
            lambda args: recorded.setdefault("provider", args.provider),
        )
        monkeypatch.setattr(sys, "argv", ["hermes", "chat", "--provider", "groq"])

        from hermes_cli.main import main

        main()

        assert recorded["provider"] == "groq"

    def test_select_provider_and_model_routes_groq_to_generic_flow(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda *args, **kwargs: None)

        def fake_prompt_provider_choice(choices, default=0):
            return next(i for i, label in enumerate(choices) if label.startswith("Groq"))

        def fake_model_flow_api_key_provider(config, provider_id, current_model=""):
            recorded["provider_id"] = provider_id

        monkeypatch.setattr("hermes_cli.main._prompt_provider_choice", fake_prompt_provider_choice)
        monkeypatch.setattr("hermes_cli.main._model_flow_api_key_provider", fake_model_flow_api_key_provider)

        from hermes_cli.main import select_provider_and_model

        select_provider_and_model()

        assert recorded["provider_id"] == "groq"

    def test_model_flow_api_key_provider_persists_groq_selection(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="llama-3.3-70b-versatile",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch(
            "builtins.input",
            return_value="",
        ):
            from hermes_cli.main import _model_flow_api_key_provider

            _model_flow_api_key_provider(load_config(), "groq", "old-model")

        import yaml
        from hermes_constants import get_hermes_home

        config = yaml.safe_load((get_hermes_home() / "config.yaml").read_text()) or {}
        model_cfg = config.get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "groq"
        assert model_cfg["default"] == "llama-3.3-70b-versatile"
        assert model_cfg["base_url"] == "https://api.groq.com/openai/v1"


class TestGroqAuthRegistry:
    def test_provider_registry_entry_exists(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "groq" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["groq"]
        assert cfg.name == "Groq"
        assert cfg.auth_type == "api_key"
        assert cfg.inference_base_url == "https://api.groq.com/openai/v1"
        assert "GROQ_API_KEY" in cfg.api_key_env_vars
        assert cfg.base_url_env_var == "GROQ_BASE_URL"

    def test_auto_detect_groq_from_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        assert resolve_provider("groq") == "groq"

    def test_auto_detect_groq_when_auto(self, monkeypatch):
        """When provider=auto and only GROQ_API_KEY is set, groq is selected."""
        monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
        result = resolve_provider("auto")
        assert result == "groq"
