"""Focused tests for GMI Cloud first-class provider wiring."""

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
from agent.model_metadata import get_model_context_length


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
        "GMI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestGmiAliases:
    @pytest.mark.parametrize("alias", ["gmi", "gmi-cloud", "gmicloud"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")
        assert resolve_provider(alias) == "gmi"

    def test_models_normalize_provider(self):
        assert normalize_provider("gmi-cloud") == "gmi"
        assert normalize_provider("gmicloud") == "gmi"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("gmi-cloud") == "gmi"
        assert normalize_provider_in_providers("gmicloud") == "gmi"


class TestGmiConfigRegistry:
    def test_optional_env_vars_include_gmi(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "GMI_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["GMI_API_KEY"]["url"] == "https://www.gmicloud.ai/"

        assert "GMI_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["GMI_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["GMI_BASE_URL"]["password"] is False
        # ENV_VARS_BY_VERSION entries are not needed for providers added after
        # _config_version 22 (the current baseline) — users discover GMI via
        # hermes model, not via upgrade prompts.


class TestGmiModelCatalog:
    def test_static_model_fallback_exists(self):
        assert "gmi" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["gmi"]
        assert "zai-org/GLM-5.1-FP8" in models
        assert "deepseek-ai/DeepSeek-V3.2" in models
        assert "moonshotai/Kimi-K2.5" in models
        assert "anthropic/claude-sonnet-4.6" in models

    def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "gmi" in slugs

    def test_provider_model_ids_prefers_live_api(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "gmi-live-key",
                "base_url": "https://api.gmi-serving.com/v1",
                "source": "GMI_API_KEY",
            },
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda api_key, base_url: [
                "openai/gpt-5.4-mini",
                "zai-org/GLM-5.1-FP8",
            ],
        )

        assert provider_model_ids("gmi") == [
            "openai/gpt-5.4-mini",
            "zai-org/GLM-5.1-FP8",
        ]

    def test_provider_model_ids_falls_back_to_static_models(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "gmi-live-key",
                "base_url": "https://api.gmi-serving.com/v1",
                "source": "GMI_API_KEY",
            },
        )
        monkeypatch.setattr("hermes_cli.models.fetch_api_models", lambda api_key, base_url: None)

        assert provider_model_ids("gmi") == list(_PROVIDER_MODELS["gmi"])


class TestGmiProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "gmi" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["gmi"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("GMI_API_KEY",)
        assert overlay.base_url_override == "https://api.gmi-serving.com/v1"
        assert overlay.base_url_env_var == "GMI_BASE_URL"
        assert not overlay.is_aggregator

    def test_provider_label(self):
        assert _PROVIDER_LABELS["gmi"] == "GMI Cloud"


class TestGmiDoctor:
    def test_provider_env_hints_include_gmi(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "GMI_API_KEY" in _PROVIDER_ENV_HINTS

    def test_run_doctor_checks_gmi_models_endpoint(self, monkeypatch, tmp_path):
        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        (home / ".env").write_text("GMI_API_KEY=***\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")

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
        assert "GMI Cloud" in out
        assert any(url == "https://api.gmi-serving.com/v1/models" for url, _, _ in calls)


class TestGmiModelMetadata:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.gmi-serving.com") == "gmi"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "gmi" in _PROVIDER_PREFIXES
        assert "gmi-cloud" in _PROVIDER_PREFIXES
        assert "gmicloud" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.gmi-serving.com/v1") == "gmi"

    def test_known_gmi_endpoint_still_uses_endpoint_metadata(self):
        with patch(
            "agent.model_metadata.get_cached_context_length",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={"anthropic/claude-opus-4.6": {"context_length": 409600}},
        ), patch(
            "agent.models_dev.lookup_models_dev_context",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_model_metadata",
            return_value={},
        ):
            result = get_model_context_length(
                "anthropic/claude-opus-4.6",
                base_url="https://api.gmi-serving.com/v1",
                api_key="gmi-test-key",
                provider="custom",
            )

        assert result == 409600


class TestGmiAuxiliary:
    def test_aux_default_model(self):
        from agent.auxiliary_client import _get_aux_model_for_provider

        assert _get_aux_model_for_provider("gmi") == "google/gemini-3.1-flash-lite-preview"

    def test_resolve_provider_client_uses_gmi_aux_default(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("gmi")

        assert client is not None
        assert model == "google/gemini-3.1-flash-lite-preview"
        assert mock_openai.call_args.kwargs["api_key"] == "gmi-test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.gmi-serving.com/v1"
        # GMI profile declares default_headers with a HermesAgent User-Agent
        # for traffic attribution. The generic profile-fallback branch in
        # resolve_provider_client should carry it through to the OpenAI client.
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent", "").startswith("HermesAgent/")

    def test_gmi_profile_declares_hermes_user_agent(self):
        """The GMI plugin sets a HermesAgent/<ver> User-Agent on its profile."""
        from providers import get_provider_profile

        profile = get_provider_profile("gmi")
        assert profile is not None
        ua = profile.default_headers.get("User-Agent", "")
        assert ua.startswith("HermesAgent/"), (
            f"expected GMI profile User-Agent to start with 'HermesAgent/', got {ua!r}"
        )

    def test_resolve_provider_client_accepts_gmi_alias(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("gmi-cloud")

        assert client is not None
        assert model == "google/gemini-3.1-flash-lite-preview"


class TestGmiMainFlow:
    def test_chat_parser_accepts_gmi_provider(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
        monkeypatch.setattr(
            "hermes_cli.main.cmd_chat",
            lambda args: recorded.setdefault("provider", args.provider),
        )
        monkeypatch.setattr(sys, "argv", ["hermes", "chat", "--provider", "gmi"])

        from hermes_cli.main import main

        main()

        assert recorded["provider"] == "gmi"

    def test_select_provider_and_model_routes_gmi_to_generic_flow(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda *args, **kwargs: None)

        def fake_prompt_provider_choice(choices, default=0):
            return next(i for i, label in enumerate(choices) if label.startswith("GMI Cloud"))

        def fake_model_flow_api_key_provider(config, provider_id, current_model=""):
            recorded["provider_id"] = provider_id

        monkeypatch.setattr("hermes_cli.main._prompt_provider_choice", fake_prompt_provider_choice)
        monkeypatch.setattr("hermes_cli.main._model_flow_api_key_provider", fake_model_flow_api_key_provider)

        from hermes_cli.main import select_provider_and_model

        select_provider_and_model()

        assert recorded["provider_id"] == "gmi"

    def test_model_flow_api_key_provider_persists_gmi_selection(self, monkeypatch):
        monkeypatch.setenv("GMI_API_KEY", "gmi-test-key")

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["zai-org/GLM-5.1-FP8", "openai/gpt-5.4-mini"],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="openai/gpt-5.4-mini",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch(
            "builtins.input",
            return_value="",
        ):
            from hermes_cli.main import _model_flow_api_key_provider

            _model_flow_api_key_provider(load_config(), "gmi", "old-model")

        import yaml
        from hermes_constants import get_hermes_home

        config = yaml.safe_load((get_hermes_home() / "config.yaml").read_text()) or {}
        model_cfg = config.get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "gmi"
        assert model_cfg["default"] == "openai/gpt-5.4-mini"
        assert model_cfg["base_url"] == "https://api.gmi-serving.com/v1"
