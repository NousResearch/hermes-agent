"""Focused tests for Qubrid AI first-class provider wiring."""

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
        "QUBRID_API_KEY",
        "QUBRID_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestQubridAliases:
    @pytest.mark.parametrize("alias", ["qubrid", "qubrid-ai", "qubrid-platform"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("QUBRID_API_KEY", "qubrid-test-key")
        assert resolve_provider(alias) == "qubrid"

    def test_models_normalize_provider(self):
        assert normalize_provider("qubrid-ai") == "qubrid"
        assert normalize_provider("qubrid-platform") == "qubrid"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("qubrid-ai") == "qubrid"
        assert normalize_provider_in_providers("qubrid-platform") == "qubrid"


class TestQubridConfigRegistry:
    def test_optional_env_vars_include_qubrid(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "QUBRID_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["QUBRID_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["QUBRID_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["QUBRID_API_KEY"]["url"] == "https://platform.qubrid.com/api-keys"

        assert "QUBRID_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["QUBRID_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["QUBRID_BASE_URL"]["password"] is False


class TestQubridModelCatalog:
    def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "qubrid" in slugs

    def test_provider_model_ids_prefers_live_api(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "qubrid-live-key",
                "base_url": "https://platform.qubrid.com/v1",
                "source": "QUBRID_API_KEY",
            },
        )

        def _fake_fetch_models(*, api_key=None, base_url=None, timeout=8.0):
            return [
                "openai/gpt-oss-120b",
                "meta-llama/Llama-3.3-70B-Instruct",
            ]

        from providers import get_provider_profile

        profile = get_provider_profile("qubrid")
        monkeypatch.setattr(profile, "fetch_models", _fake_fetch_models)

        assert provider_model_ids("qubrid") == [
            "openai/gpt-oss-120b",
            "meta-llama/Llama-3.3-70B-Instruct",
        ]

    def test_provider_model_ids_falls_back_to_static_models(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "qubrid-live-key",
                "base_url": "https://platform.qubrid.com/v1",
                "source": "QUBRID_API_KEY",
            },
        )
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda *args, **kwargs: None,
        )
        from providers import get_provider_profile

        profile = get_provider_profile("qubrid")
        monkeypatch.setattr(profile, "fetch_models", lambda *args, **kwargs: None)

        assert provider_model_ids("qubrid") == list(_PROVIDER_MODELS["qubrid"])


class TestQubridDoctor:
    def test_provider_env_hints_include_qubrid(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "QUBRID_API_KEY" in _PROVIDER_ENV_HINTS

    def test_run_doctor_checks_qubrid_models_endpoint(self, monkeypatch, tmp_path):
        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("memory: {}\n", encoding="utf-8")
        (home / ".env").write_text("QUBRID_API_KEY=***\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("QUBRID_API_KEY", "qubrid-test-key")

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
        assert "Qubrid AI" in out
        assert any(url == "https://platform.qubrid.com/v1/models" for url, _, _ in calls)


class TestQubridModelMetadata:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("platform.qubrid.com") == "qubrid"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "qubrid" in _PROVIDER_PREFIXES
        assert "qubrid-ai" in _PROVIDER_PREFIXES
        assert "qubrid-platform" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://platform.qubrid.com/v1") == "qubrid"

    def test_known_qubrid_endpoint_still_uses_endpoint_metadata(self):
        with patch(
            "agent.model_metadata.get_cached_context_length",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={"openai/gpt-oss-120b": {"context_length": 131072}},
        ), patch(
            "agent.models_dev.lookup_models_dev_context",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_model_metadata",
            return_value={},
        ):
            result = get_model_context_length(
                "openai/gpt-oss-120b",
                base_url="https://platform.qubrid.com/v1",
                api_key="qubrid-test-key",
                provider="qubrid",
            )

        assert result == 131072


class TestQubridAuxiliary:
    def test_resolve_provider_client_uses_qubrid_aux_default(self, monkeypatch):
        monkeypatch.setenv("QUBRID_API_KEY", "qubrid-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("qubrid")

        assert client is not None
        assert model == "mistralai/Mistral-7B-Instruct-v0.3"
        assert mock_openai.call_args.kwargs["api_key"] == "qubrid-test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://platform.qubrid.com/v1"
        headers = mock_openai.call_args.kwargs.get("default_headers", {})
        assert headers.get("User-Agent", "").startswith("HermesAgent/")

    def test_qubrid_profile_declares_hermes_user_agent(self):
        from providers import get_provider_profile

        profile = get_provider_profile("qubrid")
        assert profile is not None
        ua = profile.default_headers.get("User-Agent", "")
        assert ua.startswith("HermesAgent/"), (
            f"expected Qubrid profile User-Agent to start with 'HermesAgent/', got {ua!r}"
        )

    def test_resolve_provider_client_accepts_qubrid_alias(self, monkeypatch):
        monkeypatch.setenv("QUBRID_API_KEY", "qubrid-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("qubrid-ai")

        assert client is not None
        assert model == "mistralai/Mistral-7B-Instruct-v0.3"


class TestQubridMainFlow:
    def test_chat_parser_accepts_qubrid_provider(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.config.get_container_exec_info", lambda: None)
        monkeypatch.setattr(
            "hermes_cli.main.cmd_chat",
            lambda args: recorded.setdefault("provider", args.provider),
        )
        monkeypatch.setattr(sys, "argv", ["hermes", "chat", "--provider", "qubrid"])

        from hermes_cli.main import main

        main()

        assert recorded["provider"] == "qubrid"

    def test_select_provider_and_model_routes_qubrid_to_generic_flow(self, monkeypatch):
        recorded: dict[str, str] = {}

        monkeypatch.setattr("hermes_cli.auth.resolve_provider", lambda *args, **kwargs: None)

        def fake_prompt_provider_choice(choices, default=0):
            return next(i for i, label in enumerate(choices) if label.startswith("Qubrid AI"))

        def fake_model_flow_api_key_provider(config, provider_id, current_model=""):
            recorded["provider_id"] = provider_id

        monkeypatch.setattr("hermes_cli.main._prompt_provider_choice", fake_prompt_provider_choice)
        monkeypatch.setattr("hermes_cli.main._model_flow_api_key_provider", fake_model_flow_api_key_provider)

        from hermes_cli.main import select_provider_and_model

        select_provider_and_model()

        assert recorded["provider_id"] == "qubrid"

    def test_model_flow_api_key_provider_persists_qubrid_selection(self, monkeypatch):
        monkeypatch.setenv("QUBRID_API_KEY", "qubrid-test-key")

        with patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["openai/gpt-oss-120b", "meta-llama/Llama-3.3-70B-Instruct"],
        ), patch(
            "hermes_cli.auth._prompt_model_selection",
            return_value="openai/gpt-oss-120b",
        ), patch(
            "hermes_cli.auth.deactivate_provider",
        ), patch(
            "builtins.input",
            return_value="",
        ):
            from hermes_cli.main import _model_flow_api_key_provider

            _model_flow_api_key_provider(load_config(), "qubrid", "old-model")

        import yaml
        from hermes_constants import get_hermes_home

        config = yaml.safe_load((get_hermes_home() / "config.yaml").read_text()) or {}
        model_cfg = config.get("model")
        assert isinstance(model_cfg, dict)
        assert model_cfg["provider"] == "qubrid"
        assert model_cfg["default"] == "openai/gpt-oss-120b"
        assert model_cfg["base_url"] == "https://platform.qubrid.com/v1"


class TestQubridProviderLabel:
    def test_provider_label(self):
        assert _PROVIDER_LABELS["qubrid"] == "Qubrid AI"
