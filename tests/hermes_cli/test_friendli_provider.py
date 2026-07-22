"""Focused tests for Friendli first-class provider wiring.

These tests pin the wiring that makes Friendli a real provider — alias
resolution through both CLI resolvers, config/doctor/overlay registration,
and credential/base-URL resolution — without any live network calls.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from argparse import Namespace

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import resolve_api_key_provider_credentials
from hermes_cli.models import CANONICAL_PROVIDERS, _PROVIDER_LABELS, normalize_provider


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in ("FRIENDLI_API_KEY", "FRIENDLI_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


class TestFriendliAliases:
    """Both CLI resolvers must map the aliases — the plugin's aliases= tuple is
    NOT consulted by these static maps, so they need explicit coverage."""

    @pytest.mark.parametrize("alias", ["friendli", "friendliai", "friendli-ai", "FRIENDLI", " Friendli-AI "])
    def test_models_normalize_provider(self, alias):
        assert normalize_provider(alias) == "friendli"

    @pytest.mark.parametrize("alias", ["friendli", "friendliai", "friendli-ai"])
    def test_providers_normalize_provider(self, alias):
        from hermes_cli.providers import normalize_provider as normalize_in_providers

        assert normalize_in_providers(alias) == "friendli"


class TestFriendliOrdering:
    """Friendli participates in the canonical provider catalog."""

    def test_present_in_canonical_providers(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "friendli" in slugs

    def test_has_a_label(self):
        assert _PROVIDER_LABELS.get("friendli") == "Friendli"


class TestFriendliConfigRegistry:
    def test_optional_env_vars_include_friendli(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "FRIENDLI_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["FRIENDLI_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FRIENDLI_API_KEY"]["password"] is True

        assert "FRIENDLI_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["FRIENDLI_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FRIENDLI_BASE_URL"]["password"] is False


class TestFriendliOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "friendli" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["friendli"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("FRIENDLI_API_KEY",)
        assert overlay.base_url_override == "https://api.friendli.ai/serverless/v1"
        assert overlay.base_url_env_var == "FRIENDLI_BASE_URL"
        assert not overlay.is_aggregator


class TestFriendliDoctor:
    def test_provider_env_hints_include_friendli(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "FRIENDLI_API_KEY" in _PROVIDER_ENV_HINTS

    def test_slash_form_model_is_not_flagged_as_vendor_prefixed(self, monkeypatch, tmp_path):
        """Friendli's native model IDs are slash-form (deepseek-ai/DeepSeek-V3.2,
        zai-org/GLM-5.2, ...), so doctor must NOT warn that provider should be
        'openrouter' / the prefix dropped — that heuristic is for aggregator
        vendor slugs only."""
        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "model:\n"
            "  provider: friendli\n"
            "  default: deepseek-ai/DeepSeek-V3.2\n"
            "memory: {}\n",
            encoding="utf-8",
        )
        (home / ".env").write_text("FRIENDLI_API_KEY=friendli_test\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test")

        import httpx

        monkeypatch.setattr(httpx, "get", lambda *a, **k: types.SimpleNamespace(status_code=200))
        monkeypatch.setitem(
            sys.modules,
            "model_tools",
            types.SimpleNamespace(check_tool_availability=lambda *a, **k: ([], []), TOOLSET_REQUIREMENTS={}),
        )
        with contextlib.suppress(Exception):
            from hermes_cli import auth as _auth_mod

            monkeypatch.setattr(_auth_mod, "get_nous_auth_status", lambda: {})
            monkeypatch.setattr(_auth_mod, "get_codex_auth_status", lambda: {})

        buf = io.StringIO()
        with contextlib.suppress(SystemExit), contextlib.redirect_stdout(buf):
            doctor_mod.run_doctor(Namespace(fix=False))
        out = buf.getvalue()

        assert "vendor-prefixed" not in out
        assert "vendor/model slug" not in out


class TestFriendliCredentials:
    def test_resolves_default_base_url(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test_token")
        creds = resolve_api_key_provider_credentials("friendli")
        assert creds["api_key"] == "friendli_test_token"
        assert creds["base_url"] == "https://api.friendli.ai/serverless/v1"

    def test_base_url_env_override(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test_token")
        monkeypatch.setenv("FRIENDLI_BASE_URL", "https://dedicated.friendli.ai/v1")
        creds = resolve_api_key_provider_credentials("friendli")
        assert creds["base_url"] == "https://dedicated.friendli.ai/v1"


class TestFriendliAuxiliary:
    """resolve_provider_client wires the BYOK key and default aux model."""

    def _resolve(self, name):
        from unittest.mock import patch

        from agent.auxiliary_client import resolve_provider_client

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client(name)
        return client, model, mock_openai.call_args.kwargs

    def test_client_wired_with_base_url(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test_token")
        client, model, kwargs = self._resolve("friendli")
        assert client is not None
        assert kwargs["api_key"] == "friendli_test_token"
        assert kwargs["base_url"] == "https://api.friendli.ai/serverless/v1"

    def test_aux_model_default(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test_token")
        _, model, _ = self._resolve("friendli")
        assert model == "deepseek-ai/DeepSeek-V3.2"

    def test_alias_resolves_through_aux_client(self, monkeypatch):
        monkeypatch.setenv("FRIENDLI_API_KEY", "friendli_test_token")
        client, _, _ = self._resolve("friendliai")
        assert client is not None


class TestFriendliModelMetadata:
    def test_url_infers_friendli(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.friendli.ai/serverless/v1") == "friendli"
