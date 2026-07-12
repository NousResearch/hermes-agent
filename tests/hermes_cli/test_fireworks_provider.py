"""Focused tests for Fireworks AI first-class provider wiring.

These tests pin the wiring that makes Fireworks a real provider — alias
resolution through both CLI resolvers, config/doctor/overlay registration,
and credential/base-URL resolution — without
any live network calls.
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
    for key in ("FIREWORKS_API_KEY", "FIREWORKS_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


class TestFireworksAliases:
    """Both CLI resolvers must map the aliases — the plugin's aliases= tuple is
    NOT consulted by these static maps, so they need explicit coverage."""

    @pytest.mark.parametrize("alias", ["fireworks", "fireworks-ai", "fw", "FW", " Fireworks-AI "])
    def test_models_normalize_provider(self, alias):
        assert normalize_provider(alias) == "fireworks"

    @pytest.mark.parametrize("alias", ["fireworks", "fireworks-ai", "fw"])
    def test_providers_normalize_provider(self, alias):
        from hermes_cli.providers import normalize_provider as normalize_in_providers

        assert normalize_in_providers(alias) == "fireworks"


class TestFireworksOrdering:
    """Fireworks participates in the canonical provider catalog."""

    def test_present_in_canonical_providers(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "fireworks" in slugs


    def test_has_a_label(self):
        assert _PROVIDER_LABELS.get("fireworks") == "Fireworks AI"


class TestFireworksConfigRegistry:
    def test_optional_env_vars_include_fireworks(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "FIREWORKS_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["FIREWORKS_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FIREWORKS_API_KEY"]["password"] is True

        assert "FIREWORKS_BASE_URL" not in OPTIONAL_ENV_VARS


class TestFireworksOverlay:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "fireworks" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["fireworks"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_override == "https://api.fireworks.ai/inference/v1"
        assert not overlay.base_url_env_var
        assert not overlay.is_aggregator


class TestFireworksDoctor:
    def test_provider_env_hints_include_fireworks(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "FIREWORKS_API_KEY" in _PROVIDER_ENV_HINTS

    def test_slash_form_model_is_not_flagged_as_vendor_prefixed(self, monkeypatch, tmp_path):
        """Fireworks' native model IDs are slash-form (accounts/fireworks/...),
        so doctor must NOT warn that provider should be 'openrouter' / the prefix
        dropped — that heuristic is for aggregator vendor slugs only."""
        from hermes_cli import doctor as doctor_mod

        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            "model:\n"
            "  provider: fireworks\n"
            "  default: accounts/fireworks/models/kimi-k2p6\n"
            "memory: {}\n",
            encoding="utf-8",
        )
        (home / ".env").write_text("FIREWORKS_API_KEY=fw_test\n", encoding="utf-8")
        project = tmp_path / "project"
        project.mkdir()

        monkeypatch.setattr(doctor_mod, "HERMES_HOME", home)
        monkeypatch.setattr(doctor_mod, "PROJECT_ROOT", project)
        monkeypatch.setattr(doctor_mod, "_DHH", str(home))
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test")

        # Keep the run offline and cheap.
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


class TestFireworksCredentials:
    def test_resolves_default_base_url(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
        creds = resolve_api_key_provider_credentials("fireworks")
        assert creds["api_key"] == "fw_test_key"
        assert creds["base_url"] == "https://api.fireworks.ai/inference/v1"

class TestFireworksSetupPicker:
    def test_setup_uses_same_resolver_as_model_picker(self, monkeypatch):
        from unittest.mock import patch

        from hermes_cli.model_setup_flows import _model_flow_api_key_provider

        resolved = [
            "accounts/fireworks/models/kimi-k2p6",
            "accounts/fireworks/models/kimi-k2p7-code",
        ]
        captured = {}

        def capture_models(model_ids, **_kwargs):
            captured["models"] = model_ids
            return None

        monkeypatch.setattr("builtins.input", lambda *_args: "")
        with (
            patch("hermes_cli.main._prompt_api_key", return_value=("fw_test_key", False)),
            patch("hermes_cli.config.get_env_value", return_value=""),
            patch("hermes_cli.config.load_config", return_value={"model": {}}),
            patch("hermes_cli.config.save_config"),
            patch("hermes_cli.config.save_env_value"),
            patch("hermes_cli.auth._prompt_model_selection", side_effect=capture_models),
            patch("hermes_cli.auth.deactivate_provider"),
            patch("hermes_cli.models.provider_model_ids", return_value=resolved) as resolver,
        ):
            _model_flow_api_key_provider({}, "fireworks")

        resolver.assert_called_once_with("fireworks", force_refresh=True)
        assert captured["models"] == resolved


class TestFireworksAuxiliary:
    """resolve_provider_client wires the BYOK key and PAYG-safe aux model."""

    def _resolve(self, name):
        from unittest.mock import patch

        from agent.auxiliary_client import resolve_provider_client

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client(name)
        return client, model, mock_openai.call_args.kwargs

    def test_client_has_no_partner_attribution_headers(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
        client, model, kwargs = self._resolve("fireworks")
        assert client is not None
        headers = kwargs.get("default_headers", {})
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers
        assert kwargs["base_url"] == "https://api.fireworks.ai/inference/v1"

    def test_aux_model_is_payg_safe(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
        _, model, _ = self._resolve("fireworks")
        assert model.startswith("accounts/fireworks/models/")
        assert "/routers/" not in model
        assert "turbo" not in model.lower()

    def test_alias_resolves_through_aux_client(self, monkeypatch):
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw_test_key")
        client, _, _ = self._resolve("fw")
        assert client is not None


class TestFireworksModelMetadata:
    def test_url_infers_fireworks(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.fireworks.ai/inference/v1") == "fireworks"
