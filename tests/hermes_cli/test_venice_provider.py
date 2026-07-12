"""Behavior coverage for Venice provider discovery and persistence."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.config import load_config
from hermes_cli.models import provider_model_ids


def _venice_profile():
    from providers import get_provider_profile

    profile = get_provider_profile("venice")
    assert profile is not None
    return profile


def _credentials(api_key: str = "venice-test-key") -> dict[str, str]:
    return {
        "provider": "venice",
        "api_key": api_key,
        "base_url": "https://api.venice.ai/api/v1",
        "source": "VENICE_API_KEY",
    }


class TestVeniceModelDiscovery:
    def test_live_models_lead_when_available(self, monkeypatch):
        live = ["zai-org-glm-5-1", "tee-qwen3-5-122b-a10b"]
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: _credentials(),
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *args, **kwargs: live,
        )

        models = provider_model_ids("venice")

        assert models[: len(live)] == live
        assert len(models) == len(set(model.lower() for model in models))

    def test_missing_credentials_returns_curated_fallback(self, monkeypatch):
        fallback = list(_venice_profile().fallback_models)
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: _credentials(api_key=""),
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("catalog must not be probed without credentials")
            ),
        )

        assert provider_model_ids("venice") == fallback

    def test_probe_failure_returns_curated_fallback(self, monkeypatch):
        fallback = list(_venice_profile().fallback_models)
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: _credentials(),
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda *args, **kwargs: None,
        )

        assert provider_model_ids("venice") == fallback


class TestVeniceProviderSelection:
    def test_current_provider_surfaces_auto_wire_venice(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        from hermes_cli.config import OPTIONAL_ENV_VARS
        from hermes_cli.main import (
            _build_provider_choices,
            _is_profile_api_key_provider,
        )
        from hermes_cli.models import CANONICAL_PROVIDERS
        from hermes_cli.provider_catalog import provider_catalog_by_slug

        canonical = {entry.slug for entry in CANONICAL_PROVIDERS}
        descriptor = provider_catalog_by_slug()["venice"]
        assert "venice" in canonical
        assert "venice" in PROVIDER_REGISTRY
        assert "venice" in _build_provider_choices()
        assert _is_profile_api_key_provider("venice")
        assert descriptor.api_key_env_vars == ("VENICE_API_KEY",)
        assert descriptor.base_url_env_var == "VENICE_BASE_URL"
        assert OPTIONAL_ENV_VARS["VENICE_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["VENICE_API_KEY"]["password"] is True

    def test_runtime_and_auxiliary_resolution_use_profile(self, monkeypatch):
        monkeypatch.setenv("VENICE_API_KEY", "venice-test-key")

        from agent.auxiliary_client import _get_aux_model_for_provider
        from agent.model_metadata import _infer_provider_from_url
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider(requested="venice")

        assert runtime["provider"] == "venice"
        assert runtime["api_mode"] == "chat_completions"
        assert runtime["api_key"] == "venice-test-key"
        assert runtime["base_url"] == "https://api.venice.ai/api/v1"
        assert _get_aux_model_for_provider("venice") == "zai-org-glm-4.7-flash"
        assert _infer_provider_from_url("https://api.venice.ai/api/v1") == "venice"

    def test_model_flow_persists_venice_selection(self, monkeypatch):
        monkeypatch.setenv("VENICE_API_KEY", "venice-test-key")

        with (
            patch(
                "hermes_cli.models.fetch_api_models",
                return_value=["zai-org-glm-5", "zai-org-glm-4.7"],
            ),
            patch(
                "hermes_cli.auth._prompt_model_selection",
                return_value="zai-org-glm-5",
            ),
            patch("hermes_cli.auth.deactivate_provider"),
            patch("builtins.input", return_value=""),
        ):
            from hermes_cli.main import _model_flow_api_key_provider

            _model_flow_api_key_provider(load_config(), "venice", "old-model")

        import yaml
        from hermes_constants import get_hermes_home

        config = (
            yaml.safe_load(
                (get_hermes_home() / "config.yaml").read_text(encoding="utf-8")
            )
            or {}
        )
        model = config["model"]
        assert model["provider"] == "venice"
        assert model["default"] == "zai-org-glm-5"
        assert model["base_url"] == "https://api.venice.ai/api/v1"


def test_probe_api_models_encodes_catalog_query(monkeypatch):
    from hermes_cli import models as models_mod

    seen_urls: list[str] = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"data": [{"id": "zai-org-glm-5"}]}'

    def fake_open(request, *, timeout):
        seen_urls.append(request.full_url)
        return Response()

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", fake_open)

    result = models_mod.probe_api_models(
        "venice-test-key",
        "https://api.venice.ai/api/v1",
        query_params={"type": "text"},
    )

    assert result["models"] == ["zai-org-glm-5"]
    assert seen_urls == ["https://api.venice.ai/api/v1/models?type=text"]
