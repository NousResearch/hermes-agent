"""Tests for the bundled Azure Foundry image_gen plugin (GPT Image
deployments, three quality tiers, API-key + Entra ID auth)."""

from __future__ import annotations

import base64
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# The plugin directory uses a hyphen, which is not a valid Python identifier
# for the dotted-import form. Load it via importlib so tests don't need to
# touch sys.path or rename the directory (mirrors test_openai_codex_provider).
azure_plugin = importlib.import_module("plugins.image_gen.azure-foundry")


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("AZURE_FOUNDRY_IMAGE_API_KEY", "test-key")
    return azure_plugin.AzureFoundryImageGenProvider()


def _write_config(tmp_path, **azure_overrides):
    """Write config.yaml with an image_gen.azure_foundry block and reset the
    load_config cache so the next read sees it."""
    import yaml

    azure_cfg = {
        "endpoint": "https://my-resource.openai.azure.com",
        "deployment": "gpt-image-2",
        "auth_mode": "api_key",
    }
    azure_cfg.update(azure_overrides)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"image_gen": {"azure_foundry": azure_cfg}})
    )
    try:
        import hermes_cli.config as cfg_mod

        if hasattr(cfg_mod, "_invalidate_load_config_cache"):
            cfg_mod._invalidate_load_config_cache()
    except Exception:
        pass


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "azure-foundry"

    def test_display_name(self, provider):
        assert provider.display_name == "Azure Foundry"

    def test_default_model(self, provider):
        assert provider.default_model() == "azure-gpt-image-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == [
            "azure-gpt-image-low",
            "azure-gpt-image-medium",
            "azure-gpt-image-high",
        ]

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert entry["display"].startswith("GPT Image")
            assert entry["speed"]
            assert entry["strengths"]

    def test_get_setup_schema(self, provider):
        schema = provider.get_setup_schema()
        assert schema["name"] == "Azure Foundry"
        assert schema["badge"] == "paid"
        assert schema["env_vars"][0]["key"] == "AZURE_FOUNDRY_IMAGE_API_KEY"

    def test_capabilities_include_editing(self, provider):
        caps = provider.capabilities()
        assert caps["modalities"] == ["text", "image"]
        assert caps["max_reference_images"] == 16


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_key_no_config_unavailable(self, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_IMAGE_API_KEY", raising=False)
        assert azure_plugin.AzureFoundryImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_IMAGE_API_KEY", "test")
        assert azure_plugin.AzureFoundryImageGenProvider().is_available() is True

    def test_entra_mode_requires_azure_identity(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_IMAGE_API_KEY", raising=False)
        _write_config(tmp_path, auth_mode="entra_id")
        with patch.object(azure_plugin, "has_azure_identity_installed", return_value=True):
            assert azure_plugin.AzureFoundryImageGenProvider().is_available() is True
        with patch.object(azure_plugin, "has_azure_identity_installed", return_value=False):
            assert azure_plugin.AzureFoundryImageGenProvider().is_available() is False


# ── Config / model resolution ───────────────────────────────────────────────


class TestConfig:
    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("AZURE_FOUNDRY_IMAGE_MODEL", "azure-gpt-image-high")
        model_id, meta = azure_plugin._resolve_model()
        assert model_id == "azure-gpt-image-high"
        assert meta["quality"] == "high"

    def test_config_model_tier(self, tmp_path):
        _write_config(tmp_path, model="azure-gpt-image-low")
        model_id, meta = azure_plugin._resolve_model()
        assert model_id == "azure-gpt-image-low"
        assert meta["quality"] == "low"

    def test_default_model_fallback(self):
        model_id, meta = azure_plugin._resolve_model()
        assert model_id == "azure-gpt-image-medium"
        assert meta["quality"] == "medium"

    def test_deployment_default(self, tmp_path):
        _write_config(tmp_path)
        assert azure_plugin._resolve_deployment() == "gpt-image-2"

    def test_deployment_custom(self, tmp_path):
        _write_config(tmp_path, deployment="gpt-image-2-pro")
        assert azure_plugin._resolve_deployment() == "gpt-image-2-pro"

    def test_auth_mode_default_api_key(self):
        assert azure_plugin._resolve_auth_mode() == "api_key"

    def test_auth_mode_entra(self, tmp_path):
        _write_config(tmp_path, auth_mode="entra_id")
        assert azure_plugin._resolve_auth_mode() == "entra_id"

    @pytest.mark.parametrize(
        "endpoint,expected",
        [
            ("https://res.openai.azure.com", "https://res.openai.azure.com/openai/v1"),
            ("https://res.openai.azure.com/", "https://res.openai.azure.com/openai/v1"),
            ("https://res.openai.azure.com/openai", "https://res.openai.azure.com/openai/v1"),
            ("https://res.openai.azure.com/openai/v1", "https://res.openai.azure.com/openai/v1"),
            ("", ""),
        ],
    )
    def test_build_base_url(self, endpoint, expected):
        assert azure_plugin._build_base_url(endpoint) == expected


# ── Generate ────────────────────────────────────────────────────────────────


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_endpoint(self, provider):
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "endpoint" in result["error"]

    def test_missing_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_IMAGE_API_KEY", raising=False)
        _write_config(tmp_path)
        result = azure_plugin.AzureFoundryImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        _write_config(tmp_path)
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "azure-gpt-image-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "azure-foundry"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # All tiers hit the single configured Azure deployment.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # GPT Image rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    def test_client_uses_azure_base_url(self, provider, tmp_path):
        _write_config(tmp_path)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_client_holder = {}

        def _capture_client(*args, **kwargs):
            fake_client_holder["kwargs"] = kwargs
            return fake_client

        fake_openai.OpenAI.side_effect = _capture_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        assert fake_client_holder["kwargs"]["base_url"] == (
            "https://my-resource.openai.azure.com/openai/v1"
        )
        assert fake_client_holder["kwargs"]["api_key"] == "test-key"

    @pytest.mark.parametrize(
        "tier,expected_quality",
        [
            ("azure-gpt-image-low", "low"),
            ("azure-gpt-image-medium", "medium"),
            ("azure-gpt-image-high", "high"),
        ],
    )
    def test_tier_maps_to_quality(self, provider, tmp_path, monkeypatch, tier, expected_quality):
        _write_config(tmp_path)
        monkeypatch.setenv("AZURE_FOUNDRY_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        # Always the same underlying Azure deployment regardless of tier.
        assert fake_client.images.generate.call_args.kwargs["model"] == "gpt-image-2"

    @pytest.mark.parametrize(
        "aspect,expected_size",
        [
            ("landscape", "1536x1024"),
            ("square", "1024x1024"),
            ("portrait", "1024x1536"),
        ],
    )
    def test_aspect_ratio_mapping(self, provider, tmp_path, aspect, expected_size):
        _write_config(tmp_path)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_revised_prompt_passed_through(self, provider, tmp_path):
        _write_config(tmp_path)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(), revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"
        assert result["deployment"] == "gpt-image-2"

    def test_entra_id_uses_token_provider(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AZURE_FOUNDRY_IMAGE_API_KEY", raising=False)
        _write_config(tmp_path, auth_mode="entra_id")
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        captured = {}

        def _capture_client(*args, **kwargs):
            captured.update(kwargs)
            return fake_client

        fake_openai.OpenAI.side_effect = _capture_client
        fake_token_provider = MagicMock(return_value="fake-entra-token")

        with patch.dict("sys.modules", {"openai": fake_openai}), patch.object(
            azure_plugin, "build_token_provider", return_value=fake_token_provider
        ):
            result = azure_plugin.AzureFoundryImageGenProvider().generate("a cat")

        assert result["success"] is True
        # The SDK receives the callable (it mints a fresh JWT per request).
        assert captured["api_key"] is fake_token_provider
        assert captured["base_url"] == "https://my-resource.openai.azure.com/openai/v1"

    def test_edit_routes_to_images_edit(self, provider, tmp_path):
        _write_config(tmp_path)
        png_bytes = bytes.fromhex(_PNG_HEX)
        src = tmp_path / "src.png"
        src.write_bytes(png_bytes)
        fake_client = MagicMock()
        fake_client.images.edit.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("make it blue", image_url=str(src))

        assert result["success"] is True
        assert result["modality"] == "image"
        call_kwargs = fake_client.images.edit.call_args.kwargs
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"

    def test_api_error_surfaces(self, provider, tmp_path):
        _write_config(tmp_path)
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]
