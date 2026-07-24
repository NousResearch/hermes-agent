"""Tests for the bundled OpenAI image_gen plugin (gpt-image-2, three tiers)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.openai as openai_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64
    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


def _fake_response(*, b64=None, url=None, revised_prompt=None):
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for key in (
        "OPENAI_IMAGE_API_MODEL",
        "OPENAI_IMAGE_MODEL",
        "OPENAI_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


# ── Metadata ────────────────────────────────────────────────────────────────


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai"

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-2-medium"

    def test_list_models_three_tiers(self, provider):
        ids = [m["id"] for m in provider.list_models()]
        assert ids == ["gpt-image-2-low", "gpt-image-2-medium", "gpt-image-2-high"]

    def test_catalog_entries_have_display_speed_strengths(self, provider):
        for entry in provider.list_models():
            assert entry["display"].startswith("GPT Image 2")
            assert entry["speed"]
            assert entry["strengths"]


# ── Availability ────────────────────────────────────────────────────────────


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


# ── Model resolution ────────────────────────────────────────────────────────


class TestModelResolution:
    def test_default_is_medium(self):
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-medium"
        assert meta["quality"] == "medium"

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "gpt-image-2-high")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_env_var_unknown_falls_back(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "bogus-tier")
        model_id, _ = openai_plugin._resolve_model()
        assert model_id == openai_plugin.DEFAULT_MODEL

    def test_config_openai_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "gpt-image-2-low"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"

    def test_config_top_level_model(self, tmp_path):
        """``image_gen.model: gpt-image-2-high`` also works (top-level)."""
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "gpt-image-2-high"}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "gpt-image-2-high"
        assert meta["quality"] == "high"

    def test_requested_tier_from_dispatcher(self):
        model_id, meta = openai_plugin._resolve_model("gpt-image-2-low")
        assert model_id == "gpt-image-2-low"
        assert meta["quality"] == "low"

    def test_requested_non_tier_model_does_not_change_quality_tier(self):
        model_id, meta = openai_plugin._resolve_model("third-party-image-model")
        assert model_id == openai_plugin.DEFAULT_MODEL
        assert meta["quality"] == "medium"


class TestApiModelResolution:
    def test_default_api_model(self):
        assert openai_plugin._resolve_api_model() == "gpt-image-2"

    def test_env_var_api_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_API_MODEL", "krill-image-model")
        assert openai_plugin._resolve_api_model() == "krill-image-model"

    def test_config_openai_api_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"api_model": "cfg-image-model"}}})
        )
        assert openai_plugin._resolve_api_model() == "cfg-image-model"

    def test_requested_non_tier_model_becomes_api_model(self):
        assert openai_plugin._resolve_api_model("third-party-image-model") == "third-party-image-model"

    def test_requested_tier_does_not_become_api_model(self):
        assert openai_plugin._resolve_api_model("gpt-image-2-high") == "gpt-image-2"

    def test_top_level_non_tier_config_becomes_api_model(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"model": "third-party-image-model"}})
        )
        assert openai_plugin._resolve_api_model() == "third-party-image-model"

    def test_config_base_url(self, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"base_url": "https://example.test/v1"}}})
        )
        assert openai_plugin._resolve_base_url() == "https://example.test/v1"

    def test_env_base_url_beats_config(self, tmp_path, monkeypatch):
        import yaml
        monkeypatch.setenv("OPENAI_BASE_URL", "https://env.example.test/v1")
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"base_url": "https://cfg.example.test/v1"}}})
        )
        assert openai_plugin._resolve_base_url() == "https://env.example.test/v1"


# ── Generate ────────────────────────────────────────────────────────────────


class TestSourceImageLoading:
    def test_load_image_bytes_blocks_credential_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        with pytest.raises(ValueError, match="credential store"):
            openai_plugin._load_image_bytes(str(auth_json))

    def test_load_image_bytes_never_opens_blocked_credential(self, tmp_path, monkeypatch):
        """The guard must fire BEFORE the file is opened — a credential store
        must never be read into memory (#57698). Spy builtins.open and assert
        it is never called for the blocked path."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        auth_json = hermes_home / "auth.json"
        auth_json.write_text('{"api_key":"sk-secret"}', encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        import builtins

        real_open = builtins.open
        opened: list = []

        def _spy_open(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)

        monkeypatch.setattr(builtins, "open", _spy_open)
        with pytest.raises(ValueError, match="credential store"):
            openai_plugin._load_image_bytes(str(auth_json))
        assert str(auth_json) not in opened, "blocked credential must never be opened"

    def test_load_image_bytes_allows_legit_local_image(self, tmp_path, monkeypatch):
        """Negative control: a legitimate local image path is NOT blocked and
        loads normally — proves the guard doesn't over-fire on everything."""
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        img = tmp_path / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfake-image-bytes")

        data, name = openai_plugin._load_image_bytes(str(img))
        assert data == b"\x89PNG\r\n\x1a\nfake-image-bytes"
        assert name == "pic.png"

    def test_load_image_bytes_passthrough_data_uri_not_blocked(self, tmp_path, monkeypatch):
        """Negative control: data: URIs are decoded, never routed through the
        local-path guard (the guard only applies to local file reads)."""
        import base64

        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        b64 = base64.b64encode(b"xyz").decode("ascii")
        data, name = openai_plugin._load_image_bytes(f"data:image/png;base64,{b64}")
        assert data == b"xyz"
        assert name.endswith(".png")


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_b64_saves_to_cache(self, provider, tmp_path):
        png_bytes = bytes.fromhex(_PNG_HEX)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-2-medium"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"
        assert result["quality"] == "medium"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        call_kwargs = fake_client.images.generate.call_args.kwargs
        # All tiers hit the single underlying API model.
        assert call_kwargs["model"] == "gpt-image-2"
        assert call_kwargs["quality"] == "medium"
        assert call_kwargs["size"] == "1536x1024"
        # gpt-image-2 rejects response_format — we must NOT send it.
        assert "response_format" not in call_kwargs

    @pytest.mark.parametrize("tier,expected_quality", [
        ("gpt-image-2-low", "low"),
        ("gpt-image-2-medium", "medium"),
        ("gpt-image-2-high", "high"),
    ])
    def test_tier_maps_to_quality(self, provider, monkeypatch, tier, expected_quality):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", tier)
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["model"] == tier
        assert result["quality"] == expected_quality
        assert fake_client.images.generate.call_args.kwargs["quality"] == expected_quality
        # Always the same underlying API model regardless of tier.
        assert fake_client.images.generate.call_args.kwargs["model"] == "gpt-image-2"

    def test_requested_non_tier_model_sets_api_model(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", model="third-party-image-model")

        assert result["model"] == "gpt-image-2-medium"
        assert result["quality"] == "medium"
        assert fake_client.images.generate.call_args.kwargs["model"] == "third-party-image-model"
        assert fake_client.images.generate.call_args.kwargs["quality"] == "medium"

    def test_requested_tier_model_sets_quality_not_api_model(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            result = provider.generate("a cat", model="gpt-image-2-high")

        assert result["model"] == "gpt-image-2-high"
        assert result["quality"] == "high"
        assert fake_client.images.generate.call_args.kwargs["model"] == "gpt-image-2"
        assert fake_client.images.generate.call_args.kwargs["quality"] == "high"

    def test_configured_base_url_passed_to_openai_client(self, provider, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"base_url": "https://cfg.example.test/v1"}}})
        )
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            provider.generate("a cat")

        fake_openai.OpenAI.assert_called_once_with(base_url="https://cfg.example.test/v1")

    def test_client_initialization_error_returns_error_response(self, provider):
        fake_openai = MagicMock()
        fake_openai.OpenAI.side_effect = ValueError("invalid base URL")

        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "invalid base URL" in result["error"]

    def test_edit_uses_configured_api_model_and_base_url(
        self, provider, tmp_path, monkeypatch
    ):
        import yaml

        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "image_gen": {
                        "openai": {
                            "api_model": "configured-edit-model",
                            "base_url": "https://images.example.test/v1",
                        }
                    }
                }
            )
        )
        source = tmp_path / "source.png"
        source.write_bytes(bytes.fromhex(_PNG_HEX))

        fake_client = MagicMock()
        fake_client.images.edit.return_value = _fake_response(b64=_b64_png())
        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client

        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate(
                "edit this cat",
                image_url=str(source),
                model="dispatcher-image-model",
            )

        assert result["success"] is True
        assert result["modality"] == "image"
        assert result["model"] == openai_plugin.DEFAULT_MODEL
        assert result["quality"] == "medium"
        fake_openai.OpenAI.assert_called_once_with(
            base_url="https://images.example.test/v1"
        )
        assert fake_client.images.edit.call_args.kwargs["model"] == "configured-edit-model"
        fake_client.images.generate.assert_not_called()

    @pytest.mark.parametrize("aspect,expected_size", [
        ("landscape", "1536x1024"),
        ("square", "1024x1024"),
        ("portrait", "1024x1536"),
    ])
    def test_aspect_ratio_mapping(self, provider, aspect, expected_size):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=_b64_png())

        with _patched_openai(fake_client):
            provider.generate("a cat", aspect_ratio=aspect)

        assert fake_client.images.generate.call_args.kwargs["size"] == expected_size

    def test_revised_prompt_passed_through(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=_b64_png(), revised_prompt="A photo of a cat",
        )

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["revised_prompt"] == "A photo of a cat"

    def test_api_error_returns_error_response(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_empty_response_data(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = SimpleNamespace(data=[])

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_url_response_is_cached_locally(self, provider):
        """OpenAI URL response (if API ever returns one) is cached locally.

        Pre-fix this asserted the bare URL passed through; symmetric to the
        xAI #26942 fix.  Even though gpt-image-2 returns b64 today, every
        ``image_gen`` provider must guarantee the gateway gets a stable
        file path so ephemeral signed URLs can't expire mid-flight.
        """
        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            return_value=Path("/tmp/openai_gpt-image-2_20260524_000000_deadbeef.png"),
        ) as mock_save_url:
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"].startswith("/")
        assert "example.com" not in result["image"]
        mock_save_url.assert_called_once()

    def test_url_response_falls_back_to_bare_url_when_download_fails(self, provider):
        """Cache failure must not turn into a tool error — symmetric with xAI."""
        import requests as req_lib

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            b64=None, url="https://example.com/img.png",
        )

        with _patched_openai(fake_client), patch(
            "plugins.image_gen.openai.save_url_image",
            side_effect=req_lib.HTTPError("404 from CDN"),
        ):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["image"] == "https://example.com/img.png"
