"""Tests for the bundled Codex-backed image_gen plugin."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.codex as codex_plugin


# 1×1 transparent PNG — valid bytes for save_b64_image()
_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


def _b64_png() -> str:
    import base64

    return base64.b64encode(bytes.fromhex(_PNG_HEX)).decode()


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider():
    return codex_plugin.CodexImageGenProvider()


@pytest.fixture
def fake_creds():
    return {
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_key": "chatgpt-access-token",
        "auth_mode": "chatgpt",
    }


def _patched_openai(fake_client: MagicMock):
    fake_openai = MagicMock()
    fake_openai.OpenAI.return_value = fake_client
    return patch.dict("sys.modules", {"openai": fake_openai})


def _fake_response(*, result=None, revised_prompt=None, item_type="image_generation_call"):
    item = SimpleNamespace(type=item_type, result=result, revised_prompt=revised_prompt)
    return SimpleNamespace(output=[item])


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "codex"

    def test_default_model(self, provider):
        assert provider.default_model() == codex_plugin.DEFAULT_MODEL == "gpt-5.4-medium"

    def test_list_models_three_tiers(self, provider):
        models = provider.list_models()
        assert [m["id"] for m in models] == [
            "gpt-5.4-low",
            "gpt-5.4-medium",
            "gpt-5.4-high",
        ]
        assert all(m["display"] for m in models)


class TestAvailability:
    def test_unavailable_without_codex_auth(self, monkeypatch):
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(side_effect=RuntimeError("missing auth")),
        )
        assert codex_plugin.CodexImageGenProvider().is_available() is False

    def test_available_with_codex_auth(self, monkeypatch, fake_creds):
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        assert codex_plugin.CodexImageGenProvider().is_available() is True


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_codex_auth(self, monkeypatch, provider):
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(side_effect=RuntimeError("not logged in")),
        )
        result = provider.generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"
        assert "Codex" in result["error"]

    def test_success_saves_b64_result(self, provider, monkeypatch, fake_creds, tmp_path):
        fake_client = MagicMock()
        fake_stream = [
            SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
                type="image_generation_call",
                result=_b64_png(),
                revised_prompt="A cat lounging in the sun",
            )),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
        ]
        fake_client.responses.create.return_value = fake_stream
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        monkeypatch.setattr(codex_plugin, "_codex_cloudflare_headers", lambda token: {"originator": "codex_cli_rs"})

        with _patched_openai(fake_client) as patched_modules:
            result = provider.generate("a cat", aspect_ratio="portrait")
            fake_openai = patched_modules["openai"]

        assert result["success"] is True
        assert result["provider"] == "codex"
        assert result["model"] == codex_plugin.DEFAULT_MODEL == "gpt-5.4-medium"
        assert result["tool_model"] == "gpt-image-2"
        assert result["aspect_ratio"] == "portrait"
        assert result["revised_prompt"] == "A cat lounging in the sun"

        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == bytes.fromhex(_PNG_HEX)

        fake_openai.OpenAI.assert_called_once_with(
            api_key="chatgpt-access-token",
            base_url="https://chatgpt.com/backend-api/codex",
            default_headers={"originator": "codex_cli_rs"},
        )
        call_kwargs = fake_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4"
        assert call_kwargs["instructions"] == "You are a helpful assistant."
        assert call_kwargs["input"] == [{"role": "user", "content": "a cat"}]
        assert call_kwargs["stream"] is True
        assert call_kwargs["store"] is False
        assert call_kwargs["tools"] == [{
            "type": "image_generation",
            "output_format": "png",
            "quality": "medium",
        }]

    def test_dict_output_item_supported(self, provider, monkeypatch, fake_creds):
        fake_client = MagicMock()
        fake_client.responses.create.return_value = [
            SimpleNamespace(type="response.output_item.done", item={
                "type": "image_generation_call",
                "result": _b64_png(),
                "revised_prompt": "A nice cat",
            }),
            SimpleNamespace(type="response.completed", response={"output": []}),
        ]
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        monkeypatch.setattr(codex_plugin, "_codex_cloudflare_headers", lambda token: {})

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["revised_prompt"] == "A nice cat"

    @pytest.mark.parametrize(
        ("tier", "quality"),
        [
            ("gpt-5.4-low", "low"),
            ("gpt-5.4-medium", "medium"),
            ("gpt-5.4-high", "high"),
        ],
    )
    def test_tier_maps_to_quality(self, provider, monkeypatch, fake_creds, tier, quality):
        fake_client = MagicMock()
        fake_client.responses.create.return_value = [
            SimpleNamespace(type="response.output_item.done", item=SimpleNamespace(
                type="image_generation_call",
                result=_b64_png(),
                revised_prompt="tiered prompt",
            )),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
        ]
        monkeypatch.setenv("CODEX_IMAGE_MODEL", tier)
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        monkeypatch.setattr(codex_plugin, "_codex_cloudflare_headers", lambda token: {})

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is True
        assert result["model"] == tier
        assert fake_client.responses.create.call_args.kwargs["tools"] == [{
            "type": "image_generation",
            "output_format": "png",
            "quality": quality,
        }]

    def test_api_error_returns_error_response(self, provider, monkeypatch, fake_creds):
        fake_client = MagicMock()
        fake_client.responses.create.side_effect = RuntimeError("boom")
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        monkeypatch.setattr(codex_plugin, "_codex_cloudflare_headers", lambda token: {})

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_missing_image_output_returns_empty_response(self, provider, monkeypatch, fake_creds):
        fake_client = MagicMock()
        fake_client.responses.create.return_value = [
            SimpleNamespace(type="response.completed", response=SimpleNamespace(output=[])),
        ]
        monkeypatch.setattr(
            codex_plugin,
            "resolve_codex_runtime_credentials",
            MagicMock(return_value=fake_creds),
        )
        monkeypatch.setattr(codex_plugin, "_codex_cloudflare_headers", lambda token: {})

        with _patched_openai(fake_client):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"
