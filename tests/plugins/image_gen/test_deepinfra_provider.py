"""Tests for the bundled DeepInfra image_gen plugin.

Invariants only — no snapshots of specific model ids. Most surface-level
contracts (network-failure → empty list, tag filtering, no-model error)
are covered by the shared tag-filter test in
``tests/hermes_cli/test_api_key_providers.py``; these two tests pin the
plugin-specific bits that wrapper doesn't reach.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import plugins.image_gen.deepinfra as deepinfra_plugin


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
def _isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.models as _models_mod
    monkeypatch.setattr(_models_mod, "_deepinfra_catalog_cache", {})
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-key")
    yield


def test_list_models_filters_by_image_gen_tag(monkeypatch):
    """Plugin-side wiring: list_models() returns only ``image-gen``-tagged
    catalog entries and surfaces pricing + default dims when present."""
    import json
    import hermes_cli.models as models

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return json.dumps({"data": [
                {"id": "vendor/chat", "metadata": {"tags": ["chat"]}},
                {"id": "vendor/img", "metadata": {
                    "tags": ["image-gen"],
                    "pricing": {"per_image_unit": 0.005},
                    "default_width": 1024,
                }},
            ]}).encode()

    monkeypatch.setattr(
        models, "_urlopen_model_catalog_request", lambda *a, **kw: _Resp()
    )
    rows = deepinfra_plugin.DeepInfraImageGenProvider().list_models()
    ids = {row["id"] for row in rows}
    assert ids == {"vendor/img"}
    img = next(row for row in rows if row["id"] == "vendor/img")
    assert "price" in img and img["default_width"] == 1024


def test_generate_calls_openai_sdk_with_deepinfra_base_url(monkeypatch):
    """Happy path: pinned model → openai SDK called with DeepInfra
    base_url + Bearer key → b64 saved to cache."""
    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    captured: dict = {}

    class _FakeImages:
        def generate(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(data=[SimpleNamespace(b64_json=_b64_png(), url=None)])

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.images = _FakeImages()

    fake_openai = MagicMock()
    fake_openai.OpenAI = _FakeClient
    with patch.dict("sys.modules", {"openai": fake_openai}):
        result = deepinfra_plugin.DeepInfraImageGenProvider().generate(
            prompt="a cat", aspect_ratio="square",
        )

    assert result["success"] is True
    assert "deepinfra" in captured["base_url"]
    assert captured["api_key"] == "test-key"
    assert captured["kwargs"]["model"] == "vendor/test-img"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_url": "https://example.com/source.png"},
        {"reference_image_urls": ["https://example.com/reference.png"]},
    ],
)
def test_generate_rejects_unsupported_edit_inputs_without_calling_sdk(
    monkeypatch, kwargs
):
    monkeypatch.setenv("DEEPINFRA_IMAGE_MODEL", "vendor/test-img")
    fake_openai = MagicMock()
    with patch.dict("sys.modules", {"openai": fake_openai}):
        result = deepinfra_plugin.DeepInfraImageGenProvider().generate(
            prompt="edit this", **kwargs
        )

    assert result["success"] is False
    assert result["error_type"] == "modality_unsupported"
    assert result["provider"] == "deepinfra"
    fake_openai.OpenAI.assert_not_called()


def test_capabilities_advertise_text_to_image_only():
    assert deepinfra_plugin.DeepInfraImageGenProvider().capabilities() == {
        "modalities": ["text"],
        "max_reference_images": 0,
    }
