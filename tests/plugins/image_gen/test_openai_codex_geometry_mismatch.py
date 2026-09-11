"""Regression coverage for #108171: aspect_ratio mismatch reporting.

The Codex image_generation backend returns landscape geometry regardless of
the requested `size`, and unrecognized model names silently produce an image
from a default tier. The provider now infers actual geometry from pixel_size
and flags a mismatch when they differ so callers can detect the discrepancy.
"""

from __future__ import annotations

import importlib
import base64

import pytest

codex_plugin = importlib.import_module("plugins.image_gen.openai-codex")

_1536x1024_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000006000000040008060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)

_1024x1536_PNG_HEX = (
    "89504e470d0a1a0a0000000d49484452000004000000060008060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return codex_plugin.OpenAICodexImageGenProvider()


def test_portrait_request_that_returns_landscape_reports_mismatch(provider, monkeypatch):
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    
    landscape_b64 = base64.b64encode(bytes.fromhex(_1536x1024_PNG_HEX)).decode()
    monkeypatch.setattr(
        codex_plugin, "_collect_image_b64", lambda *a, **kw: {"b64": landscape_b64, "source": "final"}
    )

    result = provider.generate("a red circle", aspect_ratio="portrait")

    assert result["success"] is True
    assert result["requested_aspect_ratio"] == "portrait"
    assert result["aspect_ratio"] == "landscape"
    assert result["geometry_mismatch"] is True
    assert result["pixel_size"] == "1536x1024"


def test_matching_aspect_reports_no_mismatch(provider, monkeypatch):
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    
    landscape_b64 = base64.b64encode(bytes.fromhex(_1536x1024_PNG_HEX)).decode()
    monkeypatch.setattr(
        codex_plugin, "_collect_image_b64", lambda *a, **kw: {"b64": landscape_b64, "source": "final"}
    )

    result = provider.generate("a red circle", aspect_ratio="landscape")

    assert result["success"] is True
    assert result["requested_aspect_ratio"] == "landscape"
    assert result["aspect_ratio"] == "landscape"
    assert result["geometry_mismatch"] is False


def test_square_request_that_returns_landscape_reports_mismatch(provider, monkeypatch):
    monkeypatch.setattr(codex_plugin, "_read_codex_access_token", lambda: "codex-token")
    
    landscape_b64 = base64.b64encode(bytes.fromhex(_1536x1024_PNG_HEX)).decode()
    monkeypatch.setattr(
        codex_plugin, "_collect_image_b64", lambda *a, **kw: {"b64": landscape_b64, "source": "final"}
    )

    result = provider.generate("a red circle", aspect_ratio="square")

    assert result["success"] is True
    assert result["requested_aspect_ratio"] == "square"
    assert result["aspect_ratio"] == "landscape"
    assert result["geometry_mismatch"] is True


def test_infer_portrait_from_1024x1536():
    assert codex_plugin._infer_aspect_from_pixel_size("1024x1536") == "portrait"


def test_infer_landscape_from_1536x1024():
    assert codex_plugin._infer_aspect_from_pixel_size("1536x1024") == "landscape"


def test_infer_square_from_1024x1024():
    assert codex_plugin._infer_aspect_from_pixel_size("1024x1024") == "square"


def test_infer_defaults_to_landscape_on_parse_error():
    assert codex_plugin._infer_aspect_from_pixel_size("broken") == "landscape"
