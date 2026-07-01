#!/usr/bin/env python3
"""Tests for the MiniMax image generation provider.

Covers:
- is_available() with/without MINIMAX_API_KEY
- list_models() returns the 3 virtual model IDs
- default_model() returns minimax-image-01
- get_setup_schema() includes both env vars (GroupId marked optional)
- _resolve_model() respects env, config, defaults
- _resolve_aspect() honors the -square alias
- generate() with monkeypatched requests.post: success path, error paths
- generate() handles missing MINIMAX_API_KEY, empty prompt, bad n

The live API is NOT called — ``requests.post`` is monkeypatched in every
test that would otherwise hit the network. See the bottom-of-file
``test_live_image_generation`` for an opt-in live test (skipped by
default to keep CI deterministic).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch):
    """Ensure MINIMAX_API_KEY is set for all tests by default."""
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key-12345")


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    """Wipe config-influencing env vars between tests."""
    for var in (
        "MINIMAX_IMAGE_MODEL",
        "MINIMAX_GROUP_ID",
    ):
        monkeypatch.delenv(var, raising=False)


def _ok_url_response(url: str = "https://minimax.cdn/img.jpeg"):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "id": "test-id-abc",
        "data": {"image_urls": [url]},
        "metadata": {"failed_count": "0", "success_count": "1"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    return resp


def _ok_b64_response(b64: str = "aGVsbG8="):
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "id": "test-id-abc",
        "data": {"b64_json": [b64]},
        "metadata": {"failed_count": "0", "success_count": "1"},
        "base_resp": {"status_code": 0, "status_msg": "success"},
    }
    return resp


def _err_response(status_code: int, msg: str = "GroupId is required"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "base_resp": {"status_code": status_code, "status_msg": msg},
        "message": msg,
    }
    return resp


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider().is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        assert MiniMaxImageGenProvider().is_available() is False


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------


class TestModelCatalog:
    def test_list_models(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        provider = MiniMaxImageGenProvider()
        models = provider.list_models()
        assert [m["id"] for m in models] == [
            "minimax-image-01",
            "minimax-image-01-live",
            "minimax-image-01-square",
        ]
        for m in models:
            assert "display" in m
            assert "speed" in m
            assert "strengths" in m

    def test_default_model(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider, DEFAULT_MODEL

        assert MiniMaxImageGenProvider().default_model() == DEFAULT_MODEL
        assert DEFAULT_MODEL == "minimax-image-01"

    def test_get_setup_schema(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        schema = MiniMaxImageGenProvider().get_setup_schema()
        assert schema["name"] == "MiniMax"
        env_keys = [v["key"] for v in schema["env_vars"]]
        assert "MINIMAX_API_KEY" in env_keys
        assert "MINIMAX_GROUP_ID" in env_keys
        # GroupId should be marked optional
        group_id_var = next(
            v for v in schema["env_vars"] if v["key"] == "MINIMAX_GROUP_ID"
        )
        assert group_id_var.get("required") is False


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_IMAGE_MODEL", "minimax-image-01-live")
        from plugins.image_gen.minimax import _resolve_model

        model_id, meta = _resolve_model()
        assert model_id == "minimax-image-01-live"
        assert meta["api_model"] == "image-01-live"

    def test_default_when_no_config(self):
        from plugins.image_gen.minimax import _resolve_model, DEFAULT_MODEL

        model_id, _ = _resolve_model()
        assert model_id == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# _resolve_aspect
# ---------------------------------------------------------------------------


class TestResolveAspect:
    def test_square_alias_forces_1_1(self):
        from plugins.image_gen.minimax import _resolve_aspect

        assert _resolve_aspect("minimax-image-01-square", "landscape") == "1:1"
        assert _resolve_aspect("minimax-image-01-square", "portrait") == "1:1"
        assert _resolve_aspect("minimax-image-01-square", "square") == "1:1"

    def test_other_models_honor_kwarg(self):
        from plugins.image_gen.minimax import _resolve_aspect

        assert _resolve_aspect("minimax-image-01", "landscape") == "16:9"
        assert _resolve_aspect("minimax-image-01", "portrait") == "9:16"
        assert _resolve_aspect("minimax-image-01", "square") == "1:1"
        assert _resolve_aspect("minimax-image-01-live", "landscape") == "16:9"

    def test_unknown_aspect_defaults_to_square(self):
        from plugins.image_gen.minimax import _resolve_aspect

        assert _resolve_aspect("minimax-image-01", "weird") == "1:1"


# ---------------------------------------------------------------------------
# generate() — input validation
# ---------------------------------------------------------------------------


class TestGenerateInputValidation:
    def test_empty_prompt(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"
        assert "non-empty" in result["error"]

    def test_whitespace_only_prompt(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("   \n  ", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_invalid_n(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape", n=0)
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

        result = MiniMaxImageGenProvider().generate("a cat", "landscape", n=5)
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_invalid_response_format(self):
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate(
            "a cat", "landscape", response_format="json"
        )
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"


# ---------------------------------------------------------------------------
# generate() — request shape
# ---------------------------------------------------------------------------


class TestGenerateRequest:
    def test_request_includes_correct_url_no_groupid(self, monkeypatch):
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers", {})
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        MiniMaxImageGenProvider().generate("a tiny red square", "square")

        assert captured["url"] == "https://api.minimax.io/v1/image_generation"
        assert captured["json"]["model"] == "image-01"
        assert captured["json"]["prompt"] == "a tiny red square"
        assert captured["json"]["aspect_ratio"] == "1:1"
        assert captured["json"]["n"] == 1
        assert captured["json"]["response_format"] == "url"
        # No GroupId query string
        assert "GroupId" not in captured["url"]
        # Bearer auth header
        assert captured["headers"]["Authorization"] == "Bearer test-key-12345"

    def test_request_appends_groupid_when_set(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_GROUP_ID", "grp_abc")
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        MiniMaxImageGenProvider().generate("a cat", "landscape")

        assert captured["url"].endswith("?GroupId=grp_abc")

    def test_request_uses_live_api_model(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_IMAGE_MODEL", "minimax-image-01-live")
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        MiniMaxImageGenProvider().generate("a cat", "landscape")

        assert captured["json"]["model"] == "image-01-live"

    def test_request_uses_square_when_alias_selected(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_IMAGE_MODEL", "minimax-image-01-square")
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        # Even when caller asks for landscape, the -square alias forces 1:1
        MiniMaxImageGenProvider().generate("a cat", "landscape")

        assert captured["json"]["model"] == "image-01"
        assert captured["json"]["aspect_ratio"] == "1:1"

    def test_seed_passthrough(self, monkeypatch):
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        MiniMaxImageGenProvider().generate("a cat", "landscape", seed=42)
        assert captured["json"]["seed"] == 42

    def test_prompt_optimizer_and_watermark_passthrough(self, monkeypatch):
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _ok_url_response()

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        MiniMaxImageGenProvider().generate(
            "a cat", "landscape", prompt_optimizer=True, aigc_watermark=True
        )
        assert captured["json"]["prompt_optimizer"] is True
        assert captured["json"]["aigc_watermark"] is True


# ---------------------------------------------------------------------------
# generate() — success path
# ---------------------------------------------------------------------------


class TestGenerateSuccess:
    def test_url_response_saves_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "plugins.image_gen.minimax.requests.post",
            lambda *a, **kw: _ok_url_response(),
        )
        # Mock the URL cache to avoid real network in tests
        monkeypatch.setattr(
            "plugins.image_gen.minimax.save_url_image",
            lambda url, prefix: tmp_path / f"{prefix}_cached.jpeg",
        )
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")

        assert result["success"] is True
        assert result["provider"] == "minimax"
        assert result["model"] == "minimax-image-01"
        assert result["aspect_ratio"] == "landscape"
        # Image was cached to disk (not the bare URL)
        assert result["image"] == str(tmp_path / "minimax_minimax-image-01_cached.jpeg")
        # Extra diagnostic fields populated
        assert result["minimax_aspect"] == "16:9"
        assert result["api_model"] == "image-01"

    def test_url_response_falls_back_to_bare_url_when_cache_fails(self, monkeypatch, tmp_path):
        """If the URL fetch fails (network error, expired CDN URL, etc.),
        the provider should still return success with the bare URL rather
        than failing the whole call — same defensive pattern as the xAI
        provider (see test_xai_provider for the original rationale)."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "plugins.image_gen.minimax.requests.post",
            lambda *a, **kw: _ok_url_response(),
        )

        def _failing_cache(url, prefix):
            raise IOError("simulated network failure")

        monkeypatch.setattr("plugins.image_gen.minimax.save_url_image", _failing_cache)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is True
        assert result["image"] == "https://minimax.cdn/img.jpeg"  # bare URL fallback

    def test_b64_response_saves_to_cache(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(
            "plugins.image_gen.minimax.requests.post",
            lambda *a, **kw: _ok_b64_response(),
        )
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "square")
        assert result["success"] is True
        assert result["image"].startswith(str(tmp_path))
        # 8 bytes of "hello" — verify the bytes were actually written
        written = Path(result["image"]).read_bytes()
        assert written == b"hello"


# ---------------------------------------------------------------------------
# generate() — error paths
# ---------------------------------------------------------------------------


class TestGenerateErrors:
    def test_http_401_returns_auth_error(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.image_gen.minimax.requests.post",
            lambda *a, **kw: _err_response(401, "invalid api key"),
        )
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "401" in result["error"]
        assert "invalid api key" in result["error"]

    def test_http_404_with_missing_groupid_message(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.image_gen.minimax.requests.post",
            lambda *a, **kw: _err_response(404, "GroupId is required"),
        )
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert "GroupId" in result["error"]

    def test_empty_data_array(self, monkeypatch):
        def fake_post(*a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "data": {},
                "base_resp": {"status_code": 100, "status_msg": "no data"},
            }
            return resp

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"
        assert "no data" in result["error"]

    def test_response_with_neither_url_nor_b64(self, monkeypatch):
        def fake_post(*a, **kw):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "data": {"image_urls": [], "b64_json": []},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            }
            return resp

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "empty_response"

    def test_timeout(self, monkeypatch):
        import requests

        def fake_post(*a, **kw):
            raise requests.Timeout("read timed out")

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "timeout"

    def test_connection_error(self, monkeypatch):
        import requests

        def fake_post(*a, **kw):
            raise requests.ConnectionError("dns failure")

        monkeypatch.setattr("plugins.image_gen.minimax.requests.post", fake_post)
        from plugins.image_gen.minimax import MiniMaxImageGenProvider

        result = MiniMaxImageGenProvider().generate("a cat", "landscape")
        assert result["success"] is False
        assert result["error_type"] == "api_error"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    def test_register_wires_provider(self):
        from plugins.image_gen.minimax import register, MiniMaxImageGenProvider
        from agent import image_gen_registry

        ctx = MagicMock()
        register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        provider = ctx.register_image_gen_provider.call_args[0][0]
        assert isinstance(provider, MiniMaxImageGenProvider)
        assert provider.name == "minimax"
        # Sanity check the actual registry hook works (not just the mock)
        image_gen_registry.register_provider(provider)
        assert image_gen_registry.get_provider("minimax") is provider
        # Clean up so other tests aren't affected
        from agent.image_gen_registry import _providers as _reg
        _reg.pop("minimax", None)


# ---------------------------------------------------------------------------
# Live test (skipped by default; opt in with ``RUN_LIVE_MINIMAX=1``)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MINIMAX") != "1",
    reason="Set RUN_LIVE_MINIMAX=1 to run a real API call against MiniMax",
)
def test_live_image_generation(monkeypatch, tmp_path):
    """Real API call — only runs when RUN_LIVE_MINIMAX=1 is set.

    Useful for sanity-checking the plugin against a real MiniMax account.
    Expects ``MINIMAX_API_KEY`` (and optionally ``MINIMAX_GROUP_ID``) to be
    set in the **caller's shell environment** before pytest starts. The
    autouse ``_fake_api_key`` fixture overwrites that key for unit tests;
    we restore the real one here by reading the value that was set before
    pytest's fixtures ran. ``os.environ`` is a snapshot of the shell at
    process start, so once a fixture mutates a key, the original is gone
    — we have to use the framework-supplied monkeypatch to recover it
    via a sentinel approach: ask the user to pass the real key as a
    different env var (``MINIMAX_API_KEY_LIVE``) when running live.
    """
    # The autouse _fake_api_key fixture overwrites MINIMAX_API_KEY with
    # "test-key-12345"; we need the real key. Read it from a side-channel
    # env var that the caller sets when running live (the autouse fixture
    # doesn't touch this one). Document the contract in the test docstring.
    real_key = os.environ.get("MINIMAX_API_KEY_LIVE") or os.environ.get("MINIMAX_API_KEY", "")
    if not real_key or real_key == "test-key-12345":
        pytest.skip(
            "Live test requires MINIMAX_API_KEY_LIVE (or a real MINIMAX_API_KEY "
            "that doesn't equal the autouse fixture's 'test-key-12345')."
        )
    monkeypatch.setenv("MINIMAX_API_KEY", real_key)

    # If a real GroupId is supplied, override the autouse fixture's wipe.
    real_group = os.environ.get("MINIMAX_GROUP_ID_LIVE")
    if real_group:
        monkeypatch.setenv("MINIMAX_GROUP_ID", real_group)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.image_gen.minimax import MiniMaxImageGenProvider

    result = MiniMaxImageGenProvider().generate("a tiny red square", "square")
    assert result["success"] is True, f"Live call failed: {result}"
    assert Path(result["image"]).exists()
    assert Path(result["image"]).stat().st_size > 0
