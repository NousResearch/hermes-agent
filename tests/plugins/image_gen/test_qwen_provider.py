#!/usr/bin/env python3
"""Tests for the Wanxiang V2 (Qwen Image) provider plugin."""

from __future__ import annotations

import io, json
from unittest.mock import MagicMock, patch
import pytest
import urllib.error


@pytest.fixture(autouse=True)
def _fake_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test-key-12345")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


class TestQwenImageGenProviderSurface:
    def test_name(self):
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider().name == "qwen"

    def test_display_name(self):
        from plugins.image_gen.qwen import QwenImageProvider
        assert "Wanxiang" in QwenImageProvider().display_name

    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-xxx")
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider().is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider().is_available() is False

    def test_list_models_returns_all_models(self):
        from plugins.image_gen.qwen import QwenImageProvider, MODELS
        ids = {m["id"] for m in QwenImageProvider().list_models()}
        assert ids == set(MODELS.keys())

    def test_default_model(self):
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider().default_model() == "wan2.6-t2i"

    def test_capabilities_text_only(self):
        from plugins.image_gen.qwen import QwenImageProvider
        caps = QwenImageProvider().capabilities()
        assert caps["modalities"] == ["text"]

    def test_get_setup_schema(self):
        from plugins.image_gen.qwen import QwenImageProvider
        s = QwenImageProvider().get_setup_schema()
        assert "Wanxiang" in s["name"]
        assert s["badge"] == "paid"


class TestConfigHelpers:
    def test_resolve_model_default(self):
        from plugins.image_gen.qwen import _resolve_model
        mid, meta = _resolve_model()
        assert mid == "wan2.6-t2i"
        assert meta["sync"] is True

    def test_resolve_model_async(self):
        from plugins.image_gen.qwen import _resolve_model
        mid, meta = _resolve_model("wan2.5-t2i-preview")
        assert mid == "wan2.5-t2i-preview"
        assert meta["sync"] is False

    def test_resolve_model_invalid_fallback(self):
        from plugins.image_gen.qwen import _resolve_model
        mid, _ = _resolve_model("bogus")
        assert mid == "wan2.6-t2i"

    def test_resolve_size(self):
        from plugins.image_gen.qwen import _resolve_size
        assert _resolve_size("landscape", True) == "1696*960"
        assert _resolve_size("square", True) == "1280*1280"
        assert _resolve_size("square", False) == "1024*1024"

    def test_clamp_n(self):
        from plugins.image_gen.qwen import QwenImageProvider
        c = QwenImageProvider._clamp_n
        assert c(1) == 1
        assert c(5) == 4
        assert c(None) == 1


def _mock_sync_resp(image_url="https://oss.example.com/img.png"):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = json.dumps({
        "output": {"choices": [{"message": {"content": [{"image": image_url}]}}]}
    }).encode()
    return m


def _mock_async_submit(task_id="task-abc"):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = json.dumps({"output": {"task_id": task_id}}).encode()
    return m


def _mock_async_poll(status="SUCCEEDED", url="https://oss.example.com/img.png"):
    m = MagicMock()
    m.__enter__ = MagicMock(return_value=m)
    m.__exit__ = MagicMock(return_value=False)
    m.read.return_value = json.dumps({
        "output": {"task_status": status, "results": [{"url": url}]}
    }).encode()
    return m


class TestGenerate:
    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        from plugins.image_gen.qwen import QwenImageProvider
        r = QwenImageProvider().generate(prompt="test")
        assert r["success"] is False
        assert "DASHSCOPE_API_KEY" in r["error"]

    def test_sync_success(self):
        from plugins.image_gen.qwen import QwenImageProvider
        with patch("urllib.request.urlopen", return_value=_mock_sync_resp()):
            with patch("urllib.request.urlretrieve"):
                r = QwenImageProvider().generate(prompt="cat", aspect_ratio="square")
        assert r["success"] is True
        assert r["provider"] == "qwen"
        assert r["model"] == "wan2.6-t2i"

    def test_sync_api_error(self):
        from plugins.image_gen.qwen import QwenImageProvider
        err = urllib.error.HTTPError("url", 400, "Bad", {},
            io.BytesIO(json.dumps({"code": "BadParam", "message": "bad size"}).encode()))
        with patch("urllib.request.urlopen", side_effect=err):
            r = QwenImageProvider().generate(prompt="test")
        assert r["success"] is False
        assert r["error_type"] in ("api_error", "BadParam")

    def test_sync_empty_response(self):
        from plugins.image_gen.qwen import QwenImageProvider
        with patch("urllib.request.urlopen", return_value=_mock_sync_resp("")):
            r = QwenImageProvider().generate(prompt="test")
        assert r["success"] is False

    def test_async_success(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.image_gen.qwen._resolve_model",
            lambda mid=None: ("wan2.5-t2i-preview", {"sync": False, "display": "", "speed": "", "strengths": "", "price": ""})
        )
        from plugins.image_gen.qwen import QwenImageProvider
        calls = [0]
        def seq(req, timeout=None):
            calls[0] += 1
            if calls[0] == 1:
                return _mock_async_submit()
            return _mock_async_poll()
        with patch("urllib.request.urlopen", side_effect=seq):
            with patch("urllib.request.urlretrieve"):
                r = QwenImageProvider().generate(prompt="cat")
        assert r["success"] is True

    def test_async_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "plugins.image_gen.qwen._resolve_model",
            lambda mid=None: ("wan2.5-t2i-preview", {"sync": False, "display": "", "speed": "", "strengths": "", "price": ""})
        )
        from plugins.image_gen.qwen import QwenImageProvider
        calls = [0]
        def seq(req, timeout=None):
            calls[0] += 1
            if calls[0] == 1:
                return _mock_async_submit()
            return _mock_async_poll(status="PENDING")
        with patch("urllib.request.urlopen", side_effect=seq):
            with patch("plugins.image_gen.qwen._ASYNC_POLL_MAX_ATTEMPTS", 3):
                with patch("plugins.image_gen.qwen._ASYNC_POLL_INTERVAL", 0.01):
                    r = QwenImageProvider().generate(prompt="cat")
        assert r["success"] is False
        assert r["error_type"] == "timeout"

    def test_seed_and_n_passthrough(self):
        from plugins.image_gen.qwen import QwenImageProvider
        body = {}
        def cap(req, timeout=None):
            nonlocal body
            body = json.loads(req.data.decode())
            return _mock_sync_resp()
        with patch("urllib.request.urlopen", side_effect=cap):
            with patch("urllib.request.urlretrieve"):
                QwenImageProvider().generate(prompt="x", seed=42, n=3)
        assert body["parameters"]["seed"] == 42
        assert body["parameters"]["n"] == 3


class TestExtractImageUrl:
    def test_sync_shape(self):
        from plugins.image_gen.qwen import QwenImageProvider
        u = QwenImageProvider._extract_image_url({
            "output": {"choices": [{"message": {"content": [{"image": "https://a.png"}]}}]}
        })
        assert u == "https://a.png"

    def test_async_shape(self):
        from plugins.image_gen.qwen import QwenImageProvider
        u = QwenImageProvider._extract_image_url({
            "output": {"results": [{"url": "https://b.png"}]}
        })
        assert u == "https://b.png"

    def test_empty(self):
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider._extract_image_url({}) is None


class TestCircuitBreaker:
    def test_initially_closed(self):
        from plugins.image_gen.qwen import QwenImageProvider
        assert QwenImageProvider()._breaker_tripped() is False

    def test_trips_after_threshold(self):
        from plugins.image_gen.qwen import QwenImageProvider, _CIRCUIT_BREAKER_THRESHOLD
        p = QwenImageProvider()
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            p._record_failure()
        assert p._breaker_tripped() is True

    def test_resets_on_success(self):
        from plugins.image_gen.qwen import QwenImageProvider, _CIRCUIT_BREAKER_THRESHOLD
        p = QwenImageProvider()
        for _ in range(_CIRCUIT_BREAKER_THRESHOLD):
            p._record_failure()
        p._record_success()
        assert p._breaker_tripped() is False


class TestRegistration:
    def test_register(self):
        from plugins.image_gen.qwen import QwenImageProvider, register
        ctx = MagicMock()
        register(ctx)
        ctx.register_image_gen_provider.assert_called_once()
        assert isinstance(ctx.register_image_gen_provider.call_args[0][0], QwenImageProvider)
