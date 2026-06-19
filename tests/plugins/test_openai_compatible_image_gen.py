import base64
import importlib.util
from pathlib import Path


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "image_gen" / "openai-compatible" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("openai_compatible_image_gen_plugin", PLUGIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyResponse:
    def __init__(self, payload=None, *, headers=None, lines=None, text=""):
        self._payload = payload
        self.headers = headers or {"content-type": "application/json"}
        self._lines = lines or []
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def raise_for_status(self):
        return None


def test_provider_is_not_available_without_explicit_base_url(monkeypatch):
    plugin = load_plugin()
    monkeypatch.delenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", raising=False)
    monkeypatch.setattr(plugin, "_load_config", lambda: {})

    assert plugin.OpenAICompatibleImageGenProvider().is_available() is False


def test_provider_is_available_with_explicit_base_url(monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setattr(plugin, "_load_config", lambda: {})

    assert plugin.OpenAICompatibleImageGenProvider().is_available() is True


def test_generate_saves_b64_json_response(monkeypatch, tmp_path):
    plugin = load_plugin()
    png_bytes = b"\x89PNG\r\n\x1a\n"
    b64_image = base64.b64encode(png_bytes).decode("ascii")
    calls = []

    def fake_post(url, *, json, headers, timeout):
        calls.append((url, json, headers, timeout))
        return DummyResponse({"data": [{"b64_json": b64_image}]})

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    monkeypatch.setattr(plugin, "save_b64_image", lambda data, **kwargs: str(tmp_path / "image.png"))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "test-image-model")
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "test-key")

    result = plugin.OpenAICompatibleImageGenProvider().generate("draw cat", aspect_ratio="square")

    assert result["success"] is True
    assert result["provider"] == "openai-compatible"
    assert result["model"] == "test-image-model"
    assert result["image"].endswith("image.png")
    assert calls[0][0] == "http://localhost:8000/v1/images/generations"
    assert calls[0][1]["model"] == "test-image-model"
    assert calls[0][1]["n"] == 1
    assert calls[0][2]["Authorization"] == "Bearer test-key"
    assert "text/event-stream" in calls[0][2]["Accept"]


def test_generate_saves_url_response(monkeypatch, tmp_path):
    plugin = load_plugin()

    monkeypatch.setattr(
        plugin.requests,
        "post",
        lambda *args, **kwargs: DummyResponse({"data": [{"url": "https://example.test/image.png"}]}),
    )
    monkeypatch.setattr(plugin, "save_url_image", lambda url, **kwargs: str(tmp_path / "downloaded.png"))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")

    result = plugin.OpenAICompatibleImageGenProvider().generate("draw cat")

    assert result["success"] is True
    assert result["image"].endswith("downloaded.png")


def test_generate_parses_sse_response(monkeypatch, tmp_path):
    plugin = load_plugin()
    lines = [
        "event: ping",
        'data: {"data": [{"url": "https://example.test/sse.png"}]}',
        "data: [DONE]",
    ]

    monkeypatch.setattr(
        plugin.requests,
        "post",
        lambda *args, **kwargs: DummyResponse(
            headers={"content-type": "text/event-stream"},
            lines=lines,
        ),
    )
    monkeypatch.setattr(plugin, "save_url_image", lambda url, **kwargs: str(tmp_path / "sse.png"))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")

    result = plugin.OpenAICompatibleImageGenProvider().generate("draw cat")

    assert result["success"] is True
    assert result["image"].endswith("sse.png")


def test_generate_prefers_sse_done_event_over_partial_image(monkeypatch, tmp_path):
    plugin = load_plugin()
    lines = [
        "event: partial_image",
        'data: {"b64_json": "partial"}',
        "event: done",
        'data: {"data": [{"url": "https://example.test/done.png"}]}',
    ]

    monkeypatch.setattr(
        plugin.requests,
        "post",
        lambda *args, **kwargs: DummyResponse(
            headers={"content-type": "text/event-stream"},
            lines=lines,
        ),
    )
    monkeypatch.setattr(plugin, "save_url_image", lambda url, **kwargs: str(tmp_path / "done.png"))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")

    result = plugin.OpenAICompatibleImageGenProvider().generate("draw cat")

    assert result["success"] is True
    assert result["image"].endswith("done.png")


def test_generate_returns_sse_error_message(monkeypatch):
    plugin = load_plugin()
    lines = [
        "event: error",
        'data: {"message": "account is not entitled"}',
    ]

    monkeypatch.setattr(
        plugin.requests,
        "post",
        lambda *args, **kwargs: DummyResponse(
            headers={"content-type": "text/event-stream"},
            lines=lines,
        ),
    )
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")

    result = plugin.OpenAICompatibleImageGenProvider().generate("draw cat")

    assert result["success"] is False
    assert result["error_type"] == "request_failed"
    assert "account is not entitled" in result["error"]
