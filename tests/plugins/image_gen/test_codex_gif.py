"""Native Codex GIF contract tests. HTTP responses here are fixtures, not live inference."""
import base64
import importlib
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from tests.plugins.image_gen.test_openai_codex_provider import _b64_png

pytest_plugins = ["tests.plugins.image_gen.test_openai_codex_provider"]

codex = importlib.import_module("plugins.image_gen.openai-codex")


@pytest.fixture
def gif_bytes():
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (2, 2), color) for color in ("red", "blue")]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:], duration=200, loop=0)
    return buffer.getvalue()


@pytest.fixture
def gif_backend(codex_backend, gif_bytes):
    def respond(request):
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(gif_bytes).decode()}]})
    codex_backend["respond"] = respond
    return codex_backend


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("options", [{}, {"frame_count": 2, "frame_duration_ms": 20},
                                      {"frame_count": 16, "frame_duration_ms": 2000}])
def test_native_gif_fields_and_artifact(provider, gif_backend, gif_bytes, edit, options):
    sources = {"image_url": "data:image/png;base64," + _b64_png()} if edit else {}
    result = provider.generate("A looping dot", gif=options, **sources)
    assert result["success"] is True
    request, = gif_backend["requests"]
    assert request.url.path.endswith("images/edits" if edit else "images/generations")
    body = json.loads(request.content)
    assert body["output_format"] == "gif"
    assert "gif" not in body
    for name in ("frame_count", "frame_duration_ms"):
        assert body.get(name) == options.get(name)
        assert (name in body) == (name in options)
    path = Path(result["image"])
    assert path.suffix == ".gif"
    assert path.read_bytes() == gif_bytes
    assert result["mime_type"] == "image/gif"
    assert result["output_format"] == "gif"
    with Image.open(path) as image:
        assert image.n_frames == 2


@pytest.mark.parametrize("options", [False, True, [], "gif", 0, {"unknown": 1},
    {"frame_count": 1}, {"frame_count": 17}, {"frame_count": True}, {"frame_count": 2.0},
    {"frame_count": None}, {"frame_duration_ms": 19}, {"frame_duration_ms": 2001},
    {"frame_duration_ms": "200"}, {"frame_duration_ms": False}])
def test_invalid_gif_rejected_before_http(provider, codex_backend, options):
    result = provider.generate("A looping dot", gif=options)
    assert result["success"] is False
    assert result["error_type"] == "invalid_gif_options"
    assert codex_backend["requests"] == []


def test_static_request_omits_gif_fields(provider, codex_backend):
    result = provider.generate("A dot", gif=None)
    assert result["success"] is True
    body = json.loads(codex_backend["requests"][0].content)
    assert not {"output_format", "frame_count", "frame_duration_ms"} & body.keys()
    assert Path(result["image"]).suffix == ".png"


def test_gif_request_rejects_static_response(provider, codex_backend):
    result = provider.generate("A looping dot", gif={})
    assert result["success"] is False
    assert result["error_type"] == "unexpected_output_format"
    assert result["image"] is None


def test_actual_bytes_override_response_format_label(provider, gif_backend):
    result = provider.generate("A dot")
    assert result["success"] is True
    assert Path(result["image"]).suffix == ".gif"
    assert result["mime_type"] == "image/gif"


@pytest.mark.parametrize("encoded", ["%%%", base64.b64encode(b"not an image").decode(),
                                      base64.b64encode(b"GIF89a").decode()])
def test_bad_gif_response_does_not_write_artifact(provider, codex_backend, encoded, tmp_path):
    codex_backend["respond"] = lambda request: httpx.Response(200, json={"data": [{"b64_json": encoded}]})
    result = provider.generate("A looping dot", gif={})
    assert result["success"] is False
    assert result["image"] is None
    from hermes_constants import get_hermes_home
    assert not list((get_hermes_home() / "cache/generated/images").glob("*"))


def test_gif_request_rejects_single_frame(provider, codex_backend):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(buffer, format="GIF")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    codex_backend["respond"] = lambda request: httpx.Response(200, json={"data": [{"b64_json": encoded}]})
    result = provider.generate("A looping dot", gif={})
    assert result["success"] is False
    from hermes_constants import get_hermes_home
    assert not list((get_hermes_home() / "cache/generated/images").glob("*"))


def test_gif_cache_write_failure(provider, gif_backend, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("fixture disk full")
    monkeypatch.setattr(codex, "save_b64_image", refuse)
    result = provider.generate("A looping dot", gif={})
    assert result["success"] is False
    assert result["image"] is None
    assert result["error_type"] == "io_error"


def test_gif_backend_rejection_stays_on_native_route(provider, codex_backend):
    codex_backend["respond"] = lambda request: httpx.Response(
        403, json={"error": {"message": "GIF output is unavailable in this session."}})
    result = provider.generate("A looping dot", gif={})
    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert "HTTP 403" in result["error"]
    assert "GIF output is unavailable" in result["error"]
    assert len(codex_backend["requests"]) == 1


def test_tool_schema_and_dispatch_preserve_empty_gif(provider, gif_backend, monkeypatch):
    import tools.image_generation_tool as tool
    from tools.registry import registry

    monkeypatch.setattr(tool, "_read_configured_image_provider", lambda: "openai-codex")
    monkeypatch.setattr(tool, "_read_configured_image_model", lambda: None)
    monkeypatch.setattr(tool, "_get_plugin_provider", lambda *args, **kwargs: provider)
    schema = tool._build_dynamic_image_schema()
    gif = schema["parameters"]["properties"]["gif"]
    assert gif["type"] == "object"
    assert gif["properties"]["frame_count"]["minimum"] == 2
    assert gif["properties"]["frame_duration_ms"]["maximum"] == 2000
    result = json.loads(registry.dispatch("image_generate", {"prompt": "A looping dot", "gif": {}}))
    assert result["success"] is True
    assert result["image"].endswith(".gif")
    assert json.loads(gif_backend["requests"][0].content)["output_format"] == "gif"


def test_tool_rejects_unsupported_gif_without_static_fallback(monkeypatch):
    import tools.image_generation_tool as tool
    from tools.registry import registry

    monkeypatch.setattr(tool, "_active_image_capabilities", lambda: {"modalities": ["text"]})
    assert "gif" not in tool._build_dynamic_image_schema()["parameters"]["properties"]
    monkeypatch.setattr(tool, "_dispatch_to_plugin_provider", lambda *args, **kwargs: pytest.fail("static fallback"))
    result = json.loads(registry.dispatch("image_generate", {"prompt": "A looping dot", "gif": {}}))
    assert result["success"] is False
    assert result["error_type"] == "control_unsupported"
