import base64
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
import yaml


PLUGIN_PATH = Path(__file__).resolve().parents[2] / "plugins" / "image_gen" / "openai-compatible" / "__init__.py"


def load_plugin():
    spec = importlib.util.spec_from_file_location("openai_compatible_image_gen_plugin", PLUGIN_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lines", [
    ['event: done', 'data: {"data": [', 'data: {"b64_json": "final"}]}', ''],
    ['event: done', 'data: {"b64_json": "final"}', '', 'data: {"usage": {}}', ''],
])
def test_sse_event_boundaries_preserve_final_image(lines):
    plugin = load_plugin()
    assert plugin._parse_sse_lines(lines) == {"data": [{"b64_json": "final"}]}


@pytest.mark.parametrize("tail", [[], ['data: [DONE]', '']])
def test_sse_partial_image_is_not_a_completed_generation(tail):
    plugin = load_plugin()
    with pytest.raises(ValueError, match="final image"):
        plugin._parse_sse_lines(['event: partial_image', 'data: {"b64_json": "preview"}', '', *tail])


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

    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def test_get_setup_schema_returns_env_vars(monkeypatch):
    """get_setup_schema should return env_vars (not env/optional_env) for picker compat."""
    plugin = load_plugin()
    schema = plugin.OpenAICompatibleImageGenProvider().get_setup_schema()

    assert "env_vars" in schema, "Picker expects env_vars key"
    assert isinstance(schema["env_vars"], list)
    assert len(schema["env_vars"]) >= 1

    keys = [v["key"] for v in schema["env_vars"]]
    assert "OPENAI_COMPATIBLE_IMAGE_API_KEY" in keys
    assert "OPENAI_COMPATIBLE_IMAGE_BASE_URL" not in keys

    # Every entry must have key, prompt, and url fields
    for v in schema["env_vars"]:
        assert "key" in v
        assert "prompt" in v


def test_generate_sends_correct_size_for_aspect_ratio(monkeypatch):
    """Aspect ratio labels should be translated to OpenAI size strings (e.g. square → 1024x1024)."""
    plugin = load_plugin()
    calls = []

    def fake_post(url, *, json, headers, timeout, **kwargs):
        calls.append((url, json, headers, timeout))
        return DummyResponse({"data": [{"b64_json": "AAAA"}]})

    monkeypatch.setattr(plugin.requests, "post", fake_post)
    monkeypatch.setattr(plugin, "save_b64_image", lambda data, **kwargs: "/tmp/test.png")
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "http://localhost:8000/v1")

    # square → 1024x1024
    plugin.OpenAICompatibleImageGenProvider().generate("cat", aspect_ratio="square")
    assert calls[0][1]["size"] == "1024x1024", f"Expected 1024x1024 got {calls[0][1]['size']}"

    # landscape → 1536x1024
    plugin.OpenAICompatibleImageGenProvider().generate("cat", aspect_ratio="landscape")
    assert calls[1][1]["size"] == "1536x1024", f"Expected 1536x1024 got {calls[1][1]['size']}"

    # portrait → 1024x1536
    plugin.OpenAICompatibleImageGenProvider().generate("cat", aspect_ratio="portrait")
    assert calls[2][1]["size"] == "1024x1536", f"Expected 1024x1536 got {calls[2][1]['size']}"

    # Unknown aspect falls back to landscape (resolve_aspect_ratio default)
    plugin.OpenAICompatibleImageGenProvider().generate("cat", aspect_ratio="ultrawide")
    assert calls[3][1]["size"] == "1536x1024", f"Expected fallback 1536x1024 got {calls[3][1]['size']}"


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

    def fake_post(url, *, json, headers, timeout, **kwargs):
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
    assert calls[0][1]["size"] == "1024x1024"
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
        "",
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
        "",
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


@pytest.fixture
def endpoint_server():
    """Real HTTP and cache I/O, with deterministic synthetic endpoint responses."""
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000d49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
    )
    state: dict[str, Any] = {"requests": [], "status": 200, "content_type": "application/json", "body": b"{}", "png": png}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["requests"].append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "payload": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
            })
            self.send_response(state["status"])
            self.send_header("Content-Type", state["content_type"])
            self.send_header("Content-Length", str(len(state["body"])))
            if state["status"] == 307:
                self.send_header("Location", state["base_url"] + "/redirect-target")
            self.end_headers()
            self.wfile.write(state["body"])

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)

        def log_message(self, format: str, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state["base_url"] = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["json_b64", "json_url", "sse"])
@pytest.mark.parametrize("key", [None, "active-profile-key"])
def test_discovery_picker_dispatch_and_cache_with_scoped_optional_auth(
    tmp_path, monkeypatch, endpoint_server, mode, key,
):
    from agent import image_gen_registry, secret_scope
    from hermes_cli.config import load_config, save_config
    from hermes_cli.plugins import _ensure_plugins_discovered
    from hermes_cli.tools_config_providers import _plugin_image_gen_providers, apply_provider_selection
    from tools.image_generation_tool import _handle_image_generate

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "other-profile-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", "https://other-profile.invalid")
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_MODEL", "other-profile-model")
    config = {"image_gen": {"model": "global-model", "openai_compatible": {
        "base_url": endpoint_server["base_url"], "model": "scoped-model",
    }}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    b64 = base64.b64encode(endpoint_server["png"]).decode()
    if mode == "sse":
        endpoint_server["content_type"] = "text/event-stream"
        endpoint_server["body"] = (
            'event: partial_image\ndata: {"b64_json":"preview"}\n\n'
            'event: done\ndata: {"data": [\n'
            f'data: {{"b64_json":"{b64}"}}]}}\n\n'
            'data: {"usage": {"total_tokens":1}}\n\ndata: [DONE]\n\n'
        ).encode()
    else:
        item = {"b64_json": b64} if mode == "json_b64" else {"url": endpoint_server["base_url"] + "/image.png"}
        endpoint_server["body"] = json.dumps({"data": [item]}).encode()

    (tmp_path / ".env").write_text(f"OPENAI_COMPATIBLE_IMAGE_API_KEY={key}\n" if key else "", encoding="utf-8")
    token = secret_scope.set_secret_scope(secret_scope.build_profile_secret_scope(tmp_path))
    was_active = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    try:
        _ensure_plugins_discovered(force=True)
        row = next(r for r in _plugin_image_gen_providers() if r.get("image_gen_plugin_name") == "openai-compatible")
        assert all(v["key"].endswith("_API_KEY") for v in row["env_vars"])
        apply_provider_selection("image_gen", row["name"], config)
        save_config(config)
        assert load_config()["image_gen"]["provider"] == "openai-compatible"
        provider = image_gen_registry.get_provider("openai-compatible")
        assert provider.is_available()
        assert provider.default_model() == "scoped-model"
        result = json.loads(_handle_image_generate({"prompt": "a cat", "aspect_ratio": "square"}))
        assert result["success"], result
        path = Path(result["image"])
        assert path.is_relative_to(tmp_path / "cache" / "images")
        assert path.read_bytes() == endpoint_server["png"]
        assert result["model"] == "scoped-model"
        request, = endpoint_server["requests"]
        assert request["path"] == "/v1/images/generations"
        assert request["authorization"] == (f"Bearer {key}" if key else None)
        assert request["payload"] == {"prompt": "a cat", "model": "scoped-model", "n": 1, "size": "1024x1024"}
    finally:
        secret_scope.set_multiplex_active(was_active)
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("inputs", [
    {"image_url": "https://example.test/source.png"},
    {"reference_image_urls": ["https://example.test/reference.png"]},
])
def test_edit_input_fails_before_request(inputs, endpoint_server, monkeypatch):
    plugin = load_plugin()
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", endpoint_server["base_url"])
    result = plugin.OpenAICompatibleImageGenProvider().generate("edit cat", **inputs)
    assert result["error_type"] == "modality_unsupported"
    assert not endpoint_server["requests"]


@pytest.mark.parametrize("settings", [
    {"api_key": "legacy-config-key"}, {"timeout": 0}, {"timeout": float("nan")},
    {"base_url": "file:///tmp/image"}, {"base_url": "https://user:password@example.test/v1"},
])
def test_invalid_configuration_fails_before_request(settings, tmp_path, monkeypatch, endpoint_server):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"image_gen": {"openai_compatible": {"base_url": endpoint_server["base_url"], **settings}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    result = plugin.OpenAICompatibleImageGenProvider().generate("cat")
    assert result["error_type"] == "invalid_config"
    assert not endpoint_server["requests"]


def test_unscoped_multiplex_request_fails_closed(tmp_path, monkeypatch, endpoint_server):
    from agent import secret_scope

    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_API_KEY", "other-profile-key")
    config = {"image_gen": {"openai_compatible": {"base_url": endpoint_server["base_url"], "model": "configured"}}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    was_active = secret_scope.is_multiplex_active()
    token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    try:
        with pytest.raises(secret_scope.UnscopedSecretError):
            plugin.OpenAICompatibleImageGenProvider().generate("cat")
        assert not endpoint_server["requests"]
    finally:
        secret_scope.set_multiplex_active(was_active)
        secret_scope.reset_secret_scope(token)


@pytest.mark.parametrize("status,content_type,body", [
    (307, "application/json", b"{}"),
    (500, "application/json", b'{"error":{"message":"backend failure"}}'),
    (200, "application/json", b"not JSON"),
    (200, "text/event-stream", b'event: error\ndata: {"message":"backend failure"}\n\n'),
    (200, "text/event-stream", b'event: partial_image\ndata: {"b64_json":"preview"}\n\n'),
])
def test_failed_responses_never_become_saved_images(
    status, content_type, body, tmp_path, monkeypatch, endpoint_server,
):
    plugin = load_plugin()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENAI_COMPATIBLE_IMAGE_BASE_URL", endpoint_server["base_url"] + "/v1/")
    endpoint_server.update(status=status, content_type=content_type, body=body)
    result = plugin.OpenAICompatibleImageGenProvider().generate("cat")
    assert not result["success"]
    assert len(endpoint_server["requests"]) == 1
    assert not (tmp_path / "cache" / "images").exists()
