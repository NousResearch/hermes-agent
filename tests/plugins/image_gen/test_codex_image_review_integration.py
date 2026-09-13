"""Offline integration review of Codex image selection and error attribution.

Real bundled discovery, config, tool dispatcher, provider, SSE and image saving;
only OAuth-token lookup and HTTP transport are replaced. Fixtures are synthetic,
not evidence of live OpenAI engine selection.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from tools import image_generation_tool as tool

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
    "ae426082"
)


@pytest.fixture
def codex_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.delenv("OPENAI_IMAGE_MODEL", raising=False)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "image_gen": {"provider": "openai-codex"},
    }))
    provider = tool._get_plugin_provider("openai-codex", force=True)
    assert provider is not None
    module = sys.modules[type(provider).__module__]
    assert module.__file__ is not None
    assert Path(module.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[3])
    monkeypatch.setattr(module, "_read_codex_access_token", lambda: "synthetic-review-token")
    return tmp_path, provider


@pytest.mark.parametrize("failure", ["host-model", "size", "image-model"])
def test_backend_error_does_not_reassign_blame(codex_route, monkeypatch, failure):
    home, provider = codex_route
    requests = []
    messages = []

    def respond(request):
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        host = payload["model"]
        image_model = payload["tools"][0]["model"]
        status, param, message = {
            "host-model": (404, "model", f"Unknown model: {host}"),
            "size": (400, "tools[0].size", "Invalid model parameters: size is unsupported"),
            "image-model": (400, "tools[0].model", f"Unknown model: {image_model} is not supported"),
        }[failure]
        messages.append(message)
        return httpx.Response(status, json={"error": {
            "message": message, "type": "invalid_request_error", "param": param,
        }})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(
        *a, transport=httpx.MockTransport(respond), **kw,
    ))
    result = json.loads(tool._handle_image_generate({"prompt": "A blue circle."}))
    assert len(requests) == 1
    assert result["success"] is False
    assert result["error_type"] == "api_error"
    assert messages[0] in result["error"]
    assert not result.get("image")
    if failure != "image-model":
        image_model = requests[0]["tools"][0]["model"]
        assert f"backend rejected model {image_model!r}" not in result["error"], result["error"]


@pytest.mark.parametrize("requested", ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst"])
@pytest.mark.parametrize("editing", [False, True], ids=["generate", "edit"])
def test_configured_selection_and_unverified_echo_survive_tool_dispatch(
    codex_route, monkeypatch, requested, editing,
):
    home, provider = codex_route
    config_path = home / "config.yaml"
    config_path.write_text(yaml.safe_dump({"image_gen": {
        "provider": "openai-codex", "model": requested,
        "openai-codex": {"model": requested},
    }}))
    original_config = config_path.read_bytes()
    requests = []
    reported = "opaque-backend-image-alias"
    events = [{"type": "response.completed", "response": {
        "tools": [{"type": "image_generation", "model": reported}],
        "output": [{"type": "image_generation_call", "status": "completed",
                    "result": base64.b64encode(PNG).decode()}],
    }}]
    wire = "".join(f"data: {json.dumps(event)}\n\n" for event in events)

    def respond(request):
        assert str(request.url) == "https://chatgpt.com/backend-api/codex/responses"
        requests.append(json.loads(request.content))
        return httpx.Response(200, text=wire, headers={"content-type": "text/event-stream"})

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda *a, **kw: real_client(
        *a, transport=httpx.MockTransport(respond), **kw,
    ))
    args = {"prompt": "A blue circle.", "aspect_ratio": "square"}
    if editing:
        source = home / "reference.png"
        source.write_bytes(PNG)
        args["image_url"] = str(source)
    result = json.loads(tool._handle_image_generate(args))
    assert len(requests) == 1
    assert requests[0]["tools"][0]["model"] == requested
    parts = requests[0]["input"][0]["content"]
    assert any(part["type"] == "input_image" for part in parts) is editing
    assert result["success"] is True
    assert result["model"] == requested
    assert result["requested_model"] == requested
    assert result["reported_model"] == reported
    assert result["model_selection_verified"] is False
    assert Path(result["image"]).read_bytes() == PNG
    assert config_path.read_bytes() == original_config
