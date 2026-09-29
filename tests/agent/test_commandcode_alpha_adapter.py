"""Wire-contract tests for the Command Code /alpha/generate adapter.

Covers the protocol edge cases verified by reviewers against the live endpoint:
- image parts must be flattened and carry ``mimeType`` or the endpoint accepts the
  part and silently ignores the pixels;
- ``tool_choice="none"`` must empty the tool list, ``temperature`` must be forwarded;
- ``config.environment`` must report the real host, not a hardcoded "linux";
- a stream that ends with no output is an error (fail closed), not an empty success;
- in-stream ``{"type":"error","statusCode":402}`` must carry its status onto the
  exception so Hermes' error classifier can map 402→billing / 429→rate_limit;
- ``totalUsage.inputTokenDetails.cacheReadTokens`` must surface as
  ``usage.prompt_tokens_details.cached_tokens`` (the endpoint serves most of the
  prefix from cache; without this the caller sees zero hits).
"""

import io
import json
import platform
import urllib.error
import urllib.request

import pytest

from agent.commandcode_alpha_adapter import (
    CommandCodeAPIError,
    stream_commandcode_alpha,
)


class _FakeResponse(io.BytesIO):
    """Iterable fake of urlopen's context-manager response (NDJSON lines)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _events(lines):
    return _FakeResponse(b"".join(json.dumps(e).encode() + b"\n" for e in lines))


class _FakeAgent:
    api_key = "cmd-test-token"


@pytest.fixture()
def captured_request(monkeypatch):
    """Capture the outgoing Request and reply with a canned minimal stream."""
    box = {}

    def _fake_urlopen(req, timeout=None):
        box["req"] = req
        box["body"] = json.loads(req.data.decode("utf-8"))
        return _events([
            {"type": "text-delta", "text": "ok"},
            {"type": "finish", "finishReason": "stop", "totalUsage": {}},
        ])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return box


def _run(api_kwargs, monkeypatch, captured_request):
    resp = stream_commandcode_alpha(_FakeAgent(), api_kwargs)
    return resp


def test_image_part_is_flattened_with_mimetype(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what colour?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    image = [p for p in parts if p.get("type") == "image"]
    assert image == [{
        "type": "image",
        "image": "data:image/png;base64,AAAA",
        "mimeType": "image/png",
    }]


def test_text_only_model_gets_placeholder_not_silently_dropped(monkeypatch, captured_request):
    _run({
        "model": "deepseek/deepseek-v4-pro",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    assert [p for p in parts if p.get("type") == "image"] == []
    assert any("[image:" in p.get("text", "") for p in parts if p.get("type") == "text")


def test_temperature_forwarded_and_tool_choice_none_empties_tools(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "tool_choice": "none",
        "tools": [{"type": "function", "function": {"name": "shell", "parameters": {}}}],
    }, monkeypatch, captured_request)
    params = captured_request["body"]["params"]
    assert params["temperature"] == 0.3
    assert params["tools"] == []


def test_temperature_rejects_non_numeric(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": "0.0",
    }, monkeypatch, captured_request)
    assert "temperature" not in captured_request["body"]["params"]


def test_environment_reports_real_host(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
    }, monkeypatch, captured_request)
    expected_prefix = platform.system().lower()
    env = captured_request["body"]["config"]["environment"]
    assert env.startswith(expected_prefix)
    assert platform.python_version() in env


def test_empty_stream_fails_closed(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        return _events([{"type": "start"}])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(CommandCodeAPIError) as exc_info:
        stream_commandcode_alpha(_FakeAgent(), {
            "model": "meta/muse-spark-1.3-contributor",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert exc_info.value.status_code == 502
    assert "no output" in str(exc_info.value)


def test_instream_error_402_carries_status(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        return _events([
            {"type": "start"},
            {"type": "error", "statusCode": 402, "error": "Insufficient Balance"},
        ])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(CommandCodeAPIError) as exc_info:
        stream_commandcode_alpha(_FakeAgent(), {
            "model": "meta/muse-spark-1.3-contributor",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert exc_info.value.status_code == 402
    assert "Insufficient Balance" in str(exc_info.value)


def test_http_error_carries_status(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            "https://api.commandcode.ai/alpha/generate", 429, "Too Many Requests", {}, io.BytesIO(b"slow down"),
        )

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(CommandCodeAPIError) as exc_info:
        stream_commandcode_alpha(_FakeAgent(), {
            "model": "meta/muse-spark-1.3-contributor",
            "messages": [{"role": "user", "content": "hi"}],
        })
    assert exc_info.value.status_code == 429


def test_classifier_maps_instream_402_to_billing(monkeypatch):
    from agent.error_classifier import classify_api_error

    def _fake_urlopen(req, timeout=None):
        return _events([
            {"type": "error", "statusCode": 402, "error": "Insufficient Balance"},
        ])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(CommandCodeAPIError) as exc_info:
        stream_commandcode_alpha(_FakeAgent(), {
            "model": "meta/muse-spark-1.3-contributor",
            "messages": [{"role": "user", "content": "hi"}],
        })
    verdict = classify_api_error(exc_info.value, provider="commandcode-oauth")
    assert getattr(verdict.reason, "value", verdict.reason) == "billing"
    assert verdict.status_code == 402


def test_cache_read_tokens_surface_on_usage(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        return _events([
            {"type": "text-delta", "text": "done"},
            {"type": "finish", "finishReason": "stop", "totalUsage": {
                "inputTokens": 1000,
                "outputTokens": 50,
                "inputTokenDetails": {"cacheReadTokens": 977, "cacheWriteTokens": 23},
            }},
        ])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    resp = stream_commandcode_alpha(_FakeAgent(), {
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.usage.prompt_tokens_details.cached_tokens == 977
    assert resp.usage.prompt_tokens_details.cache_write_tokens == 23


def test_https_image_url_with_extension_is_forwarded(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://cdn.example.com/cat.PNG?x=1"}},
            ],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    image = [p for p in parts if p.get("type") == "image"]
    assert image == [{
        "type": "image",
        "image": "https://cdn.example.com/cat.PNG?x=1",
        "mimeType": "image/png",
    }]


def test_explicit_mimetype_hint_forwards_opaque_url(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://cdn.example.com/blob/9f3a"},
                "mimeType": "image/webp",
            }],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    image = [p for p in parts if p.get("type") == "image"]
    assert image and image[0]["mimeType"] == "image/webp"


def test_anthropic_source_block_becomes_data_uri_with_mime(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"},
            }],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    image = [p for p in parts if p.get("type") == "image"]
    assert image == [{
        "type": "image",
        "image": "data:image/jpeg;base64,QUJD",
        "mimeType": "image/jpeg",
    }]


def test_unknown_non_text_part_is_not_misrepresented_as_image(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "file", "file": {"filename": "notes.txt"}},
            ],
        }],
    }, monkeypatch, captured_request)
    parts = captured_request["body"]["params"]["messages"][0]["content"]
    assert [p for p in parts if p.get("type") == "image"] == []
    assert not any("[image:" in p.get("text", "") for p in parts if p.get("type") == "text")


def test_finish_step_usage_survives_empty_final_finish(monkeypatch):
    def _fake_urlopen(req, timeout=None):
        return _events([
            {"type": "text-delta", "text": "partial"},
            {"type": "finish-step", "totalUsage": {
                "inputTokens": 800, "outputTokens": 40,
                "inputTokenDetails": {"cacheReadTokens": 500},
            }},
            {"type": "finish", "finishReason": "stop"},
        ])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    resp = stream_commandcode_alpha(_FakeAgent(), {
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert resp.usage.prompt_tokens == 800
    assert resp.usage.prompt_tokens_details.cached_tokens == 500


def test_tool_choice_required_forwards_tools_with_debug(monkeypatch, captured_request):
    _run({
        "model": "meta/muse-spark-1.3-contributor",
        "messages": [{"role": "user", "content": "hi"}],
        "tool_choice": "required",
        "tools": [{"type": "function", "function": {"name": "shell", "parameters": {}}}],
    }, monkeypatch, captured_request)
    params = captured_request["body"]["params"]
    assert len(params["tools"]) == 1
