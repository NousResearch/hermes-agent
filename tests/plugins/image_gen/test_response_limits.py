from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest
import requests


class _RawStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def stream(self, chunk_size: int, decode_content: bool = False):
        _ = chunk_size, decode_content
        yield from self._chunks

    def close(self) -> None:
        return None


def _stream_response(
    chunks: list[bytes],
    *,
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.headers.update(headers or {})
    response.url = "https://provider.test/response"
    response.raw = _RawStream(chunks)  # type: ignore[assignment]
    response.request = requests.Request("POST", response.url).prepare()
    return response


def test_limited_reader_rejects_oversized_content_length():
    from agent.image_gen_provider import (
        ImageGenResponseTooLarge,
        read_response_body_with_limit,
    )

    response = _stream_response([], headers={"Content-Length": "11"})

    with pytest.raises(ImageGenResponseTooLarge, match="exceeds 10 bytes"):
        read_response_body_with_limit(response, max_bytes=10)


def test_limited_reader_rejects_streamed_body_over_limit():
    from agent.image_gen_provider import (
        ImageGenResponseTooLarge,
        read_response_body_with_limit,
    )

    response = _stream_response([b"a" * 6, b"b" * 6])

    with pytest.raises(ImageGenResponseTooLarge, match="exceeds 10 bytes"):
        read_response_body_with_limit(response, max_bytes=10)


def test_limited_reader_preserves_response_json():
    from agent.image_gen_provider import read_response_body_with_limit

    response = _stream_response([b'{"ok": ', b"true}"])

    read_response_body_with_limit(response, max_bytes=20)

    assert response.json() == {"ok": True}


def test_save_b64_image_rejects_oversized_decoded_image():
    from agent.image_gen_provider import save_b64_image

    payload = base64.b64encode(b"x" * 11).decode("ascii")

    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        save_b64_image(payload, max_bytes=10)


def test_xai_generation_rejects_oversized_response(monkeypatch):
    import agent.image_gen_provider as image_gen_provider
    from plugins.image_gen.xai import XAIImageGenProvider

    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(
        image_gen_provider, "IMAGE_GEN_RESPONSE_BODY_LIMIT_BYTES", 10
    )

    response = _stream_response([b"x" * 8, b"y" * 8])

    with patch("plugins.image_gen.xai.requests.post", return_value=response):
        result = XAIImageGenProvider().generate(prompt="test")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"
    assert "response too large" in result["error"]


def test_openrouter_generation_rejects_oversized_response(monkeypatch):
    import agent.image_gen_provider as image_gen_provider
    from plugins.image_gen.openrouter import OpenRouterCompatImageProvider

    monkeypatch.setattr(
        image_gen_provider, "IMAGE_GEN_RESPONSE_BODY_LIMIT_BYTES", 10
    )
    provider = OpenRouterCompatImageProvider(
        provider_name="openrouter",
        display_name="OpenRouter",
        runtime_name="openrouter",
        config_key="openrouter",
        model_env_var="OPENROUTER_IMAGE_MODEL",
        setup_schema={"name": "OpenRouter (image)", "badge": "paid", "env_vars": []},
    )
    runtime = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "sk-or-test",
        "source": "env",
    }
    response = _stream_response([b"x" * 8, b"y" * 8])

    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=runtime,
    ), patch("requests.post", return_value=response):
        result = provider.generate(prompt="test")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"
    assert "response too large" in result["error"]


def test_krea_poll_rejects_oversized_response(monkeypatch):
    import agent.image_gen_provider as image_gen_provider
    from plugins.image_gen.krea import KreaImageGenProvider

    monkeypatch.setenv("KREA_API_KEY", "test-key")
    monkeypatch.setattr(
        image_gen_provider, "IMAGE_GEN_RESPONSE_BODY_LIMIT_BYTES", 10
    )

    submit = MagicMock()
    submit.status_code = 200
    submit.raise_for_status = MagicMock()
    submit.json.return_value = {"job_id": "job-1"}
    poll = _stream_response([b"x" * 8, b"y" * 8])

    with patch("plugins.image_gen.krea.requests.post", return_value=submit), patch(
        "plugins.image_gen.krea.requests.get",
        return_value=poll,
    ), patch("plugins.image_gen.krea.time.sleep"):
        result = KreaImageGenProvider().generate(prompt="test")

    assert result["success"] is False
    assert result["error_type"] == "invalid_response"
    assert "poll response too large" in result["error"]
