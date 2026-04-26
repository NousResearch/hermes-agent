"""Tests for MiniMax VLM (vision) endpoint routing.

MiniMax's Anthropic-compatible chat endpoint silently ignores image blocks.
Vision requests must be routed to the dedicated /v1/coding_plan/vlm endpoint.
See GitHub issue #15715.
"""

import sys
import os
import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agent.auxiliary_client import (
    _extract_minimax_vlm_payload,
    _build_minimax_vlm_response,
    _call_minimax_vlm_endpoint,
    _async_call_minimax_vlm_endpoint,
    _ANTHROPIC_COMPAT_PROVIDERS,
    extract_content_or_reasoning,
)


# -- fixtures ----------------------------------------------------------------

SAMPLE_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
SAMPLE_DATA_URI = f"data:image/png;base64,{SAMPLE_B64}"

VISION_MESSAGES_OPENAI = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this screenshot"},
            {"type": "image_url", "image_url": {"url": SAMPLE_DATA_URI}},
        ],
    }
]

VISION_MESSAGES_TEXT_ONLY = [
    {"role": "user", "content": "What do you see?"},
]

VISION_MESSAGES_MULTI_TEXT = [
    {"role": "system", "content": "You are a vision assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "First, look at this."},
            {"type": "image_url", "image_url": {"url": SAMPLE_DATA_URI}},
            {"type": "text", "text": "Then describe it."},
        ],
    },
]

VLM_SUCCESS_RESPONSE = {
    "content": "This is a 1x1 transparent PNG image.",
    "base_resp": {"status_code": 0, "status_msg": "success"},
}

VLM_ERROR_RESPONSE = {
    "content": "",
    "base_resp": {"status_code": 1001, "status_msg": "invalid image"},
}


def _mock_client(base_url="https://api.minimax.io/v1", api_key="sk-test-123"):
    client = MagicMock()
    client.base_url = base_url
    client.api_key = api_key
    return client


# -- _extract_minimax_vlm_payload tests -------------------------------------


class TestExtractMiniMaxVlmPayload:
    def test_basic_extraction(self):
        prompt, image_url = _extract_minimax_vlm_payload(VISION_MESSAGES_OPENAI)
        assert prompt == "Describe this screenshot"
        assert image_url == SAMPLE_DATA_URI

    def test_no_image_returns_none(self):
        prompt, image_url = _extract_minimax_vlm_payload(VISION_MESSAGES_TEXT_ONLY)
        assert prompt == "What do you see?"
        assert image_url is None

    def test_multi_text_blocks_joined(self):
        prompt, image_url = _extract_minimax_vlm_payload(VISION_MESSAGES_MULTI_TEXT)
        assert "You are a vision assistant." in prompt
        assert "First, look at this." in prompt
        assert "Then describe it." in prompt
        assert image_url == SAMPLE_DATA_URI

    def test_only_first_image_used(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,FIRST"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,SECOND"}},
                ],
            }
        ]
        _, image_url = _extract_minimax_vlm_payload(msgs)
        assert image_url == "data:image/png;base64,FIRST"

    def test_empty_messages(self):
        prompt, image_url = _extract_minimax_vlm_payload([])
        assert prompt == ""
        assert image_url is None


# -- _build_minimax_vlm_response tests --------------------------------------


class TestBuildMiniMaxVlmResponse:
    def test_response_shape(self):
        resp = _build_minimax_vlm_response("hello world")
        assert resp.choices[0].message.content == "hello world"

    def test_compatible_with_extract_content(self):
        resp = _build_minimax_vlm_response("VLM analysis result")
        assert extract_content_or_reasoning(resp) == "VLM analysis result"

    def test_empty_content(self):
        resp = _build_minimax_vlm_response("")
        assert resp.choices[0].message.content == ""


# -- _call_minimax_vlm_endpoint tests ----------------------------------------


class TestCallMiniMaxVlmEndpoint:
    @patch("httpx.post")
    def test_successful_call(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = VLM_SUCCESS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = _mock_client()
        result = _call_minimax_vlm_endpoint(
            "minimax", VISION_MESSAGES_OPENAI, client, timeout=30.0,
        )

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.minimax.io/v1/coding_plan/vlm"
        payload = call_args[1]["json"]
        assert payload["prompt"] == "Describe this screenshot"
        assert payload["image_url"] == SAMPLE_DATA_URI
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test-123"

        assert result.choices[0].message.content == "This is a 1x1 transparent PNG image."

    @patch("httpx.post")
    def test_cn_provider_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = VLM_SUCCESS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = _mock_client(
            base_url="https://api.minimaxi.com/v1",
            api_key="sk-cp-test",
        )
        _call_minimax_vlm_endpoint(
            "minimax-cn", VISION_MESSAGES_OPENAI, client, timeout=30.0,
        )

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.minimaxi.com/v1/coding_plan/vlm"

    def test_no_image_raises(self):
        client = _mock_client()
        with pytest.raises(RuntimeError, match="no image found"):
            _call_minimax_vlm_endpoint(
                "minimax", VISION_MESSAGES_TEXT_ONLY, client, timeout=30.0,
            )

    @patch("httpx.post")
    def test_api_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = VLM_ERROR_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = _mock_client()
        with pytest.raises(RuntimeError, match="invalid image"):
            _call_minimax_vlm_endpoint(
                "minimax", VISION_MESSAGES_OPENAI, client, timeout=30.0,
            )

    @patch("httpx.post")
    def test_trailing_slash_in_base_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = VLM_SUCCESS_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = _mock_client(base_url="https://api.minimax.io/v1/")
        _call_minimax_vlm_endpoint(
            "minimax", VISION_MESSAGES_OPENAI, client, timeout=30.0,
        )

        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.minimax.io/v1/coding_plan/vlm"


# -- _async_call_minimax_vlm_endpoint tests ----------------------------------


class TestAsyncCallMiniMaxVlmEndpoint:
    @pytest.mark.asyncio
    async def test_no_image_raises(self):
        client = _mock_client()
        with pytest.raises(RuntimeError, match="no image found"):
            await _async_call_minimax_vlm_endpoint(
                "minimax", VISION_MESSAGES_TEXT_ONLY, client, timeout=30.0,
            )

    @pytest.mark.asyncio
    async def test_successful_call(self):
        from unittest.mock import AsyncMock

        mock_resp = MagicMock()
        mock_resp.json.return_value = VLM_SUCCESS_RESPONSE
        mock_resp.raise_for_status = MagicMock()

        mock_http_client = MagicMock()
        mock_http_client.post = AsyncMock(return_value=mock_resp)
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            result = await _async_call_minimax_vlm_endpoint(
                "minimax", VISION_MESSAGES_OPENAI, _mock_client(), timeout=30.0,
            )

        assert result.choices[0].message.content == "This is a 1x1 transparent PNG image."


# -- Provider routing tests --------------------------------------------------


class TestMiniMaxVisionRouting:
    def test_minimax_in_anthropic_compat_providers(self):
        assert "minimax" in _ANTHROPIC_COMPAT_PROVIDERS
        assert "minimax-cn" in _ANTHROPIC_COMPAT_PROVIDERS

    def test_openrouter_not_in_anthropic_compat(self):
        assert "openrouter" not in _ANTHROPIC_COMPAT_PROVIDERS
