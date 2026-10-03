"""A dropped upload with a large inline image gets one shrink attempt (#124833)."""
from __future__ import annotations

import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import openai
import pytest
from PIL import Image

from agent.conversation_compression import (
    _IMAGE_SHRINK_TARGET_BYTES,
    try_shrink_image_parts_in_messages,
)
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import recover_after_classification
from agent.turn_retry_state import TurnRetryState


def _agent():
    return SimpleNamespace(
        provider="ollama-cloud", api_mode="chat_completions", log_prefix="",
        _recover_with_credential_pool=lambda **kw: (False, kw["has_retried_429"]),
        _try_shrink_image_parts_in_messages=try_shrink_image_parts_in_messages,
        _vprint=Mock(),
    )


def _recover(agent, error, retry, messages, api_messages):
    classified = classify_api_error(error, provider=agent.provider)
    result = recover_after_classification(
        agent, error, classified, retry, status_code=getattr(error, "status_code", None),
        error_context={}, messages=messages, api_messages=api_messages,
    )
    return result, classified


def test_dropped_image_upload_shrinks_request_once_without_rewriting_history():
    image = Image.new("RGB", (1100, 1100), (20, 70, 120))
    data = io.BytesIO()
    image.save(data, format="PNG", compress_level=0)
    url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()
    assert len(url) > _IMAGE_SHRINK_TARGET_BYTES
    history = [{"role": "user", "content": [
        {"type": "text", "text": "inspect the reference"},
        {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
    ]}]
    outgoing = [message.copy() for message in history]
    agent, retry = _agent(), TurnRetryState()
    error = openai.APIConnectionError(request=httpx.Request("POST", "https://fixture.invalid/v1"))

    result, classified = _recover(agent, error, retry, history, outgoing)

    assert result == (True, False), "connection-drop recovery skipped the oversized image"
    assert classified.reason == FailoverReason.timeout  # a size cap is only a hypothesis
    assert retry.image_shrink_retry_attempted
    assert len(outgoing[0]["content"][1]["image_url"]["url"]) < _IMAGE_SHRINK_TARGET_BYTES
    assert outgoing[0]["content"][1]["image_url"]["detail"] == "high"
    assert history[0]["content"][1]["image_url"]["url"] == url
    assert outgoing[0]["content"][0] == history[0]["content"][0]
    # The guard is turn-scoped, not merely a consequence of shrinking the first payload.
    assert _recover(agent, error, retry, history, [history[0].copy()])[0] == (False, False)


@pytest.mark.parametrize("kind", ["empty", "small", "remote", "timeout", "http400", "corrupt", "resize_error"])
def test_ineligible_or_unshrinkable_upload_preserves_normal_error_recovery(kind, monkeypatch):
    request = httpx.Request("POST", "https://fixture.invalid/v1")
    error = openai.APIConnectionError(request=request)
    url = "data:image/png;base64," + "A" * (_IMAGE_SHRINK_TARGET_BYTES + 4)
    if kind == "small":
        url = "data:image/png;base64,AA=="
    elif kind == "remote":
        url = "https://fixture.invalid/large.png"
    elif kind == "timeout":
        error = openai.APITimeoutError(request=request)
    elif kind == "http400":
        error = openai.BadRequestError("failed to read request body", response=httpx.Response(400, request=request), body=None)
    elif kind == "resize_error":
        def fail(*args, **kwargs):
            raise OSError("fixture decoder failed")
        monkeypatch.setattr("tools.vision_tools._resize_image_for_vision", fail)
    outgoing = [] if kind == "empty" else [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
    ]}]
    agent, retry = _agent(), TurnRetryState()
    result, classified = _recover(agent, error, retry, [], outgoing)
    assert result == (False, False)
    assert classified.reason == (FailoverReason.format_error if kind == "http400" else FailoverReason.timeout)
    assert retry.image_shrink_retry_attempted is (kind in {"corrupt", "resize_error"})
    if outgoing:
        assert outgoing[0]["content"][0]["image_url"]["url"] == url
