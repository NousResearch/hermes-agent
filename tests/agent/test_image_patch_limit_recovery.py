"""A patch-budget rejection must resize request copies before model failover."""

import base64
import copy
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from agent.conversation_compression import try_shrink_image_parts_in_messages
from agent.error_classifier import FailoverReason, classify_api_error
from agent.turn_recovery import _image_error_max_dimension, recover_after_classification
from agent.turn_retry_state import TurnRetryState


def _patch_error(required=35721, limit=30000):
    message = (
        f"The image you provided requires {required} patches after processing, "
        f"exceeding the limit of {limit}. Please resize the image and try again."
    )
    error = RuntimeError(message)
    error.status_code = 400
    error.body = {"error": {"message": message, "param": "input", "code": "invalid_value"}}
    return error


@pytest.mark.parametrize("part_type", ["image_url", "input_image", "image"])
def test_patch_rejection_recovers_request_and_preserves_source(tmp_path, part_type):
    # Highly compressible pixels reproduce the patch limit below the byte-size
    # threshold: re-encoding a JPEG without reducing dimensions cannot fix it.
    source = tmp_path / "avatar.png"
    with Image.new("RGB", (6048, 6048), "navy") as picture:
        picture.save(source)
    original = source.read_bytes()
    encoded = base64.b64encode(original).decode("ascii")
    url = f"data:image/png;base64,{encoded}"
    if part_type == "image":
        part = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": encoded}}
    else:
        part = {"type": part_type, "image_url": {"url": url} if part_type == "image_url" else url}
    history = [{"role": "user", "content": [{"type": "text", "text": "Use my avatar."}, part]}]
    original_history = copy.deepcopy(history)
    notices = []
    agent = SimpleNamespace(
        provider="openai-codex", model="test-vision-model", log_prefix="",
        _recover_with_credential_pool=lambda **kwargs: (False, False),
        _try_shrink_image_parts_in_messages=try_shrink_image_parts_in_messages,
        _vprint=lambda message, **kwargs: notices.append(message),
    )
    error = _patch_error()
    classified = classify_api_error(error, provider=agent.provider, model=agent.model)
    assert classified.reason == FailoverReason.image_too_large
    assert not classified.should_compress
    assert not classified.should_fallback

    # A later user turn reconstructs its request from the same original history.
    # Both attempts must be recoverable without modifying that history or file.
    for _ in range(2):
        request = [dict(message) for message in history]
        retry = TurnRetryState()
        recovered, _ = recover_after_classification(
            agent, error, classified, retry, status_code=400, error_context={},
            messages=history, api_messages=request,
        )
        assert recovered and retry.image_shrink_retry_attempted
        resized = request[0]["content"][1]
        if part_type == "image":
            payload = resized["source"]["data"]
        else:
            image_url = resized["image_url"]
            payload = (image_url["url"] if isinstance(image_url, dict) else image_url).split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(payload))) as picture:
            width, height = picture.size
            assert ((width + 31) // 32) * ((height + 31) // 32) <= 30000
            assert width == height  # preserve the avatar's aspect ratio
        assert request[0]["content"][0] == history[0]["content"][0]
        assert history == original_history
        assert source.read_bytes() == original
    assert notices
    assert agent.provider == "openai-codex" and agent.model == "test-vision-model"


@pytest.mark.parametrize("limit", [1024, 10000, 30000])
def test_patch_ceiling_uses_provider_budget_from_error_body(limit):
    error = _patch_error(required=limit + 1, limit=limit)
    # SDK wrappers may keep the useful provider message only in the body.
    error.args = ("HTTP 400",)
    ceiling = _image_error_max_dimension(error)
    assert ceiling is not None
    assert ((ceiling + 31) // 32) ** 2 <= limit
    assert ((ceiling + 32) // 32) ** 2 > limit
    assert _image_error_max_dimension(RuntimeError("input token limit exceeded")) is None
