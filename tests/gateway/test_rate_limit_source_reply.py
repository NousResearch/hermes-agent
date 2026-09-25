"""Rate-limit replies attribute only sources identified by the error (#122272)."""
import pytest

from gateway.run import _gateway_provider_error_reply, _sanitize_gateway_final_response


@pytest.mark.parametrize("text,source", [
    ("HTTP 429 Discord API rate limit exceeded; retry after 600s", "discord"),
    ("HTTP 429 Too Many Requests; retry after 600s", "upstream"),
])
@pytest.mark.parametrize("platform", ["discord", "telegram", "slack"])
def test_non_model_rate_limit_is_not_a_model_quota(platform, text, source):
    reply = _sanitize_gateway_final_response(platform, text).lower()
    assert source in reply
    assert "ai model" not in reply
    assert "usage limit" not in reply
    assert "resets in" not in reply
    assert "/model" not in reply
    assert "/retry" in reply


@pytest.mark.parametrize("source", ["Codex provider", "Gemini", "custom model provider"])
def test_named_model_quota_keeps_reset_guidance(source):
    reply = _gateway_provider_error_reply(
        f"{source} quota exhausted (429); retry after 116168s"
    )
    assert "AI model service" in reply
    assert "resets in ~33h" in reply
    assert "/model" in reply


@pytest.mark.parametrize("text", [
    "HTTP 429 usage_limit_reached; retry after 600s",
    "HTTP 429 Too Many Requests",
    "HTTP 429 unknown weekly limit reached; resets in 4hr",
    "HTTP 429 Telegram API retry after 600s",
    "HTTP 429 upstream says model-router-unavailable; retry after 600s",
    "HTTP 429 hosting provider quota exhausted; retry after 600s",
    "HTTP 429 Codex quota exhausted; retry after 600s (Discord API delivery)",
])
def test_unknown_source_does_not_infer_quota_from_delay(text):
    reply = _gateway_provider_error_reply(text)
    assert "AI model" not in reply
    assert "upstream" in reply
    assert "usage limit" not in reply
    assert "resets in" not in reply


@pytest.mark.parametrize("text", [
    "OpenAI HTTP 429 rate limit exceeded; retry after 600s",
    "Anthropic HTTP 429 rate limit exceeded",
    "HTTP 429 rate limit exceeded for this model (delivery via Discord)",
])
def test_model_throttle_is_not_a_quota(text):
    reply = _gateway_provider_error_reply(text)
    assert "AI model service" in reply
    assert "usage limit" not in reply
    assert "resets in" not in reply


def test_programmatic_surface_keeps_original_error():
    text = "HTTP 429 Discord API rate limit exceeded; retry after 600s"
    assert _sanitize_gateway_final_response("local", text) == text
