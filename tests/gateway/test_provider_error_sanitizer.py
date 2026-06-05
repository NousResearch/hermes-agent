from gateway.config import Platform
from gateway.runtime.provider_error_sanitizer import (
    gateway_provider_error_reply,
    looks_like_gateway_provider_error,
    sanitize_gateway_final_response,
)


def test_provider_error_sanitizer_maps_auth_errors_without_leaking_tokens():
    raw = "Provider authentication failed: incorrect API key sk-tes...3456 Bearer abcdefghijklmnopqrstuvwxyz"

    sanitized = sanitize_gateway_final_response(Platform.TELEGRAM, raw)

    assert "authentication failed" in sanitized.lower()
    assert "sk-tes...3456" not in sanitized
    assert "abcdefghijklmnopqrstuvwxyz" not in sanitized


def test_provider_error_sanitizer_keeps_non_telegram_response_surface_unchanged():
    raw = "Provider authentication failed: incorrect API key sk-tes...3456"

    assert sanitize_gateway_final_response(Platform.DISCORD, raw) == raw
    assert sanitize_gateway_final_response("local", raw) == raw


def test_provider_error_sanitizer_preserves_short_error_envelope_heuristic_only():
    assert looks_like_gateway_provider_error("HTTP 429: quota exhausted") is True
    explanatory_answer = "HTTP 404 means a resource was not found. " + (
        "Check the route, request method, and resource identifier before retrying. " * 8
    )
    assert len(explanatory_answer) > 400
    assert looks_like_gateway_provider_error(explanatory_answer) is False


def test_provider_error_reply_categories_are_stable():
    assert "rate-limiting" in gateway_provider_error_reply("HTTP 429: rate limit").lower()
    assert "provider rejected" in gateway_provider_error_reply("policy violation: request blocked").lower()
    assert "failed after retries" in gateway_provider_error_reply("API call failed: upstream died").lower()
