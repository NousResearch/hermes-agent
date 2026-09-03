"""Tests for agent/error_surface.py — turn-error → UI layer descriptors."""

from __future__ import annotations

import pytest

from agent.error_surface import (
    LAYER_AUTH,
    LAYER_BILLING,
    LAYER_DISK,
    LAYER_ENDPOINT,
    LAYER_GATEWAY,
    LAYER_PROVIDER,
    LAYER_STREAMING,
    build_error_surface_from_exception,
    build_error_surface_from_result,
)


# ── build_error_surface_from_result ──────────────────────────────────────


def _failed_result(reason: str = "", error: str = "provider exploded", **extra) -> dict:
    result = {"completed": False, "failed": True, "error": error}
    if reason:
        result["failure_reason"] = reason
    result.update(extra)
    return result


def test_result_none_for_non_dict():
    assert build_error_surface_from_result("boom") is None
    assert build_error_surface_from_result(None) is None


def test_result_none_for_healthy_result():
    assert (
        build_error_surface_from_result({"completed": True, "final_response": "hi"})
        is None
    )


def test_result_auth_reasons_map_to_auth_layer():
    # Both auth reasons are non-retryable, matching classify_api_error's own
    # verdict (a bare retry replays the same rejected credential).
    surface = build_error_surface_from_result(_failed_result("auth"))
    assert surface == {"layer": LAYER_AUTH, "code": "auth", "retryable": False}

    surface = build_error_surface_from_result(_failed_result("auth_permanent"))
    assert surface["layer"] == LAYER_AUTH
    assert surface["retryable"] is False


def test_result_billing_block_wins():
    surface = build_error_surface_from_result(
        _failed_result("rate_limit", billing_block={"provider": "nous"})
    )
    assert surface["layer"] == LAYER_BILLING
    assert surface["retryable"] is False


def test_result_billing_reason_without_block():
    surface = build_error_surface_from_result(_failed_result("billing"))
    assert surface == {"layer": LAYER_BILLING, "code": "billing", "retryable": False}


def test_result_provider_default_for_classified_reasons():
    for reason in (
        "rate_limit",
        "server_error",
        "overloaded",
        "unknown",
        "format_error",
    ):
        surface = build_error_surface_from_result(_failed_result(reason))
        assert surface["layer"] == LAYER_PROVIDER, reason
        assert surface["code"] == reason


def test_result_non_retryable_reasons():
    for reason in (
        "auth",
        "format_error",
        "content_policy_blocked",
        "model_not_found",
        "ssl_cert_verification",
    ):
        surface = build_error_surface_from_result(_failed_result(reason))
        assert surface["retryable"] is False, reason


def test_result_prefers_classifier_retry_verdict():
    """conversation_loop stamps ``failure_retryable`` from the real
    ClassifiedError — it must win over the fallback reason set."""
    surface = build_error_surface_from_result(
        _failed_result("unknown", failure_retryable=False)
    )
    assert surface["retryable"] is False

    surface = build_error_surface_from_result(
        _failed_result("format_error", failure_retryable=True)
    )
    assert surface["retryable"] is True


def test_result_stamps_failing_session_identity():
    surface = build_error_surface_from_result(
        _failed_result("rate_limit"), provider="openrouter", model="test/m1"
    )
    assert surface["provider"] == "openrouter"
    assert surface["model"] == "test/m1"

    # Absent identity omits the keys instead of stamping empty strings.
    surface = build_error_surface_from_result(_failed_result("rate_limit"))
    assert "provider" not in surface and "model" not in surface


def test_result_timeout_on_custom_endpoint_is_endpoint_layer():
    surface = build_error_surface_from_result(
        _failed_result("timeout"), provider="custom"
    )
    assert surface["layer"] == LAYER_ENDPOINT

    # Same reason on a vendor provider stays provider-layer.
    surface = build_error_surface_from_result(
        _failed_result("timeout"), provider="anthropic"
    )
    assert surface["layer"] == LAYER_PROVIDER


def test_result_stream_drop_text_maps_to_streaming():
    surface = build_error_surface_from_result(
        _failed_result(error="The provider's stream connection keeps dropping")
    )
    assert surface["layer"] == LAYER_STREAMING
    assert surface["code"] == "stream_drop"
    assert surface["retryable"] is True


@pytest.mark.parametrize(
    "reason,message",
    [
        ("rate_limit", "Your request was not processed."),
        ("server_error", "The request could not be processed at this time."),
        ("rate_limit", "Monthly quota surpassed."),
        ("content_policy_blocked", "The request was dismissed by the safety filter."),
        ("server_error", "AssertionError: assert seq_len <= max_model_len"),
    ],
)
def test_provider_prose_with_sse_substring_stays_provider(reason, message):
    """"sse" must not match inside ordinary words (processed, surpassed, ...).

    Matched as a bare substring it reroutes provider-layer failures to the
    streaming layer, which relabels the error card and suppresses the
    switch-provider action for exactly the errors it exists for.
    """
    surface = build_error_surface_from_result(_failed_result(reason, error=message))
    assert surface["layer"] == LAYER_PROVIDER
    assert surface["code"] == reason


@pytest.mark.parametrize(
    "message",
    [
        # agent/chat_completion_helpers.py builds this exact message.
        "Provider stream returned non-JSON SSE data.",
        "SSE stream aborted",
    ],
)
def test_standalone_sse_token_still_maps_to_streaming(message):
    surface = build_error_surface_from_result(_failed_result(error=message))
    assert surface["layer"] == LAYER_STREAMING
    assert surface["code"] == "stream_drop"


def test_result_unclassified_failure_defaults_to_provider_unknown():
    surface = build_error_surface_from_result(_failed_result(error="something odd"))
    assert surface == {"layer": LAYER_PROVIDER, "code": "unknown", "retryable": True}


def test_result_disk_full_wins_over_reason():
    surface = build_error_surface_from_result(
        _failed_result(
            "server_error", error="OSError: [Errno 28] No space left on device"
        )
    )
    assert surface["layer"] == LAYER_DISK
    assert surface["retryable"] is False


# ── build_error_surface_from_exception ───────────────────────────────────


def test_exception_non_api_is_gateway_layer():
    surface = build_error_surface_from_exception(KeyError("history"))
    assert surface["layer"] == LAYER_GATEWAY
    assert surface["code"] == "KeyError"
    assert surface["retryable"] is True


def test_exception_disk_full_is_disk_layer():
    surface = build_error_surface_from_exception(OSError(28, "No space left on device"))
    assert surface["layer"] == LAYER_DISK


def test_exception_with_status_code_routes_through_classifier():
    class FakeAPIError(Exception):
        status_code = 429

    surface = build_error_surface_from_exception(
        FakeAPIError("rate limited"), provider="openrouter"
    )
    # 429 → rate_limit → provider layer via the real classifier.
    assert surface["layer"] == LAYER_PROVIDER
    assert surface["code"] in ("rate_limit", "upstream_rate_limit")


def test_anthropic_usage_limit_routes_to_billing_recovery():
    class FakeAPIError(Exception):
        status_code = 429

    surface = build_error_surface_from_exception(
        FakeAPIError("usage limit reached"),
        provider="anthropic",
        model="claude-opus-5",
    )

    assert surface == {
        "layer": LAYER_BILLING,
        "code": "billing",
        "retryable": False,
        "provider": "anthropic",
        "model": "claude-opus-5",
    }


def test_exception_auth_status_routes_to_auth_layer():
    class FakeAuthError(Exception):
        status_code = 401

    surface = build_error_surface_from_exception(FakeAuthError("invalid api key"))
    assert surface["layer"] == LAYER_AUTH


def test_exception_never_raises_on_weird_input():
    class Hostile(Exception):
        @property
        def status_code(self):  # pragma: no cover - exercised via classifier
            raise RuntimeError("hostile attribute")

    # Must not raise, whatever it returns.
    build_error_surface_from_exception(Hostile("x"))
