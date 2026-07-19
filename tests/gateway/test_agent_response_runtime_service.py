"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

See gateway/RUNTIME_SERVICES.md. Marked dead_runtime_service so suites can
optionally filter with ``-m "not dead_runtime_service"``; default still runs.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

from gateway.agent_response_runtime_service import (
    build_failed_agent_response,
    build_gateway_exception_response,
    normalize_gateway_agent_response,
)


def test_normalize_gateway_agent_response_suppresses_empty_without_fallback():
    result = normalize_gateway_agent_response(
        agent_result={
            "final_response": "(empty)",
            "suppress_reply": False,
            "failed": False,
        },
        history_len=0,
        empty_response_fallback=lambda kind: "",
    )

    assert result.response == ""
    assert result.suppress_reply is True
    assert result.response_state == "suppressed_empty"


def test_normalize_gateway_agent_response_uses_visible_fallback():
    result = normalize_gateway_agent_response(
        agent_result={
            "final_response": "[[NO_REPLY]]",
            "suppress_reply": False,
            "failed": False,
        },
        history_len=0,
        empty_response_fallback=lambda kind: "收到，你继续说。",
    )

    assert result.response == "收到，你继续说。"
    assert result.suppress_reply is False
    assert result.response_state == "qq_explicit_fallback"
    assert result.synthetic_fallback is True


def test_build_failed_agent_response_detects_context_overflow():
    response = build_failed_agent_response(
        error_detail="HTTP 400 payload too large for context",
        history_len=80,
    )

    assert "Session too large" in response
    assert "/compact" in response


def test_normalize_gateway_agent_response_surfaces_generic_failed_silently():
    result = normalize_gateway_agent_response(
        agent_result={
            "final_response": "",
            "suppress_reply": False,
            "failed": True,
            "error": "upstream exploded",
        },
        history_len=5,
        empty_response_fallback=lambda kind: "",
    )

    assert result.response_state == "failed_silent"
    assert "The request failed: upstream exploded" in result.response


class _GatewayError(Exception):
    def __init__(self, message: str, *, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


def test_build_gateway_exception_response_includes_401_hint():
    response = build_gateway_exception_response(
        error=_GatewayError("Unauthorized", status_code=401),
        history_len=3,
    )

    assert "Unauthorized" in response
    assert "run `claude /login`" in response


def test_build_gateway_exception_response_formats_usage_limit_429():
    error = _GatewayError(
        "limit reached",
        status_code=429,
        response=type(
            "_Resp",
            (),
            {
                "json": staticmethod(
                    lambda: {
                        "error": {
                            "type": "usage_limit_reached",
                            "resets_in_seconds": 3700,
                        }
                    }
                )
            },
        )(),
    )

    response = build_gateway_exception_response(
        error=error,
        history_len=3,
    )

    assert "limit reached" in response
    assert "resets in ~2h" in response


def test_build_gateway_exception_response_uses_generic_429_rate_limit_hint():
    response = build_gateway_exception_response(
        error=_GatewayError("slow down", status_code=429),
        history_len=3,
    )

    assert "slow down" in response
    assert "rate-limited" in response


def test_build_gateway_exception_response_maps_529_to_overload():
    response = build_gateway_exception_response(
        error=_GatewayError("overloaded", status_code=529),
        history_len=3,
    )

    assert "overloaded" in response
    assert "temporarily overloaded" in response


def test_build_gateway_exception_response_maps_large_400_to_context_warning():
    response = build_gateway_exception_response(
        error=_GatewayError("Bad Request", status_code=400),
        history_len=80,
    )

    assert "Session too large" in response
    assert "/compact" in response


def test_build_gateway_exception_response_maps_small_400_to_request_rejected():
    response = build_gateway_exception_response(
        error=_GatewayError("Bad Request", status_code=400),
        history_len=5,
    )

    assert "Bad Request" in response
    assert "request was rejected by the API" in response
