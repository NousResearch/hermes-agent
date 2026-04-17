from gateway.agent_response_runtime_service import (
    build_failed_agent_response,
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
