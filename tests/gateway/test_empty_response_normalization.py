"""Tests for gateway empty-response normalization."""

from gateway.run import (
    _INTERRUPT_REASON_GATEWAY_RESTART,
    _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
    _INTERRUPT_REASON_STOP,
    _normalize_empty_agent_response,
)


def test_gateway_restart_interrupt_gets_visible_fallback():
    response = _normalize_empty_agent_response(
        {
            "interrupted": True,
            "interrupt_message": _INTERRUPT_REASON_GATEWAY_RESTART,
            "api_calls": 18,
        },
        "",
    )

    assert response
    assert "gateway restart" in response.lower()
    assert "no final answer" in response.lower()


def test_gateway_shutdown_interrupt_gets_visible_fallback():
    response = _normalize_empty_agent_response(
        {
            "interrupted": True,
            "interrupt_message": _INTERRUPT_REASON_GATEWAY_SHUTDOWN,
            "api_calls": 4,
        },
        "",
    )

    assert response
    assert "gateway is shutting down" in response.lower()


def test_user_followup_interrupt_stays_silent_for_queued_message():
    response = _normalize_empty_agent_response(
        {
            "interrupted": True,
            "interrupt_message": "new user follow-up while the agent is running",
            "api_calls": 3,
        },
        "",
    )

    assert response == ""


def test_stop_control_interrupt_stays_silent_to_avoid_duplicate_stop_reply():
    response = _normalize_empty_agent_response(
        {
            "interrupted": True,
            "interrupt_message": _INTERRUPT_REASON_STOP,
            "api_calls": 3,
        },
        "",
    )

    assert response == ""


def test_non_interrupted_empty_after_api_calls_still_gets_generic_fallback():
    response = _normalize_empty_agent_response(
        {"interrupted": False, "api_calls": 2},
        "",
    )

    assert response
    assert "no response was generated" in response.lower()
