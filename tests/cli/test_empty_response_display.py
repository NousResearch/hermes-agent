"""CLI display helpers for exhausted empty model responses."""

from __future__ import annotations

from cli import (
    EMPTY_RESPONSE_EXHAUSTED_MESSAGE,
    _normalize_final_response_for_cli,
)


def test_empty_response_exhaustion_replaces_literal_empty_sentinel():
    response, is_error = _normalize_final_response_for_cli(
        {
            "final_response": "(empty)",
            "empty_response_exhausted": True,
            "error_code": "empty_response_exhausted",
        },
        "(empty)",
    )

    assert is_error is True
    assert response == EMPTY_RESPONSE_EXHAUSTED_MESSAGE
    assert "(empty)" not in response
    assert "no visible response" in response


def test_legacy_empty_response_exhaustion_code_replaces_sentinel():
    response, is_error = _normalize_final_response_for_cli(
        {
            "final_response": "(empty)",
            "error_code": "empty_response_exhausted",
        },
        "(empty)",
    )

    assert is_error is True
    assert response == EMPTY_RESPONSE_EXHAUSTED_MESSAGE


def test_non_exhausted_empty_sentinel_is_left_unchanged():
    response, is_error = _normalize_final_response_for_cli({}, "(empty)")

    assert is_error is False
    assert response == "(empty)"


def test_normal_response_is_left_unchanged():
    response, is_error = _normalize_final_response_for_cli(
        {"final_response": "Done."},
        "Done.",
    )

    assert is_error is False
    assert response == "Done."
