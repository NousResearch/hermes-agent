"""Pytest suite for parse_pasted_card.parse().

Run from the skill root:
    cd skills/commerce/personal-shopper
    PYTHONPATH=scripts pytest tests/
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import parse_pasted_card as ppc  # noqa: E402


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "1234567812345678 12/27 123",
            {"pan": "1234567812345678", "exp_month": 12, "exp_year": 27, "cvv": "123"},
        ),
        (
            "1234 5678 1234 5678 12/27 123",
            {"pan": "1234567812345678", "exp_month": 12, "exp_year": 27, "cvv": "123"},
        ),
        (
            "1234-5678-1234-5678 12/2027 1234",
            {"pan": "1234567812345678", "exp_month": 12, "exp_year": 27, "cvv": "1234"},
        ),
        (
            "4111 1111 1111 1111  03/26  999",
            {"pan": "4111111111111111", "exp_month": 3, "exp_year": 26, "cvv": "999"},
        ),
        (
            "PAN: 5555 4444 3333 2222 EXP 09/29 CVV 707",
            {"pan": "5555444433332222", "exp_month": 9, "exp_year": 29, "cvv": "707"},
        ),
    ],
)
def test_parse_valid(text: str, expected: dict[str, object]) -> None:
    out = ppc.parse(text)
    assert out is not None, f"expected match for {text!r}"
    for k, v in expected.items():
        assert out[k] == v, f"{k}: got {out[k]!r}, expected {v!r}"
    assert out["last4"] == expected["pan"][-4:]


@pytest.mark.parametrize(
    "text",
    [
        "coucou pas de carte",
        "123 12/27",  # PAN too short
        "12345678 12-27 123",  # dash separator on expiry not accepted
        "",
    ],
)
def test_parse_invalid(text: str) -> None:
    assert ppc.parse(text) is None
