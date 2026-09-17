"""Tests for TypeSafe System One protocol parsing (#113850)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.computer_use.system_one import (
    ProtocolError,
    TransportError,
    ask,
    parse_systemone_response,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "system_one"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_choice_preserves_full_distribution():
    fx = load("official-choice.json")
    parsed = parse_systemone_response(fx["response"], fx["request"]["questions"])
    ans = parsed.answers["department"]
    assert ans.choice == "technical"
    assert ans.probabilities == {"billing": 0.08, "technical": 0.85, "sales": 0.07}
    assert ans.confidence == 0.82


def test_noul_preserves_probability():
    fx = load("official-noul.json")
    parsed = parse_systemone_response(fx["response"], fx["request"]["questions"])
    ans = parsed.answers["is_urgent"]
    assert ans.noul == 0.92
    assert ans.probabilities["true"] == 0.92


def test_injected_transport_roundtrip():
    fx = load("official-choice.json")

    def transport(payload):
        assert payload["questions"] == fx["request"]["questions"]
        return fx["response"]

    parsed = ask(transport, fx["request"]["state"], fx["request"]["questions"])
    assert parsed.answers["department"].choice == "technical"


def test_transport_error_propagates():
    def boom(_payload):
        raise TransportError("down")

    with pytest.raises(TransportError):
        ask(boom, "state", load("official-choice.json")["request"]["questions"])


def test_rejects_mismatched_probability_keys():
    fx = load("official-choice.json")
    bad = json.loads(json.dumps(fx["response"]))
    bad["answers"]["department"]["probabilities"] = {"technical": 1.0}
    with pytest.raises(ProtocolError):
        parse_systemone_response(bad, fx["request"]["questions"])
