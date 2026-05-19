"""Tests for the Codex-runtime clarify IPC bridge."""

from __future__ import annotations

import pytest

from agent.transports.clarify_bridge import (
    ClarifyBridgeServer,
    request_clarify_via_bridge,
)


def test_clarify_bridge_round_trips_question_and_choices():
    captured = {}

    def callback(question, choices):
        captured["question"] = question
        captured["choices"] = choices
        return "blue"

    server = ClarifyBridgeServer(callback, token="test-token").start()
    try:
        answer = request_clarify_via_bridge(
            address=server.address,
            token="test-token",
            question="Pick a color?",
            choices=["blue", "green"],
            timeout=2.0,
        )
    finally:
        server.close()

    assert answer == "blue"
    assert captured == {
        "question": "Pick a color?",
        "choices": ["blue", "green"],
    }


def test_clarify_bridge_rejects_invalid_token():
    server = ClarifyBridgeServer(lambda _q, _c: "unused", token="real").start()
    try:
        with pytest.raises(RuntimeError, match="invalid clarify bridge token"):
            request_clarify_via_bridge(
                address=server.address,
                token="wrong",
                question="Nope?",
                timeout=2.0,
            )
    finally:
        server.close()
