"""Exercise native event parsing; fixtures are not live provider evidence."""
from types import SimpleNamespace

import pytest

from agent.codex_runtime import _consume_codex_event_stream


@pytest.mark.parametrize("terminal", ["completed", "incomplete", "failed"])
@pytest.mark.parametrize("wire_shape", [dict, SimpleNamespace])
def test_observed_model_preserves_terminal_wire_mismatch(terminal, wire_shape):
    response = _consume_codex_event_stream(iter([wire_shape(
        type=f"response.{terminal}",
        response=wire_shape(id="fixture-response", model="served-model", status=terminal),
    )]), model="requested-model")
    assert response.model == "requested-model"  # compatibility behavior unchanged
    assert response.observed_model == "served-model"
    assert response.status == terminal


@pytest.mark.parametrize("model", [None, "", "   ", 123, True, []])
def test_missing_or_invalid_wire_model_stays_unknown(model):
    response = _consume_codex_event_stream(iter([{
        "type": "response.completed", "response": {"model": model},
    }]), model="requested-model")
    assert response.observed_model is None
    assert response.model == "requested-model"


def test_partial_stream_never_uses_requested_model_as_observed():
    response = _consume_codex_event_stream(iter([{
        "type": "response.output_text.delta", "delta": "fixture text",
    }]), model="requested-model")
    assert response.output_text == "fixture text"
    assert response.observed_model is None


def test_observed_model_does_not_leak_between_requests():
    first = _consume_codex_event_stream(iter([{
        "type": "response.completed", "response": {"model": "served"},
    }]), model="requested")
    second = _consume_codex_event_stream(iter([{
        "type": "response.completed", "response": {},
    }]), model="requested")
    assert first.observed_model == "served"
    assert second.observed_model is None
