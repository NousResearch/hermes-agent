from agent.provider_response_gate import ProviderResponseGate


def test_gate_overflow_fails_open_prefix_once_then_passes_through():
    gate = ProviderResponseGate(max_bytes=5)

    consumed, fail_open = gate.capture("abc")
    assert consumed is True
    assert fail_open == []

    consumed, fail_open = gate.capture("def")
    assert consumed is True
    assert fail_open == ["abc", "def"]
    assert gate.overflowed is True
    assert gate.drain() == []

    consumed, fail_open = gate.capture("later")
    assert consumed is False
    assert fail_open == []


def test_discard_gate_never_returns_private_provider_deltas():
    gate = ProviderResponseGate(discard=True, max_bytes=1)

    consumed, fail_open = gate.capture("private finalizer output")

    assert consumed is True
    assert fail_open == []
    assert gate.text == ""
    assert gate.overflowed is False
