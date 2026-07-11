from agent.kanban_delivery_policy import ArchitectureDeliveryPolicy


def test_unresolved_gate_withholds_all_delivery_shapes():
    policy = ArchitectureDeliveryPolicy(gate_id="gate-1", state="validated_awaiting_approval")

    assert policy.stream_delta("secret streamed prose") is None
    assert policy.interim("secret interim prose") is None
    assert policy.final("secret final prose") == (
        "Architecture approval pending; output withheld (gate gate-1)."
    )
    assert "secret" not in policy.receipt


def test_human_approved_gate_preserves_delivery():
    policy = ArchitectureDeliveryPolicy(gate_id="gate-1", state="human_approved")

    assert policy.stream_delta("visible") == "visible"
    assert policy.interim("visible") == "visible"
    assert policy.final("visible") == "visible"
