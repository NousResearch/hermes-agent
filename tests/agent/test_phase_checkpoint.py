from agent.phase_checkpoint import PhaseCheckpoint


def test_phase_checkpoint_omits_raw_base64_and_truncates():
    raw = "data:image/png;base64," + ("A" * 5000)
    checkpoint = PhaseCheckpoint(
        objective="verify UI",
        changes=[f"stored screenshot {raw}"],
        verification_passed=["pytest ok"],
        next_phase_inputs=["x" * 3000],
        artifact_paths=["/tmp/capture.png"],
        stop_reason="fast path verified",
    )

    text = checkpoint.to_json()

    assert "data:image" not in text
    assert "base64,AAAA" not in text
    assert "/tmp/capture.png" in text
    assert "checkpoint field truncated" in text

