"""Provider-free contract for control-plane references across compaction."""

from unittest.mock import patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
)


def test_forced_compaction_preserves_operational_reference_envelope():
    with patch("agent.context_compressor.get_model_context_length", return_value=64_000):
        compressor = ContextCompressor(
            model="test/model",
            config_context_length=64_000,
            protect_first_n=1,
            protect_last_n=2,
            tail_mode="lean",
            quiet_mode=True,
        )
    compressor._session_id = "session-2026-001"

    refs = {
        "task_id": "task-human-001",
        "session_id": "session-2026-001",
        "worker_id": "worker-17",
        "parent_task_id": "task-parent-001",
        "child_task_id": "task-child-002",
        "browser_task_id": "browser-task-009",
        "result_ref": "sha256:result-abc123",
        "approval_state": "pending",
        "artifact_ref": "artifact-77",
        "evidence_ref": "evidence-88",
    }
    messages = [{"role": "system", "content": "system"}]
    for index in range(42):
        if index == 4:
            content = "Operational checkpoint: " + " ".join(
                f"{label}: {value}" for label, value in refs.items()
            )
        else:
            content = f"historical turn {index} " + ("x" * 1_200)
        messages.extend([
            {"role": "user", "content": content},
            {"role": "assistant", "content": f"acknowledged turn {index} " + ("y" * 1_200)},
        ])

    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("provider unavailable")):
        compacted = compressor.compress(messages, current_tokens=60_000, force=True)

    summaries = [
        message["content"]
        for message in compacted
        if message.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert "## Operational References (exact, do not paraphrase)" in summary
    for value in refs.values():
        assert value in summary
