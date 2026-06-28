from pathlib import Path

from agent.controlled_context_rebuild import (
    CONTROLLED_CONTEXT_REBUILD_HEADER,
    append_controlled_rebuild_to_summary,
    build_controlled_rebuild_packet,
    write_controlled_rebuild_checkpoint,
)


def test_packet_preserves_exact_literals_and_filters_compression_noise():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "user",
            "content": "[Your active task list was preserved across context compression]\n- [>] stale old task",
        },
        {
            "role": "user",
            "content": "давай внедри MiMo rebuild в /home/niko/.hermes/hermes-agent",
        },
        {
            "role": "assistant",
            "content": "Буду править agent/conversation_compression.py",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command":"pytest tests/agent/test_controlled_context_rebuild.py -q","workdir":"/home/niko/.hermes/hermes-agent"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "FAILED tests/test_x.py::test_y - AssertionError: TS2835 relative import",
        },
    ]

    packet = build_controlled_rebuild_packet("session-1", messages, budget=6000)

    assert packet.startswith(CONTROLLED_CONTEXT_REBUILD_HEADER)
    assert "/home/niko/.hermes/hermes-agent" in packet
    assert "agent/conversation_compression.py" in packet
    assert "pytest tests/agent/test_controlled_context_rebuild.py -q" in packet
    assert "TS2835 relative import" in packet
    assert "stale old task" not in packet


def test_checkpoint_write_is_profile_local_and_budgeted(tmp_path):
    messages = [
        {"role": "user", "content": "сохрани exact literal /tmp/example.py"},
        {"role": "assistant", "content": "done"},
    ]

    path = write_controlled_rebuild_checkpoint(
        "sid/with/slash", messages, hermes_home=tmp_path, budget=1200
    )

    assert (
        path == tmp_path / "context" / "sessions" / "sid_with_slash" / "checkpoint.md"
    )
    content = path.read_text(encoding="utf-8")
    assert CONTROLLED_CONTEXT_REBUILD_HEADER in content
    assert "/tmp/example.py" in content
    assert len(content) <= 1200


def test_append_controlled_rebuild_to_summary_is_idempotent_and_secret_safe():
    messages = [
        {"role": "user", "content": "token ghp_abcdefghijklmnopqrstuvwxyz123456"},
        {"role": "assistant", "content": "patched /repo/file.py"},
    ]

    summary = "## Goal\nShip it"
    combined = append_controlled_rebuild_to_summary(
        summary, "sid", messages, budget=4000
    )
    combined2 = append_controlled_rebuild_to_summary(
        combined, "sid", messages, budget=4000
    )

    assert combined == combined2
    assert combined.startswith(CONTROLLED_CONTEXT_REBUILD_HEADER)
    assert "## LLM Compaction Summary" in combined
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in combined
    assert "[REDACTED]" in combined
