"""End-to-end compaction smoke test against the live local Qwen server.

Skipped automatically when http://127.0.0.1:8085/v1 is unreachable.
This is a behavior gate, not a unit test — its purpose is to catch
regressions in the moe-profile path that pure-Python unit tests miss.
"""

import socket
import pytest

from agent.context_compressor import ContextCompressor


def _qwen_server_up() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 8085), timeout=0.5):
            return True
    except OSError:
        return False


qwen_required = pytest.mark.skipif(
    not _qwen_server_up(),
    reason="local-qwen moe profile not running on :8085",
)


@qwen_required
def test_full_pipeline_with_qwen_aware_flags():
    """Compaction should produce a non-empty summary using qwen-instruct.

    Constructs a synthetic conversation that exercises:
      - Pass 1.5 op-keyed dedup (two read_file calls on same path with
        DIFFERENT content — Pass 1's hash dedup won't catch them)
      - The first-assistant anchor (boundary at user msg slides forward)
      - The CompactionResult metric population
    """
    c = ContextCompressor(
        model="qwen-instruct",
        threshold_percent=0.50,
        protect_first_n=1,            # tight enough to exercise the anchor
        protect_last_n=2,
        summary_target_ratio=0.20,
        quiet_mode=True,
        base_url="http://127.0.0.1:8085/v1",
        api_key="not-needed",
        provider="local-qwen",
        api_mode="chat_completions",
        config_context_length=262_144,
        # qwen-aware flags ON
        qwen_aware_enabled=True,
        dedup_operations=True,
        anchor_first_assistant=True,
        threshold_absolute_max=80_000,
        message_threshold=200,
        turn_threshold=30,
    )
    # Build a synthetic history with redundant reads (DIFFERENT content
    # so Pass 1's hash dedup misses them and Pass 1.5 catches them).
    # Note: assistant content here is already-stripped (no <think> tags) —
    # that mirrors what _build_assistant_message produces at storage
    # boundary, so it's the realistic shape the compressor sees.
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Inspect /etc/hostname twice and summarize."},
        {"role": "assistant", "content": "I'll read it.", "tool_calls": [
            {"id": "1", "function": {"name": "read_file",
             "arguments": '{"path":"/etc/hostname"}'}},
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "first-read-content-v1\n"},
        {"role": "assistant",
         "content": "Got it; reading again.",
         "tool_calls": [
             {"id": "2", "function": {"name": "read_file",
              "arguments": '{"path":"/etc/hostname"}'}},
         ]},
        {"role": "tool", "tool_call_id": "2", "content": "second-read-content-v2\n"},
        {"role": "assistant", "content": "Both reads complete."},
        {"role": "user", "content": "Now compact and tell me the gist."},
    ]
    out = c.compress(msgs, current_tokens=85_000)
    assert len(out) <= len(msgs)
    assert c.last_compaction_result is not None

    # Pass 1.5 should have superseded the FIRST tool result with a
    # back-reference (different content from the second, so Pass 1's
    # hash dedup doesn't catch it).
    assert c.last_compaction_result.operations_deduped >= 1, (
        f"Expected op-keyed dedup ≥ 1; got {c.last_compaction_result}"
    )

    # CompactionResult should attribute the trigger to "token" since we
    # passed current_tokens=85_000 above the 80_000 absolute cap.
    assert c.last_compaction_result.triggered_by == "token", (
        f"Expected token trigger; got {c.last_compaction_result.triggered_by}"
    )
