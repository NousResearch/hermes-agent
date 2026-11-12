"""Channel state surviving context compression when a summary is merged into a tail row."""

import pytest
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        # Resolve context_length while the mock is still active so the
        # fixture returns a fully-initialized compressor.
        _ = c.context_length
        return c


def _snapshot_user(text, n=None):
    row = {"role": "user", "content": text}
    if n is not None:
        row["display_metadata"] = {"channel_state": {"n": n}}
    return row


def _plain_assistant(text):
    return {"role": "assistant", "content": text}


@pytest.mark.parametrize("messages,compress_start,compress_end,expected", [
    ([_snapshot_user("u0", 1), _plain_assistant("a0"), _snapshot_user("u1", 2), _plain_assistant("a1"),
      _snapshot_user("u2", 3), _plain_assistant("a2"), _plain_assistant("a3"), _snapshot_user("u3"),
      _plain_assistant("a4")], 3, 6,
     [("user", {"n": 1}), ("assistant", None), ("user", {"n": 2}), ("assistant", {"n": 3}), ("user", None),
      ("assistant", None)]),
    ([_snapshot_user("u0", 1), _plain_assistant("a0"), _snapshot_user("u1", 2), _plain_assistant("a1"),
      _snapshot_user("u2", 3), _plain_assistant("a2"), _plain_assistant("a3"), _snapshot_user("u3", 4),
      _plain_assistant("a4")], 3, 6,
     [("user", {"n": 1}), ("assistant", None), ("user", {"n": 2}), ("assistant", None), ("user", {"n": 4}),
      ("assistant", None)]),
    ([_snapshot_user("u0", 1), _plain_assistant("a0"), _snapshot_user("u1", 2), _plain_assistant("a1"),
      _snapshot_user("u2"), _plain_assistant("a2")], 0, 4,
     [("user", {"n": 2}), ("assistant", None)]),
], ids=["merged-into-tail", "newer-snapshot-in-tail", "merged-into-first-visible-tail-row"])
def test_summary_merged_into_a_tail_row_keeps_the_newest_channel_state(
    compressor, messages, compress_start, compress_end, expected,
):
    from agent.context_compressor import _HandoffScan

    scan = _HandoffScan(
        turns_to_summarize=messages[compress_start:compress_end], summary_indices=set(),
        tail_start=compress_end, previous_summary_before=None, has_user_turn_before=None,
    )

    compacted = compressor._assemble_compressed(messages, compress_start, compress_end, scan, "SUMMARY")

    assert [
        (m["role"], (m.get("display_metadata") or {}).get("channel_state")) for m in compacted
    ] == expected
