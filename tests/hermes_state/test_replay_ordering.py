"""Regression test for get_messages_as_conversation ordering.

A non-monotonic clock (WSL2 / NTP step) or a skewed platform-supplied
timestamp can stamp a tool result earlier than the assistant tool_call that
produced it. Ordering the replayed conversation by timestamp then places the
tool message before its assistant message, breaking the tool_call/tool_response
adjacency the provider requires and causing an HTTP 400 on resume.

PR #25774 (commit c03acca50) fixed exactly this by ordering message readers by
AUTOINCREMENT id instead of timestamp. It was later reintroduced on this replay
path by bd7fc8fdc ("inject stable human-readable message timestamps"), which is
what this test guards against.
"""
import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def test_replay_orders_by_id_not_timestamp(db):
    """A tool result stamped earlier than its assistant tool_call must still
    replay immediately after that assistant message (insertion order)."""
    sid = "s1"
    db.create_session(sid, source="cli")

    base = 1_000_000.0
    # Insertion order: user -> assistant(tool_calls) -> tool result.
    # Timestamps are deliberately NON-monotonic: the tool result (base + 1) is
    # stamped BEFORE the assistant message (base + 2) that called it, simulating
    # a clock regression between two appends within one multi-tool turn.
    db.append_message(sid, role="user", content="run the tool", timestamp=base)
    db.append_message(
        sid,
        role="assistant",
        content="",
        tool_calls=[
            {"id": "call_1", "type": "function",
             "function": {"name": "x", "arguments": "{}"}}
        ],
        timestamp=base + 2,
    )
    db.append_message(
        sid,
        role="tool",
        content="tool output",
        tool_call_id="call_1",
        tool_name="x",
        timestamp=base + 1,
    )

    conv = db.get_messages_as_conversation(sid)
    roles = [m["role"] for m in conv]

    # Must be insertion order. ORDER BY timestamp would yield
    # ["user", "tool", "assistant"] and orphan the tool result.
    assert roles == ["user", "assistant", "tool"], (
        f"replay reordered by timestamp instead of id: {roles}"
    )

    # The assistant tool_call must be immediately followed by its tool result.
    assert conv[1].get("tool_calls"), "assistant tool_calls missing from replay"
    assert conv[1]["tool_calls"][0]["id"] == "call_1"
    assert conv[2]["role"] == "tool"
    assert conv[2]["tool_call_id"] == "call_1"
