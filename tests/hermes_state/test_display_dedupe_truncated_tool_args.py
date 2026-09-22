"""A compaction generation that truncates tool-call arguments still renders one message.

``ContextCompressor._truncate_tool_call_args_at`` deliberately shrinks large tool_call
``arguments`` inside the parsed JSON before the protected tail is carried into the next
generation. ``_display_dedupe_key`` compared the serialized ``tool_calls`` column verbatim,
so the carried copy hashed to a different display identity than its original and the
transcript rendered the same assistant turn twice (once per generation).

The durable identity of a tool call is the provider-assigned call id, which survives the
truncation; the argument text does not.
"""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _assistant_with_args(args, timestamp=None, call_id="toolu_01ABC"):
    msg = {
        "role": "assistant",
        "content": "running it",
        "tool_calls": [{
            "id": call_id,
            "call_id": call_id,
            "type": "function",
            "function": {"name": "terminal", "arguments": args},
        }],
    }
    if timestamp is not None:
        msg["timestamp"] = timestamp
    return msg


FULL_ARGS = '{"command": "echo ' + "x" * 600 + '"}'
TRUNCATED_ARGS = '{"command": "echo xxx ...[truncated]"}'


def test_truncated_tool_call_args_do_not_split_one_turn(db):
    """The carried tail keeps one display row even when its arguments were shrunk."""
    sid = "s1"
    db.create_session(sid, source="cli")
    seeded = [{"role": "user", "content": "q1"}, _assistant_with_args(FULL_ARGS)]
    db.append_messages_batch(sid, seeded)
    # Compaction carries the same logical turn forward — same timestamp, arguments shrunk.
    # The original stays as display history (active=0, compacted=1) beside the live copy.
    db.archive_and_compact(sid, [
        {"role": "assistant", "content": "summary of old turns"},
        _assistant_with_args(TRUNCATED_ARGS, timestamp=seeded[1]["timestamp"]),
    ])

    rows = db.get_messages(sid, include_compacted=True)
    carried = [m for m in rows if m["role"] == "assistant" and m.get("tool_calls")]
    assert len(carried) == 1, f"tool-call turn rendered {len(carried)} times"


def test_distinct_tool_calls_still_render_separately(db):
    """Dedupe keys on the call id, so two real calls must not collapse into one."""
    sid = "s2"
    db.create_session(sid, source="cli")
    db.append_messages_batch(sid, [
        {"role": "user", "content": "q1"},
        _assistant_with_args(FULL_ARGS),
        _assistant_with_args(FULL_ARGS, call_id="toolu_01XYZ"),
    ])

    rows = db.get_messages(sid, include_compacted=True)
    call_ids = [tc["id"] for m in rows for tc in (m.get("tool_calls") or [])]
    assert call_ids == ["toolu_01ABC", "toolu_01XYZ"]
