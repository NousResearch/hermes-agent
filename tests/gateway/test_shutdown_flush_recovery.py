"""Cross-restart recovery of cap-dropped transcript spool files (#78182).

``recover_pending_to_db`` is the restart-time consumer of the same spool
``drain_transcript_spool`` drains during live operation.  These tests pin the
properties the live drain already guarantees for that spool.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gateway.shutdown_flush import (
    TRANSCRIPT_CAP_DROP_REASON,
    recover_pending_to_db,
)


@pytest.fixture
def flush_dir(tmp_path, monkeypatch):
    """A temp spool directory wired into the module under test."""
    directory = tmp_path / "pending_messages"
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "gateway.shutdown_flush._get_flush_dir", lambda: directory
    )
    return directory


def _write_spool(
    flush_dir: Path,
    name: str,
    session_id: str,
    message: dict,
    *,
    ts: int,
    seq: int,
) -> Path:
    """Write one cap-drop spool payload under an explicit file name.

    Production names these ``pending-<uuid4>.json``; the tests choose the
    names so that filename order and drop order can be made to disagree.
    """
    path = flush_dir / name
    path.write_text(
        json.dumps(
            {
                "session_key": session_id,
                "reason": TRANSCRIPT_CAP_DROP_REASON,
                "ts": ts,
                "seq": seq,
                "data": {"session_id": session_id, "message": message},
            }
        ),
        encoding="utf-8",
    )
    return path


def _contents(mock_db) -> list:
    return [c.kwargs["content"] for c in mock_db.append_message.call_args_list]


def test_replays_in_drop_order_not_file_name_order(flush_dir):
    """SessionDB restores by AUTOINCREMENT id, so append order IS the order the user sees
    after recovery. Production names are ``pending-<uuid4>.json``; these names sort opposite
    to drop order, and a burst inside one second shares ``ts``, so ``seq`` breaks the tie."""
    _write_spool(flush_dir, "pending-zzz.json", "sess-1",
                 {"role": "user", "content": "first"}, ts=100, seq=0)
    _write_spool(flush_dir, "pending-mmm.json", "sess-1",
                 {"role": "assistant", "content": "second"}, ts=100, seq=1)
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "third"}, ts=101, seq=2)

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 3
    assert _contents(mock_db) == ["first", "second", "third"]


def test_an_out_of_range_ordering_field_does_not_abort_recovery(flush_dir):
    """Ordering runs before the per-file error handling, so one file whose ``ts`` is a JSON
    integer no float can hold must not stop every other file from being recovered."""
    _write_spool(flush_dir, "pending-bad.json", "sess-1",
                 {"role": "user", "content": "huge ts"}, ts=10**400, seq=0)
    _write_spool(flush_dir, "pending-ok.json", "sess-2",
                 {"role": "user", "content": "valid"}, ts=100, seq=1)

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 2
    assert sorted(_contents(mock_db)) == ["huge ts", "valid"]


def test_a_replayed_row_is_the_row_the_live_writer_writes(flush_dir):
    """Every field the live drain persists survives the restart round trip: losing
    tool_call_id orphans a tool result, losing api_content makes the next replay diverge,
    and reasoning stays assistant-only. A missing timestamp falls back to the payload
    clock, but epoch 0 is a real timestamp."""
    tool_calls = [{"id": "call-1", "type": "function",
                   "function": {"name": "send_payment", "arguments": "{}"}}]
    _write_spool(flush_dir, "pending-ccc.json", "sess-1", {
        "role": "assistant", "content": None, "tool_calls": tool_calls,
        "reasoning": "deliberating", "reasoning_content": "chain",
        "reasoning_details": [{"type": "text"}], "codex_reasoning_items": [{"id": "r1"}],
        "codex_message_items": [{"id": "m1"}], "platform_message_id": "tg-42",
        "observed": True, "timestamp": 0, "api_content": "exact bytes sent to the API",
        "display_kind": "internal_notification",
    }, ts=100, seq=0)
    _write_spool(flush_dir, "pending-bbb.json", "sess-1", {
        "role": "tool", "content": "receipt-1", "tool_call_id": "call-1", "tool_name": "send_payment",
    }, ts=100, seq=1)
    _write_spool(flush_dir, "pending-aaa.json", "sess-1", {
        "role": "user", "content": "hi", "reasoning": "leaked", "message_id": "tg-7",
    }, ts=999, seq=2)

    mock_db = MagicMock()
    assert recover_pending_to_db(mock_db) == 3
    assistant, tool, user = (c.kwargs for c in mock_db.append_message.call_args_list)

    assert assistant["content"] is None  # a tool-call row legitimately has no content
    assert assistant["tool_calls"] == tool_calls
    assert (assistant["reasoning"], assistant["reasoning_content"]) == ("deliberating", "chain")
    assert assistant["reasoning_details"] == [{"type": "text"}]
    assert assistant["codex_reasoning_items"] == [{"id": "r1"}]
    assert assistant["codex_message_items"] == [{"id": "m1"}]
    assert (assistant["platform_message_id"], assistant["observed"]) == ("tg-42", True)
    assert assistant["api_content"] == "exact bytes sent to the API"
    assert assistant["display_kind"] == "internal_notification"
    assert assistant["timestamp"] == 0
    assert (tool["tool_call_id"], tool["tool_name"]) == ("call-1", "send_payment")
    assert user["reasoning"] is None
    assert user["platform_message_id"] == "tg-7"
    assert user["timestamp"] == 999


def test_a_failed_replay_holds_back_that_sessions_later_messages_only(flush_dir, caplog):
    """Writing "second" after "first" failed would give it a lower row id than "first" once
    "first" is retried on a later start: the inversion lands on disk for good. Another
    session is unaffected, and the pass says what it held back."""
    first = _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                         {"role": "user", "content": "first"}, ts=100, seq=0)
    second = _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                          {"role": "user", "content": "second"}, ts=101, seq=1)
    other = _write_spool(flush_dir, "pending-ccc.json", "sess-2",
                         {"role": "user", "content": "other-session"}, ts=102, seq=2)

    mock_db = MagicMock()

    def append_message(**kwargs):
        if kwargs["content"] == "first":
            raise RuntimeError("controlled database outage")
        return 1

    mock_db.append_message.side_effect = append_message

    with caplog.at_level("INFO", logger="gateway.shutdown_flush"):
        assert recover_pending_to_db(mock_db) == 1

    assert _contents(mock_db) == ["first", "other-session"]
    assert first.exists() and second.exists()
    assert not other.exists()
    assert "Held back 2 spooled transcript file(s)" in caplog.text and "sess-1 (2)" in caplog.text
