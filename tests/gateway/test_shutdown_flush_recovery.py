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


class TestSpoolReplayOrder:
    """Spool files must replay in drop order, not in file-name order."""

    def test_replays_in_drop_order_when_names_disagree(self, flush_dir):
        # Drop order is first -> second -> third; the uuid4-style names sort
        # in exactly the opposite direction, which is what a real spool does
        # on average.
        _write_spool(
            flush_dir, "pending-ccc.json", "sess-1",
            {"role": "user", "content": "first"}, ts=100, seq=0,
        )
        _write_spool(
            flush_dir, "pending-bbb.json", "sess-1",
            {"role": "assistant", "content": "second"}, ts=101, seq=1,
        )
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "third"}, ts=102, seq=2,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 3

        # SessionDB restores by AUTOINCREMENT id, so append order IS the
        # order the user will see after recovery.
        assert _contents(mock_db) == ["first", "second", "third"]

    def test_seq_breaks_ties_within_the_same_second(self, flush_dir):
        # ts has one-second resolution, so a burst of cap drops shares a ts
        # and only ``seq`` can order them.
        _write_spool(
            flush_dir, "pending-zzz.json", "sess-1",
            {"role": "user", "content": "first"}, ts=100, seq=0,
        )
        _write_spool(
            flush_dir, "pending-mmm.json", "sess-1",
            {"role": "user", "content": "second"}, ts=100, seq=1,
        )
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "third"}, ts=100, seq=2,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 3
        assert _contents(mock_db) == ["first", "second", "third"]

    def test_unparseable_payload_is_reported_and_preserved(
        self, flush_dir, caplog
    ):
        """A corrupt file must not break ordering or the recovery pass."""
        broken = flush_dir / "pending-aaa.json"
        broken.write_text("{not json", encoding="utf-8")
        _write_spool(
            flush_dir, "pending-zzz.json", "sess-1",
            {"role": "user", "content": "survivor"}, ts=100, seq=0,
        )

        mock_db = MagicMock()
        with caplog.at_level("WARNING"):
            assert recover_pending_to_db(mock_db) == 1

        assert _contents(mock_db) == ["survivor"]
        # The corrupt file is kept for the operator, and the failure is still
        # reported by the loop's own handler.
        assert broken.exists()
        assert "Failed to recover pending message" in caplog.text


class TestSpoolReplayFidelity:
    """The full transcript message must survive the restart round trip."""

    def test_structured_fields_are_preserved(self, flush_dir):
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "send_payment", "arguments": "{}"},
            }
        ]
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
            "reasoning": "deliberating",
            "reasoning_content": "chain",
            "reasoning_details": [{"type": "text"}],
            "codex_reasoning_items": [{"id": "r1"}],
            "codex_message_items": [{"id": "m1"}],
            "platform_message_id": "tg-42",
            "observed": True,
            "timestamp": 12345,
            "api_content": "exact bytes sent to the API",
        }
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1", message, ts=100, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1

        kwargs = mock_db.append_message.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["role"] == "assistant"
        # An assistant tool-call row legitimately has no content; forcing it
        # to "" would rewrite the message.
        assert kwargs["content"] is None
        assert kwargs["tool_calls"] == tool_calls
        assert kwargs["reasoning"] == "deliberating"
        assert kwargs["reasoning_content"] == "chain"
        assert kwargs["reasoning_details"] == [{"type": "text"}]
        assert kwargs["codex_reasoning_items"] == [{"id": "r1"}]
        assert kwargs["codex_message_items"] == [{"id": "m1"}]
        assert kwargs["platform_message_id"] == "tg-42"
        assert kwargs["observed"] is True
        assert kwargs["timestamp"] == 12345
        assert kwargs["api_content"] == "exact bytes sent to the API"

    def test_tool_result_keeps_its_call_id_and_name(self, flush_dir):
        """Losing tool_call_id orphans the result and breaks replay."""
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {
                "role": "tool",
                "content": "receipt-1",
                "tool_call_id": "call-1",
                "tool_name": "send_payment",
            },
            ts=100, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1

        kwargs = mock_db.append_message.call_args.kwargs
        assert kwargs["tool_call_id"] == "call-1"
        assert kwargs["tool_name"] == "send_payment"

    def test_reasoning_is_not_fabricated_for_non_assistant_roles(
        self, flush_dir
    ):
        """Mirrors the live writer, which gates reasoning on role."""
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "hi", "reasoning": "leaked"},
            ts=100, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1
        assert mock_db.append_message.call_args.kwargs["reasoning"] is None

    def test_message_id_is_used_when_platform_message_id_is_absent(
        self, flush_dir
    ):
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "hi", "message_id": "tg-7"},
            ts=100, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1
        assert (
            mock_db.append_message.call_args.kwargs["platform_message_id"]
            == "tg-7"
        )

    def test_epoch_zero_timestamp_is_not_replaced_by_the_fallback(
        self, flush_dir
    ):
        """0 is a valid timestamp; only a missing one may fall back."""
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "hi", "timestamp": 0}, ts=999, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1
        assert mock_db.append_message.call_args.kwargs["timestamp"] == 0

    def test_payload_ts_is_the_timestamp_fallback(self, flush_dir):
        _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "hi"}, ts=999, seq=0,
        )

        mock_db = MagicMock()
        assert recover_pending_to_db(mock_db) == 1
        assert mock_db.append_message.call_args.kwargs["timestamp"] == 999


class TestSpoolReplayFailure:
    """A failed replay must not let newer messages overtake older ones."""

    def test_failure_blocks_later_messages_for_that_session_only(
        self, flush_dir
    ):
        first = _write_spool(
            flush_dir, "pending-aaa.json", "sess-1",
            {"role": "user", "content": "first"}, ts=100, seq=0,
        )
        second = _write_spool(
            flush_dir, "pending-bbb.json", "sess-1",
            {"role": "user", "content": "second"}, ts=101, seq=1,
        )
        other = _write_spool(
            flush_dir, "pending-ccc.json", "sess-2",
            {"role": "user", "content": "other-session"}, ts=102, seq=2,
        )

        mock_db = MagicMock()

        def append_message(**kwargs):
            if kwargs["content"] == "first":
                raise RuntimeError("controlled database outage")
            return 1

        mock_db.append_message.side_effect = append_message

        assert recover_pending_to_db(mock_db) == 1

        # "second" must NOT be written: it would land with a lower row id
        # than "first" once "first" is retried on a later start.
        assert _contents(mock_db) == ["first", "other-session"]
        assert first.exists()
        assert second.exists()
        # A different session is unaffected by sess-1's outage.
        assert not other.exists()
