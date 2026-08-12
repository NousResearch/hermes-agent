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
