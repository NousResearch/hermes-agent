"""Durable interrupted-turn records in state.db.

A turn that starts running and never reaches a terminal frame leaves a row in
``interrupted_turns``; ``session.resume`` reads it to decide whether to continue
the interrupted prompt. These records used to live in a JSON sidecar that every
process rewrote in full and any process could delete. The properties pinned
here are the ones the file could not have:

* a write touches one conversation's row and no other's, even from a second
  process handle;
* only the process that recorded a turn may retire it, so a process that never
  ran the turn cannot delete the record of one that is still running;
* records imported from the legacy file carry no owner and stay retirable by
  anyone, and their keys are resolved to the compression-lineage root the table
  is keyed on.
"""

from __future__ import annotations

import os
import time

from hermes_state import SessionDB


def _owner(tag: str) -> str:
    return f"pid={os.getpid()}:platform={tag}"


def test_interrupted_turn_roundtrip(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    assert db.record_interrupted_turn(
        "abc", "fix the bug", attempts=1, owner=_owner("a")
    )

    record = db.read_interrupted_turn("abc")
    assert record is not None
    assert record["prompt"] == "fix the bug"
    assert record["attempts"] == 1
    assert record["owner"] == _owner("a")
    assert abs(record["started_at"] - time.time()) < 5

    assert db.clear_interrupted_turn("abc", owner=_owner("a"))
    assert db.read_interrupted_turn("abc") is None


def test_empty_prompt_records_nothing(tmp_path):
    db = SessionDB(tmp_path / "state.db")

    assert not db.record_interrupted_turn("abc", "   ", owner=_owner("a"))
    assert not db.record_interrupted_turn("", "prompt", owner=_owner("a"))
    assert db.read_interrupted_turn("abc") is None


def test_foreign_owner_cannot_retire_a_live_record(tmp_path):
    """The record of a running turn survives another process retiring it.

    This is the lease-timeout path: a second process submits on a conversation
    the first is already running, its engine waits for the turn lease, times
    out, and the gateway retires the record as it emits the terminal error
    frame. The turn it retired belongs to someone else.
    """
    path = tmp_path / "state.db"
    running = SessionDB(path)
    timing_out = SessionDB(path)

    running.record_interrupted_turn(
        "conv", "the turn that is actually running", owner=_owner("running")
    )

    assert not timing_out.clear_interrupted_turn("conv", owner=_owner("timing-out"))

    survivor = running.read_interrupted_turn("conv")
    assert survivor is not None
    assert survivor["prompt"] == "the turn that is actually running"
    assert running.clear_interrupted_turn("conv", owner=_owner("running"))


def test_write_does_not_clobber_another_conversation(tmp_path):
    """Two handles recording different conversations both keep their record.

    The sidecar loaded the whole map, changed one key and stored the map back,
    so a write for one conversation could drop a record for another. A row is
    a row.
    """
    path = tmp_path / "state.db"
    first = SessionDB(path)
    second = SessionDB(path)

    first.record_interrupted_turn("conv-a", "prompt for A", owner=_owner("first"))
    second.record_interrupted_turn("conv-b", "prompt for B", owner=_owner("second"))

    assert first.read_interrupted_turn("conv-a")["prompt"] == "prompt for A"
    assert first.read_interrupted_turn("conv-b")["prompt"] == "prompt for B"


def test_recording_takes_ownership_of_the_row(tmp_path):
    """A new turn's record replaces the spent one and carries the new owner.

    A turn only starts after the previous one ended, so the row it replaces
    describes a turn that is over — usually one whose process died, which is
    exactly the case whose attempts counter has to keep advancing for the
    crash-loop breaker to work.
    """
    path = tmp_path / "state.db"
    crashed = SessionDB(path)
    restarted = SessionDB(path)

    crashed.record_interrupted_turn(
        "conv", "the interrupted prompt", attempts=0, owner=_owner("crashed")
    )
    assert restarted.record_interrupted_turn(
        "conv", "the interrupted prompt", attempts=1, owner=_owner("restarted")
    )

    record = restarted.read_interrupted_turn("conv")
    assert record["attempts"] == 1
    assert record["owner"] == _owner("restarted")
    # The dead process's owner string no longer retires it; the live one does.
    assert not restarted.clear_interrupted_turn("conv", owner=_owner("crashed"))
    assert restarted.clear_interrupted_turn("conv", owner=_owner("restarted"))


def test_force_retires_regardless_of_owner(tmp_path):
    """The scheduler's policy deletion: past every window, actionable by none."""
    db = SessionDB(tmp_path / "state.db")
    db.record_interrupted_turn("conv", "stale prompt", owner=_owner("someone-else"))

    assert db.clear_interrupted_turn("conv", owner=_owner("scheduler"), force=True)
    assert db.read_interrupted_turn("conv") is None


def test_imported_record_has_no_owner_and_stays_retirable(tmp_path):
    """Legacy records never recorded an owner, so nobody is locked out of them."""
    db = SessionDB(tmp_path / "state.db")

    assert db.import_interrupted_turns(
        [("conv", {"prompt": "legacy prompt", "attempts": 1, "started_at": time.time()})]
    ) == 1

    record = db.read_interrupted_turn("conv")
    assert record["prompt"] == "legacy prompt"
    assert record["attempts"] == 1
    assert record["owner"] is None
    assert record["cause"] == "migrated"

    assert db.clear_interrupted_turn("conv", owner=_owner("any-process"))
    assert db.read_interrupted_turn("conv") is None


def test_import_does_not_overwrite_a_live_record(tmp_path):
    """A row written by a running process is newer than anything being imported."""
    db = SessionDB(tmp_path / "state.db")
    db.record_interrupted_turn("conv", "live prompt", owner=_owner("live"))

    assert db.import_interrupted_turns(
        [("conv", {"prompt": "legacy prompt", "started_at": time.time() - 60})]
    ) == 0

    record = db.read_interrupted_turn("conv")
    assert record["prompt"] == "live prompt"
    assert record["owner"] == _owner("live")


def test_records_are_keyed_on_the_conversation_root(tmp_path):
    """Compression segments share one record, as they share one turn lease."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")

    db.record_interrupted_turn("root", "prompt before rotation", owner=_owner("a"))

    # The resume after the rotation looks the record up under the child id.
    record = db.read_interrupted_turn("child")
    assert record is not None
    assert record["prompt"] == "prompt before rotation"


def test_import_translates_a_segment_key_to_the_root(tmp_path):
    """Legacy records were filed under the segment, not the lineage root.

    Without the translation a record written before a rotation would import to
    a row no post-rotation resume ever reads.
    """
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")

    assert db.import_interrupted_turns(
        [(
            "root",
            {"prompt": "prompt from before the rotation", "started_at": time.time()},
        )]
    ) == 1

    record = db.read_interrupted_turn("child")
    assert record is not None
    assert record["prompt"] == "prompt from before the rotation"
    assert record["owner"] is None


def test_two_legacy_segments_collapse_to_one_row_newest_first(tmp_path):
    """Both segments of a rotated conversation resolve to the same row."""
    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", source="test")
    db.end_session("root", "compression")
    db.create_session("child", source="test", parent_session_id="root")
    now = time.time()

    written = db.import_interrupted_turns([
        ("child", {"prompt": "after the rotation", "started_at": now}),
        ("root", {"prompt": "before the rotation", "started_at": now - 600}),
    ])

    assert written == 1
    assert db.read_interrupted_turn("child")["prompt"] == "after the rotation"


def test_records_older_than_a_day_are_swept_on_write(tmp_path):
    """Same bound the sidecar enforced on every write."""
    db = SessionDB(tmp_path / "state.db")
    db.import_interrupted_turns([
        ("ancient", {"prompt": "two days ago", "started_at": time.time() - 48 * 3600}),
        ("recent", {"prompt": "an hour ago", "started_at": time.time() - 3600}),
    ])
    assert db.read_interrupted_turn("ancient") is not None

    db.record_interrupted_turn("fresh", "now", owner=_owner("a"))

    assert db.read_interrupted_turn("ancient") is None
    assert db.read_interrupted_turn("recent") is not None
    assert db.read_interrupted_turn("fresh") is not None


def test_missing_record_reads_as_none(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    assert db.read_interrupted_turn("nobody") is None
    assert not db.clear_interrupted_turn("nobody", owner=_owner("a"))
