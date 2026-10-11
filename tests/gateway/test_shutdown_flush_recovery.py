"""Cross-restart recovery of cap-dropped transcript spool files (#78182).

``recover_pending_to_db`` is the restart-time consumer of the same spool
``drain_transcript_spool`` drains during live operation.  These tests pin the
properties the live drain already guarantees for that spool.
"""

import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gateway.shutdown_flush import (
    QUARANTINE_SUFFIX,
    TRANSCRIPT_CAP_DROP_REASON,
    _order_flush_files,
    drain_transcript_spool,
    is_row_rejection,
    recover_gateway_pending,
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
    monkeypatch.setattr("gateway.shutdown_flush._REPLAYED_UNREMOVABLE", set())
    return directory


@pytest.fixture
def make_store(tmp_path):
    """Build a real ``SessionStore`` whose transcript writes go to *db*. It goes through the
    constructor, so new store state does not break these tests."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    def build(db):
        with patch("gateway.session.SessionStore._ensure_loaded"):
            store = SessionStore(sessions_dir=tmp_path / "sessions", config=GatewayConfig())
        store._loaded = True
        store._db = db
        return store

    return build


@pytest.fixture
def boot_db(monkeypatch):
    """Make boot recovery's default state.db the given handle."""
    import hermes_state_registry

    def use(db):
        monkeypatch.setattr(hermes_state_registry, "acquire", lambda: db)
        monkeypatch.setattr(hermes_state_registry, "release_or_close", lambda _db: None)

    return use


def _boot(store) -> int:
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False), session_store=store)
    return recover_gateway_pending(runner)


def _write_spool(
    flush_dir: Path,
    name: str,
    session_id: str,
    message: dict,
    *,
    ts: Any,
    seq: Any,
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


def _fail_unlink_for(monkeypatch, name: str) -> None:
    """Make deleting the spool file *name* fail, as on a directory that stopped being writable."""
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self.name == name:
            raise PermissionError(13, "Permission denied", str(self))
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)


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


def test_live_drain_and_restart_pass_order_malformed_files_alike(flush_dir):
    """Every live write waits on the live drain, so a ``ts`` it cannot compare must not raise there,
    and both passes must replay a session's files in the same order."""
    _write_spool(flush_dir, "pending-a.json", "sess-1",
                 {"role": "user", "content": "string ts"}, ts="later", seq=0)
    _write_spool(flush_dir, "pending-b.json", "sess-1",
                 {"role": "user", "content": "nan ts"}, ts=math.nan, seq=1)
    _write_spool(flush_dir, "pending-c.json", "sess-1",
                 {"role": "user", "content": "ts 5"}, ts=5, seq=2)

    restart = [json.loads(path.read_text())["data"]["message"]["content"]
               for path in _order_flush_files(flush_dir.glob("*.json"))]
    live = []
    assert drain_transcript_spool("sess-1", lambda message: live.append(message["content"])) == (3, 0)

    assert live == restart == ["string ts", "nan ts", "ts 5"]


def test_a_replayed_row_is_the_row_the_live_writer_writes(flush_dir, make_store):
    """A replayed spool row must be the row the live writer would have written for the same
    message: losing tool_call_id orphans a tool result, losing api_content makes the next replay
    diverge. Only a message with no timestamp differs: it takes the payload clock, and epoch 0 is
    a real timestamp."""
    tool_calls = [{"id": "call-1", "type": "function",
                   "function": {"name": "send_payment", "arguments": "{}"}}]
    assistant = {
        "role": "assistant", "content": None, "tool_calls": tool_calls,
        "reasoning": "deliberating", "reasoning_content": "chain",
        "reasoning_details": [{"type": "text"}], "codex_reasoning_items": [{"id": "r1"}],
        "codex_message_items": [{"id": "m1"}], "platform_message_id": "tg-42",
        "observed": True, "timestamp": 0, "api_content": "exact bytes sent to the API",
        "display_kind": "internal_notification",
    }
    tool = {"role": "tool", "content": "receipt-1", "tool_call_id": "call-1", "tool_name": "send_payment"}
    user = {"role": "user", "content": "hi", "reasoning": "leaked", "message_id": "tg-7"}
    _write_spool(flush_dir, "pending-ccc.json", "sess-1", assistant, ts=100, seq=0)
    _write_spool(flush_dir, "pending-bbb.json", "sess-1", tool, ts=100, seq=1)
    _write_spool(flush_dir, "pending-aaa.json", "sess-1", user, ts=999, seq=2)

    recovery_db, live_db = MagicMock(), MagicMock()
    assert recover_pending_to_db(recovery_db) == 3
    store = make_store(live_db)
    for message in (assistant, tool, user):
        store._append_transcript_message("sess-1", message)

    replayed = [c.kwargs for c in recovery_db.append_message.call_args_list]
    written = [c.kwargs for c in live_db.append_message.call_args_list]
    assert [row.pop("timestamp") for row in replayed] == [0, 100, 999]
    assert [row.pop("timestamp") for row in written] == [0, None, None]
    assert replayed == written
    assert replayed[0]["tool_calls"] == tool_calls
    assert replayed[0]["api_content"] == "exact bytes sent to the API"
    assert (replayed[1]["tool_call_id"], replayed[1]["tool_name"]) == ("call-1", "send_payment")
    assert (replayed[2]["reasoning"], replayed[2]["platform_message_id"]) == (None, "tg-7")


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


def test_held_back_files_drain_before_that_sessions_next_live_write(flush_dir, make_store, boot_db):
    """The live writer drains a session's spool only when the store knows it has one. If boot
    recovery does not say which sessions it held back, the next live row lands ahead of them."""
    _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                 {"role": "user", "content": "old0"}, ts=100, seq=0)
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "old1"}, ts=100, seq=1)
    mock_db = MagicMock()
    mock_db.append_message.side_effect = [RuntimeError("still locked at boot"), 1, 1, 1]
    boot_db(mock_db)
    store = make_store(mock_db)

    assert _boot(store) == 0
    store.append_to_transcript("sess-1", {"role": "user", "content": "new-live"})

    assert _contents(mock_db) == ["old0", "old0", "old1", "new-live"]
    assert not list(flush_dir.glob("*.json"))


def test_live_write_waits_while_held_back_files_still_fail_to_drain(flush_dir, make_store, boot_db):
    """If the pre-write drain fails again, the live row must stay queued in memory: writing it
    while the older files are still on disk gives it a lower row id than they will get."""
    _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                 {"role": "user", "content": "old0"}, ts=100, seq=0)
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "old1"}, ts=100, seq=1)
    mock_db = MagicMock()
    mock_db.append_message.side_effect = [
        RuntimeError("still locked at boot"), RuntimeError("still locked at drain"), 1, 1, 1, 1]
    boot_db(mock_db)
    store = make_store(mock_db)

    assert _boot(store) == 0
    store.append_to_transcript("sess-1", {"role": "user", "content": "new-live"})
    assert _contents(mock_db) == ["old0", "old0"]
    assert len(list(flush_dir.glob("*.json"))) == 2

    store.append_to_transcript("sess-1", {"role": "user", "content": "new-live-2"})
    assert _contents(mock_db) == ["old0", "old0", "old0", "old1", "new-live", "new-live-2"]
    assert not list(flush_dir.glob("*.json"))
    assert not store._dirty_transcripts


def test_a_poisoned_spool_row_does_not_stop_later_live_rows(flush_dir, tmp_path, make_store, boot_db):
    """A spooled field sqlite cannot bind used to fail on every replay. With live writes waiting
    behind the spool, the session then stopped persisting for good, across restarts too."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("sess-1", source="telegram")
        _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                     {"role": "tool", "content": "result", "tool_call_id": {"bad": 1}}, ts=100, seq=0)
        _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                     {"role": "user", "content": "older"}, ts=100, seq=1)
        # sqlite cannot bind an int past 64 bits, even into a TEXT column.
        _write_spool(flush_dir, "pending-ccc.json", "sess-1",
                     {"role": "tool", "content": "big id", "tool_call_id": 2**100}, ts=100, seq=2)
        boot_db(db)
        store = make_store(db)

        assert _boot(store) == 3
        store.append_to_transcript("sess-1", {"role": "user", "content": "live"})

        assert [m["content"] for m in db.get_messages("sess-1")] == ["result", "older", "big id", "live"]
        assert not list(flush_dir.iterdir())
        assert not store._dirty_transcripts
    finally:
        db.close()


def test_a_row_the_database_rejects_is_quarantined_not_retried(flush_dir, make_store, boot_db):
    """No retry can write a row the database refuses on its own values, so at boot and in the
    live drain it is moved aside instead of holding back the session's later rows."""
    def append_message(**kwargs):
        if kwargs["content"].startswith("poison"):
            raise sqlite3.InterfaceError("Error binding parameter 3: unsupported type")
        return 1

    mock_db = MagicMock()
    mock_db.append_message.side_effect = append_message
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "poison at boot"}, ts=100, seq=0)
    _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                 {"role": "user", "content": "after"}, ts=100, seq=1)
    boot_db(mock_db)
    store = make_store(mock_db)
    assert _boot(store) == 1

    _write_spool(flush_dir, "pending-ccc.json", "sess-1",
                 {"role": "user", "content": "poison live"}, ts=200, seq=2)
    _write_spool(flush_dir, "pending-ddd.json", "sess-1",
                 {"role": "user", "content": "after live"}, ts=200, seq=3)
    store.mark_spooled_drop_sessions({"sess-1"})
    store.append_to_transcript("sess-1", {"role": "user", "content": "live"})

    assert _contents(mock_db) == ["poison at boot", "after", "poison live", "after live", "live"]
    assert sorted(p.name for p in flush_dir.iterdir()) == [
        f"pending-aaa.json{QUARANTINE_SUFFIX}", f"pending-ccc.json{QUARANTINE_SUFFIX}"]
    assert not store._dirty_transcripts


def test_only_errors_about_the_rows_own_values_are_rejections():
    """A rejection is quarantined for good, so it must be an error no retry can clear."""
    assert is_row_rejection(sqlite3.ProgrammingError("Error binding parameter 4: type 'dict' is not supported"))
    assert is_row_rejection(sqlite3.InterfaceError("Error binding parameter 3 - probably unsupported type"))
    assert is_row_rejection(OverflowError("Python int too large to convert to SQLite INTEGER"))
    assert not is_row_rejection(sqlite3.InterfaceError("no more rows available"))
    assert not is_row_rejection(sqlite3.OperationalError("database is locked"))
    assert not is_row_rejection(sqlite3.IntegrityError("FOREIGN KEY constraint failed"))
    assert not is_row_rejection(sqlite3.DatabaseError("database disk image is malformed"))


def test_transient_no_more_rows_error_keeps_the_row_for_retry(flush_dir, make_store, boot_db):
    """SessionDB retries 'no more rows available' as WAL contention and re-raises it unchanged once
    its patience runs out. The row itself is valid, so at boot and in the live drain it must stay on
    disk and be written once the contention clears, not be quarantined."""
    transient = sqlite3.InterfaceError("no more rows available")
    path = _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                        {"role": "user", "content": "old"}, ts=100, seq=0)
    mock_db = MagicMock()
    mock_db.append_message.side_effect = [transient, transient, 1, 1, 1]
    boot_db(mock_db)
    store = make_store(mock_db)

    assert _boot(store) == 0
    assert path.exists()
    store.append_to_transcript("sess-1", {"role": "user", "content": "live"})
    assert _contents(mock_db) == ["old", "old"]
    assert path.exists()

    store.append_to_transcript("sess-1", {"role": "user", "content": "live-2"})
    assert _contents(mock_db) == ["old", "old", "old", "live", "live-2"]
    assert not list(flush_dir.iterdir())
    assert not store._dirty_transcripts


def test_a_replayed_file_that_cannot_be_deleted_is_not_replayed_again(flush_dir, make_store, monkeypatch):
    """Its row is already written, so replaying it before the next live write duplicates that row.
    The files after it are still older than the live row, so they must land first."""
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "old0"}, ts=100, seq=0)
    second = _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                          {"role": "user", "content": "old1"}, ts=100, seq=1)
    _fail_unlink_for(monkeypatch, "pending-aaa.json")
    mock_db = MagicMock()
    store = make_store(mock_db)
    store.mark_spooled_drop_sessions({"sess-1"})

    store.append_to_transcript("sess-1", {"role": "user", "content": "live"})
    store.mark_spooled_drop_sessions({"sess-1"})
    store.append_to_transcript("sess-1", {"role": "user", "content": "live-2"})

    assert _contents(mock_db) == ["old0", "old1", "live", "live-2"]
    assert not second.exists()
    assert not store._dirty_transcripts


def test_boot_recovery_counts_and_never_repeats_a_file_it_could_not_delete(
        flush_dir, make_store, boot_db, monkeypatch):
    """Boot recovery wrote the row, so it is recovered; the live drain must not write it again."""
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "old0"}, ts=100, seq=0)
    _write_spool(flush_dir, "pending-bbb.json", "sess-1",
                 {"role": "user", "content": "old1"}, ts=100, seq=1)
    _fail_unlink_for(monkeypatch, "pending-aaa.json")
    mock_db = MagicMock()
    boot_db(mock_db)
    store = make_store(mock_db)

    assert _boot(store) == 2
    store.mark_spooled_drop_sessions({"sess-1"})
    store.append_to_transcript("sess-1", {"role": "user", "content": "live"})

    assert _contents(mock_db) == ["old0", "old1", "live"]


def test_repairable_spool_replay_failure_still_reaches_fts_rebuild(flush_dir, make_store):
    """A spool replay failure must keep its own type, so the FTS rebuild still runs. After the
    repair the older spooled row is written before the live one."""
    _write_spool(flush_dir, "pending-aaa.json", "sess-1",
                 {"role": "user", "content": "old"}, ts=100, seq=0)

    class RepairableDb:
        def __init__(self):
            self.broken, self.rows, self.repairs = True, [], 0

        def append_message(self, **kwargs):
            if self.broken:
                raise RuntimeError("no such table: messages_fts")
            self.rows.append(kwargs["content"])

        def rebuild_fts(self):
            self.repairs += 1
            self.broken = False
            return 1

    db = RepairableDb()
    store = make_store(db)
    store.mark_spooled_drop_sessions({"sess-1"})

    store.append_to_transcript("sess-1", {"role": "user", "content": "live"})

    assert db.repairs == 1
    assert db.rows == ["old", "live"]
    assert not list(flush_dir.glob("*.json"))
    assert not store._dirty_transcripts


def test_boot_recovery_runs_before_resume_turns_and_queued_inbound(monkeypatch):
    """Resume turns and queued inbound write live rows. The store knows nothing about the previous
    run's spool until recovery has run, so a live row written first lands ahead of it for good."""
    import asyncio

    import gateway.run as gateway_run

    order = []
    monkeypatch.setattr("gateway.shutdown_flush.recover_gateway_pending",
                        lambda runner: order.append("recover") or 0)
    monkeypatch.setattr(gateway_run, "_restart_notification_pending", lambda: False)
    monkeypatch.setattr(gateway_run, "_planned_restart_notification_pending", lambda: False)

    async def noop(*_args, **_kwargs):
        return None

    async def finish_startup_restore():
        order.append("drain inbound")

    runner = SimpleNamespace(
        _start_post_connect_services=noop, _await_startup_boot_sends=noop,
        _schedule_resume_pending_sessions=lambda: order.append("resume"),
        _finish_startup_restore=finish_startup_restore,
        _send_session_db_warning_notifications=noop, _spawn_supervised=lambda *a, **k: None,
    )
    asyncio.run(gateway_run.GatewayRunner._start_finish_wiring(runner, 0))

    assert order == ["recover", "resume", "drain inbound"]


def test_shutdown_spool_replays_in_acceptance_order_not_filename_order(
    flush_dir, monkeypatch,
):
    """Adapter teardown spools the older pending head first and the newer debounced
    buffer second, both inside the same one-second ``ts`` (#126167 review F2).
    Recovery must replay them in the order they were accepted, not by their random
    ``pending-<uuid>.json`` filenames: the shared per-process spool ``seq`` breaks
    the same-second tie.
    """
    from gateway.shutdown_flush import flush_pending_to_file

    wrote_pending = flush_pending_to_file(
        {"sess-1": SimpleNamespace(text="older-pending")}, reason="adapter_shutdown")
    wrote_debounced = flush_pending_to_file(
        {"sess-1": SimpleNamespace(text="newer-debounced")}, reason="adapter_shutdown")
    assert wrote_pending == 1 and wrote_debounced == 1

    # Recovery replays in _order_flush_files order (recover_pending_spool consumes that list
    # unchanged), so the ordered glob IS the replay order for a session.
    ordered = _order_flush_files(flush_dir.glob("*.json"))
    texts = [json.loads(p.read_text(encoding="utf-8-sig"))["data"]["text"] for p in ordered]
    assert texts == ["older-pending", "newer-debounced"]
