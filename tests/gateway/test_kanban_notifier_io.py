"""I/O and lifecycle coverage for the Kanban notifier poll guard."""

import asyncio
import os
import sqlite3
import threading

from gateway import kanban_watchers
from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    return runner


def _seed_subscription(db_path, *, title="notify", notifier_profile=None):
    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    try:
        task_id = kb.create_task(conn, title=title, assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile=notifier_profile,
        )
        return task_id
    finally:
        conn.close()


def _seed_completed_subscription(db_path, *, title="completed"):
    task_id = _seed_subscription(db_path, title=title)
    conn = kb.connect(db_path=db_path)
    try:
        kb.complete_task(conn, task_id, summary="replacement completion")
    finally:
        conn.close()
    return task_id


def _count_sqlite_opens(monkeypatch):
    opens = []
    real_connect = kb.sqlite3.connect

    def counted_connect(*args, **kwargs):
        opens.append((args, kwargs))
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(kb.sqlite3, "connect", counted_connect)
    return opens


def _run_polls(monkeypatch, runner, poll_count, on_poll_sleep=None):
    """Run exactly ``poll_count`` notifier intervals without wall-clock waits."""
    real_sleep = asyncio.sleep
    poll_sleeps = 0

    async def fake_sleep(delay):
        nonlocal poll_sleeps
        if delay == 5:
            return None
        poll_sleeps += 1
        if on_poll_sleep is not None:
            on_poll_sleep(poll_sleeps)
        if poll_sleeps >= poll_count:
            runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))
    return poll_sleeps


def test_notifier_reopens_changed_wal_and_checkpointed_board_next_poll(
    tmp_path, monkeypatch,
):
    """A committed change is seen on the next poll, including after TRUNCATE."""
    db_path = tmp_path / "changed.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    task_id = _seed_subscription(db_path, title="changed board")
    writer = kb.connect(db_path=db_path)
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)
    signatures = []
    journal_modes = []

    def change_after_first_poll(poll_sleeps):
        if poll_sleeps != 1:
            return
        before_commit = kanban_watchers._kanban_db_file_signature(db_path)
        journal_mode = str(writer.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        kb.complete_task(writer, task_id, summary="committed in WAL")
        after_commit = kanban_watchers._kanban_db_file_signature(db_path)
        checkpoint = writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        after_checkpoint = kanban_watchers._kanban_db_file_signature(db_path)
        journal_modes.append(journal_mode)
        signatures.append((before_commit, after_commit, after_checkpoint, checkpoint))

    try:
        poll_sleeps = _run_polls(
            monkeypatch, runner, 2, on_poll_sleep=change_after_first_poll,
        )
    finally:
        writer.close()

    assert poll_sleeps == 2
    assert len(adapter.sent) == 1
    assert task_id in adapter.sent[0]["text"]
    assert len(signatures) == 1
    before_commit, after_commit, after_checkpoint, checkpoint = signatures[0]
    assert checkpoint is not None
    assert after_commit != before_commit, "the committed board change must invalidate the cache"
    assert after_checkpoint != after_commit or journal_modes[0] != "wal"
    if journal_modes[0] == "wal":
        assert (
            after_commit[1] != before_commit[1]
            or after_commit[2] != before_commit[2]
        ), "a WAL commit must change a tracked SQLite sidecar"
    # Two opens for the initial probe/load, two for the changed poll, and one
    # cursor-advance connection after the event is delivered.
    assert len(opens) == 5


def test_notifier_invalidates_cache_for_subscription_remove_and_add(
    tmp_path, monkeypatch,
):
    """Subscription lifecycle changes invalidate both zero- and non-zero states."""
    db_path = tmp_path / "subscription-lifecycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    first_task = _seed_subscription(db_path, title="removed subscription")
    second_task = _seed_subscription(db_path, title="added subscription")
    # _seed_subscription intentionally adds a sub for the second task; remove
    # it before the watcher starts so the callback can test an actual add.
    conn = kb.connect(db_path=db_path)
    try:
        assert kb.remove_notify_sub(
            conn,
            task_id=second_task,
            platform="telegram",
            chat_id="chat-1",
        )
    finally:
        conn.close()

    writer = kb.connect(db_path=db_path)
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)

    def change_subscription_after_poll(poll_sleeps):
        if poll_sleeps == 1:
            assert kb.remove_notify_sub(
                writer,
                task_id=first_task,
                platform="telegram",
                chat_id="chat-1",
            )
        elif poll_sleeps == 2:
            kb.add_notify_sub(
                writer,
                task_id=second_task,
                platform="telegram",
                chat_id="chat-1",
            )
            kb.complete_task(writer, second_task, summary="subscription added")

    try:
        poll_sleeps = _run_polls(
            monkeypatch, runner, 3, on_poll_sleep=change_subscription_after_poll,
        )
    finally:
        writer.close()

    assert poll_sleeps == 3
    assert len(adapter.sent) == 1
    assert second_task in adapter.sent[0]["text"]
    # Initial query/load (2), removal leaves a zero-sub read-only probe (1),
    # and the re-added event needs query/load plus cursor advance (3).
    assert len(opens) == 6


def test_notifier_reopens_replaced_database_and_delivers_new_event(
    tmp_path, monkeypatch,
):
    """Replacing a cached DB cannot leave the watcher on stale board state."""
    db_path = tmp_path / "recreated.db"
    replacement_path = tmp_path / "recreated.db.new"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    old_task = _seed_subscription(db_path, title="old database")
    replacement_task = _seed_completed_subscription(
        replacement_path, title="replacement database",
    )
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)

    def replace_after_first_poll(poll_sleeps):
        if poll_sleeps != 1:
            return
        for suffix in ("-wal", "-shm"):
            (tmp_path / f"recreated.db{suffix}").unlink(missing_ok=True)
        os.replace(replacement_path, db_path)

    poll_sleeps = _run_polls(
        monkeypatch, runner, 2, on_poll_sleep=replace_after_first_poll,
    )

    assert poll_sleeps == 2
    assert old_task != replacement_task
    assert len(adapter.sent) == 1
    assert replacement_task in adapter.sent[0]["text"]
    # The replacement is detected by device/inode/size/timestamp metadata and
    # is queried on the immediately following poll.
    assert len(opens) == 5


def test_notifier_fails_open_when_file_signature_is_uncertain(tmp_path, monkeypatch):
    """A stat failure must query each tick rather than suppress notifications."""
    db_path = tmp_path / "stat-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    _seed_subscription(db_path, title="uncertain signature")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)

    def unavailable_signature(_db_path):
        raise OSError("simulated stat failure")

    monkeypatch.setattr(
        kanban_watchers,
        "_kanban_db_file_signature",
        unavailable_signature,
    )
    poll_sleeps = _run_polls(monkeypatch, runner, 3)

    assert poll_sleeps == 3
    assert adapter.sent == []
    # Each uncertain tick falls back to the read-only probe plus writable load.
    assert len(opens) == 6


def test_notifier_does_not_cache_through_concurrent_writer(tmp_path, monkeypatch):
    """A writer racing the query invalidates the guard without a lock error."""
    db_path = tmp_path / "concurrent-writer.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    task_id = _seed_subscription(db_path, title="concurrent board")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)
    real_list_notify_subs = kb.list_notify_subs
    writer_started = False
    writer_errors = []

    def list_with_concurrent_writer(conn, *args, **kwargs):
        nonlocal writer_started
        rows = real_list_notify_subs(conn, *args, **kwargs)
        if writer_started:
            return rows
        writer_started = True

        def write_event():
            try:
                writer = kb.connect(db_path=db_path)
                try:
                    kb.complete_task(writer, task_id, summary="concurrent commit")
                finally:
                    writer.close()
            except BaseException as exc:  # report writer failures in the assertion below
                writer_errors.append(exc)

        thread = threading.Thread(target=write_event)
        thread.start()
        thread.join()
        return rows

    monkeypatch.setattr(kb, "list_notify_subs", list_with_concurrent_writer)
    poll_sleeps = _run_polls(monkeypatch, runner, 2)

    assert poll_sleeps == 2
    assert writer_errors == []
    assert len(adapter.sent) == 1
    assert task_id in adapter.sent[0]["text"]
    # Initial probe/load (2), concurrent writer (1), next-poll probe/load and
    # cursor advance (3). The first query did not incorrectly cache its result.
    assert len(opens) == 6


def test_notifier_closes_poll_connections_on_shutdown(tmp_path, monkeypatch):
    """Normal watcher shutdown closes every connection opened by its final poll."""
    db_path = tmp_path / "shutdown.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    _seed_subscription(db_path, title="shutdown")
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    connections = []
    real_connect = kb.sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(kb.sqlite3, "connect", tracked_connect)
    poll_sleeps = _run_polls(monkeypatch, runner, 1)

    assert poll_sleeps == 1
    assert connections
    for connection in connections:
        try:
            connection.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            pass
        else:
            raise AssertionError("notifier left a SQLite connection open on shutdown")


def test_notifier_polls_each_unchanged_board_once_with_profile_context(
    tmp_path, monkeypatch,
):
    """Quiet boards are cached independently across profiles and board paths."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    kb.init_db(board="default")
    kb.create_board("secondary")
    for board, profile in (("default", "main"), ("secondary", "other")):
        conn = kb.connect(board=board)
        try:
            task_id = kb.create_task(conn, title=f"{board} quiet", assignee="worker")
            kb.add_notify_sub(
                conn,
                task_id=task_id,
                platform="telegram",
                chat_id=f"{board}-chat",
                notifier_profile=profile,
            )
        finally:
            conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"
    runner._profile_adapters = {"other": {Platform.TELEGRAM: adapter}}
    opens = _count_sqlite_opens(monkeypatch)
    poll_sleeps = _run_polls(monkeypatch, runner, 3)

    assert poll_sleeps == 3
    assert adapter.sent == []
    # Each of the two board paths gets one read-only probe and one writable
    # load; both paths are then skipped on the two unchanged ticks.
    assert len(opens) == 4


def test_notifier_drops_removed_board_from_subsequent_polls(tmp_path, monkeypatch):
    """Removing a board must not leave a cached path queried on later ticks."""
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban-home"))
    kb.init_db(board="default")
    kb.create_board("temporary")
    db_path = kb.kanban_db_path("temporary")
    _seed_subscription(db_path, title="removed board")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    opens = _count_sqlite_opens(monkeypatch)

    def remove_after_first_poll(poll_sleeps):
        if poll_sleeps == 1:
            result = kb.remove_board("temporary", archive=False)
            assert result["action"] == "deleted"

    poll_sleeps = _run_polls(
        monkeypatch, runner, 2, on_poll_sleep=remove_after_first_poll,
    )

    assert poll_sleeps == 2
    assert adapter.sent == []
    # A read-only probe can create the default board's SHM sidecar on the
    # first tick, so that board may be probed again. The removed board itself
    # must have only its initial probe/load opens and never be reopened.
    temporary_opens = [
        (args, kwargs)
        for args, kwargs in opens
        if "temporary" in str(args[0])
    ]
    assert len(temporary_opens) == 2


def test_db_file_signature_tracks_sidecar_lifecycle_and_replacement(tmp_path):
    """The signature is based on documented file metadata for all three files."""
    db_path = tmp_path / "signature.db"
    db_path.write_bytes(b"db")
    initial = kanban_watchers._kanban_db_file_signature(db_path)

    wal_path = tmp_path / "signature.db-wal"
    wal_path.write_bytes(b"wal")
    with_wal = kanban_watchers._kanban_db_file_signature(db_path)
    assert with_wal != initial

    wal_path.write_bytes(b"w")
    truncated_wal = kanban_watchers._kanban_db_file_signature(db_path)
    assert truncated_wal != with_wal

    shm_path = tmp_path / "signature.db-shm"
    shm_path.write_bytes(b"shm")
    with_shm = kanban_watchers._kanban_db_file_signature(db_path)
    assert with_shm != truncated_wal
    shm_path.unlink()
    without_shm = kanban_watchers._kanban_db_file_signature(db_path)
    assert without_shm != with_shm

    wal_path.unlink()
    before_replacement = kanban_watchers._kanban_db_file_signature(db_path)
    replacement_path = tmp_path / "signature.db.replacement"
    replacement_path.write_bytes(b"replacement database")
    os.replace(replacement_path, db_path)
    replaced = kanban_watchers._kanban_db_file_signature(db_path)
    assert replaced != before_replacement
    assert replaced[1] == ("missing",)
    assert replaced[2] == ("missing",)
