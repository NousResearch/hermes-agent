"""Tests for the kanban notifier zero-subscription early exit.

The notifier used to writable-open EVERY board DB on every tick even when a
board had zero subscriptions — paying schema init/migration on first open,
WAL/-shm sidecar creation, and checkpoint traffic for boards with nothing to
notify. Per-board work is now gated by a read-only subscription probe
(``kanban_db.count_notify_subs``), so boards with zero subscriptions are
never opened writable.

(The companion machine-global ``.notifier.lock`` singleton gate from PR
#63001 was deliberately NOT salvaged: a lock-winning default-profile gateway
cannot deliver a secondary profile's subscriptions in standalone-profile
deployments — profile routing fails closed in
``gateway/authz_mixin.py::_authorization_adapter`` — so the lock could
suppress delivery entirely. The read-only probe captures the per-tick cost
win without that risk.)
"""

import asyncio
import sqlite3

from unittest.mock import patch

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_notify as kbn


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
    return runner


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _create_completed_task(*, subscribe: bool) -> str:
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="owner gate", assignee="worker")
        if subscribe:
            kbn.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary="done")
        return tid
    finally:
        conn.close()


def test_zero_sub_board_is_never_opened_writable(tmp_path, monkeypatch):
    """A board with zero subscriptions must be skipped BEFORE `_kb.connect`."""
    db_path = tmp_path / "zero-subs.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=False)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    spy_connect.assert_not_called()
    assert adapter.sent == []


# ---------------------------------------------------------------------------
# #124389: a board with live work and zero subscribers warns instead of
# staying a DEBUG line nobody sees at INFO (#124389).
# ---------------------------------------------------------------------------


def test_count_live_tasks_excludes_finished_cards(tmp_path, monkeypatch):
    """``done``/``archived`` cards are quiet by design; every other status is
    live work that may still need a human."""
    db_path = tmp_path / "live.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kbc.connect()
    try:
        blocked = kb.create_task(conn, title="blocked", assignee="worker")
        assert kb.block_task(conn, blocked, reason="needs input")
        ready = kb.create_task(conn, title="ready", assignee="worker")
        done = kb.create_task(conn, title="done", assignee="worker")
        kb.complete_task(conn, done, summary="finished")
    finally:
        conn.close()

    # The board path resolves through the same env override the creator used.
    assert kbn.count_live_tasks(board="default") == 2
    # ...and the explicit path form agrees (no board resolution involved).
    assert kbn.count_live_tasks(db_path) == 2


def test_count_live_tasks_fails_open_on_missing_db(tmp_path, monkeypatch):
    db_path = tmp_path / "absent.db"
    assert kbn.count_live_tasks(db_path) == 0


def test_zero_sub_board_with_live_work_warns(tmp_path, monkeypatch, caplog):
    """Live cards + zero subscribers = the silent state from #124389: the
    notifier must surface it at WARNING with the subscribe command."""
    db_path = tmp_path / "live-no-subs.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="needs-input", assignee="worker")
        kb.block_task(conn, tid, reason="waiting on operator")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with caplog.at_level("WARNING"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("no subscriptions" in r.getMessage() for r in warnings), (
        "a board with blocked cards and no subscriber must warn, not stay DEBUG"
    )
    assert not adapter.sent  # zero subs still means nothing delivered


def test_zero_sub_board_with_only_finished_cards_stays_quiet(
    tmp_path, monkeypatch, caplog,
):
    """The narrower alternative to a delivery floor: finished boards do not
    nag — the warning is scoped to live work."""
    db_path = tmp_path / "finished.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    _create_completed_task(subscribe=False)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    with caplog.at_level("DEBUG"):
        asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert not [r for r in caplog.records if r.levelname == "WARNING"]
    assert not adapter.sent


def test_live_task_probe_failure_does_not_break_the_gate(tmp_path, monkeypatch, caplog):
    """The live-task probe is best effort: a failing probe must not change the
    zero-sub gate's outcome (board still skipped)."""
    db_path = tmp_path / "probe-fail.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    orig = kbn.count_live_tasks
    monkeypatch.setattr(
        kbn, "count_live_tasks",
        lambda *a, **k: (_ for _ in ()).throw(sqlite3.Error("boom")),
    )
    try:
        with patch.object(kbc, "connect", wraps=kbc.connect) as spy_connect:
            asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    finally:
        monkeypatch.setattr(kbn, "count_live_tasks", orig)

    spy_connect.assert_not_called()
    assert adapter.sent == []


def test_zero_sub_warning_is_once_per_board_and_live_count(monkeypatch, caplog):
    """The notifier ticks every few seconds: a steady board must not warn on
    every tick, while a changed backlog size re-announces."""
    from gateway import kanban_watchers_notifier as kwn

    kwn._ZERO_SUB_BOARD_WARNED.clear()
    try:
        with caplog.at_level("WARNING"):
            assert kwn._warn_zero_sub_board_once("alpha", 3, ["default"]) is True
            assert kwn._warn_zero_sub_board_once("alpha", 3, ["default"]) is True
            kwn._warn_zero_sub_board_once("alpha", 2, ["default"])
        lines = [r for r in caplog.records if "non-terminal task(s)" in r.getMessage()]
        # First announce (3) + the re-announced new size (2) — not a third.
        assert len(lines) == 2
        assert "3 non-terminal" in lines[0].getMessage()
        assert "2 non-terminal" in lines[1].getMessage()
    finally:
        kwn._ZERO_SUB_BOARD_WARNED.clear()


