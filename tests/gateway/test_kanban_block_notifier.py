"""Tests for the gateway's global block-notifier watcher."""

import asyncio
from typing import Callable, Optional

import pytest

import hermes_cli.config as cli_config
from gateway.config import HomeChannel, Platform
from hermes_cli import kanban_db as kb
from tests.gateway._kanban_helpers import (
    FailingAdapter,
    RecordingAdapter,
    make_runner as _make_runner,
)


def _patch_config(monkeypatch, *, notify_on_block: bool, channel: str = ""):
    def _fake():
        return {
            "kanban": {
                "notify_on_block": notify_on_block,
                "notify_on_block_channel": channel,
            }
        }

    monkeypatch.setattr(cli_config, "load_config", _fake)


async def _drive_ticks(
    monkeypatch,
    runner,
    *,
    ticks: int,
    interval: int = 1,
    after_tick: Optional[dict[int, Callable[[], None]]] = None,
):
    """Run ``_kanban_block_notifier_watcher`` for ``ticks`` ticks, then stop.

    ``after_tick[N]`` is called immediately after tick N completes (between
    tick N and tick N+1), letting tests mutate the DB between snapshots
    without writing per-test sleep closures.
    """
    real_sleep = asyncio.sleep
    counter = {"n": 0}
    after_tick = after_tick or {}

    async def fake_sleep(delay):
        if delay == 5:
            return None
        counter["n"] += 1
        cb = after_tick.get(counter["n"])
        if cb is not None:
            cb()
        if counter["n"] >= ticks:
            runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_block_notifier_watcher(interval=interval)


def _block_a_new_task(*, title="will-be-blocked", reason="needs love") -> str:
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title=title, assignee="worker")
        kb.block_task(conn, tid, reason=reason)
        return tid
    finally:
        conn.close()


@pytest.fixture
def board_db(tmp_path, monkeypatch):
    db_path = tmp_path / "block-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    return db_path


def test_disabled_by_default_no_delivery(board_db, monkeypatch):
    _patch_config(monkeypatch, notify_on_block=False)
    _block_a_new_task()

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})

    asyncio.run(_drive_ticks(monkeypatch, runner, ticks=1))

    assert adapter.sent == []


def test_first_tick_seeds_without_firing(board_db, monkeypatch):
    """No historical replay: tasks already blocked at boot must seed silently."""
    _patch_config(monkeypatch, notify_on_block=True)
    _block_a_new_task()

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})

    asyncio.run(_drive_ticks(monkeypatch, runner, ticks=1))

    assert adapter.sent == []


def test_block_after_seed_fires(board_db, monkeypatch):
    _patch_config(monkeypatch, notify_on_block=True)

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})

    blocked_tid: Optional[str] = None

    def block_between():
        nonlocal blocked_tid
        blocked_tid = _block_a_new_task(title="post-seed", reason="thinking")

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=2, after_tick={1: block_between},
    ))

    assert len(adapter.sent) == 1
    msg = adapter.sent[0]["text"]
    assert blocked_tid in msg
    assert "blocked" in msg.lower()
    assert "thinking" in msg


def test_explicit_channel_override_pins_platform(board_db, monkeypatch):
    _patch_config(monkeypatch, notify_on_block=True, channel="telegram")

    telegram = RecordingAdapter()
    discord = RecordingAdapter()
    runner = _make_runner(
        {Platform.TELEGRAM: telegram, Platform.DISCORD: discord},
        homes={
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM, chat_id="tg-home", name="TG",
            ),
            Platform.DISCORD: HomeChannel(
                platform=Platform.DISCORD, chat_id="dc-home", name="DC",
            ),
        },
    )

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=2,
        after_tick={1: lambda: _block_a_new_task(title="x", reason="y")},
    ))

    assert len(telegram.sent) == 1
    assert telegram.sent[0]["chat_id"] == "tg-home"
    assert discord.sent == []


def test_explicit_channel_offline_skips_delivery(board_db, monkeypatch):
    """Pinned channel offline → drop, don't fall back to a different platform."""
    _patch_config(monkeypatch, notify_on_block=True, channel="discord")

    telegram = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: telegram})

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=2,
        after_tick={1: lambda: _block_a_new_task()},
    ))

    assert telegram.sent == []


def test_default_broadcasts_to_every_home_channel(board_db, monkeypatch):
    """Empty `notify_on_block_channel` fans each transition out to every
    connected platform that has a home channel set; platforms without one
    are skipped silently.
    """
    _patch_config(monkeypatch, notify_on_block=True, channel="")

    telegram = RecordingAdapter()
    discord = RecordingAdapter()
    slack = RecordingAdapter()
    runner = _make_runner(
        {
            Platform.TELEGRAM: telegram,
            Platform.DISCORD: discord,
            Platform.SLACK: slack,
        },
        homes={
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM, chat_id="tg-home", name="TG",
            ),
            Platform.DISCORD: HomeChannel(
                platform=Platform.DISCORD, chat_id="dc-home", name="DC",
            ),
            Platform.SLACK: None,  # connected but no home configured → skipped
        },
    )

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=2,
        after_tick={1: lambda: _block_a_new_task()},
    ))

    assert len(telegram.sent) == 1
    assert telegram.sent[0]["chat_id"] == "tg-home"
    assert len(discord.sent) == 1
    assert discord.sent[0]["chat_id"] == "dc-home"
    assert slack.sent == []


def test_partial_broadcast_failure_retries_until_giveup(board_db, monkeypatch):
    """If one of N broadcast targets fails, the cursor rolls back so the
    next tick re-attempts the broadcast — up to MAX_SEND_FAILURES, after
    which the watcher gives up rather than pinning the cursor against a
    permanently-dead chat. Working targets receive duplicates during the
    retry window; that's the documented tradeoff for the simpler
    in-memory dedup model.
    """
    _patch_config(monkeypatch, notify_on_block=True, channel="")

    telegram = RecordingAdapter()
    discord = FailingAdapter()
    runner = _make_runner(
        {Platform.TELEGRAM: telegram, Platform.DISCORD: discord},
        homes={
            Platform.TELEGRAM: HomeChannel(
                platform=Platform.TELEGRAM, chat_id="tg-home", name="TG",
            ),
            Platform.DISCORD: HomeChannel(
                platform=Platform.DISCORD, chat_id="dc-home", name="DC",
            ),
        },
    )

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=5,
        after_tick={1: lambda: _block_a_new_task()},
    ))

    assert discord.attempts == 3, (
        f"FailingAdapter should be called exactly MAX_SEND_FAILURES (3) "
        f"times before give-up; got {discord.attempts}"
    )
    # Telegram receives one delivery per retry attempt; the simpler
    # in-memory dedup re-broadcasts on rollback. Three deliveries here
    # mirrors the three Discord retry attempts.
    assert len(telegram.sent) == 3


def test_reblock_after_unblock_fires_again(board_db, monkeypatch):
    """Each new block transition appends a new event id, so the diff fires fresh."""
    _patch_config(monkeypatch, notify_on_block=True)

    adapter = RecordingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})

    tid: Optional[str] = None

    def first_block():
        nonlocal tid
        tid = _block_a_new_task(title="reblock test", reason="r1")

    def reblock():
        conn = kb.connect()
        try:
            kb.unblock_task(conn, tid)
            kb.block_task(conn, tid, reason="r2")
        finally:
            conn.close()

    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=3,
        after_tick={1: first_block, 2: reblock},
    ))

    assert len(adapter.sent) == 2, (
        f"Re-block must fire a fresh notification; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "r1" in adapter.sent[0]["text"]
    assert "r2" in adapter.sent[1]["text"]


def test_send_failure_retries_then_gives_up(board_db, monkeypatch):
    """A persistently-failing adapter retries up to MAX_SEND_FAILURES, doesn't crash."""
    _patch_config(monkeypatch, notify_on_block=True)

    adapter = FailingAdapter()
    runner = _make_runner({Platform.TELEGRAM: adapter})

    # 1 seed tick + at least 3 retry ticks. Use 5 to leave headroom for the
    # final tick after MAX_SEND_FAILURES gives up (which should not retry
    # again, pinning the give-up behaviour).
    asyncio.run(_drive_ticks(
        monkeypatch, runner, ticks=5,
        after_tick={1: lambda: _block_a_new_task()},
    ))

    assert adapter.attempts == 3, (
        f"Watcher should retry exactly MAX_SEND_FAILURES (3) times before "
        f"giving up; got {adapter.attempts} attempts"
    )
