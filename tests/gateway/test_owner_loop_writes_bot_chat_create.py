"""The Bot Chat a first delivery mints commits off the owner loop (N90): a held SQLite writer must
not stall every session, Stop and socket the owner serves while the profile's Bot Chat is created;
concurrent first deliveries and a titled ``session.create`` still converge on one Bot Chat."""
import asyncio
import sqlite3
import threading
import time

import pytest

from tests.gateway import test_session_bot_target_lifecycle as lifecycle

bot = lifecycle.bot  # the shared Bot Chat owner fixture


async def _resolve(bot):
    from gateway.session_bot import _target
    return (await _target(bot.authority, bot.owner))[0].session_id


@pytest.mark.asyncio
async def test_bot_chat_creation_commits_off_the_owner_loop(bot):
    db = bot.authority.db
    # A serving owner discovered plugins at startup (_start_register_plugins_relay_hooks); this
    # bare fixture has not, so the first policy build would load every plugin ON the loop and
    # the ticker would time plugin import instead of the held writer.
    from hermes_cli.plugins import discover_plugins
    await asyncio.to_thread(discover_plugins)
    held, release = threading.Event(), threading.Event()

    def hold_writer():  # another process's writer
        other = sqlite3.connect(db.db_path, timeout=10)
        other.execute('BEGIN IMMEDIATE')
        held.set()
        release.wait(5)
        other.rollback()
        other.close()
    threading.Thread(target=hold_writer, daemon=True).start()
    assert await asyncio.to_thread(held.wait, 5)
    gaps = []

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)  # health: allow HX013 -- the ticker's period is the owner-loop stall measurement
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    tick = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    first = asyncio.ensure_future(_resolve(bot))
    second = asyncio.ensure_future(_resolve(bot))
    try:
        await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
        release.set()
        ids = await asyncio.wait_for(asyncio.gather(first, second), 30)
    finally:
        release.set()
        tick.cancel()
    assert max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind a held writer'
    assert ids[0] == ids[1] and db.get_session(ids[0])['title'] == 'Bot Chat'
    with db._read_ctx() as c:
        assert c.execute("SELECT COUNT(*) FROM sessions WHERE title='Bot Chat'").fetchone()[0] == 1
