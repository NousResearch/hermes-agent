"""Cron fire creation commits its session off the owner loop: a held SQLite writer must not stall
every session, Stop and socket the owner serves while ``cron.submit`` creates the fire's session."""
import asyncio
import sqlite3
import threading
import time
from types import SimpleNamespace

from cron import jobs
from gateway import run, session_cron
from gateway.config import GatewayConfig
from gateway.session import SessionStore
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch, list_session_admissions


def test_cron_submit_creation_commit_waits_off_the_owner_loop(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local', model='fixture')
        db = SessionDB(tmp_path / 'state.db')
        runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                                 adapters={})
        authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                     epoch=begin_runtime_epoch(db, instance_id='test'))
        monkeypatch.setattr(authority, '_schedule', lambda ref: None)
        runner.session_authority = authority

        async def probe():
            # Warm the creation path (cold imports are not the write under test).
            await session_cron.operation(authority, 'submit', {'job_id': job['id'], 'request_id': 'warm',
                                                               'extra_prompt': None})
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
            submit = asyncio.create_task(session_cron.operation(
                authority, 'submit', {'job_id': job['id'], 'request_id': 'fire', 'extra_prompt': None}))
            try:
                await asyncio.sleep(1.0)  # health: allow HX013 -- the writer hold IS the stall window the ticker measures
                release.set()
                receipt = await asyncio.wait_for(submit, 30)
            finally:
                release.set()
                tick.cancel()
            assert len(list_session_admissions(db, session_id=receipt['session_id'], pending_only=False)) == 1
            assert max(gaps) < 0.5, max(gaps)
        try:
            asyncio.run(probe())
        finally:
            db.close()
