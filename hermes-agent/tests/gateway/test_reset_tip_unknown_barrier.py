"""A local conversation reset before a crash keeps its admissions on the creation id while the
store routes the physical reset child. The unknown-turn barrier must resolve the child to that
owner: neither the crash marker release nor legacy auto-resume may treat the child as admission-
free and silently re-run a turn whose canonical admission is ``unknown``."""
import asyncio
from contextlib import closing
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
import hermes_state_runtime as rt
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source
from tests.gateway.test_unknown_resolution_physical_worker import _local_reset


def _unknown_after_reset(tmp_path, runner):
    db = SessionDB(tmp_path / 'state.db')
    epoch = rt.begin_runtime_epoch(db, instance_id='crashed')
    logical = _local_reset(db, epoch, 'logical', 'physical')
    rt.admit_session_input(db, epoch=epoch, principal_id='human', session_id=logical,
                           request_id='turn', payload={'text': 'turn'})
    rt.claim_session_input(db, epoch=epoch, session_id=logical)
    epoch = rt.begin_runtime_epoch(db, instance_id='restarted')
    rt.recover_session_inputs(db, epoch=epoch)  # the crashed turn is now 'unknown'
    return db, SessionAuthority(runner, profile_id='profile', instance_id='restarted', db=db, epoch=epoch)


def _entry(**extra):
    return SessionEntry(session_key='agent:main:telegram:dm:c', session_id='physical', created_at=datetime.now(),
                        updated_at=datetime.now(), origin=make_restart_source(chat_id='c'),
                        platform=Platform.TELEGRAM, chat_type='dm', **extra)


def test_crash_marker_on_a_reset_tip_is_released_for_its_unknown_admission(tmp_path):
    from gateway.run_runtime import release_unknown_turn_markers
    entry = _entry(active_turn_token='tok')
    cleared = []
    store = SimpleNamespace(list_sessions=lambda: [entry],
                            clear_turn_active=lambda key, token: cleared.append((key, token)))
    runner = SimpleNamespace(session_store=store, config=SimpleNamespace(multiplex_profiles=False))
    db, authority = _unknown_after_reset(tmp_path, runner)
    with closing(db):
        runner.session_authorities = [authority]
        release_unknown_turn_markers(runner)
    assert cleared == [(entry.session_key, 'tok')]


@pytest.mark.asyncio
async def test_legacy_resume_skips_a_reset_tip_whose_admission_is_unknown(tmp_path):
    runner, adapter = make_restart_runner()
    db, authority = _unknown_after_reset(tmp_path, runner)
    with closing(db):
        runner.session_authorities = [authority]
        entry = _entry(resume_pending=True, resume_reason='restart_interrupted', last_resume_marked_at=datetime.now())
        runner.session_store._entries = {entry.session_key: entry}
        adapter.handle_message = AsyncMock()
        assert runner._schedule_resume_pending_sessions() == 0
        await asyncio.sleep(0)
        adapter.handle_message.assert_not_called()
