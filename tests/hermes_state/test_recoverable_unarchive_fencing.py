"""Recoverable root lookup must fence the compression tip it will reopen."""

from contextlib import contextmanager

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError


@contextmanager
def _stores(tmp_path, monkeypatch):
    root = tmp_path / 'hermes'
    profile = root / 'profiles' / 'science'
    profile.mkdir(parents=True)
    monkeypatch.setenv('HERMES_HOME', str(root))
    local, sibling = SessionDB(root / 'state.db'), SessionDB(profile / 'state.db')
    try:
        local.create_session('root', 'cli', session_key='agent:main:local:dm:root')
        local.create_session('tip', 'cli', parent_session_id='root', session_key='agent:main:local:dm:tip')
        local._conn.execute("UPDATE sessions SET ended_at=10, end_reason='compression', archived=1 WHERE id='root'")
        local._conn.execute("UPDATE sessions SET ended_at=20, end_reason='ws_orphan_reap', archived=1 WHERE id='tip'")
        local._conn.commit()
        yield local, sibling
    finally:
        local.close()
        sibling.close()


def _rows(db):
    return tuple(tuple(row) for row in db._conn.execute('SELECT * FROM sessions ORDER BY id'))


def test_recoverable_root_refuses_foreign_tip_without_mutating_either_store(tmp_path, monkeypatch):
    with _stores(tmp_path, monkeypatch) as (local, sibling):
        # Model an existing dual-home database; fenced writers cannot create this fixture.
        sibling._conn.execute(
            "INSERT INTO sessions (id,source,started_at,profile_name,session_key,message_count) "
            "VALUES ('tip','cli',30,'science','agent:science:local:dm:tip',0)")
        sibling._conn.commit()
        before = _rows(local), _rows(sibling)
        with pytest.raises(SessionIdOwnedElsewhereError):
            local.unarchive_recoverable_session('root')
        assert (_rows(local), _rows(sibling)) == before


def test_recoverable_tip_reopens_after_old_locks_are_released(tmp_path, monkeypatch):
    import hermes_state_sessions as sessions

    original = sessions.session_lineage_guard
    active, acquisitions = set(), []

    @contextmanager
    def observed(path, ids, **kwargs):
        held = {ids} if isinstance(ids, str) else set(ids)
        assert not active, 'an expanded lock set must begin after the prior guard exits'
        acquisitions.append(held)
        active.update(held)
        try:
            with original(path, ids, **kwargs) as snapshot:
                yield snapshot
        finally:
            active.clear()

    with _stores(tmp_path, monkeypatch) as (local, sibling):
        before_sibling = _rows(sibling)
        monkeypatch.setattr(sessions, 'session_lineage_guard', observed)
        assert local.unarchive_recoverable_session('root') is True
        assert not local.get_session('root')['archived']
        tip = local.get_session('tip')
        assert not tip['archived'] and tip['ended_at'] is None and tip['end_reason'] is None
        assert local.get_session('root')['end_reason'] == 'compression'
        assert {'root', 'tip'} in acquisitions and not active
        assert _rows(sibling) == before_sibling
        local.append_message('tip', 'user', 'recovered owner remains writable')
