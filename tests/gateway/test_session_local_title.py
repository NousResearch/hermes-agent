"""Authority-side title selection: main's classic rule, without leaking foreign titles."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal
from gateway.session_local_title import resolve_titled_session
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


def _authority(db):
    epoch = begin_runtime_epoch(db, instance_id='current')
    return SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
                            instance_id='current', db=db, epoch=epoch)


def _local(db, authority, sid, *, user, title):
    db.create_session(sid, source='cli', chat_id=sid, user_id=user)
    db.set_session_title(sid, title)
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id=user, chat_type='dm')
    authority.sessions[sid] = LiveSession(source, 'route-' + sid)


def test_title_resolves_like_classic_cli_and_never_confirms_foreign_titles(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        authority = _authority(db)
        _local(db, authority, 'local-a', user='human', title='thread')
        _local(db, authority, 'local-a2', user='human', title='thread #2')
        _local(db, authority, 'local-b', user='stranger', title='private')
        # A non-local (messaging) session carrying a title is not resumable through the local client.
        db.create_session('tg-1', source='telegram', chat_id='chat', user_id='human')
        db.set_session_title('tg-1', 'telegram thread')
        human = Principal('human', 'owned', frozenset({'session:read'}), 'viewer')

        def resolve(title):
            return resolve_titled_session(authority, human, title).session_id
        # Latest "<title> #N" continuation wins over the bare title (classic lineage rule).
        assert resolve('thread') == 'local-a2'
        assert resolve('thread #2') == 'local-a2'
        # A literal id passed as title still resolves (main tries id before title).
        assert resolve('local-a') == 'local-a'
        for foreign in ('private', 'telegram thread', 'nope'):
            with pytest.raises(RuntimeStoreError, match='not_found'):
                resolve(foreign)
        assert resolve_titled_session(authority, human, 'nope', missing_ok=True) is None
        with pytest.raises(RuntimeStoreError, match='invalid_params'):
            resolve('')


def test_resolve_titled_session_skips_deliberately_archived_bot_chat(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        authority = _authority(db)
        _local(db, authority, 'local-retired', user='human', title='Bot Chat')
        assert db.set_session_hidden('local-retired', True)
        db._write_sql(
            'UPDATE sessions SET archived = 1, auto_archived = 0 WHERE id = ?', ('local-retired',))
        human = Principal('human', 'owned', frozenset({'session:read'}), 'viewer')
        assert resolve_titled_session(authority, human, 'Bot Chat', missing_ok=True) is None


def test_resolve_titled_session_finds_recoverable_auto_archived_bot_chat(tmp_path):
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        authority = _authority(db)
        _local(db, authority, 'local-reaped', user='human', title='Bot Chat')
        assert db.set_session_hidden('local-reaped', True)
        db.end_session('local-reaped', 'ws_orphan_reap')
        assert db.set_session_archived('local-reaped', True)
        human = Principal('human', 'owned', frozenset({'session:read'}), 'viewer')
        ref = resolve_titled_session(authority, human, 'Bot Chat', missing_ok=True)
        assert ref is not None
        assert ref.session_id == 'local-reaped'
