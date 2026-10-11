"""The auto-created Bot Chat target can be replaced: rename and delete spend its creation identity.

``gateway.session_bot._create_bot_chat`` mints the profile's Bot Chat when none is titled so. The
creation identity of an earlier target is immutable (a retry of it must converge, and a deleted one
must never be resurrected), so a rename or canonical delete must lead to a FRESH creation instead of
replaying the old one: retitling the renamed session back, or refusing on the retired id.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import Principal


@pytest_asyncio.fixture
async def bot(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from hermes_state import SessionDB

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    store._db = SessionDB(db_path=tmp_path / 'state.db')
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = Principal('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    try:
        yield SimpleNamespace(authority=authority, owner=owner, home=tmp_path)
    finally:
        store._db.close()


def _target_id(bot):
    from gateway.session_bot import _target
    ref, _live, _entry = _target(bot.authority, bot.owner)
    return ref.session_id


@pytest.mark.asyncio
async def test_renamed_or_deleted_bot_chat_gets_a_fresh_replacement(bot):
    db = bot.authority.db
    first = _target_id(bot)
    assert _target_id(bot) == first  # converges: resolved by title, never re-created

    db.set_session_title(first, 'Planning notes')
    second = _target_id(bot)
    assert second != first
    assert db.get_session(first)['title'] == 'Planning notes'  # the rename stands
    assert db.get_session(second)['title'] == 'Bot Chat'

    assert db.delete_session(second, sessions_dir=bot.home / 'sessions')
    bot.authority.sessions.pop(second, None)
    third = _target_id(bot)
    assert third not in (first, second)  # a replacement, not the retired id
    assert db.get_session(second) is None
    assert db.get_session(third)['title'] == 'Bot Chat'
    assert _target_id(bot) == third
