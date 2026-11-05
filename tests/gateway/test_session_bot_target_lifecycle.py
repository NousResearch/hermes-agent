"""The auto-created Bot Chat target can be replaced: rename and delete spend its creation identity.

``gateway.session_bot._create_bot_chat`` mints the profile's Bot Chat when none is titled so. The
creation identity of an earlier target is immutable (a retry of it must converge, and a deleted one
must never be resurrected), so a rename or canonical delete must lead to a FRESH creation instead of
replaying the old one: retitling the renamed session back, or refusing on the retired id.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.session_contract import SessionActor
from tests.fixtures.storage_read import session_row


@pytest_asyncio.fixture
async def bot(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # A following Bot Chat reads the profile config by path.
    monkeypatch.setattr(run, '_load_gateway_config',
                        lambda *_path: {'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='fixture')
    owner = SessionActor('uid:1000', 'default', frozenset({'session:create', 'session:read', 'session:submit'}), 'native')
    try:
        yield SimpleNamespace(authority=authority, owner=owner, home=tmp_path)
    finally:
        store.close_all_db_handles()


async def _target_id(bot):
    from gateway.session_bot import _target
    ref, _live, _entry = await _target(bot.authority, bot.owner)
    return ref.session_id


@pytest.mark.asyncio
async def test_renamed_or_deleted_bot_chat_gets_a_fresh_replacement(bot):
    from gateway.session_ingress_media import native_inputs_root
    from session_retention import after_sessions_removed
    db = bot.authority.persistence
    first = await _target_id(bot)
    assert await _target_id(bot) == first  # converges: resolved by title, never re-created

    db.sessions.set_title(first, 'Planning notes')
    second = await _target_id(bot)
    assert second != first
    assert session_row(db, first)['title'] == 'Planning notes'  # the rename stands
    assert session_row(db, second)['title'] == 'Bot Chat'

    removed = db.sessions.delete(second, media_root=str(native_inputs_root()))
    after_sessions_removed(db, db.scope.profile_dir, removed, reason='delete', primary=second,
                           sessions_dir=bot.home / 'sessions')
    assert removed
    bot.authority.sessions.pop(second, None)
    third = await _target_id(bot)
    assert third not in (first, second)  # a replacement, not the retired id
    assert session_row(db, second) is None
    assert session_row(db, third)['title'] == 'Bot Chat'
    assert await _target_id(bot) == third
