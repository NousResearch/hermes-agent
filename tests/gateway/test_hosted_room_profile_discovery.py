"""Hosted-group local membership works out of the box, without hosted_rooms.profiles config.

The canonical service used to know only its own profile plus ``hosted_rooms.profiles`` from
config; a default install has no such key, so every cross-profile roster died as
``invalid_params`` (#139002).
"""
import asyncio
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch



@pytest.mark.asyncio
async def test_sibling_profiles_are_local_group_members_without_config(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    sibling = tmp_path / 'profiles' / 'beta'
    sibling.mkdir(parents=True)
    (sibling / 'config.yaml').write_text('{}')
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
    try:
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        assert set(service.local_profiles()) == {'default', 'beta'}
        assert service.profile_homes()['beta'] == sibling
        room = service.create_room(room_id='room', name='Room', members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'beta', 'handle': 'two'}])
        assert {m['profile'] for m in room['members']} == {'default', 'beta'}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_identity_less_profile_dirs_stay_non_members(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'profiles' / 'shell').mkdir(parents=True)  # no identity marker
    db = SessionDB(tmp_path / 'state.db')
    authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
    try:
        service = CanonicalHostedRoomService(authority, asyncio.get_running_loop())
        assert service.local_profiles() == ('default',)
        with pytest.raises(ValueError, match='not local to this gateway'):
            service.create_room(room_id='room', name='Room', members=[
                {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
                {'member_id': 'two', 'profile': 'shell', 'handle': 'two'}])
    finally:
        db.close()
