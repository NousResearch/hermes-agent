"""Local authorization preserves private source identity without scanning other chats."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_detached import detached_key
from gateway.session_local import LocalSessionAdapter


@pytest.mark.parametrize('detached,chat_id', [
    (False, 'owned'), (False, 'agent:main:detached:another'), (True, 'shared-chat'),
])
@pytest.mark.parametrize('change', [
    'none', 'unregistered', 'copy', 'platform', 'chat_id', 'user_id', 'profile',
    'chat_type', 'removed', 'replaced_live_source', 'replaced_registration',
])
def test_authorization_requires_registered_unchanged_live_source(change, detached, chat_id):
    source = SessionSource(platform=Platform.TELEGRAM if detached else Platform.LOCAL,
                           chat_id=chat_id, user_id='owner', chat_type='dm',
                           detached_route=detached_key('agent:main:telegram:shared-chat', 'owned') if detached else None)
    sid = 'owned' if detached else chat_id
    live = SimpleNamespace(source=source)
    sibling = replace(source, chat_id=chat_id if detached else 'sibling',
                      detached_route=detached_key('agent:main:telegram:shared-chat', 'sibling') if detached else None)
    authority = SimpleNamespace(sessions={'sibling': SimpleNamespace(source=sibling), sid: live})
    adapter = LocalSessionAdapter(authority)
    adapter.register_source(sibling)
    adapter.register_source(source)
    candidate = replace(source) if change == 'copy' else source
    mutations = {
        'unregistered': adapter.sources.clear,
        'removed': lambda: authority.sessions.pop(sid),
        'replaced_live_source': lambda: authority.sessions.update(
            {sid: SimpleNamespace(source=replace(source))}),
        'replaced_registration': lambda: adapter.register_source(replace(source)),
    }
    if change in {'platform', 'chat_id', 'user_id', 'profile', 'chat_type'}:
        setattr(source, change, Platform.DISCORD if change == 'platform' else 'changed')
    if change in mutations:
        mutations[change]()
    assert adapter.authorize_source(candidate) is (change == 'none')


@pytest.mark.parametrize('detached,chat_id', [
    (False, 'owned'), (False, 'agent:main:detached:another'), (True, 'shared-chat'),
])
def test_authorization_inspects_only_its_own_live_source_as_sessions_grow(detached, chat_id):
    visits = []

    class LiveSource:
        def __init__(self, source):
            self._source = source

        @property
        def source(self):
            visits.append(self)
            return self._source

    for size in (1, 1024):
        source = SessionSource(platform=Platform.TELEGRAM if detached else Platform.LOCAL,
                               chat_id=chat_id, user_id='owner', chat_type='dm',
                               detached_route=detached_key('agent:main:telegram:shared-chat', 'owned') if detached else None)
        sessions = {str(i): LiveSource(replace(source, detached_route=detached_key(
            'agent:main:telegram:shared-chat', str(i))) if detached else object()) for i in range(size - 1)}
        owned = sessions['owned' if detached else chat_id] = LiveSource(source)
        adapter = LocalSessionAdapter(SimpleNamespace(sessions=sessions))
        adapter.register_source(source)
        visits.clear()
        assert adapter.authorize_source(source)
        assert visits == [owned]
