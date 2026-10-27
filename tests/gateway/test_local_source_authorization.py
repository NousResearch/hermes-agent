"""Local authorization preserves private source identity without scanning other chats."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_local import LocalSessionAdapter


@pytest.mark.parametrize('change', [
    'none', 'unregistered', 'copy', 'platform', 'chat_id', 'user_id', 'profile',
    'chat_type', 'removed', 'replaced_live_source', 'replaced_registration',
])
def test_authorization_requires_registered_unchanged_live_source(change):
    source = SessionSource(platform=Platform.LOCAL, chat_id='owned', user_id='owner', chat_type='dm')
    live = SimpleNamespace(source=source)
    authority = SimpleNamespace(sessions={source.chat_id: live})
    adapter = LocalSessionAdapter(authority)
    adapter.register_source(source)
    candidate = replace(source) if change == 'copy' else source
    mutations = {
        'unregistered': adapter.sources.clear,
        'removed': authority.sessions.clear,
        'replaced_live_source': lambda: authority.sessions.update(
            {source.chat_id: SimpleNamespace(source=replace(source))}),
        'replaced_registration': lambda: adapter.register_source(replace(source)),
    }
    if change in {'platform', 'chat_id', 'user_id', 'profile', 'chat_type'}:
        setattr(source, change, Platform.TELEGRAM if change == 'platform' else 'changed')
    if change in mutations:
        mutations[change]()
    assert adapter.authorize_source(candidate) is (change == 'none')


def test_authorization_inspects_only_its_own_live_source_as_sessions_grow():
    visits = []

    class LiveSource:
        def __init__(self, source):
            self._source = source

        @property
        def source(self):
            visits.append(self)
            return self._source

    for size in (1, 1024):
        source = SessionSource(platform=Platform.LOCAL, chat_id='owned', user_id='owner', chat_type='dm')
        sessions = {str(i): LiveSource(object()) for i in range(size - 1)}
        owned = sessions[source.chat_id] = LiveSource(source)
        adapter = LocalSessionAdapter(SimpleNamespace(sessions=sessions))
        adapter.register_source(source)
        visits.clear()
        assert adapter.authorize_source(source)
        assert visits == [owned]
