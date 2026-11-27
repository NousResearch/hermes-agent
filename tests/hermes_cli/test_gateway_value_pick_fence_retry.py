"""A value pick (model / reasoning) refused at a moved fence is re-sent at fresh fences; a CAS edit is
not. Real authority, SQLite and AuthorityConnection.dispatch; another viewer's rename lands between the
pick's fence read and its write, as it does when an editor sends model and effort back to back."""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.usefixtures("authority_cwd_in_tmp")


class _Bridge:
    """The ACP/classic client transport; ``before_mutate`` runs another viewer's write first, once."""

    def __init__(self, connection, before_mutate=None):
        self.connection, self.before_mutate, self.mutates = connection, before_mutate, []

    async def rpc(self, method, **params):
        from hermes_cli.gateway_client import GatewayClientError
        if method == 'session.mutate':
            self.mutates.append(params['operation'])
            if self.before_mutate:
                hook, self.before_mutate = self.before_mutate, None
                await hook()
        reply = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayClientError(reply['error']['message'])
        return reply['result']


async def _authority(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _set_session_reasoning_override=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='owner')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human', 'profile_id': 'fixture'})
    ref = create_local_session(authority, connection.actor, {
        'request_id': 'cli', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': []})
    return store, connection, ref.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize('operation, payload, lands', [
    ('reasoning', {'effort': 'high'}, True),   # a value pick: re-sent at the new revision
    ('rename', {'title': 'mine'}, False),      # a CAS edit of shared metadata: the refusal stands
])
async def test_a_value_pick_survives_another_viewers_write(tmp_path, monkeypatch, operation, payload, lands):
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_mutations import PreparedMutations

    store, connection, sid = await _authority(tmp_path, monkeypatch)
    try:
        other = PreparedMutations()
        bridge = _Bridge(connection)
        bridge.before_mutate = lambda: other.apply(_Bridge(connection), sid, 'rename', {'title': 'theirs'})
        if lands:
            result = await PreparedMutations().apply(bridge, sid, operation, payload)
            assert result['operation'] == operation and 'revision' in result
            assert bridge.mutates == [operation, operation]
        else:
            with pytest.raises(GatewayClientError, match='revision_conflict'):
                await PreparedMutations().apply(bridge, sid, operation, payload)
            assert bridge.mutates == [operation]
    finally:
        store.close_all_db_handles()
