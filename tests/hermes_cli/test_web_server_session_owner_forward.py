"""A standalone dashboard's mutations must reach the live gateway owner (#138915).

While a gateway serves the same home, the dashboard process is not the store owner, so its
offline mutation path can never win the exclusive-maintenance reservation: forward the
revision-fenced ``session.mutate`` verb to the owner over the control socket instead, and
keep the offline store only for an owner-less home.
"""
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException


def _standalone_request(monkeypatch, tmp_path):
    """An authenticated principal on a standalone dashboard (no composed gateway)."""
    monkeypatch.setattr('hermes_state._default_db_path', lambda: str(tmp_path / 'state.db'))
    return SimpleNamespace(
        state=SimpleNamespace(native_http_principal=None,
                              session=SimpleNamespace(user_id='user-a', provider='oidc',
                                                      issuer='https://idp.example')),
        app=SimpleNamespace(state=SimpleNamespace(auth_required=True, session_authority=None,
                                                  gateway_runner=None)))


def _discover(monkeypatch, endpoint, seen=None):
    def fake_discover(home, timeout=2.0):
        if seen is not None:
            seen['home'] = home
        return SimpleNamespace(endpoint=endpoint)
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint', fake_discover)


@pytest.mark.asyncio
async def test_standalone_mutation_forwards_to_the_live_owner(tmp_path, monkeypatch):
    from hermes_cli import web_server_sessions

    request = _standalone_request(monkeypatch, tmp_path)
    seen = {}
    calls = {}

    class FakeClient:
        async def rpc(self, method, **params):
            calls['rpc'] = (method, params)
            return {'session_id': params['session_id'], 'operation': 'delete', 'revision': 1}

    @asynccontextmanager
    async def fake_connect(endpoint=None, subject=None):
        calls['endpoint'] = endpoint
        calls['subject'] = subject
        yield FakeClient()

    offline = []
    monkeypatch.setattr(web_server_sessions, '_mutate_offline_store',
                        lambda profile, actor, params: offline.append(params))
    _discover(monkeypatch, 'owner-endpoint', seen)
    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', fake_connect)

    result = await web_server_sessions._mutate_session_request(
        request, None, 'session-1', request_id='req-1', expected_revision=0,
        operation='delete', payload={}, expected_generation=0)

    assert result == {'ok': True, 'session_id': 'session-1', 'operation': 'delete', 'revision': 1}
    # Discovery resolved the selected home's owner; the offline store never opened.
    assert seen['home'] == tmp_path.resolve()
    assert calls['endpoint'] == 'owner-endpoint'
    # The ticket names the dashboard's authenticated principal — the socket UID would be
    # refused by authorize_history on any row that stores an owner.
    assert calls['subject'] == 'auth:v1:["oidc","https://idp.example","user-a"]'
    method, params = calls['rpc']
    assert method == 'session.mutate'
    assert params == {'session_id': 'session-1', 'request_id': 'req-1', 'expected_revision': 0,
                      'operation': 'delete', 'payload': {}, 'expected_generation': 0}
    assert offline == []


@pytest.mark.asyncio
async def test_standalone_mutation_falls_back_to_offline_without_an_owner(tmp_path, monkeypatch):
    from hermes_cli import web_server_sessions

    request = _standalone_request(monkeypatch, tmp_path)
    _discover(monkeypatch, None)
    offline = []

    def fake_offline(profile, actor, params):
        offline.append(params)
        return {'session_id': params['session_id'], 'operation': 'rename', 'revision': 2}

    monkeypatch.setattr(web_server_sessions, '_mutate_offline_store', fake_offline)

    result = await web_server_sessions._mutate_session_request(
        request, None, 'session-2', request_id='req-2', expected_revision=1,
        operation='rename', payload={'title': 'renamed'})

    assert result == {'ok': True, 'session_id': 'session-2', 'operation': 'rename', 'revision': 2}
    assert len(offline) == 1
    assert offline[0]['operation'] == 'rename'


@pytest.mark.asyncio
async def test_owner_dial_failure_surfaces_a_retryable_error(tmp_path, monkeypatch):
    """A dial that dies on an endpoint discovery just identified is not owner-less (a slow first
    dial, not a stale endpoint): the offline reservation would only lose to that same owner and
    answer with the bare 409 this forward exists to remove, so a retryable 503 is the honest
    verdict — a later retry rediscovers and takes the offline store once the owner is really gone."""
    from hermes_cli import web_server_sessions

    request = _standalone_request(monkeypatch, tmp_path)
    _discover(monkeypatch, 'owner-endpoint')

    @asynccontextmanager
    async def dying_connect(endpoint=None, subject=None):
        raise TimeoutError('first dial outlived its budget')
        yield  # pragma: no cover

    offline = []
    monkeypatch.setattr(web_server_sessions, '_mutate_offline_store',
                        lambda profile, actor, params: offline.append(params))
    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', dying_connect)

    with pytest.raises(HTTPException) as exc:
        await web_server_sessions._mutate_session_request(
            request, None, 'session-3', request_id='req-3', expected_revision=0,
            operation='archive', payload={})
    assert exc.value.status_code == 503
    assert exc.value.detail == 'owner_unreachable'
    assert offline == []


@pytest.mark.asyncio
async def test_owner_refusal_maps_to_the_matching_http_status(tmp_path, monkeypatch):
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli import web_server_sessions

    request = _standalone_request(monkeypatch, tmp_path)
    _discover(monkeypatch, 'owner-endpoint')

    @asynccontextmanager
    async def refusing_connect(endpoint=None, subject=None):
        class Refuser:
            async def rpc(self, method, **params):
                raise GatewayClientError('permission_denied')
        yield Refuser()

    monkeypatch.setattr('hermes_cli.gateway_client.connect_gateway', refusing_connect)
    with pytest.raises(HTTPException) as exc:
        await web_server_sessions._mutate_session_request(
            request, None, 'session-4', request_id='req-4', expected_revision=0,
            operation='delete', payload={}, expected_generation=0)
    assert exc.value.status_code == 403
    assert exc.value.detail == 'permission_denied'
