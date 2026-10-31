"""``source: "desktop"`` — the label the gateway STORES for ``gui`` — is accepted at creation.

A Desktop chat's stored label is ``desktop`` (``SURFACES['gui']``), so a third-party client that
copies that label back was refused by the create contract (#145847). The wire accepts it as an
alias of ``gui``: the same mapping the legacy adoption lane already applies to stored rows.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.session_controls import AuthorityConnection


async def _authority(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway import run

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {'default': 'fixture'}, 'platform_toolsets': {'cli': []}})
    monkeypatch.setattr(run, '_resolve_gateway_model', lambda config: 'fixture')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = run.GatewayRunner.__new__(run.GatewayRunner)
    runner.adapters, runner.session_store, runner._session_db, runner._draining = {}, store, store._db, False
    authority = await initialize_session_authority(runner, profile_id=str(Path(store._db.db_path).parent), instance_id='test')
    authority._schedule = lambda ref: None
    return authority, store


async def _create(conn, authority, source, request_id):
    created = await conn.dispatch({'id': request_id, 'method': 'session.create',
                                   'params': {'request_id': request_id, 'source': source,
                                              'cwd': str(Path(authority.db.db_path).parent)}})
    assert 'result' in created, created
    return created['result']['stored_session_id']


@pytest.mark.asyncio
async def test_the_stored_desktop_label_creates_the_owning_gui_surface(tmp_path, monkeypatch):
    authority, store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    phone = await _create(conn, authority, 'desktop', 'phone')
    app = await _create(conn, authority, 'gui', 'app')
    described = await conn.dispatch({'id': 0, 'method': 'runtime.describe', 'params': {}})
    assert 'desktop' in described['result']['session_create']['sources']
    # Both labels commit the same row label: the alias resolves to the ``gui`` surface rather than
    # widening the stored vocabulary with a seventh one.
    rows = {sid: authority.db.get_session(sid)['source'] for sid in (phone, app)}
    assert rows[phone] == rows[app]
    assert authority.sessions[phone].source.platform == authority.sessions[app].source.platform
    # ... and the created session is the same conversation surface, so both stay in the picker.
    for sid in (phone, app):
        store._record_gateway_session_peer(sid, authority.sessions[sid].route, authority.sessions[sid].source)
    listed = await conn.dispatch({'id': 9, 'method': 'session.list', 'params': {}})
    assert sorted(row['id'] for row in listed['result']['sessions']) == sorted([phone, app])
    await conn.close()
    authority.db.close()


@pytest.mark.asyncio
async def test_a_label_that_is_not_a_known_alias_still_refuses_the_source_field(tmp_path, monkeypatch):
    authority, _store = await _authority(tmp_path, monkeypatch)
    conn = AuthorityConnection(authority, SimpleNamespace(write=lambda frame: None), {'user_id': 'owner'})
    for label in ('Desktop', 'desktops'):
        refused = await conn.dispatch({'id': label, 'method': 'session.create', 'params': {
            'request_id': label, 'source': label, 'cwd': str(Path(authority.db.db_path).parent)}})
        assert refused['error']['code'] == 4001, refused
        assert 'source' in json.dumps(refused['error']['data']), refused
    await conn.close()
    authority.db.close()


def test_build_policy_normalizes_the_alias_into_the_gui_launch(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway.session_policy import build_policy
    policy = build_policy({'source': 'desktop', 'cwd': str(tmp_path)},
                          {'model': {'default': 'fixture'}, 'platform_toolsets': {}})
    assert (policy.source, policy.platform) == ('gui', 'desktop')
    # The launch identity a later exact retry is judged by carries the aliased label too.
    assert json.loads(policy.request_json)['source'] == 'gui'


def test_the_alias_matches_the_legacy_adoption_lane():
    from gateway.session_policy import SURFACE_ALIASES
    from hermes_state_local_migration import LEGACY_CREATION_SOURCES
    assert SURFACE_ALIASES == LEGACY_CREATION_SOURCES
