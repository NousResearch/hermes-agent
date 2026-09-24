"""Explicit-router ASGI tier, not host discovery/lifecycle acceptance.

Real FastAPI parsing/endpoint/config save; synthetic auth dependency only. Missing
plugin installs no route, yielding behavioral 404 (not an import/setup error).
"""
import importlib
import json
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter('buzz')  # real-import provenance
URL = '/api/plugins/buzz-platform/policy'
KEY = 'a' * 64
FIELDS = ('allowed_users', 'allow_all_users', 'require_mention', 'thread_require_mention')
PREFIXES = ('gateway.platforms.buzz', 'gateway.platforms.buzz.extra', 'platforms.buzz',
            'platforms.buzz.extra', 'gateway.buzz', 'gateway.buzz.extra', 'buzz', 'buzz.extra')


def api_module():
    path = Path(__file__).resolve().parents[2] / 'plugins/platforms/buzz/dashboard/plugin_api.py'
    return importlib.import_module('plugins.platforms.buzz.dashboard.plugin_api') if path.exists() else None


def client():
    from fastapi import Depends
    app = FastAPI()
    app.state.enabled = True
    def auth(authorization: str | None = Header(default=None)):
        if authorization != 'Bearer synthetic-test-token':
            raise HTTPException(401)
        if not app.state.enabled:
            raise HTTPException(404)
    module = api_module()
    if module is not None:
        app.include_router(module.router, prefix='/api/plugins/buzz-platform', dependencies=[Depends(auth)])
    return TestClient(app, headers={'Authorization': 'Bearer synthetic-test-token'})


@pytest.fixture
def policy_home(tmp_path, monkeypatch):
    home = tmp_path / 'owner'; home.mkdir()
    managed = tmp_path / 'managed'; managed.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('HERMES_MANAGED_DIR', str(managed))
    for key in (*('BUZZ_' + f.upper() for f in FIELDS), 'GATEWAY_ALLOWED_USERS', 'GATEWAY_ALLOW_ALL_USERS'):
        monkeypatch.delenv(key, raising=False)
    (home / '.env').write_text('')
    (home / 'config.yaml').write_text('buzz: {allow_all_users: true}\n')
    return home


INVALID = [{}, {'policy': {}}, {'policy': None}, {'policy': {'unknown': True}},
           {'policy': {'allowed_users': KEY}}, {'policy': {'allowed_users': [1]}},
           {'policy': {'allowed_users': ['invalid-secret-input']}},
           {'policy': {'allowed_users': [KEY, 'npub1invalid']}},
           *({'policy': {field: None}} for field in FIELDS),
           *({'policy': {field: value}} for field in FIELDS[1:] for value in (0, 'false', [], {})),
           {'policy': {'allow_all_users': False}, 'unexpected': 'secret-input'}]


@pytest.mark.parametrize('body', INVALID)
def test_invalid_request_no_mutation(policy_home, body):
    path = policy_home / 'config.yaml'; before = (path.read_bytes(), path.stat().st_mtime_ns)
    response = client().put(URL, json=body)
    assert response.status_code == 422
    assert 'invalid-secret-input' not in response.text and 'npub1invalid' not in response.text
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def node(config, prefix):
    for part in prefix.split('.'):
        config = config.setdefault(part, {})
    return config


@pytest.mark.parametrize('prefix', (*PREFIXES, 'all'))
def test_partial_save_canonical_cleanup(policy_home, prefix):
    from plugins.platforms.buzz import settings
    config = {'unrelated': {'secret_ref': '${UNCHANGED}'}}
    for selected in PREFIXES if prefix == 'all' else (prefix,):
        node(config, selected).update(allowed_users=[KEY], allow_all_users=True,
                                     require_mention=False, thread_require_mention=True, transport='keep')
    path = policy_home / 'config.yaml'; path.write_text(yaml.safe_dump(config))
    response = client().put(URL, json={'policy': {'allow_all_users': False}})
    assert response.status_code == 200, response.text
    saved = yaml.safe_load(path.read_text())
    assert node(saved, PREFIXES[1]) == {**{k: v for k, v in node(config, PREFIXES[1]).items() if k not in FIELDS},
        'allowed_users': [KEY], 'allow_all_users': False, 'require_mention': False, 'thread_require_mention': True}
    for selected in PREFIXES:
        if selected != PREFIXES[1]:
            assert not set(FIELDS).intersection(node(saved, selected))
        if node(config, selected).get('transport'):
            assert node(saved, selected)['transport'] == 'keep'
    assert saved['unrelated'] == config['unrelated']
    assert settings.effective_authorization_policy(home=policy_home) == {'allowed_users': [KEY], 'allow_all_users': False}
    assert response.json()['policy'] == dict(allowed_users=[KEY], allow_all_users=False, require_mention=False, thread_require_mention=True)
    assert client().get(URL).json() == response.json()


def test_sparse_save_preserves_refs_and_explicit_empty(policy_home):
    path = policy_home / 'config.yaml'
    path.write_text('buzz: {allowed_users: ["${PRIVATE_ID}"], require_mention: "${PRIVATE_GATE}"}\n')
    (policy_home / '.env').write_text('PRIVATE_ID=' + KEY + '\nPRIVATE_GATE=false\n')
    response = client().put(URL, json={'policy': {'allow_all_users': False}})
    assert response.status_code == 200, response.text
    assert node(yaml.safe_load(path.read_text()), PREFIXES[1]) == {
        'allowed_users': ['${PRIVATE_ID}'], 'require_mention': '${PRIVATE_GATE}', 'allow_all_users': False}
    assert response.json()['policy']['allowed_users'] is None
    assert KEY not in response.text and 'PRIVATE_ID' not in response.text
    response = client().put(URL, json={'policy': {'allowed_users': []}})
    assert response.status_code == 200
    assert node(yaml.safe_load(path.read_text()), PREFIXES[1])['allowed_users'] == []


@pytest.mark.parametrize('document', ['[]', 'buzz: [', 'buzz: {extra: []}', 'platforms: []', 'gateway: {buzz: null}', 'buzz: {allowed_users: [bad]}'])
def test_malformed_user_refused(policy_home, document):
    path = policy_home / 'config.yaml'; path.write_text(document)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    response = client().put(URL, json={'policy': {'require_mention': False}})
    assert response.status_code == 409
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    response = client().get(URL)
    assert response.status_code == 200 and response.json()['locked'] is True
    assert response.json()['user_policy_unavailable'] is True
    assert all(value is None for value in response.json()['policy'].values())


@pytest.mark.parametrize('document', ['buzz: [', 'buzz: {allowed_users: []}', 'gateway: {buzz: {thread_require_mention: true}}', '[]'])
def test_managed_refused_without_leak(policy_home, document):
    import os
    (Path(os.environ['HERMES_MANAGED_DIR']) / 'config.yaml').write_text(document)
    path = policy_home / 'config.yaml'; before = path.read_bytes()
    response = client().put(URL, json={'policy': {'require_mention': False}})
    assert response.status_code == 409
    assert path.read_bytes() == before
    response = client().get(URL)
    assert response.status_code == 200 and response.json()['locked'] is True
    assert str(policy_home) not in response.text


def test_harness_auth_and_disabled_refusal(policy_home):
    c = client(); path = policy_home / 'config.yaml'; before = path.read_bytes()
    for method in ('get', 'put'):
        kwargs = {'json': {'policy': {'allow_all_users': False}}} if method == 'put' else {}
        assert getattr(c, method)(URL, headers={'Authorization': 'invalid'}, **kwargs).status_code == 401
        c.app.state.enabled = False
        assert getattr(c, method)(URL, **kwargs).status_code == 404
        c.app.state.enabled = True
    assert path.read_bytes() == before


def test_named_profile_normalizes_identity_and_metadata(policy_home, monkeypatch):
    from hermes_cli import profiles
    from tests.gateway.test_buzz_authz import SELF_NPUB
    from plugins.platforms.buzz.settings import normalize_user_ref
    decoded = normalize_user_ref(SELF_NPUB)
    assert decoded is not None
    target = policy_home / 'profiles' / 'selected'; target.mkdir(parents=True)
    (target / '.env').write_text('BUZZ_ALLOWED_USERS=' + 'b' * 64 + '\nGATEWAY_ALLOWED_USERS=' + 'c' * 64 + '\n')
    (target / 'config.yaml').write_text('buzz: {require_mention: false}\n')
    monkeypatch.setattr(profiles, '_get_profiles_root', lambda: policy_home / 'profiles')
    before = (policy_home / 'config.yaml').read_bytes()
    response = client().put(URL + '?profile=selected', json={'policy': {'allowed_users': [decoded.upper(), SELF_NPUB, decoded]}})
    assert response.status_code == 200, response.text
    assert (policy_home / 'config.yaml').read_bytes() == before
    saved = yaml.safe_load((target / 'config.yaml').read_text())
    assert node(saved, PREFIXES[1]) == {'allowed_users': [decoded], 'require_mention': False}
    payload = response.json()
    assert payload['profile'] == 'selected'
    assert payload['policy']['allowed_users'] is None
    assert payload['additional_global_grants_active'] is True
    assert 'b' * 64 not in response.text and 'c' * 64 not in response.text
    assert str(target) not in response.text
    assert client().get(URL + '?profile=selected').json() == payload


def test_coarse_managed_and_unreadable_refused(policy_home, monkeypatch):
    import hermes_cli.config as config
    path = policy_home / 'config.yaml'; before = path.read_bytes()
    monkeypatch.setattr(config, 'is_managed', lambda: True)
    assert client().put(URL, json={'policy': {'allowed_users': []}}).status_code == 409
    assert path.read_bytes() == before
    monkeypatch.setattr(config, 'is_managed', lambda: False)
    original = Path.open
    def denied(self, *args, **kwargs):
        if self == path:
            raise PermissionError('secret-path-must-not-leak')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', denied)
    response = client().put(URL, json={'policy': {'allowed_users': []}})
    assert response.status_code == 409 and 'secret-path' not in response.text
    assert client().get(URL).json()['locked'] is True
    monkeypatch.setattr(Path, 'open', original)
    assert path.read_bytes() == before
