"""Public OAuth client-ID defaults work before deferred plugin code loads."""
import hashlib
import json
import os
import re
import ssl
from dataclasses import asdict
from urllib.request import urlopen

import pytest

from hermes_cli.plugin_oauth import _declared_clients, parse_oauth_clients, _pkce_config
from tests.hermes_cli.test_platform_plugin_oauth import _plugin, _entry, _env, _start
from tests.providers.fake_pkce_idp import FakeIdP
from tests.waiting import wait_until


@pytest.mark.platforms('posix')
def test_native_auth_add_public_default_without_environment(tmp_path):
    home = tmp_path / '.hermes/profiles/goonbot'
    idp = FakeIdP().start(tls_dir=tmp_path / 'tls')
    client_id = 'https://api.example.com/oauth/client.json'
    _plugin(home, 'native', _entry(idp.base, client_secret_env='', client_id_default=client_id),
            names=('NATIVE_CLIENT_ID',))
    env = {**_env(tmp_path, home), 'SSL_CERT_FILE': str(idp.ca_pem)}
    env.pop('NATIVE_CLIENT_ID'); env.pop('NATIVE_CLIENT_SECRET')
    log = tmp_path/'auth.log'
    process = master = None
    try:
        with log.open('w') as output:
            process, master = _start(env, output)
            def authorized_url():
                match = re.search(r'https://127\.0\.0\.1:[0-9]+/authorize\?[^\s\x1b]+', log.read_text())
                assert process.poll() is None or match, log.read_text()
                return match.group() if match else None
            url = wait_until(authorized_url, timeout=30, what='default-client authorize URL')
            with urlopen(url, timeout=10, context=ssl.create_default_context(cafile=str(idp.ca_pem))) as response:
                assert response.status == 200
            process.wait(timeout=30)
        assert process.returncode == 0, log.read_text()
        assert 'native:rooms' in json.loads((home/'auth.json').read_text())['plugin_oauth']
        assert idp.token_requests[0]['client_id'] == client_id
        assert not (home/'.env').exists()  # No credential/env bootstrap or plugin execution needed.
    finally:
        if process and process.poll() is None:
            process.terminate(); process.wait(timeout=10)
        if master is not None: os.close(master)
        idp.stop()


def test_default_preserves_owned_variable_admission_and_explicit_override(tmp_path, monkeypatch):
    home = tmp_path/'.hermes'
    directory = _plugin(home, 'native', _entry('https://login.example.com', client_id_default='public-default'),
                        names=('NATIVE_CLIENT_ID', 'NATIVE_CLIENT_SECRET'))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setenv('NATIVE_CLIENT_ID', 'explicit-owned-client')
    client = _declared_clients(directory)[0]
    assert _pkce_config(client, 'native:rooms').client_id == 'explicit-owned-client'
    _plugin(home, 'other', names=('NATIVE_CLIENT_ID',))
    with pytest.raises(ValueError, match='declared by another plugin'):
        _declared_clients(directory)


@pytest.mark.parametrize('value', ['', None, 3, [], 'bad\nclient'])
def test_public_default_rejects_invalid_values(value):
    _, problems = parse_oauth_clients({'oauth': [_entry('https://login.example.com', client_id_default=value)]})
    assert any('client_id_default' in problem for problem in problems)


def test_public_default_does_not_permit_core_credential_variables():
    _, problems = parse_oauth_clients({'oauth': [_entry('https://login.example.com',
        client_id_env='OPENAI_API_KEY', client_id_default='public-default')]})
    assert any('variable Hermes uses itself' in problem for problem in problems)


def test_legacy_manifest_fingerprint_remains_unchanged():
    clients, problems = parse_oauth_clients({'oauth': [_entry('https://login.example.com')]})
    assert not problems
    old_fields = asdict(clients[0])
    old_fields.pop('client_id_default', None)
    expected = hashlib.sha256(json.dumps(old_fields, sort_keys=True).encode()).hexdigest()
    assert clients[0].fingerprint == expected
