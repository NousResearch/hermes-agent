"""The one-time import: crash recovery, bootstrap prefixes kept and a cold real-credential boot; loopback plane, real file locks."""
import contextlib
import errno
import json
import logging
import os
import subprocess
import sys

import pytest
from tests.plugins.secret_store.test_remote_secret_store import _boot
pytest_plugins = ['tests.plugins.secret_store.test_remote_secret_store']


@pytest.mark.parametrize('nested', ["scalar", "nested", "pool"])
def test_auth_crash_recovery_does_not_quarantine_already_imported_record(plane, monkeypatch, nested):
    from plugins.secret_store.remote import importer
    from hermes_secret_routing import routed_provider
    record = {'key': 'auth-marker'} if nested == "scalar" else {'key': 'auth-marker', 'scopes': ['read', {'nested': ['write']}]}
    path = plane.home / 'auth.json'
    store = {'credential_pool': {'example': [record]}} if nested == "pool" else {'providers': {'example': record}}
    path.write_text(json.dumps(store), encoding='utf-8')
    original = importer._keep_bootstrap_only
    class Crash(BaseException):
        pass
    def crash(*args):
        raise Crash()
    monkeypatch.setattr(importer, '_keep_bootstrap_only', crash)
    with pytest.raises(Crash):
        _boot(plane.home)
    assert len(plane.patches()) == 1
    monkeypatch.setattr(importer, '_keep_bootstrap_only', original)
    _boot(plane.home)
    records = routed_provider().auth_view(plane.home)
    assert records['pool:example' if nested == "pool" else 'example'] == ({'entries': [record]} if nested == "pool" else record)
    assert not (plane.home / importer.UNIMPORTED_AUTH).exists()


def test_idp_prefix_bootstrap_preserved_during_cleanup(plane):
    from hermes_secret_routing import routed_provider
    provider = routed_provider()
    key = 'GATEWAY_RELAY_IDP_AUDIENCE'
    assert provider.keeps_local(key)
    path = plane.home / '.env'
    path.write_text(f'{key}=bootstrap-marker\nIMPORT_KEY=import-marker\n', encoding='utf-8')
    _boot(plane.home)
    assert key not in provider.env_view(plane.home)
    assert f'{key}=bootstrap-marker' in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('expired', [False, True])
def test_cold_real_nous_boot_import_cleanup_poll_write(plane, monkeypatch, expired):
    from tests.plugins.config_backends.stub_plane import StubPlane, NOUS_TOKEN
    from tests.plugins.config_backends.test_remote_nous_cold_boot import _auth_json
    from tests.plugins.secret_store.stub_secrets_plane import TOKEN
    with StubPlane() as portal:
        record = _auth_json(portal, expired=expired)
        if not expired:
            record['providers']['nous']['access_token'] = TOKEN
        (plane.home / 'auth.json').write_text(json.dumps({**record, 'providers': {**record['providers'], 'example': {'key': 'local-marker'}}}), encoding='utf-8')
        (plane.home / '.env').write_text('IMPORT_KEY=import-marker\n', encoding='utf-8')
        # Refreshed StubPlane token accepted by secret-plane identity validation via test constant.
        import tests.plugins.secret_store.stub_secrets_plane as stub
        monkeypatch.setattr(stub, 'TOKEN', NOUS_TOKEN if expired else TOKEN)
        env = dict(os.environ)
        for key in list(env):
            if key.startswith('GATEWAY_RELAY_IDP_'):
                env.pop(key)
        env.update(HERMES_CONFIG_BACKEND='local', HERMES_SHARED_AUTH_DIR=str(plane.home / 'shared'),
                   HERMES_PORTAL_BASE_URL=portal.url, NOUS_PORTAL_BASE_URL=portal.url)
        source = '''
import os, sys
from pathlib import Path
assert 'hermes_cli.auth' not in sys.modules
from plugins.secret_store.remote import get_provider
p = get_provider(); h = Path(os.environ['HERMES_HOME'])
p.boot(h)
assert p.env_view(h)['IMPORT_KEY'] == 'import-marker'
assert p.auth_view(h)['example'] == {'key': 'local-marker'}
p.poll_all(); p.write_view(h, env_set={'COLD_KEY':'write-marker'})
assert p.env_view(h)['COLD_KEY'] == 'write-marker'
assert 'local-marker' not in (h/'auth.json').read_text()
assert 'IMPORT_KEY' not in (h/'.env').read_text()
print('COLD_REAL_NOUS_IMPORT_POLL_WRITE_OK')
'''
        result = subprocess.run([sys.executable, '-c', source], env=env, cwd=os.getcwd(), capture_output=True, text=True, timeout=45, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert portal.nous_refreshes == int(expired)
