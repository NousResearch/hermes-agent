"""Browser-compatible IPv6 bindings, including real encrypted storage."""
import json
import shutil
import subprocess

import pytest

from agent.vault_store import VaultError, VaultStore, normalize_origin


@pytest.mark.parametrize(('value', 'expected'), [
    ('https://[::1]:443/login?x=1#form', 'https://[::1]'),
    ('http://[0:0:0:0:0:0:0:1]:80/', 'http://[::1]'),
    ('https://[2001:0DB8:0:0:0:0:0:1]:8443/', 'https://[2001:db8::1]:8443'),
    ('https://[::ffff:192.0.2.128]:443/', 'https://[::ffff:c000:280]'),
    ('http://[::ffff:127.0.0.1]:8080/', 'http://[::ffff:7f00:1]:8080'),
    ('https://[0:0:0:0:0:0:0:0]', 'https://[::]'),
    ('https://[1:0:0:2:0:0:3:4]', 'https://[1::2:0:0:3:4]'),
    ('https://[1:2:3:4:5:6:0:8]', 'https://[1:2:3:4:5:6:0:8]'),
    ('https://[1:2:3:4:5:0:0:0]', 'https://[1:2:3:4:5::]'),
    ('https://user@Example.COM:443/login?x=1', 'https://example.com'),
    ('ftp://Example.COM/path', 'ftp://example.com'),
    ('https://Example.COM:443/login', 'https://example.com'),
    ('http://127.0.0.1:8080/login', 'http://127.0.0.1:8080'),
])
def test_origin_matches_browser_serialization(value, expected):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the URL.origin serialization oracle')
    browser_origin = subprocess.run(
        [node, '-e', 'process.stdout.write(new URL(process.argv[1]).origin)', value],
        capture_output=True, text=True, check=True,
    ).stdout
    assert browser_origin == expected
    assert normalize_origin(value) == expected
    assert normalize_origin(expected) == expected


def test_ipv6_binding_survives_encrypted_store_reload(tmp_path):
    store = VaultStore(tmp_path / 'vault')
    meta = store.add_item('login', 'IPv6 test', {'identifier_type': 'email', 'identifier': 'ipv6@example.test', 'password': 'ipv6-test-canary'},
                          origin='https://[0:0:0:0:0:0:0:1]:8443/login')
    reloaded = VaultStore(tmp_path / 'vault')
    assert reloaded.get_meta(meta.id).origin == 'https://[::1]:8443'
    assert reloaded.resolve_secret(meta.id) == {'password': 'ipv6-test-canary'}
    assert 'ipv6-test-canary' not in json.dumps(reloaded.get_meta(meta.id).to_dict())


@pytest.mark.parametrize('value', ['https://::1:8443', 'https://2001:db8::1',
                                  'http://[fe80::1%25en0]', 'https://[broken]'])
def test_ambiguous_or_non_browser_ipv6_is_not_rebound(value):
    with pytest.raises(VaultError):
        normalize_origin(value)
