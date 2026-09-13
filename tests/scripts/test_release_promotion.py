"""Promotion writes pointers only after verified candidate content exists."""
import hashlib
import io
import json
from pathlib import Path

import pytest

from scripts.bundles import release_artifacts as artifacts
from scripts.releases import r2
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


def test_publish_and_promote_use_the_same_content_before_any_channel_write(tmp_path, monkeypatch, r2_server):
    tag, commit = 'v1.2.3', 'a' * 40
    base = 'https://releases.example'
    content = {
        'app.msixbundle': b'windows transport bytes',
        'arm64.zip': b'mac arm transport bytes', 'x64.zip': b'mac x64 transport bytes',
        'apt/pool/package.deb': b'deb transport bytes',
        'apt/dists/hermes-stable/InRelease': b'signed-index transport bytes',
    }
    files = [{'path': name, 'url': f'{base}/releases/tag/{tag}/{name}', 'sha256': hashlib.sha256(data).hexdigest()} for name, data in content.items()]
    by_name = {item['path']: item for item in files}
    packages = []
    for platform in ('windows', 'macos'):
        for arch in ('x64', 'arm64'):
            file = by_name['app.msixbundle' if platform == 'windows' else arch + '.zip']
            packages.append({'platform': platform, 'arch': arch, 'tag': tag, 'commit': commit, 'identity': 'App',
                             'version': '1.2.3.0' if platform == 'windows' else '1.2.3',
                             **({'publisher': 'CN=Test', 'applicationId': 'App'} if platform == 'windows' else {'teamId': 'ABCDEFGHIJ'}),
                             'artifact': {'url': file['url'], 'sha256': file['sha256']}})
    manifest = {'schema': 1, 'tag': tag, 'commit': commit, 'files': files, 'packages': packages}

    class Response(io.BytesIO):
        def geturl(self):
            return base

    def fetch(url, **kwargs):
        if '/releases/tag/' in url:
            return Response(content[url.split(f'/releases/tag/{tag}/')[1]])
        return Response(r2_server.store[url.split(base + '/')[1]][0])

    monkeypatch.setattr('urllib.request.urlopen', fetch)
    artifacts.publish(manifest, tmp_path / 'publish', base)
    assert 'releases/termux/stable/pool/package.deb' in r2_server.store
    assert not any(key.endswith(('.appinstaller', 'InRelease')) for key in r2_server.store)

    events = []
    real_put = r2.put
    def put(**kwargs):
        events.append(kwargs['key'])
        real_put(**kwargs)
    monkeypatch.setattr(r2, 'put', put)
    monkeypatch.setattr(r2, 'finalize', lambda **kwargs: events.append('mac-feed'))
    artifacts.promote(manifest, tmp_path / 'promote', base)
    assert events[0] == 'mac-feed'
    assert events[-1] == 'releases/termux/stable/dists/hermes-stable/InRelease'
    assert 'releases/win32/stable/stable.appinstaller' in r2_server.store
    assert b'/releases/tag/v1.2.3/app.msixbundle' in r2_server.store['releases/win32/stable/stable.appinstaller'][0]

    events.clear()
    content['app.msixbundle'] = b'changed artifact'
    with pytest.raises(ValueError, match='digest mismatch'):
        artifacts.promote(manifest, tmp_path / 'broken', base)
    assert events == []
