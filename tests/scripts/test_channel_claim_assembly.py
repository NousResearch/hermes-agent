"""Channel publication crosses the real receipt, assembler and HTTP CAS boundaries."""
from __future__ import annotations

import hashlib
import json

from hermes_cli.release_channels import build_prefix, canonical_json
from scripts.releases import channel_claims, r2, stable
from tests.scripts.test_release_channels import _stage_accepted_native, object_server, publisher


def accepted_packages(objects, request):
    """Candidate rows from the staged packages, independently of the channel assembler."""
    prefix = f"releases/tag/{request['releaseTag']}/"
    bundle = next(key for key in objects if key.startswith(prefix) and key.endswith('.msixbundle'))
    packages = []
    for key, body in objects.items():
        if key.startswith(prefix + 'metadata-'):
            row = json.loads(body)
            artifact = prefix + row['filename'] if row['platform'] == 'macos' else bundle
            packages.append({**row, 'artifact': {'url': request['publicBase'] + '/' + artifact,
                                                'sha256': hashlib.sha256(objects[artifact]).hexdigest()}})
    return {'packages': packages}


def test_channel_publication_assembles_receipts_and_promotes_preview_over_http(tmp_path, monkeypatch):
    with object_server() as (url, objects, _headers, requests, _faults):
        pub = publisher(url)
        pub.create('review')
        tag = 'channel.20261028T130000Z-v1.2.3'
        request = pub.allocate_claim('review', tag, 'a' * 40, '1.2.3')
        _stage_accepted_native(tmp_path, objects, request['identity'], tag, request['commit'],
                               request['version'], request['windowsVersion'], 'stable')
        accepted = accepted_packages(objects, request)
        # Only Actions custody/candidate acceptance are supplied by the fixture. Receipt fetching,
        # request validation, native assembly, package matching and head CAS are production code.
        monkeypatch.setattr(stable, 'check_claim', lambda env: {
            'policy': 'channel', 'claim_tag': tag, 'commit': request['commit'],
            'channel': request['channel'], 'bundle_env': {}})
        monkeypatch.setattr(stable, 'read_admitted_candidate', lambda *args, **kwargs: accepted)
        monkeypatch.setattr(r2, 'credentials', lambda: (pub.store.creds, url, 'bucket'))
        monkeypatch.setattr(r2, 'public_base_url', lambda: pub.public_base)
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL', pub.public_base)
        env = {'GITHUB_REPOSITORY': pub.repository, 'CHANNEL_BUILD': request['buildId'],
               'CHANNEL_REQUEST_SHA256': hashlib.sha256(canonical_json(request)).hexdigest()}
        before = dict(objects)
        result = channel_claims.publish_claim(env, tmp_path / 'published')
        resolved = pub.reader.resolve('review')
        assert result['policy'] == 'preview'
        assert resolved.manifest['request'] == request
        assert {(row['platform'], row['arch']) for row in resolved.manifest['packages']} == {
            ('darwin', 'arm64'), ('darwin', 'x64'), ('win32', 'arm64'), ('win32', 'x64')}
        prefix = build_prefix(request['buildId'])
        assert set(objects) - set(before) == {
            prefix + 'build.json', prefix + 'darwin/stable-mac.yml', prefix + 'win32/stable.appinstaller'}
        for row in resolved.manifest['packages']:
            assert pub.reader.read_bytes(row['artifact']['key'], row['artifact']['sha256'])
            assert pub.reader.read_bytes(row['feed']['key'])
        settled = dict(objects)
        requests.clear()
        assert channel_claims.publish_claim(env, tmp_path / 'retry') == result
        assert objects == settled
        assert not any(method == 'PUT' for method, _key in requests)
