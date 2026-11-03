"""Channel publication crosses the real receipt, assembler and HTTP CAS boundaries."""
from __future__ import annotations

import hashlib
import json

import pytest

from hermes_cli.release_channels import build_prefix, canonical_json
from scripts.releases import channel_claims, lanes as lane_set, r2, stable
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
            'channel': request['channel'], 'bundle_env': {}, 'require_green': False})
        monkeypatch.setattr(stable, 'read_admitted_candidate', lambda *args, **kwargs: accepted)
        monkeypatch.setattr(r2, 'credentials', lambda: (pub.store.creds, url, 'bucket'))
        monkeypatch.setattr(r2, 'public_base_url', lambda: pub.public_base)
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL', pub.public_base)
        env = {'GITHUB_REPOSITORY': pub.repository, 'CHANNEL_BUILD': request['buildId'],
               'CHANNEL_REQUEST_SHA256': hashlib.sha256(canonical_json(request)).hexdigest(),
               'RELEASE_NEEDS': json.dumps({job: {'result': 'success'} for job in (
                   'admit', 'candidate-manifest', 'upgrade-arms')})}
        # The next build's upgrade baseline is the channel head, read back over HTTP; the manifest
        # fetch alone reads the fixture's objects (read_manifest insists on HTTPS).
        monkeypatch.setattr(stable, 'read_manifest', lambda url, **_: json.loads(
            objects[url.removeprefix(pub.public_base + '/')]))
        claim = {'policy': 'channel', 'channel': 'review', 'claim_tag': 'channel.20261029T130000Z-v1.2.3'}
        assert stable._published_baseline(env, pub.public_base, claim) is None  # no head yet
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

        candidates = f'releases/tag/{tag}/release-candidates.json'
        objects[candidates] = json.dumps({**accepted, 'tag': tag}).encode()
        assert stable._published_baseline(env, pub.public_base, claim) == {**accepted, 'tag': tag}
        # The candidate manifest must name the head's claim and hold exactly the packages it serves.
        objects[candidates] = json.dumps({**accepted, 'tag': 'channel.20261027T130000Z-v1.2.3'}).encode()
        with pytest.raises(ValueError, match='different claims'):
            stable._published_baseline(env, pub.public_base, claim)
        swapped = [{**row, 'artifact': {**row['artifact'], 'sha256': 'f' * 64}} for row in accepted['packages']]
        objects[candidates] = json.dumps({'packages': swapped, 'tag': tag}).encode()
        with pytest.raises(ValueError, match='differs from the accepted'):
            stable._published_baseline(env, pub.public_base, claim)


def test_admission_allocates_the_claims_one_request_and_every_leg_binds_to_it(tmp_path, monkeypatch):
    """stable.admit for a channel claim allocates its request over the real CAS (and owns no GitHub
    release); a rerun finds the same request; require_admitted_request refuses anything else."""
    with object_server() as (url, objects, _headers, _requests, _faults):
        pub = publisher(url)
        tag = 'channel.20261028T130000Z-v1.2.3'
        admitted = {'policy': 'channel', 'claim_tag': tag, 'tag': tag, 'commit': 'a' * 40, 'claim_object': 'b' * 40,
                    'version': '1.2.3+channel.20261028T130000Z', 'claim_epoch': stable.stamped_epoch(tag),
                    'lanes': lane_set.CHANNEL, 'require_green': False, 'bundle_env': {'HERMES_PREVIEW_FULL_CONNECTORS': '1'},
                    'channel': 'review', 'branding': 'preview'}
        monkeypatch.setattr(stable, 'check_claim', lambda env: admitted)
        monkeypatch.setattr(stable, 'output', lambda argv: pytest.fail(f'a channel claim owns no GitHub release: {argv}'))
        monkeypatch.setattr(r2, 'credentials', lambda: (pub.store.creds, url, 'bucket'))
        monkeypatch.setenv('CLOUDFLARE_R2_PUBLIC_URL', pub.public_base)
        env = {'GITHUB_REPOSITORY': pub.repository, 'GITHUB_SHA': 'c' * 40,
               'GITHUB_OUTPUT': str(tmp_path / 'output'), 'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary')}

        def admit():
            (tmp_path / 'output').unlink(missing_ok=True)
            stable.admit(env)
            return dict(line.split('=', 1) for line in (tmp_path / 'output').read_text(encoding='utf-8').splitlines())

        emitted = admit()
        pointer = json.loads(objects[f'releases/tag/{tag}/channel-request.json'])
        assert (emitted['channel-build'], emitted['channel-request-sha256']) == (pointer['buildId'], pointer['sha256'])
        assert emitted['policy'] == 'channel' and emitted['release-id'] == ''
        assert json.loads(emitted['jobs']) == lane_set.jobs_to_run(lane_set.CHANNEL, 'channel')
        # A rerun of admission finds the same request rather than allocating another.
        settled = dict(objects)
        assert admit() == emitted and objects == settled

        legs = {**env, 'CHANNEL_BUILD': emitted['channel-build'], 'CHANNEL_REQUEST_SHA256': emitted['channel-request-sha256']}
        request = channel_claims.require_admitted_request(tag, admitted, legs)
        assert (request['releaseTag'], request['channel'], request['commit']) == (tag, 'review', 'a' * 40)
        assert request['bundleEnv'] == admitted['bundle_env']
        for other in ({'CHANNEL_BUILD': 'f' * 32}, {'CHANNEL_REQUEST_SHA256': 'f' * 64}, {'CHANNEL_BUILD': ''}):
            with pytest.raises(ValueError, match='differs from the one admitted'):
                channel_claims.require_admitted_request(tag, admitted, {**legs, **other})
        for other in ({'channel': 'elsewhere'}, {'commit': 'd' * 40}, {'bundle_env': {}}):
            with pytest.raises(ValueError, match='does not match its claim'):
                channel_claims.require_admitted_request(tag, {**admitted, **other}, legs)
