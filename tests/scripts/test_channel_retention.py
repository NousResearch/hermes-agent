"""Retention retires expired preview metadata with its claim archive, not live roots."""
from datetime import UTC, datetime

import pytest

from hermes_cli.release_channels import build_prefix, canonical_json, channel_key
from scripts.releases import r2
from scripts.releases.channels import ChannelPublisher, R2ChannelStore
from tests.scripts.test_release_channels import put_build
from tests.scripts.test_release_r2 import r2_server  # shared loopback fixture

NOW = datetime(2026, 10, 28, tzinfo=UTC).timestamp()


@pytest.fixture
def retained_channels(request):
    server = request.getfixturevalue('r2_server')
    store = R2ChannelStore(*r2.credentials())
    pub = ChannelPublisher(store, 'example/hermes-agent', store.base + '/' + store.bucket,
                           authorize=lambda action, record: None, verify_build=lambda request, manifest: True)
    pub.create('review')
    builds = []
    for stamp in ('20260901T120000Z', '20260902T120000Z', '20261027T120000Z'):
        admitted = pub.allocate_claim('review', f'channel.{stamp}-v1.2.3', 'a' * 40, '1.2.3')
        builds.append(admitted)
        seed_build(server, admitted)
    # Even an expired head stays; a fresh unpublished allocation stays within the window.
    pub.promote(builds[1]['buildId'])
    return server, pub, builds


def seed_build(server, admitted):
    objects = {}
    put_build(objects, admitted)
    archive = f"releases/tag/{admitted['releaseTag']}/"
    objects[archive + 'package.msixbundle'] = b'bundle'
    objects[archive + 'package.zip'] = b'zip'
    prefix = build_prefix(admitted['buildId'])
    objects[prefix + 'win32/stable.appinstaller'] = (
        f'<AppInstaller><MainBundle Uri="{admitted["publicBase"]}/{archive}package.msixbundle"/></AppInstaller>'
    ).encode()
    objects[prefix + 'darwin/stable-mac.yml'] = canonical_json({'version': admitted['version'], 'files': []})
    server.store.update({key: (body, '"fixture"') for key, body in objects.items()})


def test_tag_retention_reads_the_same_live_heads_over_http(retained_channels):
    from scripts import release

    _server, _pub, (_old, head, _fresh) = retained_channels
    assert release._channel_head_claims() == {head['releaseTag']}


def test_prune_retires_expired_nonhead_preview_feeds_and_archive_together(retained_channels, capsys):
    server, pub, (old, head, fresh) = retained_channels
    # Protected release metadata/feeds are never retired by preview retention.
    for name, tag, policy, version, windows in (
        ('stable', 'v1.2.3', 'stable-release', '1.2.3', '1.2.3.0'),
        ('canary', 'canary.20260901T120000Z-v1.2.3', 'canary-release',
         '1.2.3+canary.20260901T120000Z', '26.901.12.0'),
    ):
        protected = pub.allocate_protected(name, 'b' * 40, '1.2.3', release_tag=tag, version=version,
                                            windows_version=windows, identity=head['identity'], policy=policy,
                                            release_gate=lambda request: True)
        seed_build(server, protected)
    before = dict(server.store)
    prefix = build_prefix(old['buildId'])
    archive = f"releases/tag/{old['releaseTag']}/"
    doomed_preview = {key for key in before if key.startswith((prefix, archive))}
    r2.prune(14, dry_run=True, now_epoch=NOW)
    selected = {line.removeprefix('(dry-run) would delete r2:') for line in capsys.readouterr().out.splitlines()}
    assert doomed_preview <= selected
    assert not any(key.startswith(build_prefix(item['buildId'])) for key in selected for item in (head, fresh))
    assert server.store == before
    r2.prune(14, now_epoch=NOW)
    assert set(server.store) == set(before) - selected
    assert pub.reader.resolve('review').manifest['request'] == head
    protected_metadata = {key for key in before if key.startswith('releases/channel-builds/')} - doomed_preview
    assert protected_metadata <= server.store.keys()


@pytest.mark.parametrize('broken', ['claim-pointer', 'request', 'head', 'record', 'head-digest', 'claim-digest'])
def test_retention_refuses_unreadable_or_changed_channel_roots(retained_channels, broken):
    server, _pub, (old, head, _fresh) = retained_channels
    key = {
        'claim-pointer': f"releases/tag/{old['releaseTag']}/channel-request.json",
        'request': build_prefix(old['buildId']) + 'request.json',
        'head': build_prefix(head['buildId']) + 'build.json',
        'record': 'releases/channels/review.json',
        'head-digest': build_prefix(head['buildId']) + 'build.json',
        'claim-digest': f"releases/tag/{old['releaseTag']}/channel-request.json",
    }[broken]
    if broken == 'head-digest':
        import json
        value = json.loads(server.store[key][0])
        value['request']['commit'] = 'f' * 40
        body = canonical_json(value)
    elif broken == 'claim-digest':
        body = canonical_json({'schema': 1, 'buildId': old['buildId'], 'sha256': '0' * 64})
    else:
        body = b'unreadable'
    server.store[key] = (body, '"changed"')
    before = dict(server.store)
    server.requests.clear()
    with pytest.raises(RuntimeError, match='refusing to prune'):
        r2.prune(14, now_epoch=NOW)
    assert server.store == before
    assert not any(method == 'DELETE' for method, _path, _headers in server.requests)


def test_retention_can_resume_after_an_interrupted_descriptor_deletion(retained_channels, monkeypatch):
    server, _pub, (old, _head, _fresh) = retained_channels
    prefix = build_prefix(old['buildId'])
    archive = f"releases/tag/{old['releaseTag']}/"
    doomed = {key for key in server.store if key.startswith((prefix, archive))}
    signed_request = r2.signed_request

    def interrupted(method, url, **kwargs):
        if method == 'DELETE' and url.endswith(prefix + 'win32/stable.appinstaller'):
            raise OSError('fixture interrupted deletion')
        return signed_request(method, url, **kwargs)

    monkeypatch.setattr(r2, 'signed_request', interrupted)
    with pytest.raises(OSError, match='interrupted deletion'):
        r2.prune(14, now_epoch=NOW)
    monkeypatch.setattr(r2, 'signed_request', signed_request)
    r2.prune(14, now_epoch=NOW)
    assert not doomed.intersection(server.store)


@pytest.mark.parametrize('field', ['head', 'lastHead', 'destinationHead'])
def test_retention_binds_each_live_head_to_its_declared_channel(retained_channels, field):
    server, pub, (_old, head, _fresh) = retained_channels
    pub.create('other')
    other = pub.allocate_claim('other', 'channel.20260903T120000Z-v1.2.3', 'b' * 40, '1.2.3')
    seed_build(server, other)
    foreign = pub.promote(other['buildId'])['head']
    if field != 'head':
        stable = pub.allocate_protected('stable', 'c' * 40, '2.0.0', release_tag='v2.0.0',
                                        version='2.0.0', windows_version='2.0.0.0',
                                        identity=head['identity'], policy='stable-release',
                                        release_gate=lambda request: True)
        seed_build(server, stable)
        pub.promote_protected(stable['buildId'], policy='stable-release', release_gate=lambda request: True)
        pub.retire('review', 'stable', '2.0.0')
        # Valid retirement pins remain live before testing the corrupt cross-channel reference.
        from scripts.releases.channel_retention import PreviewRetention
        import json
        live = PreviewRetention(lambda key: json.loads(server.store[key][0])).live_claims(list(server.store))
        assert {head['releaseTag'], stable['releaseTag']} <= live
    record = pub._read('review')[0]
    record[field] = foreign
    if field == 'lastHead':
        record['head'] = foreign  # Keep the retired record structurally valid; only ownership is wrong.
    server.store[channel_key('review')] = (canonical_json(record), '"corrupt-owner"')
    before = dict(server.store)
    server.requests.clear()
    message = 'Invalid release tag' if field == 'destinationHead' else 'identity mismatch'
    with pytest.raises(RuntimeError, match=message + '.*refusing to prune'):
        r2.prune(14, now_epoch=NOW)
    assert server.store == before
    assert not any(method == 'DELETE' for method, _path, _headers in server.requests)
