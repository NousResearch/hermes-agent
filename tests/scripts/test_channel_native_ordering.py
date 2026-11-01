"""Preview admission sequence cannot authorize a native package downgrade."""
import pytest

from hermes_cli.release_channels import ChannelError
from tests.scripts.test_release_channels import object_server, publisher, put_build


@pytest.mark.parametrize('stamp', ['20261028T120000Z', '20261028T130000Z'])
def test_later_allocation_cannot_publish_older_or_equal_native_version(stamp):
    with object_server() as (url, objects, _headers, requests, _faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        pub.create('review')
        first = pub.allocate_claim('review', 'channel.20261028T130000Z-v2.0.0', 'a' * 40, '2.0.0')
        put_build(objects, first)
        head = pub.promote(first['buildId'])
        # Even a higher source SemVer cannot override equal/older native versions.
        stale = pub.allocate_claim('review', f'channel.{stamp}-v3.0.0', 'b' * 40, '3.0.0')
        assert stale['sequence'] > first['sequence']
        put_build(objects, stale)
        requests.clear()
        with pytest.raises(ChannelError, match='native versions must increase'):
            pub.promote(stale['buildId'])
        assert not any(method == 'PUT' for method, _key in requests)
        assert pub.reader.resolve('review').terminal['head'] == head['head']
        # The exact head remains an idempotent retry despite the later allocation.
        assert pub.promote(first['buildId'])['head'] == head['head']
        # Building an older source tree later is valid: ordering is native stamp, not SemVer.
        next_build = pub.allocate_claim('review', 'channel.20261028T140000Z-v1.0.0', 'c' * 40, '1.0.0')
        put_build(objects, next_build)
        pub.promote(next_build['buildId'])
        assert pub.reader.resolve('review').manifest['request'] == next_build
        with pytest.raises(ChannelError, match='stale'):
            pub.promote(first['buildId'])


@pytest.mark.parametrize('replacement', [None, b'{}'])
def test_preview_promotion_fails_closed_when_previous_manifest_cannot_be_verified(replacement):
    with object_server() as (url, objects, _headers, requests, _faults):
        pub = publisher(url, verify_build=lambda request, manifest: True)
        pub.create('review')
        first = pub.allocate_claim('review', 'channel.20261028T130000Z-v1.0.0', 'a' * 40, '1.0.0')
        put_build(objects, first)
        head = pub.promote(first['buildId'])['head']
        second = pub.allocate_claim('review', 'channel.20261028T140000Z-v1.0.0', 'b' * 40, '1.0.0')
        put_build(objects, second)
        if replacement is None:
            del objects[head['manifestKey']]
        else:
            objects[head['manifestKey']] = replacement
        requests.clear()
        with pytest.raises(ChannelError, match='manifest is unavailable or changed'):
            pub.promote(second['buildId'])
        assert not any(method == 'PUT' for method, _key in requests)
