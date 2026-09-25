"""Configured npm mirrors must be used for the pinned download plan."""
import hashlib

import pytest

from pm.store import Store
from tests.pm._range_server import RangeHandler, dl_server, url  # noqa: F401


def test_pinned_fetch_uses_user_npm_registry_without_changing_lock(tmp_path, monkeypatch):
    from pm import downloader

    (tmp_path / '.npmrc').write_text('registry=https://mirror.example/npm/\n', encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    original = 'https://registry.npmjs.org/npm/-/npm-1.2.3.tgz'
    artifact = {'url': original, 'sha256': 'a' * 64}
    observed = []

    def capture_plan(self, **kwargs):
        observed.extend(self.sources)
        return [source.dest for source in self.sources]

    monkeypatch.setattr(downloader.Download, 'run', capture_plan)
    store = Store(tmp_path / 'store')
    with store.scratch() as scratch:
        store.fetch_many([artifact], scratch)
    assert observed[0].url == 'https://mirror.example/npm/npm/-/npm-1.2.3.tgz'
    assert observed[0].sha256 == artifact['sha256']
    assert artifact['url'] == original
    assert observed[0].fallbacks == ()


@pytest.mark.parametrize('bad_hash', [False, True])
def test_real_mirror_download_keeps_hash_and_progress_identity(tmp_path, monkeypatch, dl_server, bad_hash):
    from pm.downloader import HashError

    payload = b'locked npm archive fixture' * 100
    RangeHandler.payloads['/mirror/npm/-/npm-1.2.3.tgz'] = payload
    monkeypatch.setenv('NPM_CONFIG_REGISTRY', url(dl_server, '/mirror/'))
    original = 'https://registry.npmjs.org/npm/-/npm-1.2.3.tgz'
    digest = hashlib.sha256(b'wrong' if bad_hash else payload).hexdigest()
    store = Store(tmp_path / 'store')
    ticks = []
    with store.scratch() as scratch:
        if bad_hash:
            with pytest.raises(HashError):
                store.fetch(original, digest, scratch)
            assert not store.entry(f'fetch-{digest}').exists()
        else:
            result = store.fetch_many([{'url': original, 'sha256': digest}], scratch,
                                      progress=lambda done, total, ranges: ticks.append(ranges))
            assert result[0].read_bytes() == payload
            assert set(ticks[-1]) == {original}


def test_registry_precedence_expansion_and_unrelated_sources(tmp_path, monkeypatch):
    from pm.npm_registry import registry_url, registry_download_url
    from pm.artifact_mirror import pinned_source

    alternate = tmp_path / 'custom.npmrc'
    alternate.write_text('registry=https://${MIRROR_HOST}/nested/\n', encoding='utf-8')
    monkeypatch.setenv('MIRROR_HOST', 'mirror.example')
    monkeypatch.setenv('NPM_CONFIG_USERCONFIG', str(alternate))
    assert registry_url() == 'https://mirror.example/nested/'
    monkeypatch.setenv('NPM_CONFIG_REGISTRY', 'https://override.example')
    assert registry_url() == 'https://override.example/'
    original = 'https://example.com/archive.tgz'
    assert registry_download_url(original) == original
    assert pinned_source(original, tmp_path / 'archive', 'b' * 64).fallbacks


@pytest.mark.parametrize('value', ['file:///tmp/packages', 'https://user:secret@example.com',
                                  'https://example.com/?secret=key', '${UNSET_MIRROR}'])
def test_invalid_registry_fails_without_leaking_value(monkeypatch, value):
    from pm.npm_registry import registry_url
    monkeypatch.setenv('NPM_CONFIG_REGISTRY', value)
    with pytest.raises(ValueError) as exc:
        registry_url()
    assert value not in str(exc.value)


def test_metadata_uses_same_registry_and_empty_result(monkeypatch):
    from pm import update
    monkeypatch.setenv('NPM_CONFIG_REGISTRY', 'https://mirror.example/npm/')
    seen = []
    def get_json(address):
        seen.append(address)
        return {}
    monkeypatch.setattr(update, '_get_json', get_json)
    assert update.npm_dist_tags('@scope%2Fpkg') == {}
    assert seen == ['https://mirror.example/npm/-/package/@scope%2Fpkg/dist-tags']


def test_default_registry_preserves_public_fallback(tmp_path):
    from pm.artifact_mirror import pinned_source, mirror_url
    original = 'https://registry.npmjs.org/npm/-/npm-1.2.3.tgz'
    source = pinned_source(original, tmp_path / 'archive', 'c' * 64)
    assert source.url == original
    assert source.fallbacks == (mirror_url('c' * 64),)
