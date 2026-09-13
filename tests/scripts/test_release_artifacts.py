"""Native metadata and artifact publication use the same verified bytes."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from scripts.bundles.release_artifacts import materialize, record, stamp_matches
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


def test_windows_metadata_is_read_from_package_and_stale_stamp_is_rejected(tmp_path):
    tag, commit = 'v1.2.3', 'a' * 40
    root = tmp_path / 'release'
    root.mkdir()
    package = root / 'Product-win-x64.msix'
    manifest = '<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="Product" Publisher="CN=Test" Version="1.2.3.0" ProcessorArchitecture="x64"/><Applications><Application Id="App"/></Applications></Package>'

    def write_package(sha):
        with zipfile.ZipFile(package, 'w') as archive:
            archive.writestr('AppxManifest.xml', manifest)
            archive.writestr('app/resources/install-stamp.json', json.dumps({'tag': tag, 'commit': sha}))

    write_package(commit)
    out = root / 'metadata-windows-x64.json'
    original = package.read_bytes()
    record('windows', 'x64', root, tag, commit, out)
    metadata = json.loads(out.read_text(encoding='utf-8'))
    assert metadata['identity'] == 'Product'
    assert metadata['version'] == '1.2.3.0'
    assert metadata['publisher'] == 'CN=Test'
    assert metadata['applicationId'] == 'App'
    assert package.read_bytes() == original
    write_package('b' * 40)
    with pytest.raises(ValueError, match='provenance'):
        record('windows', 'x64', root, tag, commit, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='provenance'):
        stamp_matches({}, tag, commit)


def test_assemble_uses_staged_receipts_and_only_publishes_the_manifest(tmp_path, monkeypatch, r2_server):
    from scripts.bundles.release_artifacts import assemble
    from scripts.releases import handoff

    tag, commit, base = 'v1.2.3', 'a' * 40, 'https://releases.example'
    built = tmp_path / 'built'
    built.mkdir()
    for platform, arches in [('windows', ('x64', 'arm64')), ('macos', ('x64', 'arm64')), ('termux', ('aarch64',))]:
        for arch in arches:
            row = {'platform': platform, 'arch': arch, 'tag': tag, 'commit': commit, 'identity': 'Product'}
            if platform == 'windows':
                row.update(version='1.2.3.0', publisher='CN=Test', applicationId='App')
                package = f'Product-win-{arch}.msix'
                handoff_name = f'win32-{arch}'
            elif platform == 'macos':
                package = f'Product-mac-{arch}.zip'
                row.update(version='1.2.3', teamId='ABCDEFGHIJ', filename=package)
                handoff_name = f'darwin-{arch}'
            else:
                package = 'deb/product.deb'
                row.update(version='1.2.3-1', filename=package)
                handoff_name = 'termux'
            file = built / package
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'package transport fixture')
            metadata = built / f'metadata-{platform}-{arch}.json'
            metadata.write_text(json.dumps(row), encoding='utf-8')
            handoff.stage(tag, commit, handoff_name, built, [package, metadata.name])
    bundle = built / 'Product-win.msixbundle'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('AppxMetadata/AppxBundleManifest.xml', '<Bundle><Identity Name="Product" Publisher="CN=Test" Version="1.2.3.0"/></Bundle>')
    (built / 'Store-Product-win.msixbundle').write_bytes(b'Store bundle transport fixture')
    handoff.stage(tag, commit, 'windows-universal', built, ['*.msixbundle'])
    fetched = tmp_path / 'fetched'
    names = ['win32-x64', 'win32-arm64', 'darwin-x64', 'darwin-arm64', 'termux', 'windows-universal']
    handoff.fetch(tag, commit, names, fetched, ['metadata-*.json', '*.msixbundle'])
    r2_server.requests.clear()
    manifest = assemble(fetched, tag, commit, base, tmp_path / 'release-candidates.json')
    assert {row['platform'] + '/' + row['arch'] for row in manifest['packages']} == {
        'windows/x64', 'windows/arm64', 'macos/x64', 'macos/arm64', 'termux/aarch64'}
    assert all(not file['path'].startswith(('handoff-', 'metadata-')) for file in manifest['files'])
    puts = [path for method, path, _ in r2_server.requests if method == 'PUT']
    assert puts == [f'/hermes-releases/releases/tag/{tag}/release-candidates.json']
    assert all(key.startswith(f'releases/tag/{tag}/') for key in r2_server.store)
    receipt = fetched / 'handoff-darwin-arm64.json'
    original = receipt.read_bytes()
    receipt.unlink()
    with pytest.raises(ValueError, match='handoff'):
        assemble(fetched, tag, commit, base, tmp_path / 'missing.json')
    receipt.write_bytes(original)
    (fetched / 'metadata-windows-x64.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        assemble(fetched, tag, commit, base, tmp_path / 'changed.json')


def test_materialize_validates_the_published_file_receipt_before_using_bytes(tmp_path, monkeypatch):
    base, tag, commit = 'https://releases.example', 'v1.2.3', 'a' * 40
    data = b'package transport fixture, not native signing proof'
    digest = hashlib.sha256(data).hexdigest()
    files, packages = [], []
    for platform in ('windows', 'macos'):
        for arch in ('x64', 'arm64'):
            filename = f'{platform}-{arch}.' + ('msixbundle' if platform == 'windows' else 'zip')
            url = f'{base}/releases/tag/{tag}/{filename}'
            files.append({'path': filename, 'url': url, 'sha256': digest})
            packages.append({'platform': platform, 'arch': arch, 'identity': 'Product', 'tag': tag, 'commit': commit,
                             'version': '1.2.3.0' if platform == 'windows' else '1.2.3',
                             **({'publisher': 'CN=Test', 'applicationId': 'App'} if platform == 'windows' else {'teamId': 'ABCDEFGHIJ'}),
                             'artifact': {'url': url, 'sha256': digest}})
    manifest = {'schema': 1, 'tag': tag, 'commit': commit, 'packages': packages, 'files': files}
    class Response(io.BytesIO):
        def geturl(self):
            return base + "/package"

    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(data))
    materialize(manifest, tmp_path / 'good', public_base=base)
    assert all((tmp_path / 'good' / f['path']).read_bytes() == data for f in files)
    with pytest.raises(ValueError, match='one Store candidate'):
        materialize(manifest, tmp_path / 'store-missing', public_base=base, store_only=True)
    store = {'path': 'Store-App.msixbundle', 'url': f'{base}/releases/tag/{tag}/Store-App.msixbundle', 'sha256': digest}
    files.append(store)
    materialize(manifest, tmp_path / 'store', public_base=base, store_only=True)
    assert [p.name for p in (tmp_path / 'store').iterdir()] == [store['path']]

    from scripts.bundles import release_artifacts
    from scripts.releases import stable
    raw_manifest = json.dumps(manifest).encode()
    seen = []

    def read_remote(url, expected_hash, *, expected_origin):
        seen.append((url, expected_hash, expected_origin))
        return stable.read_manifest(url, expected_hash, expected_origin=expected_origin,
                                    opener=lambda *args, **kwargs: Response(raw_manifest))

    monkeypatch.setattr(release_artifacts, 'read_manifest', read_remote)
    monkeypatch.setenv('CANDIDATE_MANIFEST_SHA256', hashlib.sha256(raw_manifest).hexdigest())
    release_artifacts.main(['materialize', '--tag', tag, '--commit', commit, '--public-base', base,
                            '--root', str(tmp_path / 'remote-store'), '--store-only'])
    assert (tmp_path / 'remote-store' / store['path']).read_bytes() == data
    assert seen[0][0] == f'{base}/releases/tag/{tag}/release-candidates.json'
    assert seen[0][2] == base
    monkeypatch.setenv('CANDIDATE_MANIFEST_SHA256', 'f' * 64)
    with pytest.raises(ValueError, match='digest mismatch'):
        release_artifacts.main(['materialize', '--tag', tag, '--commit', commit, '--public-base', base,
                                '--root', str(tmp_path / 'wrong-manifest'), '--store-only'])
    assert not (tmp_path / 'wrong-manifest').exists()
    files[0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='receipts differ'):
        materialize(manifest, tmp_path / 'bad', public_base=base)
    files[0]['sha256'] = digest
    monkeypatch.setattr('urllib.request.urlopen', lambda *a, **kw: Response(b'changed bytes'))
    with pytest.raises(ValueError, match='digest mismatch'):
        materialize(manifest, tmp_path / 'corrupt', public_base=base)
