"""Native metadata and artifact publication use the same verified bytes."""
import hashlib
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import hermes_yaml
import pytest

from scripts.bundles.release_artifacts import materialize, record, stamp_matches
from tests.scripts.test_release_r2 import r2_server  # noqa: F401
from tests.scripts.test_stable_release import https_origin  # noqa: F401
from tests.scripts.test_release_darwin import _inputs
from scripts.bundles import release_artifacts as artifacts

ROOT = Path(__file__).resolve().parents[2]
SMOKE_RESULTS = {name: {'result': 'success'} for name in (
    'smoke-darwin-arm64', 'smoke-darwin-x64', 'smoke-win32-arm64', 'smoke-win32-x64')}
RELEASE_EPOCH = 1_787_965_323
WINDOWS_VERSION = '2026.5761.123.0'


def test_windows_metadata_is_read_from_package_and_stale_stamp_is_rejected(tmp_path):
    tag, commit = 'v1.2.3', 'a' * 40
    root = tmp_path / 'release'
    root.mkdir()
    package = root / 'Product-1.2.3-win-x64.msix'
    manifest = '<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"><Identity Name="Product" Publisher="CN=Test" Version="1.2.3.0" ProcessorArchitecture="x64"/><Applications><Application Id="App"/></Applications></Package>'

    def write_package(sha, receiver=None):
        with zipfile.ZipFile(package, 'w') as archive:
            archive.writestr('AppxManifest.xml', manifest)
            archive.writestr('app/resources/install-stamp.json', json.dumps({
                'tag': tag, 'commit': sha, 'baseVersion': tag[1:],
                'receiverProtocol': receiver,
            }))

    write_package(commit)
    out = root / 'metadata-windows-x64.json'
    original = package.read_bytes()
    record('windows', 'x64', root, tag, commit, out)
    metadata = json.loads(out.read_text(encoding='utf-8-sig'))
    assert metadata['identity'] == 'Product'
    assert metadata['version'] == '1.2.3.0'
    assert metadata['publisher'] == 'CN=Test'
    assert metadata['applicationId'] == 'App'
    assert package.read_bytes() == original
    assert 'receiverProtocol' not in metadata
    write_package(commit, receiver=1)
    record('windows', 'x64', root, tag, commit, out)
    assert json.loads(out.read_text())['receiverProtocol'] == 1
    write_package('b' * 40)
    with pytest.raises(ValueError, match='provenance'):
        record('windows', 'x64', root, tag, commit, tmp_path / 'bad.json')
    with pytest.raises(ValueError, match='provenance'):
        stamp_matches({}, tag, commit)


@pytest.fixture
def staged_candidate(tmp_path, r2_server, https_origin):
    from scripts.bundles.release_artifacts import assemble
    from scripts.releases import handoff

    tag, commit, base = 'v1.2.3', 'a' * 40, https_origin.base
    attempt = 'rc.2-v1.2.3'
    https_origin.store = r2_server.store
    legs, mac_bytes = _inputs('1.2.3')
    built = tmp_path / 'built'
    built.mkdir()
    for platform, arches in [('windows', ('x64', 'arm64')), ('macos', ('x64', 'arm64')), ('termux', ('aarch64',))]:
        for arch in arches:
            row = {
                'platform': platform, 'arch': arch, 'tag': tag, 'commit': commit,
                'baseVersion': tag[1:], 'identity': 'Product',
            }
            if platform == 'windows':
                row.update(version=WINDOWS_VERSION, executableVersion=WINDOWS_VERSION,
                           publisher='CN=Test', applicationId='App')
                package = f'HermesBundled-1.2.3-win-{arch}.msix'
                handoff_name = f'win32-{arch}'
            elif platform == 'macos':
                package = f'HermesBundled-1.2.3-mac-{arch}.zip'
                row.update(version='1.2.3', teamId='ABCDEFGHIJ', filename=package)
                handoff_name = f'darwin-{arch}'
            else:
                package = 'deb/product.deb'
                row.update(version='1.2.3-1', filename=package)
                handoff_name = 'termux'
            file = built / package
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(mac_bytes.get(f'releases/tag/{tag}/{package}', b'package transport fixture'))
            metadata = built / f'metadata-{platform}-{arch}.json'
            metadata.write_text(json.dumps(row), encoding='utf-8')
            includes = [package, metadata.name]
            if platform == 'macos':
                feed = f'{arch}-stable-mac.yml'
                (built / feed).write_text(legs[feed], encoding='utf-8')
                includes.append(feed)
            if platform == 'termux':
                for name in ('pool/package.deb', 'dists/hermes-stable/InRelease', 'dists/hermes-stable/Release'):
                    path = built / 'apt' / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f'index transport fixture: {name}'.encode())
                includes.append('apt/**/*')
            handoff.stage(attempt, commit, handoff_name, built, includes)
    bundle = built / 'Product-win.msixbundle'
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('AppxMetadata/AppxBundleManifest.xml', f'<Bundle><Identity Name="Product" Publisher="CN=Test" Version="{WINDOWS_VERSION}"/><Packages><Package Type="application" Architecture="arm64"/><Package Type="application" Architecture="x64"/></Packages></Bundle>')
    (built / 'Store-Product-win.msixbundle').write_bytes(b'Store bundle transport fixture')
    handoff.stage(attempt, commit, 'windows-universal', built, ['*.msixbundle'])
    fetched = tmp_path / 'fetched'
    names = ['win32-x64', 'win32-arm64', 'darwin-x64', 'darwin-arm64', 'termux', 'windows-universal']
    handoff.fetch(attempt, commit, names, fetched, ['metadata-*.json', '*.msixbundle'])
    r2_server.requests.clear()
    manifest = assemble(fetched, tag, commit, base, tmp_path / 'release-candidates.json',
                        smoke_results=SMOKE_RESULTS, release_epoch=RELEASE_EPOCH, archive=attempt)
    assert manifest['archive'] == attempt and manifest['tag'] == tag
    assert {row['platform'] + '/' + row['arch'] for row in manifest['packages']} == {
        'windows/x64', 'windows/arm64', 'macos/x64', 'macos/arm64', 'termux/aarch64'}
    assert all(not file['path'].startswith(('handoff-', 'metadata-')) for file in manifest['files'])
    assert all(file['url'].startswith(f'{base}/releases/tag/{attempt}/') for file in manifest['files'])
    puts = [path for method, path, _ in r2_server.requests if method == 'PUT']
    assert puts == [f'/hermes-releases/releases/tag/{attempt}/release-candidates.json']
    assert all(key.startswith(f'releases/tag/{attempt}/') for key in r2_server.store)
    assert not any(key.startswith(f'releases/tag/{tag}/') for key in r2_server.store)
    return manifest, fetched, base


def test_bootstrap_reuses_published_candidate_and_rejects_substitution(staged_candidate, r2_server, monkeypatch):
    from scripts.releases import channel_releases
    candidate, _, base = staged_candidate
    request = {'releaseTag': candidate['tag'], 'commit': candidate['commit'], 'publicBase': base,
               'identity': {'token': 'a' * 16, 'appNamePascal': 'App'}}
    packages = []
    for row in candidate['packages']:
        if row['platform'] == 'termux':
            continue
        packages.append({**row, 'platform': 'darwin' if row['platform'] == 'macos' else 'win32',
                         'artifact': {'key': row['artifact']['url'].removeprefix(base + '/'),
                                      'sha256': row['artifact']['sha256']}})
    manifest = {'request': request, 'packages': packages}
    monkeypatch.setattr(channel_releases, 'product_identity', lambda tag: request['identity'])
    published = {'draft': False, 'prerelease': False, 'published_at': 'fixture-published'}
    monkeypatch.setattr(channel_releases.stable, 'output', lambda args: candidate['commit']
                        if '/commits/' in args[2] else json.dumps(published))
    r2_server.store['releases/stable/release-candidates.json'] = r2_server.store[f"releases/tag/{candidate['archive']}/release-candidates.json"]
    assert channel_releases.verify_bootstrap(request, manifest, base, 'fixture/repo')
    substituted = copy.deepcopy(manifest)
    substituted['packages'][0]['artifact']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='accepted'):
        channel_releases.verify_bootstrap(request, substituted, base, 'fixture/repo')
    published['draft'] = True
    with pytest.raises(ValueError, match='published'):
        channel_releases.verify_bootstrap(request, manifest, base, 'fixture/repo')


def test_assemble_rejects_missing_and_changed_receipts(tmp_path, staged_candidate):
    from scripts.bundles.release_artifacts import assemble

    manifest, fetched, base = staged_candidate
    tag, commit = manifest['tag'], manifest['commit']
    receipt = fetched / 'handoff-darwin-arm64.json'
    original = receipt.read_bytes()
    receipt.unlink()
    with pytest.raises(ValueError, match='handoff'):
        assemble(fetched, tag, commit, base, tmp_path / 'missing.json',
                 smoke_results=SMOKE_RESULTS, release_epoch=RELEASE_EPOCH, archive=manifest['archive'])
    receipt.write_bytes(original)
    (fetched / 'metadata-windows-x64.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='digest'):
        assemble(fetched, tag, commit, base, tmp_path / 'changed.json',
                 smoke_results=SMOKE_RESULTS, release_epoch=RELEASE_EPOCH, archive=manifest['archive'])


def test_candidate_publication_and_store_selection(tmp_path, monkeypatch, r2_server, staged_candidate):
    manifest, _, base = staged_candidate
    tag, commit = manifest['tag'], manifest['commit']
    raw = r2_server.store[f"releases/tag/{manifest['archive']}/release-candidates.json"][0]
    args = ['--tag', tag, '--archive', manifest['archive'], '--commit', commit, '--public-base', base]
    monkeypatch.setenv('CANDIDATE_MANIFEST_SHA256', hashlib.sha256(raw).hexdigest())
    artifacts.main(['materialize', *args, '--root', str(tmp_path / 'store'), '--store-only'])
    assert [p.name for p in (tmp_path / 'store').iterdir()] == ['Store-Product-win.msixbundle']
    assert (tmp_path / 'store/Store-Product-win.msixbundle').read_bytes() == b'Store bundle transport fixture'
    monkeypatch.setenv('CANDIDATE_MANIFEST_SHA256', 'f' * 64)
    with pytest.raises(ValueError, match='digest mismatch'):
        artifacts.main(['materialize', *args, '--root', str(tmp_path / 'wrong')])
    assert not (tmp_path / 'wrong').exists()
    missing = copy.deepcopy(manifest)
    missing['files'] = [f for f in missing['files'] if not f['path'].startswith('Store-')]
    with pytest.raises(ValueError, match='one Store candidate'):
        materialize(missing, tmp_path / 'missing-store', public_base=base, store_only=True)
    changed = copy.deepcopy(manifest)
    changed['files'][0]['sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='receipts differ'):
        materialize(changed, tmp_path / 'bad', public_base=base)

    r2_server.requests.clear()
    artifacts.publish(manifest, tmp_path / 'publish', base)
    assert [path for method, path, _ in r2_server.requests if method == 'PUT'] == [
        '/hermes-releases/releases/termux/stable/pool/package.deb']
    r2_server.requests.clear()
    artifacts.promote(manifest, tmp_path / 'promote', base)
    puts = [path for method, path, _ in r2_server.requests if method == 'PUT']
    assert puts == ['/hermes-releases/' + name for name in (
        'releases/darwin/stable/stable-mac.yml', 'releases/win32/stable/stable.appinstaller',
        'releases/termux/stable/dists/hermes-stable/Release', 'releases/termux/stable/dists/hermes-stable/InRelease')]
    for item in manifest['files']:
        assert (tmp_path / 'promote' / item['path']).read_bytes() == r2_server.store[f"releases/tag/{manifest['archive']}/{item['path']}"][0]
    descriptor = ET.fromstring(r2_server.store['releases/win32/stable/stable.appinstaller'][0])
    assert descriptor.attrib == {
        'Uri': base + '/releases/win32/stable/stable.appinstaller',
        'Version': WINDOWS_VERSION,
    }
    assert descriptor.find('{*}MainBundle').attrib == {
        'Name': 'Product', 'Publisher': 'CN=Test', 'Version': WINDOWS_VERSION,
        'Uri': base + f"/releases/tag/{manifest['archive']}/Product-win.msixbundle"}
    pointer = 'releases/win32/stable/stable.appinstaller'
    original = r2_server.store[pointer]
    r2_server.corrupt_put = pointer
    r2_server.requests.clear()
    with pytest.raises(ValueError, match='Channel read-back differs'):
        artifacts.promote(manifest, tmp_path / 'bad-readback', base)
    assert [path for method, path, _ in r2_server.requests if method == 'PUT'] == ['/hermes-releases/' + pointer]
    r2_server.corrupt_put = None
    r2_server.store[pointer] = original
    before = dict(r2_server.store)
    r2_server.store[f"releases/tag/{manifest['archive']}/Product-win.msixbundle"] = (b'corrupt', '"e"')
    r2_server.requests.clear()
    with pytest.raises(ValueError, match='digest mismatch'):
        artifacts.promote(manifest, tmp_path / 'broken', base)
    assert not any(method == 'PUT' for method, _, _ in r2_server.requests)
    assert all(value == r2_server.store[key] for key, value in before.items() if not key.startswith('releases/tag/'))


def test_promote_writes_the_stable_mac_feed_from_the_attempt_archive(tmp_path, monkeypatch, r2_server, https_origin, staged_candidate):
    import urllib.request

    manifest, _, base = staged_candidate
    # urlopen caches one global opener; use the fixture's per-test context so
    # this test is order-independent against the per-test self-signed CA.
    monkeypatch.setattr(urllib.request, 'urlopen', https_origin.opener)
    artifacts.promote(manifest, tmp_path / 'promote', base)
    feed = hermes_yaml.safe_load(r2_server.store['releases/darwin/stable/stable-mac.yml'][0].decode())
    # The feed version is the plain package version, never the attempt ref.
    assert feed['version'] == '1.2.3'
    assert feed['version'] != manifest['archive']
    urls = [entry['url'] for entry in feed['files']]
    assert urls and all(url.startswith(f"/releases/tag/{manifest['archive']}/") for url in urls)
    assert f"/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-mac-arm64.zip" in urls
    assert f"/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-mac-x64.zip" in urls


def test_promote_refuses_when_one_macos_arch_is_missing(tmp_path, monkeypatch, r2_server, https_origin, staged_candidate):
    import urllib.request

    manifest, _, base = staged_candidate
    monkeypatch.setattr(urllib.request, 'urlopen', https_origin.opener)
    broken = copy.deepcopy(manifest)
    broken['files'] = [f for f in broken['files'] if f['path'] != 'x64-stable-mac.yml']
    r2_server.requests.clear()
    with pytest.raises(ValueError, match='one ARM64 and one x64 macOS feed'):
        artifacts.promote(broken, tmp_path / 'broken-arch', base)
    # The refusal happens before any channel pointer moves.
    assert not [path for method, path, _ in r2_server.requests
                if method == 'PUT' and '/releases/win32/stable/' in path]


@pytest.fixture
def candidate_workflow_step(tmp_path, r2_server, staged_candidate):
    """Run real workflow shell/CLIs; replace only service endpoints and tool setup."""
    manifest, fetched, base = staged_candidate
    jobs = hermes_yaml.safe_load((ROOT / '.github/workflows/desktop-bundled-release.yml').read_text(encoding='utf-8-sig'))['jobs']
    stable_jobs = hermes_yaml.safe_load(
        (ROOT / '.github/workflows/stable-release.yml').read_text(encoding='utf-8-sig'))['jobs']
    render = next(step for step in stable_jobs['complete']['steps']
                  if step.get('name', '').startswith('Render the admitted'))
    jobs['controller-promote'] = {
        'env': jobs['stable-publish']['env'],
        'steps': [
            {'run': 'python -m scripts.bundles.release_artifacts promote --root verified'},
            render,
        ],
    }
    shutil.copytree(fetched, tmp_path / 'candidates')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    driver = bin_dir / 'python-driver.py'
    driver.write_text(
        'import runpy,sys\n'
        f'sys.path.insert(0, {str(ROOT)!r})\n'
        'from scripts.releases import r2\n'
        f'r2.s3_endpoint=lambda _: "http://127.0.0.1:{r2_server.server_port}"\n'
        'args=sys.argv[1:]\n'
        'if args[:2] == ["-m", "scripts.ci.python_packages"]:\n'
        '    import ruamel.yaml  # already provided by the test environment\n'
        '    args=args[args.index("--")+1:]\n'
        'sys.argv=args[1:] if args[0] == "-m" else args\n'
        'if args[0] == "-m":\n'
        '    runpy.run_module(args[1],run_name="__main__")\n'
        'else:\n'
        f'    runpy.run_path(str({str(ROOT)!r}+"/"+args[0]),run_name="__main__")\n',
        encoding='utf-8')
    body_file = tmp_path / 'release-body'
    body_file.write_text('<!-- HERMES_BUILDS_TABLE -->', encoding='utf-8')
    gh = bin_dir / 'gh'
    gh.write_text(
        f'#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n'
        f'body=Path({str(body_file)!r})\n'
        'if sys.argv[1:3] == ["release", "view"]:\n'
        '    print(json.dumps({"body":body.read_text(encoding="utf-8-sig")}))\n'
        'else:\n'
        '    assert sys.argv[1:3] == ["release", "edit"], sys.argv\n'
        '    body.write_text(sys.stdin.read(), encoding="utf-8")\n', encoding='utf-8')
    gh.chmod(0o755)
    for name in ('python', 'python3'):
        command = bin_dir / name
        command.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n', encoding='utf-8')
        command.chmod(0o755)
    digest = artifacts.sha256_file(tmp_path / 'release-candidates.json')

    def run(job_name, step, needs, *, pinned=digest, ambient_needs=None):
        expressions = {
            '${{ inputs.tag }}': manifest['tag'], '${{ inputs.claim-tag }}': manifest['archive'],
            '${{ inputs.manifest-sha256 }}': pinned,
            '${{ needs.validate.outputs.sha }}': manifest['commit'], '${{ github.token }}': 'inert',
            '${{ needs.validate.outputs.release-epoch }}': str(manifest['releaseEpoch']),
            '${{ needs.admit.outputs.tag }}': manifest['tag'],
            '${{ needs.admit.outputs.claim-tag }}': manifest['archive'],
            '${{ needs.admit.outputs.commit }}': manifest['commit'],
            '${{ needs.validate.outputs.archive-tag }}': manifest['archive'],
            '${{ needs.candidates.outputs.manifest-sha256 }}': pinned,
            '${{ needs.candidate-manifest.outputs.manifest-sha256 }}': pinned,
            '${{ vars.CLOUDFLARE_R2_PUBLIC_URL }}': base, '${{ toJSON(needs) }}': json.dumps(needs),
        }
        for key in ('CLOUDFLARE_R2_ACCOUNT_ID', 'CLOUDFLARE_R2_ACCESS_KEY_ID', 'CLOUDFLARE_R2_SECRET_ACCESS_KEY', 'CLOUDFLARE_R2_BUCKET'):
            expressions['${{ ' + ('vars.' if key.endswith('_BUCKET') else 'secrets.') + key + ' }}'] = os.environ[key]
        env = {**os.environ, 'GITHUB_SHA': manifest['commit'], 'GITHUB_REPOSITORY': 'fixture/release',
               'RELEASE_CLAIM_TAG': manifest['archive'],
               'PATH': str(bin_dir) + os.pathsep + os.environ['PATH'], 'GITHUB_OUTPUT': str(tmp_path / 'outputs')}
        env.pop('RELEASE_NEEDS', None)
        if ambient_needs is not None:
            env['RELEASE_NEEDS'] = json.dumps(ambient_needs)
        for key, value in {**jobs[job_name].get('env', {}), **step.get('env', {})}.items():
            env[key] = expressions.get(value, value)
            assert '${{' not in env[key], (key, value)
        return subprocess.run(['bash', '-e', '-o', 'pipefail', '-c', step['run']], cwd=tmp_path,
                              env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)

    return jobs, run, body_file


@pytest.mark.parametrize('ambient_needs', [None, {job: {'result': 'skipped'} for job in SMOKE_RESULTS}])
@pytest.mark.platforms('posix')
def test_candidate_smoke_survives_real_promotion_and_renderer(tmp_path, r2_server, staged_candidate,
                                                           candidate_workflow_step, ambient_needs, https_origin):
    manifest, _, base = staged_candidate
    jobs, run, body_file = candidate_workflow_step
    stored = json.loads(r2_server.store[f"releases/tag/{manifest['archive']}/release-candidates.json"][0])
    assert stored['smoke_results'] == SMOKE_RESULTS
    # An unrelated orphan object must not acquire the candidate's Passed label.
    orphan = f"releases/tag/{manifest['archive']}/HermesBundled-1.2.3-linux-x64.AppImage"
    r2_server.store[orphan] = (b'orphan transport fixture', '"e"')
    for step in jobs['controller-promote']['steps']:
        if 'run' in step:
            result = run('controller-promote', step, {'validate': {'result': 'success'}},
                         ambient_needs=ambient_needs)
            assert result.returncode == 0, result.stdout + result.stderr
    assert 'releases/win32/stable/stable.appinstaller' in r2_server.store
    page = r2_server.store['releases/stable/index.html'][0].decode()
    for output in (page, body_file.read_text(encoding='utf-8-sig')):
        assert output.count('Passed') == len(SMOKE_RESULTS)
        assert 'Not run' not in output and 'Build incomplete' not in output
        assert orphan not in output
        assert base + f"/releases/tag/{manifest['archive']}/HermesBundled-1.2.3-win-x64.msix" in output
    with https_origin.opener(base + '/releases/stable/index.html', timeout=5) as response:
        assert response.read().decode() == page


@pytest.mark.platforms('posix')
def test_candidate_smoke_admission_fails_before_publication(tmp_path, r2_server, staged_candidate, candidate_workflow_step):
    manifest, _, _ = staged_candidate
    jobs, run, body_file = candidate_workflow_step
    key = f"releases/tag/{manifest['archive']}/release-candidates.json"
    raw = r2_server.store[key][0]
    for fault, message in [('legacy', 'Candidate manifest'), ('missing', 'Candidate smoke results'),
                           ('failed', 'smoke-win32-x64=failure'), ('identity', 'release identity'),
                           ('tampered', 'digest mismatch')]:
        invalid = copy.deepcopy(manifest)
        if fault == 'legacy':
            invalid['schema'] = 1
            invalid.pop('smoke_results', None)
        elif fault == 'missing':
            invalid.pop('smoke_results', None)
        elif fault == 'failed':
            invalid['smoke_results'] = {**SMOKE_RESULTS, 'smoke-win32-x64': {'result': 'failure'}}
        else:
            invalid['commit'] = 'b' * 40
        data = json.dumps(invalid).encode()
        r2_server.store[key] = (data, '"fault"')
        digest = hashlib.sha256(raw if fault == 'tampered' else data).hexdigest()
        before = dict(r2_server.store), body_file.read_bytes()
        # Exercise both commands independently, not just shell short-circuiting.
        for step in jobs['controller-promote']['steps']:
            if 'run' not in step:
                continue
            r2_server.requests.clear()
            result = run('controller-promote', step, {}, pinned=digest)
            assert result.returncode != 0, (fault, step, result.stdout, result.stderr)
            assert message in result.stderr, result.stdout + result.stderr
            assert not any(method == 'PUT' for method, _, _ in r2_server.requests)
            assert (dict(r2_server.store), body_file.read_bytes()) == before
    r2_server.store[key] = (raw, '"original"')
