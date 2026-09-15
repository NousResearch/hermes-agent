"""Tag pages and release bodies publish the same independently named objects."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from urllib.request import urlopen

import pytest

from tests.scripts.test_release_r2 import r2_server  # noqa: F401

_SPEC = importlib.util.spec_from_file_location(
    'render_builds_table', Path(__file__).resolve().parents[2] / 'scripts/render-builds-table.py')
assert _SPEC and _SPEC.loader
rbt = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rbt)


@pytest.fixture
def release_body(monkeypatch):
    state = {'body': '# Notes\n\n<!-- HERMES_BUILDS_TABLE -->\n\n## Changes\n- x\n', 'edits': []}

    def gh(argv, **kwargs):
        if argv[:3] == ['gh', 'release', 'view']:
            return subprocess.CompletedProcess(argv, 0, json.dumps({'body': state['body']}), '')
        assert argv[:3] == ['gh', 'release', 'edit']
        state['body'] = kwargs['input']
        state['edits'].append(state['body'])
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(rbt.subprocess, 'run', gh)
    return state


@pytest.mark.parametrize('tag', ['v1.2.3', 'v1.2.3-canary.20260818101010'])
def test_tag_publication_pending_rerun_stale_and_dry_run(monkeypatch, r2_server, release_body, tag):
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    version = tag[1:]
    prefix = f'releases/tag/{tag}/'
    visible = [f'HermesBundled-{version}-mac-arm64.dmg', f'HermesBundled-{version}-win-x64.msix',
               f'HermesBundled-{version}-win-arm64.msix', f'HermesBundled-{version}-linux-x64.AppImage',
               f'HermesLight-{version}-win-x64.msix']
    hidden = [f'HermesBundled-{version}-mac-arm64.zip', f'HermesBundled-{version}-win.msixbundle',
              f'HermesBundled-{version}-win-x64.msix.blockmap', 'latest.yml',
              'HermesBundled-9.9.9-win-x64.msix']
    for name in visible + hidden:
        r2_server.store[prefix + name] = (name.encode(), '"e"')
    r2_server.store['releases/tag/v9.9.9/HermesBundled-9.9.9-win-x64.msix'] = (b'neighbor', '"e"')
    monkeypatch.setenv('RELEASE_NEEDS', '{"build-win32":{"result":"success"}}')
    args = ['render-builds-table.py', '--tag', tag, '--repo', 'o/r', '--r2-base-url', base + '/']

    def render(*extra):
        monkeypatch.setattr(sys, 'argv', [*args, *extra])
        assert rbt.main() == 0

    for run in ('1', '2'):
        render('--pending-run-url', f'https://github.example/runs/{run}')
    assert 'runs/1' not in release_body['body'] and release_body['body'].count('runs/2') == 1
    assert not any(method == 'PUT' for method, _, _ in r2_server.requests)
    render()
    body = release_body['body']
    assert 'runs/2' not in body and body.count('## Downloads') == 1
    assert body.count(rbt.MARKER) == body.count(rbt.END_MARKER) == 1
    expected = {f'{base}/{prefix}{name}' for name in visible}
    assert set(re.findall(r'\]\((http[^)]+)\)', body)) == expected
    with urlopen(f'{base}/{prefix}index.html', timeout=5) as response:
        page = response.read().decode()
    assert set(re.findall(r'href="(http[^"]+)"', page)) == expected | {f'https://github.com/o/r/releases/tag/{tag}'}
    assert rbt.recorded_build(page) == tag and 'Bundle environment' not in page
    assert all(name not in page and name not in body for name in hidden)
    assert 'linux-arm64' not in page
    for url in expected:
        with urlopen(url, timeout=5) as response:
            assert response.read() == url.rsplit('/', 1)[1].encode()
    channel_key = 'releases/canary/index.html' if '-canary.' in tag else 'releases/stable/index.html'
    assert r2_server.store[channel_key][0].decode() == page
    render()
    assert release_body['body'] == body

    # Use the real page writer's marker format, independent of asset selection.
    newer = rbt.render_page('v9.9.9', {}, base).encode()
    r2_server.store[channel_key] = (newer, '"e"')
    render()
    assert r2_server.store[channel_key][0] == newer
    assert r2_server.store[prefix + 'index.html'][0].decode() == page
    before, edits = dict(r2_server.store), list(release_body['edits'])
    render('--dry-run')
    assert r2_server.store == before and release_body['edits'] == edits
    release_body['body'] = 'no marker here'
    render()
    assert release_body['body'] == 'no marker here' and release_body['edits'] == edits


@pytest.mark.parametrize('asset_present', [False, True])
@pytest.mark.parametrize('result', ['failure', 'cancelled', 'skipped'])
def test_incomplete_tag_keeps_channel_and_links_diagnostics(monkeypatch, r2_server, release_body, asset_present, result):
    tag = 'v1.2.3-canary.20260818101010'
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    key = f'releases/tag/{tag}/HermesBundled-{tag[1:]}-win-x64.msix'
    r2_server.store['releases/canary/index.html'] = (b'previous good page', '"e"')
    if asset_present:
        r2_server.store[key] = (b'transport fixture', '"e"')
    run = 'https://github.example/runs/123'
    monkeypatch.setenv('RELEASE_NEEDS', json.dumps({'publish-win32-updater': {'result': result}}))
    monkeypatch.setattr(sys, 'argv', ['renderer', '--tag', tag, '--r2-base-url', base, '--run-url', run])
    assert rbt.main() == 0
    with urlopen(f'{base}/releases/tag/{tag}/index.html', timeout=5) as response:
        page = response.read().decode()
    for output in (page, release_body['body']):
        assert 'Build incomplete' in output and f'publish-win32-updater ({result})' in output
        assert run in output
        assert (base + '/' + key in output) == asset_present
    assert ('No downloadable artifacts' in page) == (not asset_present)
    assert r2_server.store['releases/canary/index.html'][0] == b'previous good page'


@pytest.mark.parametrize('version,name', [
    ('1.2.3', 'HermesBundled-1.2.3-win-x64.msix'),
    ('1.2.3-canary.20260818', 'HermesBundled-1.2.3-canary.20260818-win-x64.msix'),
])
def test_exact_version_and_flat_name_boundaries(version, name):
    names = [name, 'HermesBundled-1.2.3-canary.20260817-win-x64.msix', 'HermesBundled-1.2.4-win-x64.msix', name + '.blockmap']
    assert rbt.filter_names_for_version(names, version) == [name]
    assert rbt.parse_assets([name])['HermesBundled'][('win', 'x64')] == (name, 'msix')


@pytest.mark.parametrize('current,tag,allowed', [
    (None, 'v1.2.3', True), ('garbage', 'v1.2.3', True),
    ('v1.2.3', 'v1.2.3-canary.20260818101010', False),
    ('v1.2.3-canary.20260818101010', 'v1.2.3-canary.20260818101009', False),
    ('v1.2.3-canary.20260818101010', 'v1.2.3-canary.20260818101011', True),
])
def test_channel_order_boundaries(current, tag, allowed):
    page = rbt.render_page(current, {}, 'https://cdn.example') if current and current != 'garbage' else current
    assert rbt.supersedes(page, tag) is allowed
