"""Execute commit staging and summary steps against a disposable object store."""
import html
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.request import urlopen
from urllib.parse import quote, unquote

import pytest

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]


def step_script(job, name):
    return next(step['run'] for step in _workflow()['jobs'][job]['steps'] if step.get('name') == name)


def shell_step(tmp_path, r2_server, job, name, env, *, script=None):
    helper = tmp_path / 'bin'
    helper.mkdir(exist_ok=True)
    driver = helper / 'python-driver.py'
    driver.write_text(
        'import runpy,sys\n'
        f'sys.path.insert(0, {str(ROOT)!r})\n'
        'from scripts.releases import r2\n'
        f'r2.s3_endpoint=lambda _: "http://127.0.0.1:{r2_server.server_port}"\n'
        'args=sys.argv[1:]\n'
        'assert args[:2] == ["-m", "scripts.releases.handoff"] or '
        'args[:1] in (["scripts/render-builds-table.py"], ["-"]), args\n'
        'if args[:1] == ["-m"]:\n'
        '    sys.argv=args[1:]\n'
        '    runpy.run_module(args[1],run_name="__main__")\n'
        'elif args[:1] == ["-"]:\n'
        '    sys.argv=args\n'
        '    exec(compile(sys.stdin.read(), "workflow-inline", "exec"))\n'
        'else:\n'
        '    sys.argv=args\n'
        f'    runpy.run_path({str(ROOT / "scripts/render-builds-table.py")!r},run_name="__main__")\n',
        encoding='utf-8',
    )
    for name_ in ['python', 'python3']:
        command = helper / name_
        command.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(driver))} "$@"\n',
                           encoding='utf-8', newline='\n')
        command.chmod(0o755)
    script_file = tmp_path / 'step.sh'
    script_file.write_text(script if script is not None else step_script(job, name), encoding='utf-8', newline='\n')
    environment = _child_env(**env)
    environment['PATH'] = str(helper) + os.pathsep + environment['PATH']
    return subprocess.run([_BASH, '-e', '-o', 'pipefail', str(script_file)], cwd=tmp_path,
                          env=environment, capture_output=True, text=True, encoding='utf-8', timeout=60)


@pytest.mark.parametrize("has_download", [True, False])
def test_failed_commit_summary_publishes_downloads_or_run_links(tmp_path, r2_server, has_download):
    sha = 'a' * 40
    run_url = 'https://github.example/o/r/actions/runs/12345'
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    summary = tmp_path / 'summary.md'
    jobs = _workflow()['jobs']
    bundle_env = {'HERMES_HOME': None, 'EMPTY': '', 'LABEL': '<script>\n"café" & value</script>'}
    env = dict(HERMES_BUILD_COMMIT=sha, HERMES_PAYLOAD_TAG='', RELEASE_COMMIT=sha,
               GITHUB_REPOSITORY='fixture-owner/fixture-repo',
               HERMES_BUNDLE_ENV_JSON=json.dumps(bundle_env), CI_SECRET='must-not-appear',
               RELEASE_PHASE='', TARGET='win32-x64', RUN_URL=run_url,
               GITHUB_STEP_SUMMARY=str(summary), CLOUDFLARE_R2_PUBLIC_URL=base,
               CLOUDFLARE_R2_ACCOUNT_ID='loopback', CLOUDFLARE_R2_ACCESS_KEY_ID='test-inert',
               CLOUDFLARE_R2_SECRET_ACCESS_KEY='test-inert', CLOUDFLARE_R2_BUCKET='hermes-releases',
               RELEASE_NEEDS=json.dumps({name: {'result': 'success' if name == 'validate' else 'failure'}
                                         for name in jobs['commit-builds-summary']['needs']}))
    if has_download:
        artifact = tmp_path / 'apps/desktop/release/HermesBundled-0.33.0-win-x64.msix'
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b'inert downloadable fixture')
        staged = shell_step(tmp_path, r2_server, 'build-win32-commit', 'Stage Windows packages to R2', env)
        assert staged.returncode == 0, staged.stdout + staged.stderr
    result = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                        'Render the full expected-binary matrix', env)
    assert result.returncode == 0, result.stdout + result.stderr
    text = summary.read_text(encoding='utf-8-sig')
    page_key = f'releases/commit/{sha}/index.html'
    with urlopen(f'{base}/{page_key}', timeout=5) as response:
        page = response.read().decode()
    assert f'href="https://github.com/fixture-owner/fixture-repo/commit/{sha}"' in page
    assert 'Bundle environment' in page and 'HERMES_HOME' in page and 'Unset' in page
    assert '<code>EMPTY</code></td><td><code>&quot;&quot;</code>' in page
    assert html.escape(json.dumps(bundle_env['LABEL'], ensure_ascii=False)) in page
    assert '<script>' not in page and 'must-not-appear' not in page and 'CI_SECRET' not in page
    links = re.findall(r'\]\((https?://[^)]+)\)', text)
    assert run_url in links
    assert text.count('✅ Built') == page.count('✅ Built') == int(has_download)
    for url in links:
        assert f'href="{url}"' in page
        if url != run_url:
            assert url.startswith(base + '/')
            with urlopen(url, timeout=5) as response:
                assert response.read() == b'inert downloadable fixture'
    for line in text.splitlines():
        if 'Not built' in line:
            assert f'[View build run]({run_url})' in line and base not in line
        elif line.startswith('| Linux'):
            assert 'Disabled' in line and '](' not in line
    assert all(key.startswith(f'releases/commit/{sha}/') for key in r2_server.store)
    for name in ('build-win32', 'build-darwin'):
        execution = f'{name}-commit'
        assert execution in jobs[name]['needs']
        assert jobs[execution]['strategy']['fail-fast'] is False
    step = next(step for step in jobs['commit-builds-summary']['steps'] if 'run' in step)
    assert step['env']['RUN_URL'] == '${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}'
    assert step['env']['HERMES_BUNDLE_ENV_JSON'] == '${{ inputs.bundle_env }}'


def test_commit_staging_and_summary_bind_every_produced_file_without_channels(tmp_path, r2_server):
    sha = 'a' * 40
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    env = dict(HERMES_BUILD_COMMIT=sha, HERMES_PAYLOAD_TAG='', RELEASE_COMMIT=sha,
               GITHUB_REPOSITORY='o/r',
               RELEASE_PHASE='', GITHUB_SHA='b' * 40, CLOUDFLARE_R2_PUBLIC_URL=base,
               CLOUDFLARE_R2_ACCOUNT_ID='loopback', CLOUDFLARE_R2_ACCESS_KEY_ID='test-inert',
               CLOUDFLARE_R2_SECRET_ACCESS_KEY='test-inert', CLOUDFLARE_R2_BUCKET='hermes-releases')
    release = tmp_path / 'apps/desktop/release'
    release.mkdir(parents=True)
    producers = [
        ('build-win32-commit', 'Stage Windows packages to R2', 'win32-x64', [
            'HermesBundled-0.33.0-win-x64.msix']),
        ('build-win32-commit', 'Stage Windows packages to R2', 'win32-arm64', [
            'HermesBundled-0.33.0-win-arm64.msix']),
        ('build-darwin-commit', 'Stage macOS packages and feed inputs to R2', 'darwin-arm64', [
            'HermesBundled-0.33.0-mac-arm64.dmg', 'HermesBundled-0.33.0-mac-arm64.zip',
            'HermesBundled-0.33.0-mac-arm64.zip.blockmap']),
        ('build-darwin-commit', 'Stage macOS packages and feed inputs to R2', 'darwin-x64', [
            'HermesBundled-0.33.0-mac-x64.dmg', 'HermesBundled-0.33.0-mac-x64.zip',
            'HermesBundled-0.33.0-mac-x64.zip.blockmap']),
        ('assemble-win32-bundle', 'Stage universal bundles to R2', 'windows-universal', [
            'HermesBundled-0.33.0.0-win.msixbundle']),
    ]
    artifact_keys = set()
    for job, name, target, names in producers:
        for file in release.iterdir():
            file.unlink()
        for filename in names:
            (release / filename).write_bytes(f'transport fixture: {filename}'.encode())
        result = shell_step(tmp_path, r2_server, job, name, {**env, 'TARGET': target})
        assert result.returncode == 0, result.stdout + result.stderr
        receipt_key = f'releases/commit/{sha}/handoff-{target}.json'
        receipt = json.loads(r2_server.store[receipt_key][0])
        assert receipt['schema'] == 2 and receipt['commit'] == sha and 'tag' not in receipt
        assert {row['path'] for row in receipt['files']} == set(names)
        artifact_keys.update(f'releases/commit/{sha}/{filename}' for filename in names)
        puts = [path for method, path, _ in r2_server.requests if method == 'PUT']
        assert puts[-1].endswith(receipt_key)

    termux_name = 'Stage the commit-build deb to R2'
    before = dict(r2_server.store)
    missing = shell_step(tmp_path, r2_server, 'termux-deb', termux_name, env)
    assert missing.returncode != 0
    assert r2_server.store == before
    deb = tmp_path / 'termux-build/deb/hermes agent_0.33.0~commit.aaaaaaaaaaaa_aarch64.deb'
    deb.parent.mkdir(parents=True)
    deb.write_bytes(b'transport fixture, not a native Debian package')
    staged = shell_step(tmp_path, r2_server, 'termux-deb', termux_name, env)
    assert staged.returncode == 0, staged.stdout + staged.stderr
    artifact_keys.add(f'releases/commit/{sha}/deb/{deb.name}')
    receipt = json.loads(r2_server.store[f'releases/commit/{sha}/handoff-termux.json'][0])
    assert {row['path'] for row in receipt['files']} == {f'deb/{deb.name}'}

    summary = tmp_path / 'summary.md'
    summary_env = {**env, 'GITHUB_STEP_SUMMARY': str(summary), 'RELEASE_NEEDS': json.dumps({
        'validate': {'result': 'success'}, 'build-win32': {'result': 'success'},
        'build-darwin': {'result': 'success'}, 'assemble-win32-bundle': {'result': 'success'},
        'termux-deb': {'result': 'success'}, 'build-linux': {'result': 'success'},
    })}
    result = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                        'Render the full expected-binary matrix', summary_env)
    assert result.returncode == 0, result.stdout + result.stderr
    text = summary.read_text(encoding='utf-8-sig')
    links = re.findall(r'\]\((http[^)]+)\)', text)
    # Blockmaps are receipt inputs; every other staged product has a download row.
    expected = {f'{base}/{quote(key, safe="/")}' for key in artifact_keys if not key.endswith('.blockmap')}
    assert set(links) == expected
    assert len(links) == len(expected)
    for url in links:
        with urlopen(url, timeout=5) as response:
            key = unquote(url.removeprefix(base + '/'))
            assert response.read() == r2_server.store[key][0]
    page_key = f'releases/commit/{sha}/index.html'
    page = r2_server.store[page_key][0].decode()
    assert all(f'href="{url}"' in page for url in expected)
    assert 'Store-' not in page and 'Linux x64' in page and 'Linux ARM64' in page
    assert all(key.startswith(f'releases/commit/{sha}/') for key in r2_server.store)
    assert not any(method == 'DELETE' for method, _, _ in r2_server.requests)

    receipt_key = f'releases/commit/{sha}/handoff-win32-x64.json'
    original = r2_server.store[receipt_key]
    r2_server.store[receipt_key] = (b'not-json', '"invalid"')
    failed = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                        'Render the full expected-binary matrix', summary_env)
    assert failed.returncode != 0
    assert summary.read_text(encoding='utf-8-sig') == text
    assert r2_server.store[page_key][0].decode() == page
    r2_server.store[receipt_key] = original

    # The actual summary command remains useful after an admitted matrix failure.
    r2_server.store.pop(f'releases/commit/{sha}/handoff-darwin-x64.json')
    summary.unlink()
    summary_env['RELEASE_NEEDS'] = json.dumps({'validate': {'result': 'success'},
                                             'build-darwin': {'result': 'failure'}})
    incomplete = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                            'Render the full expected-binary matrix', summary_env)
    assert incomplete.returncode == 0, incomplete.stdout + incomplete.stderr
    assert 'failed: build-darwin' in summary.read_text(encoding='utf-8-sig')
