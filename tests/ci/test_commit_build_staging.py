"""Execute commit staging and summary steps against a disposable object store."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.request import urlopen

from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow
from tests.scripts.test_release_r2 import r2_server  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]


def step_script(job, name):
    return next(step['run'] for step in _workflow()['jobs'][job]['steps'] if step.get('name') == name)


def shell_step(tmp_path, r2_server, job, name, env):
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
        'args[:1] == ["scripts/render-builds-table.py"] or args == ["-"], args\n'
        'if args[:1] == ["-m"]:\n'
        '    sys.argv=args[1:]\n'
        '    runpy.run_module(args[1],run_name="__main__")\n'
        'elif args == ["-"]:\n'
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
    script = tmp_path / 'step.sh'
    script.write_text(step_script(job, name), encoding='utf-8', newline='\n')
    environment = _child_env(**env)
    environment['PATH'] = str(helper) + os.pathsep + environment['PATH']
    return subprocess.run([_BASH, '-e', '-o', 'pipefail', str(script)], cwd=tmp_path,
                          env=environment, capture_output=True, text=True, encoding='utf-8', timeout=60)


def test_commit_staging_and_summary_bind_every_produced_file_without_channels(tmp_path, r2_server):
    sha = 'a' * 40
    base = f'http://127.0.0.1:{r2_server.server_port}/hermes-releases'
    env = dict(HERMES_BUILD_COMMIT=sha, HERMES_PAYLOAD_TAG='', RELEASE_COMMIT=sha,
               RELEASE_PHASE='', GITHUB_SHA='b' * 40, CLOUDFLARE_R2_PUBLIC_URL=base,
               CLOUDFLARE_R2_ACCOUNT_ID='loopback', CLOUDFLARE_R2_ACCESS_KEY_ID='test-inert',
               CLOUDFLARE_R2_SECRET_ACCESS_KEY='test-inert', CLOUDFLARE_R2_BUCKET='hermes-releases')
    release = tmp_path / 'apps/desktop/release'
    release.mkdir(parents=True)
    producers = [
        ('build-win32', 'Stage Windows packages to R2', 'win32-x64', [
            'HermesBundled-0.33.0-win-x64.msix']),
        ('build-win32', 'Stage Windows packages to R2', 'win32-arm64', [
            'HermesBundled-0.33.0-win-arm64.msix']),
        ('build-darwin', 'Stage macOS packages and feed inputs to R2', 'darwin-arm64', [
            'HermesBundled-0.33.0-mac-arm64.dmg', 'HermesBundled-0.33.0-mac-arm64.zip',
            'HermesBundled-0.33.0-mac-arm64.zip.blockmap']),
        ('build-darwin', 'Stage macOS packages and feed inputs to R2', 'darwin-x64', [
            'HermesBundled-0.33.0-mac-x64.dmg', 'HermesBundled-0.33.0-mac-x64.zip',
            'HermesBundled-0.33.0-mac-x64.zip.blockmap']),
        ('publish-win32-updater', 'Stage universal bundles to R2', 'windows-universal', [
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
    deb = tmp_path / 'termux-build/deb/hermes-agent_0.33.0~commit.aaaaaaaaaaaa_aarch64.deb'
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
        'build-darwin': {'result': 'success'}, 'publish-win32-updater': {'result': 'success'},
        'termux-deb': {'result': 'success'}, 'build-linux': {'result': 'success'},
    })}
    result = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                        'Render the full expected-binary matrix', summary_env)
    assert result.returncode == 0, result.stdout + result.stderr
    text = summary.read_text(encoding='utf-8')
    links = re.findall(r'\]\((http[^)]+)\)', text)
    # Blockmaps are receipt inputs; every other staged product has a download row.
    expected = {f'{base}/{key}' for key in artifact_keys if not key.endswith('.blockmap')}
    assert set(links) == expected
    assert len(links) == len(expected)
    for url in links:
        with urlopen(url, timeout=5) as response:
            key = url.removeprefix(base + '/')
            assert response.read() == r2_server.store[key][0]
    assert all(key.startswith(f'releases/commit/{sha}/') for key in r2_server.store)
    assert not any(method == 'DELETE' for method, _, _ in r2_server.requests)

    # The actual summary command remains useful after an admitted matrix failure.
    r2_server.store.pop(f'releases/commit/{sha}/handoff-darwin-x64.json')
    summary.unlink()
    summary_env['RELEASE_NEEDS'] = json.dumps({'validate': {'result': 'success'},
                                             'build-darwin': {'result': 'failure'}})
    incomplete = shell_step(tmp_path, r2_server, 'commit-builds-summary',
                            'Render the full expected-binary matrix', summary_env)
    assert incomplete.returncode == 0, incomplete.stdout + incomplete.stderr
    assert 'failed: build-darwin' in summary.read_text(encoding='utf-8')
