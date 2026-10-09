"""Replay artifact handoffs and publication against a fake R2."""
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import hermes_yaml
import pytest

from tests.ci.desktop_release_roles import (
    CANARY_TAG as TAG, NATIVE_TARGETS, SHA, native_builds, needs_of, phase_result, selection_gates,
    smoke_callers, stage_step, termux_builder, universal_assembler, updater_publishers,
)
from tests.ci.test_commit_build_staging import ROOT, shell_step
from tests.ci.test_desktop_release_tag_admission import _child_env, _workflow
from tests.scripts.test_release_r2 import r2_server


def smoke_workflow():
    return hermes_yaml.safe_load((ROOT / '.github/workflows/desktop-bundle-smoke.yml').read_text(encoding='utf-8-sig'))


def smoke_fetch_script():
    """The public artifact fetch every smoke runner shares."""
    scripts = {step['run'] for job in smoke_workflow()['jobs'].values()
               for step in job.get('steps', []) if step.get('id') == 'artifact'}
    (script,) = scripts
    return script
















def transport_env(tmp_path, server, *, commit=False):
    return dict(HERMES_PAYLOAD_TAG='' if commit else TAG, HERMES_BUILD_COMMIT=SHA if commit else '',
                RELEASE_TAG='' if commit else TAG, RELEASE_COMMIT=SHA, COMMIT_BUILD=str(commit).lower(),
                PUBLIC_BASE=f'http://127.0.0.1:{server.server_port}/hermes-releases',
                CLOUDFLARE_R2_PUBLIC_URL=f'http://127.0.0.1:{server.server_port}/hermes-releases',
                CLOUDFLARE_R2_ACCOUNT_ID='loopback', CLOUDFLARE_R2_ACCESS_KEY_ID='test-inert',
                CLOUDFLARE_R2_SECRET_ACCESS_KEY='test-inert', CLOUDFLARE_R2_BUCKET='hermes-releases',
                RELEASE_PHASE='', SMOKE_ROOT=str(tmp_path / 'smoke'), GITHUB_OUTPUT=str(tmp_path / 'output'))


@pytest.mark.parametrize('commit', [False, True])
@pytest.mark.parametrize('platform,fmt', [('darwin', 'dmg'), ('darwin', 'zip'), ('win32', 'msix'), ('win32', 'msixbundle')])
@pytest.mark.parametrize('arch', ['arm64', 'x64'])
def test_public_smoke_fetches_the_receipt_bound_native_format(tmp_path, r2_server, commit, platform, fmt, arch):
    env = transport_env(tmp_path, r2_server, commit=commit)
    release = tmp_path / 'apps/desktop/release'
    release.mkdir(parents=True)
    suffix = f'mac-{arch}.{fmt}' if platform == 'darwin' else ('win.msixbundle' if fmt == 'msixbundle' else f'win-{arch}.msix')
    filename = f'HermesBundled-0.28.0-{suffix}'
    payload = b'transport fixture only: not a deployable package'
    (release / filename).write_bytes(payload)
    (release / ('Store-' + filename)).write_bytes(b'not eligible')
    universal = fmt == 'msixbundle'
    jobs = _workflow()['jobs']
    producer = universal_assembler(jobs) if universal else native_builds(jobs)[(f'{platform}-{arch}', 'commit')]
    if platform == 'darwin':
        # The producer stages all of its formats together; smoke selects one.
        for ext in ('dmg', 'zip', 'zip.blockmap'):
            (release / f'HermesBundled-0.28.0-mac-{arch}.{ext}').write_bytes(payload)
        (release / f'{arch}-canary-mac.yml').write_bytes(payload)
    staged = shell_step(tmp_path, r2_server, '', '', {**env, 'TARGET': f'{platform}-{arch}'},
                        script=stage_step(jobs[producer])['run'])
    assert staged.returncode == 0, staged.stdout + staged.stderr
    assert all('/canary/' not in key for key in r2_server.store)
    script = smoke_fetch_script()
    # Deliberately remove every storage credential before executing public fetch.
    public_env = {key: '' if key.startswith('CLOUDFLARE_') else value for key, value in env.items()}
    public_env.update(PLATFORM=platform, ARCH=arch, FORMAT=fmt)
    r2_server.requests.clear()
    result = shell_step(tmp_path, r2_server, '', '', public_env, script=script)
    assert result.returncode == 0, result.stdout + result.stderr
    selected = Path((tmp_path / 'output').read_text(encoding='utf-8-sig').strip().removeprefix('path='))
    assert selected.name == filename and selected.read_bytes() == payload
    witness = json.loads((tmp_path / 'smoke/out/download.json').read_text(encoding='utf-8-sig'))
    assert witness['commit'] == SHA and witness['artifact']['sha256'] == hashlib.sha256(payload).hexdigest()
    assert not any(file.name.startswith('Store-') for file in selected.parent.iterdir())
    assert all(method == 'GET' and 'Authorization' not in headers for method, _, headers in r2_server.requests)


@pytest.mark.parametrize('fault', ['missing', 'ambiguous', 'wrong-commit', 'corrupt'])
def test_download_faults_never_export_an_accepted_artifact(tmp_path, r2_server, fault):
    env = transport_env(tmp_path, r2_server, commit=True)
    filename = 'HermesBundled-0.28.0-win-x64.msix'
    payload = b'inert integrity fixture'
    row = {'path': filename, 'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
    receipt = {'schema': 2, 'commit': 'b' * 40 if fault == 'wrong-commit' else SHA,
               'name': 'win32-x64', 'files': [row]}
    prefix = f'releases/commit/{SHA}/'
    r2_server.store[prefix + filename] = (b'corrupted' if fault == 'corrupt' else payload, '"e"')
    if fault == 'missing':
        receipt['files'] = [{**row, 'path': 'Store-' + filename}]
    elif fault == 'ambiguous':
        second = 'HermesBundled-0.29.0-win-x64.msix'
        receipt['files'].append({**row, 'path': second})
        r2_server.store[prefix + second] = (payload, '"e"')
    r2_server.store[prefix + 'handoff-win32-x64.json'] = (json.dumps(receipt).encode(), '"e"')
    script = smoke_fetch_script()
    result = shell_step(tmp_path, r2_server, '', '', {**env, 'PLATFORM': 'win32', 'ARCH': 'x64', 'FORMAT': 'msix'}, script=script)
    assert result.returncode != 0
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'smoke/out/download.json').exists()


def test_stable_phase_result_requires_smoke_but_preserves_other_phases(tmp_path, r2_server):
    jobs = _workflow()['jobs']
    gates, smokes = selection_gates(jobs), smoke_callers(jobs)
    group_jobs = {target: [gates[target], smokes[target]] for target in NATIVE_TARGETS}
    group_jobs['win32-bundle'] = [universal_assembler(jobs)]
    group_jobs['termux'] = [termux_builder(jobs)]
    verdict = phase_result(jobs)
    members = {member for group in group_jobs.values() for member in group}
    # What the verdict needs beyond admission and the build groups is the
    # publication itself; the Store submission follows the file publication.
    (publish,) = [name for name in needs_of(jobs[verdict]) if name not in members | {'validate'}
                  and not set(needs_of(jobs[name])) - {'validate'}]
    (store,) = [name for name in needs_of(jobs[verdict]) if publish in needs_of(jobs[name])]

    def run_phase(phase, selected, failed=None, skip_tests=False):
        needs = {name: {'result': 'success'} for name in needs_of(jobs[verdict])}
        needs['validate']['outputs'] = {group: ('true' if group in selected else 'false')
                                        for group in group_jobs}
        for group, group_members in group_jobs.items():
            if group not in selected:
                for member in group_members:
                    needs[member]['result'] = 'skipped'
        if skip_tests:
            for smoke in smokes.values():
                needs[smoke]['result'] = 'skipped'
        if failed:
            needs[failed]['result'] = 'cancelled'
        (step,) = [step for step in jobs[verdict]['steps'] if 'SKIP_TESTS' in (step.get('env') or {})]
        return shell_step(tmp_path, r2_server, '', '',
                          {'RELEASE_NEEDS': json.dumps(needs), 'RELEASE_PHASE': phase,
                           'SKIP_TESTS': 'true' if skip_tests else 'false'}, script=step['run'])

    every = set(group_jobs)
    assert run_phase('candidate', every).returncode == 0
    # A claim that skipped tests runs no smoke, and every build still counts.
    assert run_phase('candidate', every, skip_tests=True).returncode == 0
    assert run_phase('candidate', every, failed=gates['win32-x64'], skip_tests=True).returncode != 0
    assert run_phase('publish', every).returncode == 0
    assert run_phase('publish', every - {'termux'}).returncode == 0
    for group in group_jobs:
        # A group that was not selected is not a failure of the phase.
        assert run_phase('candidate', every - {group}).returncode == 0, group
    for failed in (smokes['darwin-arm64'], gates['win32-x64'], termux_builder(jobs)):
        assert run_phase('candidate', every, failed=failed).returncode != 0, failed
    assert run_phase('candidate', every, failed=publish).returncode == 0
    # A partial selection still judges the groups it selected.
    assert run_phase('candidate', {'darwin-arm64'}).returncode == 0
    assert run_phase('candidate', {'darwin-arm64'}, failed=smokes['darwin-arm64']).returncode != 0
    assert run_phase('publish', set(), failed=store).returncode != 0



def test_canary_publisher_consumes_staged_bytes_and_writes_pointer_last(tmp_path, r2_server):
    tag = 'v0.28.1+canary.20260818T101010Z'
    env = {**transport_env(tmp_path, r2_server), 'HERMES_DESKTOP_VARIANT': 'bundled',
           'HERMES_PAYLOAD_TAG': tag, 'RELEASE_TAG': tag}
    # Use the real assembly identity derivation with a stable base available.
    for file in ['scripts/msix-shared.mjs', 'scripts/release-content-types.json',
                 'apps/desktop/product-identity.cjs', 'apps/desktop/package.json']:
        target = tmp_path / file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / file, target)

    def git(*args, date='2026-08-18T08:10:10Z'):
        subprocess.run(['git', *args], cwd=tmp_path, check=True, capture_output=True,
                       env=_child_env(GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
                                      GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid',
                                      GIT_AUTHOR_DATE=date, GIT_COMMITTER_DATE=date,
                                      GIT_CONFIG_GLOBAL=str(tmp_path / 'no-git-config'), GIT_CONFIG_NOSYSTEM='1'))

    def assembly_identity():
        result = subprocess.run(['node', '--input-type=module', '-e',
                                 'import {appIdentity} from "./scripts/msix-shared.mjs";'
                                 'console.log(JSON.stringify(appIdentity(process.cwd()+"/apps/desktop")));'],
                                cwd=tmp_path, env=_child_env(**env), check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    git('init', '-q')
    git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'stable base')
    git('tag', 'v0.28.0')
    assembled = assembly_identity()
    version = assembled['version']
    release = tmp_path / 'apps/desktop/release'
    release.mkdir(parents=True)
    filename = f"{assembled['name']}-{version}-win.msixbundle"
    bundle = release / filename
    with zipfile.ZipFile(bundle, 'w') as archive:
        archive.writestr('AppxMetadata/AppxBundleManifest.xml',
                         '<Bundle><Identity Name="NousResearch.HermesBundledCanary" '
                         'Publisher="CN=Nous Research Inc., O=Nous Research Inc., L=Austin, S=Texas, C=US" '
                         f'Version="{version}"/></Bundle>')
    tested_bytes = bundle.read_bytes()
    jobs = _workflow()['jobs']
    publisher = updater_publishers(jobs)['win32']
    staged = shell_step(tmp_path, r2_server, '', '', env, script=stage_step(jobs[universal_assembler(jobs)])['run'])
    assert staged.returncode == 0, staged.stdout + staged.stderr
    assert all(key.startswith(f'releases/tag/{tag}/') for key in r2_server.store)
    # A newer stable becomes visible after assembly but cannot alter the
    # timestamp-derived canary identity.
    git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'new stable', date='2026-08-18T11:10:10Z')
    git('tag', 'v0.28.1')
    assert assembly_identity()['version'] == version
    for name in ('Retrieve the tested universal bundle', 'Publish identical tested bytes without rebuilding'):
        result = shell_step(tmp_path, r2_server, publisher, name, env)
        assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'staged' / filename).read_bytes() == tested_bytes
    assert r2_server.store[f'releases/win32/canary/{filename}'][0] == tested_bytes
    writes = [path for method, path, _ in r2_server.requests if method == 'PUT']
    assert writes[-1].endswith('/canary.appinstaller')
    descriptor = ET.fromstring(r2_server.store['releases/win32/canary/canary.appinstaller'][0])
    main_bundle = descriptor.find('{*}MainBundle')
    assert main_bundle is not None
    assert descriptor.get('Version') == main_bundle.get('Version') == version

    # The publication job must refuse an ambiguous envelope, even when a
    # caller accidentally broadens the receipt selector in future.
    (tmp_path / 'staged/HermesBundled-0.29.0.0-win.msixbundle').write_bytes(tested_bytes)
    before = len(writes)
    refused = shell_step(tmp_path, r2_server, publisher,
                         'Publish identical tested bytes without rebuilding', env)
    assert refused.returncode != 0
    assert len([row for row in r2_server.requests if row[0] == 'PUT']) == before
    (tmp_path / 'staged/HermesBundled-0.29.0.0-win.msixbundle').unlink()

    # Filename, tag base, and baked identity must still agree. Reading the
    # accepted assembly version is not permission to trust arbitrary metadata.
    staged_bundle = tmp_path / 'staged' / filename
    for field, wrong in [('Version', '0.28.1.0'), ('Name', 'NousResearch.Other'), ('Publisher', 'CN=Other')]:
        with zipfile.ZipFile(bundle) as archive:
            manifest = ET.fromstring(archive.read('AppxMetadata/AppxBundleManifest.xml'))
        native = manifest.find('Identity')
        assert native is not None
        native.set(field, wrong)
        with zipfile.ZipFile(staged_bundle, 'w') as archive:
            archive.writestr('AppxMetadata/AppxBundleManifest.xml', ET.tostring(manifest))
        refused = shell_step(tmp_path, r2_server, publisher,
                             'Publish identical tested bytes without rebuilding', env)
        assert refused.returncode != 0, field
        assert len([row for row in r2_server.requests if row[0] == 'PUT']) == before
    staged_bundle.write_bytes(tested_bytes)
    for wrong in ['HermesBundled-0.29.0.0-win.msixbundle', 'HermesBundled-0.28.1.65536-win.msixbundle',
                  'HermesBundled-0.28.1-canary.20260818101010-win.msixbundle']:
        renamed = staged_bundle.rename(staged_bundle.with_name(wrong))
        refused = shell_step(tmp_path, r2_server, publisher,
                             'Publish identical tested bytes without rebuilding', env)
        assert refused.returncode != 0, wrong
        assert len([row for row in r2_server.requests if row[0] == 'PUT']) == before
        renamed.rename(staged_bundle)