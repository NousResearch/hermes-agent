"""Replay artifact handoffs and publication; evaluate the actual workflow gates."""
import copy
import hashlib
import itertools
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import hermes_yaml
import pytest

from tests.ci.test_commit_build_staging import ROOT, shell_step
from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow
from tests.scripts.test_release_r2 import r2_server  # noqa: F401

SHA = 'a' * 40
TAG = 'v0.28.0-canary.20260818101010'


def smoke_workflow():
    return hermes_yaml.safe_load((ROOT / '.github/workflows/desktop-bundle-smoke.yml').read_text(encoding='utf-8-sig'))


def gate(expression, inputs, needs, *, cancelled=False, job_if=True):
    """Evaluate the workflow's boolean subset, including Actions' implicit success.

    This is not a scheduler simulation; native Actions still owns cancellation
    and skipped-ancestor propagation. Requiring a status function prevents an
    implicit success() from suppressing consumers of the skipped trust branch.
    """
    expression = expression.strip().removeprefix('${{').removesuffix('}}').strip()
    if job_if and not re.search(r'\b(always|cancelled|success|failure)\(', expression):
        if any(row.get('result') != 'success' for row in needs.values()):
            return False

    def value(match):
        bits = match[0].split('.')
        result = {'inputs': inputs, 'needs': needs}
        for bit in bits:
            result = result.get(bit, '') if isinstance(result, dict) else ''
        return repr(result)

    expression = re.sub(r'\b(?:inputs|needs)(?:\.[\w-]+)+', value, expression)
    expression = expression.replace('&&', ' and ').replace('||', ' or ')
    expression = re.sub(r'!(?!=)', ' not ', expression)
    expression = re.sub(r'\btrue\b', 'True', expression)
    expression = re.sub(r'\bfalse\b', 'False', expression)
    return bool(eval('(' + expression.strip() + ')', {'__builtins__': {}}, {
        'always': lambda: True, 'cancelled': lambda: cancelled,
        'contains': lambda value, needle: needle in value,
    }))


def admitted(jobs):
    return {name: {'result': 'success', 'outputs': {'sha': SHA}} for name in jobs}


def test_native_consumers_and_publication_fail_closed_across_trust_skips(tmp_path):
    jobs = _workflow()['jobs']
    base_inputs = {'build_commit': '', 'upload_release': True, 'release-phase': '', 'termux_only': False, 'tag': TAG}
    consumers = ['smoke-darwin', 'smoke-win32', 'smoke-win32-universal', 'assemble-win32-bundle',
                 'publish-win32-updater', 'publish-darwin-updater', 'candidate-manifest']
    for name in consumers:
        job = jobs[name]
        inputs = {**base_inputs, 'release-phase': 'candidate' if name == 'candidate-manifest' else ''}
        needs = admitted(job['needs'])
        # A skipped execution exists in the ancestry of every native result.
        needs['build-win32-commit'] = {'result': 'skipped'}
        assert gate(job['if'], inputs, needs), name
        assert not gate(job['if'], inputs, needs, cancelled=True), name
        for dependency in job['needs']:
            for result in ('failure', 'skipped', 'cancelled'):
                faulty = copy.deepcopy(needs)
                faulty[dependency]['result'] = result
                assert not gate(job['if'], inputs, faulty), (name, dependency, result)
        needs['validate']['outputs'] = {}
        assert not gate(job['if'], inputs, needs), name

    scope_jobs = consumers[:4]
    for name in scope_jobs:
        needs = admitted(jobs[name]['needs'])
        for phase, commit, upload, termux, allowed in [
            ('', SHA, False, False, True), ('candidate', '', False, False, True),
            ('', '', True, False, True), ('', '', False, False, False),
            ('publish', '', False, False, False), ('promote', '', False, False, False),
            ('', SHA, False, True, False),
        ]:
            inputs = {**base_inputs, 'release-phase': phase, 'build_commit': commit,
                      'upload_release': upload, 'termux_only': termux}
            assert gate(jobs[name]['if'], inputs, needs) is allowed, (name, inputs)
    for name in ('publish-win32-updater', 'publish-darwin-updater'):
        for key, value in [('build_commit', SHA), ('upload_release', False),
                           ('termux_only', True), ('release-phase', 'candidate'), ('release-phase', 'promote')]:
            assert not gate(jobs[name]['if'], {**base_inputs, key: value}, admitted(jobs[name]['needs']))

    for platform in ('win32', 'darwin'):
        job = jobs[f'build-{platform}']
        outcomes = ('success', 'failure', 'skipped', 'cancelled')
        for commit, release, commit_result in itertools.product(
                ('', SHA), outcomes, outcomes):
            needs = admitted(job['needs'])
            needs[f'build-{platform}-release'] = {'result': release}
            needs[f'build-{platform}-commit'] = {'result': commit_result}
            selected = gate(job['env']['SELECTED_BUILD_SUCCEEDED'], {'build_commit': commit}, needs, job_if=False)
            expected = (commit == '' and release == 'success' and commit_result == 'skipped') or (
                commit != '' and commit_result == 'success' and release == 'skipped')
            assert selected is expected
            result = subprocess.run([_BASH, '-e', '-c', job['steps'][0]['run']], cwd=tmp_path,
                                    env=_child_env(SELECTED_BUILD_SUCCEEDED=str(selected).lower()),
                                    capture_output=True, text=True, timeout=5)
            assert (result.returncode == 0) is expected


def test_signature_cache_saves_only_in_the_writable_build():
    jobs = _workflow()['jobs']
    for branch, commit in [('release', ''), ('commit', SHA)]:
        job = jobs[f'build-win32-{branch}']
        steps = job['steps']
        cache_steps = [step for step in steps
                       if 'payload-signatures' in step.get('with', {}).get('path', '')]
        assert all(not step['uses'].startswith('actions/cache@') for step in cache_steps)
        restore = next(step for step in cache_steps if step['uses'].startswith('actions/cache/restore@'))
        save = next(step for step in cache_steps if step['uses'].startswith('actions/cache/save@'))
        assert save['with']['path'] == restore['with']['path']
        assert save['with']['key'] == '${{ steps.' + restore['id'] + '.outputs.cache-primary-key }}'
        assert gate(save['if'], {'build_commit': commit}, {}, job_if=False) is (job['cache-mode'] == 'write')
        verify = next(step for step in steps if step.get('name') == 'Verify native signature cache contracts')
        assert steps.index(restore) < steps.index(verify) < steps.index(save)
    assembly = jobs['assemble-win32-bundle']
    assert assembly['cache-mode'] == 'read'
    setup = next(step for step in assembly['steps'] if step.get('uses') == './.github/actions/setup-pm')
    assert setup['with']['cache-python'] is False
    assert setup['with']['save-tools-cache'] is False and setup['with']['save-node-cache'] is False
    action = hermes_yaml.safe_load((ROOT / '.github/actions/setup-pm/action.yml').read_text())
    tools = next(step for step in action['runs']['steps'] if step.get('id') == 'tools-cache')
    assert "inputs.save-tools-cache == 'true'" in tools['if']
    assert all(not step.get('uses', '').startswith('actions/cache@') for step in assembly['steps'])


def test_smoke_matrix_native_routes_and_driver_only_dependencies():
    workflow = smoke_workflow()
    jobs = _workflow()['jobs']
    executions = []
    for name in ('smoke-darwin', 'smoke-win32', 'smoke-win32-universal'):
        caller = jobs[name]
        assert caller['strategy']['fail-fast'] is False
        assert caller['permissions'] == {'contents': 'read'} and 'secrets' not in caller
        matrix = caller['strategy']['matrix']
        for arch, fmt in itertools.product(matrix['arch'], matrix.get('format', [caller['with']['format']])):
            executions.append((caller['with']['platform'], arch, fmt))
    assert set(executions) == {(platform, arch, fmt) for arch in ('arm64', 'x64')
                               for platform, formats in [('darwin', ('dmg', 'zip')), ('win32', ('msix', 'msixbundle'))]
                               for fmt in formats}
    for name, job in workflow['jobs'].items():
        # validate is the admission job (rejects unsupported targets before any
        # runner spawns, no checkout, no runner resources); every other job is a
        # read-only smoke runner.
        if name == 'validate':
            continue
        assert job['cache-mode'] == 'read' and 'environment' not in job
        checkout = next(step for step in job['steps'] if 'actions/checkout@' in step.get('uses', ''))
        assert checkout['with']['persist-credentials'] is False
        # A channel build smokes the trusted controller's checkout, not the
        # admitted source — the runner fetches the pinned channel request itself.
        assert checkout['with']['ref'] in (
            '${{ inputs.sha }}',
            "${{ inputs.channel-build != '' && inputs.controller-sha || inputs.sha }}")
        recording = next(step for step in job['steps'] if step.get('id') == 'recording')
        assert 'save-cache' not in recording['with']
        upload = next(step for step in job['steps'] if 'actions/upload-artifact@' in step.get('uses', ''))
        assert upload['if'] == 'always()' and upload['with']['path'].endswith('/out')

    recorder = hermes_yaml.safe_load((ROOT / '.github/actions/e2e-screen-record/action.yml').read_text())
    assert all(not step.get('uses', '').startswith('actions/cache') for step in recorder['runs']['steps'])
    assert 'save-cache' not in recorder['inputs']
    # ffmpeg comes from the PM toolchain: the action must verify, not install.
    verify = next(step for step in recorder['runs']['steps'] if step.get('name') == 'Verify ffmpeg from the PM toolchain')
    assert verify['if'] == "inputs.mode == 'start'"
    for step in recorder['runs']['steps']:
        run = step.get('run', '')
        assert 'ffmpeg' not in run or 'winget' not in run and 'brew install' not in run and 'apt-get' not in run, \
            f"step {step.get('name')} installs ffmpeg through an OS package manager"

    workflows = [workflow] + [hermes_yaml.safe_load((ROOT / '.github/workflows' / name).read_text(encoding='utf-8-sig'))
                             for name in ('install-e2e-run.yml', 'install-e2e-macos-run.yml', 'install-e2e-windows-run.yml')]
    for document in workflows:
        for name, job in document['jobs'].items():
            if 'steps' not in job or name == 'validate':
                # the smoke workflow's validate job rejects unsupported targets
                # with a bare case statement; it needs no toolchain setup.
                continue
            steps = job['steps']
            setup = next(step for step in steps if step.get('uses') == './.github/actions/setup-pm')
            assert setup['with']['toolchain'] == 'all' and not setup['with'].get('extras')
            assert setup['with']['packages'] == 'ffmpeg'
            assert all(setup['with'][key] is False for key in ('cache', 'cache-node', 'cache-python'))
            install = next(step for step in steps if step.get('name') == 'Install locked chat driver dependencies')
            assert steps.index(setup) < steps.index(install)
            args = install['run'].split()
            assert args[:2] == ['npm', 'ci'] and args[args.index('--workspace') + 1] == 'tests-js'
            assert {'--include-workspace-root', '--omit=dev', '--ignore-scripts', '--no-audit', '--no-fund'} <= set(args)


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
    job = 'assemble-win32-bundle' if universal else f'build-{platform}-commit'
    stage = 'Stage universal bundles to R2' if universal else (
        'Stage Windows packages to R2' if platform == 'win32' else 'Stage macOS packages and feed inputs to R2')
    if platform == 'darwin':
        # The producer stages all of its formats together; smoke selects one.
        for ext in ('dmg', 'zip', 'zip.blockmap'):
            (release / f'HermesBundled-0.28.0-mac-{arch}.{ext}').write_bytes(payload)
        (release / f'{arch}-canary-mac.yml').write_bytes(payload)
    staged = shell_step(tmp_path, r2_server, job, stage, {**env, 'TARGET': f'{platform}-{arch}'})
    assert staged.returncode == 0, staged.stdout + staged.stderr
    assert all('/canary/' not in key for key in r2_server.store)
    script = next(step['run'] for step in smoke_workflow()['jobs']['macos']['steps']
                  if step.get('id') == 'artifact')
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
    script = next(step['run'] for step in smoke_workflow()['jobs']['macos']['steps'] if step.get('id') == 'artifact')
    result = shell_step(tmp_path, r2_server, '', '', {**env, 'PLATFORM': 'win32', 'ARCH': 'x64', 'FORMAT': 'msix'}, script=script)
    assert result.returncode != 0
    assert not (tmp_path / 'output').exists()
    assert not (tmp_path / 'smoke/out/download.json').exists()


def test_stable_phase_and_canary_gates_require_smoke_but_preserve_other_phases(tmp_path, r2_server):
    jobs = _workflow()['jobs']
    required = {
        'candidate': ['validate', 'build-win32', 'build-darwin', 'assemble-win32-bundle',
                      'smoke-darwin', 'smoke-win32', 'smoke-win32-universal', 'termux-deb', 'candidate-manifest'],
        'publish': ['validate', 'stable-publish', 'stable-store'],
        'promote': ['validate', 'stable-promote'],
    }
    for phase, selected in required.items():
        needs = {name: {'result': 'success' if name in selected else 'skipped'}
                 for name in jobs['stable-phase-result']['needs']}
        for failed in [None, *selected]:
            changed = copy.deepcopy(needs)
            if failed:
                changed[failed]['result'] = 'cancelled'
            result = shell_step(tmp_path, r2_server, 'stable-phase-result', 'Require every phase job', {
                'RELEASE_NEEDS': json.dumps(changed), 'RELEASE_PHASE': phase})
            assert (result.returncode == 0) is (failed is None), (phase, failed, result.stderr)
    inputs = {'build_commit': '', 'upload_release': True, 'tag': TAG}
    needs = admitted(jobs['publish-canary']['needs'])
    assert gate(jobs['publish-canary']['if'], inputs, needs)
    for name in ('smoke-darwin', 'smoke-win32', 'smoke-win32-universal'):
        for state in ('failure', 'skipped', 'cancelled'):
            faulty = copy.deepcopy(needs)
            faulty[name]['result'] = state
            assert not gate(jobs['publish-canary']['if'], inputs, faulty)


def test_canary_publisher_consumes_staged_bytes_and_writes_pointer_last(tmp_path, r2_server):
    tag = 'v0.28.1-canary.20260818101010'
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
    staged = shell_step(tmp_path, r2_server, 'assemble-win32-bundle', 'Stage universal bundles to R2', env)
    assert staged.returncode == 0, staged.stdout + staged.stderr
    assert all(key.startswith(f'releases/tag/{tag}/') for key in r2_server.store)
    # A newer stable becomes visible after assembly but before publication.
    git('-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-qm', 'new stable', date='2026-08-18T11:10:10Z')
    git('tag', 'v0.28.1')
    assert assembly_identity()['version'] != version
    for name in ('Retrieve the tested universal bundle', 'Publish identical tested bytes without rebuilding'):
        result = shell_step(tmp_path, r2_server, 'publish-win32-updater', name, env)
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
    refused = shell_step(tmp_path, r2_server, 'publish-win32-updater',
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
        refused = shell_step(tmp_path, r2_server, 'publish-win32-updater',
                             'Publish identical tested bytes without rebuilding', env)
        assert refused.returncode != 0, field
        assert len([row for row in r2_server.requests if row[0] == 'PUT']) == before
    staged_bundle.write_bytes(tested_bytes)
    for wrong in ['HermesBundled-0.29.0.0-win.msixbundle', 'HermesBundled-0.28.1.65536-win.msixbundle',
                  'HermesBundled-0.28.1-canary.20260818101010-win.msixbundle']:
        renamed = staged_bundle.rename(staged_bundle.with_name(wrong))
        refused = shell_step(tmp_path, r2_server, 'publish-win32-updater',
                             'Publish identical tested bytes without rebuilding', env)
        assert refused.returncode != 0, wrong
        assert len([row for row in r2_server.requests if row[0] == 'PUT']) == before
        renamed.rename(staged_bundle)