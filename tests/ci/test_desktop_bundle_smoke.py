"""Replay artifact handoffs and publication; evaluate the actual workflow gates."""
import copy
import hashlib
import itertools
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import hermes_yaml
import pytest

from tests.ci.desktop_release_roles import (
    CANARY_TAG as TAG, DOWNLOADABLE_DISPATCHES, NATIVE_TARGETS, SHA, admitted, canary_publisher,
    gate, native_builds, needs_of, phase_result, selection_gates, smoke_callers, stage_step,
    termux_builder, universal_assembler, updater_publishers,
)
from tests.ci.test_commit_build_staging import ROOT, shell_step
from tests.ci.test_desktop_release_tag_admission import _BASH, _child_env, _workflow
from tests.scripts.test_release_r2 import r2_server  # noqa: F401
from scripts.releases.job_groups import JOB_GROUPS

ALL_JOBS = ','.join(JOB_GROUPS)


def smoke_workflow():
    return hermes_yaml.safe_load((ROOT / '.github/workflows/desktop-bundle-smoke.yml').read_text(encoding='utf-8-sig'))


def smoke_fetch_script():
    """The public artifact fetch every smoke runner shares."""
    scripts = {step['run'] for job in smoke_workflow()['jobs'].values()
               for step in job.get('steps', []) if step.get('id') == 'artifact'}
    (script,) = scripts
    return script


def selected_needs(jobs, name, selected):
    """Admitted needs whose validate outputs select only `selected` groups."""
    needs = admitted(needs_of(jobs[name]))
    needs['validate']['outputs'] = {group: ('true' if group in selected else 'false')
                                    for group in JOB_GROUPS}
    return needs


def native_consumers(jobs):
    """Jobs that act on a native build's result rather than observe it.

    Every job that directly needs a selection gate or a smoke and does not
    opt into always() (observers such as summaries must survive failures).
    """
    producers = set(selection_gates(jobs).values()) | set(smoke_callers(jobs).values())
    return [name for name, job in jobs.items()
            if producers & set(needs_of(job)) and 'always()' not in str(job.get('if', ''))]


def test_native_consumers_and_publication_fail_closed_across_trust_skips(tmp_path):
    jobs = _workflow()['jobs']
    legs = native_builds(jobs)
    base_inputs = DOWNLOADABLE_DISPATCHES['tag']
    consumers = native_consumers(jobs)
    assert set(smoke_callers(jobs).values()) | {universal_assembler(jobs)} \
        | set(updater_publishers(jobs).values()) <= set(consumers), consumers
    for name in consumers:
        job = jobs[name]
        admitting = []
        for dispatch, inputs in DOWNLOADABLE_DISPATCHES.items():
            needs = admitted(needs_of(job), channel=dispatch == 'channel')
            # A skipped trust branch exists in the ancestry of every native result.
            needs.update({leg: {'result': 'skipped'} for leg in legs.values() if leg not in needs})
            if gate(job['if'], inputs, needs):
                admitting.append((dispatch, inputs, needs))
        assert admitting, f'{name} never runs for a downloadable build'
        for dispatch, inputs, needs in admitting:
            assert not gate(job['if'], inputs, needs, cancelled=True), (name, dispatch)
            for dependency in needs_of(job):
                for result in ('failure', 'skipped', 'cancelled'):
                    faulty = copy.deepcopy(needs)
                    faulty[dependency]['result'] = result
                    assert not gate(job['if'], inputs, faulty), (name, dispatch, dependency, result)
            emptied = copy.deepcopy(needs)
            emptied['validate']['outputs'] = {}
            assert not gate(job['if'], inputs, emptied), (name, dispatch)

    # Smokes and the universal assembly run for every downloadable build of
    # the build phase, and never for dry runs or later stable phases.
    scope_jobs = list(smoke_callers(jobs).values()) + [universal_assembler(jobs)]
    for name in scope_jobs:
        needs = admitted(needs_of(jobs[name]))
        for phase, commit, upload, allowed in [
            ('', SHA, False, True), ('candidate', '', False, True),
            ('', '', True, True), ('', '', False, False),
            ('publish', '', False, False), ('promote', '', False, False),
        ]:
            inputs = {**base_inputs, 'release-phase': phase, 'build_commit': commit,
                      'upload_release': upload}
            assert gate(jobs[name]['if'], inputs, needs) is allowed, (name, inputs)
    # A group the caller did not select stays skipped; its consumers refuse.
    for name in scope_jobs:
        assert not gate(jobs[name]['if'], base_inputs,
                        selected_needs(jobs, name, {'termux'})), name
    for platform, name in updater_publishers(jobs).items():
        for key, value in [('build_commit', SHA), ('upload_release', False),
                           ('release-phase', 'candidate'), ('release-phase', 'promote')]:
            assert not gate(jobs[name]['if'], {**base_inputs, key: value}, admitted(needs_of(jobs[name])))
        groups = [target for target in NATIVE_TARGETS if target.startswith(f'{platform}-')]
        if platform == 'win32':
            groups.append('win32-bundle')
        for group in groups:
            assert not gate(jobs[name]['if'], base_inputs,
                            selected_needs(jobs, name, {group})), (name, group)

    for target, name in selection_gates(jobs).items():
        job = jobs[name]
        release, commit_leg = legs[(target, 'release')], legs[(target, 'commit')]
        outcomes = ('success', 'failure', 'skipped', 'cancelled')
        for commit, release_result, commit_result in itertools.product(('', SHA), outcomes, outcomes):
            needs = admitted(needs_of(job))
            needs[release] = {'result': release_result}
            needs[commit_leg] = {'result': commit_result}
            selected = gate(job['env']['SELECTED_BUILD_SUCCEEDED'], {'build_commit': commit}, needs, job_if=False)
            expected = (commit == '' and release_result == 'success' and commit_result == 'skipped') or (
                commit != '' and commit_result == 'success' and release_result == 'skipped')
            assert selected is expected
            (check,) = [step for step in job['steps'] if 'run' in step]
            result = subprocess.run([_BASH, '-e', '-c', check['run']], cwd=tmp_path,
                                    env=_child_env(SELECTED_BUILD_SUCCEEDED=str(selected).lower()),
                                    capture_output=True, text=True, timeout=5)
            assert (result.returncode == 0) is expected


def test_only_selection_gates_consume_native_build_legs():
    """Every consumer sees a build through its selection gate, which requires
    the chosen trust branch to succeed; a direct need would accept a skip."""
    jobs = _workflow()['jobs']
    legs = set(native_builds(jobs).values())
    gates = set(selection_gates(jobs).values())
    for name, job in jobs.items():
        assert not (legs & set(needs_of(job))) or name in gates, f'{name} needs a native build leg directly'


def test_signature_cache_saves_only_in_the_writable_build():
    jobs = _workflow()['jobs']
    windows = [(name, mode) for (target, mode), name in native_builds(jobs).items() if target.startswith('win32-')]
    assert {mode for _, mode in windows} == {'release', 'commit'}
    for name, mode in windows:
        commit = SHA if mode == 'commit' else ''
        job = jobs[name]
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
    assembly = jobs[universal_assembler(jobs)]
    assert assembly['cache-mode'] == 'read'
    setup = next(step for step in assembly['steps'] if step.get('uses') == './.github/actions/setup-pm')
    assert setup['with']['cache-python'] is False
    assert setup['with']['save-tools-cache'] is False and setup['with']['save-node-cache'] is False
    action = hermes_yaml.safe_load((ROOT / '.github/actions/setup-pm/action.yml').read_text())
    tools = next(step for step in action['runs']['steps'] if step.get('id') == 'tools-cache')
    # The assembly's save-tools-cache: false must turn the tool cache save off.
    enabled = {'cache': 'true', 'save-tools-cache': 'true'}
    assert gate(tools['if'], enabled, {}, job_if=False)
    assert not gate(tools['if'], {**enabled, 'save-tools-cache': 'false'}, {}, job_if=False)
    assert all(not step.get('uses', '').startswith('actions/cache@') for step in assembly['steps'])


def test_smoke_matrix_native_routes_and_driver_only_dependencies():
    workflow = smoke_workflow()
    jobs = _workflow()['jobs']
    executions = []
    for name in smoke_callers(jobs).values():
        caller = jobs[name]
        assert caller['permissions'] == {'contents': 'read'} and 'secrets' not in caller
        formats = caller.get('strategy', {}).get('matrix', {}).get('format', [caller['with']['format']])
        for fmt in formats:
            executions.append((caller['with']['platform'], caller['with']['arch'], fmt))
    assert set(executions) == {(platform, arch, fmt) for arch in ('arm64', 'x64')
                               for platform, formats in [('darwin', ('dmg', 'zip')), ('win32', ('msix',))]
                               for fmt in formats}
    runners = {name: job for name, job in workflow['jobs'].items()
               if any('actions/checkout@' in step.get('uses', '') for step in job.get('steps', []))}
    # The admission job rejects unsupported targets before any runner spawns
    # and needs no source; every job that checks out is a read-only smoke runner.
    assert runners and len(runners) < len(workflow['jobs'])
    for name, job in runners.items():
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
        # Evidence of a failed verdict is the evidence that matters.
        assert gate(upload['if'], {}, {}, job_if=False, failed=True)
        assert upload['with']['path'].endswith('/out')
        # The chat steps are the smoke verdict. Recording stop and artifact
        # upload run after that and are evidence; a 403 there must not fail
        # the job or skip channel publication.
        stop = next(step for step in job['steps'] if step.get('name') == 'Stop screen recording')
        assert stop['continue-on-error'] is True and upload['continue-on-error'] is True
        for verdict in ('macos-chat', 'windows-chat'):
            chat = next(step for step in job['steps'] if step.get('id') == verdict)
            assert 'continue-on-error' not in chat

    recorder = hermes_yaml.safe_load((ROOT / '.github/actions/e2e-screen-record/action.yml').read_text())
    assert all(not step.get('uses', '').startswith('actions/cache') for step in recorder['runs']['steps'])
    assert 'save-cache' not in recorder['inputs']
    # ffmpeg comes from the PM toolchain: the action must verify, not install.
    verify = next(step for step in recorder['runs']['steps'] if step.get('name') == 'Verify ffmpeg from the PM toolchain')
    assert gate(verify['if'], {'mode': 'start'}, {}, job_if=False)
    assert not gate(verify['if'], {'mode': 'stop'}, {}, job_if=False)
    for step in recorder['runs']['steps']:
        run = step.get('run', '')
        assert 'ffmpeg' not in run or 'winget' not in run and 'brew install' not in run and 'apt-get' not in run, \
            f"step {step.get('name')} installs ffmpeg through an OS package manager"

    workflows = [workflow] + [hermes_yaml.safe_load((ROOT / '.github/workflows' / name).read_text(encoding='utf-8-sig'))
                             for name in ('install-e2e-run.yml', 'install-e2e-macos-run.yml', 'install-e2e-windows-run.yml')]
    for document in workflows:
        # A job that never checks out (the smoke admission job's bare case
        # statement) needs no toolchain.
        drivers = [job for job in document['jobs'].values()
                   if any('actions/checkout@' in step.get('uses', '') for step in job.get('steps', []))]
        assert drivers
        for job in drivers:
            steps = job['steps']
            setup = next(step for step in steps if step.get('uses') == './.github/actions/setup-pm')
            assert setup['with']['toolchain'] == 'all' and not setup['with'].get('extras')
            assert 'ffmpeg' in setup['with']['packages'].split(',')
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


def test_stable_phase_and_canary_gates_require_smoke_but_preserve_other_phases(tmp_path, r2_server):
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

    canary = jobs[canary_publisher(jobs)]
    inputs = {'build_commit': '', 'upload_release': True, 'tag': TAG}
    needs = admitted(needs_of(canary))
    assert gate(canary['if'], inputs, needs)
    # The publisher runs always() to report, so it must itself refuse any
    # prerequisite that did not succeed: builds, smokes, updaters, the table.
    for name in needs_of(canary):
        if name == 'validate':
            continue
        for state in ('failure', 'skipped', 'cancelled'):
            faulty = copy.deepcopy(needs)
            faulty[name]['result'] = state
            assert not gate(canary['if'], inputs, faulty), (name, state)
    emptied = copy.deepcopy(needs)
    emptied['validate']['outputs'] = {}
    assert not gate(canary['if'], inputs, emptied)
    assert not gate(canary['if'], {**inputs, 'tag': 'v0.28.0'}, needs), 'a stable tag is not a canary'


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