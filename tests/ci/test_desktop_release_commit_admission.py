"""Commit-build admission rejects mixed inputs before repository code runs."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from scripts.releases.commit_build import publish_receipt, receipt_tag
from tests.ci.test_desktop_release_tag_admission import _child_env, _git, _seed_repo, _workflow, _BASH


def _admission_script():
    return next(
        step["run"] for step in _workflow()["jobs"]["validate"]["steps"]
        if step.get("name") == "Validate tag shape, pyproject lockstep, and ancestry on origin/main"
    )


def environment(clone, commit):
    return _child_env(
        TAG='', BUILD_COMMIT=commit, RELEASE_PHASE='', UPLOAD_RELEASE='false',
        TERMUX_UPGRADE_FROM_TAG='', DEFAULT_BRANCH='main', GITHUB_REF='refs/heads/main',
        GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_REPOSITORY='fixture/repo',
        GITHUB_WORKFLOW_REF='fixture/repo/.github/workflows/desktop-bundled-release.yml@refs/heads/main',
        GITHUB_ACTOR='maintainer', GITHUB_TRIGGERING_ACTOR='maintainer',
        GITHUB_OUTPUT=str(clone / 'outputs'), GH_TOKEN='fixture-token',
        GIT_ALLOW_PROTOCOL='file', PYTHONUTF8='1',
    )


def run_admission(clone, env):
    helper = clone / 'test-bin'
    helper.mkdir(exist_ok=True)
    (helper / 'python').write_text(
        f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n', encoding='utf-8')
    (helper / 'python').chmod(0o755)
    script = clone / 'admission.sh'
    script.write_text(_admission_script(), encoding='utf-8', newline='\n')
    return subprocess.run([_BASH, '-e', '-o', 'pipefail', str(script)], cwd=clone,
                          env={**env, 'PATH': str(helper) + os.pathsep + env['PATH']},
                          capture_output=True, text=True, encoding='utf-8', timeout=60)


def test_mixed_dispatch_is_refused_before_loading_repository_code(tmp_path):
    _, clone = _seed_repo(tmp_path)
    package = clone / 'scripts/releases'
    package.mkdir(parents=True)
    witness = clone / 'module-ran'
    (package / 'commit_build.py').write_text(
        f'from pathlib import Path\nPath({str(witness)!r}).write_text("executed")\n', encoding='utf-8')
    (clone / 'outputs').write_text('prior=value\n', encoding='utf-8')
    env = environment(clone, _git('rev-parse', 'HEAD', cwd=clone))
    for extra in ({'TAG': 'v1.2.3'}, {'RELEASE_PHASE': 'candidate'}, {'UPLOAD_RELEASE': 'true'},
                  {'TERMUX_UPGRADE_FROM_TAG': 'v1.2.2'}, {'BUILD_COMMIT': 'abc123'}):
        result = run_admission(clone, {**env, **extra})
        assert result.returncode != 0, result.stdout + result.stderr
        assert not witness.exists(), 'rejected input executed the checkout admission module'
        # The R2 public-url echo precedes admission; a refused dispatch writes no sha/channel/version.
        outputs = dict(line.split('=', 1) for line in (clone / 'outputs').read_text(encoding='utf-8').splitlines())
        assert outputs['prior'] == 'value'
        assert not {'sha', 'channel', 'payload-version'} & outputs.keys()


def test_trusted_dispatch_admits_a_pushed_feature_without_switching_checkout(tmp_path):
    origin, clone = _seed_repo(tmp_path)
    main = _git('rev-parse', 'HEAD', cwd=clone)
    _git('checkout', '-qb', 'feature', cwd=origin)
    (origin / 'pyproject.toml').write_text('[project]\nname="fixture"\nversion="3.2.1"\n', encoding='utf-8')
    _git('add', 'pyproject.toml', cwd=origin)
    _git('commit', '-qm', 'feature', cwd=origin)
    commit = _git('rev-parse', 'HEAD', cwd=origin)
    _git('fetch', 'origin', cwd=clone)
    source = Path(__file__).resolve().parents[2] / 'scripts/releases'
    shutil.copytree(source, clone / 'scripts/releases', ignore=shutil.ignore_patterns('__pycache__'))
    helper = clone / 'test-bin'
    helper.mkdir()
    permission_log = clone / 'permission.jsonl'
    code = (
        'import json,sys\nfrom pathlib import Path\n'
        'assert sys.argv[1] == "api" and sys.argv[2].endswith("/permission")\n'
        f'with Path({str(permission_log)!r}).open("a",encoding="utf-8") as stream: '
        'stream.write(json.dumps(sys.argv[1:])+"\\n")\nprint("write")\n'
    )
    if os.name == 'nt':
        from scripts.build.mint_launchers import mint_one
        mint_one(str(helper), sys.executable, code, {'name': 'gh', 'module': 'fixture', 'func': 'main'})
    else:
        module = helper / 'gh.py'
        module.write_text(code, encoding='utf-8')
        (helper / 'gh').write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(module))} "$@"\n', encoding='utf-8')
        (helper / 'gh').chmod(0o755)
    result = run_admission(clone, environment(clone, commit))
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = dict(line.split('=', 1) for line in (clone / 'outputs').read_text(encoding='utf-8').splitlines())
    # public-root/public-base mirror the R2 public URL (empty outside CI); the admission
    # contract is the pinned sha, the commit channel and the payload version.
    assert {key: outputs[key] for key in ('sha', 'channel', 'payload-version')} == {
        'sha': commit, 'channel': 'commit', 'payload-version': '3.2.1'}
    assert _git('rev-parse', 'HEAD', cwd=clone) == main
    requests = [json.loads(line) for line in permission_log.read_text(encoding='utf-8').splitlines()]
    assert requests and all(row[1] == 'repos/fixture/repo/collaborators/maintainer/permission' for row in requests)
    assert not _git('tag', '--list', cwd=clone)


def test_post_build_receipts_bind_kind_commit_and_run_without_same_second_collisions(tmp_path):
    _origin, clone = _seed_repo(tmp_path)
    commit = _git('rev-parse', 'HEAD', cwd=clone)
    created_at = '2026-09-22T01:23:45Z'
    assert receipt_tag('commit', '0.0.0', created_at, '123') == \
        'v0.0.0+commit.20260922T012345Z.123'
    assert receipt_tag('commit', '0.0.0', created_at, '124') != \
        receipt_tag('commit', '0.0.0', created_at, '123')

    def run(argv, repo=None):
        if argv[:2] == ['gh', 'api'] and argv[-1] == '.permission':
            return 'write'
        if argv[:2] == ['gh', 'api'] and '/actions/runs/' in argv[2]:
            run_id = argv[2].rsplit('/', 1)[-1]
            return json.dumps({
                'id': int(run_id), 'event': 'workflow_dispatch', 'status': 'in_progress',
                'head_branch': 'main', 'head_sha': commit, 'created_at': created_at,
            })
        return subprocess.check_output(argv, cwd=repo or clone, text=True, encoding='utf-8').strip()

    base = {
        **environment(clone, commit),
        'GITHUB_ACTIONS': 'true',
        'GITHUB_SHA': commit,
        'GITHUB_RUN_ID': '123',
    }
    first = publish_receipt(
        'commit', base, version='0.0.0', commit=commit,
        details={'bundleEnv': {}}, run=run, repo=clone,
    )
    second = publish_receipt(
        'commit', {**base, 'GITHUB_RUN_ID': '124'}, version='0.0.0', commit=commit,
        details={'bundleEnv': {}}, run=run, repo=clone,
    )
    assert [first['tag'], second['tag']] == [
        'v0.0.0+commit.20260922T012345Z.123',
        'v0.0.0+commit.20260922T012345Z.124',
    ]
    assert json.loads(_git('tag', '-l', first['tag'], '--format=%(contents)', cwd=clone)) == first
    assert _git('rev-parse', f"{first['tag']}^{{commit}}", cwd=clone) == commit
    assert publish_receipt(
        'commit', base, version='0.0.0', commit=commit,
        details={'bundleEnv': {}}, run=run, repo=clone,
    ) == first
