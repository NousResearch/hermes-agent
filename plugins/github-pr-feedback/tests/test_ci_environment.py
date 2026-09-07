import json
import os
import subprocess
import sys

from github_pr_feedback.ci_environment import ci_environment


def test_safe_path_ci_imports_only_its_explicit_worktree_helpers(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    scripts = repo / 'scripts'
    scripts.mkdir(parents=True)
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'ci_summary.py').write_text('identity = "foreign"\n')
    (scripts / 'ci_summary.py').write_text('identity = "verified-worktree"\n')
    (repo / 'owner.py').write_text('identity = "verified-root"\n')
    script = scripts / 'check.py'
    script.write_text('import ci_summary, owner, json, sys\nprint(json.dumps([ci_summary.identity, owner.identity, sys.flags.safe_path]))\n')
    monkeypatch.setenv('PYTHONSAFEPATH', '1')
    monkeypatch.setenv('PYTHONPATH', str(foreign))
    result = subprocess.run([sys.executable, '-P', str(script)], cwd=foreign,
                            env=ci_environment(repo), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['verified-worktree', 'verified-root', True]


def test_ci_import_roots_do_not_follow_scripts_symlink_outside_worktree(tmp_path):
    repo = tmp_path / 'repo'
    foreign = tmp_path / 'foreign'
    repo.mkdir()
    foreign.mkdir()
    (repo / 'scripts').symlink_to(foreign, target_is_directory=True)
    env = ci_environment(repo, {'STATIC_BASE_REF': 'b' * 40})
    assert env['PYTHONPATH'].split(os.pathsep) == [str(repo.resolve())]
    assert env['STATIC_BASE_REF'] == 'b' * 40
