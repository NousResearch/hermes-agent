import hashlib
import os
import sys
from pathlib import Path

from github_pr_feedback.ci_runner import SubprocessCICommandRunner


def test_failed_command_retains_hash_bound_private_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))
    result = SubprocessCICommandRunner().run(
        (sys.executable, '-c', "import sys; print('failure detail'); sys.exit(7)"),
        cwd=tmp_path, env=dict(os.environ), timeout=10,
    )
    assert result.returncode == 7
    digest = hashlib.sha256(result.stdout.encode()).hexdigest()
    output = tmp_path / 'home/github-pr-feedback/ci-output' / (digest + '.log')
    assert output.read_text() == result.stdout
    assert output.stat().st_mode & 0o777 == 0o600
