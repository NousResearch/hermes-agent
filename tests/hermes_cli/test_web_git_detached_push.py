"""Push must not report success without attempting to push a detached checkout."""
import subprocess

import pytest

from hermes_cli import web_git


def test_detached_push_reports_failure(tmp_path):
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("commit", "--allow-empty", "-m", "baseline")
    git("checkout", "--detach", "HEAD")
    with pytest.raises(RuntimeError, match="(?i)branch|detached"):
        web_git.review_push(str(tmp_path))
