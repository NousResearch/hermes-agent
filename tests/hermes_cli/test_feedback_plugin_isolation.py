"""A worker's plugin tests must not inherit the live Kanban handoff identity."""
import os
from pathlib import Path
import subprocess
import sys


def test_plugin_handoff_tests_cannot_block_the_calling_worker(tmp_path):
    root = Path(__file__).resolve().parents[2]
    inherited_home = tmp_path / 'worker-home'
    inherited_home.mkdir()
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'plugins/github-pr-feedback/tests/test_cli.py',
         '-q', '-k', 'test_failed_audit_handoff_dispatches_the_typed_receipt_before_completion'],
        cwd=root, capture_output=True, text=True, timeout=30,
        env=dict(os.environ, HOME=str(tmp_path), HERMES_HOME=str(inherited_home),
                 HERMES_KANBAN_TASK='t_isolation_probe', HERMES_KANBAN_BOARD='worker-board',
                 HERMES_KANBAN_DB=str(inherited_home / 'kanban.db')),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (inherited_home / 'kanban.db').exists()
