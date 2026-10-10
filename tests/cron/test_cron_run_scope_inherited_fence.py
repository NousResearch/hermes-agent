"""``_CronRunScope`` must NOT open an inherited ``HERMES_DELEGATED_CHILD_CONTEXT`` fence.

(marker present, no live ContextVar) is the state of BOTH a genuinely spawned delegate
descendant — a ``delegate_task`` child that launched a separate ``hermes cron run`` process,
exactly who the fence is for — and a host process carrying stale contamination. Without a
spawn edge the two are indistinguishable, so masking inside the scope would unfence the real
descendant; the scope stays fail-closed and contaminated host entry points scrub the marker
at their own startup boundary instead (``scrub_delegate_child_env_markers``). See #136081
and the two-process regression review on PR #136152.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent import delegation_context as dc

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Runs in a REAL spawned interpreter with only the env a spawn edge granted. It asserts the
# descendant-side invariants the tool gate, the cron scope and the DB fence must all keep.
# Same argv-style pattern as tests/test_scratch_dir.py:45 (sys.executable, "-c", code).
_DESCENDANT_WITNESS = r"""
import os
import sys
from pathlib import Path

from agent import delegation_context as dc

marker = dc.DELEGATED_CHILD_ENV_MARKER
assert os.environ.get(marker), "spawned descendant must carry the granted marker"
assert dc.is_delegated_child_process_context() is True, "descendant must be fenced (tool gate)"

from cron.scheduler_run_scope import _CronRunScope
scope = _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")
try:
    scope.enter()
    assert dc.is_delegated_child_process_context() is True, \
        "cron run scope must not open a granted descendant fence"
finally:
    scope.exit()
assert dc.is_delegated_child_process_context() is True, "scope exit must keep the fence"

root = Path(os.environ.get(marker))
assert dc.kanban_path_is_fenced(root / "board.db") is True, "DB fence must hold on the lineage root"

from hermes_cli import kanban_db
try:
    kanban_db._assert_not_delegated_child_mutation(str(root / "board.db"))
except PermissionError:
    pass
else:
    sys.exit("DB-layer mutation gate let a granted descendant write")
print("DESCENDANT-FENCED")
"""

# Same interpreter, but with the marker scrubbed at the (simulated) host startup boundary.
_HOST_RESCUE_WITNESS = r"""
import os

from agent import delegation_context as dc

assert not os.environ.get(dc.DELEGATED_CHILD_ENV_MARKER), "scrubbed host env must be clean"
assert dc.is_delegated_child_process_context() is False, "scrubbed host process must be unfenced"

from cron.scheduler_run_scope import _CronRunScope
scope = _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")
try:
    scope.enter()
    assert dc.is_delegated_child_process_context() is False, "clean host cron run must stay unfenced"
finally:
    scope.exit()
print("HOST-UNFENCED")
"""


def _run_witness(script: str, env: dict):
    # argv-style, no shell (the established tests/test_scratch_dir.py:45 pattern): argv is
    # the interpreter plus a fixed module constant; env is the granted env under test; cwd
    # puts the worktree root on sys.path so the child can import the hermes packages.
    return subprocess.run(
        [sys.executable, "-c", script], env=env, cwd=str(_REPO_ROOT),
        capture_output=True, text=True, timeout=120,
    )


class TestCronRunScopeStaysFailClosed:
    def test_inherited_marker_keeps_the_run_fenced(self, monkeypatch, tmp_path):
        # Same-process contract lock: whatever put the marker there, the scope must not
        # suppress the inherited half (indistinguishable from a genuine descendant grant).
        monkeypatch.setenv(dc.DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "board"))
        from cron.scheduler_run_scope import _CronRunScope

        scope = _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")
        try:
            scope.enter()
            assert dc.is_delegated_child_process_context() is True
            assert dc.kanban_path_is_fenced(tmp_path / "board" / "x.db") is True
        finally:
            scope.exit()
        assert dc.is_delegated_child_process_context() is True

    def test_live_delegate_child_stays_fenced_inside_the_run(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dc.DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "board"))
        from cron.scheduler_run_scope import _CronRunScope

        with dc.delegated_child_context():
            scope = _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")
            try:
                scope.enter()
                assert dc.is_delegated_child_process_context() is True
            finally:
                scope.exit()

    def test_clean_environment_runs_unfenced(self, monkeypatch):
        monkeypatch.delenv(dc.DELEGATED_CHILD_ENV_MARKER, raising=False)
        from cron.scheduler_run_scope import _CronRunScope

        scope = _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")
        try:
            scope.enter()
            assert dc.is_delegated_child_process_context() is False
        finally:
            scope.exit()


class TestSpawnedDescendantTwoProcessRegression:
    def test_granted_env_spawns_a_fenced_descendant(self, monkeypatch, tmp_path):
        """A real spawn edge granting the marker (delegated_child_subprocess_env from inside
        a delegate child) must produce a descendant whose cron runs, tool gate and DB-layer
        mutation gate all stay denied."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with dc.delegated_child_context():
            granted = dc.delegated_child_subprocess_env(dict(os.environ))
        assert granted.get(dc.DELEGATED_CHILD_ENV_MARKER), "the spawn grant must carry the marker"

        result = _run_witness(_DESCENDANT_WITNESS, granted)
        assert result.returncode == 0, result.stderr
        assert "DESCENDANT-FENCED" in result.stdout

    def test_scrubbed_host_env_spawns_an_unfenced_process(self, monkeypatch, tmp_path):
        """The host-owned rescue side: after the gateway startup scrub, a respawned host
        process and its cron runs are unfenced."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with dc.delegated_child_context():
            granted = dc.delegated_child_subprocess_env(dict(os.environ))
        from hermes_cli.gateway_restart_env import scrub_delegate_child_env_markers

        scrubbed = dict(granted)
        scrub_delegate_child_env_markers(scrubbed)
        assert dc.DELEGATED_CHILD_ENV_MARKER not in scrubbed

        result = _run_witness(_HOST_RESCUE_WITNESS, scrubbed)
        assert result.returncode == 0, result.stderr
        assert "HOST-UNFENCED" in result.stdout
