"""``_CronRunScope`` must mask a ``HERMES_DELEGATED_CHILD_CONTEXT`` inherited from the
shell/session that fired the run — an in-process run has no spawn edge at which the marker
could be scrubbed, and unmasked it fails every Kanban write in the run with PermissionError.
A live in-process delegate child stays fenced. See #136081."""
from __future__ import annotations

from agent import delegation_context as dc


def _make_scope():
    from cron.scheduler_run_scope import _CronRunScope

    return _CronRunScope(job={}, job_id="job-1", execution_id="exec-1")


class TestCronRunScopeMasksInheritedFence:
    def test_inherited_marker_does_not_fence_the_run(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dc.DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "board"))
        scope = _make_scope()
        assert dc.is_delegated_child_process_context() is True
        try:
            scope.enter()
            assert dc.is_delegated_child_process_context() is False
        finally:
            scope.exit()
        # The mask is scope-local: os.environ still carries the marker afterwards.
        assert dc.is_delegated_child_process_context() is True

    def test_clean_environment_runs_without_a_mask(self, monkeypatch):
        monkeypatch.delenv(dc.DELEGATED_CHILD_ENV_MARKER, raising=False)
        scope = _make_scope()
        try:
            scope.enter()
            scope.exit()
        except Exception:
            raise

    def test_live_delegate_child_stays_fenced_inside_the_run(self, monkeypatch, tmp_path):
        monkeypatch.setenv(dc.DELEGATED_CHILD_ENV_MARKER, str(tmp_path / "board"))
        with dc.delegated_child_context():
            scope = _make_scope()
            try:
                scope.enter()
                # A genuine delegate child firing a cron job is exactly who the fence
                # is for — the scope must not open it.
                assert dc.is_delegated_child_process_context() is True
            finally:
                scope.exit()
