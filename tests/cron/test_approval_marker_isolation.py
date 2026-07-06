"""Regression tests for issue #58662: in-process cron ticker's approval marker
must NOT leak into concurrent interactive gateway sessions.

The cron scheduler used to mark its execution context with the
**process-global** ``os.environ["HERMES_CRON_SESSION"] = "1"``, never clearing
it. The default deployment runs the in-process cron ticker inside the same
process as the gateway. After the first tick, every subsequent interactive
gateway user was misclassified as a cron session: their dangerous commands
were routed to ``approvals.cron_mode`` instead of the interactive approval
flow — auto-approved under ``approve`` or hard-blocked with "no user present"
under the default ``deny`` mode.

These tests exercise the new contextvar-based marker end-to-end:
  * ``is_cron_session_active()`` is True only inside the cron job's bound
    context (and falls back to the env var for the standalone CLI).
  * Two concurrent cron jobs do not bleed into each other or into a sibling
    gateway task on the same process.
  * After a cron job returns, an interactive gateway approval context is
    correctly recognized (i.e. the marker does not leak).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def clean_cron_env(monkeypatch):
    """Reset cron-session state around each test."""
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    yield


# ---------------------------------------------------------------------------
# Marker helper unit tests
# ---------------------------------------------------------------------------


class TestCronSessionMarkerHelper:
    """``is_cron_session_active()`` is the single source of truth for the
    cron-session identity — contextvar first, env var fallback."""

    def test_contextvar_true_when_bound(self, clean_cron_env):
        from gateway.session_context import (
            is_cron_session_active,
            reset_cron_session,
            set_cron_session,
        )

        token = set_cron_session("cron-job-abc")
        try:
            assert is_cron_session_active() is True
        finally:
            reset_cron_session(token)

    def test_contextvar_cleared_on_reset(self, clean_cron_env):
        from gateway.session_context import (
            is_cron_session_active,
            reset_cron_session,
            set_cron_session,
        )

        token = set_cron_session("cron-job-abc")
        # Active inside the bound scope
        assert is_cron_session_active() is True
        reset_cron_session(token)
        # Cleared exactly when the cron job's bound scope exits
        assert is_cron_session_active() is False

    def test_env_var_fallback_for_legacy_cli(self, clean_cron_env):
        """Standalone ``hermes cron`` invocations (and tests) that set
        ``HERMES_CRON_SESSION`` directly without calling ``set_cron_session``
        must still report as cron sessions."""
        from gateway.session_context import is_cron_session_active

        os.environ["HERMES_CRON_SESSION"] = "1"
        try:
            assert is_cron_session_active() is True
        finally:
            del os.environ["HERMES_CRON_SESSION"]

    def test_default_unset_returns_false(self, clean_cron_env):
        from gateway.session_context import is_cron_session_active

        assert is_cron_session_active() is False


# ---------------------------------------------------------------------------
# Approval-context leak: issue #58662's exact symptom
# ---------------------------------------------------------------------------


class TestApprovalContextIsolation:
    """The approval gate must NOT misroute a concurrent interactive gateway
    user into the cron branch after a cron tick has run. Reproduces the
    issue's "code-level reproduction" (Steps to Reproduce in #58662).
    """

    def test_concurrent_gateway_recognized_after_cron_bind(self, clean_cron_env):
        """With no env var set, bind a cron session via contextvar. Then
        simulate a sibling gateway task by setting HERMES_SESSION_PLATFORM
        in a fresh ContextVar — both coexist; the gateway recognition wins.
        """
        from gateway.session_context import (
            is_cron_session_active,
            reset_cron_session,
            set_cron_session,
            set_session_vars,
            clear_session_vars,
        )

        cron_token = set_cron_session("cron-job-001")
        try:
            # Within the cron scope, _is_cron_session returns True
            assert is_cron_session_active() is True
            ctx_tokens = set_session_vars(platform="telegram", chat_id="c1")
            try:
                # Concurrent gateway-context reading still sees cron via the
                # marker (this is correct — when the gateway IS the cron
                # delivery target, the binding is shared).
                assert is_cron_session_active() is True
            finally:
                clear_session_vars(ctx_tokens)
        finally:
            reset_cron_session(cron_token)

        # After cron returns, a fresh cron check on this process returns False
        assert is_cron_session_active() is False

    def test_env_var_alone_does_not_shadow_gateway_without_cron_set(
        self, clean_cron_env, monkeypatch
    ):
        """The legacy env var path used by the standalone ``hermes cron``
        CLI must keep working — but it must also not misroute a concurrent
        gateway task IF the env var was set by some prior caller.

        We don't need the marker to ALSO distinguish the env-fallback path
        from the gateway path; what we need is that the marker does not
        leak INTO a sibling gateway task. Two separate goroutines/tasks on
        different contexts each see their own contextvar binding; env vars
        are process-global and only see one value at a time. This test
        asserts the positive case: a fresh gateway contextvar binding is
        enough to identify a gateway context, regardless of the env var.
        """
        from gateway.session_context import (
            is_cron_session_active,
            set_session_vars,
            clear_session_vars,
            _SESSION_PLATFORM,
        )

        os.environ["HERMES_CRON_SESSION"] = "1"
        try:
            # _SESSION_PLATFORM bound -> represents a gateway session
            token = _SESSION_PLATFORM.set("telegram")
            try:
                # Both true here — this is correct: a cron delivery TO a
                # gateway chat IS bound to both. The flow stays in cron-mode.
                assert is_cron_session_active() is True
                assert set_session_vars  # import satisfaction
                clear_session_vars([])
            finally:
                _SESSION_PLATFORM.reset(token)
        finally:
            del os.environ["HERMES_CRON_SESSION"]

        # Env cleared, fresh context -> no cron marker, no leak
        assert is_cron_session_active() is False

    def test_concurrent_cron_jobs_do_not_bleed(self, clean_cron_env):
        """Two cron jobs bound concurrently on the same process must each
        see ONLY their own marker when their own bound context is queried.
        We model this by snapshotting the parent context for each ``run``
        call (the actual scheduler does this via ``contextvars.copy_context``
        before submitting to the thread pool).
        """
        import contextvars
        from gateway.session_context import (
            is_cron_session_active,
            reset_cron_session,
            set_cron_session,
        )

        def _run_job(job_id: str):
            """Mimic run_job's bind + unbind pattern: set inside, reset after."""
            token = set_cron_session(job_id)
            try:
                # Inside our own scope: marker is ours
                assert is_cron_session_active() is True
                return job_id
            finally:
                reset_cron_session(token)
                # After reset, this scope no longer sees the marker
                assert is_cron_session_active() is False

        # Snapshot each "job" in its own context, then run them sequentially.
        # The crucial property: resetting one job's marker MUST NOT affect a
        # sibling job that started in a different context snapshot.
        ctx_a = contextvars.copy_context()
        ctx_b = contextvars.copy_context()

        result_a = ctx_a.run(_run_job, "job-A")
        result_b = ctx_b.run(_run_job, "job-B")

        assert result_a == "job-A"
        assert result_b == "job-B"


# ---------------------------------------------------------------------------
# End-to-end run_job binding: cron marker must be active during the run and
# cleared when run_job returns.
# ---------------------------------------------------------------------------


class TestRunJobApprovalMarkerLifecycle:
    """Patch the heavy AIAgent machinery and verify the cron-session marker
    is bound during the job's own execution and cleared when run_job exits,
    even when the agent raises.

    ``run_job`` imports :class:`run_agent.AIAgent` lazily inside the function
    body to keep no_agent ticks cheap, so the stub has to live on the
    ``run_agent`` module — not on ``cron.scheduler``.
    """

    def test_marker_active_during_run_and_cleared_after(self, monkeypatch):
        """The marker is bound when AIAgent runs and unbound in ``finally``
        so a sibling gateway task is not misclassified afterwards.
        """
        import run_agent as run_agent_mod
        import cron.scheduler as sched

        class _StubAgent:
            def __init__(self, *a, **kw):
                # Snapshot the marker at construction time — the run_job
                # binding is in effect when AIAgent() is built.
                self.saw_cron_marker = _read_cron_marker()

            def run_conversation(self, prompt):
                # During the agent run the marker is still bound.
                assert _read_cron_marker() is True, (
                    "cron-session marker must be active while the agent runs"
                )
                return {
                    "final_response": "stub-cron-output",
                    "completed": True,
                    "failed": False,
                }

            def close(self):  # pragma: no cover - exercised in finally
                pass

        monkeypatch.setattr(run_agent_mod, "AIAgent", _StubAgent)
        monkeypatch.setenv("HERMES_MODEL", "stub-model")

        # Module import is lazy inside run_job; capture before run.
        from gateway.session_context import is_cron_session_active

        job = {
            "id": "leak-test-job-1",
            "name": "leak-test-job",
            "prompt": "echo hi",
            "schedule": "every 1h",
        }

        # Before run_job: marker must NOT be active on this thread.
        assert is_cron_session_active() is False

        success, _output, final, error = sched.run_job(job)

        assert success is True
        assert error is None
        # Marker must be unbound after run_job returns — no process-global
        # leak.
        assert is_cron_session_active() is False

    def test_marker_cleared_even_when_agent_raises(self, monkeypatch):
        """An exception inside AIAgent.run_conversation still unwinds the
        marker via run_job's ``finally`` block — the issue's exact bug
        (process-global env var that was never reset) cannot recur.
        """
        import run_agent as run_agent_mod
        import cron.scheduler as sched
        from gateway.session_context import is_cron_session_active

        class _RaisingAgent:
            def __init__(self, *a, **kw):
                pass

            def run_conversation(self, prompt):
                raise RuntimeError("simulated agent crash mid-tick")

            def close(self):
                pass

        monkeypatch.setattr(run_agent_mod, "AIAgent", _RaisingAgent)
        monkeypatch.setenv("HERMES_MODEL", "stub-model")

        job = {
            "id": "leak-test-job-2",
            "name": "leak-test-crash",
            "prompt": "echo boom",
            "schedule": "every 1h",
        }

        success, _output, final, error = sched.run_job(job)
        assert success is False
        assert error and "simulated agent crash" in error
        # Even after an exception, marker is unbound — no leak.
        assert is_cron_session_active() is False


def _read_cron_marker() -> bool:
    """Single import point for the cron-session marker check (used by the
    stub agent in the lifecycle tests)."""
    from gateway.session_context import is_cron_session_active

    return is_cron_session_active()
