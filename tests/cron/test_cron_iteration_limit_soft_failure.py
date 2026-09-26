"""A cron run that stopped at its iteration limit is delivered — but must not be recorded "ok".

``run_conversation`` ends such a turn with ``turn_exit_reason="max_iterations_reached(N/N)"``,
``completed=False`` and a usable summary in ``final_response`` (see
``agent/turn_failure_copy.py::is_max_iteration_handoff``). The scheduler delivers that summary on
purpose — it is a resumable handoff, not a failure — but the *status* used to be written as a
completed run with ``last_status="ok"``, so "did not finish the work" reached the board looking
like "nothing to see". The delivered text is unchanged; only the recorded outcome is.

Covers:

* The flag handed to the caller is set for an iteration-limit handoff and cleared for a normal
  turn, so a later run cannot inherit a stale value.
* ``run_job`` still delivers the fallback summary (no regression to delivery behaviour).
* ``run_one_job`` records the run as a soft failure instead of "ok".
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME for each test so jobs/scripts don't leak."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "scripts").mkdir()
    (home / "cron").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))

    import importlib
    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def _max_iteration_result() -> dict:
    """The shape ``run_conversation`` returns when the turn ran out of iterations."""
    return {
        "failed": False,
        "interrupted": False,
        "completed": False,
        "turn_exit_reason": "max_iterations_reached(35/35)",
        "final_response": "I reached the iteration limit before finishing; here is what I have so far.",
        "messages": [],
        "model": "test-model",
    }


def _normal_result() -> dict:
    return {
        "failed": False,
        "completed": True,
        "turn_exit_reason": "completed",
        "final_response": "All done.",
        "messages": [],
        "model": "test-model",
    }


class _StubAgent:
    """Stands in for AIAgent where only the turn-completion explainer is touched."""

    @staticmethod
    def _format_turn_completion_explanation(*args, **kwargs):  # pragma: no cover - copy helper
        return ""


# ---------------------------------------------------------------------------
# The flag handed to the caller
# ---------------------------------------------------------------------------

def test_handoff_result_sets_the_flag(hermes_env):
    import cron.scheduler as scheduler

    scheduler._ITER_LIMIT_FALLBACK.set(False)  # a previous run's value must not be inherited

    text = scheduler._final_response_from_result(
        _max_iteration_result(), "jobid", "some job", _StubAgent)

    assert "iteration limit" in text          # still delivered
    assert scheduler._ITER_LIMIT_FALLBACK.get() is True


def test_normal_result_leaves_the_flag_clear(hermes_env):
    import cron.scheduler as scheduler

    scheduler._ITER_LIMIT_FALLBACK.set(True)  # stale True from a previous run

    text = scheduler._final_response_from_result(
        _normal_result(), "jobid", "some job", _StubAgent)

    assert text == "All done."
    assert scheduler._ITER_LIMIT_FALLBACK.get() is False


def test_failed_result_leaves_the_flag_clear(hermes_env):
    """A genuine failure raises before the flag matters; it must not be marked as a handoff."""
    import cron.scheduler as scheduler

    with pytest.raises(RuntimeError):
        scheduler._final_response_from_result(
            {"failed": True, "completed": False, "error": "boom", "final_response": ""},
            "jobid", "some job", _StubAgent)

    assert scheduler._ITER_LIMIT_FALLBACK.get() is False


# ---------------------------------------------------------------------------
# End to end through the scheduler
# ---------------------------------------------------------------------------

class _IterationLimitedAgent:
    """AIAgent stand-in whose single turn runs out of iterations."""

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, *args, **kwargs):
        return _max_iteration_result()


@pytest.fixture
def fake_run_agent(monkeypatch):
    """Inject a stub ``run_agent`` module.

    ``run_job`` imports ``AIAgent`` from ``run_agent`` late, and that import pulls the runtime
    bootstrap in with it — which walks the real HERMES_HOME and trips the suite's home guard.
    Replacing the module keeps this test about the scheduler's status handling only.
    """
    import sys
    import types

    module = types.ModuleType("run_agent")
    module.AIAgent = _IterationLimitedAgent
    monkeypatch.setitem(sys.modules, "run_agent", module)
    return module


def _agent_job(hermes_env):
    """An agent job in the isolated home, with the credential the pre-dispatch gate wants."""
    from cron.jobs import create_job

    (hermes_env / ".env").write_text("OPENROUTER_API_KEY=test-key-never-used\n")
    return create_job(
        prompt="do the thing", schedule="every 5m", deliver="local",
        model="test/model", provider="openrouter")


def test_run_job_still_delivers_the_iteration_limit_summary(hermes_env, fake_run_agent):
    import cron.scheduler as scheduler

    job = _agent_job(hermes_env)

    success, doc, final, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert "iteration limit" in final
    assert scheduler._ITER_LIMIT_FALLBACK.get() is True


def test_run_one_job_records_a_soft_failure_not_ok(hermes_env, fake_run_agent):
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    job = _agent_job(hermes_env)

    scheduler.run_one_job(job)

    stored = get_job(job["id"])
    assert stored["last_status"] != "ok"
    assert "iteration limit" in (stored["last_error"] or "")
