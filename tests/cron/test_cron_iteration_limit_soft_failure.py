"""A cron run that stopped at its iteration limit is delivered — but must not be recorded "ok".

``run_conversation`` ends such a turn with ``turn_exit_reason="max_iterations_reached(N/N)"``,
``completed=False`` and a usable summary in ``final_response`` (see
``agent/turn_failure_copy.py::is_max_iteration_handoff``). The scheduler delivers that summary on
purpose — it is a resumable handoff, not a failure — but the *status* used to be written as a
completed run with ``last_status="ok"``, so "did not finish the work" reached the board looking
like "nothing to see". The delivered text is unchanged; only the recorded outcome is.

Covers:

* ``run_job`` reports the handoff through its ``handoff_out`` out-parameter (this run's result, and
  nothing else — a later run in the same context must not inherit it).
* ``run_job`` still returns the fallback summary (no regression to delivery behaviour).
* ``run_one_job`` decides the outcome BEFORE the save/compose/deliver phase, so the incident ledger,
  ``failure_streak``, the delivery lane and the recorded status all describe the same run — and the
  job's open incident is not booked as recovered by the run that failed to finish.
* The handoff summary still reaches the operator: it is delivered *as* the failure notice, not
  replaced by the generic provider-flavoured diagnostic.
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


class _IterationLimitedAgent:
    """AIAgent stand-in whose single turn runs out of iterations."""

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, *args, **kwargs):
        return _max_iteration_result()


class _NormalAgent:
    """AIAgent stand-in whose single turn completes."""

    def __init__(self, *args, **kwargs):
        pass

    def run_conversation(self, *args, **kwargs):
        return _normal_result()


def _agent_module(agent_cls):
    import sys
    import types

    module = types.ModuleType("run_agent")
    module.AIAgent = agent_cls
    return module


@pytest.fixture
def fake_run_agent(monkeypatch):
    """Inject a stub ``run_agent`` module.

    ``run_job`` imports ``AIAgent`` from ``run_agent`` late, and that import pulls the runtime
    bootstrap in with it — which walks the real HERMES_HOME and trips the suite's home guard.
    Replacing the module keeps these tests about the scheduler's status handling only.
    """
    import sys

    monkeypatch.setitem(sys.modules, "run_agent", _agent_module(_IterationLimitedAgent))


def _agent_job(hermes_env):
    """An agent job in the isolated home, with the credential the pre-dispatch gate wants."""
    from cron.jobs import create_job

    (hermes_env / ".env").write_text("OPENROUTER_API_KEY=test-key-never-used\n")
    return create_job(
        prompt="do the thing", schedule="every 5m", deliver="local",
        model="test/model", provider="openrouter")


# ---------------------------------------------------------------------------
# What run_job reports about the run it just finished
# ---------------------------------------------------------------------------

def test_handoff_run_reports_the_iteration_limit_to_its_caller(hermes_env, fake_run_agent):
    import cron.scheduler as scheduler

    job = _agent_job(hermes_env)
    handoff: list = []

    success, doc, final, error = scheduler.run_job(job, handoff_out=handoff)

    assert success is True and error is None      # delivery behaviour unchanged
    assert "iteration limit" in final
    assert handoff == ["max_iterations_reached(35/35)"]   # and the caller can see it did not finish


def test_normal_run_reports_no_handoff(hermes_env, monkeypatch):
    """A completed turn must leave the caller's handoff slot empty."""
    import sys
    import cron.scheduler as scheduler

    monkeypatch.setitem(sys.modules, "run_agent", _agent_module(_NormalAgent))
    job = _agent_job(hermes_env)
    handoff: list = []

    success, doc, final, error = scheduler.run_job(job, handoff_out=handoff)

    assert success is True
    assert final == "All done."
    assert handoff == []


def test_result_of_a_failed_turn_is_not_a_handoff(hermes_env):
    """A genuine failure raises before anything is appended to the caller's slot."""
    import cron.scheduler as scheduler

    handoff: list = []

    with pytest.raises(RuntimeError):
        scheduler._final_response_from_result(
            {"failed": True, "completed": False, "error": "boom", "final_response": ""},
            "jobid", "some job", _StubAgent)

    assert handoff == []


# ---------------------------------------------------------------------------
# End to end through the scheduler
# ---------------------------------------------------------------------------

def test_run_one_job_records_a_soft_failure_not_ok(hermes_env, fake_run_agent, monkeypatch):
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    delivered: list = []
    monkeypatch.setattr(
        scheduler, "_deliver_result",
        lambda job, content, adapters=None, loop=None, for_failure=False: delivered.append(
            (content, for_failure)) or None)

    job = _agent_job(hermes_env)

    scheduler.run_one_job(job)

    stored = get_job(job["id"])
    assert stored["last_status"] == "error"      # pinned: not "ok", and not some other non-ok
    assert "iteration limit" in (stored["last_error"] or "")
    assert stored["failure_streak"] == 1         # booked as a failure everywhere, not as a recovery
    # The summary is the payload: it is delivered, on the failure lane, with a header that says the
    # run did not finish.
    assert delivered, "the handoff run must still deliver its summary"
    content, for_failure = delivered[-1]
    assert for_failure is True
    assert "here is what I have so far" in content
    assert "did not finish" in content


def test_a_run_that_never_reaches_the_agent_does_not_inherit_a_handoff(
        hermes_env, fake_run_agent, monkeypatch):
    """The reviewer's reproduction: run 1 hits the limit, run 2 is skipped before the agent exists.

    With process-lifetime state, run 2 inherited run 1's verdict and was booked with an error it
    never produced (``last_error = "Agent hit the iteration limit…"``).
    """
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    job = _agent_job(hermes_env)
    scheduler.run_one_job(job)
    assert get_job(job["id"])["last_status"] == "error"

    # Run 2: the pre-agent gate returns before the agent is ever built.
    monkeypatch.setattr(
        scheduler, "_prepare_job_prompt",
        lambda *a, **k: ((True, "## Response\n\nskipped\n", "skipped", None), ""))

    scheduler.run_one_job({"id": job["id"], "name": job.get("name")})

    stored = get_job(job["id"])
    assert "iteration limit" not in (stored["last_error"] or "")
    assert stored["last_status"] != "error"


def test_a_handoff_run_does_not_book_a_recovery_for_an_open_incident(monkeypatch, tmp_path):
    """The handoff must not resolve the job's open incident: no run may be booked as recovered and
    as failed at once. ``_compose_run_delivery``'s success branch is what resolves them, so the
    downgrade has to happen before it runs (cron/scheduler.py ``_run_one_job_body``)."""
    import cron.incidents as incidents
    import cron.scheduler as sched

    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    inc_id, _ = incidents.upsert_incident("job-1", "some earlier failure")

    content, *_ = sched._compose_run_delivery(
        {"id": "job-1", "name": "j1"}, success=False,
        error="Agent hit the iteration limit (max_iterations_reached(35/35)); the delivered text is a "
              "fallback, not evidence the work completed",
        final_response="Here is what I have so far.",
        output_file=None, handoff=True)

    assert incidents.get_incident(inc_id)["state"] != "resolved"
    assert "Here is what I have so far." in content


def test_the_handoff_summary_survives_the_failure_path(monkeypatch, tmp_path):
    """Composition keeps the agent's summary instead of swapping in the generic diagnostic."""
    import cron.incidents as incidents
    import cron.scheduler as sched

    monkeypatch.setattr(incidents, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")

    content, *_ = sched._compose_run_delivery(
        {"id": "job-2", "name": "j2"}, success=False,
        error="Agent hit the iteration limit (max_iterations_reached(35/35)); the delivered text is a "
              "fallback, not evidence the work completed",
        final_response="Twelve of the twenty rows were done; the rest need another pass.",
        output_file=None, handoff=True)

    assert "Twelve of the twenty rows were done" in content
    assert "did not finish" in content
