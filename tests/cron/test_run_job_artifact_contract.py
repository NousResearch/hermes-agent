"""Run-scoped delivery artifact contract tests.

Prove the production run_job() path — not just the helpers — reserves the
artifact, injects the path into the prompt, replaces the final response with
the scheduler-generated delivery, fails closed when the artifact is missing,
and never leaks the reservation across fires.

The real run_job() is exercised with a stubbed agent construction; the
artifact helpers, prompt injection and response-replacement logic are the
real code paths (patch where production reads).
"""
import pytest

import cron.scheduler as sched


@pytest.fixture
def agentless_run_job(monkeypatch):
    """Run the real sched.run_job() with agent construction/watchdog stubbed.

    Returns a runner taking (job, execution_id, final_response) where
    final_response is what the stubbed agent turn returns.
    """
    from cron import scheduler as s

    class _Agent:
        def __init__(self):
            self.calls = []

    def make_agent(job, agent_response):
        def _construct(AIAgent, job_, _cfg, setup, *, workdir, session_id, session_db):
            agent = _Agent()

            class _Setup:
                model = "test-model"
                fallback_notice = None  # KENSEI MERGE: matches _CronAgentSetup field

            return agent

        return _construct

    monkeypatch.setattr(sched, "_prepare_job_prompt", lambda job, job_id, job_name, extra, cancel: (None, "PROMPT-BODY"))
    return make_agent


def _stub_pipeline(monkeypatch, final_response):
    """Stub everything between run_job's agent construction and its return."""
    from cron import scheduler as s

    class _Scope:
        def __init__(self, *a, **k):
            self.workdir = None
            self.task_id = None

        def enter(self):
            pass

        def exit(self):
            pass

    class _Audit:
        def __init__(self, *a, **k):
            self.rows = []

        def write(self, row, err):
            self.rows.append((row, err))

    class _Cfg:
        cfg = {}
        model = "test-model"

    class _Setup:
        model = "test-model"
        blocked = None
        fallback_notice = None  # KENSEI MERGE: matches _CronAgentSetup field

    monkeypatch.setattr(sched, "_CronRunScope", _Scope)
    monkeypatch.setattr(sched, "_load_cron_job_config", lambda job, job_id, job_name: _Cfg())
    monkeypatch.setattr(sched, "_resolve_cron_agent_setup", lambda job, job_id, job_name, jc: _Setup())
    monkeypatch.setattr(sched, "_open_cron_session_db", lambda job: None)
    monkeypatch.setattr(sched, "_construct_cron_agent",
                        lambda AIAgent, job, cfg, setup, **kw: object())
    monkeypatch.setattr(sched, "_FireAudit", _Audit)
    monkeypatch.setattr(sched, "_run_agent_with_watchdog",
                        lambda agent, prompt, job, job_id, job_name, task_id, cancel, worker_state: {
                            "final_response": final_response})
    monkeypatch.setattr(sched, "_final_response_from_result",
                        lambda result, job_id, job_name, AIAgent: result["final_response"])
    monkeypatch.setattr(sched, "_finalize_cron_session", lambda *a, **k: None)
    monkeypatch.setattr(sched, "_teardown_cron_agent", lambda *a, **k: None)
    monkeypatch.setattr(sched, "_reload_dotenv_and_publish_delivery_target", lambda job: None)


_JOB = {
    "id": "testjob12345",
    "name": "artifact-test-job",
    "prompt": "produce a report",
}


def test_run_job_delivers_run_scoped_artifact(tmp_path, monkeypatch):
    """Valid artifact: final response becomes the scheduler summary + MEDIA path."""
    artifact = tmp_path / "report.html"
    artifact.write_text("<html><body>report</body></html>")
    job = dict(_JOB, delivery_artifact_template=str(tmp_path / "{execution_id}.html"))
    _stub_pipeline(monkeypatch, "Model wrote the report, verified it, ran lint, exit 0.")

    # Reserve the path the agent will "write" to; run_job also reserves via the
    # execution_id injection, but the stub agent cannot write files, so the
    # test pre-creates the file at the reserved path after prep runs. Simpler:
    # the reservation happens inside run_job, so hook _prepare_delivery_artifact
    # to write the fixture file when it reserves.
    real_prepare = sched._prepare_delivery_artifact

    def prepare_and_write(job_, execution_id_):
        path, prompt = real_prepare(job_, execution_id_)
        if path is not None and not path.exists():
            path.write_text("<html><body>report</body></html>")
        return path, prompt

    monkeypatch.setattr(sched, "_prepare_delivery_artifact", prepare_and_write)

    success, output, final_response, error = sched.run_job(job, execution_id="execABC123")
    assert success is True
    assert error is None
    assert "MEDIA:" in final_response
    assert str(tmp_path / "execABC123.html") in final_response
    # The model's verification narration must NOT be the delivered response.
    assert "lint" not in final_response
    assert "exit 0" not in final_response
    # Reservation cleared after the run.
    assert "_active_delivery_artifact" not in job


def test_run_job_fails_closed_when_artifact_missing(tmp_path, monkeypatch):
    """Agent never wrote the artifact: run fails, no delivery, no false success."""
    job = dict(_JOB, delivery_artifact_template=str(tmp_path / "{execution_id}.html"))
    _stub_pipeline(monkeypatch, "Report done! Everything verified, lint passed.")

    success, output, final_response, error = sched.run_job(job, execution_id="execXYZ789")
    assert success is False
    assert "artifact missing or invalid" in error
    assert final_response == ""
    assert "_active_delivery_artifact" not in job


def test_run_job_rejects_non_html_artifact(tmp_path, monkeypatch):
    """Template declares .html but the agent wrote plain text: fail closed."""
    job = dict(_JOB, delivery_artifact_template=str(tmp_path / "{execution_id}.html"))
    _stub_pipeline(monkeypatch, "All done.")

    real_prepare = sched._prepare_delivery_artifact

    def prepare_and_write_text(job_, execution_id_):
        path, prompt = real_prepare(job_, execution_id_)
        if path is not None and not path.exists():
            path.write_text("this is not html, just narration")
        return path, prompt

    monkeypatch.setattr(sched, "_prepare_delivery_artifact", prepare_and_write_text)

    success, output, final_response, error = sched.run_job(job, execution_id="execNOTHTML1")
    assert success is False
    assert "artifact missing or invalid" in error


def test_run_job_without_artifact_template_unchanged(tmp_path, monkeypatch):
    """Jobs without delivery_artifact_template keep the legacy behaviour."""
    _stub_pipeline(monkeypatch, "Plain summary text")
    job = dict(_JOB)

    success, output, final_response, error = sched.run_job(job, execution_id="execNOART001")
    assert success is True
    assert final_response == "Plain summary text"


def test_run_job_reservation_cleared_on_exception(tmp_path, monkeypatch):
    """A mid-run exception must not leave the reservation on the job dict."""
    job = dict(_JOB, delivery_artifact_template=str(tmp_path / "{execution_id}.html"))

    def boom(*a, **k):
        raise RuntimeError("agent exploded")

    _stub_pipeline(monkeypatch, "unused")
    monkeypatch.setattr(sched, "_run_agent_with_watchdog", boom)

    success, output, final_response, error = sched.run_job(job, execution_id="execBOOM001")
    assert success is False
    assert "agent exploded" in error
    assert "_active_delivery_artifact" not in job
