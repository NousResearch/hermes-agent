"""Per-job ``script_failure_policy`` (#123441).

A script-gated agent job whose pre-run script exits non-zero always wakes the
agent (the failure text is injected as a "Script Error" context block), and the
``wakeAgent: false`` gate only applies to successful scripts — so a script can
never signal "this run failed; do not start the model".

With ``script_failure_policy: "fail"`` a non-zero pre-run script ends the run
before any agent is constructed: the run records ok=False, no model is
invoked, and the failure is routed through the normal failure-delivery lane.
The default policy (``agent``, also the behavior for jobs that do not set the
field) preserves today's behavior byte-identically. The field is validated at
persist time (create/update) and stored only when explicitly set, so existing
job records stay byte-identical.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import cron.scheduler as scheduler
from cron import scheduler_script as sched_script


@pytest.fixture
def cron_env(tmp_path, monkeypatch):
    """Isolated cron environment with temp HERMES_HOME."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "cron").mkdir()
    (hermes_home / "cron" / "output").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import cron.jobs as jobs_mod
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")

    return hermes_home



@pytest.fixture(autouse=True)
def _stub_runtime_provider():
    """Stub ``resolve_runtime_provider``: hermetic CI has no API keys, and
    ``run_job`` resolves the runtime provider before constructing AIAgent."""
    fake_runtime = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "source": "stub",
        "requested_provider": None,
    }
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=fake_runtime,
    ):
        yield


def _make_job(**extra):
    job = {
        "id": "job_sfp",
        "name": "sfp-test",
        "prompt": "Do a thing",
        "schedule": "*/5 * * * *",
        "script": "check.py",
    }
    job.update(extra)
    return job


def _run_with_script(job, script_result):
    agent = MagicMock()
    agent.run_conversation = MagicMock(return_value={"final_response": "ok", "messages": []})
    with patch.object(sched_script, "_run_job_script", return_value=script_result), \
         patch("run_agent.AIAgent", return_value=agent) as agent_cls:
        result = scheduler.run_job(job)
    return result, agent_cls


class TestScriptFailurePolicyDefault:
    def test_failure_wakes_agent_by_default(self):
        """Default (no policy set): a failed script wakes the agent, which sees
        the failure text — today's behavior, pinned."""
        (success, doc, final, err), agent_cls = _run_with_script(
            _make_job(), (False, "boom: data collection failed"))
        agent_cls.assert_called_once()
        assert success is True
        prompt_arg = (
            agent_cls.call_args and None) or agent_cls.return_value.run_conversation.call_args
        prompt_text = prompt_arg.args[0] if prompt_arg.args else prompt_arg.kwargs.get("user_message", "")
        assert "boom" in prompt_text


class TestScriptFailurePolicyFail:
    def test_fail_policy_skips_agent_and_fails_run(self):
        """policy=fail + non-zero script: no agent, ok=False, error names the policy."""
        (success, doc, final, err), agent_cls = _run_with_script(
            _make_job(script_failure_policy="fail"), (False, "boom: data collection failed"))
        agent_cls.assert_not_called()
        assert success is False
        assert "script_failure_policy=fail" in (err or "")
        assert "Pre-run script failed" in (err or "")
        assert "the agent was NOT run" in (doc or "")
        assert "boom" in (doc or "")

    def test_fail_policy_success_still_wakes_agent(self):
        """policy=fail only changes the FAILURE path; successful scripts run the
        agent exactly as before (stdout still injected)."""
        script_output = '{"data": {"new": 3}}'
        (success, doc, final, err), agent_cls = _run_with_script(
            _make_job(script_failure_policy="fail"), (True, script_output))
        agent_cls.assert_called_once()
        assert success is True
        assert err is None

    def test_fail_policy_wake_gate_still_applies_on_success(self):
        """policy=fail + successful script returning wakeAgent=false: silent run,
        same as without the policy."""
        (success, doc, final, err), agent_cls = _run_with_script(
            _make_job(script_failure_policy="fail"),
            (True, '{"wakeAgent": false}'))
        agent_cls.assert_not_called()
        assert success is True
        from cron.scheduler import SILENT_MARKER
        assert final == SILENT_MARKER


class TestScriptFailurePolicyPersistence:
    def test_invalid_policy_rejected_on_create(self, cron_env):
        from cron.jobs import create_job
        with pytest.raises(ValueError, match="script_failure_policy"):
            create_job(prompt="x", schedule="*/5 * * * *", script_failure_policy="nope")

    def test_policy_absent_when_unset(self, cron_env):
        """Jobs that do not set the field must not grow a new key — records stay
        byte-identical to pre-feature stores."""
        from cron.jobs import create_job
        job = create_job(prompt="x", schedule="*/5 * * * *")
        assert "script_failure_policy" not in job

    def test_policy_roundtrip_and_reject_on_update(self, cron_env):
        from cron.jobs import create_job, update_job
        job = create_job(prompt="x", schedule="*/5 * * * *")
        with pytest.raises(ValueError, match="script_failure_policy"):
            update_job(job["id"], {"script_failure_policy": "nope"})
        updated = update_job(job["id"], {"script_failure_policy": "fail"})
        assert updated["script_failure_policy"] == "fail"
        cleared = update_job(job["id"], {"script_failure_policy": ""})
        assert not cleared.get("script_failure_policy")
