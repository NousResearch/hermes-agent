"""Behavior contract for the per-job cron run limits (``max_turns`` / ``max_tokens``).

The 2026-11-02 incident had a cron job whose model degenerated into a
213 KB repetition loop. Two knobs bound that class of failure at the source:

  * ``max_turns``  — how many agent iterations one run may take.
  * ``max_tokens`` — the response-token cap sent with the run's model request.

Both were stored on the job record but never read at run time (only the global
``agent.max_turns`` was consulted), so a job could not constrain itself. This
file pins the contract:

  * a per-job value wins over the global config value;
  * an absent (or explicitly null) per-job value falls back to global config;
  * both are persisted by the create/update chokepoints and validated there
    (a non-positive or non-numeric value is rejected BEFORE anything is
    stored, so a bad cap can never make every run fail);
  * the limits actually reach the constructed agent.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cron.jobs import create_job, update_job
from cron.scheduler import (
    _CronAgentSetup,
    _construct_cron_agent,
    _resolve_cron_agent_setup,
)


# --- persistence + validation ---------------------------------------------

class TestJobLimitPersistence:
    def test_create_persists_both_limits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        job = create_job(
            prompt="report", schedule="every 1h", name="limited",
            max_turns=5, max_tokens=2000)
        assert job["max_turns"] == 5
        assert job["max_tokens"] == 2000

    def test_create_without_limits_leaves_them_unset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        job = create_job(prompt="report", schedule="every 1h", name="unlimited")
        assert job.get("max_turns") is None
        assert job.get("max_tokens") is None

    def test_update_persists_both_limits(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        job = create_job(prompt="report", schedule="every 1h", name="limited")
        updated = update_job(job["id"], {"max_turns": 7, "max_tokens": 3000})
        assert updated["max_turns"] == 7
        assert updated["max_tokens"] == 3000

    @pytest.mark.parametrize("bad", [0, -1, "many", 1.5, True])
    def test_non_positive_or_non_numeric_turn_limit_is_rejected(self, tmp_path, monkeypatch, bad):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(ValueError):
            create_job(prompt="report", schedule="every 1h", name="bad", max_turns=bad)

    @pytest.mark.parametrize("bad", [0, -1, "many", 1.5, True])
    def test_non_positive_or_non_numeric_token_cap_is_rejected(self, tmp_path, monkeypatch, bad):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        with pytest.raises(ValueError):
            create_job(prompt="report", schedule="every 1h", name="bad", max_tokens=bad)

    def test_update_rejects_an_invalid_limit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        job = create_job(prompt="report", schedule="every 1h", name="bad")
        with pytest.raises(ValueError):
            update_job(job["id"], {"max_turns": 0})


# --- resolution at run time -----------------------------------------------

def _job_context(cfg):
    """The minimal ``jc`` the setup resolver reads: ``cfg`` and ``model``."""
    return SimpleNamespace(cfg=cfg, model="gpt-test")


class TestRunTimeResolution:
    """``_resolve_cron_agent_setup`` is where the job record meets the agent."""

    @staticmethod
    def _setup(job, cfg):
        return _resolve_cron_agent_setup(job, "job-1", "name", _job_context(cfg))

    def test_job_level_turn_limit_beats_the_global_config(self):
        setup = self._setup({"max_turns": 5}, {"agent": {"max_turns": 90}})
        assert setup.max_iterations == 5

    def test_job_level_turn_limit_beats_the_top_level_config(self):
        setup = self._setup({"max_turns": 5}, {"max_turns": 90})
        assert setup.max_iterations == 5

    def test_absent_job_limit_falls_back_to_the_global_config(self):
        setup = self._setup({}, {"agent": {"max_turns": 90}})
        assert setup.max_iterations == 90

    def test_absent_job_limit_falls_back_to_the_top_level_config(self):
        setup = self._setup({}, {"max_turns": 90})
        assert setup.max_iterations == 90

    def test_a_null_job_limit_falls_back_to_the_global_config(self):
        """Legacy job records carry an explicit ``max_turns: null`` — not an override."""
        setup = self._setup({"max_turns": None}, {"agent": {"max_turns": 90}})
        assert setup.max_iterations == 90

    def test_job_level_token_cap_reaches_the_setup(self):
        setup = self._setup({"max_tokens": 2000}, {})
        assert setup.max_tokens == 2000

    def test_absent_token_cap_is_left_to_the_provider_default(self):
        setup = self._setup({}, {})
        assert setup.max_tokens is None


# --- the limits reach the constructed agent ---------------------------------

class TestLimitsReachTheAgent:
    def test_constructed_agent_receives_both_limits(self):
        agent_cls = MagicMock()
        setup = _CronAgentSetup(
            model="gpt-test",
            runtime={},
            prefill_messages=None,
            max_iterations=5,
            max_tokens=2000,
            reasoning_config=None,
            fallback_model=None,
            credential_pool=None,
        )
        _construct_cron_agent(
            agent_cls, {"id": "job-1"}, {}, setup,
            workdir=None, session_id=None, session_db=None)

        kwargs = agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 5
        assert kwargs["max_tokens"] == 2000

    def test_a_job_without_limits_leaves_the_agent_defaults_alone(self):
        agent_cls = MagicMock()
        setup = _CronAgentSetup(
            model="gpt-test",
            runtime={},
            prefill_messages=None,
            max_iterations=90,
            max_tokens=None,
            reasoning_config=None,
            fallback_model=None,
            credential_pool=None,
        )
        _construct_cron_agent(
            agent_cls, {"id": "job-1"}, {}, setup,
            workdir=None, session_id=None, session_db=None)

        kwargs = agent_cls.call_args.kwargs
        assert kwargs["max_iterations"] == 90
        assert kwargs["max_tokens"] is None
