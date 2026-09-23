"""Per-job ``max_turns`` overrides the global iteration cap.

Long multi-stage jobs (collect -> analyze -> write -> publish pipelines)
legitimately need more tool rounds than the global ``agent.max_turns``
default; the global cap keeps protecting every other job. The wall-clock
bound (``agent.gateway_timeout``) remains the real safety net either way.
"""

from __future__ import annotations

import cron.scheduler as scheduler_mod


class TestPerJobMaxTurns:
    def test_job_field_overrides_global(self):
        setup = scheduler_mod._resolve_cron_agent_setup(
            {"max_turns": 110},
            "job-1",
            "long pipeline",
            _JC({"agent": {"max_turns": 60}}),
        )
        assert setup.max_iterations == 110

    def test_global_applies_without_job_field(self):
        setup = scheduler_mod._resolve_cron_agent_setup(
            {"name": "normal"},
            "job-2",
            "normal",
            _JC({"agent": {"max_turns": 60}}),
        )
        assert setup.max_iterations == 60

    def test_job_none_falls_back_to_global(self):
        setup = scheduler_mod._resolve_cron_agent_setup(
            {"max_turns": None},
            "job-3",
            "explicit none",
            _JC({"agent": {"max_turns": 60}}),
        )
        assert setup.max_iterations == 60


class _JC:
    """Minimal stand-in for the job-cfg bundle _resolve_cron_agent_setup reads."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = "test-model"
