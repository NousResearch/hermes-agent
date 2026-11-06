"""no_agent script jobs never build a prompt, so declared-skill usage must be bumped on the
script path — otherwise the curator sees no activity and archives them (#136030)."""

import json
from unittest.mock import patch

from cron.scheduler import _run_no_agent_job


class TestNoAgentSkillUsageBump:
    """Verify the no_agent script path bumps declared skills like the agent prompt path does."""

    def test_run_no_agent_job_bumps_declared_skills(self):
        def _skill_view(name: str) -> str:
            return json.dumps({"success": True, "content": f"Content for {name}."})

        job = {
            "id": "watchdog-1",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
            "skills": ["alpha", "beta"],
        }

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-1", "watchdog", None)

        assert ok is True
        assert mock_bump.call_count == 2
        assert all(
            call.kwargs == {"task_id": "watchdog-1"}
            for call in mock_bump.call_args_list
        )

    def test_no_agent_job_skips_bump_for_unresolvable_skill(self):
        """A mistyped skill name must not mint a ghost usage record (bump_use creates one)."""

        def _skill_view(name: str) -> str:
            return json.dumps({"success": False, "error": f"no skill named {name}"})

        job = {
            "id": "watchdog-2",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
            "skills": ["ghost-skill"],
        }

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-2", "watchdog", None)

        assert ok is True
        mock_bump.assert_not_called()

    def test_no_agent_job_without_declared_skills_bumps_nothing(self):
        job = {
            "id": "watchdog-3",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
        }

        with patch("tools.skills_tool.skill_view") as mock_view, \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-3", "watchdog", None)

        assert ok is True
        mock_view.assert_not_called()
        mock_bump.assert_not_called()
