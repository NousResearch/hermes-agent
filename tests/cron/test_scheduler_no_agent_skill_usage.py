"""Cron jobs refresh usage for declared skills when those skills are actually loaded."""

import json
from unittest.mock import patch

from cron.scheduler import _run_no_agent_job
from cron.scheduler_prompt import _build_job_prompt


class TestCronPromptSkillUsageBump:
    """Agent-backed cron jobs bump usage while loading skills into the prompt."""

    def test_loaded_skill_bumps_canonical_name_from_skill_view(self):
        """Cron may load by path/alias, but usage must refresh the canonical skill name."""

        def _skill_view(name: str) -> str:
            return json.dumps({
                "success": True,
                "name": "canonical-skill",
                "content": f"Content for {name}.",
            })

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use") as mock_bump:
            _build_job_prompt({
                "id": "cron-task",
                "skills": ["/tmp/profile/skills/category/skill-dir"],
                "prompt": "go",
            })

        mock_bump.assert_called_once_with("canonical-skill", task_id="cron-task")

    def test_bundle_members_bump_canonical_names(self):
        """A cron skill bundle loads member skills; each member must get a usage bump."""

        def _load_skill_payload(identifier: str, task_id: str | None = None):
            return ({"success": True, "name": f"canonical-{identifier}", "content": identifier}, None,
                    f"canonical-{identifier}")

        with patch("agent.skill_bundles.resolve_bundle_command_key", return_value="/daily"), \
             patch("agent.skill_bundles.get_skill_bundles", return_value={
                 "/daily": {"name": "daily", "skills": ["alpha", "beta"], "instruction": ""},
             }), \
             patch("agent.skill_commands._load_skill_payload", side_effect=_load_skill_payload), \
             patch("tools.skill_usage.bump_use") as mock_bump:
            _build_job_prompt({
                "id": "cron-task",
                "skills": ["daily"],
                "prompt": "go",
            })

        assert [call.args[0] for call in mock_bump.call_args_list] == [
            "canonical-alpha", "canonical-beta",
        ]
        assert all(call.kwargs == {"task_id": "cron-task"} for call in mock_bump.call_args_list)


class TestNoAgentSkillUsageBump:
    """The script-only path skips prompt assembly, so it must bump declared skills itself."""

    def test_run_no_agent_job_bumps_canonical_declared_skill(self):
        def _skill_view(name: str) -> str:
            return json.dumps({
                "success": True,
                "name": "canonical-skill",
                "content": f"Content for {name}.",
            })

        job = {
            "id": "watchdog-1",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
            "skills": ["/tmp/profile/skills/category/skill-dir"],
        }

        with patch("tools.skills_tool.skill_view", side_effect=_skill_view), \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-1", "watchdog", None)

        assert ok is True
        mock_bump.assert_called_once_with("canonical-skill", task_id="watchdog-1")

    def test_run_no_agent_job_bumps_bundle_members(self):
        def _load_skill_payload(identifier: str, task_id: str | None = None):
            return ({"success": True, "name": f"canonical-{identifier}", "content": identifier}, None,
                    f"canonical-{identifier}")

        job = {
            "id": "watchdog-2",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
            "skills": ["daily"],
        }

        with patch("agent.skill_bundles.resolve_bundle_command_key", return_value="/daily"), \
             patch("agent.skill_bundles.get_skill_bundles", return_value={
                 "/daily": {"name": "daily", "skills": ["alpha", "beta"], "instruction": ""},
             }), \
             patch("agent.skill_commands._load_skill_payload", side_effect=_load_skill_payload), \
             patch("tools.skills_tool.skill_view") as mock_view, \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-2", "watchdog", None)

        assert ok is True
        mock_view.assert_not_called()
        assert [call.args[0] for call in mock_bump.call_args_list] == [
            "canonical-alpha", "canonical-beta",
        ]
        assert all(call.kwargs == {"task_id": "watchdog-2"} for call in mock_bump.call_args_list)

    def test_no_agent_job_skips_bump_for_unresolvable_skill(self):
        """A mistyped skill name must not mint a ghost usage record (bump_use creates one)."""

        def _skill_view(name: str) -> str:
            return json.dumps({"success": False, "error": f"no skill named {name}"})

        job = {
            "id": "watchdog-3",
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
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-3", "watchdog", None)

        assert ok is True
        mock_bump.assert_not_called()

    def test_no_agent_job_without_declared_skills_bumps_nothing(self):
        job = {
            "id": "watchdog-4",
            "name": "watchdog",
            "no_agent": True,
            "script": "probe.py",
        }

        with patch("tools.skills_tool.skill_view") as mock_view, \
             patch("tools.skill_usage.bump_use") as mock_bump, \
             patch("cron.scheduler._run_job_script_with_claim_heartbeat",
                   return_value=(True, "stdout output")), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            ok, _doc, _deliver, _out = _run_no_agent_job(job, "watchdog-4", "watchdog", None)

        assert ok is True
        mock_view.assert_not_called()
        mock_bump.assert_not_called()
