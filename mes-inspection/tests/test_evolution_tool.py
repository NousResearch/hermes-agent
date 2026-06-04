"""进化工具单元测试。"""

import json
import pytest
from unittest.mock import patch, MagicMock
from evolution.evolution_tool import mes_evolution, EVOLUTION_SCHEMA


class TestEvolutionTool:
    def test_schema_structure(self):
        props = EVOLUTION_SCHEMA["parameters"]["properties"]
        assert "action" in props
        assert "evolve" in props["action"]["enum"]

    def test_list_action(self):
        result = mes_evolution(action="list")
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert len(parsed["skills"]) == 6

    @patch("evolution.evolution_tool.EvolutionRunner")
    def test_evolve_action(self, mock_cls):
        mock_runner = MagicMock()
        mock_runner.evolve_skill.return_value = {"success": True, "skill_name": "mes-nginx-check", "stdout": "OK", "stderr": "", "returncode": 0}
        mock_cls.return_value = mock_runner
        result = mes_evolution(action="evolve", skill_name="mes-nginx-check")
        parsed = json.loads(result)
        assert parsed["success"] is True

    @patch("evolution.evolution_tool.EvolutionRunner")
    def test_evolve_all_action(self, mock_cls):
        mock_runner = MagicMock()
        mock_runner.evolve_all.return_value = [{"success": True, "skill_name": "mes-nginx-check", "stdout": "", "stderr": "", "returncode": 0}]
        mock_runner.format_report.return_value = "报告"
        mock_cls.return_value = mock_runner
        result = mes_evolution(action="evolve_all")
        parsed = json.loads(result)
        assert parsed["success"] is True

    def test_missing_skill_name(self):
        result = mes_evolution(action="evolve")
        parsed = json.loads(result)
        assert parsed["success"] is False
        assert "skill_name" in parsed["error"]

    def test_unknown_action(self):
        result = mes_evolution(action="invalid")
        parsed = json.loads(result)
        assert parsed["success"] is False


class TestCronJobsConfig:
    def test_evolution_cron_job_exists(self):
        import pathlib
        cron_path = pathlib.Path(__file__).parent.parent / "config" / "cron_jobs.json"
        with open(cron_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        names = [j["name"] for j in config["jobs"]]
        assert "mes-evolution" in names

    def test_evolution_cron_job_fields(self):
        import pathlib
        cron_path = pathlib.Path(__file__).parent.parent / "config" / "cron_jobs.json"
        with open(cron_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        evo_job = next(j for j in config["jobs"] if j["name"] == "mes-evolution")
        assert "schedule" in evo_job
        assert "prompt" in evo_job
