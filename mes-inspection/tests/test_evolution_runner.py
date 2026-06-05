"""进化执行器单元测试。"""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from evolution.evolution_runner import EvolutionRunner, MES_SKILLS


class TestEvolutionRunner:
    def test_init_defaults(self):
        runner = EvolutionRunner(config={
            "iterations": 5, "eval_source": "synthetic",
            "optimizer_model": "openai/gpt-4.1", "eval_model": "openai/gpt-4.1-mini",
            "hermes_repo": "/tmp/test-repo", "skills_dir": "",
        })
        assert runner.iterations == 5
        assert runner.eval_source == "synthetic"

    def test_build_dspy_env_custom(self):
        runner = EvolutionRunner(config={
            "optimizer_model": "openai/Qwen3-235B-A22B-w8a8",
            "eval_model": "openai/Qwen3-235B-A22B-w8a8",
            "api_key_env": "QWEN_API_KEY", "base_url": "http://10.0.0.5:8000/v1",
        })
        env = runner._build_env()
        assert env.get("OPENAI_API_BASE") == "http://10.0.0.5:8000/v1"

    def test_format_result_success(self):
        runner = EvolutionRunner(config={})
        result = runner._format_result("mes-nginx-check", 0, "Evolution improved", "")
        assert result["success"] is True
        assert result["skill_name"] == "mes-nginx-check"

    def test_format_result_failure(self):
        runner = EvolutionRunner(config={})
        result = runner._format_result("mes-nginx-check", 1, "", "Error")
        assert result["success"] is False

    def test_list_evolvable_skills(self):
        runner = EvolutionRunner(config={})
        skills = runner.list_evolvable_skills()
        assert len(skills) == 6
        assert "mes-nginx-check" in skills

    def test_mes_skills_constant(self):
        assert len(MES_SKILLS) == 6
        assert all(s.startswith("mes-") for s in MES_SKILLS)
