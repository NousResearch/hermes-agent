"""Tests for Self-Evolving Swarm (PR #40125).

Verifies:
1. Orchestrator initialization and execution loop without core.* import errors.
2. WebAgent research fallback.
3. ValidatorAgent code quality scoring.
4. ToolRegistry profile-isolated storage and Hermes runtime tool registration.
5. SKILL.md skill metadata formatting and location.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from agents.orchestrator import Orchestrator, run_swarm
from agents.tool_registry import ToolRegistry
from agents.validator_agent import ValidatorAgent
from agents.web_agent import WebAgent
from hermes_constants import get_hermes_home
from tools.registry import registry as hermes_registry


def test_tool_registry_profile_isolation_and_registration(tmp_path):
    registry = ToolRegistry(registry_root=tmp_path / "tools_registry")

    tool_code = """
def sample_calc_tool(a: int, b: int) -> int:
    \"\"\"Sample calc tool.\"\"\"
    return a + b
"""
    res = registry.register_new_tool(
        tool_name="sample_calc_tool",
        code=tool_code,
        description="Sample calculation tool",
    )

    assert res["name"] == "sample_calc_tool"
    assert res["version"] == 1

    tool_info = registry.get_tool("sample_calc_tool")
    assert tool_info is not None
    assert tool_info["name"] == "sample_calc_tool"

    # Verify registered in Hermes central registry
    assert "sample_calc_tool" in hermes_registry._tools


def test_validator_agent_scoring():
    validator = ValidatorAgent()

    # Good output
    res_good = validator.run("def add(x: int, y: int) -> int:\n    return x + y", requirements="Add numbers")
    assert res_good.success is True
    assert res_good.metadata.get("passed") is True
    assert res_good.metadata.get("score", 0) >= 0.70

    # Bad output with TODO
    res_bad = validator.run("TODO: implement add function", requirements="Add numbers")
    assert res_bad.metadata.get("score", 1.0) < 0.85


def test_web_agent_fallback():
    web_agent = WebAgent()
    res = web_agent.run("Python async best practices")
    assert res.success is True
    assert "RESEARCH FINDINGS" in res.output or "Python" in res.output


def test_orchestrator_execution_loop(tmp_path):
    orch = Orchestrator(registry_root=tmp_path / "tools_registry")
    res = orch.run("Build a string helper utility")

    assert res.success is True
    assert "SOLUTION" in res.output
    assert "VALIDATION" in res.output
    assert "duration" in res.metadata


def test_skill_md_packaging():
    skill_path = (
        Path(__file__).resolve().parent.parent.parent
        / "optional-skills"
        / "autonomous-ai-agents"
        / "self-evolving-swarm"
        / "SKILL.md"
    )
    assert skill_path.exists()
    content = skill_path.read_text(encoding="utf-8")
    assert "name: self-evolving-swarm" in content
    assert "config.yaml" in content
