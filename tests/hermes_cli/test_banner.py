"""Tests for research-aware CLI banner rendering."""

from __future__ import annotations

import pytest
from rich.console import Console

pytest.importorskip("prompt_toolkit")

from hermes_cli.banner import build_welcome_banner, get_research_skill_names, is_research_variant


def test_is_research_variant_from_toolsets(monkeypatch):
    monkeypatch.delenv("HERMES_RESEARCH_ENABLED", raising=False)
    assert is_research_variant(["research"]) is True
    assert is_research_variant(["web"]) is False


def test_get_research_skill_names_includes_tinker():
    names = get_research_skill_names(
        {
            "research": ["autonomous-llm-research", "research-idea-generation"],
            "mlops": ["tinker", "axolotl"],
        }
    )
    assert "autonomous-llm-research" in names
    assert "research-idea-generation" in names
    assert "tinker" in names


def test_build_welcome_banner_shows_research_sections(monkeypatch):
    console = Console(record=True, width=180)
    monkeypatch.setattr(
        "hermes_cli.banner.get_available_skills",
        lambda: {
            "research": [
                "autonomous-llm-research",
                "research-idea-generation",
                "literature-to-experiment",
                "eval-and-ablation",
                "research-reporting",
            ],
            "mlops": ["tinker"],
        },
    )
    monkeypatch.setattr(
        "model_tools.check_tool_availability",
        lambda quiet=True: ([], []),
    )

    tools = [
        {"function": {"name": "research_state"}},
        {"function": {"name": "research_loop"}},
        {"function": {"name": "research_manager"}},
        {"function": {"name": "tinker_posttrain"}},
        {"function": {"name": "literature"}},
        {"function": {"name": "dataset"}},
        {"function": {"name": "judge"}},
        {"function": {"name": "evaluation"}},
    ]

    build_welcome_banner(
        console=console,
        model="gpt-test",
        cwd="/tmp",
        tools=tools,
        enabled_toolsets=["research"],
        session_id="sess_test",
        get_toolset_for_tool=lambda name: "research",
    )

    output = console.export_text()
    assert "Hermes Research Agent" in output
    assert "Research Tools" in output
    assert "research_manager" in output
    assert "Research Skills" in output
    assert "autonomous-llm-research" in output
