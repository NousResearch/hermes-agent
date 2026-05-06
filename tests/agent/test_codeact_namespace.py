import pytest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from agent.codeact_namespace import (
    build_codeact_workflow_guidance,
    build_tool_namespace_source,
)
from agent.codeact_skill_injector import SkillNamespaceInjector
from tools.registry import ToolRegistry


@pytest.fixture
def mock_skill_utils():
    with patch("agent.codeact_skill_injector.get_all_skill_frontmatters") as mock:
        yield mock


def test_build_tool_namespace_source_with_skills(mock_skill_utils):
    mock_skill_utils.return_value = {
        "myskill": {
            "name": "myskill",
            "description": "Does something cool.",
            "codeact_fn": "my_skill_func",
        }
    }
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = []
    injector = SkillNamespaceInjector(registry)
    source = build_tool_namespace_source(registry, skill_injector=injector)
    assert "def my_skill_func(**kwargs):" in source
    assert '"""Does something cool."""' in source
    assert "_call_tool('__skill__'" in source


def test_build_tool_namespace_source_without_injector():
    """When no injector is passed, no skill stubs are generated."""
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = []
    source = build_tool_namespace_source(registry)
    assert "Callable Skills" not in source


def test_build_tool_namespace_source_skill_names_in_protected(mock_skill_utils):
    """Skill function names should appear in __protected__."""
    mock_skill_utils.return_value = {
        "arxiv": {
            "name": "arxiv",
            "description": "Search arXiv.",
            "codeact_fn": "arxiv_search",
        }
    }
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = []
    injector = SkillNamespaceInjector(registry)
    source = build_tool_namespace_source(registry, skill_injector=injector)
    assert '"arxiv_search"' in source  # in __protected__ list
    assert "__protected__" in source


def _entry(name, toolset="vision"):
    return SimpleNamespace(
        name=name,
        toolset=toolset,
        schema={
            "name": name,
            "description": f"{name} description",
            "parameters": {"type": "object", "properties": {}},
        },
    )


def test_codeact_workflow_guidance_includes_vision_recipe():
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = [_entry("vision_analyze")]

    guidance = build_codeact_workflow_guidance(registry)

    assert "Image/OCR/translation" in guidance
    assert "vision_analyze(image_url=path" in guidance
    assert "Do not start with PIL/OCR/package installs" in guidance


def test_help_registry_includes_workflow_guidance():
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = [_entry("vision_analyze")]

    source = build_tool_namespace_source(registry)

    assert "_WORKFLOW_GUIDANCE" in source
    assert "Image/OCR/translation" in source


def test_research_recipe_is_injected_and_calls_research_gather():
    registry = MagicMock(spec=ToolRegistry)
    registry._snapshot_entries.return_value = [_entry("research_gather", "research_search")]

    source = build_tool_namespace_source(registry)
    calls = []

    def fake_call_tool(name, args):
        calls.append((name, args))
        return '{"success": true, "sources": []}'

    namespace = {"_call_tool": fake_call_tool}
    exec(source, namespace)

    result = namespace["research_web"](
        "current Detroit Pistons starting five",
        freshness="latest",
        max_sources=3,
    )

    assert result["success"] is True
    assert calls == [
        (
            "research_gather",
            {
                "question": "current Detroit Pistons starting five",
                "topic_type": "auto",
                "freshness": "latest",
                "depth": "thorough",
                "max_pages": 3,
            },
        )
    ]
    assert "research_web" in namespace["help"]()
    assert "research_web" in namespace["__protected__"]
