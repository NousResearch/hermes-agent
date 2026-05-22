"""Tests for agent/skill_utils.py — extract_skill_conditions metadata handling."""

from agent.skill_utils import extract_skill_conditions


def test_metadata_as_dict_with_hermes():
    """Normal case: metadata is a dict containing hermes keys."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": ["toolset_a"],
                "requires_toolsets": ["toolset_b"],
                "fallback_for_tools": ["tool_x"],
                "requires_tools": ["tool_y"],
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result["fallback_for_toolsets"] == ["toolset_a"]
    assert result["requires_toolsets"] == ["toolset_b"]
    assert result["fallback_for_tools"] == ["tool_x"]
    assert result["requires_tools"] == ["tool_y"]


def test_metadata_as_string_does_not_crash():
    """Bug case: metadata is a non-dict truthy value (e.g. a YAML string)."""
    frontmatter = {"metadata": "some text"}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
        "model": "",
    }


def test_metadata_as_none():
    """metadata key is present but set to null/None."""
    frontmatter = {"metadata": None}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
        "model": "",
    }


def test_metadata_missing_entirely():
    """metadata key is absent from frontmatter."""
    frontmatter = {"name": "my-skill", "description": "Does stuff."}
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
        "model": "",
    }


def test_hermes_model_field():
    """hermes.model declares the preferred model for delegation to this skill."""
    frontmatter = {
        "metadata": {"hermes": {"model": "anthropic/claude-opus-4-7"}}
    }
    result = extract_skill_conditions(frontmatter)
    assert result["model"] == "anthropic/claude-opus-4-7"


def test_hermes_model_field_absent():
    """hermes.model defaults to empty string when not declared."""
    frontmatter = {
        "metadata": {"hermes": {"requires_toolsets": ["web"]}}
    }
    result = extract_skill_conditions(frontmatter)
    assert result["model"] == ""
