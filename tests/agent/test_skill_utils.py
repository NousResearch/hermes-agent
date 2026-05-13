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
    }


# -----------------------------------------------------------------------
# yaml_load — SafeLoader (not CSafeLoader) for untrusted YAML
#
# CSafeLoader is a C extension. yaml.SafeLoader is pure Python.
# Using SafeLoader explicitly:
#   1. Removes C-extension attack surface (malformed YAML causing segfaults)
#   2. Prevents !!python/object and !!python/exe tags from executing code
#   3. Keeps behaviour consistent across Python versions / build configurations
# -----------------------------------------------------------------------


def test_yaml_load_rejects_python_object_tag():
    """SafeLoader raises ConstructorError for !!python/object — not SafeConstructor."""
    from agent.skill_utils import yaml_load
    import yaml

    payload = "!!python/object:os.system ['echo PWNED']"
    try:
        result = yaml_load(payload)
        # If it returns rather than raising, the tag must not have executed.
        assert "PWNED" not in str(result)
    except yaml.constructor.ConstructorError:
        pass  # SafeLoader rejects the tag — this is the expected behaviour.


def test_yaml_load_rejects_python_exec_tag():
    """!!python/exec (direct code execution) is also refused."""
    from agent.skill_utils import yaml_load
    import yaml

    payload = "!!python/exec 'print(1)'"
    try:
        result = yaml_load(payload)
        assert "1" not in str(result)
    except yaml.constructor.ConstructorError:
        pass  # Rejected — correct.


def test_yaml_load_parses_normal_yaml():
    """Normal YAML parses correctly after the SafeLoader change."""
    from agent.skill_utils import yaml_load

    yaml_content = """
name: my-skill
version: 1.0
metadata:
  hermes:
    requires_toolsets: [toolset_a]
"""
    result = yaml_load(yaml_content)
    assert result["name"] == "my-skill"
    assert result["version"] == 1.0  # YAML unquoted numerals are parsed as numbers
    assert result["metadata"]["hermes"]["requires_toolsets"] == ["toolset_a"]
