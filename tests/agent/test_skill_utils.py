"""Tests for agent/skill_utils.py — extract_skill_conditions metadata handling."""

from agent.skill_utils import extract_skill_conditions, parse_frontmatter


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


def test_hermes_conditions_explicit_none():
    """Regression for #23627: when each of the four hermes.conditions fields
    is present in the frontmatter but parsed as None (YAML explicit null or
    frontmatter truncation), extract_skill_conditions must still return [].
    Without `get(k) or []`, .get(k, []) would return the actual None value
    and downstream `_skill_should_show` would crash with
    TypeError: 'NoneType' object is not iterable.
    """
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": None,
                "requires_toolsets": None,
                "fallback_for_tools": None,
                "requires_tools": None,
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": [],
    }


def test_hermes_conditions_mixed_none_and_absent():
    """Some conditions fields are present-but-None, others absent entirely.
    Both cases must resolve to [] without raising."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "requires_toolsets": None,
                # fallback_for_toolsets intentionally absent
                "fallback_for_tools": ["t1"],
                # requires_tools intentionally absent
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": [],
        "requires_toolsets": [],
        "fallback_for_tools": ["t1"],
        "requires_tools": [],
    }


def test_hermes_conditions_none_iteration_does_not_crash():
    """Simulate _skill_should_show iteration over each result list.
    Before the #23627 fix, `for ts in None` raised TypeError."""
    frontmatter = {
        "metadata": {
            "hermes": {
                "fallback_for_toolsets": None,
                "requires_toolsets": None,
                "fallback_for_tools": None,
                "requires_tools": None,
            }
        }
    }
    result = extract_skill_conditions(frontmatter)
    for key, value in result.items():
        # Each value must be iterable (i.e. a list, not None).
        list(value)  # would raise TypeError on None before the fix
        assert value == [], f"{key} should be [] but is {value!r}"


def test_end_to_end_skill_md_with_empty_requires_toolsets():
    """End-to-end regression for #23627: a SKILL.md with
    metadata.hermes.requires_toolsets: [] must parse cleanly, leave the
    closing-delimiter out of the body, and produce [] from
    extract_skill_conditions without crashing.

    Two bugs together produced the original symptom:

    1. parse_frontmatter truncated yaml_content by 3 bytes (missing the
       `+ 3` offset compensation when end_match is computed on
       content[3:]). That made yaml_load see `requires_toolsets: ]`
       (missing `[`) and parse the field as None.

    2. extract_skill_conditions used .get(key, []) which returns None
       when the key is present but None-valued.

    Both must be addressed; this test guards the joint contract.
    """
    skill_md = (
        "---\n"
        "name: test-skill\n"
        "description: tests the conditions fix\n"
        "metadata:\n"
        "  hermes:\n"
        "    requires_toolsets: []\n"
        "    fallback_for_toolsets: [\"ts1\"]\n"
        "    requires_tools: [\"t1\"]\n"
        "---\n"
        "# This is the body\n"
        "rest of file\n"
    )
    frontmatter, body = parse_frontmatter(skill_md)
    # Body must start cleanly with the H1 — no closing-delimiter leakage.
    assert body.startswith("# This is the body"), (
        f"body has frontmatter leakage: {body[:40]!r}"
    )
    # requires_toolsets must be a list, not None.
    hermes = frontmatter["metadata"]["hermes"]
    assert hermes["requires_toolsets"] == []
    # extract_skill_conditions must return [] for the empty field and
    # the supplied values for the rest, without raising.
    result = extract_skill_conditions(frontmatter)
    assert result == {
        "fallback_for_toolsets": ["ts1"],
        "requires_toolsets": [],
        "fallback_for_tools": [],
        "requires_tools": ["t1"],
    }
