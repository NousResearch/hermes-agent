"""Tests for agent/skill_utils.py — skill frontmatter metadata handling."""

from pathlib import Path

from agent.skill_utils import (
    discover_all_skill_config_vars,
    extract_skill_conditions,
    extract_skill_config_vars,
    parse_frontmatter,
    resolve_skill_config_values,
)


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


def test_llm_wiki_declares_wiki_path_skill_config():
    """The bundled llm-wiki skill should participate in setup/status config discovery."""
    skill_path = Path(__file__).resolve().parents[2] / "skills" / "research" / "llm-wiki" / "SKILL.md"
    content = skill_path.read_text(encoding="utf-8")
    frontmatter, body = parse_frontmatter(content)

    config_vars = extract_skill_config_vars(frontmatter)

    assert config_vars == [
        {
            "key": "wiki_path",
            "description": "Path to the LLM Wiki markdown knowledge base directory.",
            "default": "~/wiki",
            "prompt": "LLM Wiki directory path",
        }
    ]
    assert "skills.config.wiki_path" in body
    assert "WIKI_PATH" in body


def test_llm_wiki_config_is_discoverable_and_resolves_from_config_yaml(tmp_path, monkeypatch):
    """Discovery returns logical keys, while resolution reads skills.config.<key>."""
    repo_skills_dir = Path(__file__).resolve().parents[2] / "skills"
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        "skills:\n"
        "  config:\n"
        "    wiki_path: ~/research-wiki\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("agent.skill_utils.get_all_skills_dirs", lambda: [repo_skills_dir])
    monkeypatch.setattr("agent.skill_utils.get_disabled_skill_names", lambda: set())

    discovered = discover_all_skill_config_vars()
    llm_wiki_var = next(var for var in discovered if var.get("skill") == "llm-wiki")
    resolved = resolve_skill_config_values([llm_wiki_var])

    assert llm_wiki_var["key"] == "wiki_path"
    assert resolved["wiki_path"] == str(Path.home() / "research-wiki")
