"""Tests for predefined task-agent configuration schema."""

from __future__ import annotations

import importlib

import yaml


EXPECTED_AGENT_IDS = {"research", "implementation", "review", "documentation"}


def test_default_predefined_task_agents_have_distinct_purposes():
    from hermes_cli.agent_types import PREDEFINED_TASK_AGENT_DEFINITIONS

    ids = {entry["id"] for entry in PREDEFINED_TASK_AGENT_DEFINITIONS}
    purposes = [entry["purpose"] for entry in PREDEFINED_TASK_AGENT_DEFINITIONS]

    assert ids == EXPECTED_AGENT_IDS
    assert len(purposes) == len(set(purposes))
    for entry in PREDEFINED_TASK_AGENT_DEFINITIONS:
        assert entry["enabled"] is True
        assert entry["invocation"]["entrypoint"] == "delegate_task"
        assert entry["invocation"]["task_category"] == entry["id"]
        assert "prompt" in entry["invocation"]["parameters"]["required"]


def test_valid_task_agent_definitions_load_as_dataclasses():
    from hermes_cli.agent_types import load_task_agent_definitions

    agents = load_task_agent_definitions({
        "task_agents": {
            "definitions": [
                {
                    "id": "research",
                    "purpose": "Collect and synthesize source-backed information before implementation.",
                    "enabled": True,
                    "invocation": {
                        "entrypoint": "delegate_task",
                        "task_category": "research",
                        "parameters": {"required": ["prompt"], "optional": ["context"]},
                    },
                    "runtime": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-sonnet-4",
                        "reasoning_effort": "high",
                        "max_iterations": 12,
                        "enabled_toolsets": ["web", "file"],
                    },
                },
            ]
        }
    })

    assert set(agents) == {"research"}
    assert agents["research"].id == "research"
    assert agents["research"].runtime["model"] == "anthropic/claude-sonnet-4"


def test_load_config_supplies_default_task_agents(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import hermes_constants
    import hermes_cli.config as config_mod

    importlib.reload(hermes_constants)
    importlib.reload(config_mod)

    cfg = config_mod.load_config()
    assert {entry["id"] for entry in cfg["task_agents"]["definitions"]} == EXPECTED_AGENT_IDS


def test_validate_config_structure_rejects_duplicate_task_agent_ids():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({
        "task_agents": {
            "definitions": [
                {
                    "id": "research",
                    "purpose": "Research sources.",
                    "enabled": True,
                    "invocation": {"entrypoint": "delegate_task", "task_category": "research", "parameters": {"required": ["prompt"]}},
                },
                {
                    "id": "research",
                    "purpose": "Do another research task.",
                    "enabled": True,
                    "invocation": {"entrypoint": "delegate_task", "task_category": "research", "parameters": {"required": ["prompt"]}},
                },
            ]
        }
    })

    assert any(issue.severity == "error" and "Duplicate task_agents id 'research'" in issue.message for issue in issues)


def test_validate_config_structure_rejects_unknown_or_malformed_task_agents():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({
        "task_agents": {
            "definitions": [
                {
                    "id": "social-media",
                    "purpose": "Post to social media.",
                    "enabled": "yes",
                    "invocation": {"entrypoint": "shell", "task_category": "social-media", "parameters": {}},
                },
            ]
        }
    })
    messages = "\n".join(issue.message for issue in issues)

    assert "Unknown task_agents id 'social-media'" in messages
    assert "enabled must be a boolean" in messages
    assert "invocation.entrypoint must be 'delegate_task'" in messages


def test_validate_config_structure_rejects_shared_purposes():
    from hermes_cli.config import validate_config_structure

    issues = validate_config_structure({
        "task_agents": {
            "definitions": [
                {
                    "id": "research",
                    "purpose": "Analyze the task.",
                    "enabled": True,
                    "invocation": {"entrypoint": "delegate_task", "task_category": "research", "parameters": {"required": ["prompt"]}},
                },
                {
                    "id": "review",
                    "purpose": "Analyze the task.",
                    "enabled": True,
                    "invocation": {"entrypoint": "delegate_task", "task_category": "review", "parameters": {"required": ["prompt"]}},
                },
            ]
        }
    })

    assert any("share the same purpose" in issue.message for issue in issues)


def test_documented_schema_example_is_valid_yaml_and_loads():
    from hermes_cli.agent_types import load_task_agent_definitions

    from pathlib import Path
    docs_path = Path(__file__).parents[2] / "docs" / "task-agents-config.md"
    content = docs_path.read_text(encoding="utf-8")
    start = content.index("```yaml") + len("```yaml")
    end = content.index("```", start)
    example = yaml.safe_load(content[start:end])

    agents = load_task_agent_definitions(example)
    assert set(agents) == EXPECTED_AGENT_IDS
