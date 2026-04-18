"""Tests for the agentic_stack memory provider plugin."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def brain(tmp_path: Path) -> Path:
    """Build a minimal fake agentic-stack brain on disk."""
    root = tmp_path / ".agent"
    (root / "memory" / "personal").mkdir(parents=True)
    (root / "memory" / "semantic" / "entities").mkdir(parents=True)
    (root / "memory" / "semantic" / "concepts").mkdir(parents=True)
    (root / "memory" / "working").mkdir(parents=True)
    (root / "memory" / "episodic").mkdir(parents=True)
    (root / "harness" / "hooks").mkdir(parents=True)
    (root / "tools").mkdir()

    (root / "memory" / "personal" / "PREFERENCES.md").write_text(
        "# Preferences\n\n- be concise\n- no em-dashes\n"
    )
    (root / "memory" / "semantic" / "LESSONS.md").write_text(
        "# Lessons\n\n- test before shipping\n"
    )
    (root / "memory" / "semantic" / "entities" / "alpha.md").write_text(
        "# Alpha\n\nThe alpha project ships quarterly."
    )
    (root / "memory" / "semantic" / "entities" / "beta.md").write_text(
        "# Beta\n\nBeta is our staging environment."
    )
    (root / "memory" / "working" / "REVIEW_QUEUE.md").write_text(
        "# Review Queue\n\n_No pending candidates._\n"
    )
    (root / "memory" / "episodic" / "AGENT_LEARNINGS.jsonl").write_text("")

    # Minimal log_execution stub so initialize considers the provider usable.
    (root / "harness" / "__init__.py").write_text("")
    (root / "harness" / "hooks" / "__init__.py").write_text("")
    (root / "harness" / "hooks" / "post_execution.py").write_text(
        "import json, os\n"
        "EPISODIC = os.path.join(os.path.dirname(__file__), '..', '..', "
        "'memory', 'episodic', 'AGENT_LEARNINGS.jsonl')\n"
        "def log_execution(skill_name, action, result, success, reflection='', "
        "importance=5, confidence=0.5, evidence_ids=None):\n"
        "    entry = {'skill': skill_name, 'action': action, 'result': "
        "'success' if success else 'failure', 'importance': importance, "
        "'source': {'skill': skill_name, 'profile': 'test'}}\n"
        "    with open(os.path.abspath(EPISODIC), 'a') as f:\n"
        "        f.write(json.dumps(entry) + chr(10))\n"
        "    return entry\n"
    )
    (root / "harness" / "hooks" / "on_failure.py").write_text(
        "def on_failure(*a, **kw):\n    return {}\n"
    )
    return root


def _load_plugin():
    # Ensure a fresh import each test
    for key in list(sys.modules):
        if key.startswith("plugins.memory.agentic_stack"):
            sys.modules.pop(key, None)
    return importlib.import_module("plugins.memory.agentic_stack")


def test_is_available_true_for_well_formed_brain(brain: Path) -> None:
    mod = _load_plugin()
    provider = mod.AgenticStackProvider()
    provider._config = {"brain_path": str(brain)}
    assert provider.is_available() is True


def test_is_available_false_when_harness_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty-brain"
    (empty / "memory").mkdir(parents=True)  # memory/ present, harness/ missing
    mod = _load_plugin()
    provider = mod.AgenticStackProvider()
    provider._config = {"brain_path": str(empty)}
    assert provider.is_available() is False


def test_system_prompt_block_includes_preferences_and_lessons(brain: Path) -> None:
    mod = _load_plugin()
    provider = mod.AgenticStackProvider()
    provider._config = {"brain_path": str(brain)}
    provider.initialize(
        session_id="test-1",
        hermes_home=str(brain.parent),
        platform="cli",
        agent_context="primary",
    )
    block = provider.system_prompt_block()
    assert "PREFERENCES" in block
    assert "LESSONS" in block
    # Queue is empty -> no queue section
    assert "review queue" not in block.lower()


def test_system_prompt_block_silent_on_cron_context(brain: Path) -> None:
    mod = _load_plugin()
    provider = mod.AgenticStackProvider()
    provider._config = {"brain_path": str(brain)}
    provider.initialize(
        session_id="test-2",
        hermes_home=str(brain.parent),
        platform="cli",
        agent_context="cron",
    )
    assert provider.system_prompt_block() == ""


def test_tool_schemas_are_registered(brain: Path) -> None:
    mod = _load_plugin()
    provider = mod.AgenticStackProvider()
    provider._config = {"brain_path": str(brain)}
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert names == {
        "brain_search",
        "brain_review_queue",
        "brain_graduate",
        "brain_reject",
        "brain_log",
    }
