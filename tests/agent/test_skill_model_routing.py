"""Tests for skill-level model routing.

Skills can declare a preferred model in their SKILL.md frontmatter
(``metadata.hermes.model``), and users can override that per-skill via
``config.yaml skills.model_overrides.<skill-name>``. The override applies as a
lightweight transient model swap for the skill's turn and reverts afterward.

These tests cover:
- extract_skill_model_override() frontmatter parsing
- get_skill_model_for_command() resolution order (config > frontmatter > None)
- skill_model_swap() / skill_model_restore() lightweight swap mechanics

Why: Guards the contract that skill routing honors config precedence and that
the swap is a pure model-identity flip (no client rebuild) that always reverts.
"""

from __future__ import annotations

import pytest

import agent.skill_commands as skill_commands
from agent.skill_utils import (
    extract_skill_model_override,
    skill_model_restore,
    skill_model_swap,
)


# ── extract_skill_model_override ────────────────────────────────────────────


def test_extract_model_override_present():
    fm = {"metadata": {"hermes": {"model": "deepseek/deepseek-v4-flash"}}}
    assert extract_skill_model_override(fm) == "deepseek/deepseek-v4-flash"


def test_extract_model_override_absent_returns_none():
    assert extract_skill_model_override({}) is None
    assert extract_skill_model_override({"metadata": {}}) is None
    assert extract_skill_model_override({"metadata": {"hermes": {}}}) is None


def test_extract_model_override_non_scalar_ignored():
    # A dict-shaped model (provider/model nesting) is not a usable swap slug.
    fm = {"metadata": {"hermes": {"model": {"provider": "openrouter"}}}}
    assert extract_skill_model_override(fm) is None
    # Empty/whitespace strings are treated as no override.
    assert extract_skill_model_override({"metadata": {"hermes": {"model": "  "}}}) is None


def test_extract_model_override_malformed_metadata():
    # metadata as a string (malformed YAML) must not raise.
    assert extract_skill_model_override({"metadata": "oops"}) is None


# ── get_skill_model_for_command ─────────────────────────────────────────────


@pytest.fixture
def patched_skill(monkeypatch):
    """Patch skill discovery + payload load so resolution is disk-free.

    Returns a helper to set the SKILL.md frontmatter model the fake skill
    reports, defaulting to a model declared in frontmatter.
    """
    cmd_key = "/fake-skill"
    skill_name = "fake-skill"

    monkeypatch.setattr(
        skill_commands,
        "get_skill_commands",
        lambda: {cmd_key: {"name": skill_name, "skill_dir": "/tmp/fake-skill"}},
    )

    def _set_frontmatter_model(model_line: str | None):
        if model_line is None:
            raw = "---\nname: fake-skill\n---\nbody\n"
        else:
            raw = (
                "---\n"
                "name: fake-skill\n"
                "metadata:\n"
                "  hermes:\n"
                f"    model: {model_line}\n"
                "---\nbody\n"
            )
        monkeypatch.setattr(
            skill_commands,
            "_load_skill_payload",
            lambda *_a, **_k: ({"raw_content": raw}, None, skill_name),
        )

    return cmd_key, skill_name, _set_frontmatter_model


def test_get_model_from_frontmatter(patched_skill):
    cmd_key, _name, set_model = patched_skill
    set_model("x/y")
    assert skill_commands.get_skill_model_for_command(cmd_key) == "x/y"


def test_config_override_takes_precedence(patched_skill):
    cmd_key, name, set_model = patched_skill
    set_model("x/y")  # frontmatter says x/y
    result = skill_commands.get_skill_model_for_command(cmd_key, {name: "z/override"})
    assert result == "z/override"


def test_no_model_declared_returns_none(patched_skill):
    cmd_key, _name, set_model = patched_skill
    set_model(None)  # no metadata.hermes.model
    assert skill_commands.get_skill_model_for_command(cmd_key) is None


def test_unknown_command_returns_none(monkeypatch):
    monkeypatch.setattr(skill_commands, "get_skill_commands", lambda: {})
    assert skill_commands.get_skill_model_for_command("/nope") is None


# ── skill_model_swap / skill_model_restore ──────────────────────────────────


class _FakeAgent:
    """Minimal stand-in exposing the attributes the swap helpers touch."""

    def __init__(self, model: str):
        self.model = model
        self._cached_system_prompt = "cached-prompt"


def test_swap_changes_model_and_returns_snapshot():
    agent = _FakeAgent("model-a")
    snapshot = skill_model_swap(agent, "model-b")
    assert snapshot == {"model": "model-a"}
    assert agent.model == "model-b"
    # Cached system prompt invalidated so model identity is rebuilt.
    assert agent._cached_system_prompt is None


def test_swap_noop_when_same_model():
    agent = _FakeAgent("model-a")
    assert skill_model_swap(agent, "model-a") is None
    assert agent.model == "model-a"
    # No-op must not bust the cache.
    assert agent._cached_system_prompt == "cached-prompt"


def test_swap_noop_when_empty_model():
    agent = _FakeAgent("model-a")
    assert skill_model_swap(agent, "") is None
    assert agent.model == "model-a"


def test_restore_returns_to_original():
    agent = _FakeAgent("model-a")
    snapshot = skill_model_swap(agent, "model-b")
    assert agent.model == "model-b"
    skill_model_restore(agent, snapshot)
    assert agent.model == "model-a"
    assert agent._cached_system_prompt is None


def test_restore_noop_on_none_snapshot():
    agent = _FakeAgent("model-a")
    skill_model_restore(agent, None)
    assert agent.model == "model-a"
    # None snapshot must not touch the cache.
    assert agent._cached_system_prompt == "cached-prompt"


def test_swap_then_restore_roundtrip_preserves_model():
    agent = _FakeAgent("opus")
    snap = skill_model_swap(agent, "haiku")
    assert agent.model == "haiku"
    skill_model_restore(agent, snap)
    assert agent.model == "opus"
