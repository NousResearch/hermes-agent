"""Skill-declared reasoning effort (issue #108770).

A skill that declares ``metadata.hermes.reasoning_effort`` raises the thinking level for the
turn it is invoked in, and the previous value is restored when the turn ends. The feature is
opt-in: a skill that declares nothing must not change ``agent.reasoning_config``.
"""

import json
from unittest.mock import patch

import pytest

from agent.skill_commands import (
    build_skill_invocation_message,
    build_stacked_skill_invocation_message,
    scan_skill_commands,
)
from agent.skill_reasoning import (
    apply_skill_reasoning_override,
    declaration_for_message,
    declared_reasoning_effort,
    restore_skill_reasoning_override,
)


def _make_skill(skills_dir, name, frontmatter_extra="", body="Do the thing."):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    content = f"""\
---
name: {name}
description: Description for {name}.
{frontmatter_extra}---

# {name}

{body}
"""
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def _declaring(name, level="high"):
    return f"metadata:\n  hermes:\n    reasoning_effort: {level}\n"


class _FakeAgent:
    def __init__(self, reasoning_config):
        self.reasoning_config = reasoning_config


class TestDeclaredReasoningEffort:
    def test_frontmatter_string_form_parses_to_a_level(self):
        fm = {"metadata": {"hermes": {"reasoning_effort": "high"}}}
        assert declared_reasoning_effort(fm) == {"enabled": True, "effort": "high"}

    def test_frontmatter_mapping_form_yields_its_effort(self):
        fm = {"metadata": {"hermes": {"reasoning_effort": {"effort": "max", "scope": "turn"}}}}
        assert declared_reasoning_effort(fm) == {"enabled": True, "effort": "max"}

    def test_undeclared_skill_returns_none(self):
        assert declared_reasoning_effort({"metadata": {"hermes": {"tags": []}}}) is None
        assert declared_reasoning_effort({}) is None
        assert declared_reasoning_effort({"metadata": "oops"}) is None

    def test_unknown_level_is_ignored(self):
        fm = {"metadata": {"hermes": {"reasoning_effort": "galaxy-brain"}}}
        assert declared_reasoning_effort(fm) is None

    def test_explicit_disable_is_honored(self):
        fm = {"metadata": {"hermes": {"reasoning_effort": "none"}}}
        assert declared_reasoning_effort(fm) == {"enabled": False}


class TestOverrideLifecycle:
    """Entering a declaring skill raises the level; leaving it restores the old one."""

    def _message_for(self, tmp_path, name):
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            message = build_skill_invocation_message(f"/{name}")
        assert message is not None
        return message

    def test_entering_raises_and_exiting_restores(self, tmp_path):
        _make_skill(tmp_path, "deep-research", frontmatter_extra=_declaring("deep-research", "high"))
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            message = build_skill_invocation_message("/deep-research", "write the thesis")
        assert message is not None

        agent = _FakeAgent({"enabled": True, "effort": "medium"})
        baseline = dict(agent.reasoning_config)

        snapshot = apply_skill_reasoning_override(agent, message)
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert snapshot["previous"] == baseline

        restore_skill_reasoning_override(agent, snapshot)
        assert agent.reasoning_config == baseline

    def test_skill_without_declaration_leaves_config_untouched(self, tmp_path):
        _make_skill(tmp_path, "plain", frontmatter_extra="")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            message = build_skill_invocation_message("/plain", "do a thing")
        assert message is not None

        agent = _FakeAgent({"enabled": True, "effort": "medium"})
        snapshot = apply_skill_reasoning_override(agent, message)

        assert snapshot is None
        assert agent.reasoning_config == {"enabled": True, "effort": "medium"}

    def test_non_skill_message_never_raises(self):
        agent = _FakeAgent({"enabled": True, "effort": "low"})
        assert apply_skill_reasoning_override(agent, "just a normal question") is None
        assert apply_skill_reasoning_override(agent, None) is None
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}

    def test_session_scope_survives_the_restore(self, tmp_path):
        _make_skill(
            tmp_path,
            "long-haul",
            frontmatter_extra=(
                "metadata:\n  hermes:\n    reasoning_effort:\n      effort: max\n      scope: session\n"
            ),
        )
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            message = build_skill_invocation_message("/long-haul")
        assert message is not None

        agent = _FakeAgent({"enabled": True, "effort": "medium"})
        snapshot = apply_skill_reasoning_override(agent, message)
        assert agent.reasoning_config == {"enabled": True, "effort": "max"}

        restore_skill_reasoning_override(agent, snapshot)
        # The session-scoped level is what a later turn restores, not the pre-skill level.
        assert agent.reasoning_config == {"enabled": True, "effort": "max"}


class TestMessageDetection:
    def test_single_skill_scaffold_is_detected(self, tmp_path):
        _make_skill(tmp_path, "deep-research", frontmatter_extra=_declaring("deep-research", "high"))
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            message = build_skill_invocation_message("/deep-research", "go")
        declaration = declaration_for_message(message)
        assert declaration is not None
        assert declaration["config"] == {"enabled": True, "effort": "high"}
        assert declaration["scope"] == "turn"

    def test_declaration_bypasses_idempotence_cache(self, tmp_path):
        """Two builds of the same scaffold must both resolve their declaration."""
        _make_skill(tmp_path, "deep-research", frontmatter_extra=_declaring("deep-research", "high"))
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            first = declaration_for_message(build_skill_invocation_message("/deep-research", "one"))
            second = declaration_for_message(build_skill_invocation_message("/deep-research", "two"))
        assert first is not None and second is not None
        assert first["config"] == second["config"]

    def test_plain_text_is_not_a_scaffold(self, tmp_path):
        assert declaration_for_message("hello there") is None
        assert declaration_for_message("") is None

    def test_plain_skill_scaffold_is_not_opt_in(self, tmp_path):
        """A scaffold whose skill declares nothing must not be treated as opted in."""
        _make_skill(tmp_path, "plain")
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            message = build_skill_invocation_message("/plain", "go")
        assert message is not None
        assert declaration_for_message(message) is None

    def test_stacked_bundle_resolves_the_declaring_skill(self, tmp_path):
        """A stacked `/a /b` scaffold declares per block; the declaring one wins."""
        _make_skill(tmp_path, "plain")
        _make_skill(tmp_path, "deep-research", frontmatter_extra=_declaring("deep-research", "max"))
        with patch("tools.skills_tool.SKILLS_DIR", tmp_path):
            scan_skill_commands()
            built = build_stacked_skill_invocation_message(["/plain", "/deep-research"], "go")
        assert built is not None
        declaration = declaration_for_message(built[0])
        assert declaration == {"config": {"enabled": True, "effort": "max"}, "scope": "turn"}
