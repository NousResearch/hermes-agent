"""Tests for the configurable ``agent.attribution`` identity string.

Covers the prompt_builder factories (build_agent_identity / build_help_guidance)
and their wiring into the assembled system prompt via build_system_prompt_parts.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    DEFAULT_ATTRIBUTION,
    HERMES_AGENT_HELP_GUIDANCE,
    build_agent_identity,
    build_help_guidance,
)
from agent.system_prompt import build_system_prompt_parts


# ---------------------------------------------------------------------------
# Factory-level behavior
# ---------------------------------------------------------------------------


class TestBuildAgentIdentity:
    def test_default_matches_legacy_constant(self):
        # Backward compatibility: default attribution reproduces the historical
        # constant byte-for-byte.
        assert build_agent_identity() == DEFAULT_AGENT_IDENTITY
        assert "created by Nous Research" in DEFAULT_AGENT_IDENTITY

    def test_custom_attribution_rebrands(self):
        out = build_agent_identity("Acme Corp")
        assert "created by Acme Corp." in out
        assert "Nous Research" not in out

    def test_empty_attribution_omits_clause(self):
        out = build_agent_identity("")
        assert "created by" not in out
        # The rest of the identity is preserved.
        assert "intelligent AI assistant." in out
        assert "exploration and investigations." in out

    def test_whitespace_only_treated_as_empty(self):
        out = build_agent_identity("   ")
        assert "created by" not in out

    def test_none_treated_as_empty(self):
        out = build_agent_identity(None)  # type: ignore[arg-type]
        assert "created by" not in out


class TestBuildHelpGuidance:
    def test_default_matches_legacy_constant(self):
        assert build_help_guidance() == HERMES_AGENT_HELP_GUIDANCE
        assert "(by Nous Research)" in HERMES_AGENT_HELP_GUIDANCE

    def test_custom_attribution_rebrands(self):
        out = build_help_guidance("Acme Corp")
        assert "(by Acme Corp)" in out
        assert "Nous Research" not in out

    def test_empty_attribution_drops_lead_but_keeps_pointer(self):
        out = build_help_guidance("")
        assert "(by" not in out
        # Functional docs/skill pointer must survive so help routing still works.
        assert "hermes-agent.nousresearch.com/docs" in out
        assert "skill_view(name='hermes-agent')" in out


# ---------------------------------------------------------------------------
# Integration with the assembled system prompt
# ---------------------------------------------------------------------------


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_prompt(agent):
    with (
        patch("run_agent.load_soul_md", return_value=""),
        patch("run_agent.build_nous_subscription_prompt", return_value=""),
        patch("run_agent.build_environment_hints", return_value=""),
        patch("run_agent.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestSystemPromptAttribution:
    def test_default_when_attribute_absent(self):
        # No _attribution attr at all -> getattr fallback to default.
        stable = _stable_prompt(_make_agent())
        assert "created by Nous Research" in stable
        assert "(by Nous Research)" in stable

    def test_explicit_default(self):
        stable = _stable_prompt(_make_agent(_attribution=DEFAULT_ATTRIBUTION))
        assert "created by Nous Research" in stable

    def test_custom_org(self):
        stable = _stable_prompt(_make_agent(_attribution="Acme Corp"))
        assert "created by Acme Corp." in stable
        assert "(by Acme Corp)" in stable
        assert "Nous Research" not in stable

    def test_empty_omits_attribution_keeps_help_pointer(self):
        stable = _stable_prompt(_make_agent(_attribution=""))
        assert "created by" not in stable
        assert "(by" not in stable
        # Docs pointer (functional payload) must still be present.
        assert "hermes-agent.nousresearch.com/docs" in stable
