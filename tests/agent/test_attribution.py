"""Tests for the configurable ``agent.attribution`` identity string.

Covers the prompt_builder factories and their wiring into the assembled
system prompt. Default attribution must stay byte-identical to the module
constants (prompt-cache stability). Empty attribution drops only the branding
clause; the docs/skill pointer stays.
"""

from types import SimpleNamespace
from unittest.mock import patch

from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY,
    DEFAULT_ATTRIBUTION,
    HERMES_AGENT_HELP_GUIDANCE,
    HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS,
    build_agent_identity,
    build_help_guidance,
    build_help_guidance_no_skills,
)
from agent.system_prompt import build_system_prompt_parts


class TestBuildAgentIdentity:
    def test_default_matches_legacy_constant(self):
        assert build_agent_identity() == DEFAULT_AGENT_IDENTITY
        assert "built by Nous Research" in DEFAULT_AGENT_IDENTITY

    def test_custom_attribution_rebrands(self):
        out = build_agent_identity("Acme Corp")
        assert "built by Acme Corp." in out
        assert "Nous Research" not in out
        assert "Be direct:" in out

    def test_empty_attribution_omits_clause(self):
        out = build_agent_identity("")
        assert "built by" not in out
        assert out.startswith("You are Hermes Agent. ")
        assert "Be direct:" in out

    def test_whitespace_only_treated_as_empty(self):
        assert "built by" not in build_agent_identity("   ")

    def test_none_treated_as_empty(self):
        assert "built by" not in build_agent_identity(None)  # type: ignore[arg-type]


class TestBuildHelpGuidance:
    def test_default_matches_legacy_constant(self):
        assert build_help_guidance() == HERMES_AGENT_HELP_GUIDANCE
        assert "(by Nous Research)" in HERMES_AGENT_HELP_GUIDANCE
        assert build_help_guidance_no_skills() == HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS

    def test_custom_attribution_rebrands(self):
        out = build_help_guidance("Acme Corp")
        assert "(by Acme Corp)" in out
        assert "Nous Research" not in out
        bare = build_help_guidance_no_skills("Acme Corp")
        assert "(by Acme Corp)" in bare
        assert "skill_view" not in bare

    def test_empty_attribution_drops_lead_but_keeps_pointer(self):
        out = build_help_guidance("")
        assert "(by" not in out
        assert out.startswith("You run on Hermes Agent. ")
        assert "hermes-agent.nousresearch.com/docs" in out
        assert "skill_view(name='hermes-agent')" in out
        bare = build_help_guidance_no_skills("")
        assert "(by" not in bare
        assert "hermes-agent.nousresearch.com/docs" in bare
        assert "skill_view" not in bare


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _parallel_tool_call_guidance=False,
        _tool_use_enforcement=False,
        _execution_guidance=False,
        _environment_probe=False,
        _bot_mode_protocol=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        _memory_enabled=False,
        _user_profile_enabled=False,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_args, **_kwargs: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable_prompt(agent):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=""),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
        patch(
            "agent.coding_context.coding_system_prompt_parts",
            return_value=([], [], []),
        ),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestSystemPromptAttribution:
    def test_default_when_attribute_absent(self):
        stable = _stable_prompt(_make_agent())
        assert "built by Nous Research" in stable
        assert "(by Nous Research)" in stable

    def test_explicit_default(self):
        stable = _stable_prompt(_make_agent(_attribution=DEFAULT_ATTRIBUTION))
        assert DEFAULT_AGENT_IDENTITY in stable
        assert HERMES_AGENT_HELP_GUIDANCE_NO_SKILLS in stable

    def test_custom_org(self):
        stable = _stable_prompt(_make_agent(_attribution="Acme Corp"))
        assert "built by Acme Corp." in stable
        assert "(by Acme Corp)" in stable
        assert "Nous Research" not in stable

    def test_empty_omits_attribution_keeps_help_pointer(self):
        stable = _stable_prompt(_make_agent(_attribution=""))
        assert "built by" not in stable
        assert "(by" not in stable
        assert "hermes-agent.nousresearch.com/docs" in stable

    def test_non_string_falls_back_to_default(self):
        stable = _stable_prompt(_make_agent(_attribution=123))
        assert "built by Nous Research" in stable
