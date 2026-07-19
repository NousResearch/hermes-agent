"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no separate engine: it loads the
bundled authoring guidance and builds a prompt that the live agent runs as a
normal turn, so these are the load-bearing behavior contracts.
"""

import pytest

import agent.learn_prompt as learn_prompt
from agent.skill_authoring_guidance import SkillAuthoringGuidance
from agent.learn_prompt import (
    _FALLBACK_AUTHORING_GUIDANCE,
    build_learn_prompt,
)


_TEST_AUTHORING_GUIDANCE = "LOADED HERMES AUTHORING SKILL AND CONTRACT"


class TestBuildLearnPrompt:
    @pytest.fixture(autouse=True)
    def _stub_runtime_guidance(self, monkeypatch):
        # Prompt-shape tests should not depend on a developer's active profile.
        monkeypatch.setattr(
            learn_prompt,
            "_load_authoring_guidance",
            lambda: _TEST_AUTHORING_GUIDANCE,
        )

    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt

    def test_always_includes_runtime_authoring_guidance(self):
        for req in ["", "a url https://x/y", "what we just did"]:
            assert _TEST_AUTHORING_GUIDANCE in build_learn_prompt(req)

    def test_instructs_saving_via_skill_manage_not_a_raw_file(self):
        prompt = build_learn_prompt("learn the thing")
        assert "skill_manage" in prompt

    def test_references_gather_tools_for_open_ended_sourcing(self):
        # Open-ended sourcing relies on the agent's own tools, named so it
        # knows dirs/URLs/conversation/paste all route through existing tools.
        prompt = build_learn_prompt("learn from somewhere")
        for tool in ("read_file", "search_files", "web_extract"):
            assert tool in prompt

    def test_separates_sources_from_requirements(self):
        # The reported bug (@GrenFX, Jun 2026): when a request leads with a
        # path/URL, the agent fetched it and ignored the trailing prose. The
        # prompt must tell the agent the request can MIX sources and
        # requirements, and that prose after a source is authoring guidance to
        # honor — not noise to drop.
        prompt = build_learn_prompt(
            "https://api.example.com/docs focus on the auth flow, skip deprecated bits"
        )
        low = prompt.lower()
        # Carries the whole request verbatim (no truncation at the URL).
        assert "focus on the auth flow, skip deprecated bits" in prompt
        # Explicitly distinguishes sources from requirements.
        assert "requirement" in low
        # Names the failure mode it's guarding against.
        assert "never fetch the first source" in low

    def test_empty_request_falls_back_to_the_conversation(self):
        # Bare /learn should distill "what we just did", not error.
        prompt = build_learn_prompt("")
        assert "conversation" in prompt.lower()
        # And still carries the standards + save instruction.
        assert "skill_manage" in prompt

    def test_whitespace_only_request_is_treated_as_empty(self):
        assert build_learn_prompt("   \n  ") == build_learn_prompt("")

    def test_requires_discovery_before_an_authoring_decision(self):
        prompt = build_learn_prompt("learn the thing")
        discovery = prompt.index("inspect the installed library")
        decision = prompt.index("Choose exactly one decision")
        assert discovery < decision
        for outcome in ("UPDATE", "CREATE", "CONSOLIDATE", "SKIP"):
            assert outcome in prompt
        assert "Do not default to creation." in prompt
        assert "(action=\"create\")" not in prompt

    def test_skip_is_explicitly_non_mutating(self):
        prompt = build_learn_prompt("learn the thing")
        assert "A SKIP decision performs no write." in prompt


class TestLoadAuthoringGuidance:
    def test_formats_distinct_skill_and_contract_payloads(self, monkeypatch):
        monkeypatch.setattr(
            learn_prompt,
            "load_bundled_skill_authoring_guidance",
            lambda: SkillAuthoringGuidance(
                skill_content="V2 SKILL BODY",
                contract_content="V2 CONTRACT",
            ),
        )

        guidance = learn_prompt._load_authoring_guidance()

        assert "V2 SKILL BODY" in guidance
        assert "V2 CONTRACT" in guidance

    def test_missing_bundled_skill_uses_safe_fallback(self, monkeypatch):
        monkeypatch.setattr(
            learn_prompt,
            "load_bundled_skill_authoring_guidance",
            lambda: None,
        )

        assert (
            learn_prompt._load_authoring_guidance()
            == _FALLBACK_AUTHORING_GUIDANCE
        )

    def test_skill_read_failure_uses_safe_fallback(self, monkeypatch):
        def fail(*args, **kwargs):
            raise OSError("unreadable")

        monkeypatch.setattr(
            learn_prompt,
            "load_bundled_skill_authoring_guidance",
            fail,
        )

        assert (
            learn_prompt._load_authoring_guidance()
            == _FALLBACK_AUTHORING_GUIDANCE
        )

    def test_contract_failure_keeps_skill_and_adds_fallback(self, monkeypatch):
        monkeypatch.setattr(
            learn_prompt,
            "load_bundled_skill_authoring_guidance",
            lambda: SkillAuthoringGuidance(
                skill_content="V2 SKILL BODY",
                contract_content=None,
            ),
        )

        guidance = learn_prompt._load_authoring_guidance()

        assert "V2 SKILL BODY" in guidance
        assert _FALLBACK_AUTHORING_GUIDANCE in guidance

    def test_fallback_preserves_minimum_v2_contract(self):
        fallback = _FALLBACK_AUTHORING_GUIDANCE
        for decision in ("update", "create", "consolidate", "skip"):
            assert decision in fallback
        assert "at most 60 characters" in fallback
        assert "human contributor first" in fallback
        for section in (
            "When to Use",
            "Prerequisites",
            "How to Run",
            "Quick Reference",
            "Procedure",
            "Pitfalls",
            "Verification",
        ):
            assert section in fallback


class TestLearnRegistryWiring:
    def test_learn_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None
        assert cmd.name == "learn"

    def test_learn_is_in_tools_and_skills_category(self):
        from hermes_cli.commands import resolve_command

        assert resolve_command("learn").category == "Tools & Skills"

    def test_learn_works_on_the_gateway(self):
        # /learn must reach the gateway runner (it's a both-surfaces command),
        # not be CLI-only.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "learn" in GATEWAY_KNOWN_COMMANDS

    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        assert not resolve_command("learn").cli_only
