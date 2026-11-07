"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no engine and no model tool: it
builds a standards-guided prompt that the live agent runs as a normal turn, so
these are the load-bearing behavior contracts.
"""

from agent.learn_prompt import (
    LEARN_UNAVAILABLE_MESSAGE,
    build_learn_prompt,
    skill_manage_available,
)


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt


class TestLearnRegistryWiring:
    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None and cmd.name == "learn"
        assert not cmd.cli_only


class TestLearnSkillManageGating:
    """#74249: /learn must refuse to run when skill_manage is unavailable."""

    def test_skill_manage_available_requires_tool_name(self):
        assert skill_manage_available({"skills_list", "skill_view", "skill_manage"})
        assert not skill_manage_available({"skills_list", "skill_view"})
        assert not skill_manage_available(set())
        assert not skill_manage_available(None)

    def test_unavailable_message_names_skill_manage(self):
        assert "skill_manage" in LEARN_UNAVAILABLE_MESSAGE
