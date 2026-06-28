"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no engine and no model tool: it
builds a standards-guided prompt that the live agent runs as a normal turn, so
these are the load-bearing behavior contracts.
"""

from agent.learn_prompt import (
    build_learn_prompt,
    parse_learn_request,
    _AUTHORING_STANDARDS,
)


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt

    def test_always_includes_the_authoring_standards(self):
        # The standards are what make distilled skills match house style;
        # they must travel with every prompt regardless of input.
        for req in ["", "a url https://x/y", "what we just did"]:
            assert _AUTHORING_STANDARDS in build_learn_prompt(req)

    def test_instructs_saving_via_skill_manage_not_a_raw_file(self):
        prompt = build_learn_prompt("learn the thing")
        assert "skill_manage" in prompt

    def test_references_gather_tools_for_open_ended_sourcing(self):
        # Open-ended sourcing relies on the agent's own tools, named so it
        # knows dirs/URLs/conversation/paste all route through existing tools.
        prompt = build_learn_prompt("learn from somewhere")
        for tool in ("read_file", "search_files", "web_extract"):
            assert tool in prompt

    def test_empty_request_falls_back_to_the_conversation(self):
        # Bare /learn should distill "what we just did", not error.
        prompt = build_learn_prompt("")
        assert "conversation" in prompt.lower()
        # And still carries the standards + save instruction.
        assert "skill_manage" in prompt

    def test_whitespace_only_request_is_treated_as_empty(self):
        assert build_learn_prompt("   \n  ") == build_learn_prompt("")

    def test_description_length_rule_is_in_the_standards(self):
        # The single most-violated rule must be explicit in the prompt.
        assert "60" in _AUTHORING_STANDARDS

    def test_teaches_the_full_hardline_standards(self):
        # /learn must teach ALL the CONTRIBUTING.md skill rules, not just the
        # description length — otherwise distilled skills miss platform gating,
        # author credit, and the tool-framing table. Lock the coverage in.
        std = _AUTHORING_STANDARDS.lower()
        # #1 description: the count-and-trim self-check (the reported bug).
        assert "count" in std and "60" in std
        # #3 platforms gating against OS-bound primitives.
        assert "platforms" in std
        # author is always the literal Hermes, never the host/OS identity (#52368).
        assert "author: always the literal value `hermes`" in std
        assert "never fill it from the host" in std
        # #2 Hermes-tool framing names the wrapped tools, not shell utilities.
        for tool in ("read_file", "search_files", "patch", "write_file"):
            assert tool in std
        # #6 scripts/references/templates layout.
        assert "scripts/" in _AUTHORING_STANDARDS


class TestParseLearnRequest:
    def test_plain_request_is_not_update_mode(self):
        # No --update flag → (None, stripped_request); create path untouched.
        assert parse_learn_request("the REST client in ~/proj") == (
            None,
            "the REST client in ~/proj",
        )

    def test_empty_is_not_update_mode(self):
        assert parse_learn_request("") == (None, "")
        assert parse_learn_request("   \n ") == (None, "")

    def test_update_with_skill_and_notes(self):
        assert parse_learn_request("--update arxiv-search add the ID lookup") == (
            "arxiv-search",
            "add the ID lookup",
        )

    def test_update_with_skill_only(self):
        assert parse_learn_request("--update arxiv-search") == ("arxiv-search", "")

    def test_update_without_skill_is_still_update_mode(self):
        # Bare --update routes to update mode with an empty target — it must
        # never fall through to create.
        assert parse_learn_request("--update") == ("", "")

    def test_update_only_when_flag_leads(self):
        # --update mid-string is ordinary free text, not the flag.
        assert parse_learn_request("learn how --update works") == (
            None,
            "learn how --update works",
        )


class TestBuildUpdatePrompt:
    def test_targets_the_named_skill(self):
        prompt = build_learn_prompt("--update arxiv-search")
        assert "arxiv-search" in prompt
        assert "skill_view" in prompt

    def test_uses_patch_and_edit_not_create(self):
        prompt = build_learn_prompt("--update arxiv-search fix a flag")
        assert 'action="patch"' in prompt
        assert 'action="edit"' in prompt
        # Update mode must never instruct the create action.
        assert 'action="create"' not in prompt
        # And must not carry the create-path's authoring instruction.
        assert "Author ONE SKILL.md" not in prompt

    def test_carries_the_authoring_standards(self):
        # An edited skill must still match house style.
        assert _AUTHORING_STANDARDS in build_learn_prompt("--update arxiv-search")

    def test_embeds_the_change_notes_verbatim(self):
        prompt = build_learn_prompt("--update arxiv-search add ID lookup section")
        assert "add ID lookup section" in prompt

    def test_bare_update_does_not_create(self):
        prompt = build_learn_prompt("--update")
        assert 'action="create"' not in prompt
        assert "Author ONE SKILL.md" not in prompt
        assert "skills_list" in prompt


class TestCreatePathUnchanged:
    def test_plain_request_still_uses_create(self):
        prompt = build_learn_prompt("the REST client in ~/projects/acme-sdk")
        assert 'action="create"' in prompt
        assert "Author ONE SKILL.md" in prompt

    def test_empty_request_still_falls_back_to_conversation(self):
        prompt = build_learn_prompt("")
        assert "conversation" in prompt.lower()
        assert 'action="create"' in prompt


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

    def test_learn_args_hint_and_subcommands(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert "--update" in cmd.args_hint
        assert "--update" in cmd.subcommands


class TestLearnCommandCompleter:
    def test_learn_autocomplete_suggests_update_flag(self):
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        completer = SlashCommandCompleter()
        doc = Document("/learn ", cursor_position=len("/learn "))
        completions = list(completer.get_completions(doc, None))
        assert any(c.text == "--update" for c in completions)

    def test_learn_autocomplete_suggests_skills(self, monkeypatch):
        from hermes_cli.commands import SlashCommandCompleter
        from prompt_toolkit.document import Document

        monkeypatch.setattr(
            "tools.skills_tool._find_all_skills",
            lambda *args, **kwargs: [
                {"name": "test-skill-one", "description": "Desc 1"},
                {"name": "other-skill", "description": "Desc 2"},
            ]
        )

        completer = SlashCommandCompleter()
        doc = Document("/learn --update ", cursor_position=len("/learn --update "))
        completions = list(completer.get_completions(doc, None))
        suggested = [c.text for c in completions]
        assert "test-skill-one" in suggested
        assert "other-skill" in suggested

        doc = Document("/learn --update test-s", cursor_position=len("/learn --update test-s"))
        completions = list(completer.get_completions(doc, None))
        suggested = [c.text for c in completions]
        assert "test-skill-one" in suggested
        assert "other-skill" not in suggested
