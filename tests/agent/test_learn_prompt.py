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
    _KNOWLEDGE_SKILL_STANDARDS,
    _SOURCE_HYGIENE,
)


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt




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




    def test_teaches_the_full_hardline_standards(self):
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

    def test_teaches_the_knowledge_base_layout(self):
        # Expansive sources (books, paper stacks, specs) must produce a lean
        # SKILL.md index plus per-chapter references/ files loaded on demand —
        # not one crammed file or a lossy summary. Ported from the
        # book-to-skill layout (Aug 2026).
        kb = _KNOWLEDGE_SKILL_STANDARDS.lower()
        assert "references/" in _KNOWLEDGE_SKILL_STANDARDS
        # On-demand loading goes through skill_view with a file_path.
        assert "skill_view" in _KNOWLEDGE_SKILL_STANDARDS
        # Structure, not summary — the load-bearing distillation rule.
        assert "structure" in kb and "summary" in kb
        # Copyright/quality line: synthesized notes, no verbatim reproduction.
        assert "never reproduce" in kb
        # Extend an existing skill rather than minting a near-duplicate.
        assert "fold-in" in kb
        # Large inputs must be persisted incrementally instead of overflowing
        # the live conversation context before any reference file is written.
        assert "one chapter or topic at a time" in kb
        assert "never load an entire large corpus" in kb
        assert "reconcile the skill.md index" in kb

    def test_prompt_embeds_all_three_standards_blocks(self):
        prompt = build_learn_prompt("~/books/ddia.pdf")
        assert _AUTHORING_STANDARDS in prompt
        assert _KNOWLEDGE_SKILL_STANDARDS in prompt
        assert _SOURCE_HYGIENE in prompt
        # The shape decision is explicit: small source -> one file, large
        # prose source -> knowledge-base layout.
        assert "Pick the shape by the source" in prompt
        assert "process it incrementally in step 2b" in prompt

    def test_source_hygiene_covers_invisible_unicode(self):
        # Extracted document text is an injection vector (Trojan Source /
        # invisible code points). The prompt must pin source text as data and
        # name the invisible/bidi character classes to drop.
        hyg = _SOURCE_HYGIENE.lower()
        assert "data, not instructions" in hyg
        assert "zero-width" in hyg
        assert "bidi" in hyg or "bidirectional" in hyg

    def test_existing_skill_is_extended_instead_of_created_again(self):
        prompt = build_learn_prompt("add these notes to my distributed-systems skill")
        assert "First check the available skills" in prompt
        assert "If one exists, load it with `skill_view`" in prompt
        assert "Only when no matching skill exists" in prompt
        assert 'action="create"' in prompt


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
        assert "SKILL.md" in prompt

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
