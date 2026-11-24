"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no engine and no model tool: it
builds a standards-guided prompt that the live agent runs as a normal turn, so
these are the load-bearing behavior contracts.
"""

from agent.learn_prompt import build_learn_prompt, get_skill_desc_limit


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt

    def test_measures_the_description_with_a_tool_not_by_counting(self):
        # Models operate on tokens, so "COUNT the characters" fails by
        # construction and ships over-limit descriptions (repeated create-gate
        # rejects in #110008). The prompt must demand a measured length.
        prompt = build_learn_prompt("~/books/ddia.pdf")
        low = prompt.lower()
        assert "count the characters" not in low
        assert "models cannot count" in low
        # The measuring command is present and reads the length from a file.
        assert "print(len(d))" in prompt
        assert "read_text(encoding=\"utf-8-sig\").strip()" in prompt

    def test_measuring_command_takes_no_shell_interpolated_text(self):
        # The description must never be pasted into a quoted `bash -c`
        # argument: an apostrophe there truncates the command and anything
        # after the break runs as a second command. The prompt routes the
        # sentence through a file written with `write_file` instead.
        prompt = build_learn_prompt("~/books/ddia.pdf")
        flat = " ".join(prompt.lower().split())
        assert "scratch file with `write_file`" in flat
        assert "never touches the shell" in flat
        assert "'<description>'" not in prompt

    def test_description_retry_loop_is_capped(self):
        prompt = build_learn_prompt("~/books/ddia.pdf")
        flat = " ".join(prompt.lower().split())
        assert "at most twice" in flat
        assert "never compose the skill.md body until the count passes" in flat

    def test_description_limit_matches_the_index_budget(self):
        # The instruction interpolates the live limit, not a hardcoded 60, so
        # raising SKILL_PROMPT_DESC_LIMIT does not leave /learn forcing the old
        # width.
        prompt = build_learn_prompt("~/books/ddia.pdf")
        limit = get_skill_desc_limit()
        assert f"<={limit} characters" in prompt
        assert f"truncates the description to {limit} chars" in prompt


class TestLearnRegistryWiring:
    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None and cmd.name == "learn"
        assert not cmd.cli_only
