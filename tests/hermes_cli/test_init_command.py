"""Tests for /init — generate or update AGENTS.md from a project scan.

Covers the shared prompt builder (hermes_cli.init_command.build_init_prompt)
and the slash-command registry wiring. /init has no engine and no model tool:
it builds a guidance-laden prompt that the live agent runs as a normal turn
(the /learn pattern), so these are the load-bearing behavior contracts.
"""

from hermes_cli.init_command import (
    _QUALITY_BAR,
    build_init_prompt,
    build_init_prompt_for_cwd,
)


class TestBuildInitPrompt:
    def test_includes_the_cwd(self):
        prompt = build_init_prompt("/home/alice/projects/acme")
        assert "/home/alice/projects/acme" in prompt


    def test_fresh_generation_when_no_existing_file(self):
        prompt = build_init_prompt("/tmp/proj", existing_file=None)
        assert "generate an AGENTS.md" in prompt
        # No merge discipline block for a fresh file.
        assert "MERGE DISCIPLINE" not in prompt

    def test_merge_not_overwrite_when_existing_file_passed(self):
        existing = "# My Project\n\nAlways run `make lint` before committing.\n"
        prompt = build_init_prompt("/tmp/proj", existing_file=existing)
        low = prompt.lower()
        # Update mode, with explicit merge-not-overwrite discipline.
        assert "UPDATE the existing AGENTS.md" in prompt
        assert "merge" in low
        assert "do not overwrite" in low or "not overwrite" in low
        assert "preserve" in low
        # And it carries the current content so the agent can merge.
        assert existing.strip() in prompt


    def test_includes_extra_notes_verbatim(self):
        notes = "focus on the test setup, and mention the flaky e2e suite"
        prompt = build_init_prompt("/tmp/proj", extra=notes)
        assert notes in prompt


    def test_always_includes_the_quality_bar(self):
        for existing in (None, "# old content"):
            prompt = build_init_prompt("/tmp/proj", existing_file=existing)
            assert _QUALITY_BAR in prompt

    def test_quality_bar_demands_exact_commands_and_conciseness(self):
        low = _QUALITY_BAR.lower()
        assert "100 lines" in low
        assert "never invent" in low
        assert "no generic advice" in low


class TestBuildInitPromptForCwd:

    def test_reads_existing_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "# Existing\n\nRun `tox -e py311`.\n", encoding="utf-8"
        )
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path))
        assert "UPDATE the existing AGENTS.md" in prompt
        assert "Run `tox -e py311`." in prompt

    def test_passes_extra_through(self, tmp_path):
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path), extra="keep it short")
        assert "keep it short" in prompt


class TestInitRegistryWiring:
    def test_init_is_registered_and_resolves(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("init")
        assert cmd is not None
        assert cmd.name == "init"


    def test_init_works_on_the_gateway(self):
        # /init is a both-surfaces command like /learn, not CLI-only.
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS

        assert "init" in GATEWAY_KNOWN_COMMANDS

