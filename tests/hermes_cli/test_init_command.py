"""Tests for /init — generate or update AGENTS.md from a project scan.

Covers the shared prompt builder (hermes_cli.init_command.build_init_prompt)
and the slash-command registry wiring. /init has no engine and no model tool:
it builds a guidance-laden prompt that the live agent runs as a normal turn
(the /learn pattern), so these are the load-bearing behavior contracts.
"""

from hermes_cli.init_command import (
    build_init_prompt,
    build_init_prompt_for_cwd,
)


class TestBuildInitPrompt:





    def test_includes_extra_notes_verbatim(self):
        notes = "focus on the test setup, and mention the flaky e2e suite"
        prompt = build_init_prompt("/tmp/proj", extra=notes)
        assert notes in prompt





class TestBuildInitPromptForCwd:

    def test_reads_existing_agents_md(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(
            "# Existing\n\nRun `tox -e py311`.\n", encoding="utf-8"
        )
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path))
        assert "Run `tox -e py311`." in prompt

    def test_passes_extra_through(self, tmp_path):
        prompt = build_init_prompt_for_cwd(cwd=str(tmp_path), extra="keep it short")
        assert "keep it short" in prompt



