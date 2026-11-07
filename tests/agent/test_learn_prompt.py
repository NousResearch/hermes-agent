"""Tests for /learn — open-ended skill distillation.

Covers the shared prompt builder (agent.learn_prompt.build_learn_prompt) and
the slash-command registry wiring. /learn has no engine and no model tool: it
builds a standards-guided prompt that the live agent runs as a normal turn, so
these are the load-bearing behavior contracts.
"""

from agent.learn_prompt import build_learn_prompt


class TestBuildLearnPrompt:
    def test_embeds_the_user_request_verbatim(self):
        req = "the REST client in ~/projects/acme-sdk, focus on auth"
        prompt = build_learn_prompt(req)
        assert req in prompt


class TestSourceRecording:
    """/learn must tell the agent to record where the skill came from. Nothing else
    writes this: the hub lock file only covers hub-installed skills, so a learned
    skill that records no source can never be re-checked against its material."""

    def test_prompt_carries_the_shared_sources_schema(self):
        prompt = build_learn_prompt("https://example.com/docs/guide")
        assert "metadata.hermes" in prompt
        assert "sources:" in prompt
        for key in ("url:", "repo:", "revision:", "path:", "retrieved:"):
            assert key in prompt, f"schema key {key} missing from the /learn prompt"

    def test_prompt_carries_no_real_date(self):
        # A literal date in the example gets copied into authored skills, where it
        # claims a source was read on a day it was not.
        import re

        prompt = build_learn_prompt("https://example.com/docs/guide")

        assert "YYYY-MM-DD" in prompt
        assert "today's date" in prompt.lower()
        assert not re.search(r"\b20\d\d-\d\d-\d\d\b", prompt), "the /learn prompt carries a real date"

    def test_home_paths_are_recorded_with_a_tilde(self):
        # Skills get shared: an expanded home path leaks the user's layout.
        prompt = build_learn_prompt("~/projects/app")
        assert "~/" in prompt
        assert "expanded" in prompt.lower()

    def test_conversation_and_pasted_text_record_no_source(self):
        prompt = build_learn_prompt("what we just did")
        sources_rule = [line for line in prompt.splitlines() if "source entry" in line.lower()]
        assert sources_rule, "the /learn prompt no longer says what records no source entry"
        assert any("conversation" in line.lower() and "pasted" in line.lower()
                   for line in sources_rule)

    def test_prompt_says_to_omit_revision_when_the_repo_was_not_checked_out(self):
        # The failure this prevents: reading a repo through web pages and then
        # inventing a plausible-looking SHA for a commit that was never read.
        prompt = " ".join(build_learn_prompt("https://github.com/owner/project").split()).lower()

        assert "revision" in prompt
        assert "did not check the repository out" in prompt
        assert "leave `revision` out" in prompt
        assert "invented sha" in prompt

    def test_re_reading_a_listed_source_updates_its_entry(self):
        # A refresh that keeps the old `retrieved` date looks like no refresh at all to
        # anything comparing the date, so a stale-source warning would never clear.
        prompt = " ".join(build_learn_prompt("refresh the deploy skill").split()).lower()

        assert "already listed" in prompt
        assert "update that entry's `retrieved`" in prompt
        assert "`revision` for a `repo`" in prompt

    def test_revision_is_only_attached_to_a_repo_source(self):
        """The revision rule, asserted from the example's shape rather than its wording."""
        lines = build_learn_prompt("https://github.com/owner/project").splitlines()
        revision_lines = [line for line in lines if line.strip().startswith("revision:")]

        assert len(revision_lines) == 1
        index = lines.index(revision_lines[0])
        owner = next(line for line in reversed(lines[:index]) if line.strip().startswith("- "))
        assert owner.strip().startswith("- repo:"), f"revision hangs off {owner.strip()!r}"


class TestSourcesBlockIsAcceptedByTheExistingReaders:
    """No new parser or validator is added here: the sources block has to pass the
    linter and frontmatter parsing that already exist, including when malformed."""

    SOURCES = """\
---
name: example-skill
description: Example skill for a sources test.
version: 0.1.0
author: Hermes
license: MIT
metadata:
  hermes:
    tags: [Example]
    sources:
      - url: https://example.com/docs/guide
        retrieved: 2026-10-22
      - repo: https://github.com/owner/project
        revision: 0123456789abcdef0123456789abcdef01234567
        retrieved: 2026-10-22
---

# Example

## When to Use

- A test needs a skill with a sources block.
"""

    def _lint_rules(self, content):
        from tools.skill_linter import lint_content

        return {finding.rule for finding in lint_content(content)}

    def _with_sources(self, replacement):
        """The fixture with its sources block swapped for ``replacement``."""
        original = (
            "    sources:\n"
            "      - url: https://example.com/docs/guide\n"
            "        retrieved: 2026-10-22\n"
            "      - repo: https://github.com/owner/project\n"
            "        revision: 0123456789abcdef0123456789abcdef01234567\n"
            "        retrieved: 2026-10-22")
        assert original in self.SOURCES, "fixture drift: the sources block moved"
        return self.SOURCES.replace(original, "    " + replacement)

    def test_a_valid_sources_block_raises_no_findings(self):
        assert not (self._lint_rules(self.SOURCES) & {"missing-metadata", "description-length"})

    def test_a_wrongly_shaped_sources_block_adds_no_finding(self):
        # Well-formed YAML in an unexpected shape is a reader's problem, not the linter's:
        # the block must cost the author no new warning.
        baseline = self._lint_rules(self.SOURCES)
        for broken in ("sources: not-a-list",
                       "sources:\n      - just-a-string",
                       "sources:\n      - url: https://example.com\n        path: /tmp/x"):
            assert self._lint_rules(self._with_sources(broken)) == baseline

    def test_invalid_yaml_in_a_sources_block_never_raises(self):
        # Pre-existing behaviour: unparseable YAML falls back to line splitting. The one
        # thing that must hold is that a bad sources block cannot break the linter.
        from tools.skill_linter import lint_content

        lint_content(self._with_sources("sources: [{url: [unclosed"))

    def test_frontmatter_parsing_keeps_the_block_verbatim(self):
        from agent.skill_utils import parse_frontmatter

        frontmatter, _ = parse_frontmatter(self.SOURCES)
        entries = frontmatter["metadata"]["hermes"]["sources"]
        assert [sorted(entry) for entry in entries] == [
            ["retrieved", "url"], ["repo", "retrieved", "revision"]]


class TestLearnRegistryWiring:
    def test_learn_is_not_cli_only(self):
        from hermes_cli.commands import resolve_command

        cmd = resolve_command("learn")
        assert cmd is not None and cmd.name == "learn"
        assert not cmd.cli_only
