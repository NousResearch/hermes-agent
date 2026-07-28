"""describe_skill_invocation() — recovering what the user typed from a /skill turn.

A /skill invocation expands into a message that embeds the whole skill body. Any
surface that summarizes a user turn from its raw content (session titles, sidebar
previews, the /rewind picker) otherwise shows the SKILL's prose as if the user had
written it.

Every case here builds the scaffolding with the real message builders rather than
a hand-written literal, so the description can't silently drift from the format it
parses.
"""

import pytest

import agent.skill_bundles as skill_bundles
import agent.skill_commands as skill_commands
import tools.skills_tool as skills_tool
from agent.skill_commands import (
    SKILL_EXCERPT_JOINT,
    SKILL_SCAFFOLD_SQL_LIKE,
    describe_skill_invocation,
)

SKILL_BODY = "Kick off a task in a fresh isolated git worktree instead of the current checkout."


def _write_skill(skills_dir, name, body=SKILL_BODY):
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Description for {name}\n---\n\n# {name}\n\n{body}\n"
    )
    return skill_dir


def _write_bundle(bundles_dir, slug, skills):
    bundles_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"name: {slug}", "skills:"]
    lines.extend(f"  - {skill}" for skill in skills)
    (bundles_dir / f"{slug}.yaml").write_text("\n".join(lines) + "\n")


@pytest.fixture()
def skills(tmp_path, monkeypatch):
    """Install a 'work' skill and a 'demo' bundle, with caches reset."""
    skills_dir = tmp_path / "skills"
    bundles_dir = tmp_path / "skill-bundles"
    _write_skill(skills_dir, "work")
    _write_skill(skills_dir, "clean", body="Polish your own diff by hand.")
    _write_bundle(bundles_dir, "demo", ["work", "clean"])

    monkeypatch.setattr(skills_tool, "SKILLS_DIR", skills_dir)
    monkeypatch.setenv("HERMES_BUNDLES_DIR", str(bundles_dir))
    monkeypatch.setattr(skill_commands, "_skill_commands", {})
    monkeypatch.setattr(skill_commands, "_skill_commands_platform", None)
    monkeypatch.setattr(skill_bundles, "_bundles_cache", {})
    monkeypatch.setattr(skill_bundles, "_bundles_cache_mtime", None)
    skill_commands.scan_skill_commands()
    skill_bundles.scan_bundles()
    return skills_dir


class TestDescribeSkillInvocation:
    def test_passes_through_a_normal_message(self):
        assert describe_skill_invocation("fix the title leak") is None

    def test_ignores_non_string_content(self):
        assert describe_skill_invocation(None) is None
        assert describe_skill_invocation([{"type": "text", "text": "hi"}]) is None

    def test_recovers_the_typed_instruction(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        assert describe_skill_invocation(message) == "/work — fix the title leak"

    def test_bare_invocation_describes_the_skill_alone(self, skills):
        message = skill_commands.build_skill_invocation_message("/work")
        assert describe_skill_invocation(message) == "/work"

    def test_never_surfaces_the_skill_body(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        described = describe_skill_invocation(message)
        assert "worktree" not in described
        assert "IMPORTANT" not in described

    def test_runtime_note_is_not_part_of_the_instruction(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work",
            user_instruction="fix the title leak",
            runtime_note="the runtime detail",
        )
        assert describe_skill_invocation(message) == "/work — fix the title leak"

    def test_collapses_whitespace_in_a_multiline_instruction(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the\n  title\tleak"
        )
        assert describe_skill_invocation(message) == "/work — fix the title leak"

    def test_bundle_carries_its_typed_keys(self, skills):
        result = skill_bundles.build_bundle_invocation_message(
            "/demo", user_instruction="fix the title leak"
        )
        assert result is not None
        message, _, _ = result
        described = describe_skill_invocation(message)
        assert described.endswith("— fix the title leak")
        assert "worktree" not in described

    def test_stacked_skills_describe_every_typed_key(self, skills):
        result = skill_commands.build_stacked_skill_invocation_message(
            ["/work", "/clean"], user_instruction="fix the title leak"
        )
        assert result is not None
        message, _, _ = result
        assert describe_skill_invocation(message) == "/work /clean — fix the title leak"


class TestExcerptedScaffolding:
    """Preview queries hand over a head+tail excerpt, not the whole message."""

    def _excerpt(self, message, window=400):
        """Mirror what _preview_raw_select() hands to _shape_preview()."""
        flat = message.replace("\n", " ").replace("\r", " ")
        if len(message) <= window * 2:
            return flat[: window * 2]
        return flat[:window] + SKILL_EXCERPT_JOINT + flat[-window:]

    def test_excerpt_still_recovers_the_instruction(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        described = describe_skill_invocation(self._excerpt(message))
        assert described == "/work — fix the title leak"

    def test_description_never_runs_across_the_joint(self, skills):
        # A long body pushes the head window into the middle of the skill text;
        # the instruction is only present on the tail side.
        skill_md = skills / "work" / "SKILL.md"
        skill_md.write_text(skill_md.read_text().replace(SKILL_BODY, "filler line.\n" * 200))
        skill_commands._skill_commands = {}
        skill_commands._skill_commands_platform = None
        skill_commands.scan_skill_commands()
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
        )
        described = describe_skill_invocation(self._excerpt(message))
        assert SKILL_EXCERPT_JOINT not in described
        assert "filler line" not in described


class TestSqlLikePattern:
    def test_matches_the_prefix_the_builders_emit(self, skills):
        message = skill_commands.build_skill_invocation_message("/work")
        assert message.startswith(SKILL_SCAFFOLD_SQL_LIKE.rstrip("%"))

    def test_carries_no_like_wildcards_needing_escape(self):
        # The pattern is interpolated into SQL without an ESCAPE clause, so the
        # literal part must not contain '%' or '_'.
        assert "%" not in SKILL_SCAFFOLD_SQL_LIKE[:-1]
        assert "_" not in SKILL_SCAFFOLD_SQL_LIKE[:-1]
