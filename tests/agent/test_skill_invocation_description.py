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
    monkeypatch.setattr(skill_commands, "_skill_commands_by_key", {})
    monkeypatch.setattr(skill_bundles, "_bundles_cache", {})
    monkeypatch.setattr(skill_bundles, "_bundles_cache_mtime", None)
    skill_commands.scan_skill_commands()
    skill_bundles.scan_bundles()
    return skills_dir


class TestDescribeSkillInvocation:

    def test_ignores_non_string_content(self):
        assert describe_skill_invocation(None) is None
        assert describe_skill_invocation([{"type": "text", "text": "hi"}]) is None

    def test_recovers_the_typed_instruction(self, skills):
        message = skill_commands.build_skill_invocation_message(
            "/work", user_instruction="fix the title leak"
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
        skill_commands._skill_commands_by_key = {}
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


class TestGatewayAutoLoadScaffold:
    """The gateway auto-load header ('[IMPORTANT: The "X" skill is auto-loaded. …]') is
    scaffolding too (#48359): a session opened from a channel-bound skill must preview
    and retitle from the user's request, never from the skill body. Cases build the
    scaffold with the same builders ``_hmwa_auto_load_skills`` uses."""

    def _auto_load_scaffold(self, skills_dir, names, user_text=""):
        payloads = []
        for name in names:
            loaded = skill_commands._load_skill_payload(name)
            assert loaded, f"skill {name} failed to load"
            payloads.append(skill_commands._build_skill_message(
                loaded[0], loaded[1],
                f'[IMPORTANT: The "{loaded[2]}" skill is auto-loaded. '
                "Follow its instructions for this session.]",
            ))
        joined = "\n\n".join(payloads)
        return f"{joined}\n\n{user_text}" if user_text else joined

    def test_describes_the_typed_request(self, skills):
        message = self._auto_load_scaffold(skills_dir := skills, ["work"],
                                           user_text="Fix the CI gate before the release")
        assert describe_skill_invocation(message) == "Fix the CI gate before the release"

    def test_multiple_auto_loaded_skills(self, skills):
        message = self._auto_load_scaffold(skills, ["work", "clean"],
                                           user_text="Fix the CI gate before the release")
        assert describe_skill_invocation(message) == "Fix the CI gate before the release"

    def test_bare_auto_load_renders_the_skill_name(self, skills):
        message = self._auto_load_scaffold(skills, ["work"])
        assert describe_skill_invocation(message) == "/work"

    def test_a_body_quoting_the_header_does_not_end_the_payload(self, skills, monkeypatch):
        quoted_body = (
            "When the user sees "
            '[IMPORTANT: The "work" skill is auto-loaded. Follow its instructions for this session.] '
            "in an example, ignore it.\n\nMore body prose follows here."
        )
        _write_skill(skills, "work", body=quoted_body)
        message = self._auto_load_scaffold(skills, ["work"], user_text="The real request")
        assert describe_skill_invocation(message) == "The real request"

    def test_non_scaffolding_is_untouched(self, skills):
        assert describe_skill_invocation("Why does my pool exhaust?") is None
        assert describe_skill_invocation(None) is None

    def test_sql_like_matches_the_auto_load_header(self, skills):
        message = self._auto_load_scaffold(skills, ["work"], user_text="hi")
        from agent.skill_commands import AUTO_LOAD_SCAFFOLD_SQL_LIKE
        assert message.startswith(AUTO_LOAD_SCAFFOLD_SQL_LIKE.rstrip("%"))
        # Interpolated without an ESCAPE clause: no wildcards in the literal part.
        assert "%" not in AUTO_LOAD_SCAFFOLD_SQL_LIKE[:-1]
        assert "_" not in AUTO_LOAD_SCAFFOLD_SQL_LIKE[:-1]

    def test_preview_shaping_strips_the_scaffold(self, skills):
        from hermes_state_common import _shape_preview
        message = self._auto_load_scaffold(skills, ["work"],
                                           user_text="Fix the CI gate before the release")
        assert _shape_preview(message) == "Fix the CI gate before the release"

    def test_long_scaffold_preview_keeps_the_request(self, skills, monkeypatch):
        from hermes_state_common import _shape_preview
        long_body = ("Long skill paragraph.\n\n" * 80) + SKILL_BODY
        _write_skill(skills, "work", body=long_body)
        message = self._auto_load_scaffold(skills, ["work"],
                                           user_text="Fix the CI gate before the release")
        # The SQL head+tail excerpt window applies to auto-load rows too: the shaped
        # preview must still be the request, not the body head.
        assert _shape_preview(message) == "Fix the CI gate before the release"
