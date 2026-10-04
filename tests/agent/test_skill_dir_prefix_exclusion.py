"""Underscore-prefixed skill trees must be as disabled as dot-prefixed ones (#132917).

``EXCLUDED_SKILL_DIRS`` alone only prunes an exact name list, so ``_archive`` and
``_staging-*`` trees were discovered as live skills. The prefix rule and the explicit
list now live in one predicate shared by every scanner.
"""

from pathlib import Path

from agent.skill_utils import (
    is_disabled_skill_dir,
    is_excluded_skill_path,
    iter_skill_index_files,
)


def _mk_skill(base, rel, name=None):
    skill_dir = base / Path(rel)
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name or skill_dir.name}\ndescription: test skill\n---\n",
        encoding="utf-8",
    )
    return skill_dir / "SKILL.md"


class TestDisabledSkillDirPredicate:
    def test_underscore_prefixed_component_disables(self):
        assert is_disabled_skill_dir("_archive") is True
        assert is_disabled_skill_dir("_staging-matt") is True

    def test_dot_prefixed_component_disables(self):
        assert is_disabled_skill_dir(".archive") is True
        assert is_disabled_skill_dir(".git") is True

    def test_listed_plain_names_still_disable(self):
        assert is_disabled_skill_dir("venv") is True
        assert is_disabled_skill_dir("node_modules") is True
        assert is_disabled_skill_dir("site-packages") is True

    def test_normal_names_stay_discoverable(self):
        assert is_disabled_skill_dir("computer-use") is False
        assert is_disabled_skill_dir("engineering") is False


class TestExcludedSkillPath:
    def test_issue_paths_are_excluded(self, tmp_path):
        # The two live trees from the report, scoped to the skills root.
        assert (
            is_excluded_skill_path(
                tmp_path / "_archive/20260825/prototype" / "SKILL.md", root=tmp_path
            )
            is True
        )
        assert (
            is_excluded_skill_path(
                tmp_path / "_staging-matt/engineering/pr" / "SKILL.md", root=tmp_path
            )
            is True
        )

    def test_relative_paths_apply_prefix_rule(self):
        assert is_excluded_skill_path("_archive/20260825/prototype/SKILL.md") is True
        assert is_excluded_skill_path("_staging-matt/engineering/pr/SKILL.md") is True
        assert is_excluded_skill_path("computer-use/SKILL.md") is False

    def test_absolute_path_without_root_keeps_exact_names(self, tmp_path):
        """A bare absolute path must not let ancestors like ``~/.hermes`` disable
        the tree — the prefix rule only knows components below a root (#132917)."""
        home = tmp_path / ".hermes" / "skills"
        assert is_excluded_skill_path(home / "computer-use" / "SKILL.md") is False

    def test_dot_archive_still_excluded(self, tmp_path):
        assert (
            is_excluded_skill_path(tmp_path / ".archive" / "computer-use" / "SKILL.md")
            is True
        )

    def test_normal_skill_not_excluded(self, tmp_path):
        assert (
            is_excluded_skill_path(
                tmp_path / "computer-use" / "SKILL.md", root=tmp_path
            )
            is False
        )
        assert is_excluded_skill_path(tmp_path / "computer-use" / "SKILL.md") is False

    def test_plain_listed_names_not_excluded(self, tmp_path):
        assert is_excluded_skill_path(tmp_path / "venv" / "lib" / "SKILL.md") is True
        assert (
            is_excluded_skill_path(tmp_path / "node_modules" / "pkg" / "SKILL.md")
            is True
        )

    def test_prefix_rule_applies_at_any_depth(self, tmp_path):
        assert (
            is_excluded_skill_path(
                tmp_path / "category" / "_hidden" / "SKILL.md", root=tmp_path
            )
            is True
        )

    def test_root_scoped_prefix_survives_listed_ancestor(self, tmp_path):
        """With root given, a dot-prefixed ancestor above the root is ignored and
        a prefix dir below it still disables."""
        home = tmp_path / ".hermes" / "skills"
        assert (
            is_excluded_skill_path(home / "_archive" / "old" / "SKILL.md", root=home)
            is True
        )
        assert (
            is_excluded_skill_path(home / "computer-use" / "SKILL.md", root=home)
            is False
        )


class TestDiscoveryWalkPrunesPrefixDirs:
    def test_underscore_trees_not_discovered(self, tmp_path):
        live = _mk_skill(tmp_path, "computer-use")
        _mk_skill(tmp_path, "_archive/20260825/prototype")
        _mk_skill(tmp_path, "_staging-matt/engineering/pr")

        found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

        assert found == [live]

    def test_dot_trees_not_discovered(self, tmp_path):
        live = _mk_skill(tmp_path, "bash-helper")
        _mk_skill(tmp_path, ".archive/old-skill")

        found = list(iter_skill_index_files(tmp_path, "SKILL.md"))

        assert found == [live]


class TestSkillsManifestPrunesPrefixDirs:
    def test_underscore_trees_not_in_manifest(self, tmp_path):
        from agent.prompt_builder import _build_skills_manifest

        _mk_skill(tmp_path, "computer-use")
        _mk_skill(tmp_path, "_archive/20260825/prototype")

        manifest = _build_skills_manifest(tmp_path)

        assert set(manifest) == {"computer-use/SKILL.md"}
