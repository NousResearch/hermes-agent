"""Regression for hermes-agent PR #154680: skill snapshots are not discoverable skills."""

from pathlib import Path

from agent.skill_utils import iter_skill_index_files


def test_sync_backups_do_not_duplicate_live_skill(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    live = skills_dir / "project" / "demo" / "SKILL.md"
    live.parent.mkdir(parents=True)
    live.write_text("---\nname: demo\n---\n# Demo\n", encoding="utf-8")

    sibling = skills_dir / "project" / "other" / "SKILL.md"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("---\nname: other\n---\n# Other\n", encoding="utf-8")

    before = set(iter_skill_index_files(skills_dir, "SKILL.md"))
    assert before == {live, sibling}

    backup = live.parent / ".sync-backups" / "20261112T181022Z" / "SKILL.md"
    backup.parent.mkdir(parents=True)
    backup.write_text(live.read_text(encoding="utf-8"), encoding="utf-8")

    # A snapshot must not turn an otherwise resolvable skill into a duplicate.
    assert set(iter_skill_index_files(skills_dir, "SKILL.md")) == before
