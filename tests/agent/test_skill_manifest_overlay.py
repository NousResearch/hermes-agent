"""The proposed manifest overlay must produce the same index as real filesystem writes."""

from pathlib import Path

import pytest

from agent.skill_utils import iter_skill_index_files
from utils import atomic_write_text


@pytest.mark.parametrize("sample_path", ["references/sample/SKILL.md", "nested/deep/SKILL.md", "scripts/.venv/sample/SKILL.md"])
@pytest.mark.parametrize("remove_root", [False, True])
def test_overlay_matches_native_discovery_after_actual_writes(tmp_path, sample_path, remove_root):
    root = tmp_path / "skills" / "existing"
    root.mkdir(parents=True)
    manifest = root / "SKILL.md"
    manifest.write_text("Existing manifest.")
    target = root / sample_path
    changes = {target: True}
    if remove_root:
        changes[manifest] = False
    predicted = set(iter_skill_index_files(tmp_path / "skills", "SKILL.md", manifest_changes=changes))
    # Prediction must not mutate files or create future directories.
    assert manifest.read_text() == "Existing manifest."
    assert not target.parent.exists()
    target.parent.mkdir(parents=True)
    atomic_write_text(target, "Sample manifest.")
    if remove_root:
        manifest.unlink()
    assert predicted == set(iter_skill_index_files(tmp_path / "skills", "SKILL.md"))


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("dangling", [False, True])
def test_overlay_keeps_lexical_directory_aliases(tmp_path, dangling):
    root = tmp_path / "skills" / "existing"
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text("Existing manifest.")
    target = root / "references" / "sample" / "deeper" / "SKILL.md"
    if not dangling:
        target.parent.parent.mkdir(parents=True)
    (root / "alias").symlink_to(target.parent.parent, target_is_directory=True)
    predicted = set(iter_skill_index_files(tmp_path / "skills", "SKILL.md", manifest_changes={target: True}))
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, "Sample manifest.")
    assert predicted == set(iter_skill_index_files(tmp_path / "skills", "SKILL.md"))
    assert root / "alias" / "deeper" / "SKILL.md" in predicted


@pytest.mark.platforms("posix")
def test_final_manifest_symlink_is_followed_by_atomic_write(tmp_path):
    root = tmp_path / "skills" / "existing"
    root.mkdir(parents=True)
    manifest = root / "SKILL.md"
    manifest.write_text("Existing manifest.")
    sample = root / "references" / "sample" / "SKILL.md"
    sample.parent.mkdir(parents=True)
    sample.write_text("Sample manifest.")
    link = root / "nested" / "SKILL.md"
    link.parent.mkdir()
    link.symlink_to(sample)
    # Hermes atomic writes preserve the symlink and replace its real target.
    predicted = set(iter_skill_index_files(tmp_path / "skills", "SKILL.md", manifest_changes={sample: True}))
    atomic_write_text(link, "Replaced manifest.")
    assert sample.read_text() == "Replaced manifest."
    assert link.is_symlink()
    assert predicted == set(iter_skill_index_files(tmp_path / "skills", "SKILL.md"))


@pytest.mark.platforms("posix")
def test_overlay_reports_cyclic_directory_alias_instead_of_hanging(tmp_path):
    root = tmp_path / "skills"
    root.mkdir()
    (root / "loop").symlink_to(root, target_is_directory=True)
    with pytest.raises(OSError, match="Cyclic"):
        list(iter_skill_index_files(root, "SKILL.md", manifest_changes={root / "SKILL.md": True}))
