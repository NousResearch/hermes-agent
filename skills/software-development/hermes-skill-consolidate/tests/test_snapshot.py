from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "snapshot_skills.py"
SPEC = importlib.util.spec_from_file_location("snapshot_skills", MODULE_PATH)
assert SPEC and SPEC.loader
snapshot_skills = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(snapshot_skills)


class SnapshotTests(unittest.TestCase):
    def make_skill(self, root: Path, name: str, files: dict[str, str]) -> None:
        skill = root / name
        skill.mkdir(parents=True)
        for relative, content in files.items():
            path = skill / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def test_create_and_verify_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills = tmp_path / "skills"
            skills.mkdir()
            self.make_skill(skills, "alpha-skill", {"SKILL.md": "alpha", "refs/a.md": "one"})
            self.make_skill(skills, "beta-skill", {"SKILL.md": "beta"})
            output = tmp_path / "backup"

            manifest = snapshot_skills.snapshot(
                skills, output, ["alpha-skill", "beta-skill"]
            )

            self.assertEqual(manifest["skills"], ["alpha-skill", "beta-skill"])
            self.assertEqual(len(manifest["files"]), 3)
            verified = snapshot_skills.verify(output)
            self.assertEqual(verified["files"], manifest["files"])

    def test_output_inside_live_tree_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills = tmp_path / "skills"
            skills.mkdir()
            self.make_skill(skills, "alpha-skill", {"SKILL.md": "alpha"})
            with self.assertRaisesRegex(ValueError, "outside the live skills tree"):
                snapshot_skills.snapshot(
                    skills, skills / "_backup", ["alpha-skill"]
                )

    def test_tamper_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills = tmp_path / "skills"
            skills.mkdir()
            self.make_skill(skills, "alpha-skill", {"SKILL.md": "alpha"})
            output = tmp_path / "backup"
            snapshot_skills.snapshot(skills, output, ["alpha-skill"])

            copied = output / "skills" / "alpha-skill" / "SKILL.md"
            copied.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                snapshot_skills.verify(output)

    def test_extra_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills = tmp_path / "skills"
            skills.mkdir()
            self.make_skill(skills, "alpha-skill", {"SKILL.md": "alpha"})
            output = tmp_path / "backup"
            snapshot_skills.snapshot(skills, output, ["alpha-skill"])

            extra = output / "skills" / "alpha-skill" / "extra.txt"
            extra.write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                snapshot_skills.verify(output)

    def test_invalid_skill_name_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skills = tmp_path / "skills"
            skills.mkdir()
            with self.assertRaisesRegex(ValueError, "invalid skill name"):
                snapshot_skills.snapshot(skills, tmp_path / "backup", ["../escape"])


if __name__ == "__main__":
    unittest.main()
