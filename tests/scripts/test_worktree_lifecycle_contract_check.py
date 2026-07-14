from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts" / "worktree-lifecycle" / "contract_check.py"
CONTRACT = (
    ROOT
    / "team-shortcuts"
    / "payload"
    / "skills"
    / "prompt-shortcuts"
    / "references"
    / "worktree-lifecycle-contract.md"
)


class WorktreeLifecycleContractCheckTests(unittest.TestCase):
    def run_check(self, path):
        return subprocess.run(
            [sys.executable, str(CHECKER), str(path), "--json"],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_owner_approved_contract_passes(self):
        result = self.run_check(CONTRACT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('"shortcut_count": 30', result.stdout)

    def test_shared_folder_rule_is_blocked(self):
        text = CONTRACT.read_text(encoding="utf-8")
        text += "\nหลาย Cursor/AI ใช้โฟลเดอร์เดียวกันได้\n"
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "bad-shared-folder.md"
            fixture.write_text(text, encoding="utf-8")
            result = self.run_check(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertIn("หลาย Cursor/AI", result.stdout)

    def test_multiple_writer_rule_is_blocked(self):
        text = CONTRACT.read_text(encoding="utf-8") + "\nallow_multiple_writers: true\n"
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "bad-writers.md"
            fixture.write_text(text, encoding="utf-8")
            result = self.run_check(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertIn("allow_multiple_writers", result.stdout)

    def test_cleanup_by_age_rule_is_blocked(self):
        text = CONTRACT.read_text(encoding="utf-8") + "\ncleanup_by_age: true\n"
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "bad-cleanup.md"
            fixture.write_text(text, encoding="utf-8")
            result = self.run_check(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cleanup_by_age", result.stdout)

    def test_two_zone_autonomy_policy_is_required(self):
        text = CONTRACT.read_text(encoding="utf-8").replace(
            "โซน A — AI ทำต่อเองจนจบเฟส", "โซน A ที่ถูกลบ", 1
        )
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "missing-zone-a.md"
            fixture.write_text(text, encoding="utf-8")
            result = self.run_check(fixture)
        self.assertEqual(result.returncode, 1)
        self.assertIn("โซน A — AI ทำต่อเองจนจบเฟส", result.stdout)


if __name__ == "__main__":
    unittest.main()
