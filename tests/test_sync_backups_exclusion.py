"""Test: a SKILL.md inside a skill's .sync-backups/ dir must NOT be indexed
as a second skill candidate; the live skill must still resolve.

Run from the repo root:  python3 tests/test_sync_backups_exclusion.py
(kept standalone so it can run without pytest installed)
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.skill_utils import iter_skill_index_files, EXCLUDED_SKILL_DIRS


def build_skill(root: Path, with_backup: bool) -> Path:
    skill = root / "skills" / "project" / "demo"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\n---\n# Demo\n", encoding="utf-8"
    )
    if with_backup:
        backup = skill / ".sync-backups" / "20261112T181022Z"
        backup.mkdir(parents=True)
        (backup / "SKILL.md").write_text(
            "---\nname: demo\n---\n# Demo (pre-sync baseline)\n", encoding="utf-8"
        )
    return skill


def main() -> int:
    assert ".sync-backups" in EXCLUDED_SKILL_DIRS, (
        "EXCLUDED_SKILL_DIRS must contain '.sync-backups'"
    )

    tmp = Path(tempfile.mkdtemp(prefix="sync-backups-exclusion-"))
    try:
        build_skill(tmp, with_backup=False)
        without = [str(p) for p in iter_skill_index_files(tmp / "skills", "SKILL.md")]
        assert len(without) == 1, f"expected 1 candidate without backup, got {without}"

        build_skill(tmp, with_backup=True)
        with_b = [str(p) for p in iter_skill_index_files(tmp / "skills", "SKILL.md")]
        assert len(with_b) == 1, (
            f"backup SKILL.md indexed as a second candidate: {with_b}"
        )
        assert ".sync-backups" not in with_b[0], (
            f"candidate resolved from backup dir: {with_b}"
        )

        print("OK: .sync-backups excluded from discovery; live skill resolves as the only candidate.")
        return 0
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
