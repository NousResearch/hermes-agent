"""Tests for optional-skills/productivity/memory-extension.

Covers the SKILL.md contract and the check-memory-coherence.sh script logic:
orphan detection, dangling-reference detection, and clean-state exit.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).resolve().parents[2] / "optional-skills" / "productivity" / "memory-extension"
SCRIPT = SKILL_DIR / "scripts" / "check-memory-coherence.sh"


def _bash_binary():
    """Return a working bash binary.

    On Windows, plain `bash` may resolve to the WSL relay (/usr/bin/bash),
    which fails when invoked from a subprocess with a Windows cwd. Prefer
    git-bash explicitly; fall back to PATH `bash` (Linux/macOS CI).
    """
    if sys.platform == "win32":
        candidates = [
            Path(r"C:\Program Files\Git\bin\bash.exe"),
            Path(r"C:\Program Files (x86)\Git\bin\bash.exe"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
    found = shutil.which("bash")
    if found and "wsl" not in str(found).lower():
        return found
    return None


BASH = _bash_binary()


def _frontmatter():
    content = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    return yaml.safe_load(content[3 : m.start() + 3]), content


def _run_script(home: Path) -> subprocess.CompletedProcess:
    assert BASH, "no working bash binary found"
    return subprocess.run(
        [BASH, str(SCRIPT), str(home)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestFrontmatter:
    def test_required_fields(self):
        fm, _ = _frontmatter()
        for field in ("name", "description", "version", "author", "license", "platforms"):
            assert field in fm, f"missing frontmatter field: {field}"

    def test_name_matches_directory(self):
        fm, _ = _frontmatter()
        assert fm["name"] == SKILL_DIR.name

    def test_description_hardline(self):
        fm, _ = _frontmatter()
        desc = str(fm.get("description") or "")
        assert len(desc) <= 60, f"description {len(desc)} chars (hardline 60)"
        assert desc.rstrip().endswith(".")

    def test_platforms_cover_script_shell(self):
        fm, _ = _frontmatter()
        # the script is bash; Windows is covered via git-bash/MSYS
        assert "windows" in fm.get("platforms", [])

    def test_skill_references_existing_files(self):
        _, content = _frontmatter()
        refs = re.findall(r"references/([A-Za-z0-9_-]+\.md)", content)
        refs += re.findall(r"scripts/([A-Za-z0-9_-]+\.sh)", content)
        assert refs, "SKILL.md should reference its references/ and scripts/"
        for ref in refs:
            candidates = list((SKILL_DIR / "references").glob(ref)) + list(
                (SKILL_DIR / "scripts").glob(ref)
            )
            assert candidates, f"SKILL.md references missing file: {ref}"


@pytest.fixture()
def fake_home(tmp_path):
    """Builds a coherent extended-memory layout, returns (home, write_fn)."""
    home = tmp_path / "hermes"
    memories = home / "memories"
    (memories / "extended").mkdir(parents=True)
    (memories / "MEMORY.md").write_text(
        "Topic A — hint → see extended/topic-a.md\n", encoding="utf-8"
    )
    (memories / "USER.md").write_text("", encoding="utf-8")
    (memories / "extended" / "topic-a.md").write_text("# Topic A\n", encoding="utf-8")
    (memories / "extended" / "README.md").write_text("# guide\n", encoding="utf-8")
    return home


@pytest.mark.skipif(BASH is None, reason="no working bash binary found")
class TestCoherenceScript:
    def test_clean_state_exits_zero(self, fake_home):
        r = _run_script(fake_home)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "all referenced" in r.stdout

    def test_detects_orphan_file(self, fake_home):
        (fake_home / "memories" / "extended" / "orphan.md").write_text(
            "# orphan\n", encoding="utf-8"
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "orphan.md" in r.stdout

    def test_detects_dangling_reference(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Topic A — hint → see extended/topic-a.md\n"
            "Ghost — hint → see extended/ghost.md\n",
            encoding="utf-8",
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "ghost.md" in r.stdout

    def test_ignores_readme(self, fake_home):
        # README.md is the guide copy; it must not count as an orphan
        r = _run_script(fake_home)
        assert r.returncode == 0
        assert "README.md" not in r.stdout

    def test_detects_dangling_multidot_reference(self, fake_home):
        mem = fake_home / "memories" / "MEMORY.md"
        mem.write_text(
            "Topic A — hint → see extended/topic-a.md\n"
            "Ghost — hint → see extended/ghost.bar.md\n",
            encoding="utf-8",
        )
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "ghost.bar.md" in r.stdout

    def test_missing_index_reports_error(self, fake_home):
        (fake_home / "memories" / "MEMORY.md").unlink()
        (fake_home / "memories" / "USER.md").unlink()
        r = _run_script(fake_home)
        assert r.returncode == 1
        assert "no index" in r.stdout.lower()
