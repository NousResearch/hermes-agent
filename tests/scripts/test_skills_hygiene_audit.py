"""Tests for skills-hygiene-audit.py."""
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\skills-hygiene-audit.py")
sys.path.insert(0, str(SCRIPT.parent))

import importlib.util
spec = importlib.util.spec_from_file_location("sha", SCRIPT)
sha = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sha)


def test_parse_frontmatter_basic():
    text = """---
name: my-skill
description: A skill that does things.
version: 1.0.0
---

# Body content
"""
    fm, body = sha.parse_frontmatter(text)
    assert fm["name"] == "my-skill"
    assert fm["description"] == "A skill that does things."
    assert fm["version"] == "1.0.0"
    assert "Body content" in body


def test_parse_frontmatter_no_frontmatter():
    text = "# Just a heading\nNo frontmatter here."
    fm, body = sha.parse_frontmatter(text)
    assert fm == {}
    assert "Just a heading" in body


def test_parse_frontmatter_unclosed():
    text = """---
name: orphan
no closing fence
"""
    fm, body = sha.parse_frontmatter(text)
    # No closing fence → empty frontmatter, full text in body
    assert fm == {}
    assert "name: orphan" in body


def test_parse_frontmatter_strips_quotes():
    text = """---
name: "quoted-name"
description: 'single-quoted'
---
"""
    fm, body = sha.parse_frontmatter(text)
    assert fm["name"] == "quoted-name"
    assert fm["description"] == "single-quoted"


# --- audit_skill ---

def test_audit_skill_clean():
    """A properly formatted skill returns no errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "clean-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            """---
name: clean-skill
description: A clean skill.
version: 1.0.0
---

Body.
""",
            encoding="utf-8",
        )
        issues = sha.audit_skill(skill_dir)
        errors = [i for i in issues if i["severity"] == "error"]
        assert errors == [], f"unexpected errors: {errors}"


def test_audit_skill_missing_frontmatter():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "broken-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# No frontmatter\n", encoding="utf-8")
        issues = sha.audit_skill(skill_dir)
        errors = [i for i in issues if i["severity"] == "error"]
        assert any(i["type"] == "missing-frontmatter" for i in errors)


def test_audit_skill_invalid_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "valid-dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: Invalid_Name_With_Caps\ndescription: x.\n---\n",
            encoding="utf-8",
        )
        issues = sha.audit_skill(skill_dir)
        assert any(i["type"] == "invalid-name" for i in issues)


def test_audit_skill_description_too_long():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "long-desc"
        skill_dir.mkdir()
        long_desc = "x" * 250
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: long-desc\ndescription: {long_desc}\n---\n",
            encoding="utf-8",
        )
        issues = sha.audit_skill(skill_dir)
        assert any(i["type"] == "description-too-long" for i in issues)


def test_audit_skill_size_warning():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "huge-skill"
        skill_dir.mkdir()
        # Write 60 KB of content
        big_content = "---\nname: huge-skill\ndescription: big.\n---\n" + "x" * 60_000
        (skill_dir / "SKILL.md").write_text(big_content, encoding="utf-8")
        issues = sha.audit_skill(skill_dir)
        assert any(i["type"] == "skill-md-too-large" for i in issues)


def test_audit_skill_name_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "actual-dir-name"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: different-name\ndescription: x.\n---\n",
            encoding="utf-8",
        )
        issues = sha.audit_skill(skill_dir)
        assert any(i["type"] == "name-mismatch" for i in issues)


def test_audit_skill_missing_skill_md():
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_dir = Path(tmpdir) / "no-skill-md"
        skill_dir.mkdir()
        issues = sha.audit_skill(skill_dir)
        assert any(i["type"] == "missing-skill-md" for i in issues)


# --- main / exit codes ---

def test_clean_skills_dir_returns_0(tmp_path):
    """A skills dir with only properly formatted skills returns exit 0."""
    skills = tmp_path / "skills"
    skills.mkdir()
    for i in range(3):
        sd = skills / f"skill-{i}"
        sd.mkdir()
        (sd / "SKILL.md").write_text(
            f"---\nname: skill-{i}\ndescription: Skill number {i}.\nversion: 1.0.0\n---\n",
            encoding="utf-8",
        )
    r = __import__("subprocess").run(
        [sys.executable, str(SCRIPT), "--skills-dir", str(skills)],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0
    assert "all 3 skills pass" in r.stdout


def test_dirty_skills_dir_returns_1(tmp_path):
    """A skills dir with broken skills returns exit 1."""
    skills = tmp_path / "skills"
    skills.mkdir()
    sd = skills / "broken"
    sd.mkdir()
    (sd / "SKILL.md").write_text("# no frontmatter\n", encoding="utf-8")
    r = __import__("subprocess").run(
        [sys.executable, str(SCRIPT), "--skills-dir", str(skills)],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 1


def test_missing_dir_returns_2(tmp_path):
    """A non-existent skills dir returns exit 2."""
    r = __import__("subprocess").run(
        [sys.executable, str(SCRIPT), "--skills-dir", str(tmp_path / "nope")],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 2