"""Tests for the multi-session-coordination optional skill.

Covers the skill contract (frontmatter hardline, body structure, referenced
support files) plus a compile check for the shipped CLI. No live network;
the CLI's own bash selftest suites (127 checks, scratch DBs) are exercised by
the project's upstream CI and documented in the skill's Verification section.
"""
import py_compile
import re
from pathlib import Path

import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "autonomous-ai-agents"
    / "multi-session-coordination"
)
SKILL_PATH = SKILL_DIR / "SKILL.md"


def _frontmatter_and_body():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert content.startswith("---")
    m = re.search(r"\n---\s*\n", content[3:])
    assert m, "frontmatter must close with ---"
    fm = yaml.safe_load(content[3 : m.start() + 3])
    body = content[m.end() + 3 :]
    return fm, body


def test_skill_file_exists():
    assert SKILL_PATH.is_file()


def test_frontmatter_required_fields():
    fm, _ = _frontmatter_and_body()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "multi-session-coordination"
    assert fm["metadata"]["hermes"]["tags"]


def test_description_hardline():
    fm, _ = _frontmatter_and_body()
    desc = fm["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars; hardline is 60"
    assert desc.endswith(".")


def test_author_credits_human_first():
    fm, _ = _frontmatter_and_body()
    assert not fm["author"].startswith("Hermes Agent"), "human contributor must be credited first"
    assert "Tobias Musser" in fm["author"]


def test_platforms_audited():
    fm, _ = _frontmatter_and_body()
    assert set(fm["platforms"]) <= {"linux", "macos", "windows"}
    assert "linux" in fm["platforms"] or "macos" in fm["platforms"]


def test_body_structure():
    _, body = _frontmatter_and_body()
    for section in (
        "## When to Use",
        "## Prerequisites",
        "## How to Run",
        "## Quick Reference",
        "## Procedure",
        "## Pitfalls",
        "## Verification",
    ):
        assert section in body, f"missing section: {section}"
    assert len(SKILL_PATH.read_text(encoding="utf-8")) <= 100_000


def test_procedure_has_numbered_steps_with_completion_criteria():
    _, body = _frontmatter_and_body()
    steps = re.findall(r"\*\*\d+\.\s+[^*]+?\*\*", body)
    assert len(steps) >= 4, "Procedure must have numbered steps"
    # each step must be followed by concrete, checkable content (a command
    # block or an outcome statement) before the next heading
    proc = body.split("## Procedure", 1)[1]
    blocks = re.findall(r"\*\*\d+\..*?(?=\*\*\d+\.|^### |^## )", proc, re.M | re.S)
    for block in blocks:
        assert "```" in block or "rc 75" in block or "Delivers" in block, (
            f"step lacks checkable content: {block[:80]!r}"
        )


def test_referenced_support_files_exist():
    _, body = _frontmatter_and_body()
    for ref in re.findall(r"`(?:<scripts-dir>|scripts|templates|examples)/[^`]+`", body):
        rel = ref.strip("`")
        if rel.startswith("<scripts-dir>/"):
            rel = "scripts/" + rel[len("<scripts-dir>/"):]
        assert (SKILL_DIR / rel).exists(), f"SKILL.md references missing file: {rel}"
    for sub in ("scripts", "templates", "examples"):
        assert (SKILL_DIR / sub).is_dir(), f"missing bundle subdir: {sub}"


def test_cli_compiles():
    py_compile.compile(str(SKILL_DIR / "scripts" / "session_coord.py"), doraise=True)


def test_selftests_present_and_executable():
    for suite in (
        "selftest.sh",
        "selftest_priority.sh",
        "selftest_cron.sh",
        "selftest_toggle.sh",
    ):
        p = SKILL_DIR / "scripts" / suite
        assert p.is_file(), f"missing selftest suite: {suite}"
        assert p.stat().st_mode & 0o111, f"selftest not executable: {suite}"


def test_no_machine_local_paths():
    content = SKILL_PATH.read_text(encoding="utf-8")
    assert "/home/" not in content
    assert not re.search(r"[A-Z]:\\\\Users", content)
    for p in SKILL_DIR.rglob("*"):
        if p.is_file() and p.suffix in (".md", ".sh", ".py", ".json"):
            text = p.read_text(encoding="utf-8", errors="replace")
            assert "/Users/" not in text, f"machine-local path in {p.name}"
