"""Tests for the multi-session-coordination optional skill.

Covers the skill contract (frontmatter hardline, body structure, referenced
support files) plus a compile check for the shipped CLI. No live network;
the CLI's own selftest suites use scratch databases and are documented in the
skill's Verification section. Cross-platform CI results must be verified per revision.
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
    procedure = body.split("## Procedure", 1)[1].split("\n## ", 1)[0]
    steps = re.findall(r"^### (\d+)\. (.+)$", procedure, re.M)
    assert steps, "Procedure must expose ordered, actionable sections"
    assert [int(n) for n, _ in steps] == list(range(1, len(steps) + 1))
    assert "completion criterion" in procedure.lower()
    assert "CLAIMED" in procedure and "Exit 75" in procedure


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


def test_complete_claim_is_revalidated_after_release(tmp_path, monkeypatch):
    import json
    import os
    import subprocess
    import sys

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("HERMES_COORD_DB", str(tmp_path / "board.db"))
    script = SKILL_DIR / "scripts/session_coord.py"

    def run(*args):
        return subprocess.run([sys.executable, str(script), *args], env=os.environ.copy(),
                              capture_output=True, text=True, timeout=15, check=False)

    for actor in ("holder", "waiter"):
        assert run("register", "--id", actor, "--task", "contract test").returncode == 0
    assert run("claim", "--id", "holder", "--res", "res:second").returncode == 0
    assert run("claim", "--id", "waiter", "--res", "res:first", "--res", "res:second").returncode == 75
    state = json.loads(run("status", "--json").stdout)
    assert not any(row["session_full"] == "waiter" for row in state["held_claims"])
    assert run("done", "--id", "holder").returncode == 0
    assert run("claim", "--id", "waiter", "--res", "res:first", "--res", "res:second").returncode == 0
    state = json.loads(run("status", "--json").stdout)
    assert {row["resource"] for row in state["held_claims"] if row["session_full"] == "waiter"} == {"res:first", "res:second"}
