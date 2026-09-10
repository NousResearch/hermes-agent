"""Tests for the board-gated-execution skill.

Scope: the SKILL.md contract (frontmatter, structure, no machine-local paths)
and the gate's selection logic, which is the part that must not drift. The gate
is exercised by monkeypatching the GitHub reads — no live network, per repo
policy.

The selection tests are written against the *decisions* the gate must make, not
against its current output strings, so they survive rewording.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "board-gated-execution"
)
SKILL_MD = SKILL_DIR / "SKILL.md"
SCRIPTS = SKILL_DIR / "scripts"


def load_frontmatter() -> dict:
    content = SKILL_MD.read_text()
    assert content.startswith("---"), "frontmatter must start at byte 0"
    match = re.search(r"\n---\s*\n", content[3:])
    assert match, "frontmatter must close with a --- line"
    return yaml.safe_load(content[3 : match.start() + 3])


# --------------------------------------------------------------------------
# SKILL.md contract
# --------------------------------------------------------------------------


def test_skill_md_exists():
    assert SKILL_MD.is_file()


def test_frontmatter_has_required_fields():
    fm = load_frontmatter()
    for field in ("name", "description", "version", "author", "license", "platforms"):
        assert field in fm, f"missing frontmatter field: {field}"
    assert fm["name"] == "board-gated-execution"


def test_description_fits_the_index_window():
    """The system prompt index truncates at 57 chars; the hardline is 60."""
    description = load_frontmatter()["description"]
    assert len(description) <= 60, f"{len(description)} chars"
    assert description.endswith(".")


def test_description_avoids_marketing_words():
    description = load_frontmatter()["description"].lower()
    for word in ("powerful", "comprehensive", "seamless", "advanced", "robust"):
        assert word not in description


def test_related_skills_are_declared():
    related = load_frontmatter()["metadata"]["hermes"]["related_skills"]
    assert related, "declare at least one related skill"
    assert "board-gated-execution" not in related, "a skill must not relate to itself"


def test_no_machine_local_paths():
    """A path from the author's machine breaks the skill for everyone else.

    Text files only: compiled artifacts (.pyc) embed absolute build paths and
    are not part of the contribution.
    """
    sources = [SKILL_MD, *(p for p in SCRIPTS.glob("*") if p.suffix in {".py", ".sh"})]
    for path in sources:
        text = path.read_text()
        assert "/home/" not in text, f"machine-local path in {path.name}"
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text), f"bare IP in {path.name}"


def test_body_has_the_required_sections():
    body = SKILL_MD.read_text()
    for heading in ("## When to Use", "## Prerequisites", "## Procedure",
                    "## Pitfalls", "## Verification"):
        assert heading in body, f"missing section: {heading}"


def test_counter_triggers_present():
    """`When to Use` must say when NOT to use the skill."""
    assert "Don't use for:" in SKILL_MD.read_text()


def test_scripts_referenced_by_the_skill_exist():
    body = SKILL_MD.read_text()
    for name in re.findall(r"scripts/([\w.-]+\.(?:py|sh))", body):
        assert (SCRIPTS / name).is_file(), f"SKILL.md references missing scripts/{name}"


# --------------------------------------------------------------------------
# Gate selection logic
# --------------------------------------------------------------------------


@pytest.fixture()
def gate(monkeypatch):
    """Load next-task.py with GitHub reads stubbed out."""
    spec = importlib.util.spec_from_file_location("next_task", SCRIPTS / "next-task.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "detect_repo", lambda: "owner/repo")
    return module


def issue(number: int, title: str, *labels: str) -> dict:
    return {"number": number, "title": title,
            "labels": [{"name": name} for name in labels]}


def run_gate(gate, monkeypatch, issues, capsys) -> tuple[int, str]:
    monkeypatch.setattr(gate, "fetch_issues", lambda repo: issues)
    code = gate.main(["next-task.py", "owner/repo"])
    return code, capsys.readouterr().out


def test_two_in_progress_issues_are_refused(gate, monkeypatch, capsys):
    """The WIP limit is the whole point: two in flight means one is drifting."""
    code, out = run_gate(gate, monkeypatch, [
        issue(1, "First", "in-progress", "P1"),
        issue(2, "Second", "in-progress", "P2"),
    ], capsys)
    assert code == 1
    assert "REFUSED" in out
    assert "WORK THIS" not in out, "a refusal must not also name a task"


def test_in_progress_beats_a_higher_priority_fresh_issue(gate, monkeypatch, capsys):
    code, out = run_gate(gate, monkeypatch, [
        issue(1, "Started", "in-progress", "P3"),
        issue(2, "Shiny", "P0"),
    ], capsys)
    assert code == 0
    assert "#1 Started" in out
    assert "WORK THIS" in out


def test_blocked_issues_are_never_selected(gate, monkeypatch, capsys):
    """Working a blocked issue means waiting, which is indistinguishable from drift."""
    code, out = run_gate(gate, monkeypatch, [
        issue(1, "Needs a credential", "blocked:human", "P0"),
        issue(2, "Actionable", "P3"),
    ], capsys)
    assert code == 0
    selected = out.split("BLOCKED ON A HUMAN")[0]
    assert "#2 Actionable" in selected
    assert "#1 Needs a credential" not in selected


def test_blocked_issues_are_still_surfaced(gate, monkeypatch, capsys):
    _, out = run_gate(gate, monkeypatch, [
        issue(1, "Needs a credential", "blocked:human", "P0"),
        issue(2, "Actionable", "P3"),
    ], capsys)
    assert "BLOCKED ON A HUMAN" in out
    assert "#1 Needs a credential" in out


def test_priority_order_is_respected(gate, monkeypatch, capsys):
    _, out = run_gate(gate, monkeypatch, [
        issue(3, "Someday", "P3"),
        issue(1, "Urgent", "P0"),
        issue(2, "Planned", "P2"),
    ], capsys)
    assert "#1 Urgent" in out.split("QUEUED")[0]


def test_unprioritised_issues_sort_last(gate, monkeypatch, capsys):
    _, out = run_gate(gate, monkeypatch, [
        issue(1, "No priority label"),
        issue(2, "Someday", "P3"),
    ], capsys)
    assert "#2 Someday" in out.split("QUEUED")[0]


def test_all_blocked_reports_nothing_actionable(gate, monkeypatch, capsys):
    """Saying so plainly beats inventing adjacent work."""
    code, out = run_gate(gate, monkeypatch, [
        issue(1, "Waiting on a decision", "blocked:human", "P1"),
    ], capsys)
    assert code == 0
    assert "Nothing actionable" in out
    assert "WORK THIS" not in out


def test_empty_board_is_reported_as_the_problem(gate, monkeypatch, capsys):
    code, out = run_gate(gate, monkeypatch, [], capsys)
    assert code == 0
    assert "empty" in out.lower()


def test_a_blocked_in_progress_issue_does_not_consume_the_wip_slot(gate, monkeypatch, capsys):
    """Parked-and-blocked work must not stop other work from starting."""
    code, out = run_gate(gate, monkeypatch, [
        issue(1, "Parked", "in-progress", "blocked:human", "P1"),
        issue(2, "Actionable", "P2"),
    ], capsys)
    assert code == 0
    assert "#2 Actionable" in out.split("BLOCKED ON A HUMAN")[0]
