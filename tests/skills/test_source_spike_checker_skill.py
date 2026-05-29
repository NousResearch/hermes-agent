"""Surface + behavior tests for the source-spike-artifact-checker skill."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = REPO_ROOT / "skills" / "software-development" / "source-spike-artifact-checker"
SCRIPTS_DIR = SKILL_DIR / "scripts"
TEMPLATE = SKILL_DIR / "templates" / "closure-summary-template.md"
SCHEMA = SKILL_DIR / "references" / "source-spike-artifact-schema.md"

sys.path.insert(0, str(SCRIPTS_DIR))
import source_spike_checker as ssc  # noqa: E402


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


# --- skill surface -------------------------------------------------------

def test_skill_dir_and_files_exist() -> None:
    assert SKILL_DIR.is_dir(), f"missing skill dir: {SKILL_DIR}"
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert TEMPLATE.is_file(), f"missing template: {TEMPLATE}"
    assert SCHEMA.is_file(), f"missing schema reference: {SCHEMA}"
    assert (SCRIPTS_DIR / "source_spike_checker.py").is_file()


def test_frontmatter_is_valid(frontmatter) -> None:
    assert frontmatter["name"] == "source-spike-artifact-checker"
    assert frontmatter["description"], "description must be present"
    assert len(str(frontmatter["description"])) <= 1024  # MAX_DESCRIPTION_LENGTH


def test_skill_body_uses_repo_relative_invocation() -> None:
    body = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert "skills/software-development/source-spike-artifact-checker/scripts/source_spike_checker.py" in body
    # No leftover machine-specific absolute path.
    assert "/home/filip/" not in body


# --- checker behavior ----------------------------------------------------

def test_canonical_template_passes_every_gate() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    checks = ssc.validate_text(text, source_spike_path=None)
    failed = [c.name for c in checks if not c.passed]
    assert not failed, f"template should pass all gates, missing: {failed}"


def test_template_directory_validates_pass(tmp_path: Path) -> None:
    closure = tmp_path / "closure-summary.md"
    closure.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "source-spike.md").write_text("# spike\n", encoding="utf-8")
    result = ssc.validate_target(tmp_path)
    assert result.passed, [c.name for c in result.checks if not c.passed]


def test_missing_gates_fail(tmp_path: Path) -> None:
    closure = tmp_path / "closure-summary.md"
    closure.write_text("# closure\n\nNothing useful here.\n", encoding="utf-8")
    result = ssc.validate_target(tmp_path)
    assert not result.passed
    missing = {c.name for c in result.checks if not c.passed}
    assert "verdict" in missing
    assert "CLOSE_READY explicit" in missing


def test_missing_closure_summary_is_access_error(tmp_path: Path) -> None:
    (tmp_path / "source-spike.md").write_text("# spike\n", encoding="utf-8")
    result = ssc.validate_target(tmp_path)
    assert result.access_error == "missing closure-summary.md"
    assert not result.passed


def test_exit_codes_via_main(tmp_path: Path, capsys) -> None:
    good = tmp_path / "good"
    good.mkdir()
    (good / "closure-summary.md").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    assert ssc.main([str(good)]) == 0

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "closure-summary.md").write_text("# closure\n", encoding="utf-8")
    assert ssc.main([str(bad)]) == 1

    missing = tmp_path / "missing"
    missing.mkdir()
    assert ssc.main([str(missing)]) == 2
