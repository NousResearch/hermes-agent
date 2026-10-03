from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "software-development"
    / "foundry-security-reviewer"
    / "SKILL.md"
)
SCRIPT_PATH = SKILL_PATH.parent / "scripts" / "run_review.sh"


def _frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None
    return match.group(1)


def _fake_forge(tmp_path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    forge = bin_dir / "forge"
    forge.write_text(
        "#!/bin/sh\n"
        "case \"$1\" in\n"
        "  coverage)\n"
        "    printf '%s\\n' \"$FAKE_COVERAGE_OUTPUT\"\n"
        "    exit \"$FAKE_COVERAGE_STATUS\"\n"
        "    ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    forge.chmod(0o755)
    return bin_dir


def _run_review(tmp_path, coverage_output: str, coverage_status: int):
    project = tmp_path / "project"
    project.mkdir()
    (project / "foundry.toml").write_text("[profile.default]\n", encoding="utf-8")
    fake_bin = _fake_forge(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["FAKE_COVERAGE_OUTPUT"] = coverage_output
    env["FAKE_COVERAGE_STATUS"] = str(coverage_status)

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), str(project)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return result, (project / "review_output.md").read_text(encoding="utf-8")


def test_skill_metadata_and_modern_sections():
    text = SKILL_PATH.read_text(encoding="utf-8")
    metadata = _frontmatter(text)
    description = re.search(r'^description: "([^"]+)"$', metadata, re.MULTILINE)

    assert description is not None
    assert len(description.group(1)) <= 60
    assert description.group(1).endswith(".")
    assert "author: Ahmet Osrak (Osraka), Hermes Agent" in metadata
    assert "platforms: [linux, macos]" in metadata

    sections = re.findall(r"^## (.+)$", text, re.MULTILINE)
    assert sections[:7] == [
        "When to Use",
        "Prerequisites",
        "How to Run",
        "Quick Reference",
        "Procedure",
        "Pitfalls",
        "Verification",
    ]


def test_failed_coverage_is_reported_as_unavailable(tmp_path):
    result, report = _run_review(tmp_path, "partial coverage output", 1)

    assert result.returncode == 1
    assert "Coverage threshold result unavailable" in report
    assert "No module below 80% was detected" not in report


def test_parseable_coverage_flags_low_modules(tmp_path):
    coverage = "| File | % Lines | % Statements |\n| src/Vault.sol | 72% | 80% |"
    result, report = _run_review(tmp_path, coverage, 0)

    assert result.returncode == 0
    assert "src/Vault.sol" in report
    assert "below 80%" in report
