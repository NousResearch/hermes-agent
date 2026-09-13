from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = (
    REPO_ROOT
    / "optional-skills"
    / "software-development"
    / "project-security-reviewer"
    / "SKILL.md"
)
SCRIPT_PATH = SKILL_PATH.parent / "scripts" / "detect_project.sh"


def _frontmatter(text: str) -> str:
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None
    return match.group(1)


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


def test_detector_reports_nested_supported_manifests(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    (project / "foundry.toml").write_text("[profile.default]\n", encoding="utf-8")
    nested = project / "frontend"
    nested.mkdir()
    (nested / "package.json").write_text("{}\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), str(project)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "**Git repository:** yes" in result.stdout
    assert "Foundry / Solidity" in result.stdout
    assert "`foundry.toml`" in result.stdout
    assert "Node.js / TypeScript" in result.stdout
    assert "`frontend/package.json`" in result.stdout
