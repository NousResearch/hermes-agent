from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "agent-safety"
    / "credibility-action-gate"
)


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert src.startswith("---\n")
    _, raw, _ = src.split("---", 2)
    return yaml.safe_load(raw)


def test_skill_files_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()
    assert (SKILL_DIR / "scripts" / "credibility-coordinator.mjs").is_file()
    assert (SKILL_DIR / "scripts" / "test-credibility-coordinator.mjs").is_file()
    assert (SKILL_DIR / "references" / "lane_contracts.md").is_file()
    assert (SKILL_DIR / "references" / "policy-template.json").is_file()


def test_frontmatter_matches_optional_skill_contract(frontmatter: dict) -> None:
    assert frontmatter["name"] == "credibility-action-gate"
    assert len(frontmatter["description"]) <= 60
    assert frontmatter["version"] == "1.0.0"
    assert frontmatter["license"] == "MIT"
    assert "Ales375" in frontmatter["author"]
    assert set(frontmatter["platforms"]) == {"linux", "macos", "windows"}
    hermes = frontmatter["metadata"]["hermes"]
    assert "agent-safety" in hermes["tags"]
    assert hermes["requires_toolsets"] == ["terminal"]


def test_policy_template_parses() -> None:
    policy = json.loads((SKILL_DIR / "references" / "policy-template.json").read_text(encoding="utf-8"))
    assert policy["disposition_preferences"]["missing_required_lane"] == "monitor_until_new_evidence"


def test_coordinator_has_no_network_or_secret_access() -> None:
    src = (SKILL_DIR / "scripts" / "credibility-coordinator.mjs").read_text(encoding="utf-8")
    forbidden = ["fetch(", "XMLHttpRequest", "http.request", "https.request", "process.env"]
    assert not any(token in src for token in forbidden)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for coordinator smoke tests")
def test_coordinator_scripts_parse() -> None:
    for script in [
        SKILL_DIR / "scripts" / "credibility-coordinator.mjs",
        SKILL_DIR / "scripts" / "test-credibility-coordinator.mjs",
    ]:
        subprocess.run(["node", "--check", str(script)], check=True, capture_output=True, text=True)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for coordinator smoke tests")
def test_coordinator_regression_suite_passes() -> None:
    subprocess.run(
        ["node", str(SKILL_DIR / "scripts" / "test-credibility-coordinator.mjs")],
        check=True,
        cwd=SKILL_DIR,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for coordinator smoke tests")
def test_no_lanes_fail_closed() -> None:
    result = subprocess.run(
        [
            "node",
            str(SKILL_DIR / "scripts" / "credibility-coordinator.mjs"),
            "--policy",
            str(SKILL_DIR / "references" / "policy-template.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["disposition"] == "monitor_until_new_evidence"
    assert rendered["maximum_recommended_size"] == 0
    assert rendered["missing_lanes"] == ["evidence", "external_context"]
