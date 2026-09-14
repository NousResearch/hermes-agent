"""
Smoke tests for the adversarial-debate optional skill.

Verifies:
  - SKILL.md frontmatter conforms to hardline format (description ≤60 chars)
  - All template files exist and contain valid JSON schemas
  - Template JSON is parseable and matches the expected schema
  - DQI calculation is deterministic and correct
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "software-development"
    / "adversarial-debate"
)


@pytest.fixture(scope="module")
def frontmatter() -> dict:
    src = (SKILL_DIR / "SKILL.md").read_text()
    m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
    assert m, "SKILL.md missing YAML frontmatter"
    return yaml.safe_load(m.group(1))


# ── Frontmatter checks ────────────────────────────────────────────────


def test_skill_dir_exists() -> None:
    assert SKILL_DIR.is_dir()


def test_skill_md_present() -> None:
    assert (SKILL_DIR / "SKILL.md").is_file()


def test_description_under_60_chars(frontmatter) -> None:
    desc = frontmatter["description"]
    assert len(desc) <= 60, f"description is {len(desc)} chars (hardline ≤60): {desc!r}"


def test_version_present(frontmatter) -> None:
    assert "version" in frontmatter, "missing version"


def test_author_present(frontmatter) -> None:
    assert "author" in frontmatter, "missing author"


def test_license_is_mit(frontmatter) -> None:
    assert frontmatter.get("license") == "MIT"


def test_platforms_defined(frontmatter) -> None:
    assert "platforms" in frontmatter
    assert isinstance(frontmatter["platforms"], list)
    assert len(frontmatter["platforms"]) > 0


def test_hermes_metadata_tags(frontmatter) -> None:
    meta = frontmatter.get("metadata", {}).get("hermes", {})
    assert "tags" in meta, "missing metadata.hermes.tags"
    assert isinstance(meta["tags"], list)
    assert len(meta["tags"]) >= 3


def test_hermes_metadata_related_skills(frontmatter) -> None:
    meta = frontmatter.get("metadata", {}).get("hermes", {})
    assert "related_skills" in meta, "missing metadata.hermes.related_skills"


# ── File existence checks ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "rel_path",
    [
        "templates/round-1-opening.md",
        "templates/round-2-cross-examination.md",
        "templates/round-3-convergence.md",
        "templates/synthesis-schema.json",
        "scripts/debate-orchestrator.sh",
    ],
)
def test_required_file_exists(rel_path: str) -> None:
    assert (SKILL_DIR / rel_path).is_file(), f"missing: {rel_path}"


# ── JSON schema validation ────────────────────────────────────────────


def test_synthesis_schema_is_valid_json() -> None:
    path = SKILL_DIR / "templates" / "synthesis-schema.json"
    schema = json.loads(path.read_text())
    assert "$schema" in schema
    assert "properties" in schema
    assert "required" in schema


def test_synthesis_schema_required_fields() -> None:
    path = SKILL_DIR / "templates" / "synthesis-schema.json"
    schema = json.loads(path.read_text())
    required = set(schema["required"])
    expected = {
        "debate_id", "topic", "format",
        "rounds_completed", "dqi", "dqi_assessment",
        "position_changes",
        "consensus_claims", "tensions",
        "minority_reports", "execute_automatically",
    }
    missing = expected - required
    assert not missing, f"Schema missing required fields: {missing}"


# ── DQI calculation correctness ───────────────────────────────────────


def _calculate_dqi(claims: list[dict]) -> float:
    evidence_mult = {"direct": 1.0, "inferred": 0.7, "speculative": 0.4}
    rebuttal_mult = {"unrebutted": 1.0, "conceded": 0.7, "unresolved": 0.5}
    weights = []
    for c in claims:
        conf = c.get("confidence", 50) / 100.0
        ev = evidence_mult.get(c.get("evidence_strength", "speculative"), 0.4)
        rb = rebuttal_mult.get(c.get("rebuttal_status", "unrebutted"), 1.0)
        weights.append(conf * ev * rb)
    return sum(weights) / len(weights) if weights else 0.0


def test_dqi_all_direct_unrebutted() -> None:
    claims = [
        {"confidence": 90, "evidence_strength": "direct", "rebuttal_status": "unrebutted"},
        {"confidence": 80, "evidence_strength": "direct", "rebuttal_status": "unrebutted"},
    ]
    dqi = _calculate_dqi(claims)
    expected = (0.9 * 1.0 * 1.0 + 0.8 * 1.0 * 1.0) / 2
    assert dqi == pytest.approx(expected)


def test_dqi_mixed_evidence_rebuttals() -> None:
    claims = [
        {"confidence": 100, "evidence_strength": "direct", "rebuttal_status": "unrebutted"},
        {"confidence": 50, "evidence_strength": "speculative", "rebuttal_status": "conceded"},
    ]
    dqi = _calculate_dqi(claims)
    expected = (1.0 * 1.0 * 1.0 + 0.5 * 0.4 * 0.7) / 2
    assert dqi == pytest.approx(expected)


def test_dqi_all_speculative_defended() -> None:
    claims = [
        {"confidence": 70, "evidence_strength": "speculative", "rebuttal_status": "defended"},
    ]
    dqi = _calculate_dqi(claims)
    expected = 0.7 * 0.4 * 1.0  # defended = unrebutted in this model
    assert dqi == pytest.approx(expected)


def test_dqi_empty_claims() -> None:
    assert _calculate_dqi([]) == 0.0
