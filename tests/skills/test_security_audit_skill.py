"""Invariants for the ported security-audit optional skill (cloudflare/security-audit-skill)."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SKILL_DIR = (
    Path(__file__).resolve().parents[2] / "optional-skills" / "security" / "security-audit"
)


def test_every_referenced_companion_and_validator_exists():
    """The hub links companions as references/X.md and validators as scripts/X;
    a moved or renamed file must fail here, not at audit time."""
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    links = set(re.findall(r"\]\((references/[A-Z][A-Z-]+\.md)\)", text))
    assert len(links) >= 4, links
    missing = [link for link in links if not (SKILL_DIR / link).exists()]
    assert not missing, missing
    for script in ("validate-findings.cjs", "validate-coverage-ledger.cjs", "report-schema.json"):
        assert f"scripts/{script}" in text or script == "report-schema.json"
        assert (SKILL_DIR / "scripts" / script).exists(), script
    # companion docs must point at the relocated validators, never the flat upstream path
    for doc in (SKILL_DIR / "references").glob("*.md"):
        assert "<skill-dir>/validate-" not in doc.read_text(encoding="utf-8"), doc.name


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_findings_validator_rejects_non_array_and_resolves_schema(tmp_path):
    """validate-findings.cjs loads report-schema.json via __dirname, so the two must
    stay co-located under scripts/; a non-array findings.json is a schema failure."""
    findings = tmp_path / "findings.json"
    findings.write_text(json.dumps({"not": "an array"}), encoding="utf-8")
    proc = subprocess.run(
        ["node", str(SKILL_DIR / "scripts" / "validate-findings.cjs"), str(findings)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "Failed to load report-schema.json" not in proc.stderr
    assert "expected array" in proc.stderr
