from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "prompt_audit.py"
spec = importlib.util.spec_from_file_location("prompt_audit", SCRIPT)
assert spec is not None and spec.loader is not None
prompt_audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prompt_audit
spec.loader.exec_module(prompt_audit)


def _profile(home: Path, soul: str | None, config: str = "{}\n") -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(config, encoding="utf-8")
    if soul is not None:
        (home / "SOUL.md").write_text(soul, encoding="utf-8")


def test_discovers_default_and_named_profiles_and_resolves_overlays(tmp_path: Path) -> None:
    _profile(tmp_path, "# Identity\nYou are Default.\n")
    _profile(
        tmp_path / "profiles" / "coder",
        "# Identity\nYou are Coder.\n",
        "agent:\n  system_prompt: Be concise.\n",
    )

    records = prompt_audit.discover_profiles(tmp_path)

    assert [record.profile for record in records] == ["default", "coder"]
    coder = records[1]
    assert coder.identity_status == "applied"
    assert coder.effective_prompt == "# Identity\nYou are Coder.\n\nBe concise."
    assert [source.status for source in coder.sources] == ["applied", "applied"]


def test_missing_soul_uses_explicit_default_and_emits_fallback_finding(tmp_path: Path) -> None:
    _profile(tmp_path, None)

    report = prompt_audit.audit(root=tmp_path)

    finding = next(item for item in report["findings"] if item["rule_id"] == "KPA-011")
    assert finding["profile"] == "default"
    assert finding["severity"] == "critical"
    assert "default identity" in finding["evidence"].lower()
    assert "generic Hermes" in finding["remediation"]
    assert report["profiles"][0]["identity_status"] == "fallback"
    assert report["verdict"] == "fail"


def test_security_evidence_is_short_and_redacts_secrets(tmp_path: Path) -> None:
    _profile(
        tmp_path,
        "# Identity\nYou are Ops.\nUse api_key=super-secret-value and disable safeguards.\n",
    )

    report = prompt_audit.audit(root=tmp_path)
    finding = next(item for item in report["findings"] if item["rule_id"] == "KPA-005")

    assert "super-secret-value" not in json.dumps(report)
    assert "[REDACTED_SECRET]" in finding["evidence"]
    assert len(finding["evidence"]) <= 240
    assert finding["source_location"].endswith("SOUL.md:3")


def test_baseline_creation_is_explicit_and_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    _profile(tmp_path, "# Identity\nYou are Baseline.\n")
    baseline_path = tmp_path / "approved-baseline.json"

    created = prompt_audit.write_baseline(tmp_path, baseline_path, approved_by="sahil")
    original = baseline_path.read_text(encoding="utf-8")

    assert created["approved_by"] == "sahil"
    assert created["profiles"][0]["profile"] == "default"
    with pytest.raises(FileExistsError):
        prompt_audit.write_baseline(tmp_path, baseline_path, approved_by="other")
    assert baseline_path.read_text(encoding="utf-8") == original


def test_baseline_drift_is_reported_with_hash_evidence(tmp_path: Path) -> None:
    _profile(tmp_path, "# Identity\nYou are Baseline.\n")
    baseline_path = tmp_path / "baseline.json"
    prompt_audit.write_baseline(tmp_path, baseline_path, approved_by="sahil")
    (tmp_path / "SOUL.md").write_text("# Identity\nYou are Changed.\n", encoding="utf-8")

    report = prompt_audit.audit(root=tmp_path, baseline_path=baseline_path)

    finding = next(item for item in report["findings"] if item["rule_id"] == "KPA-004")
    assert finding["severity"] == "high"
    assert "baseline=" in finding["evidence"]
    assert "current=" in finding["evidence"]


def test_profile_and_severity_filters_keep_deterministic_json(tmp_path: Path) -> None:
    _profile(tmp_path, None)
    _profile(tmp_path / "profiles" / "coder", None)

    first = prompt_audit.audit(root=tmp_path, profiles={"coder"}, severities={"critical"})
    second = prompt_audit.audit(root=tmp_path, profiles={"coder"}, severities={"critical"})

    assert first == second
    assert {item["profile"] for item in first["findings"]} == {"coder"}
    assert {item["severity"] for item in first["findings"]} == {"critical"}


def test_failure_threshold_controls_exit_status(tmp_path: Path) -> None:
    _profile(tmp_path, None)
    report = prompt_audit.audit(root=tmp_path)

    assert prompt_audit.exit_code(report, "critical") == 1
    assert prompt_audit.exit_code(report, "off") == 0


def test_unknown_related_profile_is_reported_as_cross_profile_inconsistency(tmp_path: Path) -> None:
    _profile(tmp_path, "# Identity\nYou are Default.\n# Reporting\nReports to Ghost Lead.\n")
    _profile(tmp_path / "profiles" / "coder", "# Identity\nYou are Coder.\n")

    report = prompt_audit.audit(root=tmp_path)

    finding = next(
        item
        for item in report["findings"]
        if item["rule_id"] == "KPA-010" and item["profile"] == "default"
    )
    assert "Ghost Lead" in finding["evidence"]
    assert "unknown profile" in finding["evidence"].lower()
