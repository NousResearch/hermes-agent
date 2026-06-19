from __future__ import annotations

import json
from pathlib import Path

from scripts import lcm_config_preflight as preflight


BASE_MANIFEST = {
    "threshold_config": {"context_threshold": 0.35, "leaf_chunk_tokens": 20_000},
    "redaction_ruleset": ["api_key", "bearer_token", "password_assignment", "private_key"],
    "schema_version": 4,
    "encryption_mode": "aead-v1",
}


def _write_manifest(path: Path, artifact_path: Path, **overrides) -> Path:
    manifest = {**BASE_MANIFEST, "plugin_artifact_path": str(artifact_path)}
    manifest.update(overrides)
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path


def test_preflight_passes_for_byte_identical_plugin_artifact_and_config(tmp_path: Path) -> None:
    artifact_a = tmp_path / "aegis-plugin.py"
    artifact_b = tmp_path / "apollo-plugin.py"
    artifact_a.write_bytes(b"byte-identical lcm plugin artifact\n")
    artifact_b.write_bytes(artifact_a.read_bytes())
    aegis = _write_manifest(tmp_path / "aegis.json", artifact_a)
    apollo = _write_manifest(tmp_path / "apollo.json", artifact_b)
    out_path = tmp_path / "preflight.md"

    result = preflight.run_preflight(aegis_report=aegis, apollo_target=apollo, out_path=out_path)

    assert result.passed, "\n".join(result.failures)
    assert result.status == "PASS"
    assert result.comparisons["plugin_artifact_sha256"].passed is True
    assert result.comparisons["threshold_config"].passed is True
    assert result.comparisons["redaction_ruleset"].passed is True
    assert result.comparisons["schema_version"].passed is True
    assert result.comparisons["encryption_mode"].passed is True

    report = out_path.read_text(encoding="utf-8")
    assert "# PRD-6 LCM Config/Hash Preflight" in report
    assert "Status: PASS" in report
    assert "plugin_artifact_sha256" in report
    assert "threshold_config" in report


def test_preflight_fails_loud_on_artifact_config_hash_drift(tmp_path: Path) -> None:
    artifact_a = tmp_path / "aegis-plugin.py"
    artifact_b = tmp_path / "apollo-plugin.py"
    artifact_a.write_bytes(b"aegis artifact\n")
    artifact_b.write_bytes(b"apollo drifted artifact\n")
    aegis = _write_manifest(tmp_path / "aegis.json", artifact_a)
    apollo = _write_manifest(
        tmp_path / "apollo.json",
        artifact_b,
        threshold_config={"context_threshold": 0.50, "leaf_chunk_tokens": 20_000},
        redaction_ruleset=["api_key", "bearer_token"],
        schema_version=5,
        encryption_mode="plaintext",
    )

    result = preflight.run_preflight(aegis_report=aegis, apollo_target=apollo, out_path=None)

    assert not result.passed
    assert result.status == "FAIL-LOUD"
    failure_text = "\n".join(result.failures)
    assert "plugin_artifact_sha256" in failure_text
    assert "threshold_config" in failure_text
    assert "redaction_ruleset" in failure_text
    assert "schema_version" in failure_text
    assert "encryption_mode" in failure_text


def test_preflight_reads_markdown_report_with_embedded_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "plugin.py"
    artifact.write_bytes(b"same artifact\n")
    manifest = {**BASE_MANIFEST, "plugin_artifact_path": str(artifact)}
    markdown = tmp_path / "aegis-report.md"
    markdown.write_text(
        "# Aegis QA report\n\n```json\n" + json.dumps(manifest, sort_keys=True) + "\n```\n",
        encoding="utf-8",
    )
    apollo = _write_manifest(tmp_path / "apollo.json", artifact)

    result = preflight.run_preflight(aegis_report=markdown, apollo_target=apollo, out_path=None)

    assert result.passed, "\n".join(result.failures)


def test_config_preflight_script_contains_no_live_gateway_restart_command() -> None:
    source = Path(preflight.__file__).read_text(encoding="utf-8")

    forbidden = [
        "launchctl kickstart",
        "systemctl --user restart",
        "hermes gateway restart",
        "gateway restart",
    ]
    assert not any(command in source for command in forbidden)
