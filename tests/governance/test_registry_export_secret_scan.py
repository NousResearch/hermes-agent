from pathlib import Path

from governance.export_safety import ExportMode, SecretScanner
from governance.source_registry import RegistryEntry, SourceOfTruthRegistry


def test_secret_scanner_finds_and_redacts_common_secret_shapes():
    scanner = SecretScanner()
    sample_api_label = "OPENAI_" + "API_" + "KEY"
    sample_pw_label = "pass" + "word"
    sample_a = "sk-" + "tes...cdef"
    sample_b = "hunter" + "2hunter2"
    text = f"{sample_api_label}={sample_a}\n{sample_pw_label} = \"{sample_b}\"\nnormal text"
    result = scanner.scan_text(text, mode="report")
    assert result["status"] == "redaction_required"
    assert result["findings_count"] >= 2
    redacted = scanner.redact_text(text)
    assert "sk-test" not in redacted
    assert "hunter2" not in redacted
    assert "[REDACTED:" in redacted
    report_dump = str(result["findings"])
    assert sample_a not in report_dump
    assert sample_b not in report_dump
    assert all(finding["preview"].startswith("[REDACTED:") for finding in result["findings"])


def test_secret_scanner_does_not_flag_normal_schema_identifiers():
    scanner = SecretScanner()
    text = '{"schema_version": "governance.evidence.v1", "policy_decision": "reject_policy"}'
    result = scanner.scan_text(text, mode="report")
    assert result["status"] == "clean"
    assert result["findings_count"] == 0


def test_runtime_clean_export_excludes_known_unsafe_paths():
    assert ExportMode.RUNTIME_CLEAN.allows_path("src/app.py") is True
    assert ExportMode.RUNTIME_CLEAN.allows_path(".git/config") is False
    assert ExportMode.RUNTIME_CLEAN.allows_path("node_modules/pkg/index.js") is False
    assert ExportMode.RUNTIME_CLEAN.allows_path("logs/gateway.log") is False
    assert ExportMode.RUNTIME_CLEAN.allows_path(r".git\\config") is False
    assert ExportMode.SHAREABLE_AUDIT_FULL.requires_redaction is True


def test_source_registry_requires_evidence_for_confirmed_active(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    registry = SourceOfTruthRegistry()
    entry = RegistryEntry(
        component_type="startup_file",
        component_name="Omega start",
        active_status="confirmed_active",
        confidence="verified_runtime",
        path_or_ref="/tmp/start.md",
        owner_profile="omega",
        evidence_refs=[],
    )
    result = registry.upsert(entry)
    assert result["success"] is False
    assert "evidence" in result["error"].lower()

    entry.evidence_refs = ["evidence:test"]
    result = registry.upsert(entry)
    assert result["success"] is True
    rows = registry.list_entries()
    assert rows[0]["active_status"] == "confirmed_active"


def test_governance_export_safety_is_packaged_and_not_gitignored():
    root = Path(__file__).resolve().parents[2]
    assert (root / "governance" / "export_safety.py").exists()
    assert '"governance", "governance.*"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "!governance/export_safety.py" in (root / ".gitignore").read_text(encoding="utf-8")
