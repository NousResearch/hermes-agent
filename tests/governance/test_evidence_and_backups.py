import json
from pathlib import Path

from governance.backup import create_verified_backup, has_recent_verified_backup
from governance.evidence import EvidenceLedger, EvidenceRecord, append_hash_chained_event


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_hash_chained_events_are_tamper_evident(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first = append_hash_chained_event("test_events", {"event_type": "first", "decision": "allow"})
    second = append_hash_chained_event("test_events", {"event_type": "second", "decision": "deny"})
    assert first["previous_entry_hash"] == "GENESIS"
    assert second["previous_entry_hash"] == first["entry_hash"]
    assert first["entry_hash"] != second["entry_hash"]

    rows = _read_jsonl(tmp_path / "governance" / "test_events.jsonl")
    assert rows[0]["entry_hash"] == first["entry_hash"]
    assert rows[1]["previous_entry_hash"] == rows[0]["entry_hash"]


def test_evidence_ledger_records_canonical_object(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ledger = EvidenceLedger()
    record = EvidenceRecord(
        phase="test",
        claim="policy gate blocks destructive commands",
        evidence_type="test_result",
        source="pytest",
        confidence="verified",
    )
    written = ledger.record(record)
    assert written["claim"] == "policy gate blocks destructive commands"
    assert written["redaction_status"] == "none"
    assert (tmp_path / "governance" / "evidence.jsonl").exists()


def test_create_verified_backup_preserves_bytes_and_records_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    source = tmp_path / "source.txt"
    source.write_text("important bytes\n", encoding="utf-8")
    backup_root = tmp_path / "backups"

    manifest = create_verified_backup(source, backup_root=backup_root, operation="unit test")

    backup_path = Path(manifest["backup_path"])
    assert backup_path.exists()
    assert backup_path.read_bytes() == source.read_bytes()
    assert manifest["verified_exact_match"] is True
    assert manifest["original_sha256"] == manifest["backup_sha256"]
    assert manifest["path_type"] == "regular_file"
    assert manifest["rollback_command_live_requires_approval"]
    assert manifest["rollback_argv_live_requires_approval"] == ["cp", "-a", str(backup_path), str(source)]


def test_has_recent_verified_backup_rejects_missing_or_tampered_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    source = tmp_path / "source.txt"
    source.write_text("important bytes\n", encoding="utf-8")

    manifest = create_verified_backup(source, backup_root=tmp_path / "backup-one", operation="unit test")
    assert has_recent_verified_backup(source) is True

    Path(manifest["backup_path"]).unlink()
    assert has_recent_verified_backup(source) is False

    manifest = create_verified_backup(source, backup_root=tmp_path / "backup-two", operation="unit test")
    Path(manifest["backup_path"]).write_text("tampered backup\n", encoding="utf-8")
    assert has_recent_verified_backup(source) is False


def test_has_recent_verified_backup_rejects_stale_manifest(tmp_path, monkeypatch):
    hermes_home = tmp_path / "hermes_home"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    source = tmp_path / "source.txt"
    source.write_text("important bytes\n", encoding="utf-8")

    create_verified_backup(source, backup_root=tmp_path / "backup", operation="unit test")
    log_path = hermes_home / "governance" / "backup_manifests.jsonl"
    rows = _read_jsonl(log_path)
    rows[-1]["created_at_utc"] = "1970-01-01T00:00:00Z"
    rows[-1]["timestamp_utc"] = "1970-01-01T00:00:00Z"
    log_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")

    assert has_recent_verified_backup(source) is False
