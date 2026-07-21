from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest


def _load_schemas_module():
    repo_root = Path(__file__).resolve().parents[3]
    plugin_dir = repo_root / "plugins" / "truth-ledger"
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.truth_ledger.schemas",
        plugin_dir / "schemas.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None
    assert spec.loader is not None
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    if "hermes_plugins.truth_ledger" not in sys.modules:
        pkg = types.ModuleType("hermes_plugins.truth_ledger")
        pkg.__path__ = [str(plugin_dir)]
        sys.modules["hermes_plugins.truth_ledger"] = pkg
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hermes_plugins.truth_ledger.schemas"] = mod
    spec.loader.exec_module(mod)
    return mod


def _source_envelope() -> dict[str, object]:
    return {
        "schema_name": "truth-ledger.source-envelope.v1",
        "schema_version": 1,
        "captured_at": "2026-07-19T00:00:00Z",
        "profile": "automation-operator",
        "session_id": "sess-1",
        "turn_id": "turn-1",
        "origin": {
            "platform": "cli",
            "conversation_id": "conv-1",
            "thread_id": "thread-1",
            "speaker_id": "user-1",
        },
        "input": {"user_message": "Keep responses concise."},
        "output": {"assistant_response": "Understood."},
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_pending_record(spool, *, payload_path: Path, **overrides) -> Path:
    record = {
        "schema_name": "truth-ledger.spool-record.v1",
        "schema_version": 1,
        "envelope_id": f"env_{payload_path.stem or 'injected'}",
        "state": "pending",
        "captured_at": "2026-07-19T00:00:00Z",
        "attempt_count": 0,
        "idempotency_key": f"automation-operator:sess-1:{payload_path.stem or 'turn'}",
        "source_ref": {"profile": "automation-operator", "session_id": "sess-1", "turn_id": payload_path.stem or "turn"},
        "payload_path": str(payload_path),
        "flow": {},
    }
    record.update(overrides)
    path = spool.pending_dir / f"inject-{time.time_ns()}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_spool_record_representation_is_schema_coherent_across_lifecycle(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    schemas_mod = _load_schemas_module()

    result = spool.enqueue(_source_envelope())

    assert result["ok"] is True
    pending_record_path = Path(result["path"])
    pending_record = _load_json(pending_record_path)
    schemas_mod.validate_document("spool-record.v1", pending_record)
    source_payload_path = Path(str(pending_record["payload_path"]))
    schemas_mod.validate_document("source-envelope.v1", _load_json(source_payload_path))

    claim = spool.claim_next(owner="worker-1")
    assert claim is not None
    processing_path = Path(claim["path"])
    assert processing_path.parent.name == "processing"
    processing_record = _load_json(processing_path)
    schemas_mod.validate_document("spool-record.v1", processing_record)
    assert processing_record["state"] == "processing"

    retry = spool.retry_processing(processing_path, error_code="SQLITE_BUSY")
    assert retry["ok"] is True
    retried_pending_path = Path(retry["path"])
    assert retried_pending_path.parent.name == "pending"
    retried_pending_record = _load_json(retried_pending_path)
    schemas_mod.validate_document("spool-record.v1", retried_pending_record)
    assert retried_pending_record["attempt_count"] == 1

    claim2 = spool.claim_next(owner="worker-2")
    assert claim2 is not None
    processing_path2 = Path(claim2["path"])

    dead = spool.dead_letter(processing_path2, reason="permanent")
    assert dead["ok"] is True
    dead_path = Path(dead["path"])
    assert dead_path.parent.name == "dead-letter"
    dead_record = _load_json(dead_path)
    schemas_mod.validate_document("spool-record.v1", dead_record)
    assert dead_record["state"] == "dead_lettered"
    assert dead_record["attempt_count"] == 1


def test_soft_and_hard_caps_are_enforced_without_throwing(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path, soft_count=2, hard_count=3)

    envelope = _source_envelope()
    r1 = spool.enqueue({**envelope, "turn_id": "turn-a"})
    r2 = spool.enqueue({**envelope, "turn_id": "turn-b"})
    r3 = spool.enqueue({**envelope, "turn_id": "turn-c"})
    assert r1["ok"] and r2["ok"] and r3["ok"]

    # soft cap sheds oldest pending to dead-letter
    dead_files = list((tmp_path / "spool" / "dead-letter").glob("*.json"))
    assert len(dead_files) >= 1

    # hard cap should fail-open: no exception, explicit rejection.
    strict = spool_mod.TruthSpool(tmp_path / "strict", soft_count=99, hard_count=3)
    assert strict.enqueue({**envelope, "turn_id": "strict-a"})["ok"]
    assert strict.enqueue({**envelope, "turn_id": "strict-b"})["ok"]
    assert strict.enqueue({**envelope, "turn_id": "strict-c"})["ok"]
    r4 = strict.enqueue({**envelope, "turn_id": "strict-d"})
    assert r4["ok"] is False
    assert r4["reason"] == "queue_hard_cap"


def test_recover_stale_processing_moves_back_to_pending(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    spool.enqueue({**_source_envelope(), "turn_id": "stale"})
    claim = spool.claim_next(owner="worker")
    assert claim is not None

    processing_path = Path(claim["path"])
    old = time.time() - 600
    os.utime(processing_path, (old, old))

    moved = spool.recover_stale_processing(stale_seconds=60)
    assert moved == 1
    assert not processing_path.exists()
    assert len(list((tmp_path / "spool" / "pending").glob("*.json"))) == 1


def test_ack_processing_removes_record_and_only_owned_payloads(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)

    enqueue = spool.enqueue({**_source_envelope(), "turn_id": "owned-payload"})
    assert enqueue["ok"] is True
    claim = spool.claim_next(owner="worker")
    assert claim is not None
    processing_path = Path(claim["path"])
    owned_payload_path = Path(str(claim["record"]["payload_path"]))
    assert owned_payload_path.exists()

    result = spool.ack_processing(processing_path)
    assert result["ok"] is True
    assert processing_path.exists() is False
    assert owned_payload_path.exists() is False

    foreign_payload = tmp_path / "foreign" / "payload.json"
    foreign_payload.parent.mkdir(parents=True, exist_ok=True)
    foreign_payload.write_text(json.dumps(_source_envelope()), encoding="utf-8")
    record_path = spool.processing_dir / "foreign-processing.json"
    record_path.write_text(
        json.dumps(
            {
                "schema_name": "truth-ledger.spool-record.v1",
                "schema_version": 1,
                "envelope_id": "env_foreign",
                "state": "processing",
                "captured_at": "2026-07-19T00:00:00Z",
                "attempt_count": 0,
                "idempotency_key": "automation-operator:sess-1:turn-foreign",
                "source_ref": {"profile": "automation-operator", "session_id": "sess-1", "turn_id": "turn-foreign"},
                "payload_path": str(foreign_payload),
                "flow": {},
            }
        ),
        encoding="utf-8",
    )

    result = spool.ack_processing(record_path)
    assert result["ok"] is True
    assert record_path.exists() is False
    assert foreign_payload.exists() is True


def test_ack_crash_after_processing_removal_does_not_retain_completed_payload(
    tmp_path, spool_mod, monkeypatch
):
    spool = spool_mod.TruthSpool(tmp_path)
    assert spool.enqueue({**_source_envelope(), "turn_id": "ack-crash"})["ok"] is True
    claim = spool.claim_next(owner="worker")
    assert claim is not None

    processing_path = Path(claim["path"])
    payload_path = Path(str(claim["record"]["payload_path"]))

    def _simulate_process_death(_record):
        raise SystemExit("synthetic ack process death")

    monkeypatch.setattr(spool, "_unlink_payload_if_owned", _simulate_process_death)
    with pytest.raises(SystemExit, match="synthetic ack process death"):
        spool.ack_processing(processing_path)

    assert processing_path.exists() is False
    assert payload_path.exists() is True
    assert len(list(spool.completed_dir.glob("*.json"))) == 1

    restarted = spool_mod.TruthSpool(tmp_path)
    assert restarted.recover_orphan_payloads() == 0
    assert payload_path.exists() is False
    assert list(restarted.pending_dir.glob("*.json")) == []


def test_dead_letter_and_soft_overflow_remove_payload_files(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path, soft_count=1, hard_count=5)

    first = spool.enqueue({**_source_envelope(), "turn_id": "overflow-1"})
    assert first["ok"] is True
    first_record = _load_json(Path(first["path"]))
    first_payload_path = Path(str(first_record["payload_path"]))
    assert first_payload_path.exists()

    second = spool.enqueue({**_source_envelope(), "turn_id": "overflow-2"})
    assert second["ok"] is True

    dead_files = sorted((tmp_path / "spool" / "dead-letter").glob("*.json"))
    assert dead_files
    overflow_record = _load_json(dead_files[0])
    assert overflow_record["state"] == "dead_lettered"
    assert overflow_record["flow"]["dead_letter_reason"] == "queue_overflow"
    assert overflow_record["source_ref"]["session_id"] == "sess-1"
    assert "input" not in json.dumps(overflow_record)
    assert first_payload_path.exists() is False

    claim = spool.claim_next(owner="worker")
    assert claim is not None
    processing_path = Path(claim["path"])
    payload_path = Path(str(claim["record"]["payload_path"]))
    assert payload_path.exists() is True

    dead = spool.dead_letter(processing_path, reason="permanent")
    assert dead["ok"] is True
    dead_record = _load_json(Path(dead["path"]))
    assert dead_record["state"] == "dead_lettered"
    assert dead_record["flow"]["dead_letter_reason"] == "permanent"
    assert dead_record["source_ref"]["turn_id"] == "overflow-2"
    assert "input" not in json.dumps(dead_record)
    assert payload_path.exists() is False


def test_retry_and_recovery_retain_payload_while_work_is_pending_or_processing(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    enqueue = spool.enqueue({**_source_envelope(), "turn_id": "retry-and-recover"})
    assert enqueue["ok"] is True

    pending_record = _load_json(Path(enqueue["path"]))
    payload_path = Path(str(pending_record["payload_path"]))
    assert payload_path.exists() is True

    claim = spool.claim_next(owner="worker")
    assert claim is not None
    processing_path = Path(claim["path"])
    assert payload_path.exists() is True

    retry = spool.retry_processing(processing_path, error_code="TEMP")
    assert retry["ok"] is True
    retry_record = _load_json(Path(retry["path"]))
    assert retry_record["state"] == "pending"
    assert Path(str(retry_record["payload_path"])) == payload_path
    assert payload_path.exists() is True

    claim_again = spool.claim_next(owner="worker-2")
    assert claim_again is not None
    processing_again_path = Path(claim_again["path"])
    stale = time.time() - 600
    os.utime(processing_again_path, (stale, stale))

    moved = spool.recover_stale_processing(stale_seconds=60)
    assert moved == 1
    recovered_pending = sorted(spool.pending_dir.glob("*.json"))
    assert len(recovered_pending) == 1
    recovered_record = _load_json(recovered_pending[0])
    assert recovered_record["state"] == "pending"
    assert Path(str(recovered_record["payload_path"])) == payload_path
    assert payload_path.exists() is True


def test_recover_stale_processing_tolerates_record_disappearing_mid_recovery(tmp_path, spool_mod, monkeypatch):
    spool = spool_mod.TruthSpool(tmp_path)
    spool.enqueue({**_source_envelope(), "turn_id": "race-disappear"})
    claim = spool.claim_next(owner="worker")
    assert claim is not None
    processing_path = Path(claim["path"])
    stale = time.time() - 600
    os.utime(processing_path, (stale, stale))

    original_load_record = spool._load_record

    def _load_record_with_disappearing_file(path: Path):
        if path == processing_path and path.exists():
            path.unlink()
            raise FileNotFoundError("processing record removed by concurrent close")
        return original_load_record(path)

    monkeypatch.setattr(spool, "_load_record", _load_record_with_disappearing_file)

    moved = spool.recover_stale_processing(stale_seconds=60)
    assert moved == 0
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.dead_letter_dir.glob("*.json")) == []


def test_claim_next_quarantines_malformed_and_schema_invalid_spool_records(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)

    malformed_path = spool.pending_dir / "inject-malformed.json"
    malformed_path.write_text("{not-json", encoding="utf-8")

    owned_payload = spool.payloads_dir / "owned.json"
    owned_payload.write_text(json.dumps(_source_envelope()), encoding="utf-8")
    _write_pending_record(spool, payload_path=owned_payload, unexpected="field")

    claim = spool.claim_next(owner="worker")
    assert claim is None
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.processing_dir.glob("*.json")) == []

    dead_records = [_load_json(p) for p in sorted(spool.dead_letter_dir.glob("*.json"))]
    assert len(dead_records) == 2
    assert {r["flow"]["dead_letter_reason"] for r in dead_records} == {"invalid_spool_record"}
    assert owned_payload.exists() is False


def test_claim_next_quarantines_missing_or_invalid_source_payloads(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)

    invalid_payload = spool.payloads_dir / "invalid-envelope.json"
    invalid_payload.write_text(json.dumps({"schema_name": "truth-ledger.source-envelope.v1"}), encoding="utf-8")
    _write_pending_record(spool, payload_path=invalid_payload)

    missing_payload = spool.payloads_dir / "missing-envelope.json"
    _write_pending_record(spool, payload_path=missing_payload)

    claim = spool.claim_next(owner="worker")
    assert claim is None
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.processing_dir.glob("*.json")) == []

    dead_records = [_load_json(p) for p in sorted(spool.dead_letter_dir.glob("*.json"))]
    assert len(dead_records) == 2
    assert {r["flow"]["dead_letter_reason"] for r in dead_records} == {
        "invalid_source_envelope",
        "missing_payload",
    }
    assert invalid_payload.exists() is False


def test_claim_next_rejects_payload_paths_outside_owned_root_including_symlinks(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)

    external = tmp_path / "external" / "secret.json"
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_text('{"secret":"top-secret"}', encoding="utf-8")

    _write_pending_record(spool, payload_path=external)

    symlink_path = spool.payloads_dir / "link-secret.json"
    symlink_path.symlink_to(external)
    _write_pending_record(spool, payload_path=symlink_path)

    claim = spool.claim_next(owner="worker")
    assert claim is None
    assert list(spool.pending_dir.glob("*.json")) == []
    assert list(spool.processing_dir.glob("*.json")) == []

    dead_records = [_load_json(p) for p in sorted(spool.dead_letter_dir.glob("*.json"))]
    assert len(dead_records) == 2
    assert {r["flow"]["dead_letter_reason"] for r in dead_records} == {"payload_path_out_of_root"}
    assert external.exists() is True
    assert external.read_text(encoding="utf-8") == '{"secret":"top-secret"}'
    assert symlink_path.exists() is True


def test_enqueue_rolls_back_payload_when_pending_record_write_fails(tmp_path, spool_mod, monkeypatch):
    spool = spool_mod.TruthSpool(tmp_path)
    original_write = spool_mod._write_private_json_atomic

    def _fail_pending(path, payload):
        if path.parent == spool.pending_dir:
            raise OSError("synthetic pending write failure")
        return original_write(path, payload)

    monkeypatch.setattr(spool_mod, "_write_private_json_atomic", _fail_pending)

    with pytest.raises(OSError, match="synthetic pending write failure"):
        spool.enqueue(_source_envelope())

    assert list(spool.payloads_dir.glob("*.json")) == []
    assert list(spool.pending_dir.glob("*.json")) == []


def test_enqueue_is_durably_idempotent_across_spool_instances(tmp_path, spool_mod):
    first_spool = spool_mod.TruthSpool(tmp_path)
    second_spool = spool_mod.TruthSpool(tmp_path)

    first = first_spool.enqueue(_source_envelope())
    second = second_spool.enqueue(_source_envelope())

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["reason"] == "duplicate"
    assert len(list(first_spool.pending_dir.glob("*.json"))) == 1
    assert len(list(first_spool.payloads_dir.glob("*.json"))) == 1


def test_ack_writes_durable_tombstone_that_suppresses_restart_duplicate(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    assert spool.enqueue(_source_envelope())["ok"] is True
    claim = spool.claim_next(owner="worker")
    assert claim is not None
    assert spool.ack_processing(Path(claim["path"]))["ok"] is True

    restarted = spool_mod.TruthSpool(tmp_path)
    duplicate = restarted.enqueue(_source_envelope())

    assert duplicate["ok"] is True
    assert duplicate["reason"] == "duplicate"
    assert list(restarted.pending_dir.glob("*.json")) == []
    completed = list(restarted.completed_dir.glob("*.json"))
    assert len(completed) == 1
    assert _load_json(completed[0])["state"] == "acked"


def test_recover_orphan_payload_reconstructs_pending_record_after_process_death(tmp_path, spool_mod, monkeypatch):
    spool = spool_mod.TruthSpool(tmp_path)
    original_write = spool_mod._write_private_json_atomic

    def _simulate_process_death(path, payload):
        if path.parent == spool.pending_dir:
            raise SystemExit("synthetic process death")
        return original_write(path, payload)

    monkeypatch.setattr(spool_mod, "_write_private_json_atomic", _simulate_process_death)
    with pytest.raises(SystemExit, match="synthetic process death"):
        spool.enqueue(_source_envelope())

    assert len(list(spool.payloads_dir.glob("*.json"))) == 1
    assert list(spool.pending_dir.glob("*.json")) == []

    monkeypatch.setattr(spool_mod, "_write_private_json_atomic", original_write)
    restarted = spool_mod.TruthSpool(tmp_path)
    assert restarted.recover_orphan_payloads() == 1
    assert len(list(restarted.pending_dir.glob("*.json"))) == 1


def test_claim_next_skips_retry_until_next_retry_at(tmp_path, spool_mod):
    spool = spool_mod.TruthSpool(tmp_path)
    assert spool.enqueue(_source_envelope())["ok"] is True
    claim = spool.claim_next(owner="worker")
    assert claim is not None

    retry = spool.retry_processing(Path(claim["path"]), error_code="TEMP", delay_ms=60_000)

    assert retry["ok"] is True
    assert spool.claim_next(owner="too-early") is None
