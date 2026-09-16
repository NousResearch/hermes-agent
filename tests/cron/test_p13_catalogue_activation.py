from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "p13_catalogue_activation.py"


def _module():
    spec = importlib.util.spec_from_file_location("p13_catalogue_activation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _catalogue(tmp_path: Path) -> tuple[Path, str]:
    payload = {
        "schema_version": 1,
        "status": "PRE_GO_PRIVATE_DISABLED_CATALOGUE",
        "source_denominator": 2,
        "accounting": {"registered": 0, "enabled": 0},
        "entries": [
            {
                "source_instance": "VPS:cron/jobs.json:source-a",
                "source_id": "source-a",
                "source_name": "a",
                "target_status": "ACTIVATION_CANDIDATE",
                "activation_wave": "WAVE_1_BOUNDED_SCRIPTS",
                "target_registration": False,
                "target_enabled": False,
                "blockers": [],
                "target_job": {
                    "id": "source-a",
                    "name": "a",
                    "prompt": "first",
                    "schedule": {"kind": "interval", "minutes": 15, "display": "every 15m"},
                    "repeat": {"times": None, "completed": 0},
                    "deliver": "local",
                    "skills": ["alpha"],
                    "model": "gpt-test-a",
                    "provider": "openai-test",
                    "no_agent": False,
                    "context_from": None,
                },
            },
            {
                "source_instance": "VPS:cron/jobs.json:source-b",
                "source_id": "source-b",
                "source_name": "b",
                "target_status": "ACTIVATION_CANDIDATE",
                "activation_wave": "WAVE_1_BOUNDED_SCRIPTS",
                "target_registration": False,
                "target_enabled": False,
                "blockers": [],
                "target_job": {
                    "id": "source-b",
                    "name": "b",
                    "prompt": "second",
                    "schedule": {"kind": "cron", "expr": "0 8 * * *", "display": "0 8 * * *"},
                    "repeat": {"times": None, "completed": 0},
                    "deliver": "local",
                    "skills": [],
                    "model": "gpt-test-b",
                    "provider": "openai-test",
                    "no_agent": False,
                    "context_from": ["source-a"],
                },
            },
        ],
    }
    path = tmp_path / "catalogue.json"
    text = json.dumps(payload, sort_keys=True) + "\n"
    path.write_text(text)
    path.chmod(0o600)
    return path, hashlib.sha256(text.encode()).hexdigest()


def test_load_catalogue_requires_private_mode_and_matching_checksum(tmp_path: Path) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    loaded = module.load_catalogue(path, digest)
    assert loaded["source_denominator"] == 2

    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        module.load_catalogue(path, digest)


def test_plan_is_pure_and_builds_disabled_create_contract(tmp_path: Path) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    entries = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")
    kwargs = module.build_create_kwargs(entries[0])
    assert kwargs["schedule"] == "every 15m"
    assert kwargs["enabled"] is False
    assert kwargs["context_from"] is None
    assert kwargs["skills"] == ["alpha"]


def test_stage_is_idempotent_and_remaps_context_dependencies(tmp_path: Path) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    entries = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")
    jobs: list[dict] = []
    updates: list[tuple[str, dict]] = []

    def create_job(**kwargs):
        job = {**kwargs, "id": f"target-{len(jobs) + 1}", "state": "paused"}
        jobs.append(job)
        return job

    def list_jobs(include_disabled=True):
        return list(jobs)

    def update_job(job_id, patch):
        updates.append((job_id, patch))
        return {"id": job_id, **patch}

    removed: list[str] = []
    receipt = tmp_path / "receipt.json"
    result = module.stage_wave(
        entries,
        catalogue_sha256=digest,
        disposition_sha256="disposition-test-sha",
        receipt_path=receipt,
        create_job_fn=create_job,
        list_jobs_fn=list_jobs,
        update_job_fn=update_job,
        remove_job_fn=lambda job_id: removed.append(job_id) or True,
    )
    assert result["created"] == 2
    assert updates == [("target-2", {"context_from": ["target-1"]})]
    assert all(job["enabled"] is False for job in jobs)
    assert receipt.stat().st_mode & 0o777 == 0o600

    again = module.stage_wave(
        entries,
        catalogue_sha256=digest,
        disposition_sha256="disposition-test-sha",
        receipt_path=receipt,
        create_job_fn=create_job,
        list_jobs_fn=list_jobs,
        update_job_fn=update_job,
        remove_job_fn=lambda job_id: True,
    )
    assert again["created"] == 0
    assert again["reused"] == 2


def test_mutating_actions_require_exact_go_authority() -> None:
    module = _module()
    for value in (None, "go", "!GO", " !go "):
        with pytest.raises(PermissionError):
            module.require_go_authority(value)
    module.require_go_authority("!go")


def test_stage_does_not_require_go_authority_when_matrix_is_stage_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    entries = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")
    fp_a = module._target_fingerprint(entries[0])
    fp_b = module._target_fingerprint(entries[1])
    matrix = {
        "catalogue_sha256": digest,
        "entries": [
            {"source_instance": "VPS:cron/jobs.json:source-a", "audit_no_stage_state": "STAGE_APPROVED",
             "contract_fingerprint": fp_a},
            {"source_instance": "VPS:cron/jobs.json:source-b", "audit_no_stage_state": "STAGE_APPROVED",
             "contract_fingerprint": fp_b},
        ],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    staged: list[dict] = []
    jobs_module = types.ModuleType("cron.jobs")
    for name, implementation in {
        "create_job": lambda **kwargs: kwargs,
        "list_jobs": lambda **kwargs: [],
        "pause_job": lambda *args, **kwargs: None,
        "remove_job": lambda *args, **kwargs: True,
        "resume_job": lambda *args, **kwargs: None,
        "update_job": lambda *args, **kwargs: None,
    }.items():
        setattr(jobs_module, name, implementation)
    cron_module = types.ModuleType("cron")
    cron_module.__path__ = []
    monkeypatch.setitem(sys.modules, "cron", cron_module)
    monkeypatch.setitem(sys.modules, "cron.jobs", jobs_module)
    monkeypatch.setattr(
        module,
        "stage_wave",
        lambda entries, **kwargs: staged.append({"entries": entries, **kwargs}) or {"created": 2},
    )
    monkeypatch.delenv("KENSEI_MIGRATION_AUTHORITY", raising=False)

    assert module.main([
        "stage", "--catalogue", str(path), "--sha256", digest,
        "--disposition", str(matrix_path), "--wave", "WAVE_1_BOUNDED_SCRIPTS",
        "--receipt", str(tmp_path / "receipt.json"),
    ]) == 0
    assert len(staged) == 1


def test_select_authorised_wave_rejects_entries_not_explicitly_stage_approved(
    tmp_path: Path,
) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    disposition = {
        "catalogue_sha256": digest,
        "entries": [
            {
                "source_instance": "VPS:cron/jobs.json:source-a",
                "audit_no_stage_state": "NO_STAGE_PENDING_ROW_PROOF",
            },
            {
                "source_instance": "VPS:cron/jobs.json:source-b",
                "audit_no_stage_state": "NO_STAGE_PENDING_ROW_PROOF",
            },
        ],
    }

    with pytest.raises(ValueError, match="not stage-approved"):
        module.select_authorised_wave(
            catalogue,
            disposition,
            "WAVE_1_BOUNDED_SCRIPTS",
        )


def test_main_plan_requires_a_checksum_bound_disposition_matrix(tmp_path: Path) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)

    with pytest.raises(SystemExit, match="--disposition is required"):
        module.main(
            [
                "plan",
                "--catalogue",
                str(path),
                "--sha256",
                digest,
                "--wave",
                "WAVE_1_BOUNDED_SCRIPTS",
            ]
        )


def test_select_authorised_wave_requires_complete_matrix_coverage(tmp_path: Path) -> None:
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    incomplete = {
        "catalogue_sha256": digest,
        "entries": [
            {
                "source_instance": "VPS:cron/jobs.json:source-a",
                "audit_no_stage_state": "STAGE_APPROVED",
            },
        ],
    }

    with pytest.raises(ValueError, match="entry set mismatch"):
        module.select_authorised_wave(
            catalogue,
            incomplete,
            "WAVE_1_BOUNDED_SCRIPTS",
        )


# --- Fail-closed receipt-kind validation (P13 repair) ---


def _write_receipt(path: Path, receipt: dict) -> Path:
    path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    return path


def _stage_receipt(
    tmp_path: Path,
    module,
    *,
    wave: str = "WAVE_1_BOUNDED_SCRIPTS",
    override: dict | None = None,
) -> tuple[Path, str]:
    """Build a valid p13_wave_stage_v1 receipt bound to the test catalogue."""
    path, digest = _catalogue(tmp_path)
    receipt = {
        "schema_version": 1,
        "receipt_kind": module.RECEIPT_KIND_STAGE,
        "migration": module.MIGRATION_MARKER,
        "catalogue_sha256": digest,
        "disposition_sha256": "disposition-test-sha",
        "wave": wave,
        "jobs": [
            {"source_instance": "VPS:cron/jobs.json:source-a", "source_id": "source-a", "target_id": "target-1"},
            {"source_instance": "VPS:cron/jobs.json:source-b", "source_id": "source-b", "target_id": "target-2"},
        ],
    }
    if override:
        receipt.update(override)
    receipt_path = tmp_path / "stage_receipt.json"
    return _write_receipt(receipt_path, receipt), digest


def test_read_receipt_accepts_valid_stage_receipt(tmp_path: Path) -> None:
    module = _module()
    path, digest = _stage_receipt(tmp_path, module)
    receipt = module._read_receipt(
        path, digest, expected_kind=module.RECEIPT_KIND_STAGE, expected_wave="WAVE_1_BOUNDED_SCRIPTS"
    )
    assert receipt["receipt_kind"] == module.RECEIPT_KIND_STAGE
    assert len(receipt["jobs"]) == 2


def test_read_receipt_rejects_registration_receipt(tmp_path: Path) -> None:
    """The 103-job incident: an all-registration receipt must NOT enable jobs."""
    module = _module()
    path, digest = _catalogue(tmp_path)
    registration = {
        "schema_version": 1,
        "receipt_kind": module.RECEIPT_KIND_REGISTRATION,
        "migration": module.MIGRATION_MARKER,
        "catalogue_sha256": digest,
        "summary": {"total": 103, "created": 103, "reused": 0, "skipped": 0},
        "jobs": [
            {"source_instance": f"VPS:cron/jobs.json:s{i}", "target_id": f"job-{i}"}
            for i in range(103)
        ],
        "errors": [],
    }
    receipt_path = _write_receipt(tmp_path / "reg_receipt.json", registration)
    with pytest.raises(ValueError, match="not a valid p13_wave_stage_v1 receipt"):
        module._read_receipt(
            receipt_path, digest,
            expected_kind=module.RECEIPT_KIND_STAGE,
            expected_wave="WAVE_1_BOUNDED_SCRIPTS",
        )


def test_read_receipt_rejects_legacy_receipt_without_kind(tmp_path: Path) -> None:
    """Pre-repair receipts lack receipt_kind — they must not be trusted."""
    module = _module()
    path, digest = _stage_receipt(tmp_path, module, override={"receipt_kind": None})
    with pytest.raises(ValueError, match="not a valid p13_wave_stage_v1 receipt"):
        module._read_receipt(
            path, digest,
            expected_kind=module.RECEIPT_KIND_STAGE,
            expected_wave="WAVE_1_BOUNDED_SCRIPTS",
        )


def test_read_receipt_rejects_malformed_empty_jobs(tmp_path: Path) -> None:
    module = _module()
    path, digest = _stage_receipt(tmp_path, module, override={"jobs": []})
    with pytest.raises(ValueError, match="no staged jobs"):
        module._read_receipt(path, digest, expected_kind=module.RECEIPT_KIND_STAGE)


def test_read_receipt_rejects_job_missing_target_id(tmp_path: Path) -> None:
    module = _module()
    path, digest = _stage_receipt(
        tmp_path, module,
        override={"jobs": [{"source_instance": "x", "source_id": "y"}]},
    )
    with pytest.raises(ValueError, match="missing target_id"):
        module._read_receipt(path, digest, expected_kind=module.RECEIPT_KIND_STAGE)


def test_read_receipt_rejects_job_missing_source_binding(tmp_path: Path) -> None:
    """A staged target must retain its source-instance and source-ID binding."""
    module = _module()
    path, digest = _stage_receipt(
        tmp_path, module,
        override={"jobs": [{"target_id": "target-1"}]},
    )
    with pytest.raises(ValueError, match="missing source_instance"):
        module._read_receipt(path, digest, expected_kind=module.RECEIPT_KIND_STAGE)


def test_read_receipt_rejects_missing_wave_binding(tmp_path: Path) -> None:
    module = _module()
    path, digest = _stage_receipt(tmp_path, module, override={"wave": None})
    with pytest.raises(ValueError, match="missing wave binding"):
        module._read_receipt(path, digest, expected_kind=module.RECEIPT_KIND_STAGE)


def test_read_receipt_rejects_out_of_wave_receipt(tmp_path: Path) -> None:
    module = _module()
    path, digest = _stage_receipt(tmp_path, module, wave="WAVE_2_OTHER")
    with pytest.raises(ValueError, match="does not match expected wave"):
        module._read_receipt(
            path, digest,
            expected_kind=module.RECEIPT_KIND_STAGE,
            expected_wave="WAVE_1_BOUNDED_SCRIPTS",
        )


def test_stage_receipt_written_with_strict_kind_and_disposition_binding(tmp_path: Path) -> None:
    """stage_wave must emit a p13_wave_stage_v1 receipt carrying disposition_sha256."""
    module = _module()
    path, digest = _catalogue(tmp_path)
    catalogue = module.load_catalogue(path, digest)
    entries = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")
    jobs: list[dict] = []
    receipt = tmp_path / "receipt.json"

    def create_job(**kwargs):
        job = {**kwargs, "id": f"target-{len(jobs) + 1}", "state": "paused"}
        jobs.append(job)
        return job

    module.stage_wave(
        entries,
        catalogue_sha256=digest,
        disposition_sha256="disp-abc123",
        receipt_path=receipt,
        create_job_fn=create_job,
        list_jobs_fn=lambda **kw: list(jobs),
        update_job_fn=lambda jid, patch: {"id": jid, **patch},
        remove_job_fn=lambda jid: True,
    )
    written = json.loads(receipt.read_text())
    assert written["receipt_kind"] == module.RECEIPT_KIND_STAGE
    assert written["disposition_sha256"] == "disp-abc123"
    assert written["wave"] == "WAVE_1_BOUNDED_SCRIPTS"
    assert len(written["jobs"]) == 2


def test_enable_receipt_only_resumes_staged_ids_and_rolls_back_on_failure() -> None:
    module = _module()
    receipt = {
        "receipt_kind": module.RECEIPT_KIND_STAGE,
        "migration": module.MIGRATION_MARKER,
        "catalogue_sha256": "x",
        "wave": "W1",
        "jobs": [
            {"target_id": "a"},
            {"target_id": "b"},
        ],
    }
    resumed: list[str] = []
    paused: list[str] = []

    def resume_job(target_id):
        resumed.append(target_id)
        if target_id == "b":
            return None  # simulate failure
        return {"id": target_id, "enabled": True}

    def pause_job(target_id, reason=None):
        paused.append(target_id)
        return {"id": target_id, "enabled": False}

    with pytest.raises(RuntimeError, match="failed to enable b"):
        module.enable_receipt(receipt, resume_job_fn=resume_job, pause_job_fn=pause_job)
    assert resumed == ["a", "b"]
    assert paused == ["a"]  # rollback of the one that succeeded
