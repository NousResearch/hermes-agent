"""E2E provenance-hardening tests for P13 catalogue activation.

These tests exercise the real cron.jobs store via use_cron_store(tmp_path) —
no mocks for the cron layer, no real HERMES_HOME, no network, no system state.

They cover the five provenance-hardening behaviours required by the P13
repair:
  1. Reused job validated against the expected persisted/normalized runtime
     contract before receipt creation (fail-closed on drift).
  2. Mutable origin must not let a changed job pass (origin rebind / payload
     drift rejected even though cron.jobs.update_job permits origin updates).
  3. Dict-model silent loss eliminated: legacy {model, provider} dicts
     normalized to two strings or fail closed; unpinned entries require an
     explicit P13 policy mapping, never fall back to the root global default.
  4. Stage-approved disposition row bound to source_instance and its expected
     current-contract fingerprint (not only whole catalogue hash).
  5. Receipt records enough per-job hashes to validate exact selected rows.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmarks" / "p13_catalogue_activation.py"


def _module():
    spec = importlib.util.spec_from_file_location("p13_catalogue_activation", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(
    source_instance: str = "VPS:cron/jobs.json:source-a",
    source_id: str = "source-a",
    wave: str = "WAVE_1_BOUNDED_SCRIPTS",
    *,
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a catalogue entry.  Default is a no_agent script job (no policy
    needed).  Override via ``job=`` for agent jobs or dict-model tests."""
    base_job: dict[str, Any] = {
        "id": source_id,
        "name": source_id,
        "prompt": "do work",
        "schedule": {"kind": "interval", "minutes": 15, "display": "every 15m"},
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
        "skills": [],
        "no_agent": True,
        "script": "healthcheck.sh",
        "context_from": None,
    }
    if job:
        base_job.update(job)
    return {
        "source_instance": source_instance,
        "source_id": source_id,
        "source_name": source_id,
        "target_status": "ACTIVATION_CANDIDATE",
        "activation_wave": wave,
        "target_registration": False,
        "target_enabled": False,
        "blockers": [],
        "target_job": base_job,
    }


def _catalogue_json(entries: list[dict[str, Any]]) -> tuple[str, str]:
    payload = {
        "schema_version": 1,
        "status": "PRE_GO_PRIVATE_DISABLED_CATALOGUE",
        "source_denominator": len(entries),
        "accounting": {"registered": 0, "enabled": 0},
        "entries": entries,
    }
    text = json.dumps(payload, sort_keys=True) + "\n"
    return text, hashlib.sha256(text.encode()).hexdigest()


def _write_catalogue(tmp_path: Path, entries: list[dict[str, Any]]) -> tuple[Path, str]:
    text, digest = _catalogue_json(entries)
    path = tmp_path / "catalogue.json"
    path.write_text(text)
    path.chmod(0o600)
    return path, digest


def _disposition(
    digest: str,
    entries: list[dict[str, Any]],
    *,
    row_fingerprints: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a disposition matrix with optional per-row contract fingerprints."""
    disp_entries = []
    for e in entries:
        row: dict[str, Any] = {
            "source_instance": e["source_instance"],
            "audit_no_stage_state": "STAGE_APPROVED",
        }
        if row_fingerprints and e["source_instance"] in row_fingerprints:
            row["contract_fingerprint"] = row_fingerprints[e["source_instance"]]
        disp_entries.append(row)
    return {"catalogue_sha256": digest, "entries": disp_entries}


def _write_disposition(tmp_path: Path, disp: dict[str, Any]) -> tuple[Path, str]:
    text = json.dumps(disp, sort_keys=True) + "\n"
    path = tmp_path / "disposition.json"
    path.write_text(text)
    return path, hashlib.sha256(text.encode()).hexdigest()


def _stage_once(
    module,
    tmp_path: Path,
    entry: dict[str, Any],
    *,
    p13_policy: dict[str, Any] | None = None,
) -> tuple[str, str, Path, str]:
    """Write catalogue+disposition, stage once, return (digest, disp_sha, receipt_path, fingerprint)."""
    import cron.jobs as jobs

    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")
    fp = module._target_fingerprint(entry, p13_policy=p13_policy)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})
    disp_path, disp_sha = _write_disposition(tmp_path, disp)
    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        module.stage_wave(
            selected,
            catalogue_sha256=digest,
            disposition_sha256=disp_sha,
            receipt_path=receipt_path,
            create_job_fn=jobs.create_job,
            list_jobs_fn=jobs.list_jobs,
            update_job_fn=jobs.update_job,
            remove_job_fn=jobs.remove_job,
            p13_policy=p13_policy,
        )
    return digest, disp_sha, receipt_path, fp


# ---------------------------------------------------------------------------
# 1. Successful disabled stage (E2E against real cron.jobs)
# ---------------------------------------------------------------------------

def test_e2e_successful_disabled_stage(tmp_path: Path) -> None:
    """Stage a single-entry wave against the real cron store; job must be
    persisted disabled/paused with a valid stage receipt carrying per-job
    target_fingerprint."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry()
    digest, disp_sha, receipt_path, fp = _stage_once(module, tmp_path, entry)

    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
    assert len(persisted) == 1
    job = persisted[0]
    assert job["enabled"] is False
    assert job["state"] == "paused"
    assert job["origin"]["migration"] == module.MIGRATION_MARKER
    assert job["origin"]["source_instance"] == entry["source_instance"]
    assert job["origin"]["target_fingerprint"] == fp

    written = json.loads(receipt_path.read_text())
    assert written["receipt_kind"] == module.RECEIPT_KIND_STAGE
    assert written["jobs"][0]["target_fingerprint"] == fp


# ---------------------------------------------------------------------------
# 2. Payload drift rejected on reuse
# ---------------------------------------------------------------------------

def test_e2e_payload_drift_rejected_on_reuse(tmp_path: Path) -> None:
    """A reused job whose persisted runtime fields have drifted (via
    update_job) must be rejected — the activation utility re-derives the
    contract from the *persisted* job and compares to the expected current
    contract."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry()
    digest, disp_sha, receipt_path, fp = _stage_once(module, tmp_path, entry)

    # Simulate payload drift: change the prompt via update_job.
    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
        job_id = persisted[0]["id"]
        jobs.update_job(job_id, {"prompt": "tampered prompt"})

    cat_path = tmp_path / "catalogue.json"
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    # Re-stage must reject the drifted job.
    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="persisted contract drift"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


# ---------------------------------------------------------------------------
# 3. Origin rebind rejected (genuine RED→GREEN: receipt-authoritative reuse)
# ---------------------------------------------------------------------------

def test_e2e_origin_rebind_source_instance_changed_rejected(tmp_path: Path) -> None:
    """Re-stage must reject a rebinding of an already staged source, not
    quietly create a fresh replacement.

    Vector: the prior stage receipt records source_instance→target_id.  An
    attacker rebinds the persisted job''s origin.source_instance to a
    different value (update_job permits origin updates).  On re-stage, the
    receipt still names the original target_id for this source_instance.
    That target_id still exists (the rebound job kept its id) but its
    origin now names a *different* source_instance.  Re-stage must reject
    this — not silently create a fresh replacement job.
    """
    import cron.jobs as jobs

    module = _module()
    entry = _entry()
    digest, disp_sha, receipt_path, fp = _stage_once(module, tmp_path, entry)

    # Read the original target_id from the receipt.
    original_receipt = json.loads(receipt_path.read_text())
    original_target_id = original_receipt["jobs"][0]["target_id"]

    # Rebind the persisted job''s origin.source_instance to a different value.
    # The job keeps its id, so the receipt''s target_id still exists — but its
    # origin no longer matches source-a.
    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
        assert persisted[0]["id"] == original_target_id
        original_origin = persisted[0]["origin"]
        tampered_origin = {
            **original_origin,
            "source_instance": "VPS:cron/jobs.json:DIFFERENT",
        }
        jobs.update_job(original_target_id, {"origin": tampered_origin})

    cat_path = tmp_path / "catalogue.json"
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    # Re-stage must reject: the receipt''s target_id still exists but its
    # origin has been rebound away from source-a.
    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="origin rebind|persisted contract drift|target_fingerprint"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


def test_e2e_origin_rebind_identity_theft_rejected(tmp_path: Path) -> None:
    """A *different* job''s origin is rebound to claim to be source-a
    (identity theft) while the original source-a job still exists.  The
    receipt names the original target_id; that job''s origin still matches,
    but a second job now also claims source-a.  Re-stage must use the
    receipt-authoritative target_id and reject the stolen job.
    """
    import cron.jobs as jobs

    module = _module()
    entry = _entry()
    digest, disp_sha, receipt_path, fp = _stage_once(module, tmp_path, entry)
    original_receipt = json.loads(receipt_path.read_text())
    original_target_id = original_receipt["jobs"][0]["target_id"]

    # Create a second job and rebind its origin to claim source-a.
    entry2 = _entry(source_instance="VPS:cron/jobs.json:source-b", source_id="source-b")
    (tmp_path / "b").mkdir(exist_ok=True)
    cat_path2, digest2 = _write_catalogue(tmp_path / "b", [entry, entry2])
    import shutil
    shutil.rmtree(tmp_path / "cron_home2", ignore_errors=True)
    catalogue2 = module.load_catalogue(cat_path2, digest2)
    selected2 = module.select_wave(catalogue2, "WAVE_1_BOUNDED_SCRIPTS")
    fp_a = module._target_fingerprint(entry)
    fp_b = module._target_fingerprint(entry2)
    disp2 = _disposition(digest2, [entry, entry2],
                         row_fingerprints={entry["source_instance"]: fp_a, entry2["source_instance"]: fp_b})
    disp_path2, disp_sha2 = _write_disposition(tmp_path / "b", disp2)
    receipt_path2 = tmp_path / "b" / "receipt.json"

    with jobs.use_cron_store(tmp_path / "cron_home2"):
        module.stage_wave(
            selected2,
            catalogue_sha256=digest2,
            disposition_sha256=disp_sha2,
            receipt_path=receipt_path2,
            create_job_fn=jobs.create_job,
            list_jobs_fn=jobs.list_jobs,
            update_job_fn=jobs.update_job,
            remove_job_fn=jobs.remove_job,
        )
        # Now rebind source-b''s job to claim to be source-a, with tampered payload.
        all_jobs = jobs.list_jobs(include_disabled=True)
        job_b = next(j for j in all_jobs if j["origin"]["source_instance"] == "VPS:cron/jobs.json:source-b")
        stolen_origin = {**job_b["origin"], "source_instance": "VPS:cron/jobs.json:source-a"}
        jobs.update_job(job_b["id"], {"origin": stolen_origin, "prompt": "stolen"})

    # Re-staging source-a must reject: the receipt names the original target
    # but a stolen job also claims source-a.  The receipt-authoritative target
    # must be validated and the rebind rejected.
    with jobs.use_cron_store(tmp_path / "cron_home2"):
        with pytest.raises(ValueError, match="persisted contract drift|origin rebind|target_fingerprint"):
            module.stage_wave(
                selected2,
                catalogue_sha256=digest2,
                disposition_sha256=disp_sha2,
                receipt_path=receipt_path2,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


def test_e2e_reuse_rejects_origin_target_fingerprint_mismatch(tmp_path: Path) -> None:
    """Hole 1: stage_wave reuse checks catalogue_sha256 but does NOT verify
    origin.target_fingerprint == recomputed expected fingerprint.  After
    remediation, a reused job whose persisted origin.target_fingerprint does
    not match the recomputed expected fingerprint must be rejected — even
    when the persisted contract otherwise matches (the stored fingerprint
    was tampered via update_job on the origin dict).
    """
    import cron.jobs as jobs

    module = _module()
    entry = _entry()
    digest, disp_sha, receipt_path, fp = _stage_once(module, tmp_path, entry)
    original_receipt = json.loads(receipt_path.read_text())
    original_target_id = original_receipt["jobs"][0]["target_id"]

    # Tamper ONLY the origin.target_fingerprint via update_job — leave the
    # rest of the contract intact.  The persisted contract still matches,
    # but the origin fingerprint is now wrong.
    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
        job = persisted[0]
        tampered_origin = {**job["origin"], "target_fingerprint": "0" * 64}
        jobs.update_job(job["id"], {"origin": tampered_origin})

    cat_path = tmp_path / "catalogue.json"
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="target fingerprint mismatch|target_fingerprint"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


def test_e2e_select_authorised_wave_missing_contract_fingerprint_rejected(
    tmp_path: Path,
) -> None:
    """Hole 2: select_authorised_wave must fail closed when a disposition
    row is missing its contract_fingerprint, not silently skip the check.
    """
    module = _module()
    entry = _entry()
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)

    # Disposition with NO contract_fingerprint on the row.
    disp = _disposition(digest, [entry])
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    with pytest.raises(ValueError, match="missing contract_fingerprint|fail-closed"):
        module.select_authorised_wave(catalogue, disp, "WAVE_1_BOUNDED_SCRIPTS")


def test_e2e_select_authorised_wave_policy_bound_fingerprint(tmp_path: Path) -> None:
    """Hole 2: select_authorised_wave must compute the expected fingerprint
    using the same explicit p13_policy that stage_wave will use.  A row
    fingerprint computed without the policy must NOT match.
    """
    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)

    policy = {"default": {"provider": "openai-approved", "model": "gpt-approved"}}
    # Fingerprint computed WITH the correct policy.
    fp_with_policy = module._target_fingerprint(entry, p13_policy=policy)
    # Fingerprint computed with a DIFFERENT policy (wrong — simulates the
    # old behaviour where select_authorised_wave didn''t pass p13_policy
    # and thus computed a fingerprint for a different contract).
    wrong_policy = {"default": {"provider": "openai-different", "model": "gpt-different"}}
    fp_wrong_policy = module._target_fingerprint(entry, p13_policy=wrong_policy)

    # Disposition with the WRONG (different-policy) fingerprint must be rejected.
    disp_wrong = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp_wrong_policy})
    with pytest.raises(ValueError, match="contract fingerprint mismatch"):
        module.select_authorised_wave(catalogue, disp_wrong, "WAVE_1_BOUNDED_SCRIPTS", p13_policy=policy)

    # Disposition with the CORRECT (policy-bound) fingerprint must pass.
    disp_right = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp_with_policy})
    selected = module.select_authorised_wave(catalogue, disp_right, "WAVE_1_BOUNDED_SCRIPTS", p13_policy=policy)
    assert len(selected) == 1


def test_e2e_provider_validation_disabled_provider_rejected(tmp_path: Path) -> None:
    """Hole 4: explicit P13 policy providers must be checked as not
    explicitly disabled in the effective config.  Uses a disposable
    test seam (no live user config).  A disabled provider fails closed.
    """
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    policy = {"default": {"provider": "openai-approved", "model": "gpt-approved"}}
    fp = module._target_fingerprint(entry, p13_policy=policy)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    # Disposable test seam: provider "openai-approved" is explicitly disabled.
    disabled_providers = {"openai-approved"}
    def validation_fn(provider: str) -> bool:
        return provider not in disabled_providers

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="disabled in the effective config|refusing to stage"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
                p13_policy=policy,
                provider_validation_fn=validation_fn,
            )


def test_e2e_provider_validation_enabled_provider_accepted(tmp_path: Path) -> None:
    """Hole 4 GREEN: when the policy provider is enabled in the disposable
    config seam, staging succeeds.
    """
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    policy = {"default": {"provider": "openai-approved", "model": "gpt-approved"}}
    fp = module._target_fingerprint(entry, p13_policy=policy)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    # Disposable test seam: provider is enabled.
    def validation_fn(provider: str) -> bool:
        return True

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        result = module.stage_wave(
            selected,
            catalogue_sha256=digest,
            disposition_sha256=disp_sha,
            receipt_path=receipt_path,
            create_job_fn=jobs.create_job,
            list_jobs_fn=jobs.list_jobs,
            update_job_fn=jobs.update_job,
            remove_job_fn=jobs.remove_job,
            p13_policy=policy,
            provider_validation_fn=validation_fn,
        )
    assert result["created"] == 1


def test_e2e_select_authorised_wave_provider_validation_disabled(tmp_path: Path) -> None:
    """Hole 4 (select_authorised_wave seam): the provider validation seam
    is also checked at the select_authorised_wave gate, so a disabled
    provider is rejected before staging even begins.
    """
    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)

    policy = {"default": {"provider": "openai-approved", "model": "gpt-approved"}}
    fp = module._target_fingerprint(entry, p13_policy=policy)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})

    disabled_providers = {"openai-approved"}
    def validation_fn(provider: str) -> bool:
        return provider not in disabled_providers

    with pytest.raises(ValueError, match="disabled in the effective config|refusing to stage"):
        module.select_authorised_wave(
            catalogue, disp, "WAVE_1_BOUNDED_SCRIPTS",
            p13_policy=policy,
            provider_validation_fn=validation_fn,
        )


# ---------------------------------------------------------------------------
# 4. Dict model normalized or rejected
# ---------------------------------------------------------------------------

def test_e2e_dict_model_normalized_to_strings(tmp_path: Path) -> None:
    """A legacy target_job carrying model/provider as a dict
    {model: ..., provider: ...} must be normalized to two strings before
    create_job is called — no silent loss."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={
        "no_agent": False,
        "script": None,
        "model": {"name": "gpt-custom", "provider": "openai"},
        "provider": {"name": "openai", "base_url": "https://api.openai.com/v1"},
    })
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    fp = module._target_fingerprint(entry)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        result = module.stage_wave(
            selected,
            catalogue_sha256=digest,
            disposition_sha256=disp_sha,
            receipt_path=receipt_path,
            create_job_fn=jobs.create_job,
            list_jobs_fn=jobs.list_jobs,
            update_job_fn=jobs.update_job,
            remove_job_fn=jobs.remove_job,
        )
    assert result["created"] == 1

    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
    job = persisted[0]
    # Dict values must have been normalized to strings, not silently lost.
    assert isinstance(job["model"], str)
    assert job["model"] == "gpt-custom"
    assert isinstance(job["provider"], str)
    assert job["provider"] == "openai"


def test_e2e_dict_model_unnormalizable_rejected(tmp_path: Path) -> None:
    """A dict model/provider that cannot be normalized to strings must
    fail closed rather than silently dropping the values."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={
        "no_agent": False,
        "script": None,
        "model": {"nested": {"too_deep": True}},
        "provider": {"also": {"nested": True}},
    })
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    # No valid fingerprint can be computed — disposition has no fingerprint.
    disp = _disposition(digest, [entry])
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="dict.*model|model.*dict|normalize.*model"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


# ---------------------------------------------------------------------------
# 5. Unpinned model missing explicit policy rejected
# ---------------------------------------------------------------------------

def test_e2e_unpinned_model_missing_explicit_policy_rejected(tmp_path: Path) -> None:
    """An agent job (no_agent=False) with no model/provider pin must NOT
    silently fall back to the root global default. It must require an
    explicit P13 policy mapping passed for the job or a declared default."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    # No policy mapping provided — no fingerprint computable.
    disp = _disposition(digest, [entry])
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        with pytest.raises(ValueError, match="explicit.*policy|policy.*mapping|unpinned"):
            module.stage_wave(
                selected,
                catalogue_sha256=digest,
                disposition_sha256=disp_sha,
                receipt_path=receipt_path,
                create_job_fn=jobs.create_job,
                list_jobs_fn=jobs.list_jobs,
                update_job_fn=jobs.update_job,
                remove_job_fn=jobs.remove_job,
            )


def test_e2e_explicit_policy_route_persisted(tmp_path: Path) -> None:
    """When an explicit P13 policy mapping is supplied for an unpinned
    agent job, the resolved model/provider strings are persisted on the
    staged job — no silent fall-back to the root global default."""
    import cron.jobs as jobs

    module = _module()
    entry = _entry(job={"no_agent": False, "script": None, "model": None, "provider": None})
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)
    selected = module.select_wave(catalogue, "WAVE_1_BOUNDED_SCRIPTS")

    policy = {"default": {"provider": "openai-approved", "model": "gpt-approved"}}
    fp = module._target_fingerprint(entry, p13_policy=policy)
    disp = _disposition(digest, [entry], row_fingerprints={entry["source_instance"]: fp})
    disp_path, disp_sha = _write_disposition(tmp_path, disp)

    receipt_path = tmp_path / "receipt.json"
    with jobs.use_cron_store(tmp_path / "cron_home"):
        result = module.stage_wave(
            selected,
            catalogue_sha256=digest,
            disposition_sha256=disp_sha,
            receipt_path=receipt_path,
            create_job_fn=jobs.create_job,
            list_jobs_fn=jobs.list_jobs,
            update_job_fn=jobs.update_job,
            remove_job_fn=jobs.remove_job,
            p13_policy=policy,
        )
    assert result["created"] == 1

    with jobs.use_cron_store(tmp_path / "cron_home"):
        persisted = jobs.list_jobs(include_disabled=True)
    job = persisted[0]
    assert job["provider"] == "openai-approved"
    assert job["model"] == "gpt-approved"


# ---------------------------------------------------------------------------
# 6. Per-row disposition fingerprint mismatch rejected
# ---------------------------------------------------------------------------

def test_e2e_per_row_disposition_fingerprint_mismatch_rejected(tmp_path: Path) -> None:
    """A disposition row whose contract_fingerprint does not match the
    expected current-contract fingerprint for that source_instance must
    be rejected — fail closed, not just on whole-catalogue hash."""
    module = _module()
    entry = _entry()
    cat_path, digest = _write_catalogue(tmp_path, [entry])
    catalogue = module.load_catalogue(cat_path, digest)

    # Wrong fingerprint in the disposition row.
    disp = _disposition(
        digest, [entry],
        row_fingerprints={entry["source_instance"]: "deadbeef" * 8},
    )

    with pytest.raises(ValueError, match="contract.*fingerprint|fingerprint.*mismatch"):
        module.select_authorised_wave(catalogue, disp, "WAVE_1_BOUNDED_SCRIPTS")
