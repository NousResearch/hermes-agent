"""Durable-custody contracts for the Ares CandidateStore."""

from __future__ import annotations

import io
import json
import multiprocessing
import os
import shutil
import socket
import tarfile
from pathlib import Path

import pytest

from hermes_cli.ares_candidate_store import (
    CandidateStore,
    CandidateStoreError,
    DurabilityFaultInjector,
    AUDIT_FAULT_BOUNDARIES,
    FAULT_MATRIX_SCHEMA,
    GC_FAULT_BOUNDARIES,
    PUBLICATION_FAULT_BOUNDARIES,
    canonical_json,
    generate_fault_matrix,
    sha256_bytes,
)
from hermes_cli.ares_candidate_lifecycle import CandidateLifecycleState
from ares_runtime.contracts import ActivationGrant


def _publish_in_child(root: str, source: str, queue) -> None:
    try:
        queue.put(CandidateStore(Path(root)).publish(Path(source), artifacts()).code)
    except Exception as exc:  # pragma: no cover - diagnostic returned to parent
        queue.put(f"ERROR:{type(exc).__name__}:{exc}")


def _identified(value: dict, field: str) -> dict:
    projection = dict(value)
    projection.pop(field, None)
    value[field] = sha256_bytes(canonical_json(projection))
    return value


def _json(path: Path, value: dict) -> None:
    path.write_bytes(canonical_json(value) + b"\n")


def candidate_source(
    root: Path,
    *,
    certification_overrides: dict | None = None,
    certification_mutator=None,
    certification_raw: bytes | None = None,
) -> tuple[Path, dict]:
    root.mkdir()
    payload = b"sealed-candidate-payload"
    release_entries = [
        {
            "kind": "file",
            "path": "file.txt",
            "mode": 0o644,
            "size": len(payload),
            "sha256": sha256_bytes(payload),
        }
    ]
    runtime_tree_sha256 = sha256_bytes(canonical_json(release_entries) + b"\n")
    release_manifest = (
        canonical_json({
            "schema": "AresReleaseManifestV1",
            "runtime_tree_sha256": runtime_tree_sha256,
            "files": release_entries,
        })
        + b"\n"
    )
    core = _identified(
        {
            "schema": "CandidateCoreV2",
            "canonicalization_version": "canonical-json-utf8-v1",
            "payload_files": [
                {
                    "path": "payload/file.txt",
                    "sha256": sha256_bytes(payload),
                    "size": len(payload),
                    "mode": 0o644,
                },
                {
                    "path": "payload/release-manifest.json",
                    "sha256": sha256_bytes(release_manifest),
                    "size": len(release_manifest),
                    "mode": 0o644,
                },
            ],
            "release_manifest_path": "payload/release-manifest.json",
            "release_manifest_sha256": sha256_bytes(release_manifest),
            "runtime_tree_sha256": runtime_tree_sha256,
        },
        "candidate_id",
    )

    def generation(number: int) -> dict:
        metrics = {
            "prompt_visible_provenance_bytes": 64,
            "prompt_visible_provenance_tokens": 16,
            "authoritative_provenance_bytes": 1024,
            "receipt_bytes": 2048,
            "cumulative_receipt_store_bytes": 4096,
            "compaction_latency_ms": 10.0,
            "restart_load_latency_ms": 10.0,
            "exact_expansion_latency_ms": 10.0,
            "input_tokens": 1024,
            "output_tokens": 512,
            "net_token_savings": 512,
            "budget_decision": "admit",
            "exact_expansion_hash": "a" * 64,
            "exact_expansion_expected_hash": "a" * 64,
            "exact_expansion_result": "PASS",
            "authenticated_restart_load_result": "PASS",
            "rendered_prompt_provenance_result": "PASS",
            "hmac_verification_result": "PASS",
            "key_id": "b" * 64,
        }
        samples = [
            {"phase": phase, "sample_index": index, "metrics": dict(metrics)}
            for phase, count in (("warmup", 3), ("measured", 10))
            for index in range(count)
        ]
        thresholds = []
        warnings = []
        for sample in samples:
            phase, index, values = (
                sample["phase"],
                sample["sample_index"],
                sample["metrics"],
            )
            for metric_id, limit in (
                ("prompt_visible_provenance_bytes", 512),
                ("prompt_visible_provenance_tokens", 128),
                ("authoritative_provenance_bytes", 131072),
                ("receipt_bytes", 524288),
                ("cumulative_receipt_store_bytes", number * 524288),
            ):
                thresholds.append({
                    "metric_id": metric_id,
                    "phase": phase,
                    "sample_index": index,
                    "observed": values[metric_id],
                    "hard_limit": limit,
                    "pass": True,
                })
            thresholds += [
                {
                    "metric_id": "net_token_savings",
                    "phase": phase,
                    "sample_index": index,
                    "observed": values["net_token_savings"],
                    "hard_limit": 128,
                    "pass": True,
                },
                {
                    "metric_id": "budget_decision",
                    "phase": phase,
                    "sample_index": index,
                    "observed": "admit",
                    "hard_limit": "admit",
                    "pass": True,
                },
                {
                    "metric_id": "exact_expansion",
                    "phase": phase,
                    "sample_index": index,
                    "observed": "PASS",
                    "hard_limit": "PASS",
                    "pass": True,
                },
                {
                    "metric_id": "hmac_verification",
                    "phase": phase,
                    "sample_index": index,
                    "observed": "PASS",
                    "hard_limit": "PASS",
                    "pass": True,
                },
            ]
            for metric_id, source_metric, limit in (
                ("receipt_bytes_soft_warning", "receipt_bytes", 393216),
                (
                    "cumulative_receipt_store_bytes_soft_warning",
                    "cumulative_receipt_store_bytes",
                    number * 393216,
                ),
                ("compaction_latency_soft_warning", "compaction_latency_ms", 2000),
                ("restart_load_latency_soft_warning", "restart_load_latency_ms", 100),
                (
                    "exact_expansion_latency_soft_warning",
                    "exact_expansion_latency_ms",
                    100,
                ),
            ):
                warnings.append({
                    "metric_id": metric_id,
                    "phase": phase,
                    "sample_index": index,
                    "observed": values[source_metric],
                    "warning_limit": limit,
                    "triggered": False,
                })
            warnings.append({
                "metric_id": "approximate_counter_usage",
                "phase": phase,
                "sample_index": index,
                "observed": True,
                "triggered": True,
            })
        thresholds += [
            {
                "metric_id": metric_id,
                "observed": 10.0,
                "hard_limit": limit,
                "pass": True,
            }
            for metric_id, limit in (
                ("compaction_p95_ms", 5000),
                ("restart_load_p95_ms", 500),
                ("exact_expansion_p95_ms", 500),
            )
        ]
        return {
            "generation": number,
            "warmup_runs": 3,
            "measured_runs": 10,
            "raw_measurement_samples": samples,
            "threshold_evaluations": thresholds,
            "soft_warning_evaluations": warnings,
            "failing_metric_ids": [],
            "hard_pass": True,
            "terminal_outcome": "PASS",
            "p50": {
                "compaction_latency_ms": 10.0,
                "restart_load_latency_ms": 10.0,
                "exact_expansion_latency_ms": 10.0,
            },
            "p95": {
                "compaction_latency_ms": 10.0,
                "restart_load_latency_ms": 10.0,
                "exact_expansion_latency_ms": 10.0,
            },
            "max": {
                "compaction_latency_ms": 10.0,
                "restart_load_latency_ms": 10.0,
                "exact_expansion_latency_ms": 10.0,
            },
        }

    certification = {
        "schema": "AresContextGovernorFullSealCertificationV2",
        "canonicalization_version": "canonical-json-utf8-v1",
        "certification_purpose": "FULL_SEAL",
        "certification_mode": "FULL_SEAL",
        "candidate_id": core["candidate_id"],
        "candidate_core_id": core["candidate_id"],
        "certification_set_inputs": {
            "candidate_id": core["candidate_id"],
            "candidate_core_id": core["candidate_id"],
            "required_artifact_names": [
                "gen-certification.json",
                "preseal-secret-scan.json",
                "scope-proof.json",
            ],
        },
        "required_generations": [16, 32],
        "required_warmup_runs": 3,
        "required_measured_runs": 10,
        "generations": [generation(16), generation(32)],
        "pass": True,
        "terminal_outcome": "PASS",
        "hard_pass": True,
        "failing_hard_metric_ids": [],
        "exact_expansion": "PASS",
        "authenticated_restart_load": "PASS",
        "rendered_prompt_provenance": "PASS",
        "integrity_hmac": "PASS",
        "authorization_state": "NON_AUTHORIZING",
        "non_authorizing": True,
    }
    certification.update(certification_overrides or {})
    if certification_mutator is not None:
        certification_mutator(certification)
    cert_artifact = certification_raw or (canonical_json(certification) + b"\n")
    (root / "gen-certification.json").write_bytes(cert_artifact)
    scope = {
        "schema": "AresContextGovernorScopeProofV2",
        "canonicalization_version": "canonical-json-utf8-v1",
        "candidate_id": core["candidate_id"],
        "pass": True,
    }
    preseal_scan = {
        "schema": "AresContextGovernorSecretScanV2",
        "canonicalization_version": "canonical-json-utf8-v1",
        "candidate_id": core["candidate_id"],
        "pass": True,
    }
    cert_artifacts = {
        "gen-certification.json": cert_artifact,
        "scope-proof.json": canonical_json(scope) + b"\n",
        "preseal-secret-scan.json": canonical_json(preseal_scan) + b"\n",
    }
    cert = _identified(
        {
            "schema": "CertificationSetV2",
            "canonicalization_version": "canonical-json-utf8-v1",
            "candidate_id": core["candidate_id"],
            "artifacts": [
                {"name": name, "sha256": sha256_bytes(raw)}
                for name, raw in sorted(cert_artifacts.items())
            ],
        },
        "certification_set_id",
    )
    core_raw, cert_raw = canonical_json(core) + b"\n", canonical_json(cert) + b"\n"
    archive = root / "ares-context-governor-candidate.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for name, data in (
            ("payload/file.txt", payload),
            ("payload/release-manifest.json", release_manifest),
            ("candidate-core-manifest.json", core_raw),
            ("certification-set-manifest.json", cert_raw),
            *cert_artifacts.items(),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = 0o644
            bundle.addfile(member, io.BytesIO(data))
    sealed = _identified(
        {
            "schema": "SealedCandidateV2",
            "canonicalization_version": "canonical-json-utf8-v1",
            "candidate_id": core["candidate_id"],
            "certification_set_id": cert["certification_set_id"],
            "archive_sha256": sha256_bytes(archive.read_bytes()),
        },
        "sealed_candidate_id",
    )
    post = _identified(
        {
            "schema": "PostSealEvidenceSetV1",
            "canonicalization_version": "canonical-json-utf8-v1",
            "candidate_id": core["candidate_id"],
            "certification_set_id": cert["certification_set_id"],
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "archive_sha256": sealed["archive_sha256"],
            "artifacts": [],
        },
        "post_seal_evidence_set_id",
    )
    auth = _identified(
        {
            "schema": "ActivationAuthorizationV1",
            "canonicalization_version": "canonical-json-utf8-v1",
            "candidate_id": core["candidate_id"],
            "certification_set_id": cert["certification_set_id"],
            "sealed_candidate_id": sealed["sealed_candidate_id"],
            "post_seal_evidence_set_id": post["post_seal_evidence_set_id"],
            "archive_sha256": sealed["archive_sha256"],
            "rendered_config_path": "rendered.json",
            "rendered_config_sha256": "0" * 64,
            "authorization_state": "NON_AUTHORIZING",
            "non_authorizing": True,
            "approved_release_root": f"content-addressed/{sealed['sealed_candidate_id']}",
            "governed_key_policy": {
                "snapshot_schema": "AresContextGovernorKeySnapshotV2",
                "authority": "descriptor-backed-ares-owned",
                "caller_key_material": "forbidden",
            },
        },
        "activation_authorization_id",
    )
    for name, value in (
        ("candidate-core-manifest.json", core),
        ("certification-set-manifest.json", cert),
        ("sealed-candidate-manifest.json", sealed),
        ("post-seal-evidence-set.json", post),
        ("activation-authorization.json", auth),
    ):
        _json(root / name, value)
    _json(
        root / "custody-fault-matrix-v1.json",
        {
            "schema": FAULT_MATRIX_SCHEMA,
            "canonicalization_version": "canonical-json-utf8-v1",
            "records": [],
        },
    )
    postseal_scan = {
        "schema": "AresContextGovernorSecretScanV2",
        "canonicalization_version": "canonical-json-utf8-v1",
        "candidate_id": core["candidate_id"],
        "sealed_candidate_id": sealed["sealed_candidate_id"],
        "archive_sha256": sealed["archive_sha256"],
        "pass": True,
    }
    v1 = {"schema": "AresContextGovernorV1ImmutabilityV2", "pass": True}
    dry_run = {
        "schema": "AresContextGovernorDryRunActivationV3",
        "canonicalization_version": "canonical-json-utf8-v1",
        "candidate_id": core["candidate_id"],
        "sealed_candidate_id": sealed["sealed_candidate_id"],
        "pass": True,
    }
    archive_verification = {
        "schema": "AresContextGovernorArchiveVerificationV2",
        "canonicalization_version": "canonical-json-utf8-v1",
        "candidate_id": core["candidate_id"],
        "certification_set_id": cert["certification_set_id"],
        "sealed_candidate_id": sealed["sealed_candidate_id"],
        "archive_sha256": sealed["archive_sha256"],
        "pass": True,
    }
    for name, value in (
        ("scope-proof.json", scope),
        ("preseal-secret-scan.json", preseal_scan),
        ("postseal-secret-scan.json", postseal_scan),
        ("v1-immutability.json", v1),
        ("dry-run-activation.json", dry_run),
        ("archive-verification.json", archive_verification),
    ):
        _json(root / name, value)
    return root, sealed


def artifacts() -> tuple[str, ...]:
    return (
        "ares-context-governor-candidate.tar",
        "candidate-core-manifest.json",
        "certification-set-manifest.json",
        "gen-certification.json",
        "sealed-candidate-manifest.json",
        "post-seal-evidence-set.json",
        "activation-authorization.json",
        "custody-fault-matrix-v1.json",
        "scope-proof.json",
        "preseal-secret-scan.json",
        "postseal-secret-scan.json",
        "v1-immutability.json",
        "dry-run-activation.json",
        "archive-verification.json",
    )


def publish(tmp_path: Path):
    source, sealed = candidate_source(tmp_path / "scratch")
    store = CandidateStore(tmp_path / "ares")
    return store, source, sealed, store.publish(source, artifacts())


def approval(candidate_id: str, archive_sha256: str) -> dict:
    return _identified(
        {
            "schema": "AresCandidateGcApprovalV1",
            "sealed_candidate_id": candidate_id,
            "archive_sha256": archive_sha256,
            "approved_at_unix_ns": 1,
        },
        "gc_approval_id",
    )


def activation_grant(
    store: CandidateStore, sealed_candidate_id: str
) -> ActivationGrant:
    snapshot = store.verify(sealed_candidate_id)
    event = (
        store._candidate_root(sealed_candidate_id)
        / "events"
        / f"{snapshot['lifecycle_sequence']:020d}.json"
    ).read_bytes()
    return ActivationGrant(
        candidate_id=snapshot["candidate_id"],
        certification_set_id=snapshot["certification_set_id"],
        sealed_candidate_id=snapshot["sealed_candidate_id"],
        audit_subject_id=snapshot["audit_subject"]["id"],
        audit_subject_sha256=snapshot["audit_subject"]["sha256"],
        audit_result_sha256=sha256_bytes(event),
        archive_sha256=snapshot["archive_sha256"],
        candidate_core_sha256=snapshot["candidate_core"]["sha256"],
        sealed_manifest_sha256=snapshot["sealed_candidate_manifest"]["sha256"],
        release_manifest_sha256=str(
            json.loads(
                (
                    store._candidate_root(sealed_candidate_id)
                    / "artifacts"
                    / "candidate-core-manifest.json"
                ).read_text()
            )["release_manifest_sha256"]
        ),
        runtime_tree_sha256=str(
            json.loads(
                (
                    store._candidate_root(sealed_candidate_id)
                    / "artifacts"
                    / "candidate-core-manifest.json"
                ).read_text()
            )["runtime_tree_sha256"]
        ),
        custody_event_sequence=int(snapshot["lifecycle_sequence"]) + 1,
        target_platform="linux-x86_64",
        target_release_root=str(store.root / "releases"),
        materializer_contract="AresReleaseMaterializerV1",
        activator_contract="AresReleaseActivatorV1",
        resolver_contract="AresRuntimeResolverV1",
    ).with_grant_id()


def test_publication_survives_fresh_process_and_scratch_deletion(tmp_path: Path):
    store, source, sealed, result = publish(tmp_path)
    assert result.code == "PUBLISHED"
    shutil.rmtree(source)

    fresh = CandidateStore(store.root)
    snapshot = fresh.verify(sealed["sealed_candidate_id"])
    assert snapshot["candidate_id"] == sealed["candidate_id"]
    assert snapshot["archive_sha256"] == sealed["archive_sha256"]
    assert fresh.recover()[0]["sealed_candidate_id"] == sealed["sealed_candidate_id"]


def test_missing_custody_blocks_and_corruption_fails(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    with pytest.raises(CandidateStoreError, match="CUSTODY_UNAVAILABLE"):
        store.issue_handoff("0" * 64)
    archive = result.candidate_root / "artifacts/ares-context-governor-candidate.tar"
    archive.chmod(0o600)
    archive.write_bytes(b"tampered")
    archive.chmod(0o400)
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        store.verify(sealed["sealed_candidate_id"])


def test_partial_publication_never_lists_as_sealed(tmp_path: Path, monkeypatch):
    source, _sealed = candidate_source(tmp_path / "scratch")
    store = CandidateStore(tmp_path / "ares")
    monkeypatch.setattr(
        store, "_copy_artifact", lambda *_: (_ for _ in ()).throw(OSError("fault"))
    )
    with pytest.raises(OSError):
        store.publish(source, artifacts())
    assert store.list() == []
    assert list((store.candidates_root / ".incoming").iterdir())


def test_post_rename_failure_never_enumerates_as_sealed(tmp_path: Path, monkeypatch):
    source, sealed = candidate_source(tmp_path / "scratch")
    store = CandidateStore(tmp_path / "ares")
    monkeypatch.setattr(
        store,
        "_write_event_and_snapshot_fd",
        lambda *_: (_ for _ in ()).throw(OSError("post-rename fault")),
    )
    with pytest.raises(OSError, match="post-rename"):
        store.publish(source, artifacts())
    assert store.list() == []
    assert store.verify(sealed["sealed_candidate_id"])["lifecycle_state"] == "SEALING"


def test_conflicting_publisher_cannot_replace_final_candidate(tmp_path: Path):
    store, source, sealed, result = publish(tmp_path)
    assert store.publish(source, artifacts()).code == "ALREADY_PUBLISHED_VERIFIED"
    (source / "ares-context-governor-candidate.tar").write_bytes(b"different")
    with pytest.raises(CandidateStoreError, match="PUBLICATION_CONFLICT"):
        store.publish(source, artifacts())
    assert (
        store.verify(sealed["sealed_candidate_id"])["archive_sha256"]
        == sealed["archive_sha256"]
    )


def test_concurrent_identical_publishers_have_one_commit_point(tmp_path: Path):
    source, _sealed = candidate_source(tmp_path / "scratch")
    root = tmp_path / "ares"
    queue = multiprocessing.Queue()
    workers = [
        multiprocessing.Process(
            target=_publish_in_child, args=(str(root), str(source), queue)
        )
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
        assert worker.exitcode == 0
    assert sorted(queue.get(timeout=2) for _ in workers) == [
        "ALREADY_PUBLISHED_VERIFIED",
        "PUBLISHED",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"non_authorizing": False},
        {"non_authorizing": None},
        {"authorization_state": "AUTHORIZED"},
        {"authorization_state": "UNKNOWN_FUTURE_MODE"},
    ],
)
def test_publication_rejects_non_authorizing_certification_contradictions(
    tmp_path: Path, overrides: dict
):
    source, _sealed = candidate_source(
        tmp_path / "scratch", certification_overrides=overrides
    )

    with pytest.raises(CandidateStoreError, match="INVALID_CERTIFICATION_AUTHORITY"):
        CandidateStore(tmp_path / "ares").publish(source, artifacts())


def test_publication_rejects_duplicate_conflicting_certification_fields(tmp_path: Path):
    raw = (
        b'{"schema":"AresContextGovernorGenCertificationV1",'
        b'"authorization_state":"NON_AUTHORIZING",'
        b'"non_authorizing":true,"non_authorizing":false}'
    )
    source, _sealed = candidate_source(tmp_path / "scratch", certification_raw=raw)

    with pytest.raises(CandidateStoreError, match="INVALID_CERTIFICATION_AUTHORITY"):
        CandidateStore(tmp_path / "ares").publish(source, artifacts())


@pytest.mark.parametrize(
    "overrides",
    [
        {"certification_purpose": "STAGED"},
        {"certification_mode": "dry_run"},
        {"schema": "AresContextGovernorStagedCertificationV1"},
        {"pass": False},
        {"terminal_outcome": "HARD_FAILURE"},
        {"candidate_id": "f" * 64},
        {"required_generations": [16]},
        {"failing_hard_metric_ids": ["exact_expansion"]},
        {"unknown_future_purpose": "diagnostic"},
    ],
)
def test_only_complete_candidate_bound_full_seal_can_enter_certification_set(
    tmp_path: Path, overrides: dict
):
    source, _sealed = candidate_source(
        tmp_path / "scratch", certification_overrides=overrides
    )
    with pytest.raises(CandidateStoreError, match="INVALID_CERTIFICATION_AUTHORITY"):
        CandidateStore(tmp_path / "ares").publish(source, artifacts())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda certification: certification.clear(),
        lambda certification: certification["generations"].pop(0),
        lambda certification: certification["generations"].pop(),
        lambda certification: certification["generations"][0].pop(
            "soft_warning_evaluations"
        ),
        lambda certification: certification["generations"][0][
            "raw_measurement_samples"
        ][0].pop("sample_index"),
        lambda certification: certification["generations"][0]["threshold_evaluations"][
            0
        ].pop("hard_limit"),
        lambda certification: certification["generations"][0].__setitem__(
            "warmup_runs", 2
        ),
        lambda certification: certification["generations"][0].__setitem__(
            "measured_runs", 9
        ),
        lambda certification: certification["generations"][0].__setitem__(
            "failing_metric_ids", ["exact_expansion"]
        ),
        lambda certification: certification.__setitem__("terminal_outcome", False),
        lambda certification: certification.pop("terminal_outcome"),
        lambda certification: certification.pop("exact_expansion"),
        lambda certification: certification.pop("authenticated_restart_load"),
        lambda certification: certification.pop("rendered_prompt_provenance"),
        lambda certification: certification.pop("integrity_hmac"),
        lambda certification: certification["generations"][0][
            "raw_measurement_samples"
        ][0]["metrics"].__setitem__("exact_expansion_result", "FAIL"),
        lambda certification: certification["generations"][0][
            "raw_measurement_samples"
        ][0]["metrics"].__setitem__("hmac_verification_result", "FAIL"),
        lambda certification: certification["certification_set_inputs"].__setitem__(
            "candidate_id", "f" * 64
        ),
        lambda certification: certification["certification_set_inputs"].__setitem__(
            "required_artifact_names", ["gen-certification.json"]
        ),
    ],
    ids=[
        "authority-fields-only",
        "missing-gen16",
        "missing-gen32",
        "missing-soft-warning-evidence",
        "missing-sample-binding",
        "missing-threshold-evidence",
        "wrong-warmup-count",
        "wrong-measured-count",
        "one-hard-metric-failure",
        "terminal-pass-false",
        "terminal-pass-missing",
        "missing-exact-expansion",
        "missing-restart-load",
        "missing-rendered-provenance",
        "missing-integrity-hmac",
        "exact-expansion-failure",
        "hmac-failure",
        "cross-candidate-input",
        "missing-scope-and-preseal-inputs",
    ],
)
def test_incomplete_or_malformed_full_seal_is_never_admitted(tmp_path: Path, mutate):
    source, _sealed = candidate_source(
        tmp_path / "scratch", certification_mutator=mutate
    )

    with pytest.raises(CandidateStoreError, match="INVALID_CERTIFICATION_AUTHORITY"):
        CandidateStore(tmp_path / "ares").publish(source, artifacts())


def test_duplicate_authorization_json_and_false_non_authorizing_fail_closed(
    tmp_path: Path,
):
    source, _sealed = candidate_source(tmp_path / "scratch")
    auth = (source / "activation-authorization.json").read_bytes().rstrip(b"\n")
    (source / "activation-authorization.json").write_bytes(
        auth[:-1] + b',"activation_authorization_id":"x"}\n'
    )
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        CandidateStore(tmp_path / "ares").publish(source, artifacts())

    source, _sealed = candidate_source(tmp_path / "scratch-two")
    auth = json.loads((source / "activation-authorization.json").read_text())
    auth["non_authorizing"] = False
    auth["activation_authorization_id"] = _identified(
        {
            key: value
            for key, value in auth.items()
            if key != "activation_authorization_id"
        },
        "activation_authorization_id",
    )["activation_authorization_id"]
    _json(source / "activation-authorization.json", auth)
    with pytest.raises(CandidateStoreError, match="INVALID_CERTIFICATION_AUTHORITY"):
        CandidateStore(tmp_path / "ares-two").publish(source, artifacts())


def test_audit_lease_blocks_gc_and_stale_lease_becomes_blocked(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    with pytest.raises(CandidateStoreError, match="GC_PROTECTED_LIFECYCLE"):
        store.gc_plan(
            sealed["sealed_candidate_id"],
            approval(sealed["sealed_candidate_id"], sealed["archive_sha256"]),
        )
    with pytest.raises(CandidateStoreError, match="AUDIT_LOCKED"):
        store.reject(sealed["sealed_candidate_id"])
    lease.close()  # Reboot-equivalent: OS released lock but durable lease remains.
    fresh = CandidateStore(store.root)
    recovered = fresh.recover()
    assert recovered[0]["lifecycle_state"] == "AUDIT_BLOCKED"
    assert recovered[0]["activation_authorization_state"] == "UNAUTHORIZED"


def test_only_explicit_governed_transition_can_authorize_after_audit_pass(
    tmp_path: Path,
):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    passed = store.record_audit_result(lease, passed=True, reason="test-pass")

    assert passed["lifecycle_state"] == "AUDIT_PASSED"
    assert passed["activation_authorization_state"] == "UNAUTHORIZED"
    grant = activation_grant(store, sealed["sealed_candidate_id"])
    authorized = store.authorize_activation(sealed["sealed_candidate_id"], grant=grant)
    assert authorized["lifecycle_state"] == "AWAITING_ACTIVATION"
    assert authorized["activation_authorization_state"] == "AUTHORIZED"
    assert store.read_activation_grant(sealed["sealed_candidate_id"]) == grant


@pytest.mark.parametrize("verdict", ["FAILED", "false", 1, 0, None, [], {}, object()])
def test_truthy_or_untyped_audit_verdict_never_mutates_audit_state(
    tmp_path: Path, verdict
):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    with pytest.raises(CandidateStoreError, match="INVALID_AUDIT_VERDICT"):
        store.record_audit_result(lease, passed=verdict, reason="hostile-verdict")
    assert (
        store.verify(sealed["sealed_candidate_id"])["audit_state"]
        == "HOSTILE_AUDIT_IN_PROGRESS"
    )
    lease.close()


@pytest.mark.parametrize("passed", [False])
def test_failed_audit_cannot_authorize_activation(tmp_path: Path, passed: bool):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    failed = store.record_audit_result(lease, passed=passed, reason="test-failure")

    assert failed["activation_authorization_state"] == "UNAUTHORIZED"
    with pytest.raises(CandidateStoreError, match="MISSING_AUTHORIZATION_EVIDENCE"):
        store.authorize_activation(sealed["sealed_candidate_id"], grant=object())


def test_authorization_rejects_grant_for_a_different_candidate(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    store.record_audit_result(lease, passed=True, reason="test-pass")
    grant = activation_grant(store, sealed["sealed_candidate_id"])
    wrong = ActivationGrant(**{
        **grant.__dict__,
        "candidate_id": "f" * 64,
        "grant_id": None,
    }).with_grant_id()

    with pytest.raises(CandidateStoreError, match="ACTIVATION_GRANT_MISMATCH"):
        store.authorize_activation(sealed["sealed_candidate_id"], grant=wrong)


def test_authorization_rejects_grant_with_wrong_sealed_runtime_tree(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    store.record_audit_result(lease, passed=True, reason="test-pass")
    grant = activation_grant(store, sealed["sealed_candidate_id"])
    wrong = ActivationGrant(**{
        **grant.__dict__,
        "runtime_tree_sha256": "f" * 64,
        "grant_id": None,
    }).with_grant_id()

    with pytest.raises(CandidateStoreError, match="ACTIVATION_GRANT_MISMATCH"):
        store.authorize_activation(sealed["sealed_candidate_id"], grant=wrong)


def test_activation_and_rollback_terminal_transitions_are_grant_bound(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    store.record_audit_result(lease, passed=True, reason="test-pass")
    grant = activation_grant(store, sealed["sealed_candidate_id"])
    store.authorize_activation(sealed["sealed_candidate_id"], grant=grant)

    with pytest.raises(CandidateStoreError, match="ACTIVATION_GRANT_MISMATCH"):
        store.record_activation_success(
            sealed["sealed_candidate_id"], grant_id="0" * 64, reason="bad-grant"
        )
    active = store.record_activation_success(
        sealed["sealed_candidate_id"], grant_id=grant.grant_id, reason="certified"
    )
    assert active["lifecycle_state"] == "ACTIVE"
    required = store.record_rollback_required(
        sealed["sealed_candidate_id"], grant_id=grant.grant_id, reason="health-failed"
    )
    assert required["rollback_required"] is True
    rolled_back = store.record_rollback_success(
        sealed["sealed_candidate_id"],
        grant_id=grant.grant_id,
        reason="previous-verified",
    )
    assert rolled_back["lifecycle_state"] == "ROLLED_BACK"
    assert rolled_back["activation_authorization_state"] == "UNAUTHORIZED"


def test_handoff_binds_immutable_subject_not_mutable_lifecycle(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    handoff = store.issue_handoff(sealed["sealed_candidate_id"])
    snapshot = store.verify(sealed["sealed_candidate_id"])
    assert handoff["audit_subject_id"] == snapshot["audit_subject"]["id"]
    subject = json.loads((result.candidate_root / "audit-subject.json").read_text())
    assert subject["publication_custody_sha256"] != sha256_bytes(
        (result.candidate_root / "custody.json").read_bytes()
    )
    lease = store.start_audit(sealed["sealed_candidate_id"])
    # Audit-start bookkeeping changes the snapshot sequence but cannot change
    # the already-issued immutable audit subject.
    assert (
        store.verify(sealed["sealed_candidate_id"])["audit_subject"]
        == snapshot["audit_subject"]
    )
    lease.close()
    archive = (
        result.candidate_root / "artifacts" / "ares-context-governor-candidate.tar"
    )
    archive.chmod(0o600)
    archive.write_bytes(b"changed-bytes")
    archive.chmod(0o400)
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        store.verify(sealed["sealed_candidate_id"])


def test_handoff_substitution_and_cross_candidate_binding_fail_closed(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    snapshot = store.verify(sealed["sealed_candidate_id"])
    handoff_path = result.candidate_root / snapshot["audit_handoff"]["relative_path"]
    forged = json.loads(handoff_path.read_text())
    forged["sealed_candidate_id"] = "f" * 64
    handoff_path.chmod(0o600)
    _json(handoff_path, forged)
    handoff_path.chmod(0o400)
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        store.verify(sealed["sealed_candidate_id"])


@pytest.mark.parametrize(
    "name, writer",
    [
        ("unlisted.txt", lambda path: path.write_text("unlisted", encoding="utf-8")),
        ("unlisted-dir", lambda path: path.mkdir()),
        ("unlisted-link", lambda path: path.symlink_to("candidate-core-manifest.json")),
    ],
)
def test_actual_artifact_tree_must_exactly_equal_sealed_inventory(
    tmp_path: Path, name: str, writer
):
    """An immutable digest is not meaningful unless the live tree is exact."""
    store, _source, sealed, result = publish(tmp_path)
    artifacts_root = result.candidate_root / "artifacts"
    artifacts_root.chmod(0o700)
    writer(artifacts_root / name)
    artifacts_root.chmod(0o500)
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        store.verify(sealed["sealed_candidate_id"])


def test_complete_lifecycle_history_rejects_missing_or_extra_events(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    events = result.candidate_root / "events"
    events.chmod(0o700)
    (events / "00000000000000000001.json").unlink()
    events.chmod(0o500)
    with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
        store.verify(sealed["sealed_candidate_id"])


def test_counterfeit_or_replaced_audit_lease_cannot_record_result(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    path = result.candidate_root / "audit-lease.json"
    path.chmod(0o600)
    _json(
        path,
        {
            "schema": "AresCandidateAuditLeaseV1",
            "sealed_candidate_id": sealed["sealed_candidate_id"],
        },
    )
    path.chmod(0o400)
    with pytest.raises(CandidateStoreError, match="AUDIT_LEASE|CUSTODY_CORRUPT"):
        store.record_audit_result(lease, passed=True, reason="counterfeit")
    assert path.exists(), "a failed result must not retire the only lease evidence"


def test_missing_audit_lease_recovers_to_explicit_audit_blocked(tmp_path: Path):
    store, _source, sealed, result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    lease = store.start_audit(sealed["sealed_candidate_id"])
    lease.close()
    path = result.candidate_root / "audit-lease.json"
    path.chmod(0o600)
    path.unlink()
    recovered = CandidateStore(store.root).recover()
    assert recovered[0]["lifecycle_state"] == "AUDIT_BLOCKED"


def test_recovery_never_deletes_quarantine_for_malformed_tombstone(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.reject(sealed["sealed_candidate_id"])
    store._faults = DurabilityFaultInjector(frozenset({"gc_final_removal"}))
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.gc(
            sealed["sealed_candidate_id"],
            approval(sealed["sealed_candidate_id"], sealed["archive_sha256"]),
        )
    tombstone = (
        store.candidates_root / "tombstones" / f"{sealed['sealed_candidate_id']}.json"
    )
    tombstone.chmod(0o600)
    tombstone.write_text("{}", encoding="utf-8")
    tombstone.chmod(0o400)
    recovered = CandidateStore(store.root).recover()
    assert (
        recovered
        and recovered[0]["sealed_candidate_id"] == sealed["sealed_candidate_id"]
    )


def test_explicitly_rejected_candidate_is_quarantined_with_tombstone(tmp_path: Path):
    store, _source, sealed, _result = publish(tmp_path)
    store.reject(sealed["sealed_candidate_id"])
    tombstone = store.gc(
        sealed["sealed_candidate_id"],
        approval(sealed["sealed_candidate_id"], sealed["archive_sha256"]),
    )
    assert tombstone["schema"] == "AresCandidateGcTombstoneV1"
    assert tombstone["pre_gc_lifecycle_state"] == "REJECTED"
    assert tombstone["gc_approval"]["gc_approval_id"] == tombstone["gc_approval_id"]
    assert (
        store.candidates_root / "tombstones" / f"{sealed['sealed_candidate_id']}.json"
    ).exists()


@pytest.mark.parametrize(
    ("point", "expect_restored"),
    [("gc_tombstone.write", True), ("gc_final_removal", False)],
)
def test_gc_restart_recovers_quarantine_deterministically(
    tmp_path: Path, point: str, expect_restored: bool
):
    store, _source, sealed, _result = publish(tmp_path)
    store.reject(sealed["sealed_candidate_id"])
    store._faults = DurabilityFaultInjector(frozenset({point}))
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.gc(
            sealed["sealed_candidate_id"],
            approval(sealed["sealed_candidate_id"], sealed["archive_sha256"]),
        )
    fresh = CandidateStore(store.root)
    recovered = fresh.recover()
    if expect_restored:
        assert recovered[0]["sealed_candidate_id"] == sealed["sealed_candidate_id"]
    else:
        assert recovered == []
        assert (
            store.candidates_root
            / "tombstones"
            / f"{sealed['sealed_candidate_id']}.json"
        ).exists()


@pytest.mark.parametrize(
    "state",
    [
        CandidateLifecycleState.AWAITING_HOSTILE_AUDIT,
        CandidateLifecycleState.HOSTILE_AUDIT_IN_PROGRESS,
        CandidateLifecycleState.AUDIT_BLOCKED,
        CandidateLifecycleState.AUDIT_PASSED,
        CandidateLifecycleState.AWAITING_ACTIVATION,
        CandidateLifecycleState.ACTIVE,
        CandidateLifecycleState.ROLLBACK_REQUIRED,
        CandidateLifecycleState.INCIDENT_HELD,
    ],
)
def test_gc_plan_refuses_every_protected_lifecycle_state(monkeypatch, state):
    store = CandidateStore(Path("/non-authoritative-test-root"))
    candidate_id = "a" * 64
    monkeypatch.setattr(store, "verify", lambda _: {"lifecycle_state": state.value})
    with pytest.raises(CandidateStoreError, match="GC_PROTECTED_LIFECYCLE"):
        store.gc_plan(candidate_id, approval(candidate_id, "b" * 64))


def test_hardlinked_and_symlinked_source_files_fail_closed(tmp_path: Path):
    source, _sealed = candidate_source(tmp_path / "scratch")
    hardlink = source / "gen-certification.json"
    hardlink.unlink()
    hardlink.hardlink_to(source / "candidate-core-manifest.json")
    store = CandidateStore(tmp_path / "ares")
    with pytest.raises(CandidateStoreError, match="UNSAFE_FILESYSTEM_OBJECT"):
        store.publish(source, artifacts())


@pytest.mark.parametrize("point", PUBLICATION_FAULT_BOUNDARIES)
def test_every_publication_durability_boundary_fails_closed(tmp_path: Path, point: str):
    source, sealed = candidate_source(tmp_path / "scratch")
    faults = DurabilityFaultInjector(frozenset({point}))
    store = CandidateStore(tmp_path / "ares", fault_injector=faults)
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.publish(source, artifacts())
    # The commit point is the final rename + parent fsync.  A failure before it
    # cannot be listed; a failure after it remains only SEALING and cannot be
    # handed to an auditor or activated.
    assert store.list() == []
    with pytest.raises(CandidateStoreError, match="CUSTODY_UNAVAILABLE"):
        store.issue_handoff(sealed["sealed_candidate_id"])
    fresh = CandidateStore(store.root)
    assert fresh.list() == []


def test_publication_fault_matrix_is_machine_readable(tmp_path: Path):
    records = []
    for point in PUBLICATION_FAULT_BOUNDARIES:
        source, sealed = candidate_source(tmp_path / point.replace(".", "-"))
        store = CandidateStore(
            tmp_path / f"ares-{point.replace('.', '-')}",
            fault_injector=DurabilityFaultInjector(frozenset({point})),
        )
        with pytest.raises(CandidateStoreError) as failure:
            store.publish(source, artifacts())
        fresh = CandidateStore(store.root)
        recovered = fresh.recover()
        listed = fresh.list()
        records.append({
            "boundary_id": point,
            "operation": "publication",
            "injected_error": failure.value.code,
            "previous_lifecycle_state": "CERTIFIED",
            "recovered_lifecycle_state": recovered[0]["lifecycle_state"]
            if recovered
            else None,
            "final_candidate_directory_exists": (
                store.candidates_root / sealed["sealed_candidate_id"]
            ).exists(),
            "enumerates": bool(listed),
            "authorizing": False,
            "handoff_exists": False,
            "activation_authorization_exists": False,
            "result": "PASS",
        })
    matrix = {"schema": FAULT_MATRIX_SCHEMA, "records": records}
    path = tmp_path / "publication-fault-matrix-v1.json"
    _json(path, matrix)
    assert json.loads(path.read_text()) == matrix
    assert {record["boundary_id"] for record in records} == set(
        PUBLICATION_FAULT_BOUNDARIES
    )


def test_complete_fault_matrix_is_portable_and_covers_all_operations(tmp_path: Path):
    source, _sealed = candidate_source(tmp_path / "scratch")
    matrix = generate_fault_matrix(source, artifacts())
    assert matrix["schema"] == FAULT_MATRIX_SCHEMA
    assert matrix["summary"]["all_records_pass"] is True
    assert matrix["summary"] == {
        "publication_boundaries": len(PUBLICATION_FAULT_BOUNDARIES),
        "audit_boundaries": len(AUDIT_FAULT_BOUNDARIES),
        "gc_boundaries": len(GC_FAULT_BOUNDARIES),
        "all_records_pass": True,
    }
    assert {record["operation"] for record in matrix["records"]} == {
        "publication",
        "audit_handoff",
        "audit_start",
        "gc",
    }
    assert all(
        "/tmp" not in canonical_json(record).decode() for record in matrix["records"]
    )


def test_post_commit_fault_recovers_the_exact_committed_candidate(tmp_path: Path):
    source, sealed = candidate_source(tmp_path / "scratch")
    store = CandidateStore(
        tmp_path / "ares",
        fault_injector=DurabilityFaultInjector(frozenset({"candidates_parent.fsync"})),
    )
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.publish(source, artifacts())
    # The final rename plus parent fsync is the durable commit point.  The
    # remaining SEALING receipt is non-authorizing until recovery finalizes it.
    assert store.list() == []
    fresh = CandidateStore(store.root)
    recovered = fresh.recover()
    assert recovered[0]["sealed_candidate_id"] == sealed["sealed_candidate_id"]
    assert recovered[0]["lifecycle_state"] == "SEALED"


@pytest.mark.parametrize("point", AUDIT_FAULT_BOUNDARIES[:5])
def test_handoff_persistence_failures_never_authorize_audit(tmp_path: Path, point: str):
    store, _source, sealed, _result = publish(tmp_path)
    store._faults = DurabilityFaultInjector(frozenset({point}))
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.issue_handoff(sealed["sealed_candidate_id"])
    snapshot = CandidateStore(store.root).verify(sealed["sealed_candidate_id"])
    assert snapshot["lifecycle_state"] == "SEALED"
    assert snapshot["audit_state"] == "NOT_HANDED_OFF"


@pytest.mark.parametrize("point", AUDIT_FAULT_BOUNDARIES[5:])
def test_audit_start_persistence_failures_never_authorize_result(
    tmp_path: Path, point: str
):
    store, _source, sealed, _result = publish(tmp_path)
    store.issue_handoff(sealed["sealed_candidate_id"])
    store._faults = DurabilityFaultInjector(frozenset({point}))
    with pytest.raises(CandidateStoreError, match="INJECTED_DURABILITY_FAILURE"):
        store.start_audit(sealed["sealed_candidate_id"])
    snapshot = CandidateStore(store.root).verify(sealed["sealed_candidate_id"])
    assert snapshot["activation_authorization_state"] == "UNAUTHORIZED"


def test_source_intermediate_symlink_and_special_file_fail_closed(tmp_path: Path):
    source, _sealed = candidate_source(tmp_path / "scratch")
    (source / "nested-real").mkdir()
    (source / "nested-real" / "artifact.txt").write_text("hostile path")
    (source / "nested").symlink_to(source / "nested-real", target_is_directory=True)
    with pytest.raises(CandidateStoreError, match="UNSAFE_FILESYSTEM_OBJECT"):
        CandidateStore(tmp_path / "ares").publish(
            source, (*artifacts(), "nested/artifact.txt")
        )

    source2, _sealed2 = candidate_source(tmp_path / "scratch-fifo")
    os.mkfifo(source2 / "unexpected-fifo")
    # A FIFO cannot be smuggled in as an explicitly supplied artifact either.
    with pytest.raises(CandidateStoreError, match="UNSAFE_FILESYSTEM_OBJECT"):
        CandidateStore(tmp_path / "ares2").publish(
            source2, (*artifacts(), "unexpected-fifo")
        )

    source3, _sealed3 = candidate_source(tmp_path / "scratch-socket")
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.bind(str(source3 / "unexpected.sock"))
        with pytest.raises(CandidateStoreError, match="UNSAFE_FILESYSTEM_OBJECT"):
            CandidateStore(tmp_path / "ares3").publish(
                source3, (*artifacts(), "unexpected.sock")
            )
    finally:
        sock.close()


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute/path", "a/../../escape"])
def test_unsafe_artifact_paths_fail_closed(tmp_path: Path, unsafe: str):
    source, _sealed = candidate_source(tmp_path / "scratch")
    with pytest.raises(CandidateStoreError, match="UNSAFE_PATH"):
        CandidateStore(tmp_path / "ares").publish(source, (*artifacts(), unsafe))


@pytest.mark.parametrize("extra_name", ["payload/file.txt", "PAYLOAD/FILE.TXT"])
def test_archive_duplicate_and_casefold_collision_fail_closed(
    tmp_path: Path, extra_name: str
):
    source, _sealed = candidate_source(tmp_path / "scratch")
    core_raw = (source / "candidate-core-manifest.json").read_bytes()
    cert_raw = (source / "certification-set-manifest.json").read_bytes()
    core = json.loads(core_raw)
    cert = json.loads(cert_raw)
    archive = tmp_path / "hostile.tar"
    with tarfile.open(archive, "w", format=tarfile.USTAR_FORMAT) as bundle:
        for name, data in (
            ("payload/file.txt", b"sealed-candidate-payload"),
            (extra_name, b"x"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    fd = os.open(archive, os.O_RDONLY | os.O_CLOEXEC)
    try:
        with pytest.raises(CandidateStoreError, match="CUSTODY_CORRUPT"):
            CandidateStore(tmp_path / "ares")._validate_archive_fd(
                fd, core, cert, core_raw, cert_raw
            )
    finally:
        os.close(fd)


def test_destination_symlink_and_gc_surprise_link_fail_closed(tmp_path: Path):
    source, sealed = candidate_source(tmp_path / "scratch")
    store = CandidateStore(tmp_path / "ares")
    store._prepare()
    incoming = store.candidates_root / ".incoming" / "attacker"
    incoming.symlink_to(tmp_path / "outside", target_is_directory=True)
    # Publish chooses an unguessable incoming directory, and retained custody
    # will never traverse a symlink inserted as a candidate component.
    result = store.publish(source, artifacts())
    root = result.candidate_root
    root.rename(store.candidates_root / f"{sealed['sealed_candidate_id']}-saved")
    root.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(
        CandidateStoreError, match="CUSTODY_UNAVAILABLE|UNSAFE_FILESYSTEM_OBJECT"
    ):
        store.verify(sealed["sealed_candidate_id"])


def test_held_candidate_descriptor_cannot_be_redirected_by_component_swap(
    tmp_path: Path, monkeypatch
):
    store, _source, sealed, result = publish(tmp_path)
    original = store._validate_snapshot_fd
    swapped = False

    def swap_then_validate(candidate_fd: int, sealed_id: str):
        nonlocal swapped
        if not swapped:
            swapped = True
            moved = result.candidate_root.with_name(f"{sealed_id}-moved")
            result.candidate_root.rename(moved)
            result.candidate_root.symlink_to(
                tmp_path / "attacker", target_is_directory=True
            )
        return original(candidate_fd, sealed_id)

    monkeypatch.setattr(store, "_validate_snapshot_fd", swap_then_validate)
    snapshot = store.verify(sealed["sealed_candidate_id"])
    # Verification completes against the descriptor opened before the swap,
    # never against the attacker-controlled replacement path.
    assert snapshot["sealed_candidate_id"] == sealed["sealed_candidate_id"]
