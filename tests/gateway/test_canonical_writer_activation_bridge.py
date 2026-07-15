from __future__ import annotations

import copy
import io
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_activation_bridge as bridge
from gateway.canonical_writer_host_authority import (
    EXTERNAL_IAM_RECEIPT_SCHEMA,
    OWNER_APPROVAL_RECEIPT_SCHEMA,
    ExternalIAMReceipt,
    OwnerApprovalReceipt,
)


REVISION = "a" * 40
NATIVE_PLAN_SHA = "b" * 64
FINAL_PLAN_SHA = "c" * 64
OWNER_SHA = "d" * 64
NOW = 1_800_000_000


def _approval(scope: str, plan_sha: str, nonce: str) -> OwnerApprovalReceipt:
    return OwnerApprovalReceipt.from_mapping({
        "schema": OWNER_APPROVAL_RECEIPT_SCHEMA,
        "scope": scope,
        "plan_sha256": plan_sha,
        "authority_kind": "trusted_root_bootstrap_out_of_band_owner",
        "cryptographic_owner_proof": False,
        "owner_subject_sha256": OWNER_SHA,
        "approval_source_sha256": bridge.PINNED_APPROVAL_SOURCE_SHA256,
        "nonce_sha256": nonce,
        "approved_at_unix": NOW,
        "expires_at_unix": NOW + 900,
    })


def _iam(approval: OwnerApprovalReceipt, collected: int = NOW) -> ExternalIAMReceipt:
    return ExternalIAMReceipt.from_mapping({
        "schema": EXTERNAL_IAM_RECEIPT_SCHEMA,
        "project": "adventico-ai-platform",
        "zone": "europe-west3-a",
        "instance": "muncho-canary-v2-01",
        "service_account": (
            "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
        ),
        "scopes": [
            "https://www.googleapis.com/auth/cloud-platform",
        ],
        "roles": [
            "roles/logging.logWriter",
            "roles/monitoring.metricWriter",
            (
                "projects/adventico-ai-platform/roles/"
                "munchoCanaryCloudSqlReadinessV1"
            ),
        ],
        "permissions": [
            "cloudsql.instances.get",
            "logging.logEntries.create",
            "logging.logEntries.route",
            "monitoring.metricDescriptors.create",
            "monitoring.metricDescriptors.get",
            "monitoring.metricDescriptors.list",
            "monitoring.monitoredResourceDescriptors.get",
            "monitoring.monitoredResourceDescriptors.list",
            "monitoring.timeSeries.create",
        ],
        "foundation_plan_sha256": "1" * 64,
        "host_plan_sha256": "2" * 64,
        "foundation_report_sha256": "3" * 64,
        "host_report_sha256": "4" * 64,
        "source_approval_sha256": approval.sha256,
        "collected_at_unix": collected,
        "expires_at_unix": collected + 1200,
    })


def _plan(action: str, policy_sha: str):
    if action == "stage-native-authority":
        return (
            SimpleNamespace(
                sha256=NATIVE_PLAN_SHA,
                value={
                    "revision": REVISION,
                    "external_iam_policy_sha256": policy_sha,
                },
            ),
            "native_observation",
            REVISION,
        )
    return (
        SimpleNamespace(
            sha256=FINAL_PLAN_SHA,
            revision=REVISION,
            digests=SimpleNamespace(external_iam_policy_sha256=policy_sha),
        ),
        "activation",
        REVISION,
    )


def _frame(
    action: str,
    approval: OwnerApprovalReceipt,
    iam: ExternalIAMReceipt,
    *,
    previous_approval: str | None = None,
    previous_iam: str | None = None,
) -> dict:
    scope = "native_observation" if action == "stage-native-authority" else "activation"
    plan_sha = NATIVE_PLAN_SHA if scope == "native_observation" else FINAL_PLAN_SHA
    unsigned = {
        "schema": bridge.FRAME_SCHEMA,
        "action": action,
        "scope": scope,
        "revision": REVISION,
        "plan_sha256": plan_sha,
        "owner_subject_sha256": OWNER_SHA,
        "approval_source_sha256": bridge.PINNED_APPROVAL_SOURCE_SHA256,
        "owner_approval": approval.to_mapping(),
        "external_iam_receipt": iam.to_mapping(),
        "previous_owner_approval_sha256": previous_approval,
        "previous_external_iam_receipt_sha256": previous_iam,
        "framed_at_unix": NOW,
    }
    return {
        **unsigned,
        "frame_sha256": bridge._sha256(bridge._canonical_bytes(unsigned)),
    }


def test_frame_roundtrip_uses_existing_approval_schema_and_exact_plan(monkeypatch):
    approval = _approval("native_observation", NATIVE_PLAN_SHA, "5" * 64)
    iam = _iam(approval)
    monkeypatch.setattr(bridge, "_load_plan", lambda action: _plan(action, iam.policy_sha256))
    monkeypatch.setattr(bridge.time, "time", lambda: NOW)
    value = _frame("stage-native-authority", approval, iam)

    encoded = bridge.build_frame(value)
    decoded = bridge.read_frame(io.BytesIO(encoded))

    assert decoded == value
    assert decoded["owner_approval"]["schema"] == OWNER_APPROVAL_RECEIPT_SCHEMA
    assert decoded["owner_approval"]["cryptographic_owner_proof"] is False


def test_frame_rejects_unpinned_approval_source(monkeypatch):
    approval = _approval("native_observation", NATIVE_PLAN_SHA, "5" * 64)
    iam = _iam(approval)
    monkeypatch.setattr(bridge, "_load_plan", lambda action: _plan(action, iam.policy_sha256))
    value = _frame("stage-native-authority", approval, iam)
    value["approval_source_sha256"] = "6" * 64
    unsigned = {key: copy.deepcopy(item) for key, item in value.items() if key != "frame_sha256"}
    value["frame_sha256"] = bridge._sha256(bridge._canonical_bytes(unsigned))

    with pytest.raises(PermissionError, match="not pinned"):
        bridge.validate_frame(value, now_unix=NOW)


def test_final_stage_archives_native_generation_and_finishes_stopped(monkeypatch, tmp_path):
    native_approval = _approval("native_observation", NATIVE_PLAN_SHA, "5" * 64)
    native_iam = _iam(native_approval)
    final_approval = _approval("activation", FINAL_PLAN_SHA, "6" * 64)
    final_iam = _iam(final_approval)
    assert native_iam.policy_sha256 == final_iam.policy_sha256
    monkeypatch.setattr(
        bridge,
        "_load_plan",
        lambda action: _plan(action, final_iam.policy_sha256),
    )
    monkeypatch.setattr(bridge.activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(bridge.activation, "_host_activation_lock", nullcontext)
    monkeypatch.setattr(bridge.activation, "_ensure_root_directory", lambda _path: None)
    writes: list[tuple[object, bytes]] = []
    monkeypatch.setattr(
        bridge.activation,
        "_install_exact_bytes",
        lambda path, payload, **_kwargs: writes.append((path, payload)) or True,
    )
    paths = bridge.AuthorityPaths(
        staged_owner_approval=tmp_path / "owner.json",
        staged_external_iam=tmp_path / "iam.json",
        evidence_root=tmp_path / "evidence",
    )
    state = {
        paths.staged_owner_approval: bridge._canonical_bytes(native_approval.to_mapping()),
        paths.staged_external_iam: bridge._canonical_bytes(native_iam.to_mapping()),
    }
    monkeypatch.setattr(bridge, "_read_optional", lambda path: state.get(path))
    monkeypatch.setattr(bridge, "_load_archived_previous", lambda *_args, **_kwargs: None)
    archive = {
        "owner_approval_path": "fixed-owner-archive",
        "owner_approval_file_sha256": native_approval.sha256,
        "external_iam_path": "fixed-iam-archive",
        "external_iam_file_sha256": native_iam.sha256,
    }
    monkeypatch.setattr(bridge, "_archive_previous", lambda *_args, **_kwargs: archive)

    def replace(path, *, expected_previous, payload):
        assert state.get(path) == expected_previous
        state[path] = payload
        return True

    monkeypatch.setattr(bridge, "_atomic_replace_exact", replace)
    stopped_checks: list[bool] = []
    frame = _frame(
        "replace-final-authority",
        final_approval,
        final_iam,
        previous_approval=native_approval.sha256,
        previous_iam=native_iam.sha256,
    )

    receipt = bridge.stage_authority(
        frame,
        paths=paths,
        now_unix=NOW,
        stopped_guard=lambda: stopped_checks.append(True),
    )

    assert receipt["state"] == "authority_staged_services_stopped"
    assert receipt["archive"] == archive
    assert receipt["services_started"] is False
    assert receipt["services_stopped"] is True
    assert len(stopped_checks) == 4
    assert state[paths.staged_owner_approval] == bridge._canonical_bytes(
        final_approval.to_mapping()
    )
    assert state[paths.staged_external_iam] == bridge._canonical_bytes(
        final_iam.to_mapping()
    )
    assert any(str(path).endswith("intent.json") for path, _payload in writes)
    assert any(str(path).endswith("receipt.json") for path, _payload in writes)
