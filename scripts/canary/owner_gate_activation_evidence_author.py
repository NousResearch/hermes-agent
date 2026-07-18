#!/usr/bin/env python3
"""Owner-side authoring and inert staging of post-IAM activation evidence.

The public boundary accepts only the sealed owner capabilities and one release
revision.  It consumes an already-published fresh inert transaction, authors a
fresh post-IAM observation and owner reauthentication receipt, journals the
exact canonical staging frame durably, and invokes only the fixed Stage0
activation-evidence stager.  It never publishes an activation seal and never
mutates IAM, runtime, Caddy, Cloud resources, or storage.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Never

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_activation_evidence_stager as stager
from scripts.canary import owner_gate_activation_seal as activation
from scripts.canary import owner_gate_author_journal as author_journal
from scripts.canary import owner_gate_cloud_observation_author as cloud_author
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_observation as inert
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_production_ingress_observation as ingress
from scripts.canary import owner_gate_stage0_iap as stage0_iap
from scripts.canary import owner_gate_trust_author as trust_author
from scripts.canary import production_cutover_owner_launcher as production_cutover


INTENT_SCHEMA = "muncho-owner-gate-activation-evidence-author-intent.v1"
SUCCESS_SCHEMA = "muncho-owner-gate-activation-evidence-author-success.v1"
FAILURE_SCHEMA = "muncho-owner-gate-activation-evidence-author-failure.v1"
JOURNAL_ROOT = (
    trust_author.AUTHORITY_PARENT / "owner-gate-activation-evidence-authoring"
)
JOURNAL_ARTIFACTS = frozenset({"intent", "success", "failure"})
POST_IAM_PHASE = "post_iam"
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FALSE_FACTS = (
    "activation_seal_present",
    "activation_performed",
    "runtime_started",
    "cloud_mutation_performed",
    "storage_mutation_performed",
    "iam_mutation_performed",
    "caddy_mutation_performed",
)
_STAGE0_RESPONSE_FIELDS = frozenset({
    "schema",
    "release_revision",
    "bundle_sha256",
    "receipt_sha256",
    "activation_evidence_fresh_through_unix",
    "disposition",
    "staging_state",
    *_FALSE_FACTS,
    "response_sha256",
})


def _error(code: str, _cause: BaseException | None = None) -> Never:
    raise launcher.OwnerLauncherError(code) from None


def _canonical(value: Any) -> bytes:
    try:
        return foundation.canonical_json_bytes(value)
    except foundation.OwnerGateFoundationError as exc:
        _error("owner_gate_activation_evidence_author_json_invalid", exc)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clock(now_unix: Callable[[], int]) -> int:
    value = now_unix()
    if type(value) is not int or value <= 0:
        _error("owner_gate_activation_evidence_author_time_invalid")
    return value


class ActivationEvidenceJournal(author_journal.OwnerGateAuthorJournal):
    """A separate owner-only journal with exactly three artifact names."""

    def __init__(
        self,
        *,
        _root: Path = JOURNAL_ROOT,
        _owner_uid: int = author_journal.OWNER_UID,
        _owner_gid: int = author_journal.OWNER_GID,
    ) -> None:
        super().__init__(
            _root=_root,
            _owner_uid=_owner_uid,
            _owner_gid=_owner_gid,
            _artifacts=JOURNAL_ARTIFACTS,
            _maximum_bytes=stager.MAX_FRAME_BYTES + 1024 * 1024,
        )


def _checked_frame(
    value: Mapping[str, Any],
    *,
    release_revision: str,
) -> Mapping[str, Any]:
    try:
        checked = stager.build_staging_frame(
            release_revision=release_revision,
            evidence=value.get("evidence", {}),
        )
    except (
        AttributeError,
        TypeError,
        ValueError,
        stager.OwnerGateActivationEvidenceStagingError,
    ) as exc:
        _error("owner_gate_activation_evidence_author_frame_invalid", exc)
    if _canonical(checked) != _canonical(value):
        _error("owner_gate_activation_evidence_author_frame_invalid")
    return checked


def _intent(
    *,
    release_revision: str,
    frame: Mapping[str, Any],
) -> Mapping[str, Any]:
    frame_raw = _canonical(frame)
    transaction_id = _sha256(frame_raw)
    unsigned = {
        "schema": INTENT_SCHEMA,
        "release_revision": release_revision,
        "transaction_id": transaction_id,
        "staging_frame": dict(frame),
        "staging_frame_sha256": transaction_id,
        "staging_bundle_sha256": frame["bundle_sha256"],
        "byte_identical_replay_required": True,
        "activation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
        "storage_mutation_performed": False,
    }
    return {**unsigned, "intent_sha256": foundation.sha256_json(unsigned)}


def _validate_intent(
    value: Mapping[str, Any],
    *,
    release_revision: str,
) -> tuple[str, Mapping[str, Any]]:
    fields = {
        "schema",
        "release_revision",
        "transaction_id",
        "staging_frame",
        "staging_frame_sha256",
        "staging_bundle_sha256",
        "byte_identical_replay_required",
        "activation_performed",
        "iam_mutation_performed",
        "caddy_mutation_performed",
        "storage_mutation_performed",
        "intent_sha256",
    }
    frame = value.get("staging_frame")
    if not isinstance(frame, Mapping):
        _error("owner_gate_activation_evidence_author_journal_invalid")
    checked = _checked_frame(frame, release_revision=release_revision)
    transaction_id = _sha256(_canonical(checked))
    unsigned = {key: item for key, item in value.items() if key != "intent_sha256"}
    if (
        set(value) != fields
        or value.get("schema") != INTENT_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("transaction_id") != transaction_id
        or value.get("staging_frame_sha256") != transaction_id
        or value.get("staging_bundle_sha256") != checked["bundle_sha256"]
        or value.get("byte_identical_replay_required") is not True
        or any(
            value.get(name) is not False
            for name in (
                "activation_performed",
                "iam_mutation_performed",
                "caddy_mutation_performed",
                "storage_mutation_performed",
            )
        )
        or value.get("intent_sha256") != foundation.sha256_json(unsigned)
    ):
        _error("owner_gate_activation_evidence_author_journal_invalid")
    return transaction_id, checked


def _validate_stage0_response(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    frame: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {key: item for key, item in value.items() if key != "response_sha256"}
    fresh_through = value.get("activation_evidence_fresh_through_unix")
    if (
        set(value) != _STAGE0_RESPONSE_FIELDS
        or value.get("schema") != stager.RESPONSE_SCHEMA
        or value.get("release_revision") != release_revision
        or value.get("bundle_sha256") != frame["bundle_sha256"]
        or _SHA256.fullmatch(str(value.get("receipt_sha256", ""))) is None
        or type(fresh_through) is not int
        or fresh_through <= 0
        or value.get("disposition") not in {"installed", "exact_replay"}
        or value.get("staging_state") != "complete"
        or any(value.get(name) is not False for name in _FALSE_FACTS)
        or value.get("response_sha256") != foundation.sha256_json(unsigned)
    ):
        _error("owner_gate_activation_evidence_author_response_invalid")
    return dict(value)


def _require_stage0_response_current(
    value: Mapping[str, Any],
    *,
    now_unix: int,
) -> None:
    fresh_through = value.get("activation_evidence_fresh_through_unix")
    if (
        type(now_unix) is not int
        or now_unix <= 0
        or type(fresh_through) is not int
        or fresh_through < now_unix
    ):
        _error("owner_gate_activation_evidence_author_staged_evidence_stale")


def _success(
    *,
    release_revision: str,
    transaction_id: str,
    frame: Mapping[str, Any],
    response: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": SUCCESS_SCHEMA,
        "ok": True,
        "release_revision": release_revision,
        "transaction_id": transaction_id,
        "staging_frame_sha256": transaction_id,
        "staging_bundle_sha256": frame["bundle_sha256"],
        "stage0_response": dict(response),
        "stage0_response_file_sha256": _sha256(_canonical(response)),
        "staging_state": "complete",
        "activation_seal_present": False,
        "activation_performed": False,
        "runtime_started": False,
        "cloud_mutation_performed": False,
        "storage_mutation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
    }
    return {**unsigned, "success_sha256": foundation.sha256_json(unsigned)}


def _validate_success(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    transaction_id: str,
    frame: Mapping[str, Any],
) -> Mapping[str, Any]:
    response = value.get("stage0_response")
    if not isinstance(response, Mapping):
        _error("owner_gate_activation_evidence_author_journal_invalid")
    checked_response = _validate_stage0_response(
        response,
        release_revision=release_revision,
        frame=frame,
    )
    expected = _success(
        release_revision=release_revision,
        transaction_id=transaction_id,
        frame=frame,
        response=checked_response,
    )
    if _canonical(value) != _canonical(expected):
        _error("owner_gate_activation_evidence_author_journal_invalid")
    return dict(expected)


def _failure(
    *,
    release_revision: str,
    transaction_id: str,
    frame: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": FAILURE_SCHEMA,
        "ok": False,
        "release_revision": release_revision,
        "transaction_id": transaction_id,
        "staging_frame_sha256": transaction_id,
        "staging_bundle_sha256": frame["bundle_sha256"],
        "state": "stage0_dispatch_incomplete",
        "remote_outcome_known": False,
        "byte_identical_replay_required": True,
        "activation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
        "storage_mutation_performed": False,
    }
    return {**unsigned, "failure_sha256": foundation.sha256_json(unsigned)}


def _validate_failure(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    transaction_id: str,
    frame: Mapping[str, Any],
) -> Mapping[str, Any]:
    expected = _failure(
        release_revision=release_revision,
        transaction_id=transaction_id,
        frame=frame,
    )
    if _canonical(value) != _canonical(expected):
        _error("owner_gate_activation_evidence_author_journal_invalid")
    return dict(expected)


def _assert_inert_frame_binding(
    frame: Mapping[str, Any],
    frozen: inert._FrozenInertEvidence,
) -> None:
    evidence = frame.get("evidence")
    if not isinstance(evidence, Mapping):
        _error("owner_gate_activation_evidence_author_inert_changed")
    for name in inert._EVIDENCE_NAMES:
        if (
            name not in evidence
            or _canonical(evidence[name]) != frozen.evidence_raw[name]
        ):
            _error("owner_gate_activation_evidence_author_inert_changed")


def _assert_post_iam_ready(
    *,
    report: Mapping[str, Any],
    production_ingress_observation: Mapping[str, Any],
    cloud_observation: Mapping[str, Any],
    host_observation: Mapping[str, Any],
    frozen: inert._FrozenInertEvidence,
    now_unix: int,
) -> None:
    try:
        ingress.validate_signed_production_ingress_observation(
            production_ingress_observation,
            phase=POST_IAM_PHASE,
            release_revision=frozen.binding.release_revision,
            plan_sha256=frozen.plan.sha256,
            release_public_key=frozen.binding.release_public_key,
            now_unix=now_unix,
        )
    except ingress.ProductionIngressObservationError as exc:
        _error("owner_gate_activation_evidence_author_post_iam_stale", exc)
    observed_at = report.get("observed_at_unix")
    if type(observed_at) is not int:
        _error("owner_gate_activation_evidence_author_post_iam_not_ready")
    try:
        rebuilt = preflight.build_post_iam_preflight_report(
            plan=frozen.plan,
            production_ingress_observation=production_ingress_observation,
            release_public_key=frozen.binding.release_public_key,
            cloud_observation=cloud_observation,
            host_observation=host_observation,
            cloud_collector_public_key=frozen.cloud_key,
            host_collector_public_key=frozen.host_key,
            now_unix=observed_at,
        )
    except preflight.OwnerGatePreflightError as exc:
        _error("owner_gate_activation_evidence_author_post_iam_not_ready", exc)
    if (
        _canonical(rebuilt) != _canonical(report)
        or report.get("schema") != preflight.POST_IAM_PREFLIGHT_SCHEMA
        or report.get("plan_sha256") != frozen.plan.sha256
        or report.get("release_revision") != frozen.binding.release_revision
        or report.get("effective_permissions_exact_for_fixed_probe_set") is not True
        or report.get("operation_permission_absent") is not True
        or report.get("compute_api_connectivity_verified") is not True
        or report.get("executor_activation_seal_present") is not False
        or report.get("mutation_attempted") is not False
        or report.get("topology_iam_readiness_seal_can_be_installed") is not True
        or report.get("caddy_cutover_performed") is not False
        or report.get("rollback_mode") != "pre_migration_v1_only"
        or type(observed_at) is not int
        or now_unix < observed_at
        or now_unix - observed_at
        > foundation.PREFLIGHT_MAX_AGE_SECONDS
    ):
        _error("owner_gate_activation_evidence_author_post_iam_not_ready")
    for observed in (cloud_observation, host_observation):
        collected = observed.get("collected_at_unix")
        if (
            type(collected) is not int
            or collected > now_unix
            or now_unix - collected > foundation.PREFLIGHT_MAX_AGE_SECONDS
        ):
            _error("owner_gate_activation_evidence_author_post_iam_stale")
    if (
        type(host_observation.get("completed_at_unix")) is not int
        or type(host_observation.get("fresh_through_unix")) is not int
        or host_observation["completed_at_unix"] > now_unix
        or host_observation["fresh_through_unix"] < now_unix
    ):
        _error("owner_gate_activation_evidence_author_post_iam_stale")


def _dispatch_exact_frame(
    *,
    release_revision: str,
    frame: Mapping[str, Any],
    transaction_id: str,
    frozen: inert._FrozenInertEvidence,
    transport: stage0_iap.OwnerGateStage0IapTransport,
    journal: ActivationEvidenceJournal,
    now_unix: Callable[[], int],
) -> Mapping[str, Any]:
    evidence = frame["evidence"]
    _assert_inert_frame_binding(frame, frozen)
    post_report = evidence[activation.POST_IAM_PREFLIGHT_NAME]
    post_ingress = evidence[
        activation.POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME
    ]
    post_cloud = evidence[activation.POST_IAM_CLOUD_OBSERVATION_NAME]
    post_host = evidence[activation.POST_IAM_HOST_OBSERVATION_NAME]
    current = _clock(now_unix)
    frozen.assert_stable(now_unix=current)
    _assert_post_iam_ready(
        report=post_report,
        production_ingress_observation=post_ingress,
        cloud_observation=post_cloud,
        host_observation=post_host,
        frozen=frozen,
        now_unix=current,
    )
    try:
        response = transport.stage_activation_evidence(frame)
        checked_response = _validate_stage0_response(
            response,
            release_revision=release_revision,
            frame=frame,
        )
        final_now = _clock(now_unix)
        _require_stage0_response_current(
            checked_response,
            now_unix=final_now,
        )
        frozen.assert_stable(now_unix=final_now)
        _assert_post_iam_ready(
            report=post_report,
            production_ingress_observation=post_ingress,
            cloud_observation=post_cloud,
            host_observation=post_host,
            frozen=frozen,
            now_unix=final_now,
        )
    except Exception:
        journal.publish(
            release_revision,
            transaction_id,
            "failure",
            _failure(
                release_revision=release_revision,
                transaction_id=transaction_id,
                frame=frame,
            ),
        )
        raise
    success = _success(
        release_revision=release_revision,
        transaction_id=transaction_id,
        frame=frame,
        response=checked_response,
    )
    return journal.publish(
        release_revision,
        transaction_id,
        "success",
        success,
    )


def _stage_post_iam_activation_evidence(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
    reauth_runner: owner_reauth.OwnerReauthRunner,
    now_unix: Callable[[], int] = lambda: int(time.time()),
    journal: ActivationEvidenceJournal | None = None,
) -> Mapping[str, Any]:
    journal = journal or ActivationEvidenceJournal()
    with inert._fresh_inert_evidence_snapshot(
        release_revision=release_revision,
        now_unix=_clock(now_unix),
    ) as frozen:
        transport = stage0_iap.OwnerGateStage0IapTransport(
            release_sha=release_revision,
            owner_identity=owner_identity,
            gcloud_executable=gcloud_executable,
            gcloud_configuration=gcloud_configuration,
            foundation_artifacts=frozen.loaded.raw_artifacts,
        )
        with journal.release_lease(release_revision):
            transactions = journal.list_transactions(release_revision)
            if len(transactions) > 1:
                _error(
                    "owner_gate_activation_evidence_author_manual_reconciliation_required"
                )
            if transactions:
                transaction_id, artifacts = next(iter(transactions.items()))
                if (
                    set(artifacts) - JOURNAL_ARTIFACTS
                    or "intent" not in artifacts
                ):
                    _error(
                        "owner_gate_activation_evidence_author_manual_reconciliation_required"
                    )
                checked_id, frame = _validate_intent(
                    artifacts["intent"],
                    release_revision=release_revision,
                )
                if checked_id != transaction_id:
                    _error(
                        "owner_gate_activation_evidence_author_manual_reconciliation_required"
                    )
                _assert_inert_frame_binding(frame, frozen)
                if "failure" in artifacts:
                    _validate_failure(
                        artifacts["failure"],
                        release_revision=release_revision,
                        transaction_id=transaction_id,
                        frame=frame,
                    )
                if "success" in artifacts:
                    success = _validate_success(
                        artifacts["success"],
                        release_revision=release_revision,
                        transaction_id=transaction_id,
                        frame=frame,
                    )
                    current = _clock(now_unix)
                    frozen.assert_stable(now_unix=current)
                    _assert_post_iam_ready(
                        report=frame["evidence"][
                            activation.POST_IAM_PREFLIGHT_NAME
                        ],
                        production_ingress_observation=frame["evidence"][
                            activation.
                            POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME
                        ],
                        cloud_observation=frame["evidence"][
                            activation.POST_IAM_CLOUD_OBSERVATION_NAME
                        ],
                        host_observation=frame["evidence"][
                            activation.POST_IAM_HOST_OBSERVATION_NAME
                        ],
                        frozen=frozen,
                        now_unix=current,
                    )
                    _require_stage0_response_current(
                        success["stage0_response"],
                        now_unix=current,
                    )
                    return success
                return _dispatch_exact_frame(
                    release_revision=release_revision,
                    frame=frame,
                    transaction_id=transaction_id,
                    frozen=frozen,
                    transport=transport,
                    journal=journal,
                    now_unix=now_unix,
                )

            release_private_key = inert._release_private_key(frozen.binding)
            production_transport = production_cutover.ProductionCutoverTransport(
                owner_identity,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
            )
            post_ingress = ingress.collect_and_sign_production_ingress_observation(
                ingress.OwnerGateProductionIngressTransport(production_transport),
                phase=POST_IAM_PHASE,
                release_revision=release_revision,
                plan_sha256=frozen.plan.sha256,
                release_private_key=release_private_key,
            )
            pair = cloud_author.collect_and_author_bound_pair(
                plan=frozen.plan,
                foundation_apply_chain=frozen.loaded.chain,
                final_network_evidence=frozen.network_evidence,
                final_network_collector_public_key=frozen.network_key,
                production_ingress_observation=post_ingress,
                phase=POST_IAM_PHASE,
                collected_at_unix=None,
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
                owner_identity=owner_identity,
                stage0_transport=transport,
                kit_stream=frozen.inputs.kit_stream,
                bundle_stream=frozen.inputs.bundle_stream,
            )
            post_cloud, post_host = cloud_author.consume_bound_observation_pair(
                pair,
                plan=frozen.plan,
                phase=POST_IAM_PHASE,
            )
            post_time = _clock(now_unix)
            try:
                post_report = preflight.build_post_iam_preflight_report(
                    plan=frozen.plan,
                    production_ingress_observation=post_ingress,
                    release_public_key=frozen.binding.release_public_key,
                    cloud_observation=post_cloud,
                    host_observation=post_host,
                    cloud_collector_public_key=frozen.cloud_key,
                    host_collector_public_key=frozen.host_key,
                    now_unix=post_time,
                )
            except preflight.OwnerGatePreflightError as exc:
                _error("owner_gate_activation_evidence_author_post_iam_not_ready", exc)
            _assert_post_iam_ready(
                report=post_report,
                production_ingress_observation=post_ingress,
                cloud_observation=post_cloud,
                host_observation=post_host,
                frozen=frozen,
                now_unix=post_time,
            )
            reauth = owner_reauth.produce_owner_reauth_receipt(
                runner=reauth_runner,
                private_key=release_private_key,
                now_unix=lambda: _clock(now_unix),
                gcloud_executable=gcloud_executable,
                gcloud_configuration=gcloud_configuration,
                expected_release_revision=release_revision,
            )
            validation_now = _clock(now_unix)
            try:
                checked_reauth = owner_reauth.validate_owner_reauth_receipt(
                    reauth,
                    public_key=frozen.binding.release_public_key,
                    now_unix=validation_now,
                )
            except owner_reauth.OwnerGateOwnerReauthError as exc:
                _error("owner_gate_activation_evidence_author_reauth_invalid", exc)
            if checked_reauth.get("issued_at_unix", 0) <= post_time:
                _error("owner_gate_activation_evidence_author_reauth_invalid")
            evidence = {
                activation.NETWORK_EVIDENCE_NAME: frozen.evidence[
                    inert.NETWORK_EVIDENCE_NAME
                ],
                activation.INERT_PRODUCTION_INGRESS_OBSERVATION_NAME: (
                    frozen.evidence[
                        inert.INERT_PRODUCTION_INGRESS_OBSERVATION_NAME
                    ]
                ),
                activation.INERT_CLOUD_OBSERVATION_NAME: frozen.evidence[
                    inert.INERT_CLOUD_OBSERVATION_NAME
                ],
                activation.INERT_HOST_OBSERVATION_NAME: frozen.evidence[
                    inert.INERT_HOST_OBSERVATION_NAME
                ],
                activation.INERT_PREFLIGHT_NAME: frozen.evidence[
                    inert.INERT_PREFLIGHT_NAME
                ],
                activation.POST_IAM_PRODUCTION_INGRESS_OBSERVATION_NAME: post_ingress,
                activation.POST_IAM_CLOUD_OBSERVATION_NAME: post_cloud,
                activation.POST_IAM_HOST_OBSERVATION_NAME: post_host,
                activation.POST_IAM_PREFLIGHT_NAME: post_report,
                activation.ACTIVATION_OWNER_REAUTH_NAME: checked_reauth,
            }
            if tuple(evidence) != activation.EVIDENCE_NAMES:
                _error("owner_gate_activation_evidence_author_frame_invalid")
            frame = stager.build_staging_frame(
                release_revision=release_revision,
                evidence=evidence,
            )
            intent = _intent(
                release_revision=release_revision,
                frame=frame,
            )
            transaction_id, checked_frame = _validate_intent(
                intent,
                release_revision=release_revision,
            )
            persisted_intent = journal.publish(
                release_revision,
                transaction_id,
                "intent",
                intent,
            )
            persisted_id, persisted_frame = _validate_intent(
                persisted_intent,
                release_revision=release_revision,
            )
            if (
                persisted_id != transaction_id
                or _canonical(persisted_frame) != _canonical(checked_frame)
            ):
                _error(
                    "owner_gate_activation_evidence_author_manual_reconciliation_required"
                )
            return _dispatch_exact_frame(
                release_revision=release_revision,
                frame=persisted_frame,
                transaction_id=transaction_id,
                frozen=frozen,
                transport=transport,
                journal=journal,
                now_unix=now_unix,
            )


def stage_post_iam_activation_evidence(
    *,
    release_revision: str,
    gcloud_executable: launcher.TrustedGcloudExecutable,
    gcloud_configuration: launcher.PinnedGcloudConfiguration,
    owner_identity: launcher.GcloudOwnerAccessToken,
) -> Mapping[str, Any]:
    """Sealed, pathless owner action for evidence staging only."""

    if (
        _REVISION.fullmatch(release_revision or "") is None
        or type(gcloud_executable) is not launcher.TrustedGcloudExecutable
        or type(gcloud_configuration) is not launcher.PinnedGcloudConfiguration
        or type(owner_identity) is not launcher.GcloudOwnerAccessToken
        or owner_identity.gcloud_configuration is not gcloud_configuration
        or getattr(owner_identity, "_gcloud_executable", None)
        is not gcloud_executable
    ):
        _error("owner_gate_activation_evidence_author_capability_invalid")
    launcher.require_trusted_owner_support_activation(
        gcloud_executable,
        release_sha=release_revision,
    )
    launcher.require_local_launcher_provenance(release_revision)
    receipt = _stage_post_iam_activation_evidence(
        release_revision=release_revision,
        gcloud_executable=gcloud_executable,
        gcloud_configuration=gcloud_configuration,
        owner_identity=owner_identity,
        reauth_runner=owner_reauth.SubprocessOwnerReauthRunner(),
    )
    launcher.require_trusted_owner_support_activation(
        gcloud_executable,
        release_sha=release_revision,
    )
    launcher.require_local_launcher_provenance(release_revision)
    return receipt


__all__ = [
    "ActivationEvidenceJournal",
    "FAILURE_SCHEMA",
    "INTENT_SCHEMA",
    "JOURNAL_ROOT",
    "SUCCESS_SCHEMA",
    "stage_post_iam_activation_evidence",
]
