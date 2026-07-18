from __future__ import annotations

import inspect
import io
import os
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast

import pytest

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_activation_evidence_author as author
from scripts.canary import owner_gate_activation_evidence_stager as stager
from scripts.canary import owner_gate_activation_seal as activation
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_observation as inert
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_stage0 as cloud_stage0
from scripts.canary import owner_gate_stage0_iap as stage0_iap


R1 = "1" * 40
R2 = "2" * 40
SHA = "a" * 64


def _evidence(tag: str = "one") -> dict[str, Mapping[str, Any]]:
    return {
        name: {"document": name, "tag": tag}
        for name in activation.EVIDENCE_NAMES
    }


def _journal(tmp_path: Path) -> author.ActivationEvidenceJournal:
    parent = tmp_path / "owner"
    parent.mkdir(mode=0o700)
    os.chown(
        parent,
        os.geteuid(),
        os.getegid(),
        follow_symlinks=False,
    )
    parent.chmod(0o700)
    return author.ActivationEvidenceJournal(
        _root=parent / "activation-evidence",
        _owner_uid=os.geteuid(),
        _owner_gid=os.getegid(),
    )


def _stage0_response(
    frame: Mapping[str, Any],
    *,
    release_revision: str = R1,
    disposition: str = "installed",
    fresh_through_unix: int = 10_000,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": stager.RESPONSE_SCHEMA,
        "release_revision": release_revision,
        "bundle_sha256": frame["bundle_sha256"],
        "receipt_sha256": "b" * 64,
        "activation_evidence_fresh_through_unix": fresh_through_unix,
        "disposition": disposition,
        "staging_state": "complete",
        "activation_seal_present": False,
        "activation_performed": False,
        "runtime_started": False,
        "cloud_mutation_performed": False,
        "storage_mutation_performed": False,
        "iam_mutation_performed": False,
        "caddy_mutation_performed": False,
    }
    return {**unsigned, "response_sha256": foundation.sha256_json(unsigned)}


def _frozen(
    *,
    release_revision: str,
    evidence: Mapping[str, Mapping[str, Any]],
    assert_stable: Any | None = None,
) -> SimpleNamespace:
    inert_evidence = {
        name: evidence[name] for name in inert._EVIDENCE_NAMES
    }
    return SimpleNamespace(
        evidence=inert_evidence,
        evidence_raw={
            name: author._canonical(value)
            for name, value in inert_evidence.items()
        },
        binding=SimpleNamespace(
            release_revision=release_revision,
            release_public_key=object(),
        ),
        loaded=SimpleNamespace(raw_artifacts=object(), chain=object()),
        plan=SimpleNamespace(sha256=SHA),
        inputs=SimpleNamespace(kit_stream=object(), bundle_stream=object()),
        network_evidence=object(),
        network_key=object(),
        cloud_key=object(),
        host_key=object(),
        assert_stable=assert_stable or (lambda **_kwargs: None),
    )


def _install_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    frozen: SimpleNamespace,
) -> None:
    @contextmanager
    def snapshot(*, release_revision: str, now_unix: int):
        assert release_revision == frozen.binding.release_revision
        assert now_unix > 0
        yield frozen

    monkeypatch.setattr(inert, "_fresh_inert_evidence_snapshot", snapshot)
    monkeypatch.setattr(
        stage0_iap,
        "OwnerGateStage0IapTransport",
        lambda **_kwargs: SimpleNamespace(),
    )


def test_new_authoring_persists_exact_intent_before_iap_and_reauths_after_post(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frozen = _frozen(release_revision=R1, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    journal = _journal(tmp_path)
    events: list[tuple[str, int]] = []
    current = iter(range(1000, 1100))
    clock = lambda: next(current)
    post_ingress = {"kind": "post-ingress"}
    post_cloud = {"kind": "post-cloud"}
    post_host = {"kind": "post-host"}

    monkeypatch.setattr(inert, "_release_private_key", lambda _binding: object())
    monkeypatch.setattr(
        author.production_cutover,
        "ProductionCutoverTransport",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        author.ingress,
        "OwnerGateProductionIngressTransport",
        lambda _transport: object(),
    )
    monkeypatch.setattr(
        author.ingress,
        "collect_and_sign_production_ingress_observation",
        lambda *_args, **_kwargs: post_ingress,
    )
    monkeypatch.setattr(
        author.ingress,
        "validate_signed_production_ingress_observation",
        lambda *_args, **_kwargs: post_ingress,
    )
    monkeypatch.setattr(
        author.cloud_author,
        "collect_and_author_bound_pair",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        author.cloud_author,
        "consume_bound_observation_pair",
        lambda *_args, **_kwargs: (post_cloud, post_host),
    )

    def build_post(**kwargs: Any) -> Mapping[str, Any]:
        observed = kwargs["now_unix"]
        events.append(("post", observed))
        return {
            "schema": preflight.POST_IAM_PREFLIGHT_SCHEMA,
            "plan_sha256": SHA,
            "release_revision": R1,
            "effective_permissions_exact_for_fixed_probe_set": True,
            "operation_permission_absent": True,
            "compute_api_connectivity_verified": True,
            "executor_activation_seal_present": False,
            "mutation_attempted": False,
            "topology_iam_readiness_seal_can_be_installed": True,
            "caddy_cutover_performed": False,
            "rollback_mode": "pre_migration_v1_only",
            "observed_at_unix": observed,
        }

    monkeypatch.setattr(preflight, "build_post_iam_preflight_report", build_post)

    def produce_reauth(**kwargs: Any) -> Mapping[str, Any]:
        issued = kwargs["now_unix"]()
        events.append(("reauth", issued))
        return {"issued_at_unix": issued}

    monkeypatch.setattr(
        author.owner_reauth,
        "produce_owner_reauth_receipt",
        produce_reauth,
    )
    monkeypatch.setattr(
        author.owner_reauth,
        "validate_owner_reauth_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(author, "_assert_post_iam_ready", lambda **_kwargs: None)
    dispatched: list[bytes] = []

    def dispatch(**kwargs: Any) -> Mapping[str, Any]:
        frame = kwargs["frame"]
        transaction_id = kwargs["transaction_id"]
        artifacts = journal.list_artifacts(R1, transaction_id)
        assert set(artifacts) == {"intent"}
        assert author._canonical(artifacts["intent"]["staging_frame"]) == (
            author._canonical(frame)
        )
        dispatched.append(author._canonical(frame))
        return {"journaled": True}

    monkeypatch.setattr(author, "_dispatch_exact_frame", dispatch)
    result = author._stage_post_iam_activation_evidence(
        release_revision=R1,
        gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
        gcloud_configuration=cast(launcher.PinnedGcloudConfiguration, object()),
        owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
        reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
        now_unix=clock,
        journal=journal,
    )

    assert result == {"journaled": True}
    assert events[0][0] == "post"
    assert events[1][0] == "reauth"
    assert events[1][1] > events[0][1]
    frame = stager._decode_canonical(dispatched[0])
    assert set(frame["evidence"]) == set(activation.EVIDENCE_NAMES)
    assert frame["evidence"][activation.NETWORK_EVIDENCE_NAME] == evidence[
        activation.NETWORK_EVIDENCE_NAME
    ]


def test_reauth_receipt_not_strictly_after_post_report_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frozen = _frozen(release_revision=R1, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    journal = _journal(tmp_path)
    monkeypatch.setattr(inert, "_release_private_key", lambda _binding: object())
    monkeypatch.setattr(
        author.production_cutover,
        "ProductionCutoverTransport",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        author.ingress,
        "OwnerGateProductionIngressTransport",
        lambda _transport: object(),
    )
    monkeypatch.setattr(
        author.ingress,
        "collect_and_sign_production_ingress_observation",
        lambda *_args, **_kwargs: {"post": True},
    )
    monkeypatch.setattr(
        author.ingress,
        "validate_signed_production_ingress_observation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        author.cloud_author,
        "collect_and_author_bound_pair",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        author.cloud_author,
        "consume_bound_observation_pair",
        lambda *_args, **_kwargs: ({"cloud": True}, {"host": True}),
    )
    monkeypatch.setattr(
        preflight,
        "build_post_iam_preflight_report",
        lambda **kwargs: {
            "schema": preflight.POST_IAM_PREFLIGHT_SCHEMA,
            "plan_sha256": SHA,
            "release_revision": R1,
            "effective_permissions_exact_for_fixed_probe_set": True,
            "operation_permission_absent": True,
            "compute_api_connectivity_verified": True,
            "executor_activation_seal_present": False,
            "mutation_attempted": False,
            "topology_iam_readiness_seal_can_be_installed": True,
            "caddy_cutover_performed": False,
            "rollback_mode": "pre_migration_v1_only",
            "observed_at_unix": kwargs["now_unix"],
        },
    )
    monkeypatch.setattr(
        author.owner_reauth,
        "produce_owner_reauth_receipt",
        lambda **_kwargs: {"issued_at_unix": 1001},
    )
    monkeypatch.setattr(
        author.owner_reauth,
        "validate_owner_reauth_receipt",
        lambda value, **_kwargs: value,
    )
    monkeypatch.setattr(author, "_assert_post_iam_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        author,
        "_dispatch_exact_frame",
        lambda **_kwargs: pytest.fail("IAP must not be reached"),
    )
    current = iter((1000, 1001, 1002))
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_evidence_author_reauth_invalid",
    ):
        author._stage_post_iam_activation_evidence(
            release_revision=R1,
            gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
            gcloud_configuration=cast(
                launcher.PinnedGcloudConfiguration, object()
            ),
            owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
            reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
            now_unix=lambda: next(current),
            journal=journal,
        )


def test_incomplete_intent_retries_byte_identical_even_after_failure_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frame = stager.build_staging_frame(release_revision=R1, evidence=evidence)
    intent = author._intent(release_revision=R1, frame=frame)
    transaction_id = intent["transaction_id"]
    journal = _journal(tmp_path)
    with journal.release_lease(R1):
        journal.publish(R1, transaction_id, "intent", intent)
        journal.publish(
            R1,
            transaction_id,
            "failure",
            author._failure(
                release_revision=R1,
                transaction_id=transaction_id,
                frame=frame,
            ),
        )
    frozen = _frozen(release_revision=R1, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    monkeypatch.setattr(
        author.ingress,
        "collect_and_sign_production_ingress_observation",
        lambda *_args, **_kwargs: pytest.fail("evidence must not be regenerated"),
    )
    attempts: list[bytes] = []

    def dispatch(**kwargs: Any) -> Mapping[str, Any]:
        attempts.append(author._canonical(kwargs["frame"]))
        return {"attempt": len(attempts)}

    monkeypatch.setattr(author, "_dispatch_exact_frame", dispatch)
    for _ in range(2):
        author._stage_post_iam_activation_evidence(
            release_revision=R1,
            gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
            gcloud_configuration=cast(
                launcher.PinnedGcloudConfiguration, object()
            ),
            owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
            reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
            now_unix=lambda: 1000,
            journal=journal,
        )
    assert attempts == [author._canonical(frame), author._canonical(frame)]


@pytest.mark.parametrize("mode", ["stale", "mutated"])
def test_dispatch_refuses_stale_or_mutated_inert_before_iap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    evidence = _evidence()
    frame = stager.build_staging_frame(release_revision=R1, evidence=evidence)

    def stale(**_kwargs: Any) -> None:
        raise launcher.OwnerLauncherError("owner_gate_inert_observation_stale")

    frozen = _frozen(
        release_revision=R1,
        evidence=evidence,
        assert_stable=stale if mode == "stale" else None,
    )
    if mode == "mutated":
        frozen.evidence_raw[inert.NETWORK_EVIDENCE_NAME] = b"{}"
    journal = _journal(tmp_path)
    transport = SimpleNamespace(
        stage_activation_evidence=lambda _frame: pytest.fail(
            "IAP must not be reached"
        )
    )
    monkeypatch.setattr(author, "_assert_post_iam_ready", lambda **_kwargs: None)
    with journal.release_lease(R1), pytest.raises(launcher.OwnerLauncherError):
        author._dispatch_exact_frame(
            release_revision=R1,
            frame=frame,
            transaction_id=author._sha256(author._canonical(frame)),
            frozen=cast(inert._FrozenInertEvidence, frozen),
            transport=cast(stage0_iap.OwnerGateStage0IapTransport, transport),
            journal=journal,
            now_unix=lambda: 1000,
        )


def test_success_or_intent_never_replays_across_mismatched_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frame = stager.build_staging_frame(release_revision=R1, evidence=evidence)
    intent = author._intent(release_revision=R1, frame=frame)
    response = _stage0_response(frame, release_revision=R1)
    success = author._success(
        release_revision=R1,
        transaction_id=intent["transaction_id"],
        frame=frame,
        response=response,
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_evidence_author_response_invalid",
    ):
        author._validate_success(
            success,
            release_revision=R2,
            transaction_id=intent["transaction_id"],
            frame=frame,
        )
    journal = _journal(tmp_path)
    transaction_id = intent["transaction_id"]
    with journal.release_lease(R2):
        journal.publish(R2, transaction_id, "intent", intent)
    frozen = _frozen(release_revision=R2, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    monkeypatch.setattr(
        author,
        "_dispatch_exact_frame",
        lambda **_kwargs: pytest.fail("mismatched R must not dispatch"),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_evidence_author_frame_invalid",
    ):
        author._stage_post_iam_activation_evidence(
            release_revision=R2,
            gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
            gcloud_configuration=cast(launcher.PinnedGcloudConfiguration, object()),
            owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
            reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
            now_unix=lambda: 1000,
            journal=journal,
        )


def test_expired_persisted_success_fails_closed_without_new_frame_or_iap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frame = stager.build_staging_frame(release_revision=R1, evidence=evidence)
    intent = author._intent(release_revision=R1, frame=frame)
    transaction_id = intent["transaction_id"]
    response = _stage0_response(frame, fresh_through_unix=999)
    success = author._success(
        release_revision=R1,
        transaction_id=transaction_id,
        frame=frame,
        response=response,
    )
    journal = _journal(tmp_path)
    with journal.release_lease(R1):
        journal.publish(R1, transaction_id, "intent", intent)
        journal.publish(R1, transaction_id, "success", success)
    frozen = _frozen(release_revision=R1, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    monkeypatch.setattr(author, "_assert_post_iam_ready", lambda **_kwargs: None)
    monkeypatch.setattr(
        author,
        "_dispatch_exact_frame",
        lambda **_kwargs: pytest.fail("expired success must not dispatch"),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_evidence_author_staged_evidence_stale",
    ):
        author._stage_post_iam_activation_evidence(
            release_revision=R1,
            gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
            gcloud_configuration=cast(
                launcher.PinnedGcloudConfiguration, object()
            ),
            owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
            reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
            now_unix=lambda: 1000,
            journal=journal,
        )
    with journal.release_lease(R1):
        transactions = journal.list_transactions(R1)
    assert set(transactions) == {transaction_id}
    assert set(transactions[transaction_id]) == {"intent", "success"}


def test_tampered_failure_artifact_blocks_exact_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    frame = stager.build_staging_frame(release_revision=R1, evidence=evidence)
    intent = author._intent(release_revision=R1, frame=frame)
    transaction_id = intent["transaction_id"]
    invalid_failure = dict(
        author._failure(
            release_revision=R1,
            transaction_id=transaction_id,
            frame=frame,
        )
    )
    invalid_failure["remote_outcome_known"] = True
    unsigned = {
        key: item
        for key, item in invalid_failure.items()
        if key != "failure_sha256"
    }
    invalid_failure["failure_sha256"] = foundation.sha256_json(unsigned)
    journal = _journal(tmp_path)
    with journal.release_lease(R1):
        journal.publish(R1, transaction_id, "intent", intent)
        journal.publish(R1, transaction_id, "failure", invalid_failure)
    frozen = _frozen(release_revision=R1, evidence=evidence)
    _install_snapshot(monkeypatch, frozen)
    monkeypatch.setattr(
        author,
        "_dispatch_exact_frame",
        lambda **_kwargs: pytest.fail("tampered failure must not dispatch"),
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_activation_evidence_author_journal_invalid",
    ):
        author._stage_post_iam_activation_evidence(
            release_revision=R1,
            gcloud_executable=cast(launcher.TrustedGcloudExecutable, object()),
            gcloud_configuration=cast(
                launcher.PinnedGcloudConfiguration, object()
            ),
            owner_identity=cast(launcher.GcloudOwnerAccessToken, object()),
            reauth_runner=cast(author.owner_reauth.OwnerReauthRunner, object()),
            now_unix=lambda: 1000,
            journal=journal,
        )


def test_stage0_uses_one_fixed_packaged_argv_and_rejects_path_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = stager.build_staging_frame(release_revision=R1, evidence=_evidence())
    response = _stage0_response(frame)
    transport = object.__new__(stage0_iap.OwnerGateStage0IapTransport)
    transport._release_sha = R1
    transport._timeout_seconds = 123.0
    captured: dict[str, Any] = {}

    def exchange(
        _self: Any,
        operation: Any,
        input_source: io.BytesIO,
        *,
        maximum_stdout_bytes: int,
    ) -> Any:
        captured["operation"] = operation
        captured["raw"] = input_source.read()
        captured["maximum_stdout_bytes"] = maximum_stdout_bytes
        return stage0_iap._ProcessResult(
            returncode=0,
            stdout=author._canonical(response) + b"\n",
            stderr=b"",
        )

    transport._exchange_fixed_operation = types.MethodType(exchange, transport)
    monkeypatch.setattr(stage0_iap.time, "time", lambda: 1000)
    assert transport.stage_activation_evidence(frame) == response
    release = cloud_stage0.RELEASE_BASE / R1
    assert captured["operation"].root_argv == (
        "/usr/bin/env",
        "-i",
        str(release / "venv/bin/python"),
        "-I",
        "-B",
        str(release / "bin/muncho-owner-gate-stage-activation-evidence"),
    )
    assert captured["operation"].maximum_input_bytes == stager.MAX_FRAME_BYTES
    assert captured["raw"] == author._canonical(frame)
    injected = dict(frame)
    injected["path"] = "/tmp/attacker"
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_stage0_activation_evidence_staging_invalid",
    ):
        transport.stage_activation_evidence(injected)


def test_public_action_and_cli_are_pathless() -> None:
    assert tuple(
        inspect.signature(author.stage_post_iam_activation_evidence).parameters
    ) == (
        "release_revision",
        "gcloud_executable",
        "gcloud_configuration",
        "owner_identity",
    )
    arguments = launcher._cli_parser().parse_args(
        [
            "--release-sha",
            R1,
            "--stage-owner-gate-activation-evidence",
        ]
    )
    assert arguments.stage_owner_gate_activation_evidence is True
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args(
            [
                "--release-sha",
                R1,
                "--stage-owner-gate-activation-evidence",
                "--evidence-path=/tmp/attacker",
            ]
        )
