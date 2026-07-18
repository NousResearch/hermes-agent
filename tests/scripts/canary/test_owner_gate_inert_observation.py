from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_activation_seal as activation_seal
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_observation as inert
from scripts.canary import owner_gate_outer_stage0 as outer
from tests.scripts.canary import test_owner_gate_preflight as preflight_fixture


REVISION = "a" * 40
KIT_RELEASE_ID = "b" * 64


def _owner_directory(path: Path) -> None:
    path.mkdir()
    os.chown(path, -1, os.getegid())
    path.chmod(0o700)


def _stream_source(path: Path, payload: bytes) -> Path:
    path.mkdir()
    member = path / "member.bin"
    member.write_bytes(payload)
    member.chmod(0o444)
    return path


def _fixed_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    owner_home = tmp_path / "owner"
    hermes = owner_home / ".hermes"
    input_root = hermes / "owner-gate-inert-observation-inputs"
    release_root = input_root / REVISION
    for path in (owner_home, hermes, input_root, release_root):
        _owner_directory(path)

    monkeypatch.setattr(inert, "OWNER_HOME", owner_home)
    monkeypatch.setattr(inert, "INPUT_ROOT", input_root)

    kit_build = outer.write_tree_stream(
        _stream_source(tmp_path / "kit-source", b"exact outer stage zero\n"),
        release_root / inert.KIT_STREAM_NAME,
        purpose="outer-stage0-kit",
        release_id=KIT_RELEASE_ID,
    )
    bundle_build = outer.write_tree_stream(
        _stream_source(tmp_path / "bundle-source", b"exact owner bundle\n"),
        release_root / inert.BUNDLE_STREAM_NAME,
        purpose="owner-gate-bundle",
        release_id=REVISION,
    )
    unsigned_pins = {
        "schema": inert.INPUT_PINS_SCHEMA,
        "release_revision": REVISION,
        "kit_release_id": KIT_RELEASE_ID,
        "kit_tree_manifest_sha256": kit_build["stream_manifest_sha256"],
        "kit_stream_sha256": hashlib.sha256(
            (release_root / inert.KIT_STREAM_NAME).read_bytes()
        ).hexdigest(),
        "bundle_tree_manifest_sha256": bundle_build["stream_manifest_sha256"],
        "bundle_stream_sha256": hashlib.sha256(
            (release_root / inert.BUNDLE_STREAM_NAME).read_bytes()
        ).hexdigest(),
    }
    pins = {
        **unsigned_pins,
        "pins_sha256": foundation.sha256_json(unsigned_pins),
    }
    pins_path = release_root / inert.PINS_NAME
    pins_path.write_bytes(foundation.canonical_json_bytes(pins))
    pins_path.chmod(0o400)
    return {
        "owner_home": owner_home,
        "hermes": hermes,
        "input_root": input_root,
        "release_root": release_root,
        "pins": pins,
        "pins_path": pins_path,
        "kit_path": release_root / inert.KIT_STREAM_NAME,
        "bundle_path": release_root / inert.BUNDLE_STREAM_NAME,
    }


def _rewrite(path: Path, raw: bytes, *, mode: int = 0o400) -> None:
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(mode)


def test_fixed_release_root_and_mode_0400_pins_load_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _fixed_inputs(tmp_path, monkeypatch)

    loaded = inert._PinnedObservationInputs.load(REVISION)

    assert loaded.release_root == prepared["release_root"]
    assert loaded.pins == prepared["pins"]
    assert Path(loaded.kit_stream.path) == prepared["kit_path"]
    assert Path(loaded.bundle_stream.path) == prepared["bundle_path"]
    assert stat_mode(prepared["input_root"]) == 0o700
    assert stat_mode(prepared["release_root"]) == 0o700
    assert stat_mode(prepared["pins_path"]) == 0o400
    assert stat_mode(prepared["kit_path"]) == 0o400
    assert stat_mode(prepared["bundle_path"]) == 0o400
    loaded.assert_stable()


def stat_mode(path: Path) -> int:
    return path.stat(follow_symlinks=False).st_mode & 0o777


@pytest.mark.parametrize(
    "attack",
    (
        "pins_symlink",
        "pins_hardlink",
        "pins_mode",
        "pins_tamper",
        "kit_symlink",
        "kit_hardlink",
        "kit_mode",
        "kit_tamper",
        "root_mode",
        "release_symlink",
    ),
)
def test_fixed_inputs_reject_alias_mode_and_byte_tampering(
    attack: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _fixed_inputs(tmp_path, monkeypatch)
    pins_path: Path = prepared["pins_path"]
    kit_path: Path = prepared["kit_path"]

    if attack == "pins_symlink":
        target = tmp_path / "pins-target"
        target.write_bytes(pins_path.read_bytes())
        target.chmod(0o400)
        pins_path.unlink()
        pins_path.symlink_to(target)
    elif attack == "pins_hardlink":
        os.link(pins_path, prepared["release_root"] / "pins-alias")
    elif attack == "pins_mode":
        pins_path.chmod(0o600)
    elif attack == "pins_tamper":
        _rewrite(pins_path, pins_path.read_bytes() + b"\n")
    elif attack == "kit_symlink":
        target = tmp_path / "kit-target"
        target.write_bytes(kit_path.read_bytes())
        target.chmod(0o400)
        kit_path.unlink()
        kit_path.symlink_to(target)
    elif attack == "kit_hardlink":
        os.link(kit_path, prepared["release_root"] / "kit-alias")
    elif attack == "kit_mode":
        kit_path.chmod(0o600)
    elif attack == "kit_tamper":
        raw = bytearray(kit_path.read_bytes())
        raw[-1] ^= 1
        _rewrite(kit_path, bytes(raw))
    elif attack == "root_mode":
        prepared["input_root"].chmod(0o755)
    elif attack == "release_symlink":
        release_root: Path = prepared["release_root"]
        moved = tmp_path / "moved-release"
        release_root.rename(moved)
        release_root.symlink_to(moved, target_is_directory=True)
    else:  # pragma: no cover - table is deliberately closed above
        raise AssertionError(attack)

    with pytest.raises(launcher.OwnerLauncherError):
        inert._PinnedObservationInputs.load(REVISION)


@pytest.mark.parametrize("change", ("pins", "kit", "release_inventory"))
def test_loaded_inputs_reject_every_later_identity_change(
    change: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _fixed_inputs(tmp_path, monkeypatch)
    loaded = inert._PinnedObservationInputs.load(REVISION)

    if change == "pins":
        raw = bytearray(prepared["pins_path"].read_bytes())
        raw[-1] = ord("0") if raw[-1] != ord("0") else ord("1")
        _rewrite(prepared["pins_path"], bytes(raw))
    elif change == "kit":
        raw = prepared["kit_path"].read_bytes()
        prepared["kit_path"].unlink()
        prepared["kit_path"].write_bytes(raw)
        prepared["kit_path"].chmod(0o400)
    else:
        extra = prepared["release_root"] / "unexpected"
        extra.write_bytes(b"unexpected\n")
        extra.chmod(0o400)

    with pytest.raises(launcher.OwnerLauncherError):
        loaded.assert_stable()


def test_fixed_inputs_replay_with_identical_pins_and_stream_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fixed_inputs(tmp_path, monkeypatch)

    first = inert._PinnedObservationInputs.load(REVISION)
    second = inert._PinnedObservationInputs.load(REVISION)

    assert first.pins_raw == second.pins_raw
    assert first.pins == second.pins
    assert first.kit_stream.manifest_raw == second.kit_stream.manifest_raw
    assert first.bundle_stream.manifest_raw == second.bundle_stream.manifest_raw
    assert first.kit_stream._fingerprint == second.kit_stream._fingerprint
    assert first.bundle_stream._fingerprint == second.bundle_stream._fingerprint
    first.assert_stable()
    second.assert_stable()


def _fixed_evidence_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    owner_home = tmp_path / "owner"
    hermes = owner_home / ".hermes"
    for path in (owner_home, hermes):
        _owner_directory(path)
    evidence_root = hermes / "owner-gate-inert-observation-evidence"
    monkeypatch.setattr(inert, "OWNER_HOME", owner_home)
    monkeypatch.setattr(inert, "EVIDENCE_ROOT", evidence_root)
    phase_root = inert._evidence_phase_root(REVISION)

    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = preflight_fixture._plan(network_key, cloud_key, host_key)
    cloud_observation = preflight_fixture._cloud(plan, cloud_key, iam=False)
    host_observation = preflight_fixture._host(plan, host_key, iam=False)
    report = inert.preflight.build_preflight_report(
        plan=plan,
        cloud_observation=cloud_observation,
        host_observation=host_observation,
        cloud_collector_public_key=cloud_key.public_key(),
        host_collector_public_key=host_key.public_key(),
        now_unix=preflight_fixture.NOW,
    )
    inputs = SimpleNamespace(pins={"pins_sha256": "d" * 64})
    chain = SimpleNamespace(
        foundation_source_tree_oid="c" * 40,
        pre_foundation_authority_sha256="7" * 64,
        foundation_apply_receipt_sha256="8" * 64,
    )
    loaded = SimpleNamespace(chain=chain)
    package = {"package_sha256": "3" * 64}
    receipt, payloads = inert._build_receipt(
        release_revision=REVISION,
        inputs=inputs,  # type: ignore[arg-type]
        loaded=loaded,  # type: ignore[arg-type]
        package=package,
        plan=plan,
        cloud_observation=cloud_observation,
        host_observation=host_observation,
        report=report,
    )
    return SimpleNamespace(
        phase_root=phase_root,
        inputs=inputs,
        loaded=loaded,
        package=package,
        plan=plan,
        cloud_key=cloud_key,
        host_key=host_key,
        cloud_observation=cloud_observation,
        host_observation=host_observation,
        report=report,
        receipt=receipt,
        payloads=payloads,
    )


def _load_fixed_transaction(fixed: SimpleNamespace, *, now_unix: int):
    return inert._load_evidence_transaction(
        phase_root=fixed.phase_root,
        transaction_name=fixed.receipt["evidence_set_sha256"],
        release_revision=REVISION,
        inputs=fixed.inputs,
        loaded=fixed.loaded,
        package=fixed.package,
        plan=fixed.plan,
        cloud_key=fixed.cloud_key.public_key(),
        host_key=fixed.host_key.public_key(),
        now_unix=now_unix,
    )


def test_evidence_publish_is_exact_durable_activation_named_and_compact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)

    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )

    transaction = fixed.phase_root / fixed.receipt["evidence_set_sha256"]
    assert set(os.listdir(fixed.phase_root)) == {
        fixed.receipt["evidence_set_sha256"]
    }
    assert set(os.listdir(transaction)) == inert._TRANSACTION_NAMES
    assert stat_mode(fixed.phase_root) == 0o700
    assert stat_mode(transaction) == 0o500
    assert {
        inert.INERT_CLOUD_OBSERVATION_NAME,
        inert.INERT_HOST_OBSERVATION_NAME,
        inert.INERT_PREFLIGHT_NAME,
    } == {
        activation_seal.INERT_CLOUD_OBSERVATION_NAME,
        activation_seal.INERT_HOST_OBSERVATION_NAME,
        activation_seal.INERT_PREFLIGHT_NAME,
    }
    for name in inert._TRANSACTION_NAMES:
        path = transaction / name
        assert stat_mode(path) == 0o400
        assert path.stat().st_nlink == 1
        assert path.read_bytes() == fixed.payloads[name]
    persisted, fresh = _load_fixed_transaction(
        fixed,
        now_unix=preflight_fixture.NOW,
    )
    assert fresh is True
    assert persisted == fixed.receipt
    assert not {
        "cloud_observation",
        "host_observation",
        "preflight_report",
    }.intersection(persisted)
    assert len(inert._canonical(persisted)) < launcher._MAX_JSON_LINE_BYTES


def test_fresh_replay_is_exact_and_requires_no_new_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )

    first = inert._find_fresh_replay(
        phase_root=fixed.phase_root,
        release_revision=REVISION,
        inputs=fixed.inputs,
        loaded=fixed.loaded,
        package=fixed.package,
        plan=fixed.plan,
        cloud_key=fixed.cloud_key.public_key(),
        host_key=fixed.host_key.public_key(),
        now_unix=preflight_fixture.NOW,
    )
    second = inert._find_fresh_replay(
        phase_root=fixed.phase_root,
        release_revision=REVISION,
        inputs=fixed.inputs,
        loaded=fixed.loaded,
        package=fixed.package,
        plan=fixed.plan,
        cloud_key=fixed.cloud_key.public_key(),
        host_key=fixed.host_key.public_key(),
        now_unix=preflight_fixture.NOW,
    )

    assert first == second == fixed.receipt
    assert set(os.listdir(fixed.phase_root)) == {
        fixed.receipt["evidence_set_sha256"]
    }


def test_no_replace_collision_preserves_original_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )
    transaction = fixed.phase_root / fixed.receipt["evidence_set_sha256"]
    original = {
        name: (transaction / name).read_bytes()
        for name in inert._TRANSACTION_NAMES
    }

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_manual_reconciliation_required",
    ):
        inert._publish_evidence(
            phase_root=fixed.phase_root,
            receipt=fixed.receipt,
            payloads=fixed.payloads,
        )

    assert {
        name: (transaction / name).read_bytes()
        for name in inert._TRANSACTION_NAMES
    } == original
    assert (fixed.phase_root / inert.PENDING_NAME).is_dir()


def test_stale_valid_history_is_retained_but_not_replayed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )
    stale_at = (
        preflight_fixture.NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1
    )

    replay = inert._find_fresh_replay(
        phase_root=fixed.phase_root,
        release_revision=REVISION,
        inputs=fixed.inputs,
        loaded=fixed.loaded,
        package=fixed.package,
        plan=fixed.plan,
        cloud_key=fixed.cloud_key.public_key(),
        host_key=fixed.host_key.public_key(),
        now_unix=stale_at,
    )

    assert replay is None
    assert (fixed.phase_root / fixed.receipt["evidence_set_sha256"]).is_dir()


@pytest.mark.parametrize(
    "attack",
    (
        "cloud_symlink",
        "host_hardlink",
        "preflight_mode",
        "receipt_tamper",
        "transaction_mode",
        "transaction_substitution",
        "file_owner",
    ),
)
def test_persisted_evidence_rejects_alias_mode_owner_and_substitution(
    attack: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    inert._publish_evidence(
        phase_root=fixed.phase_root,
        receipt=fixed.receipt,
        payloads=fixed.payloads,
    )
    transaction = fixed.phase_root / fixed.receipt["evidence_set_sha256"]
    transaction.chmod(0o700)
    if attack == "cloud_symlink":
        path = transaction / inert.INERT_CLOUD_OBSERVATION_NAME
        target = tmp_path / "cloud-target"
        target.write_bytes(path.read_bytes())
        target.chmod(0o400)
        path.unlink()
        path.symlink_to(target)
    elif attack == "host_hardlink":
        os.link(
            transaction / inert.INERT_HOST_OBSERVATION_NAME,
            transaction / "host-alias",
        )
    elif attack == "preflight_mode":
        (transaction / inert.INERT_PREFLIGHT_NAME).chmod(0o600)
    elif attack == "receipt_tamper":
        path = transaction / inert.RECEIPT_NAME
        _rewrite(path, path.read_bytes() + b"\n")
    elif attack == "transaction_mode":
        pass
    elif attack == "transaction_substitution":
        transaction.rename(fixed.phase_root / ("e" * 64))
    elif attack == "file_owner":
        transaction.chmod(0o500)
        monkeypatch.setattr(inert.os, "geteuid", lambda: os.getuid() + 1)
    else:  # pragma: no cover - table is deliberately closed above
        raise AssertionError(attack)
    if attack not in {"transaction_mode", "transaction_substitution", "file_owner"}:
        transaction.chmod(0o500)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_manual_reconciliation_required",
    ):
        inert._find_fresh_replay(
            phase_root=fixed.phase_root,
            release_revision=REVISION,
            inputs=fixed.inputs,
            loaded=fixed.loaded,
            package=fixed.package,
            plan=fixed.plan,
            cloud_key=fixed.cloud_key.public_key(),
            host_key=fixed.host_key.public_key(),
            now_unix=preflight_fixture.NOW,
        )


@pytest.mark.parametrize("drift", ("pending", "unknown", "partial"))
def test_partial_and_namespace_drift_require_manual_reconciliation(
    drift: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    if drift == "pending":
        _owner_directory(fixed.phase_root / inert.PENDING_NAME)
    elif drift == "unknown":
        _owner_directory(fixed.phase_root / "caller-name")
    else:
        partial = fixed.phase_root / ("f" * 64)
        _owner_directory(partial)
        partial.chmod(0o500)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_manual_reconciliation_required",
    ):
        inert._find_fresh_replay(
            phase_root=fixed.phase_root,
            release_revision=REVISION,
            inputs=fixed.inputs,
            loaded=fixed.loaded,
            package=fixed.package,
            plan=fixed.plan,
            cloud_key=fixed.cloud_key.public_key(),
            host_key=fixed.host_key.public_key(),
            now_unix=preflight_fixture.NOW,
        )


def test_interrupted_publish_leaves_pending_for_manual_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed = _fixed_evidence_store(tmp_path, monkeypatch)
    real_write = inert._write_staged_file
    writes = 0

    def interrupt_second_write(path: Path, raw: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise launcher.OwnerLauncherError("simulated_write_failure")
        real_write(path, raw)

    monkeypatch.setattr(inert, "_write_staged_file", interrupt_second_write)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_manual_reconciliation_required",
    ):
        inert._publish_evidence(
            phase_root=fixed.phase_root,
            receipt=fixed.receipt,
            payloads=fixed.payloads,
        )
    assert (fixed.phase_root / inert.PENDING_NAME).is_dir()

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_manual_reconciliation_required",
    ):
        inert._find_fresh_replay(
            phase_root=fixed.phase_root,
            release_revision=REVISION,
            inputs=fixed.inputs,
            loaded=fixed.loaded,
            package=fixed.package,
            plan=fixed.plan,
            cloud_key=fixed.cloud_key.public_key(),
            host_key=fixed.host_key.public_key(),
            now_unix=preflight_fixture.NOW,
        )


def test_production_orchestration_resamples_time_after_remote_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    cloud_public = cloud_key.public_key()
    host_public = host_key.public_key()
    plan = preflight_fixture._plan(network_key, cloud_key, host_key)
    cloud_observation = preflight_fixture._cloud(plan, cloud_key, iam=False)
    host_observation = preflight_fixture._host(plan, host_key, iam=False)
    pair = object()
    executable = object()
    configuration = object()
    owner_identity = object()
    raw_artifacts = object()
    kit_stream = object()
    bundle_stream = object()
    transport = object()
    phase_root = object()
    package = {"package_sha256": "3" * 64}
    chain = SimpleNamespace(
        foundation_source_tree_oid="a" * 40,
        pre_foundation_authority_sha256="7" * 64,
        foundation_apply_receipt_sha256="8" * 64,
    )
    events: list[str] = []
    published: dict[str, Any] = {}

    class Inputs:
        def __init__(self) -> None:
            self.pins = {"pins_sha256": "d" * 64}
            self.kit_stream = kit_stream
            self.bundle_stream = bundle_stream

        def assert_stable(self) -> None:
            events.append("stable")

    inputs = Inputs()
    loaded = SimpleNamespace(chain=chain, raw_artifacts=raw_artifacts)

    def load_inputs(release_revision: str) -> Inputs:
        assert release_revision == REVISION
        events.append("inputs")
        return inputs

    def load_foundation(release_revision: str) -> SimpleNamespace:
        assert release_revision == REVISION
        events.append("foundation")
        return loaded

    def final_plan(observed_chain: object, observed_bundle: object):
        assert observed_chain is chain
        assert observed_bundle is bundle_stream
        events.append("plan")
        return plan, package

    def build_transport(**kwargs: object) -> object:
        assert kwargs == {
            "release_sha": REVISION,
            "owner_identity": owner_identity,
            "gcloud_executable": executable,
            "gcloud_configuration": configuration,
            "foundation_artifacts": raw_artifacts,
        }
        events.append("transport")
        return transport

    def collect_pair(**kwargs: object) -> object:
        assert kwargs == {
            "plan": plan,
            "foundation_apply_chain": chain,
            "phase": "inert",
            "collected_at_unix": None,
            "gcloud_executable": executable,
            "gcloud_configuration": configuration,
            "owner_identity": owner_identity,
            "stage0_transport": transport,
            "kit_stream": kit_stream,
            "bundle_stream": bundle_stream,
        }
        events.append("collect")
        return pair

    consumes = 0

    def consume_pair(observed: object, **kwargs: object):
        nonlocal consumes
        consumes += 1
        assert consumes == 1
        assert observed is pair
        assert kwargs == {"plan": plan, "phase": "inert"}
        assert events[-1] == "collect"
        events.append("consume")
        return cloud_observation, host_observation

    def collector_key(release_revision: str, *, role: str):
        assert release_revision == REVISION
        events.append(f"key-{role}")
        return {
            "cloud": cloud_public,
            "host": host_public,
        }[role]

    real_preflight = inert.preflight.build_preflight_report

    def build_preflight(**kwargs: object):
        events.append("preflight")
        return real_preflight(**kwargs)

    class EvidenceLease:
        def __enter__(self) -> object:
            events.append("lease")
            return phase_root

        def __exit__(self, *_args: object) -> None:
            events.append("unlock")

    def find_replay(**kwargs: object):
        expected_now = (
            preflight_fixture.NOW
            if published
            else preflight_fixture.NOW - 2
        )
        assert kwargs == {
            "phase_root": phase_root,
            "release_revision": REVISION,
            "inputs": inputs,
            "loaded": loaded,
            "package": package,
            "plan": plan,
            "cloud_key": cloud_public,
            "host_key": host_public,
            "now_unix": expected_now,
        }
        events.append("scan")
        return published.get("receipt")

    def publish_evidence(**kwargs: object) -> None:
        assert kwargs["phase_root"] is phase_root
        published.update(kwargs)
        events.append("publish")

    monkeypatch.setattr(inert._PinnedObservationInputs, "load", load_inputs)
    monkeypatch.setattr(inert, "_load_successful_foundation", load_foundation)
    monkeypatch.setattr(inert, "_final_plan", final_plan)
    monkeypatch.setattr(
        inert.stage0_iap,
        "OwnerGateStage0IapTransport",
        build_transport,
    )
    monkeypatch.setattr(
        inert.cloud_author,
        "collect_and_author_bound_pair",
        collect_pair,
    )
    monkeypatch.setattr(
        inert.cloud_author,
        "consume_bound_observation_pair",
        consume_pair,
    )
    monkeypatch.setattr(inert, "_collector_key", collector_key)
    monkeypatch.setattr(inert.preflight, "build_preflight_report", build_preflight)
    monkeypatch.setattr(inert, "_evidence_lease", lambda _revision: EvidenceLease())
    monkeypatch.setattr(inert, "_find_fresh_replay", find_replay)
    monkeypatch.setattr(inert, "_publish_evidence", publish_evidence)
    clock = iter((
        preflight_fixture.NOW - 2,
        preflight_fixture.NOW,
        preflight_fixture.NOW,
    ))
    monkeypatch.setattr(inert.time, "time", lambda: float(next(clock)))

    receipt = inert._collect_inert_observation(
        release_revision=REVISION,
        gcloud_executable=executable,  # type: ignore[arg-type]
        gcloud_configuration=configuration,  # type: ignore[arg-type]
        owner_identity=owner_identity,  # type: ignore[arg-type]
    )

    assert consumes == 1
    assert events == [
        "inputs",
        "foundation",
        "plan",
        "key-cloud",
        "key-host",
        "lease",
        "scan",
        "transport",
        "collect",
        "consume",
        "preflight",
        "stable",
        "publish",
        "scan",
        "stable",
        "unlock",
    ]
    assert "cloud_observation" not in receipt
    assert "host_observation" not in receipt
    assert "preflight_report" not in receipt
    persisted_payloads = published["payloads"]
    assert inert._decode_document(
        persisted_payloads[inert.INERT_CLOUD_OBSERVATION_NAME],
        code="test",
    ) == cloud_observation
    assert inert._decode_document(
        persisted_payloads[inert.INERT_HOST_OBSERVATION_NAME],
        code="test",
    ) == host_observation
    persisted_preflight = inert._decode_document(
        persisted_payloads[inert.INERT_PREFLIGHT_NAME],
        code="test",
    )
    assert persisted_preflight["schema"] == inert.preflight.PREFLIGHT_SCHEMA
    assert persisted_preflight["mutation_performed"] is False
    assert receipt["cloud_mutation_performed"] is False
    assert receipt["service_activation_performed"] is False
    unsigned_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_sha256"
    }
    assert receipt["receipt_sha256"] == foundation.sha256_json(unsigned_receipt)
    assert inert._canonical(receipt) == foundation.canonical_json_bytes(receipt)
    assert len(inert._canonical(receipt)) < launcher._MAX_JSON_LINE_BYTES

    events.clear()
    replayed = inert._collect_inert_observation(
        release_revision=REVISION,
        gcloud_executable=executable,  # type: ignore[arg-type]
        gcloud_configuration=configuration,  # type: ignore[arg-type]
        owner_identity=owner_identity,  # type: ignore[arg-type]
    )
    assert replayed == receipt
    assert consumes == 1
    assert events == [
        "inputs",
        "foundation",
        "plan",
        "key-cloud",
        "key-host",
        "lease",
        "scan",
        "stable",
        "unlock",
    ]


def test_public_surface_has_no_path_phase_transport_or_injection_capability() -> None:
    signature = inspect.signature(inert.collect_inert_observation)

    assert tuple(signature.parameters) == (
        "release_revision",
        "gcloud_executable",
        "gcloud_configuration",
        "owner_identity",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert not hasattr(inert, "main")
    assert not {
        "path",
        "root",
        "phase",
        "evidence",
        "transport",
        "exchange",
        "popen_factory",
        "timeout_seconds",
    }.intersection(signature.parameters)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_observation_capability_invalid",
    ):
        inert.collect_inert_observation(
            release_revision=REVISION,
            gcloud_executable=object(),  # type: ignore[arg-type]
            gcloud_configuration=object(),  # type: ignore[arg-type]
            owner_identity=object(),  # type: ignore[arg-type]
        )


def test_launcher_parser_exposes_one_mutually_exclusive_pathless_action() -> None:
    parser = launcher._cli_parser()
    parsed = parser.parse_args([
        "--release-sha",
        REVISION,
        "--observe-owner-gate-inert",
    ])

    assert parsed.observe_owner_gate_inert is True
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--release-sha",
            REVISION,
            "--observe-owner-gate-inert",
            "--publish-stopped-release",
        ])
    for forbidden in (
        "--owner-gate-input-root",
        "--owner-gate-evidence",
        "--owner-gate-phase",
        "--owner-gate-transport",
        "--owner-gate-remote-command",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--release-sha",
                REVISION,
                "--observe-owner-gate-inert",
                forbidden,
                "/attacker-controlled",
            ])


def test_public_action_rejects_unsealed_runtime_before_fixed_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = object.__new__(launcher.TrustedGcloudExecutable)
    configuration = object.__new__(launcher.PinnedGcloudConfiguration)
    owner_identity = object.__new__(launcher.GcloudOwnerAccessToken)
    owner_identity._gcloud_executable = executable
    owner_identity._gcloud_configuration = configuration
    reached: list[str] = []

    def reject_unsealed(*_args: object, **_kwargs: object) -> None:
        raise launcher.OwnerLauncherError("trusted_owner_support_path_invalid")

    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        reject_unsealed,
    )
    monkeypatch.setattr(
        inert,
        "_collect_inert_observation",
        lambda **_kwargs: reached.append("inputs"),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="trusted_owner_support_path_invalid",
    ):
        inert.collect_inert_observation(
            release_revision=REVISION,
            gcloud_executable=executable,
            gcloud_configuration=configuration,
            owner_identity=owner_identity,
        )
    assert reached == []


def test_direct_launcher_canonical_bridge_is_exact_and_provenance_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = "c" * 64

    class Provenance:
        def __init__(self, *, module_path: str) -> None:
            assert module_path == str(Path(launcher.__file__).absolute())

        def __call__(self, release_sha: str) -> str:
            assert release_sha == REVISION
            return digest

    monkeypatch.setattr(launcher, "_CANONICAL_LAUNCHER_BRIDGE", None)
    monkeypatch.setattr(launcher, "__name__", "__main__")
    monkeypatch.setitem(__import__("sys").modules, "__main__", launcher)
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda release_sha: digest if release_sha == REVISION else "",
    )
    monkeypatch.setattr(launcher, "LocalLauncherProvenance", Provenance)

    launcher._install_canonical_launcher_bridge(REVISION)

    assert (
        __import__("sys").modules[launcher._CANONICAL_LAUNCHER_MODULE]
        is launcher
    )
    assert launcher._canonical_launcher_bridge_valid(launcher) is True
    assert launcher._canonical_launcher_bridge_valid(object()) is False
