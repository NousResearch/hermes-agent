from __future__ import annotations

import copy
import hashlib
import io
import inspect
import multiprocessing
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts.canary import owner_gate_activation_evidence_stager as stager
from scripts.canary import owner_gate_activation_seal as activation
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import passkey_v2_protocol as protocol
from tests.scripts.canary.test_owner_gate_activation_seal import (
    _activation_owner_reauth_receipt,
    _environment,
)
from tests.scripts.canary.test_owner_gate_foundation import NOW, REVISION


def _rename_for_test(source: Path, destination: Path) -> bool:
    if os.path.lexists(destination):
        return False
    os.rename(source, destination)
    return True


@pytest.fixture
def environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, Any]:
    activation_environment = _environment(tmp_path, monkeypatch)
    evidence_root = activation_environment["evidence_root"]
    evidence = {
        name: protocol.decode_canonical_json((evidence_root / name).read_bytes())
        for name in activation.EVIDENCE_NAMES
    }
    frame = stager.build_staging_frame(
        release_revision=REVISION,
        evidence=evidence,
    )
    evidence_root.chmod(0o700)
    shutil.rmtree(evidence_root)

    evidence_base = evidence_root.parent
    receipt_base = evidence_base.parent / "activation-evidence-staging-receipts"
    receipt_base.mkdir(mode=0o700)
    uid = evidence_base.stat().st_uid
    gid = evidence_base.stat().st_gid

    monkeypatch.setattr(stager, "EVIDENCE_BASE", evidence_base)
    monkeypatch.setattr(stager, "STAGING_RECEIPT_BASE", receipt_base)
    monkeypatch.setattr(
        stager,
        "ACTIVATION_SEAL_PATH",
        activation_environment["seal"],
    )
    monkeypatch.setattr(stager, "LOCK_PATH", activation_environment["lock"])
    monkeypatch.setattr(stager, "ROOT_UID", uid)
    monkeypatch.setattr(stager, "ROOT_GID", gid)
    monkeypatch.setattr(
        stager,
        "_installed_release",
        lambda: activation_environment["release"],
    )
    monkeypatch.setattr(stager, "_require_linux_root", lambda: None)
    monkeypatch.setattr(stager, "_rename_noreplace", _rename_for_test)
    monkeypatch.setattr(stager.time, "time", lambda: NOW)
    return {
        **activation_environment,
        "frame": frame,
        "evidence": evidence,
        "evidence_root": evidence_root,
        "evidence_base": evidence_base,
        "staging_receipt": receipt_base / f"{REVISION}.json",
        "receipt_base": receipt_base,
        "scratch_root": evidence_base / f".{REVISION}.staged",
        "receipt_scratch": receipt_base / f".{REVISION}.json.staged",
        "uid": uid,
        "gid": gid,
    }


def _stage(environment: Mapping[str, Any]) -> Mapping[str, Any]:
    return stager.stage_activation_evidence(environment["frame"])


def _assert_false_boundaries(value: Mapping[str, Any]) -> None:
    for name in (
        "activation_seal_present",
        "activation_performed",
        "runtime_started",
        "cloud_mutation_performed",
        "storage_mutation_performed",
        "iam_mutation_performed",
        "caddy_mutation_performed",
    ):
        assert value[name] is False


def _bound_validation_result(
    *,
    release: Path,
    fresh_through_unix: int,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": activation.ACTIVATION_EVIDENCE_VALIDATION_SCHEMA,
        "release_revision": release.name,
        "evidence_file_sha256": {
            name: hashlib.sha256(name.encode("ascii")).hexdigest()
            for name in activation.EVIDENCE_NAMES
        },
        "prospective_activation_seal_sha256": "a" * 64,
        "freshness_enforced": True,
        "fresh_through_unix": fresh_through_unix,
        "activation_seal_published": False,
        "cloud_mutation_performed": False,
    }
    return {
        **unsigned,
        "validation_sha256": protocol.sha256_json(unsigned),
    }


def test_exact_eight_file_bundle_is_staged_without_activation(
    environment: Mapping[str, Any],
) -> None:
    response = _stage(environment)
    assert response["disposition"] == "installed"
    assert response["staging_state"] == "complete"
    assert response["activation_evidence_fresh_through_unix"] == (
        NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
    )
    _assert_false_boundaries(response)
    assert not os.path.lexists(environment["seal"])

    root = environment["evidence_root"]
    root_state = root.lstat()
    assert stat.S_ISDIR(root_state.st_mode)
    assert stat.S_IMODE(root_state.st_mode) == 0o500
    assert root_state.st_uid == environment["uid"]
    assert root_state.st_gid == environment["gid"]
    assert {item.name for item in root.iterdir()} == set(
        activation.EVIDENCE_NAMES
    )
    for name in activation.EVIDENCE_NAMES:
        path = root / name
        state = path.lstat()
        assert stat.S_ISREG(state.st_mode)
        assert not stat.S_ISLNK(state.st_mode)
        assert stat.S_IMODE(state.st_mode) == 0o444
        assert state.st_uid == environment["uid"]
        assert state.st_gid == environment["gid"]
        assert state.st_nlink == 1
        assert path.read_bytes() == foundation.canonical_json_bytes(
            environment["evidence"][name]
        )

    receipt_raw = environment["staging_receipt"].read_bytes()
    receipt = protocol.decode_canonical_json(receipt_raw)
    assert receipt["schema"] == stager.RECEIPT_SCHEMA
    assert receipt["staging_state"] == "complete"
    assert receipt["bundle_sha256"] == environment["frame"]["bundle_sha256"]
    assert receipt["activation_evidence_fresh_through_unix"] == (
        NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
    )
    assert receipt["receipt_sha256"] == protocol.sha256_json({
        name: value
        for name, value in receipt.items()
        if name != "receipt_sha256"
    })
    _assert_false_boundaries(receipt)
    receipt_state = environment["staging_receipt"].lstat()
    assert stat.S_IMODE(receipt_state.st_mode) == 0o444
    assert receipt_state.st_nlink == 1
    assert not environment["scratch_root"].exists()
    assert not environment["receipt_scratch"].exists()


def test_strict_validation_cryptographically_binds_evidence_deadline(
    environment: Mapping[str, Any],
) -> None:
    evidence = copy.deepcopy(environment["evidence"])
    evidence[activation.ACTIVATION_OWNER_REAUTH_NAME] = (
        _activation_owner_reauth_receipt(
            environment["release_key"],
            project_number=environment["project_number"],
            expires_at_unix=NOW + 123,
        )
    )
    frame = stager.build_staging_frame(
        release_revision=REVISION,
        evidence=evidence,
    )
    response = stager.stage_activation_evidence(frame)
    validation = activation.validate_activation_evidence_strict(
        release=environment["release"],
        evidence_root=environment["evidence_root"],
        now_unix=NOW,
    )
    unsigned = {
        name: item
        for name, item in validation.items()
        if name != "validation_sha256"
    }
    assert validation["fresh_through_unix"] == NOW + 123
    assert validation["validation_sha256"] == protocol.sha256_json(unsigned)
    assert response["activation_evidence_fresh_through_unix"] == NOW + 123
    receipt = protocol.decode_canonical_json(
        environment["staging_receipt"].read_bytes()
    )
    assert receipt["activation_evidence_fresh_through_unix"] == NOW + 123
    assert receipt["activation_evidence_validation_sha256"] == validation[
        "validation_sha256"
    ]


@pytest.mark.parametrize(
    "now_unix",
    (
        NOW - 1,
        NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
    ),
    ids=("future_evidence", "stale_evidence"),
)
def test_strict_validation_rejects_future_or_stale_evidence(
    environment: Mapping[str, Any],
    now_unix: int,
) -> None:
    _stage(environment)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        activation.validate_activation_evidence_strict(
            release=environment["release"],
            evidence_root=environment["evidence_root"],
            now_unix=now_unix,
        )


def test_exact_replay_requires_identical_bytes_and_metadata(
    environment: Mapping[str, Any],
) -> None:
    first = _stage(environment)
    evidence_identity = {
        name: (environment["evidence_root"] / name).stat().st_ino
        for name in activation.EVIDENCE_NAMES
    }
    receipt_identity = environment["staging_receipt"].stat().st_ino
    second = _stage(environment)
    assert first["disposition"] == "installed"
    assert second["disposition"] == "exact_replay"
    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert evidence_identity == {
        name: (environment["evidence_root"] / name).stat().st_ino
        for name in activation.EVIDENCE_NAMES
    }
    assert environment["staging_receipt"].stat().st_ino == receipt_identity


def test_receipt_metadata_drift_blocks_exact_replay(
    environment: Mapping[str, Any],
) -> None:
    _stage(environment)
    environment["staging_receipt"].chmod(0o644)
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_receipt_invalid",
    ):
        _stage(environment)
    assert not os.path.lexists(environment["seal"])


def test_existing_bundle_conflict_never_replaces_truth(
    environment: Mapping[str, Any],
) -> None:
    _stage(environment)
    selected_name = activation.INERT_CLOUD_OBSERVATION_NAME
    original = (environment["evidence_root"] / selected_name).read_bytes()
    conflicting = copy.deepcopy(environment["frame"])
    conflicting_document = dict(conflicting["evidence"][selected_name])
    conflicting_document["schema"] = "conflicting-but-canonical.v1"
    conflicting["evidence"][selected_name] = conflicting_document
    conflicting = stager.build_staging_frame(
        release_revision=REVISION,
        evidence=conflicting["evidence"],
    )

    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_file_invalid",
    ):
        stager.stage_activation_evidence(conflicting)
    assert (environment["evidence_root"] / selected_name).read_bytes() == original
    assert not os.path.lexists(environment["seal"])


def test_revision_path_injection_is_not_a_caller_surface(
    environment: Mapping[str, Any],
) -> None:
    with pytest.raises(stager.OwnerGateActivationEvidenceStagingError):
        stager.build_staging_frame(
            release_revision="../caller-selected",
            evidence=environment["evidence"],
        )
    assert list(inspect.signature(stager.stage_activation_evidence).parameters) == [
        "value"
    ]
    assert not environment["evidence_root"].exists()


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_symlink_or_hardlink_evidence_fails_closed(
    environment: Mapping[str, Any],
    tmp_path: Path,
    kind: str,
) -> None:
    _stage(environment)
    root = environment["evidence_root"]
    selected = root / activation.INERT_HOST_OBSERVATION_NAME
    outside = tmp_path / f"outside-{kind}.json"
    if kind == "symlink":
        root.chmod(0o700)
        selected.rename(outside)
        selected.symlink_to(outside)
        root.chmod(0o500)
    else:
        os.link(selected, outside)
        assert selected.stat().st_nlink == 2

    with pytest.raises(stager.OwnerGateActivationEvidenceStagingError):
        _stage(environment)
    assert not os.path.lexists(environment["seal"])


def test_partial_publication_is_preserved_and_requires_reconciliation(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_receipt(*_args: Any, **_kwargs: Any) -> None:
        raise stager.OwnerGateActivationEvidenceStagingError(
            "injected_receipt_failure"
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(stager, "_publish_receipt_noreplace", fail_receipt)
        with pytest.raises(
            stager.OwnerGateActivationEvidenceStagingError,
            match="injected_receipt_failure",
        ):
            _stage(environment)

    assert environment["evidence_root"].exists()
    assert environment["receipt_scratch"].exists()
    assert not environment["staging_receipt"].exists()
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_partial_state",
    ):
        _stage(environment)
    assert environment["evidence_root"].exists()
    assert not os.path.lexists(environment["seal"])


def test_concurrent_first_stage_has_one_winner_and_exact_replays(
    environment: Mapping[str, Any],
) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _index: _stage(environment), range(8)))
    assert sum(item["disposition"] == "installed" for item in responses) == 1
    assert sum(
        item["disposition"] == "exact_replay" for item in responses
    ) == 7
    assert len({item["receipt_sha256"] for item in responses}) == 1
    assert environment["staging_receipt"].stat().st_nlink == 1
    assert not os.path.lexists(environment["seal"])


def test_multiprocess_concurrency_is_serialized_by_the_activation_lock(
    environment: Mapping[str, Any],
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork to inherit the fixed-path test boundary")
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()

    def child_stage() -> None:
        start.wait(timeout=20)
        try:
            results.put(("ok", dict(_stage(environment))))
        except Exception as exc:  # pragma: no cover - diagnostic transport
            results.put(("error", type(exc).__name__, str(exc)))

    processes = [context.Process(target=child_stage) for _ in range(4)]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0

    assert all(item[0] == "ok" for item in observed), observed
    responses = [item[1] for item in observed]
    assert sum(item["disposition"] == "installed" for item in responses) == 1
    assert sum(
        item["disposition"] == "exact_replay" for item in responses
    ) == 3
    assert len({item["receipt_sha256"] for item in responses}) == 1
    assert not os.path.lexists(environment["seal"])


@pytest.mark.parametrize("drift", ("schema", "file_hash", "bundle_hash"))
def test_schema_and_hash_drift_fail_before_staging(
    environment: Mapping[str, Any],
    drift: str,
) -> None:
    frame = copy.deepcopy(environment["frame"])
    if drift == "schema":
        frame["schema"] = "muncho-owner-gate-wrong.v1"
    elif drift == "file_hash":
        frame["evidence_file_sha256"][activation.NETWORK_EVIDENCE_NAME] = (
            "0" * 64
        )
        unsigned = {
            name: value for name, value in frame.items() if name != "bundle_sha256"
        }
        frame["bundle_sha256"] = protocol.sha256_json(unsigned)
    else:
        frame["bundle_sha256"] = "0" * 64

    with pytest.raises(stager.OwnerGateActivationEvidenceStagingError):
        stager.stage_activation_evidence(frame)
    assert not environment["scratch_root"].exists()
    assert not environment["evidence_root"].exists()
    assert not environment["staging_receipt"].exists()


def test_per_file_bound_is_checked_before_any_filesystem_mutation(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = copy.deepcopy(environment["evidence"])
    evidence[activation.NETWORK_EVIDENCE_NAME]["oversized"] = "x" * 128
    monkeypatch.setattr(stager, "MAX_FILE_BYTES", 64)
    with pytest.raises(stager.OwnerGateActivationEvidenceStagingError):
        stager.build_staging_frame(
            release_revision=REVISION,
            evidence=evidence,
        )
    assert not environment["scratch_root"].exists()
    assert not environment["evidence_root"].exists()


@pytest.mark.parametrize(
    ("clock_values", "error_code"),
    (
        ((NOW, NOW - 1), "owner_gate_activation_evidence_staging_time_invalid"),
        (
            (NOW, NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1),
            "owner_gate_activation_evidence_staging_freshness_expired",
        ),
    ),
    ids=("clock_rollback", "clock_forward_beyond_deadline"),
)
def test_single_derivation_fails_closed_on_clock_instability(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    clock_values: tuple[int, int],
    error_code: str,
) -> None:
    clock = iter(clock_values)
    observed_now: list[int] = []

    def validate(**kwargs: Any) -> Mapping[str, Any]:
        observed_now.append(kwargs["now_unix"])
        return _bound_validation_result(
            release=environment["release"],
            fresh_through_unix=(
                NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
            ),
        )

    monkeypatch.setattr(stager, "_current_unix_time", lambda: next(clock))
    monkeypatch.setattr(
        activation,
        "validate_activation_evidence_strict",
        validate,
    )
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match=error_code,
    ):
        stager._validate_fresh_evidence(
            release=environment["release"],
            evidence_root=environment["evidence_root"],
        )
    assert observed_now == [NOW]


def test_validation_crossing_an_integer_second_derives_only_once(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((NOW, NOW + 1))
    observed_now: list[int] = []

    def validate(**kwargs: Any) -> Mapping[str, Any]:
        observed_now.append(kwargs["now_unix"])
        return _bound_validation_result(
            release=environment["release"],
            fresh_through_unix=(
                NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
            ),
        )

    monkeypatch.setattr(stager, "_current_unix_time", lambda: next(clock))
    monkeypatch.setattr(
        activation,
        "validate_activation_evidence_strict",
        validate,
    )
    result = stager._validate_fresh_evidence(
        release=environment["release"],
        evidence_root=environment["evidence_root"],
    )
    assert result["fresh_through_unix"] == (
        NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
    )
    assert observed_now == [NOW]


def test_fresh_through_substitution_breaks_validation_digest(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = dict(_bound_validation_result(
        release=environment["release"],
        fresh_through_unix=NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS,
    ))
    validation["fresh_through_unix"] += 1
    monkeypatch.setattr(stager, "_current_unix_time", lambda: NOW)
    monkeypatch.setattr(
        activation,
        "validate_activation_evidence_strict",
        lambda **_kwargs: validation,
    )
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_validation_invalid",
    ):
        stager._validate_fresh_evidence(
            release=environment["release"],
            evidence_root=environment["evidence_root"],
        )


def test_production_size_slow_validation_stays_available_within_deadline(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production_file = b"x" * stager.MAX_FILE_BYTES
    assert len(production_file) * len(activation.EVIDENCE_NAMES) == (
        stager.MAX_BUNDLE_BYTES
    )
    clock = iter((NOW, NOW + 300))
    observed_now: list[int] = []
    observed_hashes: list[bytes] = []

    def validate(**kwargs: Any) -> Mapping[str, Any]:
        observed_now.append(kwargs["now_unix"])
        for name in activation.EVIDENCE_NAMES:
            observed_hashes.append(
                hashlib.sha256(production_file + name.encode("ascii")).digest()
            )
        return _bound_validation_result(
            release=environment["release"],
            fresh_through_unix=(
                NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
            ),
        )

    monkeypatch.setattr(stager, "_current_unix_time", lambda: next(clock))
    monkeypatch.setattr(
        activation,
        "validate_activation_evidence_strict",
        validate,
    )
    result = stager._validate_fresh_evidence(
        release=environment["release"],
        evidence_root=environment["evidence_root"],
    )
    assert result["fresh_through_unix"] == (
        NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS
    )
    assert observed_now == [NOW]
    assert len(observed_hashes) == len(activation.EVIDENCE_NAMES)
    assert len(set(observed_hashes)) == len(activation.EVIDENCE_NAMES)


def test_strict_freshness_drift_leaves_non_authorizing_partial_scratch(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stager.time,
        "time",
        lambda: NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
    )
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _stage(environment)
    assert environment["scratch_root"].exists()
    assert stat.S_IMODE(environment["scratch_root"].stat().st_mode) == 0o500
    assert not environment["evidence_root"].exists()
    assert not environment["staging_receipt"].exists()
    assert not os.path.lexists(environment["seal"])
    monkeypatch.setattr(stager.time, "time", lambda: NOW)
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_partial_state",
    ):
        _stage(environment)


def test_exact_replay_never_waives_strict_freshness(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(environment)
    monkeypatch.setattr(
        stager.time,
        "time",
        lambda: NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
    )
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _stage(environment)
    assert environment["evidence_root"].exists()
    assert environment["staging_receipt"].exists()
    assert not os.path.lexists(environment["seal"])


def test_slow_receipt_staging_rechecks_clock_before_publication(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": NOW}
    write_exact = stager._write_exact_file

    def slow_write(path: Path, **kwargs: Any) -> None:
        write_exact(path, **kwargs)
        if path == environment["receipt_scratch"]:
            clock["now"] = (
                NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1
            )

    monkeypatch.setattr(stager.time, "time", lambda: clock["now"])
    monkeypatch.setattr(stager, "_write_exact_file", slow_write)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _stage(environment)
    assert environment["scratch_root"].exists()
    assert environment["receipt_scratch"].exists()
    assert not environment["evidence_root"].exists()
    assert not environment["staging_receipt"].exists()
    assert not os.path.lexists(environment["seal"])


def test_slow_replay_read_rechecks_clock_before_acceptance(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stage(environment)
    clock = {"now": NOW}
    read_exact = stager._read_exact_file

    def slow_read(path: Path, **kwargs: Any):
        result = read_exact(path, **kwargs)
        if path == environment["staging_receipt"]:
            clock["now"] = (
                NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1
            )
        return result

    monkeypatch.setattr(stager.time, "time", lambda: clock["now"])
    monkeypatch.setattr(stager, "_read_exact_file", slow_read)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _stage(environment)
    assert environment["evidence_root"].exists()
    assert environment["staging_receipt"].exists()
    assert not os.path.lexists(environment["seal"])


def test_strict_public_validator_never_uses_a_freshness_waiver(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []
    derive = activation._derive_activation

    def tracked(**kwargs: Any):
        observed.append(kwargs["enforce_fresh"])
        return derive(**kwargs)

    monkeypatch.setattr(activation, "_derive_activation", tracked)
    _stage(environment)
    assert observed == [True, True, True, True]
    assert "validate_activation_evidence_strict" in activation.__all__


def test_preexisting_activation_seal_blocks_staging(
    environment: Mapping[str, Any],
) -> None:
    environment["seal"].write_bytes(b"preexisting-activation")
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_activation_present",
    ):
        _stage(environment)
    assert environment["seal"].read_bytes() == b"preexisting-activation"
    assert not environment["scratch_root"].exists()
    assert not environment["evidence_root"].exists()


def test_hardlinked_lock_fails_without_repairing_foreign_metadata(
    environment: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    origin = tmp_path / "foreign-lock-target"
    origin.write_bytes(b"foreign")
    origin.chmod(0o644)
    os.link(origin, environment["lock"])
    assert origin.stat().st_nlink == 2

    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_lock_invalid",
    ):
        _stage(environment)
    assert origin.read_bytes() == b"foreign"
    assert stat.S_IMODE(origin.stat().st_mode) == 0o644
    assert not environment["scratch_root"].exists()


def test_public_surface_has_no_path_or_argv_selected_destination(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert set(stager.__all__) == {
        "ACTIVATION_SEAL_PATH",
        "EVIDENCE_BASE",
        "FRAME_SCHEMA",
        "LOCK_PATH",
        "OwnerGateActivationEvidenceStagingError",
        "RECEIPT_SCHEMA",
        "RESPONSE_SCHEMA",
        "STAGING_RECEIPT_BASE",
        "build_staging_frame",
        "main",
        "stage_activation_evidence",
    }
    assert not any(
        "path" in name
        for name in inspect.signature(stager.stage_activation_evidence).parameters
    )
    assert stager.main(("/caller/selected/path",)) == 2
    assert capsys.readouterr().err == (
        '{"error_code":"owner_gate_activation_evidence_staging_failed",'
        '"ok":false}\n'
    )


def test_installed_release_is_pinned_to_module_and_venv_interpreter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_base = tmp_path / "opt/muncho-owner-gate/releases"
    release = release_base / REVISION
    module = release / "scripts/canary/owner_gate_activation_evidence_stager.py"
    interpreter = release / "venv/bin/python"
    module.parent.mkdir(parents=True)
    interpreter.parent.mkdir(parents=True)
    module.write_bytes(b"# installed\n")
    interpreter.write_bytes(b"python")
    module.chmod(0o444)
    interpreter.chmod(0o755)
    monkeypatch.setattr(stager, "__file__", str(module))
    monkeypatch.setattr(stager, "RELEASE_BASE", release_base)
    monkeypatch.setattr(stager, "ROOT_UID", module.stat().st_uid)
    monkeypatch.setattr(stager, "ROOT_GID", module.stat().st_gid)
    monkeypatch.setattr(stager.sys, "executable", str(interpreter))
    assert stager._installed_release() == release

    current = release_base.parent / "current"
    current.symlink_to(release, target_is_directory=True)
    monkeypatch.setattr(
        stager.sys,
        "executable",
        str(current / "venv/bin/python"),
    )
    assert stager._installed_release() == release

    other = tmp_path / "caller-python"
    other.write_bytes(b"python")
    other.chmod(0o755)
    monkeypatch.setattr(stager.sys, "executable", str(other))
    with pytest.raises(
        stager.OwnerGateActivationEvidenceStagingError,
        match="owner_gate_activation_evidence_staging_release_invalid",
    ):
        stager._installed_release()


def test_entrypoint_accepts_only_one_canonical_stdin_frame(
    environment: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Input:
        def __init__(self, raw: bytes) -> None:
            self.buffer = io.BytesIO(raw)

    raw = foundation.canonical_json_bytes(environment["frame"])
    monkeypatch.setattr(stager.sys, "stdin", Input(raw))
    assert stager.main(()) == 0
    output = capsys.readouterr()
    response = protocol.decode_canonical_json(output.out.encode("utf-8").strip())
    assert response["disposition"] == "installed"
    assert output.err == ""

    monkeypatch.setattr(stager.sys, "stdin", Input(raw + b"\n"))
    assert stager.main(()) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        '{"error_code":"owner_gate_activation_evidence_staging_failed",'
        '"ok":false}\n'
    )


def test_deeply_nested_stdin_is_a_stable_generic_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Input:
        def __init__(self, raw: bytes) -> None:
            self.buffer = io.BytesIO(raw)

    raw = b"[" * 2_000 + b"0" + b"]" * 2_000
    monkeypatch.setattr(stager.sys, "stdin", Input(raw))
    assert stager.main(()) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        '{"error_code":"owner_gate_activation_evidence_staging_failed",'
        '"ok":false}\n'
    )
    assert "Traceback" not in output.err


def test_canonical_recursion_at_stdin_boundary_has_no_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class Input:
        buffer = io.BytesIO(b"{}")

    nested: dict[str, Any] = {}
    cursor = nested
    for _index in range(2_000):
        child: dict[str, Any] = {}
        cursor["nested"] = child
        cursor = child
    monkeypatch.setattr(stager.sys, "stdin", Input())
    monkeypatch.setattr(stager.json, "loads", lambda *_args, **_kwargs: nested)
    assert stager.main(()) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == (
        '{"error_code":"owner_gate_activation_evidence_staging_failed",'
        '"ok":false}\n'
    )
    assert "Traceback" not in output.err


def test_linux_renameat2_publication_is_really_no_replace(
    tmp_path: Path,
) -> None:
    if not stager.sys.platform.startswith("linux"):
        pytest.skip("renameat2 contract is exercised on Linux CI")
    first = tmp_path / "first"
    destination = tmp_path / "destination"
    first.mkdir()
    assert stager._rename_noreplace(first, destination) is True
    assert destination.is_dir()
    assert not first.exists()

    second = tmp_path / "second"
    second.mkdir()
    assert stager._rename_noreplace(second, destination) is False
    assert second.is_dir()
    assert destination.is_dir()
