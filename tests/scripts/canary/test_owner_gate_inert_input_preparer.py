from __future__ import annotations

import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import full_canary_owner_launcher as launcher
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_inert_input_preparer as preparer
from scripts.canary import owner_gate_inert_observation as inert
from scripts.canary import owner_gate_outer_stage0 as outer


REVISION = "a" * 40
SOURCE_TREE = "b" * 40


def _owner_directory(path: Path, *, mode: int = 0o700) -> None:
    path.mkdir()
    os.chown(path, os.geteuid(), os.getegid())
    path.chmod(mode)


def _mode(path: Path) -> int:
    return path.stat(follow_symlinks=False).st_mode & 0o777


def _fixed_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, Any]:
    owner_home = tmp_path / "owner"
    hermes = owner_home / ".hermes"
    trusted = hermes / "trusted"
    release_base = trusted / "owner-gate-release-sources"
    bundle_base = trusted / "owner-gate-offline-bundles"
    source = release_base / REVISION
    bundle = bundle_base / REVISION
    input_root = hermes / "owner-gate-inert-observation-inputs"
    lock_root = hermes / "owner-gate-inert-input-locks"
    for path in (
        owner_home,
        hermes,
        trusted,
        release_base,
        bundle_base,
        source,
    ):
        _owner_directory(path)
    _owner_directory(bundle)

    repository = Path(outer.__file__).resolve().parents[2]
    for destination_relative, source_relative in outer.SOURCE_FILES.items():
        destination = source / source_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((repository / source_relative).read_bytes())
        destination.chmod(outer.FIXED_FILES[destination_relative])

    package_path = bundle / "package-manifest.json"
    package_path.write_bytes(b"{}")
    package_path.chmod(0o444)
    bundle.chmod(0o555)

    monkeypatch.setattr(preparer, "OWNER_HOME", owner_home)
    monkeypatch.setattr(preparer, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(preparer, "RELEASE_SOURCE_BASE", release_base)
    monkeypatch.setattr(preparer, "BUNDLE_SOURCE_BASE", bundle_base)
    monkeypatch.setattr(preparer, "LOCK_ROOT", lock_root)
    monkeypatch.setattr(inert, "OWNER_HOME", owner_home)
    monkeypatch.setattr(inert, "INPUT_ROOT", input_root)

    package_manifest = {
        "release_revision": REVISION,
        "source_tree_oid": SOURCE_TREE,
        "package_sha256": "c" * 64,
        "trust_manifest_sha256": "d" * 64,
        "credential_migration_envelope_sha256": "e" * 64,
    }
    bundle_tree = outer.build_tree_stream_manifest(
        bundle,
        purpose="owner-gate-bundle",
        release_id=REVISION,
    )
    kit_manifest = outer.build_manifest(
        source,
        release_revision=REVISION,
        source_tree_oid=SOURCE_TREE,
    )
    kit_release_id = hashlib.sha256(
        outer.canonical_json_bytes(kit_manifest)
    ).hexdigest()
    prerequisites = (
        source,
        SOURCE_TREE,
        bundle,
        package_manifest,
        bundle_tree,
        hashlib.sha256(package_path.read_bytes()).hexdigest(),
        kit_release_id,
    )
    monkeypatch.setattr(
        preparer,
        "_validated_prerequisites",
        lambda **_kwargs: prerequisites,
    )
    monkeypatch.setattr(
        inert,
        "_load_release_binding",
        lambda *_args, **_kwargs: object(),
    )
    return {
        "owner_home": owner_home,
        "hermes": hermes,
        "trusted": trusted,
        "release_base": release_base,
        "bundle_base": bundle_base,
        "source": source,
        "bundle": bundle,
        "input_root": input_root,
        "lock_root": lock_root,
        "kit_release_id": kit_release_id,
        "bundle_tree": bundle_tree,
    }


def _assert_receipt_hash(receipt: Mapping[str, Any]) -> None:
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert receipt["receipt_sha256"] == foundation.sha256_json(unsigned)


def test_owner_stage0_signature_runner_uses_sealed_cryptography(
    tmp_path: Path,
) -> None:
    private = Ed25519PrivateKey.generate()
    message = b"exact-stage-zero-message"
    paths = {
        "key": tmp_path / "public.der",
        "message": tmp_path / "message.bin",
        "signature": tmp_path / "signature.bin",
    }
    paths["key"].write_bytes(
        preparer.stage0._SPKI_ED25519_PREFIX
        + private.public_key().public_bytes_raw()
    )
    paths["message"].write_bytes(message)
    paths["signature"].write_bytes(private.sign(message))
    for path in paths.values():
        path.chmod(0o600)
    argv = (
        str(preparer.stage0.OPENSSL),
        "pkeyutl",
        "-verify",
        "-pubin",
        "-keyform",
        "DER",
        "-inkey",
        str(paths["key"]),
        "-rawin",
        "-in",
        str(paths["message"]),
        "-sigfile",
        str(paths["signature"]),
    )

    assert preparer._owner_stage0_signature_runner(argv).startswith(
        b"Signature Verified"
    )

    paths["signature"].write_bytes(b"\x00" * 64)
    with pytest.raises(
        preparer.stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_signature_invalid",
    ):
        preparer._owner_stage0_signature_runner(argv)


def test_fixed_bundle_passes_owner_signature_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    bundle_base = trusted / "owner-gate-offline-bundles"
    bundle = bundle_base / REVISION
    for path in (trusted, bundle_base, bundle):
        _owner_directory(path)
    package_path = bundle / "package-manifest.json"
    package_path.write_bytes(b"{}")
    package_path.chmod(0o444)
    bundle.chmod(0o555)
    manifest = {
        "release_revision": REVISION,
        "source_tree_oid": SOURCE_TREE,
        "package_sha256": "c" * 64,
        "credential_migration_envelope_sha256": "d" * 64,
        "trust_manifest_sha256": "e" * 64,
    }
    observed: dict[str, Any] = {}

    def verify_bundle(
        root: Path,
        *,
        expected_uid: int,
        runner: Any,
    ) -> Mapping[str, Any]:
        observed.update({
            "root": root,
            "expected_uid": expected_uid,
            "runner": runner,
        })
        return manifest

    monkeypatch.setattr(preparer, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(preparer, "BUNDLE_SOURCE_BASE", bundle_base)
    monkeypatch.setattr(preparer.stage0, "verify_bundle_stage0", verify_bundle)
    monkeypatch.setattr(
        preparer,
        "_require_exact_bundle_inventory",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        preparer.outer,
        "build_tree_stream_manifest",
        lambda *_args, **_kwargs: {"release_id": REVISION},
    )

    preparer._fixed_bundle(REVISION)

    assert observed == {
        "root": bundle,
        "expected_uid": os.geteuid(),
        "runner": preparer._owner_stage0_signature_runner,
    }


def test_prepare_publishes_exact_three_file_input_and_replays_without_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fixed_environment(tmp_path, monkeypatch)

    first = preparer.prepare_inert_observation_inputs(
        release_revision=REVISION,
        gcloud_executable=object(),  # type: ignore[arg-type]
    )

    release_root = environment["input_root"] / REVISION
    assert first["schema"] == preparer.RECEIPT_SCHEMA
    assert first["state"] == "inputs_prepared"
    assert first["inputs_ready"] is True
    assert first["input_publication_performed"] is True
    assert first["cloud_mutation_performed"] is False
    assert first["service_activation_performed"] is False
    _assert_receipt_hash(first)
    assert set(os.listdir(release_root)) == {
        inert.PINS_NAME,
        inert.KIT_STREAM_NAME,
        inert.BUNDLE_STREAM_NAME,
    }
    assert _mode(environment["input_root"]) == 0o700
    assert _mode(release_root) == 0o700
    identities = {}
    for name in (inert.PINS_NAME, inert.KIT_STREAM_NAME, inert.BUNDLE_STREAM_NAME):
        path = release_root / name
        state = path.stat(follow_symlinks=False)
        assert _mode(path) == 0o400
        assert state.st_uid == os.geteuid()
        assert state.st_gid == os.getegid()
        assert state.st_nlink == 1
        identities[name] = (state.st_dev, state.st_ino)
    loaded = inert._PinnedObservationInputs.load(REVISION)
    assert loaded.pins_raw == foundation.canonical_json_bytes(loaded.pins)
    assert loaded.pins["pins_sha256"] == foundation.sha256_json({
        key: value for key, value in loaded.pins.items() if key != "pins_sha256"
    })
    loaded.assert_stable()

    monkeypatch.setattr(
        outer,
        "materialize_kit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact replay must not rebuild the kit")
        ),
    )
    monkeypatch.setattr(
        outer,
        "write_tree_stream",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact replay must not rebuild streams")
        ),
    )
    replay = preparer.prepare_inert_observation_inputs(
        release_revision=REVISION,
        gcloud_executable=object(),  # type: ignore[arg-type]
    )

    assert replay["state"] == "exact_replay"
    assert replay["inputs_ready"] is True
    assert replay["input_publication_performed"] is False
    assert replay["pins_sha256"] == first["pins_sha256"]
    _assert_receipt_hash(replay)
    assert {
        name: (
            (release_root / name).stat(follow_symlinks=False).st_dev,
            (release_root / name).stat(follow_symlinks=False).st_ino,
        )
        for name in identities
    } == identities


def test_preflight_is_read_only_and_reports_ready_prerequisites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fixed_environment(tmp_path, monkeypatch)

    receipt = preparer.preflight_inert_observation_inputs(
        release_revision=REVISION,
        gcloud_executable=object(),  # type: ignore[arg-type]
    )

    assert receipt["schema"] == preparer.PREFLIGHT_SCHEMA
    assert receipt["state"] == "prerequisites_ready"
    assert receipt["inputs_ready"] is False
    assert receipt["input_publication_performed"] is False
    assert receipt["caller_selected_path_accepted"] is False
    assert not environment["input_root"].exists()
    assert not environment["lock_root"].exists()
    _assert_receipt_hash(receipt)


def test_partial_pending_directory_requires_manual_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fixed_environment(tmp_path, monkeypatch)
    _owner_directory(environment["input_root"])
    pending = environment["input_root"] / f".{REVISION}.pending"
    _owner_directory(pending)
    marker = pending / "partial"
    marker.write_bytes(b"partial\n")
    marker.chmod(0o400)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_input_manual_reconciliation_required",
    ):
        preparer.prepare_inert_observation_inputs(
            release_revision=REVISION,
            gcloud_executable=object(),  # type: ignore[arg-type]
        )

    assert marker.read_bytes() == b"partial\n"
    assert not (environment["input_root"] / REVISION).exists()


def test_post_rename_crash_is_a_valid_exact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fixed_environment(tmp_path, monkeypatch)
    original_fsync = preparer._fsync_directory

    def fail_after_rename(path: Path, *, code: str) -> None:
        if (
            path == environment["input_root"]
            and (environment["input_root"] / REVISION).exists()
        ):
            raise launcher.OwnerLauncherError("injected_post_rename_crash")
        original_fsync(path, code=code)

    monkeypatch.setattr(preparer, "_fsync_directory", fail_after_rename)
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="injected_post_rename_crash",
    ):
        preparer.prepare_inert_observation_inputs(
            release_revision=REVISION,
            gcloud_executable=object(),  # type: ignore[arg-type]
        )
    assert (environment["input_root"] / REVISION).is_dir()
    assert not (environment["input_root"] / f".{REVISION}.pending").exists()

    monkeypatch.setattr(preparer, "_fsync_directory", original_fsync)
    replay = preparer.prepare_inert_observation_inputs(
        release_revision=REVISION,
        gcloud_executable=object(),  # type: ignore[arg-type]
    )
    assert replay["state"] == "exact_replay"
    inert._PinnedObservationInputs.load(REVISION).assert_stable()


def test_pre_rename_crash_preserves_pending_and_blocks_fresh_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _fixed_environment(tmp_path, monkeypatch)
    original_rename = launcher._atomic_rename_no_replace
    monkeypatch.setattr(
        launcher,
        "_atomic_rename_no_replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            launcher.OwnerLauncherError("injected_pre_rename_crash")
        ),
    )

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="injected_pre_rename_crash",
    ):
        preparer.prepare_inert_observation_inputs(
            release_revision=REVISION,
            gcloud_executable=object(),  # type: ignore[arg-type]
        )
    pending = environment["input_root"] / f".{REVISION}.pending"
    assert set(os.listdir(pending)) == {
        inert.PINS_NAME,
        inert.KIT_STREAM_NAME,
        inert.BUNDLE_STREAM_NAME,
    }
    assert not (environment["input_root"] / REVISION).exists()

    monkeypatch.setattr(
        launcher,
        "_atomic_rename_no_replace",
        original_rename,
    )
    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_input_manual_reconciliation_required",
    ):
        preparer.prepare_inert_observation_inputs(
            release_revision=REVISION,
            gcloud_executable=object(),  # type: ignore[arg-type]
        )


def _inventory_bundle(path: Path) -> Mapping[str, Any]:
    _owner_directory(path)
    manifest = {
        "collector_public_key_ids": {
            "network": "1" * 64,
            "cloud": "2" * 64,
            "host": "3" * 64,
        },
        "payloads": [{"release_relative": "bin/tool", "mode": "0555"}],
        "wheels": [{"filename": "dependency.whl"}],
        "bootstrap_pip": {"filename": "pip.whl"},
    }
    files, directories = preparer._expected_bundle_inventory(manifest)
    for directory in sorted(directories, key=lambda value: len(Path(value).parts)):
        destination = path / directory
        destination.mkdir(exist_ok=True)
        destination.chmod(0o700)
    for relative, mode in files.items():
        destination = path / relative
        destination.write_bytes(f"{relative}\n".encode("ascii"))
        destination.chmod(mode)
    for directory in sorted(
        directories,
        key=lambda value: len(Path(value).parts),
        reverse=True,
    ):
        (path / directory).chmod(0o555)
    path.chmod(0o555)
    preparer._require_exact_bundle_inventory(path, manifest)
    return manifest


@pytest.mark.parametrize(
    "attack",
    ("symlink", "hardlink", "file_mode", "directory_mode", "extra"),
)
def test_bundle_inventory_rejects_alias_mode_and_extra_path_attacks(
    attack: str,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    manifest = _inventory_bundle(bundle)
    package = bundle / "package-manifest.json"
    if attack == "symlink":
        bundle.chmod(0o755)
        package.unlink()
        package.symlink_to(bundle / "trust/release-trust.json")
        bundle.chmod(0o555)
    elif attack == "hardlink":
        bundle.chmod(0o755)
        os.link(package, bundle / "package-alias.json")
        bundle.chmod(0o555)
    elif attack == "file_mode":
        package.chmod(0o400)
    elif attack == "directory_mode":
        (bundle / "trust").chmod(0o755)
    elif attack == "extra":
        bundle.chmod(0o755)
        extra = bundle / "unexpected"
        extra.write_bytes(b"unexpected\n")
        extra.chmod(0o444)
        bundle.chmod(0o555)
    else:  # pragma: no cover - closed table above
        raise AssertionError(attack)

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_input_bundle_invalid",
    ):
        preparer._require_exact_bundle_inventory(bundle, manifest)


def test_missing_fixed_release_source_is_exact_and_creates_no_input_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner_home = tmp_path / "owner"
    hermes = owner_home / ".hermes"
    trusted = hermes / "trusted"
    for path in (owner_home, hermes, trusted):
        _owner_directory(path)
    input_root = hermes / "owner-gate-inert-observation-inputs"
    lock_root = hermes / "owner-gate-inert-input-locks"
    release_base = trusted / "owner-gate-release-sources"
    bundle_base = trusted / "owner-gate-offline-bundles"
    monkeypatch.setattr(preparer, "OWNER_HOME", owner_home)
    monkeypatch.setattr(preparer, "TRUSTED_ROOT", trusted)
    monkeypatch.setattr(preparer, "RELEASE_SOURCE_BASE", release_base)
    monkeypatch.setattr(preparer, "BUNDLE_SOURCE_BASE", bundle_base)
    monkeypatch.setattr(preparer, "LOCK_ROOT", lock_root)
    monkeypatch.setattr(inert, "OWNER_HOME", owner_home)
    monkeypatch.setattr(inert, "INPUT_ROOT", input_root)
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda *_args, **_kwargs: None,
    )

    class Runtime:
        def trusted_owner_support_paths(self) -> tuple[str, str]:
            return ("/trusted/gcloud", "/trusted/root")

        def sealed_owner_support_manifest(
            self,
            *,
            expected_release_sha: str,
        ) -> Mapping[str, Any]:
            return {
                "release_sha": expected_release_sha,
                "source_tree_oid": SOURCE_TREE,
            }

    with pytest.raises(
        launcher.OwnerLauncherError,
        match="owner_gate_inert_input_release_source_missing",
    ):
        preparer.preflight_inert_observation_inputs(
            release_revision=REVISION,
            gcloud_executable=Runtime(),
        )
    assert not input_root.exists()
    assert not lock_root.exists()


def test_public_preparer_and_launcher_actions_accept_no_caller_paths() -> None:
    assert tuple(
        inspect.signature(preparer.preflight_inert_observation_inputs).parameters
    ) == ("release_revision", "gcloud_executable")
    assert tuple(
        inspect.signature(preparer.prepare_inert_observation_inputs).parameters
    ) == ("release_revision", "gcloud_executable")

    parser = launcher._cli_parser()
    required = ("--release-sha", REVISION)
    preflight = parser.parse_args((*required, "--preflight-owner-gate-inert-inputs"))
    prepare = parser.parse_args((*required, "--prepare-owner-gate-inert-inputs"))
    activation = parser.parse_args((*required, "--install-owner-gate-activation-seal"))
    assert preflight.preflight_owner_gate_inert_inputs is True
    assert prepare.prepare_owner_gate_inert_inputs is True
    assert activation.install_owner_gate_activation_seal is True
    with pytest.raises(SystemExit):
        parser.parse_args((
            *required,
            "--preflight-owner-gate-inert-inputs",
            "--prepare-owner-gate-inert-inputs",
        ))
