from __future__ import annotations

import base64
import hashlib
import multiprocessing
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from scripts.canary import owner_gate_bootstrap as bootstrap
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import owner_gate_trust as release_trust


def _secure_test_directory(path: Path, *, mode: int = 0o700) -> None:
    """Create a test-owned directory independent of BSD group inheritance."""

    path.mkdir(parents=True, mode=mode, exist_ok=True)
    path.chmod(mode)
    os.chown(path, os.geteuid(), os.getegid())


def test_executable_target_sha256_accepts_pinned_python_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python3.11"
    target.write_bytes(b"pinned-python-target\n")
    target.chmod(0o755)
    link = tmp_path / "python3"
    link.symlink_to(target.name)

    assert bootstrap._executable_target_sha256(
        link,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    ) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_activation_evidence_staging_receipt_directory_is_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    production = Path(
        "/var/lib/muncho-owner-gate/"
        "activation-evidence-staging-receipts"
    )
    assert (production, 0, 0, 0o700) in (
        bootstrap.IDENTITY_DIRECTORY_REQUIREMENTS
    )

    directory = tmp_path / "activation-evidence-staging-receipts"
    _secure_test_directory(directory)
    asset = tmp_path / "muncho-owner-gate.tmpfiles"
    asset.write_text("fixed test asset\n", encoding="ascii")
    monkeypatch.setattr(bootstrap, "IDENTITIES", ())
    monkeypatch.setattr(
        bootstrap,
        "IDENTITY_DIRECTORY_REQUIREMENTS",
        ((directory, os.geteuid(), os.getegid(), 0o700),),
    )
    monkeypatch.setattr(bootstrap, "_asset", lambda _release, _name: asset)

    receipt = bootstrap.install_identities_and_directories(
        tmp_path,
        runner=lambda _command: b"",
    )
    assert receipt["directories"] == [{
        "path": str(directory),
        "uid": os.geteuid(),
        "gid": os.getegid(),
        "mode": "0700",
    }]
    assert bootstrap._revalidate_committed_phase(
        "install_fixed_identities_and_directories",
        receipt,
        bundle=SimpleNamespace(),  # type: ignore[arg-type]
        layout=bootstrap.PRODUCTION_LAYOUT,
        runner=lambda _command: b"",
        all_evidence={},
    ) == receipt
    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_committed_phase_drift",
    ):
        bootstrap._revalidate_committed_phase(
            "install_fixed_identities_and_directories",
            {**receipt, "directories": []},
            bundle=SimpleNamespace(),  # type: ignore[arg-type]
            layout=bootstrap.PRODUCTION_LAYOUT,
            runner=lambda _command: b"",
            all_evidence={},
        )

    directory.chmod(0o755)
    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_directory_identity_invalid",
    ):
        bootstrap.install_identities_and_directories(
            tmp_path,
            runner=lambda _command: b"",
        )


def _exact_bytes_kill_worker(
    path: str,
    checkpoint: str,
    connection: object,
) -> None:
    target = Path(path)
    _secure_test_directory(target.parent)

    def stop(label: str) -> None:
        if label == checkpoint:
            connection.send(label)  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]

    bootstrap._install_exact_bytes(
        target,
        b"immutable-owner-gate-bytes\n",
        mode=0o400,
        uid=os.geteuid(),
        gid=os.getegid(),
        _checkpoint=stop,
        _write_chunk_bytes=8,
    )


def _receipt_key_kill_worker(
    root: str,
    checkpoint: str,
    connection: object,
) -> None:
    layout = bootstrap.InstallLayout(etc_root=Path(root) / "etc")
    private_path = layout.etc_root / "keys/receipt-signing-key.pem"
    public_path = layout.etc_root / "public/authority-receipt-public.pem"
    _secure_test_directory(private_path.parent)
    _secure_test_directory(public_path.parent)
    intent = {
        "schema": "muncho-owner-gate-receipt-key-phase-intent.v1",
        "targets": [
            {
                "path": str(private_path),
                "created_by_transaction": True,
                "preserve_on_rollback": True,
            },
            {
                "path": str(public_path),
                "created_by_transaction": True,
                "preserve_on_rollback": True,
            },
        ],
    }

    def stop(label: str) -> None:
        if label == checkpoint:
            connection.send(label)  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]

    bootstrap.generate_or_verify_receipt_key(
        layout=layout,
        transaction_intent=intent,
        _checkpoint=stop,
        _expected_uid=os.geteuid(),
        _expected_gid=os.getegid(),
    )


def _release_rename_kill_worker(root: str, connection: object) -> None:
    release_base = Path(root) / "releases"
    bundle = _bundle(Path(root))
    staging = release_base / f".{bundle.revision}.bootstrap"
    final = release_base / bundle.revision
    intent = {
        "schema": "muncho-owner-gate-release-phase-intent.v1",
        "staging_path": str(staging),
        "final_path": str(final),
        "created_by_transaction": True,
    }

    def stop(label: str) -> None:
        if label == "release_renamed":
            connection.send(label)  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]

    bootstrap.seal_and_publish_release(
        staging,
        bundle,
        layout=bootstrap.InstallLayout(release_base=release_base),
        transaction_intent=intent,
        _checkpoint=stop,
        _expected_uid=os.geteuid(),
        _expected_gid=os.getegid(),
    )


def _run_transaction_fixture(
    journal: Path,
    effect: Path,
    *,
    checkpoint: str | None = None,
    connection: object | None = None,
) -> dict:
    bundle = _bundle(journal.parent)
    context: dict = {}

    def first_intent() -> dict:
        if os.path.lexists(effect):
            raise RuntimeError("fixture target was not fresh")
        return {
            "schema": "bootstrap-test-file-intent.v1",
            "path": str(effect),
            "created_by_transaction": True,
        }

    builders = {
        phase: (
            first_intent
            if index == 0
            else lambda selected=phase: {
                "schema": "bootstrap-test-noop-intent.v1",
                "phase": selected,
            }
        )
        for index, phase in enumerate(bootstrap.INSTALL_PHASES)
    }

    def first_handler() -> dict:
        if checkpoint == "after_intent":
            assert connection is not None
            connection.send("after_intent")  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]
        physical = bootstrap._install_exact_bytes(
            effect,
            b"transaction-owned\n",
            mode=0o400,
            uid=os.geteuid(),
            gid=os.getegid(),
        )
        if checkpoint == "after_effect":
            assert connection is not None
            connection.send("after_effect")  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]
        return {
            **physical,
            "created": context["active_phase_intent"][
                "created_by_transaction"
            ],
            "created_by_transaction": context["active_phase_intent"][
                "created_by_transaction"
            ],
        }

    handlers = {
        phase: (
            first_handler
            if index == 0
            else lambda selected=phase: {"phase_completed": selected}
        )
        for index, phase in enumerate(bootstrap.INSTALL_PHASES)
    }
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }

    return dict(
        bootstrap.run_install_transaction(
            bundle=bundle,
            journal_path=journal,
            handlers=handlers,
            revalidators=revalidators,
            expected_uid=os.geteuid(),
            transaction_context=context,
            intent_builders=builders,
        )
    )


def _transaction_kill_worker(
    journal: str,
    effect: str,
    checkpoint: str,
    connection: object,
) -> None:
    _run_transaction_fixture(
        Path(journal),
        Path(effect),
        checkpoint=checkpoint,
        connection=connection,
    )


def _database_fixture(root: Path) -> tuple[
    bootstrap.VerifiedBundle,
    bootstrap.InstallLayout,
    dict,
    dict,
]:
    state_root = root / "state"
    etc_root = root / "etc"
    for path in (
        state_root / "authority",
        state_root / "executor",
        etc_root / "public",
    ):
        _secure_test_directory(path)
    public_path = etc_root / "public/authority-receipt-public.pem"
    public_raw = public_path.read_bytes()
    public_key = serialization.load_pem_public_key(public_raw)
    assert hasattr(public_key, "public_bytes_raw")
    migration = {
        "credential_id_b64url": base64.urlsafe_b64encode(
            b"synthetic-public-credential-id"
        )
        .rstrip(b"=")
        .decode("ascii"),
        "public_key_cose_b64url": base64.urlsafe_b64encode(
            b"synthetic-public-cose-key"
        )
        .rstrip(b"=")
        .decode("ascii"),
        "expected_user_handle_b64url": base64.urlsafe_b64encode(
            b"synthetic-user-handle"
        )
        .rstrip(b"=")
        .decode("ascii"),
        "envelope_sha256": "e" * 64,
        "collected_at_unix": 1_700_000_000,
        "initial_sign_count": 0,
        "initial_credential_backed_up": True,
    }
    bundle = bootstrap.VerifiedBundle(
        root=root,
        manifest={
            "release_revision": "a" * 40,
            "package_sha256": "b" * 64,
        },
        authority={},
        migration=migration,
    )
    layout = bootstrap.InstallLayout(
        release_base=root / "releases",
        current_link=root / "current",
        etc_root=etc_root,
        state_root=state_root,
        run_root=root / "run",
        systemd_root=root / "systemd",
        sysusers_root=root / "sysusers",
        tmpfiles_root=root / "tmpfiles",
        sudoers_root=root / "sudoers",
        python=Path("/usr/bin/python3"),
        os_release=root / "os-release",
    )
    authority_path = state_root / "authority/passkey-v2.sqlite3"
    executor_path = state_root / "executor/execution-v2.sqlite3"
    intent = {
        "schema": "muncho-owner-gate-databases-phase-intent.v1",
        "targets": [
            {
                "path": str(authority_path),
                "stage_path": str(
                    authority_path.with_name(
                        f".{authority_path.name}.fixture.stage"
                    )
                ),
                "created_by_transaction": True,
                "preserve_on_rollback": True,
            },
            {
                "path": str(executor_path),
                "stage_path": str(
                    executor_path.with_name(
                        f".{executor_path.name}.fixture.stage"
                    )
                ),
                "created_by_transaction": True,
                "preserve_on_rollback": True,
            },
        ],
        "credential_imported_by_transaction": True,
    }
    return (
        bundle,
        layout,
        {
            "public_key_path": str(public_path),
            "public_key_id": hashlib.sha256(
                public_key.public_bytes_raw()
            ).hexdigest(),
        },
        intent,
    )


def _database_kill_worker(
    root: str,
    checkpoint: str,
    connection: object,
) -> None:
    bundle, layout, key_receipt, intent = _database_fixture(Path(root))

    def stop(label: str) -> None:
        if label == checkpoint:
            connection.send(label)  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]

    bootstrap.bootstrap_and_verify_databases(
        bundle,
        layout=layout,
        key_receipt=key_receipt,
        now_unix=1_700_000_001,
        require_root=False,
        authority_uid=os.geteuid(),
        authority_gid=os.getegid(),
        executor_uid=os.geteuid(),
        executor_gid=os.getegid(),
        transaction_intent=intent,
        _checkpoint=stop,
    )


def _rollback_fixture(root: Path) -> tuple[
    bootstrap.VerifiedBundle,
    bootstrap.InstallLayout,
]:
    revision = "a" * 40
    layout = bootstrap.InstallLayout(
        release_base=root / "releases",
        current_link=root / "current",
        etc_root=root / "etc",
        state_root=root / "state",
        run_root=root / "run",
        systemd_root=root / "systemd",
        sysusers_root=root / "sysusers",
        tmpfiles_root=root / "tmpfiles",
        sudoers_root=root / "sudoers",
        python=Path("/usr/bin/python3"),
        os_release=root / "os-release",
    )
    bundle = bootstrap.VerifiedBundle(
        root=root / "bundle",
        manifest={
            "release_revision": revision,
            "package_sha256": "b" * 64,
            "payloads": [],
        },
        authority={},
        migration={},
    )
    return bundle, layout


def _prepare_rollback_fixture(root: Path) -> tuple[dict, list[dict]]:
    bundle, layout = _rollback_fixture(root)
    receipts = layout.state_root / "bootstrap-receipts"
    _secure_test_directory(receipts)
    for path in (
        layout.etc_root / "keys",
        layout.etc_root / "public",
        layout.state_root / "authority",
        layout.state_root / "executor",
        layout.systemd_root,
    ):
        _secure_test_directory(path)
    private_key = Ed25519PrivateKey.generate()
    private_raw = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path = layout.etc_root / "keys/receipt-signing-key.pem"
    public_path = layout.etc_root / "public/authority-receipt-public.pem"
    bootstrap._install_exact_bytes(
        private_path,
        private_raw,
        mode=0o400,
        uid=os.geteuid(),
        gid=os.getegid(),
    )
    bootstrap._install_exact_bytes(
        public_path,
        public_raw,
        mode=0o444,
        uid=os.geteuid(),
        gid=os.getegid(),
    )
    targets: list[dict] = []
    for path, raw in (
        (layout.etc_root / "executor.json", b"{}\n"),
        (
            layout.etc_root / bootstrap.EXECUTOR_HOSTS_FILENAME,
            bootstrap.COMPUTE_API_HOSTS_LINE,
        ),
        (layout.systemd_root / "muncho-test.service", b"[Service]\n"),
    ):
        physical = bootstrap._install_exact_bytes(
            path,
            raw,
            mode=0o444,
            uid=os.geteuid(),
            gid=os.getegid(),
        )
        targets.append(
            {
                **physical,
                "created": True,
                "created_by_transaction": True,
            }
        )
    authority_database = layout.state_root / "authority/passkey-v2.sqlite3"
    executor_database = layout.state_root / "executor/execution-v2.sqlite3"
    authority_database.write_bytes(b"preserved-authority")
    executor_database.write_bytes(b"preserved-executor")
    release = layout.release_base / bundle.revision
    _secure_test_directory(release)
    install_receipt = receipts / f"install-{bundle.revision}.json"
    install_receipt.write_bytes(b"preserved-install-receipt")

    key_evidence = {
        "schema": "muncho-owner-gate-authority-receipt-key.v1",
        "private_key_path": str(private_path),
        "public_key_path": str(public_path),
        "public_key_sha256": hashlib.sha256(public_raw).hexdigest(),
        "public_key_id": hashlib.sha256(
            private_key.public_key().public_bytes_raw()
        ).hexdigest(),
        "generated_on_target": True,
        "created": True,
        "created_by_transaction": True,
    }
    evidence = {
        bootstrap.INSTALL_PHASES[0]: {"phase": 0},
        bootstrap.INSTALL_PHASES[1]: {"phase": 1},
        bootstrap.INSTALL_PHASES[2]: key_evidence,
        bootstrap.INSTALL_PHASES[3]: {
            "schema": "muncho-owner-gate-installed-system-files.v1",
            "files": targets,
            "executor_hosts": {},
            "systemd_units_enabled": [],
            "current_release_selected": False,
            "activation_seal_created": False,
        },
        bootstrap.INSTALL_PHASES[4]: {
            "schema": "muncho-owner-gate-canonical-databases-bootstrap.v1"
        },
        bootstrap.INSTALL_PHASES[5]: {
            "release_path": str(release),
        },
        bootstrap.INSTALL_PHASES[6]: {
            "receipt_path": str(install_receipt),
        },
    }
    context: dict = {}
    handlers = {
        phase: lambda selected=phase: dict(evidence[selected])
        for phase in bootstrap.INSTALL_PHASES
    }
    revalidators = {
        phase: lambda item: dict(item)
        for phase in bootstrap.INSTALL_PHASES
    }
    intents = {}
    for phase in bootstrap.INSTALL_PHASES:
        if phase == bootstrap.INSTALL_PHASES[3]:
            intents[phase] = lambda: {
                "schema": "muncho-owner-gate-system-files-phase-intent.v1",
                "targets": [
                    {
                        "path": item["path"],
                        "created_by_transaction": True,
                        "reversible": True,
                    }
                    for item in targets
                ],
            }
        else:
            intents[phase] = lambda selected=phase: {
                "schema": "rollback-fixture-noop.v1",
                "phase": selected,
            }
    transaction = bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=(
            receipts / f"transaction-{bundle.revision}.json"
        ),
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
        transaction_context=context,
        intent_builders=intents,
        started_at_unix=1_700_000_000,
    )
    return transaction, targets


def _rollback_kill_worker(
    root: str,
    checkpoint: str,
    connection: object,
) -> None:
    fixture_root = Path(root)
    bundle, layout = _rollback_fixture(fixture_root)
    bootstrap.foundation.MUTATION_ENABLE_SEAL = fixture_root / "no-seal"
    bootstrap._revalidate_committed_phase = (  # type: ignore[assignment]
        lambda _phase, stored, **_kwargs: dict(stored)
    )

    def stop(label: str) -> None:
        if label == checkpoint:
            connection.send(label)  # type: ignore[attr-defined]
            connection.recv()  # type: ignore[attr-defined]

    bootstrap.rollback_inert_install(
        bundle.root,
        layout=layout,
        _checkpoint=stop,
        _verified_bundle=bundle,
        _expected_uid=os.geteuid(),
        _expected_gid=os.getegid(),
        _require_root=False,
    )


def _bundle(tmp_path: Path) -> bootstrap.VerifiedBundle:
    return bootstrap.VerifiedBundle(
        root=tmp_path,
        manifest={
            "release_revision": "a" * 40,
            "package_sha256": "b" * 64,
            "payloads": [],
        },
        authority={},
        migration={},
    )


def test_install_receipt_binds_exact_foundation_chain() -> None:
    bundle = bootstrap.VerifiedBundle(
        root=Path("/bundle"),
        manifest={
            "release_revision": "a" * 40,
            "package_sha256": "b" * 64,
            "source_tree_oid": "c" * 40,
        },
        authority={
            "pre_foundation_authority_sha256": "d" * 64,
            "foundation_apply_receipt_sha256": "e" * 64,
            "project_ancestry_evidence_sha256": "f" * 64,
            "project_ancestry_chain_sha256": "0" * 64,
            "resource_ancestor_chain": ["organizations/123456789012"],
        },
        migration={},
    )
    evidence = {
        phase: {"phase": phase}
        for phase in bootstrap.INSTALL_PHASES[:-1]
    }
    evidence["generate_or_verify_authority_receipt_key"] = {
        "public_key_sha256": "1" * 64,
        "public_key_id": "2" * 64,
    }
    evidence["bootstrap_and_verify_canonical_databases"] = {
        "credential_id_sha256": "3" * 64,
    }
    evidence["install_root_owned_configuration_units_firewall_and_hosts"] = {
        "executor_hosts": {"receipt_sha256": "4" * 64},
    }
    evidence["seal_and_publish_immutable_release"] = {
        "release_tree_sha256": "5" * 64,
    }

    unsigned = bootstrap._build_install_receipt_unsigned(
        bundle,
        transaction_prefix_sha256="6" * 64,
        phase_evidence=evidence,
        layout=bootstrap.PRODUCTION_LAYOUT,
        now_unix=1_800_000_000,
    )

    assert unsigned["pre_foundation_authority_sha256"] == "d" * 64
    assert unsigned["foundation_apply_receipt_sha256"] == "e" * 64
    assert unsigned["project_ancestry_evidence_sha256"] == "f" * 64
    assert unsigned["project_ancestry_chain_sha256"] == "0" * 64
    assert unsigned["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    assert unsigned["activation_performed"] is False
    assert unsigned["cloud_mutation_performed"] is False


def test_executor_hosts_file_is_exact_idempotent_and_does_not_touch_global_hosts(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[3]
    layout = bootstrap.InstallLayout(etc_root=tmp_path / "etc/muncho-owner-gate")
    global_hosts = tmp_path / "etc/hosts"
    global_hosts.parent.mkdir(parents=True)
    original_global_hosts = b"127.0.0.1 localhost\n10.80.3.2 owner-gate\n"
    global_hosts.write_bytes(original_global_hosts)

    first_file, first_receipt = bootstrap._install_executor_hosts_file(
        repository,
        layout=layout,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    second_file, second_receipt = bootstrap._install_executor_hosts_file(
        repository,
        layout=layout,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )

    path = layout.etc_root / bootstrap.EXECUTOR_HOSTS_FILENAME
    state = path.lstat()
    assert path.read_bytes() == bootstrap.COMPUTE_API_HOSTS_LINE
    assert path.read_bytes().splitlines() == [
        b"199.36.153.8 compute.googleapis.com",
        b"199.36.153.8 cloudresourcemanager.googleapis.com",
        b"199.36.153.8 iam.googleapis.com",
    ]
    assert not path.is_symlink()
    assert state.st_nlink == 1
    assert state.st_uid == os.geteuid()
    assert state.st_gid == os.getegid()
    assert state.st_mode & 0o777 == 0o444
    assert first_file == second_file
    assert first_receipt == second_receipt
    assert first_file["created"] is True
    assert first_receipt["global_etc_hosts_mutated"] is False
    assert first_receipt["bind_read_only"] == (
        "/etc/muncho-owner-gate/compute-api-hosts:/etc/hosts"
    )
    assert global_hosts.read_bytes() == original_global_hosts


@pytest.mark.parametrize("kind", ("symlink", "hardlink", "wrong_mode", "drift"))
def test_executor_hosts_file_rejects_unsafe_existing_target(
    tmp_path: Path,
    kind: str,
) -> None:
    repository = Path(__file__).parents[3]
    layout = bootstrap.InstallLayout(etc_root=tmp_path / "etc/muncho-owner-gate")
    path = layout.etc_root / bootstrap.EXECUTOR_HOSTS_FILENAME
    path.parent.mkdir(parents=True)
    source = tmp_path / "source"
    source.write_bytes(bootstrap.COMPUTE_API_HOSTS_LINE)
    source.chmod(0o444)
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    else:
        path.write_bytes(
            b"199.36.153.9 compute.googleapis.com\n"
            if kind == "drift"
            else bootstrap.COMPUTE_API_HOSTS_LINE
        )
        path.chmod(0o644 if kind == "wrong_mode" else 0o444)

    with pytest.raises(bootstrap.OwnerGateBootstrapError):
        bootstrap._install_executor_hosts_file(
            repository,
            layout=layout,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )


def test_executor_hosts_rollback_removes_only_exact_managed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[3]
    layout = bootstrap.InstallLayout(etc_root=tmp_path / "etc/muncho-owner-gate")
    file_evidence, receipt = bootstrap._install_executor_hosts_file(
        repository,
        layout=layout,
        expected_uid=os.geteuid(),
        expected_gid=os.getegid(),
    )
    # Production receipts are root:root; translate only the local ownership
    # fields so this non-root test exercises the exact removal boundary.
    local_receipt = {
        **receipt,
        "uid": 0,
        "gid": 0,
    }
    local_unsigned = {
        key: value
        for key, value in local_receipt.items()
        if key != "receipt_sha256"
    }
    local_receipt["receipt_sha256"] = bootstrap.foundation.sha256_json(
        local_unsigned
    )
    local_file = {**file_evidence, "uid": 0, "gid": 0}
    unrelated = layout.etc_root / "unrelated.conf"
    unrelated.write_bytes(b"preserve me\n")
    original_remove = bootstrap._remove_exact_installed_file

    def remove_as_local(evidence, *, allowed_roots):
        return original_remove(
            {**evidence, "uid": os.geteuid(), "gid": os.getegid()},
            allowed_roots=allowed_roots,
        )

    # The production helper fixes uid/gid at zero.  Patch only filesystem
    # ownership checks for this local test; receipt validation remains exact.
    original_validate = bootstrap._validate_executor_hosts_receipt

    def validate_as_local(value, *, path):
        translated = {
            **value,
            "uid": os.geteuid(),
            "gid": os.getegid(),
        }
        unsigned = {
            key: item
            for key, item in translated.items()
            if key != "receipt_sha256"
        }
        translated["receipt_sha256"] = bootstrap.foundation.sha256_json(unsigned)
        original_validate(
            translated,
            path=path,
            expected_uid=os.geteuid(),
            expected_gid=os.getegid(),
        )
        return dict(value)

    monkeypatch.setattr(
        bootstrap,
        "_validate_executor_hosts_receipt",
        validate_as_local,
    )
    monkeypatch.setattr(
        bootstrap,
        "_remove_exact_installed_file",
        remove_as_local,
    )
    rollback = bootstrap._rollback_executor_hosts_file(
        {"executor_hosts": local_receipt, "files": [local_file]},
        layout=layout,
    )

    assert rollback["removed"] is True
    assert rollback["global_etc_hosts_mutated"] is False
    assert not (layout.etc_root / bootstrap.EXECUTOR_HOSTS_FILENAME).exists()
    assert unrelated.read_bytes() == b"preserve me\n"


def test_stage0_openssl_verifier_uses_exact_ed25519_rawin_command(
    tmp_path: Path,
) -> None:
    private = Ed25519PrivateKey.generate()
    public = private.public_key().public_bytes_raw()
    message = b"exact canonical release manifest"
    signature = private.sign(message)
    observed: list[tuple[str, ...]] = []

    def runner(argv: tuple[str, ...]) -> bytes:
        observed.append(argv)
        assert argv[:7] == (
            "/usr/bin/openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-keyform",
            "DER",
            "-inkey",
        )
        assert argv[8:] == (
            "-rawin",
            "-in",
            argv[10],
            "-sigfile",
            argv[12],
        )
        key_der = Path(argv[7]).read_bytes()
        assert key_der == stage0._SPKI_ED25519_PREFIX + public
        private.public_key().verify(
            Path(argv[12]).read_bytes(),
            Path(argv[10]).read_bytes(),
        )
        return b"Signature Verified Successfully\n"

    stage0.verify_ed25519_with_openssl(
        public_key=public,
        signature=signature,
        message=message,
        runner=runner,
        temporary_root=tmp_path,
    )

    assert len(observed) == 1


def test_bootstrap_web_config_requires_exact_owner_binding() -> None:
    valid = {
        "schema": "muncho-owner-gate-web-config.v1",
        "listen_host": "0.0.0.0",
        "listen_port": bootstrap.foundation.WEB_LISTEN_PORT,
        "origin": "https://auth.lomliev.com",
        "rp_id": "lomliev.com",
        "owner_discord_user_id": bootstrap.OWNER_DISCORD_USER_ID,
        "authority_socket": str(bootstrap.foundation.PASSKEY_AUTHORITY_SOCKET),
    }
    assert bootstrap._validate_web_config_asset(valid) == valid

    for changed in (
        {key: value for key, value in valid.items() if key != "owner_discord_user_id"},
        {**valid, "owner_discord_user_id": "999999999999999999"},
        {**valid, "unexpected": True},
    ):
        with pytest.raises(
            bootstrap.OwnerGateBootstrapError,
            match="owner_gate_bootstrap_web_config_invalid",
        ):
            bootstrap._validate_web_config_asset(changed)


def test_stage0_fails_closed_until_reviewed_release_anchor_is_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage0, "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256", "")

    with pytest.raises(
        stage0.OwnerGateStage0Error,
        match="owner_gate_stage0_trust_anchor_unconfigured",
    ):
        stage0.verify_bundle_stage0(tmp_path, expected_uid=os.geteuid())


def test_stage0_and_runtime_share_one_reviewed_release_trust_anchor() -> None:
    assert (
        stage0.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
        == release_trust.PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256
    )


def test_verify_bundle_decodes_direct_iam_against_signed_foundation_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    release_revision = "a" * 40
    foundation_revision = "f" * 40
    foundation_tree_oid = "e" * 40
    direct_raw = b"opaque-direct-iam-authority"
    migration_raw = bootstrap.foundation.canonical_json_bytes({"fixture": True})
    collector_raw = {
        name: Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        for name in ("network", "cloud", "host")
    }
    collector_ids = {
        name: hashlib.sha256(raw).hexdigest()
        for name, raw in collector_raw.items()
    }
    authority = {
        "release_revision": release_revision,
        "foundation_source_revision": foundation_revision,
        "foundation_source_tree_oid": foundation_tree_oid,
        "collector_public_key_ids": collector_ids,
        "credential_migration_envelope_sha256": hashlib.sha256(
            migration_raw
        ).hexdigest(),
        "direct_iam_identity_authority_sha256": hashlib.sha256(
            direct_raw
        ).hexdigest(),
        "pre_foundation_authority_sha256": "1" * 64,
        "foundation_apply_receipt_sha256": "2" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
    }
    manifest = {
        "release_revision": release_revision,
        "foundation_source_revision": foundation_revision,
        "foundation_source_tree_oid": foundation_tree_oid,
        "payloads": [],
        "wheels": [],
        "direct_iam_identity_authority_sha256": authority[
            "direct_iam_identity_authority_sha256"
        ],
        "pre_foundation_authority_sha256": authority[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": authority[
            "foundation_apply_receipt_sha256"
        ],
        "resource_ancestor_chain": authority["resource_ancestor_chain"],
    }
    manifest_raw = bootstrap.foundation.canonical_json_bytes(manifest)
    by_name = {
        "package-manifest.json": manifest_raw,
        "credential.json": migration_raw,
        "direct-iam-identity-authority.json": direct_raw,
        **{
            f"{name}-observation-attestation.pub": raw
            for name, raw in collector_raw.items()
        },
    }
    monkeypatch.setattr(
        bootstrap.trust,
        "load_pinned_release_trust",
        lambda **_kwargs: authority,
    )
    monkeypatch.setattr(
        bootstrap.package,
        "validate_authorized_manifest",
        lambda _manifest, *, authority: _manifest,
    )
    monkeypatch.setattr(
        bootstrap,
        "_read_regular",
        lambda path, **_kwargs: by_name[path.name],
    )
    monkeypatch.setattr(
        bootstrap,
        "validate_migration",
        lambda value, **_kwargs: value,
    )
    decoded_against: list[str | None] = []

    def decode_direct(raw: bytes, *, release_revision: str | None = None):
        assert raw == direct_raw
        decoded_against.append(release_revision)
        return {
            "release_revision": foundation_revision,
            "pre_foundation_authority_sha256": "1" * 64,
            "foundation_apply_receipt_sha256": "2" * 64,
            "resource_ancestor_chain": ["organizations/123456789012"],
        }

    monkeypatch.setattr(bootstrap.direct_iam, "decode_canonical", decode_direct)

    verified = bootstrap.verify_bundle(root, expected_uid=os.geteuid())

    assert verified.revision == release_revision
    assert verified.direct_iam_identity["release_revision"] == (
        foundation_revision
    )
    assert decoded_against == [foundation_revision]


def test_install_transaction_crash_replays_only_uncommitted_phase(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state" / "transaction.json"
    calls: list[str] = []
    crash_phase = bootstrap.INSTALL_PHASES[2]
    should_crash = True

    def handler(phase: str):
        def run():
            nonlocal should_crash
            calls.append(phase)
            if phase == crash_phase and should_crash:
                should_crash = False
                raise RuntimeError("simulated power loss")
            return {"phase_completed": phase}

        return run

    handlers = {phase: handler(phase) for phase in bootstrap.INSTALL_PHASES}
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }
    with pytest.raises(RuntimeError, match="simulated power loss"):
        bootstrap.run_install_transaction(
            bundle=bundle,
            journal_path=journal,
            handlers=handlers,
            revalidators=revalidators,
            expected_uid=os.geteuid(),
        )

    first_prefix = list(bootstrap.INSTALL_PHASES[:2])
    assert calls == [*first_prefix, crash_phase]
    calls.clear()

    result = bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )

    assert calls == list(bootstrap.INSTALL_PHASES[2:])
    assert result["complete"] is True
    assert result["activation_performed"] is False
    assert result["cloud_mutation_performed"] is False

    calls.clear()
    replay = bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )
    assert replay == result
    assert calls == []


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "checkpoint",
    (
        "scratch_write_progress",
        "scratch_fsynced",
        "final_linked",
        "scratch_unlinked",
    ),
)
def test_sigkill_exact_byte_publication_recovers_without_partial_final(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    target = tmp_path / "private/key.pem"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_exact_bytes_kill_worker,
        args=(str(target), checkpoint, child),
    )
    process.start()
    try:
        assert parent.poll(5)
        assert parent.recv() == checkpoint
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        bootstrap._install_exact_bytes(
            target,
            b"immutable-owner-gate-bytes\n",
            mode=0o400,
            uid=os.geteuid(),
            gid=os.getegid(),
        )
        assert target.read_bytes() == b"immutable-owner-gate-bytes\n"
        assert target.stat().st_nlink == 1
        assert not list(target.parent.glob(".*.pending"))
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "checkpoint",
    (
        "private_scratch_fsynced",
        "private_final_linked",
        "private_scratch_unlinked",
        "public_scratch_fsynced",
        "public_final_linked",
        "public_scratch_unlinked",
    ),
)
def test_sigkill_receipt_key_pair_publication_reconciles_one_identity(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_receipt_key_kill_worker,
        args=(str(tmp_path), checkpoint, child),
    )
    process.start()
    try:
        assert parent.poll(5)
        assert parent.recv() == checkpoint
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        layout = bootstrap.InstallLayout(etc_root=tmp_path / "etc")
        private_path = layout.etc_root / "keys/receipt-signing-key.pem"
        public_path = layout.etc_root / "public/authority-receipt-public.pem"
        intent = {
            "schema": "muncho-owner-gate-receipt-key-phase-intent.v1",
            "targets": [
                {
                    "path": str(private_path),
                    "created_by_transaction": True,
                },
                {
                    "path": str(public_path),
                    "created_by_transaction": True,
                },
            ],
        }
        evidence = bootstrap.generate_or_verify_receipt_key(
            layout=layout,
            transaction_intent=intent,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
        )
        private_key = serialization.load_pem_private_key(
            private_path.read_bytes(),
            password=None,
        )
        assert isinstance(private_key, Ed25519PrivateKey)
        assert public_path.read_bytes() == private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        assert evidence["created"] is True
        assert evidence["created_by_transaction"] is True
        assert private_path.stat().st_nlink == 1
        assert public_path.stat().st_nlink == 1
        assert not list(layout.etc_root.rglob(".*.pending"))
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


@pytest.mark.live_system_guard_bypass
def test_sigkill_release_rename_reconciles_exact_pending_intent(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    release_base = tmp_path / "releases"
    staging = release_base / f".{bundle.revision}.bootstrap"
    _secure_test_directory(staging)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_release_rename_kill_worker,
        args=(str(tmp_path), child),
    )
    process.start()
    try:
        assert parent.poll(5)
        assert parent.recv() == "release_renamed"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        final = release_base / bundle.revision
        intent = {
            "schema": "muncho-owner-gate-release-phase-intent.v1",
            "staging_path": str(staging),
            "final_path": str(final),
            "created_by_transaction": True,
        }
        evidence = bootstrap.seal_and_publish_release(
            staging,
            bundle,
            layout=bootstrap.InstallLayout(release_base=release_base),
            transaction_intent=intent,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
        )
        assert not staging.exists()
        assert final.is_dir()
        assert final.stat().st_mode & 0o777 == 0o555
        assert evidence["created_by_transaction"] is True
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize("checkpoint", ("after_intent", "after_effect"))
def test_sigkill_transaction_replays_persisted_ownership_without_truth_flip(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    journal = tmp_path / "state/transaction.json"
    _secure_test_directory(journal.parent)
    effect = tmp_path / "managed/key.pem"
    _secure_test_directory(effect.parent)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_transaction_kill_worker,
        args=(str(journal), str(effect), checkpoint, child),
    )
    process.start()
    try:
        assert parent.poll(5)
        assert parent.recv() == checkpoint
        intent_path = journal.with_suffix(".journal") / "p0-intent.json"
        assert intent_path.is_file()
        if checkpoint == "after_intent":
            assert not effect.exists()
        else:
            assert effect.read_bytes() == b"transaction-owned\n"
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        result = _run_transaction_fixture(journal, effect)
        first = result["completed_phases"][0]["evidence"]
        assert first["created"] is True
        assert first["created_by_transaction"] is True
        assert effect.read_bytes() == b"transaction-owned\n"
        assert result["complete"] is True

        replay = _run_transaction_fixture(journal, effect)
        assert replay == result
        assert replay["completed_phases"][0]["evidence"] == first
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


def test_install_transaction_rejects_tampered_or_non_prefix_journal(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state" / "transaction.json"
    value = dict(bootstrap._new_transaction(bundle))
    value["completed_phases"] = [{
        "phase": bootstrap.INSTALL_PHASES[1],
        "evidence": {"ok": True},
        "evidence_sha256": bootstrap.foundation.sha256_json({"ok": True}),
    }]
    unsigned = {
        key: item for key, item in value.items() if key != "transaction_sha256"
    }
    value["transaction_sha256"] = bootstrap.foundation.sha256_json(unsigned)
    journal.parent.mkdir(mode=0o700)
    journal.write_bytes(bootstrap.foundation.canonical_json_bytes(value))
    journal.chmod(0o600)

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_transaction_invalid",
    ):
        bootstrap.load_or_create_transaction(
            journal,
            bundle=bundle,
            expected_uid=os.geteuid(),
        )


def test_install_transaction_rejects_semantically_tampered_immutable_phase(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state/transaction.json"
    handlers = {
        phase: lambda selected=phase: {"phase": selected}
        for phase in bootstrap.INSTALL_PHASES
    }
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }
    bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )
    phase = journal.with_suffix(".journal") / "p1-intent.json"
    phase.write_bytes(
        bootstrap.foundation.canonical_json_bytes(
            {"schema": "tampered-but-canonical.v1"}
        )
    )
    phase.chmod(0o600)

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_transaction_invalid",
    ):
        bootstrap.load_or_create_transaction(
            journal,
            bundle=bundle,
            expected_uid=os.geteuid(),
        )


@pytest.mark.parametrize(
    ("phase", "target_kind"),
    (
        (bootstrap.INSTALL_PHASES[2], "key"),
        (bootstrap.INSTALL_PHASES[3], "config"),
        (bootstrap.INSTALL_PHASES[4], "database"),
        (bootstrap.INSTALL_PHASES[5], "release"),
        (bootstrap.INSTALL_PHASES[6], "receipt"),
    ),
)
def test_fresh_vm_intent_fails_closed_on_unjournaled_managed_target(
    tmp_path: Path,
    phase: str,
    target_kind: str,
) -> None:
    bundle = _bundle(tmp_path)
    layout = bootstrap.InstallLayout(
        release_base=tmp_path / "releases",
        current_link=tmp_path / "current",
        etc_root=tmp_path / "etc",
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        systemd_root=tmp_path / "systemd",
        sysusers_root=tmp_path / "sysusers",
        tmpfiles_root=tmp_path / "tmpfiles",
        sudoers_root=tmp_path / "sudoers",
    )
    targets = {
        "key": layout.etc_root / "keys/receipt-signing-key.pem",
        "config": layout.etc_root / "executor.json",
        "database": layout.state_root / "authority/passkey-v2.sqlite3",
        "release": layout.release_base / bundle.revision,
        "receipt": (
            layout.state_root
            / "bootstrap-receipts"
            / f"install-{bundle.revision}.json"
        ),
    }
    target = targets[target_kind]
    target.parent.mkdir(parents=True)
    if target_kind == "release":
        target.mkdir()
    else:
        target.write_bytes(b"unowned")
    builders = bootstrap._production_phase_intent_builders(
        bundle=bundle,
        release=layout.release_base / f".{bundle.revision}.bootstrap",
        layout=layout,
        transaction_context={
            "transaction_id": "c" * 64,
            "next_prior_head_sha256": "d" * 64,
        },
    )

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_fresh_target_conflict",
    ):
        builders[phase]()


def test_install_transaction_checks_every_future_managed_target_before_effect(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    layout = bootstrap.InstallLayout(
        release_base=tmp_path / "releases",
        current_link=tmp_path / "current",
        etc_root=tmp_path / "etc",
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        systemd_root=tmp_path / "systemd",
        sysusers_root=tmp_path / "sysusers",
        tmpfiles_root=tmp_path / "tmpfiles",
        sudoers_root=tmp_path / "sudoers",
    )
    unexplained_receipt = (
        layout.state_root
        / "bootstrap-receipts"
        / f"install-{bundle.revision}.json"
    )
    _secure_test_directory(unexplained_receipt.parent)
    unexplained_receipt.write_bytes(b"unowned")
    journal = (
        layout.state_root
        / "bootstrap-receipts"
        / f"transaction-{bundle.revision}.json"
    )
    context: dict = {}
    handler_calls: list[str] = []
    handlers = {
        phase: (
            lambda selected=phase: (
                handler_calls.append(selected)
                or {"phase_completed": selected}
            )
        )
        for phase in bootstrap.INSTALL_PHASES
    }
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }
    builders = {
        phase: lambda selected=phase: {
            "schema": "bootstrap-test-noop-intent.v1",
            "phase": selected,
        }
        for phase in bootstrap.INSTALL_PHASES
    }

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_fresh_target_conflict",
    ):
        bootstrap.run_install_transaction(
            bundle=bundle,
            journal_path=journal,
            handlers=handlers,
            revalidators=revalidators,
            expected_uid=os.geteuid(),
            transaction_context=context,
            intent_builders=builders,
            fresh_target_guard=bootstrap._production_fresh_target_guard(
                bundle=bundle,
                layout=layout,
                transaction_context=context,
            ),
        )
    assert handler_calls == []


def test_install_transaction_exclusive_lock_rejects_concurrent_runner(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state/transaction.json"
    handlers = {
        phase: (lambda completed=phase: {"phase_completed": completed})
        for phase in bootstrap.INSTALL_PHASES
    }
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }

    with bootstrap._exclusive_transaction_lock(
        journal,
        expected_uid=os.geteuid(),
    ):
        with pytest.raises(
            bootstrap.OwnerGateBootstrapError,
            match="owner_gate_bootstrap_transaction_locked",
        ):
            bootstrap.run_install_transaction(
                bundle=bundle,
                journal_path=journal,
                handlers=handlers,
                revalidators=revalidators,
                expected_uid=os.geteuid(),
            )


def test_install_transaction_revalidates_every_committed_phase_and_fails_drift(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state/transaction.json"
    handler_calls: list[str] = []
    revalidated: list[str] = []

    def handler(phase: str):
        def invoke() -> dict[str, str]:
            handler_calls.append(phase)
            return {"phase_completed": phase}

        return invoke

    handlers = {phase: handler(phase) for phase in bootstrap.INSTALL_PHASES}
    revalidators = {
        phase: (
            lambda evidence, current=phase: (
                revalidated.append(current) or dict(evidence)
            )
        )
        for phase in bootstrap.INSTALL_PHASES
    }
    bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )
    handler_calls.clear()
    bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )
    assert handler_calls == []
    assert revalidated == list(bootstrap.INSTALL_PHASES)

    drifting = dict(revalidators)
    drifting[bootstrap.INSTALL_PHASES[0]] = lambda evidence: {
        **evidence,
        "drifted": True,
    }
    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_committed_phase_drift",
    ):
        bootstrap.run_install_transaction(
            bundle=bundle,
            journal_path=journal,
            handlers=handlers,
            revalidators=drifting,
            expected_uid=os.geteuid(),
        )
    assert handler_calls == []


def test_install_transaction_handler_failure_after_side_effect_reconciles(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    journal = tmp_path / "state/transaction.json"
    side_effect = tmp_path / "external-side-effect"
    crash_once = True

    def first_phase() -> dict[str, str]:
        nonlocal crash_once
        if not side_effect.exists():
            side_effect.write_text("created-once", encoding="ascii")
        if crash_once:
            crash_once = False
            raise RuntimeError("handler failed after side effect")
        return {
            "phase_completed": bootstrap.INSTALL_PHASES[0],
            "side_effect_sha256": hashlib.sha256(side_effect.read_bytes()).hexdigest(),
        }

    handlers = {
        phase: (
            first_phase
            if phase == bootstrap.INSTALL_PHASES[0]
            else (lambda completed=phase: {"phase_completed": completed})
        )
        for phase in bootstrap.INSTALL_PHASES
    }
    revalidators = {
        phase: lambda evidence: dict(evidence)
        for phase in bootstrap.INSTALL_PHASES
    }

    with pytest.raises(RuntimeError, match="handler failed"):
        bootstrap.run_install_transaction(
            bundle=bundle,
            journal_path=journal,
            handlers=handlers,
            revalidators=revalidators,
            expected_uid=os.geteuid(),
        )
    assert side_effect.read_text("ascii") == "created-once"
    transaction = bootstrap.load_or_create_transaction(
        journal,
        bundle=bundle,
        expected_uid=os.geteuid(),
    )
    assert transaction["completed_phases"] == []

    result = bootstrap.run_install_transaction(
        bundle=bundle,
        journal_path=journal,
        handlers=handlers,
        revalidators=revalidators,
        expected_uid=os.geteuid(),
    )
    assert result["complete"] is True
    assert side_effect.read_text("ascii") == "created-once"


def test_canonical_database_bootstrap_is_exact_and_replay_safe(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "state"
    etc_root = tmp_path / "etc"
    for path in (
        state_root / "authority",
        state_root / "executor",
        etc_root / "public",
    ):
        _secure_test_directory(path)
    private = Ed25519PrivateKey.generate()
    public_path = etc_root / "public/authority-receipt-public.pem"
    public_path.write_bytes(private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    public_path.chmod(0o444)
    credential_id = b"synthetic-public-credential-id"
    public_key_cose = b"synthetic-public-cose-key"
    user_handle = b"synthetic-user-handle"
    migration = {
        "credential_id_b64url": base64.urlsafe_b64encode(credential_id)
        .rstrip(b"=")
        .decode("ascii"),
        "public_key_cose_b64url": base64.urlsafe_b64encode(public_key_cose)
        .rstrip(b"=")
        .decode("ascii"),
        "expected_user_handle_b64url": base64.urlsafe_b64encode(user_handle)
        .rstrip(b"=")
        .decode("ascii"),
        "envelope_sha256": "e" * 64,
        "collected_at_unix": 1_700_000_000,
        "initial_sign_count": 0,
        "initial_credential_backed_up": True,
    }
    bundle = bootstrap.VerifiedBundle(
        root=tmp_path,
        manifest={
            "release_revision": "a" * 40,
            "package_sha256": "b" * 64,
        },
        authority={},
        migration=migration,
    )
    layout = bootstrap.InstallLayout(
        release_base=tmp_path / "releases",
        current_link=tmp_path / "current",
        etc_root=etc_root,
        state_root=state_root,
        run_root=tmp_path / "run",
        systemd_root=tmp_path / "systemd",
        sysusers_root=tmp_path / "sysusers",
        tmpfiles_root=tmp_path / "tmpfiles",
        sudoers_root=tmp_path / "sudoers",
            python=Path("/usr/bin/python3"),
            os_release=tmp_path / "os-release",
        )
    key_receipt = {
        "public_key_path": str(public_path),
        "public_key_id": hashlib.sha256(
            private.public_key().public_bytes_raw()
        ).hexdigest(),
    }

    first = bootstrap.bootstrap_and_verify_databases(
        bundle,
        layout=layout,
        key_receipt=key_receipt,
        now_unix=1_700_000_001,
        require_root=False,
        authority_uid=os.geteuid(),
        authority_gid=os.getegid(),
        executor_uid=os.geteuid(),
        executor_gid=os.getegid(),
    )
    replay = bootstrap.bootstrap_and_verify_databases(
        bundle,
        layout=layout,
        key_receipt=key_receipt,
        now_unix=1_700_000_999,
        require_root=False,
        authority_uid=os.geteuid(),
        authority_gid=os.getegid(),
        executor_uid=os.geteuid(),
        executor_gid=os.getegid(),
    )

    assert first["credential_count"] == 1
    assert first["credential_imported_this_attempt"] is True
    assert replay["credential_count"] == 1
    assert replay["credential_imported_this_attempt"] is False
    assert first["credential_record_sha256"] == replay["credential_record_sha256"]
    assert first["authority_preflight"] == replay["authority_preflight"]
    assert first["executor_preflight"] == replay["executor_preflight"]


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "checkpoint",
    (
        "authority_stage_validated",
        "authority_final_linked",
        "authority_stage_unlinked",
        "executor_stage_validated",
        "executor_final_linked",
        "executor_stage_unlinked",
    ),
)
def test_sigkill_database_staging_reconciles_one_import_truth(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    public_path = tmp_path / "etc/public/authority-receipt-public.pem"
    _secure_test_directory(public_path.parent)
    private_key = Ed25519PrivateKey.generate()
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_path.chmod(0o444)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_database_kill_worker,
        args=(str(tmp_path), checkpoint, child),
    )
    process.start()
    try:
        assert parent.poll(10)
        assert parent.recv() == checkpoint
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        bundle, layout, key_receipt, intent = _database_fixture(tmp_path)
        evidence = bootstrap.bootstrap_and_verify_databases(
            bundle,
            layout=layout,
            key_receipt=key_receipt,
            now_unix=1_700_000_001,
            require_root=False,
            authority_uid=os.geteuid(),
            authority_gid=os.getegid(),
            executor_uid=os.geteuid(),
            executor_gid=os.getegid(),
            transaction_intent=intent,
        )
        assert evidence["credential_count"] == 1
        assert evidence["credential_imported_this_attempt"] is True
        assert evidence["credential_imported_by_transaction"] is True
        assert evidence["authority_database_created_by_transaction"] is True
        assert evidence["executor_database_created_by_transaction"] is True
        for target in intent["targets"]:
            final = Path(target["path"])
            stage = Path(target["stage_path"])
            assert final.is_file()
            assert final.stat().st_nlink == 1
            assert not stage.exists()
            assert not any(
                Path(f"{candidate}{suffix}").exists()
                for candidate in (final, stage)
                for suffix in ("-journal", "-wal", "-shm")
            )
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


def test_inert_rollback_removes_only_digest_bound_entry_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "systemd"
    root.mkdir()
    path = root / "muncho.service"
    raw = b"[Service]\nExecStart=/bin/false\n"
    path.write_bytes(raw)
    path.chmod(0o444)
    os.chown(path, os.geteuid(), os.getegid())
    evidence = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": "0444",
        "uid": os.geteuid(),
        "gid": os.getegid(),
    }

    assert bootstrap._remove_exact_installed_file(
        evidence,
        allowed_roots=(root,),
    ) is True
    assert not path.exists()
    assert bootstrap._remove_exact_installed_file(
        evidence,
        allowed_roots=(root,),
    ) is False


def test_inert_rollback_refuses_changed_entry_and_preserves_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "systemd"
    root.mkdir()
    path = root / "muncho.service"
    path.write_bytes(b"changed")
    path.chmod(0o444)
    os.chown(path, os.geteuid(), os.getegid())
    evidence = {
        "path": str(path),
        "sha256": "0" * 64,
        "mode": "0444",
        "uid": os.geteuid(),
        "gid": os.getegid(),
    }

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_rollback_file_changed",
    ):
        bootstrap._remove_exact_installed_file(
            evidence,
            allowed_roots=(root,),
        )
    assert path.read_bytes() == b"changed"


def test_partial_rollback_does_not_claim_artifacts_never_committed(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    layout = bootstrap.InstallLayout(
        release_base=tmp_path / "releases",
        current_link=tmp_path / "current",
        etc_root=tmp_path / "etc",
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        systemd_root=tmp_path / "systemd",
        sysusers_root=tmp_path / "sysusers",
        tmpfiles_root=tmp_path / "tmpfiles",
        sudoers_root=tmp_path / "sudoers",
            python=Path("/usr/bin/python3"),
            os_release=tmp_path / "os-release",
        )
    phase_evidence = {
        "install_root_owned_configuration_units_firewall_and_hosts": {
            "schema": "muncho-owner-gate-installed-system-files.v1"
        }
    }

    projection = bootstrap._rollback_preservation_projection(
        phase_evidence,
        bundle=bundle,
        layout=layout,
    )

    assert projection == {
        "authority_database_preserved": False,
        "executor_database_preserved": False,
        "install_receipt_preserved": False,
        "immutable_release_preserved": False,
    }


def test_partial_rollback_claims_only_committed_artifacts_that_exist(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    layout = bootstrap.InstallLayout(
        release_base=tmp_path / "releases",
        current_link=tmp_path / "current",
        etc_root=tmp_path / "etc",
        state_root=tmp_path / "state",
        run_root=tmp_path / "run",
        systemd_root=tmp_path / "systemd",
        sysusers_root=tmp_path / "sysusers",
        tmpfiles_root=tmp_path / "tmpfiles",
        sudoers_root=tmp_path / "sudoers",
            python=Path("/usr/bin/python3"),
            os_release=tmp_path / "os-release",
        )
    authority = layout.state_root / "authority/passkey-v2.sqlite3"
    executor = layout.state_root / "executor/execution-v2.sqlite3"
    authority.parent.mkdir(parents=True)
    executor.parent.mkdir(parents=True)
    authority.write_bytes(b"authority")
    executor.write_bytes(b"executor")
    phase_evidence = {
        "bootstrap_and_verify_canonical_databases": {
            "schema": "muncho-owner-gate-canonical-databases-bootstrap.v1"
        }
    }

    projection = bootstrap._rollback_preservation_projection(
        phase_evidence,
        bundle=bundle,
        layout=layout,
    )

    assert projection["authority_database_preserved"] is True
    assert projection["executor_database_preserved"] is True
    assert projection["install_receipt_preserved"] is False
    assert projection["immutable_release_preserved"] is False


def test_rollback_strict_load_never_creates_empty_install_transaction(
    tmp_path: Path,
) -> None:
    bundle, layout = _rollback_fixture(tmp_path)
    receipts = layout.state_root / "bootstrap-receipts"
    _secure_test_directory(receipts)
    bootstrap.foundation.MUTATION_ENABLE_SEAL = tmp_path / "no-seal"

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_transaction_missing",
    ):
        bootstrap.rollback_inert_install(
            bundle.root,
            layout=layout,
            _verified_bundle=bundle,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
            _require_root=False,
        )
    assert not (
        receipts / f"transaction-{bundle.revision}.journal"
    ).exists()


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    "checkpoint",
    (
        "rollback_intent_published",
        "rollback_target_0_absent",
        "rollback_receipt_scratch_fsynced",
        "rollback_receipt_final_linked",
        "rollback_receipt_scratch_unlinked",
        "rollback_receipt_published",
        "rollback_success_published",
        "rollback_terminal_published",
    ),
)
def test_sigkill_rollback_reconciles_stable_target_list_and_terminal_truth(
    tmp_path: Path,
    checkpoint: str,
) -> None:
    transaction, targets = _prepare_rollback_fixture(tmp_path)
    bundle, layout = _rollback_fixture(tmp_path)
    bootstrap.foundation.MUTATION_ENABLE_SEAL = tmp_path / "no-seal"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(
        target=_rollback_kill_worker,
        args=(str(tmp_path), checkpoint, child),
    )
    process.start()
    try:
        assert parent.poll(10)
        assert parent.recv() == checkpoint
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)
        assert process.exitcode == -signal.SIGKILL

        receipt = bootstrap.rollback_inert_install(
            bundle.root,
            layout=layout,
            _verified_bundle=bundle,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
            _require_root=False,
        )
        assert receipt["transaction_sha256"] == transaction[
            "transaction_sha256"
        ]
        assert receipt["removed_entry_files"] == sorted(
            item["path"] for item in targets
        )
        assert receipt["authority_database_preserved"] is True
        assert receipt["executor_database_preserved"] is True
        assert receipt["install_receipt_preserved"] is True
        assert receipt["immutable_release_preserved"] is True
        assert receipt["current_release_selection_removed"] is False
        assert all(not Path(item["path"]).exists() for item in targets)

        replay = bootstrap.rollback_inert_install(
            bundle.root,
            layout=layout,
            _verified_bundle=bundle,
            _expected_uid=os.geteuid(),
            _expected_gid=os.getegid(),
            _require_root=False,
        )
        assert replay == receipt
        journal_root = (
            layout.state_root
            / "bootstrap-receipts"
            / f"transaction-{bundle.revision}.journal"
        )
        assert (journal_root / "rollback-intent.json").is_file()
        assert (journal_root / "rollback-success.json").is_file()
        assert (journal_root / "rollback-terminal.json").is_file()
    finally:
        if process.is_alive():
            process.kill()
        process.join(timeout=5)
        parent.close()
        child.close()


def test_release_seal_rejects_non_root_group_on_any_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    release_base = tmp_path / "releases"
    release = release_base / bundle.revision
    _secure_test_directory(release)
    child = release / "payload.txt"
    child.write_bytes(b"payload")
    layout = bootstrap.InstallLayout(release_base=release_base)
    real_lstat = Path.lstat

    def fake_lstat(path: Path):
        state = real_lstat(path)
        return SimpleNamespace(
            st_mode=state.st_mode,
            st_uid=0,
            st_gid=1 if path == child else 0,
        )

    monkeypatch.setattr(Path, "lstat", fake_lstat)

    with pytest.raises(
        bootstrap.OwnerGateBootstrapError,
        match="owner_gate_bootstrap_release_owner_invalid",
    ):
        bootstrap.seal_and_publish_release(release, bundle, layout=layout)
