from __future__ import annotations

import base64
import copy
import hashlib
import os
import shutil
import stat
import subprocess
import traceback
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_stage0 as stage0
from scripts.canary import owner_gate_trust as trust
from scripts.canary import direct_iam_identity_authority as direct_iam


ROOT = Path(__file__).parents[3]
REVISION = "a" * 40


def test_package_formatted_traceback_suppresses_raw_cause(tmp_path: Path) -> None:
    sensitive_path = tmp_path / "provider-secret-wheel-name.whl"
    try:
        package._sha256_file(sensitive_path)
    except package.OwnerGatePackageError as exc:
        rendered = "".join(traceback.format_exception(exc))
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected stable package error")

    assert "FileNotFoundError" not in rendered
    assert str(sensitive_path) not in rendered
    assert "The above exception was the direct cause" not in rendered
    assert "During handling of the above exception" not in rendered
    assert rendered.rstrip().endswith("owner_gate_package_file_unavailable")


def test_bootstrap_append_only_journal_is_in_root_runtime_inventory() -> None:
    assert (
        "scripts/canary/owner_gate_bootstrap_journal.py"
        in package.ROOT_RUNTIME_FILES
    )


def test_target_cloud_observation_signer_is_in_exact_offline_package() -> None:
    assert (
        "bin/muncho-owner-gate-cloud-observation-signer"
        in package.REQUIRED_ENTRYPOINTS
    )
    assert (
        "scripts/canary/owner_gate_cloud_observation_signer.py"
        in package.ROOT_RUNTIME_FILES
    )
    assert (
        "ops/muncho/owner-gate/bin/muncho-owner-gate-cloud-observation-signer"
        in package.REQUIRED_ASSET_FILES
    )


def test_activation_evidence_stager_is_in_exact_release_closure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = "scripts/canary/owner_gate_activation_evidence_stager.py"
    entrypoint = "bin/muncho-owner-gate-stage-activation-evidence"
    asset = (
        "ops/muncho/owner-gate/bin/"
        "muncho-owner-gate-stage-activation-evidence"
    )
    assert runtime in package.ROOT_RUNTIME_FILES
    assert entrypoint in package.REQUIRED_ENTRYPOINTS
    assert asset in package.REQUIRED_ASSET_FILES

    def local_blob(
        source_root: Path,
        release_revision: str,
        relative: str,
        *,
        required: bool,
    ):
        assert source_root == ROOT
        assert release_revision == REVISION
        selected = source_root / relative
        if not selected.is_file():
            if required:
                raise package.OwnerGatePackageError(
                    "owner_gate_package_git_object_missing"
                )
            return None
        raw = selected.read_bytes()
        mode = "100755" if selected.stat().st_mode & 0o111 else "100644"
        return raw, mode

    monkeypatch.setattr(package, "_git_blob", local_blob)
    closure = package.resolve_runtime_source_closure(
        ROOT,
        release_revision=REVISION,
    )
    assert runtime in closure
    assert "scripts/canary/owner_gate_activation_seal.py" in closure
    assert "scripts/canary/passkey_v2_protocol.py" in closure


def test_activation_evidence_staging_receipts_have_exact_tmpfiles_identity() -> None:
    tmpfiles = (
        ROOT / "ops/muncho/owner-gate/muncho-owner-gate.tmpfiles"
    ).read_text(encoding="ascii").splitlines()
    assert tmpfiles.count(
        "d /var/lib/muncho-owner-gate/"
        "activation-evidence-staging-receipts 0700 root root -"
    ) == 1


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o444)


def _stub_direct_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    path = tmp_path / "stub-direct-iam-identity-authority.json"
    _write(path, b"{}")
    monkeypatch.setattr(
        package.direct_iam,
        "decode_canonical",
        lambda _raw, *, release_revision=None: {
            "release_revision": release_revision,
            "pre_foundation_authority_sha256": "a" * 64,
            "foundation_apply_receipt_sha256": "b" * 64,
            "resource_ancestor_chain": ["organizations/123456789012"],
        },
    )
    return path


def _source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "source"
    _write(source / "scripts/__init__.py", b"")
    _write(source / "scripts/canary/__init__.py", b"")
    _write(
        source / "scripts/canary/passkey_v2_service.py",
        b"from scripts.canary import passkey_v2_sqlite\nVALUE = passkey_v2_sqlite.VALUE\n",
    )
    _write(source / "scripts/canary/passkey_v2_sqlite.py", b"VALUE = 7\n")
    monkeypatch.setattr(
        package,
        "ROOT_RUNTIME_FILES",
        ("scripts/canary/passkey_v2_service.py",),
    )
    monkeypatch.setattr(
        package,
        "_verify_clean_git_source",
        lambda _root: (REVISION, "b" * 40),
    )
    shutil.copytree(
        ROOT / "ops/muncho/owner-gate",
        source / "ops/muncho/owner-gate",
        copy_function=shutil.copy2,
    )
    for path in (source / "ops/muncho/owner-gate").rglob("*"):
        if path.is_file():
            path.chmod(0o555 if "/bin/" in str(path) else 0o444)
    def git_blob(
        root: Path,
        release_revision: str,
        relative: str,
        *,
        required: bool,
    ):
        assert root == source
        assert release_revision == REVISION
        selected = root / relative
        if not selected.is_file():
            if required:
                raise package.OwnerGatePackageError(
                    "owner_gate_package_git_object_missing"
                )
            return None
        raw = selected.read_bytes()
        mode = "100755" if selected.stat().st_mode & 0o111 else "100644"
        return raw, mode

    monkeypatch.setattr(package, "_git_blob", git_blob)
    return source


def _wheelhouse(
    tmp_path: Path,
    *,
    compiled_platforms: Mapping[str, str] | None = None,
) -> tuple[Path, dict]:
    root = tmp_path / "wheels"
    root.mkdir()
    wheels = []
    for project, version in package.REQUIRED_PROJECTS.items():
        normalized = project.replace("-", "_")
        compiled_platform = (
            (compiled_platforms or {}).get(
                project,
                "manylinux_2_28_x86_64"
                if project == "cbor2"
                else "manylinux_2_17_x86_64",
            )
        )
        if project in package.BINARY_PROJECTS:
            filename = (
                f"{normalized}-{version}-cp311-cp311-"
                f"{compiled_platform}.whl"
            )
        else:
            filename = f"{normalized}-{version}-py3-none-any.whl"
        path = root / filename
        dist_info = f"{normalized}-{version}.dist-info"
        dependencies = sorted(
            package.EXPECTED_DIRECT_DEPENDENCIES.get(project, set())
        )
        requires = "".join(
            f"Requires-Dist: {dependency}\n" for dependency in dependencies
        )
        # Both markers are false for the exact Linux/CPython 3.11 target and
        # prove that the validator evaluates PEP 508 markers instead of doing
        # brittle string matching.
        if project == "anyio":
            requires += (
                'Requires-Dist: exceptiongroup; python_version < "3.11"\n'
                'Requires-Dist: colorama; sys_platform == "win32"\n'
            )
        pure = project not in package.BINARY_PROJECTS
        tag = "py3-none-any" if pure else f"cp311-cp311-{compiled_platform}"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                f"{dist_info}/METADATA",
                (
                    "Metadata-Version: 2.1\n"
                    f"Name: {project}\n"
                    f"Version: {version}\n"
                    f"{requires}\n"
                ),
            )
            archive.writestr(
                f"{dist_info}/WHEEL",
                (
                    "Wheel-Version: 1.0\n"
                    f"Root-Is-Purelib: {'true' if pure else 'false'}\n"
                    f"Tag: {tag}\n\n"
                ),
            )
            archive.writestr(f"{normalized}/__init__.py", "")
        path.chmod(0o444)
        payload = path.read_bytes()
        wheels.append({
            "filename": filename,
            "project": project,
            "version": version,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        })
    bootstrap_version = package.LOCAL_RUNTIME_LOCK["bootstrap_pip"]["version"]
    bootstrap_filename = f"pip-{bootstrap_version}-py3-none-any.whl"
    bootstrap_path = root / bootstrap_filename
    bootstrap_dist_info = f"pip-{bootstrap_version}.dist-info"
    with zipfile.ZipFile(
        bootstrap_path,
        "w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(
            f"{bootstrap_dist_info}/METADATA",
            (
                "Metadata-Version: 2.1\n"
                "Name: pip\n"
                f"Version: {bootstrap_version}\n\n"
            ),
        )
        archive.writestr(
            f"{bootstrap_dist_info}/WHEEL",
            (
                "Wheel-Version: 1.0\n"
                "Root-Is-Purelib: true\n"
                "Tag: py3-none-any\n\n"
            ),
        )
        archive.writestr("pip/__init__.py", "")
    bootstrap_path.chmod(0o444)
    bootstrap_raw = bootstrap_path.read_bytes()
    bootstrap_pip = {
        "filename": bootstrap_filename,
        "project": "pip",
        "version": bootstrap_version,
        "sha256": hashlib.sha256(bootstrap_raw).hexdigest(),
        "size": len(bootstrap_raw),
    }
    runtime_lock = _runtime_lock_for_wheels(wheels, bootstrap_pip)
    unsigned = {
        "schema": package.WHEELHOUSE_SCHEMA,
        "python_version": package.PYTHON_VERSION,
        "platform": package.WHEELHOUSE_PLATFORM,
        "network_required": False,
        "source_build_allowed": False,
        "complete_transitive_closure": True,
        "runtime_lock_sha256": package.runtime_lock_file_sha256(runtime_lock),
        "bootstrap_pip": bootstrap_pip,
        "wheels": wheels,
    }
    return root, {
        **unsigned,
        "manifest_sha256": foundation.sha256_json(unsigned),
    }


def _runtime_lock_for_wheels(
    wheels: list[dict],
    bootstrap_pip: Mapping,
) -> dict:
    lock_unsigned = {
        "bootstrap_pip": {
            **bootstrap_pip,
            "active_dependencies": [],
        },
        "schema": package.RUNTIME_LOCK_SCHEMA,
        "python_version": package.PYTHON_VERSION,
        "platform": package.WHEELHOUSE_PLATFORM,
        "network_required": False,
        "source_build_allowed": False,
        "complete_transitive_closure": True,
        "wheels": sorted(
            (
                {
                    **item,
                    "active_dependencies": sorted(
                        package.EXPECTED_DIRECT_DEPENDENCIES.get(
                            item["project"], set()
                        )
                    ),
                }
                for item in wheels
            ),
            key=lambda item: item["project"],
        ),
    }
    return {
        **lock_unsigned,
        "lock_sha256": foundation.sha256_json(lock_unsigned),
    }


def _install_test_runtime_lock(source: Path, runtime_lock: Mapping) -> None:
    path = source / package.RUNTIME_LOCK_RELATIVE
    path.chmod(0o644)
    path.write_bytes(foundation.canonical_json_bytes(runtime_lock) + b"\n")
    path.chmod(0o444)


def _minimal_idna_wheel(
    path: Path,
    *,
    extra: zipfile.ZipInfo | str | None = None,
    extra_payload: bytes = b"",
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr(
            "idna-3.18.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: idna\nVersion: 3.18\n\n",
        )
        archive.writestr(
            "idna-3.18.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
        )
        archive.writestr("idna/__init__.py", "")
        if extra is not None:
            archive.writestr(extra, extra_payload)
    path.chmod(0o444)


def _exact_git_source(tmp_path: Path) -> tuple[Path, str]:
    source = tmp_path / "exact-git-source"
    source.mkdir()
    runtime = source / "runtime.py"
    runtime.write_bytes(b"VALUE = 1\n")
    runtime.chmod(0o644)
    commands = (
        ("init", "--quiet"),
        ("add", "--", "runtime.py"),
        (
            "-c",
            "user.name=Owner Gate Test",
            "-c",
            "user.email=owner-gate@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "exact source",
        ),
    )
    for arguments in commands:
        subprocess.run(
            ("/usr/bin/git", "-C", str(source), *arguments),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    revision = subprocess.run(
        ("/usr/bin/git", "-C", str(source), "rev-parse", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    ).stdout.decode("ascii").strip()
    return source, revision


def _git_race_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[package.PackageSpec, Path]:
    source, revision = _exact_git_source(tmp_path)
    wheelhouse = tmp_path / "empty-wheelhouse"
    wheelhouse.mkdir()
    direct = tmp_path / "direct-iam.json"
    _write(direct, b"{}")
    monkeypatch.setattr(package, "ROOT_RUNTIME_FILES", ("runtime.py",))
    monkeypatch.setattr(package, "REQUIRED_ASSET_FILES", ())
    monkeypatch.setattr(package, "validate_wheelhouse", lambda **_kwargs: ())
    monkeypatch.setattr(
        package,
        "_load_release_runtime_lock",
        lambda *_args: (
            package.LOCAL_RUNTIME_LOCK,
            package.runtime_lock_file_sha256(package.LOCAL_RUNTIME_LOCK),
        ),
    )
    monkeypatch.setattr(
        package.direct_iam,
        "decode_canonical",
        lambda _raw, *, release_revision=None: {
            "release_revision": release_revision,
            "pre_foundation_authority_sha256": "a" * 64,
            "foundation_apply_receipt_sha256": "b" * 64,
            "resource_ancestor_chain": ["organizations/123456789012"],
        },
    )
    return (
        package.PackageSpec(
            source_root=source,
            release_revision=revision,
            wheelhouse_root=wheelhouse,
            wheelhouse_manifest={"manifest_sha256": "c" * 64},
            interpreter_sha256="d" * 64,
            direct_iam_identity_authority_path=direct,
        ),
        source / "runtime.py",
    )


def _trusted_spec(
    *,
    source: Path,
    wheel_root: Path,
    wheel_manifest: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_lock: Mapping | None = None,
) -> package.PackageSpec:
    if runtime_lock is not None:
        _install_test_runtime_lock(source, runtime_lock)
    member = (
        "serviceAccount:muncho-owner-gate-executor@"
        "adventico-ai-platform.iam.gserviceaccount.com"
    )
    authority_unsigned = {
        "schema": direct_iam.SCHEMA,
        "release_revision": REVISION,
        "project_id": foundation.PROJECT,
        "project_number": direct_iam.PROJECT_NUMBER,
        "owner_gate_vm_name": foundation.VM_NAME,
        "owner_gate_vm_numeric_id": "1234567890123456789",
        "owner_gate_service_account_email": (
            direct_iam.OWNER_GATE_SERVICE_ACCOUNT_EMAIL
        ),
        "owner_gate_service_account_unique_id": "123456789012345678901",
        "target_service_account_email": direct_iam.TARGET_SERVICE_ACCOUNT_EMAIL,
        "target_service_account_unique_id": "223456789012345678901",
        "resource_ancestor_chain": ["organizations/123456789012"],
        "project_read_role": (
            f"projects/{foundation.PROJECT}/roles/"
            f"{foundation.READ_ONLY_IAM_ROLE_ID}"
        ),
        "project_read_role_title": foundation.PROJECT_READ_ROLE_TITLE,
        "project_read_role_description": foundation.PROJECT_READ_ROLE_DESCRIPTION,
        "project_read_role_etag": "project-read-role-etag",
        "project_read_permissions": list(foundation.READ_ONLY_IAM_PERMISSIONS),
        "project_read_binding_member": member,
        "project_read_binding_present": True,
        "ancestor_read_role": (
            "organizations/123456789012/roles/"
            f"{foundation.ANCESTOR_READ_ONLY_IAM_ROLE_ID}"
        ),
        "ancestor_read_role_title": foundation.ANCESTOR_READ_ROLE_TITLE,
        "ancestor_read_role_description": foundation.ANCESTOR_READ_ROLE_DESCRIPTION,
        "ancestor_read_role_etag": "ancestor-read-role-etag",
        "ancestor_read_permissions": list(
            foundation.DIRECT_IAM_ANCESTOR_PERMISSIONS
        ),
        "ancestor_binding_member": member,
        "ancestor_binding_present": True,
        "mutation_role": (
            f"projects/{foundation.PROJECT}/roles/"
            f"{foundation.MUTATION_ROLE_ID}"
        ),
        "mutation_role_title": foundation.MUTATION_ROLE_TITLE,
        "mutation_role_description": foundation.MUTATION_ROLE_DESCRIPTION,
        "mutation_role_etag": "mutation-role-etag",
        "mutation_permissions": list(foundation.MUTATION_PERMISSIONS),
        "mutation_condition": {
            "title": foundation.MUTATION_CONDITION_TITLE,
            "description": foundation.MUTATION_CONDITION_DESCRIPTION,
            "expression": foundation._condition_expression(),
        },
        "mutation_binding_member": member,
        "mutation_binding_present": False,
        "mutation_activation_seal": str(foundation.MUTATION_ENABLE_SEAL),
        "mutation_activation_seal_present": False,
        "allowed_owner_gate_impersonators": [],
        "owner_gate_user_managed_key_inventory": {
            "requested_key_types": ["USER_MANAGED"],
            "allowed_key_names": [],
        },
        "external_gcp_admin_trust_root": {
            "inventory_complete": True,
            "structural_partition_complete": True,
            "passkey_protects_against_external_gcp_admins": False,
            "passkey_protects_against_pinned_external_roots": False,
            "google_provider_control_plane_outside_passkey": True,
            "collected_under_owner_reauthentication_receipt_sha256": "8" * 64,
            "resource_policy_generations": [
                {
                    "resource": f"projects/{foundation.PROJECT}",
                    "version": 3,
                    "etag": "project-policy-etag",
                    "audit_configs": [],
                },
                {
                    "resource": "organizations/123456789012",
                    "version": 3,
                    "etag": "organization-policy-etag",
                    "audit_configs": [],
                },
            ],
            "allowed_residual_bindings": [{
                "resource": "organizations/123456789012",
                "role": "roles/resourcemanager.organizationAdmin",
                "members": ["user:owner@example.test"],
                "condition": None,
            }],
            "allowed_residual_role_definitions": [{
                "name": "roles/resourcemanager.organizationAdmin",
                "title": "Organization Administrator",
                "description": "Test fixture",
                "included_permissions": [
                    "resourcemanager.organizations.setIamPolicy"
                ],
                "stage": "GA",
                "deleted": False,
                "etag": "external-role-etag",
            }],
        },
        "metadata_oauth_scopes": list(foundation.OWNER_GATE_OAUTH_SCOPES),
        "private_google_api_hosts": list(foundation.PRIVATE_GOOGLE_API_HOSTS),
        "private_google_api_vip_range": foundation.PRIVATE_GOOGLE_API_VIP_RANGE,
        "owner_reauthentication_receipt_sha256": "8" * 64,
        "pre_foundation_authority_sha256": "a" * 64,
        "foundation_apply_receipt_sha256": "b" * 64,
        "collected_at_unix": 1_800_000_000,
    }
    authority = {
        **authority_unsigned,
        "authority_sha256": foundation.sha256_json(authority_unsigned),
    }
    authority_raw = foundation.canonical_json_bytes(authority)
    authority_path = tmp_path / "direct-iam-identity-authority.json"
    _write(authority_path, authority_raw)
    spec = package.PackageSpec(
        source_root=source,
        release_revision=REVISION,
        wheelhouse_root=wheel_root,
        wheelhouse_manifest=wheel_manifest,
        interpreter_sha256="9" * 64,
        direct_iam_identity_authority_path=authority_path,
    )
    inventory = package.build_inventory(spec)
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes_raw()
    public_key_path = tmp_path / "release-trust-signing.pub"
    _write(public_key_path, public_key)
    collector_paths = {}
    collector_ids = {}
    for name in ("network", "cloud", "host"):
        raw = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
        path = tmp_path / f"{name}-observation-attestation.pub"
        _write(path, raw)
        collector_paths[name] = path
        collector_ids[name] = hashlib.sha256(raw).hexdigest()
    migration_path = tmp_path / "credential-migration.json"
    migration_raw = foundation.canonical_json_bytes({
        "schema": "test-host-attested-public-credential",
        "secret_material": False,
    })
    _write(migration_path, migration_raw)
    unsigned = {
        "schema": trust.TRUST_SCHEMA,
        "approved_for_offline_install": True,
        "fork_repository": trust.FORK_REPOSITORY,
        "release_revision": REVISION,
        "source_tree_oid": "b" * 40,
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "boot_image_self_link": (
            "projects/debian-cloud/global/images/"
            "debian-12-bookworm-v20260609"
        ),
        "collector_public_key_ids": collector_ids,
        "credential_migration_envelope_sha256": hashlib.sha256(
            migration_raw
        ).hexdigest(),
        "direct_iam_identity_authority_sha256": hashlib.sha256(
            authority_raw
        ).hexdigest(),
        "pre_foundation_authority_sha256": inventory[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": inventory[
            "foundation_apply_receipt_sha256"
        ],
        "project_ancestry_evidence_sha256": "c" * 64,
        "project_ancestry_chain_sha256": "d" * 64,
        "resource_ancestor_chain": inventory["resource_ancestor_chain"],
        "interpreter_image": {
            "project": "debian-cloud",
            "image_name": "debian-12-bookworm-v20260609",
            "image_numeric_id": "1234567890123456789",
            "image_self_link": (
                "https://www.googleapis.com/compute/v1/projects/"
                "debian-cloud/global/images/debian-12-bookworm-v20260609"
            ),
            "python_version": package.PYTHON_VERSION,
            "interpreter_sha256": "9" * 64,
        },
        "release_attestation": {
            "purpose": trust.ATTESTATION_PURPOSE,
            "attested_at_unix": 1_800_000_000,
        },
        "signer_key_id": hashlib.sha256(public_key).hexdigest(),
    }
    signature = private_key.sign(foundation.canonical_json_bytes(unsigned))
    manifest = {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }
    manifest_raw = foundation.canonical_json_bytes(manifest)
    manifest_path = tmp_path / "release-trust.json"
    _write(manifest_path, manifest_raw)
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(public_key).hexdigest(),
    )
    return replace(
        spec,
        trust_manifest_path=manifest_path,
        trust_public_key_path=public_key_path,
        network_collector_public_key_path=collector_paths["network"],
        cloud_collector_public_key_path=collector_paths["cloud"],
        host_collector_public_key_path=collector_paths["host"],
        credential_migration_envelope_path=migration_path,
        direct_iam_identity_authority_path=authority_path,
    )


def test_package_hashes_complete_runtime_closure_and_entrypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    wheel_root, wheel_manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        wheel_manifest["wheels"], wheel_manifest["bootstrap_pip"]
    )
    manifest = package.build_manifest(_trusted_spec(
        source=source,
        wheel_root=wheel_root,
        wheel_manifest=wheel_manifest,
        runtime_lock=runtime_lock,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    ))
    assert manifest["runtime_source_closure"] == [
        "scripts/__init__.py",
        "scripts/canary/__init__.py",
        "scripts/canary/passkey_v2_service.py",
        "scripts/canary/passkey_v2_sqlite.py",
    ]
    assert "scripts/canary/passkey_v2_store.py" not in manifest["runtime_source_closure"]
    assert manifest["pre_foundation_authority_sha256"] == "a" * 64
    assert manifest["foundation_apply_receipt_sha256"] == "b" * 64
    assert manifest["project_ancestry_evidence_sha256"] == "c" * 64
    assert manifest["project_ancestry_chain_sha256"] == "d" * 64
    assert manifest["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    release_paths = {item["release_relative"] for item in manifest["payloads"]}
    for entrypoint in package.REQUIRED_ENTRYPOINTS:
        assert entrypoint in release_paths
    assert manifest["offline_bootstrap"] is True
    assert manifest["network_install_required"] is False
    assert manifest["interpreter_version"] == "3.11.2"
    assert manifest["source_tree_oid"] == "b" * 40
    assert manifest["caller_self_hash_is_authority"] is False
    assert manifest["activation_performed"] is False
    assert manifest["cloud_mutation_performed"] is False


@pytest.mark.parametrize(
    "field",
    (
        "pre_foundation_authority_sha256",
        "foundation_apply_receipt_sha256",
        "resource_ancestor_chain",
    ),
)
def test_release_trust_rejects_foundation_chain_inventory_drift(field: str) -> None:
    inventory = {
        "release_revision": REVISION,
        "source_tree_oid": "b" * 40,
        "interpreter_sha256": "9" * 64,
        "direct_iam_identity_authority_sha256": "8" * 64,
        "pre_foundation_authority_sha256": "a" * 64,
        "foundation_apply_receipt_sha256": "c" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
    }
    authority = {
        "release_revision": REVISION,
        "source_tree_oid": "b" * 40,
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "interpreter_image": {"interpreter_sha256": "9" * 64},
        "direct_iam_identity_authority_sha256": "8" * 64,
        "pre_foundation_authority_sha256": "a" * 64,
        "foundation_apply_receipt_sha256": "c" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
    }
    drifted = dict(inventory)
    drifted[field] = (
        ["organizations/999999999999"]
        if field == "resource_ancestor_chain"
        else "d" * 64
    )
    authority["package_inventory_sha256"] = foundation.sha256_json(drifted)

    with pytest.raises(
        trust.OwnerGateTrustError,
        match="owner_gate_trust_inventory_mismatch",
    ):
        trust.verify_inventory_authority(authority, inventory=drifted)


def test_inventory_rejects_mutation_after_initial_clean_git_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, runtime = _git_race_spec(tmp_path, monkeypatch)
    verify = package._verify_clean_git_source
    first = True

    def mutate_after_verify(root: Path):
        nonlocal first
        result = verify(root)
        if first:
            first = False
            runtime.write_bytes(b"VALUE = 2\n")
        return result

    monkeypatch.setattr(package, "_verify_clean_git_source", mutate_after_verify)

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_worktree_object_mismatch",
    ):
        package.build_inventory(spec)


def test_inventory_final_git_recheck_rejects_mutation_after_payload_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, runtime = _git_race_spec(tmp_path, monkeypatch)
    require_match = package._require_worktree_matches_git_blob
    mutated = False

    def mutate_after_match(path: Path, *, expected: bytes, git_mode: str):
        nonlocal mutated
        require_match(path, expected=expected, git_mode=git_mode)
        if not mutated:
            mutated = True
            runtime.write_bytes(b"VALUE = 3\n")

    monkeypatch.setattr(
        package,
        "_require_worktree_matches_git_blob",
        mutate_after_match,
    )

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_git_source_not_exact",
    ):
        package.build_inventory(spec)


def test_resolved_source_closure_imports_in_isolated_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    closure = package.resolve_runtime_source_closure(
        source,
        release_revision=REVISION,
    )
    assert "scripts/canary/passkey_v2_sqlite.py" in closure
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(source)!r});"
        "import scripts.canary.passkey_v2_service as service;"
        "assert service.VALUE == 7"
    )
    completed = subprocess.run(
        ("python3", "-I", "-B", "-c", code),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")


def test_wheelhouse_accepts_debian12_supported_mixed_manylinux_tags(
    tmp_path: Path,
) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    verified = package.validate_wheelhouse(
        root=root,
        manifest=manifest,
        runtime_lock=runtime_lock,
    )
    cbor2 = next(item for item in verified if item["project"] == "cbor2")
    assert cbor2["filename"].endswith("-manylinux_2_28_x86_64.whl")


def test_wheelhouse_rejects_platform_newer_than_pinned_debian12_contract(
    tmp_path: Path,
) -> None:
    root, manifest = _wheelhouse(
        tmp_path,
        compiled_platforms={"cbor2": "manylinux_2_37_x86_64"},
    )
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_runtime_lock_invalid",
    ):
        package.validate_wheelhouse(
            root=root,
            manifest=manifest,
            runtime_lock=runtime_lock,
        )


def test_wheelhouse_rejects_missing_transitive_dependency(tmp_path: Path) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    unsigned["wheels"] = unsigned["wheels"][:-1]
    broken = {**unsigned, "manifest_sha256": foundation.sha256_json(unsigned)}
    with pytest.raises(package.OwnerGatePackageError):
        package.validate_wheelhouse(
            root=root,
            manifest=broken,
            runtime_lock=runtime_lock,
        )


def test_wheelhouse_rejects_pure_python_claim_for_binary_project(tmp_path: Path) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    item = next(
        wheel for wheel in unsigned["wheels"] if wheel["project"] == "cryptography"
    )
    old_path = root / item["filename"]
    payload = old_path.read_bytes()
    filename = f"cryptography-{item['version']}-py3-none-any.whl"
    _write(root / filename, payload)
    item["filename"] = filename
    item["sha256"] = hashlib.sha256(payload).hexdigest()
    unsigned["wheels"] = list(unsigned["wheels"])
    broken = {**unsigned, "manifest_sha256": foundation.sha256_json(unsigned)}
    with pytest.raises(package.OwnerGatePackageError):
        package.validate_wheelhouse(
            root=root,
            manifest=broken,
            runtime_lock=runtime_lock,
        )


def test_wheelhouse_rejects_second_wheel_for_same_normalized_project(
    tmp_path: Path,
) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    unsigned = copy.deepcopy({
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    })
    original = next(
        wheel
        for wheel in unsigned["wheels"]
        if wheel["project"] == "typing-extensions"
    )
    duplicate = dict(original)
    duplicate["filename"] = (
        f"typing_extensions_duplicate-{duplicate['version']}-py3-none-any.whl"
    )
    duplicate["project"] = "typing_extensions"
    shutil.copy2(root / original["filename"], root / duplicate["filename"])
    unsigned["wheels"].append(duplicate)
    broken = {**unsigned, "manifest_sha256": foundation.sha256_json(unsigned)}

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_project_invalid",
    ):
        package.validate_wheelhouse(
            root=root,
            manifest=broken,
            runtime_lock=runtime_lock,
        )


def test_wheel_archive_rejects_symlink_member(tmp_path: Path) -> None:
    path = tmp_path / "idna-3.18-py3-none-any.whl"
    link = zipfile.ZipInfo("idna/escape")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    _minimal_idna_wheel(path, extra=link, extra_payload=b"../../etc/shadow")

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_archive_invalid",
    ):
        package._verify_wheel_archive(
            path, project="idna", version="3.18", compiled_wheel=False
        )


def test_wheel_archive_rejects_encrypted_flag(tmp_path: Path) -> None:
    path = tmp_path / "idna-3.18-py3-none-any.whl"
    _minimal_idna_wheel(path)
    raw = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        cursor = 0
        while (position := raw.find(signature, cursor)) >= 0:
            flags = int.from_bytes(raw[position + flag_offset:position + flag_offset + 2], "little")
            raw[position + flag_offset:position + flag_offset + 2] = (
                flags | 0x1
            ).to_bytes(2, "little")
            cursor = position + 4
    path.chmod(0o644)
    path.write_bytes(raw)
    path.chmod(0o444)

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_archive_invalid",
    ):
        package._verify_wheel_archive(
            path, project="idna", version="3.18", compiled_wheel=False
        )


def test_wheel_archive_rejects_excessive_compression_ratio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "idna-3.18-py3-none-any.whl"
    payload = zipfile.ZipInfo("idna/compressed.bin")
    payload.compress_type = zipfile.ZIP_DEFLATED
    _minimal_idna_wheel(
        path,
        extra=payload,
        extra_payload=b"0" * (1024 * 1024),
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_archive_invalid",
    ):
        package._verify_wheel_archive(
            path, project="idna", version="3.18", compiled_wheel=False
        )


def test_wheel_archive_rejects_entry_count_and_metadata_memory_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "idna-3.18-py3-none-any.whl"
    _minimal_idna_wheel(path)
    monkeypatch.setattr(package, "_MAX_WHEEL_ENTRIES", 2)
    with pytest.raises(package.OwnerGatePackageError):
        package._verify_wheel_archive(
            path, project="idna", version="3.18", compiled_wheel=False
        )
    monkeypatch.setattr(package, "_MAX_WHEEL_ENTRIES", 10)
    monkeypatch.setattr(package, "_MAX_WHEEL_METADATA_BYTES", 8)
    with pytest.raises(package.OwnerGatePackageError):
        package._verify_wheel_archive(
            path, project="idna", version="3.18", compiled_wheel=False
        )


@pytest.mark.parametrize(
    "member",
    (
        "idna.pth",
        "idna.data/purelib/nested/evil.pth",
        "idna.data/platlib/sitecustomize.py",
        "usercustomize/__init__.py",
        "sitecustomize.cpython-311-x86_64-linux-gnu.so",
        "idna.data/scripts/launcher",
        "idna.data/data/escape",
        "idna.data/headers/escape.h",
    ),
)
def test_wheel_archive_rejects_startup_and_external_install_members(
    tmp_path: Path,
    member: str,
) -> None:
    path = tmp_path / "idna-3.18-py3-none-any.whl"
    _minimal_idna_wheel(path, extra=member, extra_payload=b"malicious")

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_archive_invalid",
    ):
        package._verify_wheel_archive(
            path,
            project="idna",
            version="3.18",
            compiled_wheel=False,
        )


def test_canonical_runtime_lock_rejects_tamper_and_incomplete_closure() -> None:
    path = ROOT / package.RUNTIME_LOCK_RELATIVE
    raw = path.read_bytes()
    assert package.decode_runtime_lock(raw) == package.LOCAL_RUNTIME_LOCK

    changed = copy.deepcopy(package.LOCAL_RUNTIME_LOCK)
    changed["wheels"][0]["active_dependencies"] = ["missing-project"]
    unsigned = {
        key: value for key, value in changed.items() if key != "lock_sha256"
    }
    changed["lock_sha256"] = foundation.sha256_json(unsigned)
    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_runtime_lock_invalid",
    ):
        package.decode_runtime_lock(
            foundation.canonical_json_bytes(changed) + b"\n"
        )

    tampered = bytearray(raw)
    tampered[-3] = ord("0") if tampered[-3] != ord("0") else ord("1")
    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_runtime_lock_invalid",
    ):
        package.decode_runtime_lock(bytes(tampered))


def test_wheelhouse_manifest_binds_runtime_lock_file_digest(
    tmp_path: Path,
) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    unsigned["runtime_lock_sha256"] = "0" * 64
    broken = {**unsigned, "manifest_sha256": foundation.sha256_json(unsigned)}

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheelhouse_manifest_invalid",
    ):
        package.validate_wheelhouse(
            root=root,
            manifest=broken,
            runtime_lock=runtime_lock,
        )


def test_wheelhouse_rejects_bootstrap_pip_byte_or_manifest_tamper(
    tmp_path: Path,
) -> None:
    root, manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    bootstrap_path = root / manifest["bootstrap_pip"]["filename"]
    bootstrap_path.chmod(0o644)
    bootstrap_path.write_bytes(b"tampered bootstrap pip wheel")
    bootstrap_path.chmod(0o444)

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheel_digest_invalid",
    ):
        package.validate_wheelhouse(
            root=root,
            manifest=manifest,
            runtime_lock=runtime_lock,
        )

    second = tmp_path / "second"
    second.mkdir()
    root, manifest = _wheelhouse(second)
    runtime_lock = _runtime_lock_for_wheels(
        manifest["wheels"], manifest["bootstrap_pip"]
    )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    unsigned["bootstrap_pip"] = {
        **unsigned["bootstrap_pip"],
        "sha256": "0" * 64,
    }
    changed = {
        **unsigned,
        "manifest_sha256": foundation.sha256_json(unsigned),
    }
    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_wheelhouse_lock_mismatch",
    ):
        package.validate_wheelhouse(
            root=root,
            manifest=changed,
            runtime_lock=runtime_lock,
        )


def test_inventory_rejects_duplicate_source_relative_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    wheel_root, wheel_manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        wheel_manifest["wheels"], wheel_manifest["bootstrap_pip"]
    )
    _install_test_runtime_lock(source, runtime_lock)
    duplicate = package.REQUIRED_ASSET_FILES[0]
    monkeypatch.setattr(
        package,
        "REQUIRED_ASSET_FILES",
        (*package.REQUIRED_ASSET_FILES, duplicate),
    )
    spec = package.PackageSpec(
        source_root=source,
        release_revision=REVISION,
        wheelhouse_root=wheel_root,
        wheelhouse_manifest=wheel_manifest,
        interpreter_sha256="9" * 64,
        direct_iam_identity_authority_path=_stub_direct_authority(
            tmp_path, monkeypatch
        ),
    )

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_payload_path_duplicate",
    ):
        package.build_inventory(spec)


def test_inventory_rejects_duplicate_release_relative_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    wheel_root, wheel_manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        wheel_manifest["wheels"], wheel_manifest["bootstrap_pip"]
    )
    _install_test_runtime_lock(source, runtime_lock)
    _write(source / "bin/collision", b"runtime")
    _write(source / "ops/muncho/owner-gate/bin/collision", b"entrypoint")
    monkeypatch.setattr(
        package,
        "resolve_runtime_source_closure",
        lambda _root, *, release_revision: ("bin/collision",),
    )
    monkeypatch.setattr(
        package,
        "REQUIRED_ASSET_FILES",
        ("ops/muncho/owner-gate/bin/collision",),
    )
    spec = package.PackageSpec(
        source_root=source,
        release_revision=REVISION,
        wheelhouse_root=wheel_root,
        wheelhouse_manifest=wheel_manifest,
        interpreter_sha256="9" * 64,
        direct_iam_identity_authority_path=_stub_direct_authority(
            tmp_path, monkeypatch
        ),
    )

    with pytest.raises(
        package.OwnerGatePackageError,
        match="owner_gate_package_payload_path_duplicate",
    ):
        package.build_inventory(spec)


def test_descriptor_hash_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    target.chmod(0o444)
    link = tmp_path / "link"
    os.symlink(target, link)
    with pytest.raises(package.OwnerGatePackageError):
        package._sha256_file(link)


def test_materialized_bundle_contains_only_verified_fixed_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, monkeypatch)
    wheel_root, wheel_manifest = _wheelhouse(tmp_path)
    runtime_lock = _runtime_lock_for_wheels(
        wheel_manifest["wheels"], wheel_manifest["bootstrap_pip"]
    )
    spec = _trusted_spec(
        source=source,
        wheel_root=wheel_root,
        wheel_manifest=wheel_manifest,
        runtime_lock=runtime_lock,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    destination = tmp_path / "bundle"
    manifest = package.materialize_bundle(spec, destination=destination)
    assert (destination / "package-manifest.json").is_file()
    assert (destination / "trust/release-trust.json").is_file()
    assert (destination / "migration/credential.json").is_file()
    assert len(tuple((destination / "wheels").glob("*.whl"))) == len(
        package.REQUIRED_PROJECTS
    )
    bootstrap_pip = manifest["bootstrap_pip"]
    assert package._sha256_file(
        destination / "bootstrap" / bootstrap_pip["filename"]
    ) == (bootstrap_pip["sha256"], bootstrap_pip["size"])
    assert package._sha256_file(destination / "package-manifest.json")[0] == (
        hashlib.sha256(foundation.canonical_json_bytes(manifest)).hexdigest()
    )
    assert not (destination / "payload/scripts/canary/passkey_v2_store.py").exists()

    # Hand the actual materialized package layout across the Stage0 boundary.
    # Production signing keys intentionally remain unavailable to this test;
    # _trusted_spec supplies equivalent ephemeral trust for package assembly.
    assert stage0._verify_runtime_lock_payload(
        destination,
        manifest,
        expected_uid=os.getuid(),
    ) == runtime_lock
    stage0.validate_wheel_archives_for_install(
        destination,
        manifest=manifest,
        expected_uid=os.getuid(),
    )
