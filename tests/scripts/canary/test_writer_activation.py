from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.canonical_writer_bootstrap import (
    CanonicalWriterServiceConfig,
    DiscordEdgeWriterAuthorityConfig,
)
from gateway.canonical_writer_boundary import (
    DEFAULT_GATEWAY_UNIT,
    DEFAULT_SOCKET_PATH,
)
from gateway.canonical_writer_db import (
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    WriterDBConfig,
    WriterPrivilegePolicy,
)
from gateway.canonical_writer_root_collector import (
    TrustedDeploymentManifest,
    snapshot_policy_sha256,
)
from gateway import canonical_writer_host_authority as host_authority
from gateway.canonical_writer_activation import (
    ActivationPaths as PackagedActivationPaths,
    ActivationPlan as PackagedActivationPlan,
)
from gateway.canonical_writer_deployment_preflight import evaluate_snapshot
from scripts.canary import writer_activation
from scripts.canary.writer_activation import (
    ActivationDigests,
    ActivationPaths,
    ExternalNativeExecutableMapping,
    HostNumericIdentities,
    PreapprovedNativeExecutablePolicy,
    build_activation_preview as build_activation_plan,
    build_native_observation_plan,
    write_root_activation_preview as write_root_activation_plan,
    write_root_collector_manifest_preview as write_root_collector_manifest,
)
from scripts.canary.writer_release import (
    ReleaseManifest,
    TreeEntry,
    WriterOnlyUnitSpec,
    render_systemd_units,
)


REVISION = "a" * 40
SQL_PRIVATE_IP = "10.91.0.3"
SQL_TLS_SERVER_NAME = "db.muncho.internal"
APPROVED_PLAN_SHA256 = "1" * 64
WRITER_CONFIG_SHA256 = "2" * 64
GATEWAY_CONFIG_SHA256 = "3" * 64
NATIVE_A = ExternalNativeExecutableMapping(
    path="/usr/lib/muncho-reviewed/native-a.so",
    sha256="6" * 64,
)
NATIVE_B = ExternalNativeExecutableMapping(
    path="/usr/lib/muncho-reviewed/native-b.so",
    sha256="7" * 64,
)


def _file_entry(path: str, payload: bytes, *, mode: str = "0444") -> TreeEntry:
    return TreeEntry(
        path=path,
        kind="file",
        mode=mode,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _release(*, extra_entries: tuple[TreeEntry, ...] = ()) -> ReleaseManifest:
    root = Path("/opt/muncho-canary-releases") / REVISION
    stdlib = "python/cpython-3.11.15-linux-x86_64/lib/python3.11"
    site = "venv/lib/python3.11/site-packages"
    entries = (
        TreeEntry("python", "directory", "0555"),
        TreeEntry(
            "python/cpython-3.11.15-linux-x86_64",
            "directory",
            "0555",
        ),
        TreeEntry(
            "python/cpython-3.11.15-linux-x86_64/lib",
            "directory",
            "0555",
        ),
        TreeEntry(stdlib, "directory", "0555"),
        _file_entry(f"{stdlib}/os.py", b"sealed-stdlib"),
        TreeEntry("venv", "directory", "0555"),
        TreeEntry("venv/bin", "directory", "0555"),
        _file_entry("venv/bin/python", b"copied-python", mode="0555"),
        TreeEntry("venv/lib", "directory", "0555"),
        TreeEntry("venv/lib/python3.11", "directory", "0555"),
        TreeEntry(site, "directory", "0555"),
        TreeEntry(f"{site}/gateway", "directory", "0555"),
        _file_entry(
            f"{site}/gateway/canonical_writer_bootstrap.py",
            b"sealed-writer",
        ),
        _file_entry(
            f"{site}/gateway/canonical_writer_gateway_bootstrap.py",
            b"sealed-gateway",
        ),
        *extra_entries,
    )
    entries = tuple(sorted(entries, key=lambda entry: entry.path))
    provisional = ReleaseManifest(
        revision=REVISION,
        artifact_root=str(root),
        python_version="3.11.15",
        interpreter=str(root / "venv/bin/python"),
        writer_module_origin=str(
            root / "venv/lib/python3.11/site-packages/gateway/"
            "canonical_writer_bootstrap.py"
        ),
        gateway_module_origin=str(
            root / "venv/lib/python3.11/site-packages/gateway/"
            "canonical_writer_gateway_bootstrap.py"
        ),
        entries=entries,
        artifact_sha256="",
    )
    return replace(
        provisional,
        artifact_sha256=provisional.computed_artifact_sha256,
    )


def _identities() -> HostNumericIdentities:
    return HostNumericIdentities(
        gateway_uid=2101,
        gateway_gid=3101,
        gateway_home="/var/lib/hermes-gateway",
        writer_uid=2102,
        writer_gid=3102,
        writer_home="/nonexistent",
        socket_client_gid=3103,
        projector_uid=2103,
        projector_gid=3104,
        projector_home="/nonexistent",
        gateway_supplementary_gids=(3101, 3103),
        writer_supplementary_gids=(3102, 3104),
    )


def _canary_identities() -> HostNumericIdentities:
    return HostNumericIdentities(
        gateway_uid=993,
        gateway_gid=992,
        gateway_home="/var/lib/hermes-gateway",
        writer_uid=999,
        writer_gid=994,
        writer_home="/nonexistent",
        socket_client_gid=990,
        projector_uid=992,
        projector_gid=991,
        projector_home="/nonexistent",
        gateway_supplementary_gids=(990, 992),
        writer_supplementary_gids=(991, 994),
    )


def _canary_writer_config() -> CanonicalWriterServiceConfig:
    config = _writer_config()
    identities = _canary_identities()
    return replace(
        config,
        gateway_uid=identities.gateway_uid,
        writer_uid=identities.writer_uid,
        writer_gid=identities.writer_gid,
        socket_gid=identities.socket_client_gid,
        projector_gid=identities.projector_gid,
        database=replace(
            config.database,
            ca_file=Path("/etc/muncho/trust/cloudsql-server-ca.pem"),
            credential=replace(
                config.database.credential,
                expected_uid=identities.writer_uid,
                expected_gid=identities.writer_gid,
            ),
        ),
    )


def _stopped_native_receipt():
    release = _release()
    unit_spec = _unit_spec()
    digests = _digests(release, unit_spec)
    plan = build_native_observation_plan(
        release,
        unit_spec,
        _canary_writer_config(),
        _canary_identities(),
        digests,
        observation_id="11111111-1111-4111-8111-111111111111",
        boot_id_sha256="b" * 64,
        host_identity_sha256="c" * 64,
        release_manifest_file_sha256="d" * 64,
        database_ca_sha256="e" * 64,
        external_iam_policy_sha256="f" * 64,
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
    )
    observed = 1_000_000_000

    def live(label, pid):
        identities = plan.value["identities"]
        return {
            "unit_name": plan.value[f"{label}_unit"]["name"],
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "disabled",
            "main_pid": pid,
            "start_time_ticks": pid + 100,
            "argv": list(plan.value[f"{label}_argv"]),
            "external_native_mappings": [
                {
                    "path": "/usr/lib/x86_64-linux-gnu/libc.so.6",
                    "sha256": "a" * 64,
                }
            ],
            "kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
            "process_authority": {
                "pid": pid,
                "process_start_time_ticks": pid + 100,
                "effective_uid": identities[f"{label}_uid"],
                "effective_gid": identities[f"{label}_gid"],
                "supplementary_gids": identities[f"{label}_supplementary_gids"],
                "no_new_privileges": True,
                "effective_capabilities": [],
                "executable": plan.value[f"{label}_argv"][0],
            },
        }

    discord = {
        "unit_name": "muncho-discord-egress.service",
        "unit_exists": False,
        "unit_enabled": False,
        "unit_active": False,
        "main_pid": 0,
        "config_exists": False,
        "token_exists": False,
        "socket_exists": False,
        "process_pids": [],
    }
    observation = {
        "boot_id_sha256": "b" * 64,
        "host_identity_sha256": "c" * 64,
        "observed_at_unix": 2_000_000_000,
        "observed_at_boottime_ns": observed,
        "expires_at_boottime_ns": observed
        + host_authority.NATIVE_OBSERVATION_TTL_SECONDS * 1_000_000_000,
        "gateway_service": live("gateway", 4242),
        "writer_service": live("writer", 4343),
        "discord_absence": discord,
        "legacy_helper_absence": {
            "path": str(host_authority.LEGACY_CLOUD_SQL_HELPER_PATH),
            "file_exists": False,
            "file_symlink": False,
            "parent_exists": False,
            "parent_symlink": False,
            "gateway_access": {"read": False, "write": False, "execute": False},
        },
    }

    def stopped(label):
        return {
            "unit_name": plan.value[f"{label}_unit"]["name"],
            "load_state": "loaded",
            "active_state": "inactive",
            "sub_state": "dead",
            "unit_file_state": "disabled",
            "main_pid": 0,
        }

    return host_authority.NativeObservationReceipt.from_mapping({
        "schema": host_authority.NATIVE_OBSERVATION_RECEIPT_SCHEMA,
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": "1" * 64,
        "host_preparation_receipt_sha256": "2" * 64,
        "external_iam_receipt_sha256": "3" * 64,
        "plan": plan.to_mapping(),
        "observation": observation,
        "final_state": {
            "boot_id_sha256": "b" * 64,
            "host_identity_sha256": "c" * 64,
            "finalized_at_unix": 2_000_000_001,
            "finalized_at_boottime_ns": observed + 1_000_000_000,
            "gateway_service": stopped("gateway"),
            "writer_service": stopped("writer"),
            "discord_absence": discord,
        },
    })


def _hba_receipt() -> ManagedCloudSQLAdminHBAReceipt:
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=SQL_PRIVATE_IP,
        tls_server_name=SQL_TLS_SERVER_NAME,
        port=5432,
        server_certificate_sha256="4" * 64,
        database="cloudsqladmin",
        user="canonical_writer",
        observed_at_unix=2_000_000_000,
        expires_at_unix=2_000_000_300,
        sqlstate="28000",
        server_message=(
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_writer", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _writer_config() -> CanonicalWriterServiceConfig:
    identities = _identities()
    receipt = _hba_receipt()
    return CanonicalWriterServiceConfig(
        socket_path=DEFAULT_SOCKET_PATH,
        gateway_unit=DEFAULT_GATEWAY_UNIT,
        gateway_uid=identities.gateway_uid,
        writer_uid=identities.writer_uid,
        writer_gid=identities.writer_gid,
        socket_gid=identities.socket_client_gid,
        projector_gid=identities.projector_gid,
        owner_discord_user_ids=frozenset({"42"}),
        connection_timeout_seconds=5.0,
        max_connections=4,
        database=WriterDBConfig(
            host=SQL_PRIVATE_IP,
            tls_server_name=SQL_TLS_SERVER_NAME,
            port=5432,
            database="muncho_canary_brain",
            user="canonical_writer",
            ca_file=Path("/etc/muncho-canonical-writer/server-ca.pem"),
            credential=CredentialSource(
                path=Path("/etc/muncho-canonical-writer-credentials/database-password"),
                expected_uid=identities.writer_uid,
                expected_gid=identities.writer_gid,
            ),
        ),
        privileges=WriterPrivilegePolicy(
            schema="canonical_brain",
            private_schema_identity_sha256="5" * 64,
            managed_cloudsqladmin_hba_rejection_receipt=receipt,
            managed_cloudsqladmin_hba_rejection_sha256=receipt.sha256,
        ),
        discord_edge_authority=DiscordEdgeWriterAuthorityConfig(
            enabled=False,
            capability_private_key_file=None,
            edge_receipt_public_key_file=None,
            edge_receipt_public_key_id="",
            request_timeout_seconds=5,
            capability_private_key=None,
            edge_receipt_public_key=None,
        ),
    )


def _unit_spec() -> WriterOnlyUnitSpec:
    return WriterOnlyUnitSpec(database_ip_allow=(f"{SQL_PRIVATE_IP}/32",))


def _native_policy() -> PreapprovedNativeExecutablePolicy:
    return PreapprovedNativeExecutablePolicy(
        writer=(NATIVE_A, NATIVE_B),
        gateway=(NATIVE_A, NATIVE_B),
    )


def _digests(
    release: ReleaseManifest | None = None,
    unit_spec: WriterOnlyUnitSpec | None = None,
) -> ActivationDigests:
    bundle = render_systemd_units(
        release or _release(),
        unit_spec or _unit_spec(),
    )
    return ActivationDigests(
        approved_plan_sha256=APPROVED_PLAN_SHA256,
        writer_unit_sha256=hashlib.sha256(
            bundle.writer_service.encode("utf-8")
        ).hexdigest(),
        gateway_unit_sha256=hashlib.sha256(
            bundle.gateway_service.encode("utf-8")
        ).hexdigest(),
        exporter_unit_sha256=hashlib.sha256(
            bundle.exporter_service.encode("utf-8")
        ).hexdigest(),
        tmpfiles_sha256=hashlib.sha256(bundle.tmpfiles.encode("utf-8")).hexdigest(),
        writer_config_sha256=WRITER_CONFIG_SHA256,
        gateway_config_sha256=GATEWAY_CONFIG_SHA256,
    )


def _plan():
    release = _release()
    unit_spec = _unit_spec()
    return build_activation_plan(
        release,
        unit_spec,
        _writer_config(),
        _identities(),
        _digests(release, unit_spec),
        native_executable_policy=_native_policy(),
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
    )


def _activation_snapshot_with_live_code_closure():
    snapshot = json.loads(json.dumps(_plan().deployment_manifest["snapshot_template"]))
    collected_at_unix = 1_800_000_000
    snapshot["collected_at_unix"] = collected_at_unix

    for deployment_name, pid, module_origin_key in (
        ("writer_deployment", 4343, "bootstrap_module_origin"),
        ("gateway_deployment", 4242, "entry_module_origin"),
    ):
        deployment = snapshot[deployment_name]
        policy = deployment["policy"]
        import_paths = policy["import_paths"]
        interpreter = next(
            item for item in import_paths if item["kind"] == "interpreter"
        )
        approved_native_paths = [
            item["path"]
            for item in policy["preapproved_external_native_executable_mappings"]
        ]
        process = {
            "complete": True,
            "observed_at_unix": collected_at_unix,
            "pid": pid,
            "systemd_main_pid": pid,
            "process_start_time_ticks": 123_456 + pid,
            "systemd_main_pid_start_time_ticks": 123_456 + pid,
            "unit_name": policy["unit_name"],
            "cmdline": policy["exec_start"],
            "executable_path": policy["interpreter"],
            "executable_digest_sha256": interpreter["digest_sha256"],
            "revision": policy["revision"],
            "artifact_digest_sha256": policy["artifact_digest_sha256"],
            module_origin_key: policy["module_origin"],
            "effective_import_paths": sorted(
                item["path"]
                for item in import_paths
                if item["kind"] in {"site_packages", "stdlib"}
            ),
            "loaded_module_origins_complete": True,
            "loaded_module_origins": [policy["module_origin"]],
            "mapped_executable_paths_complete": True,
            "mapped_executable_paths": [
                policy["interpreter"],
                *approved_native_paths,
            ],
            "kernel_executable_mappings": list(
                policy["preapproved_kernel_executable_mappings"]
            ),
            "unexpected_import_origins": [],
            "deleted_code_mappings": [],
            "writable_code_mappings": [],
        }
        if deployment_name == "writer_deployment":
            deployment["attestation"] = {
                "complete": False,
                "artifact": {},
                "import_closure": {},
                "unit": {},
                "mounts": {},
                "process": process,
            }
        else:
            process.update({
                "dynamic_python_discovery_complete": True,
                "dynamic_python_loading_mode": "disabled",
                "dynamic_python_discovery_paths": [],
                "dynamic_python_loaded_origins": [],
                "dynamic_python_writable_paths": [],
            })
            deployment["attestation"] = {
                "complete": False,
                "unit": {},
                "mounts": {},
                "process": process,
            }
    return snapshot


def _checks_by_name(snapshot):
    return {check.name: check for check in evaluate_snapshot(snapshot).checks}


def test_preview_binds_release_host_and_sql_contracts_without_being_deployable():
    plan = _plan()
    snapshot = plan.deployment_manifest["snapshot_template"]
    host = plan.deployment_manifest["host_contract"]

    assert "approved_plan_sha256" not in plan.deployment_manifest
    assert plan.deployment_manifest["revision"] == REVISION
    assert plan.deployment_manifest["snapshot_policy_sha256"] == snapshot_policy_sha256(
        snapshot
    )
    assert plan.schema == "muncho-writer-only-activation-preview.v1"
    with pytest.raises(ValueError, match="manifest schema"):
        TrustedDeploymentManifest.from_mapping(plan.deployment_manifest)
    assert plan.sql_private_ip == SQL_PRIVATE_IP
    assert plan.sql_tls_server_name == SQL_TLS_SERVER_NAME
    assert snapshot["database"]["connection"] == {
        "host": SQL_PRIVATE_IP,
        "tls_server_name": SQL_TLS_SERVER_NAME,
        "port": 5432,
        "database": "muncho_canary_brain",
        "user": "canonical_writer",
    }
    assert snapshot["gateway_uid"] == _identities().gateway_uid
    assert snapshot["writer_uid"] == _identities().writer_uid
    assert snapshot["socket"]["expected_group_gid"] == (_identities().socket_client_gid)
    assert host["gateway_unit_fragment_sha256"] == (plan.digests.gateway_unit_sha256)
    assert host["writer_unit_fragment_sha256"] == (plan.digests.writer_unit_sha256)
    assert host["gateway_config_sha256"] == GATEWAY_CONFIG_SHA256
    assert host["writer_config_sha256"] == WRITER_CONFIG_SHA256
    assert (
        plan.unit_bundle.contract
        == render_systemd_units(_release(), _unit_spec()).contract
    )


def test_native_observation_plan_is_separate_discovery_only_and_digest_bound():
    release = _release()
    unit_spec = _unit_spec()
    digests = _digests(release, unit_spec)

    plan = build_native_observation_plan(
        release,
        unit_spec,
        _canary_writer_config(),
        _canary_identities(),
        digests,
        observation_id="11111111-1111-4111-8111-111111111111",
        boot_id_sha256="8" * 64,
        host_identity_sha256="9" * 64,
        release_manifest_file_sha256="a" * 64,
        database_ca_sha256="b" * 64,
        external_iam_policy_sha256="c" * 64,
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
    )

    value = plan.to_mapping()
    assert len(plan.sha256) == 64
    assert "external_native_mappings" not in value
    assert "deployment_manifest_sha256" not in value
    assert value["native_discovery_policy"] == {
        "allowed_roots": ["/usr/lib"],
        "allowed_kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
        "maximum_mappings": 256,
        "required_owner_uid": 0,
        "required_owner_gid": 0,
        "require_regular": True,
        "require_single_link": True,
        "forbid_symlink": True,
        "forbid_acl": True,
        "forbid_xattrs": True,
        "forbid_writable": True,
        "forbid_deleted": True,
        "exclude_artifact_root": True,
        "digest_algorithm": "sha256",
    }
    assert value["gateway_argv"] == [
        release.interpreter,
        "-I",
        "-m",
        "gateway.canonical_writer_gateway_bootstrap",
    ]
    assert value["writer_argv"] == [
        release.interpreter,
        "-I",
        "-m",
        "gateway.canonical_writer_bootstrap",
        "--config",
        "/etc/muncho-canonical-writer/writer.json",
    ]


def test_single_final_builder_round_trips_packaged_v3_from_stopped_receipt():
    release = _release()
    unit_spec = _unit_spec()
    digests = _digests(release, unit_spec)
    plan = writer_activation.build_activation_plan(
        release,
        unit_spec,
        _canary_writer_config(),
        _canary_identities(),
        _stopped_native_receipt(),
        writer_config_sha256=digests.writer_config_sha256,
        gateway_config_sha256=digests.gateway_config_sha256,
        release_manifest_file_sha256="d" * 64,
        database_ca_sha256="e" * 64,
        external_iam_policy_sha256="f" * 64,
        external_iam_receipt_path=(
            "/run/muncho-canonical-preflight/external-iam-receipt.json"
        ),
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
        paths=PackagedActivationPaths(),
    )
    encoded = json.dumps(
        plan.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
    )
    reparsed = PackagedActivationPlan.from_mapping(json.loads(encoded))

    assert plan.schema == "muncho-writer-only-activation-plan.v3"
    assert reparsed.to_mapping() == plan.to_mapping()
    assert reparsed.sha256 == plan.sha256


def test_activation_snapshot_flows_exact_linux_groups_to_packaged_preflight():
    snapshot = json.loads(json.dumps(_plan().deployment_manifest["snapshot_template"]))

    assert snapshot["gateway_supplementary_gids"] == [3101, 3103]
    assert snapshot["writer_supplementary_gids"] == [3102, 3104]
    checks = _checks_by_name(snapshot)
    assert checks["identity.gateway_socket_membership"].passed
    assert checks["identity.writer_projector_membership"].passed


@pytest.mark.parametrize(
    ("field", "missing_primary", "failed_check"),
    [
        (
            "gateway_supplementary_gids",
            [3103],
            "identity.gateway_socket_membership",
        ),
        (
            "writer_supplementary_gids",
            [3104],
            "identity.writer_projector_membership",
        ),
    ],
)
def test_packaged_preflight_rejects_linux_groups_without_primary_gid(
    field,
    missing_primary,
    failed_check,
):
    snapshot = json.loads(json.dumps(_plan().deployment_manifest["snapshot_template"]))
    snapshot[field] = missing_primary

    assert not _checks_by_name(snapshot)[failed_check].passed


def test_activation_native_mapping_policy_flows_to_packaged_code_closure():
    snapshot = _activation_snapshot_with_live_code_closure()
    expected = [
        NATIVE_A.to_mapping(_release().artifact_root),
        NATIVE_B.to_mapping(_release().artifact_root),
    ]

    assert (
        snapshot["writer_deployment"]["policy"][
            "preapproved_external_native_executable_mappings"
        ]
        == expected
    )
    assert (
        snapshot["gateway_deployment"]["policy"][
            "preapproved_external_native_executable_mappings"
        ]
        == expected
    )
    checks = _checks_by_name(snapshot)
    for name in (
        "writer_deployment.policy_valid",
        "writer_deployment.process_code_closure",
        "gateway_deployment.shared_immutable_release",
        "gateway_deployment.process_code_closure",
    ):
        assert checks[name].passed, checks[name].detail


def test_packaged_preflight_rejects_activation_native_mapping_hash_drift():
    snapshot = _activation_snapshot_with_live_code_closure()
    snapshot["writer_deployment"]["policy"][
        "preapproved_external_native_executable_mappings"
    ][0]["sha256"] = "A" * 64

    checks = _checks_by_name(snapshot)
    assert not checks["writer_deployment.policy_valid"].passed
    assert not checks["writer_deployment.process_code_closure"].passed


def test_plan_pins_minimal_immutable_import_roots_and_collector_argv():
    plan = _plan()
    snapshot = plan.deployment_manifest["snapshot_template"]
    writer_policy = snapshot["writer_deployment"]["policy"]
    gateway_policy = snapshot["gateway_deployment"]["policy"]
    import_paths = writer_policy["import_paths"]

    assert [item["kind"] for item in import_paths] == [
        "application",
        "interpreter",
        "stdlib",
        "site_packages",
    ]
    assert all(len(item["digest_sha256"]) == 64 for item in import_paths)
    assert gateway_policy["import_paths"] == import_paths
    assert gateway_policy["module"] == ("gateway.canonical_writer_gateway_bootstrap")
    assert gateway_policy["exec_start"] == [
        _release().interpreter,
        "-I",
        "-m",
        "gateway.canonical_writer_gateway_bootstrap",
    ]
    assert gateway_policy["read_write_paths"] == ["/run/hermes-cloud-gateway"]
    expected_native = [
        NATIVE_A.to_mapping(_release().artifact_root),
        NATIVE_B.to_mapping(_release().artifact_root),
    ]
    assert (
        writer_policy["preapproved_external_native_executable_mappings"]
        == expected_native
    )
    assert (
        gateway_policy["preapproved_external_native_executable_mappings"]
        == expected_native
    )
    mutated_snapshot = json.loads(json.dumps(snapshot))
    mutated_snapshot["writer_deployment"]["policy"][
        "preapproved_external_native_executable_mappings"
    ][0]["sha256"] = "8" * 64
    assert (
        snapshot_policy_sha256(mutated_snapshot)
        != (plan.deployment_manifest["snapshot_policy_sha256"])
    )
    assert writer_policy["bind_read_only_paths"] == [
        f"/opt/muncho-canary-releases/{REVISION}"
    ]
    assert plan.collector_argv == (
        f"/opt/muncho-canary-releases/{REVISION}/venv/bin/python",
        "-I",
        "-m",
        "gateway.canonical_writer_root_collector",
        "collect",
        "--manifest",
        "/etc/muncho/writer-activation/deployment-manifest.json",
        "--receipt",
        "/run/muncho-canonical-preflight/root-preflight.json",
    )
    assert plan.validator_argv[4] == "validate"
    assert all(
        item not in {"sh", "bash", "/bin/sh", "/bin/bash"}
        for item in (
            plan.collector_argv[0],
            plan.validator_argv[0],
        )
    )


def test_native_mapping_policy_is_mandatory_and_non_empty():
    release = _release()
    unit_spec = _unit_spec()
    positional = (
        release,
        unit_spec,
        _writer_config(),
        _identities(),
        _digests(release, unit_spec),
    )

    with pytest.raises(TypeError, match="native_executable_policy"):
        build_activation_plan(
            *positional,
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )
    with pytest.raises(TypeError, match="policy is required"):
        build_activation_plan(
            *positional,
            native_executable_policy=None,
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )
    for policy in (
        replace(_native_policy(), writer=()),
        replace(_native_policy(), gateway=()),
    ):
        with pytest.raises(ValueError, match="non-empty tuple"):
            build_activation_plan(
                *positional,
                native_executable_policy=policy,
                sql_private_ip=SQL_PRIVATE_IP,
                sql_tls_server_name=SQL_TLS_SERVER_NAME,
            )


@pytest.mark.parametrize(
    ("policy", "error"),
    [
        (
            replace(_native_policy(), writer=(NATIVE_A, NATIVE_A)),
            "paths must be unique",
        ),
        (
            replace(_native_policy(), gateway=(NATIVE_B, NATIVE_A)),
            "must be exactly sorted",
        ),
        (
            replace(
                _native_policy(),
                writer=(
                    ExternalNativeExecutableMapping(
                        path=(
                            f"/opt/muncho-canary-releases/{REVISION}/venv/lib/native.so"
                        ),
                        sha256="8" * 64,
                    ),
                ),
            ),
            "outside sealed release",
        ),
        (
            replace(
                _native_policy(),
                writer=(
                    ExternalNativeExecutableMapping(
                        path="usr/lib/native.so",
                        sha256="8" * 64,
                    ),
                ),
            ),
            "absolute normalized path",
        ),
        (
            replace(
                _native_policy(),
                gateway=(
                    ExternalNativeExecutableMapping(
                        path="/usr/lib/../lib/native.so",
                        sha256="8" * 64,
                    ),
                ),
            ),
            "absolute normalized path",
        ),
        (
            replace(
                _native_policy(),
                gateway=(
                    ExternalNativeExecutableMapping(
                        path="/usr/lib/native.so",
                        sha256="A" * 64,
                    ),
                ),
            ),
            "lowercase SHA-256",
        ),
    ],
)
def test_native_mapping_policy_rejects_ambiguous_or_malformed_entries(
    policy,
    error,
):
    release = _release()
    unit_spec = _unit_spec()

    with pytest.raises((TypeError, ValueError), match=error):
        build_activation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            _digests(release, unit_spec),
            native_executable_policy=policy,
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


def test_plan_contains_policy_and_paths_but_no_secret_values():
    plan = _plan()
    encoded = json.dumps(plan.to_mapping(), sort_keys=True)
    forbidden_keys = {
        "password",
        "password_value",
        "secret",
        "secret_value",
        "token",
        "api_key",
        "credential_value",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(plan.to_mapping())
    assert "database-password" not in encoded
    assert "credential_file" not in encoded
    assert "capability_private_key" not in encoded
    assert "edge_receipt_public_key" not in encoded
    assert plan.deployment_manifest["snapshot_template"]["discord_edge"] == {
        "gateway_enabled": False,
        "writer_authority_enabled": False,
        "unit_name": "muncho-discord-egress.service",
        "config_path": "/etc/muncho/discord-edge.json",
        "token_path": "/etc/muncho/discord-edge-credentials/bot-token",
        "socket_path": "/run/muncho-discord-egress/edge.sock",
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ({"sql_private_ip": "10.91.0.4"}, "plan digest"),
        (
            {"sql_tls_server_name": "other.muncho.internal"},
            "plan digest",
        ),
        (
            {"schema": "muncho-writer-only-activation-plan.v1"},
            "preview schema",
        ),
    ],
)
def test_plan_is_deterministic_and_self_digest_rejects_mutation(
    monkeypatch, mutation, error
):
    first = _plan()
    second = _plan()

    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64

    mutated = replace(first, **mutation)
    monkeypatch.setattr(writer_activation, "_effective_uid", lambda: 1000)
    with pytest.raises(ValueError, match=error):
        write_root_activation_plan(mutated)


def test_plan_rejects_reviewed_unit_digest_drift():
    release = _release()
    unit_spec = _unit_spec()
    digests = replace(
        _digests(release, unit_spec),
        writer_unit_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="unit digest"):
        build_activation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


@pytest.mark.parametrize(
    ("sql_ip", "unit_ip"),
    [
        ("127.0.0.1", "127.0.0.1/32"),
        ("10.91.0.3", "10.91.0.4/32"),
        ("2001:db8::1", "2001:db8::1/128"),
    ],
)
def test_plan_rejects_nonprivate_or_nonexact_sql_boundary(sql_ip, unit_ip):
    release = _release()
    unit_spec = WriterOnlyUnitSpec(database_ip_allow=(unit_ip,))

    with pytest.raises(ValueError, match="private IPv4|allow-list"):
        build_activation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            _digests(release, unit_spec),
            native_executable_policy=_native_policy(),
            sql_private_ip=sql_ip,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


def test_plan_rejects_writer_config_or_host_identity_drift():
    release = _release()
    unit_spec = _unit_spec()
    digests = _digests(release, unit_spec)
    config = _writer_config()

    with pytest.raises(ValueError, match="database authority"):
        build_activation_plan(
            release,
            unit_spec,
            replace(
                config,
                database=replace(config.database, host="10.91.0.4"),
            ),
            _identities(),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )

    with pytest.raises(ValueError, match="database authority"):
        build_activation_plan(
            release,
            unit_spec,
            replace(
                config,
                database=replace(
                    config.database,
                    tls_server_name="other.muncho.internal",
                ),
            ),
            _identities(),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )

    mismatched_receipt = replace(
        config.privileges.managed_cloudsqladmin_hba_rejection_receipt,
        tls_server_name="other.muncho.internal",
    )
    with pytest.raises(ValueError, match="HBA baseline"):
        build_activation_plan(
            release,
            unit_spec,
            replace(
                config,
                privileges=replace(
                    config.privileges,
                    managed_cloudsqladmin_hba_rejection_receipt=(mismatched_receipt),
                    managed_cloudsqladmin_hba_rejection_sha256=(
                        mismatched_receipt.sha256
                    ),
                ),
            ),
            _identities(),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )

    with pytest.raises(ValueError, match="host identities"):
        build_activation_plan(
            release,
            unit_spec,
            replace(config, gateway_uid=9999),
            _identities(),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )

    with pytest.raises(ValueError, match="socket-client groups"):
        build_activation_plan(
            release,
            unit_spec,
            config,
            replace(
                _identities(),
                gateway_supplementary_gids=(3101, 3103, 3104),
            ),
            digests,
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


@pytest.mark.parametrize(
    "tls_server_name",
    [" DB.MUNCHO.INTERNAL", "DB.MUNCHO.INTERNAL", "db.muncho.internal."],
)
def test_plan_rejects_nonexact_tls_server_name(tls_server_name):
    release = _release()
    unit_spec = _unit_spec()

    with pytest.raises(ValueError, match="TLS server name"):
        build_activation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            _digests(release, unit_spec),
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=tls_server_name,
        )


def test_plan_rejects_discord_authority_and_ambiguous_stdlib():
    release = _release()
    unit_spec = _unit_spec()
    config = _writer_config()

    with pytest.raises(ValueError, match="host identities"):
        build_activation_plan(
            release,
            unit_spec,
            replace(
                config,
                discord_edge_authority=replace(
                    config.discord_edge_authority,
                    enabled=True,
                ),
            ),
            _identities(),
            _digests(release, unit_spec),
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )

    ambiguous = _release(
        extra_entries=(
            TreeEntry(
                "python/second-runtime/lib/python3.11",
                "directory",
                "0555",
            ),
        )
    )
    with pytest.raises(ValueError, match="stdlib identity is ambiguous"):
        build_activation_plan(
            ambiguous,
            unit_spec,
            config,
            _identities(),
            _digests(ambiguous, unit_spec),
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


def test_activation_paths_are_production_pinned():
    with pytest.raises(ValueError, match="production-pinned"):
        build_activation_plan(
            _release(),
            _unit_spec(),
            _writer_config(),
            _identities(),
            _digests(),
            native_executable_policy=_native_policy(),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
            paths=replace(
                ActivationPaths(),
                manifest_path=Path("/tmp/deployment-manifest.json"),
            ),
        )


def test_public_root_writers_fail_before_touching_production_paths(monkeypatch):
    plan = _plan()
    calls = []
    monkeypatch.setattr(writer_activation, "_effective_uid", lambda: 1000)
    monkeypatch.setattr(
        writer_activation,
        "_validate_root_parent_chain",
        lambda _path: calls.append("parent"),
    )

    with pytest.raises(PermissionError, match="uid_0"):
        write_root_collector_manifest(plan)
    with pytest.raises(PermissionError, match="uid_0"):
        write_root_activation_plan(plan)

    assert calls == []


def test_strict_root_json_write_is_exclusive_sealed_and_canonical(
    tmp_path,
    monkeypatch,
):
    parent = tmp_path / "root-only"
    parent.mkdir(mode=0o700)
    target = parent / "manifest.json"
    real_lstat = os.lstat

    def trusted_lstat(path):
        item = real_lstat(path)
        return SimpleNamespace(
            st_mode=item.st_mode,
            st_uid=0,
            st_gid=0,
            st_nlink=item.st_nlink,
        )

    monkeypatch.setattr(writer_activation, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        writer_activation,
        "_validate_root_parent_chain",
        lambda _path: None,
    )
    monkeypatch.setattr(writer_activation.os, "fchown", lambda *_args: None)
    monkeypatch.setattr(writer_activation.os, "lstat", trusted_lstat)

    writer_activation._write_root_json(target, {"z": 1, "a": [2, 3]})

    assert target.read_bytes() == b'{"a":[2,3],"z":1}\n'
    assert stat.S_IMODE(real_lstat(target).st_mode) == 0o400
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        writer_activation._write_root_json(target, {"replacement": True})
    assert target.read_bytes() == original
