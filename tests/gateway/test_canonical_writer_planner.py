from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import canonical_writer_host_authority as host_authority
from gateway import canonical_writer_planner as planner
from gateway.canonical_writer_bootstrap import (
    CanonicalWriterServiceConfig,
    DiscordEdgeWriterAuthorityConfig,
)
from gateway.canonical_writer_boundary import DEFAULT_GATEWAY_UNIT, DEFAULT_SOCKET_PATH
from gateway.canonical_writer_db import (
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    WriterDBConfig,
    WriterPrivilegePolicy,
)
from gateway.canonical_writer_release_contract import (
    ReleaseManifest,
    TreeEntry,
    WriterOnlyUnitSpec,
    render_systemd_units,
)


REVISION = "a" * 40
SQL_PRIVATE_IP = "10.91.0.3"
SQL_TLS_SERVER_NAME = "db.muncho.internal"


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
    entries = tuple(sorted((
        TreeEntry("python", "directory", "0555"),
        TreeEntry("python/cpython-3.11.15-linux-x86_64", "directory", "0555"),
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
    ), key=lambda entry: entry.path))
    provisional = ReleaseManifest(
        revision=REVISION,
        artifact_root=str(root),
        python_version="3.11.15",
        interpreter=str(root / "venv/bin/python"),
        writer_module_origin=str(
            root
            / "venv/lib/python3.11/site-packages/gateway/"
            "canonical_writer_bootstrap.py"
        ),
        gateway_module_origin=str(
            root
            / "venv/lib/python3.11/site-packages/gateway/"
            "canonical_writer_gateway_bootstrap.py"
        ),
        entries=entries,
        artifact_sha256="",
    )
    return replace(
        provisional,
        artifact_sha256=provisional.computed_artifact_sha256,
    )


def _identities() -> planner.HostNumericIdentities:
    return planner.HostNumericIdentities(
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
            ca_file=Path("/etc/muncho/trust/cloudsql-server-ca.pem"),
            credential=CredentialSource(
                path=Path(
                    "/etc/muncho-canonical-writer-credentials/database-password"
                ),
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


def _native_digests(
    release: ReleaseManifest,
    unit_spec: WriterOnlyUnitSpec,
    *,
    writer_config_sha256: str = "2" * 64,
    gateway_config_sha256: str = "3" * 64,
) -> planner.NativeObservationDigests:
    bundle = render_systemd_units(release, unit_spec)
    return planner.NativeObservationDigests(
        writer_unit_sha256=hashlib.sha256(
            bundle.writer_service.encode()
        ).hexdigest(),
        gateway_unit_sha256=hashlib.sha256(
            bundle.gateway_service.encode()
        ).hexdigest(),
        exporter_unit_sha256=hashlib.sha256(
            bundle.exporter_service.encode()
        ).hexdigest(),
        tmpfiles_sha256=hashlib.sha256(bundle.tmpfiles.encode()).hexdigest(),
        writer_config_sha256=writer_config_sha256,
        gateway_config_sha256=gateway_config_sha256,
    )


def _native_plan(
    *,
    release: ReleaseManifest | None = None,
    writer_config_sha256: str = "2" * 64,
    gateway_config_sha256: str = "3" * 64,
    release_manifest_file_sha256: str = "d" * 64,
    database_ca_sha256: str = "e" * 64,
):
    release = release or _release()
    unit_spec = _unit_spec()
    digests = _native_digests(
        release,
        unit_spec,
        writer_config_sha256=writer_config_sha256,
        gateway_config_sha256=gateway_config_sha256,
    )
    return planner.build_native_observation_plan(
        release,
        unit_spec,
        _writer_config(),
        _identities(),
        digests,
        observation_id="11111111-1111-4111-8111-111111111111",
        boot_id_sha256="b" * 64,
        host_identity_sha256="c" * 64,
        release_manifest_file_sha256=release_manifest_file_sha256,
        config_collector_receipt_sha256="7" * 64,
        database_ca_sha256=database_ca_sha256,
        external_iam_policy_sha256="f" * 64,
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
    )


def _stopped_receipt(native_plan=None):
    plan = native_plan or _native_plan()
    observed = 1_000_000_000

    def live(label: str, pid: int):
        identities = plan.value["identities"]
        return {
            "unit_name": plan.value[f"{label}_unit"]["name"],
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "disabled",
            "main_pid": pid,
            "start_time_ticks": pid + 100,
            "argv": list(plan.value[f"{label}_argv"]),
            "external_native_mappings": [{
                "path": "/usr/lib/x86_64-linux-gnu/libc.so.6",
                "sha256": "a" * 64,
            }],
            "kernel_executable_mappings": ["[vdso]", "[vsyscall]"],
            "process_authority": {
                "pid": pid,
                "process_start_time_ticks": pid + 100,
                "effective_uid": identities[f"{label}_uid"],
                "effective_gid": identities[f"{label}_gid"],
                "supplementary_gids": identities[
                    f"{label}_supplementary_gids"
                ],
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

    def stopped(label: str):
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
        "observation": {
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
                "gateway_access": {
                    "read": False,
                    "write": False,
                    "execute": False,
                },
            },
        },
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


def _final_plan():
    return planner.build_activation_plan(
        _release(),
        _unit_spec(),
        _writer_config(),
        _identities(),
        _stopped_receipt(),
        writer_config_sha256="2" * 64,
        gateway_config_sha256="3" * 64,
        release_manifest_file_sha256="d" * 64,
        database_ca_sha256="e" * 64,
        external_iam_policy_sha256="f" * 64,
        external_iam_receipt_path=(
            "/run/muncho-canonical-preflight/external-iam-receipt.json"
        ),
        sql_private_ip=SQL_PRIVATE_IP,
        sql_tls_server_name=SQL_TLS_SERVER_NAME,
        paths=planner.ActivationPaths(),
    )


def test_native_plan_is_discovery_only_and_binds_collector_receipt():
    plan = _native_plan()

    assert plan.value["config_collector_receipt_sha256"] == "7" * 64
    assert plan.value["writer_argv"][1:3] == ["-B", "-I"]
    assert plan.value["gateway_argv"][1:3] == ["-B", "-I"]
    assert plan.value["discord"]["required_absent"] is True
    assert "approved_plan_sha256" not in json.dumps(plan.to_mapping())
    assert "external_native_mappings" not in plan.to_mapping()


def test_final_plan_is_the_only_deployable_v3_contract():
    native_receipt = _stopped_receipt()
    plan = _final_plan()

    assert plan.schema == "muncho-writer-only-activation-plan.v3"
    assert plan.digests.native_observation_receipt_sha256 == native_receipt.sha256
    assert plan.collector_argv[1:3] == ("-B", "-I")
    assert plan.validator_argv[1:3] == ("-B", "-I")
    assert plan.to_mapping()["activation_plan_sha256"] == plan.sha256
    assert "password" not in json.dumps(plan.to_mapping()).casefold()


def test_planner_cli_contracts_accept_only_digest_bound_inputs():
    native_parameters = inspect.signature(
        planner.build_and_stage_native_observation_plan
    ).parameters
    final_parameters = inspect.signature(
        planner.build_and_stage_final_activation_plan
    ).parameters

    assert set(native_parameters) == {
        "revision",
        "external_iam_policy_sha256",
        "config_collector_receipt_sha256",
    }
    assert set(final_parameters) == {"native_observation_receipt_sha256"}
    assert "approval" not in " ".join((*native_parameters, *final_parameters))


def test_final_builder_cross_binds_and_exclusively_stages_fixed_plan(monkeypatch):
    release = _release()
    release_raw = planner._canonical_bytes(release.to_mapping()) + b"\n"
    writer_raw = b"{}"
    gateway_raw = b"{}"
    database_ca_raw = b"test-ca"
    native_plan = _native_plan(
        writer_config_sha256=hashlib.sha256(writer_raw).hexdigest(),
        gateway_config_sha256=hashlib.sha256(gateway_raw).hexdigest(),
        release_manifest_file_sha256=hashlib.sha256(release_raw).hexdigest(),
        database_ca_sha256=hashlib.sha256(database_ca_raw).hexdigest(),
    )
    native_receipt = _stopped_receipt(native_plan)
    required_bindings: list[dict[str, object]] = []

    class CollectorReceipt:
        sha256 = "7" * 64
        value = {
            "database": {
                "host": SQL_PRIVATE_IP,
                "tls_server_name": SQL_TLS_SERVER_NAME,
            }
        }

        def require_bindings(self, **kwargs):
            required_bindings.append(kwargs)

        def to_mapping(self):
            return {"receipt_sha256": self.sha256, **self.value}

    collector = CollectorReceipt()
    trusted_files = {
        planner.DEFAULT_WRITER_CONFIG_SOURCE_PATH: writer_raw,
        planner.DEFAULT_GATEWAY_CONFIG_SOURCE_PATH: gateway_raw,
        planner.DEFAULT_DATABASE_CA_PATH: database_ca_raw,
    }
    staged: list[tuple[Path, bytes]] = []

    monkeypatch.setattr(planner, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        planner,
        "load_native_observation_plan",
        lambda path: native_plan,
    )
    monkeypatch.setattr(
        planner,
        "load_durable_native_observation_receipt",
        lambda plan: native_receipt,
    )
    monkeypatch.setattr(
        planner,
        "load_config_collector_receipt",
        lambda **_kwargs: collector,
    )
    monkeypatch.setattr(
        planner,
        "load_release_manifest",
        lambda revision: (release, release_raw),
    )
    monkeypatch.setattr(
        planner,
        "_read_trusted_root_file",
        lambda path, **_kwargs: trusted_files[path],
    )
    monkeypatch.setattr(planner, "load_service_config", lambda path: _writer_config())
    monkeypatch.setattr(planner, "_validate_writer_only_policy", lambda value: None)
    monkeypatch.setattr(planner.os.path, "lexists", lambda path: False)
    monkeypatch.setattr(
        planner,
        "_write_atomic_root_staged_file",
        lambda path, payload: staged.append((path, payload)),
    )

    result = planner.build_and_stage_final_activation_plan(
        native_observation_receipt_sha256=native_receipt.sha256,
    )

    assert staged[0][0] == planner.DEFAULT_STAGED_PLAN_PATH
    staged_plan = planner.ActivationPlan.from_mapping(
        json.loads(staged[0][1].decode())
    )
    assert result["activation_plan_sha256"] == staged_plan.sha256
    assert result["native_observation_plan_sha256"] == native_plan.sha256
    assert all(name.endswith("_sha256") for name in result)
    assert len(required_bindings) == 1
    assert required_bindings[0]["sql_private_ip"] == SQL_PRIVATE_IP


def test_final_builder_fails_before_inputs_when_durable_digest_differs(monkeypatch):
    native_plan = _native_plan()
    native_receipt = _stopped_receipt(native_plan)
    monkeypatch.setattr(planner, "_require_root_linux", lambda: None)
    monkeypatch.setattr(
        planner,
        "load_native_observation_plan",
        lambda path: native_plan,
    )
    monkeypatch.setattr(
        planner,
        "load_durable_native_observation_receipt",
        lambda plan: native_receipt,
    )
    monkeypatch.setattr(
        planner,
        "load_config_collector_receipt",
        lambda **_kwargs: pytest.fail("must not read later planning inputs"),
    )

    with pytest.raises(PermissionError, match="receipt digest drifted"):
        planner.build_and_stage_final_activation_plan(
            native_observation_receipt_sha256="0" * 64,
        )


def test_stager_rejects_nonproduction_output_path(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "_require_root_linux", lambda: None)

    with pytest.raises(ValueError, match="production-pinned"):
        planner._write_atomic_root_staged_file(tmp_path / "plan.json", b"{}")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "fields"),
        (
            lambda value: value.update({"artifact_root": "/tmp/release"}),
            "identity",
        ),
        (
            lambda value: value.update({"interpreter": "/usr/bin/python3"}),
            "identity",
        ),
        (
            lambda value: value["entries"][0].update({"extra": True}),
            "entry fields",
        ),
        (
            lambda value: value.update({"artifact_sha256": "0" * 64}),
            "self-digest",
        ),
    ],
)
def test_release_manifest_parser_is_exact_and_fail_closed(mutation, message):
    release = _release()
    assert planner._parse_release_manifest_mapping(
        release.to_mapping(),
        expected_revision=REVISION,
    ).to_mapping() == release.to_mapping()
    value = json.loads(json.dumps(release.to_mapping()))
    mutation(value)

    with pytest.raises(ValueError, match=message):
        planner._parse_release_manifest_mapping(
            value,
            expected_revision=REVISION,
        )


def test_native_builder_rejects_unit_config_and_identity_drift():
    release = _release()
    unit_spec = _unit_spec()
    digests = _native_digests(release, unit_spec)
    arguments = {
        "observation_id": "11111111-1111-4111-8111-111111111111",
        "boot_id_sha256": "8" * 64,
        "host_identity_sha256": "9" * 64,
        "release_manifest_file_sha256": "a" * 64,
        "config_collector_receipt_sha256": "d" * 64,
        "database_ca_sha256": "b" * 64,
        "external_iam_policy_sha256": "c" * 64,
        "sql_private_ip": SQL_PRIVATE_IP,
        "sql_tls_server_name": SQL_TLS_SERVER_NAME,
    }

    with pytest.raises(ValueError, match="unit digest drifted"):
        planner.build_native_observation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            replace(digests, writer_unit_sha256="0" * 64),
            **arguments,
        )
    with pytest.raises(ValueError, match="database binding"):
        planner.build_native_observation_plan(
            release,
            unit_spec,
            replace(
                _writer_config(),
                database=replace(
                    _writer_config().database,
                    ca_file=Path("/wrong/ca.pem"),
                ),
            ),
            _identities(),
            digests,
            **arguments,
        )
    with pytest.raises(ValueError, match="dormant identity homes"):
        planner.build_native_observation_plan(
            release,
            unit_spec,
            _writer_config(),
            replace(_identities(), projector_uid=991),
            digests,
            **arguments,
        )


@pytest.mark.parametrize(
    ("sql_ip", "unit_network", "message"),
    [
        ("127.0.0.1", "127.0.0.1/32", "SQL boundary"),
        (SQL_PRIVATE_IP, "10.91.0.4/32", "SQL boundary"),
        ("2001:db8::1", "2001:db8::1/128", "SQL boundary"),
    ],
)
def test_native_builder_rejects_nonprivate_or_nonexact_sql_boundary(
    sql_ip,
    unit_network,
    message,
):
    release = _release()
    unit_spec = WriterOnlyUnitSpec(database_ip_allow=(unit_network,))

    with pytest.raises(ValueError, match=message):
        planner.build_native_observation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            _native_digests(release, unit_spec),
            observation_id="11111111-1111-4111-8111-111111111111",
            boot_id_sha256="8" * 64,
            host_identity_sha256="9" * 64,
            release_manifest_file_sha256="a" * 64,
            config_collector_receipt_sha256="d" * 64,
            database_ca_sha256="b" * 64,
            external_iam_policy_sha256="c" * 64,
            sql_private_ip=sql_ip,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
        )


@pytest.mark.parametrize(
    "tls_server_name",
    [" DB.MUNCHO.INTERNAL", "DB.MUNCHO.INTERNAL", "db.muncho.internal."],
)
def test_native_builder_rejects_nonexact_tls_name(tls_server_name):
    release = _release()
    unit_spec = _unit_spec()

    with pytest.raises(ValueError, match="TLS server name"):
        planner.build_native_observation_plan(
            release,
            unit_spec,
            _writer_config(),
            _identities(),
            _native_digests(release, unit_spec),
            observation_id="11111111-1111-4111-8111-111111111111",
            boot_id_sha256="8" * 64,
            host_identity_sha256="9" * 64,
            release_manifest_file_sha256="a" * 64,
            config_collector_receipt_sha256="d" * 64,
            database_ca_sha256="b" * 64,
            external_iam_policy_sha256="c" * 64,
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=tls_server_name,
        )


def test_native_mapping_policy_is_mandatory_sorted_unique_and_external():
    root = _release().artifact_root
    valid = planner.ExternalNativeExecutableMapping(
        path="/usr/lib/x86_64-linux-gnu/libc.so.6",
        sha256="a" * 64,
    )
    second = planner.ExternalNativeExecutableMapping(
        path="/usr/lib/x86_64-linux-gnu/libz.so.1",
        sha256="b" * 64,
    )

    planner.PreapprovedNativeExecutablePolicy(
        writer=(valid, second),
        gateway=(valid, second),
    ).validate(root)
    with pytest.raises(ValueError, match="non-empty"):
        planner.PreapprovedNativeExecutablePolicy(
            writer=(),
            gateway=(valid,),
        ).validate(root)
    with pytest.raises(ValueError, match="exactly sorted"):
        planner.PreapprovedNativeExecutablePolicy(
            writer=(second, valid),
            gateway=(valid,),
        ).validate(root)
    with pytest.raises(ValueError, match="unique"):
        planner.PreapprovedNativeExecutablePolicy(
            writer=(valid, valid),
            gateway=(valid,),
        ).validate(root)
    with pytest.raises(ValueError, match="outside sealed release"):
        planner.PreapprovedNativeExecutablePolicy(
            writer=(planner.ExternalNativeExecutableMapping(
                path=f"{root}/venv/bin/python",
                sha256="c" * 64,
            ),),
            gateway=(valid,),
        ).validate(root)


def test_final_plan_is_deterministic_self_digesting_and_secret_free():
    first = _final_plan()
    second = _final_plan()
    value = first.to_mapping()

    assert first == second
    assert first.sha256 == second.sha256
    assert first.paths == planner.ActivationPaths()
    encoded = json.dumps(value, sort_keys=True).casefold()
    assert "super-secret" not in encoded
    assert "credential_value" not in encoded
    value["digests"]["writer_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest|sha256|drift|inconsistent"):
        planner.ActivationPlan.from_mapping(value)


def test_final_plan_rejects_writer_identity_discord_and_path_drift():
    release = _release()
    receipt = _stopped_receipt()
    base = {
        "writer_config_sha256": "2" * 64,
        "gateway_config_sha256": "3" * 64,
        "release_manifest_file_sha256": "d" * 64,
        "database_ca_sha256": "e" * 64,
        "external_iam_policy_sha256": "f" * 64,
        "external_iam_receipt_path": (
            "/run/muncho-canonical-preflight/external-iam-receipt.json"
        ),
        "sql_private_ip": SQL_PRIVATE_IP,
        "sql_tls_server_name": SQL_TLS_SERVER_NAME,
        "paths": planner.ActivationPaths(),
    }
    config = _writer_config()

    with pytest.raises(ValueError, match="database authority"):
        planner.build_activation_plan(
            release,
            _unit_spec(),
            replace(config, database=replace(config.database, host="10.91.0.4")),
            _identities(),
            receipt,
            **base,
        )
    with pytest.raises(ValueError, match="host identities"):
        planner.build_activation_plan(
            release,
            _unit_spec(),
            replace(config, gateway_uid=9999),
            _identities(),
            receipt,
            **base,
        )
    with pytest.raises(ValueError, match="host identities"):
        planner.build_activation_plan(
            release,
            _unit_spec(),
            replace(
                config,
                discord_edge_authority=replace(
                    config.discord_edge_authority,
                    enabled=True,
                ),
            ),
            _identities(),
            receipt,
            **base,
        )
    with pytest.raises(ValueError, match="production-pinned"):
        planner.ActivationPaths.from_mapping({
            **planner.ActivationPaths().to_mapping(),
            "manifest_path": "/tmp/deployment-manifest.json",
        })


def test_final_plan_rejects_ambiguous_managed_stdlib():
    ambiguous = _release(extra_entries=(
        TreeEntry(
            "python/second-runtime/lib/python3.11",
            "directory",
            "0555",
        ),
    ))

    receipt = _stopped_receipt(_native_plan(release=ambiguous))

    with pytest.raises(ValueError, match="stdlib identity is ambiguous"):
        planner.build_activation_plan(
            ambiguous,
            _unit_spec(),
            _writer_config(),
            _identities(),
            receipt,
            writer_config_sha256="2" * 64,
            gateway_config_sha256="3" * 64,
            release_manifest_file_sha256="d" * 64,
            database_ca_sha256="e" * 64,
            external_iam_policy_sha256="f" * 64,
            external_iam_receipt_path=(
                "/run/muncho-canonical-preflight/external-iam-receipt.json"
            ),
            sql_private_ip=SQL_PRIVATE_IP,
            sql_tls_server_name=SQL_TLS_SERVER_NAME,
            paths=planner.ActivationPaths(),
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"gateway_config_sha256": "0" * 64}, "gateway_config binding"),
        ({"external_iam_policy_sha256": "0" * 64}, "external_iam_policy"),
        ({"database_ca_sha256": "0" * 64}, "database binding"),
    ],
)
def test_pure_final_builder_cannot_bypass_native_evidence_bindings(
    overrides,
    message,
):
    arguments = {
        "writer_config_sha256": "2" * 64,
        "gateway_config_sha256": "3" * 64,
        "release_manifest_file_sha256": "d" * 64,
        "database_ca_sha256": "e" * 64,
        "external_iam_policy_sha256": "f" * 64,
        "external_iam_receipt_path": (
            "/run/muncho-canonical-preflight/external-iam-receipt.json"
        ),
        "sql_private_ip": SQL_PRIVATE_IP,
        "sql_tls_server_name": SQL_TLS_SERVER_NAME,
        "paths": planner.ActivationPaths(),
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        planner.build_activation_plan(
            _release(),
            _unit_spec(),
            _writer_config(),
            _identities(),
            _stopped_receipt(),
            **arguments,
        )
