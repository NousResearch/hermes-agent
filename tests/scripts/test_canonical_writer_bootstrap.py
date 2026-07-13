from __future__ import annotations

import datetime as dt
import json
import os
import socket
import stat
import time
import uuid
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_db as writer_db
from gateway import canonical_writer_bootstrap as writer_bootstrap
from gateway.canonical_writer_db import CanonicalWriterDB, QueryResult
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_CANARY_BOOTSTRAP_LOGIN,
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    PRODUCTION_CATALOG_SHA256,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_boundary import DEFAULT_SOCKET_PATH
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_edge_writer_authority import CanonicalWriterDiscordAuthority
from scripts.canonical_brain_event_projector import read_events
from gateway.canonical_writer_bootstrap import (
    CANARY_PRECLAIM_RECONCILIATION_VERSION,
    CANARY_SCOPE_BOOTSTRAP_CONSUMPTION_VERSION,
    WRITER_RUNTIME_ATTESTATION_VERSION,
    CanaryScopePreapprovalConfig,
    _canary_scope_bootstrap_consumption_receipt,
    build_service,
    export_projection_events,
    load_service_config,
    publish_writer_runtime_readiness,
    reconcile_configured_canary_preclaim,
)
from gateway.canonical_writer_readiness import readiness_receipt_sha256
from gateway.canonical_writer_service import (
    DispatchContext,
    PeerCredentials,
    SystemdCgroupV2MainPidProvider,
)


EVENT_ID_1 = "11111111-1111-4111-8111-111111111111"
EVENT_ID_2 = "22222222-2222-4222-8222-222222222222"
EVENT_ID_3 = "33333333-3333-4333-8333-333333333333"


@pytest.fixture(autouse=True)
def _tmp_path_uses_process_primary_group(tmp_path):
    """Keep BSD tmp-path group inheritance aligned with the test process."""

    os.chown(tmp_path, -1, os.getgid())


class _ProjectionScopeMixin:
    @contextmanager
    def projection_export_scope(self):
        self.snapshot_entries = getattr(self, "snapshot_entries", 0) + 1
        try:
            yield self
        except BaseException:
            self.snapshot_failures = getattr(self, "snapshot_failures", 0) + 1
            raise
        finally:
            self.snapshot_exits = getattr(self, "snapshot_exits", 0) + 1


class _ProjectionProtocolSession:
    def __init__(self, pages):
        self.pages = pages
        self.queries = []
        self.closed = False

    def query(self, sql, *, maximum_rows):
        self.queries.append((sql, maximum_rows))
        if sql.startswith("BEGIN"):
            return QueryResult((), (), "BEGIN")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock_shared"):
            return QueryResult((), ((None,),), "SELECT 1")
        if sql == "COMMIT":
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            return QueryResult((), (), "ROLLBACK")
        if "writer_projection_read_events" in sql:
            response = self.pages.pop(0)
            envelope = json.dumps({"ok": True, "result": response})
            return QueryResult(("response",), ((envelope,),), "SELECT 1")
        pytest.fail(f"unexpected projection protocol query: {sql}")

    def close(self):
        self.closed = True


def _managed_hba_receipt(*, observed_at=None, user="canonical_writer"):
    observed_at = int(time.time()) if observed_at is None else observed_at
    return {
        "version": "managed-cloudsqladmin-hba-rejection-v2",
        "host": "10.0.0.8",
        "tls_server_name": "db.internal",
        "port": 5432,
        "server_certificate_sha256": "d" * 64,
        "database": "cloudsqladmin",
        "user": user,
        "observed_at_unix": observed_at,
        "expires_at_unix": observed_at + 300,
        "sqlstate": "28000",
        "server_message": (
            'no pg_hba.conf entry for host "10.0.0.8", user '
            f'"{user}", database "cloudsqladmin", SSL encryption'
        ),
        "result": "pg_hba_rejected",
        "tls_peer_verified": True,
    }


def _config_value(tmp_path):
    writer_private = Ed25519PrivateKey.generate()
    edge_private = Ed25519PrivateKey.generate()
    writer_key_path = tmp_path / "discord-writer-capability-private.pem"
    edge_key_path = tmp_path / "discord-edge-receipt-public.pem"
    for path in (writer_key_path, edge_key_path):
        if path.exists():
            path.chmod(0o600)
    writer_key_path.write_bytes(
        writer_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    writer_key_path.chmod(0o400)
    edge_key_path.write_bytes(
        edge_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    edge_key_path.chmod(0o440)
    managed_hba_receipt = _managed_hba_receipt()
    managed_hba_digest = writer_db.managed_cloudsqladmin_hba_receipt_from_mapping(
        managed_hba_receipt
    ).sha256
    return {
        "service": {
            "socket_path": str(DEFAULT_SOCKET_PATH),
            "gateway_unit": "hermes-cloud-gateway.service",
            "gateway_uid": os.getuid() + 1,
            "writer_uid": os.getuid(),
            "writer_gid": os.getgid(),
            "socket_gid": os.getgid() + 1,
            "projector_gid": os.getgid() + 2,
            "owner_discord_user_ids": ["owner-1"],
            "connection_timeout_seconds": 20,
            "max_connections": 4,
        },
        "database": {
            "host": "10.0.0.8",
            "tls_server_name": "db.internal",
            "port": 5432,
            "database": "canonical",
            "user": "canonical_writer",
            "ca_file": str(tmp_path / "server-ca.pem"),
            "credential_file": str(tmp_path / "database-password"),
            "connect_timeout_seconds": 3,
            "io_timeout_seconds": 7,
        },
        "privileges": {
            "schema": "canonical_brain",
            "table_grants": [],
            "routine_identities": [
                {
                    "signature": signature,
                    "owner": CANONICAL_WRITER_MIGRATION_OWNER,
                    "security_definer": True,
                    "language": "plpgsql",
                    "configuration": ["search_path=pg_catalog, canonical_brain"],
                    "definition_sha256": "a" * 64,
                }
                for signature in EXPECTED_ROUTINE_SIGNATURES
            ],
            "helper_routine_identities": [
                {
                    "signature": signature,
                    "owner": CANONICAL_WRITER_MIGRATION_OWNER,
                    "security_definer": False,
                    "language": "sql",
                    "configuration": ["search_path=pg_catalog, canonical_brain"],
                    "definition_sha256": "b" * 64,
                }
                for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
            ],
            "schema_privileges": ["USAGE"],
            "database_privileges": ["CONNECT"],
            "role_memberships": [CANONICAL_WRITER_ROLE],
            "private_schema_identity_sha256": "c" * 64,
            "managed_cloudsqladmin_hba_rejection_receipt": managed_hba_receipt,
            "managed_cloudsqladmin_hba_rejection_sha256": managed_hba_digest,
            "deployment_lock_key": 4_841_739_663_211_427_921,
        },
        "discord_edge_authority": {
            "enabled": True,
            "capability_private_key_file": str(writer_key_path),
            "edge_receipt_public_key_file": str(edge_key_path),
            "edge_receipt_public_key_id": ed25519_public_key_id(
                edge_private.public_key()
            ),
            "request_timeout_seconds": 15,
        },
    }


def _config_value_with_canary(tmp_path, *, expires_at):
    value = _config_value(tmp_path)
    bootstrap_receipt = _managed_hba_receipt(
        user=CANONICAL_CANARY_BOOTSTRAP_LOGIN
    )
    bootstrap_receipt_sha256 = (
        writer_db.managed_cloudsqladmin_hba_receipt_from_mapping(
            bootstrap_receipt
        ).sha256
    )
    bootstrap_credential = tmp_path / "canary-bootstrap-password"
    bootstrap_credential.write_text("not-a-real-secret\n", encoding="utf-8")
    bootstrap_credential.chmod(0o400)
    value["canary_scope_preapproval"] = {
        "grant_id": "canary-grant-expiry-test",
        "case_id": "case:canary-expiry-test",
        "release_sha256": "1" * 64,
        "fixture_sha256": "2" * 64,
        "run_id": "canary-run-expiry-test",
        "session_key_sha256": "3" * 64,
        "expires_at": expires_at.isoformat(),
        "approved_by": "owner-1",
        "approval_source_sha256": "4" * 64,
        "provisioning_receipt_sha256": "5" * 64,
        "bootstrap_database_user": CANONICAL_CANARY_BOOTSTRAP_LOGIN,
        "bootstrap_credential_file": str(bootstrap_credential),
        "bootstrap_managed_cloudsqladmin_hba_rejection_receipt": (
            bootstrap_receipt
        ),
        "bootstrap_managed_cloudsqladmin_hba_rejection_sha256": (
            bootstrap_receipt_sha256
        ),
    }
    return value


def _write_config(tmp_path, value, *, mode=0o400):
    path = tmp_path / "writer.json"
    if path.exists():
        path.chmod(0o600)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(mode)
    return path


def _load(path):
    return load_service_config(
        path,
        _expected_owner_uid=os.getuid(),
        _require_root_owned_parents=False,
    )


def test_loads_explicit_secret_free_config_and_pins_routine_catalog(tmp_path):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))

    assert config.gateway_uid != config.writer_uid
    assert config.owner_discord_user_ids == frozenset({"owner-1"})
    assert len({config.writer_gid, config.socket_gid, config.projector_gid}) == 3
    assert config.projector_gid != config.writer_gid
    assert config.database.credential.path == tmp_path / "database-password"
    assert config.database.credential.allowed_modes == frozenset({0o400})
    assert config.database.host == "10.0.0.8"
    assert config.database.tls_server_name == "db.internal"
    assert not hasattr(config.database, "password")
    assert config.privileges.executable_routines == EXPECTED_ROUTINE_SIGNATURES
    assert {
        identity.signature
        for identity in config.privileges.dependency_routine_identities
    } == set(EXPECTED_HELPER_ROUTINE_SIGNATURES)
    assert config.privileges.table_grants == ()
    assert config.privileges.sequence_grants == ()
    assert config.privileges.schema_privileges == ("USAGE",)
    assert config.privileges.database_privileges == ("CONNECT",)
    assert config.privileges.role_memberships == (CANONICAL_WRITER_ROLE,)
    assert config.privileges.private_schema_identity_sha256 == "c" * 64
    assert config.privileges.managed_cloudsqladmin_hba_rejection_receipt is not None
    assert (
        config.privileges.managed_cloudsqladmin_hba_rejection_sha256
        == config.privileges.managed_cloudsqladmin_hba_rejection_receipt.sha256
    )
    assert config.privileges.deployment_lock_key == 4_841_739_663_211_427_921
    assert config.discord_edge_authority.enabled is True
    assert isinstance(
        config.discord_edge_authority.capability_private_key,
        Ed25519PrivateKey,
    )
    assert config.discord_edge_authority.edge_receipt_public_key_id == (
        ed25519_public_key_id(
            config.discord_edge_authority.edge_receipt_public_key
        )
    )
    assert config.discord_edge_authority.request_timeout_seconds == 15


def test_expired_canary_scope_is_only_loadable_for_reconciliation(tmp_path):
    expired_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
    path = _write_config(
        tmp_path,
        _config_value_with_canary(tmp_path, expires_at=expired_at),
    )

    with pytest.raises(ValueError, match="within the next hour"):
        _load(path)

    reconciler_config = load_service_config(
        path,
        _expected_owner_uid=os.getuid(),
        _require_root_owned_parents=False,
        _allow_expired_canary_scope=True,
    )

    assert reconciler_config.canary_scope_preapproval is not None
    assert reconciler_config.canary_scope_preapproval.expires_at == expired_at
    assert reconciler_config.source_config_path == path


def test_expired_canary_scope_explicit_cli_writes_terminal_receipt(
    tmp_path,
    monkeypatch,
    capsys,
):
    expired_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30)
    path = _write_config(
        tmp_path,
        _config_value_with_canary(tmp_path, expires_at=expired_at),
    )
    receipt_path = tmp_path / "expired-preclaim-reconciliation.json"
    real_load = writer_bootstrap.load_service_config
    real_reconcile = writer_bootstrap.reconcile_configured_canary_preclaim

    def load_config(observed_path, **kwargs):
        assert kwargs == {"_allow_expired_canary_scope": True}
        return real_load(
            observed_path,
            _expected_owner_uid=os.getuid(),
            _require_root_owned_parents=False,
            **kwargs,
        )

    def reconcile(config, *, receipt_path):
        scope = config.canary_scope_preapproval
        assert scope is not None
        assert scope.expires_at == expired_at
        return real_reconcile(
            config,
            backend=_FakePreclaimRetirementBackend([
                _preclaim_result(scope, "retired", inserted=True),
            ]),
            receipt_path=receipt_path,
            _now_unix=lambda: 1_800_000_000,
        )

    monkeypatch.setattr(
        writer_bootstrap,
        "harden_current_process_against_dumping",
        lambda: None,
    )
    monkeypatch.setattr(writer_bootstrap, "load_service_config", load_config)
    monkeypatch.setattr(
        writer_bootstrap,
        "reconcile_configured_canary_preclaim",
        reconcile,
    )
    monkeypatch.setattr(
        writer_bootstrap,
        "DEFAULT_CANARY_PRECLAIM_RECONCILIATION_RECEIPT_PATH",
        receipt_path,
    )

    assert writer_bootstrap.main([
        "--config",
        str(path),
        "--reconcile-canary-preclaim",
    ]) == 0

    emitted = json.loads(capsys.readouterr().out)
    assert emitted["result"]["outcome"] == "retired"
    assert emitted["result"]["authority_active"] is False
    assert emitted == json.loads(receipt_path.read_text(encoding="utf-8"))


def test_loaded_runtime_config_rejects_writer_writable_database_credential(tmp_path):
    value = _config_value(tmp_path)
    credential = Path(value["database"]["credential_file"])
    credential.write_text("not-a-real-secret", encoding="utf-8")
    credential.chmod(0o600)
    config = _load(_write_config(tmp_path, value))

    assert config.database.credential.allowed_modes == frozenset({0o400})
    with pytest.raises(
        writer_db.CredentialSecurityError,
        match="credential_mode_not_allowed",
    ):
        writer_db._read_credential(config.database.credential)


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o644, 0o660])
def test_config_rejects_mutable_or_world_readable_modes(tmp_path, mode):
    path = _write_config(tmp_path, _config_value(tmp_path), mode=mode)

    with pytest.raises(ValueError, match="mode"):
        _load(path)


def test_config_rejects_symlink_secret_fields_unknown_fields_and_same_uid(tmp_path):
    value = _config_value(tmp_path)
    original = _write_config(tmp_path, value)
    link = tmp_path / "link.json"
    link.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        _load(link)

    value = _config_value(tmp_path)
    value["database"]["password"] = "must-never-be-here"
    with pytest.raises(ValueError, match="secret"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["shell_command"] = "/bin/sh"
    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["gateway_uid"] = value["service"]["writer_uid"]
    with pytest.raises(ValueError, match="distinct"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["service"]["socket_gid"] = value["service"]["writer_gid"]
    with pytest.raises(ValueError, match="groups must be distinct"):
        _load(_write_config(tmp_path, value))


def test_config_requires_explicit_tls_identity_without_host_fallback(tmp_path):
    value = _config_value(tmp_path)
    del value["database"]["tls_server_name"]

    with pytest.raises(ValueError, match="database.tls_server_name"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize(
    "tls_server_name",
    [" db.internal", "DB.internal", "db.internal."],
)
def test_config_rejects_noncanonical_tls_identity(tmp_path, tls_server_name):
    value = _config_value(tmp_path)
    value["database"]["tls_server_name"] = tls_server_name

    with pytest.raises(ValueError, match="tls_server_name|TLS server name"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_whitespace_normalized_connect_host(tmp_path):
    value = _config_value(tmp_path)
    value["database"]["host"] = " 10.0.0.8"

    with pytest.raises(ValueError, match="database.host"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_hba_receipt_with_missing_or_mismatched_tls_identity(
    tmp_path,
):
    value = _config_value(tmp_path)
    del value["privileges"][
        "managed_cloudsqladmin_hba_rejection_receipt"
    ]["tls_server_name"]
    with pytest.raises(ValueError, match="receipt fields are not exact"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    receipt = value["privileges"][
        "managed_cloudsqladmin_hba_rejection_receipt"
    ]
    receipt["tls_server_name"] = "other.internal"
    value["privileges"][
        "managed_cloudsqladmin_hba_rejection_sha256"
    ] = writer_db.managed_cloudsqladmin_hba_receipt_from_mapping(receipt).sha256
    with pytest.raises(ValueError, match="does not match database coordinates"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_legacy_hba_v1_without_fallback(tmp_path):
    value = _config_value(tmp_path)
    value["privileges"][
        "managed_cloudsqladmin_hba_rejection_receipt"
    ]["version"] = "managed-cloudsqladmin-hba-rejection-v1"

    with pytest.raises(ValueError, match="receipt version is invalid"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_operator_override_of_immutable_routine_catalog(tmp_path):
    value = _config_value(tmp_path)
    value["privileges"]["executable_routines"] = ["public.anything()"]

    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        (
            "table_grants",
            [{"table": "canonical_brain.events", "privileges": ["UPDATE"]}],
            "must be empty",
        ),
        ("schema_privileges", ["USAGE", "CREATE"], "exactly USAGE"),
        ("database_privileges", ["CONNECT", "TEMP"], "exactly CONNECT"),
        ("role_memberships", [], "dedicated writer role"),
        (
            "role_memberships",
            [CANONICAL_WRITER_ROLE, "cloudsqlsuperuser"],
            "dedicated writer role",
        ),
    ],
)
def test_config_hard_pins_zero_direct_grants_and_exact_role_scope(
    tmp_path,
    field,
    value,
    reason,
):
    config = _config_value(tmp_path)
    config["privileges"][field] = value

    with pytest.raises(ValueError, match=reason):
        _load(_write_config(tmp_path, config))


def test_config_requires_exact_non_executable_helper_identity_set(tmp_path):
    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"].pop()
    with pytest.raises(ValueError, match="pinned helper catalog"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"][0][
        "security_definer"
    ] = True
    with pytest.raises(ValueError, match="SECURITY INVOKER"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["routine_identities"][0]["owner"] = "other_owner"
    with pytest.raises(ValueError, match="pinned migration owner"):
        _load(_write_config(tmp_path, config))

    config = _config_value(tmp_path)
    config["privileges"]["helper_routine_identities"][0]["configuration"] = [
        "search_path=public"
    ]
    with pytest.raises(ValueError, match="safe search_path"):
        _load(_write_config(tmp_path, config))


def test_config_rejects_sequence_grant_surface_entirely(tmp_path):
    config = _config_value(tmp_path)
    config["privileges"]["sequence_grants"] = []

    with pytest.raises(ValueError, match="unknown"):
        _load(_write_config(tmp_path, config))


def test_config_rejects_deployment_lock_key_drift(tmp_path):
    value = _config_value(tmp_path)
    value["privileges"]["deployment_lock_key"] += 1

    with pytest.raises(ValueError, match="pinned writer lock"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize("digest", [None, "c" * 63, "C" * 64, "g" * 64])
def test_config_requires_exact_private_schema_identity_digest(tmp_path, digest):
    value = _config_value(tmp_path)
    if digest is None:
        del value["privileges"]["private_schema_identity_sha256"]
    else:
        value["privileges"]["private_schema_identity_sha256"] = digest

    with pytest.raises(ValueError, match="private_schema_identity_sha256"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize("digest", ["d" * 63, "D" * 64, "g" * 64])
def test_config_rejects_invalid_managed_cloudsqladmin_hba_receipt(
    tmp_path,
    digest,
):
    value = _config_value(tmp_path)
    value["privileges"]["managed_cloudsqladmin_hba_rejection_sha256"] = digest

    with pytest.raises(ValueError, match="cloudsqladmin HBA receipt"):
        _load(_write_config(tmp_path, value))


def test_config_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "writer.json"
    path.write_text('{"service":{},"service":{}}', encoding="utf-8")
    path.chmod(0o400)

    with pytest.raises(ValueError, match="strict UTF-8 JSON"):
        _load(path)


@pytest.mark.parametrize(
    "missing_field",
    [
        "capability_private_key_file",
        "edge_receipt_public_key_file",
        "edge_receipt_public_key_id",
        "request_timeout_seconds",
    ],
)
def test_enabled_discord_edge_authority_requires_exact_key_config(
    tmp_path,
    missing_field,
):
    value = _config_value(tmp_path)
    del value["discord_edge_authority"][missing_field]

    with pytest.raises(ValueError, match="requires exact key configuration"):
        _load(_write_config(tmp_path, value))


def test_writer_config_requires_explicit_discord_edge_authority_policy(tmp_path):
    value = _config_value(tmp_path)
    del value["discord_edge_authority"]

    with pytest.raises(ValueError, match="discord_edge_authority"):
        _load(_write_config(tmp_path, value))


def test_disabled_discord_edge_authority_forbids_key_fields(tmp_path):
    value = _config_value(tmp_path)
    value["discord_edge_authority"] = {"enabled": False}
    config = _load(_write_config(tmp_path, value))

    assert config.discord_edge_authority.enabled is False
    assert config.discord_edge_authority.capability_private_key is None
    assert config.discord_edge_authority.edge_receipt_public_key is None

    value = _config_value(tmp_path)
    value["discord_edge_authority"]["enabled"] = False
    with pytest.raises(ValueError, match="only enabled=false"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize(
    "key_name,mode",
    [
        ("capability_private_key_file", 0o600),
        ("capability_private_key_file", 0o440),
        ("edge_receipt_public_key_file", 0o400),
        ("edge_receipt_public_key_file", 0o444),
    ],
)
def test_discord_edge_key_files_require_exact_modes(tmp_path, key_name, mode):
    value = _config_value(tmp_path)
    Path(value["discord_edge_authority"][key_name]).chmod(mode)

    with pytest.raises(ValueError, match="mode must"):
        _load(_write_config(tmp_path, value))


def test_discord_edge_key_files_reject_symlinks_and_hardlinks(tmp_path):
    value = _config_value(tmp_path)
    private_path = Path(
        value["discord_edge_authority"]["capability_private_key_file"]
    )
    private_link = tmp_path / "writer-private-link.pem"
    private_link.symlink_to(private_path)
    value["discord_edge_authority"]["capability_private_key_file"] = str(
        private_link
    )
    with pytest.raises(ValueError, match="regular non-symlink"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    public_path = Path(
        value["discord_edge_authority"]["edge_receipt_public_key_file"]
    )
    os.link(public_path, tmp_path / "edge-public-hardlink.pem")
    with pytest.raises(ValueError, match="exactly one filesystem link"):
        _load(_write_config(tmp_path, value))


def test_enabled_discord_edge_authority_rejects_missing_or_relative_key_path(
    tmp_path,
):
    value = _config_value(tmp_path)
    private_path = Path(
        value["discord_edge_authority"]["capability_private_key_file"]
    )
    private_path.unlink()
    with pytest.raises(ValueError, match="private key is unavailable"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["discord_edge_authority"]["edge_receipt_public_key_file"] = (
        "edge-public.pem"
    )
    with pytest.raises(ValueError, match="absolute normalized path"):
        _load(_write_config(tmp_path, value))


@pytest.mark.parametrize("identity", ["owner", "group"])
def test_discord_writer_private_key_requires_exact_writer_identity(
    tmp_path,
    identity,
):
    value = _config_value(tmp_path)
    if identity == "owner":
        value["service"]["writer_uid"] = os.getuid() + 10
        value["service"]["gateway_uid"] = os.getuid() + 11
        expected = "owner is not trusted"
    else:
        value["service"]["writer_gid"] = os.getgid() + 10
        expected = "group is not the writer group"

    with pytest.raises(ValueError, match=expected):
        _load(_write_config(tmp_path, value))


def test_discord_edge_public_key_is_pinned_and_must_be_ed25519(tmp_path):
    value = _config_value(tmp_path)
    value["discord_edge_authority"]["edge_receipt_public_key_id"] = "f" * 64
    with pytest.raises(ValueError, match="pinned key ID"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    value["discord_edge_authority"]["edge_receipt_public_key_id"] = value[
        "discord_edge_authority"
    ]["edge_receipt_public_key_id"].upper()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _load(_write_config(tmp_path, value))

    value = _config_value(tmp_path)
    public_path = Path(
        value["discord_edge_authority"]["edge_receipt_public_key_file"]
    )
    public_path.chmod(0o600)
    public_path.write_text("not a PEM key\n", encoding="utf-8")
    public_path.chmod(0o440)
    with pytest.raises(ValueError, match="public key is not PEM"):
        _load(_write_config(tmp_path, value))


def test_discord_writer_private_key_must_be_unencrypted_ed25519_pem(tmp_path):
    value = _config_value(tmp_path)
    private_path = Path(
        value["discord_edge_authority"]["capability_private_key_file"]
    )
    private_path.chmod(0o600)
    private_path.write_text("not a PEM key\n", encoding="utf-8")
    private_path.chmod(0o400)

    with pytest.raises(ValueError, match="private key is not unencrypted PEM"):
        _load(_write_config(tmp_path, value))


def test_discord_edge_keys_reject_trailing_or_noncanonical_pem_data(tmp_path):
    value = _config_value(tmp_path)
    public_path = Path(
        value["discord_edge_authority"]["edge_receipt_public_key_file"]
    )
    original = public_path.read_bytes()
    public_path.chmod(0o600)
    public_path.write_bytes(original + b"ignored trailing data\n")
    public_path.chmod(0o440)

    with pytest.raises(ValueError, match="exact SubjectPublicKeyInfo PEM"):
        _load(_write_config(tmp_path, value))


def test_discord_writer_and_edge_receipt_keys_must_be_distinct(tmp_path):
    value = _config_value(tmp_path)
    private_path = Path(
        value["discord_edge_authority"]["capability_private_key_file"]
    )
    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(),
        password=None,
    )
    assert isinstance(private_key, Ed25519PrivateKey)
    public_path = Path(
        value["discord_edge_authority"]["edge_receipt_public_key_file"]
    )
    public_path.chmod(0o600)
    public_path.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    public_path.chmod(0o440)
    value["discord_edge_authority"]["edge_receipt_public_key_id"] = (
        ed25519_public_key_id(private_key.public_key())
    )

    with pytest.raises(ValueError, match="keys must be distinct"):
        _load(_write_config(tmp_path, value))


class _FakeDatabase:
    def __init__(self, *, config, privilege_policy, statements):
        self.config = config
        self.privilege_policy = privilege_policy
        self.statement_names = statements.names
        self.statement_catalog_sha256 = statements.sha256
        self.attested = False
        self.calls = []

    def startup_attest(self):
        self.attested = True

    def query_fixed(self, statement_name, parameters):
        assert self.attested
        self.calls.append((statement_name, parameters))
        response = json.dumps({"ok": True, "result": {"pong": True}})
        return QueryResult(("response",), ((response,),), "SELECT 1")


def test_build_service_attests_before_exposing_typed_dispatch(tmp_path):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))

    bootstrap = build_service(config, _database_factory=_FakeDatabase)
    response = bootstrap.server.dispatcher.dispatch(
        CanonicalWriterOperation.PING,
        {},
        DispatchContext(
            request_id="request-1",
            sequence=1,
            deadline_unix_ms=9999999999999,
            idempotency_key=None,
            peer=PeerCredentials(10, config.gateway_uid, 20),
            runtime={"platform": "discord", "thread_id": "thread-1"},
        ),
    )

    assert bootstrap.database.attested is True
    assert bootstrap.database.statement_catalog_sha256 == PRODUCTION_CATALOG_SHA256
    assert isinstance(
        bootstrap.handlers.discord_edge_authority,
        CanonicalWriterDiscordAuthority,
    )
    assert response.status == "ok"
    assert response.result == {"pong": True}
    assert bootstrap.database.calls[0][0] == "op_ping"
    assert isinstance(
        bootstrap.server.authorizer.main_pid_provider,
        SystemdCgroupV2MainPidProvider,
    )


def test_build_service_rejects_runtime_identity_drift(tmp_path):
    value = deepcopy(_config_value(tmp_path))
    value["service"]["writer_uid"] = os.getuid() + 10
    value["discord_edge_authority"] = {"enabled": False}
    config = _load(_write_config(tmp_path, value))

    with pytest.raises(PermissionError, match="UID/GID"):
        build_service(config, _database_factory=_FakeDatabase)


def test_build_service_fails_closed_if_enabled_authority_lacks_loaded_key(
    tmp_path,
):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    broken_authority = replace(
        config.discord_edge_authority,
        capability_private_key=None,
    )
    broken_config = replace(
        config,
        discord_edge_authority=broken_authority,
    )

    with pytest.raises(RuntimeError, match="lacks writer-owned Ed25519 authority"):
        build_service(broken_config, _database_factory=_FakeDatabase)


def test_build_service_keeps_routeback_fail_closed_when_authority_disabled(
    tmp_path,
):
    value = _config_value(tmp_path)
    value["discord_edge_authority"] = {"enabled": False}
    config = _load(_write_config(tmp_path, value))

    bootstrap = build_service(config, _database_factory=_FakeDatabase)

    assert bootstrap.handlers.discord_edge_authority is None


def _canary_consumption_bootstrap(*, receipt_updates=None):
    expires_at = dt.datetime(2026, 7, 13, 12, 30, tzinfo=dt.timezone.utc)
    scope = CanaryScopePreapprovalConfig(
        grant_id="canary-grant-1",
        case_id="case:canary-1",
        release_sha256="1" * 64,
        fixture_sha256="2" * 64,
        run_id="canary-run-1",
        session_key_sha256="3" * 64,
        expires_at=expires_at,
        approved_by="1279454038731264061",
        approval_source_sha256="4" * 64,
        provisioning_receipt_sha256="5" * 64,
        bootstrap_database=SimpleNamespace(
            user=CANONICAL_CANARY_BOOTSTRAP_LOGIN,
        ),
        bootstrap_managed_hba_receipt=None,
        bootstrap_managed_hba_receipt_sha256="",
    )
    receipt = {
        "success": True,
        "grant_id": scope.grant_id,
        "case_id": scope.case_id,
        "release_sha256": scope.release_sha256,
        "fixture_sha256": scope.fixture_sha256,
        "run_id": scope.run_id,
        "session_key_sha256": scope.session_key_sha256,
        "expires_at": expires_at.isoformat(),
        "approved_by": scope.approved_by,
        "approval_source_sha256": scope.approval_source_sha256,
        "preapproved_at": "2026-07-13T12:00:00+00:00",
        "event_id": "11111111-1111-4111-8111-111111111111",
        "receipt_event_id": "11111111-1111-4111-8111-111111111111",
        "bootstrap_consumption_event_id": (
            "22222222-2222-4222-8222-222222222222"
        ),
        "bootstrap_acl_revoked": True,
        "inserted": True,
        "deduped": False,
    }
    receipt.update(receipt_updates or {})
    return SimpleNamespace(
        config=SimpleNamespace(canary_scope_preapproval=scope),
        canary_scope_preapproval_receipt=receipt,
    )


def _preclaim_result(scope, outcome, *, inserted=False, deduped=False):
    result = {
        "success": True,
        "outcome": outcome,
        "grant_id": scope.grant_id,
        "case_id": scope.case_id,
        "release_sha256": scope.release_sha256,
        "fixture_sha256": scope.fixture_sha256,
        "run_id": scope.run_id,
        "session_key_sha256": scope.session_key_sha256,
        "expires_at": scope.expires_at.isoformat(),
        "approved_by": scope.approved_by,
        "approval_source_sha256": scope.approval_source_sha256,
        "provisioning_receipt_sha256": scope.provisioning_receipt_sha256,
        "preapproval_event_id": None,
        "bootstrap_consumption_event_id": None,
        "claim_event_id": None,
        "retirement_event_id": None,
        "revocation_event_id": None,
        "claimed_at": None,
        "retired_at": None,
        "reason": "preapproval_not_committed",
        "scope_retired": False,
        "authority_active": False,
        "inserted": inserted,
        "deduped": deduped,
    }
    if outcome == "retired":
        result.update({
            "preapproval_event_id": "11111111-1111-4111-8111-111111111111",
            "bootstrap_consumption_event_id": (
                "22222222-2222-4222-8222-222222222222"
            ),
            "retirement_event_id": "33333333-3333-4333-8333-333333333333",
            "retired_at": "2026-07-13T12:01:00+00:00",
            "reason": "activation_failed_before_first_claim",
            "scope_retired": True,
        })
    elif outcome == "claimed":
        result.update({
            "preapproval_event_id": "11111111-1111-4111-8111-111111111111",
            "bootstrap_consumption_event_id": (
                "22222222-2222-4222-8222-222222222222"
            ),
            "claim_event_id": "33333333-3333-4333-8333-333333333333",
            "revocation_event_id": "44444444-4444-4444-8444-444444444444",
            "claimed_at": "2026-07-13T12:00:30+00:00",
            "reason": "claim_already_committed_session_retired",
        })
    return result


class _FakePreclaimRetirementBackend:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def retire(self, request, runtime):
        self.calls.append((request, runtime))
        if not self.results:
            pytest.fail("unexpected extra preclaim retirement call")
        return deepcopy(self.results.pop(0))


class _FakeCanaryBootstrapBackend:
    def __init__(self, receipt):
        self.receipt = receipt
        self.calls = []

    def preapprove(self, request, runtime):
        self.calls.append((request, runtime))
        return deepcopy(self.receipt)


def test_preclaim_reconciliation_receipt_binds_exact_source_and_database(
    tmp_path,
):
    source_path = _write_config(tmp_path, _config_value(tmp_path))
    config = _load(source_path)
    scope = _canary_consumption_bootstrap().config.canary_scope_preapproval
    config = replace(config, canary_scope_preapproval=scope)
    backend = _FakePreclaimRetirementBackend([
        _preclaim_result(scope, "retired", inserted=True),
    ])
    receipt_path = tmp_path / "preclaim-reconciliation.json"

    receipt = reconcile_configured_canary_preclaim(
        config,
        backend=backend,
        receipt_path=receipt_path,
        _now_unix=lambda: 1_800_000_000,
    )

    assert receipt["version"] == CANARY_PRECLAIM_RECONCILIATION_VERSION
    assert receipt["observed_at_unix"] == 1_800_000_000
    assert receipt["source_config_path"] == str(source_path)
    assert receipt["source_config_sha256"] == config.source_config_sha256
    assert receipt["database_identity"] == {
        "host": config.database.host,
        "tls_server_name": config.database.tls_server_name,
        "port": config.database.port,
        "database": config.database.database,
        "user": config.database.user,
    }
    body = dict(receipt)
    digest = body.pop("receipt_sha256")
    assert digest == readiness_receipt_sha256(body)
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert backend.calls[0][1].service_internal is True


def test_build_service_reconciles_before_bootstrap_and_retains_shutdown_backend(
    tmp_path,
):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    canary = _canary_consumption_bootstrap()
    scope = canary.config.canary_scope_preapproval
    config = replace(config, canary_scope_preapproval=scope)
    retirement = _FakePreclaimRetirementBackend([
        _preclaim_result(scope, "not_preapproved"),
    ])
    bootstrap_backend = _FakeCanaryBootstrapBackend(
        canary.canary_scope_preapproval_receipt
    )

    built = build_service(
        config,
        _database_factory=_FakeDatabase,
        _canary_bootstrap_factory=lambda **_kwargs: bootstrap_backend,
        _canary_retirement_factory=lambda **_kwargs: retirement,
    )

    assert len(retirement.calls) == 1
    assert len(bootstrap_backend.calls) == 1
    assert built.canary_preclaim_retirement_backend is retirement
    assert built.canary_scope_preapproval_receipt == (
        canary.canary_scope_preapproval_receipt
    )


def test_build_failure_after_consumption_durably_retires_preclaim(tmp_path):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    canary = _canary_consumption_bootstrap()
    scope = canary.config.canary_scope_preapproval
    config = replace(config, canary_scope_preapproval=scope)
    retirement = _FakePreclaimRetirementBackend([
        _preclaim_result(scope, "not_preapproved"),
        _preclaim_result(scope, "retired", inserted=True),
    ])
    bootstrap_backend = _FakeCanaryBootstrapBackend(
        canary.canary_scope_preapproval_receipt
    )

    class _FailingDatabase(_FakeDatabase):
        def startup_attest(self):
            raise RuntimeError("attestation failed")

    with pytest.raises(RuntimeError, match="attestation failed"):
        build_service(
            config,
            _database_factory=_FailingDatabase,
            _canary_bootstrap_factory=lambda **_kwargs: bootstrap_backend,
            _canary_retirement_factory=lambda **_kwargs: retirement,
        )

    assert len(bootstrap_backend.calls) == 1
    assert len(retirement.calls) == 2
    assert retirement.results == []


def test_build_service_rejects_stale_scope_after_prestart_reconciliation(
    tmp_path,
):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    canary = _canary_consumption_bootstrap()
    scope = canary.config.canary_scope_preapproval
    config = replace(config, canary_scope_preapproval=scope)
    retirement = _FakePreclaimRetirementBackend([
        _preclaim_result(scope, "retired", deduped=True),
    ])
    bootstrap_backend = _FakeCanaryBootstrapBackend(
        canary.canary_scope_preapproval_receipt
    )

    with pytest.raises(RuntimeError, match="config is stale"):
        build_service(
            config,
            _database_factory=_FakeDatabase,
            _canary_bootstrap_factory=lambda **_kwargs: bootstrap_backend,
            _canary_retirement_factory=lambda **_kwargs: retirement,
        )

    assert len(retirement.calls) == 1
    assert bootstrap_backend.calls == []


def test_main_start_failure_blocks_until_preclaim_reconciliation(monkeypatch):
    calls = []

    class _Server:
        def start(self):
            calls.append("start")
            raise RuntimeError("listener failed")

        def shutdown(self):
            calls.append("shutdown")

    retirement_backend = object()
    config = SimpleNamespace()
    bootstrap = SimpleNamespace(
        server=_Server(),
        canary_preclaim_retirement_backend=retirement_backend,
    )
    monkeypatch.setattr(
        writer_bootstrap,
        "harden_current_process_against_dumping",
        lambda: None,
    )
    monkeypatch.setattr(
        writer_bootstrap,
        "load_service_config",
        lambda _path, **_kwargs: config,
    )
    monkeypatch.setattr(writer_bootstrap, "build_service", lambda _config: bootstrap)

    def reconcile(observed_config, *, backend, receipt_path):
        assert observed_config is config
        assert backend is retirement_backend
        assert receipt_path == (
            writer_bootstrap.DEFAULT_CANARY_PRECLAIM_RECONCILIATION_RECEIPT_PATH
        )
        calls.append("reconciled")
        return {"success": True}

    monkeypatch.setattr(
        writer_bootstrap,
        "reconcile_configured_canary_preclaim",
        reconcile,
    )

    with pytest.raises(RuntimeError, match="listener failed"):
        writer_bootstrap.main(["--config", "/etc/muncho/writer.json"])

    assert calls == ["start", "shutdown", "reconciled"]


def test_explicit_preclaim_reconciliation_requires_preserved_scope(monkeypatch):
    config = SimpleNamespace(writer_uid=os.getuid(), writer_gid=os.getgid())
    monkeypatch.setattr(
        writer_bootstrap,
        "harden_current_process_against_dumping",
        lambda: None,
    )
    load_calls = []

    def load_config(_path, **kwargs):
        load_calls.append(kwargs)
        return config

    monkeypatch.setattr(writer_bootstrap, "load_service_config", load_config)
    monkeypatch.setattr(
        writer_bootstrap,
        "reconcile_configured_canary_preclaim",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="requires sealed scope"):
        writer_bootstrap.main([
            "--config",
            "/etc/muncho/full-canary/staged/writer.json",
            "--reconcile-canary-preclaim",
        ])
    assert load_calls == [{"_allow_expired_canary_scope": True}]


def test_canary_consumption_readiness_receipt_is_exact_secret_free_and_bound():
    receipt = _canary_scope_bootstrap_consumption_receipt(
        _canary_consumption_bootstrap()
    )

    assert receipt is not None
    assert set(receipt) == {
        "version",
        "grant_id",
        "case_id",
        "release_sha256",
        "fixture_sha256",
        "run_id",
        "session_key_sha256",
        "expires_at",
        "approved_by",
        "approval_source_sha256",
        "provisioning_receipt_sha256",
        "bootstrap_database_user",
        "preapproval_event_id",
        "consumption_event_id",
        "preapproved_at",
        "acl_revoked",
        "inserted",
        "receipt_sha256",
    }
    assert receipt["version"] == CANARY_SCOPE_BOOTSTRAP_CONSUMPTION_VERSION
    assert receipt["bootstrap_database_user"] == CANONICAL_CANARY_BOOTSTRAP_LOGIN
    assert receipt["acl_revoked"] is True
    assert receipt["inserted"] is True
    digest_body = dict(receipt)
    digest = digest_body.pop("receipt_sha256")
    assert digest == readiness_receipt_sha256(digest_body)
    serialized = json.dumps(receipt, sort_keys=True)
    assert "credential" not in serialized
    assert "password" not in serialized
    assert "managed_cloudsqladmin" not in serialized


def test_canary_consumption_readiness_is_null_without_one_shot_scope():
    assert _canary_scope_bootstrap_consumption_receipt(
        SimpleNamespace(
            config=SimpleNamespace(canary_scope_preapproval=None),
            canary_scope_preapproval_receipt=None,
        )
    ) is None


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"bootstrap_acl_revoked": False}, "not freshly consumed"),
        ({"case_id": "case:other"}, "binding mismatch"),
        ({"unexpected": "field"}, "fields are not exact"),
        ({"bootstrap_consumption_event_id": "not-a-uuid"}, "event ID"),
    ],
)
def test_canary_consumption_readiness_rejects_unproven_receipts(updates, match):
    with pytest.raises(RuntimeError, match=match):
        _canary_scope_bootstrap_consumption_receipt(
            _canary_consumption_bootstrap(receipt_updates=updates)
        )


def test_writer_runtime_readiness_binds_socket_process_and_systemd_status(
    tmp_path,
):
    socket_path = Path("/tmp") / f"cw-ready-{uuid.uuid4().hex}.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    os.chown(socket_path, -1, os.getgid())
    socket_path.chmod(0o660)
    notifications = []
    canary_bootstrap = _canary_consumption_bootstrap()
    bootstrap = SimpleNamespace(
        server=SimpleNamespace(fileno=listener.fileno()),
        config=SimpleNamespace(
            socket_path=socket_path,
            writer_uid=os.getuid(),
            socket_gid=os.getgid(),
            database=SimpleNamespace(user="canonical_writer"),
            privileges=SimpleNamespace(
                private_schema_identity_sha256="c" * 64,
                managed_cloudsqladmin_hba_rejection_sha256="d" * 64,
            ),
            discord_edge_authority=SimpleNamespace(enabled=False),
            canary_scope_preapproval=(
                canary_bootstrap.config.canary_scope_preapproval
            ),
        ),
        database=SimpleNamespace(statement_catalog_sha256="e" * 64),
        canary_scope_preapproval_receipt=(
            canary_bootstrap.canary_scope_preapproval_receipt
        ),
    )
    receipt_path = tmp_path / "writer-runtime-attestation.json"

    try:
        receipt = publish_writer_runtime_readiness(
            bootstrap,
            receipt_path=receipt_path,
            _now_unix=lambda: 1_800_000_000,
            _boot_identity_provider=lambda: ("b" * 64, 987654321),
            _process_start_time=lambda _pid: 123456,
            _notify=lambda *args, **kwargs: (
                notifications.append((args, kwargs)) or True
            ),
            _process_hardening_provider=lambda: (False, 0, 0),
            _python_runtime_provider=lambda: {
                "effective_import_paths": ["/opt/release/site-packages"],
                "unexpected_import_paths": [],
                "loaded_module_origins": [
                    "/opt/release/site-packages/gateway/"
                    "canonical_writer_bootstrap.py"
                ],
                "unexpected_import_origins": [],
                "loaded_module_origins_complete": True,
                "effective_environment_variable_names": ["NOTIFY_SOCKET"],
                "effective_environment_variable_value_sha256": {
                    "NOTIFY_SOCKET": "d" * 64
                },
            },
        )
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)

    assert receipt["version"] == WRITER_RUNTIME_ATTESTATION_VERSION
    assert receipt["writer_pid"] == os.getpid()
    assert receipt["writer_start_time_ticks"] == 123456
    assert receipt["socket_group_gid"] == os.getgid()
    assert receipt["socket_mode"] == "0660"
    assert receipt["discord_edge_authority_enabled"] is False
    assert receipt["canary_scope_bootstrap_consumption"] == (
        _canary_scope_bootstrap_consumption_receipt(canary_bootstrap)
    )
    assert receipt["writer_dumpable"] is False
    assert receipt["writer_core_soft_limit"] == 0
    assert receipt["writer_core_hard_limit"] == 0
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert notifications[0][0][0] == WRITER_RUNTIME_ATTESTATION_VERSION
    assert notifications[0][1] == {"ready": True}


def _snapshot_backend(tmp_path, monkeypatch, pages):
    config = _load(_write_config(tmp_path, _config_value(tmp_path)))
    sessions = []
    attested_sessions = []

    def session_factory(_config):
        session = _ProjectionProtocolSession(pages)
        sessions.append(session)
        return session

    def collect_attestation(session, **_kwargs):
        attested_sessions.append(session)
        return object()

    monkeypatch.setattr(
        writer_db,
        "_collect_privilege_attestation",
        collect_attestation,
    )
    monkeypatch.setattr(
        writer_db,
        "validate_privilege_attestation",
        lambda *_args, **_kwargs: None,
    )
    database = CanonicalWriterDB(
        config=config.database,
        privilege_policy=config.privileges,
        statements=PRODUCTION_STATEMENT_CATALOG,
        _session_factory=session_factory,
        _managed_hba_probe=lambda _config: (
            config.privileges.managed_cloudsqladmin_hba_rejection_receipt
        ),
    )
    database.startup_attest()
    return (
        PostgresCanonicalWriterBackend(database),
        sessions,
        attested_sessions,
    )


def test_projection_export_requires_snapshot_scope_and_leaves_no_partial_file(
    tmp_path,
):
    class _Backend:
        @staticmethod
        def projector_read(_request, _runtime):
            pytest.fail("ordinary per-page query must not be used")

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="projection snapshot"):
        export_projection_events(bootstrap, target, limit=10)

    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_multi_page_export_uses_one_attested_serializable_database_snapshot(
    tmp_path,
    monkeypatch,
):
    pages = [
        {
            "events": [{"event_id": EVENT_ID_1}],
            "has_more": True,
            "next_after_event_id": EVENT_ID_1,
        },
        {
            "events": [{"event_id": EVENT_ID_2}],
            "has_more": False,
            "next_after_event_id": EVENT_ID_2,
        },
    ]
    backend, sessions, attested_sessions = _snapshot_backend(
        tmp_path,
        monkeypatch,
        pages,
    )
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"

    assert export_projection_events(bootstrap, target, limit=10) == 2

    assert len(sessions) == 2
    assert attested_sessions == sessions
    snapshot = sessions[1]
    sql = [query for query, _maximum_rows in snapshot.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sum("writer_projection_read_events" in query for query in sql) == 2
    assert sql.count("COMMIT") == 1
    assert "ROLLBACK" not in sql
    assert snapshot.closed is True


def test_database_snapshot_rolls_back_when_projection_validation_fails(
    tmp_path,
    monkeypatch,
):
    pages = [
        {
            "events": [{"event_id": EVENT_ID_1}, "invalid-row"],
            "has_more": False,
        }
    ]
    backend, sessions, attested_sessions = _snapshot_backend(
        tmp_path,
        monkeypatch,
        pages,
    )
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"
    original = '{"events":[]}\n'
    target.write_text(original, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid row"):
        export_projection_events(bootstrap, target, limit=10)

    assert attested_sessions == sessions
    snapshot = sessions[1]
    sql = [query for query, _maximum_rows in snapshot.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sql.count("ROLLBACK") == 1
    assert "COMMIT" not in sql
    assert snapshot.closed is True
    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_privileged_projection_export_paginates_and_feeds_pure_projector(tmp_path):
    pages = [
        {
            "events": [
                {"event_id": EVENT_ID_1, "case_id": "case:1"},
                {"event_id": EVENT_ID_2, "case_id": "case:2"},
            ],
            "has_more": True,
            "next_after_event_id": EVENT_ID_2,
        },
        {
            "events": [{"event_id": EVENT_ID_3, "case_id": "case:3"}],
            "has_more": False,
            "next_after_event_id": EVENT_ID_3,
        },
    ]
    calls = []

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, runtime):
            calls.append((request, runtime))
            return pages[len(calls) - 1]

    backend = _Backend()
    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=backend,
    )
    target = tmp_path / "canonical-events.json"

    count = export_projection_events(bootstrap, target, limit=3)
    projected_rows = read_events(target, limit=10)

    assert count == 3
    assert projected_rows == [
        {"event_id": EVENT_ID_1, "case_id": "case:1"},
        {"event_id": EVENT_ID_2, "case_id": "case:2"},
        {"event_id": EVENT_ID_3, "case_id": "case:3"},
    ]
    assert [
        (request.case_id, request.after_event_id, request.limit)
        for request, _runtime in calls
    ] == [
        ("", "", 3),
        ("", EVENT_ID_2, 1),
    ]
    assert all(
        runtime.platform == "writer_service" and runtime.service_internal is True
        for _request, runtime in calls
    )
    assert backend.snapshot_entries == 1
    assert backend.snapshot_exits == 1
    assert getattr(backend, "snapshot_failures", 0) == 0


def test_projection_export_failure_removes_partial_file_and_preserves_target(tmp_path):
    target = tmp_path / "canonical-events.json"
    original = '{"events":[{"event_id":"old"}]}\n'
    target.write_text(original, encoding="utf-8")

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            return {
                "events": [{"event_id": EVENT_ID_1}, "invalid-row"],
                "has_more": False,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )

    with pytest.raises(RuntimeError, match="invalid row"):
        export_projection_events(bootstrap, target, limit=10)

    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []
    assert bootstrap.backend.snapshot_entries == 1
    assert bootstrap.backend.snapshot_exits == 1
    assert bootstrap.backend.snapshot_failures == 1


def test_projection_export_rejects_nonadvancing_cursor_and_cleans_up(tmp_path):
    calls = 0

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "events": [{"event_id": EVENT_ID_1}],
                    "has_more": True,
                    "next_after_event_id": EVENT_ID_1,
                }
            return {
                "events": [{"event_id": EVENT_ID_2}],
                "has_more": True,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="cursor did not advance"):
        export_projection_events(bootstrap, target, limit=10)

    assert calls == 2
    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_projection_export_rejects_page_larger_than_requested_bound(tmp_path):
    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, _runtime):
            assert request.limit == 1
            return {
                "events": [
                    {"event_id": EVENT_ID_1},
                    {"event_id": EVENT_ID_2},
                ],
                "has_more": False,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="exceeded its page limit"):
        export_projection_events(bootstrap, target, limit=1)

    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_projection_export_rejects_truncation_at_global_limit(tmp_path):
    target = tmp_path / "canonical-events.json"
    original = '{"events":[]}\n'
    target.write_text(original, encoding="utf-8")

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(request, _runtime):
            assert request.limit == 1
            return {
                "events": [{"event_id": EVENT_ID_1}],
                "has_more": True,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )

    with pytest.raises(RuntimeError, match="global event limit"):
        export_projection_events(bootstrap, target, limit=1)

    assert target.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []


def test_pure_projector_rejects_a_smaller_partial_read_limit(tmp_path):
    target = tmp_path / "canonical-events.json"
    target.write_text(
        json.dumps({
            "events": [
                {"event_id": EVENT_ID_1},
                {"event_id": EVENT_ID_2},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="projector limit"):
        read_events(target, limit=1)


def test_projection_export_rejects_duplicate_event_ids_across_pages(tmp_path):
    calls = 0

    class _Backend(_ProjectionScopeMixin):
        @staticmethod
        def projector_read(_request, _runtime):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "events": [{"event_id": EVENT_ID_1}],
                    "has_more": True,
                    "next_after_event_id": EVENT_ID_1,
                }
            return {
                "events": [{"event_id": EVENT_ID_1}],
                "has_more": False,
                "next_after_event_id": EVENT_ID_1,
            }

    bootstrap = SimpleNamespace(
        config=SimpleNamespace(
            writer_uid=os.getuid(),
            projector_gid=os.getgid(),
        ),
        backend=_Backend(),
    )
    target = tmp_path / "canonical-events.json"

    with pytest.raises(RuntimeError, match="duplicate or noncanonical event_id"):
        export_projection_events(bootstrap, target, limit=10)

    assert calls == 2
    assert target.exists() is False
    assert list(tmp_path.glob(".canonical-events.json.tmp.*")) == []
