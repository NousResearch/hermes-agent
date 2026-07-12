from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gateway import canonical_writer_config_collector as collector
from gateway.canonical_writer_bootstrap import load_service_config
from gateway.canonical_writer_db import (
    CANONICAL_EVENT_LOG_COLUMNS,
    CANONICAL_EVENT_LOG_OWNER,
    CANONICAL_EVENT_LOG_TABLE,
    CANONICAL_PRIVATE_WRITER_TABLES,
    CanonicalEventLogIdentity,
    CanonicalPrivateRelationIdentity,
    CanonicalPrivateSchemaIdentity,
    ManagedCloudSQLAdminHBAReceipt,
    PrivilegeAttestation,
    QueryResult,
    RoutineIdentity,
    TablePrivilegeGrant,
    _expected_canonical_non_owner_acl_grants,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
)


TLS_NAME = "14-11111111-1111-4111-8111-111111111111.europe-west3.sql.goog"
REVISION = "a" * 40


def _hba() -> ManagedCloudSQLAdminHBAReceipt:
    now = int(time.time())
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=collector.SQL_PRIVATE_IP,
        tls_server_name=TLS_NAME,
        port=collector.SQL_PORT,
        server_certificate_sha256="b" * 64,
        database="cloudsqladmin",
        user=collector.SQL_USER,
        observed_at_unix=now,
        expires_at_unix=now + 300,
        sqlstate="28000",
        server_message=(
            'no pg_hba.conf entry for host "10.90.0.2", user '
            f'"{collector.SQL_USER}", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _identity(signature: str, *, public: bool) -> RoutineIdentity:
    return RoutineIdentity(
        signature=signature,
        owner=CANONICAL_WRITER_MIGRATION_OWNER,
        security_definer=public,
        language="plpgsql" if public else "sql",
        configuration=("search_path=pg_catalog, canonical_brain",),
        definition_sha256=("c" if public else "d") * 64,
    )


def _event_identity() -> CanonicalEventLogIdentity:
    return CanonicalEventLogIdentity(
        table=CANONICAL_EVENT_LOG_TABLE,
        owner=CANONICAL_EVENT_LOG_OWNER,
        owner_dangerous=False,
        relation_kind="r",
        persistence="p",
        is_partition=False,
        access_method="heap",
        tablespace_oid=0,
        row_security=False,
        force_row_security=False,
        replica_identity="d",
        relation_options=(),
        columns=CANONICAL_EVENT_LOG_COLUMNS,
        constraints=("PRIMARY KEY (event_id)",),
        user_triggers=(),
        rewrite_rules=(),
        policies=(),
        inheritance=False,
        non_owner_acl_grants=(),
        index_count=1,
        primary_index_exact=True,
    )


def _private_identity() -> CanonicalPrivateSchemaIdentity:
    relations = tuple(
        CanonicalPrivateRelationIdentity(
            name=name,
            owner=CANONICAL_WRITER_MIGRATION_OWNER,
            owner_dangerous=False,
            relation_kind="r",
            persistence="p",
            is_partition=False,
            access_method="heap",
            tablespace_oid=0,
            row_security=False,
            force_row_security=False,
            replica_identity="d",
            relation_options=(),
            columns=(json.dumps({"name": "identity"}, separators=(",", ":")),),
            constraints=(json.dumps({"type": "p"}, separators=(",", ":")),),
            indexes=(json.dumps({"primary": True}, separators=(",", ":")),),
            index_owners=(f"{CANONICAL_WRITER_MIGRATION_OWNER}:f",),
            user_triggers=(),
            rewrite_rules=(),
            policies=(),
            inheritance=False,
        )
        for name in CANONICAL_PRIVATE_WRITER_TABLES
    )
    return CanonicalPrivateSchemaIdentity(
        schema="canonical_brain",
        owner=CANONICAL_WRITER_MIGRATION_OWNER,
        owner_dangerous=False,
        relations=relations,
    )


def _attestation() -> PrivilegeAttestation:
    hba = _hba()
    public = tuple(_identity(signature, public=True) for signature in EXPECTED_ROUTINE_SIGNATURES)
    helpers = tuple(
        _identity(signature, public=False)
        for signature in EXPECTED_HELPER_ROUTINE_SIGNATURES
    )
    private = _private_identity()
    seed = collector.WriterPrivilegePolicy(
        schema="canonical_brain",
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=public,
        dependency_routine_identities=helpers,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256=private.sha256,
        managed_cloudsqladmin_hba_rejection_receipt=hba,
        managed_cloudsqladmin_hba_rejection_sha256=hba.sha256,
    )
    return PrivilegeAttestation(
        role=collector.SQL_USER,
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=public,
        dependency_routine_identities=helpers,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        canonical_non_owner_acl_grants=(
            _expected_canonical_non_owner_acl_grants(seed)
        ),
        canonical_writer_role_inheritors=(f"{collector.SQL_USER}:1:t:f",),
        canonical_event_log_identity=_event_identity(),
        canonical_private_schema_identity=private,
    )


def test_live_attestation_builds_loadable_exact_secret_free_config(tmp_path):
    attestation = _attestation()
    policy = collector._policy_from_attestation(attestation, hba_receipt=_hba())
    config = collector._database_config(TLS_NAME)
    mapping = collector._writer_config_mapping(
        config=config,
        policy=policy,
        owner_discord_user_ids=("123456789012345678",),
    )
    path = tmp_path / "writer.json"
    path.write_bytes(collector._canonical_bytes(mapping))
    path.chmod(0o400)

    loaded = load_service_config(
        path,
        _expected_owner_uid=os.geteuid(),
        _require_root_owned_parents=False,
    )

    assert loaded.database.host == collector.SQL_PRIVATE_IP
    assert loaded.database.tls_server_name == TLS_NAME
    assert loaded.database.user == collector.SQL_USER
    assert loaded.database.credential.path == config.credential.path
    assert loaded.database.credential.expected_uid == collector.WRITER_UID
    assert loaded.database.credential.expected_gid == collector.WRITER_GID
    assert loaded.database.credential.allowed_modes == frozenset({0o400})
    assert loaded.privileges == policy
    assert loaded.discord_edge_authority.enabled is False
    assert loaded.owner_discord_user_ids == frozenset({"123456789012345678"})


def test_discord_disabled_config_accepts_no_owner_ids(tmp_path):
    attestation = _attestation()
    policy = collector._policy_from_attestation(attestation, hba_receipt=_hba())
    mapping = collector._writer_config_mapping(
        config=collector._database_config(TLS_NAME),
        policy=policy,
        owner_discord_user_ids=(),
    )
    path = tmp_path / "writer.json"
    path.write_bytes(collector._canonical_bytes(mapping))
    path.chmod(0o400)

    loaded = load_service_config(
        path,
        _expected_owner_uid=os.geteuid(),
        _require_root_owned_parents=False,
    )

    assert loaded.owner_discord_user_ids == frozenset()
    assert loaded.discord_edge_authority.enabled is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: dataclasses.replace(
                value,
                routine_identities=value.routine_identities[:-1],
            ),
            "public routine signature set",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                routine_identities=(
                    dataclasses.replace(value.routine_identities[0], owner="other_owner"),
                    *value.routine_identities[1:],
                ),
            ),
            "routine authority identity",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                routine_identities=(
                    dataclasses.replace(
                        value.routine_identities[0],
                        configuration=("search_path=public",),
                    ),
                    *value.routine_identities[1:],
                ),
            ),
            "routine authority identity",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                dependency_routine_identities=(
                    dataclasses.replace(
                        value.dependency_routine_identities[0],
                        security_definer=True,
                    ),
                    *value.dependency_routine_identities[1:],
                ),
            ),
            "helper routine execution identity",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                table_grants=(
                    TablePrivilegeGrant("public.canonical_event_log", ("SELECT",)),
                ),
            ),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(value, unexpected_privileges=("x",)),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(value, public_acl_grants=("PUBLIC",)),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(value, schema_privileges=("CREATE",)),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(value, role_memberships=()),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                canonical_event_log_identity=None,
            ),
            "privilege surface",
        ),
        (
            lambda value: dataclasses.replace(
                value,
                canonical_private_schema_identity=None,
            ),
            "privilege surface",
        ),
    ],
)
def test_policy_discovery_fails_closed_on_authority_drift(mutate, message):
    with pytest.raises(ValueError, match=message):
        collector._policy_from_attestation(
            mutate(_attestation()),
            hba_receipt=_hba(),
        )


class _Session:
    def __init__(self):
        self.queries: list[str] = []
        self.closed = False

    def query(self, sql, *, maximum_rows):
        self.queries.append(sql)
        if sql.startswith("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY"):
            return QueryResult((), (), "BEGIN")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock_shared"):
            return QueryResult((), ((None,),), "SELECT 1")
        if sql == "COMMIT":
            return QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            return QueryResult((), (), "ROLLBACK")
        raise AssertionError(sql)

    def close(self):
        self.closed = True


def test_live_collection_uses_shared_locked_read_only_transaction(monkeypatch):
    session = _Session()
    attestation = _attestation()
    monkeypatch.setattr(
        collector,
        "_collect_privilege_attestation",
        lambda *_args, **_kwargs: attestation,
    )

    policy, observed, hba = collector._collect_live_policy(
        collector._database_config(TLS_NAME),
        hba_collector=lambda _config: _hba(),
        session_factory=lambda _config: session,
    )

    assert observed == attestation
    assert hba.tls_peer_verified is True
    assert policy.private_schema_identity_sha256 == _private_identity().sha256
    assert session.queries[0] == "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY"
    assert session.queries[1].startswith(
        "SELECT pg_catalog.pg_advisory_xact_lock_shared"
    )
    assert session.queries[-1] == "COMMIT"
    assert session.closed is True


def test_failed_catalog_collection_rolls_back_read_only_transaction(monkeypatch):
    session = _Session()

    def fail(*_args, **_kwargs):
        raise RuntimeError("catalog drift")

    monkeypatch.setattr(collector, "_collect_privilege_attestation", fail)
    with pytest.raises(RuntimeError, match="catalog drift"):
        collector._collect_live_policy(
            collector._database_config(TLS_NAME),
            hba_collector=lambda _config: _hba(),
            session_factory=lambda _config: session,
        )

    assert session.queries[0] == "BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY"
    assert session.queries[-1] == "ROLLBACK"
    assert session.closed is True


def test_receipt_and_configs_never_record_credential_content_or_digest():
    secret = b"this-must-never-be-recorded-or-hashed"
    attestation = _attestation()
    hba = _hba()
    policy = collector._policy_from_attestation(attestation, hba_receipt=hba)
    artifacts = collector._build_artifacts(
        revision=REVISION,
        artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        ca_sha256="3" * 64,
        credential_provenance={
            "path": str(collector.DATABASE_CREDENTIAL_PATH),
            "device": 1,
            "inode": 2,
            "owner_uid": collector.WRITER_UID,
            "group_gid": collector.WRITER_GID,
            "mode": "0400",
            "link_count": 1,
            "modification_time_ns": 3,
            "change_time_ns": 4,
            "content_or_digest_recorded": False,
        },
        config=collector._database_config(TLS_NAME),
        policy=policy,
        attestation=attestation,
        owner_discord_user_ids=("123456789012345678",),
        collected_at_unix=int(time.time()),
    )
    rendered = (
        artifacts.writer_config
        + artifacts.gateway_config
        + collector._canonical_bytes(artifacts.receipt)
    )

    assert secret not in rendered
    assert hashlib.sha256(secret).hexdigest().encode("ascii") not in rendered
    assert b"credential_sha256" not in rendered
    assert artifacts.receipt["credential_content_or_digest_recorded"] is False
    assert artifacts.receipt["credential_provenance"][
        "content_or_digest_recorded"
    ] is False


def _valid_collector_receipt():
    attestation = _attestation()
    hba = _hba()
    policy = collector._policy_from_attestation(attestation, hba_receipt=hba)
    return collector._build_artifacts(
        revision=REVISION,
        artifact_sha256="1" * 64,
        manifest_file_sha256="2" * 64,
        ca_sha256="3" * 64,
        credential_provenance={
            "path": str(collector.DATABASE_CREDENTIAL_PATH),
            "device": 1,
            "inode": 2,
            "owner_uid": collector.WRITER_UID,
            "group_gid": collector.WRITER_GID,
            "mode": "0400",
            "link_count": 1,
            "modification_time_ns": 3,
            "change_time_ns": 4,
            "content_or_digest_recorded": False,
        },
        config=collector._database_config(TLS_NAME),
        policy=policy,
        attestation=attestation,
        owner_discord_user_ids=(),
        collected_at_unix=hba.observed_at_unix,
    ).receipt


def _collector_bindings(receipt):
    return {
        "revision": receipt["release_revision"],
        "release_artifact_sha256": receipt["release_artifact_sha256"],
        "release_manifest_file_sha256": receipt[
            "release_manifest_file_sha256"
        ],
        "writer_config_sha256": receipt["writer_config_sha256"],
        "gateway_config_sha256": receipt["gateway_config_sha256"],
        "database_ca_sha256": receipt["database"]["ca_sha256"],
        "sql_private_ip": receipt["database"]["host"],
        "sql_tls_server_name": receipt["database"]["tls_server_name"],
    }


def test_config_collector_receipt_is_exact_self_digesting_and_fresh():
    value = _valid_collector_receipt()
    receipt = collector.ConfigCollectorReceipt.from_mapping(value)

    receipt.require_fresh(value["collected_at_unix"])
    receipt.require_bindings(**_collector_bindings(value))
    assert receipt.sha256 == value["receipt_sha256"]
    assert receipt.to_mapping() == value

    extra = {**value, "unexpected": True}
    with pytest.raises(ValueError, match="fields are not exact"):
        collector.ConfigCollectorReceipt.from_mapping(extra)
    with pytest.raises(ValueError, match="stale or future-dated"):
        receipt.require_fresh(value["hba_expires_at_unix"] + 1)


def test_config_collector_receipt_loader_uses_only_derived_path(monkeypatch):
    value = _valid_collector_receipt()
    raw = collector._canonical_bytes(value)
    observed = []

    def read(path, **kwargs):
        observed.append((path, kwargs))
        return raw

    monkeypatch.setattr(collector, "_read_trusted_public_file", read)
    loaded = collector.load_config_collector_receipt(
        revision=REVISION,
        receipt_sha256=value["receipt_sha256"],
        require_fresh=True,
        now_unix=value["collected_at_unix"],
    )

    assert loaded.to_mapping() == value
    assert observed == [
        (
            collector.EVIDENCE_ROOT
            / REVISION
            / f"{value['receipt_sha256']}.json",
            {
                "uid": 0,
                "gid": 0,
                "modes": frozenset({0o400}),
                "maximum_bytes": collector._MAX_CONFIG_BYTES,
            },
        )
    ]


def test_config_collector_receipt_loader_rejects_absent_and_noncanonical(
    monkeypatch,
):
    value = _valid_collector_receipt()

    def absent(*_args, **_kwargs):
        raise FileNotFoundError("receipt absent")

    monkeypatch.setattr(collector, "_read_trusted_public_file", absent)
    with pytest.raises(FileNotFoundError, match="receipt absent"):
        collector.load_config_collector_receipt(
            revision=REVISION,
            receipt_sha256=value["receipt_sha256"],
            require_fresh=False,
        )

    monkeypatch.setattr(
        collector,
        "_read_trusted_public_file",
        lambda *_args, **_kwargs: collector._canonical_bytes(value) + b"\n",
    )
    with pytest.raises(ValueError, match="not canonical JSON"):
        collector.load_config_collector_receipt(
            revision=REVISION,
            receipt_sha256=value["receipt_sha256"],
            require_fresh=False,
        )


def test_config_collector_receipt_loader_rejects_path_digest_mismatch(
    monkeypatch,
):
    value = _valid_collector_receipt()
    monkeypatch.setattr(
        collector,
        "_read_trusted_public_file",
        lambda *_args, **_kwargs: collector._canonical_bytes(value),
    )

    with pytest.raises(ValueError, match="path digest drifted"):
        collector.load_config_collector_receipt(
            revision=REVISION,
            receipt_sha256="4" * 64,
            require_fresh=False,
        )


@pytest.mark.parametrize(
    "binding",
    (
        "release_artifact_sha256",
        "release_manifest_file_sha256",
        "writer_config_sha256",
        "gateway_config_sha256",
        "database_ca_sha256",
    ),
)
def test_config_collector_receipt_rejects_release_config_and_ca_drift(binding):
    value = _valid_collector_receipt()
    bindings = _collector_bindings(value)
    bindings[binding] = "4" * 64

    with pytest.raises(ValueError, match="binding drifted"):
        collector.ConfigCollectorReceipt.from_mapping(value).require_bindings(
            **bindings
        )


def test_cli_failure_never_reflects_or_hashes_secret(monkeypatch, capsys):
    secret = "database-password-must-not-escape"

    def fail(**_kwargs):
        raise RuntimeError(secret)

    monkeypatch.setattr(collector, "collect_and_stage", fail)
    result = collector.main(
        [
            "collect",
            "--revision",
            REVISION,
            "--release-artifact-sha256",
            "1" * 64,
            "--release-manifest-file-sha256",
            "2" * 64,
            "--tls-server-name",
            TLS_NAME,
            "--owner-discord-user-id",
            "123456789012345678",
        ]
    )
    output = capsys.readouterr()

    assert result == 1
    assert secret not in output.out + output.err
    assert hashlib.sha256(secret.encode()).hexdigest() not in output.out + output.err
    assert json.loads(output.err) == {
        "ok": False,
        "error": "trusted_config_collection_failed",
        "error_type": "RuntimeError",
    }


def test_tls_name_and_owner_ids_are_strict():
    assert collector._validate_tls_server_name(TLS_NAME) == TLS_NAME
    assert collector._owner_ids(("123456789012345678",)) == (
        "123456789012345678",
    )
    assert collector._owner_ids(()) == ()
    with pytest.raises(ValueError, match="TLS SAN"):
        collector._validate_tls_server_name("db.example.com")
    with pytest.raises(ValueError, match="duplicates"):
        collector._owner_ids(("123456789012345678", "123456789012345678"))


def test_gateway_policy_is_the_exact_credential_free_readiness_surface():
    assert yaml.safe_load(collector.GATEWAY_CONFIG_BYTES) == {
        "canonical_brain": {
            "writer_boundary": {"enabled": True},
            "discord_edge": {"enabled": False},
        },
        "plugins": {"enabled": [], "disabled": []},
        "cron": {"provider": "builtin"},
    }
    assert b"token" not in collector.GATEWAY_CONFIG_BYTES
    assert b"password" not in collector.GATEWAY_CONFIG_BYTES


def _release_tree(tmp_path):
    root = tmp_path / "release"
    library = root / "lib"
    library.mkdir(parents=True)
    module = library / "module.py"
    module.write_bytes(b"sealed-module")
    module.chmod(0o444)
    library.chmod(0o555)
    link = root / "module-link.py"
    link.symlink_to("lib/module.py")
    root.chmod(0o555)
    entries = [
        {"path": "lib", "kind": "directory", "mode": "0555"},
        {
            "path": "lib/module.py",
            "kind": "file",
            "mode": "0444",
            "size": len(b"sealed-module"),
            "sha256": hashlib.sha256(b"sealed-module").hexdigest(),
        },
        {
            "path": "module-link.py",
            "kind": "symlink",
            "mode": f"{stat.S_IMODE(link.lstat().st_mode):04o}",
            "target": "lib/module.py",
        },
    ]
    return root, module, link, entries


def test_release_entry_attestation_rejects_tampered_file(tmp_path):
    root, module, _link, entries = _release_tree(tmp_path)
    collector._verify_release_entries(
        root,
        entries,
        expected_uid=os.getuid(),
        expected_gid=os.getgid(),
    )
    module.chmod(0o644)
    module.write_bytes(b"drifted-module")
    module.chmod(0o444)

    with pytest.raises(ValueError, match="file identity drifted"):
        collector._verify_release_entries(
            root,
            entries,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_release_entry_attestation_rejects_extra_file(tmp_path):
    root, _module, _link, entries = _release_tree(tmp_path)
    root.chmod(0o755)
    extra = root / "extra.py"
    extra.write_bytes(b"extra")
    extra.chmod(0o444)
    root.chmod(0o555)

    with pytest.raises(ValueError, match="live paths differ"):
        collector._verify_release_entries(
            root,
            entries,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_release_entry_attestation_rejects_escaping_symlink(tmp_path):
    root, _module, link, entries = _release_tree(tmp_path)
    root.chmod(0o755)
    link.unlink()
    link.symlink_to("/etc/hosts")
    root.chmod(0o555)
    entries[-1] = {**entries[-1], "target": "/etc/hosts"}

    with pytest.raises(ValueError, match="escaped"):
        collector._verify_release_entries(
            root,
            entries,
            expected_uid=os.getuid(),
            expected_gid=os.getgid(),
        )


def test_credential_identity_includes_change_and_modification_times():
    base = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat.S_IFREG | 0o400,
        st_nlink=1,
        st_uid=collector.WRITER_UID,
        st_gid=collector.WRITER_GID,
        st_size=64,
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    changed = SimpleNamespace(**{**vars(base), "st_ctime_ns": 5})

    assert collector._same_file_identity(base, base) is True
    assert collector._same_file_identity(base, changed) is False
    assert collector._database_config(TLS_NAME, credential_fd=123).credential.fd == 123


def test_append_only_receipt_never_publishes_partial_canonical_path(
    monkeypatch,
    tmp_path,
):
    parent = tmp_path / "receipts"
    parent.mkdir(mode=0o700)
    monkeypatch.setattr(
        collector,
        "_validate_protected_ancestor_chain",
        lambda _path: None,
    )
    unsigned = {"schema": collector.COLLECTOR_RECEIPT_SCHEMA, "value": "exact"}
    receipt_sha = collector._sha256_json(unsigned)
    receipt = {**unsigned, "receipt_sha256": receipt_sha}
    path = parent / f"{receipt_sha}.json"
    original_write = os.write

    def interrupted_write(descriptor, value):
        original_write(descriptor, value[:1])
        raise OSError("injected interrupted write")

    monkeypatch.setattr(collector.os, "write", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        collector._write_append_only_receipt(
            path,
            receipt,
            uid=os.getuid(),
            gid=os.getgid(),
        )
    assert not path.exists()
    assert list(parent.iterdir()) == []

    monkeypatch.setattr(collector.os, "write", original_write)
    collector._write_append_only_receipt(
        path,
        receipt,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    assert path.read_bytes() == collector._canonical_bytes(receipt)
    collector._write_append_only_receipt(
        path,
        receipt,
        uid=os.getuid(),
        gid=os.getgid(),
    )
    assert path.read_bytes() == collector._canonical_bytes(receipt)


@pytest.mark.parametrize("readback_matches", [True, False])
def test_staged_config_readback_controls_receipt(
    monkeypatch,
    tmp_path,
    readback_matches,
):
    stage = tmp_path / "staged"
    stage.mkdir(mode=0o700)
    evidence = tmp_path / "evidence"
    monkeypatch.setattr(collector, "STAGING_ROOT", stage)
    monkeypatch.setattr(collector, "STAGED_WRITER_CONFIG_PATH", stage / "writer.json")
    monkeypatch.setattr(collector, "STAGED_GATEWAY_CONFIG_PATH", stage / "gateway.yaml")
    monkeypatch.setattr(collector, "EVIDENCE_ROOT", evidence)
    monkeypatch.setattr(collector, "_require_root_linux", lambda: None)
    releases: list[str] = []
    monkeypatch.setattr(
        collector,
        "_load_release_binding",
        lambda **kwargs: releases.append(kwargs["revision"]),
    )
    monkeypatch.setattr(
        collector,
        "_read_trusted_public_file",
        lambda *_args, **_kwargs: b"trusted-ca",
    )
    stat_mode = stat.S_IFREG | 0o400
    identity = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=stat_mode,
        st_nlink=1,
        st_uid=collector.WRITER_UID,
        st_gid=collector.WRITER_GID,
        st_size=32,
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    provenance = {
        "path": str(collector.DATABASE_CREDENTIAL_PATH),
        "device": 1,
        "inode": 2,
        "owner_uid": collector.WRITER_UID,
        "group_gid": collector.WRITER_GID,
        "mode": "0400",
        "link_count": 1,
        "modification_time_ns": 3,
        "change_time_ns": 4,
        "content_or_digest_recorded": False,
    }

    @contextmanager
    def pinned():
        yield 123, provenance

    monkeypatch.setattr(
        collector,
        "_pinned_credential",
        pinned,
    )
    monkeypatch.setattr(collector.os, "lstat", lambda _path: identity)
    attestation = _attestation()
    hba = _hba()
    policy = collector._policy_from_attestation(attestation, hba_receipt=hba)
    monkeypatch.setattr(
        collector,
        "_collect_live_policy",
        lambda *_args, **_kwargs: (policy, attestation, hba),
    )
    monkeypatch.setattr(
        collector,
        "_ensure_exact_directory",
        lambda path, **_kwargs: path.mkdir(parents=True, exist_ok=True),
    )
    staged: dict[Path, bytes] = {}
    monkeypatch.setattr(
        collector,
        "_atomic_replace_file",
        lambda path, payload, **_kwargs: staged.__setitem__(path, payload),
    )
    config = collector._database_config(TLS_NAME)
    loaded_database = config
    monkeypatch.setattr(
        collector,
        "load_service_config",
        lambda _path: SimpleNamespace(
            database=loaded_database,
            privileges=policy,
            gateway_uid=collector.GATEWAY_UID,
            writer_uid=collector.WRITER_UID,
            writer_gid=collector.WRITER_GID,
            socket_gid=collector.SOCKET_CLIENT_GID,
            projector_gid=collector.PROJECTOR_GID,
            owner_discord_user_ids=(
                frozenset() if readback_matches else frozenset({"unexpected"})
            ),
            discord_edge_authority=SimpleNamespace(enabled=False),
        ),
    )
    receipts: list[object] = []
    monkeypatch.setattr(
        collector,
        "_write_append_only_receipt",
        lambda *_args, **_kwargs: receipts.append(object()),
    )
    now = int(time.time())

    context = (
        nullcontext()
        if readback_matches
        else pytest.raises(RuntimeError, match="readback drifted")
    )
    with context:
        result = collector.collect_and_stage(
            revision=REVISION,
            release_artifact_sha256="1" * 64,
            release_manifest_file_sha256="2" * 64,
            tls_server_name=TLS_NAME,
            owner_discord_user_ids=(),
            _clock=lambda: now,
        )

    assert stat.S_IMODE(stat_mode) == 0o400
    assert releases == [REVISION, REVISION]
    assert set(staged) == {
        collector.STAGED_WRITER_CONFIG_PATH,
        collector.STAGED_GATEWAY_CONFIG_PATH,
    }
    if readback_matches:
        assert result["ok"] is True
        assert len(receipts) == 1
    else:
        assert receipts == []
