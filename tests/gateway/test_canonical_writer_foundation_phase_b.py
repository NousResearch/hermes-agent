from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
import sys
import threading
import time
import struct
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_foundation as foundation
from gateway import canonical_writer_foundation_phase_b as phase_b
from gateway import canonical_writer_phase_b_runtime as phase_b_runtime


pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Phase-B hardened journal execution is an explicit POSIX Cloud boundary",
)


REVISION = "a" * 40
OWNER = "b" * 64
TLS_PEER = "c" * 64
NOW = 2_000_000_000
ROLE_PAYLOAD = b"-- sealed phase-b role fixture\n"
OWNER_SIGNING_KEY = Ed25519PrivateKey.from_private_bytes(b"\x19" * 32)
OWNER_PUBLIC_RAW = OWNER_SIGNING_KEY.public_key().public_bytes(
    serialization.Encoding.Raw,
    serialization.PublicFormat.Raw,
)
OWNER_PUBLIC_FILE_SHA256 = "9" * 64


def _ssh_string(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def _sshsig(message: bytes, namespace: str) -> str:
    algorithm = b"ssh-ed25519"
    namespace_bytes = namespace.encode("ascii")
    signed = (
        b"SSHSIG"
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(hashlib.sha512(message).digest())
    )
    signature = OWNER_SIGNING_KEY.sign(signed)
    public_blob = _ssh_string(algorithm) + _ssh_string(OWNER_PUBLIC_RAW)
    signature_blob = _ssh_string(algorithm) + _ssh_string(signature)
    envelope = (
        b"SSHSIG"
        + struct.pack(">I", 1)
        + _ssh_string(public_blob)
        + _ssh_string(namespace_bytes)
        + _ssh_string(b"")
        + _ssh_string(b"sha512")
        + _ssh_string(signature_blob)
    )
    body = base64.b64encode(envelope).decode("ascii")
    lines = [body[index : index + 70] for index in range(0, len(body), 70)]
    return (
        "-----BEGIN SSH SIGNATURE-----\n"
        + "\n".join(lines)
        + "\n-----END SSH SIGNATURE-----\n"
    )


def _hashed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**value, field: phase_b._sha256_json(value)}


def _rewrite_journal_entries(
    journal: phase_b.AppendOnlyPhaseBJournal,
    plan: phase_b.PhaseBPlan,
    values: list[dict[str, Any]],
) -> None:
    entries_root = journal._entries_root(plan)
    for path in entries_root.iterdir():
        path.unlink()
    previous: str | None = None
    for sequence, value in enumerate(values):
        value["sequence"] = sequence
        value["previous_entry_sha256"] = previous
        unsigned = {
            name: item for name, item in value.items() if name != "entry_sha256"
        }
        value["entry_sha256"] = phase_b._sha256_json(unsigned)
        previous = value["entry_sha256"]
        path = entries_root / f"{sequence:08d}.json"
        path.write_bytes(phase_b._canonical_bytes(value))
        path.chmod(0o600)


def _pg_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key not in {"unsigned_receipt_jsonb_text", "receipt_sha256"}
    }

    def jsonb_order(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: jsonb_order(item[key])
                for key in sorted(
                    item,
                    key=lambda name: (
                        len(name.encode("utf-8")),
                        name.encode("utf-8"),
                    ),
                )
            }
        if isinstance(item, list):
            return [jsonb_order(child) for child in item]
        return item

    unsigned_text = json.dumps(
        jsonb_order(unsigned),
        ensure_ascii=False,
        separators=(", ", ": "),
        allow_nan=False,
    )
    return {
        **unsigned,
        "unsigned_receipt_jsonb_text": unsigned_text,
        "receipt_sha256": phase_b._sha256_bytes(unsigned_text.encode("utf-8")),
    }


def _pg_role_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _pg_hashed(value)
    unsigned_text = normalized.pop("unsigned_receipt_jsonb_text")
    return {
        "unsigned_receipt_jsonb_text": unsigned_text,
        "receipt": normalized,
    }


def _role(
    oid: int,
    name: str,
    *,
    can_login: bool,
    inherits: bool,
    create_database: bool = False,
    create_role: bool = False,
) -> dict[str, Any]:
    return {
        "oid": str(oid),
        "name": name,
        "can_login": can_login,
        "inherits": inherits,
        "superuser": False,
        "create_database": create_database,
        "create_role": create_role,
        "replication": False,
        "bypass_row_security": False,
        "connection_limit": -1,
        "validity_is_unbounded": True,
        "configuration_is_empty": True,
    }


def _acl(
    grantor_oid: int,
    grantor: str,
    grantee_oid: int,
    grantee: str,
    privilege: str,
) -> dict[str, Any]:
    return {
        "grantor_oid": str(grantor_oid),
        "grantor": grantor,
        "grantee_oid": str(grantee_oid),
        "grantee": grantee,
        "privilege": privilege,
        "grantable": False,
    }


def _database_row(
    oid: int,
    name: str,
    owner_oid: int,
    owner: str,
    acl: list[dict[str, Any]],
    *,
    public_connect: bool = False,
    public_temporary: bool = False,
    is_template: bool = False,
) -> dict[str, Any]:
    return {
        "oid": str(oid),
        "name": name,
        "owner_oid": str(owner_oid),
        "owner": owner,
        "allow_connections": True,
        "is_template": is_template,
        "connection_limit": -1,
        "acl_is_null": False,
        "acl": acl,
        "effective_public_connect": public_connect,
        "effective_public_temporary": public_temporary,
    }


def _namespaces() -> list[dict[str, Any]]:
    rows = [
        {
            "oid": "301",
            "name": "canonical_brain",
            "owner_oid": "101",
            "owner": foundation.MIGRATION_OWNER_ROLE,
            "acl_is_null": False,
            "acl": sorted(
                [
                    _acl(
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        "CREATE",
                    ),
                    _acl(
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        "USAGE",
                    ),
                    _acl(
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        102,
                        foundation.WRITER_ROLE,
                        "USAGE",
                    ),
                ],
                key=lambda row: (row["grantee"], row["privilege"], row["grantor"]),
            ),
        },
        {
            "oid": "302",
            "name": "canonical_brain_legacy_quarantine",
            "owner_oid": "107",
            "owner": "legacy_archive_source_owner",
            "acl_is_null": False,
            "acl": [
                _acl(
                    107,
                    "legacy_archive_source_owner",
                    107,
                    "legacy_archive_source_owner",
                    "CREATE",
                ),
                _acl(
                    107,
                    "legacy_archive_source_owner",
                    107,
                    "legacy_archive_source_owner",
                    "USAGE",
                ),
            ],
        },
        {
            "oid": "300",
            "name": "public",
            "owner_oid": "110",
            "owner": "pg_database_owner",
            "acl_is_null": False,
            "acl": sorted(
                [
                    _acl(110, "pg_database_owner", 110, "pg_database_owner", "CREATE"),
                    _acl(110, "pg_database_owner", 110, "pg_database_owner", "USAGE"),
                    _acl(
                        110,
                        "pg_database_owner",
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        "USAGE",
                    ),
                ],
                key=lambda row: (row["grantee"], row["privilege"], row["grantor"]),
            ),
        },
    ]
    return sorted(rows, key=lambda row: row["name"])


def _relation_column(
    position: int,
    name: str,
    type_name: str,
    *,
    not_null: bool,
    has_default: bool,
) -> dict[str, Any]:
    return {
        "position": position,
        "name": name,
        "type_oid": phase_b._TYPE_OIDS[type_name],
        "type": type_name,
        "not_null": not_null,
        "has_default": has_default,
        "default_expression_sha256": "d" * 64 if has_default else None,
        "identity": "",
        "generated": "",
        "has_missing": False,
        "is_local": True,
        "inheritance_count": 0,
        "array_dimensions": 0,
        "collation_is_type_default": True,
        "storage_is_type_default": True,
        "statistics_target": None,
        "options_are_empty": True,
        "fdw_options_are_empty": True,
        "acl_is_null": True,
        "acl": [],
    }


def _primary_constraint(oid: int, index_oid: int, name: str) -> dict[str, Any]:
    return {
        "oid": str(oid),
        "name": name,
        "type": "p",
        "validated": True,
        "deferrable": False,
        "initially_deferred": False,
        "no_inherit": True,
        "index_oid": str(index_oid),
        "parent_constraint_oid": "0",
        "column_numbers": [1],
        "column_names": ["event_id"],
        "definition_sha256": "a" * 64,
    }


def _index(
    oid: int,
    name: str,
    key_name: str,
    attribute_number: int,
    *,
    owner_oid: int,
    owner: str,
    primary: bool = False,
    unique: bool = False,
    predicate: bool = False,
    constraint_oid: int | None = None,
) -> dict[str, Any]:
    operator_name = {
        "event_id": "uuid_ops",
        "occurred_at": "timestamptz_ops",
    }.get(key_name, "text_ops")
    return {
        "oid": str(oid),
        "name": name,
        "owner_oid": str(owner_oid),
        "owner": owner,
        "relation_kind": "i",
        "persistence": "p",
        "access_method": "btree",
        "tablespace_oid": "0",
        "options_are_empty": True,
        "unique": unique,
        "nulls_not_distinct": False,
        "primary": primary,
        "exclusion": False,
        "immediate": True,
        "clustered": False,
        "valid": True,
        "check_xmin": False,
        "ready": True,
        "live": True,
        "replica_identity": False,
        "key_attribute_count": 1,
        "attribute_count": 1,
        "key_columns": [
            {
                "position": 1,
                "attribute_number": attribute_number,
                "name": key_name,
            }
        ],
        "operator_classes": [
            {
                "position": 1,
                "oid": "900",
                "schema": "pg_catalog",
                "name": operator_name,
            }
        ],
        "collation_oids": ["0"],
        "index_options": [0],
        "expressions_present": False,
        "expressions_sha256": None,
        "predicate_present": predicate,
        "predicate_sha256": "e" * 64 if predicate else None,
        "constraint_oids": [] if constraint_oid is None else [str(constraint_oid)],
    }


def _relation_acl(owner_oid: int, owner: str) -> list[dict[str, Any]]:
    return [
        _acl(owner_oid, owner, owner_oid, owner, privilege)
        for privilege in (
            "DELETE",
            "INSERT",
            "MAINTAIN",
            "REFERENCES",
            "SELECT",
            "TRIGGER",
            "TRUNCATE",
            "UPDATE",
        )
    ]


def _event_log() -> dict[str, Any]:
    columns = [
        _relation_column(
            position,
            name,
            data_type,
            not_null=True,
            has_default=False,
        )
        for position, (name, data_type) in enumerate(phase_b._EVENT_COLUMNS, start=1)
    ]
    return {
        "cardinality": 1,
        "identity": {
            "namespace_oid": "300",
            "oid": "500",
            "owner_oid": "101",
            "owner": foundation.MIGRATION_OWNER_ROLE,
            "relation_kind": "r",
            "persistence": "p",
            "is_partition": False,
            "access_method": "heap",
            "tablespace_oid": "0",
            "row_security": False,
            "force_row_security": False,
            "replica_identity": "d",
            "options_are_empty": True,
            "attribute_slots": 14,
            "relation_acl_is_null": False,
            "relation_acl": _relation_acl(101, foundation.MIGRATION_OWNER_ROLE),
            "columns": columns,
            "constraints": [_primary_constraint(501, 502, "canonical_event_log_pkey")],
            "indexes": [
                _index(
                    502,
                    "canonical_event_log_pkey",
                    "event_id",
                    1,
                    owner_oid=101,
                    owner=foundation.MIGRATION_OWNER_ROLE,
                    primary=True,
                    unique=True,
                    constraint_oid=501,
                )
            ],
            "user_triggers": [],
            "rules": [],
            "policies": [],
            "inheritance": [],
        },
    }


def _writer_ping() -> dict[str, Any]:
    return {
        "cardinality": 1,
        "routines": [
            {
                "oid": "600",
                "namespace_oid": "301",
                "owner_oid": "101",
                "owner": foundation.MIGRATION_OWNER_ROLE,
                "language": "plpgsql",
                "kind": "f",
                "argument_types": [
                    {"position": 1, "schema": "pg_catalog", "name": "jsonb"},
                    {"position": 2, "schema": "pg_catalog", "name": "jsonb"},
                ],
                "return_type": {"schema": "pg_catalog", "name": "jsonb"},
                "returns_set": False,
                "security_definer": True,
                "leakproof": False,
                "strict": False,
                "volatility": "v",
                "parallel": "u",
                "configuration_count": 1,
                "configuration_is_exact": True,
                "acl_is_null": False,
                "acl": [
                    _acl(
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        "EXECUTE",
                    ),
                    _acl(
                        101,
                        foundation.MIGRATION_OWNER_ROLE,
                        102,
                        foundation.WRITER_ROLE,
                        "EXECUTE",
                    ),
                ],
                "implementation_sha256": "d" * 64,
            }
        ],
    }


def _legacy_archive() -> dict[str, Any]:
    columns = [
        _relation_column(
            position,
            name,
            type_name,
            not_null=not_null,
            has_default=has_default,
        )
        for position, (name, type_name, not_null, has_default) in enumerate(
            phase_b._ARCHIVE_COLUMNS, start=1
        )
    ]
    return {
        "cardinality": 1,
        "identity": {
            "namespace_oid": "302",
            "oid": "700",
            "owner_oid": "107",
            "owner": "legacy_archive_source_owner",
            "relation_kind": "r",
            "persistence": "p",
            "is_partition": False,
            "access_method": "heap",
            "tablespace_oid": "0",
            "row_security": False,
            "force_row_security": False,
            "replica_identity": "d",
            "options_are_empty": True,
            "attribute_slots": 19,
            "relation_acl_is_null": False,
            "relation_acl": _relation_acl(107, "legacy_archive_source_owner"),
            "columns": columns,
            "constraints": [
                _primary_constraint(701, 702, "canonical_event_log_legacy_v1_pkey")
            ],
            "indexes": [
                _index(
                    702,
                    "canonical_event_log_legacy_v1_pkey",
                    "event_id",
                    1,
                    owner_oid=107,
                    owner="legacy_archive_source_owner",
                    primary=True,
                    unique=True,
                    constraint_oid=701,
                ),
                _index(
                    703,
                    "legacy_case_id_idx",
                    "case_id",
                    5,
                    owner_oid=107,
                    owner="legacy_archive_source_owner",
                ),
                _index(
                    704,
                    "legacy_event_type_idx",
                    "event_type",
                    3,
                    owner_oid=107,
                    owner="legacy_archive_source_owner",
                ),
                _index(
                    705,
                    "legacy_occurred_at_idx",
                    "occurred_at",
                    4,
                    owner_oid=107,
                    owner="legacy_archive_source_owner",
                ),
                _index(
                    706,
                    "legacy_idempotency_key_idx",
                    "idempotency_key",
                    16,
                    owner_oid=107,
                    owner="legacy_archive_source_owner",
                    unique=True,
                    predicate=True,
                ),
            ],
            "user_triggers": [],
            "rules": [],
            "policies": [],
            "inheritance": [],
            "owner_superuser": False,
            "owner_create_database": False,
            "owner_create_role": False,
            "owner_replication": False,
            "owner_bypass_row_security": False,
            "owner_connection_limit": -1,
            "owner_validity_is_unbounded": True,
            "owner_configuration_is_empty": True,
        },
    }


def _database_preflight(
    *,
    bootstrap_role: bool = False,
    bootstrap_login: bool = False,
    temporary_admin: str | None = None,
    auto_bridge: bool = False,
) -> dict[str, Any]:
    roles = [
        _role(101, foundation.MIGRATION_OWNER_ROLE, can_login=False, inherits=False),
        _role(102, foundation.WRITER_ROLE, can_login=False, inherits=True),
        _role(103, foundation.SQL_USER, can_login=True, inherits=True),
    ]
    if bootstrap_role:
        roles.append(
            _role(104, foundation.CANARY_BOOTSTRAP_ROLE, can_login=False, inherits=False)
        )
    if bootstrap_login:
        roles.append(
            _role(105, foundation.CANARY_BOOTSTRAP_LOGIN, can_login=True, inherits=True)
        )
    roles.sort(key=lambda row: row["name"])
    memberships = [
        {
            "granted_role": foundation.WRITER_ROLE,
            "member_role": foundation.SQL_USER,
            "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": False,
            "inherit_option": True,
            "set_option": False,
        }
    ]
    temp_roles: list[dict[str, Any]] = []
    if bootstrap_login:
        memberships.append(
            {
                "granted_role": foundation.CANARY_BOOTSTRAP_ROLE,
                "member_role": foundation.CANARY_BOOTSTRAP_LOGIN,
                "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": False,
                "inherit_option": True,
                "set_option": False,
            }
        )
    if temporary_admin is not None:
        temp_roles.append(
            _role(
                106,
                temporary_admin,
                can_login=True,
                inherits=True,
                create_database=True,
                create_role=True,
            )
        )
        memberships.append(
            {
                "granted_role": foundation.DATABASE_OWNER_ROLE,
                "member_role": temporary_admin,
                "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": False,
                "inherit_option": True,
                "set_option": True,
            }
        )
        if bootstrap_role and auto_bridge:
            memberships.append(
                {
                    "granted_role": foundation.CANARY_BOOTSTRAP_ROLE,
                    "member_role": temporary_admin,
                    "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                    "admin_option": True,
                    "inherit_option": False,
                    "set_option": False,
                }
            )
    memberships.sort(
        key=lambda row: (row["granted_role"], row["member_role"], row["grantor"])
    )
    target_acl = [
        _acl(100, foundation.DATABASE_OWNER_ROLE, 102, foundation.WRITER_ROLE, "CONNECT"),
        _acl(100, foundation.DATABASE_OWNER_ROLE, 100, foundation.DATABASE_OWNER_ROLE, "CONNECT"),
        _acl(100, foundation.DATABASE_OWNER_ROLE, 100, foundation.DATABASE_OWNER_ROLE, "CREATE"),
        _acl(100, foundation.DATABASE_OWNER_ROLE, 100, foundation.DATABASE_OWNER_ROLE, "TEMPORARY"),
    ]
    if bootstrap_role:
        target_acl.append(
            _acl(
                100,
                foundation.DATABASE_OWNER_ROLE,
                104,
                foundation.CANARY_BOOTSTRAP_ROLE,
                "CONNECT",
            )
        )
    target_acl.sort(key=lambda row: (row["grantee"], row["privilege"], row["grantor"]))
    target = _database_row(
        200, foundation.SQL_DATABASE, 100, foundation.DATABASE_OWNER_ROLE, target_acl
    )
    cloudsqladmin = _database_row(
        201,
        "cloudsqladmin",
        108,
        "cloudsqladmin",
        [
            _acl(108, "cloudsqladmin", 0, "PUBLIC", "CONNECT"),
            _acl(108, "cloudsqladmin", 0, "PUBLIC", "TEMPORARY"),
        ],
        public_connect=True,
        public_temporary=True,
    )
    postgres = _database_row(
        202,
        "postgres",
        109,
        "postgres",
        [
            _acl(109, "postgres", 109, "postgres", "CONNECT"),
            _acl(109, "postgres", 109, "postgres", "CREATE"),
            _acl(109, "postgres", 109, "postgres", "TEMPORARY"),
        ],
    )
    template = _database_row(
        203,
        "template1",
        109,
        "postgres",
        [
            _acl(109, "postgres", 109, "postgres", "CONNECT"),
            _acl(109, "postgres", 109, "postgres", "CREATE"),
            _acl(109, "postgres", 109, "postgres", "TEMPORARY"),
        ],
        is_template=True,
    )
    databases = sorted([target, cloudsqladmin, postgres, template], key=lambda row: row["name"])
    managed_privileges = [
        {
            "database_oid": row["oid"],
            "database": row["name"],
            "effective_connect": True,
            "effective_temporary": True,
            "direct_acl": [],
        }
        for row in databases
    ]
    unsigned = {
        "schema": "muncho-canonical-writer-foundation-phase-b-db-preflight.v1",
        "preflight": True,
        "terminal": False,
        "database": foundation.SQL_DATABASE,
        "database_owner": foundation.DATABASE_OWNER_ROLE,
        "postgres_version_num": 180004,
        "session_user": foundation.SQL_USER,
        "current_user": foundation.SQL_USER,
        "roles": roles,
        "memberships": memberships,
        "temporary_admin_roles": temp_roles,
        "bootstrap_role_absent": not bootstrap_role,
        "bootstrap_login_absent": not bootstrap_login,
        "namespaces": _namespaces(),
        "event_log": _event_log(),
        "writer_ping": _writer_ping(),
        "legacy_archive": _legacy_archive(),
        "target_database": target,
        "other_connectable_databases": [cloudsqladmin, postgres, template],
        "managed_cloudsqladmin": {
            "role_cardinality": 1,
            "role": _role(108, "cloudsqladmin", can_login=False, inherits=True),
            "database_privileges": managed_privileges,
        },
        "secret_material_recorded": False,
    }
    return _pg_hashed(unsigned)


def _credential() -> dict[str, Any]:
    return {
        "state": "installed",
        "path": str(foundation.DATABASE_CREDENTIAL_PATH),
        "device": 11,
        "inode": 12,
        "owner_uid": foundation.WRITER_UID,
        "group_gid": foundation.WRITER_GID,
        "mode": "0400",
        "link_count": 1,
        "modification_time_ns": 13,
        "change_time_ns": 14,
        "content_or_digest_recorded": False,
    }


def _services(
    observed_at: int = NOW,
    release_revision: str = REVISION,
) -> dict[str, Any]:
    rows = []
    for name in phase_b.SERVICE_UNITS:
        rows.append(
            {
                "name": name,
                "load_state": "not-found",
                "active_state": "inactive",
                "sub_state": "dead",
                "unit_file_state": "not-found",
                "main_pid": 0,
                "fragment_path": None,
                "drop_in_paths": [],
                "triggered_by": [],
                "triggers": [],
                "next_elapse_unix_usec": None,
            }
        )
    unsigned = {
        "schema": "muncho-canonical-writer-phase-b-services-stopped.v1",
        "release_revision": release_revision,
        "services": rows,
        "services_stopped_and_disabled": True,
        "observed_at_unix": observed_at,
    }
    return _hashed(unsigned, "attestation_sha256")


def _database(session_sha: str) -> dict[str, Any]:
    return {
        "project": foundation.PROJECT,
        "instance": foundation.SQL_INSTANCE,
        "host": foundation.SQL_HOST,
        "port": foundation.SQL_PORT,
        "database": foundation.SQL_DATABASE,
        "database_owner": foundation.DATABASE_OWNER_ROLE,
        "postgres_version_num": 180004,
        "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
        "tls_peer_certificate_sha256": TLS_PEER,
        "session_user": foundation.SQL_USER,
        "current_user": foundation.SQL_USER,
        "session_identity_sha256": session_sha,
    }


def _cloud_user_resource(
    name: str,
    *,
    database_roles: list[str],
    etag: str,
) -> dict[str, Any]:
    return {
        "databaseRoles": database_roles,
        "etag": etag,
        "host": "",
        "instance": foundation.SQL_INSTANCE,
        "name": name,
        "project": foundation.PROJECT,
        "type": "BUILT_IN",
    }


def _terminal_user_inventory(
    bootstrap_resource: Mapping[str, Any],
) -> list[dict[str, Any]]:
    resources = [
        copy.deepcopy(dict(bootstrap_resource)),
        _cloud_user_resource(
            foundation.SQL_USER,
            database_roles=[],
            etag="etag-writer",
        ),
        _cloud_user_resource(
            "postgres",
            database_roles=[],
            etag="etag-postgres",
        ),
    ]
    return sorted(resources, key=lambda item: item["name"])


def _readiness_host_identity(observed_at_unix: int) -> dict[str, Any]:
    gce = {
        "project_id": phase_b.DEDICATED_CANARY_PROJECT_ID,
        "project_number": phase_b.DEDICATED_CANARY_PROJECT_NUMBER,
        "zone": phase_b.DEDICATED_CANARY_ZONE,
        "instance_name": phase_b.DEDICATED_CANARY_INSTANCE_NAME,
        "instance_id": phase_b.DEDICATED_CANARY_INSTANCE_ID,
        "service_account_email": phase_b.DEDICATED_CANARY_SERVICE_ACCOUNT,
    }
    unsigned = {
        "schema": phase_b.FULL_CANARY_HOST_IDENTITY_SCHEMA,
        "collector_authority": "trusted_root_read_only_host_collector",
        **gce,
        "gce_identity_sha256": phase_b._sha256_json(gce),
        "machine_id_sha256": "1" * 64,
        "hostname_sha256": "2" * 64,
        "host_identity_sha256": "3" * 64,
        "boot_id_sha256": "4" * 64,
        "observed_at_unix": observed_at_unix,
    }
    scopes = list(phase_b.PHASE_B_VM_OAUTH_SCOPES)
    return {
        **unsigned,
        "receipt_sha256": phase_b._sha256_json(unsigned),
        "oauth_scopes": scopes,
        "oauth_scopes_sha256": phase_b._sha256_json(scopes),
    }


def _preflight(session_sha: str = "e" * 64) -> phase_b.PhaseBPreflight:
    artifacts = {
        phase_b.PREFLIGHT_ARTIFACT_PATH: "f" * 64,
        phase_b.ROLE_ARTIFACT_PATH: phase_b._sha256_bytes(ROLE_PAYLOAD),
    }
    unsigned = {
        "schema": phase_b.PHASE_B_PREFLIGHT_SCHEMA,
        "release_revision": REVISION,
        "release_manifest_sha256": "2" * 64,
        "release_artifacts": artifacts,
        "release_artifact_set_sha256": phase_b._sha256_json(artifacts),
        "database": _database(session_sha),
        "foundation": _database_preflight(),
        "credential": _credential(),
        "services": _services(),
        "cloud_sql": {
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "visible_users": sorted([foundation.SQL_USER, "postgres"]),
            "user_inventory_sha256": "3" * 64,
            "bootstrap_login_absent": True,
            "temporary_admin_users": [],
            "user_operations_quiescent": True,
            "operation_ledger_sha256": "4" * 64,
        },
        "observed_at_unix": NOW,
    }
    return phase_b.PhaseBPreflight.from_mapping(
        _hashed(unsigned, "observation_sha256")
    )


def _plan() -> phase_b.PhaseBPlan:
    return phase_b.build_phase_b_plan(
        _preflight(),
        owner_subject_sha256=OWNER,
        owner_resume_public_key_ed25519_hex=OWNER_PUBLIC_RAW.hex(),
        owner_resume_public_key_file_sha256=OWNER_PUBLIC_FILE_SHA256,
    )


def _signed_approval(
    plan: phase_b.PhaseBPlan,
    *,
    sequence: int,
    previous_approval_sha256: str | None,
    issued_at_unix: int,
    expires_at_unix: int,
    source_authentication_sha256: str,
) -> dict[str, Any]:
    unsigned = {
        "schema": phase_b.PHASE_B_APPROVAL_SCHEMA,
        "purpose": "initial_apply" if sequence == 0 else "resume_incomplete",
        "sequence": sequence,
        "previous_approval_sha256": previous_approval_sha256,
        "plan_sha256": plan.sha256,
        "intent_sha256": plan.value["intent_sha256"],
        "owner_subject_sha256": OWNER,
        "approval_source_sha256": "5" * 64,
        "owner_public_key_ed25519_hex": OWNER_PUBLIC_RAW.hex(),
        "owner_key_id": hashlib.sha256(OWNER_PUBLIC_RAW).hexdigest(),
        "owner_public_key_file_sha256": OWNER_PUBLIC_FILE_SHA256,
        "source_authentication_sha256": source_authentication_sha256,
        "approved": True,
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "secret_material_recorded": False,
    }
    signed = {
        **unsigned,
        "signature_sshsig": _sshsig(
            phase_b._canonical_bytes(unsigned),
            phase_b.PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
        ),
    }
    return _hashed(signed, "approval_sha256")


def _signed_source_authentication(
    plan: phase_b.PhaseBPlan,
    *,
    sequence: int,
    previous_approval_sha256: str | None,
    issued_at_unix: int,
    expires_at_unix: int,
    nonce_sha256: str,
) -> dict[str, Any]:
    unsigned = {
        "schema": phase_b.PHASE_B_SOURCE_AUTH_SCHEMA,
        "authority_kind": "skyvision_mac_ops_ed25519",
        "purpose": "initial_apply" if sequence == 0 else "resume_incomplete",
        "sequence": sequence,
        "previous_approval_sha256": previous_approval_sha256,
        "plan_sha256": plan.sha256,
        "intent_sha256": plan.value["intent_sha256"],
        "owner_subject_sha256": plan.owner_subject_sha256,
        "approval_source_sha256": "5" * 64,
        "owner_key_id": hashlib.sha256(OWNER_PUBLIC_RAW).hexdigest(),
        "requested_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "nonce_sha256": nonce_sha256,
    }
    signed = {
        **unsigned,
        "signature_sshsig": _sshsig(
            phase_b._canonical_bytes(unsigned),
            phase_b.PHASE_B_SOURCE_AUTH_SSHSIG_NAMESPACE,
        ),
    }
    return _hashed(signed, "receipt_sha256")


def _signed_approval_pair(
    plan: phase_b.PhaseBPlan,
    *,
    sequence: int,
    previous_approval_sha256: str | None,
    issued_at_unix: int,
    expires_at_unix: int,
    nonce_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _signed_source_authentication(
        plan,
        sequence=sequence,
        previous_approval_sha256=previous_approval_sha256,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        nonce_sha256=nonce_sha256,
    )
    approval = _signed_approval(
        plan,
        sequence=sequence,
        previous_approval_sha256=previous_approval_sha256,
        issued_at_unix=issued_at_unix,
        expires_at_unix=expires_at_unix,
        source_authentication_sha256=source["receipt_sha256"],
    )
    return source, approval


def _approval(plan: phase_b.PhaseBPlan) -> dict[str, Any]:
    return _signed_approval(
        plan,
        sequence=0,
        previous_approval_sha256=None,
        issued_at_unix=NOW - 1,
        expires_at_unix=NOW + 300,
        source_authentication_sha256="6" * 64,
    )


def _approved(plan: phase_b.PhaseBPlan) -> phase_b.PhaseBApproval:
    return phase_b.PhaseBApproval.from_mapping(
        _approval(plan),
        plan=plan,
        now_unix=NOW,
    )


def test_plan_is_deterministic_and_derives_fixed_admin() -> None:
    first = _plan()
    second = _plan()

    assert first.to_mapping() == second.to_mapping()
    assert first.temporary_admin_username == (
        phase_b.TEMPORARY_ADMIN_PREFIX + first.value["intent_sha256"][:16]
    )
    assert first.value["states"] == list(phase_b.JOURNAL_EVENT_ORDER)


def test_preflight_rejects_public_cross_database_or_missing_writer_acl() -> None:
    value = _preflight().to_mapping()
    value["foundation"]["other_connectable_databases"][1][
        "effective_public_connect"
    ] = True
    value["foundation"] = _pg_hashed(value["foundation"])
    value["observation_sha256"] = phase_b._sha256_json(
        {key: item for key, item in value.items() if key != "observation_sha256"}
    )
    with pytest.raises(phase_b.PhaseBError, match="cross_database"):
        phase_b.PhaseBPreflight.from_mapping(value)

    value = _preflight().to_mapping()
    value["foundation"]["target_database"]["acl"] = [
        row
        for row in value["foundation"]["target_database"]["acl"]
        if row["grantee"] != foundation.WRITER_ROLE
    ]
    value["foundation"] = _pg_hashed(value["foundation"])
    value["observation_sha256"] = phase_b._sha256_json(
        {key: item for key, item in value.items() if key != "observation_sha256"}
    )
    with pytest.raises(phase_b.PhaseBError, match="target_database"):
        phase_b.PhaseBPreflight.from_mapping(value)


def test_service_order_matches_fixed_six_unit_boundary() -> None:
    assert phase_b.SERVICE_UNITS == (
        "muncho-canary-discord-edge.service",
        "muncho-discord-egress.service",
        "muncho-canonical-writer.service",
        "muncho-canonical-writer-export.service",
        "muncho-canonical-writer-export.timer",
        "hermes-cloud-gateway.service",
    )
    assert [row["name"] for row in _services()["services"]] == list(
        phase_b.SERVICE_UNITS
    )


def test_journal_requires_lock_and_never_allows_early_terminal(tmp_path: Path) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with pytest.raises(phase_b.PhaseBError, match="lock_required"):
        journal.append(
            plan,
            approval=_approved(plan),
            event="intent",
            idempotency_key="intent",
            evidence={"safe_to_start": False},
            now_unix=NOW,
        )
    with journal.lock(plan):
        journal.append(
            plan,
            approval=_approved(plan),
            event="intent",
            idempotency_key="intent",
            evidence={"safe_to_start": False},
            now_unix=NOW,
        )
        with pytest.raises(phase_b.PhaseBError, match="prerequisite"):
            journal.append(
                plan,
                approval=_approved(plan),
                event="terminal",
                idempotency_key="terminal",
                evidence={"safe_to_start": True},
                now_unix=NOW,
            )
    assert [entry.event for entry in journal.load(plan)] == ["intent"]


def test_journal_rejects_secret_fields_symlink_lock_and_wrong_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with journal.lock(plan):
        with pytest.raises(phase_b.PhaseBError, match="secret_free"):
            journal.append(
                plan,
                approval=_approved(plan),
                event="intent",
                idempotency_key="intent",
                evidence={"password": "not-allowed"},
                now_unix=NOW,
            )

    other = phase_b.AppendOnlyPhaseBJournal(tmp_path / "other")
    other.root.mkdir(mode=0o700)
    other._plan_root(plan).mkdir(mode=0o700)
    os.chown(other.root, -1, os.getegid())
    os.chown(other._plan_root(plan), -1, os.getegid())
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    os.chmod(target, 0o600)
    other._lock_path(plan).symlink_to(target)
    with pytest.raises(phase_b.PhaseBError, match="lock_untrusted"):
        with other.lock(plan):
            pass

    current_uid = phase_b._effective_uid()
    monkeypatch.setattr(phase_b, "_effective_uid", lambda: current_uid + 1)
    with pytest.raises(phase_b.PhaseBError, match="directory_untrusted"):
        journal.load(plan)


def test_main_journal_recovers_partial_stage_after_write_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    evidence = {"safe_to_start": False}
    original_write = phase_b.os.write
    writes = 0

    def partial_then_crash(descriptor: int, payload: bytes) -> int:
        nonlocal writes
        writes += 1
        if writes == 1:
            return original_write(
                descriptor,
                payload[: max(1, len(payload) // 2)],
            )
        raise OSError("injected partial stage crash")

    monkeypatch.setattr(phase_b.os, "write", partial_then_crash)
    with pytest.raises(phase_b.PhaseBError, match="journal_write_failed"):
        with journal.lock(plan):
            journal.append(
                plan,
                approval=_approved(plan),
                event="intent",
                idempotency_key="intent",
                evidence=evidence,
                now_unix=NOW,
            )
    stages = list(journal._staging_root(plan).iterdir())
    assert len(stages) == 1
    assert stages[0].stat().st_nlink == 1
    assert list(journal._entries_root(plan).iterdir()) == []

    monkeypatch.setattr(phase_b.os, "write", original_write)
    with journal.lock(plan):
        recovered = journal.append(
            plan,
            approval=_approved(plan),
            event="intent",
            idempotency_key="intent",
            evidence=evidence,
            now_unix=NOW,
        )
    assert recovered.value["sequence"] == 0
    assert list(journal._staging_root(plan).iterdir()) == []
    assert [entry.event for entry in journal.load(plan)] == ["intent"]


def test_main_journal_recovers_linked_stage_after_publish_fsync_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    evidence = {"safe_to_start": False}
    original_link = phase_b.os.link
    original_fsync = phase_b.os.fsync
    linked = False
    failed = False

    def recording_link(*args: Any, **kwargs: Any) -> None:
        nonlocal linked
        original_link(*args, **kwargs)
        linked = True

    def fail_first_post_link_fsync(descriptor: int) -> None:
        nonlocal failed
        if linked and not failed:
            failed = True
            raise OSError("injected linked publication crash")
        original_fsync(descriptor)

    monkeypatch.setattr(phase_b.os, "link", recording_link)
    monkeypatch.setattr(phase_b.os, "fsync", fail_first_post_link_fsync)
    with pytest.raises(phase_b.PhaseBError, match="publish_failed"):
        with journal.lock(plan):
            journal.append(
                plan,
                approval=_approved(plan),
                event="intent",
                idempotency_key="intent",
                evidence=evidence,
                now_unix=NOW,
            )
    stage = next(journal._staging_root(plan).iterdir())
    final = next(journal._entries_root(plan).iterdir())
    assert stage.stat().st_nlink == final.stat().st_nlink == 2

    monkeypatch.setattr(phase_b.os, "link", original_link)
    monkeypatch.setattr(phase_b.os, "fsync", original_fsync)
    with journal.lock(plan):
        recovered = journal.append(
            plan,
            approval=_approved(plan),
            event="intent",
            idempotency_key="intent",
            evidence=evidence,
            now_unix=NOW,
        )
    assert recovered.value["sequence"] == 0
    assert list(journal._staging_root(plan).iterdir()) == []
    assert final.stat().st_nlink == 1


def test_main_journal_retries_parent_fsync_after_created_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    original_fsync = phase_b.os.fsync
    parent_status = journal.root.parent.stat()
    failed = False

    def fail_parent_once(descriptor: int) -> None:
        nonlocal failed
        status = os.fstat(descriptor)
        if (
            not failed
            and (status.st_dev, status.st_ino)
            == (parent_status.st_dev, parent_status.st_ino)
        ):
            failed = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(phase_b.os, "fsync", fail_parent_once)
    with pytest.raises(phase_b.PhaseBError, match="journal_unavailable"):
        with journal.lock(plan):
            pass
    assert journal.root.is_dir()

    with journal.lock(plan):
        journal.append(
            plan,
            approval=_approved(plan),
            event="intent",
            idempotency_key="intent",
            evidence={"safe_to_start": False},
            now_unix=NOW,
        )
    assert failed is True
    assert [entry.event for entry in journal.load(plan)] == ["intent"]


def _foundation_observation(
    plan: phase_b.PhaseBPlan,
    session_user: str,
) -> dict[str, Any]:
    roles = []
    for name, (can_login, inherits) in foundation._EXPECTED_ROLE_SHAPES.items():
        if name == foundation.SQL_USER:
            can_login = True
        roles.append(
            {
                "name": name,
                "can_login": can_login,
                "inherits": inherits,
                "superuser": False,
                "create_database": False,
                "create_role": False,
                "replication": False,
                "bypass_row_security": False,
                "connection_limit": -1,
                "validity_is_unbounded": True,
                "configuration_is_empty": True,
            }
        )
    memberships = [
        {
            "granted_role": foundation.CANARY_BOOTSTRAP_ROLE,
            "member_role": foundation.CANARY_BOOTSTRAP_LOGIN,
            "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": False,
            "inherit_option": True,
            "set_option": False,
        },
        {
            "granted_role": foundation.WRITER_ROLE,
            "member_role": foundation.SQL_USER,
            "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
            "admin_option": False,
            "inherit_option": True,
            "set_option": False,
        },
    ]
    credential = {**_credential(), "stage_path": None}
    unsigned = {
        "schema": foundation.FOUNDATION_OBSERVATION_SCHEMA,
        "database": foundation.SQL_DATABASE,
        "database_owner": foundation.DATABASE_OWNER_ROLE,
        "postgres_version_num": 180004,
        "session_user": session_user,
        "current_user": session_user,
        "temporary_admin_roles": (
            [] if session_user == foundation.SQL_USER else [session_user]
        ),
        "tls_peer_certificate_sha256": TLS_PEER,
        "roles": roles,
        "memberships": memberships,
        "event_log_shape": "canonical14",
        "event_log_owner": foundation.MIGRATION_OWNER_ROLE,
        "legacy_archive_present": True,
        "legacy_archive_identity": {
            "oid": "700",
            "owner": "legacy_archive_source_owner",
            "relation_kind": "r",
            "persistence": "p",
            "owner_superuser": False,
            "owner_create_database": False,
            "owner_create_role": False,
            "owner_replication": False,
            "owner_bypass_row_security": False,
            "owner_connection_limit": -1,
            "owner_validity_is_unbounded": True,
            "owner_configuration_is_empty": True,
        },
        "canonical_schema_owner": foundation.MIGRATION_OWNER_ROLE,
        "writer_ping_present": True,
        "database_acl": [],
        "public_schema_acl": [],
        "legacy_truth": None,
        "credential": credential,
    }
    return {
        **unsigned,
        "observation_sha256": foundation._sha256_json(unsigned),
    }


def _receipt(unsigned: Mapping[str, Any], field: str = "receipt_sha256") -> dict[str, Any]:
    return {**unsigned, field: phase_b._sha256_json(unsigned)}


class _State:
    def __init__(self, fault: str | None = None) -> None:
        self.fault = fault
        self.fired: set[str] = set()
        self.role = False
        self.bootstrap = False
        self.bootstrap_disabled = False
        self.temp_admin = False
        self.auto_bridge = False
        self.bootstrap_resource: dict[str, Any] | None = None
        self.temp_operation = 0
        self.bootstrap_operation = 0
        self.bootstrap_operation_row: list[Any] | None = None
        self.writer_session = 0
        self.pristine_collections = 0
        self.recovery_collections = 0
        self.services = 0
        self.calls: list[str] = []
        self.secrets: list[bytearray] = []

    def trip(self, point: str) -> None:
        if self.fault == point and point not in self.fired:
            self.fired.add(point)
            raise RuntimeError("injected_" + point)


class _WriterSession:
    username = foundation.SQL_USER

    def __init__(self, state: _State) -> None:
        self.state = state
        self.closed = False
        state.writer_session += 1
        self.identity = f"{state.writer_session:064x}"

    def close(self) -> None:
        self.closed = True


class _AdminSession:
    def __init__(self, state: _State, plan: phase_b.PhaseBPlan) -> None:
        self.state = state
        self.plan = plan
        self.username = plan.temporary_admin_username
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.state.trip("admin_close")

    def execute_phase_b_role_artifact(
        self,
        artifact: foundation.SealedSQLArtifact,
        *,
        bindings: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self.state.calls.append("role_mutation")
        assert bindings["muncho.canonical_writer_phase_b_approved_plan_sha256"] == self.plan.sha256
        if not self.state.role:
            self.state.role = True
            self.state.auto_bridge = self.state.temp_admin
            outcome = "created"
        elif self.state.auto_bridge:
            outcome = "adopted_same_admin_predelete"
        else:
            outcome = "adopted_zero_membership"
        self.state.trip("role")
        bridge = None
        if outcome != "adopted_zero_membership":
            bridge = {
                "granted_role": foundation.CANARY_BOOTSTRAP_ROLE,
                "member_role": self.plan.temporary_admin_username,
                "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": True,
                "inherit_option": False,
                "set_option": False,
            }
        return _pg_role_envelope({
            "schema": phase_b.PHASE_B_ROLE_RECEIPT_SCHEMA,
            "phase": "phase_b_role_and_connect",
            "preterminal": True,
            "database": foundation.SQL_DATABASE,
            "postgres_version_num": 180004,
            "session_user": self.plan.temporary_admin_username,
            "role": foundation.CANARY_BOOTSTRAP_ROLE,
            "role_outcome": outcome,
            "role_contract": {
                "can_login": False,
                "inherits": False,
                "superuser": False,
                "create_database": False,
                "create_role": False,
                "replication": False,
                "bypass_row_security": False,
                "connection_limit": -1,
                "validity_is_unbounded": True,
                "configuration_is_empty": True,
            },
            "connect_contract": {
                "database": foundation.SQL_DATABASE,
                "privilege": "CONNECT",
                "grantor": foundation.DATABASE_OWNER_ROLE,
                "grantable": False,
                "managed_cloudsqladmin_hba_boundary_separate": True,
            },
            "temporary_auto_membership": bridge,
            "temporary_admin_delete_required": True,
            "release_revision": REVISION,
            "artifact_sha256": artifact.sha256,
            "initial_observation_sha256": self.plan.preflight.sha256,
            "approved_plan_sha256": self.plan.sha256,
            "secret_material_recorded": False,
        })


class _TemporaryAdmin:
    def __init__(self, state: _State, plan: phase_b.PhaseBPlan) -> None:
        self.state = state
        self.plan = plan
        self.operation: list[Any] | None = None
        self.context: str | None = None
        self.owner: str | None = None
        self.ambiguous = False
        self.baseline_row = ["baseline-user", "UPDATE_USER", "DONE", "0" * 64, True]

    def begin_mutation_observation(
        self,
        *,
        expected_owner_subject_sha256: str,
        expected_mutation_context_sha256: str,
    ) -> None:
        self.owner = expected_owner_subject_sha256
        self.context = expected_mutation_context_sha256

    def create_or_rotate_recovery(self, username: str) -> bytearray:
        assert username == self.plan.temporary_admin_username
        operation_type = "UPDATE_USER" if self.state.temp_admin else "CREATE_USER"
        self.state.temp_admin = True
        self.state.temp_operation += 1
        self.operation = [
            f"temp-{self.state.temp_operation}",
            operation_type,
            "DONE",
            OWNER,
            True,
        ]
        self.state.calls.append("temp_admin_" + operation_type.lower())
        if self.state.fault == "ambiguous_admin" and "ambiguous_admin" not in self.state.fired:
            self.state.fired.add("ambiguous_admin")
            self.ambiguous = True
            raise RuntimeError("ambiguous_admin")
        self.ambiguous = False
        secret = bytearray(b"S" * 64)
        self.state.secrets.append(secret)
        return secret

    def mutation_reconciliation_required(self) -> bool:
        return self.ambiguous

    def require_current_authority(self, username: str) -> None:
        assert username == self.plan.temporary_admin_username
        assert self.state.temp_admin and self.operation is not None

    def temporary_admin_authority_receipt(self, username: str) -> Mapping[str, Any]:
        assert self.operation is not None
        unsigned = {
            "schema": phase_b.TEMPORARY_ADMIN_AUTHORITY_RECEIPT_SCHEMA,
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "username_sha256": phase_b._sha256_bytes(username.encode("ascii")),
            "host": "",
            "type": "BUILT_IN",
            "user_present": True,
            "owner_subject_sha256": self.owner,
            "mutation_context_sha256": self.context,
            "baseline_operation_names": ["baseline-user"],
            "baseline_user_operations": [self.baseline_row],
            "authority_operation": self.operation,
        }
        return _receipt(unsigned)

    def delete_and_confirm_absent(self, username: str) -> None:
        assert username == self.plan.temporary_admin_username
        self.state.calls.append("temp_admin_delete")
        self.state.temp_admin = False
        self.state.auto_bridge = False
        self.state.trip("delete")

    def reconciliation_receipt(self) -> Mapping[str, Any]:
        assert self.operation is not None
        delete_row = [
            f"delete-{self.state.temp_operation}",
            "DELETE_USER",
            "DONE",
            OWNER,
            True,
        ]
        unsigned = {
            "schema": phase_b.TEMPORARY_ADMIN_ABSENCE_RECEIPT_SCHEMA,
            "temporary_admin_absent": True,
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "username_sha256": phase_b._sha256_bytes(
                self.plan.temporary_admin_username.encode("ascii")
            ),
            "owner_subject_sha256": OWNER,
            "mutation_context_sha256": self.plan.sha256,
            "user_absent": True,
            "baseline_operation_names": ["baseline-user"],
            "baseline_user_operations": [self.baseline_row],
            "known_operation_names": sorted([self.operation[0], delete_row[0]]),
            "response_known_authority_operation_names": [self.operation[0]],
            "response_known_delete_operation_names": [delete_row[0]],
            "post_baseline_authority_operations": [self.operation],
            "response_known_candidate_observed": True,
            "post_baseline_authority_operation_count": 1,
            "terminal_user_operations": [self.baseline_row, self.operation, delete_row],
            "mutation_ambiguity_observed": False,
            "quiet_window_seconds": 30.0,
        }
        return _receipt(unsigned, "evidence_sha256")


class _BootstrapLogin:
    def __init__(self, state: _State, plan: phase_b.PhaseBPlan) -> None:
        self.state = state
        self.plan = plan
        self.operation_name: str | None = None
        self.operation_type: str | None = None
        self.ambiguous = False

    def describe(self) -> Mapping[str, Any] | None:
        return copy.deepcopy(self.state.bootstrap_resource)

    def create_or_rotate_recovery(self) -> bytearray:
        self.operation_type = "UPDATE_USER" if self.state.bootstrap else "CREATE_USER"
        self.state.bootstrap = True
        self.state.bootstrap_disabled = False
        self.state.bootstrap_operation += 1
        self.operation_name = f"bootstrap-{self.state.bootstrap_operation}"
        self.state.bootstrap_operation_row = [
            self.operation_name,
            self.operation_type,
            "DONE",
            OWNER,
            True,
        ]
        self.state.bootstrap_resource = {
            "databaseRoles": [foundation.CANARY_BOOTSTRAP_ROLE],
            "etag": "etag-stable",
            "host": "",
            "instance": foundation.SQL_INSTANCE,
            "name": foundation.CANARY_BOOTSTRAP_LOGIN,
            "project": foundation.PROJECT,
            "type": "BUILT_IN",
        }
        self.state.calls.append("bootstrap_" + self.operation_type.lower())
        if self.state.fault == "ambiguous_bootstrap" and "ambiguous_bootstrap" not in self.state.fired:
            self.state.fired.add("ambiguous_bootstrap")
            self.ambiguous = True
            raise RuntimeError("ambiguous_bootstrap")
        self.ambiguous = False
        secret = bytearray(b"S" * 64)
        self.state.secrets.append(secret)
        return secret

    def mutation_reconciliation_required(self) -> bool:
        return self.ambiguous

    def require_current_authority(self) -> None:
        assert self.operation_name and self.state.bootstrap_resource

    def authority_receipt(self) -> Mapping[str, Any]:
        assert self.operation_name and self.operation_type and self.state.bootstrap_resource
        unsigned = {
            "schema": phase_b.BOOTSTRAP_AUTHORITY_RECEIPT_SCHEMA,
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "name": foundation.CANARY_BOOTSTRAP_LOGIN,
            "host": "",
            "type": "BUILT_IN",
            "database_roles": [foundation.CANARY_BOOTSTRAP_ROLE],
            "etag": self.state.bootstrap_resource["etag"],
            "resource_projection_sha256": phase_b._sha256_json(
                self.state.bootstrap_resource
            ),
            "operation_name": self.operation_name,
            "operation_type": self.operation_type,
            "owner_subject_sha256": OWNER,
            "mutation_context_sha256": self.plan.sha256,
        }
        return _receipt(unsigned)


class _SelfDisable:
    def __init__(self, state: _State) -> None:
        self.state = state

    def disable_and_prove_denied(
        self,
        *,
        plan: phase_b.PhaseBPlan,
        provisional_password: bytearray,
        authority_receipt: Mapping[str, Any],
        hba_rejection_receipt: Mapping[str, Any],
        statement: str,
    ) -> Mapping[str, Any]:
        assert bytes(provisional_password) == b"S" * 64
        assert statement == phase_b.SELF_DISABLE_SQL
        self.state.calls.append("bootstrap_self_disable")
        self.state.bootstrap_disabled = True
        self.state.trip("self_disable")
        unsigned = {
            "schema": phase_b.PHASE_B_SELF_DISABLE_SCHEMA,
            "plan_sha256": plan.sha256,
            "bootstrap_authority_receipt_sha256": authority_receipt["receipt_sha256"],
            "hba_rejection_receipt_sha256": hba_rejection_receipt["receipt_sha256"],
            "user": foundation.CANARY_BOOTSTRAP_LOGIN,
            "database": foundation.SQL_DATABASE,
            "tls_peer_certificate_sha256": TLS_PEER,
            "authenticated_as_self": True,
            "statement_sha256": phase_b._sha256_bytes(statement.encode("ascii")),
            "command_tag": "ALTER ROLE",
            "password_disabled": True,
            "login_remains_true": True,
            "fresh_denial_connection": True,
            "denial_sqlstate": "28P01",
            "password_or_digest_recorded": False,
        }
        return _receipt(unsigned)


def _dependencies(state: _State, plan: phase_b.PhaseBPlan) -> phase_b.PhaseBDependencies:
    def writer_factory() -> _WriterSession:
        return _WriterSession(state)

    def pristine(session: _WriterSession) -> Mapping[str, Any]:
        state.pristine_collections += 1
        value = plan.preflight.to_mapping()
        value["database"]["session_identity_sha256"] = session.identity
        value["observed_at_unix"] += state.writer_session
        value["services"] = _services(NOW + state.writer_session)
        value["observation_sha256"] = phase_b._sha256_json(
            {key: item for key, item in value.items() if key != "observation_sha256"}
        )
        return value

    def recovery(
        session: _WriterSession,
        _plan_value: phase_b.PhaseBPlan,
        _events: frozenset[str],
    ) -> Mapping[str, Any]:
        state.recovery_collections += 1
        raw = _database_preflight(
            bootstrap_role=state.role,
            bootstrap_login=state.bootstrap,
            temporary_admin=(plan.temporary_admin_username if state.temp_admin else None),
            auto_bridge=state.auto_bridge,
        )
        cloud = {
            "project": foundation.PROJECT,
            "instance": foundation.SQL_INSTANCE,
            "bootstrap_resource": copy.deepcopy(state.bootstrap_resource),
            "temporary_admin_users": (
                [plan.temporary_admin_username] if state.temp_admin else []
            ),
            "user_inventory_sha256": "7" * 64,
            "user_operations_quiescent": True,
            "operation_ledger_sha256": "8" * 64,
        }
        unsigned = {
            "schema": phase_b.PHASE_B_RECOVERY_SCHEMA,
            "plan_sha256": plan.sha256,
            "database": _database(session.identity),
            "database_preflight": raw,
            "credential": _credential(),
            "services": _services(NOW + state.writer_session),
            "cloud_sql": cloud,
            "observed_at_unix": NOW + state.writer_session,
        }
        return _hashed(unsigned, "observation_sha256")

    def admin_factory(
        _plan_value: phase_b.PhaseBPlan, username: str, secret: bytearray
    ) -> _AdminSession:
        assert username == plan.temporary_admin_username
        assert bytes(secret) == b"S" * 64
        return _AdminSession(state, plan)

    def hba(
        _plan_value: phase_b.PhaseBPlan,
        secret: bytearray,
        authority: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        assert bytes(secret) == b"S" * 64
        state.calls.append("hba_read")
        state.trip("hba")
        unsigned = {
            "schema": phase_b.PHASE_B_HBA_RECEIPT_SCHEMA,
            "plan_sha256": plan.sha256,
            "bootstrap_authority_receipt_sha256": authority["receipt_sha256"],
            "host": foundation.SQL_HOST,
            "port": foundation.SQL_PORT,
            "tls_server_name": foundation.SQL_TLS_SERVER_NAME,
            "tls_peer_certificate_sha256": TLS_PEER,
            "user": foundation.CANARY_BOOTSTRAP_LOGIN,
            "database": "cloudsqladmin",
            "rejected": True,
            "sqlstate": "28000",
            "observed_at_unix": NOW,
            "expires_at_unix": NOW + 300,
            "secret_material_recorded": False,
        }
        return _receipt(unsigned)

    def predelete(
        session: _AdminSession,
        _plan_value: phase_b.PhaseBPlan,
        role: Mapping[str, Any],
        authority: Mapping[str, Any],
        self_disable: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        bridge = None
        if role["role_outcome"] != "adopted_zero_membership":
            bridge = {
                "granted_role": foundation.CANARY_BOOTSTRAP_ROLE,
                "member_role": plan.temporary_admin_username,
                "grantor": foundation.PERSISTENT_MEMBERSHIP_GRANTOR,
                "admin_option": True,
                "inherit_option": False,
                "set_option": False,
            }
        unsigned = {
            "schema": phase_b.PHASE_B_PREDELETE_SCHEMA,
            "plan_sha256": plan.sha256,
            "foundation_observation": _foundation_observation(
                plan, session.username
            ),
            "database_preflight": _database_preflight(
                bootstrap_role=True,
                bootstrap_login=True,
                temporary_admin=plan.temporary_admin_username,
                auto_bridge=bridge is not None,
            ),
            "bootstrap_connect_acl": {
                "database": foundation.SQL_DATABASE,
                "grantee": foundation.CANARY_BOOTSTRAP_ROLE,
                "grantor": foundation.DATABASE_OWNER_ROLE,
                "privilege": "CONNECT",
                "grantable": False,
            },
            "temporary_auto_membership": bridge,
            "other_temporary_admin_references": [],
            "role_receipt_sha256": role["receipt_sha256"],
            "bootstrap_authority_receipt_sha256": authority["receipt_sha256"],
            "self_disable_receipt_sha256": self_disable["receipt_sha256"],
            "temporary_admin_delete_required": True,
            "preterminal": True,
            "safe_to_start": False,
            "secret_material_recorded": False,
        }
        return _receipt(unsigned)

    def terminal(
        session: _WriterSession,
        _plan_value: phase_b.PhaseBPlan,
        resource: Mapping[str, Any],
        absence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        state.trip("terminal_observation")
        raw = _database_preflight(bootstrap_role=True, bootstrap_login=True)
        projection = phase_b._database_preflight_projection(raw)
        assert state.bootstrap_operation_row is not None
        inventory = _terminal_user_inventory(resource)
        operations = sorted(
            [
                *copy.deepcopy(absence["terminal_user_operations"]),
                copy.deepcopy(state.bootstrap_operation_row),
            ],
            key=lambda row: row[0],
        )
        unsigned = {
            "schema": phase_b.PHASE_B_TERMINAL_OBSERVATION_SCHEMA,
            "plan_sha256": plan.sha256,
            "foundation_observation": _foundation_observation(
                plan, foundation.SQL_USER
            ),
            "database_preflight": raw,
            "session_identity_sha256": session.identity,
            "writer_ping_identity_sha256": projection["writer_ping_identity_sha256"],
            "event_log_identity_sha256": projection["event_log_identity_sha256"],
            "legacy_archive_identity_sha256": projection["legacy_archive_identity_sha256"],
            "cross_database_acl_sha256": projection["cross_database_acl_sha256"],
            "bootstrap_connect_acl": {
                "database": foundation.SQL_DATABASE,
                "grantee": foundation.CANARY_BOOTSTRAP_ROLE,
                "grantor": foundation.DATABASE_OWNER_ROLE,
                "privilege": "CONNECT",
                "grantable": False,
            },
            "temporary_admin_references": [],
            "cloud_sql": {
                "project": foundation.PROJECT,
                "instance": foundation.SQL_INSTANCE,
                "bootstrap_resource": copy.deepcopy(resource),
                "temporary_admin_absent": True,
                "temporary_admin_username_sha256": phase_b._sha256_bytes(
                    plan.temporary_admin_username.encode("ascii")
                ),
                "user_inventory": inventory,
                "user_inventory_sha256": phase_b._sha256_json(inventory),
                "user_operations_quiescent": True,
                "relevant_user_operations": operations,
                "operation_ledger_sha256": phase_b._sha256_json(operations),
                "observed_at_unix": NOW,
            },
            "services": _services(NOW),
            "observed_at_unix": NOW,
        }
        return _hashed(unsigned, "observation_sha256")

    def services(_plan_value: phase_b.PhaseBPlan, transition: str) -> Mapping[str, Any]:
        state.services += 1
        state.calls.append("services:" + transition)
        return _services(NOW)

    def temporary_admin_factory(
        _plan_value: phase_b.PhaseBPlan,
    ) -> _TemporaryAdmin:
        state.trip("temporary_admin_factory")
        return _TemporaryAdmin(state, plan)

    return phase_b.PhaseBDependencies(
        writer_session_factory=writer_factory,
        pristine_preflight_collector=pristine,
        recovery_collector=recovery,
        temporary_admin_factory=temporary_admin_factory,
        bootstrap_login_factory=lambda _plan_value: _BootstrapLogin(state, plan),
        admin_session_factory=admin_factory,
        bootstrap_self_disable=_SelfDisable(state),
        hba_collector=hba,
        predelete_collector=predelete,
        terminal_collector=terminal,
        services_collector=services,
    )


def _artifact(plan: phase_b.PhaseBPlan, tmp_path: Path) -> foundation.SealedSQLArtifact:
    return foundation.SealedSQLArtifact(
        phase_b.ROLE_ARTIFACT_NAME,
        tmp_path / Path(phase_b.ROLE_ARTIFACT_PATH).name,
        plan.value["role_artifact_sha256"],
        ROLE_PAYLOAD,
    )


def _run(
    state: _State,
    tmp_path: Path,
    *,
    journal: phase_b.AppendOnlyPhaseBJournal | None = None,
) -> tuple[Mapping[str, Any], phase_b.AppendOnlyPhaseBJournal, phase_b.PhaseBPlan]:
    plan = _plan()
    active_journal = journal or phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    receipt = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=active_journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW,
    )
    return receipt, active_journal, plan


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_projection_sha256", "0" * 64),
        ("operation_name", "bootstrap:operation"),
        ("operation_name", "x" * 257),
        ("etag", "etag with space"),
    ],
)
def test_bootstrap_authority_rejects_forged_resource_and_cloud_identity(
    field: str,
    value: str,
) -> None:
    state = _State()
    plan = _plan()
    login = _BootstrapLogin(state, plan)
    secret = login.create_or_rotate_recovery()
    phase_b._zeroize(secret)
    forged = copy.deepcopy(dict(login.authority_receipt()))
    forged[field] = value
    if field == "etag":
        resource = copy.deepcopy(state.bootstrap_resource)
        assert resource is not None
        resource["etag"] = value
        forged["resource_projection_sha256"] = phase_b._sha256_json(resource)
    unsigned = {
        name: item for name, item in forged.items() if name != "receipt_sha256"
    }
    forged["receipt_sha256"] = phase_b._sha256_json(unsigned)

    with pytest.raises(phase_b.PhaseBError, match="bootstrap_authority_invalid"):
        phase_b._validate_bootstrap_authority(forged, plan=plan)


@pytest.mark.parametrize(
    "mutation",
    [
        "approval_digest",
        "role_digest",
        "safe_flag",
        "terminal_time",
        "nested_absence",
        "extra_field",
    ],
)
def test_terminal_replay_rejects_self_hashed_semantic_forgery(
    tmp_path: Path,
    mutation: str,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    terminal_path = sorted(journal._entries_root(plan).iterdir())[-1]
    entry = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal = entry["evidence"]["terminal_receipt"]
    if mutation == "approval_digest":
        terminal["approval_sha256"] = "0" * 64
    elif mutation == "role_digest":
        terminal["role_receipt_sha256"] = "0" * 64
    elif mutation == "safe_flag":
        terminal["safe_to_start"] = False
    elif mutation == "terminal_time":
        terminal["terminal_at_unix"] = 0
    elif mutation == "nested_absence":
        terminal["temporary_admin_absence_receipt"]["quiet_window_seconds"] = 31.0
        absence_unsigned = {
            name: item
            for name, item in terminal["temporary_admin_absence_receipt"].items()
            if name != "evidence_sha256"
        }
        terminal["temporary_admin_absence_receipt"]["evidence_sha256"] = (
            phase_b._sha256_json(absence_unsigned)
        )
        terminal["temporary_admin_absence_receipt_sha256"] = terminal[
            "temporary_admin_absence_receipt"
        ]["evidence_sha256"]
    else:
        terminal["unexpected_terminal_claim"] = True
    terminal_unsigned = {
        name: item for name, item in terminal.items() if name != "receipt_sha256"
    }
    terminal["receipt_sha256"] = phase_b._sha256_json(terminal_unsigned)
    entry_unsigned = {
        name: item for name, item in entry.items() if name != "entry_sha256"
    }
    entry["entry_sha256"] = phase_b._sha256_json(entry_unsigned)
    terminal_path.write_bytes(phase_b._canonical_bytes(entry))

    with pytest.raises(phase_b.PhaseBError, match="terminal_receipt_invalid"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )


def test_executor_reaches_terminal_only_after_two_admin_authorities(
    tmp_path: Path,
) -> None:
    state = _State()
    receipt, journal, plan = _run(state, tmp_path)

    assert receipt["safe_to_start"] is True
    assert receipt["temporary_admin_absent"] is True
    assert state.bootstrap and state.bootstrap_disabled and not state.temp_admin
    assert all(value == bytearray(b"\x00" * 64) for value in state.secrets)
    entries = journal.load(plan)
    events = [entry.event for entry in entries]
    assert events.count("temporary_admin_authority") == 1
    assert events.count("temporary_admin_predelete_authority") == 1
    assert events[-1] == "terminal"
    assert state.calls == [
        "services:temporary_admin_initial_authority",
        "temp_admin_create_user",
        "services:phase_b_role_artifact",
        "role_mutation",
        "services:bootstrap_login_authority",
        "bootstrap_create_user",
        "services:bootstrap_login_rotation_authority",
        "bootstrap_update_user",
        "hba_read",
        "services:bootstrap_password_self_disable",
        "bootstrap_self_disable",
        "services:temporary_admin_predelete_reacquire",
        "temp_admin_update_user",
        "services:temporary_admin_delete",
        "temp_admin_delete",
        "services:terminal_observation",
    ]
    serialized = b"".join(
        path.read_bytes() for path in sorted(journal._entries_root(plan).iterdir())
    )
    assert b'"password":"' not in serialized
    assert b"SSSSSS" not in serialized


@pytest.mark.parametrize(
    "fault",
    ["role", "hba", "self_disable", "delete", "terminal_observation"],
)
def test_executor_crash_replay_is_fail_closed_and_eventually_terminal(
    tmp_path: Path,
    fault: str,
) -> None:
    state = _State(fault)
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with pytest.raises(RuntimeError, match="injected"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )
    assert "terminal" not in journal.events(plan)

    receipt = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW,
    )
    assert receipt["safe_to_start"] is True
    assert journal.load(plan)[-1].event == "terminal"


def test_expired_incomplete_generation_resumes_only_with_next_signed_head(
    tmp_path: Path,
) -> None:
    state = _State("role")
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    initial = _signed_approval(
        plan,
        sequence=0,
        previous_approval_sha256=None,
        issued_at_unix=NOW - 1,
        expires_at_unix=NOW + 1,
        source_authentication_sha256="1" * 64,
    )
    with pytest.raises(RuntimeError, match="injected_role"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=[initial],
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )
    before = [entry.sha256 for entry in journal.load(plan)]
    calls_before = list(state.calls)
    with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=[initial],
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW + 2,
        )
    assert [entry.sha256 for entry in journal.load(plan)] == before
    assert state.calls == calls_before

    resumed = _signed_approval(
        plan,
        sequence=1,
        previous_approval_sha256=initial["approval_sha256"],
        issued_at_unix=NOW + 2,
        expires_at_unix=NOW + 250,
        source_authentication_sha256="2" * 64,
    )
    receipt = phase_b.execute_approved_phase_b(
        plan,
        approval=[initial, resumed],
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW + 2,
    )
    entries = journal.load(plan)
    assert receipt["approval_sha256"] == resumed["approval_sha256"]
    assert all(
        entry.value["approval_sha256"] == initial["approval_sha256"]
        for entry in entries[: len(before)]
    )
    assert all(
        entry.value["approval_sha256"] == resumed["approval_sha256"]
        for entry in entries[len(before) :]
    )
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=[initial, resumed],
        journal=journal,
    )
    assert durable.generation["approval_sequence"] == 1
    assert durable.generation["approval_chain_sha256"] == phase_b._sha256_json(
        [initial, resumed]
    )


@pytest.mark.parametrize("mutation", ["old_head", "missing_link", "fork", "source"])
def test_resume_approval_chain_rejects_rollback_fork_and_divergence(
    mutation: str,
) -> None:
    plan = _plan()
    initial = _approval(plan)
    resumed = _signed_approval(
        plan,
        sequence=1,
        previous_approval_sha256=initial["approval_sha256"],
        issued_at_unix=NOW + 301,
        expires_at_unix=NOW + 500,
        source_authentication_sha256="2" * 64,
    )
    candidate = copy.deepcopy(resumed)
    if mutation == "old_head":
        values = [resumed]
    else:
        if mutation == "missing_link":
            candidate["sequence"] = 2
        elif mutation == "fork":
            candidate["previous_approval_sha256"] = "f" * 64
        else:
            candidate["approval_source_sha256"] = "e" * 64
        unsigned = {
            name: item
            for name, item in candidate.items()
            if name not in {"signature_sshsig", "approval_sha256"}
        }
        candidate["signature_sshsig"] = _sshsig(
            phase_b._canonical_bytes(unsigned),
            phase_b.PHASE_B_APPROVAL_SSHSIG_NAMESPACE,
        )
        candidate["approval_sha256"] = phase_b._sha256_json(
            {
                name: item
                for name, item in candidate.items()
                if name != "approval_sha256"
            }
        )
        values = [initial, candidate]
    with pytest.raises(phase_b.PhaseBError, match="approval_chain_invalid"):
        phase_b.validate_phase_b_approval_chain(values, plan=plan)


def _stage_runtime_approval_authority(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> tuple[phase_b.PhaseBPlan, dict[str, Any], dict[str, Any]]:
    plan = _plan()
    initial_source, initial = _signed_approval_pair(
        plan,
        sequence=0,
        previous_approval_sha256=None,
        issued_at_unix=NOW - 300,
        expires_at_unix=NOW - 1,
        nonce_sha256="7" * 64,
    )
    authority = root / "authority"
    resumes = authority / "resume-approvals"
    authority.mkdir(mode=0o700)
    resumes.mkdir(mode=0o700)
    lock = resumes / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)
    values = {
        authority / "plan.json": plan.to_mapping(),
        authority / "owner-approval.json": initial,
        authority / "owner-approval-source.json": initial_source,
    }
    for path, value in values.items():
        path.write_bytes(phase_b._canonical_bytes(value) + b"\n")
        path.chmod(0o400)
    monkeypatch.setattr(phase_b_runtime, "PHASE_B_AUTHORITY_ROOT", authority)
    monkeypatch.setattr(phase_b_runtime, "PHASE_B_PLAN_PATH", authority / "plan.json")
    monkeypatch.setattr(
        phase_b_runtime,
        "PHASE_B_APPROVAL_PATH",
        authority / "owner-approval.json",
    )
    monkeypatch.setattr(
        phase_b_runtime,
        "PHASE_B_APPROVAL_SOURCE_PATH",
        authority / "owner-approval-source.json",
    )
    monkeypatch.setattr(phase_b_runtime, "PHASE_B_RESUME_APPROVAL_ROOT", resumes)
    monkeypatch.setattr(
        phase_b_runtime,
        "PHASE_B_RESUME_APPROVAL_LOCK_PATH",
        resumes / ".lock",
    )
    monkeypatch.setattr(
        phase_b_runtime,
        "PHASE_B_JOURNAL_ROOT",
        root / "journal",
    )
    monkeypatch.setattr(phase_b_runtime, "_ROOT_UID", os.geteuid())
    monkeypatch.setattr(phase_b_runtime, "_ROOT_GID", os.getegid())
    monkeypatch.setattr(phase_b_runtime, "_require_root_linux", lambda: None)
    monkeypatch.setattr(phase_b_runtime, "_harden_phase_b_process", lambda: None)
    monkeypatch.setattr(phase_b_runtime.time, "time", lambda: NOW)
    return plan, initial_source, initial


@pytest.mark.parametrize("crash_stage", ["none", "after_source", "after_approval"])
def test_runtime_resume_install_recovers_every_o_excl_crash_boundary(
    monkeypatch: pytest.MonkeyPatch,
    crash_stage: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="phase-b-resume-", dir=repository) as temp:
        root = Path(temp)
        plan, _initial_source, initial = _stage_runtime_approval_authority(
            monkeypatch,
            root,
        )
        source, candidate = _signed_approval_pair(
            plan,
            sequence=1,
            previous_approval_sha256=initial["approval_sha256"],
            issued_at_unix=NOW - 1,
            expires_at_unix=NOW + 120,
            nonce_sha256="8" * 64,
        )
        resumes = root / "authority" / "resume-approvals"
        if crash_stage in {"after_source", "after_approval"}:
            source_path = resumes / "00000001.source.json"
            source_path.write_bytes(phase_b._canonical_bytes(source) + b"\n")
            source_path.chmod(0o400)
        if crash_stage == "after_approval":
            approval_path = resumes / "00000001.approval.json"
            approval_path.write_bytes(
                phase_b._canonical_bytes(candidate) + b"\n"
            )
            approval_path.chmod(0o400)

        receipt = phase_b_runtime.install_fixed_phase_b_resume_approval(
            candidate,
            source,
            expected_previous_approval_sha256=initial["approval_sha256"],
        )
        assert receipt["approval_sha256"] == candidate["approval_sha256"]
        assert receipt["already_installed"] is (crash_stage == "after_approval")
        loaded = phase_b_runtime.load_fixed_phase_b_approval_chain(
            require_fresh_head=True
        )
        assert loaded == (initial, candidate)
        retry = phase_b_runtime.install_fixed_phase_b_resume_approval(
            candidate,
            source,
            expected_previous_approval_sha256=initial["approval_sha256"],
        )
        assert retry["already_installed"] is True


def test_runtime_resume_loader_rejects_bounded_overlong_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="phase-b-resume-max-", dir=repository) as temp:
        root = Path(temp)
        plan, _source, _initial = _stage_runtime_approval_authority(
            monkeypatch,
            root,
        )
        resumes = root / "authority" / "resume-approvals"
        for sequence in range(1, phase_b_runtime.PHASE_B_MAX_RESUME_APPROVALS + 2):
            for kind in ("source", "approval"):
                path = resumes / f"{sequence:08d}.{kind}.json"
                path.write_bytes(b"{}\n")
                path.chmod(0o400)
        with pytest.raises(phase_b_runtime.PhaseBRuntimeError, match="too_long"):
            phase_b_runtime._load_fixed_approval_chain_for_plan(plan)


def test_runtime_expired_trailing_source_is_completed_only_as_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(
        prefix="phase-b-resume-expired-source-", dir=repository
    ) as temp:
        root = Path(temp)
        plan, _initial_source, initial = _stage_runtime_approval_authority(
            monkeypatch,
            root,
        )
        stale_source, stale = _signed_approval_pair(
            plan,
            sequence=1,
            previous_approval_sha256=initial["approval_sha256"],
            issued_at_unix=NOW - 100,
            expires_at_unix=NOW - 1,
            nonce_sha256="8" * 64,
        )
        resumes = root / "authority" / "resume-approvals"
        source_path = resumes / "00000001.source.json"
        source_path.write_bytes(phase_b._canonical_bytes(stale_source) + b"\n")
        source_path.chmod(0o400)

        repaired = phase_b_runtime.install_fixed_phase_b_resume_approval(
            stale,
            stale_source,
            expected_previous_approval_sha256=initial["approval_sha256"],
        )
        assert repaired["completed_trailing_source_only_residue"] is True
        assert repaired["approval_fresh"] is False
        assert repaired["mutation_authorized"] is False
        assert repaired["requires_fresh_successor"] is True
        assert phase_b_runtime.load_fixed_phase_b_approval_chain() == (
            initial,
            stale,
        )
        with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
            phase_b_runtime.load_fixed_phase_b_approval_chain(
                require_fresh_head=True
            )

        fresh_source, fresh = _signed_approval_pair(
            plan,
            sequence=2,
            previous_approval_sha256=stale["approval_sha256"],
            issued_at_unix=NOW - 1,
            expires_at_unix=NOW + 120,
            nonce_sha256="9" * 64,
        )
        installed = phase_b_runtime.install_fixed_phase_b_resume_approval(
            fresh,
            fresh_source,
            expected_previous_approval_sha256=stale["approval_sha256"],
        )
        assert installed["completed_trailing_source_only_residue"] is False
        assert installed["approval_fresh"] is True
        assert installed["mutation_authorized"] is True
        assert installed["requires_fresh_successor"] is False
        assert phase_b_runtime.load_fixed_phase_b_approval_chain(
            require_fresh_head=True
        ) == (initial, stale, fresh)


def test_executor_retries_ambiguous_api_and_terminal_replay_is_idempotent(
    tmp_path: Path,
) -> None:
    state = _State("ambiguous_bootstrap")
    receipt, journal, plan = _run(state, tmp_path)
    before = [entry.sha256 for entry in journal.load(plan)]

    replay = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW,
    )

    assert replay == receipt
    assert [entry.sha256 for entry in journal.load(plan)] == before
    assert state.calls.count("bootstrap_update_user") == 1


def test_database_receipt_exact_text_survives_plan_round_trip_and_rejects_forgery() -> None:
    plan = _plan()
    replayed = phase_b.PhaseBPlan.from_mapping(
        json.loads(json.dumps(plan.to_mapping(), sort_keys=True))
    )
    assert replayed.to_mapping() == plan.to_mapping()
    assert replayed.preflight.value["foundation"][
        "unsigned_receipt_jsonb_text"
    ] == plan.preflight.value["foundation"]["unsigned_receipt_jsonb_text"]

    value = _preflight().to_mapping()
    value["foundation"]["event_log"]["identity"]["owner"] = "forged_owner"
    value["observation_sha256"] = phase_b._sha256_json(
        {key: item for key, item in value.items() if key != "observation_sha256"}
    )
    with pytest.raises(phase_b.PhaseBError, match="database_preflight"):
        phase_b.PhaseBPreflight.from_mapping(value)

    value = _preflight().to_mapping()
    unsigned_text = value["foundation"]["unsigned_receipt_jsonb_text"]
    duplicate_text = (
        unsigned_text[:-1]
        + ', "schema": "muncho-canonical-writer-foundation-phase-b-db-preflight.v1"}'
    )
    value["foundation"]["unsigned_receipt_jsonb_text"] = duplicate_text
    value["foundation"]["receipt_sha256"] = phase_b._sha256_bytes(
        duplicate_text.encode("utf-8")
    )
    value["observation_sha256"] = phase_b._sha256_json(
        {key: item for key, item in value.items() if key != "observation_sha256"}
    )
    with pytest.raises(phase_b.PhaseBError, match="database_preflight"):
        phase_b.PhaseBPreflight.from_mapping(value)


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("event_primary_key", "event_log"),
        ("canonical_namespace_owner", "namespace"),
        ("writer_namespace", "writer_ping"),
        ("legacy_trigger", "legacy_archive"),
    ],
)
def test_preflight_rejects_expanded_catalog_drift(drift: str, error: str) -> None:
    value = _preflight().to_mapping()
    foundation_value = value["foundation"]
    if drift == "event_primary_key":
        foundation_value["event_log"]["identity"]["constraints"] = []
    elif drift == "canonical_namespace_owner":
        canonical = next(
            row
            for row in foundation_value["namespaces"]
            if row["name"] == "canonical_brain"
        )
        canonical["owner"] = "legacy_archive_source_owner"
    elif drift == "writer_namespace":
        foundation_value["writer_ping"]["routines"][0]["namespace_oid"] = "300"
    else:
        foundation_value["legacy_archive"]["identity"]["user_triggers"] = [
            {"unexpected": True}
        ]
    value["foundation"] = _pg_hashed(foundation_value)
    value["observation_sha256"] = phase_b._sha256_json(
        {key: item for key, item in value.items() if key != "observation_sha256"}
    )
    with pytest.raises(phase_b.PhaseBError, match=error):
        phase_b.PhaseBPreflight.from_mapping(value)


def test_replay_after_stopped_boundary_uses_recovery_not_pristine(
    tmp_path: Path,
) -> None:
    state = _State("temporary_admin_factory")
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with pytest.raises(RuntimeError, match="temporary_admin_factory"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )
    assert state.pristine_collections == 1
    assert state.recovery_collections == 0
    assert "services_stopped" in journal.events(plan)

    receipt = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW,
    )
    assert receipt["safe_to_start"] is True
    assert state.pristine_collections == 1
    assert state.recovery_collections == 1


@pytest.mark.parametrize(
    ("fault", "counter"),
    [("ambiguous_admin", "temp_operation"), ("ambiguous_bootstrap", "bootstrap_operation")],
)
def test_approval_is_revalidated_before_every_ambiguous_mutation_retry(
    tmp_path: Path,
    fault: str,
    counter: str,
) -> None:
    state = _State(fault)
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")

    def clock() -> int:
        return NOW + 301 if fault in state.fired else NOW

    with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=clock,
        )
    assert getattr(state, counter) == 1
    assert "terminal" not in journal.events(plan)
    assert all(value == bytearray(b"\x00" * 64) for value in state.secrets)


class _RoleAndCloseFailureState(_State):
    def trip(self, point: str) -> None:
        if point in {"role", "admin_close"} and point not in self.fired:
            self.fired.add(point)
            raise RuntimeError("injected_" + point)


def test_admin_close_failure_never_masks_original_error_or_claims_closed(
    tmp_path: Path,
) -> None:
    state = _RoleAndCloseFailureState()
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with pytest.raises(RuntimeError, match="injected_role"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )
    assert state.fired == {"role", "admin_close"}
    assert "temporary_admin_closed" not in journal.events(plan)
    assert "terminal" not in journal.events(plan)


def test_admin_close_failure_after_predelete_is_recoverable_without_false_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State("admin_close")
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    with pytest.raises(phase_b.PhaseBError, match="session_close_failed"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )
    assert "predelete_verified" in journal.events(plan)
    assert "temporary_admin_closed" not in journal.events(plan)

    with pytest.raises(phase_b.PhaseBError, match="close_unproven"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW,
        )

    monkeypatch.setattr(phase_b, "_PROCESS_INSTANCE_SHA256", "9" * 64)
    receipt = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW,
    )
    assert receipt["safe_to_start"] is True
    assert "temporary_admin_closed" in journal.events(plan)


def test_forged_deletion_ledger_blocks_terminal(tmp_path: Path) -> None:
    state = _State()
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    dependencies = _dependencies(state, plan)

    class ForgingTemporaryAdmin(_TemporaryAdmin):
        def reconciliation_receipt(self) -> Mapping[str, Any]:
            receipt = super().reconciliation_receipt()
            unsigned = {
                key: copy.deepcopy(item)
                for key, item in receipt.items()
                if key != "evidence_sha256"
            }
            original_name = unsigned["post_baseline_authority_operations"][0][0]
            forged_name = "forged-authority-operation"
            unsigned["post_baseline_authority_operations"][0][0] = forged_name
            unsigned["response_known_authority_operation_names"] = [forged_name]
            unsigned["known_operation_names"] = sorted(
                forged_name if name == original_name else name
                for name in unsigned["known_operation_names"]
            )
            for operation in unsigned["terminal_user_operations"]:
                if operation[0] == original_name:
                    operation[0] = forged_name
            return _receipt(unsigned, "evidence_sha256")

    dependencies = replace(
        dependencies,
        temporary_admin_factory=lambda _plan_value: ForgingTemporaryAdmin(
            state, plan
        ),
    )
    with pytest.raises(phase_b.PhaseBError, match="admin_absence"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=dependencies,
            _clock=lambda: NOW,
        )
    assert "temporary_admin_absent" not in journal.events(plan)
    assert "terminal" not in journal.events(plan)


def _trusted_readiness_observation(
    durable: phase_b.PhaseBDurableFoundation,
    state: _State,
    *,
    release_revision: str,
    observed_at_unix: int,
) -> tuple[phase_b._PhaseBReadinessObservation, _WriterSession]:
    historical = durable.terminal_observation
    historical_cloud = historical["cloud_sql"]
    cloud = {
        "project": foundation.PROJECT,
        "instance": foundation.SQL_INSTANCE,
        "instance_projection": copy.deepcopy(
            phase_b._READINESS_INSTANCE_PROJECTION
        ),
        "instance_projection_sha256": (
            phase_b._READINESS_INSTANCE_PROJECTION_SHA256
        ),
        "user_state_authority": "postgres_pg_roles",
        "bootstrap_role_present": True,
        "bootstrap_login_present": True,
        "temporary_admin_absent": True,
        "temporary_admin_username_sha256": phase_b._sha256_bytes(
            durable.plan.temporary_admin_username.encode("ascii")
        ),
        "temporary_admin_users": [],
        "user_operations_quiescent": True,
        "relevant_user_operations": copy.deepcopy(
            historical_cloud["relevant_user_operations"]
        ),
        "operation_ledger_sha256": historical_cloud[
            "operation_ledger_sha256"
        ],
        "observed_at_unix": observed_at_unix,
    }
    services = _services(observed_at_unix, release_revision)
    writer = _WriterSession(state)
    try:
        terminal = copy.deepcopy(dict(historical))
        terminal["session_identity_sha256"] = writer.identity
        terminal["cloud_sql"] = copy.deepcopy(cloud)
        terminal["services"] = copy.deepcopy(services)
        terminal["observed_at_unix"] = observed_at_unix
        unsigned = {
            name: item
            for name, item in terminal.items()
            if name != "observation_sha256"
        }
        terminal["observation_sha256"] = phase_b._sha256_json(unsigned)
    finally:
        writer.close()
    continuity = phase_b.build_phase_b_terminal_bootstrap_continuity(
        durable,
        bootstrap_resource=historical_cloud["bootstrap_resource"],
        operation_ledger_sha256=cloud["operation_ledger_sha256"],
        observed_at_unix=observed_at_unix,
    )
    readiness = {
        "schema": phase_b.PHASE_B_READINESS_OBSERVATION_SCHEMA,
        "current_release_revision": release_revision,
        "foundation_generation_sha256": durable.generation_sha256,
        "foundation_terminal_receipt_sha256": durable.terminal_receipt[
            "receipt_sha256"
        ],
        "terminal_observation": terminal,
        "host_identity": _readiness_host_identity(observed_at_unix),
        "cloud_sql": {
            "observation": cloud,
            "observed_at_unix": observed_at_unix,
        },
        "credential": {
            "identity": _credential(),
            "observed_at_unix": observed_at_unix,
        },
        "services": services,
        "bootstrap_runtime_continuity": continuity.to_mapping(),
        "observed_at_unix": observed_at_unix,
    }
    return (
        phase_b._PhaseBReadinessObservation.from_mapping(
            _hashed(readiness, "observation_sha256"),
            foundation=durable,
            current_release_revision=release_revision,
            now_unix=observed_at_unix,
        ),
        writer,
    )


def _readiness_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_parsed_for_mechanics: bool = True,
) -> tuple[
    phase_b.AppendOnlyPhaseBReadinessJournal,
    phase_b._PhaseBReadinessWriterBoundary,
]:
    monkeypatch.setattr(
        phase_b,
        "_PHASE_B_READINESS_ROOT",
        tmp_path / "readiness",
    )
    monkeypatch.setattr(phase_b, "_READINESS_AUTHORITY_UID", os.geteuid())
    monkeypatch.setattr(
        phase_b,
        "_READINESS_AUTHORITY_GID",
        tmp_path.stat().st_gid,
    )
    if allow_parsed_for_mechanics:
        # Journal durability is tested independently of the not-yet-packaged
        # live collector.  Production deliberately has no mapping-to-authority
        # issuer; this test-local consumer override cannot escape pytest.
        monkeypatch.setattr(
            phase_b,
            "_require_trusted_readiness_observation",
            lambda value: (
                None
                if type(value) is phase_b._PhaseBReadinessObservation
                else (_ for _ in ()).throw(
                    phase_b.PhaseBError(
                        "phase_b_readiness_observation_untrusted"
                    )
                )
            ),
        )
    return (
        phase_b.AppendOnlyPhaseBReadinessJournal(),
        phase_b._PhaseBReadinessWriterBoundary(),
    )


def test_durable_loader_accepts_expired_approval_without_writes_and_defends_copies(
    tmp_path: Path,
) -> None:
    state = _State()
    receipt, journal, plan = _run(state, tmp_path)
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in journal.root.rglob("*")
    }

    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=journal,
    )
    replay = phase_b.execute_approved_phase_b(
        plan,
        approval=_approval(plan),
        role_artifact=_artifact(plan, tmp_path),
        journal=journal,
        dependencies=_dependencies(state, plan),
        _clock=lambda: NOW + 301,
    )
    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in journal.root.rglob("*")
    }

    assert replay == receipt == durable.terminal_receipt
    assert before == after
    assert durable.generation["foundation_release_revision"] == REVISION
    forged = durable.terminal_receipt
    forged["safe_to_start"] = False
    assert durable.terminal_receipt["safe_to_start"] is True
    forged_generation = durable.generation
    forged_generation["generation_sha256"] = "0" * 64
    assert durable.generation_sha256 != "0" * 64


def test_forged_durable_foundation_cannot_reach_any_authority_consumer(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=journal,
    )
    with pytest.raises(TypeError):
        phase_b.PhaseBDurableFoundation()
    with pytest.raises(AttributeError):
        durable._generation_bytes = b"{}"

    forged = object.__new__(phase_b.PhaseBDurableFoundation)
    for attribute in (
        "_plan_bytes",
        "_approval_bytes",
        "_generation_bytes",
        "_terminal_receipt_bytes",
        "_terminal_observation_bytes",
    ):
        object.__setattr__(forged, attribute, getattr(durable, attribute))
    forged_generation = durable.generation
    forged_generation["generation_sha256"] = "f" * 64
    object.__setattr__(
        forged,
        "_generation_bytes",
        phase_b._canonical_bytes(forged_generation),
    )
    object.__setattr__(
        forged,
        "_authority_seal",
        durable._authority_seal,
    )
    object.__setattr__(
        forged,
        "_authority_tag",
        durable._authority_tag,
    )
    historical_cloud = durable.terminal_observation["cloud_sql"]
    with pytest.raises(
        phase_b.PhaseBError,
        match="durable_foundation_untrusted",
    ):
        phase_b.build_phase_b_terminal_bootstrap_continuity(
            forged,
            bootstrap_resource=historical_cloud["bootstrap_resource"],
            operation_ledger_sha256=historical_cloud[
                "operation_ledger_sha256"
            ],
            observed_at_unix=NOW,
        )


def test_durable_loader_semantically_replays_intermediate_chain(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    paths = sorted(journal._entries_root(plan).iterdir())
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    service = next(value for value in values if value["event"] == "services_stopped")
    service["evidence"]["transition"] = "forged_transition"
    previous: str | None = None
    for path, value in zip(paths, values, strict=True):
        value["previous_entry_sha256"] = previous
        unsigned = {
            name: item for name, item in value.items() if name != "entry_sha256"
        }
        value["entry_sha256"] = phase_b._sha256_json(unsigned)
        previous = value["entry_sha256"]
        path.write_bytes(phase_b._canonical_bytes(value))

    with pytest.raises(phase_b.PhaseBError, match="journal_services_invalid"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_durable_loader_rejects_future_terminal_evidence_after_full_rehash(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(journal._entries_root(plan).iterdir())
    ]
    terminal_observed = next(
        value for value in values if value["event"] == "terminal_observed"
    )
    observation = terminal_observed["evidence"]["terminal_observation"]
    observation["observed_at_unix"] = NOW + 1
    observation["cloud_sql"]["observed_at_unix"] = NOW + 1
    services_unsigned = {
        name: item
        for name, item in observation["services"].items()
        if name != "attestation_sha256"
    }
    services_unsigned["observed_at_unix"] = NOW + 1
    observation["services"] = _hashed(
        services_unsigned,
        "attestation_sha256",
    )
    observation["observation_sha256"] = phase_b._sha256_json(
        {
            name: item
            for name, item in observation.items()
            if name != "observation_sha256"
        }
    )
    terminal = values[-1]["evidence"]["terminal_receipt"]
    terminal["terminal_observation_sha256"] = observation[
        "observation_sha256"
    ]
    terminal["receipt_sha256"] = phase_b._sha256_json(
        {
            name: item
            for name, item in terminal.items()
            if name != "receipt_sha256"
        }
    )
    _rewrite_journal_entries(journal, plan, values)

    with pytest.raises(phase_b.PhaseBError, match="terminal_observation_invalid"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_durable_loader_binds_approval_to_exact_terminal_append_time(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(journal._entries_root(plan).iterdir())
    ]
    terminal_entry = values[-1]
    terminal = terminal_entry["evidence"]["terminal_receipt"]
    terminal["terminal_at_unix"] = NOW + 300
    terminal["receipt_sha256"] = phase_b._sha256_json(
        {
            name: item
            for name, item in terminal.items()
            if name != "receipt_sha256"
        }
    )
    terminal_entry["recorded_at_unix"] = NOW + 300
    _rewrite_journal_entries(journal, plan, values)

    with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_durable_loader_consumes_each_service_boundary_once(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(journal._entries_root(plan).iterdir())
    ]
    values.remove(
        next(
            value
            for value in values
            if value["event"] == "services_stopped"
            and value["evidence"]["transition"] == "temporary_admin_delete"
        )
    )
    _rewrite_journal_entries(journal, plan, values)

    with pytest.raises(phase_b.PhaseBError, match="service_boundary_missing"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_durable_loader_rejects_distinct_unconsumed_service_transition(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(journal._entries_root(plan).iterdir())
    ]
    initial_index = next(
        index
        for index, value in enumerate(values)
        if value["event"] == "services_stopped"
        and value["evidence"]["transition"]
        == "temporary_admin_initial_authority"
    )
    distinct = copy.deepcopy(
        next(
            value
            for value in values
            if value["event"] == "services_stopped"
            and value["evidence"]["transition"] == "phase_b_role_artifact"
        )
    )
    distinct["idempotency_key"] = "services:forged-distinct-unconsumed"
    values.insert(initial_index + 1, distinct)
    _rewrite_journal_entries(journal, plan, values)

    with pytest.raises(phase_b.PhaseBError, match="service_boundary_stale"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_durable_loader_rejects_distinct_authority_after_self_disable(
    tmp_path: Path,
) -> None:
    state = _State()
    _receipt_value, journal, plan = _run(state, tmp_path)
    values = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(journal._entries_root(plan).iterdir())
    ]
    disabled_index = next(
        index
        for index, value in enumerate(values)
        if value["event"] == "bootstrap_password_disabled"
    )
    rotation_service = copy.deepcopy(
        next(
            value
            for value in values
            if value["event"] == "services_stopped"
            and value["evidence"]["transition"]
            == "bootstrap_login_rotation_authority"
        )
    )
    rotation_service["idempotency_key"] = "services:forged-extra-authority"
    authority_entry = copy.deepcopy(
        [
            value
            for value in values
            if value["event"] == "bootstrap_authority"
        ][-1]
    )
    authority = authority_entry["evidence"]["authority_receipt"]
    authority["operation_name"] = "bootstrap-forged-extra"
    authority["receipt_sha256"] = phase_b._sha256_json(
        {
            name: item
            for name, item in authority.items()
            if name != "receipt_sha256"
        }
    )
    authority_entry["idempotency_key"] = (
        "bootstrap-authority:" + authority["receipt_sha256"]
    )
    values[disabled_index + 1 : disabled_index + 1] = [
        rotation_service,
        authority_entry,
    ]
    _rewrite_journal_entries(journal, plan, values)

    with pytest.raises(phase_b.PhaseBError, match="authority_after_disable"):
        phase_b.load_durable_phase_b_foundation(
            plan,
            approval=_approval(plan),
            journal=journal,
        )


def test_expired_nonterminal_approval_cannot_create_journal_or_invoke_boundary(
    tmp_path: Path,
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    state = _State()
    with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=lambda: NOW + 301,
        )
    assert not journal.root.exists()
    assert state.calls == []


def test_read_only_loader_fails_closed_on_directory_disappearance_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    journal.root.mkdir(mode=0o700)
    plan_root = journal._plan_root(plan)
    plan_root.mkdir(mode=0o700)
    os.chown(journal.root, -1, os.getegid())
    os.chown(plan_root, -1, os.getegid())
    original_stat = phase_b.os.stat
    mutations: list[str] = []
    removed = False

    def racing_stat(
        path: os.PathLike[str] | str,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        nonlocal removed
        result = original_stat(path, *args, **kwargs)
        if (
            str(path) == plan.sha256
            and kwargs.get("dir_fd") is not None
            and not removed
        ):
            plan_root.rmdir()
            removed = True
        return result

    def forbidden_mkdir(*_args: Any, **_kwargs: Any) -> None:
        mutations.append("mkdir")
        raise AssertionError("read-only loader attempted mkdir")

    def forbidden_chown(*_args: Any, **_kwargs: Any) -> None:
        mutations.append("chown")
        raise AssertionError("read-only loader attempted chown")

    monkeypatch.setattr(phase_b.os, "stat", racing_stat)
    monkeypatch.setattr(phase_b.os, "mkdir", forbidden_mkdir)
    monkeypatch.setattr(phase_b.os, "chown", forbidden_chown)
    with pytest.raises(phase_b.PhaseBError, match="journal_unavailable"):
        journal.load(plan)
    assert removed is True
    assert mutations == []


def test_terminal_publication_revalidates_approval_at_terminal_time(
    tmp_path: Path,
) -> None:
    state = _State()
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")

    def clock() -> int:
        return (
            NOW + 301
            if "services:terminal_observation" in state.calls
            else NOW
        )

    with pytest.raises(phase_b.PhaseBError, match="approval_invalid"):
        phase_b.execute_approved_phase_b(
            plan,
            approval=_approval(plan),
            role_artifact=_artifact(plan, tmp_path),
            journal=journal,
            dependencies=_dependencies(state, plan),
            _clock=clock,
        )
    assert "terminal" not in journal.events(plan)


def test_readiness_publication_is_trusted_append_only_and_cross_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    current_revision = "d" * 40
    prior_revision = "c" * 40
    observed_at = NOW + 100
    first_observation, first_session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=prior_revision,
        observed_at_unix=observed_at,
    )
    readiness_journal, readiness_writer = _readiness_boundaries(
        tmp_path, monkeypatch
    )

    first = readiness_writer.publish(
        durable,
        observation=first_observation,
        current_release_revision=prior_revision,
        now_unix=observed_at,
    )
    with pytest.raises(phase_b.PhaseBError):
        readiness_journal.latest_fresh(
            durable,
            expected_current_release_revision=current_revision,
            now_unix=observed_at,
        )
    second_observation, second_session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=current_revision,
        observed_at_unix=observed_at,
    )
    second = readiness_writer.publish(
        durable,
        observation=second_observation,
        current_release_revision=current_revision,
        now_unix=observed_at,
    )
    before_validation = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in readiness_journal.root.rglob("*")
    }
    validated = phase_b.validate_published_phase_b_readiness_receipt(
        second.to_mapping(),
        foundation=durable,
        journal=readiness_journal,
        expected_current_release_revision=current_revision,
        now_unix=observed_at,
    )
    with pytest.raises(phase_b.PhaseBError, match="not_current_head"):
        phase_b.validate_published_phase_b_readiness_receipt(
            first.to_mapping(),
            foundation=durable,
            journal=readiness_journal,
            expected_current_release_revision=current_revision,
            now_unix=observed_at,
        )
    after_validation = {
        path: (path.stat().st_size, path.stat().st_mtime_ns, path.stat().st_ctime_ns)
        for path in readiness_journal.root.rglob("*")
    }

    assert first.sequence == 0
    assert second.sequence == 1
    assert second.to_mapping()["previous_receipt_sha256"] == first.sha256
    assert validated.sha256 == second.sha256
    assert durable.generation["foundation_release_revision"] == REVISION
    assert second.to_mapping()["current_release_revision"] == current_revision
    assert first_session.closed and second_session.closed
    assert before_validation == after_validation
    with pytest.raises(AttributeError):
        second._value_bytes = b"{}"
    forged_receipt = object.__new__(phase_b.PhaseBReadinessReceipt)
    forged_value = second.to_mapping()
    forged_value["safe_to_start"] = False
    object.__setattr__(
        forged_receipt,
        "_value_bytes",
        phase_b._canonical_bytes(forged_value),
    )
    object.__setattr__(
        forged_receipt,
        "_authority_seal",
        second._authority_seal,
    )
    object.__setattr__(
        forged_receipt,
        "_authority_tag",
        second._authority_tag,
    )
    with pytest.raises(phase_b.PhaseBError, match="receipt_untrusted"):
        _ = forged_receipt.sha256
    defensive = second.to_mapping()
    defensive["safe_to_start"] = False
    assert second.to_mapping()["safe_to_start"] is True


def test_readiness_writer_fsyncs_every_new_directory_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    _reader, writer = _readiness_boundaries(tmp_path, monkeypatch)
    fsynced: list[Path] = []
    original_fsync = writer._journal._fsync_directory_handle

    def recording_fsync(descriptor: int, path: Path) -> None:
        fsynced.append(path)
        original_fsync(descriptor, path)

    monkeypatch.setattr(
        phase_b._PhaseBReadinessJournalFoundation,
        "_fsync_directory_handle",
        staticmethod(recording_fsync),
    )
    writer.publish(
        durable,
        observation=observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )

    generation_root = writer._journal._generation_root(durable)
    assert writer.root.parent in fsynced
    assert writer.root in fsynced
    # lock, entries, and staging creation each durably update generation.
    assert fsynced.count(generation_root) >= 3


def test_readiness_writer_fails_closed_when_parent_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    _reader, writer = _readiness_boundaries(tmp_path, monkeypatch)
    original_fsync = writer._journal._fsync_directory_handle
    failed = False

    def failing_fsync(descriptor: int, path: Path) -> None:
        nonlocal failed
        if path == writer.root.parent and not failed:
            failed = True
            raise OSError("injected parent fsync failure")
        original_fsync(descriptor, path)

    monkeypatch.setattr(
        phase_b._PhaseBReadinessJournalFoundation,
        "_fsync_directory_handle",
        staticmethod(failing_fsync),
    )
    with pytest.raises(phase_b.PhaseBError, match="journal_unavailable"):
        writer.publish(
            durable,
            observation=observation,
            current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert not any(writer.root.rglob("*.json"))
    recovered = writer.publish(
        durable,
        observation=observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert failed is True
    assert recovered.sequence == 0


def test_readiness_publish_fsync_fault_preserves_linked_stage_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    _reader, writer = _readiness_boundaries(tmp_path, monkeypatch)
    entries_root = writer._journal._entries_root(durable)
    staging_root = writer._journal._staging_root(durable)
    original_fsync = writer._journal._fsync_directory_handle
    entries_fsyncs = 0

    def fail_entries_once(descriptor: int, path: Path) -> None:
        nonlocal entries_fsyncs
        if path == entries_root:
            entries_fsyncs += 1
            if entries_fsyncs == 2:
                raise OSError("injected entries fsync failure")
        original_fsync(descriptor, path)

    monkeypatch.setattr(
        phase_b._PhaseBReadinessJournalFoundation,
        "_fsync_directory_handle",
        staticmethod(fail_entries_once),
    )
    with pytest.raises(phase_b.PhaseBError, match="publish_failed"):
        writer.publish(
            durable,
            observation=observation,
            current_release_revision=REVISION,
            now_unix=observed_at,
        )
    stage = next(staging_root.iterdir())
    final = next(entries_root.iterdir())
    assert stage.stat().st_nlink == final.stat().st_nlink == 2

    monkeypatch.setattr(
        phase_b._PhaseBReadinessJournalFoundation,
        "_fsync_directory_handle",
        staticmethod(original_fsync),
    )
    retry_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    retry = writer.publish(
        durable,
        observation=retry_observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert retry.sequence == 1
    assert list(staging_root.iterdir()) == []


def test_readiness_current_head_validation_holds_shared_lock_through_compare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    first_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    reader, writer = _readiness_boundaries(tmp_path, monkeypatch)
    first = writer.publish(
        durable,
        observation=first_observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    second_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    entered_compare = threading.Event()
    release_compare = threading.Event()
    writer_done = threading.Event()
    failures: list[BaseException] = []
    validated: list[phase_b.PhaseBReadinessReceipt] = []
    original_latest = reader._journal._latest_fresh_unlocked

    def paused_latest(
        foundation_value: phase_b.PhaseBDurableFoundation,
        *,
        expected_current_release_revision: str,
        now_unix: int | None = None,
    ) -> phase_b.PhaseBReadinessReceipt:
        result = original_latest(
            foundation_value,
            expected_current_release_revision=(
                expected_current_release_revision
            ),
            now_unix=now_unix,
        )
        entered_compare.set()
        if not release_compare.wait(2):
            raise AssertionError("comparison release timed out")
        return result

    monkeypatch.setattr(
        reader._journal,
        "_latest_fresh_unlocked",
        paused_latest,
    )

    def validate_head() -> None:
        try:
            validated.append(
                phase_b.validate_published_phase_b_readiness_receipt(
                    first.to_mapping(),
                    foundation=durable,
                    journal=reader,
                    expected_current_release_revision=REVISION,
                    now_unix=observed_at,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)

    def append_next() -> None:
        try:
            writer.publish(
                durable,
                observation=second_observation,
                current_release_revision=REVISION,
                now_unix=observed_at,
            )
        except BaseException as exc:  # pragma: no cover - asserted below.
            failures.append(exc)
        finally:
            writer_done.set()

    validator_thread = threading.Thread(target=validate_head)
    writer_thread = threading.Thread(target=append_next)
    validator_thread.start()
    assert entered_compare.wait(2)
    writer_thread.start()
    time.sleep(0.05)
    assert not writer_done.is_set()
    release_compare.set()
    validator_thread.join(2)
    writer_thread.join(2)

    assert not validator_thread.is_alive()
    assert not writer_thread.is_alive()
    assert failures == []
    assert [receipt.sha256 for receipt in validated] == [first.sha256]
    assert writer_done.is_set()


def test_readiness_shared_lock_uses_held_dirfd_after_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    reader, writer = _readiness_boundaries(tmp_path, monkeypatch)
    receipt = writer.publish(
        durable,
        observation=observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    generation_root = writer._journal._generation_root(durable)
    displaced = generation_root.with_name(generation_root.name + ".displaced")
    original_load = (
        phase_b._PhaseBReadinessJournalFoundation._load_receipt_bytes_from_fd
    )
    swapped = False

    def swap_then_load(
        journal: phase_b._PhaseBReadinessJournalFoundation,
        foundation_value: phase_b.PhaseBDurableFoundation,
        entries_fd: int,
        *,
        linked_sequence: int | None = None,
    ) -> list[bytes]:
        nonlocal swapped
        if not swapped:
            generation_root.rename(displaced)
            generation_root.mkdir(mode=0o700)
            swapped = True
        return original_load(
            journal,
            foundation_value,
            entries_fd,
            linked_sequence=linked_sequence,
        )

    monkeypatch.setattr(
        phase_b._PhaseBReadinessJournalFoundation,
        "_load_receipt_bytes_from_fd",
        swap_then_load,
    )
    validated = reader.latest_fresh(
        durable,
        expected_current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert swapped is True
    assert validated.sha256 == receipt.sha256
    assert list(generation_root.iterdir()) == []
    assert any(displaced.iterdir())


def test_untrusted_shape_cannot_mint_or_publish_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert "PhaseBReadinessCollectors" not in phase_b.__all__
    assert "publish_phase_b_readiness" not in phase_b.__all__
    assert not hasattr(phase_b, "PhaseBReadinessCollectors")
    assert not hasattr(phase_b, "publish_phase_b_readiness")
    assert not hasattr(phase_b.PhaseBReadinessReceipt, "from_mapping")
    assert "PhaseBReadinessObservation" not in phase_b.__all__
    assert not hasattr(
        phase_b._PhaseBReadinessObservation,
        "_from_trusted_mapping",
    )
    assert not hasattr(phase_b, "_READINESS_OBSERVATION_CAPABILITY")
    with pytest.raises(TypeError):
        phase_b._PhaseBReadinessObservation({})
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    trusted, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    parsed_only = phase_b._PhaseBReadinessObservation.from_mapping(
        trusted.to_mapping(),
        foundation=durable,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    with pytest.raises(AttributeError):
        parsed_only._value_bytes = b"{}"
    journal, writer = _readiness_boundaries(
        tmp_path,
        monkeypatch,
        allow_parsed_for_mechanics=False,
    )
    assert not hasattr(journal, "append")
    assert not hasattr(journal, "lock")
    assert not hasattr(journal, "load")
    with pytest.raises(phase_b.PhaseBError, match="observation_untrusted"):
        writer.publish(
            durable,
            observation=parsed_only,
            current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert not journal.root.exists() or not any(journal.root.rglob("*.json"))


def test_fixed_collector_closure_issues_publishable_readiness_without_mapping_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    parsed, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    calls: list[tuple[str, int]] = []

    def collect_fixed(
        received: phase_b.PhaseBDurableFoundation,
        *,
        current_release_revision: str,
        observed_at_unix: int,
    ) -> Mapping[str, Any]:
        assert received is durable
        calls.append((current_release_revision, observed_at_unix))
        return parsed.to_mapping()

    monkeypatch.setattr(
        phase_b_runtime,
        "_collect_fixed_phase_b_readiness_mapping",
        collect_fixed,
    )
    issued = phase_b._collect_trusted_readiness_observation(
        durable,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    journal, writer = _readiness_boundaries(
        tmp_path,
        monkeypatch,
        allow_parsed_for_mechanics=False,
    )
    receipt = writer.publish(
        durable,
        observation=issued,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert receipt.to_mapping()["safe_to_start"] is True
    assert calls == [(REVISION, observed_at)]


def test_ordinary_caller_cannot_choose_readiness_root_or_mint_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        phase_b.AppendOnlyPhaseBReadinessJournal(tmp_path / "caller-root")
    assert not hasattr(phase_b, "PhaseBReadinessObservation")

    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    trusted, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    fixed_root = tmp_path / "fixed-root"
    monkeypatch.setattr(phase_b, "_PHASE_B_READINESS_ROOT", fixed_root)
    monkeypatch.setattr(
        phase_b,
        "_READINESS_AUTHORITY_UID",
        os.geteuid() + 1,
    )
    monkeypatch.setattr(phase_b, "_READINESS_AUTHORITY_GID", os.getegid())
    writer = phase_b._PhaseBReadinessWriterBoundary()
    with pytest.raises(phase_b.PhaseBError, match="writer_authority_required"):
        writer.publish(
            durable,
            observation=trusted,
            current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert not fixed_root.exists()


@pytest.mark.parametrize(
    "drift",
    [
        "stale",
        "stale_cloud",
        "database_session",
        "cloud_admin",
        "cloud_resource",
        "cloud_ledger",
        "credential",
        "services",
        "runtime_transition",
        "missing",
    ],
)
def test_readiness_rejects_stale_tampered_missing_or_conflicting_evidence(
    tmp_path: Path,
    drift: str,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    trusted, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    value = trusted.to_mapping()
    terminal = value["terminal_observation"]
    if drift == "database_session":
        terminal["session_identity_sha256"] = durable.terminal_observation[
            "session_identity_sha256"
        ]
    elif drift == "cloud_admin":
        value["cloud_sql"]["observation"]["temporary_admin_users"] = [
            plan.temporary_admin_username
        ]
    elif drift == "cloud_resource":
        value["cloud_sql"]["observation"]["instance_projection"][
            "state"
        ] = "SUSPENDED"
    elif drift == "cloud_ledger":
        value["cloud_sql"]["observation"]["operation_ledger_sha256"] = (
            "7" * 64
        )
        terminal["cloud_sql"]["operation_ledger_sha256"] = "7" * 64
    elif drift == "credential":
        value["credential"]["identity"]["inode"] += 1
    elif drift == "stale_cloud":
        value["cloud_sql"]["observed_at_unix"] -= 1
    elif drift == "services":
        value["services"]["services"][0]["active_state"] = "active"
        value["services"] = _hashed(
            {
                name: item
                for name, item in value["services"].items()
                if name != "attestation_sha256"
            },
            "attestation_sha256",
        )
        terminal["services"] = copy.deepcopy(value["services"])
    elif drift == "runtime_transition":
        continuity = value["bootstrap_runtime_continuity"]
        continuity["source_kind"] = "runtime_transition"
        continuity["continuity_sha256"] = phase_b._sha256_json(
            {
                name: item
                for name, item in continuity.items()
                if name != "continuity_sha256"
            }
        )
    elif drift == "missing":
        value.pop("credential")
    if drift in {"database_session", "cloud_ledger", "services"}:
        terminal["observation_sha256"] = phase_b._sha256_json(
            {
                name: item
                for name, item in terminal.items()
                if name != "observation_sha256"
            }
        )
    if drift != "missing":
        value["observation_sha256"] = phase_b._sha256_json(
            {
                name: item
                for name, item in value.items()
                if name != "observation_sha256"
            }
        )

    with pytest.raises(phase_b.PhaseBError):
        phase_b._PhaseBReadinessObservation.from_mapping(
            value,
            foundation=durable,
            current_release_revision=REVISION,
            now_unix=(
                observed_at + phase_b.PHASE_B_READINESS_MAX_AGE_SECONDS + 1
                if drift == "stale"
                else observed_at
            ),
        )


def test_readiness_journal_rejects_fork_stale_head_and_staging_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    journal, writer = _readiness_boundaries(tmp_path, monkeypatch)
    receipt = writer.publish(
        durable,
        observation=observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    with pytest.raises(phase_b.PhaseBError, match="receipt_invalid"):
        journal.latest_fresh(
            durable,
            expected_current_release_revision=REVISION,
            now_unix=(
                observed_at + phase_b.PHASE_B_READINESS_MAX_AGE_SECONDS
            ),
        )

    internal = writer._journal
    path = next(internal._entries_root(durable).iterdir())
    fork = path.with_name("00000001-" + receipt.sha256 + ".json")
    os.link(path, fork)
    with pytest.raises(phase_b.PhaseBError):
        journal.latest_fresh(
            durable,
            expected_current_release_revision=REVISION,
            now_unix=observed_at,
        )
    fork.unlink()
    staging = internal._staging_root(durable)
    residue = staging / "residue.stage"
    residue.write_text("residue", encoding="utf-8")
    before = (residue.stat().st_size, residue.stat().st_mtime_ns)
    with pytest.raises(phase_b.PhaseBError, match="staging_residue"):
        journal.latest_fresh(
            durable,
            expected_current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert residue.exists()
    assert (residue.stat().st_size, residue.stat().st_mtime_ns) == before


def test_readiness_journal_recovers_linked_and_stage_only_crashes_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _State()
    _terminal, foundation_journal, plan = _run(state, tmp_path)
    durable = phase_b.load_durable_phase_b_foundation(
        plan,
        approval=_approval(plan),
        journal=foundation_journal,
    )
    observed_at = NOW + 100
    first_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    journal, writer = _readiness_boundaries(tmp_path, monkeypatch)
    internal = writer._journal
    first = writer.publish(
        durable,
        observation=first_observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )

    first_path = internal._entries_root(durable) / (
        f"00000000-{first.sha256}.json"
    )
    linked_stage = internal._staging_root(durable) / (
        "00000000." + "1" * 32 + ".stage"
    )
    os.link(first_path, linked_stage)
    with pytest.raises(phase_b.PhaseBError, match="staging_residue"):
        journal.latest_fresh(
            durable,
            expected_current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert linked_stage.exists() and first_path.stat().st_nlink == 2

    second_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    second = writer.publish(
        durable,
        observation=second_observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert second.sequence == 1
    assert not linked_stage.exists()
    assert first_path.stat().st_nlink == 1

    third_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    stage_only = internal._staging_root(durable) / (
        "00000002." + "2" * 32 + ".stage"
    )
    stage_only.write_bytes(b'{"truncated":')
    stage_only.chmod(0o600)
    with pytest.raises(phase_b.PhaseBError, match="staging_residue"):
        journal.latest_fresh(
            durable,
            expected_current_release_revision=REVISION,
            now_unix=observed_at,
        )
    third = writer.publish(
        durable,
        observation=third_observation,
        current_release_revision=REVISION,
        now_unix=observed_at,
    )
    assert third.sequence == 2
    assert not stage_only.exists()

    conflicting_stage = internal._staging_root(durable) / (
        "00000002." + "3" * 32 + ".stage"
    )
    conflicting_stage.write_bytes(b'{"conflicting":')
    conflicting_stage.chmod(0o600)
    fourth_observation, _session = _trusted_readiness_observation(
        durable,
        state,
        release_revision=REVISION,
        observed_at_unix=observed_at,
    )
    with pytest.raises(phase_b.PhaseBError, match="staging_conflict"):
        writer.publish(
            durable,
            observation=fourth_observation,
            current_release_revision=REVISION,
            now_unix=observed_at,
        )
    assert conflicting_stage.exists()


def test_windows_import_path_fails_only_when_posix_journal_is_invoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan()
    journal = phase_b.AppendOnlyPhaseBJournal(tmp_path / "journal")
    monkeypatch.setattr(phase_b, "fcntl", None)
    with pytest.raises(phase_b.PhaseBError, match="posix_lock_unavailable"):
        with journal.lock(plan):
            pass
