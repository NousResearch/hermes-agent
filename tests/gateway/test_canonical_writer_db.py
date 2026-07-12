from __future__ import annotations

import os
import hashlib
import json
import socket
import ssl
import struct
import time
from dataclasses import replace

import pytest

from gateway import canonical_writer_db as writer_db


_ROUTINE_DEFINITION = (
    "CREATE FUNCTION canonical_brain.append_event(jsonb) RETURNS jsonb"
)
_ROUTINE_IDENTITY = writer_db.RoutineIdentity(
    signature="canonical_brain.append_event(jsonb)",
    owner="canonical_owner",
    security_definer=True,
    language="plpgsql",
    configuration=("search_path=pg_catalog, canonical_brain",),
    definition_sha256=hashlib.sha256(_ROUTINE_DEFINITION.encode()).hexdigest(),
)
_HELPER_DEFINITION = "CREATE FUNCTION canonical_brain._helper(jsonb) RETURNS jsonb"
_HELPER_IDENTITY = writer_db.RoutineIdentity(
    signature="canonical_brain._helper(jsonb)",
    owner="canonical_owner",
    security_definer=False,
    language="sql",
    configuration=("search_path=pg_catalog, canonical_brain",),
    definition_sha256=hashlib.sha256(_HELPER_DEFINITION.encode()).hexdigest(),
)
_EVENT_LOG_IDENTITY = writer_db.CanonicalEventLogIdentity(
    table=writer_db.CANONICAL_EVENT_LOG_TABLE,
    owner=writer_db.CANONICAL_EVENT_LOG_OWNER,
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
    columns=writer_db.CANONICAL_EVENT_LOG_COLUMNS,
    constraints=("PRIMARY KEY (event_id)",),
    user_triggers=(),
    rewrite_rules=(),
    policies=(),
    inheritance=False,
    non_owner_acl_grants=(),
    index_count=1,
    primary_index_exact=True,
)
_PRIVATE_RELATION_IDENTITIES = tuple(
    writer_db.CanonicalPrivateRelationIdentity(
        name=name,
        owner="canonical_owner",
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
        columns=('{"name":"id"}',),
        constraints=('{"definition":"PRIMARY KEY (id)"}',),
        indexes=('{"primary":true}',),
        index_owners=("canonical_owner:f",),
        user_triggers=(),
        rewrite_rules=(),
        policies=(),
        inheritance=False,
    )
    for name in writer_db.CANONICAL_PRIVATE_WRITER_TABLES
)
_PRIVATE_SCHEMA_IDENTITY = writer_db.CanonicalPrivateSchemaIdentity(
    schema="canonical_brain",
    owner="canonical_owner",
    owner_dangerous=False,
    relations=_PRIVATE_RELATION_IDENTITIES,
)


def _private_schema_with_relation_change(**changes):
    first, *remaining = _PRIVATE_SCHEMA_IDENTITY.relations
    return replace(
        _PRIVATE_SCHEMA_IDENTITY,
        relations=(replace(first, **changes), *remaining),
    )


def _credential(tmp_path, value: bytes = b"secret\n") -> writer_db.CredentialSource:
    path = tmp_path / "database-password"
    path.write_bytes(value)
    path.chmod(0o600)
    return writer_db.CredentialSource(path=path, expected_uid=os.getuid())


def _config(tmp_path, *, host: str = "db.internal") -> writer_db.WriterDBConfig:
    ca_file = tmp_path / "server-ca.pem"
    ca_file.write_text("test-ca", encoding="utf-8")
    ca_file.chmod(0o444)
    return writer_db.WriterDBConfig(
        host=host,
        port=5432,
        database="canonical",
        user="canonical_writer",
        ca_file=ca_file,
        credential=_credential(tmp_path),
    )


def _policy() -> writer_db.WriterPrivilegePolicy:
    return writer_db.WriterPrivilegePolicy(
        schema="canonical_brain",
        executable_routines=("canonical_brain.append_event(jsonb)",),
        routine_identities=(_ROUTINE_IDENTITY,),
        dependency_routine_identities=(_HELPER_IDENTITY,),
        canonical_owner_role="canonical_owner",
        private_schema_identity_sha256=_PRIVATE_SCHEMA_IDENTITY.sha256,
    )


def _managed_hba_receipt(
    *,
    certificate_sha256="d" * 64,
    observed_at_unix=2_000_000_000,
):
    return writer_db.ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v1",
        host="db.internal",
        port=5432,
        server_certificate_sha256=certificate_sha256,
        database="cloudsqladmin",
        user="canonical_writer",
        observed_at_unix=observed_at_unix,
        expires_at_unix=observed_at_unix + 300,
        sqlstate="28000",
        server_message=(
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_writer", database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def test_managed_hba_policy_rejects_arbitrary_digest_without_canonical_receipt():
    with pytest.raises(ValueError, match="receipt and digest must be paired"):
        replace(
            _policy(),
            managed_cloudsqladmin_hba_rejection_sha256="d" * 64,
        )

    receipt = _managed_hba_receipt()
    with pytest.raises(ValueError, match="receipt digest"):
        replace(
            _policy(),
            managed_cloudsqladmin_hba_rejection_receipt=receipt,
            managed_cloudsqladmin_hba_rejection_sha256="e" * 64,
        )


def _attestation(**changes) -> writer_db.PrivilegeAttestation:
    policy = _policy()
    value = writer_db.PrivilegeAttestation(
        role="canonical_writer",
        executable_routines=("canonical_brain.append_event(jsonb)",),
        routine_identities=(_ROUTINE_IDENTITY,),
        dependency_routine_identities=(_HELPER_IDENTITY,),
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        canonical_non_owner_acl_grants=(
            writer_db._expected_canonical_non_owner_acl_grants(policy)
        ),
        canonical_writer_role_inheritors=("canonical_writer:1:t:f",),
        canonical_event_log_identity=_EVENT_LOG_IDENTITY,
        canonical_private_schema_identity=_PRIVATE_SCHEMA_IDENTITY,
    )
    return replace(value, **changes)


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_credential_path_accepts_only_exact_private_modes(tmp_path, mode):
    source = _credential(tmp_path)
    source.path.chmod(mode)

    assert writer_db._read_credential(source) == "secret"


@pytest.mark.parametrize("mode", [0o000, 0o640, 0o660, 0o644])
def test_credential_path_rejects_non_private_modes(tmp_path, mode):
    source = _credential(tmp_path)
    source.path.chmod(mode)

    with pytest.raises(writer_db.CredentialSecurityError, match="mode"):
        writer_db._read_credential(source)


def test_credential_rejects_wrong_owner_symlink_nul_and_oversize(tmp_path):
    source = _credential(tmp_path)
    with pytest.raises(writer_db.CredentialSecurityError, match="owner"):
        writer_db._read_credential(replace(source, expected_uid=os.getuid() + 1))

    link = tmp_path / "password-link"
    link.symlink_to(source.path)
    with pytest.raises(writer_db.CredentialSecurityError):
        writer_db._read_credential(replace(source, path=link))

    source.path.write_bytes(b"bad\x00secret")
    with pytest.raises(writer_db.CredentialSecurityError, match="nul"):
        writer_db._read_credential(source)

    source.path.write_bytes(b"x" * (writer_db._MAX_CREDENTIAL_BYTES + 1))
    with pytest.raises(writer_db.CredentialSecurityError, match="length"):
        writer_db._read_credential(source)


def test_credential_fd_is_explicit_repeatable_and_not_closed(tmp_path):
    source = _credential(tmp_path)
    descriptor = os.open(source.path, os.O_RDONLY)
    try:
        fd_source = writer_db.CredentialSource(
            fd=descriptor,
            expected_uid=os.getuid(),
        )
        os.read(descriptor, 2)
        assert writer_db._read_credential(fd_source) == "secret"
        assert writer_db._read_credential(fd_source) == "secret"
        os.fstat(descriptor)
    finally:
        os.close(descriptor)


def test_credential_source_requires_exactly_one_explicit_source(tmp_path):
    with pytest.raises(ValueError, match="exactly one"):
        writer_db.CredentialSource(expected_uid=os.getuid())
    with pytest.raises(ValueError, match="exactly one"):
        writer_db.CredentialSource(
            expected_uid=os.getuid(),
            path=tmp_path / "secret",
            fd=3,
        )
    with pytest.raises(ValueError, match="mode policy"):
        writer_db.CredentialSource(
            expected_uid=os.getuid(),
            path=tmp_path / "secret",
            allowed_modes=frozenset({0o644}),
        )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"role": "postgres"}, "identity|postgres"),
        ({"superuser": True}, "dangerous"),
        ({"createdb": True}, "dangerous"),
        ({"createrole": True}, "dangerous"),
        ({"replication": True}, "dangerous"),
        ({"bypassrls": True}, "dangerous"),
        ({"table_owner": True}, "dangerous"),
        ({"routine_owner": True}, "dangerous"),
        ({"role_memberships": ("cloudsqlsuperuser",)}, "memberships"),
        ({"unexpected_privileges": ("schema:public",)}, "out_of_scope"),
        ({"public_acl_grants": ("routine:append_event:EXECUTE",)}, "public_acl"),
        (
            {"canonical_non_owner_acl_grants": (
                "function:canonical_brain._helper(jsonb)::canonical_owner:"
                "untrusted_role:EXECUTE:f",
            )},
            "canonical_non_owner_acl",
        ),
        (
            {
                "canonical_writer_role_inheritors": (
                    "canonical_writer:1:t:f",
                    "rogue_login:1:t:f",
                )
            },
            "canonical_writer_role_inheritors",
        ),
        ({"schema_privileges": ("USAGE", "CREATE")}, "schema"),
        ({"database_privileges": ("CONNECT", "TEMP")}, "database_privileges"),
        ({"executable_routines": ()}, "routine"),
        ({"routine_identities": ()}, "routine_identity"),
        ({"dependency_routine_identities": ()}, "dependency_routine_identity"),
        ({"canonical_event_log_identity": None}, "canonical_event_log_identity"),
        (
            {"canonical_private_schema_identity": None},
            "private_schema_identity",
        ),
        (
            {
                "canonical_private_schema_identity": replace(
                    _PRIVATE_SCHEMA_IDENTITY,
                    owner="rogue_owner",
                )
            },
            "private_schema_owner_identity",
        ),
        (
            {
                "canonical_private_schema_identity": replace(
                    _PRIVATE_SCHEMA_IDENTITY,
                    owner_dangerous=True,
                )
            },
            "private_schema_owner_identity",
        ),
        (
            {
                "canonical_private_schema_identity": _private_schema_with_relation_change(
                    owner="rogue_owner"
                )
            },
            "private_relation_identity",
        ),
        (
            {
                "canonical_private_schema_identity": _private_schema_with_relation_change(
                    owner_dangerous=True
                )
            },
            "private_relation_identity",
        ),
        (
            {
                "canonical_private_schema_identity": replace(
                    _PRIVATE_SCHEMA_IDENTITY,
                    relations=_PRIVATE_SCHEMA_IDENTITY.relations[:-1],
                )
            },
            "private_schema_relation_set",
        ),
        (
            {
                "canonical_private_schema_identity": replace(
                    _PRIVATE_SCHEMA_IDENTITY,
                    relations=(
                        *_PRIVATE_SCHEMA_IDENTITY.relations,
                        replace(
                            _PRIVATE_SCHEMA_IDENTITY.relations[0],
                            name="rogue_view",
                            relation_kind="v",
                            access_method="",
                            constraints=(),
                            indexes=(),
                            index_owners=(),
                        ),
                    ),
                )
            },
            "private_schema_relation_set",
        ),
        (
            {
                "canonical_private_schema_identity": _private_schema_with_relation_change(
                    relation_kind="S",
                    access_method="",
                )
            },
            "private_relation_identity",
        ),
        (
            {
                "canonical_private_schema_identity": _private_schema_with_relation_change(
                    columns=(
                        '{"name":"id"}',
                        '{"name":"forged"}',
                    )
                )
            },
            "private_schema_structure_digest",
        ),
        (
            {
                "canonical_private_schema_identity": _private_schema_with_relation_change(
                    index_owners=("rogue_owner:t",)
                )
            },
            "private_relation_identity",
        ),
        (
            {
                "canonical_event_log_identity": replace(
                    _EVENT_LOG_IDENTITY,
                    owner="legacy_owner",
                )
            },
            "canonical_event_log_identity",
        ),
        (
            {
                "canonical_event_log_identity": replace(
                    _EVENT_LOG_IDENTITY,
                    owner_dangerous=True,
                )
            },
            "canonical_event_log_identity",
        ),
        (
            {
                "canonical_event_log_identity": replace(
                    _EVENT_LOG_IDENTITY,
                    user_triggers=("forge_event",),
                )
            },
            "canonical_event_log_identity",
        ),
        (
            {
                "canonical_event_log_identity": replace(
                    _EVENT_LOG_IDENTITY,
                    primary_index_exact=False,
                )
            },
            "canonical_event_log_identity",
        ),
        (
            {
                "dependency_routine_identities": (
                    replace(
                        _HELPER_IDENTITY,
                        configuration=("search_path=public",),
                    ),
                )
            },
            "dependency_routine_identity",
        ),
        (
            {
                "routine_identities": (
                    replace(_ROUTINE_IDENTITY, owner_dangerous=True),
                )
            },
            "owner_is_dangerous",
        ),
        (
            {
                "table_grants": (
                    writer_db.TablePrivilegeGrant(
                        "canonical_brain.events", ("SELECT",)
                    ),
                )
            },
            "table",
        ),
    ],
)
def test_privilege_attestation_rejects_extra_or_missing_authority(changes, reason):
    with pytest.raises(writer_db.PrivilegeAttestationError, match=reason):
        writer_db.validate_privilege_attestation(
            _attestation(**changes),
            _policy(),
            expected_user="canonical_writer",
        )


def test_privilege_attestation_accepts_exact_policy():
    writer_db.validate_privilege_attestation(
        _attestation(),
        _policy(),
        expected_user="canonical_writer",
    )


def test_privilege_attestation_rejects_matching_but_unsafe_helper_search_path():
    unsafe = replace(
        _HELPER_IDENTITY,
        configuration=("search_path=public",),
    )
    with pytest.raises(
        writer_db.PrivilegeAttestationError,
        match="search_path_unsafe",
    ):
        writer_db.validate_privilege_attestation(
            _attestation(dependency_routine_identities=(unsafe,)),
            replace(_policy(), dependency_routine_identities=(unsafe,)),
            expected_user="canonical_writer",
        )


def test_fixed_statement_catalog_rejects_raw_or_multi_statement_sql():
    with pytest.raises(ValueError, match="schema-qualified"):
        writer_db.FixedStatement(name="raw", sql_template="SELECT * FROM events")
    with pytest.raises(ValueError, match="comment-free"):
        writer_db.FixedStatement(
            name="multi",
            sql_template="SELECT canonical_brain.append_event(); SELECT 1",
        )


def test_fixed_statement_renders_typed_values_without_sql_injection():
    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template=(
            "SELECT * FROM canonical_brain.append_event({{case_id}}, {{payload}})"
        ),
        parameters=(
            writer_db.ParameterSpec("case_id"),
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )

    rendered = writer_db._render_fixed(
        statement,
        {"case_id": "case'); DROP TABLE events; --", "payload": {"ok": True}},
    )

    assert rendered.count("convert_from(decode('") == 2
    assert "DROP TABLE" not in rendered
    assert "--" not in rendered
    with pytest.raises(writer_db.FixedStatementError, match="parameters"):
        writer_db._render_fixed(statement, {"case_id": "missing payload"})


def _private_relation_attestation_row(relation):
    def structured(values):
        return json.dumps([json.loads(item) for item in values])

    return (
        relation.name,
        relation.owner,
        "t" if relation.owner_dangerous else "f",
        relation.relation_kind,
        relation.persistence,
        "t" if relation.is_partition else "f",
        relation.access_method,
        str(relation.tablespace_oid),
        "t" if relation.row_security else "f",
        "t" if relation.force_row_security else "f",
        relation.replica_identity,
        json.dumps(list(relation.relation_options)),
        structured(relation.columns),
        structured(relation.constraints),
        structured(relation.indexes),
        json.dumps(list(relation.index_owners)),
        json.dumps(list(relation.user_triggers)),
        json.dumps(list(relation.rewrite_rules)),
        json.dumps(list(relation.policies)),
        "t" if relation.inheritance else "f",
    )


class _FakeSession:
    def __init__(self, statement_result=None):
        self.closed = False
        self.queries = []
        self.statement_result = statement_result or writer_db.QueryResult(
            ("receipt_id",), (("receipt-1",),), "SELECT 1"
        )

    def query(self, sql, *, maximum_rows):
        self.queries.append((sql, maximum_rows))
        if sql.startswith("BEGIN"):
            return writer_db.QueryResult((), (), "BEGIN")
        if sql.startswith("SELECT pg_catalog.pg_advisory_xact_lock_shared"):
            return writer_db.QueryResult((), ((None,),), "SELECT 1")
        if sql == "COMMIT":
            return writer_db.QueryResult((), (), "COMMIT")
        if sql == "ROLLBACK":
            return writer_db.QueryResult((), (), "ROLLBACK")
        if "FROM pg_catalog.pg_roles r" in sql:
            return writer_db.QueryResult(
                (), (("canonical_writer", "f", "f", "f", "f", "f"),), "SELECT 1"
            )
        if "FROM pg_catalog.pg_namespace namespace" in sql:
            return writer_db.QueryResult(
                (),
                ((
                    _PRIVATE_SCHEMA_IDENTITY.owner,
                    "t" if _PRIVATE_SCHEMA_IDENTITY.owner_dangerous else "f",
                ),),
                "SELECT 1",
            )
        if "c.relname = 'canonical_event_log'" in sql:
            return writer_db.QueryResult(
                (),
                ((
                    writer_db.CANONICAL_EVENT_LOG_OWNER,
                    "f",
                    "r",
                    "p",
                    "f",
                    "heap",
                    "0",
                    "f",
                    "f",
                    "d",
                    "[]",
                    json.dumps(list(writer_db.CANONICAL_EVENT_LOG_COLUMNS)),
                    '["PRIMARY KEY (event_id)"]',
                    "[]",
                    "[]",
                    "[]",
                    "f",
                    "[]",
                    "1",
                    "t",
                ),),
                "SELECT 1",
            )
        if "column_identity.value" in sql:
            return writer_db.QueryResult(
                (),
                tuple(
                    _private_relation_attestation_row(relation)
                    for relation in _PRIVATE_SCHEMA_IDENTITY.relations
                ),
                f"SELECT {len(_PRIVATE_SCHEMA_IDENTITY.relations)}",
            )
        if "SELECT object_acl FROM" in sql:
            return writer_db.QueryResult(
                (),
                tuple(
                    (grant,)
                    for grant in writer_db._expected_canonical_non_owner_acl_grants(
                        _policy()
                    )
                ),
                "SELECT 2",
            )
        if "FROM pg_catalog.pg_class c" in sql:
            if "has_sequence_privilege" in sql:
                return writer_db.QueryResult((), (), "SELECT 0")
            return writer_db.QueryResult((), (), "SELECT 0")
        if "FROM pg_catalog.pg_proc p" in sql:
            return writer_db.QueryResult(
                (),
                (
                    (
                        "canonical_brain.append_event(jsonb)",
                        "f",
                        "t",
                        "canonical_owner",
                        "f",
                        "f",
                        "f",
                        "f",
                        "f",
                        "t",
                        "plpgsql",
                        '["search_path=pg_catalog, canonical_brain"]',
                        _ROUTINE_DEFINITION,
                        "canonical_brain",
                    ),
                    (
                        "canonical_brain._helper(jsonb)",
                        "f",
                        "f",
                        "canonical_owner",
                        "f",
                        "f",
                        "f",
                        "f",
                        "f",
                        "f",
                        "sql",
                        '["search_path=pg_catalog, canonical_brain"]',
                        _HELPER_DEFINITION,
                        "canonical_brain",
                    ),
                ),
                "SELECT 1",
            )
        if "WITH RECURSIVE memberships" in sql:
            return writer_db.QueryResult((), (), "SELECT 0")
        if "WITH RECURSIVE inheritors" in sql:
            return writer_db.QueryResult(
                (), (("canonical_writer", "1", "t", "f"),), "SELECT 1"
            )
        if "FROM pg_catalog.pg_namespace n WHERE" in sql:
            return writer_db.QueryResult(
                (), (("canonical_brain", "f", "t", "f"),), "SELECT 1"
            )
        if "FROM pg_catalog.pg_database d" in sql:
            return writer_db.QueryResult(
                (), (("canonical", "t", "f", "t", "f", "f"),), "SELECT 1"
            )
        if "FROM pg_catalog.pg_tablespace s" in sql:
            return writer_db.QueryResult((), (), "SELECT 0")
        return self.statement_result

    def close(self):
        self.closed = True


class _FixedStatementFailureSession(_FakeSession):
    def __init__(self, sqlstate):
        super().__init__()
        self.sqlstate = sqlstate

    def query(self, sql, *, maximum_rows):
        if sql.startswith("SELECT * FROM canonical_brain.append_event"):
            self.queries.append((sql, maximum_rows))
            raise writer_db.PostgresProtocolError(
                "database_error_sqlstate:" + self.sqlstate
            )
        return super().query(sql, maximum_rows=maximum_rows)


class _ManagedHBAFakeSession(_FakeSession):
    def query(self, sql, *, maximum_rows):
        if sql.startswith("WITH database_row AS"):
            self.queries.append((sql, maximum_rows))
            return writer_db.QueryResult((), (("t",),), "SELECT 1")
        return super().query(sql, maximum_rows=maximum_rows)


class _DriftedCanonicalAclSession(_FakeSession):
    def query(self, sql, *, maximum_rows):
        if "SELECT object_acl FROM" in sql:
            expected = writer_db._expected_canonical_non_owner_acl_grants(
                _policy()
            )
            return writer_db.QueryResult(
                (),
                tuple((grant,) for grant in (
                    *expected,
                    (
                        "function:canonical_brain._helper(jsonb)::canonical_owner:"
                        "rogue_role:EXECUTE:f"
                    ),
                )),
                "SELECT 3",
            )
        return super().query(sql, maximum_rows=maximum_rows)


class _DriftedPrivateRelationOwnerSession(_FakeSession):
    def query(self, sql, *, maximum_rows):
        if "column_identity.value" in sql:
            drifted = _private_schema_with_relation_change(owner="rogue_owner")
            return writer_db.QueryResult(
                (),
                tuple(
                    _private_relation_attestation_row(relation)
                    for relation in drifted.relations
                ),
                f"SELECT {len(drifted.relations)}",
            )
        return super().query(sql, maximum_rows=maximum_rows)


def test_db_requires_startup_attestation_and_reattests_each_session(tmp_path):
    sessions = []

    def factory(_config):
        session = _FakeSession()
        sessions.append(session)
        return session

    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
        parameters=(
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=factory,
    )

    with pytest.raises(writer_db.PrivilegeAttestationError, match="startup"):
        database.execute_fixed("append_event", {"payload": {"case": 1}})
    assert database.startup_attest() == _attestation()
    result = database.query_fixed("append_event", {"payload": {"case": 1}})

    assert result.rows == (("receipt-1",),)
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert len(sessions[1].queries) == 17
    assert "canonical_brain.append_event" in sessions[1].queries[-2][0]
    event_identity_query = next(
        sql for sql, _ in sessions[1].queries
        if "c.relname = 'canonical_event_log'" in sql
    )
    private_schema_query = next(
        sql for sql, _ in sessions[1].queries
        if "FROM pg_catalog.pg_namespace namespace" in sql
    )
    private_relations_query = next(
        sql for sql, _ in sessions[1].queries
        if "column_identity.value" in sql
    )
    routine_identity_query = next(
        sql for sql, _ in sessions[1].queries
        if "FROM pg_catalog.pg_proc p" in sql
    )
    canonical_acl_query = next(
        sql for sql, _ in sessions[1].queries
        if "SELECT object_acl FROM" in sql
    )
    assert "owner.rolcanlogin" in event_identity_query
    assert "pg_catalog.pg_auth_members membership" in event_identity_query
    assert "attribute.attacl" in event_identity_query
    assert "column:%s:%s:%s:%s:%s" in event_identity_query
    assert "owner.rolcanlogin" in private_schema_query
    assert "pg_catalog.pg_auth_members membership" in private_schema_query
    assert "class.relkind IN ('r','p','S','v','m','f','c')" in (
        private_relations_query
    )
    assert "pg_catalog.pg_get_constraintdef" in private_relations_query
    assert "index_owner.rolcanlogin" in private_relations_query
    assert "NOT trigger.tgisinternal" in private_relations_query
    assert "owner.rolcanlogin" in routine_identity_query
    assert "owner_membership.member = owner.oid" in routine_identity_query
    assert "a.grantee <> n.nspowner" in canonical_acl_query
    assert "a.grantee <> c.relowner" in canonical_acl_query
    assert "attribute.attacl" in canonical_acl_query
    assert "a.is_grantable" in canonical_acl_query
    assert not hasattr(database, "query")
    assert not hasattr(database, "password")


def test_managed_hba_exception_is_actively_reprobed_on_every_startup(
    tmp_path,
    monkeypatch,
):
    now_unix = 2_000_000_000
    monkeypatch.setattr(time, "time", lambda: now_unix)
    expected = _managed_hba_receipt(observed_at_unix=now_unix)
    policy = replace(
        _policy(),
        managed_cloudsqladmin_hba_rejection_receipt=expected,
        managed_cloudsqladmin_hba_rejection_sha256=expected.sha256,
    )
    sessions = []
    probes = []

    def session_factory(_config):
        session = _ManagedHBAFakeSession()
        sessions.append(session)
        return session

    def probe(config):
        probes.append(config)
        return _managed_hba_receipt(observed_at_unix=now_unix)

    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=policy,
        statements=writer_db.StatementCatalog(()),
        _session_factory=session_factory,
        _managed_hba_probe=probe,
    )

    database.startup_attest()
    database.startup_attest()

    assert len(probes) == 2
    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    managed_acl_queries = [
        sql
        for session in sessions
        for sql, _ in session.queries
        if sql.startswith("WITH database_row AS")
    ]
    assert len(managed_acl_queries) == 2
    assert all(
        "pg_catalog.acldefault('d', database_row.datdba)" in sql
        for sql in managed_acl_queries
    )


def test_startup_uses_expired_baseline_only_as_peer_and_coordinate_pin(
    tmp_path,
    monkeypatch,
):
    now_unix = 2_000_001_000
    monkeypatch.setattr(time, "time", lambda: now_unix)
    baseline = _managed_hba_receipt(observed_at_unix=2_000_000_000)
    policy = replace(
        _policy(),
        managed_cloudsqladmin_hba_rejection_receipt=baseline,
        managed_cloudsqladmin_hba_rejection_sha256=baseline.sha256,
    )
    sessions = []
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=policy,
        statements=writer_db.StatementCatalog(()),
        _session_factory=lambda _config: (
            sessions.append(_ManagedHBAFakeSession()) or sessions[-1]
        ),
        _managed_hba_probe=lambda _config: _managed_hba_receipt(
            observed_at_unix=now_unix,
        ),
    )

    database.startup_attest()

    assert len(sessions) == 1
    assert sessions[0].closed is True


def test_managed_hba_startup_rejects_stale_or_different_peer_receipt(
    tmp_path,
    monkeypatch,
):
    now_unix = 2_000_000_000
    monkeypatch.setattr(time, "time", lambda: now_unix)
    expected = _managed_hba_receipt(observed_at_unix=now_unix)
    policy = replace(
        _policy(),
        managed_cloudsqladmin_hba_rejection_receipt=expected,
        managed_cloudsqladmin_hba_rejection_sha256=expected.sha256,
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=policy,
        statements=writer_db.StatementCatalog(()),
        _session_factory=lambda _config: pytest.fail(
            "database session must not open after HBA peer mismatch"
        ),
        _managed_hba_probe=lambda _config: _managed_hba_receipt(
            certificate_sha256="e" * 64,
            observed_at_unix=now_unix,
        ),
    )

    with pytest.raises(writer_db.PrivilegeAttestationError, match="binding"):
        database.startup_attest()


def test_managed_hba_runtime_refreshes_expired_proof_before_fixed_db_work(
    tmp_path,
    monkeypatch,
):
    current_unix = [2_000_000_000]
    monkeypatch.setattr(time, "time", lambda: current_unix[0])
    expected = _managed_hba_receipt(observed_at_unix=current_unix[0])
    policy = replace(
        _policy(),
        managed_cloudsqladmin_hba_rejection_receipt=expected,
        managed_cloudsqladmin_hba_rejection_sha256=expected.sha256,
    )
    events = []
    sessions = []

    def probe(config):
        events.append(("probe", current_unix[0], config.host))
        return _managed_hba_receipt(observed_at_unix=current_unix[0])

    def session_factory(config):
        events.append(("session", current_unix[0], config.host))
        session = _ManagedHBAFakeSession()
        sessions.append(session)
        return session

    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
        parameters=(
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=policy,
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=session_factory,
        _managed_hba_probe=probe,
    )

    database.startup_attest()
    current_unix[0] += 300
    database.query_fixed("append_event", {"payload": {"case": 1}})
    current_unix[0] += 1
    database.query_fixed("append_event", {"payload": {"case": 2}})

    assert events == [
        ("probe", 2_000_000_000, "db.internal"),
        ("session", 2_000_000_000, "db.internal"),
        ("session", 2_000_000_300, "db.internal"),
        ("probe", 2_000_000_301, "db.internal"),
        ("session", 2_000_000_301, "db.internal"),
    ]
    assert len(sessions) == 3
    assert all(session.closed for session in sessions)


@pytest.mark.parametrize("runtime_path", ["fixed", "projection"])
def test_managed_hba_runtime_refresh_failure_prevents_any_database_session(
    tmp_path,
    monkeypatch,
    runtime_path,
):
    current_unix = [2_000_000_000]
    monkeypatch.setattr(time, "time", lambda: current_unix[0])
    expected = _managed_hba_receipt(observed_at_unix=current_unix[0])
    policy = replace(
        _policy(),
        managed_cloudsqladmin_hba_rejection_receipt=expected,
        managed_cloudsqladmin_hba_rejection_sha256=expected.sha256,
    )
    probes = []
    sessions = []

    def probe(_config):
        probes.append(current_unix[0])
        return _managed_hba_receipt(
            certificate_sha256=("d" if len(probes) == 1 else "e") * 64,
            observed_at_unix=current_unix[0],
        )

    def session_factory(_config):
        session = _ManagedHBAFakeSession()
        sessions.append(session)
        return session

    if runtime_path == "fixed":
        statement = writer_db.FixedStatement(
            name="append_event",
            sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
            parameters=(
                writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
            ),
        )
    else:
        statement = writer_db.FixedStatement(
            name="op_projection_read_events",
            sql_template=(
                "SELECT * FROM canonical_brain.writer_projection_read_events"
                "({{request}})"
            ),
            parameters=(
                writer_db.ParameterSpec("request", writer_db.ParameterKind.JSON),
            ),
        )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=policy,
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=session_factory,
        _managed_hba_probe=probe,
    )
    database.startup_attest()
    current_unix[0] += 301

    with pytest.raises(writer_db.PrivilegeAttestationError, match="binding"):
        if runtime_path == "fixed":
            database.query_fixed("append_event", {"payload": {"case": 1}})
        else:
            with database.projection_read_transaction():
                pass

    assert probes == [2_000_000_000, 2_000_000_301]
    assert len(sessions) == 1
    assert sessions[0].closed is True


def _retry_test_database(tmp_path, sessions):
    available = iter(sessions)
    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
        parameters=(
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=lambda _config: next(available),
    )
    database.startup_attest()
    return database


@pytest.mark.parametrize("sqlstate", ["40001", "40P01"])
def test_fixed_transaction_retries_only_retryable_postgres_aborts(
    tmp_path,
    sqlstate,
):
    startup = _FakeSession()
    failed = _FixedStatementFailureSession(sqlstate)
    succeeded = _FakeSession()
    database = _retry_test_database(tmp_path, [startup, failed, succeeded])

    result = database.query_fixed("append_event", {"payload": {"case": 1}})

    assert result.rows == (("receipt-1",),)
    assert all(session.closed for session in (startup, failed, succeeded))
    failed_sql = [sql for sql, _maximum_rows in failed.queries]
    succeeded_sql = [sql for sql, _maximum_rows in succeeded.queries]
    assert sum("FROM pg_catalog.pg_roles r" in sql for sql in failed_sql) == 1
    assert sum("FROM pg_catalog.pg_roles r" in sql for sql in succeeded_sql) == 1
    assert failed_sql.count("ROLLBACK") == 1
    assert "COMMIT" not in failed_sql
    assert succeeded_sql.count("COMMIT") == 1


def test_fixed_transaction_stops_after_three_whole_attested_attempts(tmp_path):
    startup = _FakeSession()
    attempts = [_FixedStatementFailureSession("40001") for _ in range(3)]
    database = _retry_test_database(tmp_path, [startup, *attempts])

    with pytest.raises(
        writer_db.PostgresProtocolError,
        match="database_error_sqlstate:40001",
    ):
        database.query_fixed("append_event", {"payload": {"case": 1}})

    assert all(session.closed for session in (startup, *attempts))
    for session in attempts:
        sql = [query for query, _maximum_rows in session.queries]
        assert sum("FROM pg_catalog.pg_roles r" in query for query in sql) == 1
        assert sql.count("ROLLBACK") == 1
        assert "COMMIT" not in sql


def test_fixed_transaction_does_not_retry_other_database_errors(tmp_path):
    startup = _FakeSession()
    failed = _FixedStatementFailureSession("23505")
    database = _retry_test_database(tmp_path, [startup, failed])

    with pytest.raises(
        writer_db.PostgresProtocolError,
        match="database_error_sqlstate:23505",
    ):
        database.query_fixed("append_event", {"payload": {"case": 1}})

    assert startup.closed is True
    assert failed.closed is True
    sql = [query for query, _maximum_rows in failed.queries]
    assert sum("FROM pg_catalog.pg_roles r" in query for query in sql) == 1
    assert sql.count("ROLLBACK") == 1
    assert "COMMIT" not in sql


def test_each_runtime_call_rejects_new_arbitrary_canonical_acl(tmp_path):
    sessions = iter((_FakeSession(), _DriftedCanonicalAclSession()))
    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
        parameters=(
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=lambda _config: next(sessions),
    )
    database.startup_attest()

    with pytest.raises(
        writer_db.PrivilegeAttestationError,
        match="canonical_non_owner_acl",
    ):
        database.query_fixed("append_event", {"payload": {"case": 1}})


def test_each_runtime_call_rejects_private_relation_owner_drift(tmp_path):
    sessions = iter((_FakeSession(), _DriftedPrivateRelationOwnerSession()))
    statement = writer_db.FixedStatement(
        name="append_event",
        sql_template="SELECT * FROM canonical_brain.append_event({{payload}})",
        parameters=(
            writer_db.ParameterSpec("payload", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=lambda _config: next(sessions),
    )
    database.startup_attest()

    with pytest.raises(
        writer_db.PrivilegeAttestationError,
        match="private_relation_identity",
    ):
        database.query_fixed("append_event", {"payload": {"case": 1}})


def test_fixed_read_only_transaction_reuses_one_attested_serializable_snapshot(
    tmp_path,
):
    sessions = []

    def factory(_config):
        session = _FakeSession()
        sessions.append(session)
        return session

    statement = writer_db.FixedStatement(
        name="op_projection_read_events",
        sql_template=(
            "SELECT * FROM canonical_brain.writer_projection_read_events({{request}})"
        ),
        parameters=(
            writer_db.ParameterSpec("request", writer_db.ParameterKind.JSON),
        ),
        maximum_rows=1,
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=factory,
    )
    database.startup_attest()

    with database.projection_read_transaction() as transaction:
        first = transaction.query({"request": {"after": ""}})
        second = transaction.query({"request": {"after": "event-1"}})
        assert sessions[1].closed is False

    assert first.rows == second.rows == (("receipt-1",),)
    assert len(sessions) == 2
    snapshot_session = sessions[1]
    sql = [query for query, _maximum_rows in snapshot_session.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sum("FROM pg_catalog.pg_roles r" in query for query in sql) == 1
    assert sum("writer_projection_read_events" in query for query in sql) == 2
    assert sql.count("COMMIT") == 1
    assert "ROLLBACK" not in sql
    assert snapshot_session.closed is True
    with pytest.raises(writer_db.FixedStatementError, match="not_active"):
        transaction.query({"request": {"after": "event-2"}})


def test_fixed_read_only_transaction_rolls_back_and_closes_on_export_failure(
    tmp_path,
):
    sessions = []

    def factory(_config):
        session = _FakeSession()
        sessions.append(session)
        return session

    statement = writer_db.FixedStatement(
        name="op_projection_read_events",
        sql_template=(
            "SELECT * FROM canonical_brain.writer_projection_read_events({{request}})"
        ),
        parameters=(
            writer_db.ParameterSpec("request", writer_db.ParameterKind.JSON),
        ),
    )
    database = writer_db.CanonicalWriterDB(
        config=_config(tmp_path),
        privilege_policy=_policy(),
        statements=writer_db.StatementCatalog((statement,)),
        _session_factory=factory,
    )
    database.startup_attest()

    with pytest.raises(RuntimeError, match="projection rejected"):
        with database.projection_read_transaction() as transaction:
            transaction.query({"request": {"after": ""}})
            raise RuntimeError("projection rejected")

    snapshot_session = sessions[1]
    sql = [query for query, _maximum_rows in snapshot_session.queries]
    assert sql.count("BEGIN ISOLATION LEVEL SERIALIZABLE READ ONLY") == 1
    assert sql.count("ROLLBACK") == 1
    assert "COMMIT" not in sql
    assert snapshot_session.closed is True
    with pytest.raises(writer_db.FixedStatementError, match="not_active"):
        transaction.query({"request": {"after": "event-1"}})


class _MemorySocket:
    def __init__(self, incoming=b""):
        self.incoming = incoming
        self.sent = []
        self.timeouts = []
        self.closed = False

    def recv(self, count):
        result = self.incoming[:count]
        self.incoming = self.incoming[count:]
        return result

    def sendall(self, value):
        self.sent.append(value)

    def settimeout(self, value):
        self.timeouts.append(value)

    def getpeercert(self, *, binary_form=False):
        return b"verified-peer-certificate" if binary_form else {}

    def close(self):
        self.closed = True


def _message(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("!I", len(payload) + 4) + payload


def _error_response(*, sqlstate="28000", message=None):
    message = message or (
        'no pg_hba.conf entry for host "10.0.0.8", user '
        '"canonical_writer", database "cloudsqladmin", SSL encryption'
    )
    return _message(
        b"E",
        b"SFATAL\x00"
        + b"C"
        + sqlstate.encode("ascii")
        + b"\x00M"
        + message.encode("utf-8")
        + b"\x00\x00",
    )


@pytest.mark.parametrize(
    "message",
    [
        None,
        'pg_hba.conf rejects connection for host "10.0.0.8", user '
        '"canonical_writer", database "cloudsqladmin", SSL encryption',
    ],
)
def test_active_managed_hba_probe_parses_exact_wire_rejection_and_binds_peer(
    tmp_path,
    monkeypatch,
    message,
):
    protected = _MemorySocket(_error_response(message=message))
    monkeypatch.setattr(
        writer_db,
        "_open_verified_tls_connection",
        lambda _config: (protected, "f" * 64),
    )

    receipt = writer_db.collect_managed_cloudsqladmin_hba_receipt(
        _config(tmp_path),
        now_unix=2_000_000_000,
        ttl_seconds=60,
    )

    assert receipt.server_certificate_sha256 == "f" * 64
    assert receipt.sqlstate == "28000"
    assert receipt.result == "pg_hba_rejected"
    assert receipt.tls_peer_verified is True
    assert receipt.is_fresh(2_000_000_060)
    assert b"cloudsqladmin" in protected.sent[0]
    assert protected.closed is True


@pytest.mark.parametrize(
    "sqlstate,message",
    [
        ("28P01", None),
        ("28000", "permission denied"),
        (
            "28000",
            'no pg_hba.conf entry for host "10.0.0.8", user '
            '"canonical_writer", database "cloudsqladmin", no encryption',
        ),
    ],
)
def test_active_managed_hba_probe_rejects_non_exact_server_errors(
    tmp_path,
    monkeypatch,
    sqlstate,
    message,
):
    protected = _MemorySocket(_error_response(sqlstate=sqlstate, message=message))
    monkeypatch.setattr(
        writer_db,
        "_open_verified_tls_connection",
        lambda _config: (protected, "f" * 64),
    )

    with pytest.raises(
        writer_db.PrivilegeAttestationError,
        match="rejection_not_exact",
    ):
        writer_db.collect_managed_cloudsqladmin_hba_receipt(
            _config(tmp_path),
            now_unix=2_000_000_000,
        )


@pytest.mark.parametrize("host", ["db.internal", "10.20.30.40", "2001:db8::10"])
def test_wire_connection_uses_ca_verified_hostname_or_ip_tls(
    tmp_path, monkeypatch, host
):
    raw = _MemorySocket(b"S")
    protected = _MemorySocket(
        _message(b"R", struct.pack("!I", 0)) + _message(b"Z", b"I")
    )
    observed = {}

    class Context:
        verify_mode = None
        check_hostname = None
        minimum_version = None

        def wrap_socket(self, raw_socket, *, server_hostname):
            observed["raw"] = raw_socket
            observed["server_hostname"] = server_hostname
            return protected

    context = Context()

    def create_context(purpose, *, cafile):
        observed["purpose"] = purpose
        observed["cafile"] = cafile
        return context

    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: raw)
    monkeypatch.setattr(ssl, "create_default_context", create_context)

    session = writer_db._open_postgres_session(_config(tmp_path, host=host))
    session.close()

    assert observed["purpose"] is ssl.Purpose.SERVER_AUTH
    assert observed["cafile"].endswith("server-ca.pem")
    assert observed["server_hostname"] == host
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.minimum_version is ssl.TLSVersion.TLSv1_2
    assert raw.timeouts and protected.timeouts


def test_wire_connection_rejects_server_without_tls(tmp_path, monkeypatch):
    raw = _MemorySocket(b"N")
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: raw)

    with pytest.raises(writer_db.PostgresProtocolError, match="refused_tls"):
        writer_db._open_postgres_session(_config(tmp_path))


def test_protocol_rejects_oversized_frame_before_reading_payload():
    connection = _MemorySocket(
        b"D" + struct.pack("!I", writer_db._MAX_FRAME_BYTES + 5)
    )

    with pytest.raises(writer_db.PostgresProtocolError, match="frame_exceeds"):
        writer_db._recv_message(connection)
