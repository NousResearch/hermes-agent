"""Real PostgreSQL 18 contract tests for the privileged Canonical writer.

These tests are deliberately opt-in (``pytest -m integration``).  They start
one disposable local Docker container, enable TLS, create a dedicated
least-privilege database, apply the production migration twice, and exercise
the production PostgreSQL wire/backend path.  No Cloud service is contacted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_schema_reconciliation as schema_reconciliation
from gateway import (
    canonical_writer_schema_reconciliation_db as schema_reconciliation_db,
)
from gateway.canonical_writer_foundation import _load_source_artifacts_for_tests
from gateway.canonical_writer_db import (
    CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY,
    CanonicalWriterDB,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    PostgresProtocolError,
    RoutineIdentity,
    WriterDBConfig,
    WriterPrivilegePolicy,
    _collect_privilege_attestation,
    _open_postgres_session,
    validate_privilege_attestation,
)
from gateway.canonical_writer_handlers import (
    CanonicalWriterError,
    CanonicalWriterHandlers,
    EventAppendRequest,
    ProjectorReadRequest,
    RouteBackAuthorizeRequest,
    RouteBackTerminalRequest,
    RuntimeContext,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanonicalWriterBackend,
)
from gateway.canonical_writer_bootstrap import export_projection_events
from gateway.canonical_projection_export import validate_projection_export
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.canonical_writer_schema_reconciliation import (
    SchemaContractAsset,
    collect_schema_contract,
)
from gateway.discord_edge_protocol import (
    DiscordEdgeReceiptOutcome,
    parse_request_for_reconciliation,
    sign_receipt,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_writer_authority import (
    CanonicalWriterDiscordAuthority,
    derive_private_denial_receipt_sha256,
    derive_routeback_edge_idempotency_key,
)


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "sql" / "canonical_writer_v1.sql"
LEGACY_RECONCILIATION = (
    ROOT / "scripts" / "sql" / "canonical_writer_legacy_reconcile_v1.sql"
)
IMAGE = "postgres:18"
SCHEMA_CONTRACT_ASSET = (
    ROOT / "gateway/assets/canonical_writer_schema_contract_v1.json"
)
DATABASE = "muncho_canary_brain"
LOGIN = "muncho_canary_writer_login"
DISCORD_GUILD_ID = "1282725267068157972"
GUILD_CHANNEL = "1504852408227069993"
SYNTHETIC_PUBLIC_CANARY_CHANNEL = "1526858760100909066"
DISCORD_MESSAGE_ID = "100000000000000003"
DISCORD_BOT_ID = "100000000000000004"

SESSION = "a" * 64
EPOCH = "b" * 64
COMMAND = "c" * 64
SOURCE = "d" * 64
ROUTEBACK_CONTENT = "Real PostgreSQL route-back payload"
CONTENT = hashlib.sha256(ROUTEBACK_CONTENT.encode("utf-8")).hexdigest()
RACE_ROUTEBACK_CONTENT = "One global route-back payload"
RACE_CONTENT = hashlib.sha256(RACE_ROUTEBACK_CONTENT.encode("utf-8")).hexdigest()
WRITER_CAPABILITY_PRIVATE_KEY = Ed25519PrivateKey.generate()
EDGE_RECEIPT_PRIVATE_KEY = Ed25519PrivateKey.generate()


def _run(
    command: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 120,
    secret_values: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "command failed")[-4000:]
        for secret in secret_values:
            detail = detail.replace(secret, "<redacted>")
        raise RuntimeError(f"container command failed ({command[0]}): {detail}")
    return completed


def _wait_ready(name: str, *, timeout: int = 60) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        probe = subprocess.run(
            ["docker", "exec", name, "pg_isready", "-U", "postgres"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError("ephemeral PostgreSQL did not become ready")


def _generate_tls(directory: Path) -> tuple[Path, Path, Path]:
    ca_key = directory / "ca.key"
    ca_cert = directory / "ca.crt"
    server_key = directory / "server.key"
    server_csr = directory / "server.csr"
    server_cert = directory / "server.crt"
    extensions = directory / "server.ext"
    extensions.write_text(
        "subjectAltName=DNS:localhost\nextendedKeyUsage=serverAuth\n",
        encoding="utf-8",
    )
    _run([
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(ca_key),
        "-out",
        str(ca_cert),
        "-days",
        "1",
        "-subj",
        "/CN=Hermes Canonical Writer E2E CA",
    ])
    _run([
        "openssl",
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(server_key),
        "-out",
        str(server_csr),
        "-subj",
        "/CN=localhost",
    ])
    _run([
        "openssl",
        "x509",
        "-req",
        "-in",
        str(server_csr),
        "-CA",
        str(ca_cert),
        "-CAkey",
        str(ca_key),
        "-CAcreateserial",
        "-out",
        str(server_cert),
        "-days",
        "1",
        "-sha256",
        "-extfile",
        str(extensions),
    ])
    ca_cert.chmod(0o600)
    server_key.chmod(0o600)
    return ca_cert, server_cert, server_key


@dataclass
class RealWriterStack:
    name: str
    backend: PostgresCanonicalWriterBackend
    handlers: CanonicalWriterHandlers
    migration_runs: int


def _psql(name: str, database: str, sql: str, *, secrets_: tuple[str, ...] = ()) -> None:
    _run(
        [
            "docker", "exec", "-i", name, "psql", "-X", "-v",
            "ON_ERROR_STOP=1", "-U", "postgres", "-d", database,
        ],
        input_text=sql,
        timeout=180,
        secret_values=secrets_,
    )


def _psql_as(name: str, database: str, user: str, sql: str) -> None:
    _run(
        [
            "docker",
            "exec",
            "-i",
            name,
            "psql",
            "-X",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
        ],
        input_text=sql,
        timeout=180,
    )


def _psql_fields_as(
    name: str,
    database: str,
    user: str,
    sql: str,
) -> list[str]:
    completed = _run(
        [
            "docker",
            "exec",
            "-i",
            name,
            "psql",
            "-X",
            "-q",
            "-A",
            "-t",
            "-F",
            "|",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
        ],
        input_text=sql,
        timeout=180,
    )
    return completed.stdout.strip().split("|")


def _psql_fields(name: str, database: str, sql: str) -> list[str]:
    return _psql_fields_as(name, database, "postgres", sql)


def _canonical14_identity(name: str) -> list[str]:
    return _psql_fields(
        name,
        DATABASE,
        "WITH row_receipts AS (SELECT event.event_id, "
        "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
        "pg_catalog.jsonb_build_object("
        "'event_id', pg_catalog.to_jsonb(event)->'event_id',"
        "'schema_version', pg_catalog.to_jsonb(event)->'schema_version',"
        "'event_type', pg_catalog.to_jsonb(event)->'event_type',"
        "'occurred_at', pg_catalog.to_jsonb(event)->'occurred_at',"
        "'case_id', pg_catalog.to_jsonb(event)->'case_id',"
        "'source', pg_catalog.to_jsonb(event)->'source',"
        "'actor', pg_catalog.to_jsonb(event)->'actor',"
        "'subject', pg_catalog.to_jsonb(event)->'subject',"
        "'evidence', pg_catalog.to_jsonb(event)->'evidence',"
        "'decision', pg_catalog.to_jsonb(event)->'decision',"
        "'status', pg_catalog.to_jsonb(event)->'status',"
        "'next_action', pg_catalog.to_jsonb(event)->'next_action',"
        "'safety', pg_catalog.to_jsonb(event)->'safety',"
        "'payload', pg_catalog.to_jsonb(event)->'payload')::text, 'UTF8'"
        ")), 'hex') AS row_sha FROM public.canonical_event_log AS event) "
        "SELECT pg_catalog.count(*)::text, pg_catalog.encode(pg_catalog.sha256("
        "pg_catalog.convert_to('canonical-writer-legacy-reconcile-v1:canonical14'"
        " || E'\\n' || COALESCE(pg_catalog.string_agg(event_id::text || ':' || "
        "row_sha, E'\\n' ORDER BY event_id), ''), 'UTF8')), 'hex') "
        "FROM row_receipts;",
    )


def _jsonb_sql(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).replace("'", "''")
    return f"'{encoded}'::jsonb"


def _routine_error_code(
    stack: RealWriterStack,
    routine: str,
    request_sql: str,
    runtime_sql: str,
) -> str:
    assert routine.startswith("writer_") and routine.replace("_", "").isalnum()
    fields = _psql_fields_as(
        stack.name,
        DATABASE,
        LOGIN,
        "WITH response AS (SELECT canonical_brain."
        f"{routine}({request_sql},{runtime_sql}) AS value) "
        "SELECT COALESCE(value->'error'->>'code','<unexpected-success>') "
        "FROM response;",
    )
    assert len(fields) == 1
    return fields[0]


def _migration_invocation(
    database: str,
    migration_sql: str,
    *,
    include_cloudsqladmin_hba_receipt: bool = True,
) -> str:
    prefix = (
        "SET muncho.canonical_writer_migration_scope = 'isolated_canary_copy';\n"
        f"SET muncho.canonical_writer_migration_database = '{database}';\n"
        "SET muncho.canonical_writer_migration_approval_receipt_sha256 = "
        f"'{('a' * 64)}';\n"
    )
    if include_cloudsqladmin_hba_receipt:
        prefix += (
            "SET muncho.canonical_writer_cloudsqladmin_hba_rejection_sha256 = "
            f"'{('e' * 64)}';\n"
        )
    return prefix + migration_sql


def _test_managed_hba_receipt(config: WriterDBConfig) -> ManagedCloudSQLAdminHBAReceipt:
    observed_at = int(time.time())
    return ManagedCloudSQLAdminHBAReceipt(
        version="managed-cloudsqladmin-hba-rejection-v2",
        host=config.host,
        tls_server_name=config.tls_server_name,
        port=config.port,
        server_certificate_sha256="e" * 64,
        database="cloudsqladmin",
        user=config.user,
        observed_at_unix=observed_at,
        expires_at_unix=observed_at + 300,
        sqlstate="28000",
        server_message=(
            f'no pg_hba.conf entry for host "127.0.0.1", user "{config.user}", '
            'database "cloudsqladmin", SSL encryption'
        ),
        result="pg_hba_rejected",
        tls_peer_verified=True,
    )


def _seed_policy(config: WriterDBConfig) -> WriterPrivilegePolicy:
    def identity(signature: str, security_definer: bool) -> RoutineIdentity:
        return RoutineIdentity(
            signature=signature,
            owner=CANONICAL_WRITER_MIGRATION_OWNER,
            security_definer=security_definer,
            language="plpgsql",
            configuration=("search_path=pg_catalog, canonical_brain",),
            definition_sha256="0" * 64,
        )

    managed_hba_receipt = _test_managed_hba_receipt(config)
    return WriterPrivilegePolicy(
        schema=CANONICAL_WRITER_SCHEMA,
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=tuple(
            identity(value, True) for value in EXPECTED_ROUTINE_SIGNATURES
        ),
        dependency_routine_identities=tuple(
            identity(value, False) for value in EXPECTED_HELPER_ROUTINE_SIGNATURES
        ),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256="0" * 64,
        managed_cloudsqladmin_hba_rejection_receipt=managed_hba_receipt,
        managed_cloudsqladmin_hba_rejection_sha256=managed_hba_receipt.sha256,
    )


def _production_policy(config: WriterDBConfig) -> WriterPrivilegePolicy:
    seed = _seed_policy(config)
    session = _open_postgres_session(config)
    try:
        observed = _collect_privilege_attestation(
            session,
            config=config,
            policy=seed,
            managed_hba_receipt=(seed.managed_cloudsqladmin_hba_rejection_receipt),
        )
    finally:
        session.close()
    assert observed.canonical_private_schema_identity is not None
    policy = WriterPrivilegePolicy(
        schema=CANONICAL_WRITER_SCHEMA,
        executable_routines=observed.executable_routines,
        routine_identities=observed.routine_identities,
        dependency_routine_identities=observed.dependency_routine_identities,
        schema_privileges=("USAGE",),
        database_privileges=("CONNECT",),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256=(
            observed.canonical_private_schema_identity.sha256
        ),
        managed_cloudsqladmin_hba_rejection_receipt=(
            seed.managed_cloudsqladmin_hba_rejection_receipt
        ),
        managed_cloudsqladmin_hba_rejection_sha256=(
            seed.managed_cloudsqladmin_hba_rejection_sha256
        ),
    )
    validate_privilege_attestation(observed, policy, expected_user=LOGIN)
    return policy


@pytest.fixture(scope="module")
def real_writer_stack(tmp_path_factory: pytest.TempPathFactory) -> RealWriterStack:
    if shutil.which("docker") is None or shutil.which("openssl") is None:
        pytest.skip("Docker and OpenSSL are required")
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode:
        pytest.skip("Docker daemon is unavailable")

    directory = tmp_path_factory.mktemp("canonical-writer-real-pg")
    ca_cert, server_cert, server_key = _generate_tls(directory)
    admin_password = secrets.token_hex(32)
    writer_password = secrets.token_hex(32)
    credential = directory / "writer-password"
    credential.write_text(writer_password + "\n", encoding="utf-8")
    credential.chmod(0o600)
    name = "hermes-canonical-writer-e2e-" + uuid.uuid4().hex[:12]
    environment = dict(os.environ)
    environment["POSTGRES_PASSWORD"] = admin_password

    try:
        if subprocess.run(
            ["docker", "image", "inspect", IMAGE],
            capture_output=True,
            check=False,
        ).returncode:
            _run(["docker", "pull", IMAGE], timeout=300)
        _run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                "-e",
                "POSTGRES_PASSWORD",
                "-e",
                "POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256",
                "-p",
                "127.0.0.1::5432",
                IMAGE,
            ],
            env=environment,
            timeout=180,
            secret_values=(admin_password,),
        )
        _wait_ready(name)

        _run(["docker", "cp", str(server_cert), f"{name}:/tmp/cw-server.crt"])
        _run(["docker", "cp", str(server_key), f"{name}:/tmp/cw-server.key"])
        _run([
            "docker",
            "exec",
            "-u",
            "0",
            name,
            "sh",
            "-ec",
            "chown postgres:postgres /tmp/cw-server.crt /tmp/cw-server.key; "
            "chmod 0644 /tmp/cw-server.crt; chmod 0600 /tmp/cw-server.key",
        ])
        _psql(
            name,
            "postgres",
            "ALTER SYSTEM SET ssl = 'on';\n"
            "ALTER SYSTEM SET ssl_cert_file = '/tmp/cw-server.crt';\n"
            "ALTER SYSTEM SET ssl_key_file = '/tmp/cw-server.key';\n",
        )
        _run(["docker", "restart", name])
        _wait_ready(name)

        escaped_password = writer_password.replace("'", "''")
        _psql(
            name,
            "postgres",
            "REVOKE ALL ON DATABASE postgres FROM PUBLIC;\n"
            "REVOKE ALL ON DATABASE template1 FROM PUBLIC;\n"
            f"CREATE ROLE {CANONICAL_WRITER_MIGRATION_OWNER} NOLOGIN NOINHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_WRITER_ROLE} NOLOGIN "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {LOGIN} LOGIN INHERIT PASSWORD '{escaped_password}' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT {CANONICAL_WRITER_ROLE} TO {LOGIN} "
            "WITH ADMIN FALSE, INHERIT TRUE, SET FALSE;\n"
            "CREATE ROLE cloudsqladmin LOGIN SUPERUSER CREATEDB CREATEROLE "
            "REPLICATION BYPASSRLS;\n"
            "CREATE ROLE cloudsqlsuperuser LOGIN NOSUPERUSER CREATEDB CREATEROLE "
            "NOREPLICATION NOBYPASSRLS;\n"
            "CREATE DATABASE cloudsqladmin OWNER cloudsqladmin;\n"
            f"CREATE DATABASE {DATABASE} OWNER cloudsqlsuperuser;\n"
            f"REVOKE ALL ON DATABASE {DATABASE} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {DATABASE} TO {CANONICAL_WRITER_ROLE};\n",
            secrets_=(writer_password,),
        )
        # Cloud SQL PostgreSQL 18 currently represents the provider-owned
        # cloudsqladmin database with a NULL datacl.  PostgreSQL defines that
        # as the exact default database ACL (PUBLIC CONNECT/TEMPORARY plus all
        # owner privileges), so leave the Docker fixture in that live shape.
        assert _psql_fields(
            name,
            "postgres",
            "SELECT datacl IS NULL FROM pg_database "
            "WHERE datname = 'cloudsqladmin';",
        ) == ["t"]
        _psql(
            name,
            DATABASE,
            "REVOKE ALL ON SCHEMA public FROM PUBLIC;\n"
            f"GRANT USAGE ON SCHEMA public TO {CANONICAL_WRITER_MIGRATION_OWNER};\n"
            "CREATE TABLE public.canonical_event_log (\n"
            " event_id uuid NOT NULL, schema_version text NOT NULL,\n"
            " event_type text NOT NULL, occurred_at timestamptz NOT NULL,\n"
            " case_id text NOT NULL, source jsonb NOT NULL, actor jsonb NOT NULL,\n"
            " subject jsonb NOT NULL, evidence jsonb NOT NULL, decision jsonb NOT NULL,\n"
            " status jsonb NOT NULL, next_action jsonb NOT NULL, safety jsonb NOT NULL,\n"
            " payload jsonb NOT NULL, PRIMARY KEY (event_id)\n"
            ");\n"
            f"ALTER TABLE public.canonical_event_log OWNER TO {CANONICAL_WRITER_MIGRATION_OWNER};\n",
        )
        migration_sql = MIGRATION.read_text(encoding="utf-8")
        with pytest.raises(RuntimeError, match="can CONNECT to another database"):
            _psql(
                name,
                DATABASE,
                _migration_invocation(
                    DATABASE,
                    migration_sql,
                    include_cloudsqladmin_hba_receipt=False,
                ),
            )
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT to_regnamespace('canonical_brain') IS NULL,"
            "(SELECT count(*) FROM pg_auth_members AS membership "
            "JOIN pg_roles AS owner_role ON owner_role.rolname = "
            "'canonical_brain_migration_owner' WHERE membership.roleid = "
            "owner_role.oid OR membership.member = owner_role.oid);",
        ) == ["t", "0"]
        _psql(name, DATABASE, _migration_invocation(DATABASE, migration_sql))
        _psql(name, DATABASE, _migration_invocation(DATABASE, migration_sql))
        assert _psql_fields(
            name,
            DATABASE,
            f"SELECT pg_has_role('{LOGIN}', "
            "'canonical_brain_writer', 'USAGE'), "
            f"pg_has_role('{LOGIN}', "
            "'canonical_brain_writer', 'SET');",
        ) == ["t", "f"]
        mapping = _run(["docker", "port", name, "5432/tcp"]).stdout.strip()
        port = int(mapping.rsplit(":", 1)[1])
        config = WriterDBConfig(
            host="127.0.0.1",
            tls_server_name="localhost",
            port=port,
            database=DATABASE,
            user=LOGIN,
            ca_file=ca_cert,
            credential=CredentialSource(expected_uid=os.getuid(), path=credential),
        )
        policy = _production_policy(config)
        database = CanonicalWriterDB(
            config=config,
            privilege_policy=policy,
            statements=PRODUCTION_STATEMENT_CATALOG,
            _managed_hba_probe=lambda _config: (
                policy.managed_cloudsqladmin_hba_rejection_receipt
            ),
        )
        database.startup_attest()
        backend = PostgresCanonicalWriterBackend(database)
        yield RealWriterStack(
            name=name,
            backend=backend,
            handlers=CanonicalWriterHandlers(
                backend,
                discord_edge_authority=CanonicalWriterDiscordAuthority(
                    capability_private_key=WRITER_CAPABILITY_PRIVATE_KEY,
                    edge_receipt_public_key=(
                        EDGE_RECEIPT_PRIVATE_KEY.public_key()
                    ),
                ),
            ),
            migration_runs=2,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )


def test_release_schema_contract_asset_matches_real_postgresql_18(
    real_writer_stack: RealWriterStack,
) -> None:
    database = real_writer_stack.backend._database
    session = _open_postgres_session(database._config)
    try:
        observed = collect_schema_contract(
            session,
            config=database._config,
            policy=database._policy,
            managed_hba_receipt=(
                database._policy.managed_cloudsqladmin_hba_rejection_receipt
            ),
        )
    finally:
        session.close()

    asset = SchemaContractAsset.from_bytes(SCHEMA_CONTRACT_ASSET.read_bytes())
    assert asset.postgresql_major == 18
    assert asset.base_artifact_sha256 == hashlib.sha256(
        MIGRATION.read_bytes()
    ).hexdigest()
    assert observed.value == asset.contract.value


def _temporary_admin_config(
    writer_config: WriterDBConfig,
    tmp_path: Path,
    *,
    admin: str,
    password: str,
) -> WriterDBConfig:
    credential = tmp_path / (admin + "-password")
    credential.write_text(password + "\n", encoding="utf-8")
    credential.chmod(0o600)
    return WriterDBConfig(
        host=writer_config.host,
        tls_server_name=writer_config.tls_server_name,
        port=writer_config.port,
        database=writer_config.database,
        user=admin,
        ca_file=writer_config.ca_file,
        credential=CredentialSource(
            expected_uid=os.getuid(),
            path=credential,
        ),
        connect_timeout_seconds=writer_config.connect_timeout_seconds,
        io_timeout_seconds=writer_config.io_timeout_seconds,
        application_name="muncho-schema-reconciliation-e2e",
    )


def _schema_reconciliation_role_graph(name: str, admin: str) -> list[str]:
    return _psql_fields(
        name,
        DATABASE,
        "WITH role_graph AS (SELECT membership.roleid, membership.member, "
        "membership.grantor, membership.admin_option, "
        "membership.inherit_option, membership.set_option, "
        "granted.rolname AS granted_name, member.rolname AS member_name, "
        "grantor.rolname AS grantor_name FROM pg_catalog.pg_auth_members AS "
        "membership JOIN pg_catalog.pg_roles AS granted ON granted.oid = "
        "membership.roleid JOIN pg_catalog.pg_roles AS member ON member.oid = "
        "membership.member JOIN pg_catalog.pg_roles AS grantor ON grantor.oid "
        "= membership.grantor WHERE member.rolname IN ('"
        + admin
        + "', 'canonical_brain_migration_owner') OR granted.rolname IN ('"
        + admin
        + "', 'canonical_brain_migration_owner') OR grantor.rolname IN ('"
        + admin
        + "', 'canonical_brain_migration_owner')) SELECT "
        "pg_catalog.count(*)::text, "
        "pg_catalog.count(DISTINCT roleid)::text, "
        "pg_catalog.string_agg(granted_name, ',' ORDER BY granted_name), "
        "pg_catalog.min(member_name), pg_catalog.min(grantor_name), "
        "pg_catalog.bool_and(NOT admin_option), "
        "pg_catalog.bool_and(inherit_option), "
        "pg_catalog.bool_and(NOT set_option) FROM role_graph;",
    )


def _install_cloudsql_api_role_graph(name: str, admin: str) -> None:
    """Model the exact membership rows written by the Cloud SQL Admin API.

    Stock PostgreSQL requires the SQL grantor to hold ADMIN OPTION, while the
    managed API records ``cloudsqladmin`` directly without creating that extra
    membership.  Seed the ordinary grants first, then change only their
    catalog grantor so the production collector sees the real Cloud shape.
    """

    _psql(
        name,
        DATABASE,
        f"GRANT {CANONICAL_WRITER_MIGRATION_OWNER}, {CANONICAL_WRITER_ROLE} "
        f"TO {admin} WITH ADMIN FALSE, INHERIT TRUE, SET FALSE "
        "GRANTED BY postgres;\n"
        "WITH cloud_grantor AS (SELECT oid FROM pg_catalog.pg_roles "
        "WHERE rolname = 'cloudsqladmin'), temporary_login AS (SELECT oid "
        "FROM pg_catalog.pg_roles WHERE rolname = '"
        + admin
        + "'), expected_roles AS (SELECT oid FROM pg_catalog.pg_roles WHERE "
        "rolname IN ('canonical_brain_migration_owner', "
        "'canonical_brain_writer')) UPDATE pg_catalog.pg_auth_members AS "
        "membership SET grantor = cloud_grantor.oid FROM cloud_grantor, "
        "temporary_login WHERE membership.member = temporary_login.oid AND "
        "membership.roleid IN (SELECT oid FROM expected_roles);\n",
    )


def _create_quarantine_truth_anchors(name: str) -> None:
    _psql(
        name,
        DATABASE,
        "CREATE SCHEMA canonical_brain_legacy_quarantine "
        "AUTHORIZATION postgres;\n"
        "CREATE TABLE canonical_brain_legacy_quarantine."
        "canonical_event_log_legacy_v1 ();\n"
        "CREATE TABLE canonical_brain_legacy_quarantine."
        "reconciliation_receipts ();\n",
    )


def _trampoline_semantic_identity(name: str) -> tuple[str, str]:
    fields = _psql_fields(
        name,
        DATABASE,
        "WITH identity AS (SELECT routine.oid::text AS routine_oid, "
        "pg_catalog.jsonb_build_object('pg_proc', "
        "pg_catalog.to_jsonb(routine), 'owner_name', owner.rolname, "
        "'language_name', language.lanname, 'definition', "
        "pg_catalog.pg_get_functiondef(routine.oid), 'comment', "
        "pg_catalog.obj_description(routine.oid, 'pg_proc'), "
        "'security_labels', COALESCE((SELECT pg_catalog.jsonb_agg("
        "pg_catalog.to_jsonb(security_label) ORDER BY security_label.provider, "
        "security_label.objsubid, security_label.label) FROM "
        "pg_catalog.pg_seclabel AS security_label WHERE "
        "security_label.classoid = 'pg_catalog.pg_proc'::pg_catalog.regclass "
        "AND security_label.objoid = routine.oid), '[]'::jsonb), "
        "'dependencies', COALESCE((SELECT pg_catalog.jsonb_agg("
        "pg_catalog.to_jsonb(dependency) ORDER BY dependency.refclassid, "
        "dependency.refobjid, dependency.refobjsubid, dependency.deptype) "
        "FROM pg_catalog.pg_depend AS dependency WHERE dependency.classid = "
        "'pg_catalog.pg_proc'::pg_catalog.regclass AND dependency.objid = "
        "routine.oid AND dependency.objsubid = 0), '[]'::jsonb), "
        "'shared_dependencies', COALESCE((SELECT pg_catalog.jsonb_agg("
        "pg_catalog.to_jsonb(shared_dependency) ORDER BY "
        "shared_dependency.dbid, shared_dependency.refclassid, "
        "shared_dependency.refobjid, shared_dependency.deptype) FROM "
        "pg_catalog.pg_shdepend AS shared_dependency WHERE "
        "shared_dependency.classid = "
        "'pg_catalog.pg_proc'::pg_catalog.regclass AND "
        "shared_dependency.objid = routine.oid AND "
        "shared_dependency.objsubid = 0), '[]'::jsonb)) AS semantic FROM "
        "pg_catalog.pg_proc AS routine JOIN pg_catalog.pg_language AS language "
        "ON language.oid = routine.prolang JOIN pg_catalog.pg_roles AS owner "
        "ON owner.oid = routine.proowner WHERE routine.oid = "
        "pg_catalog.to_regprocedure("
        "'canonical_brain._deterministic_uuid(text)')) SELECT routine_oid, "
        "pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to("
        "semantic::text, 'UTF8')), 'hex') FROM identity;",
    )
    assert len(fields) == 2
    return fields[0], fields[1]


def test_real_postgresql_18_sealed_schema_reconciliation_preserves_truth(
    real_writer_stack: RealWriterStack,
    tmp_path: Path,
) -> None:
    name = real_writer_stack.name
    database = real_writer_stack.backend._database
    writer_config = database._config
    admin = "muncho_canary_admin_" + "f" * 16
    password = secrets.token_hex(32)
    escaped_password = password.replace("'", "''")
    seeded_case = "case:schema-reconciliation:4097"
    target_asset = SchemaContractAsset.from_bytes(
        SCHEMA_CONTRACT_ASSET.read_bytes()
    )
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = schema_reconciliation._build_plan_from_artifact(
        "f" * 40,
        target_asset.contract,
        artifact,
        target_asset_sha256=target_asset.sha256,
    )
    admin_config = _temporary_admin_config(
        writer_config,
        tmp_path,
        admin=admin,
        password=password,
    )
    writer_probe = _open_postgres_session(writer_config)
    try:
        tls_peer_certificate_sha256 = (
            writer_probe.tls_peer_certificate_sha256
        )
    finally:
        writer_probe.close()
    managed_hba_receipt = replace(
        _test_managed_hba_receipt(writer_config),
        server_certificate_sha256=tls_peer_certificate_sha256,
    )
    expected_role_graph = [
        "2",
        "2",
        "canonical_brain_migration_owner,canonical_brain_writer",
        admin,
        "cloudsqladmin",
        "t",
        "t",
        "t",
    ]

    assert _psql_fields(
        name,
        DATABASE,
        "SELECT pg_catalog.count(*)::text FROM public.canonical_event_log;",
    ) == ["0"]
    try:
        _psql(
            name,
            DATABASE,
            f"CREATE ROLE {admin} LOGIN INHERIT NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE NOREPLICATION NOBYPASSRLS "
            f"PASSWORD '{escaped_password}';\n"
            "DROP FUNCTION "
            "canonical_brain._discord_guild_routeback_target_valid(jsonb);\n"
            "INSERT INTO public.canonical_event_log (event_id, schema_version, "
            "event_type, occurred_at, case_id, source, actor, subject, evidence, "
            "decision, status, next_action, safety, payload) SELECT "
            "('00000000-0000-4000-8000-' || "
            "pg_catalog.lpad(source.ordinal::text, 12, '0'))::uuid, "
            "'canonical_event.v1', 'case.note', "
            "'2026-07-16T00:00:00Z'::timestamptz + "
            "source.ordinal * interval '1 microsecond', '"
            + seeded_case
            + "', pg_catalog.jsonb_build_object('system', "
            "'schema-reconciliation-e2e', 'ordinal', source.ordinal), "
            "'{\"type\":\"agent\"}'::jsonb, '{\"type\":\"case\"}'::jsonb, "
            "'[]'::jsonb, '{}'::jsonb, '{\"state\":\"noted\"}'::jsonb, "
            "'{}'::jsonb, '{}'::jsonb, pg_catalog.jsonb_build_object("
            "'summary', 'persistent reconciliation row', 'ordinal', "
            "source.ordinal) FROM pg_catalog.generate_series(1, 4097) AS "
            "source(ordinal);\n",
            secrets_=(password,),
        )
        _create_quarantine_truth_anchors(name)
        _install_cloudsql_api_role_graph(name, admin)
        assert _schema_reconciliation_role_graph(name, admin) == (
            expected_role_graph
        )
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT pg_catalog.to_regprocedure("
            "'canonical_brain._discord_guild_routeback_target_valid(jsonb)') "
            "IS NULL;",
        ) == ["t"]
        trampoline_before = _trampoline_semantic_identity(name)
        reconciliation_database = (
            schema_reconciliation_db.PostgresSchemaReconciliationDatabase(
                plan=plan,
                target=target_asset.contract,
                admin_config=admin_config,
                writer_config=writer_config,
                managed_hba_receipt=managed_hba_receipt,
            )
        )

        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as transaction:
            transaction.lock_canonical_truth()
            old_contract = transaction.observe_contract()
            truth_before = transaction.observe_canonical_truth()
            assert old_contract.value == schema_reconciliation._old_contract_value(
                target_asset.contract
            )
            assert len(truth_before.relation_receipts) == 10
            assert tuple(
                item.relation for item in truth_before.relation_receipts
            ) == schema_reconciliation.CANONICAL_TRUTH_RELATIONS
            assert truth_before.row_count == 4097
            assert truth_before.canonical_data_row_count == 4097
            assert truth_before.relation_receipts[0].row_count == 4097
            assert truth_before.relation_receipts[0].chunk_count == 2
            assert all(
                item.row_count == 0 and item.chunk_count == 0
                for item in truth_before.relation_receipts[1:]
            )
            assert tuple(
                item.anchor
                for item in truth_before.quarantine_anchor_receipts
            ) == schema_reconciliation.CANONICAL_QUARANTINE_ANCHORS
            assert tuple(
                (item.owner, item.kind, item.persistence)
                for item in truth_before.quarantine_anchor_receipts
            ) == (
                ("postgres", "n", ""),
                ("postgres", "r", "p"),
                ("postgres", "r", "p"),
            )
            assert all(
                item.object_oid > 0 and len(item.acl_sha256) == 64
                for item in truth_before.quarantine_anchor_receipts
            )
            transaction.execute_sql(plan.mutation_sql)
            target_contract = transaction.observe_contract()
            truth_after = transaction.observe_canonical_truth()
            assert target_contract.value == target_asset.contract.value
            assert target_contract.helper_catalog_identity == (
                target_asset.contract.helper_catalog_identity
            )
            assert truth_after == truth_before

        assert _trampoline_semantic_identity(name) == trampoline_before
        assert _schema_reconciliation_role_graph(name, admin) == (
            expected_role_graph
        )

        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as replay:
            replay.lock_canonical_truth()
            replay_contract = replay.observe_contract()
            replay_truth = replay.observe_canonical_truth()
            assert replay_contract.value == target_asset.contract.value
            assert replay_truth == truth_before

        assert _trampoline_semantic_identity(name) == trampoline_before
        assert _schema_reconciliation_role_graph(name, admin) == (
            expected_role_graph
        )
        _psql(name, DATABASE, f"DROP ROLE {admin};\n")

        terminal = schema_reconciliation_db.collect_post_delete_terminal_receipt(
            plan=plan,
            target=target_asset.contract,
            temporary_login=admin,
            writer_config=writer_config,
            managed_hba_receipt=managed_hba_receipt,
            pre_delete_canonical_truth=truth_before,
        )
        assert terminal.observed_contract_sha256 == target_asset.contract.sha256
        assert terminal.writer_session_identity_exact is True
        assert terminal.temporary_login_absent is True
        assert terminal.temporary_login_inventory_empty is True
        assert terminal.migration_owner_memberships_absent is True
        assert terminal.writer_authority_exact is True
        assert terminal.writer_ping_verified is True
        assert terminal.writer_ping_request_id == (
            "schema-reconciliation-post-delete-terminal-v1"
        )
        assert terminal.fresh_writer_session_closed is True
        assert terminal.tls_peer_certificate_sha256 == (
            tls_peer_certificate_sha256
        )
        assert terminal.pre_delete_canonical_truth_receipt_sha256 == (
            truth_before.sha256
        )
    finally:
        _psql(
            name,
            DATABASE,
            f"DROP ROLE IF EXISTS {admin};\n"
            + _migration_invocation(
                DATABASE,
                MIGRATION.read_text(encoding="utf-8"),
            )
            + "\nDELETE FROM public.canonical_event_log WHERE case_id = '"
            + seeded_case
            + "';\n"
            "DROP SCHEMA IF EXISTS canonical_brain_legacy_quarantine "
            "CASCADE;\n",
        )


def test_real_postgresql_18_quarantine_projection_binds_object_identity(
    real_writer_stack: RealWriterStack,
    tmp_path: Path,
) -> None:
    name = real_writer_stack.name
    database = real_writer_stack.backend._database
    writer_config = database._config
    admin = "muncho_canary_admin_" + "e" * 16
    password = secrets.token_hex(32)
    escaped_password = password.replace("'", "''")
    target_asset = SchemaContractAsset.from_bytes(
        SCHEMA_CONTRACT_ASSET.read_bytes()
    )
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = schema_reconciliation._build_plan_from_artifact(
        "e" * 40,
        target_asset.contract,
        artifact,
        target_asset_sha256=target_asset.sha256,
    )
    admin_config = _temporary_admin_config(
        writer_config,
        tmp_path,
        admin=admin,
        password=password,
    )
    writer_probe = _open_postgres_session(writer_config)
    try:
        tls_peer_certificate_sha256 = (
            writer_probe.tls_peer_certificate_sha256
        )
    finally:
        writer_probe.close()
    managed_hba_receipt = replace(
        _test_managed_hba_receipt(writer_config),
        server_certificate_sha256=tls_peer_certificate_sha256,
    )
    trampoline_before = _trampoline_semantic_identity(name)

    try:
        _psql(
            name,
            DATABASE,
            f"CREATE ROLE {admin} LOGIN INHERIT NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE NOREPLICATION NOBYPASSRLS "
            f"PASSWORD '{escaped_password}';\n",
            secrets_=(password,),
        )
        _create_quarantine_truth_anchors(name)
        _install_cloudsql_api_role_graph(name, admin)
        assert _schema_reconciliation_role_graph(name, admin) == [
            "2",
            "2",
            "canonical_brain_migration_owner,canonical_brain_writer",
            admin,
            "cloudsqladmin",
            "t",
            "t",
            "t",
        ]
        reconciliation_database = (
            schema_reconciliation_db.PostgresSchemaReconciliationDatabase(
                plan=plan,
                target=target_asset.contract,
                admin_config=admin_config,
                writer_config=writer_config,
                managed_hba_receipt=managed_hba_receipt,
            )
        )

        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as first_observation:
            first_observation.lock_canonical_truth()
            contract_a = first_observation.observe_contract()
            truth_a = first_observation.observe_canonical_truth()
        assert contract_a.value == target_asset.contract.value

        _psql(
            name,
            DATABASE,
            "DROP TABLE canonical_brain_legacy_quarantine."
            "reconciliation_receipts;\n"
            "CREATE TABLE canonical_brain_legacy_quarantine."
            "reconciliation_receipts ();\n",
        )
        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as second_observation:
            second_observation.lock_canonical_truth()
            truth_b = second_observation.observe_canonical_truth()

        assert truth_b.relation_receipts == truth_a.relation_receipts
        assert truth_b.canonical14_sha256 == truth_a.canonical14_sha256
        assert truth_b.quarantine_anchor_receipts[:2] == (
            truth_a.quarantine_anchor_receipts[:2]
        )
        anchor_a = truth_a.quarantine_anchor_receipts[2]
        anchor_b = truth_b.quarantine_anchor_receipts[2]
        assert anchor_b.object_oid != anchor_a.object_oid
        assert replace(anchor_b, object_oid=anchor_a.object_oid) == anchor_a
        assert (
            truth_b.quarantine_anchors_sha256
            != truth_a.quarantine_anchors_sha256
        )
        assert truth_b.sha256 != truth_a.sha256

        _psql(
            name,
            DATABASE,
            "CREATE TABLE canonical_brain_legacy_quarantine."
            "unexpected_data_relation ();\n",
        )
        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as rejected_observation:
            rejected_observation.lock_canonical_truth()
            assert rejected_observation.observe_contract().value == (
                target_asset.contract.value
            )
            with pytest.raises(
                PostgresProtocolError,
                match="schema_reconciliation_quarantine_anchor_invalid",
            ):
                rejected_observation.observe_canonical_truth()

        assert _trampoline_semantic_identity(name) == trampoline_before
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT pg_catalog.count(*)::text FROM public.canonical_event_log;",
        ) == ["0"]
        _psql(
            name,
            DATABASE,
            "DROP TABLE canonical_brain_legacy_quarantine."
            "unexpected_data_relation;\n",
        )
        with reconciliation_database.transaction(
            advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
        ) as final_observation:
            final_observation.lock_canonical_truth()
            assert final_observation.observe_contract().value == (
                target_asset.contract.value
            )
            assert final_observation.observe_canonical_truth() == truth_b
    finally:
        _psql(
            name,
            DATABASE,
            f"DROP ROLE IF EXISTS {admin};\n"
            "DROP SCHEMA IF EXISTS canonical_brain_legacy_quarantine "
            "CASCADE;\n",
        )


def test_stock_postgresql_18_rejects_wrong_cloud_role_graph_cleanly(
    real_writer_stack: RealWriterStack,
) -> None:
    """The exact custom-role path must not be faked with cloudsqlsuperuser."""

    name = real_writer_stack.name
    admin = "muncho_canary_admin_" + "a" * 16
    asset = SchemaContractAsset.from_bytes(SCHEMA_CONTRACT_ASSET.read_bytes())
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = schema_reconciliation._build_plan_from_artifact(
        "a" * 40,
        asset.contract,
        artifact,
        target_asset_sha256=asset.sha256,
    )
    before_truth = _canonical14_identity(name)

    _psql(
        name,
        DATABASE,
        "DROP FUNCTION "
        "canonical_brain._discord_guild_routeback_target_valid(jsonb);\n"
        f"CREATE ROLE {admin} LOGIN INHERIT NOSUPERUSER CREATEDB CREATEROLE "
        "NOREPLICATION NOBYPASSRLS;\n"
        "GRANT cloudsqlsuperuser TO cloudsqladmin "
        "WITH ADMIN TRUE, INHERIT FALSE, SET TRUE;\n"
        f"GRANT cloudsqlsuperuser TO {admin} "
        "WITH ADMIN FALSE, INHERIT TRUE, SET TRUE "
        "GRANTED BY cloudsqladmin;\n",
        )
    try:
        with pytest.raises(RuntimeError):
            _psql_as(
                name,
                DATABASE,
                admin,
                "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;\n"
                + plan.mutation_sql
                + "\nCOMMIT;\n",
            )

        assert _psql_fields(
            name,
            DATABASE,
            "SELECT pg_catalog.to_regprocedure("
            "'canonical_brain._discord_guild_routeback_target_valid(jsonb)') "
            "IS NULL, (SELECT pg_catalog.count(*) FROM "
            "pg_catalog.pg_auth_members AS membership JOIN "
            "pg_catalog.pg_roles AS owner ON owner.oid = membership.roleid "
            "OR owner.oid = membership.member WHERE owner.rolname = "
            "'canonical_brain_migration_owner')::text;",
        ) == ["t", "0"]
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT pg_catalog.count(*)::text, "
            "pg_catalog.bool_and(NOT membership.admin_option), "
            "pg_catalog.bool_and(membership.inherit_option), "
            "pg_catalog.bool_and(membership.set_option), "
            "pg_catalog.min(grantor.rolname) FROM pg_catalog.pg_auth_members "
            "AS membership JOIN pg_catalog.pg_roles AS granted ON granted.oid = "
            "membership.roleid JOIN pg_catalog.pg_roles AS member ON member.oid = "
            "membership.member JOIN pg_catalog.pg_roles AS grantor ON grantor.oid = "
            "membership.grantor WHERE granted.rolname = 'cloudsqlsuperuser' AND "
            f"member.rolname = '{admin}';",
        ) == ["1", "t", "t", "t", "cloudsqladmin"]
        assert _canonical14_identity(name) == before_truth
    finally:
        _psql(
            name,
            DATABASE,
            f"DROP ROLE IF EXISTS {admin};\n"
            "REVOKE cloudsqlsuperuser FROM cloudsqladmin "
            "GRANTED BY postgres;\n"
            + _migration_invocation(
                DATABASE,
                MIGRATION.read_text(encoding="utf-8"),
            ),
        )


def test_schema_reconciliation_rejects_indirect_unexpected_role_before_begin(
    real_writer_stack: RealWriterStack,
    tmp_path: Path,
) -> None:
    """Exact direct API edges cannot hide an inherited third role."""

    name = real_writer_stack.name
    writer_config = real_writer_stack.backend._database._config
    admin = "muncho_canary_admin_" + "b" * 16
    unexpected = "muncho_canary_unexpected_role"
    password = secrets.token_hex(32)
    escaped_password = password.replace("'", "''")
    target_asset = SchemaContractAsset.from_bytes(
        SCHEMA_CONTRACT_ASSET.read_bytes()
    )
    artifact = _load_source_artifacts_for_tests()["base_migration"]
    plan = schema_reconciliation._build_plan_from_artifact(
        "b" * 40,
        target_asset.contract,
        artifact,
        target_asset_sha256=target_asset.sha256,
    )
    admin_config = _temporary_admin_config(
        writer_config,
        tmp_path,
        admin=admin,
        password=password,
    )
    writer_probe = _open_postgres_session(writer_config)
    try:
        tls_peer_certificate_sha256 = (
            writer_probe.tls_peer_certificate_sha256
        )
    finally:
        writer_probe.close()
    managed_hba_receipt = replace(
        _test_managed_hba_receipt(writer_config),
        server_certificate_sha256=tls_peer_certificate_sha256,
    )
    trampoline_before = _trampoline_semantic_identity(name)

    try:
        _psql(
            name,
            DATABASE,
            f"CREATE ROLE {admin} LOGIN INHERIT NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE NOREPLICATION NOBYPASSRLS "
            f"PASSWORD '{escaped_password}';\n"
            f"CREATE ROLE {unexpected} NOLOGIN INHERIT NOSUPERUSER "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n",
            secrets_=(password,),
        )
        _install_cloudsql_api_role_graph(name, admin)
        _psql(
            name,
            DATABASE,
            f"GRANT {unexpected} TO canonical_brain_writer "
            "WITH ADMIN FALSE, INHERIT TRUE, SET FALSE GRANTED BY postgres;\n",
        )
        assert _schema_reconciliation_role_graph(name, admin) == [
            "2",
            "2",
            "canonical_brain_migration_owner,canonical_brain_writer",
            admin,
            "cloudsqladmin",
            "t",
            "t",
            "t",
        ]
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT pg_catalog.pg_has_role('"
            + admin
            + "', '"
            + unexpected
            + "', 'MEMBER');",
        ) == ["t"]
        reconciliation_database = (
            schema_reconciliation_db.PostgresSchemaReconciliationDatabase(
                plan=plan,
                target=target_asset.contract,
                admin_config=admin_config,
                writer_config=writer_config,
                managed_hba_receipt=managed_hba_receipt,
            )
        )

        with pytest.raises(
            schema_reconciliation.SchemaReconciliationError,
            match=(
                "schema_reconciliation_authority_role_graph_"
                "forward_closure_unexpected"
            ),
        ):
            with reconciliation_database.transaction(
                advisory_lock_key=CANONICAL_WRITER_DEPLOYMENT_LOCK_KEY
            ):
                raise AssertionError("unsafe indirect role reached a transaction")

        assert _trampoline_semantic_identity(name) == trampoline_before
        assert _schema_reconciliation_role_graph(name, admin) == [
            "2",
            "2",
            "canonical_brain_migration_owner,canonical_brain_writer",
            admin,
            "cloudsqladmin",
            "t",
            "t",
            "t",
        ]
    finally:
        _psql(
            name,
            DATABASE,
            f"REVOKE {unexpected} FROM canonical_brain_writer "
            "GRANTED BY postgres;\n"
            f"DROP ROLE IF EXISTS {admin};\n"
            f"DROP ROLE IF EXISTS {unexpected};\n",
        )


def _runtime(
    request_id: str,
    *,
    session: str = SESSION,
    epoch: str = EPOCH,
    thread_id: str = "requester-thread-e2e",
    owner: bool = True,
) -> RuntimeContext:
    return RuntimeContext(
        request_id=request_id,
        platform="discord",
        session_key_sha256=session,
        capability_epoch_sha256=epoch,
        user_id="owner-e2e",
        chat_id=thread_id,
        thread_id=thread_id,
        message_id="message-e2e",
        owner_authenticated=owner,
    )


def _plan(plan_id: str) -> dict[str, object]:
    return {
        "plan_id": plan_id,
        "revision": 1,
        "objective": "Exercise the real PostgreSQL writer boundary",
        "state": "active",
        "success_criteria": [{"id": "verified", "content": "Real DB checks pass"}],
        "steps": [{
            "id": "execute",
            "content": "Run production routines",
            "status": "in_progress",
            "depends_on": [],
        }],
        "current_step_id": "execute",
        "resume_cursor": {"next_step_id": "execute", "summary": "Continue E2E"},
    }


def _cancelled_plan(plan_id: str) -> dict[str, object]:
    plan = _plan(plan_id)
    plan.update({
        "revision": 2,
        "state": "cancelled",
        "steps": [{
            "id": "execute",
            "content": "Run production routines",
            "status": "cancelled",
            "depends_on": [],
        }],
        "current_step_id": "",
        "resume_cursor": {
            "next_step_id": "",
            "summary": "Plan was explicitly cancelled",
        },
    })
    return plan


def _dispatch(
    stack: RealWriterStack,
    operation: CanonicalWriterOperation,
    payload: dict[str, object],
    runtime: RuntimeContext,
) -> dict[str, object]:
    response = stack.handlers.dispatch(operation.value, payload, runtime=runtime)
    assert response.get("ok") is True, response
    return response["result"]


def _verified_discord_receipt(edge_request: dict[str, object]) -> dict[str, object]:
    request = parse_request_for_reconciliation(edge_request)
    capability = verify_request_capability_for_reconciliation(
        request,
        WRITER_CAPABILITY_PRIVATE_KEY.public_key(),
    )
    return sign_receipt(
        EDGE_RECEIPT_PRIVATE_KEY,
        request,
        capability,
        outcome=DiscordEdgeReceiptOutcome.VERIFIED,
        discord_object_id=DISCORD_MESSAGE_ID,
        bot_user_id=DISCORD_BOT_ID,
        adapter_accepted=True,
        readback_verified=True,
        readback_content_sha256=request.intent.content_sha256,
        blocker_code=None,
        occurred_at_unix_ms=int(time.time() * 1_000),
    ).to_message()


def _seed_plan(
    stack: RealWriterStack,
    *,
    case_id: str,
    plan_id: str,
    runtime: RuntimeContext,
    key: str,
) -> None:
    _dispatch(
        stack,
        CanonicalWriterOperation.PLAN_TRANSITION,
        {
            "case_id": case_id,
            "summary": "Real PostgreSQL active plan",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {"plan": _plan(plan_id)},
            "idempotency_key": key,
        },
        runtime,
    )


def test_migration_rerun_and_all_seventeen_public_production_routines(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("all-ops")
    case_id = "case:real-pg-all-ops"
    plan_id = "plan:real-pg-all-ops"
    seen: set[CanonicalWriterOperation] = set()

    def call(
        operation: CanonicalWriterOperation, payload: dict[str, object], rt=runtime
    ):
        seen.add(operation)
        return _dispatch(stack, operation, payload, rt)

    assert stack.migration_runs == 2
    call(CanonicalWriterOperation.PING, {})
    append_receipt = call(
        CanonicalWriterOperation.EVENT_APPEND_MODEL,
        {
            "event_type": "case.note",
            "case_id": case_id,
            "summary": "Real PostgreSQL model event",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {},
            "idempotency_key": "real-pg:event",
        },
    )
    assert append_receipt["event_type"] == "case.note"
    assert append_receipt["case_id"] == case_id
    assert append_receipt["idempotency_key"] == "real-pg:event"
    assert len(append_receipt["canonical_content_sha256"]) == 64
    assert append_receipt["readback_verified"] is True
    assert append_receipt["inserted"] is True
    retry_receipt = call(
        CanonicalWriterOperation.EVENT_APPEND_MODEL,
        {
            "event_type": "case.note",
            "case_id": case_id,
            "summary": "Real PostgreSQL model event",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {},
            "idempotency_key": "real-pg:event",
        },
    )
    assert retry_receipt["event_id"] == append_receipt["event_id"]
    assert retry_receipt["canonical_content_sha256"] == append_receipt[
        "canonical_content_sha256"
    ]
    assert retry_receipt["idempotency_key"] == "real-pg:event"
    assert retry_receipt["readback_verified"] is True
    assert retry_receipt["deduped"] is True
    call(
        CanonicalWriterOperation.PLAN_TRANSITION,
        {
            "case_id": case_id,
            "summary": "Real PostgreSQL plan",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {"plan": _plan(plan_id)},
            "idempotency_key": "real-pg:plan",
        },
    )
    call(
        CanonicalWriterOperation.VERIFICATION_APPEND,
        {
            "case_id": case_id,
            "summary": "Real PostgreSQL verification",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {
                "verification": {
                    "verification_id": "verification:real-pg",
                    "plan_id": plan_id,
                    "plan_revision": 1,
                    "summary": "Bounded real database check passed",
                    "outcome": "passed",
                    "criterion_ids": ["verified"],
                    "receipt": {"kind": "test", "ref": "pytest:real-postgres"},
                }
            },
            "idempotency_key": "real-pg:verification",
        },
    )
    call(
        CanonicalWriterOperation.CASE_QUERY,
        {
            "case_id": case_id,
            "limit": 80,
            "view": "resume_bundle",
        },
    )
    call(
        CanonicalWriterOperation.PLAN_ACTIVE_MATCH,
        {
            "case_id": case_id,
            "plan_id": plan_id,
        },
    )
    call(
        CanonicalWriterOperation.LEASE_SHADOW_RECORD,
        {
            "intent_event_id": "11111111-1111-4111-8111-111111111111",
            "intent_kind": "discord_send",
            "case": {"case_id": case_id},
            "runtime_lease_enforcement": {"blocking_effective": True},
            "enforcement_enabled": True,
            "send_path_blocking_enabled": True,
            "audit_runtime_id": "real-pg-e2e",
            "source_platform": "discord",
            "session_key_ref": "urn:hermes:session:real-pg",
        },
    )
    claim = {
        "case_id": case_id,
        "target_ref": {
            "channel_id": GUILD_CHANNEL,
            "channel_type": "guild_channel",
            "target_type": "guild_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "Real PostgreSQL route-back",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {
            "target_channel_id": GUILD_CHANNEL,
            "content_sha256": CONTENT,
        },
        "idempotency_key": "real-pg:routeback:sent",
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": GUILD_CHANNEL,
            },
            "payload": {"content": ROUTEBACK_CONTENT},
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id=case_id,
                canonical_idempotency_key="real-pg:routeback:sent",
            ),
        },
    }
    claim_result = call(CanonicalWriterOperation.ROUTEBACK_CLAIM, claim)
    edge_request = claim_result["discord_edge_request"]
    edge_receipt = _verified_discord_receipt(edge_request)
    call(
        CanonicalWriterOperation.ROUTEBACK_RECOVER,
        {
            **claim,
            "recovery_kind": "edge_evidence",
            "discord_edge_request": edge_request,
            "discord_edge_receipt": edge_receipt,
        },
    )
    call(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
        {
            **{
                key: claim[key]
                for key in (
                    "case_id",
                    "target_ref",
                    "message_summary",
                    "source_refs",
                    "execution_binding",
                    "idempotency_key",
                )
            },
            "discord_edge_request": edge_request,
            "discord_edge_receipt": edge_receipt,
        },
    )
    call(
        CanonicalWriterOperation.ROUTEBACK_CONTEXT,
        {"thread_id": GUILD_CHANNEL},
        _runtime("routeback-context", thread_id=GUILD_CHANNEL, owner=False),
    )
    call(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
        {
            "preclaim": True,
            "case_id": case_id,
            "target_ref": {"id": "unresolved-guild-target-e2e"},
            "message_summary": "Real PostgreSQL preclaim blocker",
            "source_refs": {"thread_id": runtime.thread_id},
            "blocker_reason": "guild target unresolved",
            "idempotency_key": "real-pg:routeback:blocked",
        },
    )
    call(
        CanonicalWriterOperation.CAPABILITY_GRANT,
        {
            "approval_id": "approval:real-pg",
            "case_id": case_id,
            "plan_id": plan_id,
            "plan_revision": 1,
            "approval_source_sha256": SOURCE,
            "command_hashes": [COMMAND],
            "expires_at": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
            ).isoformat(),
            "max_uses": 2,
        },
    )
    call(
        CanonicalWriterOperation.CAPABILITY_CONSUME,
        {
            "command_sha256": COMMAND,
            "idempotency_key": "real-pg:consume",
        },
    )
    call(
        CanonicalWriterOperation.CAPABILITY_REVOKE,
        {
            "plan_id": plan_id,
            "reason": "real PostgreSQL E2E complete",
        },
    )
    call(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION,
        {
            "reason": "real PostgreSQL session epoch complete",
        },
    )
    call(
        CanonicalWriterOperation.PROJECTION_READ_EVENTS,
        {
            "case_id": case_id,
            "after_event_id": "",
            "limit": 100,
        },
    )
    assert seen == set(CanonicalWriterOperation)


def test_real_postgres_workspace_index_ignores_newer_terminal_case_noise(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    # The preceding all-operations contract deliberately retires the module's
    # default session/epoch.  This query-isolation scenario owns a distinct
    # runtime generation so it cannot inherit that terminal safety fence.
    runtime = _runtime(
        "workspace-index",
        session="e" * 64,
        epoch="f" * 64,
        thread_id="workspace-index-thread",
    )
    active_case = "case:workspace-index-active"
    active_plan = "plan:workspace-index-active"
    _seed_plan(
        stack,
        case_id=active_case,
        plan_id=active_plan,
        runtime=runtime,
        key="workspace-index:active",
    )

    for index in range(12):
        case_id = f"case:workspace-index-terminal-{index}"
        plan_id = f"plan:workspace-index-terminal-{index}"
        _seed_plan(
            stack,
            case_id=case_id,
            plan_id=plan_id,
            runtime=runtime,
            key=f"workspace-index:terminal:{index}:active",
        )
        _dispatch(
            stack,
            CanonicalWriterOperation.PLAN_TRANSITION,
            {
                "case_id": case_id,
                "summary": "Real PostgreSQL terminal plan",
                "source_refs": {"thread_id": runtime.thread_id},
                "payload": {"plan": _cancelled_plan(plan_id)},
                "idempotency_key": f"workspace-index:terminal:{index}:cancelled",
            },
            runtime,
        )

    result = _dispatch(
        stack,
        CanonicalWriterOperation.CASE_QUERY,
        {
            "thread_id": runtime.thread_id,
            "limit": 5,
            "view": "workspace_candidates",
        },
        runtime,
    )

    assert result["candidate_cases_truncated"] is False
    assert result["truncated"] is False
    assert [event["case_id"] for event in result["events"]] == [active_case]
    assert result["events"][0]["payload"]["plan"]["state"] == "active"


def test_real_postgres_projection_exports_internal_provenance_only(
    real_writer_stack: RealWriterStack,
    tmp_path: Path,
) -> None:
    stack = real_writer_stack
    runtime = _runtime(
        "projection-provenance",
        session="8" * 64,
        epoch="9" * 64,
        thread_id="projection-provenance-thread",
    )
    case_id = "case:real-pg-projection-provenance"
    _seed_plan(
        stack,
        case_id=case_id,
        plan_id="plan:real-pg-projection-provenance",
        runtime=runtime,
        key="projection-provenance:plan",
    )
    request = ProjectorReadRequest(
        case_id=case_id,
        after_event_id="",
        limit=100,
    )

    external = stack.backend.projector_read(request, runtime)
    assert set(external) == {
        "events",
        "has_more",
        "next_after_event_id",
        "case_id",
        "bounded",
    }
    assert "provenance" not in external

    with stack.backend.projection_export_scope() as projection:
        internal = projection.projector_read(
            request,
            RuntimeContext(
                request_id="projection-provenance-export",
                platform="writer_service",
                service_internal=True,
            ),
        )

    assert set(internal) == {*set(external), "provenance"}
    assert len(internal["events"]) == len(internal["provenance"]) == 1
    event = internal["events"][0]
    provenance = internal["provenance"][0]
    assert set(provenance) == {
        "event_id",
        "canonical_content_sha256",
        "origin",
        "trusted_runtime",
        "appended_at",
    }
    assert provenance["event_id"] == event["event_id"]
    assert provenance["canonical_content_sha256"] == event["payload"][
        "canonical_content_sha256"
    ]
    assert provenance["trusted_runtime"] == event["source"]["observed_session"]
    assert provenance["origin"] == event["decision"]["decided_by"]
    assert provenance["appended_at"] == event["occurred_at"]

    target = tmp_path / "canonical-events.json"
    count = export_projection_events(
        SimpleNamespace(
            config=SimpleNamespace(
                writer_uid=os.getuid(),
                projector_gid=os.getgid(),
            ),
            backend=stack.backend,
        ),
        target,
    )
    exported_events, exported_provenance = validate_projection_export(
        json.loads(target.read_text(encoding="utf-8")),
        maximum_events=1_000_000,
    )
    assert count == len(exported_events) == len(exported_provenance)
    assert event["event_id"] in {row["event_id"] for row in exported_events}


def test_real_postgres_security_definer_omitted_fields_fail_closed(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    seed_runtime = _runtime(
        "null-boundary-seed",
        session="6" * 64,
        epoch="7" * 64,
        thread_id="null-boundary-thread",
    )
    case_id = "case:real-pg-null-boundary"
    plan_id = "plan:real-pg-null-boundary"
    _seed_plan(
        stack,
        case_id=case_id,
        plan_id=plan_id,
        runtime=seed_runtime,
        key="null-boundary:seed-plan",
    )
    request_only = _jsonb_sql({"request_id": "null-boundary-omitted"})

    assert _routine_error_code(
        stack,
        "writer_ping",
        "NULL::jsonb",
        request_only,
    ) == "invalid_request"
    assert _routine_error_code(
        stack,
        "writer_ping",
        "'null'::jsonb",
        request_only,
    ) == "invalid_request"
    assert _routine_error_code(
        stack,
        "writer_ping",
        "'{}'::jsonb",
        "NULL::jsonb",
    ) == "invalid_request"
    assert _routine_error_code(
        stack,
        "writer_ping",
        "'{}'::jsonb",
        "'null'::jsonb",
    ) == "invalid_request"

    projection_request = {
        "case_id": "",
        "after_event_id": "",
        "limit": 10,
    }
    assert _routine_error_code(
        stack,
        "writer_projection_read_events",
        _jsonb_sql(projection_request),
        request_only,
    ) == "service_internal_required"
    projection_request["case_id"] = case_id
    assert _routine_error_code(
        stack,
        "writer_projection_read_events",
        _jsonb_sql(projection_request),
        request_only,
    ) == "scope_mismatch"

    assert _routine_error_code(
        stack,
        "writer_case_query",
        _jsonb_sql({
            "thread_id": "arbitrary-cross-thread",
            "limit": 10,
            "view": "summary",
        }),
        _jsonb_sql({
            "request_id": "null-boundary-cross-thread",
            "thread_id": "different-runtime-thread",
        }),
    ) == "scope_mismatch"

    lease_request = {
        "intent_event_id": "71111111-1111-4111-8111-111111111111",
        "intent_kind": "discord_send",
        "case": {"case_id": case_id},
        "case_id": case_id,
        "runtime_lease_enforcement": {"blocking_effective": True},
        "enforcement_enabled": True,
        "send_path_blocking_enabled": True,
        "audit_runtime_id": "null-boundary",
        "source_platform": "discord",
        "session_key_ref": "urn:hermes:session:null-boundary",
    }
    assert _routine_error_code(
        stack,
        "writer_lease_shadow_record",
        _jsonb_sql(lease_request),
        request_only,
    ) == "scope_mismatch"

    new_case_id = "case:real-pg-null-new-case"
    assert _routine_error_code(
        stack,
        "writer_event_append_model",
        _jsonb_sql({
            "event_type": "case.note",
            "case_id": new_case_id,
            "summary": "Missing platform must not authorize a new case",
            "source_refs": {"thread_id": "null-new-thread"},
            "actors": {},
            "body": {},
            "safety": {},
            "idempotency_key": "null-boundary:new-case",
        }),
        _jsonb_sql({
            "request_id": "null-boundary-new-case",
            "session_key_sha256": "8" * 64,
            "capability_epoch_sha256": "9" * 64,
            "thread_id": "null-new-thread",
            "chat_id": "null-new-thread",
        }),
    ) == "scope_mismatch"

    approval_id = "approval:null-boundary"
    assert _routine_error_code(
        stack,
        "writer_capability_grant",
        _jsonb_sql({
            "approval_id": approval_id,
            "case_id": case_id,
            "plan_id": plan_id,
            "plan_revision": 1,
            "approval_source_sha256": "a" * 64,
            "command_hashes": ["b" * 64],
            "expires_at": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
            ).isoformat(),
            "max_uses": 1,
        }),
        _jsonb_sql({
            "request_id": "null-boundary-owner",
            "platform": "discord",
            "session_key_sha256": seed_runtime.session_key_sha256,
            "capability_epoch_sha256": seed_runtime.capability_epoch_sha256,
            "user_id": seed_runtime.user_id,
            "chat_id": seed_runtime.chat_id,
            "thread_id": seed_runtime.thread_id,
            "message_id": seed_runtime.message_id,
        }),
    ) == "owner_required"

    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT "
        f"count(*) FILTER (WHERE case_id = '{new_case_id}'),"
        f"count(*) FILTER (WHERE case_id = '{case_id}' "
        "AND event_type = 'lease.shadow.recorded') "
        "FROM public.canonical_event_log;",
    ) == ["0", "0"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM canonical_brain.writer_capability_grants "
        f"WHERE approval_id = '{approval_id}';",
    ) == ["0"]


def test_real_postgres_capability_consume_is_atomic(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("cap-seed", session="1" * 64, epoch="2" * 64)
    case_id = "case:real-pg-capability-race"
    plan_id = "plan:real-pg-capability-race"
    command = "3" * 64
    _seed_plan(
        stack, case_id=case_id, plan_id=plan_id, runtime=runtime, key="cap-race:plan"
    )
    _dispatch(
        stack,
        CanonicalWriterOperation.CAPABILITY_GRANT,
        {
            "approval_id": "approval:cap-race",
            "case_id": case_id,
            "plan_id": plan_id,
            "plan_revision": 1,
            "approval_source_sha256": "4" * 64,
            "command_hashes": [command],
            "expires_at": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
            ).isoformat(),
            "max_uses": 1,
        },
        runtime,
    )
    barrier = threading.Barrier(8)

    def consume(index: int) -> dict[str, object]:
        barrier.wait(timeout=10)
        return stack.handlers.dispatch(
            CanonicalWriterOperation.CAPABILITY_CONSUME.value,
            {"command_sha256": command, "idempotency_key": f"cap-race:{index}"},
            runtime=_runtime(f"cap-consume-{index}", session="1" * 64, epoch="2" * 64),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume, range(8)))
    assert sum(result.get("ok") is True for result in results) == 1, results
    assert {result["error"]["code"] for result in results if not result.get("ok")} == {
        "capability_exhausted"
    }


def test_real_postgres_progress_revision_preserves_only_approved_command_hash(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("cap-progress", session="3" * 64, epoch="4" * 64)
    case_id = "case:real-pg-capability-progress"
    plan_id = "plan:real-pg-capability-progress"
    approved_command = "c" * 64
    _seed_plan(
        stack,
        case_id=case_id,
        plan_id=plan_id,
        runtime=runtime,
        key="cap-progress:plan:r1",
    )
    _dispatch(
        stack,
        CanonicalWriterOperation.CAPABILITY_GRANT,
        {
            "approval_id": "approval:cap-progress",
            "case_id": case_id,
            "plan_id": plan_id,
            "plan_revision": 1,
            "approval_source_sha256": "7" * 64,
            "command_hashes": [approved_command],
            "expires_at": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
            ).isoformat(),
            "max_uses": 1,
        },
        runtime,
    )
    progress_plan = _plan(plan_id)
    progress_plan["revision"] = 2
    _dispatch(
        stack,
        CanonicalWriterOperation.PLAN_TRANSITION,
        {
            "case_id": case_id,
            "summary": "Checkpoint progress without changing approved commands",
            "source_refs": {"thread_id": runtime.thread_id},
            "payload": {"plan": progress_plan},
            "idempotency_key": "cap-progress:plan:r2",
        },
        runtime,
    )

    unapproved = stack.handlers.dispatch(
        CanonicalWriterOperation.CAPABILITY_CONSUME.value,
        {
            "command_sha256": "e" * 64,
            "idempotency_key": "cap-progress:unapproved",
        },
        runtime=runtime,
    )
    consumed = _dispatch(
        stack,
        CanonicalWriterOperation.CAPABILITY_CONSUME,
        {
            "command_sha256": approved_command,
            "idempotency_key": "cap-progress:approved",
        },
        runtime,
    )

    assert unapproved["error"]["code"] == "capability_missing"
    assert consumed["plan_revision"] == 1
    assert consumed["active_plan_revision"] == 2


def test_real_postgres_routeback_claim_has_global_identity_and_exact_pending_scope(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    case_id = "case:real-pg-routeback-race"
    seed_runtime = _runtime("route-seed", session="5" * 64, epoch="6" * 64)
    _dispatch(
        stack,
        CanonicalWriterOperation.EVENT_APPEND_MODEL,
        {
            "event_type": "case.note",
            "case_id": case_id,
            "summary": "Seed global route-back race",
            "source_refs": {"thread_id": seed_runtime.thread_id},
            "payload": {},
            "idempotency_key": "route-race:seed",
        },
        seed_runtime,
    )
    claim = {
        "case_id": case_id,
        "target_ref": {
            "channel_id": GUILD_CHANNEL,
            "channel_type": "guild_channel",
            "target_type": "guild_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "One global route-back claim",
        "source_refs": {"thread_id": seed_runtime.thread_id},
        "execution_binding": {
            "target_channel_id": GUILD_CHANNEL,
            "content_sha256": RACE_CONTENT,
        },
        "idempotency_key": "route-race:claim",
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": GUILD_CHANNEL,
            },
            "payload": {"content": RACE_ROUTEBACK_CONTENT},
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id=case_id,
                canonical_idempotency_key="route-race:claim",
            ),
        },
    }
    barrier = threading.Barrier(6)

    def claim_once(index: int) -> tuple[RuntimeContext, dict[str, object]]:
        runtime = _runtime(
            f"route-claim-{index}",
            session=f"{index + 1:x}" * 64,
            epoch=f"{index + 8:x}" * 64,
            thread_id=f"route-source-{index}",
        )
        barrier.wait(timeout=10)
        return runtime, stack.handlers.dispatch(
            CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
            claim,
            runtime=runtime,
        )

    with ThreadPoolExecutor(max_workers=6) as pool:
        attempts = list(pool.map(claim_once, range(6)))
    responses = [response for _, response in attempts]
    assert sum(response.get("ok") is True for response in responses) == 1, responses
    assert {
        response["error"]["code"]
        for response in responses
        if response.get("ok") is not True
    } == {"scope_mismatch"}
    winner_runtime = next(
        runtime
        for runtime, response in attempts
        if response.get("ok") is True
    )
    winner_result = next(
        response["result"]
        for _, response in attempts
        if response.get("ok") is True
    )
    loser_runtime = next(
        runtime
        for runtime, response in attempts
        if response.get("ok") is not True
    )
    assert winner_result["inserted"] is True
    edge_receipt = _verified_discord_receipt(
        winner_result["discord_edge_request"]
    )
    terminal = {
        **{
            key: claim[key]
            for key in (
                "case_id",
                "target_ref",
                "message_summary",
                "source_refs",
                "execution_binding",
                "idempotency_key",
            )
        },
        "discord_edge_request": winner_result["discord_edge_request"],
        "discord_edge_receipt": edge_receipt,
    }
    wrong = stack.handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        terminal,
        runtime=loser_runtime,
    )
    assert wrong["error"]["code"] == "scope_mismatch"
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM canonical_brain.writer_public_routeback_targets;",
    ) == ["0"]

    legacy_free_retry = stack.handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=winner_runtime,
    )
    assert legacy_free_retry["ok"] is True
    assert legacy_free_retry["result"]["state"] == "authorized"
    assert legacy_free_retry["result"]["deduped"] is True

    restarted_runtime = _runtime(
        "route-restart-recovery",
        session=winner_runtime.session_key_sha256,
        epoch="f" * 64,
        thread_id=winner_runtime.thread_id,
    )
    no_record_recovery = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_RECOVER,
        {
            **claim,
            "recovery_kind": "edge_no_record",
        },
        restarted_runtime,
    )
    assert no_record_recovery["recovered"] is True
    assert no_record_recovery["recovered_epoch_sha256"] == "f" * 64

    evidence_recovery = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_RECOVER,
        {
            **claim,
            "recovery_kind": "edge_evidence",
            "discord_edge_request": winner_result["discord_edge_request"],
            "discord_edge_receipt": edge_receipt,
        },
        restarted_runtime,
    )
    assert evidence_recovery["recovered"] is True
    assert evidence_recovery["recovered_epoch_sha256"] == "f" * 64

    _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
        terminal,
        restarted_runtime,
    )
    terminal_replay = stack.handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
        claim,
        runtime=loser_runtime,
    )
    assert terminal_replay["ok"] is True
    assert terminal_replay["result"]["terminal_event_type"] == (
        "route_back.sent"
    )
    assert "discord_edge_request" not in terminal_replay["result"]


def test_real_postgres_writer_accepts_thread_under_approved_root(
    real_writer_stack: RealWriterStack,
) -> None:
    runtime = _runtime(
        "approved-root-thread-target",
        session="3" * 64,
        epoch="4" * 64,
        thread_id="approved-root-thread-source",
    )
    target_ref = {
        "guild_id": DISCORD_GUILD_ID,
        "target_type": "guild_thread",
        "channel_type": "guild_thread",
        "channel_id": "1599999999999999998",
        "parent_channel_id": GUILD_CHANNEL,
    }

    claimed = real_writer_stack.backend.routeback_authorize(
        RouteBackAuthorizeRequest(
            case_id="case:real-pg-approved-root-thread",
            target_ref=target_ref,
            message_summary="Exact thread below approved backend lane",
            source_refs={"thread_id": runtime.thread_id},
            content_sha256=RACE_CONTENT,
            idempotency_key="real-pg:approved-root-thread",
        ),
        runtime,
    )

    assert claimed["inserted"] is True
    assert claimed["target_ref"] == target_ref


@pytest.mark.parametrize(
    ("label", "alias_body", "target_ref", "accepted"),
    (
        (
            "learned-root",
            {
                "alias": "trusted real postgres root",
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_id": "1527000000000000001",
            },
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_type": "guild_channel",
                "channel_id": "1527000000000000001",
            },
            False,
        ),
        (
            "learned-thread",
            {
                "alias": "trusted real postgres thread",
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_thread",
                "channel_id": "1527000000000000002",
                "parent_channel_id": GUILD_CHANNEL,
            },
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_thread",
                "channel_type": "guild_thread",
                "channel_id": "1527000000000000002",
                "parent_channel_id": GUILD_CHANNEL,
            },
            True,
        ),
    ),
)
def test_real_postgres_alias_learning_never_expands_root_permission_scope(
    real_writer_stack: RealWriterStack,
    label: str,
    alias_body: dict[str, object],
    target_ref: dict[str, object],
    accepted: bool,
) -> None:
    runtime = _runtime(
        f"{label}-event",
        session="1" * 64,
        epoch="2" * 64,
        thread_id=f"{label}-source",
    )
    event_fields = {
        "event_type": "channel.alias.learned",
        "case_id": f"case:real-pg-{label}-alias",
        "summary": "Learn one exact bounded Discord lane alias",
        "source_refs": {"thread_id": runtime.thread_id},
        "idempotency_key": f"real-pg:{label}:alias",
    }
    if accepted:
        _dispatch(
            real_writer_stack,
            CanonicalWriterOperation.EVENT_APPEND_MODEL,
            {**event_fields, "payload": alias_body},
            runtime,
        )
    else:
        # Bypass the Python handler to prove the database permission boundary
        # still rejects a provenanced but out-of-scope alias event.
        real_writer_stack.backend.event_append(
            EventAppendRequest(
                **event_fields,
                actors={},
                body=alias_body,
                safety={},
            ),
            runtime,
        )

    request = RouteBackAuthorizeRequest(
        case_id=f"case:real-pg-{label}-routeback",
        target_ref=target_ref,
        message_summary="Route through bounded learned alias",
        source_refs={"thread_id": runtime.thread_id},
        content_sha256=RACE_CONTENT,
        idempotency_key=f"real-pg:{label}:routeback",
    )
    if not accepted:
        with pytest.raises(CanonicalWriterError) as rejected:
            real_writer_stack.backend.routeback_authorize(request, runtime)
        assert rejected.value.code == "invalid_request"
        return
    claimed = real_writer_stack.backend.routeback_authorize(request, runtime)
    assert claimed["inserted"] is True
    assert claimed["target_ref"] == target_ref


def test_real_postgres_writer_accepts_only_exact_synthetic_public_canary_exception(
    real_writer_stack: RealWriterStack,
) -> None:
    runtime = _runtime(
        "synthetic-public-canary-target",
        session="e" * 64,
        epoch="0" * 64,
        thread_id="synthetic-public-canary-source",
    )
    target_ref = {
        "guild_id": DISCORD_GUILD_ID,
        "target_type": "public_guild_channel",
        "channel_type": "public_guild_channel",
        "channel_id": SYNTHETIC_PUBLIC_CANARY_CHANNEL,
    }

    claimed = real_writer_stack.backend.routeback_authorize(
        RouteBackAuthorizeRequest(
            case_id="case:real-pg-synthetic-public-canary",
            target_ref=target_ref,
            message_summary="Exact isolated synthetic canary route-back",
            source_refs={"thread_id": runtime.thread_id},
            content_sha256=RACE_CONTENT,
            idempotency_key="real-pg:synthetic-public-canary",
        ),
        runtime,
    )

    assert claimed["inserted"] is True
    assert claimed["target_ref"] == target_ref


@pytest.mark.parametrize(
    ("label", "target_ref"),
    (
        (
            "wrong-guild",
            {
                "guild_id": "1282725267068157973",
                "target_type": "guild_channel",
                "channel_type": "guild_channel",
                "channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "channel-type-mismatch",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_type": "guild_thread",
                "channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "thread-without-parent",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_thread",
                "channel_type": "guild_thread",
                "channel_id": "1504852408227069994",
            },
        ),
        (
            "dm",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "dm",
                "channel_type": "dm",
                "channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "group-dm",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "group_dm",
                "channel_type": "group_dm",
                "channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "private-thread",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "private_thread",
                "channel_type": "private_thread",
                "channel_id": "1504852408227069994",
                "parent_channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "arbitrary-public-guild-channel",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "public_guild_channel",
                "channel_type": "public_guild_channel",
                "channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "public-guild-thread",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "public_guild_thread",
                "channel_type": "public_guild_thread",
                "channel_id": "1526858760100909067",
                "parent_channel_id": SYNTHETIC_PUBLIC_CANARY_CHANNEL,
            },
        ),
        (
            "legacy-public-channel-label",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "public_channel",
                "channel_type": "public_channel",
                "channel_id": SYNTHETIC_PUBLIC_CANARY_CHANNEL,
            },
        ),
        (
            "legacy-public-thread-label",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "public_thread",
                "channel_type": "public_thread",
                "channel_id": "1526858760100909067",
                "parent_channel_id": SYNTHETIC_PUBLIC_CANARY_CHANNEL,
            },
        ),
        (
            "synthetic-canary-with-parent",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "public_guild_channel",
                "channel_type": "public_guild_channel",
                "channel_id": SYNTHETIC_PUBLIC_CANARY_CHANNEL,
                "parent_channel_id": GUILD_CHANNEL,
            },
        ),
        (
            "conflicting-legacy-thread-id",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_type": "guild_channel",
                "channel_id": GUILD_CHANNEL,
                "thread_id": "1504852408227069994",
            },
        ),
        (
            "conflicting-legacy-chat-id",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_type": "guild_channel",
                "channel_id": GUILD_CHANNEL,
                "chat_id": "1504852408227069994",
            },
        ),
        (
            "unknown-numeric-root",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_channel",
                "channel_type": "guild_channel",
                "channel_id": "1599999999999999999",
            },
        ),
        (
            "thread-under-unknown-numeric-parent",
            {
                "guild_id": DISCORD_GUILD_ID,
                "target_type": "guild_thread",
                "channel_type": "guild_thread",
                "channel_id": "1599999999999999998",
                "parent_channel_id": "1599999999999999999",
            },
        ),
    ),
)
def test_real_postgres_writer_rejects_non_adventico_or_unreviewed_target_shape(
    real_writer_stack: RealWriterStack,
    label: str,
    target_ref: dict[str, object],
) -> None:
    with pytest.raises(CanonicalWriterError) as rejected:
        real_writer_stack.backend.routeback_authorize(
            RouteBackAuthorizeRequest(
                case_id=f"case:real-pg-invalid-routeback-{label}",
                target_ref=target_ref,
                message_summary="Invalid route-back target must fail closed",
                source_refs={"thread_id": f"invalid-target-{label}"},
                content_sha256=RACE_CONTENT,
                idempotency_key=f"real-pg:invalid-routeback:{label}",
            ),
            _runtime(f"invalid-target-{label}", thread_id=f"invalid-target-{label}"),
        )

    assert rejected.value.code == "invalid_request"


def test_real_postgres_rejects_false_readback_nonempty_blocked_receipt(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime(
        "routeback-false-readback",
        session="7" * 64,
        epoch="8" * 64,
        thread_id="routeback-false-readback-thread",
    )
    case_id = "case:real-pg-false-readback-rejected"
    key = "real-pg:routeback:false-readback"
    claim = {
        "case_id": case_id,
        "target_ref": {
            "channel_id": GUILD_CHANNEL,
            "channel_type": "guild_channel",
            "target_type": "guild_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "Reject unverified nonempty receipt in SQL",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {
            "target_channel_id": GUILD_CHANNEL,
            "content_sha256": RACE_CONTENT,
        },
        "idempotency_key": key,
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": GUILD_CHANNEL,
            },
            "payload": {"content": RACE_ROUTEBACK_CONTENT},
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id=case_id,
                canonical_idempotency_key=key,
            ),
        },
    }
    claimed = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_CLAIM,
        claim,
        runtime,
    )

    with pytest.raises(CanonicalWriterError) as rejected:
        stack.backend.routeback_terminal(
            RouteBackTerminalRequest(
                authorization_id=str(claimed["authorization_id"]),
                outcome="blocked",
                receipt={
                    "platform": "discord",
                    "adapter_receipt": True,
                    "receipt_readback_verified": False,
                    "message_id": DISCORD_MESSAGE_ID,
                    "channel_id": GUILD_CHANNEL,
                    "content_sha256": RACE_CONTENT,
                },
                blocker_reason="route_back_sent_receipt_persistence_failed",
            ),
            runtime,
        )

    assert rejected.value.code == "invalid_request"
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM canonical_brain.writer_routeback_terminals "
        f"WHERE authorization_id = '{claimed['authorization_id']}';",
    ) == ["0"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        f"WHERE case_id = '{case_id}' AND event_type = 'route_back.blocked';",
    ) == ["0"]


def test_real_postgres_preserves_exact_private_denial_receipt_in_projection(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime(
        "private-denial-receipt",
        session="4" * 64,
        epoch="5" * 64,
        thread_id="private-denial-source-thread",
    )
    case_id = "case:real-pg-private-denial"
    idempotency_key = "discord:private:real-pg:é/☁:\"\\tail"
    denial_sha256 = derive_private_denial_receipt_sha256(
        probe_id=idempotency_key,
    )
    payload = {
        "preclaim": True,
        "case_id": case_id,
        "target_ref": {
            "id": "blocked-target:private",
            "target_kind": "forbidden_or_unresolved_target",
        },
        "message_summary": "Private route-back denied before dispatch",
        "source_refs": {"thread_id": runtime.thread_id},
        "blocker_reason": "discord_dm_target_forbidden",
        "idempotency_key": idempotency_key,
        "private_denial_receipt_sha256": denial_sha256,
        "dispatch_attempted": False,
    }

    terminal = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
        payload,
        runtime,
    )
    projection = _dispatch(
        stack,
        CanonicalWriterOperation.PROJECTION_READ_EVENTS,
        {"case_id": case_id},
        runtime,
    )

    expected_receipt = {
        "private_denial_receipt_sha256": denial_sha256,
        "dispatch_attempted": False,
    }
    assert terminal["receipt"] == expected_receipt
    blocked_event = next(
        event
        for event in projection["events"]
        if event["event_type"] == "route_back.blocked"
    )
    assert blocked_event["payload"]["receipt"] == expected_receipt
    assert blocked_event["payload"]["dispatch_attempted"] is False
    assert blocked_event["payload"]["route_back"]["receipt"] == (
        expected_receipt
    )

    event_count = _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        f"WHERE case_id = '{case_id}' AND event_type = 'route_back.blocked';",
    )
    _psql(
        stack.name,
        DATABASE,
        "UPDATE canonical_brain.writer_routeback_lifecycle_terminals "
        "SET receipt = '{}'::jsonb, request_sha256 = repeat('a', 64) "
        f"WHERE case_id = '{case_id}';",
    )
    legacy_replay = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
        payload,
        runtime,
    )

    assert legacy_replay["legacy_receipt_read_only"] is True
    assert legacy_replay["receipt"] == {}
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT receipt::text FROM "
        "canonical_brain.writer_routeback_lifecycle_terminals "
        f"WHERE case_id = '{case_id}';",
    ) == ["{}"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        f"WHERE case_id = '{case_id}' AND event_type = 'route_back.blocked';",
    ) == event_count

    with pytest.raises(CanonicalWriterError) as rejected:
        stack.backend.routeback_terminal(
            RouteBackTerminalRequest(
                authorization_id="",
                outcome="blocked",
                receipt={
                    "private_denial_receipt_sha256": denial_sha256,
                    "dispatch_attempted": True,
                },
                blocker_reason="discord_dm_target_forbidden",
                preclaim=True,
                case_id="case:real-pg-private-denial-tampered",
                target_ref={"id": "blocked-target:private-tampered"},
                message_summary="Tampered private denial must fail closed",
                source_refs={"thread_id": runtime.thread_id},
                idempotency_key="discord:private:real-pg:tampered",
            ),
            runtime,
        )

    assert rejected.value.code == "invalid_request"

    with pytest.raises(CanonicalWriterError) as substituted:
        stack.backend.routeback_terminal(
            RouteBackTerminalRequest(
                authorization_id="",
                outcome="blocked",
                receipt={
                    "private_denial_receipt_sha256": "e" * 64,
                    "dispatch_attempted": False,
                },
                blocker_reason="discord_dm_target_forbidden",
                preclaim=True,
                case_id="case:real-pg-private-denial-substituted",
                target_ref={"id": "blocked-target:private-substituted"},
                message_summary="Substituted private digest must fail closed",
                source_refs={"thread_id": runtime.thread_id},
                idempotency_key="discord:private:real-pg:substituted",
            ),
            runtime,
        )

    assert substituted.value.code == "invalid_request"
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        "WHERE case_id = 'case:real-pg-private-denial-substituted' "
        "AND event_type = 'route_back.blocked';",
    ) == ["0"]


def test_real_postgres_legacy_sent_receipt_is_read_only_dedupe_only(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime(
        "legacy-sent-receipt",
        session="6" * 64,
        epoch="7" * 64,
        thread_id="legacy-sent-source-thread",
    )
    case_id = "case:real-pg-legacy-sent"
    key = "real-pg:routeback:legacy-sent"
    claim = {
        "case_id": case_id,
        "target_ref": {
            "channel_id": GUILD_CHANNEL,
            "channel_type": "guild_channel",
            "target_type": "guild_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "Legacy sent receipt remains immutable",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {
            "target_channel_id": GUILD_CHANNEL,
            "content_sha256": CONTENT,
        },
        "idempotency_key": key,
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": GUILD_CHANNEL,
            },
            "payload": {"content": ROUTEBACK_CONTENT},
            "idempotency_key": derive_routeback_edge_idempotency_key(
                case_id=case_id,
                canonical_idempotency_key=key,
            ),
        },
    }
    claimed = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_CLAIM,
        claim,
        runtime,
    )
    edge_request = claimed["discord_edge_request"]
    edge_receipt = _verified_discord_receipt(edge_request)
    evidence = stack.handlers.discord_edge_authority.verify_routeback_evidence(
        request_value=edge_request,
        receipt_value=edge_receipt,
        authorization_id=str(claimed["authorization_id"]),
    )
    legacy_receipt = dict(evidence.canonical_receipt)
    legacy_receipt.pop("public_receipt_sha256")

    with pytest.raises(CanonicalWriterError) as missing_digest:
        stack.backend.routeback_terminal(
            RouteBackTerminalRequest(
                authorization_id=str(claimed["authorization_id"]),
                outcome="sent",
                receipt=legacy_receipt,
                blocker_reason="",
            ),
            runtime,
        )
    assert missing_digest.value.code == "invalid_receipt"

    terminal_payload = {
        key_name: claim[key_name]
        for key_name in (
            "case_id",
            "target_ref",
            "message_summary",
            "source_refs",
            "execution_binding",
            "idempotency_key",
        )
    } | {
        "discord_edge_request": edge_request,
        "discord_edge_receipt": edge_receipt,
    }
    current = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
        terminal_payload,
        runtime,
    )
    assert "public_receipt_sha256" in current["receipt"]
    event_count = _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        f"WHERE case_id = '{case_id}' AND event_type = 'route_back.sent';",
    )
    _psql(
        stack.name,
        DATABASE,
        "UPDATE canonical_brain.writer_routeback_terminals "
        "SET receipt = receipt - 'public_receipt_sha256', "
        "request_sha256 = repeat('f', 64) "
        f"WHERE authorization_id = '{claimed['authorization_id']}';",
    )

    replay = _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
        terminal_payload,
        runtime,
    )

    assert replay["legacy_receipt_read_only"] is True
    assert replay["receipt"] == legacy_receipt
    assert "public_receipt_sha256" not in replay["receipt"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        f"WHERE case_id = '{case_id}' AND event_type = 'route_back.sent';",
    ) == event_count


def test_real_postgres_session_epoch_revoke_linearizes_with_append(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("epoch-race", session="8" * 64, epoch="9" * 64)
    barrier = threading.Barrier(2)

    def append() -> dict[str, object]:
        barrier.wait(timeout=10)
        return stack.handlers.dispatch(
            CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
            {
                "event_type": "case.note", "case_id": "case:real-pg-epoch-race",
                "summary": "Race append with epoch retirement",
                "source_refs": {"thread_id": runtime.thread_id}, "payload": {},
                "idempotency_key": "epoch-race:first",
            },
            runtime=runtime,
        )

    def revoke() -> dict[str, object]:
        barrier.wait(timeout=10)
        return stack.handlers.dispatch(
            CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION.value,
            {"reason": "rotate real PostgreSQL epoch"},
            runtime=runtime,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        append_future = pool.submit(append)
        revoke_future = pool.submit(revoke)
        first_append = append_future.result(timeout=30)
        retired = revoke_future.result(timeout=30)
    assert retired.get("ok") is True, retired
    if first_append.get("ok") is not True:
        assert first_append["error"]["code"] == "session_epoch_retired"
    retry = stack.handlers.dispatch(
        CanonicalWriterOperation.EVENT_APPEND_MODEL.value,
        {
            "event_type": "case.note", "case_id": "case:real-pg-epoch-race",
            "summary": "Old epoch retry must remain fenced",
            "source_refs": {"thread_id": runtime.thread_id}, "payload": {},
            "idempotency_key": "epoch-race:retry",
        },
        runtime=runtime,
    )
    assert retry["error"]["code"] == "session_epoch_retired"


def test_real_postgres_legacy_reconciliation_is_atomic_and_quarantined() -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is required")

    name = f"canonical-writer-reconcile-{uuid.uuid4().hex[:10]}"
    database = "muncho_canary_brain"
    approval_sha = "b" * 64
    reconciliation_sql = LEGACY_RECONCILIATION.read_text(encoding="utf-8")
    migration_sql = MIGRATION.read_text(encoding="utf-8")
    _run([
        "docker",
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-e",
        "POSTGRES_PASSWORD=canonical-reconcile-test",
        IMAGE,
    ])
    try:
        _wait_ready(name)
        _psql(
            name,
            "postgres",
            "REVOKE ALL ON DATABASE postgres FROM PUBLIC;\n"
            "REVOKE ALL ON DATABASE template1 FROM PUBLIC;\n"
            f"CREATE ROLE {CANONICAL_WRITER_MIGRATION_OWNER} NOLOGIN NOINHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_WRITER_ROLE} NOLOGIN "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE DATABASE {database};\n"
            f"REVOKE ALL ON DATABASE {database} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {database} TO {CANONICAL_WRITER_ROLE};\n",
        )
        _psql(
            name,
            database,
            "REVOKE ALL ON SCHEMA public FROM PUBLIC;\n"
            f"GRANT USAGE ON SCHEMA public TO {CANONICAL_WRITER_MIGRATION_OWNER};\n"
            "CREATE TABLE public.canonical_event_log (\n"
            " event_id uuid PRIMARY KEY, schema_version text NOT NULL,\n"
            " event_type text NOT NULL, occurred_at timestamptz NOT NULL,\n"
            " case_id text NOT NULL, source jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " actor jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " subject jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " evidence jsonb NOT NULL DEFAULT '[]'::jsonb,\n"
            " decision jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " status jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " next_action jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " safety jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " payload jsonb NOT NULL DEFAULT '{}'::jsonb,\n"
            " inserted_at timestamptz NOT NULL DEFAULT now(),\n"
            " idempotency_key text, source_spool text,\n"
            " spool_line_number integer, raw_event_sha256 text\n"
            ");\n"
            "CREATE INDEX idx_canonical_event_log_case_id "
            "ON public.canonical_event_log(case_id);\n"
            "CREATE INDEX idx_canonical_event_log_event_type "
            "ON public.canonical_event_log(event_type);\n"
            "CREATE INDEX idx_canonical_event_log_occurred_at "
            "ON public.canonical_event_log(occurred_at);\n"
            "CREATE UNIQUE INDEX idx_canonical_event_log_idempotency_key "
            "ON public.canonical_event_log(idempotency_key) "
            "WHERE idempotency_key IS NOT NULL;\n"
            "INSERT INTO public.canonical_event_log (\n"
            " event_id,schema_version,event_type,occurred_at,case_id,source,actor,\n"
            " subject,evidence,decision,status,next_action,safety,payload,\n"
            " inserted_at,idempotency_key,source_spool,spool_line_number,raw_event_sha256\n"
            ") VALUES\n"
            " ('11111111-1111-4111-8111-111111111111','canonical_event.v1',\n"
            "  'case.note','2026-07-12T10:00:00Z','case:legacy:1',\n"
            '  \'{"system":"legacy"}\',\'{"type":"agent"}\',\n'
            "  '{\"type\":\"case\"}','[]','{}','{\"state\":\"noted\"}',\n"
            "  '{}','{}','{\"summary\":\"one\"}',\n"
            "  '2026-07-12T10:00:01Z','legacy:1','/legacy/spool',1,repeat('a',64)),\n"
            " ('22222222-2222-4222-8222-222222222222','canonical_event.v1',\n"
            "  'task.plan.updated','2026-07-12T10:01:00Z','case:legacy:2',\n"
            "  '{}','{}','{}','[]','{}','{}','{}','{}',\n"
            "  '{\"summary\":\"two\"}','2026-07-12T10:01:01Z',\n"
            "  NULL,NULL,NULL,NULL);\n",
        )

        fields = _psql_fields(
            name,
            database,
            "SET TimeZone='UTC';\n"
            "WITH row_receipts AS (\n"
            " SELECT event_id,\n"
            " encode(sha256(convert_to(jsonb_build_object(\n"
            "  'event_id',to_jsonb(e)->'event_id',\n"
            "  'schema_version',to_jsonb(e)->'schema_version',\n"
            "  'event_type',to_jsonb(e)->'event_type',\n"
            "  'occurred_at',to_jsonb(e)->'occurred_at',\n"
            "  'case_id',to_jsonb(e)->'case_id','source',to_jsonb(e)->'source',\n"
            "  'actor',to_jsonb(e)->'actor','subject',to_jsonb(e)->'subject',\n"
            "  'evidence',to_jsonb(e)->'evidence',\n"
            "  'decision',to_jsonb(e)->'decision','status',to_jsonb(e)->'status',\n"
            "  'next_action',to_jsonb(e)->'next_action',\n"
            "  'safety',to_jsonb(e)->'safety','payload',to_jsonb(e)->'payload'\n"
            " )::text,'UTF8')),'hex') AS h14,\n"
            " encode(sha256(convert_to(to_jsonb(e)::text,'UTF8')),'hex') AS h19\n"
            " FROM public.canonical_event_log AS e\n"
            ") SELECT count(*),\n"
            " encode(sha256(convert_to(\n"
            "  'canonical-writer-legacy-reconcile-v1:canonical14'||E'\\n'||\n"
            "  string_agg(event_id::text||':'||h14,E'\\n' ORDER BY event_id),\n"
            "  'UTF8')),'hex'),\n"
            " encode(sha256(convert_to(\n"
            "  'canonical-writer-legacy-reconcile-v1:extended19'||E'\\n'||\n"
            "  string_agg(event_id::text||':'||h19,E'\\n' ORDER BY event_id),\n"
            "  'UTF8')),'hex'), max(occurred_at)\n"
            " FROM row_receipts JOIN public.canonical_event_log USING(event_id);\n",
        )
        assert len(fields) == 4
        row_count, canonical_hash, extended_hash, cutoff = fields
        assert row_count == "2"

        def invocation(canonical_expectation: str = canonical_hash) -> str:
            return (
                "SET muncho.canonical_writer_reconcile_scope = "
                "'isolated_canary_copy';\n"
                f"SET muncho.canonical_writer_reconcile_database = '{database}';\n"
                "SET muncho.canonical_writer_reconcile_server_identity_sha256 = "
                f"'{('c' * 64)}';\n"
                "SET muncho.canonical_writer_reconcile_source_owner = 'postgres';\n"
                f"SET muncho.canonical_writer_reconcile_expected_row_count = '{row_count}';\n"
                "SET muncho.canonical_writer_reconcile_expected_canonical14_sha256 = "
                f"'{canonical_expectation}';\n"
                "SET muncho.canonical_writer_reconcile_expected_extended19_sha256 = "
                f"'{extended_hash}';\n"
                "SET muncho.canonical_writer_reconcile_expected_occurred_at_cutoff = "
                f"'{cutoff}';\n"
                "SET muncho.canonical_writer_reconcile_approval_receipt_sha256 = "
                f"'{approval_sha}';\n" + reconciliation_sql
            )

        with pytest.raises(RuntimeError, match="count/hash/cutoff drifted"):
            _psql(name, database, invocation("0" * 64))
        assert _psql_fields(
            name,
            database,
            "SELECT count(*), "
            "to_regclass('canonical_brain_legacy_quarantine."
            "canonical_event_log_legacy_v1') IS NULL "
            "FROM public.canonical_event_log;",
        ) == ["2", "t"]

        _psql(name, database, invocation())
        _psql(name, database, invocation())
        assert _psql_fields(
            name,
            database,
            "SELECT "
            "(SELECT count(*) FROM public.canonical_event_log),"
            "(SELECT count(*) FROM canonical_brain_legacy_quarantine."
            "canonical_event_log_legacy_v1),"
            "(SELECT count(*) FROM canonical_brain_legacy_quarantine."
            "reconciliation_receipts),"
            "(SELECT count(*) FROM pg_attribute WHERE attrelid = "
            "'public.canonical_event_log'::regclass AND attnum > 0 AND NOT attisdropped),"
            "(SELECT count(*) FROM pg_attribute WHERE attrelid = "
            "'canonical_brain_legacy_quarantine.canonical_event_log_legacy_v1'::regclass "
            "AND attnum > 0 AND NOT attisdropped),"
            "(SELECT canonical14_sha256 FROM canonical_brain_legacy_quarantine."
            "reconciliation_receipts),"
            "(SELECT extended19_sha256 FROM canonical_brain_legacy_quarantine."
            "reconciliation_receipts),"
            "pg_get_userbyid((SELECT relowner FROM pg_class WHERE oid = "
            "'public.canonical_event_log'::regclass)),"
            "(SELECT count(*) FROM pg_auth_members AS membership "
            "JOIN pg_roles AS owner_role ON owner_role.rolname = "
            "'canonical_brain_migration_owner' WHERE membership.roleid = "
            "owner_role.oid OR membership.member = owner_role.oid),"
            "has_schema_privilege('canonical_brain_migration_owner',"
            "'canonical_brain_legacy_quarantine','USAGE');",
        ) == [
            "2",
            "2",
            "1",
            "14",
            "19",
            canonical_hash,
            extended_hash,
            CANONICAL_WRITER_MIGRATION_OWNER,
            "0",
            "f",
        ]

        _psql(
            name,
            database,
            "CREATE ROLE managed_admin_probe LOGIN CREATEROLE NOSUPERUSER "
            "NOCREATEDB NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT CONNECT ON DATABASE {database} TO managed_admin_probe;\n"
            f"GRANT CREATE ON DATABASE {database} TO managed_admin_probe;\n"
            "GRANT USAGE, CREATE ON SCHEMA public TO managed_admin_probe;\n",
        )
        # Stock PostgreSQL can reproduce the post-transfer REVOKE failure.  It
        # cannot reproduce Cloud SQL's provider-specific ability for a
        # zero-membership non-superuser admin to grant the owner role; that
        # exact capability is verified by the isolated Cloud rollback probe.
        with pytest.raises(RuntimeError, match="permission denied"):
            _psql_as(
                name,
                database,
                "managed_admin_probe",
                "REVOKE ALL ON TABLE public.canonical_event_log FROM PUBLIC;",
            )
        with pytest.raises(RuntimeError, match="must be able to SET ROLE"):
            _psql_as(
                name,
                database,
                "managed_admin_probe",
                "CREATE SCHEMA managed_admin_probe_schema "
                "AUTHORIZATION canonical_brain_migration_owner;",
            )
        _psql_as(
            name,
            database,
            "managed_admin_probe",
            "CREATE TABLE public.managed_admin_owner_probe (id integer);",
        )
        with pytest.raises(RuntimeError, match="must be able to SET ROLE"):
            _psql_as(
                name,
                database,
                "managed_admin_probe",
                "ALTER TABLE public.managed_admin_owner_probe "
                "OWNER TO canonical_brain_migration_owner;",
            )
        _psql(
            name,
            database,
            "GRANT canonical_brain_migration_owner TO managed_admin_probe "
            "WITH ADMIN FALSE, INHERIT FALSE, SET TRUE;\n"
            "GRANT CREATE ON SCHEMA public "
            "TO canonical_brain_migration_owner;",
        )
        _psql_as(
            name,
            database,
            "managed_admin_probe",
            "CREATE SCHEMA managed_admin_probe_schema "
            "AUTHORIZATION canonical_brain_migration_owner;",
        )
        _psql_as(
            name,
            database,
            "managed_admin_probe",
            "ALTER TABLE public.managed_admin_owner_probe "
            "OWNER TO canonical_brain_migration_owner;",
        )
        assert _psql_fields(
            name,
            database,
            "SELECT pg_get_userbyid(nspowner) FROM pg_namespace "
            "WHERE nspname = 'managed_admin_probe_schema';",
        ) == [CANONICAL_WRITER_MIGRATION_OWNER]
        _psql(
            name,
            database,
            "DROP SCHEMA managed_admin_probe_schema;\n"
            "DROP TABLE public.managed_admin_owner_probe;\n"
            "REVOKE CREATE ON SCHEMA public "
            "FROM canonical_brain_migration_owner;\n"
            "REVOKE canonical_brain_migration_owner FROM managed_admin_probe;\n"
            "REVOKE USAGE, CREATE ON SCHEMA public FROM managed_admin_probe;\n"
            f"REVOKE CREATE, CONNECT ON DATABASE {database} "
            "FROM managed_admin_probe;\n"
            "DROP ROLE managed_admin_probe;\n",
        )

        _psql(name, database, _migration_invocation(database, migration_sql))
        _psql(name, database, _migration_invocation(database, migration_sql))
        _psql(name, database, invocation())
        assert _psql_fields(
            name,
            database,
            "WITH answer AS (SELECT canonical_brain.writer_case_query("
            '\'{"case_id":"case:legacy:1","limit":80,'
            '"view":"resume_bundle"}\'::jsonb,'
            '\'{"request_id":"legacy-reconcile-e2e",'
            '"platform":"discord","owner_authenticated":true}\'::jsonb'
            ") AS value) SELECT "
            "(value->'result'->>'event_count'),"
            "((value->'result'->'support_incomplete_reasons') "
            "? 'legacy_events_quarantined'),"
            "(SELECT count(*) FROM canonical_brain.writer_event_provenance),"
            "has_schema_privilege('canonical_brain_writer',"
            "'canonical_brain_legacy_quarantine','USAGE'),"
            "(SELECT count(*) FROM pg_auth_members AS membership "
            "JOIN pg_roles AS owner_role ON owner_role.rolname = "
            "'canonical_brain_migration_owner' WHERE membership.roleid = "
            "owner_role.oid OR membership.member = owner_role.oid),"
            "has_database_privilege('canonical_brain_migration_owner',"
            f"'{database}','TEMP') FROM answer;",
        ) == ["0", "t", "0", "f", "0", "f"]

        _psql(
            name,
            database,
            "INSERT INTO canonical_brain.writer_event_provenance ("
            "event_id,canonical_content_sha256,origin,trusted_runtime,appended_at"
            ") VALUES ('11111111-1111-4111-8111-111111111111',"
            f"'{('c' * 64)}','forged-test','{{}}'::jsonb,clock_timestamp());",
        )
        with pytest.raises(RuntimeError, match="must not be auto-promoted"):
            _psql(name, database, invocation())
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
        )
