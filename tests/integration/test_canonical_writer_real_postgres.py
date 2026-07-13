"""Real PostgreSQL 18 contract tests for the privileged Canonical writer.

These tests are deliberately opt-in (``pytest -m integration``).  They start
one disposable local Docker container, enable TLS, create a dedicated
least-privilege database, apply the production migration twice, and exercise
the production PostgreSQL wire/backend path.  No Cloud service is contacted.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import datetime as dt
import hashlib
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import uuid

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.canonical_canary_bootstrap import (
    CanaryScopeBootstrapRequest,
    CanaryScopePreclaimRetirementRequest,
)
from gateway.canonical_writer_db import (
    CanonicalWriterDB,
    CredentialSource,
    ManagedCloudSQLAdminHBAReceipt,
    PostgresServerError,
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
    RouteBackTerminalRequest,
    RuntimeContext,
)
from gateway.canonical_writer_postgres_backend import (
    CANONICAL_CANARY_BOOTSTRAP_LOGIN,
    CANONICAL_CANARY_BOOTSTRAP_ROLE,
    CANONICAL_WRITER_MIGRATION_OWNER,
    CANONICAL_WRITER_ROLE,
    CANONICAL_WRITER_SCHEMA,
    EXPECTED_HELPER_ROUTINE_SIGNATURES,
    EXPECTED_ROUTINE_SIGNATURES,
    PRODUCTION_STATEMENT_CATALOG,
    PostgresCanaryScopeBootstrapBackend,
    PostgresCanaryScopePreclaimRetirementBackend,
    PostgresCanonicalWriterBackend,
)
from gateway.canonical_writer_protocol import CanonicalWriterOperation
from gateway.discord_edge_protocol import (
    DiscordEdgeReceiptOutcome,
    parse_request_for_reconciliation,
    sign_receipt,
    verify_request_capability_for_reconciliation,
)
from gateway.discord_edge_writer_authority import (
    CanonicalWriterDiscordAuthority,
    derive_routeback_edge_idempotency_key,
)


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "sql" / "canonical_writer_v1.sql"
CANARY_BOOTSTRAP = (
    ROOT / "scripts" / "sql" / "canonical_writer_canary_bootstrap_v1.sql"
)
CANARY_BOOTSTRAP_RETIRE = (
    ROOT / "scripts" / "sql" / "canonical_writer_canary_bootstrap_retire_v1.sql"
)
LEGACY_RECONCILIATION = (
    ROOT / "scripts" / "sql" / "canonical_writer_legacy_reconcile_v1.sql"
)
IMAGE = "postgres:18"
DATABASE = "canonical_writer_e2e"
LOGIN = "canonical_writer_login"
DISCORD_GUILD_ID = "100000000000000001"
PUBLIC_CHANNEL = "100000000000000002"
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
    bootstrap_backend: PostgresCanaryScopeBootstrapBackend
    preclaim_retirement_backend: PostgresCanaryScopePreclaimRetirementBackend
    handlers: CanonicalWriterHandlers
    migration_runs: int
    canary_preapproval: dict[str, object]


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


def _psql_fields(name: str, database: str, sql: str) -> list[str]:
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
            "postgres",
            "-d",
            database,
        ],
        input_text=sql,
        timeout=180,
    )
    return completed.stdout.strip().split("|")


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


def _canary_bootstrap_invocation(
    database: str,
    bootstrap_sql: str,
    request: dict[str, object],
) -> str:
    settings = {
        "database": database,
        "grant_id": request["grant_id"],
        "case_id": request["case_id"],
        "release_sha256": request["release_sha256"],
        "fixture_sha256": request["fixture_sha256"],
        "run_id": request["run_id"],
        "session_key_sha256": request["session_key_sha256"],
        "expires_at": request["expires_at"],
        "approved_by": request["approved_by"],
        "approval_source_sha256": request["approval_source_sha256"],
        "provisioning_receipt_sha256": request[
            "provisioning_receipt_sha256"
        ],
    }
    prefix = "".join(
        "SET muncho.canonical_canary_bootstrap_"
        + name
        + " = '"
        + str(value).replace("'", "''")
        + "';\n"
        for name, value in settings.items()
    )
    return prefix + bootstrap_sql


def _canary_bootstrap_retire_invocation(
    database: str,
    retirement_sql: str,
    request: dict[str, object],
    *,
    plan_sha256: str,
    owner_approval_sha256: str,
    executor_session_identity_sha256: str,
) -> str:
    settings = {
        "database": database,
        "grant_id": request["grant_id"],
        "case_id": request["case_id"],
        "release_sha256": request["release_sha256"],
        "fixture_sha256": request["fixture_sha256"],
        "run_id": request["run_id"],
        "session_key_sha256": request["session_key_sha256"],
        "expires_at": request["expires_at"],
        "approved_by": request["approved_by"],
        "approval_source_sha256": request["approval_source_sha256"],
        "provisioning_receipt_sha256": request[
            "provisioning_receipt_sha256"
        ],
        "plan_sha256": plan_sha256,
        "owner_approval_sha256": owner_approval_sha256,
        "executor_session_identity_sha256": (
            executor_session_identity_sha256
        ),
    }
    prefix = "".join(
        "SET muncho.canonical_canary_bootstrap_"
        + name
        + " = '"
        + str(value).replace("'", "''")
        + "';\n"
        for name, value in settings.items()
    )
    return prefix + retirement_sql


def _canary_bootstrap_request(
    request: dict[str, object],
) -> CanaryScopeBootstrapRequest:
    return CanaryScopeBootstrapRequest(
        grant_id=str(request["grant_id"]),
        case_id=str(request["case_id"]),
        release_sha256=str(request["release_sha256"]),
        fixture_sha256=str(request["fixture_sha256"]),
        run_id=str(request["run_id"]),
        session_key_sha256=str(request["session_key_sha256"]),
        expires_at=dt.datetime.fromisoformat(str(request["expires_at"])),
        approved_by=str(request["approved_by"]),
        approval_source_sha256=str(request["approval_source_sha256"]),
        provisioning_receipt_sha256=str(
            request["provisioning_receipt_sha256"]
        ),
    )


def _canary_preclaim_retirement_request(
    request: dict[str, object],
) -> CanaryScopePreclaimRetirementRequest:
    return CanaryScopePreclaimRetirementRequest(
        grant_id=str(request["grant_id"]),
        case_id=str(request["case_id"]),
        release_sha256=str(request["release_sha256"]),
        fixture_sha256=str(request["fixture_sha256"]),
        run_id=str(request["run_id"]),
        session_key_sha256=str(request["session_key_sha256"]),
        expires_at=dt.datetime.fromisoformat(str(request["expires_at"])),
        approved_by=str(request["approved_by"]),
        approval_source_sha256=str(request["approval_source_sha256"]),
        provisioning_receipt_sha256=str(
            request["provisioning_receipt_sha256"]
        ),
    )


def _isolated_canary_scope(label: str) -> dict[str, object]:
    digest = lambda suffix: hashlib.sha256(
        f"{label}:{suffix}".encode("utf-8")
    ).hexdigest()
    return {
        "grant_id": f"canary-grant:{label}",
        "case_id": f"case:canary-preclaim:{label}",
        "release_sha256": digest("release"),
        "fixture_sha256": digest("fixture"),
        "run_id": f"canary-run:{label}",
        "session_key_sha256": digest("session"),
        "expires_at": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
        ).isoformat(),
        "approved_by": "owner-e2e",
        "approval_source_sha256": digest("approval"),
        "provisioning_receipt_sha256": digest("provisioning"),
    }


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
    bootstrap_password = secrets.token_hex(32)
    credential = directory / "writer-password"
    credential.write_text(writer_password + "\n", encoding="utf-8")
    credential.chmod(0o600)
    bootstrap_credential = directory / "canary-bootstrap-password"
    bootstrap_credential.write_text(bootstrap_password + "\n", encoding="utf-8")
    bootstrap_credential.chmod(0o600)
    name = "hermes-canonical-writer-e2e-" + uuid.uuid4().hex[:12]
    environment = dict(os.environ)
    environment["POSTGRES_PASSWORD"] = admin_password

    try:
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
        escaped_bootstrap_password = bootstrap_password.replace("'", "''")
        _psql(
            name,
            "postgres",
            "REVOKE ALL ON DATABASE postgres FROM PUBLIC;\n"
            "REVOKE ALL ON DATABASE template1 FROM PUBLIC;\n"
            f"CREATE ROLE {CANONICAL_WRITER_MIGRATION_OWNER} NOLOGIN NOINHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_WRITER_ROLE} NOLOGIN "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_CANARY_BOOTSTRAP_ROLE} NOLOGIN NOINHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_CANARY_BOOTSTRAP_LOGIN} LOGIN INHERIT "
            f"PASSWORD '{escaped_bootstrap_password}' NOSUPERUSER NOCREATEDB "
            "NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT {CANONICAL_CANARY_BOOTSTRAP_ROLE} TO "
            f"{CANONICAL_CANARY_BOOTSTRAP_LOGIN} "
            "WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;\n"
            f"CREATE ROLE {LOGIN} LOGIN INHERIT PASSWORD '{escaped_password}' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT {CANONICAL_WRITER_ROLE} TO {LOGIN};\n"
            "CREATE ROLE cloudsqladmin LOGIN SUPERUSER CREATEDB CREATEROLE "
            "REPLICATION BYPASSRLS;\n"
            "CREATE ROLE cloudsqlsuperuser LOGIN NOSUPERUSER CREATEDB CREATEROLE "
            "NOREPLICATION NOBYPASSRLS;\n"
            "CREATE DATABASE cloudsqladmin OWNER cloudsqladmin;\n"
            f"CREATE DATABASE {DATABASE};\n"
            f"REVOKE ALL ON DATABASE {DATABASE} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {DATABASE} TO {CANONICAL_WRITER_ROLE};\n"
            f"GRANT CONNECT ON DATABASE {DATABASE} TO "
            f"{CANONICAL_CANARY_BOOTSTRAP_ROLE};\n",
            secrets_=(writer_password, bootstrap_password),
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
        canary_preapproval: dict[str, object] = {
            "grant_id": "canary-grant:real-pg",
            "case_id": "case:real-pg-canary-scope",
            "release_sha256": "9" * 64,
            "fixture_sha256": "a" * 64,
            "run_id": "canary-run:real-pg",
            "session_key_sha256": "c" * 64,
            "expires_at": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
            ).isoformat(),
            "approved_by": "owner-e2e",
            "approval_source_sha256": "b" * 64,
            "provisioning_receipt_sha256": "8" * 64,
        }
        _psql(
            name,
            DATABASE,
            _canary_bootstrap_invocation(
                DATABASE,
                CANARY_BOOTSTRAP.read_text(encoding="utf-8"),
                canary_preapproval,
            ),
        )
        mapping = _run(["docker", "port", name, "5432/tcp"]).stdout.strip()
        port = int(mapping.rsplit(":", 1)[1])
        bootstrap_config = WriterDBConfig(
            host="127.0.0.1",
            tls_server_name="localhost",
            port=port,
            database=DATABASE,
            user=CANONICAL_CANARY_BOOTSTRAP_LOGIN,
            ca_file=ca_cert,
            credential=CredentialSource(
                expected_uid=os.getuid(),
                path=bootstrap_credential,
            ),
        )
        bootstrap_backend = PostgresCanaryScopeBootstrapBackend(
            config=bootstrap_config,
        )
        bootstrap_result = bootstrap_backend.preapprove(
            _canary_bootstrap_request(canary_preapproval),
            RuntimeContext(
                request_id="real-pg-canary-preapprove",
                platform="writer_service",
                session_key_sha256=str(
                    canary_preapproval["session_key_sha256"]
                ),
                service_internal=True,
            ),
        )
        assert bootstrap_result["bootstrap_acl_revoked"] is True
        with pytest.raises(
            PostgresServerError,
            match="database_error_sqlstate:42501",
        ):
            bootstrap_backend.preapprove(
                _canary_bootstrap_request(canary_preapproval),
                RuntimeContext(
                    request_id="real-pg-canary-preapprove-replay",
                    platform="writer_service",
                    session_key_sha256=str(
                        canary_preapproval["session_key_sha256"]
                    ),
                    service_internal=True,
                ),
            )
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT has_schema_privilege("
            "'canonical_brain_canary_bootstrap','canonical_brain','USAGE'),"
            "has_function_privilege('canonical_brain_canary_bootstrap',"
            "'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',"
            "'EXECUTE'),has_function_privilege('canonical_brain_writer',"
            "'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',"
            "'EXECUTE');",
        ) == ["f", "f", "f"]
        # A base migration rerun remains inert after consumption and cannot
        # recreate the one-shot bootstrap ACL.
        _psql(name, DATABASE, _migration_invocation(DATABASE, migration_sql))
        assert _psql_fields(
            name,
            DATABASE,
            "SELECT has_schema_privilege("
            "'canonical_brain_canary_bootstrap','canonical_brain','USAGE'),"
            "has_function_privilege('canonical_brain_canary_bootstrap',"
            "'canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb)',"
            "'EXECUTE');",
        ) == ["f", "f"]
        _psql(
            name,
            DATABASE,
            "INSERT INTO canonical_brain.writer_public_routeback_targets "
            "(channel_id,target_type,approved_by,approved_at,enabled) VALUES "
            f"('{PUBLIC_CHANNEL}','public_channel','real-pg-e2e',clock_timestamp(),true);\n",
        )

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
        preclaim_retirement_backend = (
            PostgresCanaryScopePreclaimRetirementBackend(config=config)
        )
        yield RealWriterStack(
            name=name,
            backend=backend,
            bootstrap_backend=bootstrap_backend,
            preclaim_retirement_backend=preclaim_retirement_backend,
            handlers=CanonicalWriterHandlers(
                backend,
                discord_edge_authority=CanonicalWriterDiscordAuthority(
                    capability_private_key=WRITER_CAPABILITY_PRIVATE_KEY,
                    edge_receipt_public_key=(
                        EDGE_RECEIPT_PRIVATE_KEY.public_key()
                    ),
                ),
            ),
            migration_runs=3,
            canary_preapproval=canary_preapproval,
        )
    finally:
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            text=True,
            check=False,
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


def _dispatch(
    stack: RealWriterStack,
    operation: CanonicalWriterOperation,
    payload: dict[str, object],
    runtime: RuntimeContext,
) -> dict[str, object]:
    response = stack.handlers.dispatch(operation.value, payload, runtime=runtime)
    assert response.get("ok") is True, response
    return response["result"]


def _provision_and_consume_isolated_canary(
    stack: RealWriterStack,
    request: dict[str, object],
    *,
    request_id: str,
) -> dict[str, object]:
    _psql(
        stack.name,
        DATABASE,
        _canary_bootstrap_invocation(
            DATABASE,
            CANARY_BOOTSTRAP.read_text(encoding="utf-8"),
            request,
        ),
    )
    return dict(
        stack.bootstrap_backend.preapprove(
            _canary_bootstrap_request(request),
            RuntimeContext(
                request_id=request_id,
                platform="writer_service",
                session_key_sha256=str(request["session_key_sha256"]),
                service_internal=True,
            ),
        )
    )


def _retire_isolated_canary(
    stack: RealWriterStack,
    request: dict[str, object],
    *,
    request_id: str,
) -> dict[str, object]:
    return dict(
        stack.preclaim_retirement_backend.retire(
            _canary_preclaim_retirement_request(request),
            RuntimeContext(
                request_id=request_id,
                platform="writer_service",
                session_key_sha256=str(request["session_key_sha256"]),
                service_internal=True,
            ),
        )
    )


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


def test_consumed_canary_bootstrap_is_never_falsely_retired(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request = stack.canary_preapproval
    retirement_sql = CANARY_BOOTSTRAP_RETIRE.read_text(encoding="utf-8")
    plan_sha256 = "1" * 64
    owner_approval_sha256 = "2" * 64
    executor_sha256 = "3" * 64

    result = _psql_fields(
        stack.name,
        DATABASE,
        _canary_bootstrap_retire_invocation(
            DATABASE,
            retirement_sql,
            request,
            plan_sha256=plan_sha256,
            owner_approval_sha256=owner_approval_sha256,
            executor_session_identity_sha256=executor_sha256,
        ),
    )

    assert len(result) == 13
    assert result[0:3] == ["consumed", request["grant_id"], request["case_id"]]
    assert uuid.UUID(result[3])
    assert uuid.UUID(result[4])
    assert result[5:] == [
        "",
        "f",
        plan_sha256,
        owner_approval_sha256,
        executor_sha256,
        "bootstrap_consumed",
        "t",
        "t",
    ]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log AS event "
        "WHERE event.event_type = 'canary.scope.bootstrap_retired' "
        f"AND event.case_id = '{request['case_id']}';",
    ) == ["0"]


def test_failed_canary_provision_reconciles_as_not_authorized_noop(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request: dict[str, object] = {
        "grant_id": "canary-grant:not-authorized-real-pg",
        "case_id": "case:real-pg-canary-not-authorized",
        "release_sha256": "a" * 64,
        "fixture_sha256": "b" * 64,
        "run_id": "canary-run:not-authorized-real-pg",
        "session_key_sha256": "c" * 64,
        "expires_at": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
        ).isoformat(),
        "approved_by": "owner-e2e-not-authorized",
        "approval_source_sha256": "d" * 64,
        "provisioning_receipt_sha256": "e" * 64,
    }
    plan_sha256 = "f" * 64
    owner_approval_sha256 = "0" * 64
    executor_sha256 = "1" * 64
    _psql(
        stack.name,
        DATABASE,
        "GRANT USAGE ON SCHEMA canonical_brain "
        "TO canonical_brain_canary_bootstrap; "
        "GRANT EXECUTE ON FUNCTION "
        "canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb) "
        "TO canonical_brain_canary_bootstrap;",
    )

    result = _psql_fields(
        stack.name,
        DATABASE,
        _canary_bootstrap_retire_invocation(
            DATABASE,
            CANARY_BOOTSTRAP_RETIRE.read_text(encoding="utf-8"),
            request,
            plan_sha256=plan_sha256,
            owner_approval_sha256=owner_approval_sha256,
            executor_session_identity_sha256=executor_sha256,
        ),
    )

    assert result == [
        "not_authorized",
        request["grant_id"],
        request["case_id"],
        "",
        "",
        "",
        "f",
        plan_sha256,
        owner_approval_sha256,
        executor_sha256,
        "provisioning_not_committed",
        "t",
        "t",
    ]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log AS event "
        f"WHERE event.case_id = '{request['case_id']}' "
        "AND event.event_type IN ('canary.scope.bootstrap_authorized', "
        "'canary.scope.bootstrap_consumed', "
        "'canary.scope.bootstrap_retired');",
    ) == ["0"]


def test_unconsumed_canary_bootstrap_retirement_is_exact_and_idempotent(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request: dict[str, object] = {
        "grant_id": "canary-grant:retire-real-pg",
        "case_id": "case:real-pg-canary-retire",
        "release_sha256": "1" * 64,
        "fixture_sha256": "2" * 64,
        "run_id": "canary-run:retire-real-pg",
        "session_key_sha256": "3" * 64,
        "expires_at": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=30)
        ).isoformat(),
        "approved_by": "owner-e2e-retire",
        "approval_source_sha256": "4" * 64,
        "provisioning_receipt_sha256": "5" * 64,
    }
    plan_sha256 = "6" * 64
    owner_approval_sha256 = "7" * 64
    executor_sha256 = "8" * 64
    retirement_sql = CANARY_BOOTSTRAP_RETIRE.read_text(encoding="utf-8")
    retirement_invocation = _canary_bootstrap_retire_invocation(
        DATABASE,
        retirement_sql,
        request,
        plan_sha256=plan_sha256,
        owner_approval_sha256=owner_approval_sha256,
        executor_session_identity_sha256=executor_sha256,
    )
    _psql(
        stack.name,
        DATABASE,
        _canary_bootstrap_invocation(
            DATABASE,
            CANARY_BOOTSTRAP.read_text(encoding="utf-8"),
            request,
        ),
    )

    first = _psql_fields(stack.name, DATABASE, retirement_invocation)

    assert len(first) == 13
    assert first[0:3] == ["retired", request["grant_id"], request["case_id"]]
    authorization_event_id = str(uuid.UUID(first[3]))
    assert first[4] == ""
    retirement_event_id = str(uuid.UUID(first[5]))
    assert first[6:] == [
        "t",
        plan_sha256,
        owner_approval_sha256,
        executor_sha256,
        "activation_failed_before_consumption",
        "t",
        "t",
    ]
    exact_truth = _psql_fields(
        stack.name,
        DATABASE,
        "SELECT event.event_id::text, provenance.origin, "
        "event.payload->>'idempotency_key', retirement->>'grant_id', "
        "retirement->>'case_id', retirement->>'release_sha256', "
        "retirement->>'fixture_sha256', retirement->>'run_id', "
        "retirement->>'session_key_sha256', retirement->>'expires_at', "
        "retirement->>'approved_by', retirement->>'approval_source_sha256', "
        "retirement->>'provisioning_receipt_sha256', "
        "retirement->>'authorization_event_id', retirement->>'plan_sha256', "
        "retirement->>'owner_approval_sha256', "
        "retirement->>'executor_session_identity_sha256', "
        "retirement->>'reason', retirement->>'acl_revoked' "
        "FROM public.canonical_event_log AS event "
        "JOIN canonical_brain.writer_event_provenance AS provenance "
        "ON provenance.event_id = event.event_id "
        "CROSS JOIN LATERAL (SELECT event.payload->"
        "'canary_scope_bootstrap_retirement' AS retirement) AS payload "
        "WHERE event.event_type = 'canary.scope.bootstrap_retired' "
        f"AND event.case_id = '{request['case_id']}';",
    )
    assert exact_truth[:9] == [
        retirement_event_id,
        "canary_scope_bootstrap_retire",
        f"canary-bootstrap-retire:{request['grant_id']}",
        request["grant_id"],
        request["case_id"],
        request["release_sha256"],
        request["fixture_sha256"],
        request["run_id"],
        request["session_key_sha256"],
    ]
    assert dt.datetime.fromisoformat(exact_truth[9]) == dt.datetime.fromisoformat(
        str(request["expires_at"])
    )
    assert exact_truth[10:] == [
        request["approved_by"],
        request["approval_source_sha256"],
        request["provisioning_receipt_sha256"],
        authorization_event_id,
        plan_sha256,
        owner_approval_sha256,
        executor_sha256,
        "activation_failed_before_consumption",
        "true",
    ]

    replay_executor_sha256 = "9" * 64
    second = _psql_fields(
        stack.name,
        DATABASE,
        _canary_bootstrap_retire_invocation(
            DATABASE,
            retirement_sql,
            request,
            plan_sha256=plan_sha256,
            owner_approval_sha256=owner_approval_sha256,
            executor_session_identity_sha256=replay_executor_sha256,
        ),
    )
    assert second[0:6] == [
        "retired",
        request["grant_id"],
        request["case_id"],
        authorization_event_id,
        "",
        retirement_event_id,
    ]
    assert second[6:] == [
        "f",
        plan_sha256,
        owner_approval_sha256,
        replay_executor_sha256,
        "activation_failed_before_consumption",
        "t",
        "t",
    ]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log AS event "
        "WHERE event.event_type = 'canary.scope.bootstrap_retired' "
        f"AND event.case_id = '{request['case_id']}';",
    ) == ["1"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT event.payload->'canary_scope_bootstrap_retirement'->>"
        "'executor_session_identity_sha256' "
        "FROM public.canonical_event_log AS event "
        "WHERE event.event_id = "
        f"'{retirement_event_id}'::uuid;",
    ) == [executor_sha256]

    _psql(
        stack.name,
        DATABASE,
        "GRANT USAGE ON SCHEMA canonical_brain "
        "TO canonical_brain_canary_bootstrap; "
        "GRANT EXECUTE ON FUNCTION "
        "canonical_brain.writer_canary_scope_preapprove(jsonb,jsonb) "
        "TO canonical_brain_canary_bootstrap;",
    )
    try:
        with pytest.raises(CanonicalWriterError) as failure:
            stack.bootstrap_backend.preapprove(
                _canary_bootstrap_request(request),
                RuntimeContext(
                    request_id="real-pg-retired-bootstrap-denial",
                    platform="writer_service",
                    session_key_sha256=str(request["session_key_sha256"]),
                    service_internal=True,
                ),
            )
        assert failure.value.code == "bootstrap_authorization_missing"
    finally:
        cleanup = _psql_fields(stack.name, DATABASE, retirement_invocation)
        assert cleanup[0] == "retired"
        assert cleanup[6] == "f"
        assert cleanup[-2:] == ["t", "t"]


def test_real_postgres_preclaim_retire_first_is_one_durable_tombstone(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request = _isolated_canary_scope("retire-first")
    preapproved = _provision_and_consume_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-bootstrap-retire-first",
    )

    first = _retire_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-retire-first",
    )
    replay = _retire_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-retire-replay",
    )
    rejected_claim = stack.handlers.dispatch(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
        {
            key: request[key]
            for key in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "approval_source_sha256",
            )
        },
        runtime=RuntimeContext(
            request_id="real-pg-preclaim-rejected-claim",
            platform="api_server",
            session_key_sha256=str(request["session_key_sha256"]),
            capability_epoch_sha256="7" * 64,
            chat_id="real-pg-preclaim-retire-first",
            thread_id="real-pg-preclaim-retire-first",
        ),
    )

    assert first["outcome"] == "retired"
    assert first["authority_active"] is False
    assert first["scope_retired"] is True
    assert first["inserted"] is True
    assert first["deduped"] is False
    assert first["preapproval_event_id"] == preapproved["receipt_event_id"]
    assert first["bootstrap_consumption_event_id"] == preapproved[
        "bootstrap_consumption_event_id"
    ]
    assert str(uuid.UUID(str(first["retirement_event_id"]))) == first[
        "retirement_event_id"
    ]
    assert replay["retirement_event_id"] == first["retirement_event_id"]
    assert replay["retired_at"] == first["retired_at"]
    assert replay["inserted"] is False
    assert replay["deduped"] is True
    assert rejected_claim["error"]["code"] == (
        "canary_scope_preapproval_retired"
    )
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        "WHERE event_type = 'canary.scope.preapproval_retired' "
        f"AND case_id = '{request['case_id']}';",
    ) == ["1"]


def test_real_postgres_claim_first_retires_exact_session_epoch_only(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request = _isolated_canary_scope("claim-first")
    _provision_and_consume_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-bootstrap-claim-first",
    )
    epoch = "6" * 64
    claim_runtime = RuntimeContext(
        request_id="real-pg-preclaim-claim-first",
        platform="api_server",
        session_key_sha256=str(request["session_key_sha256"]),
        capability_epoch_sha256=epoch,
        chat_id="real-pg-preclaim-claim-first",
        thread_id="real-pg-preclaim-claim-first",
    )
    claimed = _dispatch(
        stack,
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM,
        {
            key: request[key]
            for key in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "approval_source_sha256",
            )
        },
        claim_runtime,
    )

    first = _retire_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-claimed-retire-first",
    )
    replay = _retire_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-claimed-retire-replay",
    )

    assert first["outcome"] == "claimed"
    assert first["claim_event_id"] == claimed["claim_event_id"]
    assert first["retirement_event_id"] is None
    assert first["scope_retired"] is False
    assert first["authority_active"] is False
    assert first["inserted"] is True
    assert first["deduped"] is False
    assert str(uuid.UUID(str(first["revocation_event_id"]))) == first[
        "revocation_event_id"
    ]
    assert replay["revocation_event_id"] == first["revocation_event_id"]
    assert replay["inserted"] is False
    assert replay["deduped"] is True
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM public.canonical_event_log "
        "WHERE event_type = 'canary.scope.preapproval_retired' "
        f"AND case_id = '{request['case_id']}';",
    ) == ["0"]
    assert _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FROM canonical_brain."
        "writer_capability_revocation_scopes "
        "WHERE scope_type = 'session' "
        f"AND session_key_sha256 = '{request['session_key_sha256']}' "
        f"AND capability_epoch_sha256 = '{epoch}';",
    ) == ["1"]


def test_real_postgres_claim_retire_race_has_one_semantic_winner(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    request = _isolated_canary_scope("claim-retire-race")
    _provision_and_consume_isolated_canary(
        stack,
        request,
        request_id="real-pg-preclaim-bootstrap-race",
    )
    epoch = "5" * 64
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait(timeout=5)
        return stack.handlers.dispatch(
            CanonicalWriterOperation.CANARY_SCOPE_CLAIM.value,
            {
                key: request[key]
                for key in (
                    "grant_id",
                    "case_id",
                    "release_sha256",
                    "fixture_sha256",
                    "run_id",
                    "approval_source_sha256",
                )
            },
            runtime=RuntimeContext(
                request_id="real-pg-preclaim-race-claim",
                platform="api_server",
                session_key_sha256=str(request["session_key_sha256"]),
                capability_epoch_sha256=epoch,
                chat_id="real-pg-preclaim-race",
                thread_id="real-pg-preclaim-race",
            ),
        )

    def retire():
        barrier.wait(timeout=5)
        return _retire_isolated_canary(
            stack,
            request,
            request_id="real-pg-preclaim-race-retire",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future = pool.submit(claim)
        retire_future = pool.submit(retire)
        claim_result = claim_future.result(timeout=20)
        retirement = retire_future.result(timeout=20)

    event_counts = _psql_fields(
        stack.name,
        DATABASE,
        "SELECT count(*) FILTER (WHERE event_type = 'canary.scope.claimed'),"
        "count(*) FILTER (WHERE event_type = "
        "'canary.scope.preapproval_retired') "
        "FROM public.canonical_event_log "
        f"WHERE case_id = '{request['case_id']}';",
    )
    assert event_counts in (["1", "0"], ["0", "1"])
    assert retirement["authority_active"] is False
    if retirement["outcome"] == "claimed":
        assert claim_result["ok"] is True
        assert retirement["revocation_event_id"] is not None
        assert event_counts == ["1", "0"]
    else:
        assert retirement["outcome"] == "retired"
        assert claim_result["error"]["code"] == (
            "canary_scope_preapproval_retired"
        )
        assert event_counts == ["0", "1"]


def test_migration_rerun_and_all_eighteen_public_production_routines(
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

    assert stack.migration_runs == 3
    call(CanonicalWriterOperation.PING, {})
    canary_preapproval = stack.canary_preapproval
    canary_session = str(canary_preapproval["session_key_sha256"])
    canary_epoch = "d" * 64
    canary_case = str(canary_preapproval["case_id"])
    canary_runtime = RuntimeContext(
        request_id="real-pg-canary-claim",
        platform="api_server",
        session_key_sha256=canary_session,
        capability_epoch_sha256=canary_epoch,
        chat_id="real-pg-api-session",
        thread_id="real-pg-api-session",
    )
    canary_claim = call(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM,
        {
            key: canary_preapproval[key]
            for key in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "approval_source_sha256",
            )
        },
        canary_runtime,
    )
    assert canary_claim["authority_active"] is True
    assert canary_claim["expires_at"] is not None
    canary_claim_retry = call(
        CanonicalWriterOperation.CANARY_SCOPE_CLAIM,
        {
            key: canary_preapproval[key]
            for key in (
                "grant_id",
                "case_id",
                "release_sha256",
                "fixture_sha256",
                "run_id",
                "approval_source_sha256",
            )
        },
        RuntimeContext(
            request_id="real-pg-canary-claim-retry",
            platform="api_server",
            session_key_sha256=canary_session,
            capability_epoch_sha256=canary_epoch,
            chat_id="real-pg-api-session",
            thread_id="real-pg-api-session",
        ),
    )
    assert canary_claim_retry["deduped"] is True
    assert canary_claim_retry["claimed_at"] == canary_claim["claimed_at"]
    assert canary_claim_retry["expires_at"] == canary_claim["expires_at"]
    assert canary_claim_retry["claim_event_id"] == canary_claim["claim_event_id"]
    call(
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
            "channel_id": PUBLIC_CHANNEL,
            "channel_type": "public_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "Real PostgreSQL route-back",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {
            "target_channel_id": PUBLIC_CHANNEL,
            "content_sha256": CONTENT,
        },
        "idempotency_key": "real-pg:routeback:sent",
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": PUBLIC_CHANNEL,
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
        {"thread_id": PUBLIC_CHANNEL},
        _runtime("routeback-context", thread_id=PUBLIC_CHANNEL, owner=False),
    )
    call(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED,
        {
            "preclaim": True,
            "case_id": case_id,
            "target_ref": {"id": "unresolved-public-target-e2e"},
            "message_summary": "Real PostgreSQL preclaim blocker",
            "source_refs": {"thread_id": runtime.thread_id},
            "blocker_reason": "public target unresolved",
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
    canary_retirement = call(
        CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION,
        {"reason": "real PostgreSQL canary completed"},
        canary_runtime,
    )
    assert canary_retirement["canary_scopes_revoked"] == 1
    blocked_canary_read = stack.handlers.dispatch(
        CanonicalWriterOperation.CASE_QUERY.value,
        {"case_id": canary_case, "limit": 20},
        runtime=canary_runtime,
    )
    assert blocked_canary_read["error"]["code"] == "scope_mismatch"

    assert seen == set(CanonicalWriterOperation)


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
            "channel_id": PUBLIC_CHANNEL,
            "channel_type": "public_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "One global route-back claim",
        "source_refs": {"thread_id": seed_runtime.thread_id},
        "execution_binding": {
            "target_channel_id": PUBLIC_CHANNEL,
            "content_sha256": RACE_CONTENT,
        },
        "idempotency_key": "route-race:claim",
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": PUBLIC_CHANNEL,
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
    _psql(
        stack.name,
        DATABASE,
        "UPDATE canonical_brain.writer_public_routeback_targets "
        f"SET enabled = false WHERE channel_id = '{PUBLIC_CHANNEL}';",
    )
    try:
        disabled_retry = stack.handlers.dispatch(
            CanonicalWriterOperation.ROUTEBACK_CLAIM.value,
            claim,
            runtime=winner_runtime,
        )
        assert disabled_retry["error"]["code"] == "target_not_approved"

        restarted_runtime = _runtime(
            "route-restart-recovery",
            session=winner_runtime.session_key_sha256,
            epoch="f" * 64,
            thread_id=winner_runtime.thread_id,
        )
        no_record_recovery = stack.handlers.dispatch(
            CanonicalWriterOperation.ROUTEBACK_RECOVER.value,
            {
                **claim,
                "recovery_kind": "edge_no_record",
            },
            runtime=restarted_runtime,
        )
        assert no_record_recovery["error"]["code"] == "target_not_approved"

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
    finally:
        _psql(
            stack.name,
            DATABASE,
            "UPDATE canonical_brain.writer_public_routeback_targets "
            f"SET enabled = true WHERE channel_id = '{PUBLIC_CHANNEL}';",
        )


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
            "channel_id": PUBLIC_CHANNEL,
            "channel_type": "public_channel",
            "guild_id": DISCORD_GUILD_ID,
        },
        "message_summary": "Reject unverified nonempty receipt in SQL",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {
            "target_channel_id": PUBLIC_CHANNEL,
            "content_sha256": RACE_CONTENT,
        },
        "idempotency_key": key,
        "discord_edge_intent": {
            "operation": "public.message.send",
            "target": {
                "target_type": "public_guild_channel",
                "guild_id": DISCORD_GUILD_ID,
                "channel_id": PUBLIC_CHANNEL,
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
                    "channel_id": PUBLIC_CHANNEL,
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
            f"CREATE ROLE {CANONICAL_CANARY_BOOTSTRAP_ROLE} NOLOGIN NOINHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_CANARY_BOOTSTRAP_LOGIN} LOGIN INHERIT "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT {CANONICAL_CANARY_BOOTSTRAP_ROLE} "
            f"TO {CANONICAL_CANARY_BOOTSTRAP_LOGIN} "
            "WITH ADMIN FALSE, INHERIT TRUE, SET TRUE;\n"
            f"CREATE DATABASE {database};\n"
            f"REVOKE ALL ON DATABASE {database} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {database} TO {CANONICAL_WRITER_ROLE};\n"
            f"GRANT CONNECT ON DATABASE {database} "
            f"TO {CANONICAL_CANARY_BOOTSTRAP_ROLE};\n",
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
