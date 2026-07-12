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
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import uuid

import pytest

from gateway.canonical_writer_db import (
    CanonicalWriterDB,
    CredentialSource,
    RoutineIdentity,
    WriterDBConfig,
    WriterPrivilegePolicy,
    _collect_privilege_attestation,
    _open_postgres_session,
    validate_privilege_attestation,
)
from gateway.canonical_writer_handlers import (
    CanonicalWriterHandlers,
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
from gateway.canonical_writer_protocol import CanonicalWriterOperation


pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "sql" / "canonical_writer_v1.sql"
IMAGE = "postgres:18"
DATABASE = "canonical_writer_e2e"
LOGIN = "canonical_writer_login"
PUBLIC_CHANNEL = "public-channel-e2e"

SESSION = "a" * 64
EPOCH = "b" * 64
COMMAND = "c" * 64
SOURCE = "d" * 64
CONTENT = "e" * 64


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
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "extendedKeyUsage=serverAuth\n",
        encoding="utf-8",
    )
    _run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(ca_key), "-out", str(ca_cert), "-days", "1",
            "-subj", "/CN=Hermes Canonical Writer E2E CA",
        ]
    )
    _run(
        [
            "openssl", "req", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(server_key), "-out", str(server_csr),
            "-subj", "/CN=localhost",
        ]
    )
    _run(
        [
            "openssl", "x509", "-req", "-in", str(server_csr),
            "-CA", str(ca_cert), "-CAkey", str(ca_key), "-CAcreateserial",
            "-out", str(server_cert), "-days", "1", "-sha256",
            "-extfile", str(extensions),
        ]
    )
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


def _seed_policy() -> WriterPrivilegePolicy:
    def identity(signature: str, security_definer: bool) -> RoutineIdentity:
        return RoutineIdentity(
            signature=signature,
            owner=CANONICAL_WRITER_MIGRATION_OWNER,
            security_definer=security_definer,
            language="plpgsql",
            configuration=("search_path=pg_catalog, canonical_brain",),
            definition_sha256="0" * 64,
        )

    return WriterPrivilegePolicy(
        schema=CANONICAL_WRITER_SCHEMA,
        executable_routines=EXPECTED_ROUTINE_SIGNATURES,
        routine_identities=tuple(identity(value, True) for value in EXPECTED_ROUTINE_SIGNATURES),
        dependency_routine_identities=tuple(
            identity(value, False) for value in EXPECTED_HELPER_ROUTINE_SIGNATURES
        ),
        role_memberships=(CANONICAL_WRITER_ROLE,),
        private_schema_identity_sha256="0" * 64,
    )


def _production_policy(config: WriterDBConfig) -> WriterPrivilegePolicy:
    seed = _seed_policy()
    session = _open_postgres_session(config)
    try:
        observed = _collect_privilege_attestation(
            session,
            config=config,
            policy=seed,
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
    )
    validate_privilege_attestation(observed, policy, expected_user=LOGIN)
    return policy


@pytest.fixture(scope="module")
def real_writer_stack(tmp_path_factory: pytest.TempPathFactory) -> RealWriterStack:
    if shutil.which("docker") is None or shutil.which("openssl") is None:
        pytest.skip("Docker and OpenSSL are required")
    if subprocess.run(
        ["docker", "info"], capture_output=True, check=False
    ).returncode:
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
        _run(["docker", "pull", IMAGE], timeout=300)
        _run(
            [
                "docker", "run", "-d", "--name", name,
                "-e", "POSTGRES_PASSWORD", "-e", "POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256",
                "-p", "127.0.0.1::5432", IMAGE,
            ],
            env=environment,
            timeout=180,
            secret_values=(admin_password,),
        )
        _wait_ready(name)

        _run(["docker", "cp", str(server_cert), f"{name}:/tmp/cw-server.crt"])
        _run(["docker", "cp", str(server_key), f"{name}:/tmp/cw-server.key"])
        _run(
            [
                "docker", "exec", "-u", "0", name, "sh", "-ec",
                "chown postgres:postgres /tmp/cw-server.crt /tmp/cw-server.key; "
                "chmod 0644 /tmp/cw-server.crt; chmod 0600 /tmp/cw-server.key",
            ]
        )
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
            f"CREATE ROLE {CANONICAL_WRITER_MIGRATION_OWNER} NOLOGIN "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {CANONICAL_WRITER_ROLE} NOLOGIN "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"CREATE ROLE {LOGIN} LOGIN INHERIT PASSWORD '{escaped_password}' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;\n"
            f"GRANT {CANONICAL_WRITER_ROLE} TO {LOGIN};\n"
            f"CREATE DATABASE {DATABASE};\n"
            f"REVOKE ALL ON DATABASE {DATABASE} FROM PUBLIC;\n"
            f"GRANT CONNECT ON DATABASE {DATABASE} TO {CANONICAL_WRITER_ROLE};\n",
            secrets_=(writer_password,),
        )
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
        _psql(name, DATABASE, migration_sql)
        _psql(name, DATABASE, migration_sql)
        _psql(
            name,
            DATABASE,
            "INSERT INTO canonical_brain.writer_public_routeback_targets "
            "(channel_id,target_type,approved_by,approved_at,enabled) VALUES "
            f"('{PUBLIC_CHANNEL}','public_channel','real-pg-e2e',clock_timestamp(),true);\n",
        )

        mapping = _run(["docker", "port", name, "5432/tcp"]).stdout.strip()
        port = int(mapping.rsplit(":", 1)[1])
        config = WriterDBConfig(
            host="localhost",
            port=port,
            database=DATABASE,
            user=LOGIN,
            ca_file=ca_cert,
            credential=CredentialSource(expected_uid=os.getuid(), path=credential),
        )
        database = CanonicalWriterDB(
            config=config,
            privilege_policy=_production_policy(config),
            statements=PRODUCTION_STATEMENT_CATALOG,
        )
        database.startup_attest()
        backend = PostgresCanonicalWriterBackend(database)
        yield RealWriterStack(
            name=name,
            backend=backend,
            handlers=CanonicalWriterHandlers(backend),
            migration_runs=2,
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


def test_migration_twice_and_all_sixteen_production_routines(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("all-ops")
    case_id = "case:real-pg-all-ops"
    plan_id = "plan:real-pg-all-ops"
    seen: set[CanonicalWriterOperation] = set()

    def call(operation: CanonicalWriterOperation, payload: dict[str, object], rt=runtime):
        seen.add(operation)
        return _dispatch(stack, operation, payload, rt)

    assert stack.migration_runs == 2
    call(CanonicalWriterOperation.PING, {})
    call(CanonicalWriterOperation.EVENT_APPEND_MODEL, {
        "event_type": "case.note", "case_id": case_id,
        "summary": "Real PostgreSQL model event",
        "source_refs": {"thread_id": runtime.thread_id}, "payload": {},
        "idempotency_key": "real-pg:event",
    })
    call(CanonicalWriterOperation.PLAN_TRANSITION, {
        "case_id": case_id, "summary": "Real PostgreSQL plan",
        "source_refs": {"thread_id": runtime.thread_id},
        "payload": {"plan": _plan(plan_id)}, "idempotency_key": "real-pg:plan",
    })
    call(CanonicalWriterOperation.VERIFICATION_APPEND, {
        "case_id": case_id, "summary": "Real PostgreSQL verification",
        "source_refs": {"thread_id": runtime.thread_id},
        "payload": {"verification": {
            "verification_id": "verification:real-pg",
            "plan_id": plan_id, "plan_revision": 1,
            "summary": "Bounded real database check passed", "outcome": "passed",
            "criterion_ids": ["verified"],
            "receipt": {"kind": "test", "ref": "pytest:real-postgres"},
        }},
        "idempotency_key": "real-pg:verification",
    })
    call(CanonicalWriterOperation.CASE_QUERY, {
        "case_id": case_id, "limit": 80, "view": "resume_bundle",
    })
    call(CanonicalWriterOperation.PLAN_ACTIVE_MATCH, {
        "case_id": case_id, "plan_id": plan_id,
    })
    call(CanonicalWriterOperation.LEASE_SHADOW_RECORD, {
        "intent_event_id": "11111111-1111-4111-8111-111111111111",
        "intent_kind": "discord_send", "case": {"case_id": case_id},
        "runtime_lease_enforcement": {"blocking_effective": True},
        "enforcement_enabled": True, "send_path_blocking_enabled": True,
        "audit_runtime_id": "real-pg-e2e", "source_platform": "discord",
        "session_key_ref": "urn:hermes:session:real-pg",
    })
    claim = {
        "case_id": case_id,
        "target_ref": {"channel_id": PUBLIC_CHANNEL, "channel_type": "public_channel"},
        "message_summary": "Real PostgreSQL route-back",
        "source_refs": {"thread_id": runtime.thread_id},
        "execution_binding": {"target_channel_id": PUBLIC_CHANNEL, "content_sha256": CONTENT},
        "idempotency_key": "real-pg:routeback:sent",
    }
    call(CanonicalWriterOperation.ROUTEBACK_CLAIM, claim)
    call(CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT, {
        **claim,
        "receipt": {
            "platform": "discord", "adapter_receipt": True,
            "receipt_readback_verified": True, "message_id": "message-routeback-e2e",
            "channel_id": PUBLIC_CHANNEL, "content_sha256": CONTENT,
        },
    })
    call(
        CanonicalWriterOperation.ROUTEBACK_CONTEXT,
        {"thread_id": PUBLIC_CHANNEL},
        _runtime("routeback-context", thread_id=PUBLIC_CHANNEL, owner=False),
    )
    call(CanonicalWriterOperation.ROUTEBACK_FINALIZE_BLOCKED, {
        "preclaim": True, "case_id": case_id,
        "target_ref": {"id": "unresolved-public-target-e2e"},
        "message_summary": "Real PostgreSQL preclaim blocker",
        "source_refs": {"thread_id": runtime.thread_id},
        "blocker_reason": "public target unresolved",
        "idempotency_key": "real-pg:routeback:blocked",
    })
    call(CanonicalWriterOperation.CAPABILITY_GRANT, {
        "approval_id": "approval:real-pg", "case_id": case_id,
        "plan_id": plan_id, "plan_revision": 1,
        "approval_source_sha256": SOURCE, "command_hashes": [COMMAND],
        "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        "max_uses": 2,
    })
    call(CanonicalWriterOperation.CAPABILITY_CONSUME, {
        "command_sha256": COMMAND, "idempotency_key": "real-pg:consume",
    })
    call(CanonicalWriterOperation.CAPABILITY_REVOKE, {
        "plan_id": plan_id, "reason": "real PostgreSQL E2E complete",
    })
    call(CanonicalWriterOperation.CAPABILITY_REVOKE_SESSION, {
        "reason": "real PostgreSQL session epoch complete",
    })
    call(CanonicalWriterOperation.PROJECTION_READ_EVENTS, {
        "case_id": case_id, "after_event_id": "", "limit": 100,
    })

    assert seen == set(CanonicalWriterOperation)


def test_real_postgres_capability_consume_is_atomic(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    runtime = _runtime("cap-seed", session="1" * 64, epoch="2" * 64)
    case_id = "case:real-pg-capability-race"
    plan_id = "plan:real-pg-capability-race"
    command = "3" * 64
    _seed_plan(stack, case_id=case_id, plan_id=plan_id, runtime=runtime, key="cap-race:plan")
    _dispatch(stack, CanonicalWriterOperation.CAPABILITY_GRANT, {
        "approval_id": "approval:cap-race", "case_id": case_id,
        "plan_id": plan_id, "plan_revision": 1,
        "approval_source_sha256": "4" * 64, "command_hashes": [command],
        "expires_at": (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat(),
        "max_uses": 1,
    }, runtime)
    barrier = threading.Barrier(8)

    def consume(index: int) -> dict[str, object]:
        barrier.wait(timeout=10)
        return stack.handlers.dispatch(
            CanonicalWriterOperation.CAPABILITY_CONSUME.value,
            {"command_sha256": command, "idempotency_key": f"cap-race:{index}"},
            runtime=_runtime(
                f"cap-consume-{index}", session="1" * 64, epoch="2" * 64
            ),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume, range(8)))
    assert sum(result.get("ok") is True for result in results) == 1, results
    assert {
        result["error"]["code"] for result in results if not result.get("ok")
    } == {"capability_exhausted"}


def test_real_postgres_routeback_claim_is_global_across_epochs(
    real_writer_stack: RealWriterStack,
) -> None:
    stack = real_writer_stack
    case_id = "case:real-pg-routeback-race"
    seed_runtime = _runtime("route-seed", session="5" * 64, epoch="6" * 64)
    _dispatch(stack, CanonicalWriterOperation.EVENT_APPEND_MODEL, {
        "event_type": "case.note", "case_id": case_id,
        "summary": "Seed global route-back race",
        "source_refs": {"thread_id": seed_runtime.thread_id}, "payload": {},
        "idempotency_key": "route-race:seed",
    }, seed_runtime)
    claim = {
        "case_id": case_id,
        "target_ref": {"channel_id": PUBLIC_CHANNEL, "channel_type": "public_channel"},
        "message_summary": "One global route-back claim",
        "source_refs": {"thread_id": seed_runtime.thread_id},
        "execution_binding": {"target_channel_id": PUBLIC_CHANNEL, "content_sha256": "7" * 64},
        "idempotency_key": "route-race:claim",
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
    assert all(response.get("ok") is True for response in responses), responses
    assert sum(response["result"].get("inserted") is True for response in responses) == 1
    assert len({response["result"]["authorization_id"] for response in responses}) == 1
    winner_runtime = next(
        runtime
        for runtime, response in attempts
        if response["result"].get("inserted") is True
    )
    loser_runtime = next(runtime for runtime, _ in attempts if runtime != winner_runtime)
    terminal = {
        **claim,
        "receipt": {
            "platform": "discord", "adapter_receipt": True,
            "receipt_readback_verified": True, "message_id": "route-race-message",
            "channel_id": PUBLIC_CHANNEL, "content_sha256": "7" * 64,
        },
    }
    wrong = stack.handlers.dispatch(
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT.value,
        terminal,
        runtime=loser_runtime,
    )
    assert wrong["error"]["code"] == "scope_mismatch"
    _dispatch(
        stack,
        CanonicalWriterOperation.ROUTEBACK_FINALIZE_SENT,
        terminal,
        winner_runtime,
    )


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
