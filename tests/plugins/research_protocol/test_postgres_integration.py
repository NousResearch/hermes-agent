"""Disposable PostgreSQL integration gate for PR2.

Set RESEARCH_PROTOCOL_TEST_DATABASE_URL to a dedicated disposable database.
The test is skipped otherwise and never falls back to a developer database.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path

import pytest

from plugins.research_protocol.approval import (
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
)
from plugins.research_protocol.models import BudgetLimits, Capability
from plugins.research_protocol.storage.postgres import (
    ApprovalStoreTimeoutError,
    ContextReadLimitError,
    ContextReadTimeoutError,
    PostgresApprovalStore,
    PostgresContextReader,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_DSN_ENV = "RESEARCH_PROTOCOL_TEST_DATABASE_URL"
SHA_A = "a" * 64
SHA_B = "b" * 64
APPROVAL_SINGLE = "approval-single-00000001"
APPROVAL_WRONG_HASH = "approval-wrong-hash-0001"
APPROVAL_EXPIRED = "approval-expired-0000001"
APPROVAL_DENIED = "approval-denied-00000001"
APPROVAL_CONCURRENT = "approval-concurrent-0001"
APPROVAL_RESTRICTIVE_SCOPE = "approval-restrictive-0001"


def _scope(
    *,
    run_id: str = "run-integration",
    expires_at: datetime = datetime(2099, 1, 1, tzinfo=UTC),
    budget_max_executions: int = 1,
    max_executions: int = 1,
) -> ApprovalScope:
    return ApprovalScope.model_validate({
        "capability": Capability.RESEARCH_COLLECT,
        "run_id": run_id,
        "plan_sha256": SHA_B,
        "input_hashes": ["a" * 64],
        "budgets": BudgetLimits(
            max_duration_seconds=60,
            max_executions=budget_max_executions,
            max_records=10,
            max_bytes=4096,
            max_external_calls=0,
        ),
        "expires_at": expires_at,
        "max_executions": max_executions,
        "external_rights": [],
    })


def _record(
    approval_id: str,
    *,
    scope: ApprovalScope,
    created_at: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    verdict: ApprovalVerdict = ApprovalVerdict.APPROVED,
) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=approval_id,
        scope_sha256=scope.scope_sha256(),
        plan_sha256=scope.plan_sha256,
        scope_json=scope.canonical_json(),
        verdict=verdict,
        surface="integration-test",
        created_at=created_at,
        expires_at=scope.expires_at,
        max_executions=scope.max_executions,
        consumed_count=0,
    )


async def _exercise_database(dsn: str) -> None:
    import asyncpg

    admin = await asyncpg.connect(dsn)
    try:
        for name in ("0001_approval_ledger.sql", "0002_roles.sql"):
            sql = (
                REPO_ROOT / "plugins" / "research_protocol" / "migrations" / name
            ).read_text(encoding="utf-8")
            await admin.execute(sql)
        await admin.execute("TRUNCATE TABLE research_protocol.approvals")
    finally:
        await admin.close()

    collision = await asyncpg.connect(dsn)
    try:
        roles_sql = (
            REPO_ROOT
            / "plugins"
            / "research_protocol"
            / "migrations"
            / "0002_roles.sql"
        ).read_text(encoding="utf-8")
        with pytest.raises(asyncpg.DuplicateObjectError):
            await collision.execute(roles_sql)
    finally:
        await collision.close()

    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=4)
    try:
        store = PostgresApprovalStore(pool)

        valid_scope = _scope()
        await store.create_approval(_record(APPROVAL_SINGLE, scope=valid_scope))
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_SINGLE,
                scope=valid_scope,
                plan_sha256=SHA_B,
            )
            is True
        )
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_SINGLE,
                scope=valid_scope,
                plan_sha256=SHA_B,
            )
            is False
        )

        restrictive_scope = _scope(
            run_id="run-restrictive-scope",
            budget_max_executions=2,
            max_executions=1,
        )
        await store.create_approval(
            _record(APPROVAL_RESTRICTIVE_SCOPE, scope=restrictive_scope)
        )
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_RESTRICTIVE_SCOPE,
                scope=restrictive_scope,
                plan_sha256=SHA_B,
            )
            is True
        )
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_RESTRICTIVE_SCOPE,
                scope=restrictive_scope,
                plan_sha256=SHA_B,
            )
            is False
        )

        await store.create_approval(_record(APPROVAL_WRONG_HASH, scope=valid_scope))
        wrong_scope = _scope(run_id="run-other")
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_WRONG_HASH,
                scope=wrong_scope,
                plan_sha256=SHA_B,
            )
            is False
        )

        expired_scope = _scope(
            expires_at=datetime(2021, 1, 1, tzinfo=UTC),
        )
        with pytest.raises(asyncpg.InvalidParameterValueError):
            await store.create_approval(
                _record(
                    APPROVAL_EXPIRED,
                    scope=expired_scope,
                    created_at=datetime(2020, 1, 1, tzinfo=UTC),
                )
            )

        denied = _record(
            APPROVAL_DENIED,
            scope=expired_scope,
            verdict=ApprovalVerdict.DENIED,
        )
        await store.create_approval(denied)
        assert (
            await store.claim_approval(
                approval_id=APPROVAL_DENIED,
                scope=expired_scope,
                plan_sha256=SHA_B,
            )
            is False
        )

        await store.create_approval(_record(APPROVAL_CONCURRENT, scope=valid_scope))
        claims = await asyncio.gather(
            store.claim_approval(
                approval_id=APPROVAL_CONCURRENT,
                scope=valid_scope,
                plan_sha256=SHA_B,
            ),
            store.claim_approval(
                approval_id=APPROVAL_CONCURRENT,
                scope=valid_scope,
                plan_sha256=SHA_B,
            ),
        )
        assert sum(claims) == 1

        reader = PostgresContextReader(
            pool,
            max_rows=10,
            max_bytes=16_384,
            timeout_seconds=2.0,
        )
        rows = await reader.read(
            "approval_status.v1",
            {"approval_id": APPROVAL_SINGLE},
        )
        assert len(rows) == 1
        assert rows[0]["approval_id"] == APPROVAL_SINGLE
        assert rows[0]["consumed_count"] == 1

        denied_rows = await reader.read(
            "approval_status.v1",
            {"approval_id": APPROVAL_DENIED},
        )
        assert len(denied_rows) == 1
        assert denied_rows[0]["verdict"] == "denied"
        assert denied_rows[0]["consumed_count"] == 0

        run_rows = await reader.read(
            "approvals_for_run.v1",
            {"run_id": "run-integration", "limit": 10},
        )
        assert 1 <= len(run_rows) <= 10

        capped_reader = PostgresContextReader(
            pool,
            max_rows=1,
            max_bytes=16_384,
            timeout_seconds=2.0,
        )
        with pytest.raises(ContextReadLimitError, match="row cap"):
            await capped_reader.read(
                "approvals_for_run.v1",
                {"run_id": "run-integration", "limit": 1},
            )

        byte_capped_reader = PostgresContextReader(
            pool,
            max_rows=10,
            max_bytes=64,
            timeout_seconds=2.0,
        )
        with pytest.raises(ContextReadLimitError, match="byte cap"):
            await byte_capped_reader.read(
                "approval_status.v1",
                {"approval_id": APPROVAL_SINGLE},
            )

        claimant_scope = _scope(run_id="run-role-claimant")
        claimant_approval_id = "approval-role-claimant-0001"
        await store.create_approval(_record(claimant_approval_id, scope=claimant_scope))
        expiring_scope = _scope(
            run_id="run-long-transaction-expiry",
            expires_at=datetime.now(UTC) + timedelta(seconds=1),
        )
        expiring_approval_id = "approval-long-expiry-0001"
        await store.create_approval(_record(expiring_approval_id, scope=expiring_scope))

        role_connection = await pool.acquire()
        try:
            role_rows = await role_connection.fetch(
                """
                SELECT rolname, rolsuper, rolinherit, rolcreaterole,
                       rolcreatedb, rolcanlogin, rolreplication, rolbypassrls
                FROM pg_roles
                WHERE rolname = ANY($1::text[])
                ORDER BY rolname
                """,
                [
                    "research_protocol_approval_claimant",
                    "research_protocol_approval_writer",
                    "research_protocol_planner_reader",
                ],
            )
            assert len(role_rows) == 3
            for role_row in role_rows:
                assert all(
                    role_row[field] is False
                    for field in (
                        "rolsuper",
                        "rolinherit",
                        "rolcreaterole",
                        "rolcreatedb",
                        "rolcanlogin",
                        "rolreplication",
                        "rolbypassrls",
                    )
                )
            assert (
                await role_connection.fetchval(
                    """
                    SELECT count(*)
                    FROM pg_auth_members AS membership
                    JOIN pg_roles AS child ON child.oid = membership.member
                    WHERE child.rolname = ANY($1::text[])
                    """,
                    [
                        "research_protocol_approval_claimant",
                        "research_protocol_approval_writer",
                        "research_protocol_planner_reader",
                    ],
                )
                == 0
            )

            async with role_connection.transaction():
                await role_connection.execute(
                    "SET LOCAL ROLE research_protocol_planner_reader"
                )
                assert (
                    await role_connection.fetchval(
                        "SELECT count(*) FROM research_protocol.approvals"
                    )
                    >= 1
                )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_planner_reader"
                    )
                    await role_connection.execute(
                        "UPDATE research_protocol.approvals "
                        "SET consumed_count = consumed_count WHERE false"
                    )

            role_record = _record(
                "approval-role-writer-0001",
                scope=_scope(),
            )
            writer_transaction = role_connection.transaction()
            await writer_transaction.start()
            try:
                await role_connection.execute(
                    "SET LOCAL ROLE research_protocol_approval_writer"
                )
                await role_connection.execute(
                    """
                    SELECT research_protocol.store_approval(
                        $1, $2, $3, $4, $5, $6, $7, $8, $9
                    )
                    """,
                    role_record.approval_id,
                    role_record.scope_sha256,
                    role_record.plan_sha256,
                    role_record.scope_json,
                    role_record.verdict.value,
                    role_record.surface,
                    role_record.created_at,
                    role_record.expires_at,
                    role_record.max_executions,
                )
            finally:
                await writer_transaction.rollback()

            tampered_scope = _scope(run_id="run-tampered-scope-hash")
            tampered_record = _record(
                "approval-tampered-hash-0001",
                scope=tampered_scope,
            )
            with pytest.raises(asyncpg.InvalidParameterValueError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_writer"
                    )
                    await role_connection.execute(
                        """
                        SELECT research_protocol.store_approval(
                            $1, $2, $3, $4, $5, $6, $7, $8, $9
                        )
                        """,
                        tampered_record.approval_id,
                        SHA_A,
                        tampered_record.plan_sha256,
                        tampered_record.scope_json,
                        tampered_record.verdict.value,
                        tampered_record.surface,
                        tampered_record.created_at,
                        tampered_record.expires_at,
                        tampered_record.max_executions,
                    )
            assert (
                await role_connection.fetchval(
                    "SELECT count(*) FROM research_protocol.approvals "
                    "WHERE approval_id = $1",
                    tampered_record.approval_id,
                )
                == 0
            )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_writer"
                    )
                    await role_connection.execute(
                        "CREATE TABLE research_protocol.forbidden_writer_table "
                        "(id integer)"
                    )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_writer"
                    )
                    await role_connection.fetchrow(
                        "SELECT * FROM research_protocol.claim_approval($1, $2, $3)",
                        claimant_approval_id,
                        claimant_scope.scope_sha256(),
                        SHA_B,
                    )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_writer"
                    )
                    await role_connection.execute(
                        "INSERT INTO research_protocol.approvals "
                        "(last_consumed_at) VALUES (CURRENT_TIMESTAMP)"
                    )

            async with role_connection.transaction():
                await role_connection.execute(
                    "SET LOCAL ROLE research_protocol_approval_claimant"
                )
                claimant_row = await role_connection.fetchrow(
                    "SELECT approval_id, consumed_count "
                    "FROM research_protocol.claim_approval($1, $2, $3)",
                    claimant_approval_id,
                    claimant_scope.scope_sha256(),
                    SHA_B,
                )
                assert claimant_row is not None
                assert claimant_row["approval_id"] == claimant_approval_id
                assert claimant_row["consumed_count"] == 1

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_claimant"
                    )
                    await role_connection.execute(
                        "SELECT research_protocol.store_approval("
                        "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
                    )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_claimant"
                    )
                    await role_connection.execute(
                        "UPDATE research_protocol.approvals "
                        "SET consumed_count = consumed_count WHERE false"
                    )

            with pytest.raises(asyncpg.InsufficientPrivilegeError):
                async with role_connection.transaction():
                    await role_connection.execute(
                        "SET LOCAL ROLE research_protocol_approval_claimant"
                    )
                    await role_connection.execute(
                        "UPDATE research_protocol.approvals "
                        "SET verdict = verdict WHERE false"
                    )

            expiry_transaction = role_connection.transaction()
            await expiry_transaction.start()
            try:
                await role_connection.execute(
                    "SET LOCAL ROLE research_protocol_approval_claimant"
                )
                await role_connection.fetchval("SELECT CURRENT_TIMESTAMP")
                await asyncio.sleep(2)
                expired_row = await role_connection.fetchrow(
                    "SELECT approval_id FROM "
                    "research_protocol.claim_approval($1, $2, $3)",
                    expiring_approval_id,
                    expiring_scope.scope_sha256(),
                    SHA_B,
                )
                assert expired_row is None
            finally:
                await expiry_transaction.rollback()
        finally:
            await pool.release(role_connection)

        timeout_claim_scope = _scope(run_id="run-timeout-claim")
        timeout_claim_id = "approval-timeout-claim-0001"
        await store.create_approval(
            _record(timeout_claim_id, scope=timeout_claim_scope)
        )

        blocker = await pool.acquire()
        blocker_transaction = blocker.transaction()
        await blocker_transaction.start()
        try:
            await blocker.execute(
                "LOCK TABLE research_protocol.approvals IN ACCESS EXCLUSIVE MODE"
            )
            timeout_reader = PostgresContextReader(
                pool,
                max_rows=10,
                max_bytes=16_384,
                timeout_seconds=0.05,
            )
            with pytest.raises(ContextReadTimeoutError):
                await timeout_reader.read(
                    "approval_status.v1",
                    {"approval_id": APPROVAL_SINGLE},
                )

            timeout_store = PostgresApprovalStore(pool, timeout_seconds=0.05)
            with pytest.raises(ApprovalStoreTimeoutError):
                await timeout_store.create_approval(
                    _record(
                        "approval-timeout-create-0001",
                        scope=_scope(run_id="run-timeout-create"),
                    )
                )
            with pytest.raises(ApprovalStoreTimeoutError):
                await timeout_store.claim_approval(
                    approval_id=timeout_claim_id,
                    scope=timeout_claim_scope,
                    plan_sha256=SHA_B,
                )
        finally:
            await blocker_transaction.rollback()
            await pool.release(blocker)
    finally:
        await pool.close()


def test_disposable_postgres_approval_claims_and_context_reads():
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(f"{TEST_DSN_ENV} is not set to a disposable database")
    asyncio.run(_exercise_database(dsn))
