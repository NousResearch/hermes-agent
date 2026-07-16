"""Bounded PostgreSQL registry and read-only adapter tests."""

from __future__ import annotations

import asyncio
import logging
import re

import pytest

from plugins.research_protocol.storage.postgres import (
    ContextReadLimitError,
    ContextReadTimeoutError,
    PostgresApprovalStore,
    PostgresContextReader,
    UnknownContextQueryError,
)


class FakeTransaction:
    def __init__(self, owner, kwargs):
        self.owner = owner
        self.kwargs = kwargs

    async def __aenter__(self):
        self.owner.transactions.append(self.kwargs)

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeConnection:
    def __init__(self, rows=None, *, delay=0):
        self.rows = rows or []
        self.delay = delay
        self.transactions = []
        self.executions = []
        self.fetches = []

    def transaction(self, **kwargs):
        return FakeTransaction(self, kwargs)

    async def execute(self, sql, *args):
        self.executions.append((sql, args))

    async def fetch(self, sql, *args):
        self.fetches.append((sql, args))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.rows


class AcquireContext:
    def __init__(self, connection):
        self.connection = connection

    async def __aenter__(self):
        return self.connection

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakePool:
    def __init__(self, connection):
        self.connection = connection
        self.acquire_count = 0

    def acquire(self):
        self.acquire_count += 1
        return AcquireContext(self.connection)


def test_unknown_query_is_rejected_before_pool_access():
    pool = FakePool(FakeConnection())
    reader = PostgresContextReader(pool)

    with pytest.raises(UnknownContextQueryError):
        asyncio.run(reader.read("SELECT secret FROM anything", {}))

    assert pool.acquire_count == 0


def test_approval_store_rejects_unvalidated_record_before_pool_access():
    pool = FakePool(FakeConnection())
    store = PostgresApprovalStore(pool)

    with pytest.raises(TypeError, match="ApprovalRecord"):
        asyncio.run(store.create_approval(object()))

    assert pool.acquire_count == 0


def test_approval_claim_rejects_unvalidated_scope_before_pool_access():
    pool = FakePool(FakeConnection())
    store = PostgresApprovalStore(pool)

    class ForgedScope:
        @staticmethod
        def scope_sha256():
            return "a" * 64

    claimed = asyncio.run(
        store.claim_approval(
            approval_id="approval-001",
            scope=ForgedScope(),
            plan_sha256="b" * 64,
        )
    )

    assert claimed is False
    assert pool.acquire_count == 0


def test_context_read_uses_fixed_query_bound_parameters_and_readonly_transaction():
    connection = FakeConnection([
        {"approval_id": "approval-001", "verdict": "approved"}
    ])
    reader = PostgresContextReader(FakePool(connection), timeout_seconds=2.5)

    rows = asyncio.run(
        reader.read(
            "approval_status.v1",
            {"approval_id": "approval-001"},
        )
    )

    assert rows == [{"approval_id": "approval-001", "verdict": "approved"}]
    assert connection.transactions == [{"readonly": True}]
    assert len(connection.executions) == 1
    timeout_sql, timeout_args = connection.executions[0]
    assert "set_config" in timeout_sql
    assert timeout_args == ("2500ms",)
    assert len(connection.fetches) == 1
    query_sql, query_args = connection.fetches[0]
    assert "FROM research_protocol.approvals" in query_sql
    assert query_args == ("approval-001",)
    assert "approval-001" not in query_sql


def test_context_read_logs_bounded_metrics_without_parameters(caplog):
    sensitive_parameter = "postgresql://sensitive-canary.example.invalid/db"
    connection = FakeConnection([{"approval_id": sensitive_parameter}])
    reader = PostgresContextReader(FakePool(connection))

    with caplog.at_level(
        logging.INFO,
        logger="plugins.research_protocol.storage.postgres",
    ):
        rows = asyncio.run(
            reader.read(
                "approval_status.v1",
                {"approval_id": sensitive_parameter},
            )
        )

    assert rows == [{"approval_id": sensitive_parameter}]
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "query_id=approval_status.v1" in messages
    assert "row_count=1" in messages
    assert re.search(r"latency_ms=\d+(?:\.\d+)?", messages)
    assert sensitive_parameter not in messages
    assert "sensitive-canary" not in messages


def test_context_read_enforces_row_cap_after_fetch():
    connection = FakeConnection([
        {"approval_id": "one"},
        {"approval_id": "two"},
        {"approval_id": "three"},
    ])
    reader = PostgresContextReader(FakePool(connection), max_rows=2)

    with pytest.raises(ContextReadLimitError, match="row cap"):
        asyncio.run(
            reader.read(
                "approvals_for_run.v1",
                {"run_id": "run-001", "limit": 2},
            )
        )

    _sql, args = connection.fetches[0]
    assert args == ("run-001", 3)


def test_context_read_enforces_caller_limit_below_global_cap():
    connection = FakeConnection([
        {"approval_id": "one"},
        {"approval_id": "two"},
    ])
    reader = PostgresContextReader(FakePool(connection), max_rows=100)

    with pytest.raises(ContextReadLimitError, match="row cap"):
        asyncio.run(
            reader.read(
                "approvals_for_run.v1",
                {"run_id": "run-001", "limit": 1},
            )
        )

    _sql, args = connection.fetches[0]
    assert args == ("run-001", 2)


def test_context_read_enforces_serialized_byte_cap():
    connection = FakeConnection([{"approval_id": "a" * 200}])
    reader = PostgresContextReader(FakePool(connection), max_bytes=64)

    with pytest.raises(ContextReadLimitError, match="byte cap"):
        asyncio.run(
            reader.read(
                "approval_status.v1",
                {"approval_id": "approval-001"},
            )
        )


def test_context_read_enforces_wall_clock_timeout():
    connection = FakeConnection([{"approval_id": "approval-001"}], delay=0.05)
    reader = PostgresContextReader(FakePool(connection), timeout_seconds=0.01)

    with pytest.raises(ContextReadTimeoutError):
        asyncio.run(
            reader.read(
                "approval_status.v1",
                {"approval_id": "approval-001"},
            )
        )
