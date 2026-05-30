"""Regression tests for the Hindsight worker claim policy used by Hermes."""

from uuid import uuid4

import pytest

ops_postgresql = pytest.importorskip(
    "hindsight_api.engine.db.ops_postgresql",
    reason=(
        "Hindsight embedded API internals are optional and are not installed "
        "in the default hermes-agent CI environment."
    ),
)
PostgreSQLOps = ops_postgresql.PostgreSQLOps


def _task(bank_id: str) -> dict:
    return {
        "operation_id": uuid4(),
        "operation_type": "consolidation",
        "bank_id": bank_id,
        "task_payload": {"type": "consolidation", "bank_id": bank_id},
        "retry_count": 0,
    }


class _ClaimConn:
    def __init__(self, consolidation_batches: list[list[dict]]):
        self._consolidation_batches = iter(consolidation_batches)

    async def fetch(self, query: str, *args):
        if "SELECT DISTINCT bank_id" in query:
            return []
        if "operation_type = 'consolidation'" in query:
            return next(self._consolidation_batches)
        return []

    async def execute(self, query: str, *args):
        return None


@pytest.mark.asyncio
async def test_claim_batch_takes_only_one_consolidation_per_bank_from_reserved_pool():
    conn = _ClaimConn([[_task("ovyon"), _task("ovyon")]])

    rows = await PostgreSQLOps().claim_tasks(
        conn,
        "async_operations",
        "test-worker",
        {"consolidation": 2},
        0,
    )

    assert [row["bank_id"] for row in rows] == ["ovyon"]


@pytest.mark.asyncio
async def test_reserved_consolidation_blocks_same_bank_from_shared_pool_in_claim_cycle():
    conn = _ClaimConn([[_task("ovyon")], [_task("ovyon")]])

    rows = await PostgreSQLOps().claim_tasks(
        conn,
        "async_operations",
        "test-worker",
        {"consolidation": 1},
        1,
    )

    assert [row["bank_id"] for row in rows] == ["ovyon"]
