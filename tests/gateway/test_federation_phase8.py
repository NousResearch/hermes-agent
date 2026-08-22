"""Tests for federation Phase 8 — compute pool."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.federation.federation_compute_pool import (
    ChunkResult,
    ComputeCapability,
    DistributedResult,
    FederationComputePool,
)


# ========================================================================
# Data structure tests
# ========================================================================

class TestComputeCapability:
    def test_default(self):
        cap = ComputeCapability(device_id="dev-a")
        assert cap.device_id == "dev-a"
        assert cap.compute_score == 0.1  # minimum floor

    def test_score_with_cpu_and_memory(self):
        cap = ComputeCapability(
            device_id="dev-b",
            cpu_cores=8,
            memory_gb=16.0,
        )
        # score = 8*1.0 + 16*0.5 = 16.0
        assert cap.compute_score == pytest.approx(16.0, rel=0.01)

    def test_score_with_gpu(self):
        cap = ComputeCapability(
            device_id="dev-c",
            cpu_cores=4,
            memory_gb=8.0,
            gpu_type="NVIDIA RTX 4090",
        )
        # score = 4*1.0 + 8*0.5 + 5.0 = 13.0
        assert cap.compute_score == pytest.approx(13.0, rel=0.01)

    def test_load_penalty(self):
        cap = ComputeCapability(
            device_id="dev-d",
            cpu_cores=8,
            memory_gb=16.0,
            load_avg=3.0,
        )
        # score = 16.0 / (1 + 3.0) = 4.0
        assert cap.compute_score == pytest.approx(4.0, rel=0.01)


class TestChunkResult:
    def test_create(self):
        r = ChunkResult(
            chunk_id="c-001",
            device_id="dev-a",
            success=True,
            data=[1, 2, 3],
        )
        assert r.success
        assert r.data == [1, 2, 3]
        assert r.error == ""


class TestDistributedResult:
    def test_success_rate(self):
        result = DistributedResult(
            task_id="t-001",
            total_chunks=10,
            successful_chunks=8,
            failed_chunks=2,
        )
        assert result.success_rate == 0.8

    def test_empty_result(self):
        result = DistributedResult(
            task_id="t-002",
            total_chunks=0,
            successful_chunks=0,
            failed_chunks=0,
        )
        assert result.success_rate == 0.0


# ========================================================================
# FederationComputePool tests
# ========================================================================

class TestFederationComputePool:
    def _make_pool(self):
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=True)
        pool = FederationComputePool(
            device_id="dev-a",
            adapter=adapter,
            chunk_timeout=1.0,
            max_retries=1,
        )
        return pool

    def test_init(self):
        pool = self._make_pool()
        assert pool.device_id == "dev-a"
        assert pool.chunk_timeout == 1.0
        assert pool.max_retries == 1

    def test_register_local_capability(self):
        pool = self._make_pool()
        pool._register_local_capability()
        assert "dev-a" in pool._capabilities
        cap = pool._capabilities["dev-a"]
        assert cap.device_id == "dev-a"
        assert cap.cpu_cores > 0  # Should detect real CPU cores

    def test_update_peer_capability(self):
        pool = self._make_pool()
        cap = ComputeCapability(
            device_id="dev-b",
            cpu_cores=16,
            memory_gb=32.0,
        )
        pool.update_peer_capability("dev-b", cap)
        assert pool._capabilities["dev-b"] == cap

    def test_distribution_plan_single_device(self):
        pool = self._make_pool()
        pool._register_local_capability()
        plan = pool._get_distribution_plan(10)
        assert plan == {"dev-a": 10}

    def test_distribution_plan_multiple_devices(self):
        pool = self._make_pool()
        pool._capabilities["dev-a"] = ComputeCapability(
            device_id="dev-a", cpu_cores=4, memory_gb=8.0,
        )
        pool._capabilities["dev-b"] = ComputeCapability(
            device_id="dev-b", cpu_cores=8, memory_gb=16.0,
        )
        plan = pool._get_distribution_plan(10)
        # dev-b has higher score, should get more chunks
        total = sum(plan.values())
        assert total == 10
        assert "dev-a" in plan
        assert "dev-b" in plan
        assert plan["dev-b"] >= plan["dev-a"]  # Higher score = more chunks

    @pytest.mark.asyncio
    async def test_distribute_empty_items(self):
        pool = self._make_pool()
        result = await pool.distribute("test", [])
        assert result.total_chunks == 0
        assert result.successful_chunks == 0

    @pytest.mark.asyncio
    async def test_distribute_with_local_handler(self):
        pool = self._make_pool()
        pool._register_local_capability()

        # Register a handler that doubles numbers
        async def double_handler(items):
            return [x * 2 for x in items]

        pool.register_handler("double", double_handler)

        result = await pool.distribute(
            "double",
            items=[1, 2, 3, 4, 5],
            chunk_size=2,
            handler_name="double",
        )

        assert result.total_chunks == 3  # ceil(5/2)
        assert result.successful_chunks == 3
        assert result.failed_chunks == 0
        # All chunks should have been doubled
        all_data = []
        for chunk in result.aggregated_data:
            all_data.extend(chunk)
        assert sorted(all_data) == [2, 4, 6, 8, 10]

    @pytest.mark.asyncio
    async def test_distribute_with_failing_handler(self):
        pool = self._make_pool()
        pool._register_local_capability()

        async def failing_handler(items):
            raise ValueError("Test error")

        pool.register_handler("fail", failing_handler)

        result = await pool.distribute(
            "fail",
            items=[1, 2, 3],
            chunk_size=1,
            handler_name="fail",
        )

        assert result.total_chunks == 3
        assert result.successful_chunks == 0
        assert result.failed_chunks == 3

    @pytest.mark.asyncio
    async def test_handle_compute_request(self):
        pool = self._make_pool()
        pool._register_local_capability()

        async def hash_handler(items):
            return [hash(x) for x in items]

        pool.register_handler("hash", hash_handler)

        msg = MagicMock()
        msg.sender_id = "dev-b"
        msg.payload = {
            "chunk_id": "test-0",
            "task_type": "hash",
            "data": ["a", "b"],
        }

        await pool.handle_compute_request(msg)
        # Should have sent a COMPUTE_RESPONSE
        pool.adapter.send.assert_called_once()
        response_msg = pool.adapter.send.call_args[0][0]
        assert response_msg.msg_type == "compute_response"
        assert response_msg.payload["success"] is True

    def test_get_all_capabilities(self):
        pool = self._make_pool()
        pool._register_local_capability()
        pool._capabilities["dev-b"] = ComputeCapability(
            device_id="dev-b", cpu_cores=8,
        )

        caps = pool.get_all_capabilities()
        assert len(caps) == 2
        assert "dev-a" in caps
        assert "dev-b" in caps
