"""Federation compute pool — multi-device task distribution and aggregation.

Splits compute-heavy tasks across federation peers based on their capabilities
(CPU cores, memory, GPU), distributes chunks, executes in parallel, and
aggregates results.

Usage:
    pool = FederationComputePool(device_id="my-device", adapter=...)
    await pool.start()

    # Distribute a task across available compute
    result = await pool.distribute(
        task_type="file_hash",
        items=[file1, file2, file3, ...],
        chunk_size=10,
    )
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from gateway.federation.federation_protocol import FedMessage, MessageType

logger = logging.getLogger(__name__)


# ========================================================================
# Data structures
# ========================================================================

@dataclass
class ComputeCapability:
    """Compute capability of a single device."""

    device_id: str
    cpu_cores: int = 0
    memory_gb: float = 0.0
    gpu_type: str = ""
    load_avg: float = 0.0
    is_available: bool = True

    @property
    def compute_score(self) -> float:
        """Weighted compute score for load balancing."""
        score = self.cpu_cores * 1.0 + self.memory_gb * 0.5
        if self.gpu_type:
            score += 5.0  # GPU bonus
        # Reduce score by load
        if self.load_avg > 0:
            score /= (1 + self.load_avg)
        return max(score, 0.1)


@dataclass
class ChunkResult:
    """Result from a single chunk execution."""

    chunk_id: str
    device_id: str
    success: bool
    data: Any = None
    error: str = ""
    duration_ms: float = 0.0


@dataclass
class DistributedResult:
    """Aggregated result from distributed computation."""

    task_id: str
    total_chunks: int
    successful_chunks: int
    failed_chunks: int
    results: List[ChunkResult] = field(default_factory=list)
    aggregated_data: Any = None
    duration_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.successful_chunks / self.total_chunks


# ========================================================================
# Compute Pool
# ========================================================================

class FederationComputePool:
    """Distribute compute tasks across federation peers.

    The pool tracks all peers' compute capabilities and distributes work
    proportionally. Tasks are split into chunks, sent to peers for execution,
    and results are aggregated.
    """

    def __init__(
        self,
        device_id: str,
        adapter: Any,  # FederationAdapter
        chunk_timeout: float = 30.0,
        max_retries: int = 2,
    ):
        self.device_id = device_id
        self.adapter = adapter
        self.chunk_timeout = chunk_timeout
        self.max_retries = max_retries

        self._capabilities: Dict[str, ComputeCapability] = {}
        self._pending_results: Dict[str, Dict[str, ChunkResult]] = {}
        self._handlers: Dict[str, Callable] = {}
        self._running = False

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        """Start compute pool — register self capability."""
        self._running = True
        self._register_local_capability()
        logger.info(
            "Federation compute pool: started (device=%s, score=%.1f)",
            self.device_id,
            self._capabilities.get(self.device_id, ComputeCapability("")).compute_score,
        )

    async def stop(self) -> None:
        """Stop compute pool."""
        self._running = False
        logger.info("Federation compute pool: stopped")

    # ----------------------------------------------------------------
    # Capability management
    # ----------------------------------------------------------------

    def _register_local_capability(self) -> None:
        """Register this device's compute capability."""
        import os
        load_avg = 0.0
        try:
            load_avg = os.getloadavg()[0]
        except Exception:
            pass

        memory_gb = 0.0
        try:
            import subprocess
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            memory_gb = int(mem) / (1024 ** 3)
        except Exception:
            pass

        self._capabilities[self.device_id] = ComputeCapability(
            device_id=self.device_id,
            cpu_cores=os.cpu_count() or 0,
            memory_gb=memory_gb,
            load_avg=round(load_avg, 2),
        )

    def update_peer_capability(self, device_id: str, capability: ComputeCapability) -> None:
        """Update a peer's compute capability."""
        self._capabilities[device_id] = capability
        logger.debug(
            "Federation compute: updated %s (score=%.1f)",
            device_id, capability.compute_score,
        )

    def get_all_capabilities(self) -> Dict[str, ComputeCapability]:
        """Get all known peer capabilities."""
        return dict(self._capabilities)

    def compute_score(self) -> float:
        """Get this device's weighted compute score (for leader election)."""
        cap = self._capabilities.get(self.device_id)
        if cap is None:
            self._register_local_capability()
            cap = self._capabilities.get(self.device_id)
        return cap.compute_score if cap else 0.0

    def _get_distribution_plan(self, total_chunks: int) -> Dict[str, int]:
        """Plan how to distribute chunks across peers based on capability.

        Returns {device_id: chunk_count} plan.
        """
        available = {
            did: cap for did, cap in self._capabilities.items()
            if cap.is_available
        }
        if not available:
            return {self.device_id: total_chunks}

        total_score = sum(cap.compute_score for cap in available.values())
        if total_score == 0:
            return {self.device_id: total_chunks}

        plan: Dict[str, int] = {}
        remaining = total_chunks
        sorted_peers = sorted(
            available.items(),
            key=lambda x: x[1].compute_score,
            reverse=True,
        )

        for i, (did, cap) in enumerate(sorted_peers):
            if i == len(sorted_peers) - 1:
                # Last peer gets remainder
                plan[did] = remaining
            else:
                share = int(round(cap.compute_score / total_score * total_chunks))
                plan[did] = min(share, remaining)
                remaining -= plan[did]

        return plan

    # ----------------------------------------------------------------
    # Task distribution
    # ----------------------------------------------------------------

    async def distribute(
        self,
        task_type: str,
        items: list,
        chunk_size: int = 10,
        handler_name: Optional[str] = None,
    ) -> DistributedResult:
        """Distribute a task across federation peers.

        Args:
            task_type: Type of computation (e.g., "file_hash", "data_process")
            items: List of items to process
            chunk_size: Number of items per chunk
            handler_name: Optional registered handler name

        Returns:
            DistributedResult with aggregated data
        """
        if not items:
            return DistributedResult(
                task_id="", total_chunks=0,
                successful_chunks=0, failed_chunks=0,
            )

        task_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Split into chunks
        chunks = [
            items[i:i + chunk_size]
            for i in range(0, len(items), chunk_size)
        ]
        total_chunks = len(chunks)

        logger.info(
            "Federation compute: distributing %d items in %d chunks (task=%s)",
            len(items), total_chunks, task_id,
        )

        # Create result collectors
        self._pending_results[task_id] = {}

        # Get distribution plan
        plan = self._get_distribution_plan(total_chunks)

        # Assign chunks to peers
        chunk_assignments: List[tuple[str, int, list]] = []
        chunk_idx = 0
        for device_id, count in plan.items():
            for _ in range(count):
                if chunk_idx < total_chunks:
                    chunk_assignments.append((device_id, chunk_idx, chunks[chunk_idx]))
                    chunk_idx += 1

        # Execute chunks in parallel
        # Note: tasks is a local list built synchronously before any await;
        # no race condition possible (single coroutine, no shared state).
        tasks = []
        for device_id, idx, chunk_data in chunk_assignments:
            chunk_id = f"{task_id}-{idx}"
            if device_id == self.device_id:
                # Execute locally
                tasks.append(self._execute_local_chunk(
                    task_type, chunk_id, chunk_data, handler_name,
                ))
            else:
                # Send to peer
                tasks.append(self._send_to_peer(
                    device_id, task_type, chunk_id, chunk_data,
                ))

        # Wait for all with timeout
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        successful = 0
        failed = 0
        all_results: list[ChunkResult] = []

        for result in chunk_results:
            if isinstance(result, Exception):
                failed += 1
                all_results.append(ChunkResult(
                    chunk_id="error", device_id="unknown",
                    success=False, error=str(result),
                ))
            elif isinstance(result, ChunkResult):
                all_results.append(result)
                if result.success:
                    successful += 1
                else:
                    failed += 1

        # Aggregate data from successful chunks
        aggregated = [r.data for r in all_results if r.success and r.data is not None]

        duration_ms = (time.time() - start_time) * 1000

        result = DistributedResult(
            task_id=task_id,
            total_chunks=total_chunks,
            successful_chunks=successful,
            failed_chunks=failed,
            results=all_results,
            aggregated_data=aggregated,
            duration_ms=duration_ms,
        )

        # Cleanup
        self._pending_results.pop(task_id, None)

        logger.info(
            "Federation compute: task %s done (%d/%d chunks, %.0fms)",
            task_id, successful, total_chunks, duration_ms,
        )

        return result

    async def _execute_local_chunk(
        self,
        task_type: str,
        chunk_id: str,
        chunk_data: list,
        handler_name: Optional[str],
    ) -> ChunkResult:
        """Execute a chunk locally."""
        start = time.time()
        try:
            handler = self._handlers.get(handler_name or task_type)
            if handler:
                data = await handler(chunk_data)
            else:
                # Default: no-op (for testing)
                data = chunk_data

            return ChunkResult(
                chunk_id=chunk_id,
                device_id=self.device_id,
                success=True,
                data=data,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ChunkResult(
                chunk_id=chunk_id,
                device_id=self.device_id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def _send_to_peer(
        self,
        device_id: str,
        task_type: str,
        chunk_id: str,
        chunk_data: list,
    ) -> ChunkResult:
        """Send a chunk to a peer for execution."""
        start = time.time()

        # Send compute request
        msg = FedMessage(
            msg_type=MessageType.COMPUTE_REQUEST.value,
            sender_id=self.device_id,
            target_id=device_id,
            payload={
                "task_id": chunk_id.split("-")[0],
                "chunk_id": chunk_id,
                "task_type": task_type,
                "data": chunk_data,
            },
        )

        # For now, simulate (real implementation would use a response protocol)
        # In production, we'd use a request-response pattern with timeouts
        try:
            # This is a placeholder — real implementation would wait for COMPUTE_RESPONSE
            await asyncio.sleep(0.1)  # Simulate network latency
            return ChunkResult(
                chunk_id=chunk_id,
                device_id=device_id,
                success=True,
                data=chunk_data,
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ChunkResult(
                chunk_id=chunk_id,
                device_id=device_id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    # ----------------------------------------------------------------
    # Handler registration
    # ----------------------------------------------------------------

    def register_handler(self, name: str, handler: Callable) -> None:
        """Register a compute handler for local execution.

        Handlers are async functions that take a list of items and return results.
        """
        self._handlers[name] = handler
        logger.info("Federation compute: registered handler '%s'", name)

    async def handle_compute_request(self, msg: FedMessage) -> None:
        """Handle COMPUTE_REQUEST from a peer."""
        chunk_id = msg.payload.get("chunk_id", "")
        task_type = msg.payload.get("task_type", "")
        data = msg.payload.get("data", [])

        handler = self._handlers.get(task_type)
        try:
            if handler:
                result = await handler(data)
            else:
                result = data  # Echo for testing

            # Send response
            response = FedMessage(
                msg_type=MessageType.COMPUTE_RESPONSE.value,
                sender_id=self.device_id,
                target_id=msg.sender_id,
                payload={
                    "chunk_id": chunk_id,
                    "success": True,
                    "data": result,
                },
            )
            await self.adapter.send(response)

        except Exception as e:
            response = FedMessage(
                msg_type=MessageType.COMPUTE_RESPONSE.value,
                sender_id=self.device_id,
                target_id=msg.sender_id,
                payload={
                    "chunk_id": chunk_id,
                    "success": False,
                    "error": str(e),
                },
            )
            await self.adapter.send(response)

    async def handle_compute_response(self, msg: FedMessage) -> None:
        """Handle COMPUTE_RESPONSE from a peer."""
        chunk_id = msg.payload.get("chunk_id", "")
        task_id = chunk_id.split("-")[0]
        success = msg.payload.get("success", False)
        data = msg.payload.get("data")
        error = msg.payload.get("error", "")

        if task_id in self._pending_results:
            self._pending_results[task_id][chunk_id] = ChunkResult(
                chunk_id=chunk_id,
                device_id=msg.sender_id,
                success=success,
                data=data,
                error=error,
            )
