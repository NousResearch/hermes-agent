"""
SlotPool - Manages slots across Nomad allocations.

The SlotPool is the core abstraction for slot-based multiplexing:
- Tracks available/acquired slots across containers
- Handles slot acquisition and release
- Auto-scales Nomad job count based on demand
- Provides batched tool execution
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..nomad.client import (
    Allocation,
    AllocationStatus,
    NomadClient,
    create_sandbox_job,
)
from .executor import ExecutionResult, SandboxExecutor
from .slot import Slot, SlotState, create_slots_for_allocation

logger = logging.getLogger(__name__)


@dataclass
class SlotPoolConfig:
    """Configuration for SlotPool."""
    
    # Nomad settings
    nomad_address: str = "http://localhost:4646"
    job_id: str = "atropos-sandbox"
    datacenter: str = "dc1"
    
    # Container settings
    image: str = "atropos-sandbox:local"  # Use :local tag to avoid registry pull
    slots_per_container: int = 10
    privileged: bool = False
    cpu: int = 500  # MHz
    memory: int = 512  # MB
    
    # Scaling settings
    min_containers: int = 1
    max_containers: int = 10
    
    # Timeouts
    acquire_timeout: float = 30.0  # Seconds between acquire polls (also triggers scale-up attempts)
    health_check_interval: float = 30.0  # Seconds between health checks
    scale_cooldown: float = 60.0  # Seconds between scale operations

    # Job lifecycle
    purge_job_on_start: bool = False  # Purge any pre-existing job before starting (local dev/training friendly)


class SlotPool:
    """
    Manages a pool of slots across Nomad allocations.
    
    The SlotPool:
    - Deploys sandbox containers to Nomad
    - Tracks slots across all running containers
    - Handles slot acquisition/release
    - Auto-scales based on demand
    - Provides batched execution via SandboxExecutor
    
    Usage:
        config = SlotPoolConfig(
            nomad_address="http://localhost:4646",
            job_id="my-sandbox",
            slots_per_container=10,
        )
        
        pool = SlotPool(config)
        await pool.start()
        
        # Acquire a slot
        slot = await pool.acquire()
        
        # Execute tool
        result = await pool.execute(slot, "bash", {"command": "ls"})
        
        # Release slot
        await pool.release(slot)
        
        # Shutdown
        await pool.stop()
    """
    
    def __init__(self, config: Optional[SlotPoolConfig] = None):
        self.config = config or SlotPoolConfig()
        
        # Nomad client
        self.nomad = NomadClient(address=self.config.nomad_address)
        
        # Sandbox executor for tool execution
        self.executor = SandboxExecutor()
        
        # Slot tracking
        self._slots: Dict[str, Slot] = {}  # slot_key -> Slot
        self._available_queue: asyncio.Queue[str] = asyncio.Queue()
        self._lock = asyncio.Lock()
        self._scale_lock = asyncio.Lock()
        
        # State
        self._started = False
        self._health_task: Optional[asyncio.Task] = None
        self._scale_task: Optional[asyncio.Task] = None
        self._last_scale_time = 0.0
    
    def _slot_key(self, alloc_id: str, slot_id: str) -> str:
        """Generate unique key for a slot."""
        return f"{alloc_id}:{slot_id}"
    
    @property
    def total_slots(self) -> int:
        """Total number of slots in pool."""
        return len(self._slots)
    
    @property
    def available_slots(self) -> int:
        """Number of available slots."""
        return sum(1 for s in self._slots.values() if s.is_available)
    
    @property
    def acquired_slots(self) -> int:
        """Number of acquired slots."""
        return sum(1 for s in self._slots.values() if s.is_acquired)
    
    async def start(self) -> None:
        """
        Start the slot pool.
        
        - Checks if Nomad is healthy
        - Deploys sandbox job if not running
        - Discovers existing allocations
        - Starts health check background task
        """
        if self._started:
            return
        
        logger.info(f"Starting SlotPool (job_id={self.config.job_id})")

        try:
            # Check Nomad health
            if not await self.nomad.is_healthy():
                raise RuntimeError(f"Nomad is not reachable at {self.config.nomad_address}")

            if self.config.purge_job_on_start:
                logger.info(f"Purging any existing Nomad job: {self.config.job_id}")
                await self.nomad.stop_job(self.config.job_id, purge=True)

            # Check if job exists (after optional purge)
            job = await self.nomad.get_job(self.config.job_id)

            if job is None:
                # Deploy new job
                logger.info(f"Deploying sandbox job: {self.config.job_id}")
                job_spec = create_sandbox_job(
                    job_id=self.config.job_id,
                    image=self.config.image,
                    count=self.config.min_containers,
                    slots_per_container=self.config.slots_per_container,
                    privileged=self.config.privileged,
                    cpu=self.config.cpu,
                    memory=self.config.memory,
                    datacenter=self.config.datacenter,
                )
                result = await self.nomad.submit_job(job_spec)
                if "error" in result:
                    raise RuntimeError(f"Failed to submit job: {result}")

            # Wait for allocations to be running (even if the job already existed).
            await self._wait_for_healthy_allocations(self.config.min_containers)

            # Discover existing allocations and slots
            await self._refresh_slots()

            # Start health check task
            self._health_task = asyncio.create_task(self._health_check_loop())

            self._started = True
            logger.info(f"SlotPool started: {self.total_slots} slots available")
        except Exception:
            # Ensure aiohttp sessions are not leaked if we fail to start.
            await self.stop(purge_job=False)
            raise
    
    async def stop(self, purge_job: bool = False) -> None:
        """
        Stop the slot pool.
        
        Args:
            purge_job: If True, also stop the Nomad job
        """
        logger.info("Stopping SlotPool")

        # Cancel health check task
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            finally:
                self._health_task = None

        if self._scale_task:
            self._scale_task.cancel()
            try:
                await self._scale_task
            except asyncio.CancelledError:
                pass
            finally:
                self._scale_task = None

        # Optionally stop the job (do this even if start() never completed).
        if purge_job:
            logger.info(f"Stopping Nomad job: {self.config.job_id}")
            await self.nomad.stop_job(self.config.job_id, purge=True)

        # Close connections
        await self.executor.close()
        await self.nomad.close()

        self._started = False
        self._slots.clear()

        # Clear the queue
        while not self._available_queue.empty():
            try:
                self._available_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    async def acquire(self, trajectory_id: Optional[str] = None) -> Slot:
        """
        Acquire an available slot.
        
        If no slots are available, waits up to acquire_timeout seconds.
        If still no slots, attempts to scale up.
        
        Args:
            trajectory_id: Optional ID of trajectory acquiring the slot
            
        Returns:
            Acquired Slot
            
        Raises:
            asyncio.TimeoutError: If no slot becomes available
        """
        if not self._started:
            raise RuntimeError("SlotPool not started")

        while True:
            try:
                # Try to get an available slot
                slot_key = await asyncio.wait_for(
                    self._available_queue.get(),
                    timeout=self.config.acquire_timeout,
                )
            except asyncio.TimeoutError:
                # Try to scale up, but keep waiting even if scaling isn't possible.
                # In practice, slots may become available shortly (e.g. contention),
                # and scaling may be temporarily blocked by Nomad deployments.
                await self._try_scale_up()
                continue

            slot = self._slots.get(slot_key)
            if slot is None:
                # Slot was removed; discard stale queue entry and retry.
                continue

            try:
                slot.acquire(trajectory_id)
            except RuntimeError:
                # Slot isn't actually available (e.g. duplicate queue entry); retry.
                continue

            logger.debug(f"Acquired slot {slot.slot_id} (alloc={slot.alloc_id[:8]})")
            return slot
    
    async def release(self, slot: Slot, reset_workspace: bool = False) -> None:
        """
        Release a slot back to the pool.
        
        Args:
            slot: Slot to release
            reset_workspace: If True, clear the workspace files
        """
        slot_key = self._slot_key(slot.alloc_id, slot.slot_id)
        
        if slot_key not in self._slots:
            logger.warning(f"Releasing unknown slot: {slot_key}")
            return
        
        # Optionally reset workspace
        if reset_workspace:
            await self.executor.reset_slot(slot)
        
        slot.release()
        await self._available_queue.put(slot_key)
        
        logger.debug(f"Released slot {slot.slot_id}")
    
    async def execute(
        self,
        slot: Slot,
        tool_name: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> ExecutionResult:
        """
        Execute a tool in a slot's workspace.
        
        Args:
            slot: Slot to execute in
            tool_name: Name of tool (bash, read_file, write_file)
            args: Tool arguments
            timeout: Optional timeout override
            
        Returns:
            ExecutionResult
        """
        return await self.executor.execute(slot, tool_name, args, timeout)
    
    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools in parallel.
        
        This is the key optimization - batch execution across multiple slots
        maximizes container utilization.
        
        Args:
            requests: List of (slot, tool_name, args) tuples
            timeout: Optional timeout override
            
        Returns:
            List of ExecutionResults in same order
        """
        return await self.executor.execute_batch(requests, timeout)
    
    async def _refresh_slots(self) -> None:
        """Refresh slot inventory from Nomad allocations."""
        async with self._lock:
            allocs = await self.nomad.get_job_allocations(self.config.job_id)
            
            # Track which slots we've seen
            seen_keys = set()
            
            for alloc in allocs:
                if alloc.status != AllocationStatus.RUNNING:
                    continue
                
                if not alloc.http_address:
                    continue
                
                # Check container health
                healthy = await self.executor.health_check(alloc.http_address)
                if not healthy:
                    continue
                
                # Create slots for this allocation
                for i in range(self.config.slots_per_container):
                    slot_id = f"slot_{i}"
                    slot_key = self._slot_key(alloc.id, slot_id)
                    seen_keys.add(slot_key)
                    
                    if slot_key not in self._slots:
                        # New slot
                        slot = Slot(
                            slot_id=slot_id,
                            alloc_id=alloc.id,
                            container_addr=alloc.http_address,
                        )
                        self._slots[slot_key] = slot
                        await self._available_queue.put(slot_key)
                        logger.debug(f"Added slot: {slot_key}")
            
            # Remove slots from dead allocations
            for slot_key in list(self._slots.keys()):
                if slot_key not in seen_keys:
                    slot = self._slots.pop(slot_key)
                    logger.debug(f"Removed slot: {slot_key}")
    
    async def _wait_for_healthy_allocations(
        self, 
        min_count: int, 
        timeout: float = 120.0
    ) -> None:
        """Wait for allocations to become healthy."""
        import time
        start = time.time()

        def _summarize_alloc_detail(detail: Dict[str, Any]) -> str:
            task_states = detail.get("TaskStates") or {}
            parts: List[str] = []
            if isinstance(task_states, dict):
                for task_name, st in task_states.items():
                    events = (st or {}).get("Events") or []
                    if isinstance(events, list) and events:
                        # Include a few recent events; the latest can be a generic restart message
                        # while the true root cause is slightly earlier (e.g. image pull failure).
                        recent = events[-3:]
                        msgs: List[str] = []
                        for ev in recent:
                            desc = ev.get("DisplayMessage") or ev.get("Message") or ev.get("Type") or ""
                            if desc:
                                msgs.append(desc)
                        if msgs:
                            parts.append(f"{task_name}: " + " | ".join(msgs))
            return "; ".join(parts)

        def _alloc_events_lower(detail: Dict[str, Any]) -> str:
            task_states = detail.get("TaskStates") or {}
            texts: List[str] = []
            if isinstance(task_states, dict):
                for _task_name, st in task_states.items():
                    events = (st or {}).get("Events") or []
                    if isinstance(events, list):
                        for ev in events[-10:]:
                            desc = ev.get("DisplayMessage") or ev.get("Message") or ev.get("Type") or ""
                            if desc:
                                texts.append(desc)
            return " ".join(texts).lower()
        
        while time.time() - start < timeout:
            allocs = await self.nomad.get_job_allocations(self.config.job_id)
            
            healthy_count = 0
            for alloc in allocs:
                if alloc.status == AllocationStatus.RUNNING and alloc.http_address:
                    if await self.executor.health_check(alloc.http_address):
                        healthy_count += 1

                # Fast-fail on obvious driver/image errors to avoid waiting out the full timeout.
                if alloc.id:
                    detail = await self.nomad.get_allocation(alloc.id)
                    if isinstance(detail, dict):
                        summary = _summarize_alloc_detail(detail)
                        lowered = _alloc_events_lower(detail) or summary.lower()
                        if "failed to pull" in lowered or "pull access denied" in lowered:
                            raise RuntimeError(
                                "Nomad allocation failed to start due to a Docker image pull error. "
                                f"Allocation {alloc.id[:8]}: {summary}\n"
                                "If you're using a local image tag (e.g. `atropos-sandbox:local`) on macOS, "
                                "make sure the image is loaded into Docker, e.g.:\n"
                                "  docker buildx build --load -t atropos-sandbox:local -f Hermes-Agent/atropos/Dockerfile Hermes-Agent/atropos"
                            )
                        if "exceeded allowed attempts" in lowered:
                            raise RuntimeError(
                                "Nomad allocation is crash-looping and has entered restart backoff. "
                                f"Allocation {alloc.id[:8]}: {summary}\n"
                                "Inspect logs with:\n"
                                f"  nomad alloc logs -stderr -task sandbox-server {alloc.id}\n"
                                "Common causes include: missing local Docker image tag, container entrypoint error, "
                                "or sandbox-server startup failure."
                            )
            
            if healthy_count >= min_count:
                return
            
            await asyncio.sleep(2.0)

        # Timed out: include allocation status detail to help debugging.
        allocs = await self.nomad.get_job_allocations(self.config.job_id)
        alloc_lines: List[str] = []
        for alloc in allocs[:10]:
            addr = alloc.http_address or "-"
            line = f"{alloc.id[:8]} status={alloc.status.value} http={addr}"
            detail = await self.nomad.get_allocation(alloc.id)
            if isinstance(detail, dict):
                summary = _summarize_alloc_detail(detail)
                if summary:
                    line += f" detail={summary}"
            alloc_lines.append(line)

        hint = (
            "Timed out waiting for healthy sandbox allocations.\n"
            f"Job: {self.config.job_id}, desired_healthy: {min_count}\n"
            "Allocations:\n  - " + "\n  - ".join(alloc_lines)
        )
        raise RuntimeError(hint)
    
    async def _try_scale_up(self) -> bool:
        """Attempt to scale up the job."""
        import time

        async with self._scale_lock:
            # Check cooldown
            if time.time() - self._last_scale_time < self.config.scale_cooldown:
                return False

            # Check max containers
            status = await self.nomad.get_job_status(self.config.job_id)
            if status is None:
                return False

            current_count = status.count
            if current_count >= self.config.max_containers:
                logger.warning(f"Cannot scale up: already at max ({self.config.max_containers})")
                return False

            # Scale up
            new_count = min(current_count + 1, self.config.max_containers)
            logger.info(f"Scaling up from {current_count} to {new_count} containers")

            scale_resp = await self.nomad.scale_job(
                self.config.job_id,
                count=new_count,
                task_group="sandbox",
            )

            # Nomad may return non-JSON errors (e.g. plain text) with a status field.
            if isinstance(scale_resp, dict) and scale_resp.get("status", 200) >= 400:
                logger.warning(f"Scale request rejected: {scale_resp}")
                self._last_scale_time = time.time()
                return False

            self._last_scale_time = time.time()

            # Wait for new allocation in the background so contended acquires can still
            # make progress (e.g. by grabbing slots released by other trajectories).
            if self._scale_task is None or self._scale_task.done():
                self._scale_task = asyncio.create_task(self._wait_for_scale(new_count))

            return True

    async def _wait_for_scale(self, desired_count: int) -> None:
        try:
            await self._wait_for_healthy_allocations(desired_count, timeout=60.0)
            await self._refresh_slots()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Failed to scale up: {e}")
    
    async def _health_check_loop(self) -> None:
        """Background task to monitor container health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._refresh_slots()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        slots_by_state = {}
        for slot in self._slots.values():
            state = slot.state.value
            slots_by_state[state] = slots_by_state.get(state, 0) + 1

        container_count = len({s.alloc_id for s in self._slots.values()}) if self._slots else 0
        
        return {
            "total_slots": self.total_slots,
            "available_slots": self.available_slots,
            "acquired_slots": self.acquired_slots,
            "containers": container_count,
            "slots_by_state": slots_by_state,
            "started": self._started,
        }
