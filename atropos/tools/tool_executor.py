"""
ToolExecutor - queued, batched tool dispatch for multiplexed agent trajectories.

This component is responsible for:
- Maintaining trajectory -> Slot affinity (workspace continuity)
- Batching sandbox tool calls across trajectories to maximize container utilization
- Routing external tools (ToolSchema.external=True) to a ToolServer (Phase 4.5)

For now, only sandbox tools are executed:
- bash
- read_file
- write_file
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .base import (
    ArtifactArchiveRequestPayload,
    ArtifactArchiveResponsePayload,
    ArtifactListRequestPayload,
    ArtifactListResponsePayload,
    ArtifactReadRequestPayload,
    ArtifactReadResponsePayload,
    ToolCall,
    ToolCallPayload,
    ToolRegistry,
    ToolResult,
    ToolResultPayload,
    ToolServerExecuteRequest,
)
from ..backends.base import ToolBackend
from ..slots import Slot


@dataclass
class ToolExecutorConfig:
    batch_window_ms: int = 20
    max_batch_size: int = 200
    allow_network: bool = True
    require_sandbox: bool = False
    require_stateful_sandbox: bool = False
    tool_server_url: Optional[str] = None
    tool_server_token: Optional[str] = None


@dataclass
class _QueuedToolRequest:
    trajectory_id: str
    call: ToolCall
    timeout_s: Optional[float]
    future: asyncio.Future


class ToolExecutor:
    def __init__(
        self,
        backend: ToolBackend,
        tools: ToolRegistry,
        config: Optional[ToolExecutorConfig] = None,
    ) -> None:
        self.backend = backend
        self.tools = tools
        self.config = config or ToolExecutorConfig()

        self._queue: asyncio.Queue[Optional[_QueuedToolRequest]] = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()

        self._slots_lock = asyncio.Lock()
        self._slot_by_trajectory: Dict[str, Slot] = {}

        self._tool_server_client: Optional[httpx.AsyncClient] = None
        self._tool_server_lock = asyncio.Lock()

        # lightweight stats for status endpoints
        self.total_requests: int = 0
        self.total_errors: int = 0
        self.latencies_s: List[float] = []

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run_loop())

    def queue_size(self) -> int:
        return self._queue.qsize()

    async def close(self) -> None:
        self._stopping.set()
        await self._queue.put(None)
        if self._task:
            await self._task
            self._task = None

        client = self._tool_server_client
        self._tool_server_client = None
        if client is not None:
            await client.aclose()

        # Best-effort release any remaining slots.
        async with self._slots_lock:
            slots = list(self._slot_by_trajectory.items())
            self._slot_by_trajectory.clear()

        for _, slot in slots:
            try:
                await self.backend.release(slot, reset_workspace=False)
            except Exception:
                pass

    async def execute(
        self,
        trajectory_id: str,
        call: ToolCall,
        timeout_s: Optional[float] = None,
    ) -> ToolResult:
        if self._task is None:
            raise RuntimeError("ToolExecutor not started (call start() first)")

        # Allow tool args to suggest a timeout (Hermes-compatible terminal tool),
        # but never let the model choose "infinite" timeouts.
        if timeout_s is None:
            raw_timeout = call.arguments.get("timeout")
            if isinstance(raw_timeout, (int, float)):
                timeout_s = float(raw_timeout)
        if timeout_s is not None:
            timeout_s = max(1.0, min(float(timeout_s), 600.0))

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        started = time.perf_counter()
        await self._queue.put(_QueuedToolRequest(trajectory_id=trajectory_id, call=call, timeout_s=timeout_s, future=fut))
        try:
            result: ToolResult = await fut
            return result
        finally:
            self.latencies_s.append(time.perf_counter() - started)

    async def release_trajectory(self, trajectory_id: str, reset_workspace: bool = False) -> None:
        async with self._slots_lock:
            slot = self._slot_by_trajectory.pop(trajectory_id, None)

        if slot is not None:
            await self.backend.release(slot, reset_workspace=reset_workspace)

    async def _get_slot_if_present(self, trajectory_id: str) -> Optional[Slot]:
        async with self._slots_lock:
            return self._slot_by_trajectory.get(trajectory_id)

    # ---------------------------------------------------------------------
    # Artifact helpers (optional)
    # ---------------------------------------------------------------------

    async def read_artifact(self, req: ArtifactReadRequestPayload) -> ArtifactReadResponsePayload:
        slot = await self._get_slot_if_present(req.trajectory_id)
        if slot is None:
            return ArtifactReadResponsePayload(success=False, error="No active slot for trajectory (run a sandbox tool first)")
        data = await self.backend.read_artifact(
            slot,
            req.path,
            encoding=req.encoding,
            max_bytes=req.max_bytes,
            include_sha256=req.include_sha256,
        )
        if isinstance(data, dict):
            data = dict(data)
            data.pop("http_status", None)
        try:
            return ArtifactReadResponsePayload(**(data or {}))
        except Exception as e:
            return ArtifactReadResponsePayload(success=False, error=f"Invalid artifact read response: {e}")

    async def list_artifacts(self, req: ArtifactListRequestPayload) -> ArtifactListResponsePayload:
        slot = await self._get_slot_if_present(req.trajectory_id)
        if slot is None:
            return ArtifactListResponsePayload(success=False, error="No active slot for trajectory (run a sandbox tool first)")
        data = await self.backend.list_artifacts(
            slot,
            req.path,
            recursive=req.recursive,
            max_entries=req.max_entries,
        )
        if isinstance(data, dict):
            data = dict(data)
            data.pop("http_status", None)
        try:
            return ArtifactListResponsePayload(**(data or {}))
        except Exception as e:
            return ArtifactListResponsePayload(success=False, error=f"Invalid artifact list response: {e}")

    async def archive_artifacts(self, req: ArtifactArchiveRequestPayload) -> ArtifactArchiveResponsePayload:
        slot = await self._get_slot_if_present(req.trajectory_id)
        if slot is None:
            return ArtifactArchiveResponsePayload(success=False, error="No active slot for trajectory (run a sandbox tool first)")
        data = await self.backend.archive_artifacts(
            slot,
            req.path,
            archive_format=req.format,
            max_bytes=req.max_bytes,
            max_entries=req.max_entries,
        )
        if isinstance(data, dict):
            data = dict(data)
            data.pop("http_status", None)
        try:
            return ArtifactArchiveResponsePayload(**(data or {}))
        except Exception as e:
            return ArtifactArchiveResponsePayload(success=False, error=f"Invalid artifact archive response: {e}")

    async def _get_or_acquire_slot(self, trajectory_id: str) -> Slot:
        async with self._slots_lock:
            existing = self._slot_by_trajectory.get(trajectory_id)
            if existing is not None:
                return existing

        slot = await self.backend.acquire(trajectory_id)

        async with self._slots_lock:
            existing = self._slot_by_trajectory.get(trajectory_id)
            if existing is not None:
                # Another coroutine won the race; return its slot.
                await self.backend.release(slot, reset_workspace=False)
                return existing
            self._slot_by_trajectory[trajectory_id] = slot
            return slot

    async def _run_loop(self) -> None:
        pending: List[_QueuedToolRequest] = []
        deadline: Optional[float] = None

        batch_window_s = max(0.0, self.config.batch_window_ms / 1000.0)
        max_batch = max(1, self.config.max_batch_size)

        while True:
            if self._stopping.is_set() and self._queue.empty() and not pending:
                break

            timeout = None
            if pending and deadline is not None:
                timeout = max(0.0, deadline - time.perf_counter())

            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                if item is None:
                    continue
                pending.append(item)
                if len(pending) == 1:
                    deadline = time.perf_counter() + batch_window_s
                if len(pending) < max_batch:
                    continue
            except asyncio.TimeoutError:
                # batch window elapsed
                pass

            if not pending:
                deadline = None
                continue

            batch = pending
            pending = []
            deadline = None

            await self._execute_batch(batch)

    async def _get_tool_server_client(self) -> httpx.AsyncClient:
        url = self.config.tool_server_url
        if not url:
            raise RuntimeError("ToolServer not configured")

        if self._tool_server_client is not None:
            return self._tool_server_client

        async with self._tool_server_lock:
            if self._tool_server_client is None:
                self._tool_server_client = httpx.AsyncClient(base_url=url.rstrip("/"))
            return self._tool_server_client

    def _tool_server_headers(self) -> Dict[str, str]:
        token = self.config.tool_server_token
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}

    async def _execute_external(self, req: _QueuedToolRequest) -> ToolResult:
        client = await self._get_tool_server_client()
        slot_id: Optional[str] = None
        container_addr: Optional[str] = None
        slot = await self._get_slot_if_present(req.trajectory_id)
        if slot is not None:
            slot_id = slot.slot_id
            container_addr = slot.container_addr

        payload = ToolServerExecuteRequest(
            trajectory_id=req.trajectory_id,
            tool=ToolCallPayload.from_tool_call(req.call),
            timeout_s=req.timeout_s,
            slot_id=slot_id,
            container_addr=container_addr,
        )

        try:
            resp = await client.post(
                "/execute",
                json=payload.model_dump(),
                headers=self._tool_server_headers(),
                timeout=req.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            parsed = ToolResultPayload(**data)
            result = parsed.to_tool_result()
            if result.uniq_id is None:
                result.uniq_id = req.call.uniq_id
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"External tool failed: {e}",
                uniq_id=req.call.uniq_id,
            )

    async def _execute_batch(self, batch: List[_QueuedToolRequest]) -> None:
        # Resolve tool schemas once per request and separate sandbox/external/unknown.
        sandbox_items: List[_QueuedToolRequest] = []
        external_items: List[_QueuedToolRequest] = []
        unknown_items: List[_QueuedToolRequest] = []

        for it in batch:
            tool = self.tools.get(it.call.name)
            if tool is None:
                unknown_items.append(it)
                continue

            schema = tool.schema
            if not schema.external:
                sandbox_items.append(it)
            else:
                external_items.append(it)

        for it in unknown_items:
            self.total_requests += 1
            self.total_errors += 1
            if not it.future.done():
                it.future.set_result(
                    ToolResult(
                        success=False,
                        error=f"Unknown tool: {it.call.name}",
                        uniq_id=it.call.uniq_id,
                    )
                )

        if external_items:
            if not self.config.tool_server_url:
                for it in external_items:
                    self.total_requests += 1
                    self.total_errors += 1
                    if not it.future.done():
                        it.future.set_result(
                            ToolResult(
                                success=False,
                                error=f"External tool not available (ToolServer not configured): {it.call.name}",
                                uniq_id=it.call.uniq_id,
                            )
                        )
            else:
                results = await asyncio.gather(*[self._execute_external(it) for it in external_items])
                for it, res in zip(external_items, results):
                    self.total_requests += 1
                    if not getattr(res, "success", False):
                        self.total_errors += 1
                    if not it.future.done():
                        it.future.set_result(res)

        if not sandbox_items:
            return

        # Acquire slots for the distinct trajectories in this batch.
        try:
            traj_ids = list({it.trajectory_id for it in sandbox_items})
            slots = await asyncio.gather(*[self._get_or_acquire_slot(tid) for tid in traj_ids])
            slot_by_traj = dict(zip(traj_ids, slots))
        except Exception as e:
            for it in sandbox_items:
                self.total_requests += 1
                self.total_errors += 1
                if not it.future.done():
                    it.future.set_result(
                        ToolResult(
                            success=False,
                            error=f"Failed to acquire slot: {e}",
                            uniq_id=it.call.uniq_id,
                        )
                    )
            return

        # Group by timeout so we don't accidentally make short timeouts wait on long ones.
        by_timeout: Dict[float, List[_QueuedToolRequest]] = {}
        default_timeout = self.backend.default_timeout_s

        for it in sandbox_items:
            t = it.timeout_s
            if t is None:
                t = default_timeout
            if t is None:
                t = 30.0
            by_timeout.setdefault(float(t), []).append(it)

        for timeout_s, items in by_timeout.items():
            requests = []
            dispatched: List[_QueuedToolRequest] = []
            for it in items:
                slot = slot_by_traj[it.trajectory_id]
                tool_name = it.call.name
                args = dict(it.call.arguments)

                # Hermes compatibility: treat `terminal` as an alias of sandbox `bash`.
                if tool_name == "terminal":
                    if args.get("background"):
                        self.total_requests += 1
                        self.total_errors += 1
                        if not it.future.done():
                            it.future.set_result(
                                ToolResult(
                                    success=False,
                                    error="terminal background execution is not supported in sandbox",
                                    uniq_id=it.call.uniq_id,
                                )
                            )
                        continue
                    tool_name = "bash"
                    # `timeout` is handled at the ToolExecutor level, not passed to the sandbox tool args.
                    args.pop("timeout", None)
                elif tool_name == "terminal_stateful":
                    tool_name = "bash_stateful"
                    args.pop("timeout", None)
                elif tool_name == "tmux":
                    # `tmux` is a sandbox tool backed by the stateful session manager.
                    # Network policy is env-controlled.
                    args.pop("allow_network", None)

                if tool_name == "bash":
                    # Network policy is set by the environment/executor, not by the model.
                    args.pop("allow_network", None)
                    args.pop("require_sandbox", None)
                    args["allow_network"] = bool(self.config.allow_network)
                    args["require_sandbox"] = bool(self.config.require_sandbox)
                    # `timeout` is handled at the ToolExecutor level, not passed to the sandbox tool args.
                    args.pop("timeout", None)
                elif tool_name == "bash_stateful":
                    # Network policy is set by the environment/executor, not by the model.
                    args.pop("allow_network", None)
                    args.pop("require_sandbox", None)
                    args.pop("require_stateful_sandbox", None)
                    args["allow_network"] = bool(self.config.allow_network)
                    args["require_stateful_sandbox"] = bool(self.config.require_stateful_sandbox)
                    args.pop("timeout", None)
                elif tool_name == "tmux":
                    # Network policy applies to the underlying stateful session.
                    args.pop("allow_network", None)
                    args.pop("require_sandbox", None)
                    args.pop("require_stateful_sandbox", None)
                    args["allow_network"] = bool(self.config.allow_network)
                    args["require_stateful_sandbox"] = bool(self.config.require_stateful_sandbox)

                requests.append((slot, tool_name, args))
                dispatched.append(it)

            results = None
            try:
                if not dispatched:
                    continue
                results = await self.backend.execute_batch(requests, timeout_s=timeout_s)
            except Exception as e:
                for it in items:
                    self.total_requests += 1
                    self.total_errors += 1
                    if not it.future.done():
                        it.future.set_result(
                            ToolResult(
                                success=False,
                                error=f"Batch execution failed: {e}",
                                uniq_id=it.call.uniq_id,
                            )
                        )
                continue

            for it, res in zip(dispatched, results):
                self.total_requests += 1
                if not getattr(res, "success", False):
                    self.total_errors += 1
                tool_result = res.to_tool_result()
                tool_result.uniq_id = it.call.uniq_id
                if not it.future.done():
                    it.future.set_result(tool_result)
