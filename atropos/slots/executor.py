"""
SandboxExecutor - HTTP client for sandbox container communication.

Sends tool execution requests to sandbox_server.py running inside Nomad containers.
Supports single and batch execution for efficiency.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from .slot import Slot, SlotState
from ..tools.base import ToolCall, ToolResult


@dataclass
class ExecutionRequest:
    """Request to execute a tool in a slot."""
    slot: Slot
    tool_name: str
    args: Dict[str, Any]
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timeout: float = 30.0


@dataclass
class ExecutionResult:
    """Result from sandbox execution."""
    success: bool
    output: str = ""
    error: str = ""
    execution_id: str = ""
    slot_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_tool_result(self) -> ToolResult:
        """Convert to ToolResult for agent consumption."""
        return ToolResult(
            success=self.success,
            output=self.output,
            error=self.error,
            metadata=self.metadata,
            uniq_id=self.execution_id,
        )


class SandboxExecutor:
    """
    HTTP client for executing tools in sandbox containers.
    
    Communicates with sandbox_server.py running inside Nomad allocations.
    Supports both single execution and batched parallel execution.
    
    Usage:
        executor = SandboxExecutor()
        
        # Single execution
        result = await executor.execute(slot, "bash", {"command": "ls"})
        
        # Batch execution
        results = await executor.execute_batch([
            (slot1, "bash", {"command": "ls"}),
            (slot2, "write_file", {"path": "test.txt", "content": "hello"}),
        ])
    """
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
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
            ExecutionResult with output or error
        """
        execution_id = str(uuid.uuid4())
        exec_timeout = timeout or self.timeout.total or 30.0
        
        # Mark slot as executing
        original_state = slot.state
        try:
            if slot.state == SlotState.ACQUIRED:
                slot.start_execution(execution_id)
            
            result = await self._send_execute_request(
                container_addr=slot.container_addr,
                slot_id=slot.slot_id,
                tool_name=tool_name,
                args=args,
                execution_id=execution_id,
                timeout=exec_timeout,
            )
            result.slot_id = slot.slot_id
            return result
            
        finally:
            # Restore slot state
            if slot.state == SlotState.EXECUTING:
                slot.end_execution()
    
    async def _send_execute_request(
        self,
        container_addr: str,
        slot_id: str,
        tool_name: str,
        args: Dict[str, Any],
        execution_id: str,
        timeout: float,
    ) -> ExecutionResult:
        """Send execution request to sandbox server with retry logic."""
        session = await self._get_session()
        url = f"{container_addr}/execute"
        
        payload = {
            "slot_id": slot_id,
            "tool": tool_name,
            "args": args,
            "execution_id": execution_id,
            "timeout": timeout,
        }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                async with session.post(url, json=payload) as response:
                    data = await response.json()
                    
                    return ExecutionResult(
                        success=data.get("success", False),
                        output=data.get("output", ""),
                        error=data.get("error", ""),
                        execution_id=data.get("execution_id", execution_id),
                        metadata=data.get("metadata", {}),
                    )
                    
            except aiohttp.ClientError as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                continue
            except asyncio.TimeoutError:
                last_error = f"Request timed out after {timeout}s"
                break
            except Exception as e:
                last_error = str(e)
                break
        
        return ExecutionResult(
            success=False,
            error=f"Failed after {self.max_retries} attempts: {last_error}",
            execution_id=execution_id,
        )
    
    async def execute_batch(
        self,
        requests: List[Tuple[Slot, str, Dict[str, Any]]],
        timeout: Optional[float] = None,
    ) -> List[ExecutionResult]:
        """
        Execute multiple tools in parallel across slots.
        
        This is the key optimization - we batch tool calls to maximize
        container utilization while agents are waiting for LLM responses.
        
        Args:
            requests: List of (slot, tool_name, args) tuples
            timeout: Optional timeout override
            
        Returns:
            List of ExecutionResults in same order as requests
        """
        if not requests:
            return []
        
        # Group requests by container address for batch API
        by_container: Dict[str, List[Tuple[int, Slot, str, Dict[str, Any], str]]] = {}
        
        for idx, (slot, tool_name, args) in enumerate(requests):
            execution_id = str(uuid.uuid4())
            container = slot.container_addr
            
            if container not in by_container:
                by_container[container] = []
            by_container[container].append((idx, slot, tool_name, args, execution_id))
            
            # Mark slots as executing
            if slot.state == SlotState.ACQUIRED:
                slot.start_execution(execution_id)
        
        # Execute batches in parallel
        exec_timeout = timeout or self.timeout.total or 30.0
        batch_tasks = []
        
        for container_addr, batch_requests in by_container.items():
            task = self._send_batch_request(
                container_addr=container_addr,
                batch_requests=batch_requests,
                timeout=exec_timeout,
            )
            batch_tasks.append(task)
        
        # Gather all batch results
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results in original order
        results: List[Optional[ExecutionResult]] = [None] * len(requests)
        
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                # Mark all in this batch as failed
                continue
            
            for idx, result in batch_result:
                results[idx] = result
        
        # Fill in any missing results
        for idx, result in enumerate(results):
            if result is None:
                slot, tool_name, args = requests[idx]
                results[idx] = ExecutionResult(
                    success=False,
                    error="Batch execution failed",
                    slot_id=slot.slot_id,
                )
        
        # End execution on all slots
        for slot, _, _ in requests:
            if slot.state == SlotState.EXECUTING:
                slot.end_execution()
        
        return results  # type: ignore
    
    async def _send_batch_request(
        self,
        container_addr: str,
        batch_requests: List[Tuple[int, Slot, str, Dict[str, Any], str]],
        timeout: float,
    ) -> List[Tuple[int, ExecutionResult]]:
        """Send batch execution request to a single container."""
        session = await self._get_session()
        url = f"{container_addr}/batch"
        
        # Build batch payload
        payload = [
            {
                "slot_id": slot.slot_id,
                "tool": tool_name,
                "args": args,
                "execution_id": execution_id,
                "timeout": timeout,
            }
            for _, slot, tool_name, args, execution_id in batch_requests
        ]
        
        try:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                
                if not isinstance(data, list):
                    raise ValueError(f"Expected list response, got {type(data)}")
                
                results = []
                for i, (idx, slot, _, _, execution_id) in enumerate(batch_requests):
                    if i < len(data):
                        item = data[i]
                        result = ExecutionResult(
                            success=item.get("success", False),
                            output=item.get("output", ""),
                            error=item.get("error", ""),
                            execution_id=item.get("execution_id", execution_id),
                            slot_id=slot.slot_id,
                            metadata=item.get("metadata", {}),
                        )
                    else:
                        result = ExecutionResult(
                            success=False,
                            error="Missing result in batch response",
                            execution_id=execution_id,
                            slot_id=slot.slot_id,
                        )
                    results.append((idx, result))
                
                return results
                
        except Exception as e:
            # Return error for all requests in batch
            return [
                (idx, ExecutionResult(
                    success=False,
                    error=str(e),
                    execution_id=execution_id,
                    slot_id=slot.slot_id,
                ))
                for idx, slot, _, _, execution_id in batch_requests
            ]
    
    async def reset_slot(self, slot: Slot) -> ExecutionResult:
        """
        Reset a slot's workspace (delete all files).
        
        Useful when reusing a slot for a new trajectory.
        """
        session = await self._get_session()
        url = f"{slot.container_addr}/reset"
        
        try:
            async with session.post(url, json={"slot_id": slot.slot_id}) as response:
                data = await response.json()
                return ExecutionResult(
                    success=data.get("success", False),
                    output=data.get("output", ""),
                    error=data.get("error", ""),
                    slot_id=slot.slot_id,
                )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                slot_id=slot.slot_id,
            )
    
    async def health_check(self, container_addr: str) -> bool:
        """Check if a sandbox container is healthy."""
        session = await self._get_session()
        url = f"{container_addr}/health"
        
        try:
            async with session.get(url) as response:
                data = await response.json()
                return data.get("status") == "ok"
        except Exception:
            return False
    
    async def get_container_status(
        self, 
        container_addr: str
    ) -> Optional[Dict[str, Any]]:
        """Get status info from a sandbox container."""
        session = await self._get_session()
        url = f"{container_addr}/health"
        
        try:
            async with session.get(url) as response:
                return await response.json()
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Artifact helpers (optional)
    # -------------------------------------------------------------------------

    async def _post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        session = await self._get_session()
        try:
            async with session.post(url, json=payload, timeout=timeout) as response:
                data = await response.json()
                if isinstance(data, dict):
                    data.setdefault("http_status", response.status)
                    return data
                return {"success": False, "error": f"Unexpected response type: {type(data)}", "http_status": response.status}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def read_artifact(
        self,
        slot: Slot,
        path: str,
        *,
        encoding: str = "text",
        max_bytes: Optional[int] = None,
        include_sha256: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{slot.container_addr}/artifacts/read"
        payload: Dict[str, Any] = {"slot_id": slot.slot_id, "path": path, "encoding": encoding, "include_sha256": include_sha256}
        if max_bytes is not None:
            payload["max_bytes"] = max_bytes
        return await self._post_json(url, payload, timeout=timeout)

    async def list_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        recursive: bool = False,
        max_entries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{slot.container_addr}/artifacts/list"
        payload: Dict[str, Any] = {"slot_id": slot.slot_id, "path": path, "recursive": recursive}
        if max_entries is not None:
            payload["max_entries"] = max_entries
        return await self._post_json(url, payload, timeout=timeout)

    async def archive_artifacts(
        self,
        slot: Slot,
        path: str = ".",
        *,
        archive_format: str = "tar.gz",
        max_bytes: Optional[int] = None,
        max_entries: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        url = f"{slot.container_addr}/artifacts/archive"
        payload: Dict[str, Any] = {"slot_id": slot.slot_id, "path": path, "format": archive_format}
        if max_bytes is not None:
            payload["max_bytes"] = max_bytes
        if max_entries is not None:
            payload["max_entries"] = max_entries
        return await self._post_json(url, payload, timeout=timeout)
