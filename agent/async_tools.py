"""Async tool execution wrapper for Hermes Agent.

Converts synchronous tool calls into async coroutines that can run
concurrently via ``asyncio.gather()``.  This is the foundation for
Issue 3.1.x (async core) — all I/O-bound tools should be wrapped here.

Usage
-----
```python
from agent.async_tools import async_tool_call, run_concurrent_tools

# Single async tool call
result = await async_tool_call("web_search", {"query": "..."})

# Concurrent tool calls (all run in parallel)
results = await run_concurrent_tools([
    ("web_search", {"query": "topic A"}),
    ("web_search", {"query": "topic B"}),
    ("web_extract", {"url": "https://..."}),
])
```

Tool Dependency Resolution
--------------------------
When tools have dependencies (e.g. ``web_extract`` needs the URL from
``web_search`` first), use ``run_tool_chain()`` which respects the
dependency graph while maximising parallelism.
"""

import asyncio
import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Thread pool for running sync tools in async context
_tool_executor: Optional[ThreadPoolExecutor] = None
_MAX_WORKERS = 8  # cap concurrent tool executions


def _get_executor() -> ThreadPoolExecutor:
    """Get or create the tool executor thread pool."""
    global _tool_executor
    if _tool_executor is None:
        _tool_executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS,
            thread_name_prefix="hermes-tool",
        )
    return _tool_executor


def shutdown_executor() -> None:
    """Shutdown the tool executor (call during agent cleanup)."""
    global _tool_executor
    if _tool_executor is not None:
        _tool_executor.shutdown(wait=False)
        _tool_executor = None


async def async_tool_call(
    tool_name: str,
    args: dict[str, Any],
    *,
    handler: Optional[Callable] = None,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Execute a tool call asynchronously.

    Parameters
    ----------
    tool_name:
        Name of the tool to execute.
    args:
        Arguments to pass to the tool.
    handler:
        The actual tool handler function.  If None, the tool is looked up
        from the global tool registry.
    timeout:
        Maximum seconds to wait for the tool to complete.

    Returns
    -------
    dict
        The tool's JSON-parsed result, or an error dict on failure.
    """
    if handler is None:
        # Lazy import to avoid circular deps
        from model_tools import _TOOL_HANDLERS
        handler = _TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return {
                "error": f"Unknown tool: {tool_name}",
                "tool": tool_name,
            }

    loop = asyncio.get_running_loop()
    executor = _get_executor()

    # Wrap the handler to run in the thread pool
    wrapped = functools.partial(handler, args)

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, wrapped),
            timeout=timeout,
        )
        # Parse JSON if string
        if isinstance(result, str):
            import json
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"output": result}
        return result if isinstance(result, dict) else {"output": str(result)}

    except asyncio.TimeoutError:
        logger.warning("Tool %s timed out after %.1fs", tool_name, timeout)
        return {
            "error": f"Tool '{tool_name}' timed out after {timeout}s",
            "tool": tool_name,
            "timed_out": True,
        }
    except Exception as e:
        logger.error("Tool %s failed: %s", tool_name, e, exc_info=True)
        return {
            "error": f"Tool '{tool_name}' failed: {type(e).__name__}: {e}",
            "tool": tool_name,
        }


async def run_concurrent_tools(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    handlers: Optional[dict[str, Callable]] = None,
    timeout: float = 300.0,
) -> list[dict[str, Any]]:
    """Execute multiple tools concurrently.

    Parameters
    ----------
    calls:
        List of (tool_name, args) tuples.
    handlers:
        Optional dict mapping tool names to handler functions.
    timeout:
        Maximum seconds for all tools to complete.

    Returns
    -------
    list[dict]
        Results in the same order as the input calls.
    """
    if not calls:
        return []

    handlers = handlers or {}

    async def _call(name: str, args: dict) -> dict:
        return await async_tool_call(name, args, handler=handlers.get(name), timeout=timeout)

    tasks = [_call(name, args) for name, args in calls]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def run_tool_chain(
    stages: list[list[tuple[str, dict[str, Any]]]],
    *,
    handlers: Optional[dict[str, Callable]] = None,
    timeout: float = 300.0,
) -> list[list[dict[str, Any]]]:
    """Execute a chain of tool stages with intra-stage concurrency.

    Each stage is a list of tool calls that run in parallel.  Stages
    execute sequentially, so stage N+1 can depend on stage N's results.

    Parameters
    ----------
    stages:
        List of stages, where each stage is a list of (tool_name, args) tuples.
    handlers:
        Optional handler overrides.
    timeout:
        Maximum seconds per stage.

    Returns
    -------
    list[list[dict]]
        Results grouped by stage.

    Example
    -------
    >>> stages = [
    ...     [("web_search", {"query": "A"}), ("web_search", {"query": "B"})],
    ...     [("web_extract", {"url": "<url from stage 0>"})],
    ... ]
    >>> results = await run_tool_chain(stages)
    """
    all_results: list[list[dict[str, Any]]] = []

    for stage_idx, stage in enumerate(stages):
        start = time.monotonic()
        stage_results = await run_concurrent_tools(stage, handlers=handlers, timeout=timeout)
        elapsed = time.monotonic() - start
        logger.info(
            "Tool chain stage %d/%d: %d tools in %.2fs",
            stage_idx + 1, len(stages), len(stage), elapsed,
        )
        all_results.append(stage_results)

    return all_results
