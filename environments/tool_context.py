"""
ToolContext -- Unrestricted Tool Access for Reward Functions

A per-rollout handle that gives reward/verification functions direct access to
ALL hermes-agent tools, scoped to the rollout's task_id. The same task_id means
the terminal/browser session is the SAME one the model used during its rollout --
all state (files, processes, browser tabs) is preserved.

The verifier author decides which tools to use. Nothing is hardcoded or gated.

Example usage in a compute_reward():
    async def compute_reward(self, item, result, ctx):
        # Run tests in the model's terminal sandbox
        test = ctx.terminal("pytest -v")
        if test["exit_code"] == 0:
            return 1.0

        # Check if a file was created
        content = ctx.read_file("/workspace/solution.py")
        if content.get("content"):
            return 0.5

        return 0.0
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import asyncio
import concurrent.futures

from model_tools import handle_function_call
from tools.terminal_tool import cleanup_vm
from tools.browser_tool import cleanup_browser

logger = logging.getLogger(__name__)

# Thread pool for running sync tool calls that internally use asyncio.run()
_tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_tool_in_thread(tool_name: str, arguments: Dict[str, Any], task_id: str) -> str:
    """
    Run a tool call in a thread pool executor so backends that use asyncio.run()
    internally (modal, docker) get a clean event loop.

    If we're already in an async context, uses run_in_executor.
    If not (e.g., called from sync code), runs directly.
    """
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context -- need to run in thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                handle_function_call, tool_name, arguments, task_id
            )
            return future.result(timeout=300)
    except RuntimeError:
        # No running event loop -- safe to call directly
        return handle_function_call(tool_name, arguments, task_id)


class ToolContext:
    """
    Open-ended access to all hermes-agent tools for a specific rollout.

    Passed to compute_reward() so verifiers can use any tool they need:
    terminal commands, file reads/writes, web searches, browser automation, etc.
    All calls share the rollout's task_id for session isolation.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id

    # -------------------------------------------------------------------------
    # Terminal tools
    # -------------------------------------------------------------------------

    def terminal(self, command: str, timeout: int = 180) -> Dict[str, Any]:
        """
        Run a command in the rollout's terminal session.

        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds

        Returns:
            Dict with 'exit_code' (int) and 'output' (str)
        """
        import os
        backend = os.getenv("TERMINAL_ENV", "local")
        logger.debug("ToolContext.terminal [%s backend] task=%s: %s", backend, self.task_id[:8], command[:100])

        # Run in thread pool so modal/docker backends' asyncio.run() doesn't deadlock
        result = _run_tool_in_thread(
            "terminal",
            {"command": command, "timeout": timeout},
            self.task_id,
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"exit_code": -1, "output": result}

    # -------------------------------------------------------------------------
    # File tools
    # -------------------------------------------------------------------------

    def read_file(self, path: str) -> Dict[str, Any]:
        """
        Read a file from the rollout's filesystem.

        Args:
            path: File path to read

        Returns:
            Dict with file content or error
        """
        result = handle_function_call(
            "read_file", {"path": path}, task_id=self.task_id
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """
        Write a file in the rollout's filesystem.

        Args:
            path: File path to write
            content: Content to write

        Returns:
            Dict with success status or error
        """
        result = handle_function_call(
            "write_file", {"path": path, "content": content}, task_id=self.task_id
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    def search(self, query: str, path: str = ".") -> Dict[str, Any]:
        """
        Search for text in the rollout's filesystem.

        Args:
            query: Search query
            path: Directory to search in

        Returns:
            Dict with search results
        """
        result = handle_function_call(
            "search", {"query": query, "path": path}, task_id=self.task_id
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    # -------------------------------------------------------------------------
    # Web tools
    # -------------------------------------------------------------------------

    def web_search(self, query: str) -> Dict[str, Any]:
        """
        Search the web.

        Args:
            query: Search query

        Returns:
            Dict with search results
        """
        result = handle_function_call("web_search", {"query": query})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    def web_extract(self, urls: List[str]) -> Dict[str, Any]:
        """
        Extract content from URLs.

        Args:
            urls: List of URLs to extract content from

        Returns:
            Dict with extracted content
        """
        result = handle_function_call("web_extract", {"urls": urls})
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    # -------------------------------------------------------------------------
    # Browser tools
    # -------------------------------------------------------------------------

    def browser_navigate(self, url: str) -> Dict[str, Any]:
        """
        Navigate the rollout's browser session to a URL.

        Args:
            url: URL to navigate to

        Returns:
            Dict with page snapshot or error
        """
        result = handle_function_call(
            "browser_navigate", {"url": url}, task_id=self.task_id
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    def browser_snapshot(self) -> Dict[str, Any]:
        """
        Take a snapshot of the current browser page.

        Returns:
            Dict with page content/accessibility snapshot
        """
        result = handle_function_call(
            "browser_snapshot", {}, task_id=self.task_id
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"error": result}

    # -------------------------------------------------------------------------
    # Generic tool access
    # -------------------------------------------------------------------------

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Call any hermes-agent tool by name.

        This is the generic escape hatch -- if a tool doesn't have a convenience
        wrapper above, you can call it directly here.

        Args:
            tool_name: Name of the tool (e.g., "vision_analyze", "skills_list")
            arguments: Dict of arguments for the tool

        Returns:
            Raw JSON string result from the tool
        """
        return _run_tool_in_thread(tool_name, arguments, self.task_id)

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self):
        """
        Release all resources (terminal VMs, browser sessions) for this rollout.

        Called automatically by the base environment via try/finally after
        compute_reward() completes. You generally don't need to call this yourself.
        """
        try:
            cleanup_vm(self.task_id)
        except Exception as e:
            logger.debug("VM cleanup for task %s: %s", self.task_id, e)

        # Suppress browser_tool's noisy debug prints during cleanup.
        # The cleanup still runs (safe), it just doesn't spam the console.
        _prev_quiet = os.environ.get("HERMES_QUIET")
        os.environ["HERMES_QUIET"] = "1"
        try:
            cleanup_browser(self.task_id)
        except Exception as e:
            logger.debug("Browser cleanup for task %s: %s", self.task_id, e)
        finally:
            if _prev_quiet is None:
                os.environ.pop("HERMES_QUIET", None)
            else:
                os.environ["HERMES_QUIET"] = _prev_quiet
