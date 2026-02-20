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
        Write a TEXT file in the rollout's filesystem.

        Uses a shell heredoc under the hood, so this is only safe for text content.
        For binary files (images, compiled artifacts, etc.), use upload_file() instead.

        Args:
            path: File path to write
            content: Text content to write

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

    def upload_file(self, local_path: str, remote_path: str) -> Dict[str, Any]:
        """
        Upload a local file to the rollout's sandbox (binary-safe).

        Unlike write_file() which passes content through a shell heredoc (text-only),
        this method base64-encodes the file and decodes it inside the sandbox.
        Safe for any file type: binaries, images, archives, etc.

        For large files (>1MB), the content is split into chunks to avoid
        hitting shell command-length limits.

        Args:
            local_path: Path to a local file on the host
            remote_path: Destination path inside the sandbox

        Returns:
            Dict with 'exit_code' and 'output'
        """
        import base64
        from pathlib import Path as _Path

        local = _Path(local_path)
        if not local.exists():
            return {"exit_code": -1, "output": f"Local file not found: {local_path}"}

        raw = local.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")

        # Ensure parent directory exists in the sandbox
        parent = str(_Path(remote_path).parent)
        if parent not in (".", "/"):
            self.terminal(f"mkdir -p {parent}", timeout=10)

        # For small files, single command is fine
        chunk_size = 60_000  # ~60KB per chunk (well within shell limits)
        if len(b64) <= chunk_size:
            result = self.terminal(
                f"printf '%s' '{b64}' | base64 -d > {remote_path}",
                timeout=30,
            )
        else:
            # For larger files, write base64 in chunks then decode
            tmp_b64 = "/tmp/_hermes_upload.b64"
            self.terminal(f": > {tmp_b64}", timeout=5)  # truncate
            for i in range(0, len(b64), chunk_size):
                chunk = b64[i : i + chunk_size]
                self.terminal(f"printf '%s' '{chunk}' >> {tmp_b64}", timeout=15)
            result = self.terminal(
                f"base64 -d {tmp_b64} > {remote_path} && rm -f {tmp_b64}",
                timeout=30,
            )

        return result

    def upload_dir(self, local_dir: str, remote_dir: str) -> List[Dict[str, Any]]:
        """
        Upload an entire local directory to the rollout's sandbox (binary-safe).

        Recursively uploads all files, preserving directory structure.

        Args:
            local_dir: Path to a local directory on the host
            remote_dir: Destination directory inside the sandbox

        Returns:
            List of results, one per file uploaded
        """
        from pathlib import Path as _Path

        local = _Path(local_dir)
        if not local.exists() or not local.is_dir():
            return [{"exit_code": -1, "output": f"Local directory not found: {local_dir}"}]

        results = []
        for file_path in sorted(local.rglob("*")):
            if file_path.is_file():
                relative = file_path.relative_to(local)
                target = f"{remote_dir}/{relative}"
                results.append(self.upload_file(str(file_path), target))
        return results

    def download_file(self, remote_path: str, local_path: str) -> Dict[str, Any]:
        """
        Download a file from the rollout's sandbox to the host (binary-safe).

        The inverse of upload_file(). Base64-encodes the file inside the sandbox,
        reads the encoded data through the terminal, and decodes it locally.
        Safe for any file type.

        Args:
            remote_path: Path to the file inside the sandbox
            local_path: Destination path on the host

        Returns:
            Dict with 'success' (bool) and 'bytes' (int) or 'error' (str)
        """
        import base64
        from pathlib import Path as _Path

        # Base64-encode the file inside the sandbox and capture output
        result = self.terminal(
            f"base64 {remote_path} 2>/dev/null",
            timeout=30,
        )

        if result.get("exit_code", -1) != 0:
            return {
                "success": False,
                "error": f"Failed to read remote file: {result.get('output', '')}",
            }

        b64_data = result.get("output", "").strip()
        if not b64_data:
            return {"success": False, "error": f"Remote file is empty or missing: {remote_path}"}

        try:
            raw = base64.b64decode(b64_data)
        except Exception as e:
            return {"success": False, "error": f"Base64 decode failed: {e}"}

        # Write to local host filesystem
        local = _Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(raw)

        return {"success": True, "bytes": len(raw)}

    def download_dir(self, remote_dir: str, local_dir: str) -> List[Dict[str, Any]]:
        """
        Download a directory from the rollout's sandbox to the host (binary-safe).

        Lists all files in the remote directory, then downloads each one.
        Preserves directory structure.

        Args:
            remote_dir: Path to the directory inside the sandbox
            local_dir: Destination directory on the host

        Returns:
            List of results, one per file downloaded
        """
        from pathlib import Path as _Path

        # List files in the remote directory
        ls_result = self.terminal(
            f"find {remote_dir} -type f 2>/dev/null",
            timeout=15,
        )

        if ls_result.get("exit_code", -1) != 0:
            return [{"success": False, "error": f"Failed to list remote dir: {remote_dir}"}]

        file_list = ls_result.get("output", "").strip()
        if not file_list:
            return [{"success": False, "error": f"Remote directory is empty or missing: {remote_dir}"}]

        results = []
        for remote_file in file_list.splitlines():
            remote_file = remote_file.strip()
            if not remote_file:
                continue
            # Compute the relative path to preserve directory structure
            if remote_file.startswith(remote_dir):
                relative = remote_file[len(remote_dir):].lstrip("/")
            else:
                relative = _Path(remote_file).name
            local_file = str(_Path(local_dir) / relative)
            results.append(self.download_file(remote_file, local_file))

        return results

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
            "search_files", {"pattern": query, "path": path}, task_id=self.task_id
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
        Release all resources (terminal VMs, browser sessions, background processes)
        for this rollout.

        Called automatically by the base environment via try/finally after
        compute_reward() completes. You generally don't need to call this yourself.
        """
        # Kill any background processes from this rollout (safety net)
        try:
            from tools.process_registry import process_registry
            killed = process_registry.kill_all(task_id=self.task_id)
            if killed:
                logger.debug("Process cleanup for task %s: killed %d process(es)", self.task_id, killed)
        except Exception as e:
            logger.debug("Process cleanup for task %s: %s", self.task_id, e)

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
