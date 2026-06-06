#!/usr/bin/env python3
"""
Executor health checking utilities.

Each executor adapter implements ``check_health()`` on its own.
This module provides helpers for:
  - Checking command presence via ``shutil.which``
  - Checking project path accessibility
  - Bulk health checking across the registry
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from executors.types import (
    ExecutorId,
    ExecutorHealthResult,
    ExecutorHealthStatus,
    ExecutorManifest,
)

if TYPE_CHECKING:
    from executors.registry import ExecutorRegistry

logger = logging.getLogger(__name__)


async def check_command_exists(command: str) -> Tuple[bool, Optional[str]]:
    """Check if a command exists on PATH. Returns (found, path)."""
    path = shutil.which(command)
    if path:
        return True, path
    return False, None


async def check_version(command: str, version_args: str = "--version") -> Optional[str]:
    """Try to get version string from a command.

    Returns the first line of stdout on success, or None on failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            command, version_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=10.0
        )
        if proc.returncode == 0 and stdout:
            return stdout.decode("utf-8", errors="replace").strip().split("\n")[0]
        return None
    except Exception as e:
        logger.debug("Version check failed for %s: %s", command, e)
        return None


async def check_project_path(project_path: Optional[Path]) -> Tuple[bool, Optional[str]]:
    """Check if a project path exists and is accessible."""
    if project_path is None:
        return True, None  # no path to check = default OK

    path = Path(project_path)
    if not path.exists():
        return False, f"Path does not exist: {path}"
    if not path.is_dir():
        return False, f"Path is not a directory: {path}"
    # Basic accessibility check
    try:
        path.stat()
        return True, None
    except PermissionError:
        return False, f"Permission denied: {path}"
    except OSError as e:
        return False, f"OS error: {e}"


def make_unknown_health(executor_id: ExecutorId) -> ExecutorHealthResult:
    """Create a health result with UNKNOWN status."""
    return ExecutorHealthResult(
        executor_id=executor_id,
        status=ExecutorHealthStatus.UNKNOWN,
    )


def make_available_health(
    executor_id: ExecutorId,
    version: Optional[str] = None,
) -> ExecutorHealthResult:
    """Create a health result with AVAILABLE status."""
    return ExecutorHealthResult(
        executor_id=executor_id,
        status=ExecutorHealthStatus.AVAILABLE,
        version=version,
    )


def make_unavailable_health(
    executor_id: ExecutorId,
    error: str,
) -> ExecutorHealthResult:
    """Create a health result with UNAVAILABLE status."""
    return ExecutorHealthResult(
        executor_id=executor_id,
        status=ExecutorHealthStatus.UNAVAILABLE,
        error=error,
    )


async def check_executor_health(
    executor_id: ExecutorId,
    command: str,
    version_args: str = "--version",
) -> ExecutorHealthResult:
    """Generic health check: verify command exists and get version.

    Args:
        executor_id: Executor identifier for the result.
        command: The binary name to look up (e.g., "claude-code", "codex").
        version_args: Args to pass for version check.

    Returns:
        ExecutorHealthResult with status and optional version/error.
    """
    found, cmd_path = await check_command_exists(command)
    if not found:
        return make_unavailable_health(
            executor_id,
            f"Command not found: '{command}' (checked PATH)",
        )

    version = await check_version(command, version_args)
    if version:
        return make_available_health(executor_id, version=version)

    # Command exists but version check failed — still mark as available
    # (the version check might fail due to non-standard flags)
    return make_available_health(
        executor_id,
        version=f"[{cmd_path}]",
    )


async def check_all_executors_health(
    registry: "ExecutorRegistry",
) -> Dict[ExecutorId, ExecutorHealthResult]:
    """Run health checks on all registered executors.

    Calls each adapter's ``check_health()`` if implemented, otherwise falls
    back to UNKNOWN.

    Returns:
        Dict mapping executor_id to health result.
    """
    results: Dict[ExecutorId, ExecutorHealthResult] = {}

    async def _check_one(m: ExecutorManifest):
        try:
            adapter = registry.get(m.id)
            if hasattr(adapter, "check_health"):
                result = await adapter.check_health()
            else:
                result = make_unknown_health(m.id)
        except Exception as e:
            logger.warning("Health check failed for %s: %s", m.id, e)
            result = make_unavailable_health(m.id, str(e))

        registry.set_health(result)
        return m.id, result

    manifests = registry.list_executors()
    tasks = [_check_one(m) for m in manifests]
    gathered = await asyncio.gather(*tasks, return_exceptions=True)

    for item in gathered:
        if isinstance(item, Exception):
            logger.warning("Health check task raised: %s", item)
        else:
            eid, result = item
            results[eid] = result

    return results
