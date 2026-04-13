"""Configuration-driven hooks for Hermes Agent.

This module provides lightweight, declaration-based hooks that complement
the plugin system. Hooks are defined in config.yaml and executed as shell
commands or Python functions at specific lifecycle points.

Inspired by Claude Code's hooks system, but integrated with Hermes'
existing plugin architecture.

Example config.yaml:
    hooks:
      pre_tool_call:
        - matcher: "Bash"
          command: "node ~/.hermes/hooks/rtk-rewrite.js"
          timeout: 5
          description: "RTK token optimization"
        - matcher: "*"
          command: "bash ~/.hermes/hooks/observation.sh"
          async: true

      post_tool_call:
        - matcher: "Bash"
          command: "python ~/.hermes/hooks/log-cmd.py"

      pre_compact:
        - command: "bash ~/.hermes/hooks/save-state.sh"
          description: "Save state before context compression"

Hook commands receive context via stdin as JSON and can modify it by
printing JSON to stdout.
"""

import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class HookConfig:
    """Configuration for a single hook.

    Attributes:
        matcher: Tool name pattern ("*" for all, "Bash|Read" for multiple)
        command: Shell command to execute
        async_: Run asynchronously without blocking
        timeout: Maximum execution time in seconds
        description: Human-readable description for logging
    """
    command: str
    matcher: str = "*"
    async_: bool = field(default=False, repr=False)  # avoid keyword conflict
    timeout: int = 30
    description: str = ""

    def matches(self, tool_name: str) -> bool:
        """Check if this hook applies to a tool.

        Args:
            tool_name: Name of the tool being called

        Returns:
            True if the hook should execute for this tool
        """
        if self.matcher == "*":
            return True
        return tool_name in self.matcher.split("|")


class ConfigHookManager:
    """Manager for configuration-driven hooks.

    Loads hooks from config.yaml and executes them at lifecycle points.
    Hooks can modify context by printing JSON to stdout.
    """

    VALID_HOOK_TYPES = {
        "pre_tool_call",
        "post_tool_call",
        "pre_llm_call",
        "post_llm_call",
        "pre_compact",
        "on_session_start",
        "on_session_end",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dict.

        Args:
            config: Configuration dict with 'hooks' key, or None to load from file
        """
        self._hooks: Dict[str, List[HookConfig]] = {t: [] for t in self.VALID_HOOK_TYPES}

        if config is None:
            config = self._load_config()

        self._load_hooks(config)

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from Hermes config file."""
        try:
            from hermes_cli.config import load_config
            return load_config()
        except Exception as e:
            logger.debug("Could not load config for hooks: %s", e)
            return {}

    def _load_hooks(self, config: Dict[str, Any]) -> None:
        """Parse hook configurations from config dict."""
        hooks_config = config.get("hooks", {})

        for hook_type, hook_list in hooks_config.items():
            if hook_type not in self.VALID_HOOK_TYPES:
                logger.warning("Unknown hook type: %s", hook_type)
                continue

            if not isinstance(hook_list, list):
                logger.warning("Hooks for %s must be a list", hook_type)
                continue

            for hook_def in hook_list:
                if not isinstance(hook_def, dict):
                    continue

                try:
                    hook = HookConfig(
                        command=hook_def["command"],
                        matcher=hook_def.get("matcher", "*"),
                        async_=hook_def.get("async", False),
                        timeout=hook_def.get("timeout", 30),
                        description=hook_def.get("description", ""),
                    )
                    self._hooks[hook_type].append(hook)
                except KeyError as e:
                    logger.warning("Invalid hook config for %s: missing %s", hook_type, e)
                except Exception as e:
                    logger.warning("Failed to parse hook for %s: %s", hook_type, e)

    async def execute(
        self,
        hook_type: str,
        context: Dict[str, Any],
        tool_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute all hooks of a given type.

        Args:
            hook_type: Type of hook to execute (e.g., "pre_tool_call")
            context: Context dict passed to hooks via stdin
            tool_name: Optional tool name for matcher filtering

        Returns:
            Potentially modified context dict
        """
        if hook_type not in self._hooks:
            return context

        hooks = self._hooks[hook_type]

        for hook in hooks:
            # Filter by tool name for tool-related hooks
            if tool_name and not hook.matches(tool_name):
                continue

            try:
                if hook.async_:
                    # Async hooks run in background without blocking
                    asyncio.create_task(self._run_hook(hook, context, wait=False))
                else:
                    # Sync hooks can modify context
                    result = await self._run_hook(hook, context, wait=True)
                    if isinstance(result, dict):
                        # Merge hook result into context
                        context = self._merge_context(context, result)

            except asyncio.TimeoutError:
                logger.warning("Hook timeout: %s", hook.description or hook.command[:50])
            except Exception as e:
                logger.debug("Hook failed: %s - %s", hook.description or hook.command[:50], e)

        return context

    async def _run_hook(
        self,
        hook: HookConfig,
        context: Dict[str, Any],
        wait: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single hook command.

        Args:
            hook: Hook configuration
            context: Context to pass via stdin
            wait: Whether to wait for result

        Returns:
            Parsed JSON result if wait=True, None otherwise
        """
        input_data = json.dumps(context).encode('utf-8')

        try:
            if not wait:
                # Fire and forget: spawn the child process and detach it
                # from the event loop entirely so it never blocks cleanup.
                proc = subprocess.Popen(
                    hook.command,
                    shell=True,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Write stdin in a best-effort way; if the process hasn't
                # read it yet we just close the pipe.
                try:
                    proc.stdin.write(input_data)
                    proc.stdin.close()
                except Exception:
                    pass
                # Reap the child in a daemon thread to avoid zombies,
                # with a timeout so we don't leak threads.
                import threading
                def _reaper(p=proc, t=hook.timeout):
                    try:
                        p.wait(timeout=t)
                    except subprocess.TimeoutExpired:
                        p.kill()
                        p.wait()
                threading.Thread(target=_reaper, daemon=True).start()
                return None

            # Blocking path: use asyncio subprocess with timeout
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait with timeout
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_data),
                timeout=hook.timeout,
            )

            if proc.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='replace')[:200]
                logger.debug("Hook exited %d: %s", proc.returncode, stderr_text)

            # Try to parse result as JSON
            if stdout:
                try:
                    return json.loads(stdout.decode('utf-8'))
                except json.JSONDecodeError:
                    # Non-JSON output is ignored (not merged into context)
                    logger.debug("Hook produced non-JSON output, ignoring")
                    return None

            return None

        except asyncio.TimeoutError:
            logger.warning("Hook timeout after %ds: %s", hook.timeout, hook.command[:50])
            raise
        except Exception as e:
            logger.debug("Hook execution failed: %s", e)
            return None

    def _merge_context(self, original: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """Merge hook result into original context.

        Special handling for common modification patterns:
        - "args" key modifies tool arguments
        - "result" key modifies tool result
        - Other keys are merged directly
        """
        merged = original.copy()

        # Handle argument modification (pre_tool_call)
        if "args" in result and "args" in original:
            merged["args"].update(result["args"])
        elif "args" in result:
            merged["args"] = result["args"]

        # Handle result modification (post_tool_call)
        if "result" in result:
            merged["result"] = result["result"]

        # Merge other keys
        for key, value in result.items():
            if key not in ("args", "result"):
                merged[key] = value

        return merged

    def has_hooks(self, hook_type: str, tool_name: Optional[str] = None) -> bool:
        """Check if any hooks are registered for a type.

        Args:
            hook_type: Type of hook to check
            tool_name: Optional tool name for filtering

        Returns:
            True if matching hooks exist
        """
        if hook_type not in self._hooks:
            return False

        if tool_name is None:
            return len(self._hooks[hook_type]) > 0

        return any(h.matches(tool_name) for h in self._hooks[hook_type])


# Global instance for reuse
_config_hook_manager: Optional[ConfigHookManager] = None


def get_config_hook_manager(config: Optional[Dict[str, Any]] = None) -> ConfigHookManager:
    """Get or create the global ConfigHookManager instance.

    Args:
        config: Optional config dict (uses cached instance if None)

    Returns:
        ConfigHookManager singleton
    """
    global _config_hook_manager

    if config is not None or _config_hook_manager is None:
        _config_hook_manager = ConfigHookManager(config)

    return _config_hook_manager


def invalidate_hook_cache() -> None:
    """Invalidate the cached hook manager (call after config changes)."""
    global _config_hook_manager
    _config_hook_manager = None


def execute_hooks_sync(
    hook_type: str,
    context: Dict[str, Any],
    tool_name: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Synchronous wrapper for hook execution.

    This function safely executes hooks from sync contexts without
    worrying about event loop conflicts. It uses a thread pool to
    run the async hook manager.

    Args:
        hook_type: Type of hook to execute
        context: Context dict passed to hooks
        tool_name: Optional tool name for filtering
        timeout: Maximum time to wait for hooks

    Returns:
        Potentially modified context dict
    """
    hook_mgr = get_config_hook_manager()

    # Fast path: no hooks registered
    if not hook_mgr.has_hooks(hook_type, tool_name):
        return context

    # Use thread pool to avoid event loop issues in sync contexts
    import concurrent.futures

    def run_async_in_thread():
        """Run async hooks in a separate thread with its own event loop."""
        # Each thread gets its own event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                hook_mgr.execute(hook_type, context, tool_name)
            )
        finally:
            loop.close()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_async_in_thread)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("Hooks timed out after %ds", timeout)
        return context
    except Exception as e:
        logger.debug("Hook execution failed: %s", e)
        return context
