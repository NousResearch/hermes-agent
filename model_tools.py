#!/usr/bin/env python3
"""
Model Tools Module

Thin orchestration layer over the tool registry. Each tool file in tools/
self-registers its schema, handler, and metadata via tools.registry.register().
This module triggers discovery (by importing all tool modules), then provides
the public API that run_agent.py, cli.py, batch_runner.py, and the RL
environments consume.

Public API (signatures preserved from the original 2,400-line version):
    get_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode) -> list
    handle_function_call(function_name, function_args, task_id, user_task) -> str
    TOOL_TO_TOOLSET_MAP: dict          (for batch_runner.py)
    TOOLSET_REQUIREMENTS: dict         (for cli.py, doctor.py)
    get_all_tool_names() -> list
    get_toolset_for_tool(name) -> str
    get_available_toolsets() -> dict
    check_toolset_requirements() -> dict
    check_tool_availability(quiet) -> tuple
"""

import json
import asyncio
import os
import logging
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================

def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.

    This is the single source of truth for sync->async bridging in tool
    handlers. The RL paths (agent_loop.py, tool_context.py) also provide
    outer thread-pool wrapping as defense-in-depth, but each handler is
    self-protecting via this function.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result(timeout=300)
    return asyncio.run(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

def _discover_tools():
    """Import all tool modules to trigger their registry.register() calls.

    Wrapped in a function so import errors in optional tools (e.g., fal_client
    not installed) don't prevent the rest from loading.
    """
    _modules = [
        "tools.web_tools",
        "tools.terminal_tool",
        "tools.file_tools",
        "tools.vision_tools",
        "tools.mixture_of_agents_tool",
        "tools.image_generation_tool",
        "tools.skills_tool",
        "tools.skill_manager_tool",
        "tools.browser_tool",
        "tools.cronjob_tools",
        "tools.rl_training_tool",
        "tools.tts_tool",
        "tools.todo_tool",
        "tools.memory_tool",
        "tools.session_search_tool",
        "tools.clarify_tool",
        "tools.code_execution_tool",
        "tools.delegate_tool",
        "tools.process_registry",
        "tools.send_message_tool",
        "tools.honcho_tools",
        "tools.homeassistant_tool",
    ]
    import importlib
    for mod_name in _modules:
        try:
            importlib.import_module(mod_name)
        except Exception as e:
            logger.debug("Could not import %s: %s", mod_name, e)


_discover_tools()


# =============================================================================
# Backward-compat constants  (built once after discovery)
# =============================================================================

TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# Resolved tool names from the last get_tool_definitions() call.
# Used by code_execution_tool to know which tools are available in this session.
_last_resolved_tool_names: List[str] = []


# =============================================================================
# Legacy toolset name mapping  (old _tools-suffixed names -> tool name lists)
# =============================================================================

_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "moa_tools": ["mixture_of_agents"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_type", "browser_scroll", "browser_back",
        "browser_press", "browser_close", "browser_get_images",
        "browser_vision"
    ],
    "cronjob_tools": ["schedule_cronjob", "list_cronjobs", "remove_cronjob"],
    "rl_tools": [
        "rl_list_environments", "rl_select_environment",
        "rl_get_current_config", "rl_edit_config",
        "rl_start_training", "rl_check_status",
        "rl_stop_training", "rl_get_results",
        "rl_list_runs", "rl_test_inference"
    ],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# =============================================================================
# get_tool_definitions  (the main schema provider)
# =============================================================================

def get_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get tool definitions for model API calls with toolset-based filtering.

    All tools must be part of a toolset to be accessible.

    Args:
        enabled_toolsets: Only include tools from these toolsets.
        disabled_toolsets: Exclude tools from these toolsets (if enabled_toolsets is None).
        quiet_mode: Suppress status prints.

    Returns:
        Filtered list of OpenAI-format tool definitions.
    """
    # Determine which tool names the caller wants
    tools_to_include: set = set()

    if enabled_toolsets:
        for toolset_name in enabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.update(resolved)
                if not quiet_mode:
                    print(f"✅ Enabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.update(legacy_tools)
                if not quiet_mode:
                    print(f"✅ Enabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            else:
                if not quiet_mode:
                    print(f"⚠️  Unknown toolset: {toolset_name}")

    elif disabled_toolsets:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

        for toolset_name in disabled_toolsets:
            if validate_toolset(toolset_name):
                resolved = resolve_toolset(toolset_name)
                tools_to_include.difference_update(resolved)
                if not quiet_mode:
                    print(f"🚫 Disabled toolset '{toolset_name}': {', '.join(resolved) if resolved else 'no tools'}")
            elif toolset_name in _LEGACY_TOOLSET_MAP:
                legacy_tools = _LEGACY_TOOLSET_MAP[toolset_name]
                tools_to_include.difference_update(legacy_tools)
                if not quiet_mode:
                    print(f"🚫 Disabled legacy toolset '{toolset_name}': {', '.join(legacy_tools)}")
            else:
                if not quiet_mode:
                    print(f"⚠️  Unknown toolset: {toolset_name}")
    else:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # Ask the registry for schemas (only returns tools whose check_fn passes)
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    return filtered_tools


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools whose execution is intercepted by the agent loop (run_agent.py)
# because they need agent-level state (TodoStore, MemoryStore, etc.).
# The registry still holds their schemas; dispatch just returns a stub error
# so if something slips through, the LLM sees a sensible message.
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    user_task: Optional[str] = None,
) -> str:
    """
    Main function call dispatcher that routes calls to the tool registry.

    Args:
        function_name: Name of the function to call.
        function_args: Arguments for the function.
        task_id: Unique identifier for terminal/browser session isolation.
        user_task: The user's original task (for browser_snapshot context).

    Returns:
        Function result as a JSON string.
    """
    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        if function_name == "execute_code":
            return registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                enabled_tools=_last_resolved_tool_names,
            )

        return registry.dispatch(
            function_name, function_args,
            task_id=task_id,
            user_task=user_task,
        )

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)


# =============================================================================
# Backward-compat wrapper functions
# =============================================================================

def get_all_tool_names() -> List[str]:
    """Return all registered tool names."""
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    """Return the toolset a tool belongs to."""
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Return toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """Return {toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """Return (available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
