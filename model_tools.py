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
import logging
import re
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from tools.registry import discover_builtin_tools, registry
from toolsets import resolve_toolset, validate_toolset

logger = logging.getLogger(__name__)


# =============================================================================
# Async Bridging  (single source of truth -- used by registry.dispatch too)
# =============================================================================

_tool_loop = None          # persistent loop for the main (CLI) thread
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # per-worker-thread persistent loops


def _get_tool_loop():
    """Return a long-lived event loop for running async tool handlers.

    Using a persistent loop (instead of asyncio.run() which creates and
    *closes* a fresh loop every time) prevents "Event loop is closed"
    errors that occur when cached httpx/AsyncOpenAI clients attempt to
    close their transport on a dead loop during garbage collection.
    """
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Return a persistent event loop for the current worker thread.

    Each worker thread (e.g., delegate_task's ThreadPoolExecutor threads)
    gets its own long-lived loop stored in thread-local storage.  This
    prevents the "Event loop is closed" errors that occurred when
    asyncio.run() was used per-call: asyncio.run() creates a loop, runs
    the coroutine, then *closes* the loop — but cached httpx/AsyncOpenAI
    clients remain bound to that now-dead loop and raise RuntimeError
    during garbage collection or subsequent use.

    By keeping the loop alive for the thread's lifetime, cached clients
    stay valid and their cleanup runs on a live loop.
    """
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run an async coroutine from a sync context.

    If the current thread already has a running event loop (e.g., inside
    the gateway's async stack or Atropos's event loop), we spin up a
    disposable thread so asyncio.run() can create its own loop without
    conflicting.

    For the common CLI path (no running loop), we use a persistent event
    loop so that cached async clients (httpx / AsyncOpenAI) remain bound
    to a live loop and don't trigger "Event loop is closed" on GC.

    When called from a worker thread (parallel tool execution), we use a
    per-thread persistent loop to avoid both contention with the main
    thread's shared loop AND the "Event loop is closed" errors caused by
    asyncio.run()'s create-and-destroy lifecycle.

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
        # Inside an async context (gateway, RL env) — run in a fresh thread
        # with its own event loop we own a reference to, so on timeout we
        # can cancel the task inside that loop (ThreadPoolExecutor.cancel()
        # only works on not-yet-started futures — it's a no-op on a running
        # worker, which previously leaked the thread on every 300 s timeout).
        import concurrent.futures

        worker_loop: Optional[asyncio.AbstractEventLoop] = None
        loop_ready = threading.Event()

        def _run_in_worker():
            nonlocal worker_loop
            worker_loop = asyncio.new_event_loop()
            loop_ready.set()
            try:
                asyncio.set_event_loop(worker_loop)
                return worker_loop.run_until_complete(coro)
            finally:
                try:
                    # Cancel anything still pending (e.g. task cancelled
                    # externally via call_soon_threadsafe on timeout).
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_run_in_worker)
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # Cancel the coroutine inside its own loop so the worker thread
            # can wind down instead of running forever.
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    # Loop already closed — nothing to cancel.
                    pass
            raise
        finally:
            # wait=False: don't block the caller on a stuck coroutine. We've
            # already requested cancellation above; the worker will exit
            # once the coroutine observes it (usually at the next await).
            pool.shutdown(wait=False)

    # If we're on a worker thread (e.g., parallel tool execution in
    # delegate_task), use a per-thread persistent loop.  This avoids
    # contention with the main thread's shared loop while keeping cached
    # httpx/AsyncOpenAI clients bound to a live loop for the thread's
    # lifetime — preventing "Event loop is closed" on GC cleanup.
    if threading.current_thread() is not threading.main_thread():
        worker_loop = _get_worker_loop()
        return worker_loop.run_until_complete(coro)

    tool_loop = _get_tool_loop()
    return tool_loop.run_until_complete(coro)


# =============================================================================
# Tool Discovery  (importing each module triggers its registry.register calls)
# =============================================================================

discover_builtin_tools()

# MCP tool discovery (external MCP servers from config) used to run here as
# a module-level side effect.  It was removed because discover_mcp_tools()
# internally uses a blocking future.result(timeout=120) wait, and the
# gateway lazy-imports this module from inside the asyncio event loop on
# the first user message — freezing Discord/Telegram heartbeats for up to
# 120s whenever any configured MCP server was slow or unreachable (#16856).
#
# Each entry point now runs discovery explicitly at its own startup:
#   - gateway/run.py            -> start_gateway() uses run_in_executor
#   - cli.py, hermes_cli/*      -> inline on startup (no event loop)
#   - tui_gateway/server.py     -> inline on startup (no event loop)
#   - acp_adapter/server.py     -> asyncio.to_thread on session init

# Plugin tool discovery (user/project/pip plugins)
try:
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


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
        "browser_press", "browser_get_images",
        "browser_vision", "browser_console"
    ],
    "cronjob_tools": ["cronjob"],
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

# Module-level memoization for get_tool_definitions(). Keyed on
# (frozenset(enabled_toolsets), frozenset(disabled_toolsets), registry._generation).
# Hot callers (gateway runner, AIAgent.__init__) invoke this on every turn
# with quiet_mode=True; caching avoids ~7 ms of registry walking + schema
# filtering + check_fn probing per call. Only active when quiet_mode=True
# because quiet_mode=False has stdout side effects (tool-selection prints).
#
# Invalidation happens transparently via the registry's _generation counter,
# which bumps on register() / deregister() / register_toolset_alias(). The
# inner check_fn TTL cache in registry.py handles environment drift (Docker
# daemon start/stop, env var changes, etc.) on a 30 s horizon.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}


def _clear_tool_defs_cache() -> None:
    """Drop memoized get_tool_definitions() results. Called when dynamic
    schema dependencies change (e.g. discord capability cache reset,
    execute_code sandbox reconfigured)."""
    _tool_defs_cache.clear()


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
    # Fast path: memoized result when the caller doesn't need stdout prints.
    # The cache key captures every argument-level input; the registry
    # generation captures registry mutations (MCP refresh, plugin load).
    # check_fn results are TTL-cached one level down, inside
    # registry.get_definitions. The config-mtime fingerprint below captures
    # user-visible config edits that affect dynamic schemas (execute_code
    # mode, discord action allowlist, etc.) without needing an explicit
    # invalidate hook on every config-writer.
    if quiet_mode:
        try:
            from hermes_cli.config import get_config_path
            cfg_path = get_config_path()
            cfg_stat = cfg_path.stat()
            cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
        except (FileNotFoundError, OSError, ImportError):
            cfg_fp = None
        cache_key = (
            frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
            frozenset(disabled_toolsets) if disabled_toolsets else None,
            registry._generation,
            cfg_fp,
        )
        cached = _tool_defs_cache.get(cache_key)
        if cached is not None:
            # Update _last_resolved_tool_names so downstream callers see
            # consistent state even on a cache hit.
            global _last_resolved_tool_names
            _last_resolved_tool_names = [t["function"]["name"] for t in cached]
            # Return a shallow copy of the list but share the dict references —
            # schemas are treated as read-only by all known callers.
            return list(cached)

    result = _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode)
    if quiet_mode:
        # Cache the freshly-computed list, but hand callers a shallow copy so
        # downstream mutations (e.g. run_agent appending memory/LCM tool
        # schemas to self.tools) don't poison the cache. Without this, a
        # long-lived Gateway process accumulates duplicate tool names across
        # agent inits and providers that enforce unique tool names
        # (DeepSeek, Xiaomi MiMo, Moonshot Kimi) reject the request with
        # HTTP 400. Mirrors the cache-hit path above. (issue #17335)
        _tool_defs_cache[cache_key] = result
        return list(result)
    return result


def _compute_tool_definitions(
    enabled_toolsets: List[str] = None,
    disabled_toolsets: List[str] = None,
    quiet_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`."""
    # Determine which tool names the caller wants
    tools_to_include: set = set()

    if enabled_toolsets is not None:
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
    else:
        # Default: start with everything
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools_to_include.update(resolve_toolset(ts_name))

    # Always apply disabled toolsets as a subtraction step at the end.
    # This ensures that even if a composite toolset (like hermes-cli)
    # is enabled, any tools belonging to a disabled toolset are strictly
    # stripped out. See issue #17309.
    if disabled_toolsets:
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

    # Plugin-registered tools are now resolved through the normal toolset
    # path — validate_toolset() / resolve_toolset() / get_all_toolsets()
    # all check the tool registry for plugin-provided toolsets.  No bypass
    # needed; plugins respect enabled_toolsets / disabled_toolsets like any
    # other toolset.

    # Ask the registry for schemas (only returns tools whose check_fn passes)
    filtered_tools = registry.get_definitions(tools_to_include, quiet=quiet_mode)

    # The set of tool names that actually passed check_fn filtering.
    # Use this (not tools_to_include) for any downstream schema that references
    # other tools by name — otherwise the model sees tools mentioned in
    # descriptions that don't actually exist, and hallucinates calls to them.
    available_tool_names = {t["function"]["name"] for t in filtered_tools}

    # Rebuild execute_code schema to only list sandbox tools that are actually
    # available.  Without this, the model sees "web_search is available in
    # execute_code" even when the API key isn't configured or the toolset is
    # disabled (#560-discord).
    if "execute_code" in available_tool_names:
        from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
        sandbox_enabled = SANDBOX_ALLOWED_TOOLS & available_tool_names
        dynamic_schema = build_execute_code_schema(sandbox_enabled, mode=_get_execution_mode())
        for i, td in enumerate(filtered_tools):
            if td.get("function", {}).get("name") == "execute_code":
                filtered_tools[i] = {"type": "function", "function": dynamic_schema}
                break

    # Rebuild discord / discord_admin schemas based on the bot's privileged
    # intents (detected from GET /applications/@me) and the user's action
    # allowlist in config.  Hides actions the bot's intents don't support so
    # the model never attempts them, and annotates fetch_messages when the
    # MESSAGE_CONTENT intent is missing.
    _discord_schema_fns = {
        "discord": "get_dynamic_schema_core",
        "discord_admin": "get_dynamic_schema_admin",
    }
    for discord_tool_name in _discord_schema_fns:
        if discord_tool_name in available_tool_names:
            try:
                from tools import discord_tool as _dt
                schema_fn = getattr(_dt, _discord_schema_fns[discord_tool_name])
                dynamic = schema_fn()
            except Exception:
                dynamic = None
            if dynamic is None:
                filtered_tools = [
                    t for t in filtered_tools
                    if t.get("function", {}).get("name") != discord_tool_name
                ]
                available_tool_names.discard(discord_tool_name)
            else:
                for i, td in enumerate(filtered_tools):
                    if td.get("function", {}).get("name") == discord_tool_name:
                        filtered_tools[i] = {"type": "function", "function": dynamic}
                        break

    # Strip web tool cross-references from browser_navigate description when
    # web_search / web_extract are not available.  The static schema says
    # "prefer web_search or web_extract" which causes the model to hallucinate
    # those tools when they're missing.
    if "browser_navigate" in available_tool_names:
        web_tools_available = {"web_search", "web_extract"} & available_tool_names
        if not web_tools_available:
            for i, td in enumerate(filtered_tools):
                if td.get("function", {}).get("name") == "browser_navigate":
                    desc = td["function"].get("description", "")
                    desc = desc.replace(
                        " For simple information retrieval, prefer web_search or web_extract (faster, cheaper).",
                        "",
                    )
                    filtered_tools[i] = {
                        "type": "function",
                        "function": {**td["function"], "description": desc},
                    }
                    break

    if not quiet_mode:
        if filtered_tools:
            tool_names = [t["function"]["name"] for t in filtered_tools]
            print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(tool_names)}")
        else:
            print("🛠️  No tools selected (all filtered out or unavailable)")

    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    # Sanitize schemas for broad backend compatibility. llama.cpp's
    # json-schema-to-grammar converter (used by its OAI server to build
    # GBNF tool-call parsers) rejects some shapes that cloud providers
    # silently accept — bare "type": "object" with no properties,
    # string-valued schema nodes from malformed MCP servers, etc. This
    # is a no-op for schemas that are already well-formed.
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    return filtered_tools


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Tools whose execution is intercepted by the agent loop (run_agent.py)
# because they need agent-level state (TodoStore, MemoryStore, etc.).
# The registry still holds their schemas; dispatch just returns a stub error
# so if something slips through, the LLM sees a sensible message.
_AGENT_LOOP_TOOLS = {"todo", "memory", "session_search", "delegate_task"}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# =========================================================================
# Tool Input Repair Layer
#
# Validates-then-repairs model tool arguments for common structural and
# semantic errors produced by open-weight LLMs (DeepSeek, Qwen, GLM, etc.).
# Prevents the "repeat the same broken call forever" failure mode.
#
# Design:
#   1. validate-then-repair — pass through unchanged if valid
#   2. 4 basic structural fixers + semantic per-tool fixers
#   3. No jsonschema dependency — all hand-written rules
#   4. repair_log is injected into tool result for model feedback
# =========================================================================

# ── Sentinel for "remove this parameter entirely" ──
_UNSET = object()


# ── Basic structural repair functions ──

def _repair_strip_null(value, schema, param_name):
    """If a parameter is optional and value is None → _UNSET (remove it)."""
    if value is None:
        required = schema.get("_required", False) if isinstance(schema, dict) else False
        if not required:
            return _UNSET
    return value


def _repair_parse_json_array(value, schema, param_name):
    """If schema expects an array but value is a JSON-stringified array → parse it."""
    if not isinstance(value, str):
        return value
    expected_type = schema.get("type") if isinstance(schema, dict) else None
    if expected_type != "array" and (not isinstance(expected_type, list) or "array" not in expected_type):
        return value
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, TypeError):
            pass
    return value


def _repair_unwrap_empty_object(value, schema, param_name):
    """If schema expects an array and value is {} → [] (common LLM error)."""
    if value == {}:
        expected_type = schema.get("type") if isinstance(schema, dict) else None
        if expected_type == "array" or (isinstance(expected_type, list) and "array" in expected_type):
            return []
    return value


def _repair_wrap_bare_string(value, schema, param_name):
    """If schema expects string[] and value is a bare string → wrap in list."""
    if not isinstance(value, str):
        return value
    expected_type = schema.get("type") if isinstance(schema, dict) else None
    items_schema = schema.get("items", {}) if isinstance(schema, dict) else {}
    items_type = items_schema.get("type") if isinstance(items_schema, dict) else None
    if expected_type == "array" and items_type == "string":
        stripped = value.strip()
        if not stripped.startswith("["):
            return [stripped]
    return value


# ── Repair rules registry (priority-ordered) ──
_REPAIR_RULES = [
    ("stripNull", _repair_strip_null),
    ("parseJsonArray", _repair_parse_json_array),
    ("unwrapEmptyObject", _repair_unwrap_empty_object),
    ("wrapBareString", _repair_wrap_bare_string),
]


def _repair_tool_args(tool_name: str, args: Dict[str, Any]) -> tuple:
    """Apply structural repair rules to tool arguments.

    Returns:
        (repaired_args, repair_log_or_None)
    """
    if not args or not isinstance(args, dict):
        return args, None

    schema = registry.get_schema(tool_name)
    if not schema:
        return args, None

    properties = (schema.get("parameters") or {}).get("properties")
    required = (schema.get("parameters") or {}).get("required", [])
    if not properties:
        return args, None

    repair_log = None
    repaired = dict(args)

    for key, value in list(repaired.items()):
        prop_schema = properties.get(key)
        if prop_schema is None:
            continue

        # Inject required info so repair functions can use it
        if isinstance(prop_schema, dict):
            prop_schema = {**prop_schema, "_required": key in required}

        for rule_name, rule_fn in _REPAIR_RULES:
            try:
                repaired_value = rule_fn(repaired.get(key, _UNSET), prop_schema, key)
            except Exception:
                continue

            if repaired_value is _UNSET:
                if repair_log is None:
                    repair_log = []
                repair_log.append({
                    "param": key,
                    "from": value,
                    "to": None,
                    "repair": rule_name,
                })
                del repaired[key]
                break
            elif repaired_value is not repaired.get(key):
                if repair_log is None:
                    repair_log = []
                repair_log.append({
                    "param": key,
                    "from": value,
                    "to": repaired_value,
                    "repair": rule_name,
                })
                repaired[key] = repaired_value
                break

    return repaired, repair_log


# ── Semantic repair functions ──

_MD_LINK_RE = re.compile(r'^\[([^\]]+)\]\([^)]*\)$')
_PATH_TRAVERSAL_RE = re.compile(r'(\.\./|\.\.\\)')


def _repair_markdown_link(value):
    """Detect ``[real_path](display_text)`` pattern and extract real_path."""
    if not isinstance(value, str):
        return value
    m = _MD_LINK_RE.match(value.strip())
    if m:
        return m.group(1).strip()
    return value


def _repair_expand_user(value):
    """Expand ``~`` to the user's home directory."""
    if not isinstance(value, str):
        return value
    if value.startswith("~"):
        import os
        return os.path.expanduser(value)
    return value


def _repair_check_path_traversal(value):
    """Check for path traversal patterns and log a warning if found."""
    if not isinstance(value, str):
        return value
    if _PATH_TRAVERSAL_RE.search(value):
        logger.warning("Path traversal pattern detected: %s", repr(value[:200]))
    return value


def _repair_strip_shell_prompt(value):
    """Strip shell prompt prefixes (``$ ``, ``> ``, etc.) from values."""
    if not isinstance(value, str):
        return value
    stripped = value.lstrip()
    for prefix in ("$ ", "> ", "% ", "# "):
        if stripped.startswith(prefix):
            return stripped[len(prefix):]
    return value


def _repair_fix_double_escape(value):
    """Fix double-escaped backslashes in command strings (2 backslashes → 1)."""
    if not isinstance(value, str):
        return value
    return value.replace('\\\\', '\\')


# ── Desktop-tool semantic repair functions for OpenClaw (GLM-5-Turbo) ──


def _repair_desktop_coord_int(value):
    """Convert float or string coordinates to int for desktop_click/desktop_move.

    GLM-5-Turbo occasionally emits coordinates as floats (``100.0``) or
    string digits (``"500"``) instead of plain ints.
    """
    # Bug fix: bool is int subclass in Python — True -> 1, False -> 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            # Bug fix: handle float strings like "100.5" -> 100
            f = float(value.strip())
            if f != f or f == float("inf") or f == float("-inf"):
                return value
            return int(f)
        except (ValueError, TypeError):
            return value
    # None stays None — required param, let handler produce a clear error
    return value


def _repair_desktop_clipboard_action(value):
    """Fix ``action: null`` → ``action: "read"`` for desktop_clipboard.

    ``action`` is a required param.  GLM-5-Turbo sometimes sends null
    for required string enums; defaulting to ``"read"`` is safe because
    read is side-effect-free.
    """
    if value is None:
        return "read"
    return value


def _repair_desktop_clipboard_text_string(value):
    """Convert non-string ``text`` to string for desktop_clipboard write.

    GLM-5-Turbo occasionally passes ``text: 123`` (int) instead of
    ``text: "123"`` (string).  The existing coerce layer only converts
    *string → target_type*, not the reverse, so this covers the gap.
    """
    # Bug fix: None must not become "None" — handler validates null text
    # and returns a clear error.
    if value is None:
        return value
    if not isinstance(value, str):
        return str(value)
    return value


def _repair_desktop_scroll_amount(value):
    """Convert float or string ``amount`` to int for desktop_scroll.

    ``coerce_tool_args`` handles string→int, but float→int is missed
    because the coerce layer only acts on string values.  This catches
    ``amount: 5.0`` emitted as a JSON number by the model.
    """
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            # Bug fix: handle float strings like "-3.5" -> -3
            f = float(value.strip())
            if f != f or f == float("inf") or f == float("-inf"):
                return value
            return int(f)
        except (ValueError, TypeError):
            return value
    return value


# ── Semantic repair map (per-tool, per-parameter) ──
# Keyed by (tool_name, param_name) tuple.
_SEMANTIC_REPAIR_MAP = {
    ("read_file", "path"): [
        ("strip_markdown_link", _repair_markdown_link),
        ("expand_tilde", _repair_expand_user),
    ],
    ("write_file", "path"): [
        ("strip_markdown_link", _repair_markdown_link),
        ("expand_tilde", _repair_expand_user),
    ],
    ("terminal", "command"): [
        ("fix_double_escape", _repair_fix_double_escape),
        ("strip_shell_prompt", _repair_strip_shell_prompt),
    ],
    ("patch", "path"): [
        ("strip_markdown_link", _repair_markdown_link),
        ("expand_tilde", _repair_expand_user),
    ],
    ("search_files", "path"): [
        ("strip_markdown_link", _repair_markdown_link),
        ("expand_tilde", _repair_expand_user),
    ],
    # ── Desktop tools (OpenClaw / GLM-5-Turbo) ──
    # desktop_type.text must NEVER be modified — content is user-authored text
    ("desktop_type", "text"): [
        ("safeguard_desktop_type_text", lambda v: v),
    ],
    ("desktop_click", "x"): [
        ("fix_coord_type", _repair_desktop_coord_int),
    ],
    ("desktop_click", "y"): [
        ("fix_coord_type", _repair_desktop_coord_int),
    ],
    ("desktop_move", "x"): [
        ("fix_coord_type", _repair_desktop_coord_int),
    ],
    ("desktop_move", "y"): [
        ("fix_coord_type", _repair_desktop_coord_int),
    ],
    ("desktop_clipboard", "action"): [
        ("fix_clipboard_action_null", _repair_desktop_clipboard_action),
    ],
    ("desktop_clipboard", "text"): [
        ("fix_clipboard_text_type", _repair_desktop_clipboard_text_string),
    ],
    ("desktop_scroll", "amount"): [
        ("fix_amount_type", _repair_desktop_scroll_amount),
    ],
}


def _repair_semantic_args(tool_name: str, args: Dict[str, Any]) -> tuple:
    """Apply per-tool, per-parameter semantic repairs.

    Returns:
        (repaired_args, repair_log_or_None)
    """
    if not args or not isinstance(args, dict):
        return args, None

    repair_log = None
    repaired = dict(args)

    for (tname, pname), rules in _SEMANTIC_REPAIR_MAP.items():
        if tname != tool_name:
            continue
        if pname not in repaired:
            continue
        value = repaired[pname]
        original = value
        for rule_name, rule_fn in rules:
            try:
                new_value = rule_fn(value)
            except Exception:
                continue
            if new_value is not value:
                if repair_log is None:
                    repair_log = []
                repair_log.append({
                    "param": pname,
                    "from": value,
                    "to": new_value,
                    "repair": rule_name,
                })
                value = new_value
        if value is not original:
            repaired[pname] = value

    return repaired, repair_log


# ── Error formatting for model-facing error messages ──

def _format_tool_error_for_model(e: Exception, function_name: str, function_args: dict) -> str:
    """Format a Python tool error into a model-readable message."""
    error_str = str(e)
    if len(error_str) > 500:
        error_str = error_str[:500] + "... (truncated)"
    return (
        f"Exception type: {type(e).__name__}\n"
        f"Message: {error_str}\n"
        f"Arguments: {json.dumps(function_args, ensure_ascii=False, default=str)[:300]}"
    )


def _get_repair_hint(function_name: str, function_args: dict) -> str:
    """Generate a repair hint for common error patterns."""
    hints = []
    for key, value in function_args.items():
        if isinstance(value, str) and value.strip().startswith("["):
            hints.append(
                f"Parameter '{key}' looks like a JSON array string. "
                "Try passing it as a native array instead of a string."
            )
    if hints:
        return " | ".join(hints)
    return "Check parameter types and values."


# =========================================================================
# Tool argument type coercion
# =========================================================================

def coerce_tool_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce tool call arguments to match their JSON Schema types.

    LLMs frequently return numbers as strings (``"42"`` instead of ``42``)
    and booleans as strings (``"true"`` instead of ``true``).  This compares
    each argument value against the tool's registered JSON Schema and attempts
    safe coercion when the value is a string but the schema expects a different
    type.  Original values are preserved when coercion fails.

    Handles ``"type": "integer"``, ``"type": "number"``, ``"type": "boolean"``,
    and union types (``"type": ["integer", "string"]``).

    Also wraps bare scalar values in a single-element list when the schema
    declares ``"type": "array"``.  Open-weight models (DeepSeek, Qwen, GLM)
    sometimes emit ``{"urls": "https://a.com"}`` when the tool expects
    ``{"urls": ["https://a.com"]}``; wrapping here avoids a confusing tool
    failure on what is otherwise a well-formed call.
    """
    if not args or not isinstance(args, dict):
        return args

    schema = registry.get_schema(tool_name)
    if not schema:
        return args

    properties = (schema.get("parameters") or {}).get("properties")
    if not properties:
        return args

    for key, value in list(args.items()):
        prop_schema = properties.get(key)
        if not prop_schema:
            continue
        expected = prop_schema.get("type")

        # Wrap bare non-list values when the schema declares ``array``.
        # Strings still go through _coerce_value first so JSON-encoded
        # arrays (``'["a","b"]'``) get parsed and nullable ``"null"``
        # becomes ``None`` rather than ``["null"]``.
        # ``None`` itself is preserved — we don't know whether the model
        # meant "omit" or "empty list", and tools with sensible defaults
        # (e.g. read_file's normalize_read_pagination) already handle it.
        if expected == "array" and value is not None and not isinstance(value, (list, tuple)):
            if isinstance(value, str):
                coerced = _coerce_value(value, expected, schema=prop_schema)
                if coerced is not value:
                    # _coerce_value handled it (JSON-parsed list or
                    # nullable "null" → None).
                    args[key] = coerced
                    continue
                args[key] = [value]
                logger.info(
                    "coerce_tool_args: wrapped bare string in list for %s.%s",
                    tool_name, key,
                )
                continue
            args[key] = [value]
            logger.info(
                "coerce_tool_args: wrapped bare %s in list for %s.%s",
                type(value).__name__, tool_name, key,
            )
            continue

        if not isinstance(value, str):
            continue
        if not expected and not _schema_allows_null(prop_schema):
            continue
        coerced = _coerce_value(value, expected, schema=prop_schema)
        if coerced is not value:
            args[key] = coerced

    return args


def _coerce_value(value: str, expected_type, schema: dict | None = None):
    """Attempt to coerce a string *value* to *expected_type*.

    Returns the original string when coercion is not applicable or fails.
    """
    if _schema_allows_null(schema) and value.strip().lower() == "null":
        return None

    if isinstance(expected_type, list):
        # Union type — try each in order, return first successful coercion
        for t in expected_type:
            result = _coerce_value(value, t, schema=schema)
            if result is not value:
                return result
        return value

    if expected_type in ("integer", "number"):
        return _coerce_number(value, integer_only=(expected_type == "integer"))
    if expected_type == "boolean":
        return _coerce_boolean(value)
    if expected_type == "array":
        return _coerce_json(value, list)
    if expected_type == "object":
        return _coerce_json(value, dict)
    if expected_type == "null" and value.strip().lower() == "null":
        return None
    return value


def _schema_allows_null(schema: dict | None) -> bool:
    """Return True when a JSON Schema fragment explicitly permits null."""
    if not isinstance(schema, dict):
        return False

    schema_type = schema.get("type")
    if schema_type == "null":
        return True
    if isinstance(schema_type, list) and "null" in schema_type:
        return True
    if schema.get("nullable") is True:
        return True

    for union_key in ("anyOf", "oneOf"):
        variants = schema.get(union_key)
        if not isinstance(variants, list):
            continue
        for variant in variants:
            if isinstance(variant, dict) and variant.get("type") == "null":
                return True

    return False


def _coerce_json(value: str, expected_python_type: type):
    """Parse *value* as JSON when the schema expects an array or object.

    Handles model output drift where a complex oneOf/discriminated-union schema
    causes the LLM to emit the array/object as a JSON string instead of a native
    structure.  Returns the original string if parsing fails or yields the wrong
    Python type.
    """
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(parsed, expected_python_type):
        logger.debug(
            "coerce_tool_args: coerced string to %s via json.loads",
            expected_python_type.__name__,
        )
        return parsed
    return value


def _coerce_number(value: str, integer_only: bool = False):
    """Try to parse *value* as a number.  Returns original string on failure."""
    try:
        f = float(value)
    except (ValueError, OverflowError):
        return value
    # Guard against inf/nan — not JSON-serializable, keep original string
    if f != f or f == float("inf") or f == float("-inf"):
        return value
    # If it looks like an integer (no fractional part), return int
    if f == int(f):
        return int(f)
    if integer_only:
        # Schema wants an integer but value has decimals — keep as string
        return value
    return f


def _coerce_boolean(value: str):
    """Try to parse *value* as a boolean.  Returns original string on failure."""
    low = value.strip().lower()
    if low == "true":
        return True
    if low == "false":
        return False
    return value


def handle_function_call(
    function_name: str,
    function_args: Dict[str, Any],
    task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_task: Optional[str] = None,
    enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False,
) -> str:
    """
    Main function call dispatcher that routes calls to the tool registry.

    Args:
        function_name: Name of the function to call.
        function_args: Arguments for the function.
        task_id: Unique identifier for terminal/browser session isolation.
        user_task: The user's original task (for browser_snapshot context).
        enabled_tools: Tool names enabled for this session.  When provided,
                       execute_code uses this list to determine which sandbox
                       tools to generate.  Falls back to the process-global
                       ``_last_resolved_tool_names`` for backward compat.

    Returns:
        Function result as a JSON string.
    """
    # ── Tool Input Repair: structural + semantic (before coerce) ──
    # Since both repair passes return (repaired, log) and we want a single
    # merged log, we apply semantic first, then structural, then merge.
    function_args, sem_repair_log = _repair_semantic_args(function_name, function_args)
    function_args, struct_repair_log = _repair_tool_args(function_name, function_args)

    # Merge repair logs
    repair_log = None
    if struct_repair_log or sem_repair_log:
        repair_log = []
        if sem_repair_log:
            repair_log.extend(sem_repair_log)
        if struct_repair_log:
            repair_log.extend(struct_repair_log)

    # Log repair events to the session event store (event logging is optional)
    if repair_log and session_id and task_id:
        try:
            from agent.session_event_log import EventLog
            el = EventLog()
            el.log_tool_input_repaired(
                task_id=task_id,
                session_id=session_id,
                tool_name=function_name,
                repair_log=repair_log,
            )
        except Exception:
            pass

    # Coerce string arguments to their schema-declared types (e.g. "42"→42)
    function_args = coerce_tool_args(function_name, function_args)

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return json.dumps({"error": f"{function_name} must be handled by the agent loop"})

        # Check plugin hooks for a block directive (unless caller already
        # checked — e.g. run_agent._invoke_tool passes skip=True to
        # avoid double-firing the hook).
        #
        # Single-fire contract: pre_tool_call fires exactly once per tool
        # execution. get_pre_tool_call_block_message() internally calls
        # invoke_hook("pre_tool_call", ...) and returns the first block
        # directive (if any), so observer plugins see the hook on that same
        # pass. When skip=True, the caller already fired it — do nothing
        # here.
        if not skip_pre_tool_call_hook:
            block_message: Optional[str] = None
            try:
                from hermes_cli.plugins import get_pre_tool_call_block_message
                block_message = get_pre_tool_call_block_message(
                    function_name,
                    function_args,
                    task_id=task_id or "",
                    session_id=session_id or "",
                    tool_call_id=tool_call_id or "",
                )
            except Exception as _hook_err:
                logger.debug("pre_tool_call hook error: %s", _hook_err)

            if block_message is not None:
                return json.dumps({"error": block_message}, ensure_ascii=False)

        # Notify the read-loop tracker when a non-read/search tool runs,
        # so the *consecutive* counter resets (reads after other work are fine).
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # Measure tool dispatch latency so post_tool_call and
        # transform_tool_result hooks can observe per-tool duration.
        # Inspired by Claude Code 2.1.119, which added ``duration_ms`` to
        # PostToolUse hook inputs so plugin authors can build latency
        # dashboards, budget alerts, and regression canaries without having
        # to wrap every tool manually.  We use monotonic() so the value is
        # unaffected by wall-clock adjustments during the call.
        _dispatch_start = time.monotonic()
        if function_name == "execute_code":
            # Prefer the caller-provided list so subagents can't overwrite
            # the parent's tool set via the process-global.
            sandbox_enabled = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                enabled_tools=sandbox_enabled,
            )
        else:
            result = registry.dispatch(
                function_name, function_args,
                task_id=task_id,
                user_task=user_task,
            )
        duration_ms = int((time.monotonic() - _dispatch_start) * 1000)

        try:
            from hermes_cli.plugins import invoke_hook
            invoke_hook(
                "post_tool_call",
                tool_name=function_name,
                args=function_args,
                result=result,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                duration_ms=duration_ms,
            )
        except Exception as _hook_err:
            logger.debug("post_tool_call hook error: %s", _hook_err)

        # Generic tool-result canonicalization seam: plugins receive the
        # final result string (JSON, usually) and may replace it by
        # returning a string from transform_tool_result. Runs after
        # post_tool_call (which stays observational) and before the result
        # is appended back into conversation context. Fail-open; the first
        # valid string return wins; non-string returns are ignored.
        try:
            from hermes_cli.plugins import invoke_hook
            hook_results = invoke_hook(
                "transform_tool_result",
                tool_name=function_name,
                args=function_args,
                result=result,
                task_id=task_id or "",
                session_id=session_id or "",
                tool_call_id=tool_call_id or "",
                duration_ms=duration_ms,
            )
            for hook_result in hook_results:
                if isinstance(hook_result, str):
                    result = hook_result
                    break
        except Exception as _hook_err:
            logger.debug("transform_tool_result hook error: %s", _hook_err)

        # ── Inject repair_log into tool result (feedback loop) ──
        if repair_log:
            try:
                # If result is already a JSON string, append _repair field
                parsed = json.loads(result) if isinstance(result, str) else result
                if isinstance(parsed, dict):
                    parsed["_repair"] = repair_log
                    result = json.dumps(parsed, ensure_ascii=False)
                elif isinstance(parsed, list):
                    result = json.dumps({
                        "result": parsed,
                        "_repair": repair_log,
                    }, ensure_ascii=False)
            except (ValueError, TypeError):
                # Non-JSON result — wrap it
                result = json.dumps({
                    "result": result,
                    "_repair": repair_log,
                }, ensure_ascii=False)

        return result

    except Exception as e:
        error_detail = _format_tool_error_for_model(e, function_name, function_args)
        error_msg = f"Tool '{function_name}' execution failed."
        logger.exception("Error executing %s: %s", function_name, str(e))
        result_dict = {
            "error": error_msg,
            "detail": error_detail,
            "hint": _get_repair_hint(function_name, function_args),
        }
        # Include repair_log in error responses too
        if repair_log:
            result_dict["_repair"] = repair_log
        return json.dumps(result_dict, ensure_ascii=False)


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
