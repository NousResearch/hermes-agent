"""Thin orchestration layer over the tool registry.

Importing runs tool discovery (each tools/*.py self-registers via
tools.registry.register()); exposes get_tool_definitions() (toolset-filtered
schemas sent to the model) and handle_function_call() (dispatch with
hooks/middleware) plus registry pass-throughs.
"""

import os
import json
import re
import asyncio
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from contextvars import ContextVar
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Tuple

from agent.tool_result_classification import is_skill_view_dedup_result
from tools.registry import CHECK_FN_CACHE_BYPASS, check_fn_cache_scope, discover_builtin_tools, registry, tool_error
from tools.registry import _MAX_TOOL_ERROR_CHARS as _TOOL_ERROR_MAX_LEN
from toolsets import resolve_toolset, validate_toolset
from tools.arg_coercion import coerce_tool_args

logger = logging.getLogger(__name__)

_post_tool_call_hook_suppressed: ContextVar[bool] = ContextVar("post_tool_call_hook_suppressed", default=False)


@contextmanager
def suppress_post_tool_call_hook():
    """Let an outer executor own the terminal post-tool event."""
    token = _post_tool_call_hook_suppressed.set(True)
    try:
        yield
    finally:
        _post_tool_call_hook_suppressed.reset(token)

# Platform-bundle names already flagged in disabled_toolsets (advisory logged once per name).
_WARNED_DISABLED_BUNDLES: set = set()


def _is_delegated_child_context() -> bool:
    try:
        from agent.delegation_context import is_delegated_child_context
        return is_delegated_child_context()
    except Exception:
        return False


def _is_dispatcher_owned_worker() -> bool:
    """False when HERMES_KANBAN_* is present but this execution does not own it
    (delegate_task child, or a cron job fired in-process from a worker)."""
    try:
        from agent.delegation_context import is_dispatcher_owned_worker_context
        return is_dispatcher_owned_worker_context()
    except Exception:
        return True


# --- Async bridging (single source of truth; registry.dispatch uses it too) ---
# Loops are persistent (never asyncio.run per call): cached httpx/AsyncOpenAI
# clients stay bound to a live loop, so their GC cleanup can't hit "Event loop
# is closed". Main thread shares one loop; worker threads own thread-local loops.

_tool_loop = None          # persistent loop for the main (CLI) thread
_tool_loop_lock = threading.Lock()
_worker_thread_local = threading.local()  # per-worker-thread persistent loops


def _get_tool_loop():
    """Long-lived event loop for async tool handlers on the main thread."""
    global _tool_loop
    with _tool_loop_lock:
        if _tool_loop is None or _tool_loop.is_closed():
            _tool_loop = asyncio.new_event_loop()
        return _tool_loop


def _get_worker_loop():
    """Persistent event loop for the current worker thread (thread-local)."""
    loop = getattr(_worker_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_thread_local.loop = loop
    return loop


def _run_async(coro):
    """Run a coroutine from sync code; safe under a running loop (gateway/RL env)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Inside a running loop: run in a fresh thread whose loop we keep a
        # reference to, so on timeout we can cancel the task inside it
        # (ThreadPoolExecutor.cancel() is a no-op on a running worker).
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
                try:  # drain tasks still pending after an external cancel
                    pending = asyncio.all_tasks(worker_loop)
                    for t in pending:
                        t.cancel()
                    if pending:
                        worker_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                except Exception:
                    pass
                worker_loop.close()

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Carry profile + approval/sudo context so get_hermes_home() resolves correctly.
        from tools.thread_context import propagate_context_to_thread
        future = pool.submit(propagate_context_to_thread(_run_in_worker))
        try:
            return future.result(timeout=300)
        except concurrent.futures.TimeoutError:
            # Cancel inside the worker's own loop so the thread can wind down.
            if loop_ready.wait(timeout=1.0) and worker_loop is not None:
                try:
                    for t in asyncio.all_tasks(worker_loop):
                        worker_loop.call_soon_threadsafe(t.cancel)
                except RuntimeError:
                    pass  # loop already closed
            raise
        finally:
            pool.shutdown(wait=False)  # never block the caller on a stuck coroutine

    if threading.current_thread() is not threading.main_thread():
        return _get_worker_loop().run_until_complete(coro)
    return _get_tool_loop().run_until_complete(coro)


# --- Tool discovery (importing each tools/*.py triggers registry.register) ---
discover_builtin_tools()

# MCP discovery is deliberately NOT run here: it blocks up to 120 s and the
# gateway lazy-imports this module inside its event loop; each entry point
# (gateway/run.py, cli.py, tui_gateway, acp_adapter) runs it at startup.
try:  # plugin tool discovery (user/project/pip plugins)
    # MCP tool discovery (external MCP servers from config) used to run here as a module-level side effect.
    # It was removed because discover_mcp_tools() internally uses a blocking future.result(timeout=120)
    # wait, and the gateway lazy-imports this module from inside the asyncio event loop on the first user
    # message — freezing Discord/Telegram heartbeats for up to 120s whenever any configured MCP server was
    # slow or unreachable (#16856). - gateway/run.py            -> start_gateway() uses run_in_executor -
    # acp_adapter/server.py     -> asyncio.to_thread on session init
    from hermes_cli.plugins import discover_plugins
    discover_plugins()
except Exception as e:
    logger.debug("Plugin discovery failed: %s", e)


# Backward-compat constants (built once after discovery)
TOOL_TO_TOOLSET_MAP: Dict[str, str] = registry.get_tool_to_toolset_map()

TOOLSET_REQUIREMENTS: Dict[str, dict] = registry.get_toolset_requirements()

# Tool names from the last get_tool_definitions() call (execute_code sandbox fallback).
_last_resolved_tool_names: List[str] = []


# Legacy toolset names (old _tools-suffixed names -> tool name lists)
_LEGACY_TOOLSET_MAP = {
    "web_tools": ["web_search", "web_extract"],
    "terminal_tools": ["terminal"],
    "vision_tools": ["vision_analyze"],
    "image_tools": ["image_generate"],
    "skills_tools": ["skills_list", "skill_view", "skill_manage"],
    "browser_tools": ["browser_navigate", "browser_snapshot", "browser_click", "browser_type", "browser_scroll",
                      "browser_back", "browser_press", "browser_get_images", "browser_vision", "browser_console"],
    "cronjob_tools": ["cronjob_manage"],
    "file_tools": ["read_file", "write_file", "patch", "search_files"],
    "tts_tools": ["text_to_speech"],
}


# --- get_tool_definitions (the main schema provider) --------------------------
# Memo for get_tool_definitions(), active only with quiet_mode=True (the
# non-quiet path prints). Hot callers (gateway runner, AIAgent.__init__) hit it
# every turn; a miss costs ~7 ms of registry walk + check_fn probing. The key
# includes registry._generation (bumped on register/deregister/alias) so
# invalidation is transparent; check_fn drift is handled by registry.py's 30 s TTL.
_tool_defs_cache: Dict[tuple, List[Dict[str, Any]]] = {}
_tool_defs_cache_lock = threading.Lock()
# FIFO cap: 8 covers a long-lived gateway's warm set of platform/toolset combos.
# Hard cap on memoized get_tool_definitions() results. A long-lived Gateway process sees many distinct
# toolset/config fingerprints over its lifetime (per-session toolset sets, config edits, kanban-task
# toggles); without a bound the cache grows unboundedly. 8 comfortably covers the warm working set (the
# handful of distinct platform/toolset combos a gateway actually serves) while keeping the cap small.
# (#19251)
_TOOL_DEFS_CACHE_MAX = 8


def _clear_tool_defs_cache() -> None:
    """Drop memoized results when a dynamic-schema dependency changes (discord caps, sandbox mode)."""
    with _tool_defs_cache_lock:
        _tool_defs_cache.clear()


def get_tool_definitions(enabled_toolsets: Optional[List[str]] = None, disabled_toolsets: Optional[List[str]] = None,
                         quiet_mode: bool = False, skip_tool_search_assembly: bool = False) -> List[Dict[str, Any]]:
    """Tool definitions for model API calls, filtered by toolset.

    enabled_toolsets None = all; disabled_toolsets are subtracted after enabling.
    quiet_mode suppresses status prints and enables memoization.
    skip_tool_search_assembly returns raw schemas for every enabled tool — only
    the tool_search bridge should use it (it reads the real, uncollapsed catalog).
    """
    def compute():
        return _compute_tool_definitions(enabled_toolsets, disabled_toolsets, quiet_mode,
                                         skip_tool_search_assembly=skip_tool_search_assembly)
    if not quiet_mode:
        return compute()
    cache_key = _tool_defs_cache_key(enabled_toolsets, disabled_toolsets, skip_tool_search_assembly)
    # Cache the freshly-computed list, but hand callers a shallow copy so downstream mutations (e.g.
    # run_agent appending memory/LCM tool schemas to self.tools) don't poison the cache. Without this, a
    # long-lived Gateway process accumulates duplicate tool names across agent inits and providers that
    # enforce unique tool names (DeepSeek, Xiaomi MiMo, Moonshot Kimi) reject the request with HTTP 400.
    # Mirrors the cache-hit path above. (issue #17335) Bound the cache with LRU eviction so a long-lived
    # Gateway process doesn't accumulate entries unboundedly across the many distinct toolset/config
    # fingerprints it sees over its lifetime (#19251).
    with _tool_defs_cache_lock:
        cached = _tool_defs_cache.get(cache_key) if cache_key is not None else None
    if cached is None:
        result = compute()
        if cache_key is None:
            return list(result)
        with _tool_defs_cache_lock:
            cached = _tool_defs_cache.get(cache_key)  # another thread may have filled it meanwhile
            if cached is None:
                if len(_tool_defs_cache) >= _TOOL_DEFS_CACHE_MAX:
                    _tool_defs_cache.pop(next(iter(_tool_defs_cache)))
                _tool_defs_cache[cache_key] = cached = result
    else:
        global _last_resolved_tool_names
        _last_resolved_tool_names = [t["function"]["name"] for t in cached]
    # Always a shallow copy: run_agent appends memory/LCM schemas to its list; a
    # shared list would accumulate duplicate names (HTTP 400 from DeepSeek/Kimi/MiMo).
    return list(cached)


def _tool_defs_cache_key(
    enabled_toolsets: Optional[List[str]], disabled_toolsets: Optional[List[str]], skip_tool_search_assembly: bool,
) -> Optional[tuple]:
    """Memo key for get_tool_definitions, or None when caching must be bypassed.

    Covers every argument plus everything that changes the result without one:
    registry generation, config.yaml mtime/size (dynamic schemas), kanban
    context, profile scope. check_fn results are TTL-cached in the registry.
    """
    profile_scope = check_fn_cache_scope()
    if profile_scope == CHECK_FN_CACHE_BYPASS:
        return None
    try:
        from hermes_cli.config import get_config_path
        cfg_stat = get_config_path().stat()
        cfg_fp = (cfg_stat.st_mtime_ns, cfg_stat.st_size)
    except (FileNotFoundError, OSError, ImportError):
        cfg_fp = None
    return (
        registry.current_scope_key(), frozenset(enabled_toolsets) if enabled_toolsets is not None else None,
        frozenset(disabled_toolsets) if disabled_toolsets else None, registry._generation, cfg_fp,
        bool(os.environ.get("HERMES_KANBAN_TASK")), bool(skip_tool_search_assembly),
        _is_delegated_child_context(), _is_dispatcher_owned_worker(), profile_scope,
    )


def _apply_toolset_selection(tools: set, names: List[str], quiet_mode: bool, *, disable: bool) -> None:
    """Add (or subtract) every toolset in *names* to/from *tools*, printing the selection unless quiet."""
    from toolsets import bundle_non_core_tools, get_toolset
    verb, icon = ("Disabled", "🚫") if disable else ("Enabled", "✅")
    for name in names:
        if validate_toolset(name):
            label = f"{verb} toolset"
            if disable and (name.startswith("hermes-") or (get_toolset(name) or {}).get("posture")):
                # Bundles/postures re-list the core tools without owning them;
                # subtracting the whole set would empty the list — remove only the non-core delta.
                resolved = sorted(bundle_non_core_tools(name))
                if not quiet_mode and name.startswith("hermes-") and name not in _WARNED_DISABLED_BUNDLES:
                    _WARNED_DISABLED_BUNDLES.add(name)
                    logger.info(
                        "agent.disabled_toolsets contains platform-bundle name '%s'; core tools are "
                        "preserved and only its platform-specific tools (%s) are removed. Bundle names "
                        "usually belong in `toolsets:`, not `disabled_toolsets` (#33924).",
                        name, ", ".join(resolved) if resolved else "none",
                    )
            else:
                resolved = resolve_toolset(name)
        elif name in _LEGACY_TOOLSET_MAP:
            label = f"{verb} legacy toolset"
            resolved = _LEGACY_TOOLSET_MAP[name]
        else:
            if not quiet_mode:
                print(f"⚠️  Unknown toolset: {name}")
            continue
        (tools.difference_update if disable else tools.update)(resolved)
        if not quiet_mode:
            print(f"{icon} {label} '{name}': {', '.join(resolved) if resolved else 'no tools'}")


def _select_tool_names(enabled_toolsets: Optional[List[str]], disabled_toolsets: Optional[List[str]], quiet_mode: bool) -> set:
    """Tool names requested by the toolset selection (before check_fn filtering)."""
    tools: set = set()
    if enabled_toolsets is not None:
        enabled = list(enabled_toolsets)
        # Dispatcher-spawned kanban workers always get the lifecycle handoff
        # tools, even when the assignee profile restricts its chat toolsets.
        if (os.environ.get("HERMES_KANBAN_TASK") and not _is_delegated_child_context()
                and _is_dispatcher_owned_worker() and "kanban" not in enabled):
            enabled.append("kanban")
        _apply_toolset_selection(tools, enabled, quiet_mode, disable=False)
    else:
        from toolsets import get_all_toolsets
        for ts_name in get_all_toolsets():
            tools.update(resolve_toolset(ts_name))
    # Disabled toolsets are always subtracted LAST, so a tool in a disabled
    # toolset is stripped even when a composite (hermes-cli) re-enables it.
    # This ensures that even if a composite toolset (like hermes-cli) is enabled, any tools belonging to a
    # disabled toolset are strictly stripped out. See issue #17309.
    if disabled_toolsets:
        _apply_toolset_selection(tools, disabled_toolsets, quiet_mode, disable=True)
    return tools


# --- Dynamic schema rewrites -------------------------------------------------
# Each rewriter gets (tool definition, set of tool names that passed check_fn)
# and returns the (possibly replaced) definition, or None to drop the tool.
# Cross-references must use that set so the model never hears of an absent tool.

_BROWSER_NAVIGATE_WEB_HINT = " For simple information retrieval, prefer web_search or web_extract (faster, cheaper)."


def _fn_def(schema: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "function", "function": schema}


def _rewrite_execute_code(td: Dict[str, Any], available: set) -> Optional[Dict[str, Any]]:
    """List only sandbox tools that are actually available."""
    # Without this, the model sees "web_search is available in execute_code" even when the API key isn't
    # configured or the toolset is disabled (#560-discord).
    from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS, build_execute_code_schema, _get_execution_mode
    return _fn_def(build_execute_code_schema(SANDBOX_ALLOWED_TOOLS & available, mode=_get_execution_mode()))


def _discord_rewriter(schema_fn_name: str):
    """Schema depends on the bot's privileged intents and the config action allowlist; None drops the tool."""
    def _rewrite(td: Dict[str, Any], available: set) -> Optional[Dict[str, Any]]:
        try:
            from tools import discord_tool as _dt
            dynamic = getattr(_dt, schema_fn_name)()
        except Exception:
            dynamic = None
        return None if dynamic is None else _fn_def(dynamic)
    return _rewrite


def _rewrite_browser_navigate(td: Dict[str, Any], available: set) -> Optional[Dict[str, Any]]:
    """Drop the "prefer web_search or web_extract" hint when neither web tool is present (else the model hallucinates them)."""
    if {"web_search", "web_extract"} & available:
        return td
    desc = td["function"].get("description", "").replace(_BROWSER_NAVIGATE_WEB_HINT, "")
    return _fn_def({**td["function"], "description": desc})


def _rewrite_browser_exec(td: Dict[str, Any], available: set) -> Optional[Dict[str, Any]]:
    """browser_exec runs arbitrary host Python: a session without the terminal surface
    must not regain host execution via the browser toolset. Session-level gate rather
    than a check_fn because check_fns are TTL-cached process-wide across sessions."""
    return td if "terminal" in available else None


def _rewrite_delegate_task(td: Dict[str, Any], available: set) -> Optional[Dict[str, Any]]:
    """Trim the child-restrictions line to sibling tools actually present, or drop
    the line when none apply, so the model never learns ghost vocabulary. Two
    source variants exist (depth-off also names delegate_task itself); test the
    longer one first because the sibling list is a substring of it."""
    blocked_present = [t for t in ("clarify", "memory", "cronjob_manage") if t in available]
    if len(blocked_present) == 3:
        return td
    fn = td.get("function", {})
    desc = fn.get("description", "")
    for full, self_named in (("delegate_task, clarify, memory, or cronjob", True), ("clarify, memory, or cronjob", False)):
        if full in desc:
            break
    else:
        return td
    if blocked_present:
        names = (["delegate_task"] if self_named else []) + blocked_present
        replacement = " or ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + ", or " + names[-1]
        desc = desc.replace(full, replacement)
    else:
        # Both variants end at the following newline.
        start = desc.find("- Children cannot call " + full)
        if start != -1:
            desc = desc[:start] + desc[desc.index("\n", start) + 1:]
    return {**td, "function": {**fn, "description": desc}}


_DYNAMIC_SCHEMA_REWRITERS = {
    "execute_code": _rewrite_execute_code,
    "discord": _discord_rewriter("get_dynamic_schema_core"),
    "discord_admin": _discord_rewriter("get_dynamic_schema_admin"),
    "browser_navigate": _rewrite_browser_navigate,
    "browser_exec": _rewrite_browser_exec,
    "delegate_task": _rewrite_delegate_task,
}


def _apply_dynamic_schemas(tool_defs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply _DYNAMIC_SCHEMA_REWRITERS in list order; the availability set is a
    snapshot taken before any rewrite (no rewriter's inputs are droppable)."""
    available = {t["function"]["name"] for t in tool_defs}
    out = []
    for td in tool_defs:
        rewrite = _DYNAMIC_SCHEMA_REWRITERS.get(td["function"]["name"])
        if rewrite is not None:
            td = rewrite(td, available)
        if td is not None:
            out.append(td)
    return out


_TOOL_SEARCH_LISTING_FORMS = {
    "full": "catalog listing embedded",
    "names": "names-only listing embedded",
    "mixed": "listing embedded (oversized servers summarized)",
    "groups": "server summary embedded (search-only discovery)",
    "none": "no listing (search-only)",
}


def _compute_tool_definitions(enabled_toolsets: Optional[List[str]] = None, disabled_toolsets: Optional[List[str]] = None,
                              quiet_mode: bool = False, skip_tool_search_assembly: bool = False) -> List[Dict[str, Any]]:
    """Uncached implementation of :func:`get_tool_definitions`."""
    tools_to_include = _select_tool_names(enabled_toolsets, disabled_toolsets, quiet_mode)
    # Registry returns only tools whose check_fn passes.
    filtered_tools = _apply_dynamic_schemas(registry.get_definitions(tools_to_include, quiet=quiet_mode))
    global _last_resolved_tool_names
    _last_resolved_tool_names = [t["function"]["name"] for t in filtered_tools]

    if not quiet_mode:
        print(f"🛠️  Final tool selection ({len(filtered_tools)} tools): {', '.join(_last_resolved_tool_names)}"
              if filtered_tools else "🛠️  No tools selected (all filtered out or unavailable)")
    # Normalize schema shapes llama.cpp's grammar converter rejects (bare
    # "type": "object", string-valued nodes from malformed MCP servers).
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
        filtered_tools = sanitize_tool_schemas(filtered_tools)
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("Schema sanitization skipped: %s", e)

    # Tool Search (progressive disclosure): replace MCP/plugin tools with the
    # tool_search/describe/call bridge when the deferrable surface exceeds the
    # configured share of the context window. Core tools are never deferred.
    # Must be the LAST step (after sanitization); idempotent if called twice.
    try:
        from tools.tool_search import assemble_tool_defs, load_config as _load_ts_config
        ts_cfg = _load_ts_config()
        if not skip_tool_search_assembly and ts_cfg.enabled != "off":
            assembly = assemble_tool_defs(filtered_tools, context_length=_resolve_active_context_length(), config=ts_cfg)
            if assembly.activated and not quiet_mode:
                print(f"🔎 Tool Search (tier {assembly.tier}): {assembly.deferred_count} "
                      f"MCP/plugin tools deferred (~{assembly.deferred_tokens} tokens) behind "
                      f"tool_search/describe/call — "
                      f"{_TOOL_SEARCH_LISTING_FORMS.get(assembly.listing_form, assembly.listing_form)}.")
            filtered_tools = assembly.tool_defs
    except Exception as e:  # pragma: no cover — never break tool loading
        logger.warning("Tool search assembly skipped: %s", e)

    return filtered_tools


def _active_model_config() -> Tuple[str, Dict[str, Any]]:
    """(model_id, model section) from config.yaml; model_id is "" when unset."""
    from hermes_cli.config import load_config
    cfg = load_config() or {}
    model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    raw_model_id = model_cfg.get("model") or model_cfg.get("default") or ""
    if isinstance(raw_model_id, dict):
        from hermes_cli.config import split_model_config_default
        raw_model_id, _ = split_model_config_default(raw_model_id)
    return str(raw_model_id).strip(), model_cfg


def _resolve_active_context_length() -> int:
    """Active model's context length for the tool-search gate (0 if unresolvable).

    Order: explicit `model.context_length`; provider-aware resolution (Codex OAuth
    enforces a smaller window than the direct API for the same slug); the on-disk
    metadata cache (slightly stale is fine for picking a tier and avoids a ~200 ms
    /models probe per CLI startup); then the full live resolver.
    """
    try:
        model_id, model_cfg = _active_model_config()
        if not model_id:
            return 0
        from agent.model_metadata import get_cached_context_length, get_model_context_length
        # Honor explicit `model.context_length` in config.yaml — short-circuits the OpenRouter /models probe
        # at get_model_context_length step 0, so non-OpenRouter providers don't pay the ~2-3s OpenRouter
        # fetch at every CLI startup. See issue #46620.
        raw_ctx = model_cfg.get("context_length")
        config_ctx = raw_ctx if isinstance(raw_ctx, int) and raw_ctx > 0 else None
        provider = str(model_cfg.get("provider") or "").strip()
        base_url = str(model_cfg.get("base_url") or "").strip()
        api_key = ""
        if provider:
            # Credential resolution failing (offline, no keys) degrades to a
            # provider+base_url-only lookup so static fallbacks still apply.
            try:
                from hermes_cli.runtime_provider import resolve_runtime_provider
                rt = resolve_runtime_provider(requested=provider, target_model=model_id) or {}
                base_url = str(rt.get("base_url") or base_url or "").strip()
                api_key = str(rt.get("api_key") or "").strip()
            except Exception as rt_exc:
                logger.debug("Runtime credential resolution failed for tool-search "
                             "context gate (provider=%s): %s — using config values only", provider, rt_exc)
        if config_ctx is None and base_url:
            try:
                cached_ctx = get_cached_context_length(model_id, base_url)
                if isinstance(cached_ctx, int) and cached_ctx > 0:
                    return cached_ctx
            except Exception:
                pass
        return int(get_model_context_length(model_id, base_url=base_url, api_key=api_key,
                                            config_context_length=config_ctx, provider=provider) or 0)
    except Exception as e:
        logger.debug("Could not resolve active context length: %s", e)
        return 0


# =============================================================================
# handle_function_call  (the main dispatcher)
# =============================================================================

# Intercepted by the agent loop (need agent-level state); dispatch returns a stub error.
_AGENT_LOOP_TOOLS = {"todo_list", "memory", "session_search", "delegate_task"}

# Legacy tool-name aliases accepted at every dispatch seam (old sessions/saved
# prompts keep working); schemas advertise only new names.
_LEGACY_TOOL_ALIASES = {
    "todo": "todo_list", "cronjob": "cronjob_manage", "process": "process_manage",
    "tour": "gui_tour", "tip": "show_tip",
}
_READ_SEARCH_TOOLS = {"read_file", "search_files"}


# --- Tool error sanitization --------------------------------------------------
# Defense-in-depth: strip role tags / CDATA / code fences from exception text the
# model will read, and cap length (cap shared with tools/registry.py so text never
# passes two different caps with two different markers).
_TOOL_ERROR_STRIP_RES = (
    re.compile(r'</?(?:tool_call|function_call|result|response|output|input|system|assistant|user)>', re.IGNORECASE),
    re.compile(r'^\s*```(?:json|xml|html|markdown)?\s*', re.MULTILINE),
    re.compile(r'\s*```\s*$', re.MULTILINE),
    re.compile(r'<!\[CDATA\[.*?\]\]>', re.DOTALL),
)


def _sanitize_tool_error(error_msg: str) -> str:
    """Strip structural framing tokens from a tool error before the model sees it."""
    if not error_msg:
        return "[TOOL_ERROR] "
    sanitized = error_msg
    for pattern in _TOOL_ERROR_STRIP_RES:
        sanitized = pattern.sub("", sanitized)
    if len(sanitized) > _TOOL_ERROR_MAX_LEN:
        sanitized = sanitized[:_TOOL_ERROR_MAX_LEN - 3] + "..."
    return f"[TOOL_ERROR] {sanitized}"


@dataclass(frozen=True)
class _CallIds:
    """Identity fields of one tool call, threaded through hooks and middleware."""
    task_id: Optional[str] = None
    session_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    turn_id: Optional[str] = None
    api_request_id: Optional[str] = None

    def hook_kwargs(self) -> Dict[str, str]:
        """Same fields with None -> "" (hook/middleware wire contract)."""
        return {k: v or "" for k, v in asdict(self).items()}


def _tool_result_observer_fields(tool_name: str, result: Any) -> tuple[str, Optional[str], Optional[str]]:
    """Derive (status, error_type, error_message) from a tool result for observer hooks."""
    if is_skill_view_dedup_result(tool_name, result):
        return "ok", None, None
    try:
        parsed_result = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed_result, dict) and parsed_result.get("error"):
            return "error", "tool_error", str(parsed_result.get("error"))
    except Exception:
        pass
    try:
        from agent.display import _detect_tool_failure
        failed, suffix = _detect_tool_failure(tool_name, result)
        if failed:
            return "error", "tool_error", suffix.strip().strip("[]") or None
    except Exception:
        pass
    return "ok", None, None


def _emit_post_tool_call_hook(
    *, function_name: str, function_args: Dict[str, Any], result: Any,
    task_id: Optional[str] = None, session_id: Optional[str] = None, tool_call_id: Optional[str] = None,
    turn_id: Optional[str] = None, api_request_id: Optional[str] = None, duration_ms: int = 0,
    status: Optional[str] = None, error_type: Optional[str] = None, error_message: Optional[str] = None,
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Emit the ``post_tool_call`` observer hook; gated on has_hook, and ok/error
    fields are derived from the result only past that gate when status is None."""
    if _post_tool_call_hook_suppressed.get():
        return
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if not has_hook("post_tool_call"):
            return
        if status is None:
            status, error_type, error_message = _tool_result_observer_fields(function_name, result)
        invoke_hook(
            "post_tool_call", tool_name=function_name, args=function_args, result=result,
            **_CallIds(task_id, session_id, tool_call_id, turn_id, api_request_id).hook_kwargs(),
            duration_ms=duration_ms, status=status, error_type=error_type, error_message=error_message,
            middleware_trace=list(middleware_trace or []),
        )
    except Exception as _hook_err:
        logger.debug("post_tool_call hook error: %s", _hook_err)


def _dispatch_bridge_tool(function_name: str, function_args: Dict[str, Any],
                          enabled_toolsets: Optional[List[str]], disabled_toolsets: Optional[List[str]]):
    """Handle a Tool Search bridge call (tool_search / tool_describe / tool_call).

    None when *function_name* is not a bridge tool; ``(result, None)`` for a
    finished catalog read or error; ``(None, (name, args))`` when a validated
    tool_call should be re-dispatched as the real tool.
    """
    try:
        from tools import tool_search as ts
    except Exception:
        return None
    if not ts.is_bridge_tool(function_name):
        return None
    # Un-collapsed catalog scoped to the session's toolsets, so a restricted
    # session (subagent, kanban worker) can't reach the whole registry via the bridge.
    try:
        current_defs = get_tool_definitions(enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
                                            quiet_mode=True, skip_tool_search_assembly=True) or []
    except Exception:
        current_defs = []
    args = function_args or {}
    if function_name == ts.TOOL_SEARCH_NAME:
        return ts.dispatch_tool_search(args, current_tool_defs=current_defs), None
    if function_name == ts.TOOL_DESCRIBE_NAME:
        return ts.dispatch_tool_describe(args, current_tool_defs=current_defs), None
    underlying_name, underlying_args, err = ts.resolve_underlying_call(args)
    if err or not underlying_name:
        return tool_error(err or "tool_call could not be resolved"), None
    # Defense in depth: resolve_underlying_call only checks the global
    # registry; also require membership in the session-scoped catalog.
    if underlying_name not in ts.scoped_deferrable_names(current_defs):
        return tool_error(f"'{underlying_name}' is not available in this session. "
                          "Use tool_search to find tools you can call."), None
    # Validate against the deferred tool's concrete schema — the generic
    # ``arguments: object`` bridge schema can't enforce it.
    probe_err = ts.validate_deferred_call_args(underlying_name, underlying_args)
    if probe_err is not None:
        return probe_err, None
    return None, (underlying_name, underlying_args)


def _apply_request_middleware(
    function_name: str, function_args: Dict[str, Any], ids: _CallIds, trace: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """tool_request middleware: returns (args, original_args, trace); fail-open."""
    try:
        from hermes_cli.middleware import apply_tool_request_middleware
        mw = apply_tool_request_middleware(function_name, function_args, **ids.hook_kwargs())
        return mw.payload, mw.original_payload, mw.trace
    except Exception as _mw_err:
        logger.debug("tool_request middleware error: %s", _mw_err)
        return function_args, dict(function_args), trace


def _pre_dispatch_guards(function_name: str, function_args: Dict[str, Any], skip_pre_tool_call_hook: bool,
                         ids: _CallIds, middleware_trace: List[Dict[str, Any]],
                         ) -> Tuple[Dict[str, Any], Optional[Tuple[Any, str, Optional[str]]]]:
    """Plugin pre_tool_call hook, then ACP edit approval.

    ``(args, None)`` to proceed (args possibly plugin-modified), or
    ``(args, (result, error_type, error_message))`` when blocked.
    """
    # pre_tool_call fires exactly once per execution: one invoke_hook pass yields
    # both the block message and modified args. skip=True: caller already fired it.
    if not skip_pre_tool_call_hook:
        block_message: Optional[str] = None
        try:
            from hermes_cli.plugins import _dispatch_pre_tool_call_hooks
            block_message, modified_args = _dispatch_pre_tool_call_hooks(
                function_name, function_args, middleware_trace=list(middleware_trace), **ids.hook_kwargs(),
            )
            if modified_args is not None:
                function_args = modified_args
        except Exception as _hook_err:
            logger.debug("pre_tool_call hook error: %s", _hook_err)
        if block_message is not None:
            return function_args, (tool_error(block_message), "plugin_block", block_message)

    # ACP/Zed edit approval before any file mutation. The requester is bound
    # via ContextVar only for ACP sessions, so CLI/gateway paths are unaffected.
    try:
        from acp_adapter.edit_approval import maybe_require_edit_approval
        edit_block_message = maybe_require_edit_approval(function_name, function_args)
        if edit_block_message is not None:
            return function_args, (edit_block_message, "edit_approval_denied", None)
    except Exception as _edit_approval_err:
        logger.debug("ACP edit approval guard error: %s", _edit_approval_err)
        if function_name in {"write_file", "patch"}:
            return function_args, (tool_error("Edit approval denied: approval guard failed"), "edit_approval_error", None)
    return function_args, None


@contextmanager
def _approval_observability(ids: _CallIds):
    """Bind the approval observability context (turn/tool_call/session ids) for the block."""
    try:
        from tools.approval_context import reset_current_observability_context, set_current_observability_context
        tokens = set_current_observability_context(turn_id=ids.turn_id or "", tool_call_id=ids.tool_call_id or "",
                                                   session_id=ids.session_id or "")
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            reset_current_observability_context(tokens)
        except Exception:
            pass


def _execute_tool(function_name: str, function_args: Dict[str, Any], original_args: Dict[str, Any], ids: _CallIds,
                  *, user_task: Optional[str], enabled_tools: Optional[List[str]], skip_tool_execution_middleware: bool) -> Any:
    """Run the registry handler (through tool-execution middleware unless skipped)
    with the approval observability context bound for the duration."""
    dispatch_kwargs: Dict[str, Any] = {"task_id": ids.task_id, "session_id": ids.session_id}
    if function_name == "execute_code":
        # Prefer the caller's list so subagents can't overwrite the parent's
        # tool set via the process-global.
        dispatch_kwargs["enabled_tools"] = enabled_tools if enabled_tools is not None else _last_resolved_tool_names
    else:
        dispatch_kwargs["user_task"] = user_task

    def _dispatch(next_args: Dict[str, Any]) -> Any:
        return registry.dispatch(function_name, next_args, **dispatch_kwargs)

    with _approval_observability(ids):
        if skip_tool_execution_middleware:
            return _dispatch(function_args)
        from hermes_cli.middleware import run_tool_execution_middleware
        return run_tool_execution_middleware(function_name, function_args, _dispatch, original_args=original_args,
                                             **ids.hook_kwargs())


def _apply_transform_tool_result_hook(function_name: str, function_args: Dict[str, Any], result: Any, duration_ms: int,
                                      ids: _CallIds) -> Any:
    """transform_tool_result: plugins may replace the final result string.

    Runs after post_tool_call and before the result enters context. Fail-open;
    first string return wins. Gated on has_hook so the no-listener path is cheap.
    """
    try:
        from hermes_cli.lifecycle import has_hook, invoke_hook
        if has_hook("transform_tool_result"):
            status, error_type, error_message = _tool_result_observer_fields(function_name, result)
            hook_results = invoke_hook("transform_tool_result", tool_name=function_name, args=function_args,
                                       result=result, **ids.hook_kwargs(), duration_ms=duration_ms,
                                       status=status, error_type=error_type, error_message=error_message)
            return next((r for r in hook_results if isinstance(r, str)), result)
    except Exception as _hook_err:
        logger.debug("transform_tool_result hook error: %s", _hook_err)
    return result


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def handle_function_call(
    function_name: str, function_args: Dict[str, Any], task_id: Optional[str] = None,
    tool_call_id: Optional[str] = None, session_id: Optional[str] = None, turn_id: Optional[str] = None,
    api_request_id: Optional[str] = None, user_task: Optional[str] = None, enabled_tools: Optional[List[str]] = None,
    skip_pre_tool_call_hook: bool = False, skip_tool_request_middleware: bool = False,
    skip_tool_execution_middleware: bool = False, tool_request_middleware_trace: Optional[List[Dict[str, Any]]] = None,
    enabled_toolsets: Optional[List[str]] = None, disabled_toolsets: Optional[List[str]] = None,
) -> str:
    """Route a tool call through hooks/middleware to the registry; returns a JSON string.

    task_id isolates terminal/browser sessions; user_task feeds browser_snapshot.
    enabled_tools picks execute_code's sandbox tools (default: the process-global
    ``_last_resolved_tool_names``). skip_pre_tool_call_hook: caller already fired
    it (single-fire contract). enabled/disabled_toolsets scope the Tool Search
    bridge catalog to this session's grant (None = unrestricted).
    """
    function_args = coerce_tool_args(function_name, function_args)
    if not isinstance(function_args, dict):
        function_args = {}
    trace = list(tool_request_middleware_trace or [])
    function_name = _LEGACY_TOOL_ALIASES.get(function_name, function_name)
    ids = _CallIds(task_id, session_id, tool_call_id, turn_id, api_request_id)
    start = time.monotonic()

    def _emit(result: Any, **extra: Any) -> Any:
        """Emit post_tool_call with this call's identity fields; returns *result*."""
        _emit_post_tool_call_hook(function_name=function_name, function_args=function_args, result=result,
                                  **asdict(ids), middleware_trace=list(trace), **extra)
        return result

    # Tool Search bridge: tool_search / tool_describe are catalog reads handled
    # inline; tool_call is unwrapped so every downstream hook (pre/post, edit
    # approval, guardrails) sees the real tool name, never the bridge.
    bridged = _dispatch_bridge_tool(function_name, function_args, enabled_toolsets, disabled_toolsets)
    if bridged is not None:
        result, underlying = bridged
        if underlying is None:
            return _emit(result, duration_ms=_elapsed_ms(start))
        return handle_function_call(
            *underlying, **asdict(ids), user_task=user_task, enabled_tools=enabled_tools,
            skip_pre_tool_call_hook=skip_pre_tool_call_hook, skip_tool_request_middleware=skip_tool_request_middleware,
            skip_tool_execution_middleware=skip_tool_execution_middleware, tool_request_middleware_trace=list(trace),
            enabled_toolsets=enabled_toolsets, disabled_toolsets=disabled_toolsets,
        )

    original_args = dict(function_args)
    if not skip_tool_request_middleware:
        function_args, original_args, trace = _apply_request_middleware(function_name, function_args, ids, trace)

    try:
        if function_name in _AGENT_LOOP_TOOLS:
            return tool_error(f"{function_name} must be handled by the agent loop")

        function_args, blocked = _pre_dispatch_guards(function_name, function_args, skip_pre_tool_call_hook, ids, trace)
        if blocked is not None:
            result, error_type, error_message = blocked
            return _emit(result, status="blocked", error_type=error_type, error_message=error_message)

        # Any non-read/search tool resets the consecutive-read-loop counter.
        if function_name not in _READ_SEARCH_TOOLS:
            try:
                from tools.file_tools_read_tracking import notify_other_tool_call
                notify_other_tool_call(task_id or "default")
            except Exception:
                pass  # file_tools may not be loaded yet

        # duration_ms (monotonic) is exposed to post_tool_call / transform_tool_result.
        start = time.monotonic()
        result = _execute_tool(function_name, function_args, original_args, ids, user_task=user_task,
                               enabled_tools=enabled_tools, skip_tool_execution_middleware=skip_tool_execution_middleware)
        duration_ms = _elapsed_ms(start)
        _emit(result, duration_ms=duration_ms)
        return _apply_transform_tool_result_hook(function_name, function_args, result, duration_ms, ids)

    except Exception as e:
        error_msg = f"Error executing {function_name}: {str(e)}"
        logger.exception(error_msg)
        return _emit(tool_error(_sanitize_tool_error(error_msg)), duration_ms=_elapsed_ms(start),
                     status="error", error_type=type(e).__name__, error_message=str(e))


# =============================================================================
# Backward-compat wrapper functions (registry pass-throughs)
# =============================================================================

def get_all_tool_names() -> List[str]:
    return registry.get_all_tool_names()


def get_toolset_for_tool(tool_name: str) -> Optional[str]:
    return registry.get_toolset_for_tool(tool_name)


def get_available_toolsets() -> Dict[str, dict]:
    """Toolset availability info for UI display."""
    return registry.get_available_toolsets()


def check_toolset_requirements() -> Dict[str, bool]:
    """{toolset: available_bool} for every registered toolset."""
    return registry.check_toolset_requirements()


def check_tool_availability(quiet: bool = False) -> Tuple[List[str], List[dict]]:
    """(available_toolsets, unavailable_info)."""
    return registry.check_tool_availability(quiet=quiet)
