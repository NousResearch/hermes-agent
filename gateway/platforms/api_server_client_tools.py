"""Split-runtime helpers for ``/v1/runs``: tools supplied by the API caller are relayed to it
for execution instead of running on the API-server host.

The route handler and the SSE notify bridge live in ``api_server_runs``; the suspend/resume
registry is ``tools.client_tool_gateway``. Off by default: the relay path only activates when
split-runtime is enabled AND a run carries a non-empty ``tools`` array.
"""

import json
import logging
import os
from functools import partial
from typing import Any, Optional

from gateway.platforms.api_server_room_grants import _json_error

logger = logging.getLogger("gateway.platforms.api_server")


def adapter_split_runtime(extra: dict) -> bool:
    """Adapter-wide flag: ``platforms.api_server.extra.split_runtime``, else the
    ``API_SERVER_SPLIT_RUNTIME`` environment variable."""
    from gateway.platforms import api_server as _api_server  # called from the adapter's __init__: no cycle

    return _api_server._coerce_request_bool(
        extra.get("split_runtime", os.getenv("API_SERVER_SPLIT_RUNTIME", "")))


def split_runtime_enabled(self) -> bool:
    """Whether split-runtime (client-tool relay) is enabled.

    Resolved from, in order:

      1. ``platforms.api_server.extra.split_runtime`` (adapter config),
      2. the ``API_SERVER_SPLIT_RUNTIME`` environment variable,
      3. the profile config's top-level ``api_server: split_runtime:`` block — the same place
         ``gateway_timeout`` and the other api_server knobs live, and the form the docs describe.

    (3) is a fallback rather than the primary source because the adapter's ``extra`` is the
    canonical platform-config channel; it exists because the top-level block is where an
    operator naturally puts this and the shared-key bridge in ``gateway.config`` does not carry
    the key into ``extra``.

    (1) and (2) are adapter-wide. (3) is resolved **per profile**: under
    ``gateway.multiplex_profiles`` one adapter serves every profile, and the request middleware
    has already entered the routed profile's runtime scope, so ``load_config()`` here reads that
    profile's config. The result is memoized per profile name — caching a single value would let
    whichever profile issued the first run decide the flag for all of them.
    """
    if self._split_runtime:
        return True
    from gateway.platforms import api_server as _api_server  # bound as an adapter method: no cycle at call time

    # "" is the un-prefixed/single-profile case (the default profile).
    profile_key = _api_server._api_request_profile.get() or ""
    cache = getattr(self, "_split_runtime_config_cache", None)
    if cache is None:
        cache = self._split_runtime_config_cache = {}
    if profile_key not in cache:
        try:
            from hermes_cli.config import load_config

            api_cfg = (load_config() or {}).get("api_server", {}) or {}
            cache[profile_key] = _api_server._coerce_request_bool(api_cfg.get("split_runtime", False))
        except Exception:
            logger.debug("split_runtime config lookup failed; treating as off", exc_info=True)
            cache[profile_key] = False
    return cache[profile_key]


def runtime_capabilities(enabled: bool) -> dict[str, Any]:
    """The ``runtime`` block of ``/v1/capabilities``."""
    return {
        "mode": "server_agent", "tool_execution": "split" if enabled else "server",
        "split_runtime": enabled,
        "description": (
            "The API server creates a server-side Hermes AIAgent. Host toolsets execute on the "
            "API-server host; when split_runtime is enabled, tools supplied in a run's 'tools' "
            "array are relayed to the client via tool_call.request events and resolved via "
            "/v1/runs/{run_id}/tool_result.")}


def _reserved_tool_names() -> set[str]:
    """Host tool names a client tool may not take. Parallel/path-scoped tools are batched before
    dispatch, and agent-level tools own their dispatch inside the executor, so a same-named
    client tool could not be intercepted with the same locality on every execution path."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from agent.tool_dispatch_helpers import _PARALLEL_SAFE_TOOLS, _PATH_SCOPED_TOOLS

    return set(_PARALLEL_SAFE_TOOLS) | set(_PATH_SCOPED_TOOLS) | set(INLINE_TOOL_EXECUTORS)


def parse_client_tools(self, body: dict, *, _openai_error) -> tuple[list[dict[str, Any]], Any]:
    """Validate a run body's ``tools`` array; returns ``(client_tools, error_response)``.
    Empty (and no error) when split-runtime is off or the run supplies no tools."""
    raw_tools = body.get("tools") if self._split_runtime_enabled() else None
    if not raw_tools:
        return [], None
    if not isinstance(raw_tools, list):
        return [], _json_error(_openai_error, "'tools' must be an array of tool definitions", status=400)
    reserved = _reserved_tool_names()
    seen: set[str] = set()
    client_tools: list[dict[str, Any]] = []
    for i, tool_def in enumerate(raw_tools):
        if not isinstance(tool_def, dict) or tool_def.get("type") != "function":
            return [], _json_error(
                _openai_error, f"tools[{i}] must be an object with type 'function'", status=400)
        name = (tool_def.get("function") or {}).get("name")
        if not name or not isinstance(name, str):
            return [], _json_error(_openai_error, f"tools[{i}].function.name is required", status=400)
        if name in reserved:
            return [], _json_error(
                _openai_error,
                f"tools[{i}].function.name '{name}' collides with a reserved host tool and "
                "cannot be a client tool", status=400)
        if name in seen:
            return [], _json_error(_openai_error, f"duplicate client tool name '{name}'", status=400)
        seen.add(name)
        client_tools.append(tool_def)
    return client_tools, None


def apply_client_tools(agent, client_tools: Optional[list[dict[str, Any]]], session_key: str) -> None:
    """Make *client_tools* the agent's client tools for this turn, tagged so the executor
    suspends and relays them instead of dispatching on the host.

    Canonical API turns reuse a cached agent, so this replaces whatever an earlier turn applied
    (``None`` or ``[]`` removes it) and restores host tools a client tool shadowed. The cached
    system prompt is only invalidated when the client tool set actually changes, so a client
    that sends the same tools every turn keeps its prompt-cache prefix."""
    previous = getattr(agent, "_client_tool_defs", None) or []
    defs = [fn for fn in ((t or {}).get("function", {}) or {} for t in client_tools or ()) if fn.get("name")]
    agent._client_tool_session_key = session_key or ""
    if defs == previous:
        return
    old_names = getattr(agent, "_client_tool_names", None) or set()
    # Copy before mutating: the tool structures may be shared with other agents.
    tools = [t for t in (getattr(agent, "tools", None) or [])
             if (t.get("function", {}) or {}).get("name") not in old_names]
    tools.extend(getattr(agent, "_client_tool_shadowed", None) or [])
    valid_names = (set(getattr(agent, "valid_tool_names", None) or set()) - old_names) | {
        (t.get("function", {}) or {}).get("name") for t in tools}
    names = {fn["name"] for fn in defs}
    # A client tool shadowing a same-named host tool wins: park the host entry so the relay path
    # handles the name, and give it back once the client stops supplying that tool.
    shadowed = [t for t in tools if (t.get("function", {}) or {}).get("name") in names]
    tools = [t for t in tools if (t.get("function", {}) or {}).get("name") not in names]
    tools.extend({"type": "function", "function": fn} for fn in defs)
    agent.tools, agent.valid_tool_names = tools, valid_names | names
    agent._client_tool_names, agent._client_tool_shadowed = names, shadowed
    # Stored so tools.mcp_tool_agent._reinject_post_build_tools re-appends them when a
    # turn-boundary rebuild refreshes agent.tools.
    agent._client_tool_defs = defs
    # The system prompt (with its <tools> block) is cached at build time: invalidate it so the
    # changed client tools appear in the model's tool documentation.
    try:
        agent._invalidate_system_prompt()
    except Exception:
        logger.debug("system prompt invalidation failed; clearing the cache directly", exc_info=True)
        agent._cached_system_prompt = None


def run_observer(self, run_id: str, client_tools: Optional[list[dict[str, Any]]],
                 session_key: str) -> dict[str, Any]:
    """Observer fields a canonical (session-authority) /v1/runs turn carries: a sink for its
    agent's memory-provider tool names (``classify_server_tool_kind`` needs them, and the run
    never holds the agent), plus the run's client tools when it supplied any. Request-owned
    like the SSE sinks: a turn recovered after a restart has no caller to relay to, so it runs
    without them."""
    fields: dict[str, Any] = {"memory_tool_names_callback": partial(self._run_memory_tool_names.__setitem__, run_id)}
    if client_tools:
        fields.update(client_tools=client_tools, client_tool_session_key=session_key)
    return fields


def wire_turn_observers(agent, observers) -> None:
    """Per-turn wiring of a canonical API turn's cached agent: report its memory-tool names to
    the run, and apply this turn's client tools (or none, dropping an earlier turn's)."""
    manager = getattr(agent, "_memory_manager", None)
    memory_names = frozenset(manager.get_all_tool_names()) if manager is not None else frozenset()
    for observer in observers:
        report = observer.get("memory_tool_names_callback")
        if report is not None:
            report(memory_names)
    observer = next((o for o in observers if o.get("client_tools")), {})
    apply_client_tools(agent, observer.get("client_tools"), observer.get("client_tool_session_key", ""))


def classify_server_tool_kind(self, run_id: str, tool_name: Optional[str]) -> Optional[str]:
    """Best-effort provenance tag for a server-executed tool, surfaced on the Runs
    ``tool.started`` event so split-runtime shells can render a distinct badge. Returns
    ``"mcp"``, ``"memory"``, or ``None`` (plain server tool); never raises."""
    if not tool_name:
        return None
    # MCP tools are registered under the ``mcp_{server}_{tool}`` namespace.
    if tool_name.startswith("mcp_"):
        return "mcp"
    # Memory-provider tools are only knowable from the agent's memory manager: reported by a
    # canonical turn's wiring, or read from the agent an in-process run built itself.
    if tool_name in self._run_memory_tool_names.get(run_id, ()):
        return "memory"
    memory_manager = getattr(self._active_run_agents.get(run_id), "_memory_manager", None)
    try:
        if memory_manager is not None and memory_manager.has_tool(tool_name):
            return "memory"
    except Exception:
        logger.debug("memory-tool provenance lookup failed for %s", tool_name, exc_info=True)
    return None


def tool_result_text(body: dict, *, _coerce_request_bool) -> str:
    """Normalize a ``tool_result`` body into the result string handed to the model: accepts
    ``output`` | ``result`` | ``content``; dict/list become JSON; an error flag wraps the
    payload as ``{"error": ...}``."""
    is_error = any(_coerce_request_bool(body.get(k), default=False) for k in ("error", "is_error"))
    raw_output = next((body[k] for k in ("output", "result", "content") if body.get(k) is not None), None)
    if isinstance(raw_output, (dict, list)):
        return json.dumps(raw_output, ensure_ascii=False)
    if raw_output is None:
        return json.dumps(
            {"error": "client tool returned no output"} if is_error else {"result": None}, ensure_ascii=False)
    return json.dumps({"error": str(raw_output)}, ensure_ascii=False) if is_error else str(raw_output)
