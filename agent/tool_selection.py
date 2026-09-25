"""Opt-in tool working sets, before provider conversion and cache decoration.

Selection changes disclosure, never authority. The same fresh session catalog supplies
selection, discovery and dispatch admission. No task text or callback errors are logged.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import time

from hermes_constants import get_hermes_home, hermes_home_key, profile_name_for_home
from tools.tool_search import (
    BRIDGE_TOOL_NAMES, assemble_tool_defs, bridge_tool_schemas,
    estimate_tokens_from_schemas, load_config_readonly,
)

logger = logging.getLogger(__name__)
SELECTION_SCHEMA_VERSION = "hermes.tool-selection.v1"
SELECTION_DEADLINE_SECONDS = 0.750
MAX_CATALOG_TOOLS = 4096
MAX_CATALOG_CHARS = 4 * 1024 * 1024


def invoke_selection_callbacks(callbacks, kwargs):
    """Cooperative deadline, bounded owners, isolated inputs, fail-closed arbitration.

    Python callbacks cannot be safely preempted. Reject late results without spawning
    abandoned workers; plugins MUST bound their own synchronous work and network I/O.
    """
    if len(callbacks) > 8:
        return [False]
    deadline = time.monotonic() + SELECTION_DEADLINE_SECONDS
    results = []
    for callback in tuple(callbacks):
        try:
            if time.monotonic() >= deadline:
                return [False]
            result = callback(**copy.deepcopy(kwargs))
            if time.monotonic() >= deadline:
                logger.warning("Tool selection deadline exceeded; callback was not preempted")
                return [False]
            if result is not None:
                results.append(result)
                if len(results) > 1:
                    return [False]  # Never union or let registration order choose an owner.
        except (Exception, SystemExit):
            # Exception messages can contain the private task; do not use the generic
            # middleware error reporter (which includes callback exception text).
            logger.warning("Tool selection callback failed; using discovery only")
            return [False]
    return results


def _plain_text(value, cap):
    return (isinstance(value, str) and len(value) <= cap
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in value))


def bounded_task_context(current, messages, current_index):
    """Only actual current input and prior plain conversational text, never tool bodies."""
    if not _plain_text(current, 128 * 1024) or not current:
        return None
    if not isinstance(messages, list) or not isinstance(current_index, int) or not 0 <= current_index <= len(messages):
        return None
    from agent.context_compressor import ContextCompressor
    recent, size = [], 0
    partial = current_index > 4096
    for row in reversed(messages[max(0, current_index - 4096):current_index]):
        if not isinstance(row, dict) or row.get("role") not in ("user", "assistant"):
            partial = True
            continue
        text = row.get("content")
        # Tool-call/reasoning turns aren't plain assistant conversation. Omit whole
        # rows rather than accidentally exporting embedded internal reasoning.
        if (row.get("tool_calls") or row.get("reasoning") or row.get("reasoning_content")
                or row.get("is_summary") or not isinstance(text, str) or not _plain_text(text, 32768)):
            partial = True
            continue
        if (ContextCompressor._is_context_summary_message(row)
                or ContextCompressor._is_synthetic_compression_user_turn(row)
                or "<think>" in text or "<analysis>" in text):
            partial = True
            continue
        if len(recent) == 4 or size + len(text) > 32768:
            partial = True
            continue
        recent.append({"role": row["role"], "content": text})
        size += len(text)
    return {"current_user_message": current, "recent_messages": list(reversed(recent)), "partial": partial}


def capture_selection_context(agent, current, messages, current_index):
    cfg = load_config_readonly()
    if not cfg.selection_enabled:
        agent._tool_selection_context = None
        agent._tool_selection_cache = None
        return
    if not cfg.defer_all or cfg.enabled == "off":
        raise ValueError("tools.tool_search.selection requires enabled: on and defer: all")
    if getattr(agent, "api_mode", "") == "codex_app_server" or getattr(agent, "provider", "") == "moa":
        raise ValueError("Tool selection is not supported by codex_app_server or MoA")
    agent._tool_selection_context = bounded_task_context(current, messages, current_index)
    agent._tool_selection_context_turn = getattr(agent, "_current_turn_id", "")
    agent._tool_selection_context_home = hermes_home_key()
    agent._tool_selection_cache = None


def _dynamic_schemas(agent, registry_defs):
    from agent.memory_manager import memory_provider_tools_enabled, normalize_tool_schema
    from tools.bot_mode_dm import message_agent_authorized, message_agent_tool_schema

    enabled = getattr(agent, "enabled_toolsets", None)
    disabled = getattr(agent, "disabled_toolsets", None)
    by_name = {s["function"]["name"]: s for s in registry_defs}
    providers = []
    manager = getattr(agent, "_memory_manager", None)
    if manager and memory_provider_tools_enabled(enabled, disabled, memory_tool_present="memory" in by_name):
        providers.append((manager.get_all_tool_schemas, False))
    engine = getattr(agent, "context_compressor", None)
    if engine and (enabled is None or "context_engine" in enabled) and "context_engine" not in (disabled or ()):
        providers.append((engine.get_tool_schemas, True))
    engine_names = set()
    for get_schemas, is_engine in providers:
        schemas = get_schemas()
        if not isinstance(schemas, (list, tuple)) or len(schemas) > MAX_CATALOG_TOOLS:
            raise ValueError("unavailable dynamic catalog")
        for raw in schemas:
            schema = normalize_tool_schema(raw)
            if schema is None or schema["name"] in BRIDGE_TOOL_NAMES:
                raise ValueError("invalid dynamic schema")
            name = schema["name"]
            if name not in by_name:
                by_name[name] = {"type": "function", "function": schema}
                if is_engine:
                    engine_names.add(name)
    if message_agent_authorized(agent):
        by_name["message_agent"] = message_agent_tool_schema()
    agent._context_engine_tool_names = engine_names
    return list(by_name.values())


def authorized_catalog(agent):
    """Fresh host-owned registry + dynamic inventory, respecting the session's scope.

    Never take an inventory from tool arguments or fall back to stale agent.tools on
    failure. This routine does not call routing plugins and is safe at dispatch time.
    """
    import model_tools
    from agent.oneshot_footprint import prune_oneshot_tools
    from tools.connectors.turn import side_agent_tool_drops

    try:
        if getattr(agent, "_tool_catalog_home", hermes_home_key()) != hermes_home_key():
            raise ValueError("wrong profile for session catalog")
        schemas = model_tools.get_tool_definitions(
            enabled_toolsets=getattr(agent, "enabled_toolsets", None),
            disabled_toolsets=getattr(agent, "disabled_toolsets", None),
            quiet_mode=True, skip_tool_search_assembly=True,
        ) or []
        schemas = _dynamic_schemas(agent, schemas)
        drops = side_agent_tool_drops(agent)
        schemas = [s for s in prune_oneshot_tools(schemas) if s["function"]["name"] not in drops]
        if len(schemas) > MAX_CATALOG_TOOLS:
            raise ValueError("catalog too large")
        if any(s.get("type") != "function" or not isinstance(s["function"].get("description", ""), str)
               or not isinstance(s["function"].get("parameters"), dict) for s in schemas):
            raise ValueError("invalid function schema")
        names = [s["function"]["name"] for s in schemas]
        if len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("invalid catalog")
        # JSON is also the detachment boundary: schemas must be finite JSON data.
        raw = json.dumps(schemas, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(raw) > MAX_CATALOG_CHARS:
            raise ValueError("catalog too large")
        raw.encode("utf-8")  # Refuse lone surrogates before revision hashing/provider serialization.
        schemas = json.loads(raw)
    except Exception:
        logger.warning("Session tool catalog unavailable; no task tools authorized")
        schemas = []
    agent._authorized_tool_catalog = schemas
    agent.valid_tool_names = {s["function"]["name"] for s in schemas} | set(BRIDGE_TOOL_NAMES)
    return schemas


def _select(agent, catalog, revision, cfg):
    from hermes_cli.middleware import middleware_payload
    from hermes_cli.plugins import invoke_middleware

    context = getattr(agent, "_tool_selection_context", None)
    if (context is None or getattr(agent, "_tool_selection_context_turn", None) != getattr(agent, "_current_turn_id", "")
            or getattr(agent, "_tool_selection_context_home", None) != hermes_home_key()):
        return []
    budget = {"max_tools": cfg.selection_max_tools, "max_schema_tokens": cfg.selection_max_schema_tokens}
    try:
        results = invoke_middleware("tool_selection", **middleware_payload(
            tool_selection_schema_version=SELECTION_SCHEMA_VERSION,
            catalog=copy.deepcopy(catalog), catalog_revision=revision, task_context=copy.deepcopy(context),
            session_id=getattr(agent, "session_id", "") or "",
            turn_id=getattr(agent, "_current_turn_id", "") or "",
            profile_name=profile_name_for_home(get_hermes_home()) or "default", budget=budget,
        ))
        if len(results) != 1 or not isinstance(results[0], dict):
            return []
        result = results[0]
        names = result.get("selected_tools")
        if (result.get("catalog_revision") != revision or not isinstance(names, list)
                or len(names) > cfg.selection_max_tools
                or any(not isinstance(n, str) for n in names)
                or len(set(names)) != len(names)
                or not _plain_text(result.get("source"), 128)
                or not _plain_text(result.get("reason"), 512)):
            return []
        by_name = {row["name"]: row["schema"] for row in catalog}
        if any(n not in by_name for n in names):
            return []
        selected = [by_name[n] for n in names]
        if estimate_tokens_from_schemas(selected) > cfg.selection_max_schema_tokens:
            return []
        return names
    except (Exception, SystemExit):
        logger.warning("Tool selection unavailable; using discovery only")
        return []


def _preserve_tool_choice(agent, offered, schemas):
    """Refuse unknown/unsupported choices rather than silently weakening an explicit target."""
    overrides = getattr(agent, "request_overrides", None) or {}
    extra = overrides.get("extra_body") or {}
    if "tools" in overrides or "tools" in extra:
        raise ValueError("Tool working sets cannot be combined with request_overrides.tools")
    choices = [overrides.get("tool_choice"), extra.get("tool_choice")]
    by_name = {s["function"]["name"]: s for s in schemas + offered}
    for choice in choices:
        if choice is None or choice in ("auto", "none", "required"):
            continue
        if not isinstance(choice, dict) or choice.get("type") != "function":
            raise ValueError("Unsupported tool_choice with tool working sets")
        mode = getattr(agent, "api_mode", "")
        if mode not in ("chat_completions", "codex_responses"):
            raise ValueError("Explicit function tool_choice is unsupported by this working-set transport")
        if mode == "codex_responses" and "function" in choice:
            raise ValueError("Responses tool_choice requires a flat name, not Chat function shape")
        if mode == "chat_completions" and not isinstance(choice.get("function"), dict):
            raise ValueError("Chat tool_choice requires a function object")
        target = choice.get("name") or (choice.get("function") or {}).get("name")
        if target == "tool_search":
            raise ValueError("Explicit tool_search choice is unsupported with provider bridge aliases")
        if not isinstance(target, str) or target not in by_name:
            raise ValueError("Unauthorized tool_choice target with tool working sets")
        if target not in {s["function"]["name"] for s in offered}:
            offered.append(by_name[target])
    return offered


def tools_for_request(agent):
    cfg = load_config_readonly()
    if cfg.selection_enabled and (not cfg.defer_all or cfg.enabled == "off"):
        raise ValueError("tools.tool_search.selection requires enabled: on and defer: all")
    if not cfg.defer_all:
        return agent.tools  # Native default, including cache identity, is unchanged.
    if getattr(agent, "api_mode", "") == "codex_app_server" or getattr(agent, "provider", "") == "moa":
        raise ValueError("defer: all is not supported by codex_app_server or MoA")
    schemas = authorized_catalog(agent)
    if cfg.enabled == "off":
        return schemas
    if not cfg.selection_enabled:
        return _preserve_tool_choice(agent, assemble_tool_defs(schemas, config=cfg).tool_defs, schemas)
    catalog = [{"name": s["function"]["name"], "description": s["function"].get("description", ""), "schema": s}
               for s in sorted(schemas, key=lambda s: s["function"]["name"])]
    raw = json.dumps(catalog, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    revision = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    key = (hermes_home_key(), getattr(agent, "_current_turn_id", ""), revision,
           cfg.selection_max_tools, cfg.selection_max_schema_tokens)
    cached = getattr(agent, "_tool_selection_cache", None)
    if cached is None or cached[0] != key:
        names = _select(agent, catalog, revision, cfg)
        fresh = authorized_catalog(agent)
        if fresh != schemas:
            # Refuse the stale decision. Next assembly sees the new revision.
            schemas, names = fresh, []
            agent._tool_selection_cache = None
        else:
            agent._tool_selection_cache = (key, names)
    else:
        names = cached[1]
    from tools.connectors.search import connections_in_scope
    offered = bridge_tool_schemas(len(schemas), connections_granted=connections_in_scope(schemas))
    by_name = {s["function"]["name"]: s for s in schemas}
    offered.extend(by_name[n] for n in names)
    offered = _preserve_tool_choice(agent, offered, schemas)
    task_tools = [s for s in offered if s["function"]["name"] not in BRIDGE_TOOL_NAMES]
    if (len(task_tools) > cfg.selection_max_tools
            or estimate_tokens_from_schemas(task_tools) > cfg.selection_max_schema_tokens):
        raise ValueError("Explicit tool_choice exceeds tool selection budget")
    agent._request_visible_tool_names = {s["function"]["name"] for s in offered}
    return copy.deepcopy(offered)


def dispatch_scope_error(agent, name):
    """Admission check for direct and unwrapped calls; never grants a handler shortcut."""
    if not load_config_readonly().defer_all:
        return None
    schemas = authorized_catalog(agent)
    if name not in BRIDGE_TOOL_NAMES and name not in {s["function"]["name"] for s in schemas}:
        return "Tool is not available in this session. Use tool_search to refresh."
    return None


def catalog_lookup_executor(name):
    if name not in ("tool_search", "tool_describe") or not load_config_readonly().defer_all:
        return None
    return lambda agent, args, ctx: dispatch_catalog_lookup(agent, name, args, ctx)


def dispatch_catalog_lookup(agent, name, args, ctx):
    """Native inline executor: policy/hooks run before this host-owned catalog read."""
    from tools import tool_search as ts
    dispatch = {"tool_search": ts.dispatch_tool_search, "tool_describe": ts.dispatch_tool_describe}[name]
    return dispatch(args, current_tool_defs=authorized_catalog(agent))
