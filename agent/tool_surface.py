"""Non-mutating assembly of the complete model-facing tool surface.

Registry tools are resolved first. External memory providers and context
engines expose schemas outside the registry, so callers pass those families
here before schema sanitization and tool-search replacement. Keeping this as
one final assembly step prevents diagnostics, agent initialization, and MCP
refreshes from disagreeing about the tools sent to the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FullToolSurface:
    """Result and diagnostics from one complete tool-surface assembly."""

    tool_defs: List[Dict[str, Any]]
    pre_assembly_tool_defs: List[Dict[str, Any]]
    injected_names: Dict[str, List[str]] = field(default_factory=dict)
    skipped: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    tool_search_activated: bool = False
    deferred_names: List[str] = field(default_factory=list)
    deferred_tokens: int = 0
    threshold_tokens: int = 0


def _tool_name(tool_def: Dict[str, Any]) -> str:
    function = tool_def.get("function") if isinstance(tool_def, dict) else None
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return name if isinstance(name, str) else ""


def _family_enabled(
    family: str,
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
) -> bool:
    if disabled_toolsets:
        if family in disabled_toolsets:
            return False
        try:
            from toolsets import get_toolset, resolve_toolset

            for toolset in disabled_toolsets:
                # Match model_tools subtraction semantics: platform bundles and
                # posture toolsets preserve shared core tools.
                if toolset.startswith("hermes-") or (
                    get_toolset(toolset) or {}
                ).get("posture"):
                    continue
                if family in resolve_toolset(toolset):
                    return False
        except Exception:
            logger.debug(
                "Failed to resolve disabled toolsets for %s tools",
                family,
                exc_info=True,
            )
    if family == "memory":
        from agent.memory_manager import memory_provider_tools_enabled

        return memory_provider_tools_enabled(enabled_toolsets)
    if family == "context_engine":
        return enabled_toolsets is None or "context_engine" in enabled_toolsets
    return True


def _append_family(
    tool_defs: List[Dict[str, Any]],
    existing_names: set[str],
    *,
    family: str,
    schemas: Iterable[Any],
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
    injected_names: Dict[str, List[str]],
    skipped: Dict[str, List[Dict[str, str]]],
) -> None:
    from agent.memory_manager import normalize_tool_schema

    allowed = _family_enabled(family, enabled_toolsets, disabled_toolsets)
    for raw_schema in schemas:
        schema = normalize_tool_schema(raw_schema)
        if schema is None:
            logger.warning(
                "%s returned a tool schema with no resolvable name; skipping (%r)",
                family.replace("_", " ").title(),
                raw_schema,
            )
            skipped[family].append({"tool": "", "reason": "invalid schema"})
            continue
        name = schema["name"]
        if not allowed:
            skipped[family].append({"tool": name, "reason": "toolset disabled"})
            continue
        if name in existing_names:
            skipped[family].append({"tool": name, "reason": "duplicate tool name"})
            continue
        tool_defs.append({"type": "function", "function": schema})
        existing_names.add(name)
        injected_names[family].append(name)


def assemble_full_tool_surface(
    base_tool_defs: Iterable[Dict[str, Any]],
    *,
    enabled_toolsets: Optional[List[str]] = None,
    disabled_toolsets: Optional[List[str]] = None,
    memory_tool_schemas: Iterable[Any] = (),
    context_engine_tool_schemas: Iterable[Any] = (),
    apply_tool_search: bool = True,
    context_length: Optional[int] = None,
    tool_search_config: Any = None,
    quiet_mode: bool = True,
) -> FullToolSurface:
    """Return the final model-facing tool list without mutating any input.

    The order matches runtime behavior: copy registry definitions, normalize
    and deduplicate externally injected families, sanitize the combined schema
    list, then apply tool-search replacement once at the end.
    """
    from tools.schema_sanitizer import sanitize_tool_schemas

    # The sanitizer deep-copies each definition. Running it after injection
    # gives every family identical backend-compatibility treatment and makes
    # the non-mutation guarantee explicit.
    staged = [tool for tool in base_tool_defs if isinstance(tool, dict)]
    existing_names = {name for name in map(_tool_name, staged) if name}
    injected_names = {"memory": [], "context_engine": []}
    skipped = {"memory": [], "context_engine": []}

    _append_family(
        staged,
        existing_names,
        family="memory",
        schemas=memory_tool_schemas,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        injected_names=injected_names,
        skipped=skipped,
    )
    _append_family(
        staged,
        existing_names,
        family="context_engine",
        schemas=context_engine_tool_schemas,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        injected_names=injected_names,
        skipped=skipped,
    )

    try:
        sanitized = sanitize_tool_schemas(staged)
    except Exception as exc:  # pragma: no cover - preserve fail-soft loading
        logger.warning("Schema sanitization skipped: %s", exc)
        sanitized = list(staged)
    pre_assembly = list(sanitized)
    if not apply_tool_search:
        return FullToolSurface(
            tool_defs=sanitized,
            pre_assembly_tool_defs=pre_assembly,
            injected_names=injected_names,
            skipped=skipped,
        )

    try:
        from tools.tool_search import (
            assemble_tool_defs,
            classify_tools,
            load_config as load_tool_search_config,
        )

        config = tool_search_config or load_tool_search_config()
        assembly = assemble_tool_defs(
            sanitized,
            context_length=context_length,
            config=config,
        )
        deferred_names: List[str] = []
        if assembly.activated:
            _, deferred = classify_tools(sanitized)
            deferred_names = sorted(
                name for name in map(_tool_name, deferred) if name
            )
            if not quiet_mode:
                print(
                    f"🔎 Tool Search: {assembly.deferred_count} MCP/plugin tools deferred "
                    f"(~{assembly.deferred_tokens} tokens) behind tool_search/describe/call. "
                    f"Threshold ~{assembly.threshold_tokens} tokens."
                )
        return FullToolSurface(
            tool_defs=assembly.tool_defs,
            pre_assembly_tool_defs=pre_assembly,
            injected_names=injected_names,
            skipped=skipped,
            tool_search_activated=assembly.activated,
            deferred_names=deferred_names,
            deferred_tokens=assembly.deferred_tokens,
            threshold_tokens=assembly.threshold_tokens,
        )
    except Exception as exc:  # pragma: no cover - tool loading must fail soft
        logger.warning("Tool search assembly skipped: %s", exc)
        return FullToolSurface(
            tool_defs=sanitized,
            pre_assembly_tool_defs=pre_assembly,
            injected_names=injected_names,
            skipped=skipped,
        )


def assemble_agent_tool_surface(
    agent: Any,
    base_tool_defs: Iterable[Dict[str, Any]],
    *,
    quiet_mode: bool = True,
) -> FullToolSurface:
    """Read an agent's external schema providers and assemble a staged copy."""
    memory_schemas: List[Dict[str, Any]] = []
    try:
        memory_manager = getattr(agent, "_memory_manager", None)
        get_schemas = getattr(
            memory_manager, "get_all_tool_schemas", None
        ) if memory_manager else None
        if callable(get_schemas):
            result = get_schemas()
            if isinstance(result, (list, tuple)):
                memory_schemas = list(result)
    except Exception:
        logger.debug("Memory-provider tool collection skipped", exc_info=True)

    context_schemas: List[Dict[str, Any]] = []
    try:
        compressor = getattr(agent, "context_compressor", None)
        get_schemas = getattr(
            compressor, "get_tool_schemas", None
        ) if compressor else None
        if callable(get_schemas):
            result = get_schemas()
            if isinstance(result, (list, tuple)):
                context_schemas = list(result)
    except Exception:
        logger.debug("Context-engine tool collection skipped", exc_info=True)

    return assemble_full_tool_surface(
        base_tool_defs,
        enabled_toolsets=getattr(agent, "enabled_toolsets", None),
        disabled_toolsets=getattr(agent, "disabled_toolsets", None),
        memory_tool_schemas=memory_schemas,
        context_engine_tool_schemas=context_schemas,
        context_length=getattr(
            getattr(agent, "context_compressor", None), "context_length", None
        ),
        quiet_mode=quiet_mode,
    )
