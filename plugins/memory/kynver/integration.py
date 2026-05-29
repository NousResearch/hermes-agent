"""Wire Kynver operating substrate via generic Hermes plugin provider seams."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from .agentos_bridge import KynverAgentOSClient, load_kynver_agentos_config
from .context_envelope import format_context_envelope_block, load_context_envelope
from .operating_context import load_operating_context
from .session_manager import KynverSessionManager
from .skills_bridge import format_agentos_skills_index, list_agentos_skill_manifest
from .substrate import allow_local_fallback, resolve_memory_provider_name, substrate_active
from .todo_store import KynverTodoStore

logger = logging.getLogger(__name__)

_health_cache: dict[str, bool] = {}


def _health_cache_key(config: Mapping[str, Any] | None) -> str:
    cfg = load_kynver_agentos_config()
    return f"{cfg.api_url}:{cfg.slug}:{bool(cfg.api_key)}"


def provide_kynver_todo_store(agent: Any, config: Mapping[str, Any] | None = None) -> Any | None:
    if not substrate_active(config=config, client=None):
        return None
    client = KynverAgentOSClient()
    ctx = load_operating_context(config=config)
    return KynverTodoStore(
        client,
        operating_context=ctx,
        allow_fallback=allow_local_fallback(config),
        hermes_session_id=getattr(agent, "session_id", "") or "",
    )


def configure_agent_extension(
    agent: Any,
    config: Mapping[str, Any],
    *,
    platform: str | None = None,
    skip_memory: bool = False,
) -> None:
    """Attach Kynver session metadata and prompt substrate state to an AIAgent."""

    agent._kynver_active = False
    agent._kynver_degraded = False
    agent._kynver_context_block = ""
    agent._kynver_skills_block = ""
    agent._kynver_session_manager = None
    agent._kynver_client = None
    agent._kynver_operating_context = load_operating_context(config=config)

    if skip_memory or not substrate_active(config=config):
        return

    client = KynverAgentOSClient()
    agent._kynver_client = client
    agent._kynver_active = True
    _health_cache[_health_cache_key(config)] = True

    session_mgr = KynverSessionManager(client)
    agent._kynver_session_manager = session_mgr

    channel = platform or getattr(agent, "platform", None) or "hermes"
    model = getattr(agent, "model", "") or ""
    session_mgr.open_session(
        channel=str(channel),
        model=str(model),
        hermes_session_id=getattr(agent, "session_id", "") or "",
        metadata={"source": "hermes-forge"},
    )

    envelope = load_context_envelope(client, agent._kynver_operating_context)
    agent._kynver_context_block = format_context_envelope_block(envelope)

    manifest = list_agentos_skill_manifest(client)
    agent._kynver_skills_block = format_agentos_skills_index(manifest)

    store = getattr(agent, "_todo_store", None)
    if isinstance(store, KynverTodoStore):
        agent._kynver_degraded = bool(store.degraded)


def prompt_context_blocks(agent: Any) -> List[str]:
    blocks: List[str] = []
    ctx_block = getattr(agent, "_kynver_context_block", "") or ""
    if ctx_block:
        blocks.append(ctx_block)
    skills_block = getattr(agent, "_kynver_skills_block", "") or ""
    if skills_block:
        blocks.append(skills_block)
    store = getattr(agent, "_todo_store", None)
    if isinstance(store, KynverTodoStore) and store.degraded:
        blocks.append(
            "[Kynver: DEGRADED — todo/current-focus using local Hermes fallback; "
            "AgentOS ownership inactive for this session]"
        )
    elif getattr(agent, "_kynver_degraded", False):
        blocks.append(
            "[Kynver: degraded mode — some operating state is using local Hermes fallback]"
        )
    return blocks


def memory_provider_init_kwargs_extension(agent: Any) -> Dict[str, Any]:
    mgr = getattr(agent, "_kynver_session_manager", None)
    if mgr and mgr.agentos_session_id:
        return {"agentos_session_id": mgr.agentos_session_id}
    return {}


_seams_registered = False


def ensure_kynver_operating_seams_registered() -> None:
    """Idempotent registration for exclusive memory plugins (skip PluginManager)."""

    global _seams_registered
    if _seams_registered:
        return
    register_operating_providers(_MinimalPluginContext())
    _seams_registered = True


class _MinimalPluginContext:
    """Shim ctx when Kynver loads outside PluginManager.discover_plugins()."""


def register_operating_providers(ctx) -> None:
    """Register Kynver on generic Hermes plugin provider seams."""

    from hermes_cli.plugins import (
        register_plugin_agent_configure_hook,
        register_plugin_memory_init_kwargs_provider,
        register_plugin_memory_provider_resolver,
        register_plugin_prompt_context_provider,
        register_plugin_todo_store_provider,
    )

    from .operating_hooks import register_operating_hooks

    plugin_name = "kynver"
    register_plugin_todo_store_provider(plugin_name, provide_kynver_todo_store, priority=100)
    register_plugin_prompt_context_provider(plugin_name, prompt_context_blocks, priority=100)
    register_plugin_memory_provider_resolver(plugin_name, resolve_memory_provider_name, priority=100)
    register_plugin_memory_init_kwargs_provider(
        plugin_name,
        memory_provider_init_kwargs_extension,
        priority=100,
    )
    register_plugin_agent_configure_hook(plugin_name, configure_agent_extension, priority=100)

    if hasattr(ctx, "register_todo_store_provider"):
        ctx.register_todo_store_provider(provide_kynver_todo_store, priority=100)
    register_operating_hooks(ctx)
