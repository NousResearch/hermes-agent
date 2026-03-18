"""
Plugin Hook System

Implements OpenClaw's plugin hook architecture in Python. Plugins register
handlers for named hooks that fire at specific points in the agent pipeline.

Four execution modes (mirroring OpenClaw):
  VOID       -- Fire all handlers in parallel, ignore return values
  MODIFYING  -- Fire handlers sequentially, each can modify a shared result
  CLAIMING   -- Fire handlers in priority order, first non-None return wins
  SYNC_ONLY  -- Same as VOID but guaranteed sequential (for hooks needing order)

The 25 plugin hooks (ported from OpenClaw src/plugins/types.ts):

  Context injection:
    before_prompt_build     -- MODIFYING  inject text into system prompt
    before_model_resolve    -- CLAIMING   override model/provider selection

  Message lifecycle:
    before_inbound_message  -- MODIFYING  transform/block inbound messages
    after_inbound_message   -- VOID       observe inbound after processing
    before_outbound_message -- MODIFYING  transform outbound message text
    after_outbound_message  -- VOID       observe outbound after delivery

  Tool lifecycle:
    before_tool_call        -- MODIFYING  intercept/modify tool calls
    after_tool_call         -- VOID       observe tool results
    claim_tool              -- CLAIMING   plugins can claim ownership of a tool name

  Session lifecycle:
    on_session_start        -- VOID       new session created
    on_session_end          -- VOID       session ended
    on_session_compact      -- MODIFYING  modify compact summary text

  Agent lifecycle:
    on_agent_start          -- VOID       agent turn begins
    on_agent_end            -- VOID       agent turn ends

  Provider/model:
    register_provider       -- CLAIMING   register a custom LLM provider
    register_tts            -- CLAIMING   register a custom TTS engine
    register_web_search     -- CLAIMING   register a custom web search provider
    register_image_gen      -- CLAIMING   register a custom image gen provider

  Memory:
    memory_read             -- CLAIMING   read from custom memory backend
    memory_write            -- VOID       write to custom memory backend
    memory_search           -- CLAIMING   search custom memory backend

  Subagent:
    before_subagent_spawn   -- MODIFYING  modify subagent config before spawn
    after_subagent_result   -- VOID       observe subagent result

  Gateway:
    on_gateway_start        -- VOID       gateway process starts
    on_gateway_stop         -- VOID       gateway process stops

Errors in plugin hooks are caught and logged but never block the pipeline.
"""

import asyncio
import importlib.util
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

PLUGINS_DIR = get_hermes_home() / "plugins"


# =============================================================================
# Execution modes
# =============================================================================

class HookMode(Enum):
    VOID      = "void"       # parallel, ignore returns
    MODIFYING = "modifying"  # sequential, each can modify result
    CLAIMING  = "claiming"   # priority order, first non-None wins
    SYNC_ONLY = "sync_only"  # sequential, ignore returns


# =============================================================================
# Hook definitions
# =============================================================================

@dataclass
class HookDef:
    name: str
    mode: HookMode
    description: str


HOOK_DEFINITIONS: List[HookDef] = [
    # Context injection
    HookDef("before_prompt_build",     HookMode.MODIFYING, "Inject text into agent system prompt"),
    HookDef("before_model_resolve",    HookMode.CLAIMING,  "Override model/provider selection"),
    # Message lifecycle
    HookDef("before_inbound_message",  HookMode.MODIFYING, "Transform or block inbound messages"),
    HookDef("after_inbound_message",   HookMode.VOID,      "Observe inbound messages after processing"),
    HookDef("before_outbound_message", HookMode.MODIFYING, "Transform outbound message text"),
    HookDef("after_outbound_message",  HookMode.VOID,      "Observe outbound messages after delivery"),
    # Tool lifecycle
    HookDef("before_tool_call",        HookMode.MODIFYING, "Intercept or modify tool calls"),
    HookDef("after_tool_call",         HookMode.VOID,      "Observe tool results"),
    HookDef("claim_tool",              HookMode.CLAIMING,  "Claim ownership of a tool name"),
    # Session lifecycle
    HookDef("on_session_start",        HookMode.VOID,      "New session created"),
    HookDef("on_session_end",          HookMode.VOID,      "Session ended"),
    HookDef("on_session_compact",      HookMode.MODIFYING, "Modify compact summary text"),
    # Agent lifecycle
    HookDef("on_agent_start",          HookMode.VOID,      "Agent turn begins"),
    HookDef("on_agent_end",            HookMode.VOID,      "Agent turn ends"),
    # Provider registration
    HookDef("register_provider",       HookMode.CLAIMING,  "Register a custom LLM provider"),
    HookDef("register_tts",            HookMode.CLAIMING,  "Register a custom TTS engine"),
    HookDef("register_web_search",     HookMode.CLAIMING,  "Register a custom web search provider"),
    HookDef("register_image_gen",      HookMode.CLAIMING,  "Register a custom image generation provider"),
    # Memory
    HookDef("memory_read",             HookMode.CLAIMING,  "Read from custom memory backend"),
    HookDef("memory_write",            HookMode.VOID,      "Write to custom memory backend"),
    HookDef("memory_search",           HookMode.CLAIMING,  "Search custom memory backend"),
    # Subagent
    HookDef("before_subagent_spawn",   HookMode.MODIFYING, "Modify subagent config before spawn"),
    HookDef("after_subagent_result",   HookMode.VOID,      "Observe subagent result"),
    # Gateway
    HookDef("on_gateway_start",        HookMode.VOID,      "Gateway process starts"),
    HookDef("on_gateway_stop",         HookMode.VOID,      "Gateway process stops"),
]

HOOK_DEFS_BY_NAME: Dict[str, HookDef] = {h.name: h for h in HOOK_DEFINITIONS}


# =============================================================================
# Plugin registration entry
# =============================================================================

@dataclass
class PluginHookEntry:
    hook_name: str
    handler: Callable
    priority: int = 0       # Higher priority fires first
    plugin_name: str = ""


# =============================================================================
# Plugin registry
# =============================================================================

class PluginRegistry:
    """
    Discovers, loads, and fires plugin hooks.

    Usage:
        registry = PluginRegistry()
        registry.discover_and_load()

        # VOID hook (fire-and-forget)
        await registry.call("on_session_start", {"platform": "telegram"})

        # MODIFYING hook (returns modified value)
        prompt = await registry.call("before_prompt_build", ctx, initial="")

        # CLAIMING hook (first non-None return wins)
        provider = await registry.call("before_model_resolve", ctx)
    """

    def __init__(self):
        self._entries: Dict[str, List[PluginHookEntry]] = {}
        self._loaded_plugins: List[dict] = []

    @property
    def loaded_plugins(self) -> List[dict]:
        return list(self._loaded_plugins)

    def register(
        self,
        hook_name: str,
        handler: Callable,
        priority: int = 0,
        plugin_name: str = "",
    ) -> None:
        """Programmatically register a hook handler."""
        if hook_name not in HOOK_DEFS_BY_NAME:
            logger.warning("[plugins] Unknown hook name: %s", hook_name)
        entry = PluginHookEntry(
            hook_name=hook_name,
            handler=handler,
            priority=priority,
            plugin_name=plugin_name,
        )
        bucket = self._entries.setdefault(hook_name, [])
        bucket.append(entry)
        bucket.sort(key=lambda e: -e.priority)  # highest priority first

    def discover_and_load(self) -> None:
        """
        Scan PLUGINS_DIR for plugin directories and load their handlers.

        Each plugin directory must contain:
          - PLUGIN.yaml  with 'name', 'hooks' list
          - handler.py   with functions named after each hook
        """
        if not PLUGINS_DIR.exists():
            return

        for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
            if not plugin_dir.is_dir():
                continue

            manifest_path = plugin_dir / "PLUGIN.yaml"
            handler_path = plugin_dir / "handler.py"

            if not manifest_path.exists() or not handler_path.exists():
                continue

            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not manifest or not isinstance(manifest, dict):
                    logger.warning("[plugins] Skipping %s: invalid PLUGIN.yaml", plugin_dir.name)
                    continue

                plugin_name = manifest.get("name", plugin_dir.name)
                hooks = manifest.get("hooks", [])
                if not hooks:
                    logger.warning("[plugins] Skipping %s: no hooks declared", plugin_name)
                    continue

                # Security: ensure plugin path doesn't escape plugins dir
                try:
                    plugin_dir.resolve().relative_to(PLUGINS_DIR.resolve())
                except ValueError:
                    logger.error("[plugins] Path escape detected for %s — skipping", plugin_name)
                    continue

                spec = importlib.util.spec_from_file_location(
                    f"hermes_plugin_{plugin_name}", handler_path
                )
                if spec is None or spec.loader is None:
                    logger.warning("[plugins] Skipping %s: could not load handler.py", plugin_name)
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                registered = []
                priority = manifest.get("priority", 0)
                for hook_name in hooks:
                    fn = getattr(module, hook_name, None)
                    if fn is None:
                        logger.warning("[plugins] %s: no function '%s' in handler.py", plugin_name, hook_name)
                        continue
                    self.register(hook_name, fn, priority=priority, plugin_name=plugin_name)
                    registered.append(hook_name)

                if registered:
                    self._loaded_plugins.append({
                        "name": plugin_name,
                        "description": manifest.get("description", ""),
                        "hooks": registered,
                        "path": str(plugin_dir),
                    })
                    logger.info("[plugins] Loaded plugin '%s' for hooks: %s", plugin_name, registered)

            except Exception as e:
                logger.error("[plugins] Error loading plugin %s: %s", plugin_dir.name, e)

    async def call(
        self,
        hook_name: str,
        context: Optional[Dict[str, Any]] = None,
        initial: Any = None,
    ) -> Any:
        """
        Fire all handlers for a hook according to its execution mode.

        Args:
            hook_name: The hook to fire (e.g. "before_prompt_build").
            context:   Dict of context data passed to each handler.
            initial:   Starting value for MODIFYING hooks.

        Returns:
            VOID/SYNC_ONLY  → None
            MODIFYING       → final accumulated value (starts from `initial`)
            CLAIMING        → first non-None return value, or None
        """
        if context is None:
            context = {}

        entries = self._entries.get(hook_name, [])
        if not entries:
            return initial

        hook_def = HOOK_DEFS_BY_NAME.get(hook_name)
        mode = hook_def.mode if hook_def else HookMode.VOID

        async def _run(entry: PluginHookEntry, *args):
            try:
                result = entry.handler(context, *args)
                if asyncio.iscoroutine(result):
                    return await result
                return result
            except Exception as e:
                logger.error(
                    "[plugins] Error in plugin '%s' hook '%s': %s",
                    entry.plugin_name, hook_name, e
                )
                return None

        if mode == HookMode.VOID:
            await asyncio.gather(*[_run(e) for e in entries], return_exceptions=True)
            return None

        elif mode == HookMode.SYNC_ONLY:
            for entry in entries:
                await _run(entry)
            return None

        elif mode == HookMode.MODIFYING:
            value = initial
            for entry in entries:
                result = await _run(entry, value)
                if result is not None:
                    value = result
            return value

        elif mode == HookMode.CLAIMING:
            for entry in entries:
                result = await _run(entry)
                if result is not None:
                    return result
            return None

        return None
