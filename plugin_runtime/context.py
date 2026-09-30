"""Plugin-facing runtime registration context.

This module owns the context object handed to plugin register() functions. Host integrations
remain lazy at their real subsystem boundaries; plugin runtime contracts and config access stay
owned by plugin_runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import re
from contextlib import suppress
from functools import cached_property, wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from hermes_constants import get_hermes_home, get_process_hermes_home, hermes_home_key
from plugin_runtime.capabilities import plugin_capability_granted
from plugin_runtime.config_bridge import (
    load_plugin_config_for_home,
    plugin_gateway_injection_allowed,
    plugin_setting_segments,
    read_plugin_mcp_allowlist,
    read_plugin_setting,
    write_plugin_setting,
)
from plugin_runtime.dispatch import (
    DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    HERMES_EVENT_NAMESPACE,
    MAX_SYSTEM_PROMPT_SECTION_CHARS,
    SYSTEM_PROMPT_SECTION_POSITIONS,
    VALID_HOOKS,
    VALID_MIDDLEWARE,
    PluginSystemPromptSection,
    is_valid_system_prompt_section_id,
)
from plugin_runtime.loading import _serialized_replacement
from plugin_runtime.manifest import PluginManifest, manifest_key
from plugin_runtime.registration import PluginRegistration, replacement_coordinator
from plugin_runtime.state import PluginState
from plugin_runtime.contracts import RegisteredApprovalTransport
from plugin_runtime.host_bindings import get_plugin_host_callback


logger = logging.getLogger("hermes_cli.plugins")
_UNSET = object()


class PluginToolOverrideError(PermissionError):
    """Plugin tried to override a built-in tool without ``plugins.entries.<id>.allow_tool_override``."""


class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        self._llm: Any = None  # lazy; tests preseed it (see ``llm``)
        # Set when this context's load overran ``plugins.load_timeout_seconds``: the abandoned worker may
        # still be running register(), and nothing it registers from then on may reach a registry.
        self._load_abandoned = False

    def _abandon_load(self) -> None:
        """Mark this load as timed out; every later ``register_*``/``subscribe``/``on_unload`` is ignored."""
        self._load_abandoned = True

    @property
    def plugin_id(self) -> str:
        """Return the effective registry id used for this plugin's namespaces."""
        return manifest_key(self.manifest)

    def has_plugin(self, plugin_id: str) -> bool:
        """Return True when another plugin is loaded and enabled (runtime probe for advisory
        ``requires_plugins``). Matches on registry key or manifest name.

        See #64165.
        """
        return any(
            loaded.enabled and (key == plugin_id or loaded.manifest.name == plugin_id)
            for key, loaded in self._manager._plugins.items()
        )

    def _segments(self, key: str) -> tuple[str, ...]:
        """Validated plugin-relative settings path (warn + re-raise on rejection)."""
        try:
            return plugin_setting_segments(key)
        except ValueError:
            logger.warning("Rejected config path %r from plugin %s", key, self.plugin_id)
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read plugin-relative ``plugins.entries.<plugin_id>.settings.<key>`` (falls back to the
        legacy ``config`` subtree for migration compatibility)."""
        return read_plugin_setting(self.plugin_id, self._segments(key), default)

    def set_config(self, key: str, value: Any) -> None:
        """Atomically write one value in this plugin's ``settings`` subtree."""
        write_plugin_setting(self.plugin_id, self._segments(key), value)

    @cached_property
    def state(self) -> PluginState:
        """This plugin's profile-scoped durable JSON state facade."""
        return PluginState(self.plugin_id, self.manifest.skill_namespace)

    @cached_property
    def platform_actions(self):
        """Capability-gated platform action facade (``add_reaction``, ``set_thread_title``). Every call
        re-checks ``gateway.platform_actions`` (legacy ``plugins.entries.<id>.allow_platform_actions``,
        default OFF) and returns ``{"ok": bool, ...}`` — verbs never raise into hook dispatch; no adapter
        handles or raw SDK objects."""
        factory = get_plugin_host_callback("platform_actions_factory")
        if factory is None:
            raise RuntimeError("platform actions host is unavailable")
        return factory(self.plugin_id)

    def _wrong_type(self, obj: Any, base_class: type, label: str, article: str = "a") -> bool:
        """Warn-and-ignore gate shared by every registrar that requires a base class."""
        if isinstance(obj, base_class):
            return False
        logger.warning("Plugin '%s' tried to register %s %s that does not inherit from %s. Ignoring.",
                       self.manifest.name, article, label, base_class.__name__)
        return True

    def _refuse(self, what: str) -> ValueError:
        """``ValueError`` for a malformed registration (``what`` completes "tried to register ...")."""
        return ValueError(f"Plugin '{self.manifest.name}' tried to register {what}.")

    def _track(
        self, kind: str, key: str, release: Callable[[], None], *, persistent: bool = False,
    ) -> PluginRegistration:
        """Record host-owned cleanup for a successful registration (see
        :meth:`PluginManager._track_registration` for ``persistent``)."""
        return self._manager._track_registration(self.manifest, kind, key, release, persistent=persistent)

    def _track_replacement(
        self, kind: str, key: str, *, slot: tuple, current: Any, previous: Any,
        restore: Callable[[Any], bool],
    ) -> PluginRegistration:
        """Track one generation in a replaceable manager-local registration slot."""
        lease = replacement_coordinator.acquire(slot, current=current, previous=previous, restore=restore)
        return self._track(kind, key, lease.dispose)

    def _track_mapping_entry(
        self, kind: str, key: str, mapping: Dict[str, Any], entry: Any, previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Store ``entry`` under ``key`` in a manager-local mapping and lease the slot; unload restores
        ``previous`` (default: the displaced entry, or removes the key) only while ``entry`` is still
        current."""
        if previous is _UNSET:
            previous = mapping.get(key)
        mapping[key] = entry
        return self._track_replacement(
            kind, key, slot=("manager_mapping", id(mapping), key), current=entry, previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(mapping, key, entry, replacement),
        )

    def _register_entry(
        self, kind: str, key: str, mapping: Dict[str, Any], entry: Any, log_fmt: str, *log_args: Any,
        previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Shared tail of the manager-mapping registrars: store + lease the entry, then log
        ``log_fmt % (plugin name, *log_args)`` at debug."""
        handle = self._track_mapping_entry(kind, key, mapping, entry, previous)
        logger.debug(log_fmt, self.manifest.name, *log_args)
        return handle

    def _register_scoped_provider(
        self, provider: Any, *, kind: str, base_class: type, registry: Any, label: str,
        article: str = "a", normalize: Optional[Callable[[str], str]] = lambda n: n.strip(),
        register: Optional[Callable[..., Any]] = None, reject_message: Optional[str] = None,
    ) -> Optional[PluginRegistration]:
        """Shared body of the ``register_<category>_provider`` methods: type-check (warn + ignore),
        register in the scope-keyed ``registry``, lease the slot so unload restores the displaced entry.
        ``None`` when the registry refused/replaced the provider (``ValueError`` with ``reject_message``
        set, or a falsy ``register``)."""
        if self._wrong_type(provider, base_class, label, article):
            return None
        registry_name = provider.name if normalize is None else normalize(provider.name)
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(registry_name, scope=scope)
        try:
            accepted = (register or registry.register_provider)(provider, scope=scope)
        except ValueError as exc:
            if reject_message is None:
                raise
            logger.warning(reject_message, self.manifest.name, exc)
            return None
        if (register is not None and not accepted) or registry.snapshot_registration(
            registry_name, scope=scope
        ) is not provider:
            return None
        handle = self._manager._track_scoped_registration(
            self.manifest, kind, registry_name, registry, provider, previous
        )
        logger.info("Plugin '%s' registered %s: %s", self.manifest.name, label, registry_name)
        return handle

    @property
    def llm(self) -> Any:
        """Host-owned :class:`agent.plugin_llm.PluginLlm` facade: completions on the user's active
        model/auth. Overrides (model, agent id, auth profile) are fail-closed, gated by
        ``plugins.entries.<plugin_id>.llm.*``."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            self._llm = PluginLlm(plugin_id=self.plugin_id)
        return self._llm

    @cached_property
    def subagent_lifecycle(self) -> Any:
        """Plugin-safe subagent lifecycle service: serializable handles and immutable snapshots,
        never a live agent or private registry."""
        from agent.subagent_lifecycle import SubagentLifecycleService, get_active_subagent_parent
        return SubagentLifecycleService(get_active_subagent_parent)

    @property
    def profile_name(self) -> str:
        """Active profile name (``"default"``, the ``~/.hermes/profiles/<name>`` id, or ``"custom"``),
        derived from ``HERMES_HOME`` — not ``_cli_ref``, which is None outside the interactive CLI —
        so gateway and kanban workers get it too."""
        try:
            from hermes_constants import get_default_hermes_root

            home = self._manager.home_path.resolve()
            default_home = get_default_hermes_root().resolve()
            if home == default_home:
                return "default"
            parts = home.relative_to(default_home / "profiles").parts
            if len(parts) == 1 and parts[0]:
                return parts[0]
            return "custom"
        except (OSError, RuntimeError, ValueError):
            return "custom"

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register a cleanup callback for unload: runs in reverse acquisition order interleaved
        with registration teardown; exceptions are logged, never propagated."""
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro, *, name: Optional[str] = None) -> "asyncio.Task":
        """Spawn a supervised asyncio task; unload/force reload cancels it. Needs a running loop."""
        if not asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        loop = asyncio.get_running_loop()
        task_name = name or f"plugin:{self.plugin_id}:task"
        task = loop.create_task(coro, name=task_name)
        handle = self._track("background_task", task_name, lambda: task.done() or task.cancel())
        task.add_done_callback(lambda _t: handle.dispose())
        logger.debug("Plugin %s spawned supervised task: %s", self.manifest.name, task_name)
        return task

    def register_approval_transport(self, name: str, present_fn: Callable) -> None:
        """Register a human approval transport, inactive until ``security.approval.transport:
        <name>`` selects it. It receives a redacted ``ApprovalRequest`` and returns only a
        correlated decision; policy and persistence stay host-owned. ``present_fn`` may be async."""
        transports = self._manager._approval_transports
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in transports:
            owner = transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        entry = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=self.plugin_id,
            profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", self.plugin_id, clean)
        # Duplicate names are rejected above, so there is never a displaced previous entry to restore;
        # tracking makes unload/force-reload remove this transport.
        self._track_mapping_entry("approval_transport", clean, transports, entry, None)

    @_serialized_replacement
    def register_tool(
        self, name: str, toolset: str, schema: dict, handler: Callable,
        check_fn: Callable | None = None, requires_env: list | None = None, is_async: bool = False,
        description: str = "", emoji: str = "", override: bool = False,
    ) -> Optional[PluginRegistration]:
        """Register a tool in the global registry and track it as plugin-provided. ``override=True``
        replaces a same-named built-in (without it a name claimed by another toolset is rejected) and
        needs operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override: true`` — otherwise
        any enabled plugin could silently replace a privileged built-in like ``write_file``.

        ``override=True`` against a built-in tool requires the operator to opt in via
        ``plugins.entries.<plugin_id>.allow_tool_override: true`` in config.yaml — mirrors the trust gate
        pattern used for ``ctx.llm`` provider/model overrides (#23194).
        """
        if override and not self._tool_override_allowed(name):
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool {name!r}. Set "
                f"plugins.entries.{self.plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )
        from tools.registry import registry
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(name, scope=scope)
        if previous is None and not override and registry.get_entry(name, scope=scope) is not None:
            logger.warning("Plugin %s tried to shadow global tool %s without override=True",
                           self.manifest.name, name)
            return None
        registry.register(
            name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
            requires_env=requires_env, is_async=is_async, description=description, emoji=emoji,
            override=override, scope=scope,
        )
        registered = registry.snapshot_registration(name, scope=scope)
        handle = None
        if registered is not None and registered is not previous and registered.handler is handler:
            self._manager._plugin_tool_names.add(name)
            handle = self._manager._track_scoped_registration(
                self.manifest, "tool", name, registry, registered, previous,
                finalize=lambda: self._manager._remove_tool_name_if_unowned(name),
            )
        logger.debug("Plugin %s registered tool: %s%s", self.manifest.name, name,
                     " (override)" if override else "")
        return handle

    # -- capability probing (#64228) -----------------------------------------
    def has_capability(self, capability: str) -> bool:
        """True when *capability* is live for this plugin (probe, then degrade gracefully). Bundled
        plugins are trusted for ``tools.override``; otherwise granted_capabilities or the legacy
        ``allow_*`` key decides. Unknown ids / unreadable consent -> False (fail closed)."""
        if self.manifest.source == "bundled" and capability == "tools.override":
            return True
        return plugin_capability_granted(self.plugin_id, capability)

    def call_mcp(
        self, server: str, tool: str, arguments: Optional[Dict[str, Any]] = None,
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """Call ``tool`` on MCP ``server`` synchronously through :mod:`tools.mcp_tool`'s native client
        (same trust gates, breaker, reconnect — never a parallel connection). Servers not in
        ``plugins.entries.<plugin_id>.mcp_allowlist`` raise ``PermissionError`` (default-deny). ``timeout``
        clamps to 1–600s; results over ~64KB are truncated with a marker.

        This is a per-server grant, deliberately not ambient authority over every configured server.
        TODO(#64228): swap the per-server allowlist for the declared capability model once it lands
        (per-tool grants, expiry, ro/rw).
        """
        if server not in self._mcp_allowlist(self.plugin_id):
            raise PermissionError(
                f"Plugin {self.manifest.name!r} is not allowed to call MCP "
                f"server {server!r}. Add it to "
                f"plugins.entries.{self.plugin_id}.mcp_allowlist in config.yaml "
                f"to grant access (default is no MCP access)."
            )
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        timeout = max(1.0, min(timeout, 600.0))
        from tools.mcp_tool_handlers import _make_tool_handler
        raw = _make_tool_handler(server, tool, timeout)(dict(arguments or {}))
        logger.debug("Plugin %s called MCP %s/%s (timeout=%ss, %d chars returned)",
                     self.manifest.name, server, tool, timeout, len(raw or ""))
        return self._mcp_envelope(raw)

    _MCP_RESULT_CHAR_CAP = 65536

    @classmethod
    def _mcp_envelope(cls, raw: Any) -> Dict[str, Any]:
        """Normalize an MCP handler result string into a stable envelope."""
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        truncated = len(raw) > cls._MCP_RESULT_CHAR_CAP
        if truncated:
            raw = raw[: cls._MCP_RESULT_CHAR_CAP] + "… [truncated]"
        try:
            parsed: Any = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            envelope: Dict[str, Any] = {"ok": False, "error": parsed["error"]}
        elif isinstance(parsed, dict) and "result" in parsed:
            envelope = {"ok": True, "result": parsed["result"]}
            if "structuredContent" in parsed:
                envelope["structuredContent"] = parsed["structuredContent"]
        else:
            envelope = {"ok": True, "result": parsed if parsed is not None else raw}
        return {**envelope, "truncated": True} if truncated else envelope

    @staticmethod
    def _mcp_allowlist(plugin_id: str) -> List[str]:
        """Operator-granted MCP server allowlist; missing/unreadable -> [] (default-deny)."""
        return read_plugin_mcp_allowlist(plugin_id)

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Whether this plugin may override built-in tools: bundled plugins are trusted (a maintainer
        choice, not privilege escalation); others need ``tools.override`` via
        :func:`plugin_capability_granted` (granted_capabilities OR legacy ``allow_tool_override: true``).

        Bundled plugins (shipped with Hermes core) are trusted by default — an override there is a
        deliberate maintainer choice, not a third-party plugin trying to elevate privilege. For every other
        source, the canonical check is :func:`plugin_capability_granted` with the ``tools.override``
        capability — satisfied by EITHER the consent-flow grant
        (``plugins.entries.<plugin_id>.granted_capabilities``) OR the deprecated legacy key
        ``allow_tool_override: true`` (still honored for backward compatibility; #64228 reference
        migration).
        """
        if self.manifest.source == "bundled":
            return True
        try:
            cfg = load_plugin_config_for_home(self._manager.home_path)
        except Exception:
            return False  # fail closed: better to break the override than silently grant it
        # Pass THIS manager's profile-scoped config so a multi-profile process never consults the
        # active profile's consent state instead.
        return plugin_capability_granted(self.plugin_id, "tools.override", config=cfg)

    # Fail-closed by construction: any failure to read consent state inside plugin_capability_granted
    # returns False. The profile-scoped config is passed through so a multi-profile process consults THIS
    # manager's home, never the active profile's (#65593 constraint).
    def inject_message(
        self, content: str, role: str = "user", *, session_key: str | None = None,
    ) -> bool:
        """Inject a message into a CLI, Ink TUI/desktop, or messaging-gateway conversation.

        CLI uses the attached REPL queues. Ink TUI and desktop use a separate injector
        from the messaging gateway and queue onto the live session named by ``session_key``
        (the durable key, not the ephemeral UI session id). Non-CLI injection needs that
        ``session_key`` plus ``plugins.entries.<plugin_id>.allow_gateway_injection``.
        ``True`` means a host accepted the request, not that the turn completed.
        """
        cli = self._manager._cli_ref
        msg = content if role == "user" else f"[{role}] {content}"
        if cli is not None:
            queue_ = cli._interrupt_queue if getattr(cli, "_agent_running", False) else cli._pending_input
            queue_.put(msg)
            return True
        if not session_key:
            logger.warning("inject_message: gateway mode requires an existing session_key")
            return False
        if not self._gateway_injection_allowed():
            logger.warning("inject_message: gateway injection denied for plugin %s; set "
                           "plugins.entries.%s.allow_gateway_injection: true to allow it",
                           self.plugin_id, self.plugin_id)
            return False
        # TUI/desktop host is a different slot. It accepts only when it owns this
        # session_key; a miss falls through so a co-resident messaging gateway
        # still receives its own keys. An exception fails closed — do not also
        # hand the same text to the gateway.
        if self._manager.has_tui_message_injector:
            try:
                if self._manager.inject_tui_message(
                    session_key=session_key, content=msg, plugin_id=self.plugin_id,
                ):
                    return True
            except Exception:
                logger.warning("inject_message: TUI scheduling failed for plugin %s", self.plugin_id,
                               exc_info=True)
                return False
        if not self._manager.has_gateway_message_injector:
            logger.warning("inject_message: no live gateway is available")
            return False
        try:
            return bool(self._manager.inject_gateway_message(
                session_key=session_key, content=msg, plugin_id=self.plugin_id,
            ))
        except Exception:
            logger.warning("inject_message: gateway scheduling failed for plugin %s", self.plugin_id,
                           exc_info=True)
            return False

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns."""
        return plugin_gateway_injection_allowed(self.plugin_id)

    @_serialized_replacement
    def register_cli_command(
        self, name: str, help: str, setup_fn: Callable, handler_fn: Callable | None = None,
        description: str = "",
    ) -> PluginRegistration:
        """Register a CLI subcommand (``hermes <name> ...``). *setup_fn* receives the argparse
        subparser; *handler_fn* becomes ``set_defaults(func=...)``."""
        entry = {
            "name": name, "help": help, "description": description, "setup_fn": setup_fn,
            "handler_fn": handler_fn, "plugin": self.manifest.name, "plugin_key": self.plugin_id,
        }
        return self._register_entry("cli_command", name, self._manager._cli_commands, entry,
                                    "Plugin %s registered CLI command: %s", name)

    @_serialized_replacement
    def register_command(
        self, name: str, handler: Callable, description: str = "", args_hint: str = "",
        argument_mode: str | None = None,
    ) -> Optional[PluginRegistration]:
        """Register an in-session slash command (``/name``); handler ``fn(raw_args: str) -> str | None``
        (sync or async). ``args_hint`` (e.g. ``"<file>"``) lets adapters like Discord surface an argument
        field; without it the command registers parameterless there but still accepts trailing text."""
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning("Plugin '%s' tried to register a command with an empty name.", self.manifest.name)
            return
        with suppress(Exception):  # reject if it conflicts with a built-in command
            resolver = get_plugin_host_callback("command_resolver")
            if resolver is not None and resolver(clean) is not None:
                logger.warning("Plugin '%s' tried to register command '/%s' which conflicts "
                               "with a built-in command. Skipping.", self.manifest.name, clean)
                return
        hint = (args_hint or "").strip()
        entry = {
            "handler": handler, "description": description or "Plugin command",
            "plugin": self.manifest.name, "plugin_key": self.plugin_id, "args_hint": hint,
            "argument_mode": argument_mode if argument_mode in {"options", "text", "mixed"}
            else ("text" if hint else None),
        }
        return self._register_entry("command", clean, self._manager._plugin_commands, entry,
                                    "Plugin %s registered command: /%s", clean)

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry with the parent agent (when available)
        resolved automatically; returns the handler's JSON string. ``kwargs`` forward to dispatch."""
        from tools.registry import registry
        # In gateway mode _cli_ref is None — tools degrade gracefully (no spinner, TERMINAL_CWD).
        if "parent_agent" not in kwargs:
            agent = getattr(self._manager._cli_ref, "agent", None)
            if agent is not None:
                kwargs["parent_agent"] = agent
        return registry.dispatch(tool_name, args, scope=self._manager.scope_key, **kwargs)

    @_serialized_replacement
    def register_context_engine(self, engine) -> Optional[PluginRegistration]:
        """Register the (single) ``agent.context_engine.ContextEngine`` replacing the built-in
        ContextCompressor; a second registration is rejected with a warning."""
        if self._manager._context_engine is not None:
            logger.warning("Plugin '%s' tried to register a context engine, but one is "
                           "already registered. Only one context engine plugin is allowed.",
                           self.manifest.name)
            return
        from agent.context_engine import ContextEngine
        if self._wrong_type(engine, ContextEngine, "context engine"):
            return
        previous = self._manager._context_engine  # always None here; kept for the restore contract
        self._manager._context_engine = engine
        handle = self._track_replacement(
            "context_engine", engine.name, slot=("manager_value", id(self._manager), "_context_engine"),
            current=engine, previous=previous,
            restore=lambda replacement: self._manager._restore_value("_context_engine", engine, replacement),
        )
        logger.info("Plugin '%s' registered context engine: %s", self.manifest.name, engine.name)
        return handle

    def register_context_reference(self, provider) -> None:
        """Register a :class:`agent.context_references.ContextReferenceProvider`; ``provider.prefix``
        defines ``@<prefix>:``. Built-in prefixes (diff, staged, file, folder, git, url) are
        rejected."""
        from agent.context_references import (
            ContextReferenceProvider as _CRP, register_context_reference_provider as _register,
        )
        if self._wrong_type(provider, _CRP, "context reference provider"):
            return
        try:
            _register(provider)
        except ValueError as exc:
            logger.warning("Plugin '%s' context reference registration failed: %s", self.manifest.name, exc)
            return
        logger.info("Plugin '%s' registered context reference: @%s:", self.manifest.name, provider.prefix)

    def register_memory_provider(self, provider) -> None:
        """Record a memory provider (inert). Activation is owned by ``plugins/memory`` via
        ``memory.provider``; a provider reaching here was loaded by the general manager, and
        without this method its ``register()`` would fail on a missing attribute."""
        from agent.memory_provider import MemoryProvider
        if self._wrong_type(provider, MemoryProvider, "memory provider"):
            return
        self._memory_provider = provider
        logger.debug("Plugin '%s' registered memory provider: %s", self.manifest.name,
                     getattr(provider, "name", "?"))

    @_serialized_replacement
    def register_dashboard_auth_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a :class:`hermes_cli.dashboard_auth.DashboardAuthProvider` for the dashboard
        auth gate (non-loopback bind without ``--insecure``). Wrong type / duplicate name warn and
        are ignored, never raised."""
        provider_type = get_plugin_host_callback("dashboard_auth_provider_type")
        register_global_provider = get_plugin_host_callback("dashboard_auth_register")
        unregister_global_provider = get_plugin_host_callback("dashboard_auth_unregister")
        if provider_type is None or register_global_provider is None or unregister_global_provider is None:
            logger.warning("Plugin '%s' tried to register dashboard-auth provider but the host is unavailable",
                           self.manifest.name)
            return
        if self._wrong_type(provider, provider_type, "dashboard-auth provider"):
            return
        launch_scope = hermes_home_key(get_process_hermes_home())
        if self._manager.scope_key != launch_scope:
            logger.warning(
                "Plugin '%s' tried to register dashboard-auth provider %r "
                "from profile scope %s; ignoring it because dashboard auth "
                "is owned by launch scope %s.",
                self.manifest.name,
                provider.name,
                self._manager.scope_key,
                launch_scope,
            )
            return
        registry_name = provider.name
        # The auth registry is process-global (lifetime = web server). Disposing it on a routine
        # per-home manager teardown emptied it for the WHOLE process and disabled sign-in until
        # restart — so upsert and keep it out of reverse-order teardown (``persistent=True``).
        try:
            # A per-home manager is torn down routinely (profile-scoped dashboard activity, force
            # re-discovery), and disposing this registration on that teardown emptied the auth registry for
            # the WHOLE process, permanently disabling sign-in until restart (#91701). The handle still
            # disposes explicitly (identity- conditional), and a forced re-discovery rotates the provider in
            # place via the upsert.
            register_global_provider(provider)
        except (TypeError, ValueError) as e:
            logger.warning("Plugin '%s' failed to register dashboard-auth provider %r: %s",
                           self.manifest.name, getattr(provider, "name", "?"), e)
            return
        handle = self._track("dashboard_auth_provider", registry_name,
                             lambda: unregister_global_provider(registry_name, provider), persistent=True)
        logger.info("Plugin '%s' registered dashboard-auth provider: %s (%s)", self.manifest.name,
                    registry_name, provider.display_name)
        return handle

    @_serialized_replacement
    def register_platform(
        self, name: str, label: str, adapter_factory: Callable, check_fn: Callable,
        validate_config: Callable | None = None, required_env: list | None = None,
        install_hint: str = "", **entry_kwargs: Any,
    ) -> Optional[PluginRegistration]:
        """Register a gateway platform adapter (``adapter_factory(PlatformConfig) -> BasePlatformAdapter``).
        ``check_fn`` is a PASSIVE "deps importable?" probe that must never install (status displays call
        it freely); an ACTIVE installer goes in ``ensure_deps_fn`` (called from ``create_adapter()`` when
        ``check_fn`` is False). Extra kwargs (``setup_fn``, ``emoji``, ``allowed_users_env``,
        ``platform_hint``, ``ensure_deps_fn``) forward to ``PlatformEntry``; unknown keys raise TypeError."""
        from plugin_runtime.platform_registry import platform_registry, PlatformEntry
        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name, label=label, adapter_factory=adapter_factory, check_fn=check_fn,
            validate_config=validate_config, required_env=required_env or [],
            install_hint=install_hint, source="plugin", **entry_kwargs,
        )
        scope = self._manager.scope_key
        previous = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.register(entry, scope=scope)
        current = platform_registry.snapshot_registration(name, scope=scope)
        if current[0] is not entry or current[1] is not None:
            return None
        self._manager._plugin_platform_names.add(name)
        handle = self._manager._track_scoped_registration(
            self.manifest, "platform", name, platform_registry, current, previous,
            finalize=lambda: self._manager._remove_platform_name_if_unowned(name),
        )
        logger.debug("Plugin %s registered platform: %s", self.manifest.name, name)
        return handle

    def register_slack_action_handler(
        self, action_id: Any, callback: Callable,
    ) -> PluginRegistration:
        """Register a Slack Block Kit action handler, wired into ``slack_bolt.AsyncApp`` at connect.
        ``action_id`` is anything ``slack_bolt.App.action()`` accepts; ``callback`` is
        ``async def handler(ack, body, action)`` (``await ack()`` within 3s). Raises ``ValueError`` for
        a non-callable callback or empty ``action_id``."""
        if not callable(callback):
            raise self._refuse("a Slack action handler with a non-callable callback")
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise self._refuse("a Slack action handler with an empty action_id")
        entry = (action_id, callback, self.manifest.name)
        handlers = self._manager._slack_action_handlers
        handlers.append(entry)
        handle = self._track("slack_action_handler", repr(action_id),
                             lambda: self._manager._remove_identity(handlers, entry))
        logger.debug("Plugin %s registered Slack action handler: %s", self.manifest.name, action_id)
        return handle

    def register_platform_handler(self, platform: str, factory: Callable) -> None:
        """Register ``factory(native, adapter)``, invoked at ``connect()`` before/as the core handlers
        register (``adapter`` read-only). ``native``: telegram PTB ``Application``, discord
        ``commands.Bot``, slack ``AsyncApp``, matrix client, teams ``App``, dingtalk
        ``DingTalkStreamClient``, line aiohttp ``web.Application``, others ``None``. Keep SDK imports
        inside the factory; exceptions are logged and the platform still connects. Scope handlers in
        first-match dispatch tables so core flows keep working. Raises ``ValueError`` when not callable
        or platform is empty."""
        if not callable(factory):
            raise self._refuse("a platform handler factory with a non-callable factory")
        key = (platform or "").strip().lower()
        if not key:
            raise self._refuse("a platform handler factory with an empty platform name")
        self._manager._platform_handler_factories.setdefault(key, []).append((factory, self.manifest.name))
        logger.debug("Plugin %s registered %s handler factory: %s", self.manifest.name, key,
                     getattr(factory, "__name__", repr(factory)))

    def register_telegram_handler(self, factory: Callable) -> None:
        """``register_platform_handler("telegram", factory)``. PTB dispatches only the FIRST matching
        handler per group and core registers a catch-all ``CallbackQueryHandler`` — always scope with
        ``pattern=`` or you swallow the core button flows."""
        self.register_platform_handler("telegram", factory)

    @_serialized_replacement
    def register_auxiliary_task(
        self, key: str, *, display_name: str, description: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> PluginRegistration:
        """Register an auxiliary LLM task with its own ``auxiliary.<key>`` config block (picker entry,
        ``AUXILIARY_<KEY>_*`` env bridge, defaults merged into loaded configs). ``defaults`` may
        override provider/model/base_url/api_key/timeout/extra_body (unknown keys kept verbatim).
        Raises ``ValueError`` for an empty/invalid key, a built-in key, or another plugin's key."""
        me = self.manifest.name
        if not key or not isinstance(key, str):
            raise ValueError(f"Plugin '{me}' tried to register auxiliary task with invalid key {key!r}")
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(f"Plugin '{me}' auxiliary task key {key!r} "
                             f"must contain only alphanumeric characters and underscores")
        builtin_keys = get_plugin_host_callback("builtin_auxiliary_task_keys")
        reserved = set(builtin_keys()) if builtin_keys is not None else set()
        if key in reserved:
            raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — that key is reserved "
                             f"for a built-in task. Pick a plugin-namespaced key (e.g. '{me}_{key}').")
        # Owner is the canonical id ``ctx.llm`` is bound to, so agent/plugin_llm.py can match it.
        owner_id = self.plugin_id
        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != owner_id:
            raise ValueError(f"Plugin '{me}' cannot register auxiliary task {key!r} — already registered "
                             f"by plugin '{existing.get('plugin')}'")
        # Plugin owns the schema; routing fields are guaranteed present so consumers don't crash.
        entry = {
            "key": key, "display_name": display_name, "description": description,
            "defaults": {"provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": 60,
                         "extra_body": {}, **(defaults or {})},
            "plugin": owner_id, "plugin_key": owner_id,
        }
        return self._register_entry("auxiliary_task", key, self._manager._aux_tasks, entry,
                                    "Plugin %s registered auxiliary task: %s (%s)", key, display_name,
                                    previous=existing)

    def register_redaction_patterns(self, patterns) -> int:
        """Additively register secret-token regexes with :mod:`agent.redact`; returns the count accepted.
        Plugins can over-redact, never weaken built-ins; ``security.redact_secrets: false`` applies
        equally. Each pattern must compile and start with >= 2 literal characters; invalid entries warn
        and are skipped."""
        from agent.redact import register_redaction_patterns as _register
        try:
            count = _register(patterns, source=f"plugin:{self.manifest.name}")
        except Exception as exc:
            logger.warning("Plugin '%s' redaction pattern registration failed: %s", self.manifest.name, exc)
            return 0
        logger.debug("Plugin %s registered %d redaction pattern(s)", self.manifest.name, count)
        return count

    def register_hook(self, hook_name: str, callback: Callable) -> PluginRegistration:
        """Register a lifecycle hook callback (unknown names warn but are still stored)."""
        return self._track_callback("hook", hook_name, callback, self._manager._hooks, VALID_HOOKS)

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """Register behavior-changing middleware (request kinds rewrite the payload, execution kinds
        wrap the callback). Unknown kinds warn but are stored."""
        return self._track_callback(
            "middleware", kind, callback, self._manager._middleware, VALID_MIDDLEWARE
        )

    def _track_callback(
        self, kind: str, key: str, callback: Callable, mapping: Dict[str, List[Callable]],
        valid: Set[str],
    ) -> PluginRegistration:
        """Append ``callback`` under ``key`` (warning on unknown ``key``) and lease its removal."""
        if key not in valid:
            logger.warning("Plugin '%s' registered unknown %s '%s' (valid: %s)", self.manifest.name, kind,
                           key, ", ".join(sorted(valid)))
        mapping.setdefault(key, []).append(callback)
        handle = self._track(kind, key, lambda: self._manager._remove_callback(mapping, key, callback))
        logger.debug("Plugin %s registered %s: %s", self.manifest.name, kind, key)
        return handle

    def register_system_prompt_section(
        self, id: str, content: Union[str, Callable[[Mapping[str, Any]], str]], *,
        position: str = "after_memory", max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context frozen into each new session prompt. Callables receive a
        read-only session-info mapping; the rendered prompt is persisted by core verbatim."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError("system prompt section id must be 1-128 lowercase characters "
                             "using letters, numbers, '.', '_', or '-'")
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError("system prompt section position must be one of: "
                             + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS)))
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS):
            raise ValueError(f"system prompt section max_chars must be between 1 and {MAX_SYSTEM_PROMPT_SECTION_CHARS}")
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(f"system prompt section {id!r} is already registered by plugin {existing.plugin!r}")
        section = PluginSystemPromptSection(
            id=id, content=content, position=position, max_chars=max_chars, plugin=self.plugin_id,
        )
        return self._register_entry("system_prompt_section", id, self._manager._system_prompt_sections,
                                    section, "Plugin %s registered system prompt section: %s", id,
                                    previous=existing)

    def emit(self, event: str, payload: Optional[dict] = None) -> int:
        """Publish bare *event* as ``<plugin_key>:<event>`` (namespace FORCED to this plugin); return
        the subscriber count scheduled. Any ``':'`` in the name (``hermes:x`` is reserved for core,
        foreign namespaces forbidden) raises ``ValueError``. Delivery is fire-and-forget via a
        single-worker queue: order preserved, a blocking subscriber cannot stall the emitter."""
        plugin_key = self.plugin_id
        if not event or not isinstance(event, str):
            logger.warning("Plugin '%s' tried to emit an invalid event name %r", plugin_key, event)
            raise ValueError(f"Plugin '{plugin_key}' emit() requires a non-empty event name")
        if ":" in event:
            logger.warning("Plugin '%s' tried to emit namespaced/reserved event '%s' — a plugin may only emit "
                           "bare event names under its own '%s:' namespace (the '%s:' prefix is reserved "
                           "for core, and foreign namespaces are forbidden)",
                           plugin_key, event, plugin_key, HERMES_EVENT_NAMESPACE)
            raise ValueError(f"Plugin '{plugin_key}' may not emit '{event}': emit only the bare event name; "
                             f"the namespace is forced to '{plugin_key}:' and the '{HERMES_EVENT_NAMESPACE}:' "
                             f"prefix is reserved for core")
        if payload is not None and not isinstance(payload, dict):
            raise TypeError(f"Plugin '{plugin_key}' emit() payload must be a dict or None")
        return self._manager._dispatch_event(f"{plugin_key}:{event}", payload or {})

    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to a fully-qualified ``<plugin_key>:<event>`` name (unrestricted — only
        emitting is namespace-gated). Owner-tagged so unload removes zombie callbacks."""
        if not event or not isinstance(event, str):
            raise ValueError(f"Plugin '{self.manifest.name}' subscribe() requires a non-empty event name")
        self._manager._subscribe_event(self.plugin_id, event, callback)
        logger.debug("Plugin %s subscribed to event: %s", self.manifest.name, event)

    @_serialized_replacement
    def register_skill(
        self, name: str, path: Path, description: str = "",
        frontmatter: Optional[Mapping[str, Any]] = None,
    ) -> PluginRegistration:
        """Register a read-only skill resolvable as ``'<plugin_name>:<name>'`` via ``skill_view()``
        and listed by ``skills_list``. Not copied into ``~/.hermes/skills/`` and not in the system
        prompt's ``<available_skills>``. Raises ``ValueError`` (``':'``/invalid chars) or
        ``FileNotFoundError``."""
        from agent.skill_utils import _NAMESPACE_RE
        if ":" in name:
            raise ValueError(f"Skill name '{name}' must not contain ':' (the namespace is derived from the "
                             f"plugin name '{self.manifest.name}' automatically).")
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+.")
        # Plugin register() helpers commonly pass the SKILL.md location as str
        # (PluginManifest.path is stored as str); the registry and find_plugin_skill()
        # promise a Path downstream.
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")
        namespace = self.manifest.skill_namespace or self.manifest.name
        qualified = f"{namespace}:{name}"
        if self.manifest.portable and qualified in self._manager._plugin_skills:
            raise ValueError(f"Plugin skill '{qualified}' is already registered")
        entry = {
            "path": path, "plugin": namespace, "plugin_key": self.plugin_id, "bare_name": name,
            "description": description, "frontmatter": dict(frontmatter or {}),
        }
        return self._register_entry("skill", qualified, self._manager._plugin_skills, entry,
                                    "Plugin %s registered skill: %s", qualified)


# -- scoped provider registrars ------------------------------------------------------------------
# Every ``register_<category>_provider`` shares one body (:meth:`PluginContext._register_scoped_provider`):
# type-check, register in the scope-keyed process-global registry, lease the slot so unload restores
# the displaced entry. Rows: (method, kind, registry module, base-class module:attr, label, docstring,
# options). ``normalize``: ``strip`` (default), ``lower`` (strip+lowercase) or ``None`` (raw name).
_SCOPED_PROVIDER_REGISTRARS: Tuple[Tuple[str, str, str, str, str, str, Dict[str, Any]], ...] = (
    ("register_image_gen_provider", "image_gen_provider", "agent.image_gen_registry",
     "agent.image_gen_provider:ImageGenProvider", "image_gen provider",
     "Register an :class:`agent.image_gen_provider.ImageGenProvider`; "
     "``provider.name`` is matched by ``image_gen.provider``.", {"article": "an"}),
    ("register_video_gen_provider", "video_gen_provider", "agent.video_gen_registry",
     "agent.video_gen_provider:VideoGenProvider", "video_gen provider",
     "Register an :class:`agent.video_gen_provider.VideoGenProvider`; "
     "``provider.name`` is matched by ``video_gen.provider``.", {}),
    ("register_web_search_provider", "web_search_provider", "agent.web_search_registry",
     "agent.web_search_provider:WebSearchProvider", "web provider",
     "Register an :class:`agent.web_search_provider.WebSearchProvider`; "
     "``provider.name`` is matched by ``web.search_backend`` / ``web.extract_backend`` / ``web.backend``.",
     {}),
    ("register_browser_provider", "browser_provider", "agent.browser_registry",
     "agent.browser_provider:BrowserProvider", "browser provider",
     "Register an :class:`agent.browser_provider.BrowserProvider`; "
     "``provider.name`` is matched by ``browser.cloud_provider`` (consulted by "
     "``tools.browser_tool_cloud._get_cloud_provider``).", {}),
    ("register_terminal_environment_provider", "terminal_environment_provider",
     "agent.terminal_env_registry", "agent.terminal_env_provider:TerminalEnvironmentProvider",
     "terminal environment provider",
     "Register a :class:`agent.terminal_env_provider.TerminalEnvironmentProvider`; ``provider.name`` "
     "is matched by ``terminal.backend`` when no built-in backend has that name. Built-in names (local, "
     "docker, singularity, modal, daytona, vercel_sandbox, ssh) are rejected — plugins never shadow "
     "in-tree backends.",
     {"normalize": "lower", "reject_message": "Plugin '%s' terminal environment provider rejected: %s"}),
    ("register_secret_source", "secret_source", "agent.secret_sources.registry",
     "agent.secret_sources.base:SecretSource", "secret source",
     "Register a :class:`agent.secret_sources.base.SecretSource`, run by ``load_hermes_dotenv()`` "
     "(after ``~/.hermes/.env``, before credentials are read) when ``secrets.<name>`` is enabled. The "
     "orchestrator owns ordering/precedence/provenance; the source only fetches. Since dotenv usually "
     "loads before discovery, the manager re-pulls enabled plugin sources afterwards.",
     {"normalize": None, "register": "register_source", "param": "source"}),
    ("register_tts_provider", "tts_provider", "agent.tts_registry",
     "agent.tts_provider:TTSProvider", "TTS provider",
     "Register an :class:`agent.tts_provider.TTSProvider`; ``provider.name`` is matched by "
     "``tts.provider`` unless it is a built-in name (rejected with a warning) or a "
     "``tts.providers.<name>: type: command`` entry shares it (command-providers win).",
     {"normalize": "lower"}),
    ("register_transcription_provider", "transcription_provider", "agent.transcription_registry",
     "agent.transcription_provider:TranscriptionProvider", "transcription provider",
     "Register an :class:`agent.transcription_provider.TranscriptionProvider`; ``provider.name`` is "
     "matched by ``stt.provider`` unless it is a built-in name (rejected) or a ``stt.providers.<name>: "
     "type: command`` entry shares it (command-providers win).", {"normalize": "lower"}),
)

_NAME_NORMALIZERS: Dict[Optional[str], Optional[Callable[[str], str]]] = {
    "strip": lambda n: n.strip(), "lower": lambda n: n.strip().lower(), None: None,
}


def _make_scoped_provider_registrar(method_name, kind, registry_mod, base_ref, label, doc, options):
    """Build one ``register_<category>_provider`` method from a ``_SCOPED_PROVIDER_REGISTRARS`` row."""
    base_mod, base_attr = base_ref.split(":")
    normalize_fn = _NAME_NORMALIZERS[options.get("normalize", "strip")]
    register_name = options.get("register")

    def register(self, provider) -> Optional[PluginRegistration]:
        registry = importlib.import_module(registry_mod)
        return self._register_scoped_provider(
            provider, kind=kind, base_class=getattr(importlib.import_module(base_mod), base_attr),
            registry=registry, label=label, article=options.get("article", "a"), normalize=normalize_fn,
            register=getattr(registry, register_name) if register_name else None,
            reject_message=options.get("reject_message"),
        )

    def register_source(self, source) -> Optional[PluginRegistration]:  # secret sources: ``source``
        return register(self, source)

    method = register_source if options.get("param") == "source" else register
    method.__name__, method.__qualname__, method.__doc__ = method_name, f"PluginContext.{method_name}", doc
    return _serialized_replacement(method)


for _row in _SCOPED_PROVIDER_REGISTRARS:
    setattr(PluginContext, _row[0], _make_scoped_provider_registrar(*_row))
del _row


def _ignore_after_abandoned_load(method):
    """Turn a registrar into a no-op once the context's load timed out: the abandoned worker thread may
    still be executing register(), and a late registration would land in registries that the failure
    path already swept (#108139)."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_load_abandoned", False):
            logger.warning(
                "Plugin '%s' called %s() after its load timed out; ignored", self.manifest.name,
                method.__name__,
            )
            return None
        return method(self, *args, **kwargs)

    return wrapped


# Every mutating entry point plugins reach through ``ctx`` during register(); applied by name so the
# guard cannot drift from the surface as registrars are added.
for _name, _method in list(vars(PluginContext).items()):
    if callable(_method) and (_name.startswith("register_") or _name in {"subscribe", "on_unload"}):
        setattr(PluginContext, _name, _ignore_after_abandoned_load(_method))
del _name, _method
