"""``PluginContext``: the ``ctx`` every plugin's ``register(ctx)`` receives.

Import it from :mod:`hermes_cli.plugins` (the public path, re-exported there). Every name the method
bodies read that ``hermes_cli.plugins`` also binds (functions, classes, constants, ``logger``,
``replacement_coordinator``, stdlib modules) is read from that facade at call time (``_facade()``), so
a patch on ``hermes_cli.plugins.<name>`` reaches ctx methods as it did when the class lived there.
"""

from __future__ import annotations

import asyncio
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

from hermes_cli.plugins_capability_ctx import PluginCapabilityCtxMixin
from hermes_cli.plugins_dispatch import DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS
from hermes_cli.plugins_ledger import PluginRegistration
from hermes_cli.plugins_loader import _serialized_replacement
from hermes_cli.plugins_manifest import PluginManifest
from hermes_cli.plugins_registrars import register_scoped_provider, scoped_provider_registrar
from hermes_cli.plugins_state import PluginState

if TYPE_CHECKING:
    from hermes_cli.plugins import PluginManager

_UNSET = object()


def _facade():
    """``hermes_cli.plugins``, late-imported (it imports this module)."""
    import hermes_cli.plugins as facade
    return facade


class PluginContext(PluginCapabilityCtxMixin):
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: PluginManager):
        self.manifest = manifest
        self._manager = manager
        self._llm: Any = None  # lazy; tests preseed it (see ``llm``)
        # Set when this context's load overran ``plugins.load_timeout_seconds``: the abandoned worker may
        # still be running register(), and nothing it registers from then on may reach a registry.
        self._load_abandoned = False
        self._staged: list[PluginRegistration] = []  # this load's staged registrations, published if it succeeds

    def _abandon_load(self) -> None:
        """Mark this load as timed out; every later ``register_*``/``subscribe``/``on_unload`` is ignored."""
        self._load_abandoned = True

    @property
    def plugin_id(self) -> str:
        """Return the effective registry id used for this plugin's namespaces."""
        return _facade().manifest_key(self.manifest)

    def has_plugin(self, plugin_id: str) -> bool:
        """Return True when another plugin is loaded and enabled (runtime probe for advisory
        ``requires_plugins``, #64165). Matches on registry key or manifest name."""
        return any(
            loaded.enabled and (key == plugin_id or loaded.manifest.name == plugin_id)
            for key, loaded in self._manager._plugins.items()
        )

    def _segments(self, key: str) -> tuple[str, ...]:
        """Validated plugin-relative settings path (warn + re-raise on rejection)."""
        try:
            return _facade()._plugin_relative_segments(key)
        except ValueError:
            _facade().logger.warning("Rejected config path %r from plugin %s", key, self.plugin_id)
            raise

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read plugin-relative ``plugins.entries.<plugin_id>.settings.<key>`` (falls back to the
        legacy ``config`` subtree for migration compatibility)."""
        segments = self._segments(key)
        entry = _facade()._plugin_settings_entry(_facade().load_config_readonly() or {}, self.plugin_id)
        if entry is None:
            return default
        value = _facade()._nested_plugin_value(entry.get("settings"), segments, _facade()._UNSET)
        if value is not _facade()._UNSET:
            return value
        return _facade()._nested_plugin_value(entry.get("config"), segments, default)

    def set_config(self, key: str, value: Any) -> None:
        """Atomically write one value in this plugin's ``settings`` subtree."""
        _facade().save_plugin_setting(self.plugin_id, self._segments(key), value)

    @cached_property
    def state(self) -> PluginState:
        """This plugin's profile-scoped durable JSON state facade."""
        return _facade().PluginState(self.plugin_id, self.manifest.data_namespace)

    @cached_property
    def platform_actions(self):
        """Capability-gated platform action facade (``add_reaction``, ``set_thread_title``). Every call
        re-checks ``gateway.platform_actions`` (legacy ``plugins.entries.<id>.allow_platform_actions``,
        default OFF) and returns ``{"ok": bool, ...}`` — verbs never raise into hook dispatch; no adapter
        handles or raw SDK objects."""
        from hermes_cli.platform_actions import PlatformActions
        return PlatformActions(self.plugin_id)

    def _wrong_type(self, obj: Any, base_class: type, label: str, article: str = "a") -> bool:
        """Warn-and-ignore gate shared by every registrar that requires a base class."""
        if isinstance(obj, base_class):
            return False
        _facade().logger.warning("Plugin '%s' tried to register %s %s that does not inherit from %s. Ignoring.",
                                 self.manifest.name, article, label, base_class.__name__)
        return True

    def _refuse(self, what: str) -> ValueError:
        """``ValueError`` for a malformed registration (``what`` completes "tried to register ...")."""
        return ValueError(f"Plugin '{self.manifest.name}' tried to register {what}.")

    def _track(
        self, kind: str, key: str, release: Callable[[], None], *, persistent: bool = False,
    ) -> PluginRegistration:
        """Record host-owned cleanup for a registration (``persistent``: see ``_track_registration``)."""
        return self._manager._track_registration(self.manifest, kind, key, release, persistent=persistent)

    def _track_replacement(
        self, kind: str, key: str, *, slot: tuple, current: Any, previous: Any,
        restore: Callable[[Any], bool],
    ) -> PluginRegistration:
        """Track one generation in a replaceable manager-local registration slot."""
        lease = _facade().replacement_coordinator.acquire(slot, current=current, previous=previous, restore=restore)
        return self._track(kind, key, lease.dispose)

    def _track_mapping_entry(
        self, kind: str, key: str, mapping: dict[str, Any], entry: Any, previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Store ``entry`` under ``key`` in a manager-local mapping and lease the slot; unload restores
        ``previous`` (default: the displaced entry, or removes the key) only while ``entry`` is still
        current."""
        if previous is _facade()._UNSET:
            previous = mapping.get(key)
        mapping[key] = entry
        return self._track_replacement(
            kind, key, slot=("manager_mapping", id(mapping), key), current=entry, previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(mapping, key, entry, replacement),
        )

    def _register_entry(
        self, kind: str, key: str, mapping: dict[str, Any], entry: Any, log_fmt: str, *log_args: Any,
        previous: Any = _UNSET,
    ) -> PluginRegistration:
        """Store + lease a manager-mapping entry, then debug-log ``log_fmt % (plugin name, *log_args)``."""
        handle = self._track_mapping_entry(kind, key, mapping, entry, previous)
        _facade().logger.debug(log_fmt, self.manifest.name, *log_args)
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
            _facade().logger.warning(reject_message, self.manifest.name, exc)
            return None
        if (register is not None and not accepted) or registry.snapshot_registration(
            registry_name, scope=scope
        ) is not provider:
            return None
        handle = self._manager._track_scoped_registration(
            self.manifest, kind, registry_name, registry, provider, previous
        )
        _facade().logger.info("Plugin '%s' registered %s: %s", self.manifest.name, label, registry_name)
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

    @cached_property
    def delegation_backends(self) -> Any:
        """``agent.delegation_backends.DelegationBackends``: list delegation backends and save one, optionally
        rebuilding the calling session's tools now (``/delegation``); a host remote facade on the owner process."""
        from agent.delegation_backends import DelegationBackends
        return DelegationBackends()

    @property
    def profile_name(self) -> str:
        """Active profile name (``"default"``, the ``~/.hermes/profiles/<name>`` id, or ``"custom"``),
        derived from ``HERMES_HOME`` — not ``_cli_ref``, which is None outside the interactive CLI —
        so gateway and kanban workers get it too."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    def current_cron_execution(self) -> Any:
        """The cron run executing now (``hermes_agent.cron.execution_identity.CronExecution``), else ``None``."""
        from hermes_agent.cron.execution_identity import current_cron_execution
        return current_cron_execution()

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register a cleanup callback for unload: runs in reverse acquisition order interleaved
        with registration teardown; exceptions are logged, never propagated."""
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        _facade().logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro, *, name: Optional[str] = None) -> asyncio.Task:
        """Spawn a supervised asyncio task; unload/force reload cancels it. Needs a running loop."""
        if not _facade().asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        loop = _facade().asyncio.get_running_loop()
        task_name = name or f"plugin:{self.plugin_id}:task"
        task = loop.create_task(coro, name=task_name)
        handle = self._track("background_task", task_name, lambda: task.done() or task.cancel())
        task.add_done_callback(lambda _t: handle.dispose())
        _facade().logger.debug("Plugin %s spawned supervised task: %s", self.manifest.name, task_name)
        return task

    def register_approval_transport(self, name: str, present_fn: Callable) -> None:
        """Register a human approval transport, inactive until ``security.approval.transport:
        <name>`` selects it. It receives a redacted ``ApprovalRequest`` and returns only a
        correlated decision; policy and persistence stay host-owned. ``present_fn`` may be async."""
        from hermes_cli.approval_transport import RegisteredApprovalTransport
        transports = self._manager._approval_transports
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not _facade().re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in transports:
            owner = transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        entry = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=self.plugin_id,
            profile_home=str(_facade().get_hermes_home().resolve()),
        )
        _facade().logger.info("Plugin %s registered approval transport: %s", self.plugin_id, clean)
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
        any enabled plugin could silently replace a privileged built-in like ``write_file`` (#23194)."""
        if override and not self._tool_override_allowed(name):
            raise _facade().PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool {name!r}. Set "
                f"plugins.entries.{self.plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )
        from tools.registry import registry
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(name, scope=scope)
        if previous is None and not override and registry.get_entry(name, scope=scope) is not None:
            _facade().logger.warning("Plugin %s tried to shadow global tool %s without override=True",
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
        _facade().logger.debug("Plugin %s registered tool: %s%s", self.manifest.name, name,
                               " (override)" if override else "")
        return handle

    def call_mcp(
        self, server: str, tool: str, arguments: Optional[dict[str, Any]] = None,
        timeout: float = 30,
    ) -> dict[str, Any]:
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
        _facade().logger.debug("Plugin %s called MCP %s/%s (timeout=%ss, %d chars returned)",
                               self.manifest.name, server, tool, timeout, len(raw or ""))
        return self._mcp_envelope(raw)

    _MCP_RESULT_CHAR_CAP = 65536

    @classmethod
    def _mcp_envelope(cls, raw: Any) -> dict[str, Any]:
        """Normalize an MCP handler result string into a stable envelope."""
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        truncated = len(raw) > cls._MCP_RESULT_CHAR_CAP
        if truncated:
            raw = raw[: cls._MCP_RESULT_CHAR_CAP] + "… [truncated]"
        try:
            parsed: Any = _facade().json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            envelope: dict[str, Any] = {"ok": False, "error": parsed["error"]}
        elif isinstance(parsed, dict) and "result" in parsed:
            envelope = {"ok": True, "result": parsed["result"]}
            if "structuredContent" in parsed:
                envelope["structuredContent"] = parsed["structuredContent"]
        else:
            envelope = {"ok": True, "result": parsed if parsed is not None else raw}
        return {**envelope, "truncated": True} if truncated else envelope

    @staticmethod
    def _mcp_allowlist(plugin_id: str) -> list[str]:
        """Operator-granted MCP server allowlist; missing/unreadable -> [] (default-deny)."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            return []
        allowlist = (_facade()._plugin_settings_entry(cfg, plugin_id) or {}).get("mcp_allowlist")
        return [str(item) for item in allowlist] if isinstance(allowlist, list) else []

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Whether this plugin may override built-in tools: bundled plugins are trusted (a maintainer
        choice, not privilege escalation); others need ``tools.override`` via
        :func:`plugin_capability_granted` (granted_capabilities OR legacy ``allow_tool_override: true``,
        still honored for backward compatibility; #64228)."""
        if self.manifest.source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config
            with _facade()._plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:
            return False  # fail closed: better to break the override than silently grant it
        # Pass THIS manager's profile-scoped config so a multi-profile process never consults the
        # active profile's consent state instead.
        return _facade().plugin_capability_granted(self.plugin_id, "tools.override", config=cfg)

    # Fail-closed by construction: any failure to read consent state inside plugin_capability_granted
    # returns False. The profile-scoped config is passed through so a multi-profile process consults THIS
    # manager's home, never the active profile's (#65593 constraint).
    def inject_message(
        self, content: str, role: str = "user", *, session_key: str | None = None,
        origin: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue *content* for a session (``session_key``) or a chat in this plugin's own profile
        (``origin``); ``True`` means a host accepted it. Contract: plugins_injection.py, plugin docs."""
        from hermes_cli.plugins_injection import inject_plugin_message
        return inject_plugin_message(self, content, role, session_key=session_key, origin=origin)

    def session_cwd(self) -> str:
        """The calling session's recorded cwd, never the gateway's own (``plugin_session``)."""
        from hermes_cli.plugin_session import context_cwd
        return context_cwd(self._manager)

    def session_surface(self) -> str:
        """The calling session's client (``cli``/``tui``/``desktop``/``acp``) or messaging platform."""
        from hermes_cli.plugin_session import context_surface
        return context_surface(self._manager)

    def post_message(self, text: str) -> Any:
        """Post *text* into the session bound to this call; the handle's ``edit(text)`` updates it in
        place. Never raises (``ok=False`` + ``error`` with no conversation): ``plugin_messages.py``."""
        from hermes_cli.plugin_messages import post
        return post(self, text)

    async def apost_message(self, text: str) -> Any:
        """``post_message`` for async code on the gateway loop (edit with ``await handle.aedit``)."""
        from hermes_cli.plugin_messages import apost
        return await apost(self, text)

    def ask_choice(self, question: str, choices: list) -> Optional[str]:
        """From a sync slash-command handler: ask its user to pick one of *choices* (``str`` or
        ``{value, label?, description?}``); the value, or None if unanswerable: ``plugin_choices.py``."""
        from hermes_cli.plugin_choices import ask_choice
        return ask_choice(question, choices)

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns (read from the plugin's
        own profile config, never the calling thread's ambient home)."""
        try:
            with _facade()._plugin_home_scope(self._manager.home_path):
                cfg = _facade().load_config_readonly() or {}
        except Exception:
            return False
        return (_facade()._plugin_settings_entry(cfg, self.plugin_id) or {}).get("allow_gateway_injection") is True

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
        (sync or async), plus the session-context keywords it declares (``plugin_command_dispatch``). ``args_hint`` (e.g. ``"<file>"``) lets adapters like Discord surface an argument
        field; without it the command registers parameterless there but still accepts trailing text."""
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            _facade().logger.warning("Plugin '%s' tried to register a command with an empty name.", self.manifest.name)
            return
        with _facade().suppress(Exception):  # reject if it conflicts with a built-in command
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                _facade().logger.warning("Plugin '%s' tried to register command '/%s' which conflicts "
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
            _facade().logger.warning("Plugin '%s' tried to register a context engine, but one is "
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
        _facade().logger.info("Plugin '%s' registered context engine: %s", self.manifest.name, engine.name)
        return handle

    def register_context_reference(self, provider) -> None:
        """Register a :class:`agent.context_references.ContextReferenceProvider`; ``provider.prefix`` defines
        ``@<prefix>:``. Built-in prefixes (diff, staged, file, folder, git, url) are rejected."""
        from agent.context_references import (
            ContextReferenceProvider as _CRP, register_context_reference_provider as _register,
        )
        if self._wrong_type(provider, _CRP, "context reference provider"):
            return
        try:
            self._staged.append(self._manager._track_context_reference(self.manifest, provider, _register))
        except ValueError as exc:
            _facade().logger.warning("Plugin '%s' context reference registration failed: %s", self.manifest.name, exc)
            return
        _facade().logger.info("Plugin '%s' registered context reference: @%s:", self.manifest.name, provider.prefix)

    def register_memory_provider(self, provider) -> None:
        """Record a memory provider (inert). Activation is owned by ``plugins/memory`` via
        ``memory.provider``; a provider reaching here was loaded by the general manager, and
        without this method its ``register()`` would fail on a missing attribute."""
        from agent.memory_provider import MemoryProvider
        if self._wrong_type(provider, MemoryProvider, "memory provider"):
            return
        self._memory_provider = provider
        _facade().logger.debug("Plugin '%s' registered memory provider: %s", self.manifest.name,
                               getattr(provider, "name", "?"))

    @_serialized_replacement
    def register_dashboard_auth_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a :class:`hermes_cli.dashboard_auth.DashboardAuthProvider` for the dashboard
        auth gate (non-loopback bind without ``--insecure``). Wrong type / duplicate name warn and
        are ignored, never raised."""
        from hermes_cli.dashboard_auth import DashboardAuthProvider
        from hermes_cli.dashboard_auth.registry import register_global_provider, unregister_global_provider
        if self._wrong_type(provider, DashboardAuthProvider, "dashboard-auth provider"):
            return
        launch_scope = _facade().hermes_home_key(_facade().get_process_hermes_home())
        if self._manager.scope_key != launch_scope:
            _facade().logger.warning(
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
            _facade().logger.warning("Plugin '%s' failed to register dashboard-auth provider %r: %s",
                                     self.manifest.name, getattr(provider, "name", "?"), e)
            return
        handle = self._track("dashboard_auth_provider", registry_name,
                             lambda: unregister_global_provider(registry_name, provider), persistent=True)
        _facade().logger.info("Plugin '%s' registered dashboard-auth provider: %s (%s)", self.manifest.name,
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
        from hermes_agent.gateway.platform_registry import core_ships_platform, platform_registry, PlatformEntry
        if entry_kwargs.get("trusted_inbound") and self.manifest.source != "bundled" and core_ships_platform(name):
            raise self._refuse(f"core platform '{name}' with trusted_inbound (it would waive allowlists and pairing)")
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
        _facade().logger.debug("Plugin %s registered platform: %s", self.manifest.name, name)
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
        _facade().logger.debug("Plugin %s registered Slack action handler: %s", self.manifest.name, action_id)
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
        _facade().logger.debug("Plugin %s registered %s handler factory: %s", self.manifest.name, key,
                               getattr(factory, "__name__", repr(factory)))

    def register_telegram_handler(self, factory: Callable) -> None:
        """``register_platform_handler("telegram", factory)``. PTB dispatches only the FIRST matching
        handler per group and core registers a catch-all ``CallbackQueryHandler`` — always scope with
        ``pattern=`` or you swallow the core button flows."""
        self.register_platform_handler("telegram", factory)

    @_serialized_replacement
    def register_auxiliary_task(
        self, key: str, *, display_name: str, description: str,
        defaults: Optional[dict[str, Any]] = None,
        inherit_from: Optional[str] = None,
    ) -> PluginRegistration:
        """Register an auxiliary LLM task with its own ``auxiliary.<key>`` config block (picker entry,
        ``AUXILIARY_<KEY>_*`` env bridge, defaults merged into loaded configs). ``defaults`` may
        override provider/model/base_url/api_key/timeout/extra_body (unknown keys kept verbatim).
        ``inherit_from`` names a built-in or already-registered auxiliary task whose effective
        configuration becomes the base for this one; it is resolved at read time, so the task tracks
        the base's current config instead of snapshotting it (precedence: inherited base, then
        ``defaults``, then user config in ``auxiliary.<key>``). An unknown or self-referential
        ``inherit_from`` logs a warning and registers the task without inheritance.
        Raises ``ValueError`` for an empty/invalid key, a built-in key, or another plugin's key."""
        from hermes_cli.plugins_aux_tasks import build_auxiliary_task_entry
        existing = self._manager._aux_tasks.get(key)
        entry = build_auxiliary_task_entry(
            self.manifest.name, self.plugin_id, key, display_name=display_name, description=description,
            defaults=defaults, inherit_from=inherit_from, registered=self._manager._aux_tasks)
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
            _facade().logger.warning("Plugin '%s' redaction pattern registration failed: %s", self.manifest.name, exc)
            return 0
        _facade().logger.debug("Plugin %s registered %d redaction pattern(s)", self.manifest.name, count)
        return count

    def register_locale(
        self, lang: str, source: str | Path | Mapping[str, Any], *, endonym: Optional[str] = None,
        rtl: bool = False, surface: str = "core",
    ) -> PluginRegistration:
        """Register a language-pack layer for ``lang`` (``pl``, ``pt-br``): ``source`` is a YAML file path or
        a nested/flat mapping of ``dotted.key: text``. ``surface`` is ``core`` (Python ``t()``), ``tui`` or
        ``desktop`` (served to the renderers over ``i18n.catalog``). Partial catalogs are fine; the last
        registration wins key by key. Resets the i18n caches; never changes ``display.language``. Raises
        ``ValueError`` for a malformed id/surface/file and ``FileNotFoundError`` for a missing path."""
        from agent.i18n_layers import (
            SURFACES, is_language_id, load_locale_source, normalize_language_id, register_pack, unregister_pack,
        )
        lang_id = normalize_language_id(lang)
        if not is_language_id(lang_id):
            raise self._refuse(f"locale with invalid language id {lang!r} (expected e.g. 'pl', 'pt-br')")
        if surface not in SURFACES:
            raise self._refuse(f"locale {lang_id!r} for unknown surface {surface!r} (one of {', '.join(SURFACES)})")
        messages = load_locale_source(source)
        entry = register_pack(lang_id, surface, messages, source=f"plugin:{self.manifest.name}",
                              endonym=endonym, rtl=rtl)
        handle = self._track("locale", f"{lang_id}.{surface}", lambda: unregister_pack(entry))
        _facade().logger.debug("Plugin %s registered locale: %s/%s (%d keys)", self.manifest.name, lang_id, surface,
                               len(messages))
        return handle

    def register_locale_dir(
        self, path: str | Path, *, metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> list[PluginRegistration]:
        """Register every ``<lang>[.tui|.desktop].yaml`` under ``path`` (a pack's ``locales/`` dir). The
        loader calls this for plugins declaring ``provides_locales``; ``metadata`` maps ids to
        ``{endonym, rtl}``. A broken file is skipped with a warning so one typo never disables the pack."""
        from agent.i18n_layers import scan_locale_dir
        handles: list[PluginRegistration] = []
        for lang_id, surface, file in scan_locale_dir(_facade().Path(path)):
            meta = dict((metadata or {}).get(lang_id) or {})
            try:
                handles.append(self.register_locale(
                    lang_id, file, surface=surface, endonym=meta.get("endonym"), rtl=bool(meta.get("rtl", False)),
                ))
            except Exception as exc:
                _facade().logger.warning("Plugin '%s' locale file %s skipped: %s", self.manifest.name, file, exc)
        return handles

    def register_hook(self, hook_name: str, callback: Callable, *, hold_stream: bool = False) -> PluginRegistration:
        """Register a lifecycle hook callback (``PluginDispatchMixin.track_hook``)."""
        return self._manager.track_hook(self, hook_name, callback, hold_stream=hold_stream)

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """Register behavior-changing middleware (request kinds rewrite the payload, execution kinds
        wrap the callback). Unknown kinds warn but are stored."""
        return self._track_callback(
            "middleware", kind, callback, self._manager._middleware, _facade().VALID_MIDDLEWARE
        )

    def register_session_env(self, names, resolver: Callable[[str], Any]) -> PluginRegistration:
        """Each session's tool subprocesses get ``resolver(session_id)``'s ``names``, resolved at spawn."""
        from hermes_cli.plugins_session_env import register_session_env
        release = register_session_env(self._manager.scope_key, names, resolver, owner=self.plugin_id)
        return self._track("session_env", getattr(resolver, "__qualname__", "resolver"), release)

    def _track_callback(
        self, kind: str, key: str, callback: Callable, mapping: dict[str, list[Callable]],
        valid: set[str],
    ) -> PluginRegistration:
        """Append ``callback`` under ``key`` (warning on unknown ``key``) and lease its removal."""
        if key not in valid:
            _facade().logger.warning("Plugin '%s' registered unknown %s '%s' (valid: %s)", self.manifest.name, kind,
                                     key, ", ".join(sorted(valid)))
        mapping.setdefault(key, []).append(callback)
        handle = self._track(kind, key, lambda: self._manager._remove_callback(mapping, key, callback))
        _facade().logger.debug("Plugin %s registered %s: %s", self.manifest.name, kind, key)
        return handle

    def register_system_prompt_section(
        self, id: str, content: str | Callable[[Mapping[str, Any]], str], *,
        position: str = "after_memory", max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context frozen into each new session prompt. Callables receive a
        read-only session-info mapping; the rendered prompt is persisted by core verbatim."""
        if not _facade().is_valid_system_prompt_section_id(id):
            raise ValueError("system prompt section id must be 1-128 lowercase characters "
                             "using letters, numbers, '.', '_', or '-'")
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in _facade().SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError("system prompt section position must be one of: "
                             + ", ".join(sorted(_facade().SYSTEM_PROMPT_SECTION_POSITIONS)))
        if (isinstance(max_chars, bool) or not isinstance(max_chars, int)
                or not 0 < max_chars <= _facade().MAX_SYSTEM_PROMPT_SECTION_CHARS):
            raise ValueError(f"system prompt section max_chars must be between 1 and {_facade().MAX_SYSTEM_PROMPT_SECTION_CHARS}")
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(f"system prompt section {id!r} is already registered by plugin {existing.plugin!r}")
        section = _facade().PluginSystemPromptSection(
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
            _facade().logger.warning("Plugin '%s' tried to emit an invalid event name %r", plugin_key, event)
            raise ValueError(f"Plugin '{plugin_key}' emit() requires a non-empty event name")
        if ":" in event:
            _facade().logger.warning("Plugin '%s' tried to emit namespaced/reserved event '%s' — a plugin may only emit "
                                     "bare event names under its own '%s:' namespace (the '%s:' prefix is reserved "
                                     "for core, and foreign namespaces are forbidden)",
                                     plugin_key, event, plugin_key, _facade().HERMES_EVENT_NAMESPACE)
            raise ValueError(f"Plugin '{plugin_key}' may not emit '{event}': emit only the bare event name; "
                             f"the namespace is forced to '{plugin_key}:' and the '{_facade().HERMES_EVENT_NAMESPACE}:' "
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
        _facade().logger.debug("Plugin %s subscribed to event: %s", self.manifest.name, event)

    from hermes_cli.plugins_content import register_automation_blueprint, register_skill
    from hermes_cli.plugin_rpc import emit_to_session, register_rpc

    # -- scoped provider registrars ----------------------------------------------------------------
    # One method per row of plugins_registrars.py's table, which also holds each method's docstring.

    @scoped_provider_registrar
    def register_login_backend(self, provider):
        return register_scoped_provider(self, "register_login_backend", provider)

    @scoped_provider_registrar
    def register_browser_page_backend(self, provider):
        return register_scoped_provider(self, "register_browser_page_backend", provider)

    @scoped_provider_registrar
    def register_image_gen_provider(self, provider):
        return register_scoped_provider(self, "register_image_gen_provider", provider)

    @scoped_provider_registrar
    def register_video_gen_provider(self, provider):
        return register_scoped_provider(self, "register_video_gen_provider", provider)

    @scoped_provider_registrar
    def register_web_search_provider(self, provider):
        return register_scoped_provider(self, "register_web_search_provider", provider)

    @scoped_provider_registrar
    def register_browser_provider(self, provider):
        return register_scoped_provider(self, "register_browser_provider", provider)

    @scoped_provider_registrar
    def register_terminal_environment_provider(self, provider):
        return register_scoped_provider(self, "register_terminal_environment_provider", provider)

    @scoped_provider_registrar
    def register_secret_source(self, source):
        return register_scoped_provider(self, "register_secret_source", source)

    @scoped_provider_registrar
    def register_tts_provider(self, provider):
        return register_scoped_provider(self, "register_tts_provider", provider)

    @scoped_provider_registrar
    def register_transcription_provider(self, provider):
        return register_scoped_provider(self, "register_transcription_provider", provider)

    @scoped_provider_registrar
    def register_delegation_provider(self, provider):
        return register_scoped_provider(self, "register_delegation_provider", provider)


def _ignore_after_abandoned_load(method):
    """Turn a registrar into a no-op once the context's load timed out: the abandoned worker thread may
    still be executing register(), and a late registration would land in registries that the failure
    path already swept (#108139)."""
    @_facade().wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_load_abandoned", False):
            _facade().logger.warning(
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

