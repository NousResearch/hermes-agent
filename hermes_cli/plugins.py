"""
Hermes Plugin System
====================

Discovers, loads, and manages plugins from four sources:

1. **Bundled plugins** – ``<repo>/plugins/<name>/`` (shipped with hermes-agent;
   ``memory/`` and ``context_engine/`` subdirs are excluded — they have their
   own discovery paths)
2. **User plugins**   – ``~/.hermes/plugins/<name>/``
3. **Project plugins** – ``./.hermes/plugins/<name>/`` (opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``)
4. **Pip plugins**     – packages that expose the ``hermes_agent.plugins``
   entry-point group.

Later sources override earlier ones on name collision, so a user or project
plugin with the same name as a bundled plugin replaces it.

Each directory plugin must contain a ``plugin.yaml`` manifest **and** an
``__init__.py`` with a ``register(ctx)`` function.

Lifecycle hooks
---------------
Plugins may register callbacks for any of the hooks in ``VALID_HOOKS``.
The agent core calls ``invoke_hook(name, **kwargs)`` at the appropriate
points.

Tool registration
-----------------
``PluginContext.register_tool()`` delegates to ``tools.registry.register()``
so plugin-defined tools appear alongside the built-in tools.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import importlib.util
import inspect
import logging
import os
import sys
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from hermes_constants import get_hermes_home
from utils import env_var_enabled
from hermes_cli.config import cfg_get


def get_bundled_plugins_dir() -> Path:
    """Locate the bundled ``plugins/`` directory.

    Honours ``HERMES_BUNDLED_PLUGINS`` (set by the Nix wrapper / packaged
    installs) so read-only store paths are consulted first.  Falls back to
    the in-repo path used during development.
    """
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"

try:
    import yaml
except ImportError:  # pragma: no cover – yaml is optional at import time
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin developer debug logging
# ---------------------------------------------------------------------------
#
# Set ``HERMES_PLUGINS_DEBUG=1`` to surface verbose plugin-discovery logs to
# stderr in addition to ~/.hermes/logs/agent.log. Aimed at plugin authors
# trying to figure out why their plugin isn't showing up: which directories
# were scanned, which manifests parsed, which plugins were skipped (and why),
# what each ``register(ctx)`` call registered, and full tracebacks on load
# failure.
#
# The env var is read once at import time; tests that need to flip it
# mid-process can call ``_install_plugin_debug_handler(force=True)``.

_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG.

    Idempotent: only attaches the handler once per process unless ``force``
    is passed. Does not touch the root logger or other Hermes loggers.
    """
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on",
        }
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    # Don't double-emit through the root logger when the central logging
    # config also writes to stderr. agent.log still captures everything.
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug(
        "HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled"
    )


_install_plugin_debug_handler()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_HOOKS: Set[str] = {
    "pre_tool_call",
    "post_tool_call",
    "transform_terminal_output",
    "transform_tool_result",
    # Transform LLM output before it's returned to the user.
    # Plugins return a string to replace the response text, or None/empty to leave unchanged.
    # First non-None string wins. Useful for vocabulary/personality transformation.
    "transform_llm_output",
    "pre_llm_call",
    "post_llm_call",
    "pre_api_request",
    "post_api_request",
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "subagent_stop",
    # Gateway pre-dispatch hook. Fired once per incoming MessageEvent
    # after the internal-event guard but BEFORE auth/pairing and agent
    # dispatch. Plugins may return a dict to influence flow:
    #   {"action": "skip",    "reason": "..."}  -> drop message (no reply)
    #   {"action": "rewrite", "text": "..."}    -> replace event.text, continue
    #   {"action": "allow"}  /  None             -> normal dispatch
    # Kwargs: event: MessageEvent, gateway: GatewayRunner, session_store.
    "pre_gateway_dispatch",
    # Approval lifecycle hooks. Fired by tools/approval.py when a dangerous
    # command needs user approval -- fires BOTH for CLI-interactive prompts
    # and for gateway/ACP approvals (Telegram, Discord, Slack, TUI, etc.).
    # Observers only: return values are ignored. Plugins cannot veto or
    # pre-answer an approval from these hooks (use pre_tool_call to block
    # a tool before it reaches approval).
    #
    # Kwargs for pre_approval_request:
    #   command: str, description: str, pattern_key: str, pattern_keys: list[str],
    #   session_key: str, surface: "cli" | "gateway"
    # Kwargs for post_approval_response: same as above plus
    #   choice: "once" | "session" | "always" | "deny" | "timeout"
    "pre_approval_request",
    "post_approval_response",
}

ENTRY_POINTS_GROUP = "hermes_agent.plugins"

_NS_PARENT = "hermes_plugins"


def _env_enabled(name: str) -> bool:
    """Return True when an env var is set to a truthy opt-in value."""
    return env_var_enabled(name)


def _get_disabled_plugins() -> set:
    """Read the disabled plugins list from config.yaml.

    Kept for backward compat and explicit deny-list semantics. A plugin
    name in this set will never load, even if it appears in
    ``plugins.enabled``.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        disabled = cfg_get(config, "plugins", "disabled", default=[])
        return set(disabled) if isinstance(disabled, list) else set()
    except Exception:
        return set()


def _get_enabled_plugins() -> Optional[set]:
    """Read the enabled-plugins allow-list from config.yaml.

    Plugins are opt-in by default — only plugins whose name appears in
    this set are loaded. Returns:

    * ``None`` — the key is missing or malformed. Callers should treat
      this as "nothing enabled yet" (the opt-in default); the first
      ``migrate_config`` run populates the key with a grandfathered set
      of currently-installed user plugins so existing setups don't
      break on upgrade.
    * ``set()`` — an empty list was explicitly set; nothing loads.
    * ``set(...)`` — the concrete allow-list.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
        plugins_cfg = config.get("plugins")
        if not isinstance(plugins_cfg, dict):
            return None
        if "enabled" not in plugins_cfg:
            return None
        enabled = plugins_cfg.get("enabled")
        if not isinstance(enabled, list):
            return None
        return set(enabled)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_VALID_PLUGIN_KINDS: Set[str] = {"standalone", "backend", "exclusive", "platform", "model-provider"}


@dataclass
class PluginManifest:
    """Parsed representation of a plugin.yaml manifest."""

    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    requires_env: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[str] = field(default_factory=list)
    source: str = ""        # "user", "project", or "entrypoint"
    path: Optional[str] = None
    # Plugin kind — see plugins.py module docstring for semantics.
    # ``standalone`` (default): hooks/tools of its own; opt-in via
    #                           ``plugins.enabled``.
    # ``backend``: pluggable backend for an existing core tool (e.g.
    #              image_gen). Built-in (bundled) backends auto-load;
    #              user-installed still gated by ``plugins.enabled``.
    # ``exclusive``: category with exactly one active provider (memory).
    #              Selection via ``<category>.provider`` config key; the
    #              category's own discovery system handles loading and the
    #              general scanner skips these.
    # ``platform``: gateway messaging platform adapter (e.g. IRC). Bundled
    #              platform plugins auto-load so every shipped platform is
    #              available out of the box; user-installed platform plugins
    #              in ~/.hermes/plugins/ still gated by ``plugins.enabled``
    #              (untrusted code).
    kind: str = "standalone"
    # Registry key — path-derived, used by ``plugins.enabled``/``disabled``
    # lookups and by ``hermes plugins list``. For a flat plugin at
    # ``plugins/disk-cleanup/`` the key is ``disk-cleanup``; for a nested
    # category plugin at ``plugins/image_gen/openai/`` the key is
    # ``image_gen/openai``. When empty, falls back to ``name``.
    key: str = ""


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    commands_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# PluginContext  – handed to each plugin's ``register()`` function
# ---------------------------------------------------------------------------

class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        # Lazy-built host-owned LLM facade — see ctx.llm property below.
        self._llm: Any = None

    # -- host-owned LLM access ----------------------------------------------

    @property
    def llm(self) -> Any:
        """Return the plugin's :class:`agent.plugin_llm.PluginLlm` facade.

        Lets trusted plugins run host-owned chat or structured completions
        against the user's active model and auth without bringing their
        own provider keys. Override capability (model, agent id, auth
        profile) is fail-closed by default and gated through
        ``plugins.entries.<plugin_id>.llm.*`` config keys.

        See :mod:`agent.plugin_llm` for the full surface."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            plugin_id = self.manifest.key or self.manifest.name
            self._llm = PluginLlm(plugin_id=plugin_id)
        return self._llm

    # -- tool registration --------------------------------------------------

    def register_tool(
        self,
        name: str,
        toolset: str,
        schema: dict,
        handler: Callable,
        check_fn: Callable | None = None,
        requires_env: list | None = None,
        is_async: bool = False,
        description: str = "",
        emoji: str = "",
    ) -> None:
        """Register a tool in the global registry **and** track it as plugin-provided."""
        from tools.registry import registry

        registry.register(
            name=name,
            toolset=toolset,
            schema=schema,
            handler=handler,
            check_fn=check_fn,
            requires_env=requires_env,
            is_async=is_async,
            description=description,
            emoji=emoji,
        )
        self._manager._plugin_tool_names.add(name)
        logger.debug("Plugin %s registered tool: %s", self.manifest.name, name)

    # -- message injection --------------------------------------------------

    def inject_message(self, content: str, role: str = "user") -> bool:
        """Inject a message into the active conversation.

        If the agent is idle (waiting for user input), this starts a new turn.
        If the agent is running, this interrupts and injects the message.

        This enables plugins (e.g. remote control viewers, messaging bridges)
        to send messages into the conversation from external sources.

        Returns True if the message was queued successfully.
        """
        cli = self._manager._cli_ref
        if cli is None:
            logger.warning("inject_message: no CLI reference (not available in gateway mode)")
            return False

        msg = content if role == "user" else f"[{role}] {content}"

        if getattr(cli, "_agent_running", False):
            # Agent is mid-turn — interrupt with the message
            cli._interrupt_queue.put(msg)
        else:
            # Agent is idle — queue as next input
            cli._pending_input.put(msg)
        return True

    # -- CLI command registration --------------------------------------------

    def register_cli_command(
        self,
        name: str,
        help: str,
        setup_fn: Callable,
        handler_fn: Callable | None = None,
        description: str = "",
    ) -> None:
        """Register a CLI subcommand (e.g. ``hermes honcho ...``).

        The *setup_fn* receives an argparse subparser and should add any
        arguments/sub-subparsers.  If *handler_fn* is provided it is set
        as the default dispatch function via ``set_defaults(func=...)``."""
        self._manager._cli_commands[name] = {
            "name": name,
            "help": help,
            "description": description,
            "setup_fn": setup_fn,
            "handler_fn": handler_fn,
            "plugin": self.manifest.name,
        }
        logger.debug("Plugin %s registered CLI command: %s", self.manifest.name, name)

    # -- slash command registration -------------------------------------------

    def register_command(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        args_hint: str = "",
    ) -> None:
        """Register a slash command (e.g. ``/lcm``) available in CLI and gateway sessions.

        The handler signature is ``fn(raw_args: str) -> str | None``.
        It may also be an async callable — the gateway dispatch handles both.

        Unlike ``register_cli_command()`` (which creates ``hermes <subcommand>``
        terminal commands), this registers in-session slash commands that users
        invoke during a conversation.

        ``args_hint`` is an optional short string (e.g. ``"<file>"`` or
        ``"dias:7 formato:json"``) used by gateway adapters to surface the
        command with an argument field — for example Discord's native slash
        command picker. Plugin commands without ``args_hint`` register as
        parameterless in Discord and still accept trailing text when invoked
        as free-form chat.

        Names conflicting with built-in commands are rejected with a warning.
        """
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.",
                self.manifest.name,
            )
            return

        # Reject if it conflicts with a built-in command
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Plugin '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.",
                    self.manifest.name, clean,
                )
                return
        except Exception:
            pass  # If commands module isn't available, skip the check

        self._manager._plugin_commands[clean] = {
            "handler": handler,
            "description": description or "Plugin command",
            "plugin": self.manifest.name,
            "args_hint": (args_hint or "").strip(),
        }
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)

    # -- tool dispatch -------------------------------------------------------

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry, with parent agent context.

        This is the public interface for plugin slash commands that need to call
        tools like ``delegate_task`` without reaching into framework internals.
        The parent agent (if available) is resolved automatically — plugins never
        need to access the agent directly.

        Args:
            tool_name: Registry name of the tool (e.g. ``"delegate_task"``).
            args: Tool arguments dict (same as what the model would pass).
            **kwargs: Extra keyword args forwarded to the registry dispatch.

        Returns:
            JSON string from the tool handler (same format as model tool calls).
        """
        from tools.registry import registry

        # Wire up parent agent context when available (CLI mode).
        # In gateway mode _cli_ref is None — tools degrade gracefully
        # (workspace hints fall back to TERMINAL_CWD, no spinner).
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return registry.dispatch(tool_name, args, **kwargs)

    # -- context engine registration -----------------------------------------

    def register_context_engine(self, engine) -> None:
        """Register a context engine to replace the built-in ContextCompressor.

        Only one context engine plugin is allowed. If a second plugin tries
        to register one, it is rejected with a warning.

        The engine must be an instance of ``agent.context_engine.ContextEngine``.
        """
        if self._manager._context_engine is not None:
            logger.warning(
                "Plugin '%s' tried to register a context engine, but one is "
                "already registered. Only one context engine plugin is allowed.",
                self.manifest.name,
            )
            return
        # Defer the import to avoid circular deps at module level
        from agent.context_engine import ContextEngine
        if not isinstance(engine, ContextEngine):
            logger.warning(
                "Plugin '%s' tried to register a context engine that does not "
                "inherit from ContextEngine. Ignoring.",
                self.manifest.name,
            )
            return
        self._manager._context_engine = engine
        logger.info(
            "Plugin '%s' registered context engine: %s",
            self.manifest.name, engine.name,
        )

    # -- image gen provider registration ------------------------------------

    def register_image_gen_provider(self, provider) -> None:
        """Register an image generation backend.

        ``provider`` must be an instance of
        :class:`agent.image_gen_provider.ImageGenProvider`. The
        ``provider.name`` attribute is what ``image_gen.provider`` in
        ``config.yaml`` matches against when routing ``image_generate``
        tool calls.
        """
        from agent.image_gen_provider import ImageGenProvider
        from agent.image_gen_registry import register_provider

        if not isinstance(provider, ImageGenProvider):
            logger.warning(
                "Plugin '%s' tried to register an image_gen provider that does "
                "not inherit from ImageGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        register_provider(provider)
        logger.info(
            "Plugin '%s' registered image_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- video gen provider registration -------------------------------------

    def register_video_gen_provider(self, provider) -> None:
        """Register a video generation backend.

        ``provider`` must be an instance of
        :class:`agent.video_gen_provider.VideoGenProvider`. The
        ``provider.name`` attribute is what ``video_gen.provider`` in
        ``config.yaml`` matches against when routing ``video_generate``
        tool calls.
        """
        from agent.video_gen_provider import VideoGenProvider
        from agent.video_gen_registry import register_provider as _register_video_provider

        if not isinstance(provider, VideoGenProvider):
            logger.warning(
                "Plugin '%s' tried to register a video_gen provider that does "
                "not inherit from VideoGenProvider. Ignoring.",
                self.manifest.name,
            )
            return
        _register_video_provider(provider)
        logger.info(
            "Plugin '%s' registered video_gen provider: %s",
            self.manifest.name, provider.name,
        )

    # -- platform adapter registration ---------------------------------------

    def register_platform(
        self,
        name: str,
        label: str,
        adapter_factory: Callable,
        check_fn: Callable,
        validate_config: Callable | None = None,
        required_env: list | None = None,
        install_hint: str = "",
        **entry_kwargs: Any,
    ) -> None:
        """Register a gateway platform adapter.

        The adapter_factory receives a ``PlatformConfig`` and returns a
        ``BasePlatformAdapter`` subclass instance.  The gateway calls
        ``check_fn()`` before instantiation to verify dependencies.

        Extra keyword arguments are forwarded to ``PlatformEntry`` (e.g.
        ``setup_fn``, ``emoji``, ``allowed_users_env``, ``platform_hint``).
        Unknown keys raise TypeError from the dataclass constructor.

        Example::

            ctx.register_platform(
                name="irc",
                label="IRC",
                adapter_factory=lambda cfg: IRCAdapter(cfg),
                check_fn=lambda: True,
                emoji="💬",
                setup_fn=irc_interactive_setup,
            )
        """
        from gateway.platform_registry import platform_registry, PlatformEntry

        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name,
            label=label,
            adapter_factory=adapter_factory,
            check_fn=check_fn,
            validate_config=validate_config,
            required_env=required_env or [],
            install_hint=install_hint,
            source="plugin",
            **entry_kwargs,
        )
        platform_registry.register(entry)
        self._manager._plugin_platform_names.add(name)
        logger.debug(
            "Plugin %s registered platform: %s",
            self.manifest.name,
            name,
        )

    # -- hook registration --------------------------------------------------

    def register_hook(self, hook_name: str, callback: Callable) -> None:
        """Register a lifecycle hook callback.

        Unknown hook names produce a warning but are still stored so
        forward-compatible plugins don't break.
        """
        if hook_name not in VALID_HOOKS:
            logger.warning(
                "Plugin '%s' registered unknown hook '%s' "
                "(valid: %s)",
                self.manifest.name,
                hook_name,
                ", ".join(sorted(VALID_HOOKS)),
            )
        self._manager._hooks.setdefault(hook_name, []).append(callback)
        logger.debug("Plugin %s registered hook: %s", self.manifest.name, hook_name)

    # -- skill registration -------------------------------------------------

    def register_skill(
        self,
        name: str,
        path: Path,
        description: str = "",
    ) -> None:
        """Register a read-only skill provided by this plugin.

        The skill becomes resolvable as ``'<plugin_name>:<name>'`` via
        ``skill_view()``.  It does **not** enter the flat
        ``~/.hermes/skills/`` tree and is **not** listed in the system
        prompt's ``<available_skills>`` index — plugin skills are
        opt-in explicit loads only.

        Raises:
            ValueError: if *name* contains ``':'`` or invalid characters.
            FileNotFoundError: if *path* does not exist.
        """
        from agent.skill_utils import _NAMESPACE_RE

        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(
                f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+."
            )
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        qualified = f"{self.manifest.name}:{name}"
        self._manager._plugin_skills[qualified] = {
            "path": path,
            "plugin": self.manifest.name,
            "bare_name": name,
            "description": description,
        }
        logger.debug(
            "Plugin %s registered skill: %s",
            self.manifest.name, qualified,
        )


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------

class PluginManager:
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self) -> None:
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: Dict[str, dict] = {}  # Slash commands registered by plugins
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        # Plugin skill registry: qualified name → metadata dict.
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found.

        When ``force`` is true, clear cached discovery state first so config
        changes or newly-added bundled backends become visible in long-lived
        sessions without requiring a full agent restart.
        """
        if self._discovered and not force:
            return
        if force:
            self._plugins.clear()
            self._hooks.clear()
            self._plugin_tool_names.clear()
            self._cli_commands.clear()
            self._plugin_commands.clear()
            self._plugin_skills.clear()
            self._context_engine = None
        self._discovered = True

        manifests: List[PluginManifest] = []

        # 1. Bundled plugins (<repo>/plugins/<name>/)
        #
        # Repo-shipped plugins live next to hermes_cli/. Two layouts are
        # supported (see ``_scan_directory`` for details):
        #
        #   - flat: ``plugins/disk-cleanup/plugin.yaml`` (standalone)
        #   - category: ``plugins/image_gen/openai/plugin.yaml`` (backend)
        #
        # ``memory/``, ``context_engine/``, and ``model-providers/`` are
        # skipped at the top level — they have their own discovery systems
        # (plugins/memory/__init__.py, providers/__init__.py). ``platforms/``
        # is a category holding platform adapters (scanned one level deeper
        # below).
        repo_plugins = get_bundled_plugins_dir()
        logger.debug("Scanning bundled plugins: %s", repo_plugins)
        bundled = self._scan_directory(
            repo_plugins,
            source="bundled",
            skip_names={"memory", "context_engine", "platforms", "model-providers"},
        )
        logger.debug("  bundled (top-level): %d manifest(s)", len(bundled))
        manifests.extend(bundled)
        bundled_platforms = self._scan_directory(
            repo_plugins / "platforms", source="bundled"
        )
        logger.debug("  bundled/platforms: %d manifest(s)", len(bundled_platforms))
        manifests.extend(bundled_platforms)

        # 2. User plugins (~/.hermes/plugins/)
        user_dir = get_hermes_home() / "plugins"
        logger.debug("Scanning user plugins: %s", user_dir)
        user_manifests = self._scan_directory(user_dir, source="user")
        logger.debug("  user: %d manifest(s)", len(user_manifests))
        manifests.extend(user_manifests)

        # 3. Project plugins (./.hermes/plugins/)
        if _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            project_dir = Path.cwd() / ".hermes" / "plugins"
            logger.debug("Scanning project plugins: %s", project_dir)
            project_manifests = self._scan_directory(project_dir, source="project")
            logger.debug("  project: %d manifest(s)", len(project_manifests))
            manifests.extend(project_manifests)
        else:
            logger.debug(
                "Project plugins disabled (set HERMES_ENABLE_PROJECT_PLUGINS=1 to enable)"
            )

        # 4. Pip / entry-point plugins
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        # Load each manifest (skip user-disabled plugins).
        # Later sources override earlier ones on key collision — user
        # plugins take precedence over bundled, project plugins take
        # precedence over user. Dedup here so we only load the final
        # winner. Keys are path-derived (``image_gen/openai``,
        # ``disk-cleanup``) so ``tts/openai`` and ``image_gen/openai``
        # don't collide even when both manifests say ``name: openai``.
        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        winners: Dict[str, PluginManifest] = {}
        for manifest in manifests:
            winners[manifest.key or manifest.name] = manifest
        for manifest in winners.values():
            lookup_key = manifest.key or manifest.name

            # Explicit disable always wins (matches on key or on legacy
            # bare name for back-compat with existing user configs).
            if lookup_key in disabled or manifest.name in disabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = "disabled via config"
                self._plugins[lookup_key] = loaded
                logger.debug("Skipping disabled plugin '%s'", lookup_key)
                continue

            # Exclusive plugins (memory providers) have their own
            # discovery/activation path. The general loader records the
            # manifest for introspection but does not load the module.
            if manifest.kind == "exclusive":
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "exclusive plugin — activate via <category>.provider config"
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (exclusive, handled by category discovery)",
                    lookup_key,
                )
                continue

            # Model provider plugins are loaded by providers/__init__.py
            # (its own lazy discovery keyed off first get_provider_profile()
            # call). We record the manifest here for introspection but do
            # not import the module — a second import would create two
            # ProviderProfile instances and break the "last writer wins"
            # override semantics between bundled and user plugins.
            if manifest.kind == "model-provider":
                loaded = LoadedPlugin(manifest=manifest, enabled=True)
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (model-provider, handled by providers/ discovery)",
                    lookup_key,
                )
                continue

            # Built-in backends auto-load — they ship with hermes and must
            # just work. Selection among them (e.g. which image_gen backend
            # services calls) is driven by ``<category>.provider`` config,
            # enforced by the tool wrapper.
            #
            # Bundled platform plugins (gateway adapters like IRC) auto-load
            # for the same reason: every platform Hermes ships must be
            # available out of the box without the user having to opt in.
            if manifest.source == "bundled" and manifest.kind in {"backend", "platform"}:
                self._load_plugin(manifest)
                continue

            # Everything else (standalone, user-installed backends,
            # entry-point plugins) is opt-in via plugins.enabled.
            # Accept both the path-derived key and the legacy bare name
            # so existing configs keep working.
            is_enabled = (
                enabled is not None
                and (lookup_key in enabled or manifest.name in enabled)
            )
            if not is_enabled:
                loaded = LoadedPlugin(manifest=manifest, enabled=False)
                loaded.error = (
                    "not enabled in config (run `hermes plugins enable {}` to activate)"
                    .format(lookup_key)
                )
                self._plugins[lookup_key] = loaded
                logger.debug(
                    "Skipping '%s' (not in plugins.enabled)", lookup_key
                )
                continue
            self._load_plugin(manifest)

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled",
                len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    # -----------------------------------------------------------------------
    # Directory scanning
    # -----------------------------------------------------------------------

    def _scan_directory(
        self,
        path: Path,
        source: str,
        skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read ``plugin.yaml`` manifests from subdirectories of *path*.

        Supports two layouts, mixed freely:

        * **Flat** — ``<root>/<plugin-name>/plugin.yaml``. Key is
          ``<plugin-name>`` (e.g. ``disk-cleanup``).
        * **Category** — ``<root>/<category>/<plugin-name>/plugin.yaml``,
          where the ``<category>`` directory itself has no ``plugin.yaml``.
          Key is ``<category>/<plugin-name>`` (e.g. ``image_gen/openai``).
          Depth is capped at two segments.

        *skip_names* is an optional allow-list of names to ignore at the
        top level (kept for back-compat; the current call sites no longer
        pass it now that categories are first-class).
        """
        return self._scan_directory_level(
            path, source, skip_names=skip_names, prefix="", depth=0
        )

    def _scan_directory_level(
        self,
        path: Path,
        source: str,
        *,
        skip_names: Optional[Set[str]],
        prefix: str,
        depth: int,
    ) -> List[PluginManifest]:
        """Recursive implementation of :meth:`_scan_directory`.

        ``prefix`` is the category path already accumulated ("" at root,
        "image_gen" one level in). ``depth`` is the recursion depth; we
        cap at 2 so ``<root>/a/b/c/`` is ignored.
        """
        manifests: List[PluginManifest] = []
        if not path.is_dir():
            return manifests

        for child in sorted(path.iterdir()):
            if not child.is_dir():
                continue
            if depth == 0 and skip_names and child.name in skip_names:
                continue
            manifest_file = child / "plugin.yaml"
            if not manifest_file.exists():
                manifest_file = child / "plugin.yml"

            if manifest_file.exists():
                manifest = self._parse_manifest(
                    manifest_file, child, source, prefix
                )
                if manifest is not None:
                    manifests.append(manifest)
                continue

            # No manifest at this level. If we're still within the depth
            # cap, treat this directory as a category namespace and recurse
            # one level in looking for children with manifests.
            if depth >= 1:
                logger.debug("Skipping %s (no plugin.yaml, depth cap reached)", child)
                continue

            sub_prefix = f"{prefix}/{child.name}" if prefix else child.name
            manifests.extend(
                self._scan_directory_level(
                    child,
                    source,
                    skip_names=None,
                    prefix=sub_prefix,
                    depth=depth + 1,
                )
            )

        return manifests

    def _parse_manifest(
        self,
        manifest_file: Path,
        plugin_dir: Path,
        source: str,
        prefix: str,
    ) -> Optional[PluginManifest]:
        """Parse a single ``plugin.yaml`` into a :class:`PluginManifest`.

        Returns ``None`` on parse failure (logs a warning).
        """
        try:
            if yaml is None:
                logger.warning("PyYAML not installed – cannot load %s", manifest_file)
                return None
            data = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}

            name = data.get("name", plugin_dir.name)
            key = f"{prefix}/{plugin_dir.name}" if prefix else name

            raw_kind = data.get("kind", "standalone")
            if not isinstance(raw_kind, str):
                raw_kind = "standalone"
            kind = raw_kind.strip().lower()
            if kind not in _VALID_PLUGIN_KINDS:
                logger.warning(
                    "Plugin %s: unknown kind '%s' (valid: %s); treating as 'standalone'",
                    key, raw_kind, ", ".join(sorted(_VALID_PLUGIN_KINDS)),
                )
                kind = "standalone"

            # Auto-coerce user-installed memory providers to kind="exclusive"
            # so they're routed to plugins/memory discovery instead of being
            # loaded by the general PluginManager (which has no
            # register_memory_provider on PluginContext). Mirrors the
            # heuristic in plugins/memory/__init__.py:_is_memory_provider_dir.
            # Bundled memory providers are already skipped via skip_names.
            if kind == "standalone" and "kind" not in data:
                init_file = plugin_dir / "__init__.py"
                if init_file.exists():
                    try:
                        source_text = init_file.read_text(errors="replace")[:8192]
                        if (
                            "register_memory_provider" in source_text
                            or "MemoryProvider" in source_text
                        ):
                            kind = "exclusive"
                            logger.debug(
                                "Plugin %s: detected memory provider, "
                                "treating as kind='exclusive'",
                                key,
                            )
                        elif (
                            "register_provider" in source_text
                            and "ProviderProfile" in source_text
                        ):
                            # Model provider plugin (calls register_provider()
                            # from ``providers`` with a ProviderProfile). Route
                            # to providers/__init__.py discovery.
                            kind = "model-provider"
                            logger.debug(
                                "Plugin %s: detected model provider, "
                                "treating as kind='model-provider'",
                                key,
                            )
                    except Exception:
                        pass

            logger.debug(
                "Parsed manifest: key=%s name=%s kind=%s source=%s path=%s",
                key, name, kind, source, plugin_dir,
            )
            return PluginManifest(
                name=name,
                version=str(data.get("version", "")),
                description=data.get("description", ""),
                author=data.get("author", ""),
                requires_env=data.get("requires_env", []),
                provides_tools=data.get("provides_tools", []),
                provides_hooks=data.get("provides_hooks", []),
                source=source,
                path=str(plugin_dir),
                kind=kind,
                key=key,
            )
        except Exception as exc:
            logger.warning(
                "Failed to parse %s: %s", manifest_file, exc, exc_info=_PLUGINS_DEBUG,
            )
            return None

    # -----------------------------------------------------------------------
    # Entry-point scanning
    # -----------------------------------------------------------------------

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Check ``importlib.metadata`` for pip-installed plugins."""
        manifests: List[PluginManifest] = []
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                group_eps = eps.select(group=ENTRY_POINTS_GROUP)
            elif isinstance(eps, dict):
                group_eps = eps.get(ENTRY_POINTS_GROUP, [])
            else:
                group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

            for ep in group_eps:
                manifest = PluginManifest(
                    name=ep.name,
                    source="entrypoint",
                    path=ep.value,
                    key=ep.name,
                )
                manifests.append(manifest)
        except Exception as exc:
            logger.debug("Entry-point scan failed: %s", exc)

        return manifests

    # -----------------------------------------------------------------------
    # Loading
    # -----------------------------------------------------------------------

    def _load_plugin(self, manifest: PluginManifest) -> None:
        """Import a plugin module and call its ``register(ctx)`` function."""
        loaded = LoadedPlugin(manifest=manifest)
        logger.debug(
            "Loading plugin '%s' (source=%s, kind=%s, path=%s)",
            manifest.key or manifest.name, manifest.source, manifest.kind, manifest.path,
        )

        try:
            if manifest.source in {"user", "project", "bundled"}:
                module = self._load_directory_module(manifest)
            else:
                module = self._load_entrypoint_module(manifest)

            loaded.module = module

            # Call register()
            register_fn = getattr(module, "register", None)
            if register_fn is None:
                loaded.error = "no register() function"
                logger.warning("Plugin '%s' has no register() function", manifest.name)
            else:
                ctx = PluginContext(manifest, self)
                register_fn(ctx)
                loaded.tools_registered = [
                    t for t in self._plugin_tool_names
                    if t not in {
                        n
                        for name, p in self._plugins.items()
                        for n in p.tools_registered
                    }
                ]
                loaded.hooks_registered = list(
                    {
                        h
                        for h, cbs in self._hooks.items()
                        if cbs  # non-empty
                    }
                    - {
                        h
                        for name, p in self._plugins.items()
                        for h in p.hooks_registered
                    }
                )
                loaded.commands_registered = [
                    c for c in self._plugin_commands
                    if self._plugin_commands[c].get("plugin") == manifest.name
                ]
                loaded.enabled = True
                logger.debug(
                    "  registered: %d tool(s), %d hook(s), %d slash command(s), %d CLI command(s)",
                    len(loaded.tools_registered),
                    len(loaded.hooks_registered),
                    len(loaded.commands_registered),
                    sum(
                        1 for c in self._cli_commands
                        if self._cli_commands[c].get("plugin") == manifest.name
                    ),
                )

        except Exception as exc:
            loaded.error = str(exc)
            logger.warning(
                "Failed to load plugin '%s': %s",
                manifest.name, exc, exc_info=_PLUGINS_DEBUG,
            )

        self._plugins[manifest.key or manifest.name] = loaded

    def _load_directory_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Import a directory-based plugin as ``hermes_plugins.<slug>``.

        The module slug is derived from ``manifest.key`` so category-namespaced
        plugins (``image_gen/openai``) import as
        ``hermes_plugins.image_gen__openai`` without colliding with any
        future ``tts/openai``.
        """
        plugin_dir = Path(manifest.path)  # type: ignore[arg-type]
        init_file = plugin_dir / "__init__.py"
        if not init_file.exists():
            raise FileNotFoundError(f"No __init__.py in {plugin_dir}")

        # Ensure the namespace parent package exists
        if _NS_PARENT not in sys.modules:
            ns_pkg = types.ModuleType(_NS_PARENT)
            ns_pkg.__path__ = []  # type: ignore[attr-defined]
            ns_pkg.__package__ = _NS_PARENT
            sys.modules[_NS_PARENT] = ns_pkg

        key = manifest.key or manifest.name
        slug = key.replace("/", "__").replace("-", "_")
        module_name = f"{_NS_PARENT}.{slug}"
        spec = importlib.util.spec_from_file_location(
            module_name,
            init_file,
            submodule_search_locations=[str(plugin_dir)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {init_file}")

        module = importlib.util.module_from_spec(spec)
        module.__package__ = module_name
        module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_entrypoint_module(self, manifest: PluginManifest) -> types.ModuleType:
        """Load a pip-installed plugin via its entry-point reference."""
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            group_eps = eps.select(group=ENTRY_POINTS_GROUP)
        elif isinstance(eps, dict):
            group_eps = eps.get(ENTRY_POINTS_GROUP, [])
        else:
            group_eps = [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]

        for ep in group_eps:
            if ep.name == manifest.name:
                return ep.load()

        raise ImportError(
            f"Entry point '{manifest.name}' not found in group '{ENTRY_POINTS_GROUP}'"
        )

    # -----------------------------------------------------------------------
    # Hook invocation
    # -----------------------------------------------------------------------

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> List[Any]:
        """Call all registered callbacks for *hook_name*.

        Each callback is wrapped in its own try/except so a misbehaving
        plugin cannot break the core agent loop.

        Returns a list of non-``None`` return values from callbacks.

        For ``pre_llm_call``, callbacks may return a dict describing
        context to inject into the current turn's user message::

            {"context": "recalled text..."}
            "recalled text..."          # plain string, equivalent

        Context is ALWAYS injected into the user message, never the
        system prompt.  This preserves the prompt cache prefix — the
        system prompt stays identical across turns so cached tokens
        are reused.  All injected context is ephemeral — never
        persisted to session DB.
        """
        callbacks = self._hooks.get(hook_name, [])
        results: List[Any] = []
        for cb in callbacks:
            try:
                ret = cb(**kwargs)
                if ret is not None:
                    results.append(ret)
            except Exception as exc:
                logger.warning(
                    "Hook '%s' callback %s raised: %s",
                    hook_name,
                    getattr(cb, "__name__", repr(cb)),
                    exc,
                )
        return results

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: List[Dict[str, Any]] = []
        for key, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": loaded.manifest.name,
                    "key": loaded.manifest.key or loaded.manifest.name,
                    "kind": loaded.manifest.kind,
                    "version": loaded.manifest.version,
                    "description": loaded.manifest.description,
                    "source": loaded.manifest.source,
                    "enabled": loaded.enabled,
                    "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "commands": len(loaded.commands_registered),
                    "error": loaded.error,
                }
            )
        return result

    # -----------------------------------------------------------------------
    # Plugin skill lookups
    # -----------------------------------------------------------------------

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(
            e["bare_name"]
            for qn, e in self._plugin_skills.items()
            if qn.startswith(prefix)
        )

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# ---------------------------------------------------------------------------
# Module-level singleton & convenience functions
# ---------------------------------------------------------------------------

_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """Return (and lazily create) the global PluginManager singleton."""
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins.

    Default behavior is idempotent. Pass ``force=True`` to rescan plugin
    manifests and reload state in the current process.
    """
    get_plugin_manager().discover_and_load(force=force)


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook on all loaded plugins.

    Returns a list of non-``None`` return values from plugin callbacks.
    """
    return get_plugin_manager().invoke_hook(hook_name, **kwargs)



def get_pre_tool_call_block_message(
    tool_name: str,
    args: Optional[Dict[str, Any]],
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
) -> Optional[str]:
    """Check ``pre_tool_call`` hooks for a blocking directive.

    Plugins that need to enforce policy (rate limiting, security
    restrictions, approval workflows) can return::

        {"action": "block", "message": "Reason the tool was blocked"}

    from their ``pre_tool_call`` callback.  The first valid block
    directive wins.  Invalid or irrelevant hook return values are
    silently ignored so existing observer-only hooks are unaffected.
    """
    hook_results = invoke_hook(
        "pre_tool_call",
        tool_name=tool_name,
        args=args if isinstance(args, dict) else {},
        task_id=task_id,
        session_id=session_id,
        tool_call_id=tool_call_id,
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        if result.get("action") != "block":
            continue
        message = result.get("message")
        if isinstance(message, str) and message:
            return message

    return None


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after ensuring plugin discovery has run.

    Pass ``force=True`` to rescan in the current process.
    """
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return _ensure_plugins_discovered()._context_engine


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = _ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command return value, awaiting async handlers when needed.

    Sync CLI/TUI dispatch sites call plugin handlers from plain functions.
    If a handler is async, await it directly when no loop is running; if
    we're already inside an active loop, run it in a helper thread with its
    own loop so the caller still gets a concrete result synchronously. The
    threaded path is bounded by a 30s timeout so a hung async handler cannot
    wedge the terminal indefinitely.
    """
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(
        target=_runner,
        name="hermes-plugin-command-await",
        daemon=True,
    )
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Return the full plugin commands dict (name → {handler, description, plugin}).

    Triggers idempotent plugin discovery so callers can use plugin commands
    before any explicit discover_plugins() call.
    """
    return _ensure_plugins_discovered()._plugin_commands


def get_plugin_toolsets() -> List[tuple]:
    """Return plugin toolsets as ``(key, label, description)`` tuples.

    Used by the ``hermes tools`` TUI so plugin-provided toolsets appear
    alongside the built-in ones and can be toggled on/off per platform.
    """
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if not entry:
            continue
        ts = entry.toolset
        toolset_tools.setdefault(ts, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    for _name, loaded in manager._plugins.items():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        label = f"🔌 {ts_key.replace('_', ' ').title()}"
        if plugin and plugin.manifest.description:
            desc = plugin.manifest.description
        else:
            desc = ", ".join(sorted(toolset_tools[ts_key]))
        result.append((ts_key, label, desc))

    return result
