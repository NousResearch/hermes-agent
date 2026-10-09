"""Where third-party Python plugin code is allowed to run (``plugins.isolation``).

``in_process`` (default) imports every plugin into the Hermes process, as always. ``host`` keeps
third-party plugin code OUT of it: general plugins load in one plugin-host process per profile
(:mod:`hermes_cli.plugin_host`) and talk to Hermes only through ``ctx``; every other path that would
import a user-supplied module in-process (category providers, model-provider profiles, dashboard
plugin APIs) refuses with a stated reason. Bundled plugins ship with Hermes and stay in-process.

That second half is the property a pooled deployment needs: one Hermes process serving many
profiles must never share an interpreter with a profile's plugin code. The same boundary gives a
single-user install crash isolation (a plugin that segfaults or hangs takes down its host, not the
gateway), which is why it is a plain config value and not a cloud-only switch.

The ctx surface tables below are the single source of truth for both the runtime boundary
(:mod:`hermes_cli.plugin_host_child` refuses these methods) and the static audit
(:mod:`hermes_cli.plugin_isolation_audit`), so "the audit says pool-safe" and "it loads in the host"
cannot drift apart.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, FrozenSet, List, Mapping, Optional

logger = logging.getLogger("hermes_cli.plugins")

ISOLATION_IN_PROCESS = "in_process"
ISOLATION_HOST = "host"
_ISOLATION_MODES = frozenset({ISOLATION_IN_PROCESS, ISOLATION_HOST})

# ctx methods that hand a plugin (or receive from it) live objects no process boundary can carry:
# raw platform SDK clients, argparse parsers, adapter classes bound to the gateway's event loop, the
# launch-scope dashboard auth registry. A plugin calling one of these runs only in-process.
HOST_UNSUPPORTED_CTX_METHODS: Dict[str, str] = {
    "register_platform": "platform adapters run on the gateway's event loop with native SDK clients",
    "register_platform_handler": "the handler factory receives the platform's native SDK client",
    "register_telegram_handler": "the handler factory receives the native Telegram Application",
    "register_slack_action_handler": "the callback receives native slack_bolt ack/body objects",
    "register_dashboard_auth_provider": "dashboard auth is owned by the launch process, not a profile",
    "register_approval_transport": "approval transports hold the live approval request/response channel",
}

# Registered in-process only; skipped (with a warning, the plugin still loads) inside the host
# because the surface they extend does not exist in a server process.
HOST_SKIPPED_CTX_METHODS: Dict[str, str] = {
    "register_cli_command": "`hermes <command>` subcommands are wired into the local CLI parser",
}

# Live ctx facades the host reaches by method call; each call returns plain data.
HOST_REMOTE_FACADES: FrozenSet[str] = frozenset({"state", "llm", "platform_actions", "subagent_lifecycle"})

# Object-taking ctx methods and the ABC the parent checks. The host sends the object's method table
# and the parent builds a subclass of that ABC whose methods run in the host.
HOST_OBJECT_BASES: Dict[str, str] = {
    "register_image_gen_provider": "agent.image_gen_provider:ImageGenProvider",
    "register_video_gen_provider": "agent.video_gen_provider:VideoGenProvider",
    "register_web_search_provider": "agent.web_search_provider:WebSearchProvider",
    "register_browser_provider": "agent.browser_provider:BrowserProvider",
    "register_terminal_environment_provider": "agent.terminal_env_provider:TerminalEnvironmentProvider",
    "register_secret_source": "agent.secret_sources.base:SecretSource",
    "register_tts_provider": "agent.tts_provider:TTSProvider",
    "register_transcription_provider": "agent.transcription_provider:TranscriptionProvider",
    "register_context_engine": "agent.context_engine:ContextEngine",
    "register_context_reference": "agent.context_references:ContextReferenceProvider",
    "register_memory_provider": "agent.memory_provider:MemoryProvider",
}

# Hooks whose payload carries live handles (the gateway runner, the session store). In the host the
# callback receives placeholders for those fields, so a plugin that uses them needs in-process.
HOST_DEGRADED_HOOKS: Dict[str, str] = {
    "pre_gateway_dispatch": "receives the live gateway runner and session store",
}


def _plugins_config(config: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
    if config is None:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly() or {}
    section = config.get("plugins") if isinstance(config, Mapping) else None
    return section if isinstance(section, Mapping) else {}


# Set in the plugin host's environment: inside the host, plugin code is already out of Hermes, so
# every loader there imports directly (a host must never spawn a host of its own).
HOST_PROCESS_ENV = "HERMES_PLUGIN_HOST_PROCESS"


def isolation_mode(config: Optional[Mapping[str, Any]] = None) -> str:
    """``plugins.isolation`` for the active profile; an unknown value warns and means in-process."""
    if os.environ.get(HOST_PROCESS_ENV) == "1":
        return ISOLATION_IN_PROCESS
    raw = _plugins_config(config).get("isolation", ISOLATION_IN_PROCESS)
    mode = str(raw or ISOLATION_IN_PROCESS).strip().lower().replace("-", "_")
    if mode not in _ISOLATION_MODES:
        logger.warning("plugins.isolation=%r is not one of %s; loading plugins in-process",
                       raw, ", ".join(sorted(_ISOLATION_MODES)))
        return ISOLATION_IN_PROCESS
    return mode


def host_launcher(config: Optional[Mapping[str, Any]] = None) -> List[str]:
    """``plugins.host.launcher``: argv prefix the host process runs under (a sandbox runner)."""
    host = _plugins_config(config).get("host")
    launcher = host.get("launcher") if isinstance(host, Mapping) else None
    if not launcher:
        return []
    if isinstance(launcher, str):
        import shlex
        return shlex.split(launcher)
    return [str(part) for part in launcher]


def in_process_import_refusal(what: str, *, source: str = "user",
                              config: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """Reason an in-process import of third-party code must not happen, or ``None`` when allowed.

    Every loader that imports a user-supplied module into the Hermes process asks this first;
    bundled code always passes.
    """
    if source == "bundled" or isolation_mode(config) != ISOLATION_HOST:
        return None
    return (f"{what} runs only in-process, and plugins.isolation is 'host' (third-party plugin code "
            f"never runs inside the Hermes process); set plugins.isolation: in_process to load it")


def user_plugin_host() -> Any:
    """The active profile's plugin host when ``plugins.isolation`` is ``host``, else ``None``.

    Category loaders (memory providers, context engines, cron schedulers) ask this before importing
    a user plugin directory, so the same plugins keep working with their code in the host.
    """
    if isolation_mode() != ISOLATION_HOST:
        return None
    from hermes_cli.plugins import get_plugin_manager
    return get_plugin_manager()._plugin_host()
