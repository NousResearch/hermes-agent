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

import contextlib
import importlib.util
import logging
import os
import sys
from types import ModuleType
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional

logger = logging.getLogger("hermes_cli.plugins")

ISOLATION_IN_PROCESS = "in_process"
ISOLATION_HOST = "host"
_ISOLATION_MODES = frozenset({ISOLATION_IN_PROCESS, ISOLATION_HOST})

# ctx methods that hand a plugin (or receive from it) live objects no process boundary can carry:
# raw platform SDK clients, argparse parsers, adapter classes bound to the gateway's event loop, the
# launch-scope dashboard auth registry. A plugin calling one of these runs only in-process.
HOST_UNSUPPORTED_CTX_METHODS: dict[str, str] = {
    "register_login_backend": "browser-vault fills hold a server-local context manager",
    "register_browser_page_backend": "browser-vault fill scripts carry credentials that stay in the Hermes process",
    "register_platform": "platform adapters run on the gateway's event loop with native SDK clients",
    "register_platform_handler": "the handler factory receives the platform's native SDK client",
    "register_telegram_handler": "the handler factory receives the native Telegram Application",
    "register_slack_action_handler": "the callback receives native slack_bolt ack/body objects",
    "register_dashboard_auth_provider": "dashboard auth is owned by the launch process, not a profile",
    "register_approval_transport": "approval transports hold the live approval request/response channel",
    "post_message": "the returned handle is bound to the gateway's event loop and live adapter",
    "apost_message": "the returned handle is bound to the gateway's event loop and live adapter",
}

# Registered in-process only; skipped (with a warning, the plugin still loads) inside the host
# because the surface they extend does not exist in a server process.
HOST_SKIPPED_CTX_METHODS: dict[str, str] = {
    "register_cli_command": "`hermes <command>` subcommands are wired into the local CLI parser",
}

# Live ctx facades the host reaches by method call; each call returns plain data.
HOST_REMOTE_FACADES: frozenset[str] = frozenset({
    "state", "llm", "platform_actions", "subagent_lifecycle", "delegation_backends",
})

# Object-taking ctx methods and the ABC the parent checks. The host sends the object's method table
# and the parent builds a subclass of that ABC whose methods run in the host.
HOST_OBJECT_BASES: dict[str, str] = {
    "register_image_gen_provider": "hermes_agent.agent.image_gen_provider:ImageGenProvider",
    "register_video_gen_provider": "hermes_agent.agent.video_gen_provider:VideoGenProvider",
    "register_web_search_provider": "hermes_agent.agent.web_search_provider:WebSearchProvider",
    "register_browser_provider": "hermes_agent.agent.browser_provider:BrowserProvider",
    "register_terminal_environment_provider": "hermes_agent.agent.terminal_env_provider:TerminalEnvironmentProvider",
    "register_secret_source": "hermes_agent.agent.secret_sources.base:SecretSource",
    "register_tts_provider": "hermes_agent.agent.tts_provider:TTSProvider",
    "register_transcription_provider": "hermes_agent.agent.transcription_provider:TranscriptionProvider",
    "register_delegation_provider": "hermes_agent.agent.delegation_provider:DelegationProvider",
    "register_context_engine": "hermes_agent.agent.context_engine:ContextEngine",
    "register_context_reference": "hermes_agent.agent.context_references:ContextReferenceProvider",
    "register_memory_provider": "hermes_agent.agent.memory_provider:MemoryProvider",
}

# Hooks whose payload carries live handles (the gateway runner, the session store). In the host the
# callback receives placeholders for those fields, so a plugin that uses them needs in-process.
HOST_DEGRADED_HOOKS: dict[str, str] = {
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


def host_launcher(config: Optional[Mapping[str, Any]] = None) -> list[str]:
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


def import_user_code(path: Any, module_name: str) -> Optional[ModuleType]:
    """Import the Python file *path* as *module_name* into this process: shared import steps for
    loaders that execute a single file of plugin code in-process. ``None`` when *path* has no module
    spec.

    The module sits in ``sys.modules`` while its body runs (forward references under ``from
    __future__ import annotations`` resolve by module name) and is removed again if the body raises.
    It applies no ``plugins.isolation`` check; callers keep their own.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def user_plugin_host() -> Any:
    """The active profile's plugin host when ``plugins.isolation`` is ``host``, else ``None``.

    Category loaders (memory providers, context engines, cron schedulers) ask this before importing
    a user plugin directory, so the same plugins keep working with their code in the host.
    """
    if isolation_mode() != ISOLATION_HOST:
        return None
    from hermes_cli.plugins import get_plugin_manager
    return get_plugin_manager()._plugin_host()


@contextlib.contextmanager
def transient_plugin_host(before_stop: Optional[Any] = None) -> Iterator[None]:
    """Stop the active profile's plugin host on exit when it was started inside the block: a one-shot
    call for another profile (a profile lifecycle hook) must not leave a process behind. A host
    that was already running is left alone; a stopped one restarts on its next use.
    ``before_stop`` runs first, to drop registrations that would point at the stopped host.

    A request that starts using the same profile's host while the block runs (the dashboard
    serving that new profile at that moment) can lose its in-flight call; its next call restarts it."""
    if isolation_mode() != ISOLATION_HOST:
        yield
        return
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    was_running = manager._running_plugin_host() is not None
    try:
        yield
    finally:
        host = manager._running_plugin_host()
        if host is not None and not was_running:
            if before_stop is not None:
                before_stop()
            host.shutdown()
