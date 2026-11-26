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
import contextvars
import importlib.util
import logging
import os
import sys
from pathlib import Path
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
    "fork_session": "the returned handle tracks a fork thread of the live session agent in the Hermes process",
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
    "register_image_gen_provider": "agent.image_gen_provider:ImageGenProvider",
    "register_video_gen_provider": "agent.video_gen_provider:VideoGenProvider",
    "register_web_search_provider": "agent.web_search_provider:WebSearchProvider",
    "register_browser_provider": "agent.browser_provider:BrowserProvider",
    "register_terminal_environment_provider": "agent.terminal_env_provider:TerminalEnvironmentProvider",
    "register_secret_source": "agent.secret_sources.base:SecretSource",
    "register_tts_provider": "agent.tts_provider:TTSProvider",
    "register_transcription_provider": "agent.transcription_provider:TranscriptionProvider",
    "register_delegation_provider": "agent.delegation_provider:DelegationProvider",
    "register_context_engine": "agent.context_engine:ContextEngine",
    "register_context_reference": "agent.context_references:ContextReferenceProvider",
    "register_memory_provider": "agent.memory_provider:MemoryProvider",
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


def host_launcher(config: Optional[Mapping[str, Any]] = None, *, home: Any = None) -> list[str]:
    """``plugins.host.launcher``: argv prefix the host process runs under (a sandbox runner). Read from the
    config (no *config*), a launcher the managed overlay sets wins, even one that expands to nothing, its
    ``${VAR}``s expanded against the process env as ``apply_managed_overlay`` documents, whichever profile's
    secret scope the caller has bound. For a profile a plugin runs elsewhere (*home*, else this process's)
    only that one counts: that backend writes the profile's config.yaml."""
    from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal
    if config is None:
        from hermes_cli.managed_scope import apply_managed_overlay
        managed = contextvars.Context().run(apply_managed_overlay, {})  # no scope bound
        host = _plugins_config(managed).get("host")
        if (isinstance(host, Mapping) and "launcher" in host) or run_elsewhere_refusal(home) is not None:
            return _launcher_in(managed)
    return _launcher_in(config)


def _launcher_in(config: Optional[Mapping[str, Any]]) -> list[str]:
    host = _plugins_config(config).get("host")
    launcher = host.get("launcher") if isinstance(host, Mapping) else None
    if not launcher:
        return []
    if isinstance(launcher, str):
        import shlex
        return shlex.split(launcher)
    return [str(part) for part in launcher]


# The launcher a host start in progress chose, lent to a provider extraction that start runs (a cold one, while it
# builds the child's env, in the host's context, so for the same home) so it does not choose again.
_START_LAUNCHER: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "plugin_host_start_launcher", default=None)


def is_launch_home(home: Path) -> bool:
    """Is *home* the launch profile's? The one answer a host start uses for its launcher and its child's env:
    by the key the process remembered for its own path, as ``serves_routed_profile`` asks, not by where that
    path leads now. After that link is renamed *home* is the new name and the old one leads to no files; after
    it is repointed at another profile's directory, that profile is still another's."""
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    return hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home())


def home_moved_refusal(home: Path) -> Optional[str]:
    """Why plugin code must not start for *home* now, or None: its name leads to another directory than the one
    this process remembered it by (``hermes_home_key``), so ``is_launch_home`` and the launcher it picks would
    apply that directory's settings, and the remembered profile's values, to another directory's files."""
    from hermes_constants import hermes_home_key
    if os.path.normcase(os.path.realpath(home)) == hermes_home_key(home):
        return None
    return f"{home} leads to another directory than the one this process opened it as: restart Hermes"


def launcher_for(home: Path, own: bool) -> tuple[list, Optional[dict[str, str]]]:
    """*home*'s ``plugins.host.launcher``, chosen in a fresh context bound to *home*: a start can come from a
    caller bound to another profile, or from a crash restart's thread, which has no binding at all. With it, the
    env the launch profile's ``.env`` templates resolved against when they did not resolve against the live env
    (the frozen launch env, under multiplexing), for the host child's env to resolve that ``.env`` the same way;
    else None.

    A profile a plugin runs elsewhere expands its ``${VAR}``s against the process env, as ``host_launcher``
    documents for the managed overlay's launcher. A start inside a launch-profile body (a caller bound to the
    launch profile's own scope: unstamped, or stamped with a home keyed to it, as the dashboard binds it) expands
    the launch profile's launcher through the launch scope (``launch_secret_scope``: its ``.env`` over the launch
    env), as that body reads, under its task-local multiplexing too, so a name the launch env froze without stays
    unresolved; the TUI gateway and the dashboard bind it even single-profile. That scope is built again from
    *home* rather than borrowed: the body read its files through a name a link moved since can have led
    elsewhere. No caller's scope is ever lent. With no launch-profile body the launch profile outside
    multiplexing uses the process env, so a restart reads what the process loaded; under multiplexing it uses the
    launch scope too. The process env holds the launch profile's values, so any other profile's launcher expands
    through its own files (``build_profile_secret_scope``), multiplexing or not: a desktop backend's cron ticker
    serves a sibling under task-local multiplexing, which a fresh context does not carry. *own* is whether *home*
    is the launch profile's: ``is_launch_home``'s answer, made once per start for the host child's env too (where
    ``owner_secret_scope``, asking where the process's own name leads now, would take a renamed launch profile
    for another and hand a profile its link was repointed at the launch env). No terminal policy: the launcher
    is only read here."""
    from agent.secret_scope import (current_secret_scope, current_secret_scope_home, is_multiplex_active,
                                    set_multiplex_context)
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    launch_key = hermes_home_key(get_routing_process_hermes_home())
    stamp = current_secret_scope_home()
    launch_body = current_secret_scope() is not None and (stamp is None or hermes_home_key(stamp) == launch_key)
    multiplexed = is_multiplex_active()

    def choose() -> tuple[list, Optional[dict[str, str]]]:
        from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal
        if launch_body and multiplexed:
            set_multiplex_context(True)  # this context's own copy, dropped with it
        if run_elsewhere_refusal(home) is not None or (own and not launch_body and not is_multiplex_active()):
            from hermes_cli.plugins_loader import _plugin_home_scope
            with _plugin_home_scope(home):
                return host_launcher(home=home), None
        from agent.runtime_scope import bind_state, scope_for_profile
        from agent.secret_scope import build_profile_secret_scope
        from tui_gateway.launch_profile_policy import launch_secret_scope, launch_template_env
        # The launch profile's scope with its files read through *home*, the name that leads there now; any other
        # profile's from its own files, by ``own`` rather than by where the process's own name leads now (its
        # ``.env`` templates included).
        templates = launch_template_env() if own and is_multiplex_active() else None
        secrets = (launch_secret_scope(home, template_env=templates, process_home=True) if own
                   else build_profile_secret_scope(Path(home), process_home=False))
        with bind_state(scope_for_profile(home), secrets=secrets, terminal=False):
            return host_launcher(home=home), templates

    return contextvars.Context().run(choose)


def extraction_launcher(home: Path) -> list:
    """The launcher a provider-extraction child for *home* runs under: the one the host start for *home* in progress
    chose (``_START_LAUNCHER``) while *home* still runs here, else the one a host start for *home* would choose
    (``launcher_for``, with ``is_launch_home``'s answer), whichever context the extraction runs in. A profile
    marked to run elsewhere since its start chose so takes a marked profile's launcher. Only the launcher: that
    child's env carries no credentials."""
    from hermes_cli.profiles_run_elsewhere import run_elsewhere_refusal
    lent = _START_LAUNCHER.get()
    if lent is not None and run_elsewhere_refusal(home) is None:
        return list(lent)
    return launcher_for(home, is_launch_home(home))[0]


@contextlib.contextmanager
def lending_launcher(launcher: list) -> Iterator[None]:
    """Lend *launcher*, the one a host start chose, to a provider extraction that start runs meanwhile."""
    token = _START_LAUNCHER.set(launcher)
    try:
        yield
    finally:
        _START_LAUNCHER.reset(token)


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
    return requested_plugin_host()


def requested_plugin_host() -> Any:
    """The active profile's shared plugin host, for a caller that hands it plugin paths read through the active
    home's name (:func:`refuse_moved_home`)."""
    from hermes_cli.plugins import get_plugin_manager
    from hermes_constants import get_hermes_home
    refuse_moved_home(get_hermes_home())
    return get_plugin_manager()._plugin_host()


def refuse_moved_home(home: Path) -> None:
    """``PluginHomeMoved`` when *home*'s name leads to another directory than the one the process opened it as.
    Managers are found by the remembered key, so the active one is that directory's, possibly running under
    another name (a renamed profile's) with that profile's environment: plugin paths read through *home*'s name
    are another directory's plugin code."""
    from hermes_cli.plugins_loader import PluginHomeMoved
    if (moved := home_moved_refusal(Path(home).absolute())) is not None:
        raise PluginHomeMoved(moved)


@contextlib.contextmanager
def transient_plugin_host() -> Iterator[Any]:
    """A throwaway plugin manager for a one-shot load in the active profile (a profile lifecycle
    hook), or ``None`` when ``plugins.isolation`` is not ``host``. Its own scope key and host
    process keep it apart from the profile's shared manager, so the teardown on exit (dispose its
    registrations, stop its host) cannot touch a concurrent user of that profile's shared host."""
    if isolation_mode() != ISOLATION_HOST:
        yield None
        return
    from hermes_constants import get_hermes_home
    refuse_moved_home(get_hermes_home())  # its manager keys the remembered directory too
    import uuid

    from hermes_cli.plugins import PluginManager
    manager = PluginManager()
    # One scope per manager, so two overlapping hook loads never share registry slots; home_path stays
    # the profile's home.
    manager.scope_key = f"{manager.scope_key}#transient-{uuid.uuid4().hex}"
    try:
        yield manager
    finally:
        try:
            manager.unload()
        finally:
            # Any host this manager made, alive or not: one whose channel closed can still have a running
            # process, and shutdown() waits for or kills it.
            host = getattr(manager, "_plugin_host_instance", None)
            if host is not None:
                host.shutdown()
