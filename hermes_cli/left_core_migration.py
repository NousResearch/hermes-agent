"""Move a home onto the catalog plugin of a feature that left core.

A feature (gateway platform, toolset, bundled plugin) that moves from core into a standalone catalog
plugin keeps its names, config keys and env vars, so migrating is only "install the plugin". Each row of
:data:`LEFT_CORE` names the catalog plugin and a read-only predicate "does this home use it". The
contract is the one memory providers established (``memory_provider_migration``, whose helpers this
reuses; memory is not a row because its plugin name comes from ``memory.provider``, not a table):

* ``hermes update`` installs the plugin for every profile home sharing the venv that uses the feature.
* Agent start and gateway start retry once per process for the active home (Desktop users update
  through the app and never run ``hermes update``), honouring ``security.allow_lazy_installs``.
* After a failed attempt (catalog miss or unreachable, failed install) starts skip the row for
  an hour, doubling with each further failure up to a week, so an offline
  machine pays the network round trip once, not on every ``hermes chat``, and a home whose install
  cannot succeed stops failing every hour; ``hermes update`` always retries.
* A home gets the plugin automatically at most once (``_left_core_installed`` in its config.yaml):
  ``hermes plugins remove`` afterwards is the user's choice and sticks.

Installs go through the normal catalog install path at the reviewed pin (kill list, dependency
constraints, enable). Unattended dependency consent covers only these rows: the feature shipped in
core, so its users already accepted its dependencies. Agent/gateway start always installs with it
(nothing can answer a prompt there); ``hermes update`` in a terminal still asks. Every outcome
reaches the user (terminal, Desktop, chat); an outcome no terminal shows waits in the home for its next
turn someone sees (:func:`claim_kept_notices`).
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.memory_provider_migration import (
    _failed_recently, _home_consent, _home_label, _install_command, _interactive,
    _note_failure, _unattended_consent,
)

logger = logging.getLogger(__name__)

# ``terminal.*`` keys a left-core plugin still reads -> the ``.env`` copy core's ``hermes config set`` also
# wrote. Not in DEFAULT_CONFIG (a seeded default would hide the plugin's env fallback), so ``config
# set/get`` recognise them here, and ``config unset`` removes that copy too: the plugin falls back to it,
# and the old value would return instead of the default.
LEFT_CORE_TERMINAL_KEYS = {"terminal.singularity_image": "TERMINAL_SINGULARITY_IMAGE"}


def _read_config(home: Path) -> dict:
    from hermes_cli.config_backend import read_config_doc
    try:
        data = read_config_doc(home / "config.yaml")
    except Exception:  # missing/unreadable/invalid YAML: nothing we can see is in use
        return {}
    return data if isinstance(data, dict) else {}


def platform_configured_on(home: Path, name: str) -> bool:
    """Platform *name* is on in *home*'s config files as the gateway loader reads them: legacy
    ``gateway.json`` base layer, then config.yaml's ``gateway.platforms`` / ``platforms`` /
    ``gateway.<name>`` blocks merged in the loader's order (managed overlay and ``${VAR}`` applied),
    so the last explicit ``enabled`` wins. A stored token with no ``enabled`` anywhere counts too.
    Env credentials are the caller's to check. Read-only; works before the platform's plugin loads."""
    from gateway import config_loader
    from gateway.config import PlatformConfig
    gw_data = config_loader.load_legacy_gateway_json(home)
    gw_data = gw_data if isinstance(gw_data, dict) else {}
    try:
        yaml_cfg = config_loader.read_yaml_layers(home)
    except Exception:  # malformed config.yaml: the gateway falls back to gateway.json alone too
        yaml_cfg = {}
    yaml_cfg = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    block = config_loader.merge_platform_sections(
        yaml_cfg, yaml_cfg.get("gateway"), gw_data, also=frozenset({name})).get(name)
    if not isinstance(block, dict):
        return False
    platform = PlatformConfig.from_dict(block)
    return platform.enabled or ("enabled" not in block and bool(str(platform.token or "").strip()))


def _toolset_listed(config: dict, names: frozenset[str], *, own_platform: str = "") -> bool:
    """A toolset in *names* is selected in ``platform_toolsets`` or the top-level ``toolsets`` list.

    *own_platform*'s row is skipped: it scopes that platform's own sessions, which only exist once the
    platform is on (:func:`platform_configured_on`). The installers copy cli-config.yaml.example into
    config.yaml, and it carried ``homeassistant: [hermes-homeassistant]`` until Home Assistant left core,
    so that row alone is not use."""
    rows = config.get("platform_toolsets")
    selections = [sel for platform, sel in rows.items() if platform != own_platform] if isinstance(rows, dict) else []
    selections.append(config.get("toolsets"))
    return any(isinstance(sel, list) and any(str(item) in names for item in sel) for sel in selections)


def homeassistant_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core run Home Assistant for *home*: ``HASS_TOKEN`` in its ``.env`` (it enabled both
    the gateway platform and the tools), the platform on in its gateway config
    (:func:`platform_configured_on`), or the ``homeassistant`` / ``hermes-homeassistant`` toolset selected for
    another platform or in the top-level ``toolsets`` list. *process_env* (the active home at startup only)
    also counts a ``HASS_TOKEN`` the process received from its environment (systemd unit, Docker, shell export)."""
    from agent.secret_scope import load_env_file
    if (load_env_file(home / ".env").get("HASS_TOKEN") or "").strip():
        return True
    if process_env:
        try:
            from agent.secret_scope import get_secret
            if (get_secret("HASS_TOKEN", "") or "").strip():
                return True
        except Exception:
            pass
    if platform_configured_on(home, "homeassistant"):
        return True
    return _toolset_listed(
        _read_config(home), frozenset({"homeassistant", "hermes-homeassistant"}), own_platform="homeassistant")


def _auth_store(home: Path) -> dict:
    """*home*'s ``auth.json``; ``{}`` when missing, unreadable or invalid (no login we can see)."""
    import json
    try:
        store = json.loads((home / "auth.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return store if isinstance(store, dict) else {}


def spotify_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core's Spotify tools usable for *home*: a login stored in its ``auth.json``
    (``providers.spotify``, written by ``hermes auth spotify``) or the ``spotify`` toolset selected
    for a platform. The tools were opt-in and login-gated, so a client id alone is not use."""
    providers = _auth_store(home).get("providers")
    state = providers.get("spotify") if isinstance(providers, dict) else None
    if isinstance(state, dict) and (state.get("access_token") or state.get("refresh_token")):
        return True
    return _toolset_listed(_read_config(home), frozenset({"spotify"}))


def _plugin_lists(home: Path, plugin: str) -> tuple[bool, bool]:
    """Whether *plugin*'s key or name is in *home*'s ``plugins.enabled`` / ``plugins.disabled``, read with the
    managed overlay applied like the loader's config, so an administrator's pinned list counts."""
    from gateway import config_loader
    feature = next(f for f in LEFT_CORE if f.plugin == plugin)
    try:
        plugins = config_loader.read_yaml_layers(home).get("plugins")
    except Exception:  # malformed config.yaml: the loader enables nothing from it either
        logger.debug("left-core: unreadable config for %s", home, exc_info=True)
        return False, False
    plugins = plugins if isinstance(plugins, dict) else {}

    def listed(key: str) -> bool:
        names = plugins.get(key)
        return isinstance(names, list) and any(str(n) in (feature.plugin_key, feature.plugin) for n in names)
    return listed("enabled"), listed("disabled")


def _bundled_plugin_selected(home: Path, plugin: str) -> bool:
    """*plugin*'s bundled copy was on for *home*: in ``plugins.enabled`` and not in ``plugins.disabled``
    (which wins, as in the loader)."""
    enabled, disabled = _plugin_lists(home, plugin)
    return enabled and not disabled


def langfuse_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core trace to Langfuse for *home*: the bundled ``observability/langfuse`` selected
    (:func:`_bundled_plugin_selected`). The plugin was opt-in, so keys in ``.env`` alone are not use."""
    return _bundled_plugin_selected(home, "langfuse")


def singularity_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core run its Singularity backend for *home*: ``terminal.backend`` (or the legacy
    ``terminal.env_type``) set to ``singularity`` in its config (managed overlay applied, as the bridge
    reads it), else ``TERMINAL_ENV=singularity`` from its dotenv layers as ``load_hermes_dotenv`` resolves
    them (the home's ``.env``, then the administrator-managed ``.env`` with override, ``${VAR}``
    references expanded: :func:`~hermes_cli.env_loader.resolve_home_dotenv`). config.yaml wins over
    both, as in the bridge. *process_env* (startup) also counts a ``TERMINAL_ENV`` the process received
    from its environment (systemd unit, Docker, shell export), for the launch home only: ``TERMINAL_*``
    is process-global (``get_secret`` never scopes it), so for any other served home it is the launch
    profile's backend, not this one's."""
    from gateway import config_loader
    try:
        terminal = config_loader.read_yaml_layers(home).get("terminal")
    except Exception:  # malformed config.yaml: the bridge reads nothing from it either
        logger.debug("left-core: unreadable config for %s", home, exc_info=True)
        terminal = None
    terminal = terminal if isinstance(terminal, dict) else {}
    configured = str(terminal.get("backend") or terminal.get("env_type") or "").strip()
    if not configured:
        from hermes_cli.env_loader import resolve_home_dotenv
        configured = resolve_home_dotenv(home).get("TERMINAL_ENV", "").strip()
    if not configured and process_env:
        from hermes_constants import get_routing_process_hermes_home, hermes_home_key
        if hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home()):
            configured = os.environ.get("TERMINAL_ENV", "").strip()
    return configured.lower() == "singularity"


def agent_imports_in_use(home: Path, *, process_env: bool = False) -> bool:
    """*home* registered a ``hermes import-agent`` source for ``--sync`` (its ``import-sync.json``,
    which scheduled syncs read). A one-shot import leaves no other trace, and installing the plugin
    for every home would add a command nobody asked for; the old spelling names the install."""
    return (home / "import-sync.json").is_file()



def _home_env(home: Path, name: str, *, process_env: bool) -> str:
    """*name* as *home*'s dotenv layers resolve it (:func:`~hermes_cli.env_loader.resolve_home_dotenv`),
    else, with *process_env* and for the launch home only, the process environment."""
    from hermes_cli.env_loader import resolve_home_dotenv
    value = resolve_home_dotenv(home).get(name, "").strip()
    if not value and process_env:
        from hermes_constants import get_routing_process_hermes_home, hermes_home_key
        if hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home()):
            value = os.environ.get(name, "").strip()
    return value


def camofox_in_use(home: Path, *, process_env: bool = False) -> bool:
    """What made core run its Camofox backend for *home*, read the way core selected it
    (``tools/browser_camofox.is_camofox_mode``): a CDP endpoint (``browser.cdp_url`` or
    ``BROWSER_CDP_URL``) won over Camofox; a stored selection (``browser.cloud_provider``, or
    ``use_gateway`` = the managed row) decided alone, Camofox only when it names ``camofox``; with
    none stored, ``CAMOFOX_URL`` in its dotenv layers (the process env too for the launch home with
    *process_env*). ``browser.camofox`` settings never selected it: a leftover block without an
    address ran browser_exec. Config is the raw config.yaml, as core read it; read-only."""
    from utils import is_truthy_value
    browser = _read_config(home).get("browser")
    browser = browser if isinstance(browser, dict) else {}
    if str(browser.get("cdp_url") or "").strip() or _home_env(home, "BROWSER_CDP_URL", process_env=process_env):
        return False
    if is_truthy_value(browser.get("use_gateway")):
        return False
    provider = browser.get("cloud_provider")
    selected = str(provider).strip().lower() if provider is not None else ""
    if selected:
        return selected == "camofox"
    return bool(_home_env(home, "CAMOFOX_URL", process_env=process_env))


def _route_slots(home: Path) -> list[dict]:
    """Every inference route *home* configures, read with core's own loaders: config.yaml with the managed
    overlay (``read_yaml_layers``): ``model``, the fallback chain (``get_fallback_chain``: legacy
    ``fallback_model`` merged), each ``auxiliary.<task>``, ``delegation`` and its fallback chain, every
    delegation backend route (:func:`_delegation_backend_routes`) and its chain, every MoA preset's reference
    models and aggregator (``normalize_moa_config``), the cron-wide ``cron.model_provider``, each cron job's pin
    and the user's own ``providers:`` / ``custom_providers:`` endpoints (``get_compatible_custom_providers``)."""
    from cron.scheduler_preflight import cron_default_provider
    from gateway import config_loader
    from hermes_cli.config_providers import get_compatible_custom_providers
    from hermes_cli.fallback_config import get_fallback_chain
    from hermes_cli.moa_config import normalize_moa_config
    try:
        cfg = config_loader.read_yaml_layers(home)
    except Exception:  # malformed config.yaml: the agent reads nothing from it either
        logger.debug("left-core: unreadable config for %s", home, exc_info=True)
        cfg = {}
    cfg = cfg if isinstance(cfg, dict) else {}

    def block(value) -> dict:
        return value if isinstance(value, dict) else {}

    def with_chain(slot: dict) -> list[dict]:
        return [slot, *get_fallback_chain({"fallback_providers": slot.get("fallback_providers")})]
    delegation = block(cfg.get("delegation"))
    slots = [block(cfg.get("model")), *get_fallback_chain(cfg), *map(block, block(cfg.get("auxiliary")).values()),
             *with_chain(delegation), {"provider": cron_default_provider(cfg)}]
    for preset in normalize_moa_config(cfg.get("moa"))["presets"].values():
        slots += [*map(block, preset.get("reference_models") or []), block(preset.get("aggregator"))]
    for route in _delegation_backend_routes(delegation):
        slots += with_chain(route)
    slots += [{"base_url": entry.get("base_url")} for entry in get_compatible_custom_providers(cfg)]
    import json
    try:
        jobs = json.loads((home / "cron" / "jobs.json").read_text(encoding="utf-8-sig")).get("jobs")
    except (OSError, ValueError, AttributeError):  # no (readable) job store: no pins
        jobs = None
    return slots + ([block(job) for job in jobs] if isinstance(jobs, list) else [])


def _provider_choices(home: Path) -> set[str]:
    """Every provider *home* routes inference through (:func:`_route_slots`), plus the ``auth.json``
    ``active_provider`` the resolver's auto rung follows."""
    chosen = {str(slot.get("provider") or "").strip().lower() for slot in _route_slots(home)}
    chosen.add(str(_auth_store(home).get("active_provider") or "").strip().lower())
    return chosen - {""}


def routes_at_hosts(home: Path, hosts: tuple[str, ...]) -> bool:
    """A route *home* configures (:func:`_route_slots`) has its ``base_url`` at one of *hosts* (the host or
    a subdomain), whatever provider name it goes by (``custom``, a named endpoint)."""
    urls = {str(slot.get("base_url") or "").strip() for slot in _route_slots(home)} - {""} if hosts else set()
    return any(_at_host(url, host) for url in urls for host in hosts)


def _at_host(url: str, host: str) -> bool:
    """*url* is at *host* or a subdomain. A URL ``urlparse`` rejects (an unclosed IPv6 bracket) is at no host:
    that route fails on its own, and one bad slot must not abort the home's other migrations."""
    from utils import base_url_host_matches
    try:
        return base_url_host_matches(url, host)
    except ValueError:
        return False


def _delegation_backend_routes(delegation: dict) -> list[dict]:
    """The routes of every ``delegation.backends.<name>`` block: each entry of its ``options`` list (the
    ``model_map`` backend's titled provider/model routes, the shape ``web_routers/delegation.py`` writes).
    While a backend is active, these are the only routes delegated children take. The backends are
    plugins, so no core loader reads them; any backend using the ``options`` shape counts."""
    backends = delegation.get("backends")
    blocks = [b for b in backends.values() if isinstance(b, dict)] if isinstance(backends, dict) else []
    return [o for b in blocks if isinstance(b.get("options"), list) for o in b["options"] if isinstance(o, dict)]


def providers_selected(home: Path, names: tuple[str, ...], *, process_env: bool = False) -> set[str]:
    """The provider *names* (ids or aliases) *home* routes inference through (:func:`_provider_choices`),
    or names in its ``HERMES_INFERENCE_PROVIDER`` (dotenv layers; the process env too for the launch home
    with *process_env*). A credential alone is not a choice: core never routed to a provider for a key, and
    a name that is the user's own custom endpoint (``custom_providers: [{name: github, ...}]``) is not one."""
    chosen = _provider_choices(home) | {_home_env(home, "HERMES_INFERENCE_PROVIDER", process_env=process_env).lower()}
    chosen &= {name.lower() for name in names}
    return {name for name in chosen if not _names_custom_endpoint(home, name)} if chosen else chosen


def _names_custom_endpoint(home: Path, name: str) -> bool:
    """*name* resolves to one of *home*'s own ``providers:`` / ``custom_providers:`` endpoints (the
    resolver's named-custom rung: ``named_custom_provider_config_path``), not to the provider it aliases."""
    from gateway import config_loader
    from hermes_cli.runtime_provider_custom import named_custom_provider_config_path
    try:
        return named_custom_provider_config_path(name, config_loader.read_yaml_layers(home)) is not None
    except Exception:  # malformed config.yaml: the resolver finds no custom entry in it either
        logger.debug("left-core: unreadable config for %s", home, exc_info=True)
        return False


# GitHub Copilot: the API provider and the Copilot CLI over ACP, ids and every alias their profiles declared.
_COPILOT_API = ("copilot", "github-copilot", "github-models", "github-model", "github")
_COPILOT_ACP = ("copilot-acp", "github-copilot-acp", "copilot-acp-agent")


def copilot_in_use(home: Path, *, process_env: bool = False) -> bool:
    """Core routed inference through GitHub Copilot for *home*: ``copilot`` or ``copilot-acp`` (any alias)
    is a provider it chose (:func:`providers_selected`). A GH_TOKEN in ``.env`` alone is not use."""
    return bool(providers_selected(home, _COPILOT_API + _COPILOT_ACP, process_env=process_env))


def copilot_relogin_notice(home: Path, *, process_env: bool = False) -> str:
    """The Copilot API provider is in use in *home* with no token the plugin reads: no
    COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN in its dotenv layers (process env too for the launch home
    with *process_env*) and no Hermes login in its ``auth.json`` (a ``manual`` credential-pool row from
    ``hermes auth add copilot``, or ``providers.copilot``). Core also took ``gh auth token`` and the
    GitHub/Copilot CLI's token files, which the plugin does not; read-only, never runs ``gh``."""
    if not providers_selected(home, _COPILOT_API, process_env=process_env):
        return ""
    if any(_home_env(home, name, process_env=process_env)
           for name in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")):
        return ""
    store = _auth_store(home)
    pool = store.get("credential_pool")
    rows = pool.get("copilot") if isinstance(pool, dict) else None
    if isinstance(rows, list) and any(
            isinstance(row, dict) and str(row.get("source") or "").split(":", 1)[0] == "manual" for row in rows):
        return ""
    providers = store.get("providers")
    if isinstance(providers, dict) and isinstance(providers.get("copilot"), dict) and providers["copilot"]:
        return ""
    return ("This profile's Copilot login came from `gh auth token` or the GitHub/Copilot CLI, which the plugin "
            "no longer reads: run `hermes auth add copilot` to sign in.")


def _no_notice(home: Path, *, process_env: bool = False) -> str:
    return ""


@dataclass(frozen=True)
class LeftCoreFeature:
    plugin: str                               # catalog entry name == installed plugin name
    label: str                                # user-facing feature name
    in_use: Callable[..., bool]               # (home, *, process_env=False) -> bool, read-only
    unchanged: str                            # what migrated users keep, shown on success
    # Credentials core stripped from every child process while it shipped the feature. Core keeps
    # stripping them (tools/environments/local_env_policy.py): with the plugin absent (migration
    # pending, declined or failed) no manifest declares them, and they would reach every child.
    secret_env: tuple[str, ...] = ()
    # Non-secret settings core kept out of children by default (provider blocklist, Tier 2).
    private_env: tuple[str, ...] = ()
    # Toolsets the plugin registers. Core resolved them like any built-in toolset (on only where the
    # platform's saved list or default composite carried them); a plugin toolset is on for every
    # platform whose known_plugin_toolsets does not name it, so :func:`_record_migration` records
    # them there wherever core had them off.
    toolsets: tuple[str, ...] = ()
    # Platforms whose core default composite (``hermes-<platform>``) never carried those toolsets.
    off_platforms: tuple[str, ...] = ()
    # Core-era toolsets the plugin's toolsets replace while core keeps the name for something else
    # (core's ``browser`` is browser_exec now; Camofox's tools live in the plugin's ``camofox``). The
    # one-time scope conversion treats them as the plugin's toolsets for "did core carry it" and turns
    # them off in the home: a saved platform list naming one names the plugin's toolsets instead, and
    # ``agent.disabled_toolsets`` takes them for platforms on a default composite.
    replaces_toolsets: tuple[str, ...] = ()
    # Gateway platform the plugin ships: a running gateway serves it only after a restart.
    platform: str = ""
    # Channel ownership core declared for that platform, kept while the plugin is absent (migration
    # pending, declined, failed) so a channel-less profile clone still strips the source's identity
    # (hermes_cli/profile_channels.py): the env whose presence enabled the adapter, and the env
    # prefixes it shared with the feature's tools (stripped only when the source runs the adapter).
    enable_env: tuple[str, ...] = ()
    channel_env_prefixes: tuple[str, ...] = ()
    # Top-level command the plugin registers in place of core's ``hermes auth <cli>`` login
    # (``hermes <cli> login|status|logout``); :func:`moved_command_hint` points the old spelling at it.
    cli: str = ""
    # Top-level command the plugin registers: core's own command kept under its name (``hermes
    # import-agent``), or in place of core's nested ``old_command`` (``hermes mcp serve`` -> ``hermes
    # mcp-serve``); :func:`moved_command_hint` points the old spelling at it.
    command: str = ""
    old_command: str = ""
    # Other top-level commands core shipped that the same plugin now registers (``hermes claw``, the
    # OpenClaw migration's spelling before ``hermes import-agent openclaw``); hinted like *command*.
    old_cli: tuple[str, ...] = ()
    # Key of the bundled plugin core shipped (``observability/langfuse``). It stays an alias of the
    # catalog plugin in ``plugins.enabled`` / ``plugins.disabled`` (the loader's gate, an
    # administrator's pinned list) and for every `hermes plugins <verb>` (:func:`legacy_plugin_alias`).
    plugin_key: str = ""
    # ``terminal.backend`` value the plugin registers: unknown-backend errors name the install command.
    terminal_backend: str = ""
    # Config schema version that shipped without the feature, for a row whose conversion changes what a
    # home runs (``replaces_toolsets``): only a home from before it ran the core feature, so the row fires
    # only while the home's config is below it or when that version's config step found the feature in
    # use (:func:`record_core_use`). A newer home that sets the plugin up itself keeps core's toolsets.
    left_at: int = 0
    # Provider ids and aliases the plugin's model-provider profiles register. A name nothing resolves to
    # gets :func:`provider_install_hint` appended to its "unknown provider" error, runtime resolution
    # installs the plugin first for a home that uses it (:func:`recover_for_provider`), and those names
    # never resolve through another rung (the models.dev catalog) while the plugin is not loaded.
    providers: tuple[str, ...] = ()
    # Env names the plugin's providers read that core kept profile-managed: scrubbed when inherited but
    # absent from the profile's .env (env_loader._PROFILE_MANAGED_ENV_KEYS) and routed to .env by `hermes
    # config set` (config._EXTRA_ENV_KEYS). Read statically: env_loader runs before provider discovery.
    profile_env: tuple[str, ...] = ()
    # The subset of those names that choose an executable or its argv: the generic env writers refuse them
    # on every home (:func:`launch_env_names`), as core did. A loaded plugin's profile declares them itself;
    # a home whose plugin is absent, disabled, failed to load or quarantined would otherwise accept a value
    # that runs as the provider's binary once the plugin loads.
    launch_env: tuple[str, ...] = ()
    # Hosts the plugin's providers serve (a host covers its subdomains). Core attached their wire quirks by
    # host for any provider name, so a home with a route at one (a ``custom`` endpoint included) uses the
    # feature (:func:`feature_in_use`), and a request rejected there names the install
    # (:func:`host_install_hint`).
    hosts: tuple[str, ...] = ()
    # (home, *, process_env=False) -> str: one more sentence for every migration outcome in *home* (a
    # credential source core read that the plugin does not, and the command that replaces it), "" = none.
    notice: Callable[..., str] = _no_notice


LEFT_CORE: tuple[LeftCoreFeature, ...] = (
    LeftCoreFeature(
        plugin="homeassistant", label="Home Assistant", in_use=homeassistant_in_use,
        unchanged="HASS_TOKEN/HASS_URL, platforms.homeassistant and the ha_* tool names are unchanged",
        secret_env=("HASS_TOKEN",), private_env=("HASS_URL",),
        toolsets=("homeassistant",), off_platforms=("acp", "webhook"), platform="homeassistant",
        enable_env=("HASS_TOKEN",), channel_env_prefixes=("HASS_",),
    ),
    LeftCoreFeature(
        plugin="spotify", label="Spotify", in_use=spotify_in_use,
        unchanged="your Spotify login, the spotify toolset and the spotify_* tools are unchanged; "
                  "`hermes auth spotify` is now `hermes spotify login`",
        cli="spotify",
    ),
    LeftCoreFeature(
        plugin="langfuse", label="Langfuse", in_use=langfuse_in_use,
        unchanged="your Langfuse keys and HERMES_LANGFUSE_* settings are unchanged; the plugin is now "
                  "`langfuse` (was observability/langfuse)",
        secret_env=("HERMES_LANGFUSE_SECRET_KEY",),
        private_env=("HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_BASE_URL"),
        plugin_key="observability/langfuse",
    ),
    LeftCoreFeature(
        # MCP clients that launch `hermes mcp serve` are configured outside Hermes, so no home shows the
        # feature in use: nothing installs automatically, and the old spelling prints where it went.
        plugin="mcp-serve", label="Hermes as an MCP server", in_use=lambda home, **_: False,
        unchanged="the conversation tools keep their names; MCP clients change their args to [\"mcp-serve\"]",
        command="mcp-serve", old_command="hermes mcp serve",
    ),
    LeftCoreFeature(
        # CAMOFOX_URL is an address, not a credential: core kept it out of children only through its
        # `category: tool` OPTIONAL_ENV_VARS entry (the Tier 2 provider blocklist), as HASS_URL. The
        # bearer token is a credential.
        plugin="camofox", label="Camofox browser backend", in_use=camofox_in_use,
        unchanged="CAMOFOX_URL, the browser.camofox settings and the browser_navigate/browser_click/... tool "
                  "names are unchanged; they are the `camofox` toolset now, and core's `browser` toolset "
                  "(browser_exec) is turned off in this profile",
        secret_env=("CAMOFOX_API_KEY",), private_env=("CAMOFOX_URL",),
        toolsets=("camofox",), off_platforms=("webhook",), replaces_toolsets=("browser",), left_at=52,
    ),
    LeftCoreFeature(
        plugin="singularity", label="Singularity/Apptainer terminal backend", in_use=singularity_in_use,
        unchanged="terminal.backend: singularity, terminal.singularity_image (TERMINAL_SINGULARITY_IMAGE) "
                  "and the container_* limits are unchanged",
        terminal_backend="singularity",
    ),
    LeftCoreFeature(
        plugin="agent-importers", label="`hermes import-agent`", in_use=agent_imports_in_use,
        unchanged="`hermes import-agent`, its flags and your import-sync.json sync sources are unchanged",
        command="import-agent", old_cli=("claw",),
    ),
    LeftCoreFeature(
        # Core kept COPILOT_GITHUB_TOKEN and the two base-URL overrides out of children as provider env
        # (Tier 2: credentialed CLIs still get them); GH_TOKEN/GITHUB_TOKEN stay on core's static lists.
        plugin="github-copilot", label="GitHub Copilot", in_use=copilot_in_use,
        unchanged="your COPILOT_GITHUB_TOKEN / GH_TOKEN / GITHUB_TOKEN and `hermes auth add copilot` logins, model "
                  "names and copilot-acp settings are unchanged",
        private_env=("COPILOT_GITHUB_TOKEN", "COPILOT_API_BASE_URL", "COPILOT_ACP_BASE_URL"),
        providers=_COPILOT_API + _COPILOT_ACP, hosts=("githubcopilot.com", "models.github.ai"),
        profile_env=("HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH",
                     "COPILOT_ACP_BASE_URL"),
        launch_env=("HERMES_COPILOT_ACP_COMMAND", "HERMES_COPILOT_ACP_ARGS", "COPILOT_CLI_PATH"),
        notice=copilot_relogin_notice,
    ),
)

def _install_sentence(feature: LeftCoreFeature) -> str:
    from hermes_constants import get_hermes_home
    return (f"{feature.label} moved out of core into the '{feature.plugin}' plugin: install it with "
            f"`{_install_command(feature.plugin, Path(get_hermes_home()))}`")


def platform_install_hint(platform: str) -> str:
    """For an unknown platform name that left core: the sentence that names its install command for
    the active home, else ``""``. Appended to "unknown platform" errors (``send_message``)."""
    feature = next((f for f in LEFT_CORE if f.platform and f.platform == platform), None)
    return f". {_install_sentence(feature)}" if feature else ""


def terminal_backend_install_hint(backend: str) -> str:
    """For a ``terminal.backend`` value that left core: the sentence that names its install command for
    the active home, else ``""``. Appended to "unknown terminal backend" errors."""
    feature = next((f for f in LEFT_CORE if f.terminal_backend and f.terminal_backend == backend), None)
    return f". {_install_sentence(feature)}" if feature else ""


def _provider_feature(name: str) -> Optional[LeftCoreFeature]:
    key = str(name or "").strip().lower()
    return next((f for f in LEFT_CORE if key and key in f.providers), None)


def provider_install_hint(name: str) -> str:
    """For a provider name nothing resolves to that a left-core plugin supplies: the sentence naming its
    install command for the active home (its enable command when installed but disabled; its load error and
    update command when installed but it failed to load), else ``""``.
    Appended to every "unknown provider" error (runtime resolution, ``/model``, ``hermes model``, ``hermes
    auth add``)."""
    feature = _provider_feature(name)
    if feature is None:
        return ""
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    if not plugin_present(feature.plugin, home):
        return f"{_install_sentence(feature)}."
    command = _install_command(feature.plugin, home)
    if _plugin_lists(home, feature.plugin)[1]:
        return (f"{feature.label} moved out of core into the '{feature.plugin}' plugin, which is installed but "
                f"disabled: run `{command.replace(' install ', ' enable ')}`.")
    # A model-provider plugin loads without plugins.enabled: present, not disabled and no profile = its load failed.
    error = _load_error(feature)
    return (f"{feature.label} moved out of core into the '{feature.plugin}' plugin, which is installed but failed "
            f"to load{f' ({error})' if error else ''}: run `{command.replace(' install ', ' update ')}`, or "
            f"see `hermes logs` for the cause.")


def _load_error(feature: LeftCoreFeature) -> str:
    """The error *feature*'s provider plugin raised on import in the active home, else ``""``."""
    from providers import plugin_load_error
    return plugin_load_error(feature.plugin)


def provider_left_core(name: str) -> bool:
    """*name* is a provider a left-core plugin supplies and no registered profile serves it now: no other
    resolution rung (the models.dev catalog) may stand in for it (it would send the user's token to the
    vendor's endpoint through a generic client)."""
    if _provider_feature(name) is None:
        return False
    from providers import get_provider_profile
    return get_provider_profile(str(name).strip().lower()) is None


def feature_in_use(feature: LeftCoreFeature, home: Path, *, process_env: bool = False) -> bool:
    """*home* uses *feature*: its :attr:`~LeftCoreFeature.in_use` predicate, or a configured route at one of
    its :attr:`~LeftCoreFeature.hosts` (:func:`routes_at_hosts`)."""
    return feature.in_use(home, process_env=process_env) or bool(feature.hosts and routes_at_hosts(home, feature.hosts))


def host_install_hint(base_url: str) -> str:
    """For a request rejected by an endpoint at a host a left-core plugin serves while none of that plugin's
    providers is registered (core attached the host's wire quirks itself): the install (or enable) sentence
    for the active home, else ``""``. Printed with non-retryable request errors."""
    feature = next((f for f in LEFT_CORE if any(_at_host(str(base_url or ""), h) for h in f.hosts)), None)
    if feature is None or not all(provider_left_core(name) for name in feature.providers):
        return ""
    return provider_install_hint(feature.providers[0])


def recover_for_provider(name: Optional[str]) -> None:
    """Runtime resolution of *name*, a provider a left-core plugin supplies, in a home that uses it and lacks
    the plugin: run :func:`recover_at_startup` now. The CLI and Desktop resolve the provider before the agent
    (whose start runs the migration) exists, so they would otherwise stop at "unknown provider" on a home no
    ``hermes update`` migrated. Outcomes print in a declared terminal front end (:func:`serve_notices_on_stderr`),
    else wait for the home's next turn someone sees."""
    from hermes_constants import get_hermes_home, hermes_home_key
    from providers import get_provider_profile
    home = Path(get_hermes_home())
    name = str(name or "").strip().lower()
    if name in ("", "auto"):  # the auto rung follows auth.json's active_provider (a core-era choice)
        name = str(_auth_store(home).get("active_provider") or "").strip().lower()
    feature = _provider_feature(name)
    if feature is None or _installs_held.get():  # held: a read-only diagnostic consumes no kept notice either
        return
    # A terminal front end shows outcomes now, also those another process (the gateway this CLI started) kept;
    # otherwise (Desktop, gateway, a detached turn) the home's next watched turn in any process gets them. Never
    # inferred from a TTY: a foreground `hermes gateway run` has one, and its resolutions serve the chat user.
    say = (lambda message: print(message, file=sys.stderr)) if _terminal_front_end else None
    if say is not None:
        for message in _claim_notices(home):
            say(message)
    if hermes_home_key(home) in _attempted:  # this process already tried: skip the config/auth/.env reads
        return
    if plugin_present(feature.plugin, home) or get_provider_profile(name) is not None:
        return
    if feature_in_use(feature, home, process_env=True):
        recover_at_startup(say=say)


_terminal_front_end = False


def serve_notices_on_stderr() -> None:
    """This process is a terminal front end (the classic CLI, whose turns run in the gateway): outcomes of
    :func:`recover_for_provider` print to its stderr, also when its output is piped (``hermes chat -q``),
    since a resolution that fails before any agent exists (no credentials) leaves no agent to deliver them.
    Only a declaration counts: a foreground gateway has a TTY too, and its outcomes belong to the chat user."""
    global _terminal_front_end
    _terminal_front_end = True


def profile_env_names() -> frozenset[str]:
    """Every row's :attr:`LeftCoreFeature.profile_env` (static: no discovery)."""
    return frozenset(name for feature in LEFT_CORE for name in feature.profile_env)


def launch_env_names() -> frozenset[str]:
    """Every row's :attr:`LeftCoreFeature.launch_env` (static): the generic env writers refuse them."""
    return frozenset(name for feature in LEFT_CORE for name in feature.launch_env)


def legacy_plugin_alias(name: str) -> str:
    """The catalog plugin a bundled key that left core now names (``observability/langfuse`` ->
    ``langfuse``), else ``""``."""
    return next((f.plugin for f in LEFT_CORE if f.plugin_key and f.plugin_key == name), "")


def legacy_plugin_keys(plugin: str) -> set[str]:
    """Bundled keys that now name catalog *plugin* (inverse of :func:`legacy_plugin_alias`)."""
    return {f.plugin_key for f in LEFT_CORE if f.plugin_key and f.plugin == plugin}


def moved_command_hint(prog: str, value: str, action: str = "") -> str:
    """For a command a left-core plugin took over: ``hermes auth [<action>] <cli>`` (core's old login
    spelling), core's old nested spelling (``hermes mcp serve``), or the plugin's own ``hermes <cli>`` /
    ``hermes <command>`` / ``hermes <old_cli>`` while the plugin is not loaded, the line saying where it
    went and how to get it for the active home, else ``""``."""
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    moved = next((f for f in LEFT_CORE if f.old_command and f.old_command == f"{prog} {value}"), None)
    if moved is not None:
        present = plugin_present(moved.plugin, home)
        get = "" if present else f" Install it with: `{_install_command(moved.plugin, home)}`."
        return (f"{moved.label} moved out of core into the '{moved.plugin}' plugin: `{moved.old_command}` is now "
                f"`hermes {moved.command}`.{get}")
    if prog == "hermes auth":
        feature = next((f for f in LEFT_CORE if f.cli and f.cli == value), None)
    elif prog == "hermes":
        feature = next((f for f in LEFT_CORE if value and value in (f.cli, f.command, *f.old_cli)), None)
    else:
        feature = None
    if feature is None:
        return ""
    install = _install_command(feature.plugin, home)
    if not plugin_present(feature.plugin, home):
        get = f" Install it with: `{install}`."
    else:
        get = f" Enable it with: `{install.replace(' install ', ' enable ')}`." if prog == "hermes" else ""
    if prog == "hermes auth":
        old = " ".join(p for p in ("hermes auth", action, value) if p)
        return (f"{feature.label} moved out of core into the '{feature.plugin}' plugin: `{old}` is now "
                f"`hermes {value} {action or 'login'}`.{get}")
    label = f"`hermes {value}`" if value in feature.old_cli else feature.label
    return f"{label} moved out of core into the '{feature.plugin}' plugin, which is not loaded.{get}"


def moved_plugin_hint(name: str) -> str:
    """For a plugin name nothing resolves to (so the plugin is not installed): the bundled key of a
    plugin that left core (``observability/langfuse``) or its catalog name, the line naming the install
    command for the active home, else ``""``. Appended to "No plugin named ..." errors."""
    feature = next((f for f in LEFT_CORE if f.plugin_key and name in (f.plugin_key, f.plugin)), None)
    return f"{_install_sentence(feature)}." if feature else ""


_attempted: set[str] = set()
_installs_held: ContextVar[bool] = ContextVar("left_core_installs_held", default=False)


@contextmanager
def installs_held():
    """No migration attempt inside (a read-only diagnostic such as ``hermes doctor``, whose provider checks run
    the runtime resolver): the unknown-provider and host hints still name the install command."""
    token = _installs_held.set(True)
    try:
        yield
    finally:
        _installs_held.reset(token)


def defer_startup_notice(message: str) -> None:
    """Hand *message* to the active home's next turn someone sees (:func:`claim_kept_notices`), once.
    For boot-path work that runs before any agent exists."""
    from hermes_constants import get_hermes_home
    logger.warning(message)
    _queue_notice(Path(get_hermes_home()), message)


def _notice_file(home: Path) -> Path:
    return home / "cache" / "left-core-notices.jsonl"


@contextmanager
def _notices_locked(home: Path):
    """The cross-process lock every append and claim of *home*'s kept notices holds: an append never lands
    in a file a claimant already read, and a claimant never reads a half-written line."""
    import threading
    from hermes_cli.auth import _file_lock
    path = _notice_file(home)
    # A fresh holder: the helper's per-thread reentrancy must never skip one home's lock inside another's.
    with _file_lock(path.with_name(f"{path.name}.lock"), threading.local(), 5.0,
                    f"Timed out waiting for the left-core notice lock ({path})"):
        yield path


def _queue_notice(home: Path, message: str) -> None:
    """Keep *message* for *home*'s next watched turn in ANY process (:func:`_claim_notices`): the process that
    produced it (a CLI resolving the provider, a gateway start) often runs no agent for that home itself.
    A line already kept is not kept again: every start with ``allow_lazy_installs`` off reports the same row."""
    import json
    line = json.dumps(message)
    try:
        with _notices_locked(home) as path:
            if path.exists() and line in path.read_text(encoding="utf-8-sig").splitlines():
                return
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError as exc:  # unwritable home or lock timeout: the outcome is still logged
        logger.debug("left-core notice not kept for %s: %s", home, exc)


def _claim_notices(home: Path) -> list[str]:
    """Take every notice kept for *home*, exactly once across processes: read and removed under the notice
    lock, so of several agents starting at once only one gets (and delivers) them. A crash before the
    removal leaves them for the next claimant rather than losing them."""
    import json
    if not _notice_file(home).exists():  # the common case: no lock taken
        return []
    try:
        with _notices_locked(home) as path:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            path.unlink()
    except OSError:  # another claimant took them first, or the lock timed out (they stay kept)
        return []
    messages = []
    for line in dict.fromkeys(lines):
        try:
            messages.append(str(json.loads(line)))
        except ValueError:  # a line from outside Hermes: skip it only
            continue
    return messages


def claim_kept_notices() -> list[str]:
    """Every notice kept for the active home in any process, for a turn someone sees (exactly once)."""
    from hermes_constants import get_hermes_home
    return _claim_notices(Path(get_hermes_home()))


def plugin_present(plugin: str, home: Path) -> bool:
    """Installed in *home* (enabled or not: a user who disabled it chose to). Read-only."""
    return (home / "plugins" / plugin).is_dir()


def _core_carried(feature: LeftCoreFeature, selection: list) -> bool:
    """Whether core resolved *feature*'s toolsets on for a platform selecting *selection*: named
    directly, or through a ``hermes`` / ``hermes-<platform>`` composite that included them."""
    def carries(name: str) -> bool:
        if name in feature.toolsets or name in feature.replaces_toolsets or name == "hermes":
            return True
        return name.startswith("hermes-") and name[len("hermes-"):].replace("-", "_") not in feature.off_platforms
    return any(carries(str(name)) for name in selection)


_SCOPED_KEY = "_left_core_scoped"  # config.yaml: rows whose toolset scope this home already converted
# config.yaml: rows this home has had installed (by the migration, or by the user before it ran).
# Never written before the plugin dir exists, so a home whose install never succeeded keeps retrying.
_INSTALLED_KEY = "_left_core_installed"


# config.yaml: rows whose ``left_at`` config step found the core feature in use in this home.
_USED_KEY = "_left_core_used"


def record_core_use(home: Path, config: dict, version: int) -> bool:
    """Config step *version* hook, before it writes *config* (*home*'s raw config.yaml, still showing the
    core-era setup): mark every row that left core at *version* and that *home* uses. Returns whether
    *config* changed. The launch home's process env counts, as at agent start."""
    used = [feature for feature in LEFT_CORE if feature.left_at == version
            and not _marked(config, _USED_KEY, feature) and feature_in_use(feature, home, process_env=True)]
    for feature in used:
        _mark(config, _USED_KEY, feature)
    return bool(used)


def _predates_split(home: Path, feature: LeftCoreFeature) -> bool:
    """*home* ran *feature* in core: its config is still below ``left_at`` (this pass ran before the
    config step), that step recorded the feature in use, or config.yaml is missing or unstamped. Every
    config.yaml Hermes writes carries a stamp, and the first save that stamps one records the rows in use
    (``config_version_stamp.note_first_stamp``), so only a home from before the split lacks it; core ran
    the feature there from its .env alone."""
    if not feature.left_at:
        return True
    config = _read_config(home)
    if _marked(config, _USED_KEY, feature) or "_config_version" not in config:
        return True
    try:
        return int(config["_config_version"]) < feature.left_at
    except (TypeError, ValueError):
        return False


def _marked(config: dict, key: str, feature: LeftCoreFeature) -> bool:
    done = config.get(key)
    return isinstance(done, list) and feature.plugin in map(str, done)


def _scope_kept(config: dict, feature: LeftCoreFeature) -> bool:
    return not feature.toolsets or _marked(config, _SCOPED_KEY, feature)


def _mark(config: dict, key: str, feature: LeftCoreFeature) -> None:
    done = config.get(key)
    config[key] = sorted({*(map(str, done) if isinstance(done, list) else ()), feature.plugin})


def _turn_off_replaced(config: dict, feature: LeftCoreFeature) -> None:
    """Turn *feature*'s ``replaces_toolsets`` off in *config*: every saved platform list naming one names
    the plugin's toolsets instead (so turning core's toolset back on later stays that platform's choice),
    and ``agent.disabled_toolsets`` takes them, which also covers platforms on a default composite. One the
    user had already disabled there disables the plugin's toolsets too (they are on by default)."""
    from agent.skill_utils import parse_config_string_list
    from hermes_cli.toolset_validation import parse_platform_toolsets_value
    replaced = set(feature.replaces_toolsets)
    saved = config.get("platform_toolsets")
    for platform, value in (saved.items() if isinstance(saved, dict) else ()):
        selection = parse_platform_toolsets_value(value)
        if selection is not None and replaced & set(map(str, selection)):
            kept = [item for item in selection if str(item) not in replaced]
            saved[platform] = kept + [ts for ts in feature.toolsets if ts not in map(str, kept)]
    agent = config.get("agent") if isinstance(config.get("agent"), dict) else {}
    disabled = parse_config_string_list(agent.get("disabled_toolsets")) if agent.get("disabled_toolsets") else []
    turned_off = replaced if replaced.isdisjoint(disabled) else replaced | set(feature.toolsets)
    agent["disabled_toolsets"] = [*disabled, *sorted(turned_off - set(disabled))]
    config["agent"] = agent


def _record_migration(home: Path, feature: LeftCoreFeature) -> None:
    """Keep *feature*'s core-era toolset scope, and mark it installed once its plugin is in *home*,
    in one config write.

    Scope: record the toolsets in ``known_plugin_toolsets[platform]`` (= off) for every platform
    where core had them off: a saved ``platform_toolsets`` list that does not carry them, or no list
    on an ``off_platforms`` platform. Platforms core resolved them on for are left alone, except that
    a row's ``replaces_toolsets`` go off everywhere (:func:`_turn_off_replaced`). A one-time
    conversion per home and row, independent of installation: a plugin installed before the update
    (it stays inert on a core that still ships the feature) still needs it. Completion is marked
    under ``_left_core_scoped`` so a later run never undoes a choice the user made since
    (``hermes tools``). Raises when config.yaml cannot be read or written."""
    from hermes_cli.config import atomic_config_write, read_user_config_raw
    from hermes_cli.config_backend import config_exists
    from hermes_cli.toolset_validation import parse_platform_toolsets_value
    path = home / "config.yaml"
    created = not config_exists(path)
    config = read_user_config_raw(path)
    installed = plugin_present(feature.plugin, home) and not _marked(config, _INSTALLED_KEY, feature)
    if _scope_kept(config, feature) and not installed:
        return
    if installed:
        _mark(config, _INSTALLED_KEY, feature)
    if not _scope_kept(config, feature):
        saved = config.get("platform_toolsets")
        saved = saved if isinstance(saved, dict) else {}
        selections = {str(p): parse_platform_toolsets_value(v) for p, v in saved.items()}
        for platform in feature.off_platforms:
            selections.setdefault(platform, None)
        known = config.get("known_plugin_toolsets")
        known = known if isinstance(known, dict) else {}
        for platform, selection in selections.items():
            if selection is None and platform not in feature.off_platforms:
                continue  # no (valid) saved list: core's default composite carried the toolsets
            if selection is not None and _core_carried(feature, selection):
                continue
            current = known.get(platform) if isinstance(known.get(platform), list) else []
            if missing := [ts for ts in feature.toolsets if ts not in current]:
                known[platform] = sorted({*map(str, current), *missing})
        if known:
            config["known_plugin_toolsets"] = known
        if feature.replaces_toolsets:
            _turn_off_replaced(config, feature)
        _mark(config, _SCOPED_KEY, feature)
    if created:
        _stamp_created(home, config)
    atomic_config_write(path, config)


def _stamp_created(home: Path, config: dict) -> None:
    """A missing config.yaml reads as the current schema, so no config step ever ran on this home: the file
    the conversion creates carries that stamp (unstamped, every boot warns it predates the support floor)
    and, as a plain save's first stamp does, the rows in use, or the stamp makes the home look post-split."""
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    from hermes_cli.config_version_stamp import note_first_stamp
    config["_config_version"] = DEFAULT_CONFIG["_config_version"]
    note_first_stamp(home, {}, config)


def _pending(home: Path, *, say: Callable[[str], None], process_env: bool = False,
             backoff: bool = False) -> list[LeftCoreFeature]:
    """Rows *home* uses whose plugin it never had and that the catalog ships (a catalog miss is
    reported through *say*). Converts the toolset scope of every row *home* uses once
    (:func:`_record_migration`), installed or not; a row whose scope cannot be recorded is
    reported and skipped, never installed unscoped. A row marked installed is done for good.
    *backoff* (startup) skips a row whose last attempt failed recently, before any network."""
    from hermes_cli.memory_provider_migration import catalog_source
    out = []
    for feature in LEFT_CORE:
        if _marked(_read_config(home), _INSTALLED_KEY, feature):
            continue
        if not feature_in_use(feature, home, process_env=process_env):
            continue
        if not _predates_split(home, feature):
            _say_post_split(home, feature, say=say, backoff=backoff, process_env=process_env)
            continue
        present = plugin_present(feature.plugin, home)
        try:
            _record_migration(home, feature)
        except Exception as exc:
            say(f"  ⚠ {feature.label} moved out of core into the '{feature.plugin}' plugin: its per-platform "
                f"toolset selection could not be kept ({exc}). Check `hermes tools`"
                + ("." if present else f" and run `{_install_command(feature.plugin, home)}`."))
            continue
        if present:
            continue
        if backoff and _failed_recently(home, feature.plugin):
            logger.info("%s plugin install failed recently for %s; retrying later or on `hermes update`",
                        feature.label, home)
            continue
        if catalog_source(feature.plugin) is None:
            _note_failure(home, feature.plugin)
            say(_noticed(f"  ⚠ {feature.label} moved out of core into the '{feature.plugin}' plugin, which this "
                         f"Hermes cannot find in the plugin catalog yet. Run "
                         f"`{_install_command(feature.plugin, home)}` once it is listed.", home, feature, process_env))
            continue
        out.append(feature)
    return out


def _say_post_split(home: Path, feature: LeftCoreFeature, *, say: Callable[[str], None], backoff: bool,
                    process_env: bool = False) -> None:
    """*home* uses *feature* while its stamp says it never ran it in core (a newer home, or a Nix-managed
    config.yaml stamped at activation): nothing is converted or installed, but a home without the plugin is
    told where the feature went. Starts repeat it on the install backoff (``_note_failure``)."""
    if plugin_present(feature.plugin, home) or (backoff and _failed_recently(home, feature.plugin)):
        return
    _note_failure(home, feature.plugin)
    say(_noticed(f"  ⚠ {feature.label} moved out of core into the '{feature.plugin}' plugin, which this profile "
                 f"does not have, so it is off. Run `{_install_command(feature.plugin, home)}`.",
                 home, feature, process_env))


def _noticed(message: str, home: Path, feature: LeftCoreFeature, process_env: bool) -> str:
    """*message* with *feature*'s :attr:`~LeftCoreFeature.notice` for *home* appended."""
    extra = feature.notice(home, process_env=process_env)
    return f"{message} {extra}" if extra else message


def _install_into(home: Path, *, consent: Optional[bool] = None) -> Callable[[str], dict]:
    def _install(name: str) -> dict:
        from hermes_cli.plugins_cmd import dashboard_install_plugin
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(home)
        try:
            return dashboard_install_plugin("", force=False, enable=True, catalog_name=name,
                                            assume_deps_consent=_unattended_consent() if consent is None
                                            else consent)
        finally:
            reset_hermes_home_override(token)
    return _install


def _first_cause(error: str) -> str:
    """One line for a notice: git's own ``fatal:`` / ``error:`` line when *error* ends with raw git
    stderr (a failed clone), else its first non-empty line."""
    lines = [line.strip() for line in error.splitlines() if line.strip()]
    cause = next((line.split(":", 1)[1].strip() for line in lines if line.startswith(("fatal:", "error:"))), None)
    return (cause or (lines[0] if lines else "unknown error")).rstrip(". ")


def _install_one(home: Path, feature: LeftCoreFeature, *, install: Callable[[str], dict],
                 say: Callable[[str], None], process_env: bool = False) -> bool:
    try:
        result = install(feature.plugin)
    except Exception as exc:  # network, uv, kill list — report, do not raise
        result = {"ok": False, "error": str(exc)}
    if result.get("ok"):
        from providers import refresh_bound_home
        refresh_bound_home()  # a provider plugin resolves in this same call, not after the stamp TTL
        try:
            _record_migration(home, feature)
        except Exception as exc:  # the next run sees the plugin dir and records it then
            logger.debug("left-core install marker not written for %s: %s", home, exc)
        say(_noticed(f"  ✓ {feature.label} moved out of core — installed the '{feature.plugin}' plugin from the "
                     f"catalog ({feature.unchanged}).", home, feature, process_env))
        return True
    _note_failure(home, feature.plugin)
    error = _first_cause(str(result.get("error") or ""))
    say(_noticed(f"  ⚠ {feature.label} moved out of core and its '{feature.plugin}' plugin could not be installed "
                 f"automatically: {error}. Run `{_install_command(feature.plugin, home)}`.", home, feature, process_env))
    return False


def migrate_home(home: Path, *, install: Callable[[str], dict], say: Callable[[str], None] = print,
                 process_env: bool = False) -> list[str]:
    """Install every left-core plugin *home* uses and lacks. Returns installed plugin names; never raises."""
    installed = []
    for feature in _pending(home, say=say, process_env=process_env):
        if _install_one(home, feature, install=install, say=say, process_env=process_env):
            installed.append(feature.plugin)
    return installed


def migrate_all_homes(*, say: Callable[[str], None] = print) -> list[str]:
    """``hermes update`` hook: every profile home sharing this venv. Grouped by plugin and each home's
    unattended consent like the memory migration: homes of one group share dependency answers, and a
    failure names the rest of its group in one line instead of failing them one by one."""
    from hermes_cli.plugins_cmd_install import shared_dependency_answers
    from hermes_constants import get_routing_process_hermes_home, hermes_home_key
    from pm.plugins_state import dependency_homes

    def labelled(home: Path) -> Callable[[str], None]:
        return lambda message: say(f"  [{_home_label(home)}] {message.lstrip()}")

    def launched(home: Path) -> bool:  # this process's env (a shell export) is the launch home's, as at start
        return hermes_home_key(home) == hermes_home_key(get_routing_process_hermes_home())

    pending: dict[tuple[str, bool], tuple[LeftCoreFeature, list[Path]]] = {}
    for home in dependency_homes():
        try:
            features = _pending(home, say=labelled(home), process_env=launched(home))
            consent = bool(features) and _home_consent(home)
        except Exception as exc:
            logger.debug("left-core migration skipped for %s: %s", home, exc)
            continue
        for feature in features:
            pending.setdefault((feature.plugin, consent), (feature, []))[1].append(home)

    installed: list[str] = []
    try:
        for feature, homes in pending.values():
            if len(homes) > 1 and _interactive():
                say(f"  {feature.label} is used in {len(homes)} profiles "
                    f"({', '.join(_home_label(h) for h in homes)}); your answers to its dependency "
                    f"questions apply to all of them.")
            with shared_dependency_answers():
                for index, home in enumerate(homes):
                    if _install_one(home, feature, install=_install_into(home), say=labelled(home),
                                    process_env=launched(home)):
                        installed.append(feature.plugin)
                        continue
                    rest = homes[index + 1:]
                    if rest:
                        say(f"  ⚠ The '{feature.plugin}' plugin was not installed for "
                            f"{', '.join(_home_label(h) for h in rest)} either. Run "
                            + ", ".join(f"`{_install_command(feature.plugin, h)}`" for h in rest) + ".")
                    break
    except KeyboardInterrupt:
        say("  ⚠ Plugin migration cancelled. Profiles already migrated keep their plugin; run "
            "`hermes plugins install <name>` (with `-p <profile>`) for the rest.")
    return installed


def _gateway_serves(home: Path) -> bool:
    """A live gateway serves *home*: it loaded its platforms before this agent-start install."""
    try:
        from gateway.status import resolve_gateway_liveness
        return resolve_gateway_liveness(profile_dir=home, use_cache=False).running
    except Exception:
        return False


def recover_at_startup(*, say: Optional[Callable[[str], None]] = None, gateway_start: bool = False) -> list[str]:
    """Agent/gateway start hook for the active home: one attempt per process per home. With *say* (a
    terminal's sink) outcomes are delivered now, together with any an earlier attempt or boot maintenance
    (:func:`defer_startup_notice`) kept for this home in any process; without it they are logged and kept in
    the home (:func:`_queue_notice`) for its next turn someone sees (:func:`claim_kept_notices`), in
    whichever process. *gateway_start*: the gateway itself, which needs no restart hint. Returns installed
    plugin names."""
    from agent.safe_worker_policy import worker_config_snapshot
    if _installs_held.get():
        return []
    if worker_config_snapshot() is not None:
        # A frozen safe/ignore-user-config worker never reads the profile's config.yaml and never
        # installs customizations; the owner process runs this migration for the home.
        return []
    from hermes_constants import get_hermes_home, hermes_home_key

    home = Path(get_hermes_home())
    key = hermes_home_key(home)

    delivered: set[str] = set()

    def deliver(message: str) -> None:
        if say is None:
            _queue_notice(home, message)
            return
        if message in delivered:  # a kept copy of what this attempt reports again (allow_lazy_installs off)
            return
        delivered.add(message)
        try:
            say(message)
        except Exception:
            logger.debug("left-core migration notification failed", exc_info=True)

    if say is not None:
        for message in _claim_notices(home):
            deliver(message)
    if key in _attempted:
        return []
    _attempted.add(key)

    def report(message: str) -> None:
        message = message.strip()
        logger.warning(message)
        deliver(message)

    try:
        features = _pending(home, say=report, process_env=True, backoff=True)
        if not features:
            return []
        from pm.install import lazy_installs_allowed
        if not lazy_installs_allowed():
            for feature in features:
                report(_noticed(f"⚠ {feature.label} moved out of core and its '{feature.plugin}' plugin is not "
                                f"installed, so it is off. security.allow_lazy_installs is off, so Hermes did not "
                                f"fetch it: run `{_install_command(feature.plugin, home)}`.", home, feature, True))
            return []
        installed = []
        # Agent init cannot answer a dependency prompt: under the CLI the chat input owns the terminal,
        # so the question hangs at "Initializing agent...". Every row shipped in core, so its users
        # already accepted its dependencies (as memory_provider_migration.recover_at_startup does).
        for feature in features:
            if _install_one(home, feature, install=_install_into(home, consent=True), say=report, process_env=True):
                installed.append(feature.plugin)
                if not gateway_start and feature.platform and _gateway_serves(home):
                    report(f"Restart the gateway (`hermes gateway restart`) so it serves {feature.label}.")
        return installed
    except Exception as exc:  # never take agent/gateway start down
        logger.warning("left-core plugin migration failed: %s", exc, exc_info=True)
        return []


def recover_for_command(name: Optional[str]) -> None:
    """``hermes <name>`` names a command a left-core plugin registers, and the active home uses that
    feature but lacks the plugin: run :func:`recover_at_startup` now, before plugin commands are
    discovered, so the command it installs runs in this same invocation. A home nothing migrated yet
    (a Docker image swap runs no ``hermes update``) whose only Hermes process is a scheduled
    ``hermes import-agent --sync`` never starts an agent or a gateway to do it."""
    from hermes_constants import get_hermes_home
    home = Path(get_hermes_home())
    if any(name and name in (f.cli, f.command, *f.old_cli) and not plugin_present(f.plugin, home)
           and feature_in_use(f, home) for f in LEFT_CORE):
        recover_at_startup(say=lambda message: print(message, file=sys.stderr))
