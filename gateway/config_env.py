"""Environment-variable overrides for the gateway config (``_apply_env_overrides``).

Runs after ``GatewayConfig.from_dict`` so env always wins over config.yaml / gateway.json.
Most platforms follow one shape — "credentials present in env ⇒ enable the platform and
copy the values into ``extra``" — declared by each platform plugin as ``PlatformEntry.env_overrides``
(``gateway.platform_env.EnvCred`` rows; see that module). Core-only platforms (api_server, webhook,
msgraph_webhook, relay) keep their small functions here.
"""

import copy
import logging
import os
from contextvars import ContextVar
from typing import Optional

from gateway.config import (
    _BUILTIN_PLATFORM_VALUES,
    _discovered_registry_entry,
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    _getenv_str,
    _has_usable_api_server_key,
    SHARED_LISTENER_MIRROR_PLATFORMS,
)
from gateway.platform_env import INT, csv_extras, env_extras
from utils import is_truthy_value

# Logger name parity with the origin module: records stay under "gateway.config".
logger = logging.getLogger("gateway.config")

getenv = _getenv_str

# Platforms already warned "explicitly disabled in config.yaml, but credentials present in
# env": config reloads every turn, so the notice is one-time per platform per process.
_EXPLICIT_DISABLE_WARNED: set = set()

# Env var(s) whose presence drives a CORE platform's env-enable branch, named in the explicit-disable
# WARNING. Plugin platforms declare theirs as ``PlatformEntry.env_enable_credentials``.
_ENV_ENABLE_CREDENTIALS: dict = {
    Platform.RELAY: ("GATEWAY_RELAY_URL",),
}


def _enable_credential_names(platform: Platform) -> tuple:
    """WARNING credential names: the core row, else the platform entry's declaration. The entry is
    already materialized (its own env step is what is warning), so the lookup never imports."""
    if platform in _ENV_ENABLE_CREDENTIALS:
        return _ENV_ENABLE_CREDENTIALS[platform]
    from gateway.platform_registry import platform_registry
    entry = platform_registry.get(platform.value)
    names = tuple(getattr(entry, "env_enable_credentials", ()) or ())
    if not names:
        from gateway.platform_env import bundled_env_rows
        names = (bundled_env_rows(platform.value) or ((), ()))[1]
    return names


# Set by builtin_platform_status: a status read for another profile must neither repeat the gateway's
# operational logs on every dashboard poll nor use up its one-time warnings.
_STATUS_READ: ContextVar[bool] = ContextVar("gateway_env_status_read", default=False)


def _warn_explicit_disable_beats_env(platform: Platform) -> None:
    """One-time WARNING: ``platforms.<x>.enabled: false`` wins over env creds (which
    used to force-enable), so users relying on "creds in .env = on" learn why it went dark."""
    if platform in _EXPLICIT_DISABLE_WARNED or _STATUS_READ.get():
        return
    _EXPLICIT_DISABLE_WARNED.add(platform)
    names = _enable_credential_names(platform)
    present = [n for n in names if (os.environ.get(n) or "").strip()]
    creds = ", ".join(present or names) or "its credentials"
    logger.warning(
        "Platform '%s' is explicitly disabled by platforms.%s.enabled: false in "
        "config.yaml, so the credentials found in the environment (%s) will NOT "
        "start its adapter. Environment credentials no longer override an "
        "explicit disable. Remove the key or set platforms.%s.enabled: true to "
        "turn it back on.",
        platform.value, platform.value, creds, platform.value,
    )


def _loading_secondary_under_multiplexer() -> bool:
    """True while a multiplexer loads a NON-default profile's config (``_profile_runtime_scope`` sets the
    home override; the runner sets the multiplex flag). Same signal ``gateway.config`` uses for scoped reads."""
    from agent.secret_scope import is_multiplex_active
    from hermes_constants import get_hermes_home_override, profile_name_for_home
    override = get_hermes_home_override()
    return bool(override) and is_multiplex_active() and profile_name_for_home(override) != "default"


def _enable_from_env(
    config: GatewayConfig, platform: Platform, *, pop_marker: bool = False, warn: bool = True
) -> PlatformConfig:
    """Enable *platform* on env credentials unless config.yaml explicitly disabled it.

    A multiplex secondary profile may pin ``enabled: false`` yet inherit the process env; without
    this guard env presence would force-enable it. By default the ``_enabled_explicit`` marker is
    READ (the plugin-enable and relay passes still need it) and the disable is warned once;
    api_server/webhook POP it (terminal branch) and stay silent.
    """
    platform_config = config.platforms.setdefault(platform, PlatformConfig())
    extra = platform_config.extra
    explicit = extra.pop("_enabled_explicit", False) if pop_marker else extra.get("_enabled_explicit", False)
    if platform_config.enabled:
        return platform_config
    if not explicit and not (
        platform.value in SHARED_LISTENER_MIRROR_PLATFORMS and _loading_secondary_under_multiplexer()
    ):
        # A secondary's API_SERVER_KEY / WEBHOOK_ENABLED (the docs require the key in its .env for
        # /p/<profile>/ auth) must not turn into listener intent: the default profile's listener already
        # mirrors those two at /p/<profile>/ (#100397). The credential still lands in ``extra`` for it.
        # Every other inbound-port platform IS enabled for a secondary: it runs in shared-listener mode.
        platform_config.enabled = True
    elif warn:
        _warn_explicit_disable_beats_env(platform)
    return platform_config


def _api_server(config: GatewayConfig) -> None:
    """Require a usable key: an unauthenticated adapter refuses to start and the reconnect watcher would spin."""
    key = getenv("API_SERVER_KEY")
    if not _has_usable_api_server_key(key):
        return
    extra = _enable_from_env(config, Platform.API_SERVER, pop_marker=True, warn=False).extra
    extra["key"] = key
    csv_extras(extra, (("cors_origins", getenv("API_SERVER_CORS_ORIGINS")),))
    env_extras(extra, (("port", "API_SERVER_PORT", INT), ("host", "API_SERVER_HOST"), ("model_name", "API_SERVER_MODEL_NAME")))


def _webhook(config: GatewayConfig) -> None:
    enabled = is_truthy_value(getenv("WEBHOOK_ENABLED"))
    if not (enabled or Platform.WEBHOOK in config.platforms):
        return
    webhook_config = (
        _enable_from_env(config, Platform.WEBHOOK, pop_marker=True, warn=False)
        if enabled
        else config.platforms[Platform.WEBHOOK]
    )
    env_extras(webhook_config.extra, (("port", "WEBHOOK_PORT", INT), ("secret", "WEBHOOK_SECRET")))


def _msgraph_webhook(config: GatewayConfig) -> None:
    enabled = is_truthy_value(getenv("MSGRAPH_WEBHOOK_ENABLED"))
    client_state = getenv("MSGRAPH_WEBHOOK_CLIENT_STATE")
    resources = getenv("MSGRAPH_WEBHOOK_ACCEPTED_RESOURCES")
    allowed_cidrs = getenv("MSGRAPH_WEBHOOK_ALLOWED_SOURCE_CIDRS")
    if not (enabled or Platform.MSGRAPH_WEBHOOK in config.platforms or getenv("MSGRAPH_WEBHOOK_PORT")
            or client_state or resources or allowed_cidrs):
        return
    msgraph_cfg = config.platforms.setdefault(Platform.MSGRAPH_WEBHOOK, PlatformConfig())
    if enabled:  # same guard as webhook, but READ (don't pop) the marker: the relay pass still consults it
        _enable_from_env(config, Platform.MSGRAPH_WEBHOOK, warn=False)
    env_extras(msgraph_cfg.extra, (("port", "MSGRAPH_WEBHOOK_PORT", INT),))
    if client_state:
        msgraph_cfg.extra["client_state"] = client_state
    csv_extras(msgraph_cfg.extra, (("accepted_resources", resources), ("allowed_source_cidrs", allowed_cidrs)))


def _plugin_probe_seed(entry) -> Optional[dict]:
    """``env_enablement_fn()`` result as a non-empty dict, else None."""
    if entry.env_enablement_fn is None:
        return None
    try:
        seed = entry.env_enablement_fn()
    except Exception as e:
        logger.debug("env_enablement_fn for %s raised: %s", entry.name, e)
        return None
    return seed if isinstance(seed, dict) and seed else None


def _plugin_is_configured(entry, existing_extra: dict, seed: Optional[dict]) -> bool:
    """``entry.is_connected`` on a transient ``enabled=True`` view seeded with env extras (never the real config)."""
    try:
        for k, v in (seed or {}).items():
            if k != "home_channel":
                existing_extra.setdefault(k, v)
        configured = bool(entry.is_connected(PlatformConfig(enabled=True, extra=existing_extra)))
    except Exception as exc:
        logger.debug("is_connected for %s raised: %s — skipping enablement", entry.name, exc)
        return False
    if not configured:
        logger.debug(
            "Plugin platform '%s' available but not configured "
            "(is_connected returned False) — skipping enable",
            entry.name,
        )
    return configured


def _enable_plugin_platform(config: GatewayConfig, entry) -> None:
    try:
        platform = Platform(entry.name)
    except Exception as e:
        logger.debug("unknown platform name %r: %s", entry.name, e)
        return
    existing_cfg = config.platforms.get(platform)
    already_enabled = existing_cfg is not None and existing_cfg.enabled
    existing_extra = dict(existing_cfg.extra or {}) if existing_cfg is not None else {}
    # Never re-enable a platform the user explicitly disabled (marker set by the YAML loader).
    if existing_cfg is not None and not already_enabled and existing_extra.get("_enabled_explicit", False):
        return
    seed = _plugin_probe_seed(entry)
    # is_connected gates only platforms not already enabled by YAML/env.
    if not already_enabled and entry.is_connected is not None and not _plugin_is_configured(entry, existing_extra, seed):
        return
    # Dependencies LAST — only for platforms already enabled or past the credential gate.
    try:
        # ``check_fn`` is a PASSIVE probe (never installs); a platform whose deps are missing but which
        # registered ``ensure_deps_fn`` still gets enabled here — the registry's ``create_adapter()`` runs
        # the active installer at gateway start, when the user actually wants the platform up. Historically
        # the ACTIVE installer was wired as ``check_fn`` and this sweep pip-installed
        # Discord/Telegram/Slack/Feishu/Dingtalk SDKs on every ``load_gateway_config()`` call — including
        # the desktop/dashboard readiness probe (``GET /api/status``) — blocking startup until every install
        # finished and boot-looping the desktop app at 94%. The check_fn/ensure_deps_fn split (#79812) makes
        # that impossible by construction.
        deps_ok = bool(entry.check_fn())
    except Exception as e:
        logger.debug("check_fn for %s raised: %s", entry.name, e)
        deps_ok = False
    if not deps_ok and entry.ensure_deps_fn is None:
        return
    platform_config = config.platforms.setdefault(platform, PlatformConfig())
    platform_config.enabled = True
    if seed:  # commit the probe's env-seeded extras (env_enablement_fn is never called twice)
        seed = dict(seed)
        home = seed.pop("home_channel", None)
        platform_config.extra.update(seed)
        if isinstance(home, dict) and home.get("chat_id"):
            platform_config.home_channel = HomeChannel(
                platform=platform, chat_id=str(home["chat_id"]), name=str(home.get("name") or "Home"),
                thread_id=str(home["thread_id"]) if home.get("thread_id") else None,
            )


def _configured_plugin_entries(config: GatewayConfig) -> list:
    """Plugin entries with activation evidence (config block, env/.env, credential pool); only those
    are imported — an unconfigured adapter has nothing to enable. See gateway.platform_activation.
    config.yaml blocks already materialized their adapters in load_yaml_layer (and every merged
    block is in config.platforms), so the YAML is not parsed a second time here."""
    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        from gateway.platform_activation import configured_plugin_entries
        return configured_plugin_entries(platform_registry, yaml_cfg={}, platform_names=config.platforms)
    except Exception as e:  # health: allow BLE001 -- moved from the plugin pass: discovery never breaks config load
        logger.debug("Plugin platform discovery for env overrides failed: %s", e)
        return []


def _bundled_env_fallbacks(config: GatewayConfig, entries: list, *, only: Optional[Platform] = None) -> list:
    """Env rows of bundled platforms with activation evidence but no registry entry (discovery
    suppressed or the plugin disabled), read from their ``gateway_env`` files; see
    ``gateway.platform_env.bundled_env_rows``. They only run the env steps, never the enable pass."""
    from types import SimpleNamespace
    from gateway.platform_activation import activation_predicate
    from gateway.platform_env import _BUNDLED_ENV_SIBLINGS, bundled_env_rows
    have = {getattr(e, "name", "") for e in entries}
    if only is not None:  # bundled_env_rows() answers for one name without scanning every manifest
        names = {only.value} - have
    else:
        names = (set(Platform._scan_bundled_plugin_platforms()[0]) | set(_BUNDLED_ENV_SIBLINGS)) - have
    if not names:
        return []
    wanted = activation_predicate(yaml_cfg={}, platform_names=config.platforms)
    out = []
    for name in sorted(names):
        rows = bundled_env_rows(name) if wanted(name) else None
        if rows and rows[0]:
            out.append(SimpleNamespace(name=name, env_overrides=rows[0], env_enable_credentials=rows[1]))
    return out


def _platform_rank(entry) -> int:
    # Static ``Platform`` members first, in enum order (the order the env rows always ran in), so
    # ``config.platforms`` insertion order stays deterministic and independent of registry load order.
    members = list(Platform.__members__.values())
    try:
        return members.index(Platform(entry.name))
    except (ValueError, KeyError):
        return len(members)


def _apply_plugin_env_overrides(config: GatewayConfig, entries: list) -> None:
    """Run each plugin platform's declared ``env_overrides`` steps (credential enable, home channel,
    reply mode, ...). Every step touches only its own platform, so only the order WITHIN an entry is
    significant (e.g. Telegram's reply mode may create the entry its home channel then attaches to)."""
    for entry in sorted(entries, key=_platform_rank):
        for step in getattr(entry, "env_overrides", ()) or ():
            try:
                step(config)
            except Exception as e:  # health: allow BLE001 -- a broken plugin env row must not break config load
                logger.debug("env override for %s raised: %s", entry.name, e)


def _enable_plugin_platforms_from_env(config: GatewayConfig, entries: list) -> None:
    """Registry-driven enable for plugin platforms whose ``env_overrides`` did not already enable them.

    Enabled when credentials are configured (``is_connected`` MUST gate: ``check_fn`` alone would
    enable unconfigured platforms that retry-connect forever) and deps are present (``check_fn``) or
    installable later by ``create_adapter()`` — never here: installing in this sweep boot-looped the app.
    """
    for entry in entries:
        try:
            _enable_plugin_platform(config, entry)
        except Exception as e:
            logger.debug("Plugin platform enable pass failed for %s: %s", entry.name, e)


def _relay(config: GatewayConfig) -> None:
    """Relay (connector-fronted, EXPERIMENTAL): enabled by GATEWAY_RELAY_URL or gateway.relay_url;
    the URL is mirrored into extra["relay_url"] for the connected-checker.

    Relay-exclusive: the GATEWAY_RELAY_URL env stamp means the connector owns every platform
    connection; a directly-connected adapter would be a second unmanaged ingress (duplicate
    deliveries, split sessions, a socket disarming scale-to-zero). So the env stamp disables all
    other messaging platforms — even explicitly enabled ones — except non-messaging surfaces (local,
    api_server, webhook; same set as the scale-to-zero arm gate). relay_url from YAML only keeps the
    additive behavior. Opt out with GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true.
    """
    relay_url_env = getenv("GATEWAY_RELAY_URL").strip()
    existing_relay = config.platforms.get(Platform.RELAY)
    relay_url_yaml = str(existing_relay.extra.get("relay_url") or "").strip() if existing_relay else ""
    relay_url_val = relay_url_env or relay_url_yaml
    if relay_url_val:
        relay_config = _enable_from_env(config, Platform.RELAY)
        relay_config.extra["relay_url"] = relay_url_val.rstrip("/")
        # An opted-out relay does not own this profile's native connections.
        if not relay_config.enabled:
            return

    if not relay_url_env or is_truthy_value(getenv("GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS")):
        return
    skip = {Platform.RELAY, Platform.LOCAL, Platform.API_SERVER, Platform.WEBHOOK}
    for platform, platform_config in config.platforms.items():
        if platform in skip or not platform_config.enabled:
            continue
        if _STATUS_READ.get():
            pass  # status reads repeat this on every poll
        elif platform_config.extra.get("_enabled_explicit"):
            logger.warning(
                "Relay connector is configured via GATEWAY_RELAY_URL; "
                "disabling directly-connected platform '%s' even though "
                "it is explicitly enabled in this profile's configuration. "
                "All messaging goes through the connector on this "
                "deployment. Set GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true "
                "to keep direct platforms alongside the relay.",
                platform.value,
            )
        else:
            logger.info(
                "Relay connector is configured via GATEWAY_RELAY_URL; "
                "disabling directly-connected platform '%s'.",
                platform.value,
            )
        platform_config.enabled = False


def _scrub_explicit_markers(config: GatewayConfig) -> None:
    for platform_config in config.platforms.values():
        platform_config.extra.pop("_enabled_explicit", None)


def _apply_env_overrides(config: GatewayConfig, *, only: Optional[Platform] = None) -> None:
    """Apply environment variable overrides to *config*.

    Order: plugin platforms' declared env rows, then the core-only rows (api_server, webhook,
    msgraph_webhook; each touches only its own platform), then the registry enable pass, relay
    disabling (needs every platform's final ``enabled``), and the marker scrub, which must be last.
    ``only`` limits the plugin passes to that built-in platform's own entry and bundled env rows
    (no per-call manifest scan; see ``builtin_platform_status``).
    """
    if only is None:
        entries = _configured_plugin_entries(config)
    else:
        entry = _discovered_registry_entry(only)
        entries = [entry] if entry is not None else []
    _apply_plugin_env_overrides(config, entries + _bundled_env_fallbacks(config, entries, only=only))
    _api_server(config)
    _webhook(config)
    _msgraph_webhook(config)
    _enable_plugin_platforms_from_env(config, entries)
    _relay(config)
    _scrub_explicit_markers(config)


def builtin_platform_status(platform_id: str, yaml_cfg: dict) -> Optional[tuple[bool, bool]]:
    """``(enabled, connected)`` as the gateway decides them for one built-in platform of the profile
    whose config.yaml is *yaml_cfg*, with env read through the bound secret scope; None for a
    plugin-only id.

    For judging one platform of a profile that is not the process's own (the dashboard's Channels
    page): ``load_gateway_config()`` there enumerates that profile's platform plugins and constructs
    ``Platform(<plugin id>)`` for them, process-wide pseudo-members every other profile then sees.
    So only this platform's and the relay's blocks (the relay can disable it) go through the
    loader's own steps (section merge, administrator-pinned keys, top-level bridge; see
    ``load_yaml_layer``), which also record an explicit ``enabled`` for the env pass. The env pass's
    logs and one-time warnings are held back.
    """
    if platform_id not in _BUILTIN_PLATFORM_VALUES:
        return None
    from gateway.config_loader import (
        _apply_managed_extra, bridge_platform_shared_keys, merge_platform_sections, snapshot_authored_extra,
    )
    from hermes_cli.managed_scope import apply_managed_overlay

    platform = Platform(platform_id)
    names = {platform_id, Platform.RELAY.value}
    top, nested = _platform_layers(yaml_cfg, names)
    platforms = merge_platform_sections(top, nested, gw_data={})
    authored = snapshot_authored_extra(platforms)  # no gateway.json layer here: the merge is config.yaml only
    managed_top, managed_nested = _platform_layers(apply_managed_overlay({}), names)
    _apply_managed_extra(authored, {**managed_top, "gateway": managed_nested}, platforms)
    bridge_platform_shared_keys(top, nested["platforms"], {}, platforms, [platform, Platform.RELAY], authored=authored)
    for block in platforms.values():
        block.pop("home_channel", None)  # status needs none; HomeChannel.from_dict can raise or name another platform
    config = GatewayConfig.from_dict({"platforms": platforms})
    token = _STATUS_READ.set(True)
    try:
        _apply_env_overrides(config, only=platform)
        platform_config = config.platforms.get(platform)
        if platform_config is None:
            return False, False
        return bool(platform_config.enabled), config._is_platform_connected(platform, platform_config)
    finally:
        _STATUS_READ.reset(token)


def _platform_layers(yaml_cfg: dict, names: set) -> tuple[dict, dict]:
    """*yaml_cfg*'s platform blocks for *names* only, as the loader's two inputs: the root layer
    (``platforms.<id>`` and top-level ``<id>:``) and the ``gateway:`` layer."""
    gateway_section = yaml_cfg.get("gateway") if isinstance(yaml_cfg.get("gateway"), dict) else {}
    top = {"platforms": _only_keys(yaml_cfg.get("platforms"), names), **_only_keys(yaml_cfg, names)}
    nested = {"platforms": _only_keys(gateway_section.get("platforms"), names), **_only_keys(gateway_section, names)}
    return top, nested


def _only_keys(section, names: set) -> dict:
    """Copies of *section*'s entries named in *names* (no others reach the loader's ``Platform(key)``
    probes, and the merge and bridge, which write into blocks they reuse, never touch the caller's)."""
    if not isinstance(section, dict):
        return {}
    return {key: copy.deepcopy(value) for key, value in section.items() if key in names}
