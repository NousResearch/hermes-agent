"""config.yaml / gateway.json → ``GatewayConfig.from_dict`` schema (the ``load_gateway_config`` phases).

Precedence for top-level keys: key-presence at the TOP LEVEL of config.yaml wins; the nested
``gateway.<key>`` form (what ``hermes config set gateway.<key>`` produces) is consulted only when the
top-level key is absent — not merely falsy/mistyped — so a present-but-empty top-level value is never
silently replaced by the nested one. Both overwrite whatever legacy gateway.json set.
"""

import contextlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from gateway.config import Platform, _dict_slot, _normalize_choice

# Logger name parity with the origin module: records stay under "gateway.config".
logger = logging.getLogger("gateway.config")


def load_legacy_gateway_json(home: Path) -> Any:
    """Legacy ``gateway.json`` base layer (config.yaml keys always win). Malformed → ``{}`` + warning."""
    path = home / "gateway.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        logger.info("Loaded legacy %s — consider moving settings to config.yaml", path)
        return data
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


# --- top-level key bridging ----------------------------------------------------
#
# Top-level settings are also accepted nested under ``gateway:`` (what ``hermes config set
# gateway.<key>`` produces). This loader builds gw_data FLAT and never forwards the yaml ``gateway:``
# section, so even keys GatewayConfig.from_dict can fall back on itself (loop_watchdog*,
# multiplex_profiles, ...) must be bridged here or they are silently ignored on real startup.
#
# Fallback modes (how the nested ``gateway.<key>`` form is consulted):
#   "presence": top-level key present → its value; else nested key present.
#   "gwdata":   like "presence", but nested only when NOTHING (config.yaml or gateway.json) set it yet.
#   "none":     top-level VALUE is None → nested value.
#   "dict":     top-level value is not a mapping → nested value; accepted only if a mapping.
#   "nested":   nested form only (no top-level spelling is bridged).

def _quick_commands_ok(value: Any) -> bool:
    if isinstance(value, dict):
        return True
    logger.warning(
        "Ignoring invalid quick_commands in config.yaml (expected mapping, got %s)",
        type(value).__name__,
    )
    return False


def _dm_behavior_choice(value: Any, default: str = "pair") -> str:
    return _normalize_choice(value, {"pair", "ignore"}, default)


def _presence(*keys: str) -> tuple:
    return tuple((k, k, "presence", None, None) for k in keys)


# (yaml key, gw_data key, mode, accept(value) -> bool, transform(value))
_TOPLEVEL_BRIDGE: tuple = (
    ("session_reset", "default_reset_policy", "presence", lambda v: bool(v) and isinstance(v, dict), None),
    ("quick_commands", "quick_commands", "none", _quick_commands_ok, None),
    ("stt", "stt", "presence", lambda v: isinstance(v, dict), None),
    *_presence("stt_echo_transcripts", "group_sessions_per_user", "thread_sessions_per_user"),
    ("multiplex_profiles", "multiplex_profiles", "gwdata", None, None),
    *_presence("multiplex_profile_allowlist", "room_link_url"),
    ("profile_routes", "profile_routes", "none", lambda v: isinstance(v, list), None),
    *_presence("max_concurrent_sessions"),
    ("systemd_watchdog_seconds", "systemd_watchdog_seconds", "nested", None, None),
    ("streaming", "streaming", "dict", None, None),
    *_presence(
        "reset_triggers", "always_log_local", "write_sessions_json", "hosted_rooms_enabled",
        "loop_watchdog",
        "loop_watchdog_probe_interval_s", "loop_watchdog_probe_timeout_s", "loop_watchdog_max_strikes",
        "filter_silence_narration",
    ),
    ("unauthorized_dm_behavior", "unauthorized_dm_behavior", "presence", None, _dm_behavior_choice),
)


def _bridge_lookup(yaml_cfg: dict, gateway_section: Any, gw_data: dict, key: str, mode: str) -> tuple:
    """Return ``(found, value)`` for *key* under the given fallback *mode*."""
    nested = isinstance(gateway_section, dict)
    if mode in ("presence", "gwdata"):
        if key in yaml_cfg:
            return True, yaml_cfg[key]
        if nested and key in gateway_section and (mode == "presence" or key not in gw_data):
            return True, gateway_section[key]
        return False, None
    if mode == "nested":
        return (True, gateway_section[key]) if nested and key in gateway_section else (False, None)
    value = yaml_cfg.get(key)
    if mode == "none":
        if value is None and nested:
            value = gateway_section.get(key)
        return value is not None, value
    # "dict"
    if not isinstance(value, dict) and nested:
        value = gateway_section.get(key)
    return isinstance(value, dict), value


def bridge_toplevel_keys(yaml_cfg: dict, gateway_section: Any, gw_data: dict) -> None:
    for yaml_key, gw_key, mode, accept, transform in _TOPLEVEL_BRIDGE:
        found, value = _bridge_lookup(yaml_cfg, gateway_section, gw_data, yaml_key, mode)
        if not found or (accept is not None and not accept(value)):
            continue
        gw_data[gw_key] = transform(value) if transform else value


# --- platform sections -----------------------------------------------------------

def merge_platform_sections(yaml_cfg: dict, gateway_cfg: Any, gw_data: dict) -> dict:
    """Merge every place a platform block may live into ``gw_data["platforms"]`` and return it.

    Order (later wins on shared keys, ``extra`` deep-merged so gateway.json defaults survive):
    ``gateway.platforms.*`` → top-level ``platforms.*`` → ``gateway.<platform>`` subsections (nested
    first so top-level config keeps precedence, matching the gateway.streaming fallback). An
    ``enabled`` key in any block sets the ``_enabled_explicit`` marker consumed by the env pass.
    Finally api_server's port/key/host/cors_origins/model_name are bridged into ``extra`` so
    ``gateway.api_server.port: 8642`` reaches the adapter (mirrors the env path).
    """
    platforms_data = _dict_slot(gw_data, "platforms")

    def merge(source_platforms: Any) -> None:
        if not isinstance(source_platforms, dict):
            return
        for plat_name, plat_block in source_platforms.items():
            if not isinstance(plat_block, dict):
                continue
            existing = platforms_data.get(plat_name, {})
            existing = existing if isinstance(existing, dict) else {}
            merged_extra = {**existing.get("extra", {}), **plat_block.get("extra", {})}
            if "enabled" in plat_block:
                merged_extra["_enabled_explicit"] = True
            merged = {**existing, **plat_block}
            if merged_extra:
                merged["extra"] = merged_extra
            platforms_data[plat_name] = merged

    nested_gateway = gateway_cfg if isinstance(gateway_cfg, dict) else {}
    merge(nested_gateway.get("platforms"))
    merge(yaml_cfg.get("platforms"))
    merge({k: v for k, v in nested_gateway.items() if k != "platforms" and isinstance(v, dict) and _is_platform_name(k)})

    api_plat = platforms_data.get("api_server")
    if isinstance(api_plat, dict):
        api_extra = _dict_slot(api_plat, "extra")
        for key in ("port", "key", "host", "cors_origins", "model_name"):
            if key in api_plat and key not in api_extra:
                api_extra[key] = api_plat.pop(key)
    return platforms_data


def _is_platform_name(key: Any) -> bool:
    try:
        Platform(key)
    except (ValueError, AttributeError):
        return False
    return True


def platform_section(yaml_cfg: dict, name: str, gateway_platforms: Any) -> tuple:
    """``(section, is_toplevel)`` for platform *name*: a top-level ``<name>:`` block wins; otherwise
    the block under ``gateway.platforms`` / ``platforms`` so shared-key bridging and adapter hooks
    still run for nested-only configs."""
    section = yaml_cfg.get(name)
    toplevel = isinstance(section, dict)
    if not toplevel:
        nested = (src[name] for src in (gateway_platforms, yaml_cfg.get("platforms")) if isinstance(src, dict) and isinstance(src.get(name), dict))
        section = next(nested, section)
    return section, toplevel


def _str_keyed(value: Any) -> Any:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else value


_TELEGRAM = frozenset({Platform.TELEGRAM})
_DISCORD_SLACK = frozenset({Platform.DISCORD, Platform.SLACK})

def _plain(*keys: str) -> tuple:
    """Keys copied verbatim into ``extra`` for every platform."""
    return tuple((k, None, None) for k in keys)


# (key, platforms-or-None, transform-or-None), in ``extra`` insertion order.
# ``"dm"`` defers to the global unauthorized_dm_behavior.
_SHARED_KEYS: tuple = (
    ("unauthorized_dm_behavior", None, "dm"),
    ("notice_delivery", None, lambda v: _normalize_choice(v, {"public", "private"}, "public")),
    *_plain("reply_prefix", "reply_in_thread", "cron_continuable_surface", "require_mention", "send_read_receipts"),
    ("allowed_chats", _TELEGRAM, None),
    ("group_allowed_chats", _TELEGRAM, None),
    ("allowed_topics", _TELEGRAM, None),
    *_plain("free_response_channels", "mention_patterns", "exclusive_bot_mentions"),
    ("observe_unmentioned_group_messages", _TELEGRAM, None),
    *_plain(
        "dm_policy", "allow_from", "allow_admin_from", "user_allowed_commands",
        "group_policy", "group_allow_from", "group_allow_admin_from", "group_user_allowed_commands",
    ),
    ("channel_skill_bindings", _DISCORD_SLACK, None),
    ("channel_prompts", None, _str_keyed),
    *_plain("gateway_restart_notification", "typing_indicator", "typing_status_text"),
)

# Top-level port/host/secret bridged into ``extra`` for adapters that read them from config.extra
# (PlatformConfig.from_dict only reads the ``extra:`` sub-key, so ``platforms.webhook.port`` would be lost).
_PORT_BRIDGE_KEYS: dict = {
    Platform.WEBHOOK: ("port", "host", "secret"),
    Platform.MSGRAPH_WEBHOOK: ("port", "host", "secret"),
    Platform.API_SERVER: ("port", "host"),
}


def _bridged_keys(plat: Platform, platform_cfg: dict, gw_data: dict) -> dict:
    bridged: dict = {}
    for key, only, transform in _SHARED_KEYS:
        if key not in platform_cfg or (only is not None and plat not in only):
            continue
        if transform == "dm":
            bridged[key] = _dm_behavior_choice(platform_cfg[key], gw_data.get("unauthorized_dm_behavior", "pair"))
        else:
            bridged[key] = transform(platform_cfg[key]) if transform else platform_cfg[key]
    for key in _PORT_BRIDGE_KEYS.get(plat, ()):
        if key in platform_cfg and key not in platform_cfg.get("extra", {}):
            bridged[key] = platform_cfg[key]
    return bridged


def shared_loop_targets(registry) -> list:
    """Built-in platforms plus registered plugin platforms (so plugin authors get shared-key bridging)."""
    targets: list = list(Platform)
    for entry in registry.plugin_entries() if registry is not None else ():
        with contextlib.suppress(ValueError, KeyError):
            if (plat := Platform(entry.name)) not in targets:
                targets.append(plat)
    return targets


def bridge_platform_shared_keys(
    yaml_cfg: dict, gateway_platforms: Any, gw_data: dict, platforms_data: dict, targets: list
) -> None:
    """Copy shared keys (allow_from, require_mention, …) from each platform's YAML section into ``extra``.

    ``enabled`` is only written from a TOP-LEVEL block: for nested-only configs ``merge_platform_sections``
    already merged it with the correct precedence. An explicit top-level enable/disable sets
    ``_enabled_explicit`` so the env pass honors ``enabled: false`` for migrated plugin platforms
    instead of re-enabling them on token/SDK presence.
    """
    for plat in targets:
        if plat == Platform.LOCAL:
            continue
        platform_cfg, cfg_toplevel = platform_section(yaml_cfg, plat.value, gateway_platforms)
        if not isinstance(platform_cfg, dict):
            continue
        bridged = _bridged_keys(plat, platform_cfg, gw_data)
        has_channel_overrides = "channel_overrides" in platform_cfg
        if has_channel_overrides and isinstance(platform_cfg.get("channel_overrides"), dict):
            plat_data = _dict_slot(platforms_data, plat.value)
            _dict_slot(plat_data, "extra")
            plat_data["channel_overrides"] = {
                str(cid): ov_data
                for cid, ov_data in platform_cfg["channel_overrides"].items()
                if isinstance(ov_data, dict)
            }
        enabled_was_explicit = cfg_toplevel and "enabled" in platform_cfg
        if not bridged and not enabled_was_explicit and not has_channel_overrides:
            continue
        plat_data = _dict_slot(platforms_data, plat.value)
        extra = _dict_slot(plat_data, "extra")
        if enabled_was_explicit:
            plat_data["enabled"] = platform_cfg["enabled"]
            extra["_enabled_explicit"] = True
        extra.update(bridged)


def apply_plugin_yaml_hooks(yaml_cfg: dict, gateway_platforms: Any, platforms_data: dict, registry) -> None:
    """Plugin-owned YAML→env config bridges (``PlatformEntry.apply_yaml_config_fn``). Order: shared-key
    loop → this dispatch → core-only bridges (require_mention/signal) → ``_apply_env_overrides()``."""
    if registry is None:
        return
    for entry in registry.all_entries():
        # Plugin-owned YAML→env config bridges (#24836). See ``PlatformEntry.apply_yaml_config_fn`` for the
        # hook contract. Order: shared-key loop (above) → this dispatch → legacy hardcoded blocks (below;
        # no-op when a hook already set their env var) → ``_apply_env_overrides()`` after
        # ``GatewayConfig.from_dict``.
        if entry.apply_yaml_config_fn is None:
            continue
        platform_cfg, _ = platform_section(yaml_cfg, entry.name, gateway_platforms)
        if not isinstance(platform_cfg, dict):
            continue
        try:
            seeded = entry.apply_yaml_config_fn(yaml_cfg, platform_cfg)
        except Exception as e:
            logger.debug("apply_yaml_config_fn for %s raised: %s", entry.name, e)
            continue
        if isinstance(seeded, dict) and seeded:
            _dict_slot(_dict_slot(platforms_data, entry.name), "extra").update(seeded)


def bridge_core_env_settings(yaml_cfg: dict, platforms_data: dict) -> None:
    """The two YAML→env bridges that stay in core (per-platform ones live in plugin hooks).

    Top-level ``require_mention`` → Telegram when the ``telegram:`` section has none: users write it
    alongside ``group_sessions_per_user`` expecting it to work, and the telegram plugin's hook only
    runs when a telegram block exists. Signal ``require_mention`` → ``SIGNAL_REQUIRE_MENTION`` (env wins).
    """
    tl_require_mention = yaml_cfg.get("require_mention")
    if tl_require_mention is not None and "require_mention" not in (yaml_cfg.get("telegram") or {}):
        tg_plat = platforms_data.setdefault(Platform.TELEGRAM.value, {})
        tg_plat.setdefault("extra", {}).setdefault("require_mention", tl_require_mention)
        # Also bridge to the TELEGRAM_REQUIRE_MENTION env var that the adapter reads at runtime. This used
        # to live in the telegram_cfg block in core; it stays in core because it keys off the TOP-LEVEL
        # require_mention (not a telegram: block), so the telegram plugin's apply_yaml_config_fn hook —
        # which only runs when a telegram config block exists — can't cover the no-telegram-block case
        # (#3979).
        if not os.getenv("TELEGRAM_REQUIRE_MENTION"):
            os.environ["TELEGRAM_REQUIRE_MENTION"] = str(tl_require_mention).lower()

    # Telegram settings → env vars / extra: migrated to the telegram plugin's apply_yaml_config_fn hook
    # (plugins/platforms/telegram/adapter.py). #41112 / #3823.
    # WhatsApp settings → env vars: migrated to the whatsapp plugin's apply_yaml_config_fn hook
    # (plugins/platforms/whatsapp/adapter.py). #41112 / #3823.
    signal_cfg = yaml_cfg.get("signal", {})
    if isinstance(signal_cfg, dict) and "require_mention" in signal_cfg and not os.getenv("SIGNAL_REQUIRE_MENTION"):
        os.environ["SIGNAL_REQUIRE_MENTION"] = str(signal_cfg["require_mention"]).lower()


def load_yaml_layer(home: Path, gw_data: dict) -> None:
    """Overlay ``config.yaml`` onto *gw_data* in place. Raises on any failure (caller warns + falls back)."""
    import yaml

    config_yaml_path = home / "config.yaml"
    if not config_yaml_path.exists():
        return
    with open(config_yaml_path, encoding="utf-8") as f:
        yaml_cfg = yaml.safe_load(f) or {}

    # Managed scope: overlay administrator-pinned values (this loader bypasses
    # hermes_cli.config.load_config, so a managed session_reset / quick_commands / stt would otherwise be ignored).
    from hermes_cli import managed_scope
    yaml_cfg = managed_scope.apply_managed_overlay(yaml_cfg)

    gateway_section = yaml_cfg.get("gateway")
    bridge_toplevel_keys(yaml_cfg, gateway_section, gw_data)
    gateway_platforms = gateway_section.get("platforms") if isinstance(gateway_section, dict) else None
    platforms_data = merge_platform_sections(yaml_cfg, gateway_section, gw_data)

    try:
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry as registry
    except Exception as e:
        logger.debug("plugin discovery skipped: %s", e)
        registry = None

    targets = shared_loop_targets(registry)
    bridge_platform_shared_keys(yaml_cfg, gateway_platforms, gw_data, platforms_data, targets)
    apply_plugin_yaml_hooks(yaml_cfg, gateway_platforms, platforms_data, registry)
    bridge_core_env_settings(yaml_cfg, platforms_data)
