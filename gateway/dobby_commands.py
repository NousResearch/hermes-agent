"""Read-only Dobby command center helpers for gateway slash commands."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gateway.config import Platform

try:
    from scripts.dobby_package import redact_text
except Exception:  # pragma: no cover - package helpers may be absent in lean installs
    def redact_text(text: str, env: Mapping[str, Any] | None = None) -> str:
        return text


BROAD_VALUES = frozenset({"*", "all", "any", "everyone", "@everyone", "public"})
CORE_DOBBY_PLATFORMS = frozenset({"discord", "webhook", "api_server"})
PENDING_FEATURES = (
    "health",
    "quota",
    "research",
    "reminders",
    "attachment review",
    "repo helper",
    "memory controls",
)


def handle_dobby_command(
    args: str,
    *,
    config: Any = None,
    adapters: Mapping[Any, Any] | None = None,
    hermes_home: str | Path | None = None,
    profile_name: str | None = None,
    package_root: str | Path | None = None,
    cron_count: int | None = None,
) -> str:
    """Handle the read-only ``/dobby`` gateway command."""

    parts = str(args or "").strip().split()
    subcommand = parts[0].lower() if parts else "status"

    if subcommand in {"status", "help"}:
        if subcommand == "help":
            return _redact_output(
                "Dobby command center\n"
                "- /dobby status: read-only package readiness status\n"
                f"- Pending: {', '.join(PENDING_FEATURES)}"
            )
        return render_dobby_status(
            config=config,
            adapters=adapters,
            hermes_home=hermes_home,
            profile_name=profile_name,
            package_root=package_root,
            cron_count=cron_count,
        )

    return _redact_output(
        f"Dobby subcommand `{subcommand}` is not implemented yet. "
        "Try `/dobby status`."
    )


def render_dobby_status(
    *,
    config: Any = None,
    adapters: Mapping[Any, Any] | None = None,
    hermes_home: str | Path | None = None,
    profile_name: str | None = None,
    package_root: str | Path | None = None,
    cron_count: int | None = None,
) -> str:
    """Render safe high-level Dobby package readiness status."""

    package_state = _path_state(_default_package_root(package_root))
    home_state = _path_state(Path(hermes_home).expanduser() if hermes_home else None)
    profile_state = "present" if profile_name else "unknown"
    if profile_name == "default":
        profile_state = "default"
    elif profile_name == "custom":
        profile_state = "custom"
    elif profile_name:
        profile_state = "named"

    enabled_platforms = _enabled_platform_names(config)
    discord_allowlist = _discord_allowlist_state(config, adapters or {})
    webhook_policy = _webhook_policy_state(config, adapters or {})
    browser_state = _browser_state(config)
    broad_state = _broad_integrations_state(config, enabled_platforms)
    memory_state = _memory_boundary_state(config)
    cron_state = str(cron_count) if isinstance(cron_count, int) and cron_count >= 0 else "unknown"

    readiness = _readiness_state(
        package_state=package_state,
        home_state=home_state,
        discord_allowlist=discord_allowlist,
        webhook_policy=webhook_policy,
        browser_state=browser_state,
        broad_state=broad_state,
        memory_state=memory_state,
    )

    platforms = ", ".join(enabled_platforms) if enabled_platforms else "none"
    lines = [
        "Dobby Package Status",
        f"- Readiness: {readiness}",
        f"- Package artifacts: {package_state}",
        f"- Profile: {profile_state}",
        f"- Hermes home: {home_state}",
        f"- Enabled platforms: {platforms}",
        f"- Discord allowlist: {discord_allowlist}",
        f"- Webhook strict policy: {webhook_policy}",
        f"- Browser integration: {browser_state}",
        f"- Broad integrations: {broad_state}",
        f"- Memory/Honcho boundary: {memory_state}",
        f"- Cron jobs: {cron_state}",
        f"- Pending: {', '.join(PENDING_FEATURES)}",
        "No secret values or live checks are shown.",
    ]
    return _redact_output("\n".join(lines))


def _default_package_root(package_root: str | Path | None) -> Path:
    if package_root is not None:
        return Path(package_root)
    return Path.cwd() / "packaging" / "dobby-package"


def _path_state(path: Path | None) -> str:
    if path is None:
        return "unknown"
    try:
        return "present" if path.exists() else "missing"
    except OSError:
        return "unknown"


def _enabled_platform_names(config: Any) -> list[str]:
    platforms = _platform_configs(config)
    names = [name for name, platform_config in platforms.items() if _is_enabled(platform_config)]
    return sorted(names)


def _platform_configs(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        platforms = config.get("platforms", {})
        if isinstance(platforms, Mapping):
            return {str(_platform_name(k)): v for k, v in platforms.items()}
        return {}
    platforms = getattr(config, "platforms", {})
    if isinstance(platforms, Mapping):
        return {str(_platform_name(k)): v for k, v in platforms.items()}
    return {}


def _platform_name(value: Any) -> str:
    if isinstance(value, Platform):
        return value.value
    return str(getattr(value, "value", value))


def _platform_config(config: Any, name: str) -> Any:
    return _platform_configs(config).get(name)


def _is_enabled(platform_config: Any) -> bool:
    if isinstance(platform_config, Mapping):
        return platform_config.get("enabled") is True
    return getattr(platform_config, "enabled", False) is True


def _extra(platform_config: Any) -> Mapping[str, Any]:
    if isinstance(platform_config, Mapping):
        extra = platform_config.get("extra", {})
    else:
        extra = getattr(platform_config, "extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _root_mapping(config: Any, key: str) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        value = config.get(key, {})
        return value if isinstance(value, Mapping) else {}
    value = getattr(config, key, {})
    return value if isinstance(value, Mapping) else {}


def _discord_allowlist_state(config: Any, adapters: Mapping[Any, Any]) -> str:
    adapter = _adapter_for(adapters, "discord")
    if adapter is not None:
        users = getattr(adapter, "_allowed_user_ids", set()) or set()
        roles = getattr(adapter, "_allowed_role_ids", set()) or set()
        if users or roles:
            return "present"

    discord_cfg = _platform_config(config, "discord")
    extra = _extra(discord_cfg)
    root_discord = _root_mapping(config, "discord")
    root_gateway = _root_mapping(config, "gateway")

    candidates = (
        extra.get("allowed_users"),
        extra.get("allowed_user_ids"),
        extra.get("allowed_roles"),
        extra.get("allowed_role_ids"),
        extra.get("allowed_channels"),
        root_discord.get("allowed_users"),
        root_discord.get("allowed_channels"),
        root_gateway.get("allowed_users"),
        root_gateway.get("allowed_channels"),
    )
    if any(_explicit_list(value) for value in candidates):
        return "present"
    return "not present"


def _webhook_policy_state(config: Any, adapters: Mapping[Any, Any]) -> str:
    adapter = _adapter_for(adapters, "webhook")
    if adapter is not None:
        routes = getattr(adapter, "_routes", None)
        global_secret = getattr(adapter, "_global_secret", "")
        if isinstance(routes, Mapping):
            return "present" if _strict_webhook_routes(routes, global_secret) else "not present"

    webhook_cfg = _platform_config(config, "webhook")
    extra = _extra(webhook_cfg)
    routes = extra.get("routes")
    if isinstance(routes, Mapping):
        return "present" if _strict_webhook_routes(routes, extra.get("secret", "")) else "not present"

    root_webhook = _root_mapping(config, "webhook")
    if root_webhook:
        return "present" if _strict_webhook_root(root_webhook) else "not present"
    return "not present"


def _strict_webhook_routes(routes: Mapping[Any, Any], global_secret: Any = "") -> bool:
    if not routes:
        return False
    for route_name, route_config in routes.items():
        if not isinstance(route_config, Mapping):
            return False
        if not _explicit_route_name(route_name):
            return False
        secret = route_config.get("secret", global_secret)
        if not _configured_secret(secret):
            return False
        if route_config.get("require_signature") is not True:
            return False
        if route_config.get("signature_algorithm") != "hmac-sha256":
            return False
        if not route_config.get("signature_header") or not route_config.get("timestamp_header"):
            return False
        if not _positive_int(route_config.get("replay_window_seconds")):
            return False
    return True


def _strict_webhook_root(webhook: Mapping[str, Any]) -> bool:
    routes = webhook.get("allowed_routes")
    route_list = _as_list(routes)
    return (
        webhook.get("enabled") is True
        and webhook.get("require_signature") is True
        and webhook.get("signature_algorithm") == "hmac-sha256"
        and webhook.get("unsigned_requests", "deny") == "deny"
        and bool(route_list)
        and all(_explicit_route_name(route) for route in route_list)
        and bool(webhook.get("signature_header"))
        and bool(webhook.get("timestamp_header"))
        and _positive_int(webhook.get("replay_window_seconds"))
    )


def _browser_state(config: Any) -> str:
    browser = _root_mapping(config, "browser")
    if not browser:
        return "unknown"
    return "enabled" if browser.get("enabled") is True else "disabled"


def _broad_integrations_state(config: Any, enabled_platforms: list[str]) -> str:
    root = _root_mapping(config, "external_memory_providers")
    if root.get("enabled") is True:
        return "enabled"
    broad_platforms = [name for name in enabled_platforms if name not in CORE_DOBBY_PLATFORMS]
    return "enabled" if broad_platforms else "disabled"


def _memory_boundary_state(config: Any) -> str:
    memory = _root_mapping(config, "memory")
    honcho = _root_mapping(config, "honcho")
    external = _root_mapping(config, "external_memory_providers")
    if not memory and not honcho and not external:
        return "unknown"

    memory_on = (
        memory.get("memory_enabled") is True
        or memory.get("user_profile_enabled") is True
        or bool(str(memory.get("provider", "")).strip())
    )
    honcho_on = any(value not in (False, None, "", [], {}) for value in honcho.values())
    external_on = external.get("enabled") is True

    if memory_on or honcho_on or external_on:
        enabled = []
        if memory_on:
            enabled.append("memory")
        if honcho_on:
            enabled.append("Honcho")
        if external_on:
            enabled.append("external memory")
        return "enabled (" + ", ".join(enabled) + ")"
    return "native off; Honcho off"


def _readiness_state(
    *,
    package_state: str,
    home_state: str,
    discord_allowlist: str,
    webhook_policy: str,
    browser_state: str,
    broad_state: str,
    memory_state: str,
) -> str:
    hard_not_ready = {
        package_state == "missing",
        home_state == "missing",
        discord_allowlist == "not present",
        webhook_policy == "not present",
        browser_state == "enabled",
        broad_state == "enabled",
        memory_state.startswith("enabled"),
    }
    if any(hard_not_ready):
        return "not ready"
    if "unknown" in {package_state, home_state, browser_state, memory_state}:
        return "unknown"
    return "ready"


def _adapter_for(adapters: Mapping[Any, Any], platform_name: str) -> Any | None:
    for platform, adapter in adapters.items():
        if _platform_name(platform) == platform_name:
            return adapter
    return None


def _explicit_list(value: Any) -> bool:
    values = _as_list(value)
    return bool(values) and all(_is_explicit(value) for value in values)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _is_explicit(value: str) -> bool:
    text = str(value).strip()
    return bool(text) and text.lower() not in BROAD_VALUES and not (text.startswith("<") and text.endswith(">"))


def _explicit_route_name(value: Any) -> bool:
    text = str(value).strip()
    return bool(text) and "*" not in text and text not in {"/", "/*"}


def _configured_secret(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text != "INSECURE_NO_AUTH" and not (text.startswith("<") and text.endswith(">"))


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _redact_output(text: str) -> str:
    return redact_text(text, {})
