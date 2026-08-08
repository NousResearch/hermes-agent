"""
Platform routing helpers extracted from ``gateway/run.py``.

These functions answer "which platform is this / where does this go" questions
used by the gateway runner and the ``/sethome`` slash command: normalizing a
platform identity, deciding which surfaces must keep raw text, mapping a
platform to its config key or home-channel env var, and checking whether a
token-authenticated platform actually has a usable credential.

They were extracted verbatim from the gateway god-file so the routing policy
lives in one focused, independently testable module. ``gateway/run.py``
re-exports every name for backward compatibility — external importers
(``gateway/slash_commands.py``, tests) keep working unchanged.
"""

from typing import Any, Dict, Optional

from gateway.config import Platform

# Surfaces that consume gateway text programmatically (CLI/TUI "local"
# diagnostics, API JSON, webhook payloads) and therefore must keep RAW
# status/error text. EVERY other platform is a human-facing chat surface
# where operational lifecycle/provider-error noise (and any secrets in it)
# must be suppressed or sanitized. Widens #28533's Telegram-only filter to
# all chat gateways (#39293). Fail-closed: unknown/empty platform -> chat.
_GATEWAY_RAW_TEXT_PLATFORMS = frozenset(
    {"local", "api_server", "webhook", "msgraph_webhook"}
)


def _gateway_platform_value(platform: Any) -> str:
    """Return a normalized gateway platform value for enums or raw strings."""
    return str(getattr(platform, "value", platform) or "").strip().lower()


def _gateway_surface_passes_raw_text(platform: Any) -> bool:
    """True only for programmatic/local surfaces that must keep raw text."""
    return _gateway_platform_value(platform) in _GATEWAY_RAW_TEXT_PLATFORMS


def _non_conversational_metadata(
    metadata: Optional[Dict[str, Any]] = None,
    *,
    platform: Any = None,
) -> Optional[Dict[str, Any]]:
    """Mark Discord lifecycle/status sends without changing other platforms."""
    if _gateway_platform_value(platform) != "discord":
        return metadata
    merged = dict(metadata or {})
    merged["non_conversational"] = True
    return merged


def _home_target_env_var(platform_name: str) -> str:
    """Return the configured home-target env var for a platform.

    Consults built-in ``_HOME_TARGET_ENV_VARS`` first, then the plugin
    registry via ``cron.scheduler._resolve_home_env_var``, then falls back
    to ``<PLATFORM>_HOME_CHANNEL`` for unknown names.
    """
    from cron.scheduler import _resolve_home_env_var

    resolved = _resolve_home_env_var(platform_name)
    if resolved:
        return resolved
    return f"{platform_name.upper()}_HOME_CHANNEL"


def _home_thread_env_var(platform_name: str) -> str:
    """Return the optional thread/topic env var for a platform home target."""
    return f"{_home_target_env_var(platform_name)}_THREAD_ID"


def _platform_has_bot_credential(platform: "Platform", platform_config: "PlatformConfig") -> bool:
    """Return True when a token-authenticated platform has a usable bot credential.

    Platforms that do not use ``PlatformConfig.token`` always return True so we
    never skip them here (Signal session paths, port-binding HTTP adapters, etc.).
    """
    from gateway.config import PLATFORM_TOKEN_ENV_NAMES

    if platform not in PLATFORM_TOKEN_ENV_NAMES:
        return True
    token = getattr(platform_config, "token", None) or ""
    if isinstance(token, str) and token.strip():
        return True
    # Some adapters also accept api_key as the primary credential.
    api_key = getattr(platform_config, "api_key", None) or ""
    if isinstance(api_key, str) and api_key.strip():
        return True
    return False


def _platform_config_key(platform: "Platform") -> str:
    """Map a Platform enum to its config.yaml key (LOCAL→"cli", rest→enum value)."""
    return "cli" if platform == Platform.LOCAL else platform.value
