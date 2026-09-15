"""Delivery policy for engine diagnostics, never assistant or command responses."""

from gateway.display_config import resolve_display_setting


# Lifecycle is an older mixed progress/diagnostic rail. Match only engine-owned
# diagnostic headlines here; ordinary heartbeats and compression progress survive.
_DIAGNOSTIC_PREFIXES = (
    "⚠", "❌", "🚫", "↻", "Content filter terminated stream",
    "🔌 Detected stale connections", "✅ Primary model restored",
    "⏳ Retrying", "⏱️ Rate limited", "⏱️ Provider overloaded",
    "🔐 Authentication failed", "⏳ Your Nous account",
    "ℹ️ Estimated cost of these empty attempts",
    "📐 Compression could not reduce the request further",
)


def is_warning_status(event_type: str, message: str) -> bool:
    return event_type == "warn" or (
        event_type == "lifecycle" and str(message or "").lstrip().startswith(_DIAGNOSTIC_PREFIXES)
    )


def warning_notifications_enabled(platform, user_config=None) -> bool:
    """Use the turn snapshot when supplied, otherwise the active profile's effective config.

    Programmatic/local surfaces keep their diagnostic stream. Unknown values never
    silently opt an operator out; null inherits via the canonical display resolver.
    """
    from gateway.run import _gateway_surface_passes_raw_text, _load_gateway_config

    if _gateway_surface_passes_raw_text(platform):
        return True
    if user_config is None:
        user_config = _load_gateway_config()
    platform_key = getattr(platform, "value", platform)
    return resolve_display_setting(user_config, platform_key, "warning_notifications", True)
