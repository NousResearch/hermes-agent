"""Delivery policy for engine diagnostics, never assistant or command responses."""

from gateway.display_config import resolve_display_setting


class DiagnosticText(str):
    """Producer-owned classification on the legacy two-argument status callback.

    String behavior and event kind stay unchanged for existing plugin renderers.
    Durable carriers must serialize their own category, not this in-memory marker.
    """


def is_warning_status(event_type: str, message: str) -> bool:
    return event_type == "warn" or isinstance(message, DiagnosticText)


def is_diagnostic_notice(notice) -> bool:
    """Out-of-band ``AgentNotice`` classification shared by the gateway, TUI and CLI sinks.

    Credit-service notices are automatic diagnostics at every level (usage bands are ``info``,
    ``credits.restored`` is ``success``); everything else is diagnostic only when it warns.
    """
    return (getattr(notice, "level", None) in {"warn", "error"}
            or str(getattr(notice, "key", "") or "").startswith("credits."))


def diagnostic_metadata(event) -> dict:
    """``{"notification_category": "diagnostic"}`` for a trusted diagnostic-only wake, else ``{}``.

    Only internal wakes may classify a turn; human content never does.
    """
    if getattr(event, "internal", False) and (getattr(event, "metadata", None) or {}).get(
            "notification_category") == "diagnostic":
        return {"notification_category": "diagnostic"}
    return {}


def effective_user_config() -> dict:
    """The active profile's effective config, or ``{}`` when it cannot be read (presentation fails open)."""
    from hermes_cli.config_effective import load_user_config_effective
    try:
        config = load_user_config_effective()
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def diagnostic_turn_muted(display_metadata, platform, user_config=None) -> bool:
    """One admission rule for every surface: a diagnostic-category wake mutes its turn's
    presentation only when the owning policy hides diagnostics. Human content never mutes."""
    return ((display_metadata or {}).get("notification_category") == "diagnostic"
            and not warning_notifications_enabled(platform, user_config))


def diagnostic_wake_muted(event, user_config=None) -> bool:
    """Only trusted diagnostic-only wakes can mute a turn, never human content."""
    snapshot = getattr(event, "_notification_reply_muted", None)
    if isinstance(snapshot, bool):
        return snapshot
    return bool(getattr(event, "internal", False)) and diagnostic_turn_muted(
        getattr(event, "metadata", None), event.source.platform, user_config)


def render_notification(render, *, platform, diagnostic=True, user_config=None) -> bool:
    """Invoke a synchronous UI renderer only when its classified content is visible.

    Return whether the renderer ran, not whether a transport delivered anything.
    Call only at a presentation sink, never around producer callbacks or persistence.
    The caller supplies the owning scope/turn snapshot; exceptions remain its policy.
    """
    if diagnostic and not warning_notifications_enabled(platform, user_config):
        return False
    render()
    return True


async def present_notification(present, *, platform, diagnostic=True, user_config=None) -> bool:
    """Async twin of :func:`render_notification` for lifecycle emitters that await a send.

    Return whether the presenter ran; its own receipt/exception is the caller's to interpret.
    The caller binds the owning profile scope BEFORE calling (watchers/shutdown fan-outs).
    """
    if diagnostic and not warning_notifications_enabled(platform, user_config):
        return False
    await present()
    return True


def warning_notifications_enabled(platform, user_config=None) -> bool:
    """Use the turn snapshot when supplied, otherwise the active profile's effective config.

    No surface exemption: direct command/API outcomes are not notifications.
    Unknown values never opt in; null inherits via the canonical display resolver.
    """
    if user_config is None:
        user_config = effective_user_config()
    elif not isinstance(user_config, dict):
        user_config = {}
    platform_key = getattr(platform, "value", platform)
    return not resolve_display_setting(user_config, platform_key, "suppress_warning_notifications", False)
