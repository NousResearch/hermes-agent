"""Which webhook ``deliver`` targets the gateway can dispatch: one side-effect-free check shared by the
webhook adapter (route load) and ``hermes webhook`` (subscribe / list). Plugin discovery stays with the caller."""

from contextlib import suppress

BUILTIN_DELIVER_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "signal", "sms", "whatsapp", "matrix", "mattermost",
    "email", "dingtalk", "wecom", "wecom_callback", "weixin",
    "bluebubbles", "qqbot"})
_NON_PLATFORM_TARGETS = frozenset({"log", "github_comment"})


def is_known_platform(name: str) -> bool:
    """Cross-platform delivery target: built-in names, plugin-registered platforms, or a platform that left
    core for a catalog plugin. A route saved while core shipped it stays loaded: a secondary profile's plugin
    is not in the launch registry this checks, and a skipped route is not re-read until its file changes.
    Without the plugin, the send fails naming the install (webhook ``_deliver_cross_platform``)."""
    if name in BUILTIN_DELIVER_PLATFORMS:
        return True
    from hermes_cli.left_core_migration import LEFT_CORE
    if any(f.platform == name for f in LEFT_CORE):
        return True
    with suppress(Exception):
        from gateway.platform_registry import platform_registry
        return platform_registry.is_registered(name)
    return False


def is_dispatchable_target(target: object) -> bool:
    """A saved route's ``deliver`` value the gateway can send to (a hand-edited non-string is not)."""
    return isinstance(target, str) and (target in _NON_PLATFORM_TARGETS or is_known_platform(target))
