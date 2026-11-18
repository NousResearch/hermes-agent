"""Which webhook ``deliver`` targets the gateway can dispatch: one side-effect-free check shared by the
webhook adapter (route load) and ``hermes webhook`` (subscribe / list). Plugin discovery stays with the caller."""

from contextlib import suppress

BUILTIN_DELIVER_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "signal", "sms", "whatsapp", "matrix", "mattermost",
    "email", "dingtalk", "feishu", "wecom", "wecom_callback", "weixin",
    "bluebubbles", "qqbot"})
_NON_PLATFORM_TARGETS = frozenset({"log", "github_comment"})


def is_known_platform(name: str) -> bool:
    """Cross-platform delivery target: built-in names or plugin-registered platforms."""
    if name in BUILTIN_DELIVER_PLATFORMS:
        return True
    with suppress(Exception):
        from gateway.platform_registry import platform_registry
        return platform_registry.is_registered(name)
    return False


def is_dispatchable_target(target: object) -> bool:
    """A saved route's ``deliver`` value the gateway can send to (a hand-edited non-string is not)."""
    return isinstance(target, str) and (target in _NON_PLATFORM_TARGETS or is_known_platform(target))
