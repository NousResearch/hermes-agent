"""Where a ``/handoff`` lands: the profile's home channel on a platform the gateway delivers to.

One check for every surface that queues a handoff (the classic CLI's own store, the shared owner's
``handoff.request``), run before the row is written: a platform the watcher could never deliver to
would otherwise sit ``pending`` until the client gives up. The watcher re-resolves the live transport
before dispatch (``GatewayStartupMixin._handoff_resolve_destination``).
"""
from gateway import config as gateway_config
from gateway.config import HomeChannel, Platform


class HandoffTargetError(ValueError):
    """``reason`` is ``unknown_platform``, ``platform_not_enabled`` or ``no_home_channel``."""

    def __init__(self, reason: str, platform: str):
        super().__init__(reason)
        self.reason, self.platform = reason, platform


def _relay_fronts(config, platform_name: str) -> bool:
    """A relay-fronted gateway has only a RELAY block, yet the platforms it fronts are deliverable."""
    relay = config.platforms.get(Platform.RELAY)
    if not (relay and relay.enabled):
        return False
    from gateway.relay import relay_platform_identities
    return platform_name in {name for name, _bot in relay_platform_identities()}


def resolve_handoff_home(platform_name: str) -> HomeChannel:
    """The current profile's home channel for *platform_name*; raises ``HandoffTargetError``.
    Reads the scoped profile's config, so the caller binds the owning profile first."""
    try:
        platform = Platform(platform_name)
    except ValueError:
        raise HandoffTargetError('unknown_platform', platform_name) from None
    config = gateway_config.load_gateway_config()
    native = config.platforms.get(platform)
    if not (native and native.enabled) and not _relay_fronts(config, platform_name):
        raise HandoffTargetError('platform_not_enabled', platform_name)
    home = config.get_home_channel(platform)
    if not home or not home.chat_id:
        raise HandoffTargetError('no_home_channel', platform_name)
    return home
