"""Configured memory bounds for inbound gateway media."""


def get_inbound_media_max_bytes() -> int:
    """Max inbound media bytes held in memory (``gateway.max_inbound_media_bytes``);
    ``0`` / negative / unparseable disables the cap; unreadable config → default."""
    from gateway.platforms.base import _or_default, _config_section, DEFAULT_INBOUND_MEDIA_MAX_BYTES

    return _or_default(lambda: int(_config_section("gateway")["max_inbound_media_bytes"]),
                       DEFAULT_INBOUND_MEDIA_MAX_BYTES, (KeyError, TypeError, ValueError))
