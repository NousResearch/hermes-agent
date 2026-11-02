"""Configured memory bounds for inbound gateway media."""


def get_inbound_media_max_bytes() -> int:
    """Return the configured inbound media byte limit.

    Zero and negative values disable the limit. Missing, unreadable or
    unparseable configuration uses the default limit.
    """
    from gateway.platforms.base import _or_default, _config_section, DEFAULT_INBOUND_MEDIA_MAX_BYTES

    value = _or_default(lambda: int(_config_section("gateway")["max_inbound_media_bytes"]),
                        DEFAULT_INBOUND_MEDIA_MAX_BYTES, (KeyError, TypeError, ValueError))
    return 0 if value <= 0 else value
