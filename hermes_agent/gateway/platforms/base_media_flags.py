"""Per-attachment inlining flags shared by inbound media transformations."""
from typing import Optional

from gateway.platforms.event import MessageEvent


def pad_inline_flags(event: MessageEvent) -> list[Optional[bool]]:
    """Represent unspecified legacy flags as None before appending or filtering media."""
    flags = list(getattr(event, "media_text_inlined", []) or [])
    return flags + [None] * (len(event.media_urls) - len(flags))
