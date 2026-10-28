"""Localize Relay attachments while preserving their parallel metadata."""
import logging
from typing import Optional

from gateway.platforms.base_media_flags import pad_inline_flags

logger = logging.getLogger(__name__)


async def localize_inbound_media(event, get_media_client) -> None:
    """Download connector re-hosted attachments to local temp paths — every NATIVE
    adapter presents inbound media as LOCAL FILE PATHS (vision/file tools consume
    paths). Best-effort per entry: a failed download drops that entry, never the
    message; with no client only re-host URLs are dropped (they'd 401 downstream)."""
    try:
        urls = list(getattr(event, "media_urls", None) or [])
        if not urls:
            return
        # Filter all per-attachment fields together so survivors keep their own semantics.
        types = list(getattr(event, "media_types", None) or []) + [""] * len(urls)
        names = list(getattr(event, "media_filenames", None) or []) + [""] * len(urls)
        flags = pad_inline_flags(event)
        client = get_media_client()
        localized: list[tuple[str, str, str, Optional[bool]]] = []
        for url, mime, name, inlined in zip(urls, types, names, flags):
            if not isinstance(url, str) or not url:
                continue
            if client is None:
                if "/relay/media/" not in url:
                    localized.append((url, mime, name, inlined))
                continue
            # The path is a random temp name; the original name rides beside it.
            got = await client.download_with_name(url, suggested_name=name or None)
            if got:
                localized.append((got[0], mime, got[1], inlined))
            elif "/relay/media/" not in url:
                # A public URL still has value as a URL; a dead re-host does not.
                localized.append((url, mime, name, inlined))
        event.media_urls, event.media_types, event.media_filenames, event.media_text_inlined = (
            [entry[i] for entry in localized] for i in range(4))
    except Exception:
        logger.debug("relay inbound media localization failed", exc_info=True)
