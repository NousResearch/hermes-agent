"""Transfer adapter-cached attachments into the routed profile."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from gateway.platforms.event import MessageEvent

logger = logging.getLogger("gateway.run")


def rehome_inbound_media(event: MessageEvent) -> None:
    """Move adapter-cached attachments into the ACTIVE profile's ``cache/`` and repoint the event.

    Adapters download and cache an attachment BEFORE the gateway routes the event to a profile, so
    on a multiplexed gateway the file lands under the launch home while the routed turn's sandbox
    mounts (``get_cache_directory_mounts``) and vision's ``_media_cache_roots`` resolve the routed
    profile's ``cache/`` — the agent is handed a mounted, empty directory (#101134). Runs inside the
    routed scope at the shared preprocessing choke point (every adapter, every media kind); a no-op
    when the active home is the launch home, and idempotent (a moved entry is no longer under it).
    """
    if not event.media_urls:
        return
    from tools.credential_files import to_agent_visible_cache_path
    rewritten = list(event.media_urls)
    for i, raw in enumerate(event.media_urls):
        src, dest = Path(raw), Path(rehomed_media_path(raw))
        if dest == src:
            continue
        try:
            if not src.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        except OSError:
            logger.warning("Could not move inbound attachment %s into the routed profile's cache", raw, exc_info=True)
            continue
        rewritten[i] = str(dest)
        if event.text and raw in event.text:  # note an adapter already baked in (observed/replied media)
            event.text = event.text.replace(raw, to_agent_visible_cache_path(str(dest)))
    event.media_urls = rewritten


def rehomed_media_path(raw: str) -> str:
    """Return where ``rehome_inbound_media`` puts the attachment at *raw* for the active profile."""
    from hermes_constants import get_hermes_home, get_routing_process_hermes_home, hermes_home_key
    active, launch = Path(get_hermes_home()), Path(get_routing_process_hermes_home())
    if hermes_home_key(active) == hermes_home_key(launch):
        return raw
    try:
        return str(active / "cache" / Path(raw).relative_to(launch / "cache"))
    except ValueError:
        return raw
