"""Inbound attachment re-homing for multiplexed gateways (moved out of ``gateway/run_inbound.py``)."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from gateway.platforms.event import MessageEvent

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


def rehome_inbound_media(event: MessageEvent) -> None:
    """Rehome adapter-cached attachments into the active profile's cache.

    Quoted images share a cache entry with other prepared inputs. Copy them so
    those inputs retain their source file. Authored attachments move into the
    routed profile. The caller binds that profile before media preprocessing.
    """
    if not event.media_urls:
        return
    from tools.credential_files import to_agent_visible_cache_path
    rewritten = list(event.media_urls)
    quoted = {dependency.media_index for dependency in event._quoted_media_dependencies}
    for i, raw in enumerate(event.media_urls):
        src, dest = Path(raw), Path(rehomed_media_path(raw))
        if dest == src:
            continue
        try:
            if not src.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if i not in quoted:
                shutil.move(str(src), str(dest))
            else:
                with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f".{dest.name}.", delete=False) as file:
                    temporary = Path(file.name)
                try:
                    shutil.copy2(src, temporary)
                    os.replace(temporary, dest)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not rehome inbound attachment %s into the routed profile's cache", raw, exc_info=True)
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
