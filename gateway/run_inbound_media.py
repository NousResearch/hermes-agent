"""Inbound attachment re-homing for multiplexed gateways (moved out of ``gateway/run_inbound.py``)."""

from __future__ import annotations

import dataclasses
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
    failed: set[int] = set()
    quoted = {dependency.media_index for dependency in event._quoted_media_dependencies}
    for i, raw in enumerate(event.media_urls):
        src, dest = Path(raw), Path(rehomed_media_path(raw))
        if dest == src:
            continue
        try:
            if not src.is_file():
                failed.add(i)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            if i not in quoted:
                shutil.move(str(src), str(dest))
            else:
                with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".inbound-", delete=False) as file:
                    temporary = Path(file.name)
                try:
                    shutil.copy2(src, temporary)
                    os.replace(temporary, dest)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            failed.add(i)
            logger.warning("Could not rehome inbound attachment %s into the routed profile's cache", raw, exc_info=True)
            continue
        rewritten[i] = str(dest)
        if event.text and raw in event.text:  # note an adapter already baked in (observed/replied media)
            event.text = event.text.replace(raw, to_agent_visible_cache_path(str(dest)))
    transfers = {raw: rewritten[index] for index, raw in enumerate(event.media_urls) if index not in failed}
    _rehome_pending_provenance(event, transfers)
    if not failed:
        event.media_urls = rewritten
        return

    retained = [index for index in range(len(rewritten)) if index not in failed]
    positions = {index: position for position, index in enumerate(retained)}
    if event.text:
        for index in failed:
            event.text = event.text.replace(event.media_urls[index], "[attachment unavailable]")
    event.media_urls = [rewritten[index] for index in retained]
    event.media_types = [event.media_types[index] for index in retained if index < len(event.media_types)]
    event.media_text_inlined = [
        event.media_text_inlined[index] if index < len(event.media_text_inlined) else None
        for index in retained
    ]
    event._quoted_media_dependencies = tuple(
        dataclasses.replace(dependency, media_index=positions[dependency.media_index])
        for dependency in event._quoted_media_dependencies
        if dependency.media_index in positions
    )


def _rehome_pending_provenance(event: MessageEvent, paths: dict[str, str]) -> None:
    from tools.credential_files import to_agent_visible_cache_path

    if event._pending_native_input is not None:
        event._pending_native_input = event._pending_native_input.rehome_attachments(paths)
    for part, _merge in event._merged_parts:
        for original in part.media_urls:
            if original in paths and original in part.text:
                part.text = part.text.replace(original, to_agent_visible_cache_path(paths[original]))
        part.media_urls = [paths.get(path, path) for path in part.media_urls]
        _rehome_pending_provenance(part, paths)


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


def _build_media_placeholder(event: MessageEvent) -> str:
    """Render media-only pending input in the active profile."""
    from gateway.run import _event_media_is_image, _event_media_is_audio, _event_media_is_video
    from tools.credential_files import to_agent_visible_cache_path

    if not event.media_urls:
        return ""

    rehome_inbound_media(event)
    if not event.media_urls:
        return "[attachment unavailable]"

    parts: list[str] = []
    for i, raw in enumerate(event.media_urls):
        url = to_agent_visible_cache_path(raw)
        if _event_media_is_image(event, i):
            parts.append(f"[User sent an image: {url}]")
        elif _event_media_is_audio(event, i):
            parts.append(f"[User sent audio: {url}]")
        elif _event_media_is_video(event, i):
            parts.append(f"[User sent a video: {url}]")
        else:
            parts.append(f"[User sent a file: {url}]")
    return "\n".join(parts)
