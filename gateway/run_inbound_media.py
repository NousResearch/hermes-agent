"""Inbound attachment re-homing for multiplexed gateways (moved out of ``gateway/run_inbound.py``)."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from gateway.platforms.event import MessageEvent

# Log-record parity with the origin module.
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
    from hermes_constants import get_hermes_home, get_routing_process_hermes_home, hermes_home_key
    active, launch = Path(get_hermes_home()), Path(get_routing_process_hermes_home())
    if hermes_home_key(active) == hermes_home_key(launch):
        return
    from tools.credential_files import to_agent_visible_cache_path
    rewritten = list(event.media_urls)
    for i, raw in enumerate(event.media_urls):
        src = Path(raw)
        try:
            rel = src.relative_to(launch / "cache")
        except ValueError:
            continue
        dest = active / "cache" / rel
        try:
            if not src.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        except OSError:
            logger.warning("Could not move inbound attachment %s into the routed profile's cache", raw, exc_info=True)
            continue
        rewritten[i] = str(dest)
        agent_path = to_agent_visible_cache_path(str(dest))
        if event.text and raw in event.text:  # note an adapter already baked in (observed/replied media)
            event.text = event.text.replace(raw, agent_path)
        # A queued voice note's transcript (``_gateway_pending_stt_text``) was built before this move
        # and carries the audio marker's old path; it replaces ``event.text`` for the turn.
        pending = getattr(event, "_gateway_pending_stt_text", None)
        old_agent_path = to_agent_visible_cache_path(raw)
        if pending and old_agent_path in pending:
            event._gateway_pending_stt_text = pending.replace(old_agent_path, agent_path)
    event.media_urls = rewritten


def with_attachment_notes(runner_cls, event: MessageEvent, text: str) -> str:
    """*text* with the path note a normal turn gives each attachment, for a reply consumed outside a
    turn (a clarify answer). STT voice is excluded: the caller already transcribed it into *text*.
    Images get a path note, not pre-analysis: the answer must not wait on a vision call."""
    rehome_inbound_media(event)
    if not event.media_urls:
        return text
    image_paths, _stt, audio_files, videos = runner_cls._classify_inbound_media(event, True)
    names = runner_cls._inbound_original_names(event)
    text = runner_cls._prepend_inbound_media_file_notes(text, audio_files, videos, names)
    text = runner_cls._prepend_inbound_document_notes(event, text)
    for path in reversed(image_paths):
        _display, agent_path = runner_cls._inbound_attachment_display_name(path, names.get(path, ""))
        text = (f"[The user sent an image. It is saved at: {agent_path}. Look at it with vision_analyze "
                f"(image_url: {agent_path}) when the answer depends on it.]\n\n{text}")
    return text.strip()
