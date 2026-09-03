"""Re-home inbound attachments into the routed profile's cache.

Platform adapters download and cache an inbound attachment *before* the
gateway knows which multiplexed profile owns the turn, so the bytes land in
the cache of the process/launch Hermes home.  The turn then runs inside
``_profile_runtime_scope``, where ``get_cache_directory_mounts()`` — and the
sandbox bind mounts built from it — resolve the *profile's* cache instead.

The two halves disagree in the worst possible way: the agent is handed a path
that exists, is mounted, and is empty, so it reports "file not found" and asks
for a re-send that lands in the same unmounted directory.  Host-side vision
reads fail the same way from the other end, because
``tools/image_source._media_cache_roots()`` only authorises paths under the
*active* home (#101134).

This module moves those files into the active profile's matching cache
directory at the one point where the routed profile is known and no media
consumer (vision, STT, document notes) has run yet.  Single-profile gateways
resolve the same home twice and return before touching disk.

Path form: entries are rewritten to **host** paths.  That is the coordinate
system every host-side consumer already needs (STT opens the file, vision
resolves it against the cache roots), and the document-note builder in
``gateway/run.py`` performs the single host→sandbox translation under the
routed scope.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _is_existing_file(path: Path) -> bool:
    """``path.is_file()`` that survives an unreadable parent directory.

    ``pathlib`` swallows ``ENOENT``/``ENOTDIR`` but not ``EACCES``, and the
    sandbox-form paths this module inspects start at ``/root`` — mode 0700 for
    a gateway that does not run as root.  Probing one would raise instead of
    answering ``False``.
    """
    try:
        return path.is_file()
    except OSError:
        return False


def _source_host_path(
    entry: str,
    source_mounts: Sequence[Dict[str, str]],
) -> Optional[Path]:
    """Return the host file named by *entry*, or ``None`` if it is not one.

    Accepts both coordinate systems: a host path (what the cache primitives
    return) and a sandbox path minted by ``to_agent_visible_cache_path`` under
    the source home (what ``CachedMedia`` used to carry, and what an
    out-of-tree adapter may still push into ``media_urls``).
    """
    if not entry:
        return None
    candidate = Path(entry)
    if _is_existing_file(candidate):
        return candidate
    posix_entry = PurePosixPath(entry)
    for mount in source_mounts:
        try:
            rel = posix_entry.relative_to(mount["container_path"])
        except ValueError:
            continue
        translated = Path(mount["host_path"]) / Path(str(rel))
        if _is_existing_file(translated):
            return translated
    return None


def _relocate(source: Path, destination: Path) -> None:
    """Move *source* onto *destination*, falling back to copy across devices."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
        return
    except OSError:
        # Cross-device rename, or a Windows handle still open on the source.
        shutil.copy2(source, destination)
    try:
        source.unlink()
    except OSError:
        # The copy landed; leaving the original behind is untidy, not broken.
        logger.debug("inbound media source not removable: %s", source, exc_info=True)


def _rehome_one(
    entry: str,
    source_mounts: Sequence[Dict[str, str]],
    destination_roots: Dict[str, str],
) -> Optional[str]:
    """Return the relocated host path for *entry*, or ``None`` to keep it."""
    host_path = _source_host_path(entry, source_mounts)
    if host_path is None:
        return None
    for mount in source_mounts:
        try:
            rel = host_path.relative_to(Path(mount["host_path"]))
        except ValueError:
            continue
        destination_root = destination_roots.get(mount["container_path"])
        if not destination_root:
            return None
        root = Path(destination_root)
        destination = root / rel
        try:
            # ``rel`` comes from relative_to, so it cannot escape on its own;
            # a symlinked cache entry still can.
            destination.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            return None
        if _is_existing_file(destination):
            # Same file name already staged for this profile (a redelivered
            # event): keep the staged copy rather than clobbering it.
            return str(destination)
        _relocate(host_path, destination)
        return str(destination)
    return None


def rehome_media_paths(paths: Sequence[str]) -> List[str]:
    """Return *paths* with adapter-cached files moved into the active profile.

    Entries that are not adapter-cached files, or that already live under the
    active home's cache, are returned untouched — which also makes the whole
    pass idempotent.
    """
    from hermes_constants import (
        get_hermes_home,
        get_process_hermes_home,
        hermes_home_key,
    )
    from tools.credential_files import get_cache_directory_mounts

    active_home = Path(get_hermes_home())
    source_home = Path(get_process_hermes_home())
    if hermes_home_key(active_home) == hermes_home_key(source_home):
        return list(paths)

    source_mounts = get_cache_directory_mounts(home=source_home)
    destination_roots = {
        mount["container_path"]: mount["host_path"]
        for mount in get_cache_directory_mounts(home=active_home)
    }

    rewritten: List[str] = []
    for entry in paths:
        try:
            moved = _rehome_one(entry, source_mounts, destination_roots)
        except OSError:
            logger.warning(
                "Could not move inbound attachment into the routed profile "
                "cache; the agent may not be able to open it",
                exc_info=True,
            )
            moved = None
        rewritten.append(moved or entry)
    return rewritten


def _rewrite_baked_paths(
    event: Any,
    original: Sequence[str],
    rewritten: Sequence[str],
) -> None:
    """Repoint paths an adapter already interpolated into ``event.text``.

    ``CachedMedia.context_note()`` is appended to the observed-group
    transcript at ingress, naming the pre-move location.
    """
    text = getattr(event, "text", None)
    if not text:
        return
    from tools.credential_files import to_agent_visible_cache_path

    for old, new in zip(original, rewritten):
        if old == new or old not in text:
            continue
        text = text.replace(old, to_agent_visible_cache_path(new))
    event.text = text


def rehome_event_media(event: Any) -> None:
    """Move one inbound event's cached attachments into the active profile.

    Call from inside the routed profile's runtime scope, before any consumer
    reads ``event.media_urls`` or ``event.text``.
    """
    original = list(getattr(event, "media_urls", None) or [])
    if not original:
        return
    try:
        rewritten = rehome_media_paths(original)
    except Exception:
        # Never fail an inbound turn over cache placement: the worst case is
        # the pre-fix behaviour.
        logger.debug("inbound media rehome failed", exc_info=True)
        return
    if rewritten == original:
        return
    event.media_urls = rewritten
    _rewrite_baked_paths(event, original, rewritten)
