"""Check-time delivery roots for native media attachment.

Split from :mod:`gateway.platforms.base` (a facade whose file size is
ratcheted): the roots that let the gateway attach a file the agent produced
live here.

Two kinds of directory may back an attached file:

* the transient media caches (``cache/images``, ``cache/audio``, ``cache/videos``,
  ``cache/documents``, ``cache/screenshots``, ``cache/generated``), which the
  hourly housekeeping sweeps once they go idle; and
* ``<home>/deliverables`` — the durable home for user-facing output (#126419).

A deliverable is the one a human is meant to keep, so nothing prunes its tree:
scratch and the media caches are swept, and a chart parked there is gone a day
after its last write. Because the sweep does not touch ``deliverables/``, the
delivery allowlist has to vouch for it by name — on a strict gateway with
recency trust off, an allowlisted root is the only thing that still lets a
day-old file attach.

Every profile home (``<root>/profiles/<name>``) gets its own deliverables dir.
These roots are enumerated at check time so a profile created after startup
counts, and they are allowlisted BEFORE the ``/root`` denylist, which otherwise
wins when HERMES_HOME is symlinked under a denied prefix (#31733).
"""

from __future__ import annotations

from pathlib import Path

# Durable, never-pruned home for user-facing output, relative to a Hermes home.
DELIVERABLES_SUBDIR = "deliverables"


def deliverables_root(home: Path) -> Path:
    """The deliverables dir of one Hermes home (``<home>/deliverables``)."""
    return home / DELIVERABLES_SUBDIR


def profile_delivery_roots() -> list[Path]:
    """Every check-time delivery root outside the import-time safe roots.

    Per-profile cache subdirs, each profile's durable deliverables dir, the
    launch home's deliverables dir, and every home's connector downloads dir.
    """
    from gateway.platforms.base import _MEDIA_DELIVERY_CACHE_SUBDIRS, _profile_dirs
    from hermes_constants import CONNECTOR_DOWNLOADS_SUBDIR, get_hermes_home

    profiles = _profile_dirs()
    homes = (get_hermes_home(), *profiles)
    return [
        *(p / "cache" / subdir for p in profiles for subdir in _MEDIA_DELIVERY_CACHE_SUBDIRS),
        *(deliverables_root(p) for p in homes),
        *(p / CONNECTOR_DOWNLOADS_SUBDIR for p in homes),
    ]
