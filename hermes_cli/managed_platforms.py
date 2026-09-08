"""Host-declared ownership of messaging channels.

A hosting layer that configures channels on the operator's behalf (the Nous
Portal for hosted agents) stamps the container with two deployment values:

    HERMES_MANAGED_PLATFORMS=telegram:native,discord:relay
    HERMES_MANAGED_PLATFORMS_LABEL=Nous Portal

The dashboard renders the listed channels read-only and refuses writes to
them. ``native`` means the host supplies the platform's own credentials;
``relay`` means the platform is fronted by the host's connector and the native
adapter is off by design. The link back to the host is
``HERMES_DASHBOARD_PORTAL_URL``, which hosted deployments already set.

These are internal deployment stamps, not user settings. Absent means no
channel is managed and every surface behaves as before.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Mapping, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

PLATFORMS_ENV = "HERMES_MANAGED_PLATFORMS"
LABEL_ENV = "HERMES_MANAGED_PLATFORMS_LABEL"
URL_ENV = "HERMES_DASHBOARD_PORTAL_URL"

KINDS = ("native", "relay")
DEFAULT_KIND = "native"
DEFAULT_LABEL = "your hosting provider"
MAX_LABEL_LENGTH = 64


@dataclass(frozen=True)
class ManagedPlatforms:
    """Platform ids owned by the host, keyed to their kind, plus display data."""

    platforms: Dict[str, str] = field(default_factory=dict)
    label: str = DEFAULT_LABEL
    url: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.platforms)

    def kind_of(self, platform_id: str) -> Optional[str]:
        return self.platforms.get(platform_id)

    def manages_relay(self) -> bool:
        return "relay" in self.platforms.values()

    def record_for(self, platform_id: str) -> Optional[dict]:
        kind = self.platforms.get(platform_id)
        if kind is None:
            return None
        return {"kind": kind, "label": self.label, "url": self.url}


def load_managed_platforms(environ: Optional[Mapping[str, str]] = None) -> ManagedPlatforms:
    env = os.environ if environ is None else environ
    return _parse(env.get(PLATFORMS_ENV, ""), env.get(LABEL_ENV, ""), env.get(URL_ENV, ""))


@lru_cache(maxsize=16)
def _parse(raw_platforms: str, raw_label: str, raw_url: str) -> ManagedPlatforms:
    """Cached per distinct stamp so a malformed value is logged once, not per request."""
    platforms = _parse_platforms(raw_platforms)
    if not platforms:
        return ManagedPlatforms()
    return ManagedPlatforms(
        platforms=platforms, label=_parse_label(raw_label), url=_parse_url(raw_url)
    )


def _parse_platforms(raw: str) -> Dict[str, str]:
    platforms: Dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip().lower()
        if not entry:
            continue
        platform_id, _, kind = entry.partition(":")
        platform_id = platform_id.strip()
        kind = kind.strip() or DEFAULT_KIND
        if not platform_id:
            continue
        if kind not in KINDS:
            logger.warning(
                "%s: unknown kind %r for platform %r, treating it as %s",
                PLATFORMS_ENV, kind, platform_id, DEFAULT_KIND,
            )
            kind = DEFAULT_KIND
        platforms.setdefault(platform_id, kind)
    return platforms


def _parse_label(raw: str) -> str:
    label = " ".join(raw.split())
    if not label:
        return DEFAULT_LABEL
    return label[:MAX_LABEL_LENGTH]


def _parse_url(raw: str) -> Optional[str]:
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return candidate
