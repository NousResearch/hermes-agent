"""Public, content-addressed copies of reviewed binary inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote, urlsplit
import re

from pm.downloader import Source
from pm.index_config import npm_registry_url

_LAYOUT = json.loads(Path(__file__).with_name("artifact-mirror.json").read_text(encoding="utf-8-sig"))
KEY_PREFIX = _LAYOUT["prefix"]
PUBLIC_PREFIX = _LAYOUT["origin"] + "/" + KEY_PREFIX

# The Termux main pool deletes a package's previous archive whenever it
# rebuilds it (pm/termux_libs.py), so a pool pin rots silently. The Internet
# Archive mirrors pool/main by the same layout and keeps the retired file.
# Both the CI archiver and the client fallback ladder derive this URL from
# here, so the seed source and the client rung can never disagree.
_POOL_HOST = "packages.termux.dev"
_POOL_PREFIX = "/apt/termux-main/pool/main/"


def object_key(sha256: str) -> str:
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("A pinned input requires a full lowercase SHA256")
    return KEY_PREFIX + sha256


def mirror_url(sha256: str) -> str:
    object_key(sha256)
    return PUBLIC_PREFIX + sha256


def historical_url(url: str) -> str | None:
    """Internet Archive twin of a Termux pool archive, or None.

    Applies only to pool URLs: the archive keeps pool/main by
    (group, package, filename). Any other source returns None.
    """
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != _POOL_HOST or not parsed.path.startswith(_POOL_PREFIX):
        return None
    parts = parsed.path[len(_POOL_PREFIX):].split("/")
    if len(parts) != 3 or any(not part or part in (".", "..") for part in parts):
        return None
    group, package, filename = (quote(part, safe="") for part in parts)
    return f"https://archive.org/download/termux_pkgs_archive_{group}/{package}/{filename}"


def pinned_source(url: str, dest: Path, sha256: str) -> Source:
    archive = mirror_url(sha256)
    # The lock records registry.npmjs.org; a user's npm mirror serves the same pinned bytes (#123132).
    url = npm_registry_url(url, os.environ)
    fallbacks: list[str] = []
    if url != archive:
        fallbacks.append(archive)
        # A pool archive the pool itself dropped stays fetchable under the
        # same hash gate: the historical twin either serves the pinned bytes
        # (sha256 verifies) or it does not.
        historical = historical_url(url)
        if historical and historical not in fallbacks:
            fallbacks.append(historical)
    return Source(url, dest, sha256, fallbacks=tuple(fallbacks))
