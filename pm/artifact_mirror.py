"""Public, content-addressed copies of reviewed binary inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re

from pm.downloader import Source
from pm.index_config import npm_registry_url

_LAYOUT = json.loads(Path(__file__).with_name("artifact-mirror.json").read_text(encoding="utf-8-sig"))
KEY_PREFIX = _LAYOUT["prefix"]
PUBLIC_PREFIX = _LAYOUT["origin"] + "/" + KEY_PREFIX


def object_key(sha256: str) -> str:
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise ValueError("A pinned input requires a full lowercase SHA256")
    return KEY_PREFIX + sha256


def mirror_url(sha256: str) -> str:
    object_key(sha256)
    return PUBLIC_PREFIX + sha256


def pinned_source(url: str, dest: Path, sha256: str) -> Source:
    archive = mirror_url(sha256)
    # The lock records registry.npmjs.org; a user's npm mirror serves the same pinned bytes (#123132).
    mirrored = npm_registry_url(url, os.environ)
    return Source(mirrored, dest, sha256,
                  fallbacks=() if mirrored == archive else (archive,),
                  # A closed network's registry is plain http as often as not; the user
                  # configured that origin, and the lock's SHA256 still verifies the bytes.
                  allow_plain_http=mirrored != url and mirrored.startswith("http://"))
