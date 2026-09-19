"""Public, content-addressed copies of reviewed binary inputs."""
from __future__ import annotations

import json
from pathlib import Path
import re

from pm.downloader import Source

_LAYOUT = json.loads(Path(__file__).with_name("artifact-mirror.json").read_text(encoding="utf-8"))
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
    return Source(url, dest, sha256, fallbacks=() if url == archive else (archive,))
