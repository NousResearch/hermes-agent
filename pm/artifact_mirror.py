"""Public, content-addressed copies of reviewed binary inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlsplit

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


def _github_layout() -> dict:
    # Read at call time: setup-hermes.sh (awk) and the nix expressions read the R2
    # keys of this same file, and older checkouts have no "github" object.
    try:
        return _LAYOUT["github"]
    except KeyError:
        raise ValueError("pm/artifact-mirror.json has no \"github\" layout") from None


def github_repository() -> str:
    return _github_layout()["repository"]


def github_release_tag(sha256: str) -> str:
    """Name the existing release without creating a tag or changing /releases/latest."""
    object_key(sha256)
    return _github_layout()["release_tag"]


def github_asset_url(sha256: str) -> str:
    return (f"https://github.com/{github_repository()}/releases/download/"
            f"{github_release_tag(sha256)}/{sha256}")


def wheel_source(filename: str, sha256: str, dest: Path) -> Source:
    """The existing release is read-only; the pinned R2 object is its fallback."""
    release = (f"https://github.com/{github_repository()}/releases/download/"
               f"{github_release_tag(sha256)}/{quote(filename, safe='')}")
    return Source(release, dest, sha256, fallbacks=(mirror_url(sha256),))


def pinned_source(url: str, dest: Path, sha256: str) -> Source:
    archive = mirror_url(sha256)
    # Loopback pins are local test/development inputs, not published artifacts.
    # Keep them local instead of trying a public release (or requiring its layout).
    if urlsplit(url).hostname in ("127.0.0.1", "localhost", "::1"):
        return Source(url, dest, sha256, fallbacks=() if url == archive else (archive,))
    release = github_asset_url(sha256)
    # A configured npm registry remains first (#123132). The existing release
    # is read-only; R2 holds the same pinned bytes when the release lacks an asset.
    registry_url = npm_registry_url(url, os.environ)
    candidates = (registry_url, release, archive, url) if registry_url != url else (release, archive, url)
    first, *fallbacks = dict.fromkeys(candidates)
    return Source(first, dest, sha256, fallbacks=tuple(fallbacks))
