"""Validate and conditionally publish macOS update feeds."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable

from . import r2 as r2_module
from .semver import compare, is_valid_version

ARCHES = ("arm64", "x64")
_HASH_PATTERN = re.compile(r"^[A-Za-z0-9+/]{86}==$")
_TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+(?:-canary\.\d{14})?$")


def _darwin_feed(channel: str, light: bool = False) -> dict[str, Any]:
    """Read the same feed facts as the desktop runtime."""
    with (Path(__file__).resolve().parents[2] / "apps/desktop/update-feed.json").open(encoding="utf-8") as file:
        feeds = json.load(file)
    if channel not in feeds:
        raise TypeError(f"Unknown update channel: {channel}")
    return feeds[channel]["light" if light else "bundled"]


# ---------------------------------------------------------------------------
# Feed parsing / merging (pure)
# ---------------------------------------------------------------------------

def parse_mac_feed(text: str) -> dict[str, Any]:
    import hermes_yaml as yaml  # lazy: YAML support loads only on the feed path

    feed = yaml.safe_load(text)
    if (
        not isinstance(feed, dict)
        or not is_valid_version(str(feed.get("version", "")))
        or not isinstance(feed.get("files"), list)
        or not feed["files"]
    ):
        raise ValueError("Invalid macOS update feed")
    for file_entry in feed["files"]:
        if (
            not isinstance(file_entry, dict)
            or not isinstance(file_entry.get("url"), str)
            or not isinstance(file_entry.get("sha512"), str)
            or not _HASH_PATTERN.match(file_entry["sha512"])
            or not isinstance(file_entry.get("size"), int)
            or isinstance(file_entry.get("size"), bool)
            or file_entry["size"] <= 0
        ):
            raise ValueError("Invalid macOS artifact metadata")
    if feed.get("path") and not any(
        isinstance(f, dict)
        and f.get("url") == feed.get("path")
        and f.get("sha512") == feed.get("sha512")
        for f in feed["files"]
    ):
        raise ValueError("Legacy feed path/hash disagrees with files")
    return feed


_MAC_URL_PATTERN = re.compile(
    r"^/releases/tag/v[0-9A-Za-z.+-]+/[0-9A-Za-z._+-]+\.(zip|dmg)$"
)


def mac_feed_references(text: str) -> list[str]:
    """Every bucket key a published feed references (artifact + .blockmap)."""
    feed = parse_mac_feed(text)
    references: list[str] = []
    for file_entry in feed["files"]:
        # Only our immutable artifact namespace is eligible for publication
        # or pruning; anything else fails loudly instead of half-publishing.
        if not _MAC_URL_PATTERN.match(file_entry["url"]):
            raise ValueError(f"Invalid macOS artifact path: {file_entry['url']}")
        key = file_entry["url"][1:]
        references.extend([key, f"{key}.blockmap"])
    return references


def merge_mac_feeds(legs: dict[str, str], tag: str, light: bool = False) -> dict[str, Any]:
    """Validate both native legs, merge them, and rewrite artifact URLs into
    the immutable per-release tag namespace."""
    import hermes_yaml as yaml  # lazy

    version = tag[1:] if isinstance(tag, str) and tag.startswith("v") else ""
    if not is_valid_version(version) or not _TAG_PATTERN.match(tag):
        raise ValueError("Invalid macOS release tag")
    selection = _darwin_feed("canary" if "-canary." in version else "stable", light)
    expected = [f"{arch}-{selection['fileName']}" for arch in ARCHES]
    if sorted(legs.keys()) != sorted(expected):
        raise ValueError("Expected exactly one ARM64 and one x64 macOS feed")
    files: dict[str, dict[str, Any]] = {}
    first: dict[str, Any] | None = None
    for index, name in enumerate(expected):
        leg = parse_mac_feed(legs[name])
        if str(leg.get("version")) != version:
            raise ValueError(f"Feed version does not match {tag}")
        prefix = f"{'HermesLight' if light else 'HermesBundled'}-{version}-mac-{ARCHES[index]}"
        if not any(f.get("url") == f"{prefix}.zip" for f in leg["files"]):
            raise ValueError(f"Missing native ZIP for {ARCHES[index]}")
        for file_entry in leg["files"]:
            if file_entry.get("url") not in (f"{prefix}.zip", f"{prefix}.dmg"):
                raise ValueError(f"Wrong variant or architecture: {file_entry.get('url')}")
            rewritten = {**file_entry, "url": f"/releases/tag/{tag}/{file_entry['url']}"}
            prior = files.get(file_entry["url"])
            if prior is not None and prior != rewritten:
                raise ValueError(f"Conflicting artifact: {file_entry['url']}")
            files[file_entry["url"]] = rewritten
        if first is None:
            first = leg
    assert first is not None
    merged = {**first, "files": list(files.values())}
    if merged.get("path"):
        merged["path"] = f"/releases/tag/{tag}/{merged['path']}"
    text = yaml.safe_dump(merged, width=100000, default_flow_style=False, sort_keys=False)
    mac_feed_references(text)  # publish only a feed that parses back clean
    return {
        "key": f"{selection['directory']}/{selection['fileName']}",
        "text": text,
        "files": merged["files"],
        "version": version,
    }


# ---------------------------------------------------------------------------
# Publication (transport-verified)
# ---------------------------------------------------------------------------

def publish_mac_feed(
    plan: dict[str, Any],
    transport: dict[str, Callable[..., Any]],
) -> None:
    """The transport verifies immutable bytes and conditionally replaces one
    pointer. Downgrade rejection: semver.compare against the live feed."""
    read = transport["read"]
    verify = transport["verify"]
    write = transport["write"]
    live = read(plan["key"])
    if live:
        old_feed = parse_mac_feed(live["text"])
        next_feed = parse_mac_feed(plan["text"])
        order = compare(plan["version"], str(old_feed["version"]))
        if order < 0:
            raise ValueError("Refusing to move the macOS feed backward")
        if order == 0:
            if old_feed != next_feed:
                raise ValueError("Refusing to replace published version with different artifacts")
            return
    for file_entry in plan["files"]:
        verify(file_entry["url"][1:], file_entry)
    write(plan["key"], plan["text"], live["etag"] if live else None)
    published = read(plan["key"])
    if not published or published["text"] != plan["text"]:
        raise ValueError("macOS feed readback differs from publication")


# ---------------------------------------------------------------------------
# finalize (real signed transport)
# ---------------------------------------------------------------------------

def finalize(tag: str, dir: str, variant: str | None = None) -> None:
    """Validate both native legs, verify their streamed bytes, then replace
    the feed pointer (conditional write, then readback)."""
    if variant and variant != "light":
        raise ValueError("Unknown macOS variant")
    creds, base, bucket = r2_module.credentials()
    now = r2_module.amz_timestamp()
    import os

    legs = {
        name: open(os.path.join(dir, name), encoding="utf-8").read()
        for name in sorted(os.listdir(dir))
        if name.endswith("-mac.yml")
    }
    plan = merge_mac_feeds(legs, tag, variant == "light")

    def read(key: str) -> dict[str, str] | None:
        try:
            response = r2_module.signed_request(
                "GET", f"{base}/{bucket}/{r2_module.encode_key_path(key)}", creds=creds, now=now
            )
        except r2_module.R2RequestError as err:
            if err.status == 404:
                return None  # first publish: no live feed yet
            raise
        if response.status >= 400:
            raise r2_module.R2RequestError("GET", key, response.status)
        etag = response.header("etag")
        if not etag:
            raise ValueError(f"No ETag for {key}")
        return {"text": response.text(), "etag": etag}

    def verify(key: str, file_entry: dict[str, Any]) -> None:
        r2_module.verify_remote_artifact(
            f"{base}/{bucket}/{r2_module.encode_key_path(key)}",
            creds, now, file_entry["size"], file_entry["sha512"],
        )

    def write(key: str, text: str, etag: str | None) -> None:
        r2_module.put_object(
            creds, base, bucket, key, text.encode("utf-8"), now,
            "application/yaml",
            {"If-Match": etag} if etag else {"If-None-Match": "*"},
        )

    publish_mac_feed(plan, {"read": read, "verify": verify, "write": write})
    print(f"OK r2: finalized {tag} -> {plan['key']}")
