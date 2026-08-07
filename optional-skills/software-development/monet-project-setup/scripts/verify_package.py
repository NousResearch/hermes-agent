#!/usr/bin/env python3
"""Verify a secret-free Project Primer or Agent Preview Pack before handoff."""

from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from project_primer_runtime import (  # type: ignore[import-not-found]
    load_project_primer,
    reject_secret_material,
)

MAX_TOTAL_BYTES = 600_000_000
MAX_PACKAGE_BYTES = MAX_TOTAL_BYTES
MAX_MEMBER_BYTES = 64_000_000
MAX_MEMBERS = 10_000
BASE_MEMBERS = {"manifest.json", "project.json", "primer.json"}
PREVIEW_VERSION_JSON = "versions/hermes-preview/version.json"
PAGE_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _safe_member(info: zipfile.ZipInfo) -> bool:
    member = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    is_symlink = (mode & 0o170000) == 0o120000
    return (
        bool(info.filename)
        and not member.is_absolute()
        and ".." not in member.parts
        and "\\" not in info.filename
        and "\0" not in info.filename
        and info.file_size <= MAX_MEMBER_BYTES
        and not is_symlink
        and not (info.flag_bits & 0x1)
        and not info.comment
    )


_DISALLOWED_PNG_CHUNKS = {b"tEXt", b"zTXt", b"iTXt"}


def _verify_png_member(archive: zipfile.ZipFile, member: str) -> tuple[int, int]:
    """Require a structurally sound PNG with no text chunks or trailing bytes.

    Screenshots are binary members the JSON secret scan never sees, so the
    classic smuggling channels (data after IEND, tEXt/zTXt/iTXt chunks) are
    rejected outright and the raw bytes are scanned for credential patterns.
    """

    data = archive.read(member)
    if len(data) < 45 or data[:8] != PNG_SIGNATURE:
        raise ValueError(f"Preview screenshot is not a PNG: {member}")
    width = height = 0
    offset = 8
    first = True
    ended = False
    while offset < len(data):
        if ended or offset + 12 > len(data):
            raise ValueError(f"Preview screenshot has trailing or malformed data: {member}")
        (length,) = struct.unpack(">I", data[offset : offset + 4])
        chunk_type = data[offset + 4 : offset + 8]
        body_end = offset + 8 + length
        if body_end + 4 > len(data):
            raise ValueError(f"Preview screenshot has trailing or malformed data: {member}")
        body = data[offset + 8 : body_end]
        (crc,) = struct.unpack(">I", data[body_end : body_end + 4])
        if zlib.crc32(chunk_type + body) & 0xFFFFFFFF != crc:
            raise ValueError(f"Preview screenshot fails its PNG integrity check: {member}")
        if first:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError(f"Preview screenshot is not a PNG: {member}")
            width, height = struct.unpack(">II", body[:8])
            if width == 0 or height == 0 or width * height > 100_000_000:
                raise ValueError(f"Preview screenshot dimensions are unsafe: {member}")
            first = False
        elif chunk_type in _DISALLOWED_PNG_CHUNKS:
            raise ValueError(f"Preview screenshot carries embedded text metadata: {member}")
        if chunk_type == b"IEND":
            ended = True
        offset = body_end + 4
    if first or not ended:
        raise ValueError(f"Preview screenshot is not a PNG: {member}")
    reject_secret_material(data.decode("latin-1"), label=member)
    return width, height


def _reject_embedded_config(project: dict) -> None:
    """Decode the base64 JSON blobs the builder writes and scan their content."""

    def scan(value: object, label: str, *, allow_slot_metadata: bool) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError(f"{label} must be base64 JSON.")
        try:
            decoded = json.loads(base64.b64decode(value, validate=True))
        except ValueError:
            raise ValueError(f"{label} is not decodable base64 JSON.") from None
        reject_secret_material(decoded, label=label, allow_slot_metadata=allow_slot_metadata)

    scan(project.get("sourcesJSON"), "project.json.sourcesJSON", allow_slot_metadata=False)
    connectors = project.get("connectors")
    if isinstance(connectors, list):
        for index, connector in enumerate(connectors):
            if isinstance(connector, dict):
                scan(
                    connector.get("configJSON"),
                    f"project.json.connectors.{index}.configJSON",
                    allow_slot_metadata=True,
                )


def _configured_hosts(primer: object) -> set[str]:
    project = primer.project
    hosts: set[str] = set()
    for value in (project.live_url, project.dev_url, project.local_url):
        if value and (host := urlparse(value).hostname):
            hosts.add(host.lower().removeprefix("www."))
    return hosts


def _verify_preview(
    archive: zipfile.ZipFile,
    names: set[str],
    primer: object,
) -> tuple[str, int]:
    if PREVIEW_VERSION_JSON not in names:
        raise ValueError("Agent Preview Pack is missing its version metadata.")
    version = json.loads(archive.read(PREVIEW_VERSION_JSON))
    reject_secret_material(version, label=PREVIEW_VERSION_JSON)
    if version.get("captureKind") != "agent-preview":
        raise ValueError("Agent Preview Pack has the wrong capture provenance.")
    if version.get("capturedWith") not in {"chromium", "chrome", "edge", "firefox", "webkit"}:
        raise ValueError("Agent Preview Pack has an unsupported capture browser.")
    pages = version.get("pages")
    if not isinstance(pages, list) or not pages or len(pages) > 100:
        raise ValueError("Agent Preview Pack must contain 1 to 100 pages.")
    if len(pages) > primer.capture.max_pages:
        raise ValueError("Agent Preview Pack exceeds the Primer page policy.")

    allowed = set(BASE_MEMBERS) | {PREVIEW_VERSION_JSON}
    slugs: set[str] = set()
    hosts = _configured_hosts(primer)
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError("Agent Preview page metadata is invalid.")
        slug = page.get("pageSlug")
        if not isinstance(slug, str) or not PAGE_SLUG.fullmatch(slug) or slug in slugs:
            raise ValueError("Agent Preview page slugs must be unique and path-safe.")
        slugs.add(slug)
        if page.get("reviewOrder") != index:
            raise ValueError("Agent Preview review order must be contiguous and deterministic.")
        url = page.get("url")
        parsed_url = urlparse(url) if isinstance(url, str) else None
        host = parsed_url.hostname if parsed_url is not None else None
        if (
            parsed_url is None
            or parsed_url.scheme not in {"http", "https"}
            or not host
            or (hosts and host.lower().removeprefix("www.") not in hosts)
        ):
            raise ValueError("Agent Preview page URL does not match the configured project.")
        screenshot = f"versions/hermes-preview/pages/{slug}.png"
        if screenshot not in names:
            raise ValueError(f"Agent Preview screenshot is missing: {slug}.png")
        _verify_png_member(archive, screenshot)
        allowed.add(screenshot)

    if names != allowed:
        raise ValueError("Agent Preview Pack contains undeclared members.")
    return str(version.get("label") or "Hermes Preview"), len(pages)


def verify(path: Path) -> dict[str, object]:
    filename = path.name.lower()
    if not (filename.endswith(".monetproj") or filename.endswith(".monetproj.zip")):
        raise ValueError("Expected an existing .monetproj or .monetproj.zip file.")
    if not path.is_file():
        raise ValueError("Monet project package does not exist.")
    if path.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError("Monet project package exceeds the safe compressed size limit.")

    with zipfile.ZipFile(path) as archive:
        if archive.comment:
            raise ValueError("Monet project package must not carry an archive comment.")
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError("Monet project package contains too many members.")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(not _safe_member(info) for info in infos):
            raise ValueError("Monet project package contains an unsafe member.")
        if sum(info.file_size for info in infos) > MAX_TOTAL_BYTES:
            raise ValueError("Monet project package exceeds the safe expanded size limit.")
        if not BASE_MEMBERS.issubset(names):
            raise ValueError("Monet project package is missing required metadata.")

        # Every JSON member is scanned so the reported "containsSecrets": false
        # reflects inspection of this file, not trust in how it was built.
        manifest = json.loads(archive.read("manifest.json"))
        reject_secret_material(manifest, label="manifest.json")
        project = json.loads(archive.read("project.json"))
        reject_secret_material(project, label="project.json")
        _reject_embedded_config(project)
        primer = load_project_primer(archive.read("primer.json"))
        version_count = manifest.get("versionCount")
        preview_label: str | None = None
        preview_pages = 0
        if version_count == 0:
            if set(names) != BASE_MEMBERS:
                raise ValueError("Primer package must contain exactly three expected members.")
        elif version_count == 1:
            preview_label, preview_pages = _verify_preview(archive, set(names), primer)
        else:
            raise ValueError("Monet project package manifest has an invalid version count.")
        if archive.testzip() is not None:
            raise ValueError("Monet project package failed its ZIP integrity check.")

    if manifest.get("formatVersion") != 1:
        raise ValueError("Monet project package manifest is invalid.")
    if project.get("slug") != primer.project.slug:
        raise ValueError("Project and Primer slugs do not match.")
    return {
        "valid": True,
        "containsSecrets": False,
        "projectSlug": primer.project.slug,
        "previewVersion": preview_label,
        "previewPages": preview_pages,
        "members": sorted(names),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.package), indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
