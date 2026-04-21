#!/usr/bin/env python3
"""Find and download blank meme templates from Imgflip.

This script focuses on source lookup:
- search popular Imgflip templates by name or ID
- download the blank template image to a local path

Usage:
    python imgflip_download.py --search "absolute cinema"
    python imgflip_download.py --download "absolute cinema" /tmp/template.png
    python imgflip_download.py --download 505705955 /tmp/template.png
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import BytesIO
from pathlib import Path

try:
    import requests as _requests
except ImportError:  # pragma: no cover - requests is part of the base repo deps
    _requests = None

from PIL import Image

SCRIPT_DIR = Path(__file__).parent
IMGFLIP_API = "https://api.imgflip.com/get_memes"
IMGFLIP_CACHE_FILE = SCRIPT_DIR / ".cache" / "imgflip_memes.json"
IMGFLIP_CACHE_MAX_AGE = 86400


def _fetch_url(url: str, timeout: int = 15) -> bytes:
    if _requests is not None:
        resp = _requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    import urllib.request

    return urllib.request.urlopen(url, timeout=timeout).read()


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def fetch_imgflip_templates() -> list[dict]:
    """Fetch popular meme templates from Imgflip, cached for 24h."""
    import time

    IMGFLIP_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if IMGFLIP_CACHE_FILE.exists():
        age = time.time() - IMGFLIP_CACHE_FILE.stat().st_mtime
        if age < IMGFLIP_CACHE_MAX_AGE:
            with IMGFLIP_CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)

    try:
        data = json.loads(_fetch_url(IMGFLIP_API))
        memes = data.get("data", {}).get("memes", [])
        with IMGFLIP_CACHE_FILE.open("w", encoding="utf-8") as f:
            json.dump(memes, f)
        return memes
    except Exception as exc:  # pragma: no cover - network failure fallback
        if IMGFLIP_CACHE_FILE.exists():
            with IMGFLIP_CACHE_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        print(f"Warning: could not fetch Imgflip templates: {exc}", file=sys.stderr)
        return []


def _score_template(query: str, meme: dict) -> tuple[int, int, str]:
    """Return a ranking tuple where lower is better."""
    normalized_query = _normalize(query)
    normalized_name = _normalize(meme.get("name", ""))
    meme_id = str(meme.get("id", ""))

    if normalized_query == normalized_name or query.strip() == meme_id:
        return (0, 0, normalized_name)
    if normalized_name.startswith(normalized_query) or meme_id.startswith(query.strip()):
        return (1, len(normalized_name), normalized_name)
    if normalized_query in normalized_name:
        return (2, len(normalized_name), normalized_name)

    query_tokens = set(normalized_query.split())
    name_tokens = set(normalized_name.split())
    overlap = len(query_tokens & name_tokens)
    return (10 - overlap, len(normalized_name), normalized_name)


def search_templates(query: str, limit: int = 10) -> list[dict]:
    """Search Imgflip templates by name or ID."""
    memes = fetch_imgflip_templates()
    ranked = sorted(memes, key=lambda meme: _score_template(query, meme))
    normalized_query = _normalize(query)

    matches: list[dict] = []
    for meme in ranked:
        if not normalized_query:
            break
        name = meme.get("name", "")
        meme_id = str(meme.get("id", ""))
        normalized_name = _normalize(name)
        if (
            normalized_query == normalized_name
            or query.strip() == meme_id
            or normalized_name.startswith(normalized_query)
            or normalized_query in normalized_name
            or len(set(normalized_query.split()) & set(normalized_name.split())) > 0
        ):
            matches.append(meme)
            if len(matches) >= limit:
                break
    return matches


def resolve_template(query: str) -> dict | None:
    """Resolve one Imgflip template from a search query or numeric ID."""
    matches = search_templates(query, limit=1)
    if not matches:
        return None
    meme = matches[0]
    return {
        "id": meme.get("id"),
        "name": meme.get("name"),
        "url": meme.get("url"),
        "box_count": meme.get("box_count", 2),
        "source": "imgflip",
    }


def download_template(query: str, output_path: str | Path) -> str:
    """Download the best matching blank template and save it locally."""
    template = resolve_template(query)
    if template is None:
        raise SystemExit(f"No Imgflip templates found for: {query}")

    data = _fetch_url(template["url"])
    image = Image.open(BytesIO(data)).convert("RGBA")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")
    image.save(output)
    return str(output)


def _format_match(meme: dict) -> str:
    return f"{meme.get('id')}\t{meme.get('box_count', 2)}\t{meme.get('name')}\t{meme.get('url')}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search and download blank Imgflip meme templates.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search", metavar="QUERY", help="Search Imgflip templates by name or ID")
    group.add_argument(
        "--download",
        nargs=2,
        metavar=("QUERY", "OUTPUT"),
        help="Download the best matching template to OUTPUT",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.search:
        matches = search_templates(args.search)
        if not matches:
            print(f"No Imgflip templates found for: {args.search}")
            return 1
        for meme in matches:
            print(_format_match(meme))
        return 0

    query, output = args.download
    result = download_template(query, output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
