#!/usr/bin/env python3
"""Fetch a site's DESIGN.md design-system doc by slug from
VoltAgent/awesome-design-md.

Standard library only (urllib). Prints the DESIGN.md text to stdout so it can
be consumed as agent context. On network failure falls back to an optional
on-disk cache; without any usable cache it exits non-zero with a clear stderr
message. No API keys, no build step, no dependencies.

Typical invocation (from the hermes-agent checkout):

    python3 optional-skills/creative/design-md-pull/scripts/fetch_design_md.py cursor
    python3 optional-skills/creative/design-md-pull/scripts/fetch_design_md.py vercel --ttl-seconds 86400

Framed through Hermes as:

    terminal(command="python3 scripts/fetch_design_md.py cursor",
             cwd="<hermes-agent checkout>", timeout=30)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/VoltAgent/awesome-design-md/"
    "main/design-md/{slug}/DESIGN.md"
)
DEFAULT_TTL_SECONDS = 24 * 60 * 60
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = (
    "Hermes-Agent/design-md-pull "
    "(skill; https://github.com/NousResearch/hermes-agent)"
)


class FetchError(Exception):
    """Raised when a DESIGN.md cannot be obtained and no cache can serve."""


def _cache_dir(cache_dir: str | None) -> Path:
    if cache_dir:
        return Path(cache_dir)
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hermes-agent" / "design-md-pull"


def _slug_file_name(slug: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in slug)


def _cache_paths(cache_dir: Path, slug: str) -> tuple[Path, Path]:
    safe = _slug_file_name(slug)
    return cache_dir / f"{safe}.md", cache_dir / f"{safe}.meta.json"


def _read_cache(cache_dir: Path, slug: str, ttl_seconds: float) -> str | None:
    """Return cached body if present and within TTL, else None."""
    data_path, meta_path = _cache_paths(cache_dir, slug)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fetched_at = float(meta.get("fetched_at", 0))
    except (OSError, ValueError, TypeError):
        return None
    if time.time() - fetched_at > ttl_seconds:
        return None
    try:
        return data_path.read_text(encoding="utf-8")
    except OSError:
        return None


def _write_cache(cache_dir: Path, slug: str, body: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _cache_paths(cache_dir, slug)
    data_path.write_text(body, encoding="utf-8")
    meta_path.write_text(
        json.dumps({"fetched_at": time.time(), "slug": slug}), encoding="utf-8"
    )


def _fetch_url(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        body = response.read()
    return body.decode("utf-8")


def fetch_design_md(
    slug: str,
    cache_dir: str | None = None,
    ttl_seconds: float = DEFAULT_TTL_SECONDS,
) -> tuple[str, str]:
    """Return (body, source).

    source is "fresh" (live fetch), "cache-fresh" (within TTL), or
    "cache-stale (<reason>" (upstream failed, stale cache served).
    """
    directory = _cache_dir(cache_dir)
    url = BASE_URL.format(slug=slug)

    fresh = _read_cache(directory, slug, ttl_seconds)
    if fresh is not None:
        return fresh, "cache-fresh"

    try:
        body = _fetch_url(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FetchError(
                f"design-md-pull: site slug '{slug}' is not in the "
                f"VoltAgent/awesome-design-md catalog (HTTP 404 at {url})"
            ) from exc
        message = f"design-md-pull: HTTP {exc.code} fetching {url}"
        stale = _read_cache(directory, slug, ttl_seconds=float("inf"))
        if stale is not None:
            return stale, f"cache-stale ({message})"
        raise FetchError(f"{message}; no usable cache") from exc
    except (urllib.error.URLError, OSError) as exc:
        message = f"design-md-pull: network error fetching {url}: {exc}"
        stale = _read_cache(directory, slug, ttl_seconds=float("inf"))
        if stale is not None:
            return stale, f"cache-stale ({message})"
        raise FetchError(message) from exc

    _write_cache(directory, slug, body)
    return body, "fresh"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="design-md-pull",
        description="Fetch a site's DESIGN.md from VoltAgent/awesome-design-md by slug.",
    )
    parser.add_argument("slug", help="Site slug, e.g. cursor, vercel, linear.app")
    parser.add_argument(
        "--cache-dir",
        help="Cache directory (default: XDG cache/hermes-agent/design-md-pull)",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=float,
        default=DEFAULT_TTL_SECONDS,
        help="Cache freshness TTL in seconds (default 86400)",
    )
    parser.add_argument(
        "--output",
        help="Write the DESIGN.md to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    try:
        body, source = fetch_design_md(
            args.slug, cache_dir=args.cache_dir, ttl_seconds=args.ttl_seconds
        )
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        Path(args.output).write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
        if not body.endswith("\n"):
            sys.stdout.write("\n")

    if source.startswith("cache-stale"):
        print(
            f"# (note: upstream unavailable; serving stale cache. {source[len('cache-stale '):]})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
