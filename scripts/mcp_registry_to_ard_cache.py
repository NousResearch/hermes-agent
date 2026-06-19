#!/usr/bin/env python3
"""Import the official MCP Registry into Hermes ARD cache format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import httpx

from hermes_constants import get_hermes_home

DEFAULT_REGISTRY_URL = "https://registry.modelcontextprotocol.io/v0.1/servers"
ARD_TYPE_MCP_SERVER_CARD = "application/mcp-server-card+json"


def _slug(value: str) -> str:
    value = value.strip().replace("/", ":")
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-") or "unknown"


def _official_meta(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("_meta") if isinstance(item.get("_meta"), dict) else {}
    official = meta.get("io.modelcontextprotocol.registry/official") if isinstance(meta, dict) else None
    return official if isinstance(official, dict) else {}


def _first_remote(server: dict[str, Any]) -> dict[str, Any] | None:
    remotes = server.get("remotes")
    if not isinstance(remotes, list):
        return None
    for remote in remotes:
        if not isinstance(remote, dict):
            continue
        url = remote.get("url")
        if isinstance(url, str) and url.startswith(("https://", "http://")):
            return remote
    return None


def registry_item_to_ard_entry(item: dict[str, Any], *, include_non_latest: bool = False) -> dict[str, Any] | None:
    server = item.get("server")
    if not isinstance(server, dict):
        return None
    official = _official_meta(item)
    if not include_non_latest and official.get("isLatest") is False:
        return None
    remote = _first_remote(server)
    if remote is None:
        return None
    name = server.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    title = server.get("title") if isinstance(server.get("title"), str) else name
    description = server.get("description") if isinstance(server.get("description"), str) else ""
    repository = server.get("repository") if isinstance(server.get("repository"), dict) else None
    metadata = {
        "source": "official-mcp-registry",
        "transport": remote.get("type", "streamable-http"),
        "mcpRegistry": {
            "name": name,
            "version": server.get("version"),
            "isLatest": official.get("isLatest"),
            "status": official.get("status"),
            "publishedAt": official.get("publishedAt"),
            "updatedAt": official.get("updatedAt"),
            "websiteUrl": server.get("websiteUrl"),
            "repository": repository,
        },
    }
    tags = ["mcp", "official-mcp-registry"]
    remote_type = remote.get("type")
    if isinstance(remote_type, str):
        tags.append(remote_type)
    return {
        "identifier": f"urn:ai:registry.modelcontextprotocol.io:mcp:{_slug(name)}",
        "displayName": title,
        "type": ARD_TYPE_MCP_SERVER_CARD,
        "url": remote["url"],
        "description": description,
        "tags": tags,
        "metadata": metadata,
    }


def fetch_registry_page(registry_url: str, *, limit: int = 100, cursor: str | None = None, timeout: float = 30) -> dict[str, Any]:
    params: dict[str, Any] = {"limit": limit}
    if cursor:
        params["cursor"] = cursor
    response = httpx.get(
        registry_url,
        params=params,
        headers={"Accept": "application/json, application/problem+json"},
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    data = response.json()
    items = data.get("servers") if isinstance(data, dict) else []
    metadata = data.get("metadata") if isinstance(data, dict) and isinstance(data.get("metadata"), dict) else {}
    return {
        "items": [item for item in items if isinstance(item, dict)] if isinstance(items, list) else [],
        "next_cursor": metadata.get("nextCursor"),
        "metadata": metadata,
    }


def fetch_registry_entries(
    registry_url: str = DEFAULT_REGISTRY_URL,
    *,
    page_limit: int = 100,
    max_pages: int = 5,
    include_non_latest: bool = False,
) -> list[dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    for _ in range(max_pages):
        page = fetch_registry_page(registry_url, limit=page_limit, cursor=cursor)
        for item in page["items"]:
            entry = registry_item_to_ard_entry(item, include_non_latest=include_non_latest)
            if entry and entry["identifier"] not in entries:
                entries[entry["identifier"]] = entry
        cursor = page["next_cursor"]
        if not cursor:
            break
    return list(entries.values())


def write_ard_cache(entries: list[dict[str, Any]], output: Path, meta_output: Path) -> dict[str, Any]:
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        ident = entry.get("identifier")
        if isinstance(ident, str) and ident not in deduped:
            deduped[ident] = entry
    output.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    cache = {
        "version": 1,
        "specVersion": "1.0",
        "host": {
            "displayName": "Official MCP Registry Import",
            "identifier": "did:web:registry.modelcontextprotocol.io",
        },
        "timestamp": now,
        "total_entries": len(deduped),
        "entries": list(deduped.values()),
    }
    output.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "timestamp": now,
        "cache_file": str(output),
        "total_entries": len(deduped),
        "sources": {"mcp_registry": {"entries": len(deduped)}},
    }
    meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import official MCP Registry entries into ARD cache format")
    parser.add_argument("--registry-url", default=DEFAULT_REGISTRY_URL)
    parser.add_argument("--page-limit", type=int, default=100)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--include-non-latest", action="store_true")
    parser.add_argument("--output", type=Path, default=get_hermes_home() / ".hub" / "ard-mcp-registry-cache.json")
    parser.add_argument("--meta-output", type=Path, default=get_hermes_home() / ".hub" / "ard-mcp-registry-cache.meta.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    entries = fetch_registry_entries(
        args.registry_url,
        page_limit=args.page_limit,
        max_pages=args.max_pages,
        include_non_latest=args.include_non_latest,
    )
    cache = write_ard_cache(entries, args.output, args.meta_output)
    summary = {"ok": True, "output": str(args.output), "meta_output": str(args.meta_output), "total_entries": cache["total_entries"]}
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote {cache['total_entries']} MCP Registry ARD entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
