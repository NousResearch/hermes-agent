#!/usr/bin/env python3
"""Publish GitDB/GitHub tool candidates as private ARD entries."""

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
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hermes_constants import get_hermes_home

TOOL_CANDIDATE_TYPE = "application/vnd.hermes.tool-candidate+json"
DEFAULT_BUCKETS = {"adopt_now", "watch"}


def _slug(value: str) -> str:
    value = value.strip().replace("/", ":")
    return re.sub(r"[^A-Za-z0-9_.:-]+", "-", value).strip("-") or "unknown"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = data.get("items") or data.get("selected_items") or data.get("repos") or data.get("findings") or []
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)]


def gitdb_item_to_ard_entry(item: dict[str, Any]) -> dict[str, Any]:
    full_name = item.get("full_name") or item.get("repo") or item.get("name") or "unknown/unknown"
    full_name = str(full_name)
    url = item.get("url") or item.get("html_url") or f"https://github.com/{full_name}"
    description = item.get("description") if isinstance(item.get("description"), str) else ""
    bucket = str(item.get("triage_bucket") or item.get("bucket") or item.get("posture") or "watch")
    topics = [str(t) for t in _as_list(item.get("topics"))]
    focus_tags = [str(t) for t in _as_list(item.get("focus_tags") or item.get("focus_terms"))]
    tags = sorted({"gitdb", "tool-candidate", bucket, *topics[:10], *focus_tags})
    metadata = {
        "source": "gitdb-github-watch",
        "gitdb": {
            "full_name": full_name,
            "triage_bucket": bucket,
            "triage_reason": item.get("triage_reason"),
            "triage_signals": _as_list(item.get("triage_signals")),
            "focus_score": item.get("focus_score"),
            "focus_terms": _as_list(item.get("focus_terms")),
            "selected": item.get("selected"),
            "stars": item.get("stars"),
            "forks": item.get("forks"),
            "language": item.get("language"),
            "license": item.get("license"),
            "last_push": item.get("last_push"),
        },
    }
    return {
        "identifier": f"urn:ai:gitdb.local:tool-candidate:{_slug(full_name)}",
        "displayName": full_name,
        "type": TOOL_CANDIDATE_TYPE,
        "url": str(url),
        "description": description,
        "tags": tags,
        "metadata": metadata,
    }


def build_catalog(data: Any, *, buckets: set[str] | None = None) -> dict[str, Any]:
    wanted = buckets or DEFAULT_BUCKETS
    entries: dict[str, dict[str, Any]] = {}
    for item in _items(data):
        bucket = str(item.get("triage_bucket") or item.get("bucket") or item.get("posture") or "watch")
        if bucket not in wanted:
            continue
        entry = gitdb_item_to_ard_entry(item)
        entries.setdefault(entry["identifier"], entry)
    return {
        "specVersion": "1.0",
        "host": {
            "displayName": "Hermes GitDB Tool Candidates",
            "identifier": "did:web:gitdb.local",
        },
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "entries": list(entries.values()),
    }


def write_catalog(catalog: dict[str, Any], output: Path) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "output": str(output), "entries": len(catalog.get("entries", []))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert GitDB watch candidates into a private ARD catalog")
    parser.add_argument("--input", type=Path, default=WORKSPACE_ROOT / "docs" / "reports" / "red" / "gitdb_watch_latest.json")
    parser.add_argument("--output", type=Path, default=get_hermes_home() / ".hub" / "ard-gitdb-candidates.json")
    parser.add_argument("--bucket", action="append", help="Bucket to include; repeatable. Default: adopt_now, watch")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(args.input.read_text(encoding="utf-8"))
    catalog = build_catalog(data, buckets=set(args.bucket) if args.bucket else DEFAULT_BUCKETS)
    summary = write_catalog(catalog, args.output)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Wrote {summary['entries']} GitDB ARD candidate entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
