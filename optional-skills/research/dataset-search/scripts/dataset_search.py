#!/usr/bin/env python3
"""
Dataset Search CLI — search and discover open datasets via HuggingFace Datasets API.

Usage:
    python3 dataset_search.py search "chest xray"
    python3 dataset_search.py search "text classification" --task text-classification --limit 10
    python3 dataset_search.py popular --limit 10
    python3 dataset_search.py detail "keremberke/chest-xray-classification"

No API key required. Uses HuggingFace Datasets API.
"""

import json
import sys
import urllib.parse
import urllib.request

HF_API = "https://huggingface.co/api/datasets"
USER_AGENT = "Mozilla/5.0"

# HuggingFace tags a dataset's size bucket as e.g. ``"size_categories:10K<n<100K"``.
# The size_categories family is curated by HF and stable. We map the user-facing
# `--size` flag values to those tag prefixes and filter results client-side
# because the search endpoint doesn't accept a size_categories parameter.
_SIZE_BUCKETS = {
    "small":  {"size_categories:n<1K", "size_categories:1K<n<10K"},
    "medium": {"size_categories:10K<n<100K"},
    "large":  {
        "size_categories:100K<n<1M",
        "size_categories:1M<n<10M",
        "size_categories:10M<n<100M",
        "size_categories:100M<n<1B",
        "size_categories:1B<n<10B",
        "size_categories:n>10B",
    },
}


def _filter_by_size(results: list, size: str) -> list:
    """Client-side filter applying a size bucket to a search result list.

    HuggingFace's search endpoint returns each dataset's size bucket as a
    tag of the form ``size_categories:<range>``. We restrict to whatever
    buckets the chosen ``--size`` value names. Datasets with NO size tag are
    excluded while a size filter is active (they surface when ``--size`` is
    omitted), matching the documented Pitfalls behaviour.

    Args:
        results: list of dataset dicts (the JSON from the search endpoint).
        size:    one of "small"|"medium"|"large".

    Returns:
        The subset of results whose tag list contains at least one of
        the canonical ``size_categories:<bucket>`` tags for the bucket.
        Datasets that publish no size tag are excluded (treated as
        "unknown" — not matched).
    """
    allowed = _SIZE_BUCKETS[size]
    kept = []
    for ds in results:
        tags = ds.get("tags") or []
        # Datasets that publish no size_categories tag are excluded when a
        # size filter is set (documented in SKILL.md Pitfalls). Drop --size to
        # surface them.
        if any(t in allowed for t in tags):
            kept.append(ds)
    return kept


def api_request(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def search_datasets(params: dict) -> list:
    url = f"{HF_API}?{urllib.parse.urlencode(params)}"
    data = api_request(url)
    results = []
    for ds in data:
        results.append({
            "id": ds.get("id", ""),
            "likes": ds.get("likes", 0),
            "downloads": ds.get("downloads", 0),
            "tags": ds.get("tags", [])[:8],
            "siblings": len(ds.get("siblings", [])),
            "created_at": ds.get("createdAt", ""),
            "last_modified": ds.get("lastModified", ""),
        })
    return results


def cmd_search(args):
    params = {
        "search": args.query,
        "sort": "likes",
        "direction": -1,
        "limit": args.limit,
    }
    if args.task:
        params["task_categories"] = args.task
    if args.modality:
        params["modality"] = args.modality
    if args.lang:
        params["language"] = args.lang

    results = search_datasets(params)
    # Size filter is client-side because the HF search endpoint doesn't
    # accept size_categories as a query param (per teknium1 review on
    # PR #45710 — see SKILL.md Size Filter section for rationale).
    if args.size:
        results = _filter_by_size(results, args.size)
    output = {"query": args.query, "total": len(results), "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_popular(args):
    params = {"sort": "likes", "direction": -1, "limit": args.limit}
    results = search_datasets(params)
    output = {"popular": True, "results": results}
    print(json.dumps(output, indent=2, ensure_ascii=False))


def cmd_detail(args):
    url = f"{HF_API}/{urllib.parse.quote(args.dataset_id, safe='/')}"
    data = api_request(url)

    detail = {
        "id": data.get("id"),
        "description": (data.get("description") or "")[:2000],
        "likes": data.get("likes", 0),
        "downloads": data.get("downloads", 0),
        "tags": data.get("tags", []),
        "citation": (data.get("citation") or "")[:500] if data.get("citation") else "",
        "card_data": data.get("cardData", {}),
        "configs": [
            c.get("config_name")
            for c in data.get("configs", [])
            if c.get("config_name")
        ],
        "siblings": len(data.get("siblings", [])),
        "paper_url": data.get("paperUrl"),
        "created_at": data.get("createdAt"),
        "last_modified": data.get("lastModified"),
    }

    # Extract useful cardData fields
    card = detail["card_data"] or {}
    detail["license"] = card.get("license", "")
    detail["size"] = card.get("size_categories", [])
    detail["task"] = card.get("task_categories", [])
    detail["modality"] = card.get("modality", [])
    detail["language"] = card.get("language", [])

    print(json.dumps(detail, indent=2, ensure_ascii=False))


def print_usage():
    print(__doc__)


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    command = sys.argv[1]

    if command == "search":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("query")
        p.add_argument(
            "--task",
            help="Task category (e.g. image-classification, text-classification)",
        )
        p.add_argument("--modality", help="Data modality (e.g. image, text, audio)")
        p.add_argument(
            "--size",
            choices=["small", "medium", "large"],
            help="Size bucket (parsed from the dataset's `size_categories` tag)",
        )
        p.add_argument("--lang", help="Language code (e.g. en, tr)")
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_search(args)
    elif command == "popular":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument("--limit", type=int, default=10)
        args = p.parse_args(sys.argv[2:])
        cmd_popular(args)
    elif command == "detail":
        import argparse

        p = argparse.ArgumentParser()
        p.add_argument(
            "dataset_id", help="Dataset ID (e.g. keremberke/chest-xray-classification)"
        )
        args = p.parse_args(sys.argv[2:])
        cmd_detail(args)
    elif command in ("--help", "-h"):
        print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
