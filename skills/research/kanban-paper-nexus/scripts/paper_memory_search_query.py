#!/usr/bin/env python3
"""Build a short search_memory query: canonical arXiv id + paper title only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from paper_doc_registry import canonical_paper_id  # noqa: E402

MAX_TITLE_CHARS = 80
MAX_QUERY_CHARS = 120

# Substrings that must never appear in search_memory query for this workflow.
_FORBIDDEN_SUBSTRINGS = (
    "kanban-feishu",
    "finance-kanban",
    "paper-reading-framework",
    "SKILL.md",
    "abstract",
    "【待填】",
    "workflow_id:",
)


def build_search_query(canonical_id: str, title: str | None = None) -> str:
    """Return query string: ``<canonical_id>`` or ``<canonical_id> <short title>``."""
    cid = canonical_paper_id(canonical_id)
    if not cid:
        raise ValueError(f"invalid canonical_id: {canonical_id!r}")
    if not title or not str(title).strip():
        return cid
    t = " ".join(str(title).split())
    if len(t) > MAX_TITLE_CHARS:
        t = t[:MAX_TITLE_CHARS].rsplit(" ", 1)[0].strip() or t[:MAX_TITLE_CHARS].strip()
    q = f"{cid} {t}".strip()
    if len(q) > MAX_QUERY_CHARS:
        q = q[:MAX_QUERY_CHARS].rsplit(" ", 1)[0].strip() or q[:MAX_QUERY_CHARS].strip()
    return q


def validate_query(query: str) -> None:
    q = query.strip()
    if not q:
        raise ValueError("empty query")
    if len(q) > MAX_QUERY_CHARS:
        raise ValueError(f"query too long ({len(q)} > {MAX_QUERY_CHARS})")
    if "\n" in q:
        raise ValueError("query must be a single line (no newlines)")
    low = q.lower()
    for bad in _FORBIDDEN_SUBSTRINGS:
        if bad.lower() in low:
            raise ValueError(f"forbidden substring in query: {bad!r}")
    # Reject pasted arXiv abstract-length blobs (heuristic).
    if len(q.split()) > 24:
        raise ValueError("query has too many tokens; use canonical_id + short title only")


def build_args(
    canonical_id: str,
    title: str | None = None,
    *,
    limit: int = 3,
) -> dict:
    cid = canonical_paper_id(canonical_id)
    query = build_search_query(cid, title)
    validate_query(query)
    return {
        "query": query,
        "workflow_id": f"paper-nexus:{cid}",
        "limit": max(1, min(int(limit), 8)),
        "rules": {
            "allowed": "canonical id (arXiv or s2:<hash>) + paper title (truncated)",
            "forbidden": "skill text, PDF/abstract, kanban design docs, full handoff JSON",
        },
    }


def _load_title_from_meta(meta: dict) -> str | None:
    t = meta.get("title") or meta.get("paper_title")
    return str(t).strip() if t else None


def main() -> int:
    p = argparse.ArgumentParser(description="Build search_memory args for paper-nexus")
    p.add_argument("paper_id", help="arXiv or Semantic Scholar URL/id")
    p.add_argument("--title", help="paper title (optional if --meta-json given)")
    p.add_argument(
        "--meta-json",
        help="path to paper_nexus_metadata.py JSON output",
    )
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()

    from paper_nexus_metadata import resolve_and_fetch, resolve_canonical_id  # noqa: E402

    cid = canonical_paper_id(resolve_canonical_id(args.paper_id))
    title = args.title
    if args.meta_json:
        data = json.loads(Path(args.meta_json).read_text(encoding="utf-8"))
        title = title or _load_title_from_meta(data)
        cid = canonical_paper_id(data.get("canonical_id") or cid)
    elif not title:
        try:
            data = resolve_and_fetch(args.paper_id)
            title = _load_title_from_meta(data)
            cid = canonical_paper_id(data.get("canonical_id") or cid)
        except Exception:
            # S2 rate limits etc.: canonical-only query is still valid for search_memory
            title = None

    out = build_args(cid, title, limit=args.limit)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
