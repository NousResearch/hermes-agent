#!/usr/bin/env python3
"""Durable idea-concept backlog — bridges research-synthesis ideas → blog approval review.

The research paper-synthesis cron classifies papers into write_now / ask_first /
file / skip, but those candidates were never persisted for follow-up: an
ask_first or write_now idea that wasn't acted on in the run that produced it
was simply lost. This module gives those ideas a durable home so they surface
as *standalone idea concepts* in the blog approval review bundle, where Sahil
can approve (→ move to the per-stream content backlog for generation) or reject.

Design:
- One JSONL append-only store, one record per idea concept.
- Fields: id, title, concept (why it matters / what to build), angles (list),
  stream (ai|pm|builder), source (paper id / PR / digest item), post_thesis,
  concrete_takeaway, evidence_anchor, gap_claim, stream_format_rationale,
  created_at, status (pending|approved|rejected|generated), decided_at.
- CLI (used by the synthesis cron):
    add        --title --concept --post-thesis --concrete-takeaway
               --evidence-anchor --gap-claim --stream-format-rationale
               [--angle "..."] [--stream] [--source]
    approve    <id>      # move into the per-stream backlog (blog_router topic)
    reject     <id>
    list       [--status pending]
- Imported by build_review_bundle.py to render pending ideas as standalone
  cards alongside blog/X/LinkedIn approvals (no new channel, no separate flow).

Avoids redundant repetition: dedupes by (source, title) fingerprint on add, so
re-running a day's synthesis can't stack duplicate idea cards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

STORE_PATH = Path.home() / ".hermes" / "research" / "idea-backlog.jsonl"
BLOG_TOPICS_DIR = Path(__file__).resolve().parent.parent / "blog_topics"
# This backlog feeds SahilBlog. Social-only ideas stay in their own pipeline.
ALLOWED_STREAMS = ("ai", "pm", "builder")
REQUIRED_PURPOSE_FIELDS = (
    "post_thesis",
    "concrete_takeaway",
    "evidence_anchor",
    "gap_claim",
    "stream_format_rationale",
)


# ── Store ────────────────────────────────────────────────────────────────────

def _read() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    out = []
    for line in STORE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _write(records: list[dict]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _fingerprint(title: str, source: str = "") -> str:
    raw = f"{source}|{title}".strip().lower()
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add(title: str, concept: str, angles: Optional[list[str]] = None,
        stream: str = "ai", source: str = "", status: str = "pending", *,
        post_thesis: str = "", concrete_takeaway: str = "",
        evidence_anchor: str = "", gap_claim: str = "",
        stream_format_rationale: str = "") -> dict:
    """Append one purpose-led idea, deduped by (source,title) fingerprint.

    New intake is deliberately fail-closed: each purpose field must contain
    local, non-whitespace text. This validates completeness only; it performs
    no semantic-similarity check and makes no model call.
    """
    purpose_fields = {
        "post_thesis": post_thesis,
        "concrete_takeaway": concrete_takeaway,
        "evidence_anchor": evidence_anchor,
        "gap_claim": gap_claim,
        "stream_format_rationale": stream_format_rationale,
    }
    missing = [field for field in REQUIRED_PURPOSE_FIELDS
               if not isinstance(purpose_fields[field], str) or not purpose_fields[field].strip()]
    if missing:
        return {"status": "invalid", "missing_fields": missing}
    records = _read()
    fp = _fingerprint(title, source)
    for r in records:
        if r.get("_fp") == fp:
            return {"status": "duplicate", "id": r["id"]}
    rec = {
        "id": f"idea-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{fp}",
        "_fp": fp,
        "title": title,
        "concept": concept,
        "angles": angles or [],
        "stream": stream if stream in ALLOWED_STREAMS else "ai",
        "source": source,
        **purpose_fields,
        "status": status,
        "created_at": datetime.now(UTC).isoformat(),
        "decided_at": None,
    }
    records.append(rec)
    _write(records)
    return {"status": "added", "id": rec["id"]}


def _set_status(idea_id: str, status: str) -> Optional[dict]:
    records = _read()
    for r in records:
        if r["id"] == idea_id:
            r["status"] = status
            r["decided_at"] = datetime.now(UTC).isoformat()
            _write(records)
            return r
    return None


def approve(idea_id: str) -> dict:
    """Mark an idea approved and enqueue it where ``blog_router`` reads it."""
    rec = _set_status(idea_id, "approved")
    if not rec:
        return {"status": "not_found", "id": idea_id}
    # ``blog_router._manual_queue_path`` reads BLOG_TOPICS_DIR/<stream>.jsonl.
    # Do not route approved ideas through the legacy blog_topics/backlog path.
    BLOG_TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    q = BLOG_TOPICS_DIR / f"{rec['stream']}.jsonl"
    entry = {
        "topic_id": rec["id"],
        "title_hint": rec["title"],
        "stream": rec["stream"],
        "source": rec.get("source", ""),
        "concept": rec.get("concept", ""),
        "angles": rec.get("angles", []),
        "post_thesis": rec.get("post_thesis", ""),
        "concrete_takeaway": rec.get("concrete_takeaway", ""),
        "evidence_anchor": rec.get("evidence_anchor", ""),
        "gap_claim": rec.get("gap_claim", ""),
        "stream_format_rationale": rec.get("stream_format_rationale", ""),
        "priority": 9,
        "approved_at": datetime.now(UTC).isoformat(),
    }
    with open(q, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "approved", "id": idea_id, "queued": str(q)}


def reject(idea_id: str) -> dict:
    rec = _set_status(idea_id, "rejected")
    if not rec:
        return {"status": "not_found", "id": idea_id}
    return {"status": "rejected", "id": idea_id}


def list_pending() -> list[dict]:
    return [r for r in _read() if r.get("status") == "pending"]


# ── HTML rendering (imported by build_review_bundle) ─────────────────────────

def idea_cards(max_items: Optional[int] = None) -> list[dict]:
    """Return pending ideas as render-ready panes for the review bundle.

    Each card mirrors the approval-bundle pane shape {id,title,pane,group} so
    build_review_bundle can drop it straight into the tabbed document without
    new plumbing. Group label is 'IDEAS' (a new sidebar group).
    """
    pending = list_pending()
    if max_items:
        pending = pending[:max_items]
    from html import escape as _esc
    cards = []
    for i, r in enumerate(pending, 1):
        angles = "".join(f'<span class="angle-tag">{_esc(a)}</span>' for a in r.get("angles", []))
        meta = f'<dl><dt>Stream</dt><dd>{_esc(r.get("stream","ai"))}</dd>' \
               f'<dt>Source</dt><dd>{_esc(r.get("source","—"))}</dd></dl>'
        pane = (
            f"<h1>{i}. {_esc(r['title'])}</h1>{meta}"
            f'<div class="angles">{angles}</div>'
            f'<p style="color:#d6d3d1;font-size:.92rem;line-height:1.55;">{_esc(r.get("concept",""))}</p>'
            f'<div class="actions"><code>!approve-idea {_esc(r["id"])}</code>'
            f'<code>!reject-idea {_esc(r["id"])}</code></div>'
        )
        cards.append({"id": r["id"], "title": r["title"], "pane": pane, "group": "IDEAS", "stream": r.get("stream", "ai")})
    return cards


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> int:
    p = argparse.ArgumentParser(description="Idea-concept backlog (research → blog review)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add")
    a.add_argument("--title", required=True)
    a.add_argument("--concept", required=True)
    a.add_argument("--angle", action="append", default=[])
    a.add_argument("--stream", default="ai")
    a.add_argument("--source", default="")
    a.add_argument("--post-thesis", required=True)
    a.add_argument("--concrete-takeaway", required=True)
    a.add_argument("--evidence-anchor", required=True)
    a.add_argument("--gap-claim", required=True)
    a.add_argument("--stream-format-rationale", required=True)

    ap = sub.add_parser("approve"); ap.add_argument("id")
    rj = sub.add_parser("reject"); rj.add_argument("id")
    ls = sub.add_parser("list"); ls.add_argument("--status", default="pending")

    args = p.parse_args()
    if args.cmd == "add":
        print(json.dumps(add(
            args.title, args.concept, args.angle, args.stream, args.source,
            post_thesis=args.post_thesis,
            concrete_takeaway=args.concrete_takeaway,
            evidence_anchor=args.evidence_anchor,
            gap_claim=args.gap_claim,
            stream_format_rationale=args.stream_format_rationale,
        )))
    elif args.cmd == "approve":
        print(json.dumps(approve(args.id)))
    elif args.cmd == "reject":
        print(json.dumps(reject(args.id)))
    elif args.cmd == "list":
        rows = _read() if args.status == "all" else [r for r in _read() if r.get("status") == args.status]
        for r in rows:
            print(f"{r['id']}  [{r['status']}]  {r.get('stream','ai')}  {r['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
