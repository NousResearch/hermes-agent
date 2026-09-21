"""One-off blog back-population orchestrator for SahilBlog.

Iterates the approved topics (backfill_topics.TOPICS) per stream, generates
each post through the shared pipeline (generator → illustrator → assembler →
publisher), stages as "approved:false" draft MDX in the SahilBlog repo.

Key behaviours:
  - Idempotent: skips topics whose slug MDX already exists (by collision check).
  - Budget-capped: checks BACKFILL_SPEND_CAP_GBP before each image batch using
    the standard budget module (label prefix "backfill:").
  - Dry-run: ``--dry-run`` generates text only, skips all image/FAL calls.
  - Per-stream: ``--stream ai|pm|builder`` or all.
  - Limit: ``--limit N`` stops after N generated posts.
  - AI stream named events: runs ``news_verify`` for topics flagged
    ``needs_verification`` and threads the result into the LLM prompt.
  - No push: drafts staged as git commits only; Sahil deploys separately.

Usage:
  PYTHONPATH=. ../.venv/bin/python -m blog.backfill --stream ai --limit 1 --dry-run
  PYTHONPATH=. ../.venv/bin/python -m blog.backfill --stream pm
  PYTHONPATH=. ../.venv/bin/python -m blog.backfill  # all streams
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import budget
import config
from blog.backfill_topics import TOPICS, needs_verification, topics_for
from blog.blog_assembler import assemble
from blog.blog_generator import ReviewUnavailable, write_with_gate
from blog.blog_illustrator import illustrate
from blog.blog_publisher import stage_draft
from blog.exclusions import ExcludedContentError, assert_allowed

if sys.version_info >= (3, 11):
    from typing import assert_never  # noqa: F401
else:
    pass


def _stagger_dates(num_posts: int, span_days: int = 21) -> list[str]:
    """Spread ``num_posts`` dates evenly across the past ``span_days`` days.

    Returns a list of YYYY-MM-DD strings, oldest first. The most recent date
    is yesterday (today is reserved for already-generated validation posts).
    """
    from datetime import date, timedelta
    today = date.today()
    if num_posts <= 0:
        return []
    dates = []
    for i in range(num_posts):
        # Even spread: post 0 -> oldest, post N-1 -> most recent (yesterday)
        offset_days = span_days - round(i * span_days / max(num_posts - 1, 1))
        d = today - timedelta(days=offset_days)
        dates.append(d.isoformat())
    return dates


def _budget_check(cost_gbp: float) -> bool:
    """Check if a backfill spend stays within the one-off envelope.

    Uses the separate backfill ledger so one-off spend does not count against
    the recurring monthly cap.
    """
    return budget.can_spend(cost_gbp, cap_gbp=config.BACKFILL_SPEND_CAP_GBP,
                            ledger_path=config.BACKFILL_LEDGER_PATH)


def _verify_if_needed(topic: dict) -> Optional[dict]:
    """For AI stream topics with ``needs_verification``, run news_verify.

    Returns the verification dict or None if verification is not needed.
    Returns ``{"verified": False, "snippets": [], "query": claim}`` on network
    failure (so the generator reframes instead of fabricating).
    """
    if needs_verification(topic) and topic.get("claim"):
        from blog.news_verify import verify_event
        result = verify_event(topic["claim"])
        print(f"[backfill] verified '{topic['claim']}': {result['verified']}")
        return result
    return None


def _build_plan(topic: dict, stream: str) -> dict:
    """Build a plan dict that blog_generator.write_with_gate expects.

    Mirrors the shape returned by blog_router.choose (signals + title_hint +
    tags + source).
    """
    from blog.blog_streams import STREAMS, tags_for
    topic_tags = topic.get("tags", [])
    s = STREAMS[stream]
    return {
        "topic_id": topic["title"],  # unique-ish; used for dedup
        "title_hint": f"{topic['title']}: {topic['angle']}",
        "tags": tags_for(stream, topic_tags),
        "source": s["source"],
        "signals": [{
            "signal_id": topic["title"],
            "summary": f"{topic['title']}: {topic['angle']}",
            "priority": 10,
        }],
    }


def _slug_from_title(title: str) -> str:
    """Convert a title to a blog slug for collision detection."""
    from blog.blog_slug import slugify
    return slugify(title)


def _exists_on_disk(slug: str) -> bool:
    """Check if a post with this slug already exists in the repo.

    Checks both the exact slug and slug-N variants. This makes the backfill
    idempotent across re-runs.
    """
    repo = Path(config.SAHILBLOG_REPO)
    posts_dir = repo / "src/content/blog"
    if not posts_dir.exists():
        return False
    # Check if any file starts with the slug prefix
    for f in posts_dir.iterdir():
        if f.stem == slug or f.stem.startswith(f"{slug}-"):
            return True
    return False


def run(stream: Optional[str] = None, limit: Optional[int] = None,
        dry_run: bool = False) -> dict:
    """Run the back-population orchestrator.

    Args:
        stream: One of "ai", "pm", "builder", or None for all.
        limit: Max posts to generate (per-stream if stream is set, total if all).
        dry_run: Skip image generation, text only.

    Returns a result dict with {
        "status": "ok" | "partial" | "error",
        "generated": int,
        "skipped": int,
        "errors": int,
        "total_images": int,
        "total_spend_gbp": float,
        "results": [...],
    }
    """
    streams_to_run = [stream] if stream else list(TOPICS)
    result = {
        "status": "ok",
        "generated": 0,
        "skipped": 0,
        "errors": 0,
        "total_images": 0,
        "total_spend_gbp": 0.0,
        "results": [],
    }
    total_limit = limit or float("inf")
    # Blog images now use Codex CLI via ChatGPT subscription, so there is no
    # per-image FAL budget to reserve. Keep spend fields for backwards-compatible
    # reporting, but do not block backfill on image cost.
    per_batch_cost = 0.0

    # Build a flat list of all topics to assign staggered pubDates. This makes
    # the blog read as naturally evolving instead of 36 posts all dated today.
    all_topics_flat: list[tuple[str, dict]] = []
    for s in streams_to_run:
        for topic in topics_for(s):
            all_topics_flat.append((s, topic))
    # Assign dates spread across the past 3 weeks, oldest first.
    date_map: dict[str, str] = {}  # topic_title -> YYYY-MM-DD
    if all_topics_flat:
        dates = _stagger_dates(len(all_topics_flat))
        for (s_, t_), d_ in zip(all_topics_flat, dates):
            date_map[t_["title"]] = d_

    for s in streams_to_run:
        if result["generated"] >= total_limit:
            break
        for topic in topics_for(s):
            if result["generated"] >= total_limit:
                break

            title = topic["title"]
            slug = _slug_from_title(title)

            try:
                assert_allowed(title=title, slug=slug, topic_id=title)
            except ExcludedContentError as exc:
                print(f"[backfill] EXCLUDE [{s}] {title}: {exc}")
                result["skipped"] += 1
                result["results"].append(
                    {"stream": s, "title": title, "status": "excluded", "reason": str(exc)}
                )
                continue

            # Idempotency check.
            if _exists_on_disk(slug):
                print(f"[backfill] SKIP (exists) [{s}] {title}")
                result["skipped"] += 1
                result["results"].append(
                    {"stream": s, "title": title, "status": "skipped"}
                )
                continue

            # Budget check (skip per-batch when over cap).
            if not dry_run and not _budget_check(per_batch_cost):
                print(f"[backfill] BUDGET CAP HIT at {result['total_spend_gbp']:.3f} GBP; stopping")
                result["status"] = "partial"
                break

            # Build plan + optional verification.
            plan = _build_plan(topic, s)
            verification = _verify_if_needed(topic) if s == "ai" else None

            # Generate the draft (text only). Strict mode: if the editorial
            # reviewer degrades (LLM unavailable), halt the stream rather than
            # staging an unreviewed draft.
            print(f"[backfill] generating [{s}] {title}...")
            try:
                draft = write_with_gate(plan, stream=s,
                                        verification=verification,
                                        strict_review=True)
            except ReviewUnavailable as exc:
                print(f"[backfill] HALT [{s}] {title}: {exc}")
                print(f"[backfill] stopping stream '{s}' (strict review unavailable)")
                result["errors"] += 1
                result["results"].append(
                    {"stream": s, "title": title, "status": "review_unavailable"}
                )
                result["status"] = "partial"
                break
            if not draft:
                print(f"[backfill] FAIL (generator) [{s}] {title}")
                result["errors"] += 1
                result["results"].append(
                    {"stream": s, "title": title, "status": "generator_failed"}
                )
                continue

            # Set section count for full posts (hero + 2 sections).
            if not dry_run:
                draft["_max_sections"] = 2
            else:
                draft["_max_sections"] = 0

            # Illustrate (skip in dry-run mode). imagery_transplant.generate
            # already records each spend to the ledger, so we only track counts
            # here without double-recording.
            if dry_run:
                images = {"hero_path": None, "section_paths": {}}
            else:
                print(f"[backfill] illustrating [{s}] {title}...")
                images = illustrate(draft, max_sections=draft["_max_sections"])
                if images.get("hero_path"):
                    result["total_images"] += 1
                for heading, path in images.get("section_paths", {}).items():
                    result["total_images"] += 1

            # Assemble MDX and stage as draft.
            pub_date = date_map.get(title)
            mdx_path = assemble(draft, images, repo=config.SAHILBLOG_REPO,
                                pub_date=pub_date)
            try:
                slug_staged = stage_draft(str(mdx_path), repo=config.SAHILBLOG_REPO)
            except ExcludedContentError as exc:
                print(f"[backfill] EXCLUDE [{s}] {title}: {exc}")
                result["skipped"] += 1
                result["results"].append(
                    {"stream": s, "title": title, "status": "excluded", "reason": str(exc)}
                )
                continue
            print(f"[backfill] staged [{s}] {slug_staged}")
            result["generated"] += 1
            result["results"].append({
                "stream": s,
                "title": title,
                "slug": slug_staged,
                "status": "ok",
                "mdx_path": str(mdx_path),
            })

    return result


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="SahilBlog one-off back-population orchestrator",
    )
    parser.add_argument(
        "--stream", choices=["ai", "pm", "builder", "all"], default="all",
        help="Stream to backfill (default: all)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max posts to generate",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Text only, skip all image generation",
    )
    args = parser.parse_args()

    stream = args.stream if args.stream != "all" else None
    result = run(stream=stream, limit=args.limit, dry_run=args.dry_run)

    print(f"[backfill] done: {result['generated']} generated, "
          f"{result['skipped']} skipped, {result['errors']} errors")
    if not args.dry_run:
        print(f"[backfill] spend: {result['total_images']} images, "
              f"£{result['total_spend_gbp']:.3f} GBP")
    for r in result.get("results", []):
        if r["status"] == "ok":
            print(f"  OK  [{r['stream']}] {r['title']} -> {r.get('mdx_path', '')}")
        else:
            print(f"  {r['status'].upper()} [{r['stream']}] {r['title']}")

    return 0 if result["status"] in ("ok", "partial") else 1


if __name__ == "__main__":
    sys.exit(_cli())
