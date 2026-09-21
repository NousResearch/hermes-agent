"""Blog pipeline orchestrator - run_stream / run_all + CLI.

Order of operations (mirrors article_pipeline):
  1. blog_router.choose(stream)  -> plan or None (skip)
  2. blog_generator.write_with_gate(plan, stream)  -> draft (or None)
  3. blog_illustrator.illustrate(draft)  -> {hero_path, section_paths}
  4. Check for failed images (hero AND all sections None -> set aside)
  5. blog_assembler.assemble(draft, images, repo)  -> mdx Path
  6. blog_publisher.stage_draft(mdx_path, repo)  -> slug
  7. blog_router.record(stream, topic_id, title)  -> (on success only)

Status values:
  - "skipped_disabled"    - BLOG_ENABLED is False
  - "skipped_router"      - router returned None
  - "skipped_excluded"    - exclusion-list policy blocked the item
  - "skipped_generator"   - generator returned None (LLM dead or gate fail);
                            reason/issues/attempts included; topic is
                            quarantined out of choose() after
                            GENERATOR_FAILURE_THRESHOLD consecutive failures
  - "ok"                  - draft staged + topic recorded
  - "failed_images"       - all images failed; post set aside for retry
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from config import BLOG_ENABLED, BLOG_STREAMS, SAHILBLOG_REPO
from blog.blog_router import (
    choose, record, reserve, release,
    track_failed_generator, GENERATOR_FAILURE_THRESHOLD,
)
from blog.blog_generator import write_with_gate
from blog.blog_illustrator import illustrate
from blog.blog_assembler import assemble
from blog.blog_publisher import stage_draft
from blog.exclusions import ExcludedContentError, assert_allowed


# Path for tracking posts with failed images.
FAILED_IMAGES_PATH = Path(__file__).parent.parent / "blog_topics" / "failed_images.jsonl"
# Full draft (title/description/body/tier/tags) is saved here when images fail so
# the retry cron can regenerate the FULL art-directed image set (hero + sections)
# rather than a thin slug-only image. Nothing imageless is ever published.
PENDING_DRAFTS_DIR = Path(__file__).parent.parent / "blog_topics" / "pending_images"
STALE_THRESHOLD_DAYS = 7

# Keys assemble()/illustrate() need to rebuild a post from a held draft.
_DRAFT_PERSIST_KEYS = (
    "title", "description", "body_md", "slug", "tier", "tags",
    "format", "source", "stream",
)


def _save_pending_draft(draft: dict, stream: str) -> None:
    """Persist the full draft so a later retry can regenerate quality images."""
    slug = draft.get("slug", "")
    if not slug:
        return
    PENDING_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {k: draft.get(k) for k in _DRAFT_PERSIST_KEYS if draft.get(k) is not None}
    payload["stream"] = stream
    (PENDING_DRAFTS_DIR / f"{slug}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _load_pending_draft(slug: str) -> Optional[dict]:
    """Load a persisted draft for retry; None if it was never saved (legacy)."""
    path = PENDING_DRAFTS_DIR / f"{slug}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _delete_pending_draft(slug: str) -> None:
    path = PENDING_DRAFTS_DIR / f"{slug}.json"
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _track_failed_image(slug: str, stream: str, error: str) -> None:
    """Append a failed-image record to the tracking JSONL file."""
    FAILED_IMAGES_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "slug": slug,
        "stream": stream,
        "date": date.today().isoformat(),
        "attempts": 1,
        "last_error": error,
        "first_failure": date.today().isoformat(),
    }
    # Check if slug already tracked — if so, increment attempts.
    existing: list[dict] = []
    if FAILED_IMAGES_PATH.exists():
        for line in FAILED_IMAGES_PATH.read_text().splitlines():
            if line.strip():
                try:
                    existing.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Find existing entry for this slug.
    found = False
    for e in existing:
        if e.get("slug") == slug:
            e["attempts"] = e.get("attempts", 0) + 1
            e["last_error"] = error
            e["date"] = date.today().isoformat()
            found = True
            break
    if not found:
        existing.append(entry)
    # Rewrite the file.
    with open(FAILED_IMAGES_PATH, "w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")


def _remove_from_failed(slug: str) -> None:
    """Remove a slug from the failed-images tracking file."""
    if not FAILED_IMAGES_PATH.exists():
        return
    existing: list[dict] = []
    for line in FAILED_IMAGES_PATH.read_text().splitlines():
        if line.strip():
            try:
                existing.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    existing = [e for e in existing if e.get("slug") != slug]
    with open(FAILED_IMAGES_PATH, "w") as f:
        for e in existing:
            f.write(json.dumps(e) + "\n")


def _get_failed_entries() -> list[dict]:
    """Read all failed-image entries from the tracking file."""
    if not FAILED_IMAGES_PATH.exists():
        return []
    entries = []
    for line in FAILED_IMAGES_PATH.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _is_stale(entry: dict, threshold_days: int = STALE_THRESHOLD_DAYS) -> bool:
    """Check if a failed-image entry is stale (older than threshold_days)."""
    first = entry.get("first_failure") or entry.get("date", "")
    if not first:
        return False
    try:
        first_dt = date.fromisoformat(first)
    except ValueError:
        return False
    return (date.today() - first_dt).days > threshold_days


def get_stale_failed_images() -> list[dict]:
    """Return entries that have been in failed_images for > threshold_days.

    Used by the weekly audit cron to flag posts stuck in failed-image state.
    """
    return [e for e in _get_failed_entries() if _is_stale(e)]


def _maybe_request_approval(draft: dict, stream: str, slug: str, mdx_path: str) -> None:
    """Request Discord approval for a successfully staged draft (Process 1).

    Non-blocking: logs error on failure but never crashes the pipeline.
    Approval is optional — posts proceed as drafts regardless.
    """
    try:
        from blog.blog_approval import request as req_approval
        req_approval(
            slug=slug,
            title=draft.get("title", ""),
            stream=stream,
            tier=draft.get("tier", ""),
            mdx_path=mdx_path,
        )
    except Exception as exc:
        import logging
        logging.getLogger("blog_pipeline").warning(
            "Approval request failed for %s (non-blocking): %s", slug, exc)


def run_stream(stream: str, repo: Optional[str] = None,
               pub_date: Optional[str] = None,
               case_study_exempt: bool = False) -> dict:
    """Drive one blog post through the pipeline for a single stream.

    pub_date (YYYY-MM-DD) backdates the post; defaults to today. Used by the
    backlog pre-generation driver to scatter dates across a historical window.
    case_study_exempt: skip the AI-stream "named company + number" hard gate
    (backlog pregen passes True; research/PR-derived topics aren't case studies).
    """
    if not BLOG_ENABLED:
        return {"status": "skipped_disabled", "stream": stream}

    repo_path = repo or str(SAHILBLOG_REPO)

    # 1. Router.
    plan = choose(stream)
    if not plan:
        return {"status": "skipped_router", "stream": stream}

    try:
        assert_allowed(title=plan.get("title_hint", ""),
                       topic_id=plan.get("topic_id", ""))
    except ExcludedContentError as exc:
        return {
            "status": "skipped_excluded", "stream": stream,
            "topic_id": plan.get("topic_id", ""),
            "reason": str(exc),
        }

    # Reserve temporarily so other streams/processes don't pick it while this
    # run is in flight. This is released on generator/image failure and after
    # successful permanent record(), avoiding topic-burn.
    reservation_token = reserve(stream, plan.get("topic_id", ""), plan.get("title_hint", ""))

    # 2. Generator (with gate + retry).
    diagnostics: dict = {}
    draft = write_with_gate(plan, stream=stream, case_study_exempt=case_study_exempt,
                             diagnostics=diagnostics)
    if not draft:
        release(reservation_token)
        topic_id = plan.get("topic_id", "")
        reason = diagnostics.get("reason", "rejected by quality gate")
        attempts = track_failed_generator(topic_id, stream, reason)
        if attempts >= GENERATOR_FAILURE_THRESHOLD:
            import logging
            logging.getLogger("blog_pipeline").warning(
                "Topic %s quarantined after %d generator failures (stream=%s, reason=%s)",
                topic_id, attempts, stream, reason,
            )
        return {"status": "skipped_generator", "stream": stream,
                "topic_id": topic_id, "reason": reason,
                "issues": diagnostics.get("issues", []), "attempts": attempts}

    # 3. Illustrator.
    images = illustrate(draft)

    # 4. Failed-image check: if hero AND all sections are None, set aside.
    hero_ok = images.get("hero_path") is not None
    sections_ok = bool(images.get("section_paths"))
    if not hero_ok and not sections_ok:
        slug = draft.get("slug", "")
        error = "All image generation attempts failed (hero + sections)"
        _track_failed_image(slug, stream, error)
        # Persist the full draft so the retry cron can regenerate the complete
        # art-directed set (hero + sections), not a thin slug-only image.
        _save_pending_draft(draft, stream)
        release(reservation_token)
        return {
            "status": "failed_images", "stream": stream,
            "slug": slug, "title": draft.get("title", ""),
            "topic_id": plan.get("topic_id", ""),
            "error": error,
        }

    # 5. Assembler (accept partial imagery — hero or some sections succeeded).
    mdx_path = assemble(draft, images, repo=repo_path, pub_date=pub_date)

    # 6. Publisher (stage draft, no push).
    try:
        slug = stage_draft(str(mdx_path), repo=repo_path)
    except ExcludedContentError as exc:
        release(reservation_token)
        return {
            "status": "skipped_excluded", "stream": stream,
            "topic_id": plan.get("topic_id", ""),
            "reason": str(exc),
        }

    # 7. Record topic permanently (on success only), then release reservation.
    record(stream, plan.get("topic_id", ""), draft.get("title", ""))
    release(reservation_token)

    # 8. Optionally request Discord approval (Process 1).
    # This doesn't change the publish flow — approval is tracked separately
    # and handled by the approval-requester cron.
    _maybe_request_approval(draft, stream, slug, str(mdx_path))

    return {
        "status": "ok", "stream": stream, "slug": slug,
        "title": draft.get("title", ""), "topic_id": plan.get("topic_id", ""),
        "mdx_path": str(mdx_path),
    }


def run_stage_draft_only(
    stream: str,
    repo: Optional[str] = None,
    *,
    max_new_drafts: int = 1,
    max_images: int = 3,
    dry_run: bool = False,
    case_study_exempt: bool = False,
) -> dict:
    """Create at most one uncommitted worktree draft with at most three images.

    This is the sole scheduler-facing seam.  It deliberately stops after
    assembly: it does not invoke ``stage_draft``, approval, publishing, git
    commit/push, or any delivery surface.  ``dry_run`` validates the request
    without calling router, generator, illustrator, or other providers.
    """
    if max_new_drafts != 1 or max_images != 3:
        raise ValueError("stage-only scheduler seam requires exactly one draft and three images")
    if stream not in BLOG_STREAMS:
        raise ValueError(f"unknown blog stream: {stream}")

    repo_path = str(repo or SAHILBLOG_REPO)
    if not Path(repo_path).is_dir():
        raise ValueError(f"blog repo does not exist: {repo_path}")
    if dry_run:
        return {
            "status": "dry_run",
            "stage_only": True,
            "provider_execution": "refused",
            "stream": stream,
            "repo": repo_path,
            "max_new_drafts": max_new_drafts,
            "max_images": max_images,
        }
    if not BLOG_ENABLED:
        return {"status": "skipped_disabled", "stream": stream, "stage_only": True}

    plan = choose(stream)
    if not plan:
        return {"status": "skipped_router", "stream": stream, "stage_only": True}
    try:
        assert_allowed(title=plan.get("title_hint", ""), topic_id=plan.get("topic_id", ""))
    except ExcludedContentError as exc:
        return {
            "status": "skipped_excluded", "stream": stream, "stage_only": True,
            "topic_id": plan.get("topic_id", ""), "reason": str(exc),
        }

    reservation_token = reserve(stream, plan.get("topic_id", ""), plan.get("title_hint", ""))
    try:
        draft = write_with_gate(plan, stream=stream, case_study_exempt=case_study_exempt)
        if not draft:
            return {
                "status": "skipped_generator", "stream": stream, "stage_only": True,
                "topic_id": plan.get("topic_id", ""),
            }

        # Hero plus two sections is the complete, non-overridable three-image cap.
        images = illustrate(draft, max_sections=max_images - 1)
        hero_ok = images.get("hero_path") is not None
        sections_ok = bool(images.get("section_paths"))
        if not hero_ok and not sections_ok:
            return {
                "status": "failed_images", "stream": stream, "stage_only": True,
                "slug": draft.get("slug", ""), "topic_id": plan.get("topic_id", ""),
            }

        mdx_path = assemble(draft, images, repo=repo_path)
        return {
            "status": "staged", "stage_only": True, "stream": stream,
            "slug": draft.get("slug", ""), "topic_id": plan.get("topic_id", ""),
            "mdx_path": str(mdx_path), "drafts_created": 1,
            "images_requested": max_images,
        }
    finally:
        # Stage-only work is intentionally uncommitted, so it must not burn a topic.
        release(reservation_token)


def retry_failed_images(slug: str, stream: str,
                        repo: Optional[str] = None,
                        raise_on_cap: bool = False) -> dict:
    """Re-attempt image generation for one held post, then assemble + stage it.

    Uses the persisted full draft (body + headings) so the regenerated set is
    the full art-directed hero + sections, not a thin slug-only image. On
    success: re-assembles, re-stages, clears both trackers. On failure:
    increments the attempt count. With raise_on_cap=True, a Codex usage cap
    propagates (CodexCapExceeded) so a batch caller can defer.

    Returns {"status": "ok"|"failed_images"|"no_draft"|"not_found", ...}
    """
    repo_path = repo or str(SAHILBLOG_REPO)
    entries = _get_failed_entries()
    entry = next((e for e in entries if e.get("slug") == slug), None)
    if not entry:
        return {"status": "not_found", "slug": slug}

    draft = _load_pending_draft(slug)
    if not draft:
        # Legacy entry with no persisted draft — cannot rebuild quality images.
        return {"status": "no_draft", "slug": slug,
                "note": "no persisted draft; re-run the stream to regenerate"}
    draft.setdefault("stream", stream)

    images = illustrate(draft, raise_on_cap=raise_on_cap)
    hero_ok = images.get("hero_path") is not None
    sections_ok = bool(images.get("section_paths"))

    if not hero_ok and not sections_ok:
        _track_failed_image(slug, stream, "Retry: all images still failing")
        return {"status": "failed_images", "slug": slug}

    # Success — assemble, stage, and clear from both trackers.
    mdx_path = assemble(draft, images, repo=repo_path)
    try:
        staged_slug = stage_draft(str(mdx_path), repo=repo_path)
    except ExcludedContentError as exc:
        return {"status": "skipped_excluded", "slug": slug, "reason": str(exc)}
    _remove_from_failed(slug)
    _delete_pending_draft(slug)
    return {"status": "ok", "slug": staged_slug,
            "hero_path": images.get("hero_path")}


def retry_all_pending_images(repo: Optional[str] = None) -> dict:
    """Holistic retry: process EVERY held post, deferring if Codex is capped.

    Codex-cap-aware: the first cap signal (CodexCapExceeded) stops the whole run
    and returns status "deferred" — no attempt counts are burned, so held posts
    survive intact until the cap resets. Otherwise each held post is retried,
    assembled, and staged. Legacy entries without a persisted draft are reported
    so they can be regenerated by re-running the stream.

    Returns {"status": "ok"|"deferred"|"idle", "recovered": [...], ...}
    """
    from blog.blog_illustrator import CodexCapExceeded

    entries = _get_failed_entries()
    if not entries:
        return {"status": "idle", "recovered": [], "still_failed": [],
                "no_draft": []}

    recovered: list[str] = []
    still_failed: list[str] = []
    no_draft: list[str] = []
    pruned: list[str] = []

    for entry in entries:
        slug = entry.get("slug", "")
        stream = entry.get("stream", "ai")
        if not slug:
            continue
        try:
            res = retry_failed_images(slug, stream, repo=repo, raise_on_cap=True)
        except CodexCapExceeded:
            return {
                "status": "deferred", "reason": "codex_capped",
                "recovered": recovered, "still_failed": still_failed,
                "no_draft": no_draft, "pruned": pruned,
                "pending": [e.get("slug") for e in entries
                            if e.get("slug") not in recovered],
            }
        status = res.get("status")
        if status == "ok":
            recovered.append(slug)
        elif status == "no_draft":
            # Legacy entry with an unrecoverable draft. Give it a grace window
            # to be re-picked by the router; prune once abandoned (stale) so it
            # doesn't report forever.
            if _is_stale(entry):
                _remove_from_failed(slug)
                _delete_pending_draft(slug)
                pruned.append(slug)
            else:
                no_draft.append(slug)
        else:
            still_failed.append(slug)

    return {
        "status": "ok", "recovered": recovered,
        "still_failed": still_failed, "no_draft": no_draft, "pruned": pruned,
    }


def run_all(streams: tuple = BLOG_STREAMS, repo: Optional[str] = None) -> dict:
    """Run each configured stream and return per-stream results."""
    results = {}
    for stream in streams:
        try:
            results[stream] = run_stream(stream, repo=repo)
        except Exception as exc:
            results[stream] = {"status": "error", "stream": stream, "error": str(exc)}
    any_ok = any(r.get("status") == "ok" for r in results.values())
    return {
        "status": "ok" if any_ok else "skipped_all",
        "results": results,
    }


def _cli():
    """CLI entry point: python -m blog.blog_pipeline --stream ai|pm|builder|research|all"""
    parser = argparse.ArgumentParser(description="SahilBlog content pipeline")
    parser.add_argument("--stream", default="all",
                        choices=["ai", "pm", "builder", "research", "all"],
                        help="Stream to run (default: all)")
    parser.add_argument("--repo", default=None,
                        help="Path to SahilBlog repo (default: config.SAHILBLOG_REPO)")
    parser.add_argument("--retry", action="store_true",
                        help="Retry all held (failed-image) posts; defers if Codex is capped")
    args = parser.parse_args()

    if args.retry:
        result = retry_all_pending_images(repo=args.repo)
        print(f"retry_all_pending_images: {result['status']}")
        if result.get("reason"):
            print(f"  reason: {result['reason']}")
        for key in ("recovered", "still_failed", "no_draft", "pruned", "pending"):
            if result.get(key):
                print(f"  {key}: {', '.join(result[key])}")
        # Deferred (capped) is not an error — the run simply waits for reset.
        return 0 if result["status"] in ("ok", "idle", "deferred") else 1

    if args.stream == "all":
        result = run_all(repo=args.repo)
        print(f"run_all: {result['status']}")
        for stream, r in result.get("results", {}).items():
            print(f"  {stream}: {r.get('status')} {r.get('slug', '')}")
        return 0 if result["status"] == "ok" else 1
    else:
        result = run_stream(args.stream, repo=args.repo)
        print(f"run_stream({args.stream}): {result['status']}")
        if result.get("slug"):
            print(f"  slug: {result['slug']}")
            print(f"  mdx:  {result.get('mdx_path', '')}")
        return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(_cli())