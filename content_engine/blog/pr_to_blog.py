#!/usr/bin/env python3
"""
PR-to-Blog Pipeline — Daily automated article generation from merged PRs.

Scans for newly merged PRs by Sahil-SS9 across configured GitHub repos.
For each merged PR found since the last run:
  1. Fetches PR metadata (title, body, commits, diffstat, merged_at)
  2. Generates a Builder's Log blog post via the existing blog pipeline
     (blog_generator.write_with_gate + blog_illustrator + blog_assembler)
  3. Stages as approved:false draft MDX in SahilBlog
  4. Registers the draft for Discord approval in #blog-management
  5. Does NOT auto-publish; publishing requires !approve <slug>
  6. Social distribution is deferred until the post is approved/published

Idempotent: tracks processed PRs in a state file. Safe to run repeatedly.

Usage:
  PYTHONPATH=. ../.venv/bin/python -m blog.pr_to_blog
  PYTHONPATH=. ../.venv/bin/python -m blog.pr_to_blog --dry-run
  PYTHONPATH=. ../.venv/bin/python -m blog.pr_to_blog --pr 49064  # specific PR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ──────────────────────────────────────────────────────────

GITHUB_REPOS = [
    "NousResearch/hermes-agent",
]
# Sahil's GitHub handle. His upstream PRs are merged by teknium1 under teknium1's
# authorship, so the daily scan finds his work by searching PR bodies for this
# handle (the salvage / co-author marker), not by --author. See fetch_merged_prs.
SAHIL_HANDLE = "Sahil-SS9"
STATE_FILE = Path(__file__).parent / ".." / "data" / "pr-to-blog-state.json"
LINKEDIN_DRAFTS_DIR = Path(__file__).parent / ".." / "output" / "linkedin-drafts"
BUILDER_STREAM = "builder"

# ── State Management ────────────────────────────────────────────────

def _load_state() -> dict:
    """Load the processed-PR state file."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"processed_prs": {}, "last_run": None}


def _save_state(state: dict) -> None:
    """Save the processed-PR state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _is_processed(state: dict, repo: str, pr_number: int) -> bool:
    key = f"{repo}#{pr_number}"
    return key in state["processed_prs"]


def _mark_processed(state: dict, repo: str, pr_number: int, slug: str) -> None:
    key = f"{repo}#{pr_number}"
    state["processed_prs"][key] = {
        "slug": slug,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── GitHub PR Discovery ─────────────────────────────────────────────

def _gh_exec(args: list[str]) -> dict:
    """Run a gh CLI command and return parsed JSON."""
    result = subprocess.run(
        ["gh", *args],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print(f"  gh error: {result.stderr.strip()}", file=sys.stderr)
        return {}
    return json.loads(result.stdout) if result.stdout.strip() else {}


def fetch_merged_prs(repo: str, limit: int = 10) -> list[dict]:
    """Fetch recently merged PRs that are Sahil's contributions.

    Sahil's upstream work is merged by teknium1 *under teknium1's authorship*,
    so an author filter cannot find it — and `--author teknium1` would scoop the
    entire upstream firehose (300+ PRs/week of other people's work). Instead we
    search merged PRs whose body credits ``Sahil-SS9`` (the salvage / co-author
    marker carried on his contributions). Specific-PR backfills bypass this via
    ``fetch_pr_detail``.
    """
    prs = _gh_exec([
        "pr", "list",
        "--repo", repo,
        "--state", "merged",
        "--search", f"{SAHIL_HANDLE} in:body",
        "--json", "number,title,body,mergedAt,additions,deletions,files,url",
        "--limit", str(limit),
    ])
    if not isinstance(prs, list):
        return []
    # Deduplicate by PR number (search can surface a PR more than once).
    seen: set = set()
    unique_prs: list[dict] = []
    for p in prs:
        if p.get("number") not in seen:
            seen.add(p.get("number"))
            unique_prs.append(p)
    return unique_prs


def fetch_pr_detail(repo: str, pr_number: int) -> dict:
    """Fetch full PR detail including commits."""
    return _gh_exec([
        "pr", "view", str(pr_number),
        "--repo", repo,
        "--json", "number,title,body,mergedAt,additions,deletions,files,commits,url,baseRefName,headRefName",
    ])


# ── Blog Post Generation ────────────────────────────────────────────

_IMAGE_URL_RE = re.compile(
    r'!\[[^\]]*\]\((https?://[^)]+)\)|'
    r'(https?://[^\s)]+(?:\.png|\.jpg|\.jpeg|\.webp|/assets/[^\s)]+|user-attachments/assets/[^\s)]+))',
    re.IGNORECASE,
)


def _extract_pr_infographic_url(pr: dict) -> str:
    """Return the first PR-attached infographic/image URL, if present."""
    body = pr.get("body") or ""
    for match in _IMAGE_URL_RE.findall(body):
        url = next((x for x in match if x), "") if isinstance(match, tuple) else match
        if url:
            return url.strip()
    return ""


def _download_pr_infographic(pr: dict, slug_hint: str) -> Optional[str]:
    """Download a merged PR's attached infographic for use as the blog hero."""
    url = _extract_pr_infographic_url(pr)
    if not url:
        return None
    out_dir = Path(__file__).parent.parent / "output" / "pr_infographics"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    out = out_dir / f"{slug_hint}{suffix}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 10_000:
            print(f"  WARN: PR infographic too small from {url}")
            return None
        out.write_bytes(data)
        print(f"  Using PR-attached infographic: {url}")
        return str(out)
    except Exception as exc:
        print(f"  WARN: could not download PR infographic {url}: {exc}")
        return None


def _build_pr_plan(pr: dict, repo: str) -> dict:
    """Build a blog pipeline plan dict from a merged PR.

    The plan dict is what blog_generator.write_with_gate expects:
    {title_hint, title, tags, source, signals, ...}
    """
    title = pr.get("title", "")
    pr_number = pr.get("number", "")
    body = pr.get("body", "")
    additions = pr.get("additions", 0)
    deletions = pr.get("deletions", 0)
    files = pr.get("files", [])
    commits = pr.get("commits", [])
    merged_at = pr.get("mergedAt", "")
    url = pr.get("url", "")

    file_list = ", ".join(f.get("path", "?") for f in files[:10]) if files else "unknown"
    commit_count = len(commits) if commits else 0

    # Build a rich angle/context string for the generator prompt
    angle = (
        f"GitHub PR #{pr_number} merged into {repo} on {merged_at[:10]}. "
        f"Title: {title}. "
        f"Diffstat: +{additions}/-{deletions} across {len(files)} files. "
        f"Files: {file_list}. "
        f"Commits: {commit_count}. "
        f"PR body: {body[:2000] if body else 'no body'}. "
        f"PR URL: {url}."
    )

    # Build a hermes_pr signal that context_enrich can use.
    # The generator requires plan["signals"] to be non-empty.
    # Use the first commit SHA if available so context_enrich can git-show it.
    first_sha = ""
    if commits and isinstance(commits, list):
        first_commit = commits[0] if commits else {}
        first_sha = first_commit.get("oid", first_commit.get("sha", "")) if isinstance(first_commit, dict) else ""

    signal = {
        "signal_type": "hermes_pr",
        "summary": f"{title} (+{additions}/-{deletions}, {len(files)} files, {commit_count} commits)",
        "repo": repo.split("/")[-1],
        "sha": first_sha,
        "body": body[:1000] if body else "",
        "pr_url": url,
        "pr_number": pr_number,
        "merged_at": merged_at,
        "repo_full": repo,
        "file_list": file_list,
    }

    return {
        "topic_id": f"pr-{repo.split('/')[-1]}-{pr_number}",
        "title_hint": _pr_title_to_blog_title(title),
        "title": _pr_title_to_blog_title(title),
        "angle": angle,
        "tags": ["kensei", "hermes", "devops", "github", "build",
                 "reality-check", "production", "reliability"],
        "source": "gitradar",
        "signals": [signal],
        "stream": BUILDER_STREAM,
    }


def _pr_title_to_blog_title(pr_title: str) -> str:
    """Turn a conventional-commit PR title into a distinct, blog-ready headline.

    Drops the type(scope) prefix and any trailing salvage ref, then frames the
    summary by commit type so each post gets a unique, honest title. The old
    behaviour prepended a blanket 'How I fixed a Hermes Agent bug:' to every
    post, which collided on near-identical slugs and mislabelled feat/perf work.

    'fix(kanban): hold reclaim while the worker is still alive'
      -> 'Hold reclaim while the worker is still alive: a kanban fix'
    'feat(simplify-code): risk-tiered application'
      -> 'Risk-tiered application: a simplify-code feature'
    """
    import re
    m = re.match(
        r'^(?P<type>fix|feat|chore|docs|test|refactor|perf|style|ci|build|revert)'
        r'(\((?P<scope>[^)]+)\))?:\s*(?P<rest>.*)$',
        pr_title.strip(),
    )
    if not m:
        return pr_title.strip().rstrip('.')
    ctype = m.group('type')
    scope = (m.group('scope') or '').replace('_', ' ').replace('-', ' ').strip()
    # Drop a trailing salvage ref like " (#41448)" for a cleaner headline.
    rest = re.sub(r'\s*\(#\d+\)\s*$', '', m.group('rest').strip()).rstrip('.')
    summary = (rest[:1].upper() + rest[1:]) if rest else rest
    area = scope or ctype
    kind = "feature" if ctype == "feat" else ("speedup" if ctype == "perf" else "fix")
    return f"{summary}: a {area} {kind}"


def generate_blog_post(pr: dict, repo: str, dry_run: bool = False) -> Optional[dict]:
    """Generate a full blog post from a merged PR through the pipeline.

    Returns {slug, mdx_path, title} on success, None on failure.
    """
    from blog.blog_generator import write_with_gate
    from blog.blog_illustrator import illustrate
    from blog.blog_assembler import assemble
    from blog.blog_publisher import stage_draft
    from blog.blog_approval import request as request_approval
    from config import SAHILBLOG_REPO

    plan = _build_pr_plan(pr, repo)
    print(f"  Plan: {plan['topic_id']} -> {plan['title']}")

    # 1. Generate draft through the gate
    # Use strict_review=False (graceful degradation) for the daily cron path.
    # If the reviewer LLM is down (429, network), the pipeline continues on
    # the deterministic gate alone. The PR is already merged so the work is
    # verified; the blog post is a narrative wrapper, not the fix itself.
    print("  Generating draft...")
    draft = write_with_gate(plan, stream=BUILDER_STREAM, strict_review=False)
    if not draft:
        reason = "draft_gate_or_generator_returned_none"
        print(f"  FAIL: {reason}")
        return {"error": reason}

    if dry_run:
        print("  DRY RUN: skipping illustration, assembly, staging")
        return {"slug": "dry-run", "mdx_path": None, "title": draft.get("title", "?")}

    # 2. Resolve imagery. For upstream PR posts, prefer the merged PR's
    # attached Teknium/NousResearch infographic. Generate via Codex only when
    # no source image is attached or downloadable.
    print("  Resolving hero image...")
    slug_hint = plan["topic_id"].replace("/", "-")
    pr_image = _download_pr_infographic(pr, slug_hint)
    if pr_image:
        images = {"hero_path": pr_image, "section_paths": {}}
        image_source = "upstream_pr_infographic"
    else:
        print("  Generating images with Codex CLI...")
        images = illustrate(draft)
        image_source = "codex_cli"
        if not images.get("hero_path"):
            print("  WARN: no hero image generated; draft will still require approval")

    # 3. Assemble MDX
    print("  Assembling MDX...")
    mdx_path = assemble(draft, images, repo=str(SAHILBLOG_REPO))
    if not mdx_path:
        print("  FAIL: assembler returned None")
        return None

    # 4. Stage as draft and request approval. Never auto-publish PR posts:
    # merged code proves the code landed, not that the write-up is approved.
    print("  Staging draft...")
    slug = stage_draft(str(mdx_path), repo=str(SAHILBLOG_REPO))
    approval_id = request_approval(slug, draft.get("title", "?"), BUILDER_STREAM,
                                   draft.get("tier", "builder"), str(mdx_path))
    print(f"  Staged for approval: {slug} ({approval_id})")

    return {"slug": slug, "mdx_path": str(mdx_path),
            "title": draft.get("title", "?"), "approved": False,
            "approval_id": approval_id, "image_source": image_source,
            "url": f"https://algorithmiccompass.com/blog/{slug}"}


# ── Social Distribution ─────────────────────────────────────────────

def post_to_x(title: str, slug: str, pr_url: str, dry_run: bool = False) -> Optional[str]:
    """Post a tweet about the published blog post.

    Returns the tweet URL on success, None on failure.
    """
    blog_url = f"https://algorithmiccompass.com/blog/{slug}"
    # Build a concise build-in-public tweet
    tweet = (
        f"Shipped a fix into Hermes Agent upstream.\n\n"
        f"{title}\n\n"
        f"Full writeup: {blog_url}\n"
        f"PR: {pr_url}"
    )

    if len(tweet) > 280:
        # Truncate title if needed
        max_title_len = 280 - len(f"Shipped a fix into Hermes Agent upstream.\n\n\nFull writeup: {blog_url}\nPR: {pr_url}") - 10
        truncated = title[:max_title_len] + "..." if len(title) > max_title_len else title
        tweet = (
            f"Shipped a fix into Hermes Agent upstream.\n\n"
            f"{truncated}\n\n"
            f"Full writeup: {blog_url}\n"
            f"PR: {pr_url}"
        )

    if dry_run:
        print(f"  DRY RUN X post: {tweet[:100]}...")
        return None

    result = subprocess.run(
        ["xurl", "post", tweet],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"  X post FAIL: {result.stderr.strip()}")
        return None

    # Parse tweet URL from xurl output
    output = result.stdout.strip()
    # xurl outputs the tweet URL or JSON with the tweet ID
    if "x.com" in output or "twitter.com" in output:
        return output.split("\n")[0]
    try:
        data = json.loads(output)
        tweet_id = data.get("data", {}).get("id", "")
        if tweet_id:
            return f"https://x.com/Sahil_SS9/status/{tweet_id}"
    except (json.JSONDecodeError, KeyError):
        pass
    return output[:200] if output else None


def draft_linkedin_post(title: str, slug: str, pr: dict, repo: str,
                        dry_run: bool = False) -> str:
    """Draft a LinkedIn post and save it for review.

    LinkedIn API requires OAuth approval. For now, we draft the post
    and save it as a file for manual posting or future browser automation.

    Returns the draft file path.
    """
    LINKEDIN_DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
    blog_url = f"https://algorithmiccompass.com/blog/{slug}"
    pr_url = pr.get("url", "")
    pr_number = pr.get("number", "")
    additions = pr.get("additions", 0)
    deletions = pr.get("deletions", 0)
    files_count = len(pr.get("files", []))

    # LinkedIn voice: authoritative + specific + opinion-having
    # Per sahil-linkedin-voice skill
    draft = f"""# LinkedIn Draft — PR #{pr_number} Blog Post

## Post

I shipped a fix into Hermes Agent upstream this week.

{title}

The problem: a kanban dispatcher bug that could spawn 100+ duplicate workers on a single board when a worker gets stuck in uninterruptible D-state. Load average 70, swap exhausted, the whole VPS on its knees.

The fix: defer the reclaim instead of releasing the claim when the worker is still alive. 84 lines of Python, 165 lines of tests. Self-correcting: not spawning a duplicate is what relieves the memory pressure so the stuck worker can finally die.

Diffstat: +{additions}/-{deletions} across {files_count} files.

PR: {pr_url}
Full writeup: {blog_url}

This is the real build log. Not the polished version. The one where the VPS was on its knees and the fix was one boolean check the callers had been ignoring all along.

#buildinpublic #opensource #hermesagent #aiagents

---

## Metadata
- PR: {repo}#{pr_number}
- Blog slug: {slug}
- Generated: {datetime.now(timezone.utc).isoformat()}
- Status: DRAFT (review before posting)
"""

    draft_path = LINKEDIN_DRAFTS_DIR / f"{slug}.md"
    draft_path.write_text(draft)
    print(f"  LinkedIn draft saved: {draft_path}")
    return str(draft_path)


# ── Main Pipeline ───────────────────────────────────────────────────

def run(dry_run: bool = False, specific_pr: Optional[int] = None,
        source_repo: Optional[str] = None) -> dict:
    """Run the PR-to-blog pipeline.

    source_repo (owner/name) targets a single repo for a specific PR, so
    cross-repo backfills (e.g. mnemosyne-oss/mnemosyne#265) resolve against the
    right repo instead of the default hermes-agent list. Returns a summary dict.
    """
    state = _load_state()
    results = {"processed": [], "skipped": [], "failed": []}

    repos = [source_repo] if source_repo else GITHUB_REPOS
    for repo in repos:
        print(f"\n=== Scanning {repo} for merged PRs crediting {SAHIL_HANDLE} ===")

        if specific_pr:
            pr = fetch_pr_detail(repo, specific_pr)
            if not pr:
                print(f"  Could not fetch PR #{specific_pr}")
                results["failed"].append({"repo": repo, "pr": specific_pr, "error": "fetch_failed"})
                continue
            prs = [pr]
        else:
            prs = fetch_merged_prs(repo)
            if not prs:
                print("  No merged PRs found")
                continue

        for pr in prs:
            pr_number = pr.get("number", 0)
            title = pr.get("title", "?")
            merged_at = pr.get("mergedAt", "")

            print(f"\n--- PR #{pr_number}: {title} (merged {merged_at[:10]}) ---")

            if _is_processed(state, repo, pr_number):
                print("  SKIP: already processed")
                results["skipped"].append({"repo": repo, "pr": pr_number, "title": title})
                continue

            # Fetch full PR detail if we only have the list version
            if "commits" not in pr:
                print("  Fetching full PR detail...")
                pr = fetch_pr_detail(repo, pr_number)
                if not pr:
                    print("  FAIL: could not fetch PR detail")
                    results["failed"].append({"repo": repo, "pr": pr_number, "error": "detail_fetch_failed"})
                    continue

            # Generate blog post
            blog_result = generate_blog_post(pr, repo, dry_run=dry_run)
            if not blog_result or blog_result.get("error"):
                reason = (blog_result or {}).get("error", "blog_generation_returned_no_result")
                print(f"  FAIL: blog generation failed: {reason}")
                results["failed"].append({"repo": repo, "pr": pr_number, "error": reason})
                continue

            slug = blog_result["slug"]

            # Distribution is deliberately owned by the dedicated X and LinkedIn
            # managers. The blog pipeline only produces the blog artifact and must
            # never post to X or create personal LinkedIn drafts as a side effect.

            # Mark as processed
            if not dry_run:
                _mark_processed(state, repo, pr_number, slug)
                _save_state(state)

            results["processed"].append({
                "repo": repo, "pr": pr_number, "title": title,
                "slug": slug, "approved": blog_result.get("approved", False),
            })

    if not dry_run:
        _save_state(state)

    # Summary
    print(f"\n=== Pipeline complete ===")
    print(f"  Processed: {len(results['processed'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Failed: {len(results['failed'])}")

    return results


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PR-to-Blog daily pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip image generation, staging, approval, and social posting")
    parser.add_argument("--pr", type=int, metavar="N",
                        help="Process a specific PR number (default: scan for merged)")
    parser.add_argument("--repo", metavar="OWNER/NAME", default=None,
                        help="Target repo for --pr (e.g. mnemosyne-oss/mnemosyne). "
                             "Defaults to the built-in hermes-agent list.")
    args = parser.parse_args()

    results = run(dry_run=args.dry_run, specific_pr=args.pr, source_repo=args.repo)

    # A daily batch that staged at least one draft has made useful progress.
    # Report failed PRs explicitly but reserve non-zero exit for total failure.
    if results["failed"]:
        for item in results["failed"]:
            print(f"  Failed PR {item['repo']}#{item['pr']}: {item['error']}")
    if results["failed"] and not results["processed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()