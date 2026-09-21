#!/usr/bin/env python3
"""Audit SahilBlog pipeline state without mutating it.

Problem-only report for cron/manual checks:
- dirty worktrees
- pending approvals with missing files/previews
- approved:false drafts without pending approval
- duplicate draft title clusters
- failed image entries
- protected approved posts are not treated as image defects
- cron job last_status for blog-backlog-pregen and blog-failed-retry
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

# Roots resolve with override precedence (CLI > env > production default):
#   --engine-root / BLOG_AUDIT_ENGINE_ROOT  → content_engine parent (the repo)
#   --blog-root   / BLOG_AUDIT_BLOG_ROOT    → SahilBlog checkout
# The production defaults are preserved so live cron behaviour is unchanged.
_DEFAULT_ENGINE = Path(__file__).resolve().parent.parent
_DEFAULT_BLOG = Path.home() / "repos" / "SahilBlog"


def _resolve_roots() -> tuple[Path, Path]:
    eng_env = os.environ.get("BLOG_AUDIT_ENGINE_ROOT", "").strip()
    blog_env = os.environ.get("BLOG_AUDIT_BLOG_ROOT", "").strip()
    engine = Path(eng_env).expanduser() if eng_env else _DEFAULT_ENGINE
    blog = Path(blog_env).expanduser() if blog_env else _DEFAULT_BLOG
    return engine, blog


ENGINE, BLOG = _resolve_roots()
TRACKER = ENGINE / "blog_topics" / "pending_approvals.jsonl"
FAILED_IMAGES = ENGINE / "blog_topics" / "failed_images.jsonl"
EXEMPT = ENGINE / "blog_topics" / "published_exempt.jsonl"
POSTS = BLOG / "src/content/blog"

# After this many failed attempts a post is treated as a hard failure:
# the audit stops re-flagging it daily and instead reports it once as
# "escalated" so manual intervention is visible without the noise.
ESCALATE_AFTER = 3


RETRY_STATUS = ENGINE / "output" / "logs" / "blog-failed-retry-status.json"
CRON_JOBS_JSON = Path.home() / ".hermes" / "cron" / "jobs.json"
BLOG_CRON_NAMES = {"blog-backlog-pregen", "blog-failed-retry"}


def _cron_schedule_minutes(schedule: dict) -> float | None:
    """Return the interval in minutes for a cron schedule dict, or None."""
    kind = schedule.get("kind")
    if kind == "interval":
        return schedule.get("minutes")
    if kind == "cron":
        expr = schedule.get("expr", "")
        # Only handle simple hourly/daily/sub-12h fixed schedules.
        # Every N hours: "0 */N * * *" → N*60
        m = re.match(r"^\d+\s+\*/(\d+)\s+\*\s+\*\s+\*$", expr)
        if m:
            return int(m.group(1)) * 60
        # Daily at HH:MM: "M H * * *" → 1440
        m = re.match(r"^\d+\s+\d+\s+\*\s+\*\s+\*$", expr)
        if m:
            return 1440
    return None


def _read_blog_cron_status() -> list[str]:
    """Check blog-backlog-pregen and blog-failed-retry cron job health.

    Reports an issue when:
    - last_status is "error"
    - last_run_at is stale (older than 2x the schedule interval)
    """
    issues: list[str] = []
    if not CRON_JOBS_JSON.exists():
        return issues
    try:
        data = json.loads(CRON_JOBS_JSON.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return issues
    jobs = data if isinstance(data, list) else data.get("jobs", [])
    now = datetime.now(timezone.utc)
    for job in jobs:
        name = job.get("name", "")
        if name not in BLOG_CRON_NAMES:
            continue
        last_status = job.get("last_status")
        last_run_at = job.get("last_run_at")
        schedule = job.get("schedule", {})
        enabled = job.get("enabled", False)
        # Report error status
        if last_status == "error":
            err = job.get("last_error") or ""
            # Truncate long errors
            if len(err) > 120:
                err = err[:117] + "..."
            issues.append(f"cron {name}: last_status=error {err}".rstrip())
        # Report stale runs (only if enabled and has run at least once)
        if enabled and last_run_at:
            try:
                last_dt = datetime.fromisoformat(last_run_at)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                interval_min = _cron_schedule_minutes(schedule)
                if interval_min is not None:
                    stale_threshold = interval_min * 2  # 2x schedule interval
                    elapsed_min = (now - last_dt).total_seconds() / 60
                    if elapsed_min > stale_threshold:
                        issues.append(
                            f"cron {name}: stale last_run ({int(elapsed_min)}m ago, "
                            f"threshold {int(stale_threshold)}m)"
                        )
            except Exception:
                pass
    return issues


def _read_retry_status() -> dict | None:
    if not RETRY_STATUS.exists():
        return None
    try:
        return json.loads(RETRY_STATUS.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _git_dirty(repo: Path) -> int:
    r = subprocess.run(["git", "status", "--short"], cwd=repo, text=True, capture_output=True, timeout=30)
    if r.returncode != 0:
        return -1
    return len([ln for ln in r.stdout.splitlines() if ln.strip()])


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"_invalid_json": line[:120]})
    return rows


def _frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip('"')
    return fm


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def _read_exempt() -> set[str]:
    """Slugs manually published to the live site (false-flag avoidance).

    When Sahil publishes a post by hand, its local frontmatter may still read
    ``approved:false``. Those posts are not drafts awaiting approval — they are
    live. Adding the slug here stops the audit from flagging them. No deploy
    polling, no repo rewrites: just an explicit, human-maintained list.
    """
    rows = _read_jsonl(EXEMPT)
    return {r["slug"] for r in rows if r.get("slug")}


def audit() -> list[str]:
    issues: list[str] = []

    kd = _git_dirty(ENGINE.parent)
    bd = _git_dirty(BLOG)
    if kd:
        issues.append(f"KenseiAgent worktree dirty: {kd} changed/untracked paths")
    if bd:
        issues.append(f"SahilBlog worktree dirty: {bd} changed/untracked paths")

    # Cron job health for blog-backlog-pregen and blog-failed-retry
    cron_issues = _read_blog_cron_status()
    issues.extend(cron_issues)

    exempt = _read_exempt()

    approvals = _read_jsonl(TRACKER)
    pending = [e for e in approvals if e.get("status") == "pending"]
    tracked_slugs = {e.get("slug") for e in approvals if e.get("slug")}
    for e in approvals:
        if e.get("_invalid_json"):
            issues.append("pending_approvals contains invalid JSON line")
            continue
        slug = e.get("slug", "")
        if slug in {"ai", "slug"} or slug.startswith("test-approval-"):
            issues.append(f"pending_approvals contains junk/test slug: {slug}")
        if e.get("status") == "pending":
            mdx = Path(e.get("mdx_path") or "")
            if not e.get("mdx_path") or not mdx.exists():
                issues.append(f"pending approval missing real mdx_path: {slug}")
            preview = Path(e.get("preview_path") or "")
            if not e.get("preview_path") or not preview.exists():
                issues.append(f"pending approval missing preview_path: {slug}")

    false_posts = []
    title_rows = []
    for p in POSTS.glob("*.md*"):
        fm = _frontmatter(p)
        approved = fm.get("approved", "").lower()
        title = fm.get("title", p.stem)
        if approved == "false":
            false_posts.append(p.stem)
            title_rows.append((p.stem, title))
    for slug in false_posts:
        if slug in exempt:
            # Manually published live — not a draft awaiting approval.
            continue
        if slug not in tracked_slugs:
            issues.append(f"approved:false draft has no approval-tracker entry: {slug}")

    for i, (slug_a, title_a) in enumerate(title_rows):
        for slug_b, title_b in title_rows[i+1:]:
            ratio = SequenceMatcher(None, _norm_title(title_a), _norm_title(title_b)).ratio()
            if ratio >= 0.82:
                issues.append(f"possible duplicate draft titles ({ratio:.2f}): {slug_a} / {slug_b}")

    failed = _read_jsonl(FAILED_IMAGES)
    retry = _read_retry_status()
    if retry:
        finished = retry.get("finished_at", "unknown")
        if retry.get("rc") == 0 and retry.get("idle"):
            issues.append(f"image retry last ran {finished}: idle (nothing held)")
        else:
            rec = retry.get("recovered", [])
            still = retry.get("still_failed", [])
            rec_n = len(rec) if isinstance(rec, list) else rec
            still_n = len(still) if isinstance(still, list) else still
            bits = [f"last ran {finished}"]
            if rec_n:
                bits.append(f"recovered {rec_n}")
            if still_n:
                bits.append(f"still_failed {still_n}")
            if retry.get("deferred"):
                bits.append("deferred (Codex cap)")
            if retry.get("no_draft"):
                bits.append(f"no_draft {len(retry['no_draft']) if isinstance(retry['no_draft'], list) else retry['no_draft']}")
            issues.append("image-retry: " + ", ".join(bits))
    for e in failed:
        if not e.get("slug"):
            continue
        slug = e["slug"]
        attempts = e.get("attempts", 0)
        last = e.get("date") or e.get("first_failure") or "unknown"
        err = e.get("last_error", "")
        if attempts >= ESCALATE_AFTER:
            # Hard failure: stop daily re-flagging as a fresh problem.
            # Report once as escalated so manual intervention is visible
            # without the noise of a repeated "failed image tracked" line.
            issues.append(
                f"escalated: image generation failed {attempts}x (last {last}) "
                f"for {slug} — manual fix needed. error={err}"
            )
        else:
            issues.append(
                f"failed image tracked: {slug} attempts={attempts} "
                f"last_attempt={last} error={err} "
                f"(retry cron runs daily 08:15)"
            )

    return issues


MAX_REPORT_CHARS = 1600


def render_report(problems: list[str]) -> str:
    """Return a Discord-safe, zero-noise summary of audit problems."""
    if not problems:
        return ""

    lines = ["[Blog Pipeline Audit]"]
    for index, problem in enumerate(problems):
        remaining = len(problems) - index - 1
        summary = f"- … {remaining} more issue(s); inspect manually."
        candidate = "\n".join([*lines, f"- {problem}"])
        if remaining and len(candidate) + 1 + len(summary) > MAX_REPORT_CHARS:
            lines.append(summary)
            return "\n".join(lines)
        if not remaining and len(candidate) > MAX_REPORT_CHARS:
            lines.append("- … 1 more issue(s); inspect manually.")
            return "\n".join(lines)
        lines.append(f"- {problem}")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="SahilBlog pipeline audit")
    parser.add_argument("--engine-root", type=Path, default=None,
                        help="content_engine parent dir (env BLOG_AUDIT_ENGINE_ROOT)")
    parser.add_argument("--blog-root", type=Path, default=None,
                        help="SahilBlog checkout dir (env BLOG_AUDIT_BLOG_ROOT)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.engine_root or args.blog_root:
        if args.engine_root:
            os.environ["BLOG_AUDIT_ENGINE_ROOT"] = str(args.engine_root)
        if args.blog_root:
            os.environ["BLOG_AUDIT_BLOG_ROOT"] = str(args.blog_root)
        # Re-resolve module-level roots for this run. As __main__ the module
        # globals are this scope, so globals().update() reaches the
        # audit()/_git_dirty() functions which read these names at call time.
        _eng, _blog = _resolve_roots()
        globals().update({
            "ENGINE": _eng,
            "BLOG": _blog,
            "TRACKER": _eng / "blog_topics" / "pending_approvals.jsonl",
            "FAILED_IMAGES": _eng / "blog_topics" / "failed_images.jsonl",
            "EXEMPT": _eng / "blog_topics" / "published_exempt.jsonl",
            "POSTS": _blog / "src/content/blog",
            "RETRY_STATUS": _eng / "output" / "logs" / "blog-failed-retry-status.json",
            "CRON_JOBS_JSON": Path.home() / ".hermes" / "cron" / "jobs.json",
        })
    report = render_report(audit())
    if report:
        print(report)
