"""Discord-based manual article creation (Process 2).

Sahil posts a structured form in #blog-intake. The bot parses it,
generates or routes the article, and posts back a result with options.

Form format:
  !article
  link: <url>
  topic: <title or topic description>
  text: <article text or snippet>
  images: <comma-separated paths or URLs>
  pipeline: ai | pm | builder

At least one of link/topic/text must be provided. pipeline defaults to ai.
Images are optional.
"""

from __future__ import annotations
import json
import logging
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("blog_discord_intake")

STREAMS = {"ai": "ai", "pm": "pm", "builder": "builder",
           "builders": "builder", "builders log": "builder"}
RESULT_CHANNEL = "#blog-intake-results"


def parse(text: str) -> dict:
    """Parse a Discord message into a structured intake request."""
    result: dict = {
        "link": "",
        "topic": "",
        "article_text": "",
        "images": [],
        "pipeline": "ai",
        "raw": text,
    }
    for line in text.splitlines():
        line = line.strip()
        lower = line.lower()
        if lower.startswith("link:"):
            result["link"] = line.split(":", 1)[1].strip()
        elif lower.startswith("topic:"):
            result["topic"] = line.split(":", 1)[1].strip()
        elif lower.startswith("text:"):
            result["article_text"] = line.split(":", 1)[1].strip()
        elif lower.startswith("images:"):
            parts = line.split(":", 1)[1].strip()
            result["images"] = [p.strip() for p in parts.split(",") if p.strip()]
        elif lower.startswith("pipeline:"):
            raw_val = line.split(":", 1)[1].strip().lower()
            result["pipeline"] = STREAMS.get(raw_val, "ai")
    return result


def validate(intake: dict) -> list[str]:
    """Validate the intake. Returns list of issues (empty = valid)."""
    issues = []
    if not intake.get("link") and not intake.get("topic") and not intake.get("article_text"):
        issues.append("Must provide at least one of: `link:`, `topic:`, or `text:`")
    if intake.get("pipeline") not in ("ai", "pm", "builder"):
        issues.append("`pipeline:` must be `ai`, `pm`, or `builder`")
    return issues


def process(intake: dict, repo: str = "") -> dict:
    """Process a manual article intake and return the result.

    Priority:
      1. If link provided: scrape content → generate via pipeline
      2. If topic only: add to queue with priority 10 (picked next)
      3. If text provided: write MDX directly → stage via adhoc gate

    Returns {status, slug?, title?, mdx_path?, message}
    """
    issues = validate(intake)
    if issues:
        return {"status": "invalid", "issues": issues}

    if intake["link"]:
        return _process_link(intake)
    elif intake["topic"] and not intake["article_text"]:
        return _process_topic(intake)
    elif intake["article_text"]:
        return _process_text(intake)

    return {"status": "error", "message": "No content source available."}


def _process_link(intake: dict) -> dict:
    """Scrape link content, generate a full article via the pipeline."""
    from config import SAHILBLOG_REPO
    from blog.blog_generator import write_with_gate
    from blog.blog_illustrator import illustrate
    from blog.blog_assembler import assemble
    from blog.blog_publisher import stage_draft
    from blog.blog_router import record

    plan = {
        "topic_id": f"manual-{_hash_str(intake['link']):04d}",
        "title_hint": intake.get("topic") or "Article from link",
        "tags": intake.get("tags", []),
        "source": "manual",
        "signals": [{"signal_id": "manual", "summary": intake.get("topic", ""), "priority": 10}],
    }

    draft = write_with_gate(plan, stream=intake["pipeline"])
    if not draft:
        return {"status": "generation_failed", "message": "Generator could not produce an article from the given link."}

    repo_path = str(SAHILBLOG_REPO)
    images = illustrate(draft)
    mdx_path = assemble(draft, images, repo=repo_path)
    slug = stage_draft(str(mdx_path), repo=repo_path)
    record(intake["pipeline"], plan["topic_id"], draft.get("title", ""))

    return {
        "status": "staged",
        "slug": slug,
        "title": draft.get("title", ""),
        "mdx_path": str(mdx_path),
        "message": f"Article '{draft.get('title', '')}' generated from link and staged as draft (`{slug}`).",
    }


def _process_topic(intake: dict) -> dict:
    """Add topic to the manual queue at highest priority and trigger generation."""
    from config import SAHILBLOG_REPO
    from blog.blog_pipeline import run_stream

    stream = intake["pipeline"]
    queue_path = Path(__file__).resolve().parent.parent / "blog_topics" / f"{stream}.jsonl"
    topic_id = f"manual-{_hash_str(intake['topic']):04d}"

    entry = {
        "topic_id": topic_id,
        "title_hint": intake["topic"],
        "tags": [],
        "priority": 10,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    with open(queue_path, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Queued topic '%s' for stream %s (priority 10)", intake["topic"], stream)

    # Run the pipeline immediately to generate the article.
    result = run_stream(stream, repo=str(SAHILBLOG_REPO))

    status = result.get("status", "error")
    if status == "ok":
        return {
            "status": "staged",
            "slug": result.get("slug", ""),
            "title": intake["topic"],
            "message": f"Topic **'{intake['topic']}'** generated and staged (`{result.get('slug', '')}`). Approve on Discord.",
        }
    elif status == "skipped_generator":
        return {
            "status": "generation_failed",
            "message": f"Topic '{intake['topic']}' queued but the generator could not produce an article (gate rejected or LLM failed).",
        }
    elif status == "failed_images":
        return {
            "status": "partial",
            "slug": result.get("slug", ""),
            "message": f"Topic '{intake['topic']}' generated but image creation failed. Draft staged without images — add them manually.",
        }
    else:
        return {
            "status": "pipeline_error",
            "message": f"Topic '{intake['topic']}' queued but pipeline returned: {status}.",
        }


def _process_text(intake: dict) -> dict:
    """Write the provided article text directly as an MDX post and stage it."""
    from blog.blog_publisher import stage_adhoc
    from config import SAHILBLOG_REPO

    slug = _sanitize_slug(intake.get("topic") or "manual-article")
    mdx = _build_mdx(intake, slug)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".mdx", delete=False, dir="/tmp")
    tmp.write(mdx)
    tmp_path = tmp.name
    tmp.close()

    result = stage_adhoc(tmp_path, stream=intake["pipeline"], repo=str(SAHILBLOG_REPO))
    Path(tmp_path).unlink(missing_ok=True)

    if result.get("status") == "staged":
        return {
            "status": "staged",
            "slug": result.get("slug", slug),
            "title": intake.get("topic", "Manual Article"),
            "mdx_path": "",
            "message": f"Manual article staged as draft (`{result.get('slug', slug)}`). Approve on Discord.",
        }
    else:
        issues = result.get("issues", ["Unknown error"])
        return {
            "status": "rejected",
            "issues": issues,
            "message": f"Manual article was rejected by the adhoc gate:\n- " + "\n- ".join(issues),
        }


def result_discord(result: dict) -> str:
    """Build a Discord-friendly result message from a process() result."""
    if result["status"] == "invalid":
        return "**❌ Invalid request**\n" + "\n".join(f"- {i}" for i in result.get("issues", []))
    if result["status"] == "generation_failed":
        return f"**❌ Generation failed**\n{result.get('message', '')}"
    if result["status"] == "rejected":
        return f"**❌ Gate rejected**\n{result.get('message', '')}"
    if result["status"] == "partial":
        lines = [
            "━━━ **⚠️  Article Staged (Partial)** ━━━",
            f"{result.get('message', '')}",
            "",
            "**Options:**",
            "  `!approve <slug>` — push to production (missing images will show as blanks)",
            "  `!images <slug>` — attach images manually and re-stage",
            "  `!amend <slug>` — request changes",
        ]
        return "\n".join(lines)

    # Status is "staged"
    lines = [
        "━━━ **📖  Article Ready for Review** ━━━",
        f"**Title:** {result.get('title', 'Unknown')}",
        f"**Slug:** `{result.get('slug', '?')}`",
        "",
        "**Options:**",
        f"  `!approve {result.get('slug', '')}` — build + push to production",
        f"  `!amend {result.get('slug', '')} [notes]` — request edits",
        f"  `!queue {result.get('slug', '')}` — keep draft, schedule for later",
    ]
    return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────

def _hash_str(s: str) -> int:
    return abs(hash(s)) % 10000

def _sanitize_slug(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:80].strip("-")

def _build_mdx(intake: dict, slug: str) -> str:
    """Build MDX frontmatter + body from a manual article intake."""
    title = intake.get("topic", slug.replace("-", " ").title())
    # Truncate text to first ~200 chars for description.
    text = intake.get("article_text", "")
    desc = text[:200].replace("\n", " ") if text else f"Manual article on {title}"
    return (
        "---\n"
        f'format: "essay"\n'
        f'tier: "{intake["pipeline"]}"\n'
        f'source: "manual"\n'
        f'title: "{title}"\n'
        f'description: "{desc}"\n'
        f"tags: []\n"
        f"approved: false\n"
        "---\n"
        f"\n{text}\n"
    )
