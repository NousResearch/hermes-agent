"""Discord-based approval workflow for blog posts.

State machine:
  staged → pending_approval → approved → build → push (live)
                             → rejected → [removed from tracker, draft stays hidden]
                             → amended → [re-staged, re-requested]

Tracker file: blog_topics/pending_approvals.jsonl
"""

from __future__ import annotations
import json
import logging
import re
from datetime import UTC, datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger("blog_approval")

TRACKER_PATH = Path(__file__).resolve().parent.parent / "blog_topics" / "pending_approvals.jsonl"
PREVIEWS_DIR = Path.home() / ".hermes" / "reports" / "blog-previews"
APPROVAL_CHANNEL = "#blog-management"

# ── State management ──────────────────────────────────────────────

def _read_tracker() -> list[dict]:
    if not TRACKER_PATH.exists():
        return []
    entries = []
    for line in TRACKER_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries

def _write_tracker(entries: list[dict]) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACKER_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

def _approval_id(slug: str) -> str:
    return f"{date.today().isoformat()}-{slug}"


_TOPIC_CLUSTER_RULES = [
    (re.compile(r"\btoken[-\s]?efficiency\b|\bllm spend\b|\btoken cost\b", re.I), "token-efficiency-frontier"),
    (re.compile(r"\bagent memory\b|\bvector (database|store)\b|\bmemory architecture\b", re.I), "agent-memory-architecture"),
    (re.compile(r"\bmodel routing\b|\brouting work across models\b", re.I), "model-routing"),
    (re.compile(r"\bevaluation harness\b|\beval(s|uation)? design\b", re.I), "evaluation-harness"),
]
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "for", "from", "how", "in", "is",
    "of", "on", "the", "to", "what", "where", "who", "why", "with", "your",
    "framework", "practical", "builder", "builders", "pm", "pms", "map",
}


def topic_cluster(title: str) -> str:
    """Return a coarse editorial topic cluster for approval-batch dedup."""
    title = title or ""
    for pattern, cluster in _TOPIC_CLUSTER_RULES:
        if pattern.search(title):
            return cluster
    words = re.findall(r"[a-z0-9]+", title.lower())
    kept = [w for w in words if w not in _STOPWORDS]
    return "-".join(kept[:4]) or "untitled"


def batch_validation_issues(entries: Optional[list[dict]] = None) -> list[str]:
    """Validate pending approval batch before it is posted to Discord.

    Repetition is allowed only when every repeated entry declares the same
    explicit series key. Otherwise repeated topic clusters are blocked from the
    approval queue so Sahil is not asked to review accidental duplicates.
    """
    entries = entries if entries is not None else _read_tracker()
    pending_entries = [e for e in entries if e.get("status") == "pending"]
    by_cluster: dict[str, list[dict]] = {}
    for e in pending_entries:
        cluster = topic_cluster(e.get("title") or e.get("slug") or "")
        by_cluster.setdefault(cluster, []).append(e)

    issues: list[str] = []
    for cluster, grouped in sorted(by_cluster.items()):
        if len(grouped) <= 1:
            continue
        series_keys = {g.get("series") or "" for g in grouped}
        if len(series_keys) == 1 and next(iter(series_keys)):
            continue
        slugs = ", ".join(g.get("slug", "?") for g in grouped)
        issues.append(f"duplicate topic cluster '{cluster}' in pending approvals: {slugs}")
    return issues


def _find_pending_topic_conflict(entries: list[dict], candidate: dict) -> Optional[dict]:
    candidate_cluster = topic_cluster(candidate.get("title") or candidate.get("slug") or "")
    candidate_series = candidate.get("series") or ""
    for e in entries:
        if e.get("status") != "pending":
            continue
        if topic_cluster(e.get("title") or e.get("slug") or "") != candidate_cluster:
            continue
        existing_series = e.get("series") or ""
        if candidate_series and candidate_series == existing_series:
            continue
        return e
    return None

# ── Commands ──────────────────────────────────────────────────────

def _generate_preview(slug: str, mdx_path: str = "") -> str:
    """Generate a standalone HTML preview for a draft when possible."""
    try:
        from blog.preview import build_preview_html, parse_frontmatter
        p = Path(mdx_path) if mdx_path else None
        if p and p.exists():
            text = p.read_text(encoding="utf-8", errors="replace")
            fm, body = parse_frontmatter(text)
            tags_raw = fm.get("tags", "[]")
            try:
                tags = json.loads(tags_raw) if tags_raw.startswith("[") else [t.strip().strip('"\'') for t in tags_raw.strip("[]").split(",") if t.strip()]
            except Exception:
                tags = []
            section_images = [m[1] for m in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', body)[:6]]
            html = build_preview_html(
                slug=slug,
                title=fm.get("title") or slug,
                description=fm.get("description", ""),
                stream="builder" if fm.get("tier") == "builder" else fm.get("tier", "ai"),
                tier=fm.get("tier", "ai"),
                tags=tags,
                body_md=body,
                pub_date=str(fm.get("pubDate", "")),
                hero_src=fm.get("heroImage", ""),
                section_images=section_images,
                status="pending",
                embed_images=True,  # images compressed via Pillow before embedding
            )
        else:
            from blog.preview import _read_mdx, build_preview_html
            post = _read_mdx(slug)
            if not post:
                return ""
            html = build_preview_html(**post)
        out_dir = PREVIEWS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{slug}.html"
        out_path.write_text(html, encoding="utf-8")
        return str(out_path)
    except Exception as exc:
        logger.warning("Preview generation failed for %s: %s", slug, exc)
        return ""


def request(slug: str, title: str, stream: str, tier: str = "",
            mdx_path: str = "") -> str:
    """Register a draft for Discord approval. Returns approval_id.

    Production entries must point to a real draft file so approval messages are
    auditable and junk/test entries cannot leak into #blog-management.
    """
    if not mdx_path or not Path(mdx_path).exists():
        raise ValueError(f"Approval request for {slug!r} requires an existing mdx_path")

    entries = _read_tracker()
    if any(e.get("slug") == slug and e.get("status") == "pending" for e in entries):
        logger.info("Approval already pending for %s", slug)
    else:
        candidate = {"slug": slug, "title": title, "stream": stream, "tier": tier}
        conflict = _find_pending_topic_conflict(entries, candidate)
        if conflict:
            raise ValueError(
                "Approval request would duplicate pending topic cluster "
                f"'{topic_cluster(title)}': {slug!r} conflicts with {conflict.get('slug')!r}. "
                "Use explicit series metadata or amend/reject the duplicate first."
            )
        preview_path = _generate_preview(slug, mdx_path)
        entry = {
            "approval_id": _approval_id(slug),
            "slug": slug,
            "title": title,
            "stream": stream,
            "tier": tier,
            "mdx_path": mdx_path,
            "preview_path": preview_path,
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "approved_at": None,
            "discord_message_id": None,
            "notes": "",
        }
        entries.append(entry)
        _write_tracker(entries)
        logger.info("Approval requested: %s (%s)", _approval_id(slug), title)
    return _approval_id(slug)


def pending() -> list[dict]:
    """Return all entries in pending state."""
    return [e for e in _read_tracker() if e.get("status") == "pending"]


def approve(slug: str) -> bool:
    """Mark a pending slug as approved. Returns False if not found."""
    entries = _read_tracker()
    for e in entries:
        if e.get("slug") == slug and e.get("status") == "pending":
            e["status"] = "approved"
            e["approved_at"] = datetime.now(UTC).isoformat()
            _write_tracker(entries)
            logger.info("Approved: %s", slug)
            return True
    logger.warning("No pending approval found for slug: %s", slug)
    return False


def reject(slug: str, reason: str = "") -> None:
    entries = _read_tracker()
    for e in entries:
        if e.get("slug") == slug and e.get("status") == "pending":
            e["status"] = "rejected"
            e["notes"] = reason
            _write_tracker(entries)
            logger.info("Rejected: %s (%s)", slug, reason)
            return


def amend(slug: str, notes: str = "") -> None:
    """Mark an item for amendment (not yet rejected — sahil will edit and re-stage)."""
    entries = _read_tracker()
    for e in entries:
        if e.get("slug") == slug and e.get("status") == "pending":
            e["status"] = "amend"
            e["notes"] = notes
            _write_tracker(entries)
            logger.info("Amendment requested: %s", slug)
            return


def remove(slug: str) -> None:
    """Remove an entry entirely (after successful publish or full discard)."""
    entries = _read_tracker()
    entries = [e for e in entries if e.get("slug") != slug]
    _write_tracker(entries)


def _preview_size_ok(preview_path: str, max_mb: int = 7) -> bool:
    """Check if preview file is small enough for Discord attachment."""
    try:
        return Path(preview_path).stat().st_size < max_mb * 1024 * 1024
    except OSError:
        return False


_ATTACHMENT_OVERSIZE_MSG = "Preview too large for Discord attachment. Use `!preview <slug>` to generate and receive a lightweight preview."


def summary(entry: dict) -> str:
    """Build a Discord-friendly approval request message."""
    preview = entry.get("preview_path") or ""
    preview_ok = bool(preview and Path(preview).exists() and _preview_size_ok(preview))
    preview_oversize = bool(preview and Path(preview).exists() and not _preview_size_ok(preview))

    lines = [
        "[Blog Approval Request]",
        "━━━ **📝  Draft Ready for Review** ━━━",
        f"**Title:** {entry.get('title', 'Untitled')}",
        f"**Stream:** `{entry.get('stream', '?')}`  **Tier:** `{entry.get('tier', '?')}`",
        f"**Slug:** `{entry.get('slug', '?')}`",
        f"**Approval ID:** `{entry.get('approval_id', '?')}`",
        f"**File:** `{entry.get('mdx_path', '?')}`",
    ]
    if preview_ok:
        lines.append(f"MEDIA:{preview}")
    elif preview_oversize:
        lines.append(f"> {_ATTACHMENT_OVERSIZE_MSG}")
    lines.extend([
        "",
        "**Reply in this thread:**",
        "  `!approve <slug>` — build + push to production",
        "  `!reject <slug> [reason]` — block, draft stays hidden",
        "  `!amend <slug> [notes]` — needs edits before approval",
        "",
        "Need to see the visual preview? Reply `!preview <slug>` for a lightweight text preview with image placeholders.",
    ])
    return "\n".join(lines)


def publish(slug: str) -> dict:
    """Build + push an approved slug. Clears from tracker on success."""
    from blog.blog_publisher import approve as publish_approve
    result = publish_approve(slug)
    if result.get("status") == "ok":
        remove(slug)
    return result


# ── Command parsing for Discord replies ──────────────────────────

def parse_discord_command(text: str) -> Optional[dict]:
    """Parse a Discord reply into a structured command.

    Accepted formats:
      !approve <slug>
      !reject <slug> [reason...]
      !amend <slug> [notes...]
      !approve-idea <idea-id>
      !reject-idea <idea-id> [reason...]

    Returns dict with keys: {'command': str, 'slug': str, 'args': str} or None.
    """
    m = re.match(r"^!(approve-idea|reject-idea|approve|reject|amend|preview)\s+(\S+)\s*(.*)", text.strip(), re.IGNORECASE)
    if not m:
        return None
    return {
        "command": m.group(1).lower(),
        "slug": m.group(2),
        "args": m.group(3).strip(),
    }


def handle_discord_command(text: str) -> dict:
    """Process a Discord command string and return a result dict.

    Returns:
        {"handled": True, "action": str, "slug": str, "message": str} or
        {"handled": False, "message": str}
    """
    cmd = parse_discord_command(text)
    if not cmd:
        return {"handled": False, "message": "Unknown command. Use `!approve <slug>`, `!reject <slug>`, or `!amend <slug>`."}

    slug = cmd["slug"]
    action = cmd["command"]

    if action == "approve-idea":
        from blog.idea_backlog import approve as _idea_approve
        # route: idea id may collide with blog slug names, so keep them distinct.
        res = _idea_approve(slug)
        if res.get("status") == "approved":
            return {"handled": True, "action": "idea_approved", "slug": slug,
                    "message": f"💡 Idea **{slug}** approved and queued to its stream backlog for generation."}
        if res.get("status") == "not_found":
            return {"handled": True, "action": "not_found", "slug": slug,
                    "message": f"❌ No pending idea found for `{slug}`."}
        return {"handled": True, "action": "idea_already", "slug": slug,
                "message": f"ℹ️ Idea `{slug}` already processed ({res.get('status')})."}

    elif action == "reject-idea":
        from blog.idea_backlog import reject as _idea_reject
        res = _idea_reject(slug)
        if res.get("status") == "rejected":
            return {"handled": True, "action": "idea_rejected", "slug": slug,
                    "message": f"💡 Idea **{slug}** rejected."}
        return {"handled": True, "action": "not_found", "slug": slug,
                "message": f"❌ No pending idea found for `{slug}`."}

    if action == "approve":
        if approve(slug):
            result = publish(slug)
            if result.get("status") == "ok":
                return {
                    "handled": True, "action": "approved", "slug": slug,
                    "message": f"✅ **{slug}** approved and published (build + push OK).",
                }
            else:
                return {
                    "handled": True, "action": "build_failed", "slug": slug,
                    "message": f"⚠️ **{slug}** approved but **build failed** (rc={result.get('build_rc', '?')}). Draft staged, review locally.",
                }
        return {"handled": True, "action": "not_found", "slug": slug,
                "message": f"❌ No pending approval found for `{slug}`."}

    elif action == "reject":
        reject(slug, cmd["args"])
        return {"handled": True, "action": "rejected", "slug": slug,
                "message": f"❌ **{slug}** rejected."}

    elif action == "amend":
        amend(slug, cmd["args"])
        return {"handled": True, "action": "amended", "slug": slug,
                "message": f"✏️ **{slug}** marked for amendment. Edit the MDX locally and re-stage."}

    elif action == "preview":
        # Find the pending entry
        entries = _read_tracker()
        entry = next((e for e in entries if e.get("slug") == slug), None)
        if not entry:
            return {"handled": False, "message": f"❌ No approval entry found for `{slug}`."}
        mdx_path = entry.get("mdx_path", "")
        if not mdx_path or not Path(mdx_path).exists():
            return {"handled": False, "message": f"❌ MDX file not found for `{slug}`: {mdx_path}"}
        # Generate lightweight text-only preview
        try:
            from blog.preview import build_preview_html, parse_frontmatter
            text = Path(mdx_path).read_text(encoding="utf-8", errors="replace")
            fm, body = parse_frontmatter(text)
            import json, re
            tags_raw = fm.get("tags", "[]")
            try:
                tags = json.loads(tags_raw) if tags_raw.startswith("[") else [t.strip().strip("\"'") for t in tags_raw.strip("[]").split(",") if t.strip()]
            except Exception:
                tags = []
            section_images = [m[1] for m in re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', body)[:6]]
            html = build_preview_html(
                slug=slug,
                title=fm.get("title") or slug,
                description=fm.get("description", ""),
                stream="builder" if fm.get("tier") == "builder" else fm.get("tier", "ai"),
                tier=fm.get("tier", "ai"),
                tags=tags,
                body_md=body,
                pub_date=str(fm.get("pubDate", "")),
                hero_src=fm.get("heroImage", ""),
                section_images=section_images,
                status="pending",
                embed_images=True,
            )
            out_dir = PREVIEWS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{slug}-light.html"
            out_path.write_text(html, encoding="utf-8")
            return {
                "handled": True, "action": "preview", "slug": slug,
                "message": f"MEDIA:{out_path}\n📄 **{slug}** — lightweight preview generated.",
            }
        except Exception as exc:
            return {"handled": True, "action": "preview_failed", "slug": slug,
                    "message": f"⚠️ Preview generation failed for `{slug}`: {exc}"}

    return {"handled": False, "message": "Unexpected error."}
