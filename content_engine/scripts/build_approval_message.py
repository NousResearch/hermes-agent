"""Build the all-in-one approval message with MEDIA: lines for all previews."""
import json
from pathlib import Path

previews = []

# 1. Blog posts (pending)
tracker = Path("/home/kensei/repos/KenseiAgent/content_engine/blog_topics/pending_approvals.jsonl")
for line in tracker.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    e = json.loads(line)
    if e.get("status") == "pending":
        previews.append({
            "type": "blog",
            "title": e["title"],
            "slug": e["slug"],
            "preview": e.get("preview_path", ""),
        })

# 2. X/Twitter article previews
preview_dir = Path("/home/kensei/.hermes/reports/blog-previews")
for p in sorted(preview_dir.glob("xarticle-*.html")):
    slug = p.stem.replace("xarticle-", "")
    md_path = Path(f"/home/kensei/repos/KenseiAgent/content_engine/output/articles/{slug}/article.md")
    title = slug.replace("-", " ").title()
    if md_path.exists():
        title = md_path.read_text().splitlines()[0].replace("# ", "").strip()
    previews.append({
        "type": "xarticle",
        "title": title,
        "slug": slug,
        "preview": str(p),
    })

# Build message
out = [f"**Review queue - {len(previews)} items, previews attached**", ""]
for p in previews:
    short_title = p["title"][:60]
    out.append(f"**{short_title}**")
    out.append(f"  type: {p['type']} | slug: {p['slug']}")
    out.append(f"MEDIA:{p['preview']}")
    out.append("")

msg = "\n".join(out)
Path("/tmp/approval_msg.txt").write_text(msg)
print(f"Built message: {len(msg)} chars, {len(previews)} items")
