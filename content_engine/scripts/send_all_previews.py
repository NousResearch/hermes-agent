"""Build and send approval messages to Discord, split if needed."""
import json
import subprocess
import sys
from pathlib import Path

CHANNEL = "discord:#blog-management"
MAX_CHARS = 1900  # Discord limit is 2000, leave room for safe margin

previews = []

# Blog posts
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

# X/Twitter article previews
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


def format_item(p, idx=None):
    short_title = p["title"][:60]
    lines = [f"**{short_title}**", f"  type: {p['type']} | slug: {p['slug']}", f"MEDIA:{p['preview']}"]
    if p["type"] == "blog":
        lines.append("  -> !approve <slug> | !reject <slug> | !amend <slug>")
    return "\n".join(lines)


# Split into batches: blog first (small), then X articles in 2-3 batches
blog_items = [p for p in previews if p["type"] == "blog"]
x_items = [p for p in previews if p["type"] == "xarticle"]

batches = []

# Batch 1: header + all blog items
blog_batch = "**SahilBlog posts awaiting review**\n\n" + "\n\n".join(format_item(p) for p in blog_items)
batches.append(blog_batch)

# Batch 2+: X articles, ~5 per batch
x_chunk_size = 4
for i in range(0, len(x_items), x_chunk_size):
    chunk = x_items[i : i + x_chunk_size]
    header = f"**X/Twitter article previews ({i + 1}-{i + len(chunk)} of {len(x_items)})**"
    batch = header + "\n\n" + "\n\n".join(format_item(p) for p in chunk)
    batches.append(batch)

print(f"Total items: {len(previews)} ({len(blog_items)} blog + {len(x_items)} x-article)")
print(f"Split into {len(batches)} batches")
for i, b in enumerate(batches, 1):
    print(f"  Batch {i}: {len(b)} chars")

# Send each batch
for i, batch in enumerate(batches, 1):
    print(f"\nSending batch {i}/{len(batches)} ({len(batch)} chars)...")
    result = subprocess.run(
        ["hermes", "send", "--to", CHANNEL, "-q", batch],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode == 0:
        print(f"  [OK] batch {i} sent")
    else:
        print(f"  [FAIL] batch {i}: {result.stderr[:200]}")
        if result.stdout:
            print(f"  stdout: {result.stdout[:200]}")
