#!/usr/bin/env python3
"""Batch-generate Codex CLI hero images for backfill blog posts.

Reads all draft posts (approved: false) from the gap dates 18-27 JUN,
generates a Codex CLI hero image for each, copies it to public/blog/<slug>/,
and updates the frontmatter with the heroImage path.

Usage:
  set -a && . ~/.hermes/.env && set +a && PYTHONPATH=content_engine python3 scripts/generate-hero-images.py
"""
import sys, os, re, time, json, shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content_engine"))
os.chdir(str(Path.home() / "repos" / "SahilBlog"))

from blog.codex_image_gen import generate_hero

SAHILBLOG = Path.home() / "repos" / "SahilBlog"
POSTS_DIR = SAHILBLOG / "src" / "content" / "blog"
PUBLIC_DIR = SAHILBLOG / "public" / "blog"

# Find all draft posts from gap dates
posts = []
for f in sorted(POSTS_DIR.glob("*.md*")):
    content = f.read_text()
    fm = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm: continue
    fmd = fm.group(1)
    pub = re.search(r"^pubDate:\s*(\d{4}-\d{2}-\d{2})", fmd, re.M)
    approved = re.search(r"^approved:\s*(\S+)", fmd, re.M)
    hero = re.search(r'^heroImage:\s*["\']?(\S+)["\']?', fmd, re.M)
    title_m = re.search(r'^title:\s*["\']?([^"\']+)', fmd, re.M)
    desc_m = re.search(r'^description:\s*["\']?([^"\']+)', fmd, re.M)
    
    if not pub: continue
    ds = pub.group(1)
    if ds < "2026-06-18" or ds > "2026-06-27": continue
    if approved and approved.group(1) == "true": continue
    
    # Check if hero already exists
    slug = f.stem
    hero_dir = PUBLIC_DIR / slug
    hero_path = hero_dir / "hero.png"
    has_hero = hero_path.exists() and hero_path.stat().st_size > 100000
    
    posts.append({
        "file": f,
        "slug": slug,
        "title": title_m.group(1) if title_m else slug,
        "description": desc_m.group(1) if desc_m else "",
        "date": ds,
        "has_hero": has_hero,
        "hero_dir": hero_dir,
        "hero_path": hero_path,
    })

print(f"Found {len(posts)} draft posts needing hero images")
print(f"Already have hero: {sum(1 for p in posts if p['has_hero'])}")
print(f"Need generation:   {sum(1 for p in posts if not p['has_hero'])}")

# Generate images
results = {"ok": 0, "failed": 0, "skipped": 0}
for i, p in enumerate(posts, 1):
    if p["has_hero"]:
        print(f"[{i}/{len(posts)}] SKIP {p['slug']} (already has hero)")
        results["skipped"] += 1
        continue
    
    print(f"[{i}/{len(posts)}] {p['date']} {p['slug']}")
    print(f"  Title: {p['title'][:60]}")
    sys.stdout.flush()
    
    out_path = str(p["hero_path"])
    t0 = time.time()
    result = generate_hero(
        title=p["title"],
        description=p["description"],
        out_path=out_path,
        timeout=180,
        workdir=str(SAHILBLOG)
    )
    elapsed = time.time() - t0
    
    if result and Path(result).exists() and Path(result).stat().st_size > 100000:
        size_kb = Path(result).stat().st_size // 1024
        print(f"  ✅ {size_kb}KB in {elapsed:.0f}s")
        results["ok"] += 1
    else:
        print(f"  ❌ failed in {elapsed:.0f}s")
        results["failed"] += 1

print(f"\n{'='*50}")
print(f"COMPLETE: {results['ok']} OK, {results['failed']} failed, {results['skipped']} skipped")
print(f"{'='*50}")
