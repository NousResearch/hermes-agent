#!/usr/bin/env python3
"""Regenerate square/stub hero images with corrected 16:9 prompt."""
import sys, os, re, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "content_engine"))
os.chdir(str(Path.home() / "repos" / "SahilBlog"))

from PIL import Image
from blog.codex_image_gen import generate_hero

public = Path("/home/kensei/repos/SahilBlog/public/blog")
posts_dir = Path("/home/kensei/repos/SahilBlog/src/content/blog")

total = 0
for f in sorted(posts_dir.glob("*.md*")):
    content = f.read_text()
    fm = re.search(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not fm: continue
    fmd = fm.group(1)
    pub = re.search(r"^pubDate:\s*(\d{4}-\d{2}-\d{2})", fmd, re.M)
    approved = re.search(r"^approved:\s*(\S+)", fmd, re.M)
    if not pub: continue
    if approved and approved.group(1) == "true": continue

    slug = f.stem
    hero = public / slug / "hero.png"
    if not hero.exists(): continue

    try:
        img = Image.open(hero)
        w, h = img.size
        size_kb = hero.stat().st_size // 1024
    except: continue

    ratio = w / h
    if ratio > 1.5 and size_kb > 100: continue

    title_m = re.search(r'^title:\s*["\']?([^"\']+)', fmd, re.M)
    desc_m = re.search(r'^description:\s*["\']?([^"\']+)', fmd, re.M)
    title = title_m.group(1) if title_m else slug
    desc = desc_m.group(1) if desc_m else ""

    total += 1
    print(f"[{total}] {slug} ({w}x{h} {size_kb}KB)")
    sys.stdout.flush()
    t0 = time.time()
    result = generate_hero(title, desc, str(hero), timeout=180, workdir=str(Path.home() / "repos" / "SahilBlog"))
    elapsed = time.time() - t0
    if result and Path(result).exists():
        new_size = Path(result).stat().st_size // 1024
        print(f"  -> {new_size}KB in {elapsed:.0f}s")
    else:
        print(f"  -> FAILED in {elapsed:.0f}s")

print(f"\nDone: {total} images regenerated")
