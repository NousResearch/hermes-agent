"""Gap-fill missing section images for SahilBlog posts using Codex CLI.

Reads each blog post, identifies missing section images, and generates
them via Codex CLI (art-directed prompts from art_director module).

Skips hero if already present. Only generates section images that don't exist.
"""
from __future__ import annotations

import os, sys, re, shutil, subprocess, time, glob
from pathlib import Path

# Add content_engine to path
sys.path.insert(0, "/home/kensei/repos/KenseiAgent/content_engine")

import config
from blog import art_director
from blog.art_director import build_art_brief, fallback_brief, compose_prompt

BLOG_DIR = Path(config.SAHILBLOG_REPO).expanduser()
SRC_DIR = BLOG_DIR / "src" / "content" / "blog"
PUB_DIR = BLOG_DIR / "public" / "blog"
CODEX_IMAGES_DIR = Path.home() / ".codex" / "generated_images"

# How many section images to generate per post in this run
# BLOG_MAX_SECTION_IMAGES is normally 2 (budget), but we override for gap-filling
MAX_NEW_SECTIONS = 3  # Generate up to 3 new sections per post per run

# Rotation state
ROTATION_STATE_PATH = Path("/home/kensei/repos/KenseiAgent/content_engine/blog_topics/skill_rotation.json")

def load_recent_styles():
    try:
        import json
        data = json.loads(ROTATION_STATE_PATH.read_text())
        history = data.get("history", []) if isinstance(data, dict) else data
    except:
        return []
    if not isinstance(history, list):
        return []
    valid = art_director.STYLE_IDS
    return [x for x in history if isinstance(x, str) and x in valid][-6:]

def record_style(style_id):
    import json
    history = load_recent_styles()
    history.append(style_id)
    ROTATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"history": history[-6:], "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ROTATION_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")

def find_latest_codex_image(after_ts=None):
    if not CODEX_IMAGES_DIR.exists():
        return None
    candidates = []
    for session_dir in CODEX_IMAGES_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        for img in session_dir.glob("*"):
            if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            try:
                mtime = img.stat().st_mtime
            except OSError:
                continue
            if after_ts is None or mtime >= after_ts:
                candidates.append((mtime, str(img)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

CODEX_CAP_SIGNALS = ("usage_limit_reached", "hit your usage limit", "usage limit has been reached", "resets_in_seconds")

def output_shows_cap(text):
    low = (text or "").lower()
    return any(sig in low for sig in CODEX_CAP_SIGNALS)

def generate_codex_image(full_prompt, out_path, timeout=300, retry_timeout=360):
    """Generate via Codex CLI, copy to out_path. Returns True on success."""
    for attempt, t in enumerate([timeout, retry_timeout], 1):
        before_ts = time.time()
        try:
            result = subprocess.run(
                ["codex", "exec", "--disable", "use_linux_sandbox_bwrap", full_prompt],
                capture_output=True, text=True, timeout=t,
                cwd=str(BLOG_DIR),
            )
            print(f"  codex exit={result.returncode} (attempt {attempt})")
        except subprocess.TimeoutExpired:
            print(f"  codex timed out after {t}s (attempt {attempt})")
            continue
        except Exception as exc:
            print(f"  codex error: {exc}")
            continue

        if output_shows_cap(result.stdout) or output_shows_cap(result.stderr):
            print("  >>> CODEX USAGE CAP REACHED <<<")
            return "cap"

        img = find_latest_codex_image(after_ts=before_ts)
        if not img or not Path(img).exists():
            print(f"  no image found after codex run")
            continue

        size_kb = Path(img).stat().st_size // 1024
        print(f"  generated {size_kb}KB -> {img}")
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_path)
        except Exception as exc:
            print(f"  copy failed: {exc}")
            continue

        if Path(out_path).exists():
            print(f"  verified: {Path(out_path).stat().st_size} bytes")
            return True

    print(f"  all attempts failed")
    return False

def generate_webp(png_path):
    webp_path = Path(png_path).with_suffix(".webp")
    try:
        result = subprocess.run(
            ["node", "-e",
             f"const s=require('sharp'); s('{png_path}').webp({{quality:90,effort:6}}).toFile('{webp_path}').then(()=>console.log('ok'))"],
            capture_output=True, timeout=60,
            cwd=str(BLOG_DIR),
        )
        if result.returncode == 0 and webp_path.exists():
            print(f"  webp: {webp_path} ({webp_path.stat().st_size} bytes)")
            return str(webp_path)
    except Exception as exc:
        print(f"  webp skipped: {exc}")
    return None

def get_missing_sections(post_file):
    """Return list of (idx, heading) tuples for missing section images."""
    content = post_file.read_text()
    h2s = re.findall(r"^##\s+(.+?)\s*$", content, re.MULTILINE)
    
    slug = post_file.stem
    pub_dir = PUB_DIR / slug
    
    missing = []
    for idx, heading in enumerate(h2s, 1):
        # Check if image exists
        patterns = [
            pub_dir / f"{idx:02d}.png",
            pub_dir / f"{idx:02d}.webp",
            pub_dir / f"section_{idx:02d}.png",
            pub_dir / f"section_{idx:02d}.webp",
        ]
        if any(p.exists() for p in patterns):
            continue
        missing.append((idx, heading))
    
    return missing

def generate_section_images(post_file, max_new=MAX_NEW_SECTIONS):
    """Generate missing section images for a blog post via Codex CLI."""
    content = post_file.read_text()
    slug = post_file.stem
    
    # Extract title from frontmatter
    title_match = re.search(r'^title:\s*(.+)', content, re.MULTILINE)
    title = title_match.group(1).strip().strip('"') if title_match else slug
    
    # Get stream
    first_tag = ""
    tags_match = re.search(r"^tags:\s*\[(.+?)\]", content, re.MULTILINE)
    if tags_match:
        tags = [t.strip().strip('"') for t in tags_match.group(1).split(",")]
        first_tag = tags[0].lower() if tags else ""
    
    # Map to stream
    stream = "ai"
    if "pm" in first_tag or "product" in first_tag:
        stream = "pm"
    elif "builder" in first_tag or "build" in first_tag:
        stream = "builder"
    
    # Remove frontmatter for body
    body_match = re.search(r'^---\s*\n(.*?)\n---\s*\n', content, re.DOTALL)
    body_md = content[body_match.end():] if body_match else content
    
    missing = get_missing_sections(post_file)
    if not missing:
        print(f"\n{'='*60}\n{slug}: ALL SECTIONS PRESENT ({len(re.findall(r'^## ', content, re.MULTILINE))} sections)")
        return 0
    
    print(f"\n{'='*60}\n{slug}: {len(missing)} missing sections (of {len(re.findall(r'^## ', content, re.MULTILINE))})")
    
    new = missing[:max_new]
    print(f"  Generating {len(new)} of {len(missing)} missing: {[h for _, h in new]}")
    
    # Build draft dict for art_director
    draft = {
        "title": title,
        "body_md": body_md,
        "stream": stream,
    }
    
    all_h2s = re.findall(r"^##\s+(.+?)\s*$", content, re.MULTILINE)
    
    # Get art brief
    recent = load_recent_styles()
    brief = build_art_brief(draft, all_h2s, recent_styles=recent)
    if brief is None:
        print("  art brief LLM unavailable - using fallback")
        brief = fallback_brief(draft, all_h2s, recent_styles=recent)
    print(f"  style={brief['style']}")
    record_style(brief["style"])
    
    generated = 0
    out_dir = PUB_DIR / slug
    
    for idx, heading in new:
        out_file = str(out_dir / f"{idx:02d}.png")
        concept = heading  # Use heading as concept
        prompt = compose_prompt(concept, brief)
        print(f"  [{idx:02d}] generating: {heading[:60]}...")
        
        result = generate_codex_image(prompt, out_file)
        if result == "cap":
            print("  CAP HIT - stopping batch")
            break
        elif result:
            generate_webp(out_file)
            generated += 1
    
    print(f"  Generated {generated} new section images for {slug}")
    return generated

def main():
    posts = sorted(SRC_DIR.glob("*.md*"))
    
    total = 0
    capped = False
    
    for post_file in posts:
        if capped:
            print(f"\nSKIP {post_file.stem}: Codex cap reached, deferring remaining")
            continue
        
        # Check if post needs images
        missing = get_missing_sections(post_file)
        if not missing:
            continue
        
        gen = generate_section_images(post_file, max_new=MAX_NEW_SECTIONS)
        total += gen
        if gen == 0:
            continue
        
        # Small delay between posts
        time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"TOTAL NEW IMAGES GENERATED: {total}")
    if capped:
        print("NOTE: Batch was capped. Run again later to continue.")

if __name__ == "__main__":
    main()
