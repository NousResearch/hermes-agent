#!/usr/bin/env python3
"""Run the full inbox-fetch workflow using direct Python imports."""
import json
import os
import sys
import uuid
from pathlib import Path

# Setup path and env
WORKDIR = "/home/kensei/repos/KenseiAgent/content_engine"
ENV_FILE = "/home/kensei/.hermes/.env"

sys.path.insert(0, WORKDIR)
os.chdir(WORKDIR)

# Load .env
if os.path.exists(ENV_FILE):
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ[key] = val

os.environ["PYTHONPATH"] = WORKDIR

# Now import engine modules
from database import init_db, insert_draft, update_draft_ai_image_path, list_recent_drafts
from repurpose import fetch_inbox

# Step 1: inbox-fetch
print("=== STEP 1: inbox-fetch ===", flush=True)
init_db()

try:
    items = fetch_inbox()
except Exception as e:
    print(f"ERROR fetching inbox: {e}")
    items = []

print(f"Raw items returned: {len(items)}", flush=True)

if not items:
    print("[SILENT]")
    print("=== No inbox items to process ===")
    sys.exit(0)

# Print each item as JSON for the record
for it in items:
    print(f"ITEM: {json.dumps(it, indent=2)[:500]}", flush=True)

draft_results = []

# Brand/platform map
PLATFORM_MAP = {
    "sahil_twitter": "x",
    "sahil_linkedin": "linkedin",
    "plenishd": "instagram",
    "coachos": "instagram",
    "matchdaymaestro": "x",
}

def detect_brand(caption):
    caption_lower = caption.lower()
    if any(kw in caption_lower for kw in ["plenish", "cook", "kitchen", "recipe", "meal", "food", "dish", "ingredient", "bake", "roast"]):
        return "plenishd"
    if any(kw in caption_lower for kw in ["coach", "coaching", "client", "training", "fitness", "session", "workout", "drill"]):
        return "coachos"
    if any(kw in caption_lower for kw in ["matchday", "maestro", "football", "soccer", "gameweek", "fantasy", "predict", "stadium", "pitch", "goalscorer", "premier league", "champions league"]):
        return "matchdaymaestro"
    if any(kw in caption_lower for kw in ["product", "pm ", "leadership", "strategy", "roadmap", "stakeholder", "team", "vision", "shipped", "build"]):
        return "sahil_linkedin"
    return "sahil_twitter"

def gen_copy(brand, caption):
    """On-brand copy, British English, no em-dashes, no AI filler."""
    copies = {
        "sahil_twitter": f"{caption}\n\nshipped it. no hype. just the numbers and what i learned.",
        "sahil_linkedin": f"{caption}\n\nproduct is about asking better questions, not having all the answers. here is what that looks like when you build in the open.",
        "plenishd": f"{caption}\n\nsunday prep sorted. a full week of flavour with half the effort. this is how we do it in my kitchen.",
        "coachos": f"{caption}\n\nyour clients already know what to do. the gap is execution, not education. here is the system that closes it.",
        "matchdaymaestro": f"{caption}\n\nplace your predictions, stake your coins, and watch the table shift. no betting talk here, just pure football banter and smart picks.",
    }
    return copies.get(brand, f"{caption}\n\nbuilding in public, raw and real.")

for idx, item in enumerate(items):
    print(f"\n{'='*60}", flush=True)
    print(f"Processing item {idx+1}/{len(items)}", flush=True)
    
    msg_id = item.get("msg_id", "unknown")
    caption = item.get("caption", "")
    images = item.get("images", [])
    links = item.get("links", [])
    videos = item.get("videos", [])
    
    print(f"msg_id: {msg_id}", flush=True)
    print(f"caption: {caption[:200] if caption else '(empty)'}", flush=True)
    print(f"images: {images}", flush=True)
    print(f"links: {links[:3]}", flush=True)
    print(f"videos: {videos}", flush=True)
    
    brand = detect_brand(caption)
    platform = PLATFORM_MAP.get(brand, "x")
    print(f"Detected brand: {brand}, platform: {platform}", flush=True)
    
    title_words = caption.strip().split()[:7]
    short_title = " ".join(title_words) if title_words else "repurposed post"
    if len(short_title) > 60:
        short_title = short_title[:60]
    
    copy = gen_copy(brand, caption)
    print(f"Generated copy:\n{copy}", flush=True)
    
    # Create draft directly via database
    draft_id = f"{brand[:4]}_{uuid.uuid4().hex[:8]}"
    insert_draft(
        draft_id=draft_id,
        brand=brand,
        platform=platform,
        pillar="repurpose",
        topic="repurpose",
        title=short_title,
        body_text=copy,
        content_type="text+image",
    )
    print(f"Draft created: {draft_id}", flush=True)
    
    # Generate image. No scene_prompt: the Track A prompt engine (or the
    # infographic router) art-directs from the draft itself, so every post
    # gets a varied, brand-distinct prompt instead of a static one-liner.
    try:
        from draft_media import generate_post_image
        from database import get_draft
        d = get_draft(draft_id)
        if d:
            d = dict(d)
            d.setdefault("visual_description", caption.strip()[:120])
            path = generate_post_image(d)
            if path:
                update_draft_ai_image_path(draft_id, path)
                print(f"Image generated: {path}", flush=True)
            else:
                print(f"Image generation returned None", flush=True)
        else:
            print(f"Draft {draft_id} not found after insert", flush=True)
    except Exception as e:
        print(f"Image generation error: {e}", flush=True)
    
    draft_results.append((brand, draft_id, short_title))

# Step 5: deliver-discord
print(f"\n{'='*60}", flush=True)
print("=== STEP 5: deliver-discord ===", flush=True)

try:
    from discord_digest import deliver_discord_digest
    drafts = list_recent_drafts(90)
    print(f"Found {len(drafts)} recent drafts for digest", flush=True)
    ok = deliver_discord_digest(drafts)
    print(f"Deliver result: {'OK' if ok else 'FAILED'}", flush=True)
except Exception as e:
    print(f"deliver-discord error: {e}", flush=True)

# Summary
print(f"\n{'='*60}", flush=True)
print("=== SUMMARY ===", flush=True)
for brand, did, title in draft_results:
    print(f"{brand} {did} {title}", flush=True)