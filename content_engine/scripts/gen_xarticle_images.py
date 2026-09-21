"""Generate missing images for X/Twitter Article bundles using Codex CLI.

Uses the same art-directed flow as blog_illustrator:
  1. Find ![](imgs/XX-*) refs in article.md
  2. Generate via codex exec with art-directed prompts
  3. Save to bundle imgs/ directory

Paces at ~3 images/post to stay under Codex 50/hr cap.
"""
from __future__ import annotations

import os, re, shutil, subprocess, time, json, glob
from pathlib import Path
from typing import Optional

# Paths
BUNDLES_DIR = Path("/home/kensei/repos/KenseiAgent/content_engine/output/articles")
CODEX_IMAGES_DIR = Path.home() / ".codex" / "generated_images"
ROTATION_PATH = Path("/home/kensei/repos/KenseiAgent/content_engine/blog_topics/skill_rotation.json")

# ── Style rotation (same as blog) ──
STYLE_ORDER = [
    "signal-hud", "ninth-observatory", "mythic-tech-codex",
    "baoyu-infographic", "chromatic-institute", "ink-ember-studio",
    "cosmic-postcard", "saga-noir", "pixel-art",
]

def load_style_index() -> int:
    try:
        data = json.loads(ROTATION_PATH.read_text())
        return data.get("x_article_style_idx", 0)
    except:
        return 0

def save_style_index(idx: int):
    ROTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    if ROTATION_PATH.exists():
        try:
            payload = json.loads(ROTATION_PATH.read_text())
        except:
            pass
    payload["x_article_style_idx"] = idx
    ROTATION_PATH.write_text(json.dumps(payload, indent=2) + "\n")

def next_style() -> tuple[str, str]:
    idx = load_style_index()
    style = STYLE_ORDER[idx % len(STYLE_ORDER)]
    save_style_index(idx + 1)
    return style, _style_guidance(style)

def _style_guidance(style_id: str) -> str:
    """Brief direction for each style."""
    guides = {
        "signal-hud": "dark technical dashboard aesthetic; neon data-flows on near-black; one glowing focal element; diagnostic mood; no text/UI",
        "ninth-observatory": "systems-as-architecture; vast built spaces, cross-sections of halls/conveyors; muted stone-and-brass palette; single warm focal light",
        "mythic-tech-codex": "Edwardian/Victorian scientific-illustration plate; ink linework and antique watercolour; sepia and aged-paper tones; museum encyclopedia feel",
        "baoyu-infographic": "clean modern explainer infographic; flat/isometric structured panels, flow arrows; soft palette teal/coral/navy/amber on cream; legible diagrammatic storytelling",
        "chromatic-institute": "clean modern research abstraction; networked nodes and fields; confident saturated colour on light ground; Bauhaus-ish geometry",
        "ink-ember-studio": "painterly human-centred fine art; warm emotive palette; real people and gesture; editorial-magazine feel",
        "cosmic-postcard": "mid-century Space Age retro-futurism; travel-poster composition; warm oranges/teals/cream; cinematic horizon",
        "saga-noir": "bold graphic-novel mythology; high-contrast ink; dramatic silhouettes; limited spot colour",
        "pixel-art": "deliberate pixel-art; visible pixel blocks, limited palette; SNES/PICO-8 energy with optional CRT glow",
    }
    return guides.get(style_id, "modern editorial illustration; confident colour; high contrast; no text")

GLOBAL_RULES = (
    "No text, letters, numbers, captions, logos, watermarks, or UI screenshots "
    "anywhere in the image. 16:9 landscape. High detail, intentional editorial "
    "composition, not generic AI art."
)

# ── Codex generation ──
CODEX_CAP_SIGNALS = ("usage_limit_reached", "hit your usage limit", "usage limit has been reached", "resets_in_seconds")

def output_shows_cap(text: str) -> bool:
    low = (text or "").lower()
    return any(sig in low for sig in CODEX_CAP_SIGNALS)

def find_latest_codex_image(after_ts: Optional[float] = None) -> Optional[str]:
    if not CODEX_IMAGES_DIR.exists():
        return None
    candidates: list[tuple[float, str]] = []
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

def generate_codex_image(prompt: str, out_path: str, timeout: int = 300) -> str | None:
    """Generate via codex exec, copy to out_path. Returns out_path, or 'cap' if limit hit."""
    for attempt, t in enumerate([timeout, timeout + 60], 1):
        before_ts = time.time()
        try:
            result = subprocess.run(
                ["codex", "exec", "--disable", "use_linux_sandbox_bwrap", prompt],
                capture_output=True, text=True, timeout=t,
                cwd=str(BUNDLES_DIR),
            )
            print(f"    codex exit={result.returncode} (attempt {attempt})")
        except subprocess.TimeoutExpired:
            print(f"    codex timed out after {t}s (attempt {attempt})")
            continue
        except Exception as exc:
            print(f"    codex error: {exc}")
            continue

        if output_shows_cap(result.stdout) or output_shows_cap(result.stderr):
            print("    >>> CODEX CAP REACHED <<<")
            return "cap"

        img = find_latest_codex_image(after_ts=before_ts)
        if not img or not Path(img).exists():
            print(f"    no image found after codex run")
            continue

        size_kb = Path(img).stat().st_size // 1024
        print(f"    generated {size_kb}KB")
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_path)
            if Path(out_path).exists():
                print(f"    saved: {Path(out_path).stat().st_size} bytes")
                return out_path
        except Exception as exc:
            print(f"    copy failed: {exc}")
            continue

    print(f"    all attempts failed")
    return None

def build_prompt(alt_text: str, style_id: str, style_guide: str, article_context: str) -> str:
    """Build an art-directed Codex image prompt."""
    # Use the alt text as the subject
    return (
        f"Illustration for a technical article section titled '{alt_text}'. "
        f"{article_context[:200]}. "
        f"Style: {style_guide}. {GLOBAL_RULES}"
    )

def process_bundle(bundle_dir: Path) -> int:
    """Generate missing images for one article bundle. Returns count generated."""
    article_md = bundle_dir / "article.md"
    if not article_md.exists():
        print(f"  SKIP: no article.md")
        return 0

    content = article_md.read_text()
    name = bundle_dir.name

    # Find all image references: ![alt](imgs/XX-*.png)
    refs = re.findall(r'!\[(.*?)\]\(imgs/(\d+.*?\.(?:png|webp))\)', content)
    if not refs:
        print(f"  SKIP: no image refs in article")
        return 0

    imgs_dir = bundle_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    # Filter to missing images only
    missing = []
    for alt_text, filename in refs:
        out_path = imgs_dir / filename
        if not out_path.exists():
            missing.append((alt_text, filename, out_path))

    if not missing:
        print(f"  DONE: all {len(refs)} images present")
        return 0

    print(f"  {len(missing)}/{len(refs)} images missing: {[f for _, f, _ in missing]}")

    # Get article context (first few paragraphs, skip image refs)
    article_context = re.sub(r'!\[.*?\]\(.*?\)', '', content)[:500]

    generated = 0
    caps = False

    for alt_text, filename, out_path in missing:
        if caps:
            print(f"  SKIP {filename}: cap hit earlier in batch")
            continue

        style_id, style_guide = next_style()
        print(f"  [{filename}] style={style_id} alt='{alt_text[:60]}'")

        prompt = build_prompt(alt_text, style_id, style_guide, article_context)
        result = generate_codex_image(prompt, str(out_path))

        if result == "cap":
            caps = True
            print(f"  >>> CAP HIT - stopping batch <<<")
            break
        elif result:
            generated += 1

        # Pace: ~2s between images
        time.sleep(2)

    print(f"  Generated {generated} images for {name}")
    return generated

def main():
    bundles = sorted(BUNDLES_DIR.glob("2*"))
    bundles = [b for b in bundles if b.is_dir() and (b / "article.md").exists()]

    print(f"=== X/Twitter Articles: {len(bundles)} bundles ===\n")

    total = 0
    caps = False

    for bundle in bundles:
        if caps:
            print(f"SKIP {bundle.name}: Codex cap reached, deferring")
            continue

        print(f"\n--- {bundle.name} ---")
        gen = process_bundle(bundle)
        total += gen

    print(f"\n{'='*60}")
    print(f"TOTAL IMAGES GENERATED: {total}")
    if caps:
        print("NOTE: Codex cap hit mid-batch. Resume later.")

if __name__ == "__main__":
    main()
