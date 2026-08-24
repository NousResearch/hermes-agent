"""Article illustrator — baoyu-method illustrations for a finished article.

Workflow (port of the on-box baoyu-article-illustrator skill):
  1. analyse sections (markdown H2s)
  2. write outline.md (YAML frontmatter + one entry per illustration)
  3. for each illustration: write a prompt file first, then generate the
     image, then OCR-verify if expected labels are present
  4. insert ![alt](imgs/NN-<type>-<slug>.png) after the relevant section

Density: per-section (default) / balanced / hero-only. Capped at
ARTICLE_MAX_IMAGES so a long article cannot blow the budget.

Generation: existing FAL chain (seedream45 -> nano_banana) via
draft_media.generate_post_image; budget-gated via budget.can_spend;
OCR-verified via text integrity (already degraded when deps absent).
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import baoyu_loader as bl
import budget
import draft_media
import text_integrity
from config import ARTICLE_MAX_IMAGES

# Local aliases so tests can monkeypatch article_illustrator.X directly.
generate_post_image = draft_media.generate_post_image
verify_text = text_integrity.verify_text


# Map a signal_type (or mode) to a baoyu preset name. Falls back to
# edu-visual when the signal is unknown.
_PRESET_FOR = {
    "harness_change": "tech-explainer",
    "github_push": "tech-explainer",
    "hermes_pr": "tech-explainer",
    "hermes_skill": "tech-explainer",
    "research_signal": "science-paper",
    "gitradar_repo": "versus",
    "research_tool": "versus",
    "architecture": "architecture",
    "manifesto": "ink-notes-compare",
    "digest": "edu-visual",
}


def preset_for_signal(signal_or_mode: str) -> str:
    """Resolve a signal type or a mode to a baoyu preset name."""
    return _PRESET_FOR.get(signal_or_mode, "edu-visual")


def _h2_sections(body_md: str) -> list[str]:
    """Return the body text of every `## H2` section (heading + content),
    excluding the takeaway heading (per spec — the takeaway is a short
    prose sign-off, not a visual moment).
    """
    sections: list[str] = []
    current: list[str] = []
    in_section = False
    for line in body_md.splitlines():
        if line.startswith("## "):
            if in_section:
                sections.append("\n".join(current))
            current = [line]
            in_section = True
        elif in_section:
            current.append(line)
    if in_section:
        sections.append("\n".join(current))
    return [s for s in sections if not _is_takeaway_heading(_section_heading(s))]


def _is_takeaway_heading(heading: str) -> bool:
    """Heuristic: the takeaway is the prose sign-off, not a visual moment."""
    h = heading.lower()
    return ("try next" in h) or ("takeaway" in h) or ("try this" in h)


def _section_heading(section_md: str) -> str:
    for line in section_md.splitlines():
        if line.startswith("## "):
            return line[3:].strip()
    return ""


def _section_slug(text: str, max_words: int = 3) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return "-".join(words[:max_words]) or "section"


def _make_image_draft(preset_name: str, heading: str, slug: str, idx: int,
                      brand: str, is_hero: bool) -> dict:
    """Build the draft dict that build_image_prompt / generate_post_image consume."""
    title = (heading or slug).strip()
    body = (
        f"{preset_name} illustration for an article. "
        f"Section: {title}. Hero composition with one clear focal point." if is_hero
        else f"{preset_name} illustration for an article section. "
             f"Section heading: {title}. Visualise the concept, not the literal heading."
    )
    return {
        "id": f"art-{idx:02d}", "brand": brand, "platform": "twitter",
        "content_type": "infographic" if not is_hero else "hero",
        "pillar": preset_name, "topic": title, "title": title,
        "body_text": body, "visual_description": title,
    }


def write_outline(draft: dict, out_dir: Path, hero: bool = True) -> list[dict]:
    """Write outline.md and return the parsed list of illustration entries.

    Each entry: {"index": int, "type": str, "style": str, "slug": str,
                 "heading": str, "preset": str, "is_hero": bool}.
    """
    preset_name = preset_for_signal(
        draft.get("signals", [{}])[0].get("signal_type")
        if draft.get("signals") else "digest"
    )
    preset = bl.preset(preset_name) or {}
    sections = _h2_sections(draft.get("body_md", ""))
    entries: list[dict] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)

    idx = 1
    if hero:
        slug = _section_slug(draft.get("title", "hero"), max_words=4) or "hero"
        entries.append({
            "index": idx, "type": preset.get("type", "hero"),
            "style": preset.get("style", "warm"),
            "slug": slug, "heading": draft.get("title", "Hero"),
            "preset": preset_name, "is_hero": True,
        })
        idx += 1

    for sec in sections:
        head = _section_heading(sec) or "section"
        slug = _section_slug(head) or "section"
        entries.append({
            "index": idx, "type": preset.get("type", "infographic"),
            "style": preset.get("style", "vector-illustration"),
            "slug": slug, "heading": head,
            "preset": preset_name, "is_hero": False,
        })
        idx += 1

    # Write outline.md (YAML frontmatter + one entry per illustration).
    frontmatter = (
        "---\n"
        f"title: {draft.get('title', 'Article')}\n"
        f"mode: {draft.get('mode', 'deep_dive')}\n"
        f"pillar: {draft.get('pillar', 'general')}\n"
        f"density: hero+per-section\n"
        f"image_count: {len(entries)}\n"
        f"preset: {preset_name}\n"
        f"date: {os.environ.get('ARTICLE_DATE', '')}\n"
        "---\n\n"
    )
    body_lines = ["# Illustration Outline", ""]
    for e in entries:
        body_lines.append(f"## Illustration {e['index']}")
        body_lines.append(f"**Position**: {('hero' if e['is_hero'] else 'after section: ' + e['heading'])}")
        body_lines.append(f"**Purpose**: {('hero composition' if e['is_hero'] else 'illustrate the section concept')}")
        body_lines.append(f"**Visual Content**: type={e['type']}, style={e['style']}, heading=\"{e['heading']}\"")
        body_lines.append(f"**Filename**: {e['index']:02d}-{e['type']}-{e['slug']}.png")
        body_lines.append("")
    (out_dir / "outline.md").write_text(frontmatter + "\n".join(body_lines), encoding="utf-8")
    return entries


def _write_prompt_file(out_dir: Path, entry: dict, draft: dict) -> Path:
    """Write prompts/NN-<type>-<slug>.md (YAML frontmatter + structured prompt).

    BLOCKING: every image generation must have a saved prompt file. The
    file is the reproducibility / regen record.
    """
    preset = bl.preset(entry["preset"]) or {}
    style_block = bl.style_block(preset.get("style", "vector-illustration"))
    palette_block = bl.palette_block(preset.get("palette", "")) if preset.get("palette") else ""
    universal = bl.universal_rules()

    pcm = bl.prompt_construction_block()
    # Pick the type-specific template from prompt-construction.md (best
    # effort — fallback to a generic template when the upstream text is
    # not in the runtime cache).
    type_template = ""
    if pcm:
        m = re.search(rf"###\s+{re.escape(entry['type'].capitalize())}\s*\n(.*?)(?=\n###\s|\Z)",
                      pcm, re.DOTALL)
        if m:
            type_template = m.group(1).strip()

    frontmatter = (
        "---\n"
        f"illustration_id: {entry['index']:02d}\n"
        f"type: {entry['type']}\n"
        f"style: {entry['style']}\n"
        + (f"palette: {preset['palette']}\n" if preset.get("palette") else "")
        + "---\n\n"
    )
    body_parts = [
        f"# {entry['heading']}",
        "",
        "## Type-specific template",
        type_template or "Visualise the section concept in this article.",
        "",
        "## Style",
        style_block or "(style block unavailable)",
    ]
    if palette_block:
        body_parts += ["", "## Palette", palette_block]
    body_parts += ["", "## Universal rules", universal]
    body_parts += [
        "",
        "## Article context",
        (draft.get("context", "") or "")[:800],
        "",
        "## Section heading",
        entry["heading"],
    ]
    path = out_dir / "prompts" / f"{entry['index']:02d}-{entry['type']}-{entry['slug']}.md"
    path.write_text(frontmatter + "\n".join(body_parts), encoding="utf-8")
    return path


def _build_prompt(draft: dict) -> str:
    """Build the dimensional prompt via the existing prompt_engine."""
    from prompt_engine import build_image_prompt
    prompt, _aspect, _model, _expected = build_image_prompt(draft)
    return prompt


def _verify(path: Optional[str], expected: list[str]) -> bool:
    if not path or not expected:
        return True
    ok, _missing = text_integrity.verify_text(path, expected)
    return bool(ok)


def budget_can_spend(cost: float) -> bool:
    return budget.can_spend(cost)


# Per-position preset rotation so hero + sections never share one treatment.
_POSITION_PRESETS = ["science-paper", "architecture", "ink-notes-flow", "data-report"]
_POSITION_PALETTES = [None, None, "mono-ink", None]


def _section_subject(entry: dict, base_draft: dict) -> str:
    """Derive a concrete visual subject from the section's actual content.

    Uses up to ~400 chars of the section body (falling back to the heading)
    so the image model renders the section's specific argument instead of an
    abstract machine archetype.
    """
    body = (entry.get("body_md") or "").strip()
    if not body:
        body = str(base_draft.get("body_text") or "")
    # strip markdown noise for a cleaner subject line
    body = re.sub(r"[#*`>\[\]]", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) > 400:
        body = body[:400].rsplit(" ", 1)[0]
    if len(body) < 40:
        return entry["heading"]
    return f"{entry['heading']} — specifically: {body}"


def _entry_to_draft(entry: dict, base_draft: dict, is_hero: bool) -> dict:
    """Compose a self-contained draft dict from an outline entry.

    Inherits the article-level brand + platform + context but uses the
    entry's heading as the title and a CONTENT-DERIVED visual subject.
    The preset rotates per position so consecutive images differ in both
    composition and treatment (fixes the funnel-machine convergence).
    """
    idx = int(entry.get("index", 0))
    pos = max(0, idx - 1)  # 0-based position within the article
    preset_name = entry["preset"] if is_hero else _POSITION_PRESETS[pos % len(_POSITION_PRESETS)]
    palette = None if is_hero else _POSITION_PALETTES[pos % len(_POSITION_PALETTES)]
    subject = entry["heading"] if is_hero else _section_subject(entry, base_draft)
    return {
        **base_draft,
        "id": f"art-{entry['index']:02d}",
        "content_type": "hero" if is_hero else "infographic",
        "pillar": preset_name,
        "_preset_override": preset_name,
        "_palette_override": palette,
        "topic": entry["heading"],
        "title": entry["heading"],
        "body_text": (f"{preset_name} illustration for an article section. "
                      f"Visualise this specific argument: {subject[:300]}"),
        "visual_description": subject,
    }


def illustrate(draft: dict, out_dir: Path, density: str = "per-section",
               max_images: int = ARTICLE_MAX_IMAGES) -> str:
    """Illustrate a finished article. Writes outline + prompts + images,
    returns the body markdown with image references inserted.

    Density rules:
      - per-section: hero + 1 image per H2 (capped at max_images total)
      - balanced: hero + min(3, H2) section images
      - hero-only: hero only
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "imgs").mkdir(parents=True, exist_ok=True)
    (out_dir / "prompts").mkdir(parents=True, exist_ok=True)

    sections = _h2_sections(draft.get("body_md", ""))
    hero_count = 1 if density in ("per-section", "balanced", "hero-only") else 0
    if density == "hero-only":
        section_cap = 0
    elif density == "balanced":
        section_cap = min(3, len(sections))
    else:  # per-section
        section_cap = max(0, max_images - hero_count)
        section_cap = min(section_cap, len(sections))
    chosen_sections = sections[:section_cap]

    # Budget cap: if we cannot spend on at least one image, fall back to
    # hero-only and emit a warning.
    if not budget_can_spend(0.05):
        print("[article_illustrator] budget cap hit, degraded to hero-only",
              file=sys.stderr)
        chosen_sections = []
        density = "hero-only"

    # Build outline with only the chosen illustrations.
    out = {
        "title": draft.get("title", "Article"), "body_md": draft.get("body_md", ""),
        "mode": draft.get("mode", "deep_dive"), "pillar": draft.get("pillar", "general"),
        "signals": draft.get("signals", []), "context": draft.get("context", ""),
    }
    entries = write_outline(out, out_dir, hero=(hero_count == 1))
    # Filter entries to the chosen set.
    chosen = []
    if hero_count == 1:
        chosen.append(entries[0])
    chosen.extend(entries[1:1 + len(chosen_sections)])

    # Write prompt files (BLOCKING) before generating any image.
    for entry in chosen:
        _write_prompt_file(out_dir, entry, out)

    # Generate each image.
    base_draft = {
        # Thread the article's real brand (e.g. sahilblog) so prompt_engine
        # applies the brand's own BRAND_STYLE_MAP entry instead of collapsing
        # every article to the sahil_twitter preset (funnel-machine bug).
        "brand": draft.get("brand") or "sahilblog", "platform": "twitter",
        "context": out.get("context", ""), "kb_snippets": draft.get("kb_snippets", []),
    }
    for entry in chosen:
        is_hero = bool(entry.get("is_hero"))
        img_draft = _entry_to_draft(entry, base_draft, is_hero=is_hero)
        try:
            img_path = generate_post_image(img_draft)
        except Exception as exc:  # noqa: BLE001
            print(f"[article_illustrator] {entry['index']:02d} image failed: {exc}",
                  file=sys.stderr)
            continue
        if not img_path:
            continue
        # Move/copy into out_dir/imgs/NN-type-slug.png.
        target = out_dir / "imgs" / f"{entry['index']:02d}-{entry['type']}-{entry['slug']}.png"
        try:
            target.write_bytes(Path(img_path).read_bytes())
        except Exception as exc:  # noqa: BLE001
            print(f"[article_illustrator] image copy failed: {exc}", file=sys.stderr)

    # Insert image references into the body markdown.
    body = out.get("body_md", "")
    body = _insert_images(body, chosen, out_dir)
    return body


def _insert_images(body_md: str, chosen: list[dict], out_dir: Path) -> str:
    """Insert ![alt](imgs/NN-type-slug.png) after each section heading (or
    after the title for the hero).
    """
    lines = body_md.splitlines()
    if not lines:
        return body_md
    out_lines: list[str] = []
    inserted: set[int] = set()

    def _image_line(entry: dict) -> str:
        return f"![{entry['heading']}](imgs/{entry['index']:02d}-{entry['type']}-{entry['slug']}.png)"

    for line in lines:
        out_lines.append(line)
        # Hero goes after the first `# ` (title).
        if line.startswith("# ") and any(e.get("is_hero") for e in chosen):
            for e in chosen:
                if e.get("is_hero") and e["index"] not in inserted:
                    out_lines.append("")
                    out_lines.append(_image_line(e))
                    inserted.add(e["index"])
        elif line.startswith("## "):
            head = line[3:].strip()
            for e in chosen:
                if not e.get("is_hero") and e["heading"] == head and e["index"] not in inserted:
                    out_lines.append("")
                    out_lines.append(_image_line(e))
                    inserted.add(e["index"])
    return "\n".join(out_lines)
