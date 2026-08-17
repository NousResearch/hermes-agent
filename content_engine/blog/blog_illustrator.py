"""Blog illustrator — art-directed image set with a provider-free P11 seam.

P11 persists an immutable visual plan and a planned provenance manifest before
any legacy image generator runs. It does not change the current provider,
provider configuration or publishing path; P10 owns that future replacement.
"""
from __future__ import annotations


import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import config
from blog import art_director
from blog.art_director import build_art_brief, fallback_brief, compose_prompt
from blog.asset_manifest import AssetManifestError, save_asset_manifest
from blog.reference_catalog import CatalogIntegrityError
from blog.visual_plan import VisualPlanError, save_visual_plan


# ── Rotation state (persisted so variety survives cron restarts) ──
ROTATION_STATE_PATH = Path(__file__).parent.parent / "blog_topics" / "skill_rotation.json"
ROTATION_HISTORY_LIMIT = 6


def _load_recent_styles() -> list[str]:
    try:
        data = json.loads(ROTATION_STATE_PATH.read_text())
    except Exception:
        return []
    history = data.get("history", []) if isinstance(data, dict) else data
    if not isinstance(history, list):
        return []
    valid = art_director.STYLE_IDS
    return [x for x in history if isinstance(x, str) and x in valid][-ROTATION_HISTORY_LIMIT:]


def _record_style(style_id: str) -> None:
    history = _load_recent_styles()
    history.append(style_id)
    ROTATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"history": history[-ROTATION_HISTORY_LIMIT:],
               "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    ROTATION_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n")


# ── Codex CLI generation ────────────────────────────────────────
CODEX_IMAGES_DIR = Path.home() / ".codex" / "generated_images"


# Substrings that mean the ChatGPT/Codex image quota is exhausted (both the raw
# HTTP 429 form and the friendly weekly-cap CLI message). A capped run is NOT a
# per-image failure — the whole batch should defer, not count attempts.
_CODEX_CAP_SIGNALS = (
    "usage_limit_reached",
    "hit your usage limit",
    "usage limit has been reached",
    "resets_in_seconds",
)


class CodexCapExceeded(Exception):
    """Raised (when raise_on_cap=True) when Codex reports its usage cap."""


def _output_shows_cap(text: str) -> bool:
    low = (text or "").lower()
    return any(sig in low for sig in _CODEX_CAP_SIGNALS)


def _generate_codex_image(full_prompt: str, out_path: str,
                          timeout: int = 300,
                          retry_timeout: int = 360,
                          raise_on_cap: bool = False) -> Optional[str]:
    """Generate an image via Codex CLI and copy it to out_path.

    Codex generates internally then we copy the newest image out of
    ~/.codex/generated_images/. Retries once with a longer timeout.
    Returns out_path on success, None on failure. When raise_on_cap is set and
    Codex reports its usage cap, raises CodexCapExceeded so batch callers can
    defer instead of burning retries (a capped 429 does not consume quota).
    """
    for attempt, current_timeout in enumerate([timeout, retry_timeout], 1):
        before_ts = time.time()
        try:
            result = subprocess.run(
                ["codex", "exec", "--disable", "use_linux_sandbox_bwrap",
                 full_prompt],
                capture_output=True, text=True, timeout=current_timeout,
                cwd=str(config.SAHILBLOG_REPO),
            )
            print(f"[blog_illustrator] codex exec exit={result.returncode} (attempt {attempt})")
            if result.returncode != 0:
                print(f"[blog_illustrator] codex stdout: {(result.stdout or '')[-500:]}")
                print(f"[blog_illustrator] codex stderr: {(result.stderr or '')[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"[blog_illustrator] codex timed out after {current_timeout}s (attempt {attempt})")
            continue
        except Exception as exc:
            print(f"[blog_illustrator] codex execution error: {exc}")
            continue

        if _output_shows_cap(result.stdout) or _output_shows_cap(result.stderr):
            print("[blog_illustrator] Codex usage cap reached")
            if raise_on_cap:
                raise CodexCapExceeded("Codex image usage cap reached")
            return None

        img = _find_latest_codex_image(after_ts=before_ts)
        if not img or not Path(img).exists():
            print(f"[blog_illustrator] no image found after codex run (attempt {attempt})")
            continue

        size_kb = Path(img).stat().st_size // 1024
        print(f"[blog_illustrator] generated {size_kb}KB -> {img}")
        try:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, out_path)
        except Exception as exc:
            print(f"[blog_illustrator] copy failed (attempt {attempt}): {exc}")
            continue
        if Path(out_path).exists():
            print(f"[blog_illustrator] verified: {Path(out_path).stat().st_size} bytes")
            return out_path

    print(f"[blog_illustrator] all Codex attempts failed for {out_path}")
    return None


def _find_latest_codex_image(
    after_ts: Optional[float] = None,
    images_dir: Path = CODEX_IMAGES_DIR,
) -> Optional[str]:
    """Find the newest Codex-generated image after a timestamp (by mtime)."""
    if not images_dir.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for session_dir in images_dir.iterdir():
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


def _generate_webp(png_path: str) -> Optional[str]:
    """Generate a WebP copy alongside the PNG for the Astro <picture> tag.

    Always overwrites: PostLayout serves the .webp first, so a stale webp left
    beside a freshly regenerated png would keep the OLD image live.
    """
    webp_path = Path(png_path).with_suffix(".webp")
    try:
        result = subprocess.run(
            [
                "node", "-e",
                f"const s=require('sharp');"
                f" s('{png_path}').webp({{quality:90,effort:6}}).toFile('{webp_path}')"
                f" .then(()=>console.log('ok'))",
            ],
            capture_output=True, timeout=60,
            cwd=str(config.SAHILBLOG_REPO),
        )
        if result.returncode == 0 and webp_path.exists():
            print(f"[blog_illustrator] webp generated: {webp_path} ({webp_path.stat().st_size} bytes)")
            return str(webp_path)
    except Exception as exc:
        print(f"[blog_illustrator] webp generation skipped: {exc}")
    return None


# ── Section parsing ─────────────────────────────────────────────

def _extract_h2_headings(body_md: str) -> list[str]:
    """Extract H2 heading texts from markdown body."""
    out = []
    for line in body_md.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


def _extract_diagram_spec(draft: dict) -> Optional[str]:
    """Extract Mermaid diagram code from a blueprint-format draft body."""
    body = draft.get("body_md", "") or ""
    m = re.search(r"```mermaid\n(.*?)```", body, re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_section_text(body_lines: list[str], heading: str) -> str:
    """Return the body text under a given H2 heading (for fallback concepts)."""
    found = False
    lines: list[str] = []
    for line in body_lines:
        if re.match(r"^##\s+" + re.escape(heading) + r"\s*$", line):
            found = True
            continue
        if found:
            if line.startswith("## "):
                break
            lines.append(line)
    return " ".join(l.strip() for l in lines if l.strip())[:400]


# ── Public API ──────────────────────────────────────────────────

# One hero plus this hard maximum preserves the scheduler's max_images=3 bound,
# even when a direct caller supplies an unsafe config or max_sections override.
MAX_SECTION_IMAGES = 2


def illustrate(
    draft: dict,
    out_dir: Optional[Path] = None,
    max_sections: Optional[int] = None,
    workdir: Optional[str] = None,
    raise_on_cap: bool = False,
) -> dict:
    """Generate an art-directed hero + section image set via Codex CLI.

    One art brief (style + locked palette/motif + shared direction) drives the
    whole post; each image gets a unique, article-grounded prompt composed
    against that shared direction, so the set is consistent but not repetitive.

    Returns {hero_path: str|None, section_paths: {h2_heading: path}}.
    """
    stream = draft.get("stream", "ai")
    if max_sections is None:
        max_sections = config.BLOG_MAX_SECTION_IMAGES
    max_sections = min(max(0, max_sections), MAX_SECTION_IMAGES)
    out_path = Path(out_dir) if out_dir else Path(config.OUTPUT_DIR) / "blog_images"
    out_path.mkdir(parents=True, exist_ok=True)

    result: dict = {"hero_path": None, "section_paths": {}}

    body_md = draft.get("body_md", "")
    headings = _extract_h2_headings(body_md)[:max_sections] if max_sections > 0 else []

    # One art brief for the whole post (LLM). HARD FAIL if unavailable.
    # The fallback_brief produces generic, article-disconnected images with no
    # palette, motif, or per-section art direction. Silently using it shipped
    # dozens of terrible images to production. Now we stop and report instead.
    recent = _load_recent_styles()
    brief = build_art_brief(draft, headings, recent_styles=recent)
    if brief is None:
        print("[blog_illustrator] ⚠️  ART DIRECTOR FAILED — refusing to generate images.")
        print("[blog_illustrator] The LLM art brief is mandatory. Fallback brief produces")
        print("[blog_illustrator] generic images with no article-specific art direction.")
        print(f"[blog_illustrator] SKIPPED: {draft.get('title', '?')}")
        return result
    print(f"[blog_illustrator] art brief: style={brief['style']} "
          f"seed={brief.get('selection_seed', 'n/a')} "
          f"layout={brief.get('layout', '')[:80]!r} "
          f"palette={brief.get('palette','')[:60]!r} motif={brief.get('motif','')[:60]!r}")
    _record_style(brief["style"])

    # P11 contract seam: select only reviewed core references, write the plan
    # and planned provenance before the unchanged legacy generator is reached.
    asset_layouts = brief.get("asset_layouts", {})
    planned_prompts = {
        "hero": compose_prompt(
            brief["hero_prompt"], brief, assigned_layout=asset_layouts.get("hero")
        )
    }
    planned_outputs = {"hero": "hero.png"}
    section_prompts = brief.get("section_prompts", {})
    body_lines = body_md.splitlines()
    for index, heading in enumerate(headings, 1):
        concept = section_prompts.get(heading) or _extract_section_text(body_lines, heading) or heading
        key = f"section-{index:02d}"
        planned_prompts[key] = compose_prompt(
            concept, brief, assigned_layout=asset_layouts.get(key)
        )
        planned_outputs[key] = f"section_{index:02d}.png"
    try:
        visual_plan = art_director.build_visual_plan_from_brief(
            draft,
            headings,
            brief,
            catalog_root=Path(config.IMAGERY_ANCHORS_DIR),
        )
        planned_manifest = art_director.build_planned_asset_manifest_from_plan(
            visual_plan,
            planned_prompts,
            planned_outputs,
            text_policy=str(brief.get("text_policy", "none")),
        )
        visual_plan_path = save_visual_plan(visual_plan, out_path / "visual-plan.json")
        asset_manifest_path = save_asset_manifest(
            planned_manifest, out_path / "asset-manifest.json"
        )
    except (AssetManifestError, CatalogIntegrityError, OSError, ValueError, VisualPlanError) as exc:
        print(f"[blog_illustrator] P11 visual contract failed — refusing generation: {exc}")
        return result
    result["visual_plan_path"] = str(visual_plan_path)
    result["asset_manifest_path"] = str(asset_manifest_path)

    # ── Hero ────────────────────────────────────────────────
    hero_out = str(out_path / "hero.png")
    hero_prompt = planned_prompts["hero"]
    gen = _generate_codex_image(hero_prompt, hero_out, raise_on_cap=raise_on_cap)
    if gen and Path(gen).exists():
        result["hero_path"] = gen
        _generate_webp(gen)
        print(f"[blog_illustrator] hero via {gen}")
    else:
        print(f"[blog_illustrator] hero generation failed for '{draft.get('title','')}'")

    # ── Sections ────────────────────────────────────────────
    if headings:
        for idx, heading in enumerate(headings, 1):
            key = f"section-{idx:02d}"
            section_out = str(out_path / planned_outputs[key])
            prompt = planned_prompts[key]
            gen_sec = _generate_codex_image(prompt, section_out, raise_on_cap=raise_on_cap)
            if gen_sec and Path(gen_sec).exists():
                result["section_paths"][heading] = gen_sec
                _generate_webp(gen_sec)
                print(f"[blog_illustrator] section {idx} via {gen_sec}")
            else:
                print(f"[blog_illustrator] section '{heading}' generation failed")

    return result
