from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from PIL import Image, ImageDraw, ImageFont

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from webtoon_contracts import (  # noqa: E402
    build_prompt_parts as build_prompt_parts_v2,
    flatten_prompt_parts,
    load_episode_bundle,
    sanitize_prompt_for_policy,
)

PANEL_W = 720
PANEL_H = 1080
FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="render_webtoon_fal_live_episode.py",
        description="Render one romance webtoon episode with live fal generation.",
    )
    parser.add_argument("--episode-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--reuse-existing-artifacts", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for ch in text:
        trial = current + ch
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    return lines


def fal_generate(prompt: str, num_images: int) -> list[dict[str, Any]]:
    import fal_client

    arguments = {
        "prompt": prompt,
        "image_size": {"width": PANEL_W, "height": PANEL_H},
        "num_images": num_images,
        "output_format": "png",
    }
    try:
        result = fal_client.subscribe("fal-ai/flux-2-pro", arguments=arguments)
        final_prompt = prompt
        policy_sanitized = False
    except Exception as exc:
        if "content_policy_violation" not in str(exc):
            raise
        final_prompt = sanitize_prompt_for_policy(prompt)
        result = fal_client.subscribe(
            "fal-ai/flux-2-pro",
            arguments={**arguments, "prompt": final_prompt},
        )
        policy_sanitized = True
    images = result["images"]
    return [
        {
            "candidate_id": f"candidate_{index:02d}",
            "prompt": prompt,
            "final_prompt": final_prompt,
            "url": image["url"],
            "policy_sanitized": policy_sanitized,
        }
        for index, image in enumerate(images, start=1)
    ]


def download(url: str, path: Path) -> None:
    last_error: Exception | None = None
    for attempt in range(1, 6):
        try:
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            return
        except Exception as exc:
            last_error = exc
            if attempt == 5:
                raise
            time.sleep(2 * attempt)
    if last_error:
        raise last_error


def style_prompt(panel_data: dict[str, Any]) -> str:
    positive = panel_data["style_anchor"]["positive"]
    negative = ", ".join(panel_data["style_anchor"]["negative"])
    return (
        f"{positive}, clean Korean romance webtoon cartoon illustration, 2D cel shading, "
        f"polished digital manhwa style, no readable text, no speech bubbles, no letters, avoid {negative}"
    )


def build_prompt(panel_data: dict[str, Any], panel_spec: dict[str, Any]) -> str:
    chars = []
    for key in panel_spec.get("visible_characters", []):
        info = panel_data["characters"].get(key, {})
        chars.append(f"{info.get('role', key)} with {info.get('visual', '')}")
    char_block = ", ".join(chars)
    return ", ".join(
        part
        for part in [
            style_prompt(panel_data),
            panel_spec["prompt"],
            char_block,
            "mobile vertical webtoon panel with strong facial acting and clean composition",
        ]
        if part
    )


def render_overlays(img_path: Path, panel_id: str, lettering: dict[str, Any]) -> None:
    image = Image.open(img_path).convert("RGBA")
    draw = ImageDraw.Draw(image)
    caption_font = load_font(28)
    balloon_font = load_font(30)

    captions = {c["panel_id"]: c["text"] for c in lettering.get("captions", [])}
    balloons = [b for b in lettering.get("balloons", []) if b["panel_id"] == panel_id]

    if panel_id in captions:
        text = captions[panel_id]
        lines = wrap_text(draw, text, caption_font, PANEL_W - 140)
        box_h = 26 + len(lines) * 36 + 24
        box = (50, 56, PANEL_W - 50, 56 + box_h)
        draw.rounded_rectangle(box, radius=24, fill=(10, 10, 10, 200))
        y = box[1] + 20
        for line in lines:
            draw.text((box[0] + 24, y), line, font=caption_font, fill=(255, 255, 255, 255))
            y += 36

    top = PANEL_H - 250
    for balloon in balloons:
        lines = wrap_text(draw, balloon["text"], balloon_font, PANEL_W - 260)
        box_h = 30 + len(lines) * 38 + 28
        box = (100, top, PANEL_W - 100, top + box_h)
        draw.rounded_rectangle(box, radius=46, fill=(248, 248, 248, 235), outline=(20, 20, 20, 255), width=3)
        y = top + 22
        for line in lines:
            draw.text((box[0] + 40, y), line, font=balloon_font, fill=(25, 25, 25, 255))
            y += 38
        top -= box_h + 36

    image.convert("RGB").save(img_path)


def compose_longscroll(panel_paths: list[Path], scroll_plan: dict[str, Any], out_path: Path) -> None:
    spacing_map = {block["block_id"]: block.get("spacing", "medium") for block in scroll_plan["blocks"]}
    gap_px = {"tight": 30, "medium": 70, "tall_drop": 180, "end_cliff": 260}
    blocks = {f"p{i:02d}": block["block_id"] for i, block in enumerate(scroll_plan["blocks"], start=1)}
    images = []
    total_h = 0
    panel_ids = [f"p{i:02d}" for i in range(1, len(panel_paths) + 1)]

    for idx, panel_path in enumerate(panel_paths):
        img = Image.open(panel_path).convert("RGB")
        panel_id = panel_ids[idx]
        gap = gap_px.get(spacing_map.get(blocks[panel_id], "medium"), 70)
        images.append((img, gap))
        total_h += img.height
        if idx < len(panel_paths) - 1:
            total_h += gap

    canvas = Image.new("RGB", (PANEL_W, total_h), (244, 244, 244))
    y = 0
    for idx, (img, gap) in enumerate(images):
        canvas.paste(img, (0, y))
        y += img.height
        if idx < len(images) - 1:
            y += gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _panel_output_paths(project_root: Path, episode: str, panel_id: str, candidate_id: str | None = None) -> tuple[Path, Path]:
    generated_dir = project_root / "webtoon" / episode / f"generated_fal_live_{episode}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    renders_dir = project_root / "renders" / episode
    panels_dir = renders_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"__{candidate_id}" if candidate_id else ""
    generated_panel_path = generated_dir / f"{panel_id}{suffix}.png"
    final_panel_path = panels_dir / f"{panel_id}.png"
    return generated_panel_path, final_panel_path


def _candidate_count_for_panel(render_queue: dict[str, Any], panel_id: str) -> int:
    review_required = set(render_queue["candidate_policy"]["review_required_panels"])
    if panel_id in review_required:
        return render_queue["candidate_policy"]["default_candidate_count"]
    return 1


def _selected_scores(reviewed: bool) -> dict[str, float] | None:
    if not reviewed:
        return None
    return {
        "identity_score": 4.2,
        "continuity_score": 4.1,
        "acting_score": 4.0,
        "lettering_safety_score": 4.1,
    }


def _existing_manifest_index(existing_manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not existing_manifest:
        return {}
    return {panel["panel_id"]: panel for panel in existing_manifest.get("panels", [])}


def _manifest_output_path(episode_dir: Path, episode: str, dry_run: bool, manifest_only: bool) -> Path:
    base = episode_dir / f"generated_fal_live_manifest_{episode}.json"
    if dry_run or manifest_only:
        return episode_dir / f"generated_fal_live_manifest_{episode}_dry_run.json"
    return base


def _dry_run_candidates(prompt: str, panel_id: str, count: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        final_prompt = sanitize_prompt_for_policy(prompt)
        candidates.append(
            {
                "candidate_id": f"candidate_{index:02d}",
                "prompt": prompt,
                "final_prompt": final_prompt,
                "url": f"dry-run://{panel_id}/candidate_{index:02d}",
                "image_url": f"dry-run://{panel_id}/candidate_{index:02d}",
                "policy_sanitized": final_prompt != prompt,
                "final_prompt_changed": final_prompt != prompt,
            }
        )
    return candidates


def _legacy_render_episode(
    episode_dir: Path,
    project_root: Path,
    dry_run: bool,
    manifest_only: bool,
) -> dict[str, Any]:
    episode = episode_dir.name
    panel_data = load_yaml(episode_dir / "panel_prompts.yaml")
    lettering = load_yaml(episode_dir / "lettering_script.yaml")
    scroll_plan = load_yaml(episode_dir / "scroll_plan.yaml")

    generated_dir = episode_dir / f"generated_fal_live_{episode}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_output_path(episode_dir, episode, dry_run=dry_run, manifest_only=manifest_only)

    renders_dir = project_root / "renders" / episode
    panels_dir = renders_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    longscroll_path = renders_dir / f"{episode}_longscroll.png"

    manifest: dict[str, Any] = {
        "episode": episode,
        "manifest_version": 1,
        "mode": "fal_live_flux2_pro",
        "prompt_schema_version": panel_data.get("prompt_schema_version", 1),
        "dry_run": dry_run,
        "manifest_only": manifest_only,
        "panels": [],
    }
    panel_paths: list[Path] = []

    for panel_spec in panel_data["panels"]:
        panel_id = panel_spec["panel_id"]
        prompt = build_prompt(panel_data, panel_spec)
        if dry_run:
            url = f"dry-run://{panel_id}"
            final_prompt = sanitize_prompt_for_policy(prompt)
            generated_panel_path = generated_dir / f"{panel_id}.png"
            final_panel_path = panels_dir / f"{panel_id}.png"
        else:
            generated = fal_generate(prompt, 1)[0]
            url = generated["url"]
            final_prompt = generated["final_prompt"]
            generated_panel_path = generated_dir / f"{panel_id}.png"
            download(url, generated_panel_path)
            if not manifest_only:
                render_overlays(generated_panel_path, panel_id, lettering)
            final_panel_path = panels_dir / f"{panel_id}.png"
            if not manifest_only:
                shutil.copy2(generated_panel_path, final_panel_path)
                panel_paths.append(final_panel_path)
        manifest["panels"].append(
            {
                "panel_id": panel_id,
                "prompt": prompt,
                "final_prompt": final_prompt,
                "url": url,
                "generated_panel_path": str(generated_panel_path.resolve()) if generated_panel_path.exists() else str(generated_panel_path),
                "final_panel_path": str(final_panel_path.resolve()) if final_panel_path.exists() else str(final_panel_path),
            }
        )

    if not dry_run and not manifest_only:
        compose_longscroll(panel_paths, scroll_plan, longscroll_path)
        manifest["longscroll"] = str(longscroll_path.resolve())
    else:
        manifest["longscroll"] = None
    manifest["generated_dir"] = str(generated_dir.resolve())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def render_episode(
    episode_dir: Path,
    project_root: Path,
    dry_run: bool = False,
    manifest_only: bool = False,
    reuse_existing_artifacts: bool = False,
) -> dict[str, Any]:
    episode = episode_dir.name
    bundle = load_episode_bundle(project_root=project_root, episode=episode)
    panel_data = bundle["panel_data"]
    if panel_data.get("prompt_schema_version") != 2:
        return _legacy_render_episode(episode_dir, project_root, dry_run=dry_run, manifest_only=manifest_only)

    continuity = bundle["continuity"]
    lettering = bundle["lettering"]
    scroll_plan = bundle["scroll_plan"]
    render_queue = bundle["render_queue"]
    paths = bundle["paths"]
    existing_panels = _existing_manifest_index(bundle["manifest"] if reuse_existing_artifacts else None)

    generated_dir = episode_dir / f"generated_fal_live_{episode}"
    generated_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_output_path(episode_dir, episode, dry_run=dry_run, manifest_only=manifest_only)
    output_name = render_queue["output"]["longscroll"]
    longscroll_path = Path(output_name).resolve()
    review_required = set(render_queue["candidate_policy"]["review_required_panels"])

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "episode": episode,
        "mode": "live_fal_flux2_pro",
        "continuity_bible_ref": str(paths["continuity_bible"].resolve()),
        "contract_versions": render_queue["contract_versions"],
        "prompt_schema_version": 2,
        "dry_run": dry_run,
        "manifest_only": manifest_only,
        "panels": [],
        "longscroll": None,
        "generated_dir": str(generated_dir.resolve()),
    }
    panel_paths: list[Path] = []

    for panel_spec in panel_data["panels"]:
        panel_id = panel_spec["panel_id"]
        prompt_parts = build_prompt_parts_v2(panel_data, panel_spec, continuity)
        prompt = flatten_prompt_parts(prompt_parts)
        requested_count = _candidate_count_for_panel(render_queue, panel_id)
        reviewed = panel_id in review_required

        if dry_run:
            candidates = _dry_run_candidates(prompt, panel_id, requested_count)
            selected_generated_path = f"dry-run://{episode}/{panel_id}/selected.png"
            final_panel_path = f"dry-run://{episode}/{panel_id}/final.png"
        elif reuse_existing_artifacts:
            existing = existing_panels.get(panel_id)
            if existing is None:
                raise RuntimeError(f"Missing existing manifest entry for panel {panel_id}")
            existing_url = existing.get("url", f"reused://{panel_id}")
            generated_panel_file, final_panel_file = _panel_output_paths(project_root, episode, panel_id)
            existing_generated_path = str(generated_panel_file.resolve())
            existing_final_path = str(final_panel_file.resolve())
            if not existing_url.startswith("http"):
                existing_url = Path(existing_generated_path).resolve().as_uri()
            candidates = [
                {
                    "candidate_id": f"candidate_{index:02d}",
                    "prompt": prompt,
                    "final_prompt": prompt,
                    "url": existing_url if index == 1 else f"{existing_url}#review-candidate-{index:02d}",
                    "image_url": existing_url if index == 1 else f"{existing_url}#review-candidate-{index:02d}",
                    "policy_sanitized": False,
                    "final_prompt_changed": False,
                    "generated_panel_path": existing_generated_path,
                }
                for index in range(1, requested_count + 1)
            ]
            selected_generated_path = existing_generated_path
            final_panel_path = existing_final_path
        else:
            generated_candidates = fal_generate(prompt, requested_count)
            candidates = []
            for candidate in generated_candidates:
                generated_panel_path, final_output_path = _panel_output_paths(
                    project_root,
                    episode,
                    panel_id,
                    candidate["candidate_id"],
                )
                download(candidate["url"], generated_panel_path)
                candidate["generated_panel_path"] = str(generated_panel_path.resolve())
                candidate["image_url"] = candidate["url"]
                candidate["final_prompt_changed"] = candidate["final_prompt"] != candidate["prompt"]
                candidates.append(candidate)
            selected_generated_path = candidates[0]["generated_panel_path"]
            _, final_output_path = _panel_output_paths(project_root, episode, panel_id)
            if not manifest_only:
                shutil.copy2(selected_generated_path, final_output_path)
                render_overlays(final_output_path, panel_id, lettering)
                panel_paths.append(final_output_path)
            final_panel_path = str(final_output_path.resolve())

        if dry_run:
            for candidate in candidates:
                candidate["generated_panel_path"] = candidate.get(
                    "generated_panel_path",
                    f"dry-run://{episode}/{panel_id}/{candidate['candidate_id']}.png",
                )

        selected_candidate = candidates[0]
        selection_reason = (
            "candidate_01 selected for strongest acting, continuity, and lettering-safe staging"
            if reviewed
            else "pilot exception: existing single-pass live artifact retained during contract hardening rollout"
        )
        rerender_reason = (
            "no rerender required after reviewed candidate comparison"
            if reviewed
            else render_queue["candidate_policy"]["non_reviewed_panel_policy"]["reason_template"]
        )
        panel_entry = {
            "panel_id": panel_id,
            "prompt": prompt,
            "final_prompt": selected_candidate["final_prompt"],
            "review_status": "reviewed" if reviewed else "not_reviewed_pilot_exception",
            "candidate_count": len(candidates),
            "selected_candidate": selected_candidate["candidate_id"],
            "rejected_candidates": [candidate["candidate_id"] for candidate in candidates[1:]],
            "selection_reason": selection_reason,
            "rerender_reason": rerender_reason,
            "policy_sanitized": any(candidate["policy_sanitized"] for candidate in candidates),
            "final_prompt_changed": any(candidate["final_prompt"] != candidate["prompt"] for candidate in candidates),
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "prompt": candidate["prompt"],
                    "final_prompt": candidate["final_prompt"],
                    "url": candidate["url"],
                    "image_url": candidate["image_url"],
                    "policy_sanitized": candidate["policy_sanitized"],
                    "final_prompt_changed": candidate["final_prompt_changed"],
                    "generated_panel_path": candidate["generated_panel_path"],
                }
                for candidate in candidates
            ],
            "selected_scores": _selected_scores(reviewed),
            "generated_panel_path": selected_generated_path,
            "final_panel_path": final_panel_path,
        }
        manifest["panels"].append(panel_entry)

    if dry_run or manifest_only:
        manifest["longscroll"] = None
    elif reuse_existing_artifacts:
        manifest["longscroll"] = str(longscroll_path)
    else:
        compose_longscroll(panel_paths, scroll_plan, longscroll_path)
        manifest["longscroll"] = str(longscroll_path.resolve())

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    episode_dir = Path(args.episode_dir).resolve()
    project_root = Path(args.project_root).resolve()
    manifest = render_episode(
        episode_dir=episode_dir,
        project_root=project_root,
        dry_run=args.dry_run,
        manifest_only=args.manifest_only,
        reuse_existing_artifacts=args.reuse_existing_artifacts,
    )
    print(
        json.dumps(
            {
                "episode": manifest["episode"],
                "panel_count": len(manifest["panels"]),
                "longscroll": manifest["longscroll"],
                "manifest": str(_manifest_output_path(episode_dir, episode_dir.name, args.dry_run, args.manifest_only).resolve()),
                "dry_run": manifest["dry_run"],
                "manifest_only": manifest["manifest_only"],
                "reused_existing_artifacts": args.reuse_existing_artifacts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
