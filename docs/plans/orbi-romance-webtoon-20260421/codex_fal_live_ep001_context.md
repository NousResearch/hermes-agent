# Codex context — Orbi romance EP001 fal live render

## Goal
`docs/plans/orbi-romance-webtoon-20260421` 패키지의 EP001을 **fal 실렌더**로 뽑아 실제 longscroll PNG까지 만든다.

## Current state
Already exists:
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001/scroll_plan.yaml`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001/panel_prompts.yaml`
- `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001/lettering_script.yaml`
- storyboard fallback outputs already exist, but user explicitly asked for fal rendering.

Existing reusable references in repo:
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_webtoon_fal_v3.py`
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_balloons.py`
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_layout_utils.py`
- live generated assets under the same lane

## Runtime facts already verified
- `fal_client` import works
- `FAL_KEY` is present via `~/.config/environment.d/fal.conf`
- `fal_client.auth.fetch_auth_credentials()` succeeds
- installed Korean font exists: `Noto Sans CJK KR`

## Constraints
- Work only inside `/home/orbibot/.zeroclaw/workspace/hermes-agent`
- Keep new files under `docs/plans/orbi-romance-webtoon-20260421/`
- Do not touch unrelated lanes
- Deliver a **real fal-generated EP001** package, not storyboard fallback
- Prefer `fal-ai/flux-2-pro` and `fal-ai/flux-2-pro/edit` when continuity chaining helps, but do not overengineer if the current prompt schema is too thin; a coherent first live render is better than a perfect but stalled continuity system
- No readable Korean text should be generated inside images; captions/balloons should be postprocessed with Pillow
- Final deliverables must include a longscroll PNG

## Required outputs
Inside `docs/plans/orbi-romance-webtoon-20260421/webtoon/ep001/` create or update:
- `render_webtoon_fal_live.py` — actual live render script
- `generated_fal_live_ep001/` panel PNGs
- `generated_fal_live_ep001/ep001_fal_live_longscroll.png`
- `generated_fal_live_manifest.json`
- if post-lettering is added separately, include output dir + manifest path clearly

## Preferred implementation shape
1. Reuse ideas from existing live fal renderer rather than starting from scratch
2. Read the romance lane panel prompts and lettering script
3. Generate 8 panels for EP001 via fal
4. Download PNGs locally
5. Add simple Korean captions/balloons in postprocess using Pillow
6. Stitch to one longscroll PNG
7. Save a manifest recording prompts, output URLs, local paths, and longscroll path
8. Verify all output files exist before finishing

## Verification
Run and capture evidence for:
- render script exits 0
- 8 panel PNGs exist
- longscroll PNG exists
- manifest exists
- if possible, visually inspect one representative output path or at least confirm dimensions with PIL

## User-facing requirement
The result must be something Hermes can immediately return with a file path, ideally the final longscroll PNG path.
