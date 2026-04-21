# Heuristic Webtoon UI

`heuristic_webtoon_ui.py` renders captions and dialogue overlays onto panel images without hardcoded per-panel bubble coordinates.

What it does:
- Reads `lettering_script.yaml` and panel images from an input directory.
- Places captions and balloons with speaker-aware safe zones such as `top-left`, `top-right`, and `upper-mid`.
- Wraps Korean text by measured width through Pillow font metrics instead of fixed character counts.
- Avoids overlap with previously placed UI boxes and clamps every placement inside the canvas with padding.
- Adds tails for spoken dialogue and skips tails for captions/internal notes.
- Applies soft rounded masks to likely on-image text regions before drawing UI.
- Writes a `placement_manifest.json` debug file describing chosen boxes, masks, zones, and font sizes.

Example:

```bash
python3 docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/heuristic_webtoon_ui.py \
  --input-dir docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/generated_fal_v2 \
  --output-dir docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/generated_fal_ui \
  --lettering docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/lettering_script.yaml \
  --scroll-plan docs/plans/orbi-trend-webnovel-webtoon-20260417/webtoon/ep001/scroll_plan.yaml \
  --compose-longscroll \
  --mode balanced \
  --mask-strength 0.45
```

Useful flags:
- `--input-dir`: source panel folder such as `generated_fal_v2`
- `--output-dir`: target folder for rendered panels
- `--lettering`: YAML dialogue/caption script
- `--scroll-plan`: optional spacing metadata for longscroll assembly
- `--mode`: `compact`, `balanced`, or `airy`
- `--mask-strength`: `0.0` to disable auto masks, higher values to mask more aggressively
- `--font-path`: override the detected Korean font
