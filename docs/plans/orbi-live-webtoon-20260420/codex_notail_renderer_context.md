# Codex context — disable speech-balloon tails in live webtoon renderer

## Goal
Change the live EP001 balloon renderer so speech balloons render **without tails**. The user explicitly wants the renderer changed, not just a one-off artifact edit.

## Source of truth
Repo root: `/home/orbibot/.zeroclaw/workspace/hermes-agent`
Target episode dir: `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001`
Primary renderer: `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_balloons.py`
Related utilities/tests:
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_layout_utils.py`
- `tests/test_balloon_pipeline_ep001.py`

## Current issue
The current renderer still draws tails for speech balloons.
Confirmed in code:
- `render_balloons.py`
- `def should_draw_tail(placement): return bool(placement.tail_points) and placement.template == "speech"`
- `render_shape(...)` draws `draw.polygon(placement.tail_points, ...)` when `should_draw_tail(placement)` is true.

Confirmed in the latest live output manifest:
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live_ballooned/placement_manifest.json`
- `p02 / l01` -> `has_tail_points: true`
- `p06 / l02` -> `has_tail_points: true`
- `p08 / l05` -> `has_tail_points: true`

## Required outcome
1. Update renderer behavior so speech balloons are rendered without tails.
2. Preserve the rest of the balloon/caption pipeline.
3. Update/add regression tests so this behavior is locked in.
4. Keep diff minimal and localized.
5. Do not add dependencies.

## User preference / workflow constraint
The user explicitly prefers code changes via Codex `$ralplan` + `$ralph`, not manual patch-only editing.

## Good implementation shape
Preferred direction:
- Make tail drawing disabled by renderer contract for this lane.
- If useful, keep placement metadata generation intact, but rendering must not draw tails.
- Tests should verify the rendered output for speech balloons no longer depends on tail polygons being drawn.
- If there is an existing tail regression test, update it to reflect the new desired behavior.

## Verification requirements
Run and report at least:
- relevant pytest coverage, especially `tests/test_balloon_pipeline_ep001.py`
- re-render command for the live episode after code change:
  1. `python3 docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/analyze_balloon_zones.py --manifest docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_manifest_live.json --lettering docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/lettering_script.yaml --scroll-plan docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/scroll_plan.yaml --output docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_analysis_ep001.yaml`
  2. `python3 docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_balloons.py --input-dir docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live --analysis docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_analysis_ep001.yaml --lettering docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/lettering_script.yaml --scroll-plan docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/scroll_plan.yaml --output-dir docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live_ballooned --compose-longscroll`
- Confirm the new placement/render output is generated successfully.

## Acceptance criteria
- Speech balloons in the live renderer are tail-less.
- Tests pass.
- The live ballooned longscroll is regenerated successfully.
- Final report identifies changed files and any residual manual-review flags unrelated to tails.
