# Codex context — tail-less balloon annotator update

## Goal
말풍선 어노테이터를 **tail(꼬리) 없는 기본 정책**으로 정리합니다.

사용자 최신 지시:
- "말풍선 꼬리를 아예 없애자"
- "codex를 이용해서 말풍선 어노테이터를 그 방향으로 수정" 
- **Codex + OMX `$ralplan` → `$ralph`** 순서로 진행

## Source-of-truth lane
작업 대상은 아래 live annotator lane입니다.

- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_balloons.py`
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_layout_utils.py`
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/analyze_balloon_zones.py`
- `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_analysis_ep001.yaml`
- input panels: `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live/`
- current output: `docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live_ballooned/`

## Important current state already verified
1. `render_balloons.py` already has:
   - `should_draw_tail()` returning `False`
   - so the final renderer currently does **not** draw the tail polygon.
2. But the pipeline is still semantically tail-based in several places:
   - `PlacementCandidate` still carries `tail_points`, `tail_entry_edge`, `tail_cross_ratio`
   - placement scoring still uses `resolve_tail_route(...)`
   - score breakdown still includes `tail_cross_penalty`
   - manifest output still writes `tail_points`, `tail_entry_edge`, `tail_cross_ratio`
   - `analyze_balloon_zones.py` still emits tail-oriented contract fields like:
     - `tail_policy`
     - `tail_overrides`
     - speaker anchors with `preferred_tail_side`
3. Therefore the live lane is visually no-tail already, but **analysis / scoring / manifest contract** is still built around tails.

## Desired outcome
Make the annotator contract genuinely tail-less, not just visually tail-hidden.

Concretely:
- speech balloon placement should no longer depend on tail routing logic
- attachment quality should be evaluated by **box placement relative to speaker-local zones / anchor distance / placement side / overlap**, not by tail-cross metrics
- manifests/debug output should stop pretending tails exist, or clearly normalize them away
- panel analysis YAML generation should stop centering tail policy as a first-class concept
- keep speaker disambiguation through:
  - zone choice
  - box position
  - speaker-local zones
  - placement-side heuristics
  - optional anchor distance scoring
- do **not** regress text fitting, overlap avoidance, longscroll composition, or mixed chat/caption modes

## Likely implementation direction
Codex should inspect and decide exact diff, but the intended direction is roughly:

### render_balloons.py
- remove or minimize tail-specific route selection in scoring path
- replace `resolve_tail_route(...)` usage with a tail-less attachment evaluation
- keep anchor-based proximity/placement-side scoring if useful
- either:
  - delete `tail_points`, `tail_entry_edge`, `tail_cross_ratio` from `PlacementCandidate` and manifest, or
  - hard-normalize them to `None` / `0.0` only if removing them causes too much churn
- if kept temporarily for backward compatibility, do **not** let them influence ranking except perhaps as dead-compat fields

### analyze_balloon_zones.py
- reduce/remove tail-centric structures from emitted render hints
- preferred output should describe attachment in box-placement terms, not tail-routing terms
- keep speaker anchors if they still help score box proximity, but rename/reframe if needed
- avoid `preferred_tail_side`, `tail_policy`, `tail_overrides` being central unless required for compatibility

### balloon_layout_utils.py
- if any style flags or helpers are tail-only and now unused, remove/simplify them

## Constraints
- Use the existing repo patterns; no new dependencies.
- Keep diffs focused on the live annotator lane above.
- Prefer deletion/simplification over adding another abstraction layer.
- Preserve ability to run the existing EP001 render pipeline end-to-end.
- Do not touch unrelated repo files.

## Verification expectations
At minimum, after implementation:
1. Re-run the live EP001 analysis/render pipeline (or the minimal relevant subset) so outputs/manifests reflect the new tail-less contract.
2. Verify no rendered speech balloons include tails.
3. Verify manifest/debug data no longer contains misleading tail-driven scoring semantics, or that such fields are removed/neutralized intentionally.
4. Verify the pipeline still completes successfully.

## Suggested commands to consider
Inspect first; then run whatever matches the current scripts.

Likely useful commands:
- `python docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/analyze_balloon_zones.py --manifest docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_manifest_live.json --lettering docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/lettering_script.yaml --scroll-plan docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/scroll_plan.yaml --output docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_analysis_ep001.yaml`
- `python docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/render_balloons.py --input-dir docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live --analysis docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/balloon_analysis_ep001.yaml --lettering docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/lettering_script.yaml --scroll-plan docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/scroll_plan.yaml --output-dir docs/plans/orbi-live-webtoon-20260420/webtoon/ep001/generated_fal_live_ballooned --compose-longscroll`

## Deliverables expected from Codex
1. `.omx/context/...` and `.omx/plans/...` artifacts from `$ralplan`
2. implementation via `$ralph`
3. updated source files
4. regenerated analysis/render outputs if needed
5. verification evidence (commands + outcome)

## Notes for Codex
- This is not a request to merely hide the tail drawing. That is already mostly true.
- The real task is to make the **annotator logic and contract** match the no-tail design direction.
- If you find one or two compatibility fields must remain, keep them only as thin compatibility shims and document why.
