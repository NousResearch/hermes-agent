# Video Library Duration Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Supply MoneyPrinter with an ordered pool of distinct semantic clips before its existing repeat fallback, so narration longer than the primary matched shots still produces continuous video with the least possible repetition.

**Architecture:** Add a deterministic round-robin pool planner at the named video-library edge. Desktop uses the planner to place one primary clip per script segment first, then supplemental unused clips in later rounds; MoneyPrinter remains a sequential renderer and only cycles after the complete pool is exhausted. Extend the deterministic acceptance runner to exercise the same contract against the real beef-noodle library and a longer local narration.

**Tech Stack:** TypeScript, React hooks, Vitest/jsdom, Python 3.11, pytest, SQLite-backed video library, MoneyPrinterTurbo, FFmpeg/ffprobe.

---

## File map

- Modify `apps/desktop/src/app/video-studio/named-library-matching.ts`: own pure deterministic candidate-pool planning.
- Modify `apps/desktop/src/app/video-studio/named-library-matching.test.ts`: lock primary order, supplemental rounds, and deduplication.
- Modify `apps/desktop/src/app/video-studio/use-named-video-library.ts`: request a bounded expanded candidate set and create a full automatic timeline.
- Modify `apps/desktop/src/app/video-studio/use-named-video-library.test.tsx`: verify the hook sends primary and supplemental rows in order.
- Modify `scripts/video_library_acceptance.py`: create a multi-round semantic pool for local acceptance.
- Modify `tests/capabilities/test_video_library_acceptance.py`: verify acceptance pool order and renderer handoff.
- Do not modify the concurrent browser-Agent files currently dirty in the worktree.

## Task 1: Deterministic round-robin shot pool

**Files:**

- Modify: `apps/desktop/src/app/video-studio/named-library-matching.ts`
- Test: `apps/desktop/src/app/video-studio/named-library-matching.test.ts`

- [ ] **Step 1: Write the failing pool-order test**

Import `planAutomaticClipPool` and add a test with two segments and two candidates per segment:

```ts
it('plans every primary beat before round-robin supplemental shots', () => {
  const s1Primary = { ...clip, id: 's1-primary', asset_id: 'asset-1', score: 1 }
  const s1Extra = { ...clip, id: 's1-extra', asset_id: 'asset-3', score: 0.7 }
  const s2Primary = { ...clip, id: 's2-primary', asset_id: 'asset-2', score: 0.9 }
  const s2Extra = { ...clip, id: 's2-extra', asset_id: 'asset-4', score: 0.6 }
  const segments = [
    { id: 'segment-1', text: '第一段' },
    { id: 'segment-2', text: '第二段' }
  ]

  const pool = planAutomaticClipPool(segments, {
    'segment-1': [s1Extra, s1Primary],
    'segment-2': [s2Extra, s2Primary]
  })

  expect(pool.map(item => [item.round, item.segment.id, item.clip.id])).toEqual([
    [0, 'segment-1', 's1-primary'],
    [0, 'segment-2', 's2-primary'],
    [1, 'segment-1', 's1-extra'],
    [1, 'segment-2', 's2-extra']
  ])
})
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/named-library-matching.test.ts
```

Expected: FAIL because `planAutomaticClipPool` is not exported.

- [ ] **Step 3: Implement the minimal pool planner**

Add this public shape and planner:

```ts
export interface PlannedSegmentClip {
  clip: VideoLibraryClip
  round: number
  segment: ScriptSegment
}

export function planAutomaticClipPool(
  segments: ScriptSegment[],
  candidatesBySegment: Record<string, VideoLibraryClip[]>
): PlannedSegmentClip[] {
  const ranked = Object.fromEntries(
    segments.map(segment => [
      segment.id,
      [...(candidatesBySegment[segment.id] || [])].sort((left, right) => clipRank(right) - clipRank(left))
    ])
  )
  const planned: PlannedSegmentClip[] = []
  const usedAssets = new Set<string>()
  const usedClips = new Set<string>()

  for (let round = 0; ; round += 1) {
    let added = false
    for (const segment of segments) {
      const remaining = ranked[segment.id].filter(candidate => !usedClips.has(candidate.id))
      const selected = remaining.find(candidate => !usedAssets.has(candidate.asset_id)) || remaining[0]
      if (!selected) {continue}
      planned.push({ clip: selected, round, segment })
      usedClips.add(selected.id)
      usedAssets.add(selected.asset_id)
      added = true
    }
    if (!added) {return planned}
  }
}
```

Make `automaticallySelectClips()` return the first planned item for each segment, preserving its existing API while inheriting duplicate-clip protection.

- [ ] **Step 4: Add the deduplication/new-asset-preference test**

Add a test where two segments share the same top clip and a lower-ranked different asset exists. Assert that clip IDs are unique and the lower-ranked new asset is selected before another clip from a used asset.

- [ ] **Step 5: Run planner tests and verify GREEN**

Run the Step 2 command. Expected: all tests in the file pass.

- [ ] **Step 6: Commit Task 1 only**

```bash
git add apps/desktop/src/app/video-studio/named-library-matching.ts \
  apps/desktop/src/app/video-studio/named-library-matching.test.ts
git commit -m "feat(video-library): plan supplemental semantic shots"
```

## Task 2: Use the full pool in automatic Desktop timelines

**Files:**

- Modify: `apps/desktop/src/app/video-studio/use-named-video-library.ts`
- Test: `apps/desktop/src/app/video-studio/use-named-video-library.test.tsx`

- [ ] **Step 1: Write the failing automatic-timeline test**

Extend the existing automatic test with two candidates per segment. Expect `listClips` query calls to use `limit: 12`, and expect `createTimeline` to receive:

```ts
[
  'segment-1-primary',
  'segment-2-primary',
  'segment-1-extra',
  'segment-2-extra'
]
```

with script rows:

```ts
[
  { id: 'segment-1', text: '第一段。' },
  { id: 'segment-2', text: '第二段。' },
  { id: 'segment-1', text: '第一段。' },
  { id: 'segment-2', text: '第二段。' }
]
```

- [ ] **Step 2: Run the hook test and verify RED**

```bash
npm --workspace apps/desktop run test:ui -- src/app/video-studio/use-named-video-library.test.tsx
```

Expected: FAIL because the hook still sends one clip per segment and limit 5.

- [ ] **Step 3: Integrate the planner**

Import `planAutomaticClipPool`, add `const AUTO_MATCH_CANDIDATE_LIMIT = 12`, and in `createAutomaticTimeline()`:

```ts
const pool = planAutomaticClipPool(segments, candidatesBySegment)
const primary = Object.fromEntries(
  pool.filter(item => item.round === 0).map(item => [item.segment.id, item.clip.id])
)
if (Object.keys(primary).length !== segments.length) {
  throw new Error('AI 无法为全部文案匹配镜头')
}
setMatches({ candidatesBySegment, confirmedBySegment: primary, errorsBySegment: {} })

const result = requireData(
  await client.createTimeline(
    selectedLibraryId,
    pool.map(item => item.clip.id),
    aspect,
    pool.map(item => ({ id: item.segment.id, text: item.segment.text }))
  ),
  '素材时间线创建失败'
)
```

Use `AUTO_MATCH_CANDIDATE_LIMIT` for both semantic and unfiltered fallback queries.

- [ ] **Step 4: Run all named-library Desktop tests**

```bash
npm --workspace apps/desktop run test:ui -- \
  src/app/video-studio/named-library-matching.test.ts \
  src/app/video-studio/use-named-video-library.test.tsx \
  src/app/video-studio/unified-material-library-panel.test.tsx
npm --workspace apps/desktop run typecheck
```

Expected: all selected tests and TypeScript typecheck pass.

- [ ] **Step 5: Commit Task 2 only**

```bash
git add apps/desktop/src/app/video-studio/use-named-video-library.ts \
  apps/desktop/src/app/video-studio/use-named-video-library.test.tsx
git commit -m "feat(video-studio): exhaust unique shots before repeat"
```

## Task 3: Expand deterministic acceptance coverage

**Files:**

- Modify: `scripts/video_library_acceptance.py`
- Test: `tests/capabilities/test_video_library_acceptance.py`

- [ ] **Step 1: Write the failing multi-round acceptance test**

Create a fake client returning two ranked candidates for each required intent and assert:

```python
assert [row["query"] for row in plan["selections"]] == [
    "前厅顾客吃面",
    "员工端碗上餐",
    "成品牛肉面特写",
    "前厅顾客吃面",
    "员工端碗上餐",
    "成品牛肉面特写",
]
assert len({row["id"] for row in plan["selections"]}) == 6
```

- [ ] **Step 2: Run the acceptance test and verify RED**

```bash
.venv/bin/pytest tests/capabilities/test_video_library_acceptance.py -q
```

Expected: FAIL because `build_acceptance_plan()` still picks one candidate per intent.

- [ ] **Step 3: Implement round-robin acceptance planning**

Add `_rank()` and `_build_round_robin_selections()` mirroring the production contract: sort each query's candidates, prefer unused assets, never repeat a clip ID, and emit one item per query per round until all returned candidates are exhausted. Change `build_acceptance_plan()` to request `limit=12` and return that pool.

- [ ] **Step 4: Keep renderer provenance assertions dynamic**

Update the render test to compare cached material count and selected-source count with `len(plan["selections"])` instead of the old fixed value `3`.

- [ ] **Step 5: Run focused Python regression**

```bash
.venv/bin/pytest \
  tests/capabilities/test_video_library_acceptance.py \
  tests/capabilities/test_video_library_e2e.py \
  tests/capabilities/test_moneyprinter_adapter.py \
  tests/capabilities/test_moneyprinter_mcp_tools.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit Task 3 only**

```bash
git add scripts/video_library_acceptance.py tests/capabilities/test_video_library_acceptance.py
git commit -m "test(video-library): cover long narration with unique shots"
```

## Task 4: Real long-narration regression

**Runtime writes only; do not commit:**

- `/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/04_素材分析/验收/agent-e2e-result.json`
- `/Users/ruoyu/Documents/Obsidian Vault/01-内容生产/牛肉面资产库/timelines/`
- `external/MoneyPrinterTurbo/storage/local_videos/`
- `external/MoneyPrinterTurbo/storage/custom_audio/`
- `external/MoneyPrinterTurbo/storage/tasks/<task-id>/`

- [ ] **Step 1: Verify current dev ownership and authenticated sidecar**

Run `hermes-dev-desktop status`, inspect port 8080, and call `adapter.health()` under the isolated Hermes Dev `HERMES_HOME`. Do not restart or interrupt the concurrent browser Agent unless the service itself is unavailable.

- [ ] **Step 2: Create a 9-second local silent narration**

```bash
ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i 'anullsrc=r=44100:cl=stereo' -t 9 -q:a 9 \
  /tmp/beef-noodle-duration-coverage.mp3
```

- [ ] **Step 3: Run real automatic planning and rendering**

```bash
HERMES_HOME='/Users/ruoyu/Library/Application Support/Hermes Dev/hermes-home' \
  .venv/bin/python scripts/video_library_acceptance.py \
  --library beef-noodle \
  --render \
  --audio /tmp/beef-noodle-duration-coverage.mp3 \
  --timeout 300
```

Expected: `state=complete`; more than three distinct clip IDs are cached; the first repeated source, if any, occurs only after every supplied unique path.

- [ ] **Step 4: Verify media and visual order**

Use `ffprobe` to confirm H.264 video, AAC audio, 1080x1920 portrait dimensions, and duration at least nine seconds. Extract frames around the primary and supplemental boundaries and inspect them for premature reuse.

- [ ] **Step 5: Run final clean verification**

```bash
git diff --check
git status --short
```

Expected: video-duration commits are clean; the pre-existing browser-Agent changes remain present and untouched.
