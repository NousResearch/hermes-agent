# Stage 12-A — Responsive/mobile readability evidence and CSS/SVG posture

Date: 2026-05-09
Branch: `ai-office-stage6-7-cleanup-20260508`

## Goal

Close the next non-renderer slice after Stage 11-C by making the `/office` RPG map explicitly report and style responsive/mobile readability posture without adding a renderer, backend API changes, persistent storage, mutation controls, or raw record projection.

## Constraints

- Keep CSS/SVG as the renderer path.
- Do not add Phaser, PixiJS, canvas renderer, sprite assets, or DeskRPG code/assets.
- Keep `/office` read-only.
- Do not add backend/API/schema changes, mutation controls, persistent browser storage, cron/Kanban/topic/gateway/NAS/Obsidian writes, or individual task identity exposure.
- Derive responsive posture only from browser-local viewport width plus existing safe `OfficeMapDensityPlan`.
- Keep Korean-first UI copy.

## Implementation

Stage 12-A adds a small, testable responsive readability plan:

- `web/src/pages/officeView.ts`
  - Added `OfficeResponsiveReadabilityPlan`.
  - Added `buildOfficeResponsiveReadabilityPlan(densityPlan, { viewportWidth })`.
  - Narrow viewport policy uses `viewportWidth < 640` and recommends `summary` mode, mobile-readable map class, and stacked rail class.
  - Desktop policy preserves the current density mode and desktop rail posture.
- `web/src/pages/OfficePage.tsx`
  - Builds the responsive plan from existing browser-local `viewportWidth` state.
  - Applies smoke hooks:
    - `data-office-responsive="true"`
    - `data-office-responsive-mode`
    - `data-office-responsive-recommended-density`
    - `data-office-responsive-rail="true"`
  - Renders a Korean Stage 12-A note inside the existing safe map legend.
- `web/src/index.css`
  - Adds CSS-only mobile-readable map/rail classes.
  - Keeps horizontal containment and stacked lower-rail posture on narrow screens.
- `web/src/pages/OfficePage.test.ts`
  - Adds RED/GREEN coverage for `buildOfficeResponsiveReadabilityPlan` and raw-term exclusion.

## Decision

No renderer adoption is justified by this slice. Stage 12-A treats narrow-screen readability as a CSS/SVG layout posture issue that can be represented through DOM hooks, Korean notes, and responsive CSS while preserving accessibility and safe DTO boundaries.

## Verification

Timestamp: 2026-05-09 13:21 KST

Results:

```text
cd /Users/lidises/dev/hermes-agent/web
npm test -- --run OfficePage.test.ts
# 28 passed

./node_modules/.bin/eslint src/pages/OfficePage.tsx src/pages/officeView.ts src/pages/OfficePage.test.ts
# passed

npm run build
# passed; JS 1,256.07 kB / gzip 367.41 kB; CSS 127.36 kB / gzip 20.43 kB
# existing Vite >500 kB chunk warning remains

cd /Users/lidises/dev/hermes-agent
source .venv/bin/activate
scripts/run_tests.sh tests/hermes_cli/test_office_redaction.py tests/hermes_cli/test_office_state_adapters.py tests/hermes_cli/test_office_api.py -q --tb=short
# 18 passed in 1.12s

git diff --check
# passed
```

Browser smoke target:

- `http://127.0.0.1:8765/office?stage12a=responsive`

Browser smoke evidence:

- Desktop/standard: responsive hook present, `mode=desktop`, `recommended=standard`, polish hook true, responsive rail present, 4 jump targets, 12 safe character inspect buttons, raw leak regex false.
- Narrow simulated viewport + summary: `mode=narrow`, `recommended=summary`, label mode `minimal`, 6 safe character inspect buttons, recent target `#office-map-recent-collapsed`, collapsed recent rail present, raw leak regex false.
- Browser console JS errors: none.
- Visual smoke after scrolling the `main` container: labels/rail remained readable enough, no severe overlap, and the result looked like CSS responsive compression rather than a renderer failure.

## Next candidates after Stage 12-A

1. Office empty-source copy polish.
2. Small non-mutating PR/handoff summary pass.
3. Only if new measured evidence appears: reopen a renderer spike proposal under the Stage 11 hard gates.
