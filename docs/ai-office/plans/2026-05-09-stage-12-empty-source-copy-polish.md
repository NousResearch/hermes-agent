# Stage 12-B — Empty-source copy polish

Updated: 2026-05-09 13:44 KST

## Scope

Stage 12-B is a small non-renderer dashboard/product polish slice after Stage 12-A. It makes the source-state empty condition easier to read without changing backend/API/schema behavior.

Constraints kept:

- `/office` stays read-only.
- No backend/API/schema changes.
- No mutation controls.
- No cron/Kanban/topic/gateway/NAS/Obsidian writes.
- No persistent browser storage.
- No Phaser, PixiJS, canvas renderer, sprite assets, or DeskRPG code/assets.
- No raw prompt, transcript, task body, cron script, log, auth, secret, token, model/provider identity, or individual task identity projection.

## Implementation

- `web/src/pages/officeView.ts`
  - Added `OfficeEmptySourceCopyPlan` / `OfficeEmptySourceCopyItem`.
  - Added `buildOfficeEmptySourceCopyPlan(state)`.
  - The helper summarizes missing expected source ids and the read-only/safe DTO boundary from source-health counts only.
- `web/src/pages/OfficePage.tsx`
  - Computes `emptySourceCopy` from current safe `OfficeState`.
  - When `state.data_sources` is empty, renders a Korean explanatory panel with:
    - `data-office-empty-source-copy="true"`
    - per-item `data-office-empty-source-item`
  - Existing source cards are unchanged when source records exist.
- `web/src/pages/OfficePage.test.ts`
  - Added a Stage 12-B RED/GREEN helper test covering Korean copy, missing-source count, read-only wording, and raw-term exclusion.

## Safety notes

The Stage 12-B helper only uses `buildOfficeSourceHealthSummary(state)`, `state.data_sources.length`, and the expected source-id gap. It does not inspect adapter body fields, raw content, credentials, scripts, logs, or task identities. The UI panel is informational only and offers no run/edit/connect controls.

## Verification

Verification 2026-05-09 13:44 KST:

- RED verified first: Stage 12-B test failed because `buildOfficeEmptySourceCopyPlan` did not exist.
- GREEN focused frontend test passed: `OfficePage.test.ts` 29 passed.
- ESLint passed for `OfficePage.tsx`, `officeView.ts`, and `OfficePage.test.ts`.
- `npm run build` passed with the existing Vite large-chunk warning; current build size was JS `1,257.73 kB` / gzip `367.75 kB`, CSS `127.84 kB` / gzip `20.50 kB`.
- Backend focused office tests passed: `18 passed in 1.00s`.
- Browser smoke `/office?stage12b=empty-source-copy`: live source fixture had 5 reported source cards, so `data-office-empty-source-copy` was correctly absent; source names `kanban`, `cron`, `sessions`, `topics`, and `provenance` were visible, raw leak regex false, console JS errors none.
- Empty-source copy rendering is covered by the focused helper test because the live API fixture is currently non-empty.
