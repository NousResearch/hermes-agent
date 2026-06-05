# Mission Control Blueprint Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the Claude Agent setup guide into a real Mission Control cockpit for this Hermes runtime: every guide step and feature picker item is mapped to live evidence, an honest readiness state, and a premium responsive UI.

**Architecture:** Add a server-only `hermes_cli.mission_control` intelligence module that compacts local Hermes state into a privacy-minimized snapshot. Expose it through `/api/mission-control/blueprint`, then add a React dashboard route `/mission-control` with Apple-level responsive cards, coverage matrices, and operational panels.

**Tech Stack:** Python/FastAPI backend, SQLite `state.db`, Hermes config/env files, Vite + React + TypeScript frontend, existing Hermes dashboard design system.

---

### Task 1: Add backend snapshot tests first

**Objective:** Lock the required behavior before production code: complete blueprint coverage, sanitized paths/secrets, and real runtime metrics shape.

**Files:**
- Create: `tests/hermes_cli/test_mission_control.py`

**Steps:**
1. Write tests importing `hermes_cli.mission_control`.
2. Assert `build_mission_control_snapshot()` includes all 27 guide steps (including step 22.5), H1-H11, O1-O10.
3. Assert `mission_control_summary()` returns counts/readiness and never leaks raw home paths or secret-looking env values.
4. Run targeted tests and verify RED.

### Task 2: Implement server-only Mission Control intelligence layer

**Objective:** Build deterministic, privacy-minimized snapshot generation from real Hermes files and SQLite state.

**Files:**
- Create: `hermes_cli/mission_control.py`

**Steps:**
1. Add static blueprint source model.
2. Add safe helpers for path compaction, ages, config/env redaction, file counts, SQLite session metrics, cron/skill/MCP/tool/gateway evidence.
3. Derive route coverage and readiness scores.
4. Run tests and verify GREEN.

### Task 3: Expose API endpoint

**Objective:** Serve the snapshot to the dashboard as controlled JSON.

**Files:**
- Modify: `hermes_cli/web_server.py`
- Extend tests: `tests/hermes_cli/test_mission_control.py`

**Steps:**
1. Add `GET /api/mission-control/blueprint` returning snapshot.
2. Add TestClient assertion for 200 response and no raw path/secret leak.
3. Run tests.

### Task 4: Add frontend API types and route

**Objective:** Make Mission Control a first-class dashboard page.

**Files:**
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/App.tsx`
- Create: `web/src/pages/MissionControlPage.tsx`

**Steps:**
1. Add TypeScript interfaces and `api.getMissionControlBlueprint()`.
2. Add route `/mission-control` and sidebar item.
3. Build responsive page sections: command hero, readiness rings, blueprint coverage, live operations, safety/privacy, next actions, troubleshooting.

### Task 5: Polish and verify

**Objective:** Ship a working artifact with build/test/browser evidence.

**Steps:**
1. Run targeted Python tests.
2. Run `npm run build` in `web/`.
3. Start dashboard server locally.
4. Browser smoke `/mission-control` on desktop and mobile widths, check console errors.
5. Commit changes if verification passes.

### Task 6: Phase 2 source-complete Mission Control

**Objective:** Cover the source guide beyond the original 27 steps and feature picker, while making runtime evidence more honest and privacy-safe.

**Backend scope:**
1. Add static source sections for architecture pieces, prerequisites, data-flow surfaces, pre-flight checks, customization checklist, next-tool ideas, glossary, troubleshooting, and resources.
2. Split static support from runtime evidence: env key booleans/counts, SQLite/FTS/cache/actual-cost/API-call metrics, semantic memory config, memory tiers, skill trust/compliance counts, cron cadence/reflection/heartbeat counts, gateway whitelist counts, prompt-injection/approval posture, hosting, quality hooks, production readiness, and action queue.
3. Keep secrets, raw IDs, session titles, tool output, cron prompts, skill names, absolute paths, and vector index names out of the snapshot.

**Frontend scope:**
1. Add production readiness, pre-flight, customization, data-flow, source extras, and device-proof sections.
2. Improve responsive shell with a max-width cockpit, stable `data-testid` anchors, progressbar ARIA, loading/error test IDs, focus-visible links, and React Router internal navigation.
3. Keep dense details as compact cards so mobile/tablet/desktop do not overflow.

**Verification gates:**
1. Targeted Mission Control Python tests must cover new counts and redaction guarantees.
2. `npm run build` must pass.
3. Full Python suite must pass.
4. API smoke must verify JSON shape and no sensitive strings.
5. Browser desktop/mobile smoke must verify no horizontal overflow and no console errors.
