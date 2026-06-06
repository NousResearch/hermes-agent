# Mission Control Claude Agent 2 Completion Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement every sensible Mission Control feature implied by https://claude-agent-2.vercel.app/ as a verified, privacy-safe, Apple++ responsive Hermes dashboard.

**Architecture:** Keep `hermes_cli/mission_control.py` as the only server-side aggregation/redaction boundary. Return counts, booleans, known-family labels, safe status enums, and generic placeholders; never return raw session contents, prompts, commands, IDs, paths, private skill names, MCP names, toolset names, or custom provider labels. Keep React as a display layer only, with designed bento layout and responsive behavior.

**Tech Stack:** Python/FastAPI snapshot endpoint, SQLite session DB, Hermes config/env/runtime files, React + TypeScript + Tailwind dashboard, pytest + npm build + browser smoke.

---

### Task 1: Add privacy regression tests for unsafe labels

**Objective:** Prove Mission Control does not leak custom toolset, gateway platform, provider, session DB, cron, MCP, or command labels.

**Files:**
- Modify: `tests/hermes_cli/test_mission_control.py`

**Steps:**
1. Write failing tests with unique canaries in config, env, cron, MCP, and SQLite columns.
2. Run targeted pytest and verify failures.
3. Implement redaction helpers and runtime changes.
4. Re-run targeted pytest until green.

### Task 2: Harden runtime label sanitization and add safer live signals

**Objective:** Return richer Mission Control data without leaking operator/private names.

**Files:**
- Modify: `hermes_cli/mission_control.py`

**Expected changes:**
- Add known-family sanitizers for platform, provider, voice, semantic, terminal/backend labels.
- Collapse custom toolsets to counts/buckets (`builtin`, `mcp`, `plugin`, `custom`, `unknown`) and never serialize raw toolset names.
- Collapse custom gateway platform keys to known families or `other` plus counts.
- Sanitize provider-like config fields to allowlisted families or `custom`/`local`/`other`.
- Read SQLite in read-only mode where possible.
- Add safe aggregation for end reasons, delegation, complex sessions, role counts, handoff categories, rewind totals, and API call stats.
- Expand cron signals with safe status counts, last-run/next-run age buckets, overdue/failed counts, timezone state, and reflection freshness.
- Expand MCP config with redacted placeholders and transport/status counts only.
- Add safe approval/tool-output-limits summary when available.

### Task 3: Update API typings and frontend cockpit layout

**Objective:** Turn the page from an audit wall into an Apple++ command center that works on all devices.

**Files:**
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/pages/MissionControlPage.tsx` or split colocated components if practical.

**Expected changes:**
- Hero shows readiness, production, blockers/watch count, top action, and last generated timestamp.
- Use an explicit bento grid for production, operator queue, runtime pulse, and heatmap.
- Add mobile section nav and disclosure/search/filter for dense source coverage.
- Add larger tap targets (`min-h-11`) for real controls.
- Improve ARIA for section headings, progress bars, decorative icons, and focus order.
- Keep evidence useful but collapse dense evidence on mobile.

### Task 4: Verify production behavior

**Objective:** Prove the dashboard works and does not leak.

**Commands:**
- `python -m pytest tests/hermes_cli/test_mission_control.py -q`
- broader relevant pytest suite
- `npm run build` in `web/`
- Browser smoke on desktop/tablet/mobile:
  - `/mission-control` renders
  - no horizontal overflow
  - console has no errors
  - sensitive canaries absent from DOM/API JSON
  - buttons/links reachable and mobile tap targets acceptable

### Task 5: Independent review, commit, push

**Objective:** Ship verified work.

**Steps:**
1. Dispatch spec/privacy/UI review subagents.
2. Fix blockers.
3. Run final tests/build/smoke.
4. Commit and push to the active PR branch.
5. Leave goal active only if blockers remain; mark complete if verified.
