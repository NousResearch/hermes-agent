# Hermes Masterclass Roadmap Implementation Plan

> **For Hermes:** Treat this as a documentation + orchestration roadmap. Use the board to track the work, and use GitHub issues as the durable source of truth.

**Goal:** Convert the Hermes masterclass checklist into a clean, ordered roadmap that is tracked in GitHub issues and mirrored on the Kanban board.

**Architecture:**
The roadmap is organized around the checklist themes surfaced in the transcript: identity/memory, skills lifecycle, profile specialization, prompt optimization, Telegram as control plane, and environment hygiene. Existing GitHub issues are reused wherever possible so we do not duplicate active work. The Kanban board then sequences the documentation and linking work so the roadmap is easy to execute later.

**Tech Stack:**
- GitHub issues via `gh`
- Hermes Kanban CLI
- Markdown docs in `docs/plans/`
- Existing open issues in `NousResearch/hermes-agent`

---

## Checklist → GitHub Issue Map

### 1) SOUL.md should evolve with usage
- Existing issue: #11919
- Scope: keep `SOUL.md` lean, but allow behavior changes to be promoted into it when a lesson is stable.

### 2) Memory should be layered and explicit
- Existing issues: #25456, #32726, #8457, #22612, #32064
- Scope: formalize durable/user/session memory boundaries and make retrieval more predictable.

### 3) Skills should behave like a living library
- Existing issues: #27997, #26655, #29017, #20352, #28213, #33314
- Scope: curate stale skills, protect critical ones, and surface lifecycle status/diff information.

### 4) Profiles should specialize by role and route cleanly
- Existing issues: #10143, #9514, #21574, #30652
- Scope: keep programmer/designer/researcher behavior separated and route by channel/topic/task complexity.

### 5) Prompt / budget optimization should happen before weight changes
- Existing issues: #10164, #508, #18092, #11719
- Scope: improve prompts, budgets, and response quality loops before model-weight tuning.

### 6) Telegram should remain the control plane
- Existing issues: #10143, #10452, #21461, #27870
- Scope: keep Telegram topic routing, board mapping, and startup notifications predictable.

### 7) Environment hygiene / duplicate installs / stale paths
- Existing issues: #24186, #30151, #33367, #6447, #25272
- Scope: reduce config drift, eliminate stale hardcoded paths, and keep setup reproducible.

---

## Ordered Execution Plan

### Task 1: Normalize roadmap references

**Objective:** Convert the transcript-derived checklist into a single canonical roadmap doc and issue map.

**Files:**
- Create: `docs/plans/2026-05-27-hermes-masterclass-roadmap.md`
- Update: `ops/_active_task.md`

**Acceptance:**
- All checklist themes are mapped to GH issue references.
- No duplicate issue is created for work that already exists.

### Task 2: Create/update the umbrella GitHub issue

**Objective:** Publish one GH issue that links the checklist themes and points at the existing issues.

**Acceptance:**
- One umbrella issue exists in the repo.
- It includes the transcript-derived checklist and links to the existing issue set.

### Task 3: Create Kanban cards for the roadmap

**Objective:** Mirror the roadmap in the default Kanban board so the work is ordered and visible.

**Acceptance:**
- One parent intake card exists.
- Child cards are grouped by theme and reference the GH issue map.

### Task 4: Verify consistency

**Objective:** Confirm the GH issue map, the markdown plan, and the Kanban board all match.

**Acceptance:**
- No missing checklist item.
- No duplicated task family.
- The board is ready for later execution.

---

## Definition of Done

- [ ] Repo doc saved and committed
- [ ] Umbrella GitHub issue created or updated
- [ ] Kanban tasks created on the default board
- [ ] Existing issues linked rather than duplicated
- [ ] Final summary reports the GH references and next execution step
