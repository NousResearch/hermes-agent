# Cross-Repo Plan Consistency Pass — Final
**Date:** 2026-05-18 (updated after agentic-hub discovery)
**Repos covered:** hermes-agent · army-of-one (Atlas) · rooben-pro · **agentic-hub** (previously missed)
**Purpose:** Full inventory + dependency map + authoritative next steps across all four repos.

---

## 1. Corrected Architecture Finding

The previous pass was wrong because `~/Documents/agentic-hub/` was not scanned. That repo is the **integration glue layer** and changes the architecture picture significantly.

### What Actually Exists

| Capability | Where it lives |
|---|---|
| `/refine` adaptive Q&A + `Specification` + `LLMPlanner` | `agentic-hub/rooben-planning/` — a forked, stripped copy of rooben-pro's planning core. Zero HTTP/auth/DB. Used in-process. |
| DAG dispatcher (spec → Kanban tasks) | `agentic-hub/bridges/hermes-rooben/dag-dispatcher.py` — calls `rooben_planning` in-process, writes Kanban tasks, ingests Atlas context briefs per task. |
| Hermes routing to the above | `hermes-agent/gateway/run.py` Plan 005-B — messages starting with `plan:` or `/workflow ` are routed via `_dispatch_to_rooben()` → `dag-dispatcher.py`. |
| Hermes subagent DAG (free-form) | `hermes-agent/tools/delegate_task` — the LLM agent reasons about decomposition in-context and calls `delegate_task(tasks=[...])` directly. No Rooben involved. |
| Kanban dag-dispatcher coordination | `hermes-agent/hermes_cli/kanban_db.py` — SQLite WAL board, YAML frontmatter with `HERMES_NODE_ID` / `HERMES_CHILD_FENCE` / `HERMES_CORRELATION_ID`. |
| Rooben Pro (HTTP SaaS product) | `rooben-pro/` — independent product, still has its own `/api/refine/*` HTTP routes. NOT a running service in the personal stack anymore (removed Plan 005-F). |

### Current Personal Stack Architecture

```
Hermes (hermes-agent, launchd on Mac)
  │
  ├── "plan:" / "/workflow" prefix ──► _dispatch_to_rooben() [Plan 005-B]
  │       └──► agentic-hub/bridges/hermes-rooben/dag-dispatcher.py
  │               ├── import rooben_planning (agentic-hub/rooben-planning/) [in-process, 005-F]
  │               │       ├── RefinementEngine  ← adaptive Q&A → Specification
  │               │       └── LLMPlanner        ← Specification → WorkflowState (DAG)
  │               ├── kanban_dispatch_dag()  ─────────────────────────► Hermes Kanban workers
  │               └── Atlas ingest_research_output() per task  ────────► Atlas (port 8000)
  │
  └── all other messages ──► free-form agent run
          └── delegate_task(tasks=[...]) ──► subagent pool (implicit DAG)
                                             └── Atlas MCP tools ────────► Atlas (port 8000)

Docker compose (make dev) → atlas only (1 container, ~865 MB saved vs original 6)
Rooben Pro → NOT a running service. Planning core vendored in-process via agentic-hub/rooben-planning/.
```

### Two-Track Planning Model

| Trigger | Path | Planning style |
|---|---|---|
| `plan:` / `/workflow ` prefix | Rooben planning in-process → structured DAG → Kanban | Explicit (spec → tasks → execution) |
| All other messages | Hermes LLM agent in-context → delegate_task | Implicit (agent reasons and decomposes) |

---

## 2. Full Plan Inventory — All Four Repos

### 2A. agentic-hub (`~/Documents/agentic-hub/plans/`) — PERSONAL STACK

| Plan | Title | Status |
|------|-------|--------|
| 001 | Personal Agentic Stack | **COMPLETE** 2026-05-13 — 7 phases; manual ACs (AC1 4-week, AC6 monthly cost, outbound soak) still running on calendar |
| 002 | Multi-Agentic Coherence on the Hermes Backbone | **COMPLETE** 2026-05-13 — 5 phases; 411+ tests; W3C traceparent propagates end-to-end |
| 003 | Live Stack UAT + Daily-Driver Polish + Code-Gen MVP | **COMPLETE** 2026-05-14 — 10 live-stack bugs fixed; Scenario B clean |
| 004 | Rooben Decoupling — Service-Token Auth + Portkey | **COMPLETE** 2026-05-14 — auth bypass + Portkey direct AWS; Scenario A passes |
| 005 | Close Workaround Gaps + Architecture Trim + Stack Slimdown | **COMPLETE** 2026-05-15 — 6→1 service; ~865 MB saved; Rooben in-process (005-F) |
| 006 | Close Plan 005 Deferrals — Persistence, Atlas, Subagent Depth | **COMPLETE** 2026-05-15 — rooben-pro PR #22 merged; Atlas fence column live |
| 007 | Close Plan 006 Deferrals — Atlas LLM, Gmail, Portkey UAT | **COMPLETE** 2026-05-15 — Atlas Portkey wired; Gmail MCP registered; Scenario A 0 FAIL |
| 008 | Atlas-as-Sole-Memory — UAT + Fix Silent Ingest Break | **COMPLETE** 2026-05-16 — session ingest fixed; `hermes sessions finalize` CLI shipped; shadow-mode observability live |

**Open follow-ups from agentic-hub completion (not blocking anything):**
- `hermes slack inject` CLI doesn't exist — blocks full UAT automation for 008-D green-pass
- Plan 007 follow-up #3: `llm_extraction` still degrades; tracked in agentic-hub/bridges
- `SYNC_DRIFT.md` baseline exists in `rooben-planning/` — periodic drift review recommended
- Manual ACs from Plan 001 running on calendar: log Hermes message-event count weekly; check monthly AWS cost; confirm outbound soak in 2 weeks

---

### 2B. Hermes Agent (`~/Documents/hermes-agent/plans/`) — SAAS PRODUCTIZATION

| Plan | Title | Status | Format |
|------|-------|--------|--------|
| 001 | Multi-User SaaS — Identity, Scoped Skills & Memory, Cloud Storage | DRAFT | ⚠️ Pre-canonical: `saas-multi-user/` not `001-saas-multi-user/phases/` |
| 002 | Hermes Self-Organization — Directory Layout, MCP Pool, Runtime Isolation | DRAFT | ✅ Canonical |
| 003 | Skills Service — Scoped Registry, MCP Surface, Git-Backed | **APPROVED** | ✅ Canonical |
| 004 | Self-Improvement Service (Hermes↔Skills↔Atlas feedback loop) | STUB | ✅ Placeholder only |

**Orphaned file:** `plans/gemini-oauth-provider.md` — standalone proposal, no number, not tracked. Recommend: assign Plan 005-hermes or archive.

---

### 2C. Atlas (`~/Documents/army-of-one/plans/`) — MEMORY SUBSTRATE EVOLUTION

| Plan | Title | Status | Blocked By |
|------|-------|--------|------------|
| 001–008 | Foundation, second brain, intelligence, quality, ontology | **COMPLETE** (archived) | — |
| 005 | Atlas Personal AWS Deployment | APPROVED, NOT STARTED | None — await bulk ingest completion |
| 009 | Ontology Expressiveness & Content-Type Coverage | DRAFT | None (starts after Atlas 014 for sequencing) |
| 010 | Memory Self-Organization (Dreams-equivalent) | DRAFT | Atlas 009-A |
| 011 | Life-Comprehensive World Ontology | DRAFT | Atlas 009-0, 010-0/B |
| 012 | Hermes → Atlas Memory Backend Connector | APPROVED DRAFT | Atlas 011-A, 009-0, 010-B |
| 013 | Generalize Atlas Ontology Beyond VC | **DOC-REFORM COMPLETE** | — (no standalone dir; applied inline to 011 + 012) |
| 014 | Atlas Ontology Fitness, Evolution & Self-Improvement | **IN PROGRESS** (A–F done; G–I remain) | None (parallel) |

**Format issues in Atlas:**
- Plans 010, 011, 012 have no `phases/` subdirectory — only a master doc + STATUS.md. Needs `phases/` + phase files generated before the execution loop runs.
- Plan 013 has no directory (correct — doc-reform only, confirmed complete).

---

### 2D. Rooben Pro (`~/Documents/rooben-pro/docs/plans/`) — SAAS PRODUCT ROADMAP

| Plan | Title | Status |
|------|-------|--------|
| 008–020 | Infra, templates, UX, journeys, onboarding | **COMPLETE** (archived) |
| 021 | Multi-Modal Ingest Pipeline | Not started — no blockers |
| 022 | Spec / DAG / Execution Studio | Not started — no blockers |
| 005 | Unified Adaptive Builder | **⚠️ ORPHANED** — superseded by v0.4.0 extension DB migration |
| 006 | Workflows-as-Agents + Hierarchical Composition | **⚠️ ORPHANED** — partially shipped in v0.5.0; residual scope unclear |
| 023 / `2026-04-29-mcp-gateway-sse-bridge.md` | MCP Gateway SSE Bridge | **⚠️ ORPHANED** — concrete plan, untracked |

**agentic-hub/rooben-planning/ drift note:** This is a frozen fork of rooben-pro at SHA `82bdb1d`. `SYNC_DRIFT.md` tracks divergence. As rooben-pro's planning core evolves (Plans 021/022), periodically review whether the fork needs a sync.

---

## 3. Cross-Repo Dependency Map

```
══════════════════════════════════════════════════════════════════════
PERSONAL STACK (agentic-hub) — ALL COMPLETE
══════════════════════════════════════════════════════════════════════

001 → 002 → 003 → 004 → 005 → 006 → 007 → 008 ✅ DONE
                                                   │
                              Follow-ups (non-blocking) only


══════════════════════════════════════════════════════════════════════
ATLAS MEMORY EVOLUTION (army-of-one) — ACTIVE CRITICAL PATH
══════════════════════════════════════════════════════════════════════

014 (IN PROGRESS: A-F done)
  └──► 014-G → 014-H → 014-I

Atlas 005 (AWS deployment) — independent, unblocked

009-0 ──► 009-A ──► 009-B ──► 009-C ──► 009-D
  │           │
  │           └───────────────────────────────────────┐
  ▼                                                   │
010-0 ──► 010-A ──► 010-B ──► 010-C                 │
                       │                              │
                       └──────────────────────────────┼──►  011 (all phases A–M)
                                                      │          │
                                              009-0 ──┘          └──► 012 (all phases 0–G)
                                                                         │
                                                         (Atlas 012 = Hermes↔Atlas connector)


══════════════════════════════════════════════════════════════════════
HERMES SAAS PRODUCTIZATION (hermes-agent) — FUTURE WORK
══════════════════════════════════════════════════════════════════════

001-0 (HermesIdentity dataclass) ─────────────────────────────────────┐
  │                                                                     │
  ├──► 001-A → 001-B → 001-C → 001-D → 001-E                         │
  ├──► 002-B → 002-C → 002-D                                          │
  └──► 003-A → 003-B → 003-C → 003-D                                  │
                                    │                                   │
       003-E (parallelizable after 003-A config) ──────────────────────┘
       003-F (after 003-C + 001-D)

[002-A and 003-E are unblocked today — can start before 001-0]


══════════════════════════════════════════════════════════════════════
CROSS-REPO BRIDGE
══════════════════════════════════════════════════════════════════════

Atlas 012 (Hermes↔Atlas connector) <──── depends on Atlas 011-A + 009-0 + 010-B
Hermes 004 (self-improvement) ──────────────────────────────────────────────────────►  needs Hermes 003 + Atlas 012


══════════════════════════════════════════════════════════════════════
ROOBEN PRO (INDEPENDENT SAAS PRODUCT)
══════════════════════════════════════════════════════════════════════

021 (Multi-Modal Ingest) — no blockers — independent
022 (Spec/DAG Studio) ────── no blockers — independent
```

---

## 4. Consistency Issues

### ✅ Issue 1 — agentic-hub Repo Was Missing From First Pass
**Corrected.** All Plans 001–008 are COMPLETE. The repo is the personal stack integration layer.

### ⚠️ Issue 2 — Hermes Plan 001 Format Mismatch
`plans/saas-multi-user/` instead of `plans/001-saas-multi-user/phases/`. Phase files are at the directory root, not in `phases/`. Will break any build:execute-plan looping skill.

**Fix:**
```bash
cd ~/Documents/hermes-agent/plans/
mv saas-multi-user/ 001-saas-multi-user/
mkdir 001-saas-multi-user/phases/
mv 001-saas-multi-user/phase-*.md 001-saas-multi-user/phases/
mv 001-saas-multi-user/MASTER-PLAN.md 001-saas-multi-user/001-saas-multi-user.md
# Update PROGRESS.md path reference
```

### ⚠️ Issue 3 — Atlas Plans 010, 011, 012 Missing `phases/` Directories
These plans have only a master doc + STATUS.md. No `phases/` directory. Low urgency until execution begins.

**Fix:**
```bash
cd ~/Documents/army-of-one/plans/
for p in 010-memory-self-organization 011-world-ontology 012-hermes-atlas-memory-connector; do
  mkdir -p "$p/phases" && touch "$p/phases/.gitkeep"
done
```

### ⚠️ Issue 4 — Rooben Pro Plans 005 + 006 Orphaned
Not in PROGRESS.md, not archived.
- **Plan 005** (Unified Adaptive Builder): Superseded by v0.4.0 extension DB migration. **Archive.**
- **Plan 006** (Workflows-as-Agents): Partially shipped in v0.5.0. Needs Blake's review — archive or extract residual scope.

### ⚠️ Issue 5 — Rooben Pro MCP Gateway Plan Untracked
`docs/plans/2026-04-29-mcp-gateway-sse-bridge.md` is a concrete implementation plan with no number, not in PROGRESS.md. **Assign as Plan 023 or archive.**

### ⚠️ Issue 6 — agentic-hub/rooben-planning Fork Drift
`rooben-planning/` is a frozen fork of rooben-pro at SHA `82bdb1d`. `SYNC_DRIFT.md` tracks this. As rooben-pro evolves (Plans 021/022), the fork may diverge. **Recommend:** Review `SYNC_DRIFT.md` after each rooben-pro plan completion and decide whether to sync.

### ✅ Issue 7 — Port Assignments Consistent
Atlas=8000, Skills Service=8001, Self-Improvement TBD=8002. No conflicts across all repos.

### ✅ Issue 8 — HermesIdentity Is Absolute Hermes Critical Path (confirmed)
Every Hermes SaaS plan except 002-A and 003-E is blocked on Hermes 001-0. Only those two can start immediately.

### ✅ Issue 9 — Atlas 009 Is Absolute Atlas Critical Path (confirmed)
Atlas 009-0 and 009-A gate Plans 010, 011, and 012. Atlas 005 (AWS) and Atlas 014 are independent.

### ✅ Issue 10 — Atlas 014-F Output Timing (safe to proceed)
014-F (evolution algorithms) feeds the Plan 011-H proposal queue, but Plan 011-H is the last phase of Plan 011. The 014-F output will queue proposals that 011-H eventually consumes. No blocking issue. Continue 014-F as planned.

### ⚠️ Issue 11 — `hermes slack inject` CLI Missing (blocks UAT automation)
Surfaced in agentic-hub Plan 008-D. Full UAT automation for Atlas memory recall requires `hermes slack inject` to push test messages into Slack. This CLI command doesn't exist in hermes-agent. Green-pass UAT requires Blake's 2 manual Slack DMs until it's built. **Recommend:** Add as a sub-task under Hermes 002 (Self-Organization) or as a standalone small fix.

---

## 5. Recommended Execution Order

### Tier 0 — Unblocked Today

| Action | Repo | Why |
|--------|------|-----|
| **Atlas 014-G → 014-H → 014-I** | army-of-one | IN PROGRESS — continue to completion. Next: OntoClean linter (014-G). |
| **Hermes 001-0** (HermesIdentity dataclass) | hermes-agent | Zero blockers — first thing to unblock all Hermes SaaS plans. ~1 day. |
| **Hermes 002-A** (directory layout migration) | hermes-agent | Zero blockers — safe to start alongside 001-0. |
| **Rooben Pro 021** (Multi-Modal Ingest) | rooben-pro | No blockers — independent product. |
| **Rooben Pro 022** (Spec/DAG Studio) | rooben-pro | No blockers — independent product. |
| **Atlas 005** (Personal AWS Deployment) | army-of-one | APPROVED, unblocked — start after bulk ingest completes. |
| Format fixes (Issues 2 + 3 above) | hermes-agent, army-of-one | 10-minute housekeeping; unblocks execution loops. |
| Rooben 005/006 archive decision | rooben-pro | Clear the orphaned plan clutter. |

### Tier 1 — After Atlas 014 Completes

| Action | Repo | Why |
|--------|------|-----|
| **Atlas 009** (009-0 → 009-A → 009-B → 009-C → 009-D) | army-of-one | Unlocks all Atlas memory evolution plans (010, 011, 012). |

### Tier 2 — After Hermes 001-0 and Atlas 009-0

| Action | Repo | Why |
|--------|------|-----|
| **Atlas 010** (010-0 → 010-A → 010-B) | army-of-one | Unlocks Atlas 011 + 012. |
| **Hermes 001** (A → B → C → D → E) | hermes-agent | HermesIdentity unlocked. |
| **Hermes 002** (B → C → D) | hermes-agent | 001-0 complete. |
| **Hermes 003** (A → B → C → D; 003-E parallel after A) | hermes-agent | 001-0 complete. |

### Tier 3 — After Atlas 009 + 010

| Action | Repo | Why |
|--------|------|-----|
| **Atlas 011** (all phases A through M) | army-of-one | Needs 009-0, 010-0/B. |
| **Atlas 012** (0 → A → B → C → D → E → F → G) | army-of-one | Needs 011-A, 009-0, 010-B. |

### Tier 4 — After Hermes 003 + Atlas 012

| Action | Repo | Why |
|--------|------|-----|
| **Hermes 004** (Self-Improvement Service) | hermes-agent | Needs Skills Service + Hermes↔Atlas connector both live. |

---

## 6. Summary Table — All Active Plans

| Plan | Repo | Title | Status | Tier | Blocked By |
|------|------|-------|--------|------|------------|
| agentic-hub 001–008 | agentic-hub | Personal Agentic Stack (all) | ✅ COMPLETE | — | — |
| Atlas 014 (G–I) | army-of-one | Ontology Fitness & Evolution | IN PROGRESS | 0 | None |
| Atlas 005 | army-of-one | Personal AWS Deployment | APPROVED | 0 | Bulk ingest completion |
| Rooben 021 | rooben-pro | Multi-Modal Ingest Pipeline | Not started | 0 | None |
| Rooben 022 | rooben-pro | Spec / DAG / Execution Studio | Not started | 0 | None |
| Hermes 001-0 | hermes-agent | HermesIdentity Dataclass | Not started | 0 | **NOTHING — start now** |
| Hermes 002-A | hermes-agent | Directory Layout Migration | Not started | 0 | None |
| Atlas 009 | army-of-one | Ontology Expressiveness | DRAFT | 1 | Atlas 014 completion (sequencing) |
| Atlas 010 | army-of-one | Memory Self-Organization | DRAFT | 2 | Atlas 009-A |
| Hermes 001 (A–E) | hermes-agent | Multi-User SaaS | DRAFT | 2 | Hermes 001-0 |
| Hermes 002 (B–D) | hermes-agent | Self-Organization (MCP, Cloud) | DRAFT | 2 | Hermes 001-0 |
| Hermes 003 (A–F) | hermes-agent | Skills Service | APPROVED | 2 | Hermes 001-0 |
| Atlas 011 | army-of-one | Life-Comprehensive World Ontology | DRAFT | 3 | Atlas 009-0, 010-0/B |
| Atlas 012 | army-of-one | Hermes → Atlas Memory Connector | APPROVED DRAFT | 3 | Atlas 011-A, 009-0, 010-B |
| Hermes 004 | hermes-agent | Self-Improvement Service | STUB | 4 | Hermes 003 + Atlas 012 |

---

## 7. Open Questions — RESOLVED 2026-05-18

| # | Question | Resolution |
|---|----------|------------|
| 1 | Rooben 005 + 006 | **ARCHIVED.** Rooben Pro is out of scope for agentic-hub, hermes, and atlas going forward. Plans 005/006/023 marked archived in rooben-pro PROGRESS.md. |
| 2 | Rooben 023 MCP Gateway | **ARCHIVED.** Same scope decision as item 1. |
| 3 | `hermes slack inject` CLI | **Not needed.** It's a testing harness for frontend messaging validation. Slack is an integration, not a critical system. No action required. |
| 4 | rooben-planning fork sync cadence | **Moot.** Rooben out of scope (see item 1). `agentic-hub/rooben-planning/` remains as-is — no future syncs needed unless the personal stack planning pipeline breaks. |
| 5 | Atlas 005 AWS deployment status | **Blocked — awaiting Blake's decisions.** Bulk Claude CC ingest ran 2026-04-22 (1,353 sessions, 633 MB DB). Drive ingest has NOT been done. Open questions before 005-A can start: (a) AWS account + IAM ready? (b) Region: us-east-1 or us-west-2? (c) Tailscale auth key, or skip in favor of ALB + IP allow-list? (d) Terraform state bucket name (`s3://atlas-tfstate-blake` proposed). Drive ingest should happen first. |
| 6 | Hermes 001-0 timing + execution order | **Resolution: local-first, SaaS last.** Full dependency evaluation below. Everything improves locally before SaaS deployment. |

---

## 8. Full Dependency-Resolved Execution Order (Local-First)

> **Principle:** Improve locally → validate → SaaS deployment is Tier 4 (last).
> Everything in Tiers 0–3 runs on your Mac. No AWS required until Tier 4.

### Tier 0 — Unblocked Today (parallel)

| # | Action | Repo | Notes |
|---|--------|------|-------|
| T0-A | **Atlas 014-G → 014-H → 014-I** | army-of-one | Continue IN PROGRESS plan to completion. OntoClean linter next. |
| T0-B | **Hermes 001-0** — HermesIdentity dataclass | hermes-agent | Zero blockers. ~1 day. Single highest-leverage Hermes action — unlocks T2 entirely. |
| T0-C | **Hermes 002-A** — Directory layout migration | hermes-agent | Zero blockers. Run alongside T0-B. |
| T0-D | **Drive bulk ingest** (manual, Blake) | army-of-one | Run `second-brain-ingest.sh` with Drive source. Prerequisite for Atlas 005. |

### Tier 1 — After Atlas 014 Complete

| # | Action | Repo | Notes |
|---|--------|------|-------|
| T1-A | **Atlas 009** — 009-0 → 009-A → 009-B → 009-C → 009-D | army-of-one | Unlocks all Atlas memory evolution plans. |

### Tier 2 — After T0-B (Hermes 001-0) + T1-A (Atlas 009-0)

These all unblock simultaneously and can run in parallel streams:

| # | Action | Repo | Blocked by |
|---|--------|------|------------|
| T2-A | **Atlas 010** — 010-0 → 010-A → 010-B → 010-C | army-of-one | Atlas 009-A |
| T2-B | **Hermes 001** — A → B → C → D → E | hermes-agent | Hermes 001-0 |
| T2-C | **Hermes 002** — B → C → D | hermes-agent | Hermes 001-0 |
| T2-D | **Hermes 003** — A → B → C → D (003-E parallel after A) | hermes-agent | Hermes 001-0 |
| T2-E | **Atlas 005** (AWS deployment) | army-of-one | Drive ingest done + Blake's 3 config decisions |

### Tier 3 — After Atlas 009 + 010

| # | Action | Repo | Blocked by |
|---|--------|------|------------|
| T3-A | **Atlas 011** — all phases A through M | army-of-one | Atlas 009-0, 010-0/B |
| T3-B | **Atlas 012** — Hermes↔Atlas Memory Connector | army-of-one | Atlas 011-A, 009-0, 010-B |

### Tier 4 — SaaS Deployment (After Hermes 003 + Atlas 012)

| # | Action | Repo | Notes |
|---|--------|------|-------|
| T4-A | **Hermes 004** — Self-Improvement Service | hermes-agent | Needs Skills Service (003) + Atlas connector (012) both live locally first |
| T4-B | **Cloud deployment** — Neon, S3, ECS/Fargate | hermes-agent | Only after local multi-user works end-to-end |

---

## 9. Self-Improvement Pipeline — LIVE

The `#product-feedback` channel (Slack ID: `C0B4EHQFHS5`) is now wired into two cron jobs:

| Job | Schedule | What it does |
|-----|----------|--------------|
| `feedback-triage-nightly` (ID: `9ca1ab28753d`) | Nightly 10pm | Reads new messages → triages → annotates existing plan/phase files → DMs summary to Blake |
| `feedback-to-plans-weekly` (ID: `b755658d3c55`) | Monday 9am | Full weekly batch → groups themes → generates new plan/phase files → posts structured summary with priority recommendation |

**Skill:** `devops/slack-feedback-to-plans` — defines the full pipeline, triage taxonomy, routing rules, and monthly review cadence.

**Future expansion** (once Hermes 003 + Atlas 012 are live): the monthly cadence can trigger a skill promotion review — patterns that appear repeatedly in feedback become candidates for new skills or skill updates, creating a full self-improvement loop: `feedback → plan → code → skill → Atlas memory → improved behavior`.


1. **Rooben 005 + 006**: Archive both? Or extract residual scope from 006 into a 023?
2. **Rooben 023** (MCP Gateway SSE Bridge): Was it implemented in agentic-hub's linear-bridge / MCP work? Or still needed?
3. **`hermes slack inject`**: Build as part of Hermes 002, or standalone fix now?
4. **agentic-hub/rooben-planning/ sync cadence**: When Rooben 021/022 ship changes to the planning core, how often do we sync the fork? On every rooben-pro release, or as-needed?
5. **Atlas 005 trigger**: Has the bulk Claude + Drive ingest completed? If yes, Atlas 005 (AWS deployment) can begin immediately.
6. **Hermes 001-0 timing**: This is the single highest-leverage Hermes action. Ready to execute now?
