# [Project Name] — Overnight Navigation Map

**Purpose:** Entry doc for long-running autonomous sessions. Read this, then fan out into the mind palace based on what's most actionable.

---

## Quick Trunk Reference

| What | Where |
|------|-------|
| Master plan | `.hermes/mind-palace/plans/master_impl_plan_v2.md` |
| Phase detail | `.hermes/mind-palace/tier3-impl/10-training-loop/README.md` |
| Architecture | `.hermes/presentation/3-architecture.md` |
| Per-phase status | `.hermes/presentation/4-implementation-status.md` |
| Current state log | `.hermes/session-end-YYYY-MM-DD.md` |
| Fresh start prompt | `.hermes/mind-palace/fresh_start_prompt.md` |

## Where We Are (YYYY-MM-DD)

**Phases [N-M] complete. Phase [K] active.**

**Done — VERIFIED:**
- [Accomplishment 1 — verified by running binary]
- [Accomplishment 2 — verified by checking file state]

**CLAIMS VERIFIED — real runtime output from actual binaries:**

| Stream | Result | Status |
|--------|--------|--------|
| A — [name] | [actual numbers from running binary] | ✅ PASS |
| B — [name] | [actual numbers or state] | ✅ DONE |
| C — [name] | [blocked state] | ❌ BLOCKED |

**ROOT BLOCKER:**
[One paragraph explaining what blocks everything else]

## Workstreams (pick one per session)

### P0 — [Root Blocker Fix]
[Description of the single highest-impact task. Include file paths.]

### A — [Verified Stream — no more work needed]
[One line summary. Don't repeat work.]

### B — [Next Stream — BLOCKED on P0]
[Description of what's waiting.]

### C — [Independent Stream]
[Description of something that doesn't depend on P0.]

## Data You Should Not Re-Derive

- [Fact 1 — e.g., R=0.956 for Poincaré ball]
- [Fact 2 — e.g., SSM recurrence formula]
- [Fact 3 — e.g., tensor layout quirk from target map]
- [Hardware fact if relevant]
- [Path fact — e.g., where models live]

## Fallback

If all streams look blocked, pick the one with the clearest next step and make a `plan.md` in `.hermes/plans/`. Better to document a plan than sit idle.
