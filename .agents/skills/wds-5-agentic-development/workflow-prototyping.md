---
name: prototyping
description: Build interactive prototypes from specifications
---

# [P] Prototyping — Build Interactive Prototype from Specs

**Goal:** Enable non-technical designers to build production-ready code through structured AI collaboration.

**When to use:** Page specifications are complete and approved. Ready to build working implementations. Want iterative development with approval gates.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### Essential Guides

- **[Feedback Protocol](data/guides/FEEDBACK-PROTOCOL.md)** — Classify feedback before acting
- **[Session Protocol](data/guides/SESSION-PROTOCOL.md)** — Read dialog, verify plan, present status
- **[Execution Principles](data/guides/EXECUTION-PRINCIPLES.md)** — Document-first, sketch fidelity

---

## STEPS

Execute steps in `./steps-p/`:

| Step | File | Purpose |
|------|------|---------|
| 1 | 1-prototype-setup.md | Set up prototype environment |
| 2 | 2-scenario-analysis.md | Analyze scenario, identify views |
| 3 | 3-logical-view-breakdown.md | Break view into sections |
| 4a | 4a-announce-and-gather.md | Announce section, gather context |
| 4b | 4b-create-story-file.md | Create story file |
| 4c | 4c-implement-section.md | Implement the section |
| 4d | 4d-present-for-testing.md | Present for user testing |
| 4e | 4e-handle-issue.md | Handle reported issues |
| 4f | 4f-handle-improvement.md | Handle improvements |
| 4g | 4g-section-approved.md | Section approved, next |
| 5 | 5-finalization.md | Integration test, final approval |

**Flow:** 1 → 2 → 3 → [4a-4g loop per section] → 5

### Critical Rules

- **ALWAYS** complete Phase 1 setup before starting
- **ALWAYS** analyze scenario before selecting views
- **ALWAYS** use section-by-section approach
- **ALWAYS** get approval before next section
- **ALWAYS** create story files just-in-time (not upfront)
- **ALWAYS** verify with Puppeteer before presenting to user
- **ALWAYS** capture baseline before modifying existing features

---

## DESIGN LOG REPORTING POINTS

This workflow has TWO reporting points. Both append to the Design Loop Status table in `{output_folder}/_progress/00-design-log.md`:

**1. On entry (step 1 complete):** Append status `building`
```
| [Scenario slug] | [NN.X] | [Page name] | building | [YYYY-MM-DD] |
```

**2. On completion (step 5 approved):** Append status `built`
```
| [Scenario slug] | [NN.X] | [Page name] | built | [YYYY-MM-DD] |
```

Do NOT skip these updates. The design log drives Phase 4's adaptive dashboard.

## AFTER COMPLETION

Design log already updated with `built` status (see above). Present the transition:

<output>
**"[page name]" is built!**

1. **Run acceptance testing** — validate against the specification
2. **Explore the next scenario step** — [next page name]
</output>
