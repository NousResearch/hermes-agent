---
name: handover
description: Package complete testable flows and hand off to development
---

# [H] Handover — Package DD-XXX and Hand Off to BMad

**Goal:** Package a complete testable flow into a Design Delivery and hand off to development.

**When to use:** A scenario flow is fully designed, all specifications exist, and you are ready to hand off to BMad for implementation.

---

## INITIALIZATION

Read design log at `{output_folder}/_progress/00-design-log.md` before starting.

---

## STEPS

Execute steps in `./steps-h/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-detect-completion.md | Verify flow is complete and testable |
| 02 | step-02-create-delivery.md | Package into DD-XXX Design Delivery |
| 03 | step-03-create-test-scenario.md | Define validation tests |
| 04 | step-04-handoff-dialog.md | Structured handoff conversation with BMad |
| 05 | step-05-hand-off.md | Official handoff to BMad |
| 06 | step-06-continue.md | Return to design or next flow |

**Reference data:**
- `./data/delivery-templates.md`
- `./data/handoff-dialog-scripts.md`
- `./data/design-deliveries-guide.md`

---

## AFTER COMPLETION

1. Append a progress entry to `{output_folder}/_progress/00-design-log.md` under `## Progress`:
   `### [date] — Design Delivery: [what was packaged]`
2. Suggest next action: Phase 5 prototyping or next scenario
