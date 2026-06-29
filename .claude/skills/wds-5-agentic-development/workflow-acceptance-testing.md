---
name: acceptance-testing
description: Design and run acceptance tests from specification criteria
---

# [T] Acceptance Testing — Design & Run Tests from Spec Criteria

**Goal:** Validate that implementation matches design specifications through structured testing.

**When to use:** Implementation is complete (prototype or production), ready for validation against specs.

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

---

## STEPS

Execute steps in `./steps-t/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-prepare.md | Gather materials, set up environment |
| 02 | step-02-execute.md | Run all test categories |
| 03 | step-03-document-issues.md | Create issue tickets |
| 04 | step-04-report.md | Compile test report |
| 05 | step-05-iterate.md | Iterate fixes or approve |

**Reference data:**
- `./data/testing-guide.md`
- `./data/test-result-templates.md`
- `./data/issue-templates.md`

---

## DESIGN LOG REPORTING POINT

When all tests pass and the user approves, append to the Design Loop Status table in `{output_folder}/_progress/00-design-log.md`:

```
| [Scenario slug] | [NN.X] | [Page name] | approved | [YYYY-MM-DD] |
```

Do NOT skip this. The design log drives Phase 4's adaptive dashboard.

## AFTER COMPLETION

**If all tests pass:**

Design log updated with `approved` status (see above). Present the transition:

<output>
**"[page name]" is approved!**

1. **Explore the next scenario step** — [next page name]
2. **Design delivery** — package for development handoff (Phase 4 [H])
</output>

**If issues found:**

Status stays `built`. Present:

<output>
**"[page name]" has [N] issues to fix.**

1. **Fix the issues** — route to bugfixing
2. **Explore the next scenario step** — fix later, continue designing
</output>
