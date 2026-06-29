---
name: development
description: Write production code from approved specifications
---

# [D] Development — Write Production Code

**Goal:** Write production-quality code from approved specifications using structured agent collaboration.

**When to use:**
- An approved specification exists (page spec, feature spec, or component spec)
- Prototype has been validated (if prototyping was part of the process)
- The codebase and tech stack are established (this is NOT for greenfield project setup)
- You need to turn a spec into committed, tested, production-ready code

**When NOT to use:**
- No approved spec exists yet — use [A] Analysis or spec writing first
- You need to explore or understand an existing codebase — use [R] Reverse Engineering
- You are fixing a bug in existing code — use [F] Bugfixing
- You need a throwaway prototype to validate ideas — use [P] Prototyping

---

## CORE PRINCIPLES

1. **Spec-driven.** The approved specification is the source of truth. Every implementation decision traces back to the spec. If the spec is ambiguous, clarify before coding — do not guess.

2. **Incremental.** Implement one feature or component at a time. Commit after each meaningful unit of work. Never let uncommitted changes grow large.

3. **Test as you go.** Run tests after each significant change. Do not batch all testing to the end. A failing test discovered early is cheap; discovered late it is expensive.

4. **Follow existing patterns.** Match the codebase's conventions for file structure, naming, styling, state management, and error handling. Consistency with the existing code matters more than personal preference.

5. **Document deviations.** If you must deviate from the spec (technical constraint, discovered issue), document what you changed and why before moving on.

---

## REFERENCE MATERIAL

Guides in `./data/guides/` support this workflow:

| Guide | Use When |
|-------|----------|
| EXECUTION-PRINCIPLES.md | Core execution discipline (document before acting, sketch fidelity, plan-then-execute) |
| INLINE-TESTING-GUIDE.md | Self-verifying implementation with Puppeteer before presenting to user |
| SEO-VALIDATION-GUIDE.md | Public-facing pages that need SEO compliance |
| SESSION-PROTOCOL.md | Managing agent sessions and handoffs |
| FEEDBACK-PROTOCOL.md | Handling user feedback during development |

---

## INITIALIZATION

### Design Log
Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

---

## STEPS

Execute steps in `./steps-d/`:

| Step | File | Purpose |
|------|------|---------|
| 01 | step-01-scope-and-plan.md | Read spec, identify work items, create implementation order |
| 02 | step-02-setup-environment.md | Verify environment, install dependencies, establish baseline |
| 03 | step-03-implement.md | Build features one at a time from the plan |
| 04 | step-04-verify.md | Walk through every acceptance criterion |
| 05 | step-05-finalize.md | Clean up, test suite, PR preparation |

---

## DESIGN LOG REPORTING POINTS

Append to the Design Loop Status table in `{output_folder}/_progress/00-design-log.md`:

**On entry (step 01 complete):** `| [Scenario slug] | [NN.X] | [Page name] | building | [YYYY-MM-DD] |`
**On completion (step 05 approved):** `| [Scenario slug] | [NN.X] | [Page name] | built | [YYYY-MM-DD] |`

## AFTER COMPLETION

Design log updated with `built` status (see above). Present the transition:

<output>
**"[page name]" is built!**

1. **Run acceptance testing** — validate against the specification
2. **Explore the next scenario step** — [next page name]
</output>
