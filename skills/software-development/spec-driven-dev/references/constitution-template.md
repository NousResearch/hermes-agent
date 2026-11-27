# Constitution Template

The constitution is **frozen** once accepted. Amendments require an explicit
note in the file (see `## Amendment log` at the bottom).

## Sections

### 1. Purpose
Why this project exists. 2-4 sentences. Should not change often.

### 2. Core invariants
Non-negotiable rules the implementation must always honor. Each line is a
single rule with a one-line rationale.

### 3. Quality gates
Hard checks that block merge: coverage baselines, dependency pinning, lint.

### 4. Out of scope
Things this project explicitly does NOT do. Prevents scope creep at spec time.

### 5. Amendment log
| Date | Section | What changed | Why |
|------|---------|--------------|-----|

## Authoring notes

- Keep it under 60 lines total.
- Each invariant must be verifiable in <30 seconds by a reviewer with `read_file` or `terminal`.
- Do not include implementation specifics (those belong in plan.md).
- If you cannot write a verification step, the invariant is too vague — reword it.
