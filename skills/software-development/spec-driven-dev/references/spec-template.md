# Spec Template (`spec.md`)

The spec describes **WHAT** the feature does — observable behavior and
acceptance criteria. It is **frozen** after sign-off. No code, no file paths,
no implementation tactics.

## Sections

### 1. Title & one-sentence summary
Single sentence describing the feature from the user perspective.

### 2. User stories
Given/When/Then. One paragraph per story. Number them (`US-1`, `US-2`, ...).
Stories are testable by hand without code knowledge.

### 3. Acceptance criteria
A numbered, **checkable** list. Each criterion is one yes/no question a
reviewer can answer in under 30 seconds by reading the diff or running one
command. Format:

```
AC-1: [criterion]
  Verify: [exact command or observation]
```

Bad: "The feature is fast."  
Good: "AC-3: P95 latency on the new endpoint <= 200ms with 100 concurrent
users. Verify: `locust -u 100 -r 10 --headless --run-time 60s` shows p95 <= 200ms."

### 4. Edge cases & error states
What happens when inputs are malformed, concurrent, or adversarial. Each
edge case maps to at least one acceptance criterion.

### 5. Out of scope
Bullet list.

### 6. Open questions
Numbered list of questions that block planning. If empty by spec-freeze, delete
the section.

## Authoring notes

- **No file paths.** File paths are a planning concern.
- **No API shapes.** API shapes belong in `plan.md`.
- **No tests.** Tests belong in `tasks.md` per slice.
- **Each acceptance criterion is binary.** If it cannot be yes/no, split it.
- **Reject "works correctly", "is intuitive", "performs well".** Replace with
  measurable language.

## Anti-patterns

- Acceptance criteria that read like a tutorial — those are user stories.
- Spec that names specific classes or functions — that leaked from planning.
- "TBD" as a placeholder — block spec-freeze until resolved, or move to
  `Open questions` and freeze the rest.
