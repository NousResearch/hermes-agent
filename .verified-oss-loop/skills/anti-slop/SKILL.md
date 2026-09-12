# Anti-slop

The smallest complete change that matches this tree. Style of the files you touch wins over a generic “clean code” essay.

## Patterns

- One issue, one PR, one concern.
- Match naming, imports, error handling, and test style already in the file.
- Fail, then pass (`skills/tdd/SKILL.md`). Assert behavior, not the implementation’s private shape.
- Names from the domain, not `handleData2`.
- Impact, then edit (`skills/orient/SKILL.md`). Check sibling surfaces that read the same state.
- Comments explain why or a constraint. Delete comments that restate the next line.
- Delete dead code you added. Do not leave a helper used once “for later”.

## Anti-patterns

- Narrating comments (`// increment the counter`), emoji, or “As an AI…”.
- Speculative abstractions: a base class, factory, or helper for a single call site.
- Drive-by refactors, format-only churn, or rewriting files you did not need.
- Tests that only echo the implementation, compare duplicated fixtures, or never fail for the bug.
- Catch-all `except` / `type: ignore` / empty `catch` to make CI green.
- Unasked README, badges, or a second `AGENTS.md` / `CLAUDE.md`.
- Extra dependencies for a one-liner the stdlib or existing stack already covers.
- `gitnexus analyze` (or any indexer) injecting a second H1 into `AGENTS.md`.
- Inventing a second loop, a parallel skill tree, or TODO instead of finishing the claim.
- Broad glob rewrites (`**/*`) when the blast radius is one module.

## Stop and shrink

If the diff teaches a new architecture, you overshot. Split or delete until a reviewer can name the issue in the first screen.
