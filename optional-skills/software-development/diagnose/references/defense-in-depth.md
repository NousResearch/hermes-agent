# Defense-in-Depth Validation

Adapted from `obra/superpowers` (skills/systematic-debugging). Used by `diagnose` in Phase 5/6 when the root cause was invalid data or a violated invariant that travelled far.

## Core principle

Fixing the root cause stops *this* bug. But the bad value already proved it can cross the system unchecked — a different code path, a refactor, or a mock can carry the next one. **Validate at EVERY layer the data passes through. One check fixes the bug; layers make the bug class impossible.**

## The four layers

1. **Entry point** — reject obviously invalid input at the API boundary (null/empty, existence, type, shape).
2. **Business logic** — reject values that are syntactically fine but wrong for *this* operation.
3. **Environment guard** — refuse dangerous operations in the wrong context (e.g. in tests, refuse destructive commands outside the temp dir; in prod, refuse admin paths on a read replica).
4. **Debug instrumentation** — capture context (args, cwd, stack) right before the risky operation, so the next failure carries its own evidence.

```typescript
// Example spread across layers for "empty projectDir → git init in source tree"
// Layer 1 — entry:      createProject(name, dir): dir must be non-empty, exist, be a directory
// Layer 2 — business:   initializeWorkspace(dir): dir required for workspace init
// Layer 3 — env guard:  in tests, gitInit(dir) refuses dirs outside os.tmpdir()
// Layer 4 — debug log:  log { dir, cwd, stack } immediately before the git init call
```

## Applying the pattern

1. **Trace the data flow** — where did the bad value originate, and what did it cross? (See `root-cause-tracing.md`.)
2. **Map every checkpoint** — list each boundary the value passed through unchecked.
3. **Add validation at each layer** — each layer defends against the failures the others miss: alternate code paths bypass entry checks, mocks bypass business checks, platform edge cases need env guards.
4. **Test each layer** — try to bypass layer 1 deliberately; verify layer 2 or 3 catches it.

## When to apply — and when not to

Apply when the diagnosed bug moved bad data through multiple components. Skip it for a purely local logic bug — layering checks around a self-contained mistake is just noise.
