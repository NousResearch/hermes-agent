# Root-Cause Tracing

Adapted from `obra/superpowers` (skills/systematic-debugging). Used by `diagnose` during Phase 3/4 when the error surfaces deep in the call stack.

## Core principle

Bugs manifest deep in the call stack — a file created in the wrong directory, a DB opened with the wrong path, a null dereference ten calls away from the bad input. The instinct is to fix where the error appears; that treats the symptom. **Trace backward through the call chain to the original trigger, and fix at the source.**

## When to use

- Error happens deep in execution, not at the entry point
- Stack trace shows a long call chain
- It is unclear where the invalid value originated
- A stray file/state appears and you don't know which test or caller creates it

## The tracing process

1. **Observe the symptom** — quote the exact error and throw site.
2. **Find the immediate cause** — which line directly produces the failure?
3. **Ask: what called this, with what value?** — walk one level up, note the argument/state that was bad.
4. **Repeat until the value is *created*, not just passed on** — the root cause is where the bad value was born, not the last function that touched it.
5. **Fix at the source** — then consider `defense-in-depth.md` so no layer can pass that value through again.

## Example trace (real pattern)

Symptom: `.git` directory created inside the source tree.

```
git init runs with cwd = process.cwd()        ← empty cwd parameter
  ↑ WorktreeManager.createSessionWorktree(projectDir)   — projectDir = ""
  ↑ Session.create() passed empty string
  ↑ Test read context.tempDir before beforeEach ran
  ↑ setupCoreTest() initialises tempDir to ""
```

Root cause: initialization order — a value consumed before it was assigned. Fix at the setup/getter, not inside `git init`.

## When you can't trace manually — capture the stack

Add temporary instrumentation at the operation that fails, log the inputs **and** the call stack, run the Phase 1 loop, grep for the tag:

| Stack | Stack capture |
|---|---|
| JS/TS | `console.error('[DEBUG-x1]', { arg }, new Error().stack)` — use `console.error`, not the logger (loggers get suppressed in tests) |
| Python | `import traceback; print("[DEBUG-x1]", arg); traceback.print_stack()` |
| .NET | `Console.Error.WriteLine($"[DEBUG-x1] {arg}\n{Environment.StackTrace}")` |

Tips:

- Log **before** the dangerous operation, not after it fails.
- Include context: arguments, cwd, env vars, correlation id.
- Look for the *pattern* across runs — same test, same parameter, same caller.

## Finding which test causes pollution

If stray files/state appear during a test run and you don't know the culprit, bisect test files with `scripts/find-polluter.sh`:

```bash
scripts/find-polluter.sh '.git' 'src/**/*.test.ts' 'npx vitest run {file}'
```

It runs each matching test file in isolation and stops at the first one that leaves the pollution behind.
