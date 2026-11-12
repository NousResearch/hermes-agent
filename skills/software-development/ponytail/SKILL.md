---
name: ponytail
description: Choose the smallest safe implementation that works.
version: 1.0.0
author: SeoYeonKim (@westkite1201), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [coding, simplification, yagni, minimalism, refactor]
    related_skills: [simplify-code, requesting-code-review, test-driven-development]
---

# Ponytail Skill

Ponytail is an opt-in coding discipline for finding the smallest safe solution.
It removes speculative complexity without removing correctness, security, data
safety, accessibility, verification, or an explicit user requirement.

## When to Use

Use this skill only when the user explicitly asks for Ponytail, YAGNI, a
minimal solution, the shortest safe path, or an aggressive simplification.
It also fits a requested implementation that specifically calls out avoiding
dependencies, wrappers, factories, configuration, or future-proofing.

Do not auto-run Ponytail for every coding task. Do not use it to skip reading
the relevant code or diagnosing a bug's root cause.

## Prerequisites

- Access to the task, nearby code, and the repository's own instructions.
- A clear statement of the behavior that must remain true.
- For a bug, a confirmed root cause. Use `systematic-debugging` or the
  repository's normal debugging workflow before simplifying the fix.
- Use `search_files` to look for existing helpers and `read_file` to inspect
  the actual call path before proposing new code.

No external service or dependency is required.

## How to Run

State the requested intensity when useful:

- **lite** — implement the request normally and name a simpler alternative.
- **full** — default; enforce the ladder for the current task.
- **ultra** — challenge unneeded requirements and prefer deletion when the
  user explicitly asks for aggressive YAGNI.

The skill is task-scoped. Never claim that Ponytail remains enabled for future
sessions.

## Quick Reference

Stop at the first rung that fully satisfies the requested behavior:

| Rung | Question | Preferred result |
|---|---|---|
| 1 | Does this need to exist? | Skip speculative work. |
| 2 | Does the codebase already solve it? | Reuse the existing path. |
| 3 | Does the standard library solve it? | Use the built-in. |
| 4 | Does the platform or framework solve it? | Use native behavior. |
| 5 | Does an installed dependency solve it? | Reuse it without adding one. |
| 6 | Can clear local code solve it? | Write the fewest readable lines. |

Complexity is justified when it protects a concrete requirement. In
particular, do not simplify away:

- trust-boundary validation, authentication, authorization, or secret handling
- error handling that prevents corrupt state or data loss
- accessibility basics in user-facing interfaces
- migrations and compatibility paths that protect existing users or data
- operational observability that production support depends on
- tests for non-trivial, security-sensitive, monetary, parsing, or concurrent
  behavior
- calibration, tolerances, or safety margins for physical systems

## Procedure

1. **Fix the boundary.** Restate the required behavior and explicit constraints.
   Completion: every proposed deletion can be checked against that boundary.
2. **Read the path.** Inspect the relevant implementation and callers before
   editing. Completion: the real shared change point or root cause is known.
3. **Climb the ladder.** Check existing code, standard library, native features,
   and installed dependencies in order. Completion: the highest viable rung is
   selected with evidence.
4. **Make the smallest safe change.** Prefer one shared-path fix, deletion over
   addition, and direct local code over a one-use abstraction. Completion: no
   requested behavior or safety invariant was removed.
5. **Verify narrowly.** Run the repository's smallest relevant test, lint,
   type, or build command. Add one focused regression test for non-trivial
   behavior when a test suite exists. Completion: the changed behavior is
   exercised, not merely imported.
6. **Report the tradeoff.** State what changed, what complexity was skipped,
   and the concrete trigger for adding it later. Completion: the user can tell
   when the minimal design would stop being sufficient.

Suggested response shape:

```text
changed: <smallest working fix>
skipped: <dependency, abstraction, or config not added>
add when: <specific future condition that justifies it>
```

If the user explicitly chooses a fuller version after seeing the simpler path,
implement the requested version without repeatedly arguing against it.

## Pitfalls

1. **Small diff, wrong path.** Minimalism starts after comprehension; patch the
   shared cause rather than several symptoms.
2. **Safety labeled as bloat.** Validation, auth, data protection,
   accessibility, and meaningful tests are constraints, not ornament.
3. **One-use architecture.** Wait for a second concrete implementation before
   adding an interface, factory, registry, or configuration layer.
4. **A new dependency for a tiny helper.** Prefer existing dependencies,
   standard library, or a few clear local lines.
5. **Clever compression.** Fewer lines are not better when the result is harder
   to read, debug, or operate.
6. **No verification.** A minimal implementation still needs the smallest
   check that proves its behavior.

## Verification

- [ ] The user explicitly requested the minimal/YAGNI mode.
- [ ] The relevant code path or bug root cause was inspected first.
- [ ] Existing code, standard library, and native features were checked.
- [ ] No safety, compatibility, accessibility, or data invariant was removed.
- [ ] No speculative dependency or abstraction was introduced.
- [ ] The smallest relevant runnable verification passed.
- [ ] Skipped complexity has a concrete future trigger.

## Attribution

Adapted from [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail)
under the MIT License.

```text
MIT License

Copyright (c) 2026 DietrichGebert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
