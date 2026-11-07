---
name: ponytail-minimal-code
description: "Use when writing or reviewing code to minimize complexity."
version: 1.0.0
author: Het / @hetdev, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [coding, yagni, minimal-code, code-review, best-practices, ponytail]
    related_skills: [simplify-code, requesting-code-review, test-driven-development]
---

# Ponytail Minimal Code Skill

Ponytail makes implementation deliberately smaller by questioning whether
code is needed before choosing how to write it. It applies YAGNI and a
decision ladder while keeping validation, security, accessibility, and data
safety intact.

**Credits:** This skill adapts the MIT-licensed
[Ponytail project](https://github.com/DietrichGebert/ponytail) by
[Dietrich Gebert](https://github.com/DietrichGebert) for the Hermes Agent skill
system.

## When to Use

- Use before implementing, refactoring, or reviewing code.
- Use when choosing between a standard-library feature, a platform feature,
  an installed dependency, or new code.
- Do not use minimalism to skip a real security, validation, accessibility,
  error-handling, or data-loss requirement.

## Prerequisites

- No setup or external runtime dependency is required.
- Read the relevant code with `read_file` and locate existing solutions with
  `search_files`.
- Check the project’s existing dependencies and platform capabilities before
  proposing a new dependency.
- Use `terminal` for commands and tests; use `patch` for focused edits.

## How to Run

Apply the decision ladder in order before writing code. Stop at the first step
that satisfies the requirement, then use `terminal` to run the relevant tests.

## Quick Reference

| Step | Question | Action |
| --- | --- | --- |
| 1 | Does this need to exist? | Skip it when the answer is no. |
| 2 | Does the standard library do it? | Use the standard library. |
| 3 | Does the native platform feature do it? | Use the platform feature. |
| 4 | Does an installed dependency do it? | Reuse the dependency. |
| 5 | Is it a one-liner? | Write one line. |
| 6 | Is new code still required? | Write the minimum that works. |

## Procedure

1. Question the requirement. Remove hypothetical features, abstractions,
   utility files, and configuration that no real requirement needs.
2. Check the standard library before writing a custom implementation.
3. Check native platform features before adding a library.
4. Check installed dependencies before adding a new dependency.
5. Prefer a one-liner when it remains clear and correct.
6. Write the minimum implementation that handles real edge cases.
7. Inline small functions, and avoid a class, factory, interface, or plugin
   architecture when one implementation or caller is sufficient.
8. Preserve trust-boundary validation, data-loss prevention, security,
   accessibility, and error handling for real failure modes.
9. During review, ask whether the file can be deleted, the function replaced
   by a standard-library call, the class made a function, the abstraction
   inlined, or the dependency removed.
10. When delegating through `delegate_task`, include the rule to use the
    standard library first, existing dependencies second, and the minimum code
    that works.

## Pitfalls

- Do not install a package for a standard-library operation.
- Do not create an abstraction layer with one concrete implementation.
- Do not build infrastructure for hypothetical future requirements.
- Do not add error handling that silently hides failures.
- Do not confuse “lazy” with skipping validation, security, accessibility, or
  data-loss safeguards.

## Verification

- Confirm that every new file, abstraction, dependency, and configuration value
  answers a current requirement.
- Confirm that standard-library, platform, and installed-dependency options
  were considered in that order.
- Run the focused tests with `terminal` and confirm that real edge cases and
  safety requirements remain covered.
- Confirm that the final implementation is the smallest clear solution that
  satisfies the requirement.
