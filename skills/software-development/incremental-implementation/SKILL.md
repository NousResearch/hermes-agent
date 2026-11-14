---
name: incremental-implementation
description: 'Deliver changes incrementally across files. Use when implementing any feature or change that touches more than one file, or when a task feels too big to land in one step.'
version: 1.0.0
author: Hermes Agent (adapted from obra/superpowers)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [implementation, incremental, workflow, discipline]
    related_skills: [test-driven-development, executing-plans, karpathy-guidelines]
---

# Incremental Implementation

Deliver changes incrementally. One file, one change, one commit at a time.

## When to Use

- When implementing any feature or change that touches more than one file
- When about to write a large amount of code at once
- When a task feels too big to land in one step
- When you need to keep changes reviewable

## The Process

### 1. Break Down First

Before touching code, list every file that needs to change and what changes in each. Order them so each change builds on the previous one.

### 2. One Change at a Time

For each file:

1. Make the minimal change
2. Run relevant tests
3. Verify the change works in isolation
4. Commit with a descriptive message
5. Move to the next file

### 3. Commit Discipline

- **Atomic commits**: each commit is one logical change
- **Descriptive messages**: what changed and why
- **No broken intermediate states**: every commit should pass tests
- **Build after each commit**: ensure you haven't broken the build

### 4. Review Checkpoints

After every 3-5 commits:

- Review the cumulative diff
- Check for consistency across files
- Verify the feature is coming together as designed
- Adjust the remaining plan if needed

## Anti-Patterns

| Don't                          | Do                                        |
| ------------------------------ | ----------------------------------------- |
| Change 5 files, then test      | Change 1 file, test, commit, repeat       |
| "I'll clean it up later"       | Clean up now, before the next change      |
| One giant commit               | One commit per logical change             |
| Skip tests "to save time"      | Tests save time by catching issues early  |
| Refactor while adding features | Separate refactors into their own commits |

## Hermes Integration

- Use `write_file` / `replace_string_in_file` for each incremental change
- Use `terminal` for tests after each commit
- Use `search_files` to verify no unintended side effects
- Combine with `test-driven-development` for test-first increments
- Combine with `karpathy-guidelines` for surgical change discipline
- Use `git-workflow-and-versioning` skill for commit conventions
