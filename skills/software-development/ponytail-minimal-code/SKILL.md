---
name: ponytail-minimal-code
description: "Use when writing or reviewing code. Forces the lazy-senior-dev decision ladder: question if code is needed, use stdlib/platform/existing deps first, write the minimum that works. Inspired by Ponytail (YAGNI for AI agents)."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [coding, yagni, minimal-code, code-review, best-practices, ponytail]
    related_skills: [simplify-code, requesting-code-review, test-driven-development]
---

# Ponytail — Write Less Code

Makes you think like the laziest senior dev in the room. The best code is
the code you never wrote.

> He says nothing. He writes one line. It works.

## The Decision Ladder

Before writing ANY code, run through this ladder **in order**. Stop at the
first step that satisfies the requirement:

```
1. Does this need to exist?   → NO: skip it entirely (YAGNI)
2. Stdlib does it?            → USE IT — don't reinvent
3. Native platform feature?   → USE IT — browser/OS/framework built-ins
4. Installed dependency?      → USE IT — check what's already in the project
5. One-liner?                 → WRITE ONE LINE
6. Only then:                 → Write the MINIMUM that works
```

### 1. Question everything

- "Do we actually need this feature / file / abstraction?" — if unsure, don't create it.
- Don't pre-build for hypothetical future requirements.
- Don't create utility files "just in case".
- Don't add config files nobody asked for.

### 2. Stdlib first

If the standard library can do it, never write a custom implementation.

| Language | Prefer |
|----------|--------|
| Python | `collections`, `itertools`, `pathlib`, `dataclasses`, `json`, `csv`, `hashlib`, `unittest.mock` |
| JavaScript / TypeScript | `Array.prototype.*`, `Object.*`, `Intl`, `URL`, `fetch`, `crypto.subtle` |
| Go | `strings`, `slices`, `maps`, `encoding/json`, `net/http` |
| Rust | `std::collections`, `std::fs`, `std::process` |

### 3. Platform features

- `<input type="date">` not a date-picker library.
- `<dialog>` not a modal library.
- CSS `grid` / `flexbox` not a layout framework.
- Python `venv` + `pip` when a full dependency manager isn't needed.
- React `useState` / `useReducer` not a global state library for local state.

### 4. Existing dependencies

- Check `package.json`, `requirements.txt`, `pyproject.toml`, `go.mod` before adding new deps.
- Don't install `lodash` if native array methods suffice.
- Don't install `axios` if `fetch` is available.
- Don't install `moment` / `dayjs` if `Intl.DateTimeFormat` works.

### 5. One-liners

- If a task is a single expression, write a single expression.
- Don't wrap a one-liner in a class with a factory pattern.
- Don't add error handling to a line that can't fail.

### 6. Minimum viable code

- Write the smallest implementation that passes tests and handles **real** edge cases.
- Inline small functions rather than creating util files.
- Skip the interface / abstract base class if there's one implementation.
- Skip the factory if there's one product.

## What NOT to do

- ❌ Install a package for something the stdlib does.
- ❌ Create an abstraction layer with one concrete implementation.
- ❌ Write a config system when constants work.
- ❌ Add a wrapper class around a function call.
- ❌ Build a plugin architecture nobody will extend.
- ❌ Create separate files for 3-line utilities.
- ❌ Add error handling that swallows errors silently.
- ❌ Write retry logic for non-transient failures.

## Lazy, not negligent

This philosophy is **not** about cutting corners on:

- ✅ Trust-boundary validation (auth, input sanitization, SQL parameterization).
- ✅ Data-loss prevention (transactions, backups, atomic writes).
- ✅ Security (sanitization, CSRF, parameterized queries).
- ✅ Accessibility (semantic HTML, ARIA, keyboard nav).
- ✅ Error handling for **real** failure modes.

It **is** about:

- ✅ Not building infrastructure for features that don't exist yet.
- ✅ Not abstracting code that has one caller.
- ✅ Not importing libraries for trivial operations.
- ✅ Not writing boilerplate the framework already handles.

## Code review with Ponytail

When reviewing code — yours, a teammate's, or subagent output — ask:

1. *Could this entire file be deleted?* → Delete it.
2. *Could this function be a stdlib call?* → Replace it.
3. *Could this class be a function?* → Simplify it.
4. *Could this abstraction be inlined?* → Inline it.
5. *Is this dependency necessary?* → Remove it if stdlib works.
6. *Is this handling a real edge case or an imaginary one?* → Remove imaginary handling.

## When delegating to subagents

Include this directive in the delegation `context`:

> Follow Ponytail rules: use stdlib first, existing deps second, write the
> minimum code that works. Don't create unnecessary abstractions, files,
> or dependencies. Question whether each piece of code needs to exist.

## Attribution

Inspired by [Ponytail](https://github.com/DietrichGebert/ponytail) by
Dietrich Gebert — "Makes your AI agent think like the laziest senior dev
in the room." This skill adapts Ponytail's decision ladder and YAGNI
principles for the Hermes Agent skill system.
