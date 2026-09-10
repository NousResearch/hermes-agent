---
name: explain-error
description: Diagnose an error or stack trace and suggest a fix.
version: 0.1.0
author: Burak Koç (@HeLLGURD), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
category: software-development
triggers:
  - "explain this error"
  - "what does this error mean"
  - "why is this failing"
  - "debug this stack trace"
  - "what's wrong with this traceback"
  - "fix this exception"
  - "I'm getting an error"
  - "help me understand this crash"
toolsets:
  - terminal
  - file
metadata:
  hermes:
    tags: [Debugging, Errors, Stack-Trace, Exceptions, Troubleshooting]
    related_skills: [code-wiki, rest-graphql-debug]
---

# Explain Error Skill

Paste any error message, exception, or stack trace and get a plain-English
explanation: what broke, *why*, exactly where, and how to fix it. It reads the
trace in the correct order for the language at hand — including Python's
bottom-up layout — and points at the line in *your* code. It does not profile
slow-but-working code, write new features, or audit for security issues.

## When to Use

- The user pastes an error, traceback, or stack trace and wants it explained
- A command fails and the output is cryptic
- A new Hermes user hits an error during setup and is stuck
- An exception shows up in logs and the root cause isn't obvious

Do NOT use for:
- Performance problems (slow, not broken) — that's profiling
- Writing new features — this is for diagnosing what's already broken
- Security vulnerability review — use the `web-pentest` skill
- Errors the user has already diagnosed — go straight to the fix

## Prerequisites

- None. The skill works on pasted text alone.
- Optional: access to the project source, so the failing line can be read
  directly with `read_file` for a more precise diagnosis.

## How to Run

Answer in four fixed parts so the user always knows what to expect:

1. **TL;DR** — one sentence: what broke, in plain words.
2. **Root cause** — *why* it happened, traced to the real trigger, not the
   surface symptom.
3. **Location** — the exact file and line that matters (the user's code).
4. **Fix** — a concrete, copy-pasteable change; lead with the recommended one.

## Quick Reference

Identify the language from the error's signature:

| Language / Runtime | Tell-tale signature |
|---|---|
| Python | `Traceback (most recent call last):`, `File "...", line N` |
| JavaScript / Node | `at Object.<anonymous>`, `ReferenceError`, `.js:N:M` |
| TypeScript | `TS2322`, `error TS####`, `.ts(N,M)` |
| Java | `Exception in thread`, `at com.foo.Bar.method(Bar.java:N)` |
| Go | `panic:`, `goroutine N [running]:`, `.go:N` |
| Rust | `thread 'main' panicked at`, `error[E####]` |
| C / C++ | `segmentation fault`, `undefined reference to` |
| Ruby | `... (RuntimeError)`, `from file.rb:N:in` |

## Procedure

### Step 1 — Read the trace in the right order

Stack traces are mostly framework frames; two things matter, and **their
position depends on the language** — read it the right way round:

- **Python** prints the frames first and the **exception type + message
  LAST, at the very bottom**. The header `Traceback (most recent call last)`
  says exactly that: the calls are listed oldest first, so the **last frame
  listed** — the one directly above the exception line — is where the error was
  actually raised. Read it bottom-up: the final line is the *what*, and the
  frame just above it is the *where*.
- **Java / JavaScript** put the exception type + message on the **first** line,
  with the `at ...` frames below it. Read those top-down.

From the raising frame, walk through the frames toward the user's own code,
skipping library frames (`site-packages`, `node_modules`, `vendor/`, stdlib),
until you reach a path inside the project. That frame is the *where*.

### Step 2 — Diagnose the root cause

Map the error to its real trigger:

| Error pattern | Usually means |
|---|---|
| `NoneType has no attribute X` / `undefined is not an object` | A value expected to exist is null/None — check what produced it |
| `KeyError` / undefined property | Accessing a key/field that isn't there — typo or missing data |
| `IndexError` / `index out of range` | Indexing past the end of a list/array |
| `ImportError` / `ModuleNotFoundError` | Dependency not installed, wrong venv, or wrong import path |
| `TypeError: expected X got Y` | Wrong argument type — often str vs int, or None where a value is required |
| `ConnectionError` / `ECONNREFUSED` | Service not running, wrong host/port, or firewall |
| `segmentation fault` | Invalid memory access — null pointer, buffer overrun, use-after-free |

Don't stop at the symptom. `NoneType has no attribute 'name'` isn't "add a null
check" — it's "*why* is this None when the code assumed it wasn't?" Trace it
back.

### Step 3 — Confirm against the source

If the project is on disk, confirm instead of guessing. Use `read_file` on the
failing file with an offset around the reported line (a few before through a few
after) to see it in context. To locate where a symbol is defined or used
elsewhere, use `search_files`. Reading the real code turns a probable
explanation into a certain one.

### Step 4 — Deliver the four-part answer

Format the response using the four parts above. Example:

```
TL;DR — You're calling .strip() on a value that's None, so Python crashes.

Root cause — config.get("name") returns None when the "name" key is missing.
The next line assumes it's a string and calls .strip(), which None lacks.

Location — src/loader.py, line 42:
    name = config.get("name").strip()

Fix — Provide a default so the value is always a string:
    name = (config.get("name") or "").strip()
```

## Pitfalls

- **Chained exceptions.** Python prints two joined tracebacks — `During
  handling of the above exception, another exception occurred` (implicit) or
  `The above exception was the direct cause of the following exception`
  (`raise ... from`). Python prints them oldest-first, so the **original cause
  is the FIRST/top block** and the exception that finally propagated is the
  **LAST/bottom block**. Both phrases point back at "the above exception" — the
  earlier block. Start from the top block; that's the real cause, and the
  final block is usually fallout.
- **Truncated trace.** Work with what's shown, but say what's missing: "the
  trace is cut off above this frame — share the full output to pinpoint it."
- **Minified JS** (`at t.default (bundle.min.js:1:48291)`). The frame is
  useless; recommend source maps, or work from the message and the user-code
  entry point.
- **Compiler error wall** (dozens of C++/Rust errors). Fix the **first** error
  only — the rest are usually cascading fallout. Say so.
- **No stack trace, just a message.** Diagnose from the message plus the command
  that produced it; ask for the command if it's missing.
- **Calibrate the depth.** Beginner phrasing ("what is a traceback") → explain
  the concept too. Expert phrasing (bare trace, precise terms) → lead with the
  root cause and fix, and stay terse.

## Verification

- If the project is runnable, give the exact command to confirm the fix and run
  it with `terminal`, e.g. `python -m pytest tests/test_loader.py -x`.
- Don't claim the fix works — say "this should resolve it; re-run X to confirm,"
  then let the re-run prove it.
