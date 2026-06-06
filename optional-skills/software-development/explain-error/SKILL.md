---
name: explain-error
description: |
  Decode any error message, exception, or stack trace — identify the language
  and framework, explain the root cause in plain English, point at the exact
  failing line, and give a concrete fix. Works for any language, any stack.
version: 0.1.0
author: HeLLGURD
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
    tags: [Debugging, Errors, Stack-Trace, Exceptions, Troubleshooting, Beginner-Friendly]
    related_skills: [git-workflow, code-wiki, rest-graphql-debug]
---

# Explain Error

Paste any error message, exception, or stack trace and get a clear,
plain-English explanation: what went wrong, *why*, exactly where, and how to
fix it. No jargon dumps — the explanation is calibrated to be useful whether
you're brand-new to the language or a senior engineer triaging an unfamiliar
stack.

Works for any language and any runtime. Uses only `terminal` and `file` tools —
no external services.

---

## When to Use

- User pastes an error, traceback, or stack trace and wants to understand it
- A command fails and the output is cryptic
- A new Hermes user hits an error during setup or a task and is stuck
- An exception appears in logs and the root cause isn't obvious

Do NOT use for:
- Designing new code from scratch — that's not debugging
- Performance profiling (slow, not broken) — different problem
- Errors the user has already diagnosed and just wants help fixing — go
  straight to the fix

---

## Prerequisites

- None required. The skill works on pasted text alone.
- Optional: access to the project source (so the failing line can be read
  directly via `read_file` for a more precise diagnosis).

---

## The Four-Part Answer

Every explanation follows the same structure so the user always knows what to
expect:

1. **TL;DR** — one sentence: what broke, in plain words.
2. **Root cause** — *why* it happened, traced to the actual trigger (not just
   the surface symptom).
3. **Location** — the exact file and line that matters (the user's code, not
   the framework internals).
4. **Fix** — a concrete, copy-pasteable change. If there are multiple valid
   fixes, lead with the recommended one.

---

## Procedure

### Step 1 — Identify the language and runtime

From the error format, determine the stack. Signature patterns:

| Language / Runtime | Tell-tale signature |
|---|---|
| Python | `Traceback (most recent call last):`, `File "...", line N` |
| JavaScript / Node | `at Object.<anonymous>`, `ReferenceError`, `TypeError`, `.js:N:M` |
| TypeScript | `TS2322`, `error TS####`, `.ts(N,M)` |
| Java | `Exception in thread`, `at com.foo.Bar.method(Bar.java:N)` |
| Go | `panic:`, `goroutine N [running]:`, `.go:N` |
| Rust | `thread 'main' panicked at`, `error[E####]` |
| C / C++ | `segmentation fault`, `undefined reference to`, `error:` from gcc/clang |
| Ruby | `... (RuntimeError)`, `from file.rb:N:in` |
| Shell | `command not found`, `No such file or directory`, non-zero exit codes |
| Database | `SQLSTATE`, `ORA-####`, `ERROR: ...` from psql/mysql |

If the language is ambiguous, ask the user — but most of the time the
signature is unmistakable.

### Step 2 — Find the real error in the noise

Stack traces are mostly framework/library frames. The line that matters is
usually:
- The **deepest frame in the user's own code** (not in `site-packages`,
  `node_modules`, `vendor/`, stdlib, etc.)
- The **actual exception type + message** at the top (Python) or bottom (Java)
  of the trace

Strategy:
1. Read the exception type and message first — that's the *what*.
2. Walk the stack from the top, skipping framework frames, until you hit a
   path inside the user's project. That frame is almost always the *where*.
3. If source is available, `read_file` that line + 5 lines of context to
   confirm the *why*.

### Step 3 — Diagnose the root cause

Map the error to its real trigger. Common error families and what they
actually mean:

| Error pattern | Usually means |
|---|---|
| `NoneType has no attribute X` / `undefined is not an object` | A value you expected to exist is null/None — check what produced it |
| `KeyError` / `undefined` property | Accessing a key/field that isn't there — typo or missing data |
| `IndexError` / `index out of range` | Looping or indexing past the end of a list/array |
| `ImportError` / `ModuleNotFoundError` / `Cannot find module` | Dependency not installed, wrong venv, or wrong import path |
| `TypeError: expected X got Y` | Wrong argument type — often a string vs int, or None passed where a value is required |
| `ConnectionError` / `ECONNREFUSED` | Target service not running, wrong host/port, or firewall |
| `PermissionError` / `EACCES` | File/dir permissions, or writing to a protected path |
| `TS2322` / `is not assignable to type` | A value's type doesn't match the expected type — often a missing field or wrong shape |
| `segmentation fault` | Invalid memory access — null pointer, buffer overrun, use-after-free |
| `panic: runtime error: index out of range` | Go slice/array bounds violation |

Don't stop at the surface symptom. `NoneType has no attribute 'name'` isn't
"add a null check" — it's "*why* is this None when the code assumed it
wouldn't be?" Trace it back.

### Step 4 — Read the source if available

If the project is on disk, confirm the diagnosis instead of guessing:

```bash
# Confirm the failing file exists
ls -la path/to/failing_file.py

# Read the failing line with context
sed -n '80,95p' path/to/failing_file.py    # or use read_file with offset
```

Reading the actual code turns a probable explanation into a certain one.

### Step 5 — Deliver the four-part answer

Format the response using the structure from "The Four-Part Answer" above.
Example:

```
TL;DR — You're calling .strip() on a value that's None, so Python crashes.

Root cause — config.get("name") returns None when the "name" key is missing
from the config file. The next line assumes it's always a string and calls
.strip() on it, which None doesn't support.

Location — src/loader.py, line 42:
    name = config.get("name").strip()

Fix — Provide a default so the value is always a string:
    name = (config.get("name") or "").strip()

Or, if a missing name should be an error, fail loudly with a clear message:
    name = config.get("name")
    if name is None:
        raise ValueError("config is missing required 'name' field")
    name = name.strip()
```

### Step 6 — Verify the fix (when possible)

If the project is runnable, suggest the exact command to confirm the fix:

```bash
# Re-run the failing command / test
python -m pytest tests/test_loader.py::test_missing_name -x
```

Don't claim the fix works — say "this should resolve it; re-run X to confirm."

---

## Calibrating the Explanation

Read the user's apparent level from how they phrase the question and adjust:

- **Beginner signals** ("I don't understand", "what is a traceback", "new to
  Python") → explain the concept too (what a NoneType is, what a stack trace
  represents), avoid unexplained jargon.
- **Expert signals** (pastes only the trace, uses precise terms) → skip the
  basics, lead with root cause and fix, keep it terse.

When unsure, default to clear-but-not-condescending: explain the *why* without
assuming they already know it.

---

## Edge Cases

**Truncated stack trace:**
Work with what's given, but flag what's missing: "The trace is cut off above
this frame — if you can share the full output, I can pinpoint the exact
trigger." Often the visible part is still enough.

**Multiple chained exceptions** (Python `During handling of the above
exception, another exception occurred`):
Explain the *original* exception first (the bottom one) — that's the real
cause. The later one is often just fallout.

**Minified / bundled JS error** (`at t.default (bundle.min.js:1:48291)`):
The trace points at minified code. Recommend enabling source maps and
re-running, or look at the error *message* and the user-code entry point
instead of the minified frame.

**Compiler error wall** (dozens of C++/Rust errors):
Fix the *first* error only. Most of the rest are cascading fallout that
disappears once the first one is resolved. Say so explicitly.

**Error with no stack trace** (just a message):
Diagnose from the message + the command that produced it. Ask for the command
if it wasn't provided.

---

## What This Skill Does NOT Cover

- Performance issues (slow code that still works) — that's profiling, not
  error analysis
- Writing new features — this is for diagnosing what's broken
- Security vulnerability analysis — use the `web-pentest` skill for that
- Flaky/intermittent failures that need reproduction — those need a different,
  investigative approach
