---
name: self-sufficient-coding-agent
description: >
  Operate as a single self-sufficient coding agent — no external agent calls, no handoffs.
  Handles any coding, research, writing, or automation task end-to-end using only native tools.
  Use when the user wants "just you, no Claude Code, no delegation" for any task.
tags: [meta, agent-pattern, coding, autonomous]
---

# Self-Sufficient Coding Agent

## When to use
- User says "just do it yourself" / "no delegation" / "single agent"
- Any coding task that would normally go to Claude Code
- Complex multi-step work requiring iterative edit-test loops
- Research + implementation combined tasks

## Core Loop Pattern

Every task follows this cycle until done:

```
1. ORIENT  — read_file / search_files / web_search to understand context
2. PLAN    — break task into concrete steps (don't announce, just do)
3. EDIT    — patch() for targeted changes, write_file() for new files
4. VERIFY  — terminal() to run tests, lint, build, or validate
5. FIX     — if verify fails, loop back to EDIT with the error as context
6. DONE    — report what was done with real output evidence
```

## Tool Mapping (Claude Code → Hermes)

| Claude Code Action    | Hermes Equivalent                                    |
|-----------------------|------------------------------------------------------|
| `bash(command)`       | `terminal(command)`                                  |
| `str_replace(file,old,new)` | `patch(path, old_string, new_string)`          |
| `read(file, offset, limit)` | `read_file(path, offset, limit)`               |
| `glob(pattern)`       | `search_files(pattern, target="files")`             |
| `grep(pattern)`       | `search_files(pattern, target="content")`           |
| `write(file, content)`| `write_file(path, content)`                         |
| ReAct loop            | Just keep calling tools — I am the loop              |
| Todo tracking         | `todo()` — track subtasks                           |
| Background process    | `terminal(background=true)` + `process(action="poll")` |

## Multi-Step Code Changes

For large refactors or multi-file edits:
1. `search_files` to find all affected files
2. `read_file` each file to understand context
3. `patch()` each file with targeted changes
4. `terminal` to verify (lint, typecheck, build, test)
5. Fix any failures in the loop

## Research + Implementation

For tasks requiring external knowledge:
1. `web_search` for relevant docs/examples
2. `web_extract` to read the actual documentation
3. Implement using findings
4. Verify the implementation works

## Parallel Work

For independent subtasks:
1. `execute_code()` with multiple tool calls in one script
2. Or `delegate_task(tasks=[...])` for parallel subagent work (only when truly parallel and independent)

## Learning from Experience

Before implementing, check if you've solved something similar:
- `session_search(query="<task keywords>")` for past approaches
- `fact_store(action="search", query="<domain>")` for stored patterns
- Note what worked and what failed last time

After completing, self-score:
- Tests passing: +0.2
- Zero lint errors: +0.15
- Documentation present: +0.1
- Score < 0.7 = needs improvement before delivery

## Pitfalls

- Don't fabricate output — if a tool call fails, report it honestly
- Don't describe what you'll do — just do it and show the result
- Use `patch()` not `sed` for file edits — it's safer with fuzzy matching
- Use `read_file()` not `cat` — it handles pagination and line numbers
- Use `search_files()` not `grep`/`find` — it's ripgrep-backed and faster
- For CRLF-sensitive files on /mnt/c/, use Python binary mode (see memory)
- Always verify after editing — never assume an edit succeeded

## Verification Template

After completing any task, report as:
```
TASK: [what was requested]
STATUS: DONE / BLOCKED / PARTIAL

CHANGES:
- file1.ts: [what changed]
- file2.ts: [what changed]

VERIFICATION:
- [test command]: PASS/FAIL
- [build command]: PASS/FAIL
- [lint command]: PASS/FAIL

OUTPUT:
[actual terminal output showing results]
```
