---
name: codebase-index
description: Use when working on any codebase task — finding symbols, tracing dependencies, searching code. Replaces grep/cat with index-backed queries that use 100x fewer tokens.
version: 1.0.0
author: Mibayy
license: MIT
metadata:
  hermes:
    tags: [codebase, code-navigation, tokens, mcp, search, refactoring]
    related_skills: [systematic-debugging, code-review, writing-plans]
---

# Codebase Index

## Prerequisites

**This skill requires mcp-codebase-index to be installed and configured.**

Without the MCP server running, the tools below do not exist and the skill has
no effect. See the [Codebase Index setup guide](../../../website/docs/user-guide/features/codebase-index.md)
for installation and configuration instructions.

## When to Use

Use this skill at the start of any session involving code navigation:

- Finding where a function or class is defined
- Tracing what calls what (impact analysis before refactoring)
- Searching for a pattern across a large codebase
- Understanding a file's structure without reading all of it
- Reviewing a PR diff and needing context on affected symbols

## Core Rule

**Never use `terminal("grep ...")` or `read_file` for code navigation when
the codebase-index tools are available.**

| Instead of | Use |
|------------|-----|
| `terminal("grep -r 'foo' .")` | `search_codebase(pattern="foo")` |
| `read_file("src/run_agent.py")` | `get_function_source("function_name")` or `get_class_source("ClassName")` |
| `terminal("cat file.py \| grep class")` | `get_structure_summary(file_path="src/file.py")` |
| `read_file` + manual search | `find_symbol("symbol_name")` |

## Workflow

### 1. Orient (first tool call of any coding session)

```
list_projects()          → see what's indexed
switch_project("name")   → set the active project
get_project_summary()    → file count, top classes/functions
```

### 2. Find symbols

```
find_symbol("_honcho_prefetch")
→ returns: file, line range, signature, 20-line preview
```

### 3. Read only what you need

```
get_function_source("MyClass.my_method")
→ returns: full source of that method only
```

### 4. Trace dependencies before touching anything

```
get_dependencies("send_message")     → what does it call?
get_dependents("send_message")       → what calls it?
get_change_impact("send_message")    → transitive blast radius
```

### 5. Search precisely

```
search_codebase(pattern="contextTokens", file_glob="*.py")
→ returns: file, line number, matching line — no noise
```

## Pairs well with delegate_task

When used with `delegate_task` (PR #3387), pass `skills=["codebase-index"]` in
a task to give the subagent index access. The subagent gets the full skill
content injected into its system prompt and will call `search_codebase` /
`find_symbol` instead of grep/cat -- the token savings apply inside delegated
tasks too.

```json
{
  "tasks": [
    {
      "goal": "Find all callers of send_message and check for missing error handling",
      "skills": ["codebase-index"],
      "toolsets": ["terminal", "file"]
    }
  ]
}
```

This requires mcp-codebase-index to be running in the parent session. The MCP
tools are not available inside the subagent's isolated terminal, but the skill
content guides the subagent's reasoning about what to query via the parent's
tool proxy.

## Pairs well with structured memory

codebase-index works well alongside the native structured memory toolset
(PR #3093). It provides a typed, searchable fact store built directly into
Hermes — no external process, no pip install, no config block. Enable it with
`- structured_memory` in the enabled toolsets.

- **codebase-index** controls what the agent reads — symbols instead of files,
  so the context stays small during exploration.
- **structured memory** controls what the agent remembers — constraints, decisions,
  and values stored as typed facts (`C[]`, `D[]`, `V[]`) that survive context
  compression and are searchable across turns via FTS5.

Neither replaces the other. Running both is the recommended setup for extended
development sessions.

## Pitfalls

- **Cold start**: first index takes a few seconds per project. Normal.
- **Stale index**: if files changed outside git (direct writes), call `reindex()`.
- **Multi-project**: always check `list_projects()` and `switch_project()` if
  working across repos in the same session.
- **get_usage_stats()**: shows cumulative token savings per project — useful
  to verify the server is working.

## What not to do

- Do not `read_file` an entire source file to find one function.
- Do not `terminal("grep -r")` when `search_codebase` exists.
- Do not call `get_project_summary` repeatedly — once per session is enough.
- Do not skip `switch_project` when moving between repos — queries will hit
  the wrong index.
