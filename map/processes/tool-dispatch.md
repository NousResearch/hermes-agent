---
id: tool-dispatch
kind: process
universe: runtime
name: Tool Dispatch
summary: >
  Resolve a model-requested tool call into a handler invocation:
  schema assembly, bridge routing, middleware, hooks, registry dispatch, result canonicalization.
aliases: []
tags: [tools, dispatch, middleware]
shape: process
steps:
  - id: step.1
    summary: >
      Coerce `function_args` to schema-declared types and short-circuit bridge
      tools (`tool_search`, `tool_describe`, `tool_call`) through the deferred catalog.
  - id: step.2
    summary: >
      Apply tool request middleware, then reject agent-loop-reserved tool names
      before registry dispatch.
  - id: step.3
    summary: >
      Fire `pre_tool_call` plugin hooks; return block/approve directives immediately.
  - id: step.4
    summary: >
      Run ACP/Zed edit approval for mutating file tools, then dispatch through
      `registry.dispatch()` inside tool execution middleware.
  - id: step.5
    summary: >
      Emit `post_tool_call` hook with duration and middleware trace, then apply
      `transform_tool_result` if registered.
entrypoints: [step.1]
produces: []
consumes: [repo:model_tools.py, repo:tools/registry.py]
---

# Tool Dispatch

1. Coerce `function_args` to schema-declared types via `coerce_tool_args` (`model_tools.py:1234`).
2. If the tool is a bridge tool (`tool_search`, `tool_describe`, `tool_call`), resolve the underlying tool name and recurse with the real name after scope-checking against the session's toolset catalog (`model_tools.py:1267-1347`).
3. Apply tool request middleware if not skipped (`model_tools.py:1350-1367`).
4. Reject agent-loop-reserved tool names before registry dispatch (`model_tools.py:1370-1371`).
5. Fire `pre_tool_call` plugin hooks exactly once; if a plugin returns `block` or `approve` directive, return the block message immediately (`model_tools.py:1384-1419`).
6. Run ACP/Zed edit approval for mutating file tools (`write_file`, `patch`) before execution (`model_tools.py:1421-1460`).
7. Dispatch through `registry.dispatch()` inside tool execution middleware, with async bridging via a persistent event loop (`model_tools.py:1493-1527`).
8. Emit `post_tool_call` hook with duration and middleware trace (`model_tools.py:1536-1547`).
9. Apply `transform_tool_result` plugin hook if registered; first valid string result replaces the tool output (`model_tools.py:1557-1585`).
10. On exception, emit error hook and return sanitized error payload (`model_tools.py:1588-1612`).

## Human check

Confirm `handle_function_call` still rejects `_AGENT_LOOP_TOOLS`, fires `pre_tool_call` before execution, and emits both pre and post hooks with the same `task_id`/`session_id`/`turn_id`/`api_request_id` context.

## Deterministic validation

```bash
grep -n "def handle_function_call" model_tools.py
grep -n "_AGENT_LOOP_TOOLS" model_tools.py
grep -n "_emit_post_tool_call_hook" model_tools.py | head -20
grep -n "transform_tool_result" model_tools.py
```

Expected: `handle_function_call` at line 1192, `_AGENT_LOOP_TOOLS` rejection around 1370, post-hook emission around 1536, and transform hook around 1557.
