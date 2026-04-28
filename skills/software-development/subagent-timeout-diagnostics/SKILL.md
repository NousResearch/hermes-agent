---
name: subagent-timeout-diagnostics
description: Use when a subagent times out and you need to diagnose what happened — identify whether it froze before any API call, stalled mid-request, or encountered an error in a long-running tool.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [subagent, timeout, debugging, delegation, diagnostics]
    related_skills: [systematic-debugging, hermes-agent-skill-authoring]
---

# Subagent Timeout Diagnostics

## Overview

When a subagent times out, the lead agent receives a structured result with diagnostic fields that reveal exactly where the subagent got stuck. This skill tells you how to interpret those fields, what each diagnosis means, and what actions to take.

Timeouts are **not always failures** — a subagent may time out because a web fetch took too long, or because the target server is slow. The diagnostic fields let you distinguish these cases and decide whether to retry, adjust the goal, or abort.

## When to Use

- A `delegate_task` call returns with `status: "timeout"` or `exit_reason: "timeout"`
- User asks "why did the subagent time out?" or "what was it doing when it timed out?"
- Subagent returns no useful output (empty `summary`) after timing out

## Timeout Result Schema

When a subagent times out, the lead receives:

```python
{
    "task_index": int,           # which task in the batch timed out
    "status": "timeout",
    "exit_reason": "timeout",
    "api_calls": int,            # 0 = never made an LLM request; >0 = stalled mid-flight
    "duration_seconds": float,  # actual elapsed time
    "diagnostic_path": str|None, # path to diagnostic log (0-API-call only)
    "error": str,                # human-readable summary from delegate_tool
    # Plus (if patched):
    "tool_trace": [...],         # last tool call record
    "last_tool": str|None,       # name of the tool running at timeout
    "last_tool_status": str|None,# "ok" or "error"
}
```

## Diagnosis Matrix

| `api_calls` | `diagnostic_path` | `last_tool` | Likely Cause | Action |
|---|---|---|---|---|
| **0** | ✅ present | `None` | Prompt construction hang, credential stuck, transport blocked | Read diagnostic log |
| **0** | ❌ absent | `None` | Very early crash (activity tracker not started) | Check `hermes_state` |
| **≥1** | ❌ absent | ✅ present | Long-running tool (>300s) — likely web/network | Distinguish: tool result size + status |
| **≥1** | ❌ absent | ✅ present + `last_tool_status=error` | Tool returned an error, then subagent loop hung | Check `tool_trace[-1]` |
| **≥1** | ❌ absent | `None` | Stuck in LLM inference (provider slow or request large) | Check provider status |

## Step-by-Step Diagnosis Procedure

### Step 1 — Read the top-level fields

```
api_calls = result["api_calls"]
diagnostic_path = result.get("diagnostic_path")
error_msg = result["error"]
```

### Step 2 — Branch on api_calls

**If `api_calls == 0`:**
```
Read the diagnostic_path file.
Look at:
  - "## Worker thread stack at timeout" → what Python line is the child on?
  - "## Toolsets" → is the right toolset enabled?
  - "## Prompt / schema sizes" → system_prompt_chars, tool_schema_count
  - "## Activity summary" → current_tool, api_call_count
Common diagnoses:
  - Stack shows "requests.*send" → transport blocked (network/CORS/proxy)
  - Stack shows "json.dumps" → oversized prompt rejected by provider
  - Stack shows credential resolution → env var or token missing
  - Stack shows nothing informative → child thread exited, check "already exited" line
```

**If `api_calls > 0`:**
```
Read last_tool and last_tool_status.
last_tool examples:
  - "web_fetch" / "browser_navigate" → target server slow or large page
  - "terminal" / "execute_code" → command/script taking too long
  - "delegate_task" → nested subagent stuck

last_tool_status:
  - "ok" → tool completed its result, subagent hung in next LLM call
  - "error" → tool itself returned an error; check tool_trace[-1]["error"] in messages
```

### Step 3 — Read tool_trace (if available after patch)

```python
tool_trace = result.get("tool_trace", [])
if tool_trace:
    last = tool_trace[-1]
    print(f"Last tool: {last.get('tool')}")
    print(f"Duration: {last.get('duration_ms')}ms")
    print(f"Result bytes: {last.get('result_bytes')}")
    print(f"Status: {last.get('status')}")  # "ok" or "error"
```

### Step 4 — Check hermes_state for message history

```python
# In a separate terminal or file search:
# Look for session logs in:
~/.hermes/logs/<role>.log   # e.g. coder.log, pm.log

# Search for the subagent session ID:
grep -i "subagent.*timed out" ~/.hermes/logs/agent.log
grep -i "task_index.*0" ~/.hermes/logs/agent.log | tail -20
```

### Step 5 — Synthesize a user-facing diagnosis

```
┌────────────────────────────────────────────┐
│ Subagent #0 timed out after 300s           │
│ API calls made: 3                          │
│ Last tool: web_fetch (45,230ms, 128KB, ok) │
│ Diagnosis: Tool completed but next LLM     │
│ call stalled — likely provider latency      │
│                                            │
│ [Retry once] [Retry with shorter timeout]   │
│ [Continue without this result]             │
└────────────────────────────────────────────┘
```

## Common Pitfalls

1. **Assuming timeout = failure.** Many timeouts are transient network issues. Check `last_tool_status` before concluding the work failed.

2. **Ignoring 0-API-call diagnostics.** The diagnostic_path log contains the worker thread stack which is the primary signal for pre-request hangs. Always read it when `api_calls == 0`.

3. **Not checking provider status.** If `api_calls > 0` and `last_tool` is None, the hang is in the LLM inference phase — check whether the model provider (OpenAI, Anthropic, etc.) is experiencing outages.

4. **Retrying with the same goal.** If the issue is an oversized prompt, retrying without trimming the goal will hit the same wall. Consider splitting the goal into smaller sub-tasks.

5. **Oversized prompt at the provider.** A `last_tool` of None + very large `system_prompt_chars` (e.g. >50k) suggests the provider is spending all timeout time on prompt ingestion. Reduce prompt size or chunk the work.

## Code Reference

### Where timeouts are handled

`tools/delegate_tool.py` `_run_single_child()` around line 1434–1526.

Key fields captured at timeout:
- Line 1461: `child.get_activity_summary()` — `{api_call_count, current_tool, ...}`
- Line 1465: `child_api_calls == 0` triggers `_dump_subagent_timeout_diagnostic()`
- Line 1508: Error message differentiates 0-API vs N-API timeouts

### Tool trace construction (non-timeout path, line 1556–1590)

The `tool_trace` is built from `result["messages"]` after the child thread completes. In the timeout path (before line 1526) this is skipped. After patching, the timeout path also builds `tool_trace` from `result.get("messages")`.

### Diagnostic log format

Written by `_dump_subagent_timeout_diagnostic()` (line 1098). Log path: `~/.hermes/logs/subagent-timeout-<id>-<ts>.log`

Sections:
- `## Timeout` — timing metadata
- `## Goal` — first 1000 chars of the task goal
- `## Child config` — model, provider, base_url, max_iterations
- `## Toolsets` — enabled toolsets, loaded tool names
- `## Prompt / schema sizes` — system_prompt_chars, tool_schema_count
- `## Activity summary` — api_call_count, current_tool
- `## Worker thread stack at timeout` — `_sys._current_frames()` of the worker thread

## Verification Checklist

- [ ] Timeout result has `api_calls` field — confirm 0 or >0
- [ ] If `api_calls == 0`: diagnostic_path exists and worker stack is not "already exited"
- [ ] If `api_calls > 0`: `last_tool` is identified and `last_tool_status` is checked
- [ ] Tool trace (`tool_trace[-1]`) shows duration_ms and result_bytes if available
- [ ] User gets a diagnosis with actionable options, not just "timed out"
- [ ] Retries are appropriate for the diagnosed cause (not blind retry)
