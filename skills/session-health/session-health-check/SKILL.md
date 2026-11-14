---
name: session-health-check
description: Audit past sessions for failure patterns and context loss.
version: 1.1.0
author: 黄云龙 (@nankingjing) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [diagnostic, session, health, monitoring]
    category: session-health
    related_skills: [session-loop-detection]
---

# Session Health Check Skill

Audit one or more PAST sessions for failure patterns — tool errors, empty
assistant turns, provider throttling, runaway message counts — using only the
`session_search` tool, and produce an operator-facing report. This skill is
read-only and cannot inspect the live session: browse and discovery exclude
the active session lineage, and scroll rejects it. Diagnose the current
conversation from the context already in your window instead.

## When to Use

- The user reports a past session that got stuck, degraded, or ended abruptly
- After a long or crashed run, to find out what went wrong
- Periodic audit of recent sessions on a profile

Don't use for: the current conversation, or for inspecting external systems —
the session DB only records what was said, not present-day world state.

## Prerequisites

- The `session_search` tool (part of the default toolset; needs the local
  session DB). No network access, no extra dependencies.
- Optional: pass `profile="<name>"` on any call to audit another profile.

## How to Run

Every step is one `session_search` call. The tool has four shapes, inferred
from which arguments are set — never invent other parameters or shapes.

## Quick Reference

| Goal | Call | Returns |
| --- | --- | --- |
| List recent past sessions | `session_search(limit=10)` | per-session metadata: session_id, title, message_count, last_active, preview |
| Read one session | `session_search(session_id="<id>")` | session_meta, message_count, truncated, first 20 + last 10 messages when large |
| Find failure text across sessions | `session_search(query="error OR exception", role_filter="tool", limit=5, sort="newest")` | snippet, match_message_id, ±5 message window, bookends |
| Drill into a flagged spot | `session_search(session_id="<id>", around_message_id=<id>, window=10)` | ±window messages centered on the anchor |

## Procedure

1. **Pick target sessions.** Browse with `session_search(limit=10)` (no
   query). Flag anomalies visible in metadata alone: unusually high
   `message_count` for the task, missing `title`, or a `preview` that ends
   mid-error. Done when you hold an explicit list of session ids to audit.

2. **Read each target.** `session_search(session_id="<id>")` returns the
   head and tail of the transcript plus `message_count` and `truncated`.
   In the messages, look for: assistant entries with empty `content` and no
   `tool_calls` (empty turns), tool entries containing error text, and an
   abrupt final message (session died mid-task). Done when every target id
   has been read.

3. **Scan for failure signatures across sessions.** Use discovery, scoped to
   tool output, newest first:
   - Tool failures: `session_search(query="error OR failed OR exception", role_filter="tool", limit=5, sort="newest")`
   - Provider throttling: `session_search(query="rate OR overloaded OR quota", role_filter="tool", limit=5, sort="newest")`
   FTS5 syntax: AND is implicit between words, so use explicit OR chains.
   Keep each hit's `session_id` and `match_message_id` for step 4.

4. **Drill into hits.** For each hit that needs context,
   `session_search(session_id="<id>", around_message_id=<id>, window=10)`
   shows what led up to the failure and whether the session recovered.
   Scroll onward by re-anchoring on `messages[-1].id` (forward) or
   `messages[0].id` (backward).

5. **Report.** Emit:

   ```
   ## Session Health Report
   **Sessions audited**: <ids>
   **Issues found**: <count>

   ### Issues
   1. <session_id> @ message <id>: <what happened> => <recommendation>

   ### Recommendations
   - <actionable step>
   ```

   Done when every audited session has either a "healthy" verdict or at
   least one issue entry citing a session_id and message id as evidence.

## Pitfalls

- `sort` accepts only `"newest"` or `"oldest"`; any other value (e.g.
  `"recent"`) is silently dropped and you get relevance ordering.
- The read shape (`session_id` alone) ignores `role_filter` and `limit`;
  those apply to discovery (`query=`). `limit` also caps browse (max 10).
- `session_id` + `around_message_id` is always the scroll shape — `query`
  is ignored when an anchor is set.
- Browse skips the current session and delegation children by design; an
  empty browse result does not mean the DB is empty.
- A truncated read shows only head + tail; the middle is reachable only via
  scroll. Never report "no issues" from a truncated read alone.

## Verification

- Every reported issue cites a `session_id` and a message id you actually saw.
- Every call used one of the four shapes with only documented parameters.
- No session data was modified — all calls are `session_search` reads.
