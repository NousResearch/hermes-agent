---
name: session-loop-detection
description: Spot repeated tool-call loops in past agent sessions.
version: 1.1.0
author: 黄云龙 (@nankingjing) + Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [diagnostic, session, loop, anti-pattern]
    category: session-health
    related_skills: [session-health-check]
---

# Session Loop Detection Skill

Find turn-level loops — the same tool called with identical or
near-identical arguments across consecutive assistant turns — in a PAST
session, classify the root cause, and recommend a mitigation. Uses only the
read-only `session_search` tool; it cannot inspect the live session (that
transcript is already in your context — scan it there directly).

## When to Use

- The user reports a session that was "stuck in a loop" or burned tokens
  without making progress
- A past session shows an oversized `message_count` for a simple task
- After `session-health-check` flags repeated tool errors in one session

## Prerequisites

- The `session_search` tool (part of the default toolset; needs the local
  session DB). No network access, no extra dependencies.
- The suspect session's id, or enough topic keywords to find it.

## How to Run

Every step is one `session_search` call (browse, discovery, read, or
scroll — inferred from arguments). There is no per-turn filter shape:
enumerate messages via read/scroll and compare the turns yourself.

## Quick Reference

Loop signature — flag only when ALL three hold:

1. Same `tool_name` with identical or trivially-different arguments on >=3
   consecutive assistant turns
2. The intervening tool results did not change (or kept failing identically)
3. No user message between the repeats

| Goal | Call |
| --- | --- |
| Find the suspect session | `session_search(limit=10)` or `session_search(query="<task keywords>", sort="newest")` |
| Read head + tail | `session_search(session_id="<id>")` |
| Walk the transcript | `session_search(session_id="<id>", around_message_id=<id>, window=20)` |

## Procedure

1. **Locate the session.** Browse recent past sessions with
   `session_search(limit=10)` — a looping session usually stands out by
   `message_count` — or discover by topic with
   `session_search(query="<task keywords>", sort="newest")`. Done when you
   hold the target `session_id`.

2. **Read the tail.** `session_search(session_id="<id>")`. Assistant
   entries carry `tool_calls` (tool name + arguments); tool entries carry
   `tool_name`. Compare consecutive assistant turns in the tail against the
   loop signature — loops usually persist to the end of the session, so the
   last 10 messages give a first verdict.

3. **Walk backward if truncated.** For a large session (`truncated: true`),
   anchor on the earliest tail message id:
   `session_search(session_id="<id>", around_message_id=<id>, window=20)`,
   then keep re-anchoring on `messages[0].id`. When `messages_before` is
   less than the window you have reached the session start. Done when you
   have found the first repeated call, or ruled the loop out back to the
   last user message.

4. **Classify the root cause** from the surrounding messages:
   - **Perception failure** — tool results changed, but the agent repeated
     the call anyway (misread output)
   - **Goal ambiguity** — the last user message is underspecified; the agent
     oscillates between interpretations
   - **Tool failure** — the tool returned the same error every time and the
     agent never treated it as terminal
   - **Context saturation** — repeats begin right after a compaction or
     summary marker in the transcript

5. **Report and recommend.** State the session id, the repeated tool, the
   argument diff, first/last message ids of the repetition, the root-cause
   class, and one mitigation:
   - Perception failure => quote the misread tool result; add an explicit
     check of that field before retrying
   - Goal ambiguity => split the request into discrete, verifiable sub-goals
   - Tool failure => surface the error verbatim; stop after two identical
     failures instead of retrying
   - Context saturation => start a fresh session or compact before retrying

## Pitfalls

- `role_filter` and `limit` apply only to discovery (`query=`); the read
  shape ignores them and browse ignores `role_filter`. Filtering assistant
  turns is your job after reading.
- `sort` accepts only `"newest"` or `"oldest"` — `"recent"` is silently
  dropped.
- Scroll rejects the current session lineage; this workflow is for past
  sessions only.
- Two similar calls are often a legitimate retry. Require >=3 repeats with
  an unchanged result before declaring a loop.

## Verification

- The verdict cites message ids for the first and last repeated call.
- The root-cause class is backed by a quoted message from the transcript.
- No session data was modified — all calls are `session_search` reads.
