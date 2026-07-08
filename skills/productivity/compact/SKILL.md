---
name: compact
description: "Use when the conversation exceeds 30+ turns or when context is getting long — manually compact conversation context by summarizing, saving state to memory, and recommending a fresh session. Inspired by Claude Code /compact."
version: 1.0.0
author: Hermes Agent (adapted from Claude Code compaction pattern)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [compact, context, compaction, summarize, memory, performance]
    related_skills: [project-context]
---

# Compact — Manual Context Compaction

## Overview

Long conversations waste tokens, slow responses, and risk hitting provider context limits (causing HTTP 400 errors). This skill provides a systematic way to compact the conversation context — preserving what matters while shedding what doesn't.

Inspired by Claude Code's `/compact` command.

**Core principle:** Every 30-40 turns, compress the conversation into a dense summary so future turns are fast and reliable.

## When to Use

- User explicitly asks to compact
- Conversation exceeds 30 turns
- Responses become noticeably slow
- You suspect you're approaching the provider's token limit
- After completing a major subtask — compact before starting the next

## Compaction Workflow

### Step 1: Inventory Current State

Scan the conversation and identify:

- **Active tasks**: What are we working on right now? What's in progress?
- **Key decisions**: What choices were made and why?
- **Important file paths**: Which files have been created/modified?
- **Pending items**: What still needs to be done?
- **User preferences expressed**: Any style choices, corrections, or conventions stated?

### Step 2: Save Critical State to Memory

Use the `memory` tool to persist anything that should survive the compaction:

```
memory(action='add', target='memory', content='<fact>')
```

Save ONLY:
- Project conventions discovered this session
- User preferences/corrections
- Tool quirks or workarounds discovered
- Stable environment facts

Do NOT save:
- Task progress ("halfway through feature X")
- Temporary config changes
- Stuff that will be stale in a week

### Step 3: Summarize for the User

Provide a compact summary in this format:

```
=== SESSION SUMMARY ===

In Progress:
- [One-line summary of each active task]

Key Decisions:
- [Decision 1 with brief rationale]
- [Decision 2 with brief rationale]

Files Touched:
- path/to/file1 — [what changed]
- path/to/file2 — [what changed]

Pending:
- [ ] [Next action needed]
- [ ] [Another pending item]

Memory Saved:
- [What was saved to memory]
```

### Step 4: Recommend Next Steps

Tell the user:

```
Recommendation: Start a fresh session with /new.
The new session will auto-load memories and skills — 
no context will be lost. The next session will start 
with a clean slate and fast responses.

To continue, in the new session say:
"[One-liner of what to continue working on]"
```

### Step 5: Verify

Before ending:
- [ ] All critical state saved to `memory`
- [ ] Summary is complete and accurate
- [ ] User knows what to say in the next session
- [ ] No stale task progress saved to memory

## Common Pitfalls

1. **Saving too much to memory** — memory is for durable facts, not session state. 5-10 entries max.
2. **Overly detailed summary** — the summary should be a launchpad for the next session, not a transcript. 300 words max.
3. **Forgetting to save user preferences** — "user prefers X" is the most valuable memory, don't lose it.
4. **Compacting too early** — don't compact before the 20-turn mark; let context build up some useful history first.
5. **Compacting too late** — if you hit HTTP 400, you waited too long. Compact proactively at 30 turns.

## Verification Checklist

- [ ] Active tasks listed with current status
- [ ] Key decisions documented with rationale
- [ ] File paths are absolute (survive session change)
- [ ] Memory saved for durable facts only
- [ ] Summary is under 300 words
- [ ] User knows what to say in next session
