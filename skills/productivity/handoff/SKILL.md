---
name: handoff
description: "HANDOFF.md — cheapest cross-session memory for AI agents. Write before closing a session, read at the start of the next."
version: 1.0.0
author: baoyu0
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [handoff, workflow, session, continuity, productivity]
    homepage: https://github.com/baoyu0/hermes-agent
---

# Handoff — Cross-Session Continuity

Write a structured handoff document as `HANDOFF.md` in the **project root** before closing a long session. The next session starts by reading it — instantly back in context.

**Why this exists:** AI memory is session-bounded. Projects are week-bounded. The gap between them is usually bridged by the user re-telling context — exhausting and error-prone. One markdown file in the project root is the cheapest cross-session memory there is.

## The Ritual

```
# Before closing a long session
/load handoff
handoff "next step: deploy the API gateway"

# At the start of a new session
Read HANDOFF.md
```

## Required Sections

```markdown
# HANDOFF

## Current Task
One-liner: what are we building/fixing?

## Done
- [x] What's finished (link PRs, file paths, commits)

## Blockers
- What's stuck, with error messages / log paths

## Next Steps
1. Prioritised action items

## Landmines
> ❌ Trap: what happened
> ✅ Root cause: why it happened
> ✅ Fix: what to do instead

## Suggested Skills
- skill-name: why
```

## Principles

- **Save to `HANDOFF.md`** in the project root — not a temp dir. The next session must find it.
- **Don't duplicate** PRDs, PRs, issues, or commit messages — reference them by path/URL.
- **Landmines must include root cause** — not "hit error X", but "because Y, so Z, don't do X again".
- **Redact secrets** — no API keys, passwords, tokens, internal addresses.
- **End with** `▶ Next session: read HANDOFF.md first`.

## Tips

- For short single-session tasks, skip the handoff. Only write it for work that spans >1 session.
- The file accumulates across sessions — append, don't rewrite, so you can trace the project arc.
- Other AI agents (Codex, Claude Code, Cursor) can all read this file — zero lock-in.
