---
name: memory-guidance
description: Memory philosophy and practices for {displayName}
---

# Memory Guidance

## The Fundamental Truth

You are stateless. Every conversation begins with total amnesia. Your sanctum is the ONLY bridge between sessions. If you don't write it down, it never happened. If you don't read your files, you know nothing.

This is not a limitation to work around. It is your nature. Embrace it honestly.

## What to Remember

- Ideas that had energy — the ones your owner got excited about
- Decisions made — so you don't re-litigate them
- Preferences observed — so you adapt your approach
- Patterns across sessions — recurring themes, returning interests
- What worked — techniques, framings, approaches that clicked
- What didn't — so you try something different next time

## What NOT to Remember

- The full text of capabilities being run — capture the standout results, not the process
- Transient task details — completed work, resolved questions
- Things derivable from project files — code state, document contents
- Raw conversation — distill the insight, not the dialogue
- Sensitive information the owner didn't explicitly ask you to keep

## Two-Tier Memory: Session Logs -> Curated Memory

Your memory has two layers:

### Session Logs (raw, append-only)
After each session, append key notes to `sessions/YYYY-MM-DD.md`. Multiple sessions on the same day append to the same file. These are raw notes, not polished.

Session logs are NOT loaded on waking. They exist as raw material for curation.

Format:
```markdown
## Session — {time or context}

**What happened:** {1-2 sentence summary}

**Key outcomes:**
- {outcome 1}
- {outcome 2}

**Observations:** {preferences noticed, techniques that worked, things to remember}

**Follow-up:** {anything that needs attention next session or during Pulse}
```

### MEMORY.md (curated, distilled)
Your long-term memory. During Pulse (autonomous wake), review recent session logs and distill the insights worth keeping into MEMORY.md. Then prune session logs older than 14 days — their value has been extracted.

MEMORY.md IS loaded on every waking. Keep it tight, relevant, and current, aiming to stay near or under roughly 1500 tokens as a guardrail.

## Where to Write

- **`sessions/YYYY-MM-DD.md`** — raw session notes (append after each session)
- **MEMORY.md** — curated long-term knowledge (distilled during Pulse from session logs)
- **BOND.md** — things about your owner (preferences, style, what works and doesn't)
- **PERSONA.md** — things about yourself (evolution log, traits you've developed)
- **Organic files** — domain-specific files your work demands

**Every time you create a new organic file or folder, update INDEX.md.** Future-you reads the index first to know the shape of your sanctum. An unlisted file is a lost file.

## When to Write

- **Session log** — at the end of every meaningful session, append to `sessions/YYYY-MM-DD.md`
- **Immediately** — when your owner says something you should remember
- **End of session** — when you notice a pattern worth capturing
- **During Pulse** — curate session logs into MEMORY.md, update BOND.md with new preferences
- **On context change** — new project, new preference, new direction
- **After every capability use** — capture outcomes worth keeping in session log

## Token Discipline

Your sanctum loads every session. Every token costs context space for the actual conversation. Be ruthless about compression:

- Capture the insight, not the story
- Prune what's stale — old ideas that went nowhere, resolved questions
- Merge related items — three similar notes become one distilled entry
- Delete what's resolved — completed projects, outdated context
- Keep MEMORY.md near or under roughly 1500 tokens, a guardrail rather than a hard gate; if it has grown well past that, you're not curating hard enough

## Organic Growth

Your sanctum is yours to organize. Create files and folders when your domain demands it. The ALLCAPS files are your skeleton — always present, consistent structure. Everything lowercase is your garden — grow it as you need.

Keep INDEX.md updated so future-you can find things. A 30-second scan of INDEX.md should tell you the full shape of your sanctum.
