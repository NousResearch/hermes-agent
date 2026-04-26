---
name: agent-dreaming
description: "Background memory consolidation — reviews recent sessions, scores candidates, promotes durable insights to MEMORY.md, and extracts patterns. Modeled on OpenClaw's three-phase dreaming (Light/Deep/REM). Run via cron or manually."
tags: [memory, consolidation, dreaming, introspection, maintenance]
triggers: ["dream", "consolidate memory", "run dreaming", "memory consolidation"]
---

# Agent Dreaming

Three-phase background memory consolidation. Reviews recent session transcripts, scores what's worth keeping, promotes durable insights, and extracts recurring patterns.

**When to run:** Scheduled cron (recommended: every 6–8 hours) or manually after a burst of activity.

**Composition with memory-lean-check:** Dreaming *produces* better memory entries. Lean check *trims and verifies* them. Run dreaming first, then lean check if memory is near capacity.

**Autonomous vs interactive:** Light and Deep phases run autonomously in both cron and manual modes. REM phase sends a message to the user with proposed structural actions and waits for response — it never creates wiki pages or skills without approval.

---

## Phase 1: Light (Ingest + Stage)

**Goal:** Pull raw material from recent sessions, identify candidates for promotion.

### Steps

1. **Get recent sessions.** Call `session_search()` with no arguments. This returns recent session titles, previews, and timestamps. Note the count — if zero sessions, skip dreaming entirely and report "nothing to dream about."

2. **Resolve wiki path.** Read `wiki.path` from `${HERMES_HOME}/config.yaml` (e.g. `grep wiki.path ${HERMES_HOME}/config.yaml`). Fall back to `${HERMES_HOME}/wiki` if not configured. Never use `$HOME/wiki` — that's shared across profiles and breaks isolation. This path is needed for checking existing wiki pages when evaluating pointer candidates.

3. **Read current memory.** Read `$HERMES_HOME/memories/MEMORY.md`. Count existing entries (split on `§`, count non-empty segments). Note the char usage. This is your baseline.

3. **Filter sessions.** Skip cron sessions (`session.source == "cron"`) — they're fully automated with no user interaction and produce zero correction/preference signal, unless they logged errors.

4. **Deep-dive sessions with signal.** For each remaining session that looks substantive (not just a quick question), call `session_search(query="<session topic keywords>")` to get a full summary. Look for:
   - **User corrections** — "actually it's X not Y", "don't do that", "remember this"
   - **New preferences** — tool choices, style, workflow habits
   - **Environment discoveries** — ports, paths, service names, config quirks
   - **Recurring problems** — things that broke, workarounds found
   - **Resolved ambiguities** — cases where you had to ask clarifying questions and got definitive answers

5. **Stage candidates.** Write a dream artifact to `$HERMES_HOME/dreams/YYYY-MM-DD-HHMM.md` with this structure:

```markdown
# Dream Artifact — YYYY-MM-DD HH:MM

## Sessions Reviewed
- [session_id] title (timestamp)
- ...

## Current Memory State
- [N] entries, [chars] / 2,200 chars ([%] full)

## Candidates
### New Entries (genuinely new info not in MEMORY.md)
- [candidate 1]: description + source session
- ...

### Replacements (corrections to existing entries)
- [existing entry text] → [proposed replacement] + source session
- ...

### Removals (stale/broken info)
- [entry text] — reason (contradicted by session X, wiki page deleted, etc.)

## Patterns Noted
- [any recurring themes across sessions]
```

**Do NOT write to MEMORY.md yet.** This phase only stages candidates.

---

## Phase 2: Deep (Score + Promote)

**Goal:** Evaluate staged candidates against current memory, promote winners.

### Steps

1. **Load the dream artifact** created in Phase 1. If it has no candidates, skip to Phase 3.

2. **Score each candidate.** For each candidate, assess four dimensions. A candidate must pass ALL four to be promoted — any single failure means skip:

   - **Novelty:** Is this genuinely new, or does it overlap with an existing entry? (Check MEMORY.md manually — search for keywords) → FAIL if a similar entry already exists.
   - **Durability:** Will this still be true in 30 days? User preferences and environment facts score high. Task progress, TODOs, and session outcomes score low. → FAIL if the fact is temporary or likely to change.
   - **Specificity:** Is this precise enough to be actionable? "User prefers X" is good. "User might like X" is vague. → FAIL if the entry would require guessing to act on.
   - **Reduction:** Does promoting this let you *remove or shorten* an existing entry? This is the lean check synergy. → Not a hard fail, but candidates with reduction potential get priority.

   **For replacements:** Only assess Novelty (is the new version genuinely better?) and Durability (is the replacement still accurate?). Specificity and Reduction don't apply.

3. **Execute promotions.** For each candidate that scores well on all four dimensions:

   - **New entries:** Call `memory(action='add', target='memory', content='...')` — use wiki pointers (relative to `$WIKI`) where a wiki page exists or should exist. Keep entries under 150 chars ideally.
   - **Replacements:** Call `memory(action='replace', target='memory', old_text='...', new_text='...')` — the old_text must be a unique substring of the existing entry.
   - **Removals:** Call `memory(action='remove', target='memory', old_text='...')` — only if genuinely stale/contradicted.

4. **Log promotions to dream diary.** Append to `$HERMES_HOME/dreams/diary.md`:

```markdown
### YYYY-MM-DD HH:MM — Deep Phase
- PROMOTED: [entry text] (from: [session summary])
- REPLACED: [old] → [new] (reason: [correction source])
- SKIPPED: [candidate] (reason: [novelty/durability/specificity failure])
- REMOVED: [entry] (reason: [stale/contradicted])
```

5. **Post-promotion check.** Re-read MEMORY.md. Verify entry count is correct. Check char usage against thresholds:
   - Under 60%: healthy, no action needed
   - 60–80%: flag in diary, allow replacements but defer new additions next cycle
   - Over 80%: flag in diary as critical, recommend running `memory-lean-check` before next dreaming cycle

---

## Phase 3: REM (Pattern Extract)

**Goal:** Look across recent dream artifacts for recurring themes that warrant structural action.

### Steps

1. **Read recent dream artifacts.** List files in `$HERMES_HOME/dreams/` (excluding `diary.md`). Read the last 3–5 artifacts. Look for:
   - **Repeated corrections** — Same thing corrected 2+ times → candidate for a wiki entry or skill update
   - **Repeated patterns** — Same problem solved 2+ times → candidate for a new skill
   - **Memory gaps** — Topics that came up frequently but have no memory entry → candidate for promotion in next cycle
   - **Skill staleness** — Skills referenced in sessions but with outdated info → candidate for `skill_manage(action='patch')`

2. **For patterns that warrant action, report — don't act.** REM patterns involve structural changes (creating wiki pages, skills, patching skills) that benefit from human review. Instead of executing directly:

   - **Log to dream artifact and diary** as usual (traceability)
   - **Send a message** to the user's home channel summarizing the patterns and proposed actions. Format:

   ```
   🌙 Dreaming REM — patterns found:

   • [PATTERN]: [description]
     → Proposed: [create wiki page / create skill / patch skill / queue for next cycle]

   • [PATTERN]: ...
     → Proposed: ...

   Reply with which to act on, or "skip all" to defer.
   ```

   - **Wait for user response** before executing any structural changes
   - If running as a cron job with no interactive session, send the message and stop — the user can trigger the actions manually or reply in chat

3. **Write REM summary** to the dream artifact (append):

```markdown
## REM Phase — Patterns
- PATTERN: [description] → ACTION: [create wiki page / create skill / patch skill / queue for next cycle]
- ...
```

4. **Update dream diary** with REM findings.

---

## Rules

- **Never fabricate session content.** If `session_search` doesn't return useful detail for a session, skip it. Do not infer or hallucinate what happened.
- **Never promote speculative entries.** Every promoted memory must trace to a specific session interaction. "I think the user might prefer X" is not promotion-worthy.
- **Pointer = entry.** If an entry has a wiki/skill pointer (`see wiki/...` or `see skill '...'`), the pointer IS the entry. Inline detail that duplicates what the pointer targets should be removed. Details belong at the destination, not in memory. Wiki pointers are relative to the resolved `$WIKI` path (from config.yaml, falling back to `${HERMES_HOME}/wiki`).
- **Preserve the § delimiter.** When reading/writing MEMORY.md directly (for counting), never corrupt the entry separator.
- **Keep dreams compact.** Dream artifacts should be under 200 lines. If a session generated 20+ candidates, pick the top 5 by durability score.
- **Respect the char limit.** MEMORY.md has a 2,200 char limit. Over 60% (1,320 chars): defer *new additions* but still allow *replacements that reduce char count*. Over 80%: defer everything and run `memory-lean-check` first.
- **Dream diary is append-only.** Never overwrite previous diary entries. New entries go at the bottom (chronological order).
- **Session IDs in diary.** Always note which session(s) informed each promotion for traceability.

---

## Verification

After a dreaming run:
1. MEMORY.md entry count matches expected (baseline + additions - removals)
2. MEMORY.md char usage is documented in diary
3. Dream artifact exists with all three phases documented
4. Dream diary has a new entry
5. No entries were promoted without a source session reference
