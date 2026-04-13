# Hermes + Obsidian Dreaming MVP

Goal: add a low-risk, local-only “dreaming” layer inspired by OpenClaw/OpenClaw-style background reflection without replacing Hermes’ built-in memory system or introducing cloud storage.

## Bottom line

Yes: dreaming + Obsidian is a good combo if Obsidian stays a structured scratch/continuity layer and Hermes’ built-in memory remains the canonical durable store.

Bad combo: free-form autonomous journaling into the vault.
Good combo: bounded, overwrite-heavy, reviewable notes that help Hermes recover context when the user has memory issues.

## Design principles

1. Local-only by default
   - Use the existing Obsidian provider and local vault path.
   - No cloud vector DB, no external sync requirement, no hosted memory backend.
   - Optional Obsidian Sync/iCloud/Git are user choices, not part of the MVP.

2. Built-in memory remains canonical
   - USER.md and MEMORY.md are the only durable memory Hermes should truly “trust”.
   - Obsidian is a working memory / reflection layer.
   - Promotion into MEMORY.md must be explicit and conservative.

3. Structured, bounded, reversible
   - Prefer overwrite-in-place notes over append-only streams.
   - Prefer a few fixed notes over many autonomous notes.
   - Keep every note short enough for easy manual inspection.

4. Help with real memory failures
   - Focus on “what was I doing?”, “what changed?”, “what matters tomorrow?”, and “what should become durable memory?”
   - Avoid writing private raw conversation dumps or speculative psychologizing.

## What already exists in Hermes today

The current Obsidian memory provider is already close to a safe foundation:
- Config is local and active in `~/.hermes/config.yaml`
  - `memory.provider: obsidian`
  - `memory.obsidian.vault_path: /Users/joshuafeuer/Documents/Obsidian Vault`
  - `memory.obsidian.workspace: Hermes`
- The provider already creates and maintains four structured notes:
  - `user-profile.md`
  - `active-projects.md`
  - `decisions-log.md`
  - `current-focus.md`
- It already mirrors built-in memory into Obsidian and writes a bounded `current-focus.md` snapshot at session end.

That means the safest MVP is not “build a whole new memory system”, but “add a small dreaming pass on top of the existing structured provider behavior”.

## Proposed MVP shape

Add one new generated note and one promotion rule:

### 1. Keep existing notes as-is
- `user-profile.md`
  - Mirror of durable user facts from USER.md.
- `decisions-log.md`
  - Mirror of durable project/system facts from MEMORY.md.
- `current-focus.md`
  - Overwritten session handoff note.
- `active-projects.md`
  - Manual/high-signal project snapshot note.

### 2. Add one new note: `dream-review.md`
Purpose: a bounded nightly/idle synthesis note that answers:
- What was active recently?
- What appears stable vs transient?
- What might deserve promotion to durable memory?
- What should be surfaced tomorrow morning?

Suggested structure:

```md
---
title: Dream Review
managed_by: hermes-obsidian-provider
updated: 2026-...
retention: overwrite-rolling
---

# Dream Review

## Stable candidates
- Repeated preference/fact that appeared across sessions
- Repeated project constraint or environment fact

## Open loops
- Things likely to be resumed soon
- Blockers or unfinished tasks

## Tomorrow cue
- 3-5 bullets max

## Do not promote yet
- Fresh hypotheses
- Emotional venting
- One-off task debris
```

Behavior:
- Overwrite or keep only the last 3 versions.
- Never append forever.
- Never store raw transcript chunks.

## What to write

Write only high-signal, low-ambiguity summaries:

1. Durable user preferences
   - Communication preferences
   - Recurring workflow preferences
   - Accessibility or memory-support preferences
   - Stable tooling choices

2. Durable project facts
   - Repo paths, key commands, important routes, architecture decisions
   - Long-lived constraints and conventions
   - Naming/location facts the user repeatedly forgets

3. Near-term continuity
   - Current focus
   - Open loops
   - Next-step reminders
   - Active blockers

4. Promotion candidates
   - Facts repeated across multiple sessions
   - Facts that would clearly help future recall
   - Facts specific enough to be actionable later

## What NOT to write

Do not write:

1. Raw conversation transcripts
   - No full chat logs in Obsidian dreaming notes.
   - No giant pasted tool outputs.

2. Secrets or sensitive data
   - API keys
   - tokens
   - passwords
   - private identifiers unless already intentionally stored in local memory

3. Unverified inferences about the user
   - No armchair psychology
   - No speculative labels like “the user is anxious/avoidant/etc.”
   - No medical or mental-health interpretations

4. Ephemeral noise
   - One-off errands
   - Temporary URLs
   - throwaway debugging details unless repeatedly useful
   - stale task fragments once resolved

5. Ambiguous claims
   - If confidence is low, keep it in `dream-review.md` under “Do not promote yet”, not MEMORY.md.

## Schedule

Use a very small schedule, not constant background processing.

### Recommended MVP schedule

1. Session end
   - Already happening: write `current-focus.md`.
   - This is the most important memory-support feature.

2. Nightly dream pass: once per day
   - Best default: 1 run overnight or during a low-activity window.
   - Input sources:
     - current built-in USER.md / MEMORY.md
     - `current-focus.md`
     - `active-projects.md`
     - recent session summaries if available
   - Output:
     - overwrite `dream-review.md`
     - optionally propose 0-3 promotions

3. Weekly cleanup pass: once per week
   - Deduplicate stale open loops
   - Prune old/project-resolved items from `active-projects.md`
   - Review whether recurring dream candidates should be promoted or discarded

### Avoid
- Running dream synthesis every few turns
- Auto-creating lots of notes
- Large append-only daily journals
- Any schedule that silently grows token/storage cost

## Promotion to MEMORY.md

Promotion should be explicit, thresholded, and conservative.

### Canonical rule
Only promote from dreaming into MEMORY.md when a candidate is:
- repeated, or
- clearly durable, and
- useful for future assistance, and
- safe to store.

### Minimal promotion heuristic
Promote only if at least one of these is true:
1. It appeared in 2+ sessions, or
2. The user explicitly said it is important to remember, or
3. It is a stable environment/project fact likely to save future time.

### Promotion workflow
1. Dream pass generates “Stable candidates”.
2. Each candidate gets a status:
   - promote
   - hold
   - discard
3. Only `promote` items are written into built-in memory.
4. After promotion, the mirrored Obsidian note updates naturally from USER.md / MEMORY.md.

### Good promotion examples
- “User prefers concise updates and dislikes long preambles.”
- “Obsidian vault path is /Users/joshuafeuer/Documents/Obsidian Vault.”
- “Primary local Hermes workspace is /Users/joshuafeuer/.hermes/hermes-agent.”
- “Current active memory provider is Obsidian.”
- “User benefits from explicit next-step handoff summaries because of memory issues.”

### Bad promotion examples
- “User sounded frustrated today.”
- “Maybe this repo is moving toward X architecture.”
- “Temporary bug is probably caused by Y.”
- “Need groceries / random one-off reminder.”

## How Obsidian should be used

Use Obsidian as a human-readable memory dashboard, not a second autonomous brain.

### Best use
- One folder: `Hermes/`
- 4-5 structured notes total
- Fast manual review in the morning
- Manual edits allowed in `active-projects.md`
- Hermes-managed notes clearly marked in frontmatter

### Suggested division of labor
- Hermes owns:
  - `user-profile.md`
  - `decisions-log.md`
  - `current-focus.md`
  - `dream-review.md`
- Shared human + Hermes note:
  - `active-projects.md`

### Human workflow in Obsidian
- Open `current-focus.md` when resuming work
- Open `dream-review.md` once daily for cueing
- Edit `active-projects.md` when priorities materially change
- Do not manually edit the mirrored durable notes unless intentionally fixing memory

## Top risks and pitfalls

1. Obsidian becoming a junk drawer
   - If Hermes starts spraying notes, recall quality drops fast.
   - Mitigation: fixed small note set, overwrite-heavy behavior.

2. Confusing scratch memory with durable memory
   - If `dream-review.md` is treated as canonical truth, memory quality degrades.
   - Mitigation: MEMORY.md/USER.md remain source of truth.

3. Over-promotion
   - Writing too much into MEMORY.md will pollute recall.
   - Mitigation: require repetition/durability thresholds; cap promotions per day.

4. Privacy drift
   - Even local-only systems can become oversharing systems.
   - Mitigation: never store secrets/raw transcripts; prefer summaries only.

5. Manual edits fighting automatic sync
   - If users edit mirrored notes directly, Hermes may overwrite them.
   - Mitigation: clearly label which notes are mirrored vs manually curated.

6. Stale open loops
   - “Next steps” that never clear become background noise.
   - Mitigation: weekly cleanup and overwrite rather than append.

7. False confidence from weak synthesis
   - A dream pass can hallucinate patterns.
   - Mitigation: promotion only for explicit, repeated, or stable facts.

## Recommended MVP decision

Yes, proceed — but keep it narrow.

The best MVP is:
- local-only
- built on the current Obsidian provider
- one additional bounded `dream-review.md`
- nightly or idle-only synthesis
- conservative promotion into built-in memory
- Obsidian used as a reviewable continuity dashboard, not as the canonical memory store

## Practical implementation summary

If implementing next:
1. Extend the Obsidian provider to manage `dream-review.md`.
2. Add a cron-triggered nightly “dreaming” prompt using existing scheduler infrastructure.
3. Read only:
   - USER.md / MEMORY.md
   - `current-focus.md`
   - `active-projects.md`
   - maybe recent session summaries
4. Write only:
   - overwrite `dream-review.md`
   - optionally add at most 0-3 memory promotions
5. Keep all data local on disk in the existing Obsidian vault and Hermes memory files.

This gives the user the main OpenClaw-like benefit — better continuity and memory support — without adding a new cloud dependency or a high-maintenance memory architecture.
