# Voice Entity Resolution — Absorbed from domain/voice-entity-resolution

## ⚠️ `entity_resolver` can fail outright — fall through to `contact_resolver` (observed 2026-09-21)

Calling `entity_resolver(query=..., limit=15)` returned:

```
[TOOL_ERROR] Tool execution failed: TypeError: _handle() got an unexpected keyword argument 'task_id'
```

This is a defect in the tool wrapper, not a bad argument — do NOT retry with different
params, and do not conclude the entity doesn't exist. The normal routing path
(`entity_resolver` → `contact_resolver` for people) degrades cleanly: **go straight to
`contact_resolver`** for person lookups. For a PROJECT/land/entity target while
`entity_resolver` is down, resolve via `contact_resolver` on the associated person (e.g. the
broker who brought the land), or read the registry sheets directly.

Report the defect to the user only if it actually blocks resolution. The same fault was
already recorded in the `google-workspace` skill (2026-09-18), so treat it as persistent
rather than transient and don't burn turns re-testing it.

## What This Reference Covers

Handles voice message entity resolution — the full 4-phase workflow for extracting project/land/entity context from voice messages before routing to contacts.

**Skill status:** Absorbed into `messaging-drafts` umbrella (2026-05-29). Original at `domain/voice-entity-resolution/`.

## When to Use

Activate on every voice message:
```
[The user sent a voice message~ Here's what they said: "..."]
```

## 4-Phase Resolution Workflow

### Phase 0 — Inline Number Detection
Before any resolution, scan for `+91` or 10-digit Indian mobile patterns.
If name AND number provided together → skip Phase 2/3, proceed directly.

### Phase 1 — Project/Land/Entity Context
1. Check for ordinal references ("folder #2", "file #3") — these are **session-specific**, never reuse numbering from prior sessions
2. Search projects, land_proposals, entities sheets for fuzzy match
3. Confirm before proceeding

**Spreadsheet:** `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`

### Phase 2 — Contact Resolution via Entity Context
For each confirmed project/entity row, read its associated_contacts column to narrow who the message is about.

### Phase 3 — Fallback Full Contact Search
Use `contacts-lookup` skill when person not in entity's associated contacts.

### Phase 4 — Mandatory Final Confirmation Before Any Action
Present complete summary of ALL entities and contacts before executing any send/write tool.

## Key Rules

- **NEVER** start Phase 2 without confirming Phase 1
- **NEVER** call any send/write tool before Phase 4 confirmation
- **NEVER** use People API (`contacts people search`) — it is disabled; use `contact_resolver` instead
- **NEVER** use the STT/user-typed name in drafted messages — always use the canonical name
- **ALWAYS** extract ALL person references from transcript — there may be more than one

## Known STT Corrections (pre-loaded)

| STT heard | Correct | Notes |
|-----------|---------|-------|
| "Chirchiganapalini" | Chichuraganapalli, Tamil Nadu 635103 | Serenity Estate, NOT Bangalore |
| "Chikabhо Navatvara" | Chichuraganapalli, Tamil Nadu 635103 | Cyrillic artifact in transcription |
| "reverse stolen" | Riverstone Farms | STT error from prior session |
| "ragoo" | Raghu Iyer | Voice phonetic variant |
| "Unverson" / "Umbhurasen" | Anbarasan Murugaperumal (Anbu) | STT garbled full name — Anbu is DRA employee, on Telegram |
| "Anjali Muruga Perumal" | Anjali Murugaperumal | NOT Anbarasan — this is his **sister** (different person, same surname). They share the Murugaperumal surname. Phone: +919791179561 (old) / user may have newer number. |
| "Shranta" | Piyush | Advocate name — "Piyush" misheard as "Shranta". Confirm with client before sending. |
| "Somaya" | Sumeya / Sumaya | User phonetically spelled "SU-MA-Y" syllable-by-syllable to correct. When user syllables out a name (he-she-they-it says "X-Y-Z" or "SU-MA-Y"), capture it as the canonical spelling. |
| "Akbar" | Akber | M.Akber Hussain (akber@ahindia.com) — Miller's Road lease co-owner. Voice adds intrusive 'r'. PA Director Atheeq (padirector@ahindia.com). |

## ⚠️ Pitfall — Surname Equivalence Trap (Same Surname ≠ Same Person)

**Trigger:** User says a full name (e.g., "Anjali Murugaperumal") and you find someone in recent context who shares the same surname ("Anbarasan Murugaperumal").

**Problem:** You've been interacting with Person A (same surname) in the recent conversation. When the user mentions Person B (different given name, same surname), your recent-context bias makes you assume they're the same person or a voice-garbled variant of Person A. This wastes time and confuses the user.

**Common Indian family name pattern:** Multiple family members (siblings, parents, cousins) share the same surname. A voice transcription of a sibling's name sounds like a garbled version of the person you've already been talking to.

**Fix:**
1. When the user provides a FULL NAME (given + surname), trust that it's a distinct person — don't map it to a known contact just because the surname matches.
2. Cross-check the given name (first name) independently in contacts and Gmail.
3. If the expected person (the one you've been talking to) has a DIFFERENT given name, they are likely a different individual.
4. Search for the full given name in People API and contacts before assuming it's a variant.
5. Only if no record exists at all should you ask the user for clarification.

**Real example (June 2026):** User said "Anjali Muruga Perumal" (sister of Anbu). Because I had just messaged "Anbarasan Murugaperumal (Anbu)" moments earlier, I assumed "Anjali" was a mispronunciation of "Anbarasan." The user corrected: it was his sister, not him. Had I searched People API for "Anjali" independently, I would have found her separate contact entry with her own phone number.

## How to Save Corrections

After task completion, if user corrected any misheard name:
```python
from skills.noun_learner import handle_correction
handle_correction(sheet_type="projects", row=ROW, misspelling="reverse stolen")
```
